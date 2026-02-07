//! Metal differentiable NLL session for profiled fitting on Apple Silicon.
//!
//! Mirrors `ProfiledDifferentiableSession` (CUDA) but:
//! - All GPU computation in f32 (no hardware f64 on Apple Silicon)
//! - Signal uploaded from CPU (no PyTorch zero-copy)
//! - Tolerance relaxed for L-BFGS-B convergence (f32 precision)

use ns_compute::metal_differentiable::{MetalDifferentiableAccelerator, SignalSampleInfo};
use ns_compute::metal_types::MetalModelData;
use ns_core::traits::LogDensityModel;
use ns_core::Result;
use ns_translate::pyhf::HistFactoryModel;
use statrs::function::gamma::ln_gamma;

use crate::lbfgs::LbfgsState;

/// Metal GPU session for profiled significance with envelope theorem gradient.
///
/// Same algorithm as `ProfiledDifferentiableSession` (CUDA) but:
/// - f32 GPU computation (f64↔f32 at boundary)
/// - Signal uploaded via `upload_signal()` (CPU → GPU), not raw CUDA pointers
/// - L-BFGS-B tolerance relaxed to 1e-3 for f32 precision
pub struct MetalProfiledDifferentiableSession {
    accel: MetalDifferentiableAccelerator,
    poi_index: usize,
    n_params: usize,
    signal_n_bins: usize,
    init_params: Vec<f64>,
    bounds: Vec<(f64, f64)>,
}

impl MetalProfiledDifferentiableSession {
    /// Create a profiled session for the given model and signal sample.
    pub fn new(model: &HistFactoryModel, signal_sample_name: &str) -> Result<Self> {
        let poi_index = model.poi_index().ok_or_else(|| {
            ns_core::Error::Validation("No POI defined — cannot compute profiled q₀".into())
        })?;

        let entries_raw = model.signal_sample_gpu_info_all(signal_sample_name)?;
        let signal_entries: Vec<SignalSampleInfo> = entries_raw
            .iter()
            .map(|&(idx, first, n)| SignalSampleInfo {
                sample_idx: idx,
                first_bin: first,
                n_bins: n,
            })
            .collect();
        let total_signal_bins: usize = signal_entries.iter().map(|e| e.n_bins as usize).sum();

        let gpu_data = model.serialize_for_gpu()?;
        let n_params = gpu_data.n_params;
        let metal_data = MetalModelData::from_gpu_data(&gpu_data);
        let mut accel = MetalDifferentiableAccelerator::from_metal_data(&metal_data, &signal_entries)?;

        let (observed, ln_facts, obs_mask) = prepare_observed(model);
        accel.upload_observed(&observed, &ln_facts, &obs_mask)?;

        let init_params = model.parameter_init();
        let bounds = model.parameter_bounds();

        Ok(Self {
            accel,
            poi_index,
            n_params,
            signal_n_bins: total_signal_bins,
            init_params,
            bounds,
        })
    }

    /// Upload signal histogram (f64 → f32 at boundary).
    pub fn upload_signal(&mut self, signal: &[f64]) -> Result<()> {
        self.accel.upload_signal(signal)
    }

    /// Compute profiled q₀ and its gradient w.r.t. signal bins.
    ///
    /// Signal must be uploaded via `upload_signal()` before calling.
    ///
    /// Returns `(q0, grad_signal)`. Gradient uses the envelope theorem.
    pub fn profiled_q0_and_grad(&mut self) -> Result<(f64, Vec<f64>)> {
        // 1. Free fit (unconditional MLE)
        let (nll_hat, free_params) = self.profile_fit(&self.bounds.clone(), None)?;
        let mu_hat = free_params[self.poi_index];

        // 2. Conditional fit at μ=0
        let mut bounds0 = self.bounds.clone();
        bounds0[self.poi_index] = (0.0, 0.0);
        let mut warm0 = free_params.clone();
        warm0[self.poi_index] = 0.0;
        let (nll_fixed, fixed_params) = self.profile_fit(&bounds0, Some(&warm0))?;

        // 3. q₀ = 2·(NLL_fixed − NLL_free), clamped at 0
        let q0 = (2.0 * (nll_fixed - nll_hat)).max(0.0);

        // One-sided discovery: if μ̂ < 0 or q₀ = 0, gradient is zero
        if mu_hat < 0.0 || q0 == 0.0 {
            return Ok((0.0, vec![0.0; self.signal_n_bins]));
        }

        // 4. Envelope gradient: ∂q₀/∂s = 2·(∂NLL/∂s|_{θ̂₀} − ∂NLL/∂s|_{θ̂})
        let grad_free = self.signal_grad_at(&free_params)?;
        let grad_fixed = self.signal_grad_at(&fixed_params)?;

        let mut grad = vec![0.0; self.signal_n_bins];
        for i in 0..self.signal_n_bins {
            grad[i] = 2.0 * (grad_fixed[i] - grad_free[i]);
        }

        Ok((q0, grad))
    }

    /// Compute profiled qμ and its gradient w.r.t. signal bins.
    ///
    /// Signal must be uploaded via `upload_signal()` before calling.
    pub fn profiled_qmu_and_grad(&mut self, mu_test: f64) -> Result<(f64, Vec<f64>)> {
        // 1. Free fit
        let (nll_hat, free_params) = self.profile_fit(&self.bounds.clone(), None)?;
        let mu_hat = free_params[self.poi_index];

        // 2. Conditional fit at μ=μ_test
        let mut bounds_mu = self.bounds.clone();
        bounds_mu[self.poi_index] = (mu_test, mu_test);
        let mut warm_mu = free_params.clone();
        warm_mu[self.poi_index] = mu_test;
        let (nll_fixed, fixed_params) = self.profile_fit(&bounds_mu, Some(&warm_mu))?;

        // 3. qμ
        let qmu = (2.0 * (nll_fixed - nll_hat)).max(0.0);

        if mu_hat > mu_test || qmu == 0.0 {
            return Ok((0.0, vec![0.0; self.signal_n_bins]));
        }

        // 4. Envelope gradient
        let grad_free = self.signal_grad_at(&free_params)?;
        let grad_fixed = self.signal_grad_at(&fixed_params)?;

        let mut grad = vec![0.0; self.signal_n_bins];
        for i in 0..self.signal_n_bins {
            grad[i] = 2.0 * (grad_fixed[i] - grad_free[i]);
        }

        Ok((qmu, grad))
    }

    /// Run L-BFGS-B fit using Metal GPU NLL+grad evaluations.
    fn profile_fit(
        &mut self,
        bounds: &[(f64, f64)],
        warm_start: Option<&[f64]>,
    ) -> Result<(f64, Vec<f64>)> {
        let x0 = match warm_start {
            Some(w) => w.to_vec(),
            None => self.init_params.clone(),
        };

        let max_iter = 200;
        // Relaxed tolerance for f32 precision
        let mut lbfgs = LbfgsState::new(x0, bounds.to_vec(), 10, 1e-3);

        for _ in 0..max_iter {
            let params = lbfgs.x.clone();
            let (nll, grad_params) = self.accel.nll_and_grad_params(&params)?;
            lbfgs.step(nll, &grad_params);
            if lbfgs.converged {
                break;
            }
        }

        if !lbfgs.converged {
            return Err(ns_core::Error::Validation(format!(
                "Metal profile fit did not converge after {} iterations (NLL={:.6})",
                max_iter, lbfgs.fval
            )));
        }

        Ok((lbfgs.fval, lbfgs.x.clone()))
    }

    /// Compute ∂NLL/∂signal at given parameter values.
    fn signal_grad_at(&mut self, params: &[f64]) -> Result<Vec<f64>> {
        self.accel
            .nll_grad_all(params)
            .map(|(_nll, _grad_params, grad_signal)| grad_signal)
    }

    /// Number of signal bins.
    pub fn signal_n_bins(&self) -> usize {
        self.signal_n_bins
    }

    /// Number of model parameters.
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// Default initial parameter values.
    pub fn parameter_init(&self) -> &[f64] {
        &self.init_params
    }

    /// Compute profiled qμ for multiple mu_test values (sequential, session reuse).
    ///
    /// Signal must be uploaded via `upload_signal()` before calling.
    /// Returns `Vec<(qmu, grad_signal)>` — one entry per mu_test value.
    pub fn batch_profiled_qmu(
        &mut self,
        mu_values: &[f64],
    ) -> Result<Vec<(f64, Vec<f64>)>> {
        mu_values
            .iter()
            .map(|&mu| self.profiled_qmu_and_grad(mu))
            .collect()
    }
}

/// Prepare observed data arrays from model.
fn prepare_observed(model: &HistFactoryModel) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut observed = Vec::new();
    let mut ln_facts = Vec::new();
    let mut obs_mask = Vec::new();

    for ch_data in model.observed_main_by_channel() {
        for &obs in &ch_data.y {
            observed.push(obs);
            ln_facts.push(ln_gamma(obs + 1.0));
            obs_mask.push(if obs > 0.0 { 1.0 } else { 0.0 });
        }
    }

    (observed, ln_facts, obs_mask)
}
