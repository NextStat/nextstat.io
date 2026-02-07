//! Differentiable NLL session for PyTorch integration.
//!
//! Wraps `DifferentiableAccelerator` from ns-compute with model-level logic
//! (serialization, observed data upload, signal sample lookup).
//!
//! Two session types:
//! - [`DifferentiableSession`] — Phase 1: NLL at fixed nuisance parameters
//! - [`ProfiledDifferentiableSession`] — Phase 2: Profiled q₀/qμ with
//!   envelope theorem gradients and GPU-accelerated L-BFGS-B fits

use ns_compute::differentiable::{DifferentiableAccelerator, SignalSampleInfo};
use ns_core::traits::LogDensityModel;
use ns_core::Result;
use ns_translate::pyhf::HistFactoryModel;
use statrs::function::gamma::ln_gamma;

use crate::lbfgs::LbfgsState;

// ---------------------------------------------------------------------------
// DifferentiableSession — Phase 1: NLL at fixed nuisance parameters
// ---------------------------------------------------------------------------

/// GPU session for differentiable NLL computation with PyTorch zero-copy.
///
/// Created once per model+signal combination. Holds all GPU state.
/// Thread-safety: NOT thread-safe (single CUDA stream).
pub struct DifferentiableSession {
    accel: DifferentiableAccelerator,
    n_params: usize,
    signal_n_bins: usize,
    init_params: Vec<f64>,
}

impl DifferentiableSession {
    /// Create a new session for the given model and signal sample.
    ///
    /// Serializes the model, uploads it to GPU, and uploads observed data.
    pub fn new(model: &HistFactoryModel, signal_sample_name: &str) -> Result<Self> {
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
        let mut accel = DifferentiableAccelerator::from_gpu_data(&gpu_data, &signal_entries)?;

        // Upload observed data
        let (observed, ln_facts, obs_mask) = Self::prepare_observed(model);
        accel.upload_observed(&observed, &ln_facts, &obs_mask)?;

        let init_params = model.parameter_init();

        Ok(Self {
            accel,
            n_params,
            signal_n_bins: total_signal_bins,
            init_params,
        })
    }

    /// Compute NLL + write signal gradient into PyTorch tensor (zero-copy).
    ///
    /// `signal_ptr` and `grad_signal_ptr` are raw CUDA device pointers
    /// from `torch.Tensor.data_ptr()`.
    ///
    /// Returns NLL scalar. The gradient w.r.t. signal bins is written
    /// directly into `grad_signal_ptr`.
    pub fn nll_grad_signal(
        &mut self,
        params: &[f64],
        signal_ptr: u64,
        grad_signal_ptr: u64,
    ) -> Result<f64> {
        let (nll, _grad_params) =
            self.accel.nll_grad_wrt_signal(params, signal_ptr, grad_signal_ptr)?;
        Ok(nll)
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

    pub(crate) fn prepare_observed(model: &HistFactoryModel) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
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
}

// ---------------------------------------------------------------------------
// ProfiledDifferentiableSession — Phase 2: profiled q₀/qμ + envelope gradient
// ---------------------------------------------------------------------------

/// GPU session for profiled significance (q₀, qμ) with envelope theorem gradient.
///
/// Combines `DifferentiableAccelerator` (GPU NLL + signal/params gradient) with
/// CPU-side L-BFGS-B for profiled fits. The gradient ∂q₀/∂signal uses the
/// envelope theorem:
///
/// ```text
/// ∂q₀/∂s = 2 · (∂NLL/∂s|_{θ=θ̂₀} − ∂NLL/∂s|_{θ=θ̂})
/// ```
///
/// This is exact when the fits converge (∂NLL/∂θ = 0 at optima).
///
/// The signal gradient is returned to Python as a `Vec<f64>` (not zero-copy).
/// This is acceptable because the gradient vector is small (~100 doubles = 800 bytes)
/// and the cost is negligible compared to the two L-BFGS-B fits per forward pass.
pub struct ProfiledDifferentiableSession {
    accel: DifferentiableAccelerator,
    poi_index: usize,
    n_params: usize,
    signal_n_bins: usize,
    init_params: Vec<f64>,
    bounds: Vec<(f64, f64)>,
}

impl ProfiledDifferentiableSession {
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
        let mut accel = DifferentiableAccelerator::from_gpu_data(&gpu_data, &signal_entries)?;

        let (observed, ln_facts, obs_mask) = DifferentiableSession::prepare_observed(model);
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

    /// Compute profiled q₀ and its gradient w.r.t. signal bins.
    ///
    /// Discovery test statistic: q₀ = 2·(NLL(μ=0,θ̂₀) − NLL(μ̂,θ̂))
    ///
    /// `signal_ptr` is a raw CUDA device pointer from `torch.Tensor.data_ptr()`.
    ///
    /// Returns `(q0, grad_signal)`. Gradient uses the envelope theorem.
    pub fn profiled_q0_and_grad(&mut self, signal_ptr: u64) -> Result<(f64, Vec<f64>)> {
        // 1. Free fit (unconditional MLE)
        let (nll_hat, free_params) = self.profile_fit(signal_ptr, &self.bounds.clone(), None)?;
        let mu_hat = free_params[self.poi_index];

        // 2. Conditional fit at μ=0
        let mut bounds0 = self.bounds.clone();
        bounds0[self.poi_index] = (0.0, 0.0);
        let mut warm0 = free_params.clone();
        warm0[self.poi_index] = 0.0;
        let (nll_fixed, fixed_params) = self.profile_fit(signal_ptr, &bounds0, Some(&warm0))?;

        // 3. q₀ = 2·(NLL_fixed − NLL_free), clamped at 0
        let q0 = (2.0 * (nll_fixed - nll_hat)).max(0.0);

        // One-sided discovery: if μ̂ < 0 or q₀ = 0, gradient is zero
        if mu_hat < 0.0 || q0 == 0.0 {
            return Ok((0.0, vec![0.0; self.signal_n_bins]));
        }

        // 4. Envelope gradient: ∂q₀/∂s = 2·(∂NLL/∂s|_{θ̂₀} − ∂NLL/∂s|_{θ̂})
        let grad_free = self.signal_grad_at(&free_params, signal_ptr)?;
        let grad_fixed = self.signal_grad_at(&fixed_params, signal_ptr)?;

        let mut grad = vec![0.0; self.signal_n_bins];
        for i in 0..self.signal_n_bins {
            grad[i] = 2.0 * (grad_fixed[i] - grad_free[i]);
        }

        Ok((q0, grad))
    }

    /// Compute profiled qμ and its gradient w.r.t. signal bins.
    ///
    /// Upper-limit test statistic: qμ = 2·(NLL(μ_test,θ̂_μ) − NLL(μ̂,θ̂))
    /// with clamping: qμ = 0 if μ̂ > μ_test.
    pub fn profiled_qmu_and_grad(
        &mut self,
        mu_test: f64,
        signal_ptr: u64,
    ) -> Result<(f64, Vec<f64>)> {
        // 1. Free fit
        let (nll_hat, free_params) = self.profile_fit(signal_ptr, &self.bounds.clone(), None)?;
        let mu_hat = free_params[self.poi_index];

        // 2. Conditional fit at μ=μ_test
        let mut bounds_mu = self.bounds.clone();
        bounds_mu[self.poi_index] = (mu_test, mu_test);
        let mut warm_mu = free_params.clone();
        warm_mu[self.poi_index] = mu_test;
        let (nll_fixed, fixed_params) =
            self.profile_fit(signal_ptr, &bounds_mu, Some(&warm_mu))?;

        // 3. qμ
        let qmu = (2.0 * (nll_fixed - nll_hat)).max(0.0);

        if mu_hat > mu_test || qmu == 0.0 {
            return Ok((0.0, vec![0.0; self.signal_n_bins]));
        }

        // 4. Envelope gradient
        let grad_free = self.signal_grad_at(&free_params, signal_ptr)?;
        let grad_fixed = self.signal_grad_at(&fixed_params, signal_ptr)?;

        let mut grad = vec![0.0; self.signal_n_bins];
        for i in 0..self.signal_n_bins {
            grad[i] = 2.0 * (grad_fixed[i] - grad_free[i]);
        }

        Ok((qmu, grad))
    }

    /// Run L-BFGS-B fit using GPU NLL+grad evaluations.
    ///
    /// Returns `(nll_at_minimum, parameter_values)`.
    /// Returns an error if the fit fails to converge (envelope theorem
    /// requires ∂NLL/∂θ = 0 at the optima for exact gradients).
    fn profile_fit(
        &mut self,
        signal_ptr: u64,
        bounds: &[(f64, f64)],
        warm_start: Option<&[f64]>,
    ) -> Result<(f64, Vec<f64>)> {
        let x0 = match warm_start {
            Some(w) => w.to_vec(),
            None => self.init_params.clone(),
        };

        let max_iter = 200;
        let mut lbfgs = LbfgsState::new(x0, bounds.to_vec(), 10, 1e-6);

        for _ in 0..max_iter {
            let params = lbfgs.x.clone();
            let (nll, grad_params) = self.accel.nll_and_grad_params(&params, signal_ptr)?;
            lbfgs.step(nll, &grad_params);
            if lbfgs.converged {
                break;
            }
        }

        if !lbfgs.converged {
            return Err(ns_core::Error::Validation(format!(
                "GPU profile fit did not converge after {} iterations (NLL={:.6})",
                max_iter, lbfgs.fval
            )));
        }

        Ok((lbfgs.fval, lbfgs.x.clone()))
    }

    /// Compute ∂NLL/∂signal at given parameter values using the internal buffer.
    fn signal_grad_at(&mut self, params: &[f64], signal_ptr: u64) -> Result<Vec<f64>> {
        self.accel
            .nll_grad_all(params, signal_ptr)
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
}
