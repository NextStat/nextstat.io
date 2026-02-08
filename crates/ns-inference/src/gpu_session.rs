//! Generic GPU-accelerated single-model fit path (CUDA f64, Metal f32).
//!
//! Provides `GpuSession<A>` (shared GPU state) and `GpuObjective` (fused NLL+gradient
//! with caching) for GPU-accelerated MLE fits of HistFactory models.
//!
//! # Architecture
//!
//! ```text
//! argmin L-BFGS iteration:
//!   1. cost(x_k)     → GpuObjective::eval(x_k)
//!                        → GPU kernel(x_k) → (nll, grad)
//!                        → cache = (x_k, nll, grad)
//!                        → return nll
//!
//!   2. gradient(x_k) → GpuObjective::gradient(x_k)
//!                        → x_k == cached (argmin contract)
//!                        → return cached grad  ← ZERO cost!
//! ```
//!
//! Result: 1 GPU kernel launch per iteration (not 2), because argmin's
//! L-BFGS always calls `cost(x)` then `gradient(x)` with the same params.

use crate::optimizer::{LbfgsbOptimizer, ObjectiveFunction, OptimizationResult, OptimizerConfig};
use ns_compute::gpu_accel::GpuAccelerator;
use ns_core::traits::LogDensityModel;
use ns_core::Result;
use ns_translate::pyhf::HistFactoryModel;
use statrs::function::gamma::ln_gamma;
use std::cell::RefCell;

/// Shared GPU session for multiple fits on the same model structure.
///
/// Static model data (nominal, modifiers, constraints) is uploaded once.
/// Only params and observed data change between fits.
pub struct GpuSession<A: GpuAccelerator> {
    accel: RefCell<A>,
    n_params: usize,
    n_main_bins: usize,
}

impl<A: GpuAccelerator> GpuSession<A> {
    /// Create a session from a pre-constructed accelerator and upload the model’s observed data.
    pub fn from_accelerator(mut accel: A, model: &HistFactoryModel) -> Result<Self> {
        let n_params = accel.n_params();
        let n_main_bins = accel.n_main_bins();

        let (observed, ln_facts, obs_mask) = Self::prepare_observed(model);
        accel.upload_observed_single(&observed, &ln_facts, &obs_mask)?;

        Ok(Self { accel: RefCell::new(accel), n_params, n_main_bins })
    }

    /// Upload new observed data (e.g. after `model.with_observed_main()`).
    pub fn upload_observed(&self, model: &HistFactoryModel) -> Result<()> {
        let (observed, ln_facts, obs_mask) = Self::prepare_observed(model);
        self.accel.borrow_mut().upload_observed_single(&observed, &ln_facts, &obs_mask)
    }

    /// Single NLL + gradient evaluation on GPU.
    pub fn nll_grad(&self, params: &[f64]) -> Result<(f64, Vec<f64>)> {
        self.accel.borrow_mut().single_nll_grad(params)
    }

    /// Run optimizer with GPU objective (no Hessian).
    pub fn fit_minimum(
        &self,
        model: &HistFactoryModel,
        config: &OptimizerConfig,
    ) -> Result<OptimizationResult> {
        let init = model.parameter_init();
        let bounds = model.parameter_bounds();
        self.fit_minimum_from_with_bounds(model, &init, &bounds, config)
    }

    /// Run optimizer with warm-start.
    pub fn fit_minimum_from(
        &self,
        model: &HistFactoryModel,
        initial_params: &[f64],
        config: &OptimizerConfig,
    ) -> Result<OptimizationResult> {
        let bounds = model.parameter_bounds();
        self.fit_minimum_from_with_bounds(model, initial_params, &bounds, config)
    }

    /// Run optimizer with warm-start + custom bounds (for fixed-param fits).
    pub fn fit_minimum_from_with_bounds(
        &self,
        _model: &HistFactoryModel,
        initial_params: &[f64],
        bounds: &[(f64, f64)],
        config: &OptimizerConfig,
    ) -> Result<OptimizationResult> {
        let objective = GpuObjective {
            session: self,
            cache: RefCell::new(GpuCache {
                params: Vec::new(),
                nll: 0.0,
                grad: Vec::new(),
                valid: false,
            }),
        };

        let optimizer = LbfgsbOptimizer::new(config.clone());
        optimizer.minimize(&objective, initial_params, bounds)
    }

    /// Number of model parameters.
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// Number of main bins.
    pub fn n_main_bins(&self) -> usize {
        self.n_main_bins
    }

    /// Prepare observed data arrays from a HistFactory model.
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
}

/// Cached GPU evaluation result.
struct GpuCache {
    params: Vec<f64>,
    nll: f64,
    grad: Vec<f64>,
    valid: bool,
}

/// GPU-backed objective function with fused NLL+gradient caching.
///
/// When argmin calls `cost(x)` followed by `gradient(x)` with the same params,
/// the GPU kernel is launched only once (on `cost`), and `gradient` returns
/// the cached result for free.
struct GpuObjective<'a, A: GpuAccelerator> {
    session: &'a GpuSession<A>,
    cache: RefCell<GpuCache>,
}

// SAFETY: L-BFGS-B optimizer is single-threaded within one minimize() call.
// The RefCell is never shared across threads.
unsafe impl<A: GpuAccelerator> Send for GpuObjective<'_, A> {}
unsafe impl<A: GpuAccelerator> Sync for GpuObjective<'_, A> {}

impl<A: GpuAccelerator> GpuObjective<'_, A> {
    /// Ensure GPU results are computed and cached for the given params.
    fn ensure_computed(&self, params: &[f64]) -> Result<()> {
        let mut cache = self.cache.borrow_mut();
        if cache.valid && cache.params == params {
            return Ok(());
        }

        let (nll, grad) = self.session.nll_grad(params)?;
        cache.params = params.to_vec();
        cache.nll = nll;
        cache.grad = grad;
        cache.valid = true;
        Ok(())
    }
}

impl<A: GpuAccelerator> ObjectiveFunction for GpuObjective<'_, A> {
    fn eval(&self, params: &[f64]) -> Result<f64> {
        self.ensure_computed(params)?;
        Ok(self.cache.borrow().nll)
    }

    fn gradient(&self, params: &[f64]) -> Result<Vec<f64>> {
        self.ensure_computed(params)?;
        Ok(self.cache.borrow().grad.clone())
    }
}

// ---------------------------------------------------------------------------
// Backend constructors + availability checks
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub type CudaGpuSession = GpuSession<ns_compute::cuda_batch::CudaBatchAccelerator>;

#[cfg(feature = "cuda")]
pub fn cuda_session(model: &HistFactoryModel) -> Result<CudaGpuSession> {
    let gpu_data = model.serialize_for_gpu()?;
    let accel = ns_compute::cuda_batch::CudaBatchAccelerator::from_gpu_data(&gpu_data, 1)?;
    GpuSession::from_accelerator(accel, model)
}

#[cfg(feature = "cuda")]
pub fn is_cuda_single_available() -> bool {
    ns_compute::cuda_batch::CudaBatchAccelerator::is_available()
}

#[cfg(feature = "metal")]
pub type MetalGpuSession = GpuSession<ns_compute::metal_batch::MetalBatchAccelerator>;

#[cfg(feature = "metal")]
pub fn metal_session(model: &HistFactoryModel) -> Result<MetalGpuSession> {
    let gpu_data = model.serialize_for_gpu()?;
    let metal_data = ns_compute::metal_types::MetalModelData::from_gpu_data(&gpu_data);
    let accel = ns_compute::metal_batch::MetalBatchAccelerator::from_metal_data(&metal_data, 1)?;
    GpuSession::from_accelerator(accel, model)
}

#[cfg(feature = "metal")]
pub fn is_metal_single_available() -> bool {
    ns_compute::metal_batch::MetalBatchAccelerator::is_available()
}

// ---------------------------------------------------------------------------
// CUDA parity regression tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use ns_translate::pyhf::Workspace;
    use std::path::PathBuf;

    fn load_model_from_fixture(name: &str) -> HistFactoryModel {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures").join(name);
        let json = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!("failed to read fixture {}: {e}", path.display());
        });
        let ws: Workspace = serde_json::from_str(&json).unwrap();
        HistFactoryModel::from_workspace(&ws).unwrap()
    }

    #[test]
    fn cuda_nll_matches_cpu_for_tchannel_workspace() {
        if !is_cuda_single_available() {
            // Allow running the `cuda` feature build on non-CUDA machines.
            return;
        }

        let model = load_model_from_fixture("tchannel_workspace.json");
        let session = cuda_session(&model).unwrap();

        // Init params
        let params = model.parameter_init();
        let cpu_nll = model.nll(&params).unwrap();
        let (gpu_nll, _grad) = session.nll_grad(&params).unwrap();
        assert!(
            (gpu_nll - cpu_nll).abs() < 1e-6,
            "NLL mismatch at init: gpu={gpu_nll:.12}, cpu={cpu_nll:.12}, diff={:.2e}",
            (gpu_nll - cpu_nll).abs()
        );

        // A deterministic perturbed point (stresses interpolation paths)
        let mut perturbed = params.clone();
        for (i, p) in perturbed.iter_mut().enumerate() {
            if i % 3 == 0 {
                *p *= 0.9;
            } else if i % 3 == 1 {
                *p += 0.1;
            } else {
                *p -= 0.1;
            }
        }
        let cpu_nll2 = model.nll(&perturbed).unwrap();
        let (gpu_nll2, _) = session.nll_grad(&perturbed).unwrap();
        assert!(
            (gpu_nll2 - cpu_nll2).abs() < 1e-6,
            "NLL mismatch at perturbed: gpu={gpu_nll2:.12}, cpu={cpu_nll2:.12}, diff={:.2e}",
            (gpu_nll2 - cpu_nll2).abs()
        );
    }
}
