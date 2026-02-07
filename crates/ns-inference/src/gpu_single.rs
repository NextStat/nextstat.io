//! GPU-accelerated single-model fit path.
//!
//! Provides `GpuSession` (shared GPU state) and `GpuObjective` (fused NLL+gradient
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

use crate::optimizer::{LbfgsbOptimizer, ObjectiveFunction, OptimizerConfig, OptimizationResult};
use ns_compute::cuda_batch::CudaBatchAccelerator;
use ns_core::Result;
use ns_translate::pyhf::HistFactoryModel;
use statrs::function::gamma::ln_gamma;
use std::cell::RefCell;

/// Shared GPU session for multiple fits on the same model structure.
///
/// Static model data (nominal, modifiers, constraints) is uploaded once.
/// Only params and observed data change between fits.
pub struct GpuSession {
    accel: RefCell<CudaBatchAccelerator>,
    n_params: usize,
    n_main_bins: usize,
}

impl GpuSession {
    /// Create a GPU session from a HistFactory model.
    ///
    /// Serializes model data and uploads to GPU. Also uploads the model's
    /// observed data so it's ready for immediate use.
    pub fn new(model: &HistFactoryModel) -> Result<Self> {
        let gpu_data = model.serialize_for_gpu()?;
        let n_params = gpu_data.n_params;
        let n_main_bins = gpu_data.n_main_bins;

        // max_batch=1 since we only need single-model evaluation
        let mut accel = CudaBatchAccelerator::from_gpu_data(&gpu_data, 1)?;

        // Upload the model's observed data
        let (observed, ln_facts, obs_mask) = Self::prepare_observed(model);
        accel.upload_observed_single(&observed, &ln_facts, &obs_mask)?;

        Ok(Self {
            accel: RefCell::new(accel),
            n_params,
            n_main_bins,
        })
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
struct GpuObjective<'a> {
    session: &'a GpuSession,
    cache: RefCell<GpuCache>,
}

// SAFETY: L-BFGS-B optimizer is single-threaded within one minimize() call.
// The RefCell is never shared across threads (same justification as HFObjective).
unsafe impl Send for GpuObjective<'_> {}
unsafe impl Sync for GpuObjective<'_> {}

impl GpuObjective<'_> {
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

impl ObjectiveFunction for GpuObjective<'_> {
    fn eval(&self, params: &[f64]) -> Result<f64> {
        self.ensure_computed(params)?;
        Ok(self.cache.borrow().nll)
    }

    fn gradient(&self, params: &[f64]) -> Result<Vec<f64>> {
        self.ensure_computed(params)?;
        Ok(self.cache.borrow().grad.clone())
    }
}

/// Check if CUDA single-model GPU acceleration is available at runtime.
pub fn is_cuda_single_available() -> bool {
    CudaBatchAccelerator::is_available()
}
