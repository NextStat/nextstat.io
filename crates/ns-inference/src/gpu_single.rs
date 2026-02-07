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

use crate::optimizer::{LbfgsbOptimizer, ObjectiveFunction, OptimizationResult, OptimizerConfig};
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

// ---------------------------------------------------------------------------
// GPU parity contract — tolerance constants for integration tests
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GPU parity contract — tolerance constants
// ---------------------------------------------------------------------------

/// NLL at same params: |gpu − cpu| < atol
pub const GPU_NLL_ATOL: f64 = 1e-8;
/// Per-element gradient (slightly relaxed for reduction order)
pub const GPU_GRAD_ATOL: f64 = 1e-5;
/// Best-fit parameter values
pub const GPU_PARAM_ATOL: f64 = 2e-4;
/// NLL at best-fit point
pub const GPU_FIT_NLL_ATOL: f64 = 1e-6;

// ---------------------------------------------------------------------------
// Integration tests — run on CUDA machines only
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::OptimizerConfig;
    use ns_translate::pyhf::{HistFactoryModel, Workspace};

    fn load_simple_model() -> HistFactoryModel {
        let json = include_str!("../../../tests/fixtures/simple_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        HistFactoryModel::from_workspace(&ws).unwrap()
    }

    fn load_complex_model() -> HistFactoryModel {
        let json = include_str!("../../../tests/fixtures/complex_workspace.json");
        let ws: Workspace = serde_json::from_str(json).unwrap();
        HistFactoryModel::from_workspace(&ws).unwrap()
    }

    #[test]
    fn test_gpu_nll_matches_cpu() {
        let model = load_simple_model();
        let session = GpuSession::new(&model).unwrap();

        // Test at init params
        let params = model.parameter_init();
        let cpu_nll = model.nll(&params).unwrap();
        let (gpu_nll, _grad) = session.nll_grad(&params).unwrap();
        assert!(
            (gpu_nll - cpu_nll).abs() < GPU_NLL_ATOL,
            "NLL mismatch at init: gpu={gpu_nll:.15}, cpu={cpu_nll:.15}, diff={:.2e}",
            (gpu_nll - cpu_nll).abs()
        );

        // Test at perturbed params
        let mut perturbed = params.clone();
        for p in &mut perturbed {
            *p *= 1.1;
        }
        let cpu_nll2 = model.nll(&perturbed).unwrap();
        let (gpu_nll2, _) = session.nll_grad(&perturbed).unwrap();
        assert!(
            (gpu_nll2 - cpu_nll2).abs() < GPU_NLL_ATOL,
            "NLL mismatch at perturbed: gpu={gpu_nll2:.15}, cpu={cpu_nll2:.15}, diff={:.2e}",
            (gpu_nll2 - cpu_nll2).abs()
        );
    }

    #[test]
    fn test_gpu_grad_matches_cpu() {
        let model = load_simple_model();
        let session = GpuSession::new(&model).unwrap();
        let params = model.parameter_init();

        let cpu_grad = model.gradient_reverse(&params).unwrap();
        let (_nll, gpu_grad) = session.nll_grad(&params).unwrap();

        assert_eq!(cpu_grad.len(), gpu_grad.len());
        for (i, (&cpu_g, &gpu_g)) in cpu_grad.iter().zip(gpu_grad.iter()).enumerate() {
            assert!(
                (gpu_g - cpu_g).abs() < GPU_GRAD_ATOL,
                "Gradient[{i}] mismatch: gpu={gpu_g:.10}, cpu={cpu_g:.10}, diff={:.2e}",
                (gpu_g - cpu_g).abs()
            );
        }
    }

    #[test]
    fn test_gpu_fit_matches_cpu() {
        let model = load_simple_model();
        let config = OptimizerConfig::default();

        // CPU fit
        let cpu_result = crate::mle::MaximumLikelihoodEstimator::new()
            .fit_histfactory(&model)
            .unwrap();

        // GPU fit
        let session = GpuSession::new(&model).unwrap();
        let gpu_result = session.fit_minimum(&model, &config).unwrap();

        // NLL at minimum
        assert!(
            (gpu_result.fval - cpu_result.nll).abs() < GPU_FIT_NLL_ATOL,
            "Fit NLL mismatch: gpu={:.10}, cpu={:.10}, diff={:.2e}",
            gpu_result.fval, cpu_result.nll,
            (gpu_result.fval - cpu_result.nll).abs()
        );

        // Parameters
        for (i, (&gpu_p, &cpu_p)) in gpu_result.parameters.iter()
            .zip(cpu_result.parameters.iter()).enumerate()
        {
            assert!(
                (gpu_p - cpu_p).abs() < GPU_PARAM_ATOL,
                "Param[{i}] mismatch: gpu={gpu_p:.8}, cpu={cpu_p:.8}, diff={:.2e}",
                (gpu_p - cpu_p).abs()
            );
        }
    }

    #[test]
    fn test_gpu_session_reuse() {
        let model = load_simple_model();
        let config = OptimizerConfig::default();
        let session = GpuSession::new(&model).unwrap();

        // First fit (unconditional)
        let r1 = session.fit_minimum(&model, &config).unwrap();

        // Conditional fit (fix POI)
        let mut bounds = model.parameter_bounds();
        let poi = model.poi_index().unwrap();
        bounds[poi] = (0.0, 0.0);
        let mut warm = r1.parameters.clone();
        warm[poi] = 0.0;
        let r2 = session.fit_minimum_from_with_bounds(&model, &warm, &bounds, &config).unwrap();

        // Third fit (unconditional again — must match first)
        let r3 = session.fit_minimum(&model, &config).unwrap();

        assert!(
            (r1.fval - r3.fval).abs() < 1e-10,
            "Session reuse non-deterministic: first={:.15}, third={:.15}",
            r1.fval, r3.fval
        );
        assert!(r2.fval >= r1.fval - 1e-10, "Conditional NLL should be >= free NLL");
    }

    #[test]
    fn test_gpu_complex_workspace() {
        let model = load_complex_model();
        let session = GpuSession::new(&model).unwrap();
        let config = OptimizerConfig::default();

        // NLL parity
        let params = model.parameter_init();
        let cpu_nll = model.nll(&params).unwrap();
        let (gpu_nll, _) = session.nll_grad(&params).unwrap();
        assert!(
            (gpu_nll - cpu_nll).abs() < GPU_NLL_ATOL,
            "Complex NLL mismatch: gpu={gpu_nll:.15}, cpu={cpu_nll:.15}, diff={:.2e}",
            (gpu_nll - cpu_nll).abs()
        );

        // Fit parity
        let cpu_result = crate::mle::MaximumLikelihoodEstimator::new()
            .fit_histfactory(&model)
            .unwrap();
        let gpu_result = session.fit_minimum(&model, &config).unwrap();

        assert!(
            (gpu_result.fval - cpu_result.nll).abs() < GPU_FIT_NLL_ATOL,
            "Complex fit NLL mismatch: gpu={:.10}, cpu={:.10}, diff={:.2e}",
            gpu_result.fval, cpu_result.nll,
            (gpu_result.fval - cpu_result.nll).abs()
        );
    }

    #[test]
    fn test_gpu_nll_and_grad_at_multiple_points() {
        let model = load_simple_model();
        let session = GpuSession::new(&model).unwrap();

        // Test at several parameter configurations to stress-test the kernel
        let init = model.parameter_init();
        let n = init.len();

        let test_points: Vec<Vec<f64>> = vec![
            init.clone(),
            // All params shifted up
            init.iter().map(|&p| p * 1.2).collect(),
            // All params shifted down
            init.iter().map(|&p| p * 0.8).collect(),
            // Alternating shifts
            init.iter().enumerate().map(|(i, &p)| {
                if i % 2 == 0 { p * 1.3 } else { p * 0.7 }
            }).collect(),
        ];

        for (j, params) in test_points.iter().enumerate() {
            let cpu_nll = model.nll(params).unwrap();
            let cpu_grad = model.gradient_reverse(params).unwrap();
            let (gpu_nll, gpu_grad) = session.nll_grad(params).unwrap();

            assert!(
                (gpu_nll - cpu_nll).abs() < GPU_NLL_ATOL,
                "Point {j}: NLL gpu={gpu_nll:.15}, cpu={cpu_nll:.15}, diff={:.2e}",
                (gpu_nll - cpu_nll).abs()
            );

            for (i, (&gg, &cg)) in gpu_grad.iter().zip(cpu_grad.iter()).enumerate() {
                assert!(
                    (gg - cg).abs() < GPU_GRAD_ATOL,
                    "Point {j}, grad[{i}]: gpu={gg:.10}, cpu={cg:.10}, diff={:.2e}",
                    (gg - cg).abs()
                );
            }
        }
    }
}
