//! Maximum Likelihood Estimation

use crate::optimizer::{LbfgsbOptimizer, ObjectiveFunction, OptimizerConfig};
use nalgebra::DMatrix;
use ns_core::traits::{LogDensityModel, PreparedNll};
use ns_core::{FitResult, Result};
use ns_translate::pyhf::{HistFactoryModel, NllScratch};
use std::cell::RefCell;

/// Diagnostics extracted from an `OptimizationResult`.
struct OptDiagnostics {
    reason: String,
    grad_norm: f64,
    initial_nll: f64,
    n_active: usize,
    edm: f64,
}

/// Compute diagnostics from an `OptimizationResult` and parameter bounds.
fn diagnostics_from_opt(
    opt: &crate::optimizer::OptimizationResult,
    bounds: &[(f64, f64)],
) -> OptDiagnostics {
    let grad_norm = opt
        .final_gradient
        .as_ref()
        .map(|g| g.iter().map(|x| x * x).sum::<f64>().sqrt())
        .unwrap_or(f64::NAN);
    let n_active = opt
        .parameters
        .iter()
        .zip(bounds.iter())
        .filter(|(x, (lo, hi))| (**x - lo).abs() < 1e-10 || (**x - hi).abs() < 1e-10)
        .count();
    OptDiagnostics {
        reason: opt.message.clone(),
        grad_norm,
        initial_nll: opt.initial_cost,
        n_active,
        edm: opt.edm,
    }
}

/// Apply diagnostics to a `FitResult`.
fn apply_diagnostics(fr: FitResult, d: OptDiagnostics) -> FitResult {
    fr.with_diagnostics(d.reason, d.grad_norm, d.initial_nll, d.n_active).with_edm(d.edm)
}

/// Check for identifiability issues based on the Hessian and uncertainties.
///
/// Returns a list of human-readable warning strings (empty if model is well-identified).
pub fn identifiability_warnings(
    hessian: &DMatrix<f64>,
    n: usize,
    param_names: &[String],
    uncertainties: &[f64],
) -> Vec<String> {
    let mut warnings = Vec::new();

    // 1. Near-singular Hessian: condition number via SVD
    if n > 0 {
        let svd = hessian.clone().svd(false, false);
        let svals = &svd.singular_values;
        let s_max = svals.iter().fold(0.0_f64, |a, &b| a.max(b));
        let s_min = svals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        if s_min > 0.0 {
            let cond = s_max / s_min;
            if cond > 1e8 {
                warnings.push(format!(
                    "Hessian condition number = {:.1e}: model may be poorly identified",
                    cond
                ));
            }
        } else {
            warnings.push("Hessian is singular: model is not identifiable".into());
        }
    }

    // 2. Individual parameter: NaN/Inf uncertainty
    for i in 0..n.min(param_names.len()).min(uncertainties.len()) {
        if uncertainties[i].is_nan() || uncertainties[i].is_infinite() {
            warnings.push(format!(
                "Parameter '{}': uncertainty is {}",
                param_names[i], uncertainties[i]
            ));
        }
    }

    // 3. Near-zero Hessian diagonal → parameter not identifiable
    for i in 0..n.min(param_names.len()) {
        if hessian[(i, i)].abs() < 1e-12 {
            warnings.push(format!(
                "Parameter '{}': near-zero Hessian diagonal — not identifiable",
                param_names[i]
            ));
        }
    }

    warnings
}

/// Maximum Likelihood Estimator
///
/// Fits statistical models by minimizing negative log-likelihood.
#[derive(Clone)]
pub struct MaximumLikelihoodEstimator {
    config: OptimizerConfig,
}

impl MaximumLikelihoodEstimator {
    /// Create a new MLE with default configuration
    pub fn new() -> Self {
        Self { config: OptimizerConfig::default() }
    }

    /// Create MLE with custom optimizer configuration
    pub fn with_config(config: OptimizerConfig) -> Self {
        Self { config }
    }

    /// Access the optimizer configuration.
    pub fn config(&self) -> &OptimizerConfig {
        &self.config
    }

    /// Fit any [`LogDensityModel`] by minimizing negative log-likelihood.
    ///
    /// # Arguments
    /// * `model` - Statistical model to fit
    ///
    /// # Returns
    /// FitResult with best-fit parameters, uncertainties, covariance, and fit quality
    pub fn fit<M: LogDensityModel>(&self, model: &M) -> Result<FitResult> {
        let result = self.fit_minimum(model)?;
        let bounds = model.parameter_bounds();
        let diag = diagnostics_from_opt(&result, &bounds);

        // Compute full Hessian and covariance matrix
        let hessian = self.compute_hessian(model, &result.parameters)?;
        let n = result.parameters.len();
        let diag_uncertainties = self.diagonal_uncertainties(&hessian, n);

        let fr = match self.invert_hessian(&hessian, n) {
            Some(covariance) => {
                // Uncertainties from diagonal of covariance matrix
                let mut all_variances_ok = true;
                let mut uncertainties = Vec::with_capacity(n);
                for i in 0..n {
                    let var = covariance[(i, i)];
                    if var.is_finite() && var > 0.0 {
                        uncertainties.push(var.sqrt());
                    } else {
                        all_variances_ok = false;
                        uncertainties.push(diag_uncertainties[i]);
                    }
                }

                if all_variances_ok {
                    // Store covariance as row-major flat Vec
                    let cov_flat: Vec<f64> = covariance.iter().copied().collect();

                    FitResult::with_covariance(
                        result.parameters,
                        uncertainties,
                        cov_flat,
                        result.fval,
                        result.converged,
                        result.n_iter as usize,
                        result.n_fev,
                        result.n_gev,
                    )
                } else {
                    log::warn!("Invalid covariance diagonal; omitting covariance matrix");
                    FitResult::new(
                        result.parameters,
                        uncertainties,
                        result.fval,
                        result.converged,
                        result.n_iter as usize,
                        result.n_fev,
                        result.n_gev,
                    )
                }
            }
            None => {
                // Hessian inversion failed; fall back to diagonal estimate
                log::warn!("Hessian inversion failed, using diagonal approximation");
                let uncertainties = self.diagonal_uncertainties(&hessian, n);
                FitResult::new(
                    result.parameters,
                    uncertainties,
                    result.fval,
                    result.converged,
                    result.n_iter as usize,
                    result.n_fev,
                    result.n_gev,
                )
            }
        };
        let mut fr = apply_diagnostics(fr, diag);

        // Identifiability warnings
        let param_names = model.parameter_names();
        let warns = identifiability_warnings(&hessian, n, &param_names, &fr.uncertainties);
        fr.warnings = warns;

        Ok(fr)
    }

    /// Fit from an explicit starting point (warm-start) with full Hessian/covariance.
    ///
    /// Like [`fit`], but uses `initial_params` instead of `model.parameter_init()`.
    pub fn fit_from<M: LogDensityModel>(
        &self,
        model: &M,
        initial_params: &[f64],
    ) -> Result<FitResult> {
        let result = self.fit_minimum_from(model, initial_params)?;
        let bounds = model.parameter_bounds();
        let diag = diagnostics_from_opt(&result, &bounds);

        let hessian = self.compute_hessian(model, &result.parameters)?;
        let n = result.parameters.len();
        let diag_uncertainties = self.diagonal_uncertainties(&hessian, n);

        let fr = match self.invert_hessian(&hessian, n) {
            Some(covariance) => {
                let mut all_variances_ok = true;
                let mut uncertainties = Vec::with_capacity(n);
                for i in 0..n {
                    let var = covariance[(i, i)];
                    if var.is_finite() && var > 0.0 {
                        uncertainties.push(var.sqrt());
                    } else {
                        all_variances_ok = false;
                        uncertainties.push(diag_uncertainties[i]);
                    }
                }
                if all_variances_ok {
                    let cov_flat: Vec<f64> = covariance.iter().copied().collect();
                    FitResult::with_covariance(
                        result.parameters,
                        uncertainties,
                        cov_flat,
                        result.fval,
                        result.converged,
                        result.n_iter as usize,
                        result.n_fev,
                        result.n_gev,
                    )
                } else {
                    log::warn!("Invalid covariance diagonal; omitting covariance matrix");
                    FitResult::new(
                        result.parameters,
                        uncertainties,
                        result.fval,
                        result.converged,
                        result.n_iter as usize,
                        result.n_fev,
                        result.n_gev,
                    )
                }
            }
            None => {
                log::warn!("Hessian inversion failed, using diagonal approximation");
                let uncertainties = self.diagonal_uncertainties(&hessian, n);
                FitResult::new(
                    result.parameters,
                    uncertainties,
                    result.fval,
                    result.converged,
                    result.n_iter as usize,
                    result.n_fev,
                    result.n_gev,
                )
            }
        };
        let mut fr = apply_diagnostics(fr, diag);

        let param_names = model.parameter_names();
        let warns = identifiability_warnings(&hessian, n, &param_names, &fr.uncertainties);
        fr.warnings = warns;

        Ok(fr)
    }

    /// Minimize NLL and return the optimizer result.
    ///
    /// Fast path: does not compute Hessian/covariance. Intended for repeated minimizations
    /// (profile likelihood scans, hypotest/limits).
    pub fn fit_minimum(
        &self,
        model: &impl LogDensityModel,
    ) -> Result<crate::optimizer::OptimizationResult> {
        let initial_params: Vec<f64> = model.parameter_init();
        self.fit_minimum_from(model, &initial_params)
    }

    /// Minimize NLL from an explicit starting point (warm-start).
    ///
    /// This is important for profile scans / CLs scans where consecutive points
    /// are highly correlated and re-starting from `parameter_init()` is slow.
    pub fn fit_minimum_from(
        &self,
        model: &impl LogDensityModel,
        initial_params: &[f64],
    ) -> Result<crate::optimizer::OptimizationResult> {
        if initial_params.len() != model.dim() {
            return Err(ns_core::Error::Validation(format!(
                "fit_minimum_from: initial_params length {} != model.dim() {}",
                initial_params.len(),
                model.dim()
            )));
        }
        let bounds: Vec<(f64, f64)> = model.parameter_bounds();
        let prepared = model.prepared();

        #[derive(Default)]
        struct Cache {
            params: Vec<f64>,
            nll: f64,
            grad: Vec<f64>,
            nll_valid: bool,
            grad_valid: bool,
        }

        struct CachedObjective<'a, M: LogDensityModel + ?Sized> {
            prepared: M::Prepared<'a>,
            model: &'a M,
            prefer_fused: bool,
            cache: RefCell<Cache>,
        }

        // SAFETY: L-BFGS-B optimizer is single-threaded within one minimize() call.
        // The RefCell is never shared across threads.
        unsafe impl<M: LogDensityModel + ?Sized> Send for CachedObjective<'_, M> {}
        unsafe impl<M: LogDensityModel + ?Sized> Sync for CachedObjective<'_, M> {}

        impl<M: LogDensityModel + ?Sized> CachedObjective<'_, M> {
            fn ensure_cost(&self, params: &[f64]) -> Result<()> {
                let mut cache = self.cache.borrow_mut();
                if cache.nll_valid && cache.params == params {
                    return Ok(());
                }

                if self.prefer_fused {
                    let (nll, grad) = self.model.nll_grad_prepared(&self.prepared, params)?;
                    cache.params = params.to_vec();
                    cache.nll = nll;
                    cache.grad = grad;
                    cache.nll_valid = true;
                    cache.grad_valid = true;
                } else {
                    let nll = self.prepared.nll(params)?;
                    cache.params = params.to_vec();
                    cache.nll = nll;
                    cache.nll_valid = true;
                    cache.grad_valid = false;
                }
                Ok(())
            }

            fn ensure_grad(&self, params: &[f64]) -> Result<()> {
                let mut cache = self.cache.borrow_mut();
                if cache.grad_valid && cache.params == params {
                    return Ok(());
                }

                if self.prefer_fused {
                    let (nll, grad) = self.model.nll_grad_prepared(&self.prepared, params)?;
                    cache.params = params.to_vec();
                    cache.nll = nll;
                    cache.grad = grad;
                    cache.nll_valid = true;
                    cache.grad_valid = true;
                } else {
                    let grad = self.model.grad_nll(params)?;
                    cache.params = params.to_vec();
                    cache.grad = grad;
                    cache.nll_valid = false;
                    cache.grad_valid = true;
                }
                Ok(())
            }
        }

        impl<M: LogDensityModel + ?Sized> ObjectiveFunction for CachedObjective<'_, M> {
            fn eval(&self, params: &[f64]) -> Result<f64> {
                self.ensure_cost(params)?;
                Ok(self.cache.borrow().nll)
            }

            fn gradient(&self, params: &[f64]) -> Result<Vec<f64>> {
                self.ensure_grad(params)?;
                Ok(self.cache.borrow().grad.clone())
            }
        }

        let objective = CachedObjective {
            prepared,
            model,
            prefer_fused: model.prefer_fused_eval_grad(),
            cache: RefCell::new(Cache::default()),
        };
        let optimizer = LbfgsbOptimizer::new(self.config.clone());
        optimizer.minimize(&objective, initial_params, &bounds)
    }

    /// Minimize NLL for a [`HistFactoryModel`], reusing the AD tape across gradient calls.
    ///
    /// Convenience wrapper that creates a fresh tape internally.
    /// For batch fitting, prefer [`fit_minimum_histfactory_with_tape`] with a
    /// per-thread tape via Rayon `map_init`.
    pub fn fit_minimum_histfactory(
        &self,
        model: &HistFactoryModel,
    ) -> Result<crate::optimizer::OptimizationResult> {
        let mut tape = ns_ad::tape::Tape::with_capacity(model.n_params() * 20);
        self.fit_minimum_histfactory_with_tape(model, &mut tape)
    }

    /// Minimize NLL for a [`HistFactoryModel`], reusing a caller-provided AD tape.
    ///
    /// The tape is cleared and reused on every gradient evaluation, avoiding
    /// heap allocation.  Pass a per-thread tape from Rayon `map_init` to get
    /// one allocation per thread instead of per fit.
    pub fn fit_minimum_histfactory_with_tape(
        &self,
        model: &HistFactoryModel,
        tape: &mut ns_ad::tape::Tape,
    ) -> Result<crate::optimizer::OptimizationResult> {
        let initial_params: Vec<f64> = model.parameter_init();
        let bounds: Vec<(f64, f64)> = model.parameter_bounds();
        let prepared = model.prepare();
        let tape_cell = RefCell::new(std::mem::take(tape));
        let scratch_cell = RefCell::new(NllScratch::for_model(model));

        struct HFObjective<'a> {
            prepared: ns_translate::pyhf::PreparedModel<'a>,
            model: &'a HistFactoryModel,
            tape: RefCell<ns_ad::tape::Tape>,
            scratch: RefCell<NllScratch>,
        }

        // SAFETY: L-BFGS-B optimizer is single-threaded within one minimize() call.
        // The RefCell is never shared across threads.
        unsafe impl Send for HFObjective<'_> {}
        unsafe impl Sync for HFObjective<'_> {}

        impl ObjectiveFunction for HFObjective<'_> {
            fn eval(&self, params: &[f64]) -> Result<f64> {
                let mut s = self.scratch.borrow_mut();
                self.prepared.nll_reuse(params, &mut s)
            }

            fn gradient(&self, params: &[f64]) -> Result<Vec<f64>> {
                let mut t = self.tape.borrow_mut();
                self.model.gradient_reverse_reuse(params, &mut t)
            }
        }

        let objective = HFObjective { prepared, model, tape: tape_cell, scratch: scratch_cell };
        let optimizer = LbfgsbOptimizer::new(self.config.clone());
        let result = optimizer.minimize(&objective, &initial_params, &bounds);

        // Return tape to caller so capacity is preserved across fits
        *tape = objective.tape.into_inner();

        result
    }

    /// Minimize NLL for a [`HistFactoryModel`], reusing both the AD tape and NllScratch.
    ///
    /// This is the zero-alloc hot-loop entrypoint for workflows that repeatedly refit the same
    /// model structure (profile scans, differentiable objectives, training loops).
    pub fn fit_minimum_histfactory_with_tape_reuse(
        &self,
        model: &HistFactoryModel,
        tape: &mut ns_ad::tape::Tape,
        scratch: &mut NllScratch,
    ) -> Result<crate::optimizer::OptimizationResult> {
        let initial_params: Vec<f64> = model.parameter_init();
        let bounds: Vec<(f64, f64)> = model.parameter_bounds();
        self.fit_minimum_histfactory_from_with_bounds_reuse(
            model,
            &initial_params,
            &bounds,
            tape,
            scratch,
        )
    }

    /// Minimize NLL for a [`HistFactoryModel`] from an explicit starting point, reusing a caller-provided AD tape.
    ///
    /// This is the HistFactory equivalent of [`fit_minimum_from`] and is intended for warm-started
    /// workflows (toys, profile scans, limits) where consecutive minimizations are correlated.
    pub fn fit_minimum_histfactory_from_with_tape(
        &self,
        model: &HistFactoryModel,
        initial_params: &[f64],
        tape: &mut ns_ad::tape::Tape,
    ) -> Result<crate::optimizer::OptimizationResult> {
        if initial_params.len() != model.dim() {
            return Err(ns_core::Error::Validation(format!(
                "fit_minimum_histfactory_from_with_tape: initial_params length {} != model.dim() {}",
                initial_params.len(),
                model.dim()
            )));
        }

        let bounds: Vec<(f64, f64)> = model.parameter_bounds();
        self.fit_minimum_histfactory_from_with_bounds_with_tape(
            model,
            initial_params,
            &bounds,
            tape,
        )
    }

    /// Minimize NLL for a [`HistFactoryModel`] from an explicit starting point and explicit bounds,
    /// reusing a caller-provided AD tape.
    ///
    /// This enables fast fixed-parameter fits without cloning the full model: clamp the bounds
    /// of the fixed parameter to `(value, value)` and pass them here.
    pub fn fit_minimum_histfactory_from_with_bounds_with_tape(
        &self,
        model: &HistFactoryModel,
        initial_params: &[f64],
        bounds: &[(f64, f64)],
        tape: &mut ns_ad::tape::Tape,
    ) -> Result<crate::optimizer::OptimizationResult> {
        let mut scratch = NllScratch::for_model(model);
        self.fit_minimum_histfactory_from_with_bounds_reuse(
            model,
            initial_params,
            bounds,
            tape,
            &mut scratch,
        )
    }

    /// Minimize NLL for a [`HistFactoryModel`] reusing both tape and NllScratch.
    ///
    /// Zero-alloc variant for hot loops (ranking, profile scan). Pass per-thread
    /// tape and scratch from Rayon `map_init` to avoid any heap allocation.
    pub fn fit_minimum_histfactory_from_with_bounds_reuse(
        &self,
        model: &HistFactoryModel,
        initial_params: &[f64],
        bounds: &[(f64, f64)],
        tape: &mut ns_ad::tape::Tape,
        scratch: &mut NllScratch,
    ) -> Result<crate::optimizer::OptimizationResult> {
        if initial_params.len() != model.dim() {
            return Err(ns_core::Error::Validation(format!(
                "fit_minimum_histfactory_from_with_bounds_reuse: initial_params length {} != model.dim() {}",
                initial_params.len(),
                model.dim()
            )));
        }
        if bounds.len() != model.dim() {
            return Err(ns_core::Error::Validation(format!(
                "fit_minimum_histfactory_from_with_bounds_reuse: bounds length {} != model.dim() {}",
                bounds.len(),
                model.dim()
            )));
        }

        let prepared = model.prepare();
        let tape_cell = RefCell::new(std::mem::take(tape));
        let scratch_cell = RefCell::new(std::mem::replace(scratch, NllScratch::empty()));

        struct HFObjective<'a> {
            prepared: ns_translate::pyhf::PreparedModel<'a>,
            model: &'a HistFactoryModel,
            tape: RefCell<ns_ad::tape::Tape>,
            scratch: RefCell<NllScratch>,
        }

        // SAFETY: L-BFGS-B optimizer is single-threaded within one minimize() call.
        // The RefCell is never shared across threads.
        unsafe impl Send for HFObjective<'_> {}
        unsafe impl Sync for HFObjective<'_> {}

        impl ObjectiveFunction for HFObjective<'_> {
            fn eval(&self, params: &[f64]) -> Result<f64> {
                let mut s = self.scratch.borrow_mut();
                self.prepared.nll_reuse(params, &mut s)
            }

            fn gradient(&self, params: &[f64]) -> Result<Vec<f64>> {
                let mut t = self.tape.borrow_mut();
                self.model.gradient_reverse_reuse(params, &mut t)
            }
        }

        let objective = HFObjective { prepared, model, tape: tape_cell, scratch: scratch_cell };
        let optimizer = LbfgsbOptimizer::new(self.config.clone());
        let result = optimizer.minimize(&objective, initial_params, bounds);

        // Return tape + scratch to caller so capacity is preserved across fits
        *tape = objective.tape.into_inner();
        *scratch = objective.scratch.into_inner();

        result
    }

    /// Compute full Hessian matrix using forward differences of analytical gradient.
    ///
    /// H_{ij} = (∂g_i/∂x_j) ≈ (g_i(x + ε·e_j) − g_i(x)) / ε
    ///
    /// Cost: N+1 gradient evaluations (each O(1) via reverse-mode AD).
    fn compute_hessian(
        &self,
        model: &impl LogDensityModel,
        best_params: &[f64],
    ) -> Result<DMatrix<f64>> {
        let n = best_params.len();
        let grad_center = model.grad_nll(best_params)?;

        let mut hessian = DMatrix::zeros(n, n);

        for j in 0..n {
            let eps = 1e-4 * best_params[j].abs().max(1.0);

            let mut params_plus = best_params.to_vec();
            params_plus[j] += eps;
            let grad_plus = model.grad_nll(&params_plus)?;

            for i in 0..n {
                hessian[(i, j)] = (grad_plus[i] - grad_center[i]) / eps;
            }
        }

        // Symmetrise: H = (H + H^T) / 2
        let ht = hessian.transpose();
        hessian = (&hessian + &ht) * 0.5;

        Ok(hessian)
    }

    /// Invert Hessian to get covariance matrix via Cholesky decomposition.
    ///
    /// Returns `None` if the Hessian is not positive definite.
    fn invert_hessian(&self, hessian: &DMatrix<f64>, n: usize) -> Option<DMatrix<f64>> {
        // We want a positive-(semi)definite covariance; even at a valid minimum the
        // numerically estimated Hessian can be slightly indefinite. Prefer a damped
        // Cholesky solve to avoid negative variances (which then explode to 1e6).
        let identity = DMatrix::identity(n, n);

        // Scale damping to the Hessian diagonal to be unit-ish across models.
        let diag_scale = (0..n).map(|i| hessian[(i, i)].abs()).fold(0.0_f64, f64::max).max(1.0);

        let mut h_damped = hessian.clone();
        let mut damping = 0.0_f64;
        let max_attempts = 10;

        for attempt in 0..max_attempts {
            if let Some(chol) = nalgebra::linalg::Cholesky::new(h_damped.clone()) {
                return Some(chol.solve(&identity));
            }

            if attempt + 1 == max_attempts {
                break;
            }

            // Increase diagonal damping geometrically.
            let next_damping = if damping == 0.0 { diag_scale * 1e-9 } else { damping * 10.0 };
            let add = next_damping - damping;
            for i in 0..n {
                h_damped[(i, i)] += add;
            }
            damping = next_damping;
        }

        let cov = h_damped.lu().try_inverse()?;
        // Reject clearly-bad inverses (negative/NaN variances).
        for i in 0..n {
            let v = cov[(i, i)];
            if !(v.is_finite() && v > 0.0) {
                return None;
            }
        }
        Some(cov)
    }

    /// Extract uncertainties from Hessian diagonal (fallback).
    fn diagonal_uncertainties(&self, hessian: &DMatrix<f64>, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let hess_ii = hessian[(i, i)];
                let denom = hess_ii.abs().max(1e-12);
                1.0 / denom.sqrt()
            })
            .collect()
    }

    /// Run multiple independent fits in parallel using Rayon.
    ///
    /// Returns one `FitResult` per model.
    pub fn fit_batch<M: LogDensityModel + Sync>(&self, models: &[M]) -> Vec<Result<FitResult>> {
        use rayon::prelude::*;

        models.par_iter().map(|model| self.fit(model)).collect()
    }

    /// Full fit for a [`HistFactoryModel`], reusing the AD tape across all gradient
    /// calls (minimization + Hessian).
    ///
    /// For a model with N params this saves N+1+~30 tape allocations per fit.
    pub fn fit_histfactory(&self, model: &HistFactoryModel) -> Result<FitResult> {
        let mut tape = ns_ad::tape::Tape::with_capacity(model.n_params() * 20);
        self.fit_histfactory_with_tape(model, &mut tape)
    }

    /// Full fit for a [`HistFactoryModel`] with a caller-provided tape.
    pub fn fit_histfactory_with_tape(
        &self,
        model: &HistFactoryModel,
        tape: &mut ns_ad::tape::Tape,
    ) -> Result<FitResult> {
        let result = self.fit_minimum_histfactory_with_tape(model, tape)?;
        let bounds = model.parameter_bounds();
        let diag = diagnostics_from_opt(&result, &bounds);

        let hessian = self.compute_hessian_histfactory(model, &result.parameters, tape)?;
        let n = result.parameters.len();
        let diag_uncertainties = self.diagonal_uncertainties(&hessian, n);

        let fr = match self.invert_hessian(&hessian, n) {
            Some(covariance) => {
                let mut all_variances_ok = true;
                let mut uncertainties = Vec::with_capacity(n);
                for i in 0..n {
                    let var = covariance[(i, i)];
                    if var.is_finite() && var > 0.0 {
                        uncertainties.push(var.sqrt());
                    } else {
                        all_variances_ok = false;
                        uncertainties.push(diag_uncertainties[i]);
                    }
                }
                if all_variances_ok {
                    let cov_flat: Vec<f64> = covariance.iter().copied().collect();
                    FitResult::with_covariance(
                        result.parameters,
                        uncertainties,
                        cov_flat,
                        result.fval,
                        result.converged,
                        result.n_iter as usize,
                        result.n_fev,
                        result.n_gev,
                    )
                } else {
                    log::warn!("Invalid covariance diagonal; omitting covariance matrix");
                    FitResult::new(
                        result.parameters,
                        uncertainties,
                        result.fval,
                        result.converged,
                        result.n_iter as usize,
                        result.n_fev,
                        result.n_gev,
                    )
                }
            }
            None => {
                log::warn!("Hessian inversion failed, using diagonal approximation");
                FitResult::new(
                    result.parameters,
                    diag_uncertainties,
                    result.fval,
                    result.converged,
                    result.n_iter as usize,
                    result.n_fev,
                    result.n_gev,
                )
            }
        };
        Ok(apply_diagnostics(fr, diag))
    }

    /// Compute Hessian for a [`HistFactoryModel`], reusing a caller-provided tape.
    fn compute_hessian_histfactory(
        &self,
        model: &HistFactoryModel,
        best_params: &[f64],
        tape: &mut ns_ad::tape::Tape,
    ) -> Result<DMatrix<f64>> {
        let n = best_params.len();
        let grad_center = model.gradient_reverse_reuse(best_params, tape)?;

        let mut hessian = DMatrix::zeros(n, n);

        for j in 0..n {
            let eps = 1e-4 * best_params[j].abs().max(1.0);

            let mut params_plus = best_params.to_vec();
            params_plus[j] += eps;
            let grad_plus = model.gradient_reverse_reuse(&params_plus, tape)?;

            for i in 0..n {
                hessian[(i, j)] = (grad_plus[i] - grad_center[i]) / eps;
            }
        }

        // Symmetrise: H = (H + H^T) / 2
        let ht = hessian.transpose();
        hessian = (&hessian + &ht) * 0.5;

        Ok(hessian)
    }

    /// Generate toy pseudo-experiments and fit each one.
    ///
    /// Uses tape-reusing gradient path via Rayon `map_init`.
    ///
    /// # Arguments
    /// * `model` - Base model (used for expected data and parameter structure)
    /// * `params` - Parameters to generate toys at (e.g., best-fit or Asimov)
    /// * `n_toys` - Number of pseudo-experiments
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Returns
    /// Vector of fit results, one per toy
    pub fn fit_toys(
        &self,
        model: &HistFactoryModel,
        params: &[f64],
        n_toys: usize,
        seed: u64,
    ) -> Vec<Result<FitResult>> {
        use rayon::prelude::*;

        // Generate expected main data at given parameters (pyhf ordering)
        let expected = match model.expected_data_pyhf_main(params) {
            Ok(e) => e,
            Err(e) => return vec![Err(e)],
        };

        let tape_capacity = model.n_params() * 20;

        // Generate toy datasets in parallel with deterministic seeds.
        // map_init creates one Tape per Rayon worker thread.
        (0..n_toys)
            .into_par_iter()
            .map_init(
                || ns_ad::tape::Tape::with_capacity(tape_capacity),
                |tape, toy_idx| {
                    let toy_seed = seed.wrapping_add(toy_idx as u64);
                    let toy_data = crate::toys::poisson_main_from_expected(&expected, toy_seed);

                    let toy_model = model.with_observed_main(&toy_data)?;
                    self.fit_histfactory_with_tape(&toy_model, tape)
                },
            )
            .collect()
    }

    /// Compute ranking: impact of each nuisance parameter on POI.
    ///
    /// For each constrained NP, fixes it at ±1σ and re-fits (NLL-only,
    /// no Hessian). The shift in POI measures the NP's impact.
    ///
    /// Performance: uses bounds-clamping (no model clone), warm-start from
    /// nominal fit, and per-thread tape + NllScratch reuse.
    ///
    /// # Returns
    /// `RankingEntry` per constrained NP, sorted by |impact| descending.
    pub fn ranking(&self, model: &HistFactoryModel) -> Result<Vec<RankingEntry>> {
        use rayon::prelude::*;
        use std::collections::HashSet;

        let poi_idx = model
            .poi_index()
            .ok_or_else(|| ns_core::Error::Validation("No POI defined".to_string()))?;

        // Nominal fit WITH Hessian (need uncertainties for constraint = σ̂/σ).
        let nominal_result = self.fit_histfactory(model)?;
        let mu_hat = nominal_result.parameters[poi_idx];
        let base_bounds = model.parameter_bounds();

        // Ranking applies to nuisance parameters constrained by either:
        // - explicit Gaussian constraints (constraint_width.is_some()), OR
        // - auxiliary Poisson constraints (Barlow–Beeston ShapeSys).
        //
        // We intentionally *exclude* unconstrained normfactors/shapefactors: they have
        // neither a Gaussian width nor an aux-poisson sigma.
        let poisson_sigmas = model.poisson_constraint_sigmas();
        let mut np_set: HashSet<usize> = model
            .parameters()
            .iter()
            .enumerate()
            .filter(|(i, p)| *i != poi_idx && p.constraint_width.is_some())
            .map(|(i, _)| i)
            .collect();
        for &idx in poisson_sigmas.keys() {
            if idx != poi_idx {
                np_set.insert(idx);
            }
        }
        let mut np_indices: Vec<usize> = np_set.into_iter().collect();
        np_indices.sort_unstable();

        let tape_capacity = model.n_params() * 20;

        // Per-NP refits: NLL-only (no Hessian), warm-start, bounds-clamp.
        // map_init allocates one Tape + NllScratch per Rayon worker thread.
        let entries: Vec<Result<RankingEntry>> = np_indices
            .par_iter()
            .map_init(
                || (ns_ad::tape::Tape::with_capacity(tape_capacity), NllScratch::for_model(model)),
                |(tape, scratch), &np_idx| {
                    let param = &model.parameters()[np_idx];
                    let center = param.constraint_center.unwrap_or(param.init);
                    let sigma = param
                        .constraint_width
                        .or_else(|| poisson_sigmas.get(&np_idx).copied())
                        .unwrap_or(0.1);

                    // --- +1σ: fix NP via bounds-clamping (no model clone) ---
                    let (b_lo, b_hi) = base_bounds[np_idx];
                    let val_up = (center + sigma).min(b_hi);
                    let mut bounds_up = base_bounds.clone();
                    bounds_up[np_idx] = (val_up, val_up);
                    let mut warm = nominal_result.parameters.clone();
                    warm[np_idx] = val_up;

                    let result_up = self.fit_minimum_histfactory_from_with_bounds_reuse(
                        model, &warm, &bounds_up, tape, scratch,
                    )?;
                    let mu_up = result_up.parameters[poi_idx];

                    // --- -1σ ---
                    let val_down = (center - sigma).max(b_lo);
                    let mut bounds_down = base_bounds.clone();
                    bounds_down[np_idx] = (val_down, val_down);
                    warm = nominal_result.parameters.clone();
                    warm[np_idx] = val_down;

                    let result_down = self.fit_minimum_histfactory_from_with_bounds_reuse(
                        model,
                        &warm,
                        &bounds_down,
                        tape,
                        scratch,
                    )?;
                    let mu_down = result_down.parameters[poi_idx];

                    // Pull: (θ̂ - θ₀) / σ
                    let theta_hat = nominal_result.parameters[np_idx];
                    let pull = (theta_hat - center) / sigma;

                    // Constraint: σ̂ / σ (should be ≤ 1)
                    let constraint = nominal_result.uncertainties[np_idx] / sigma;

                    Ok(RankingEntry {
                        name: param.name.clone(),
                        delta_mu_up: mu_up - mu_hat,
                        delta_mu_down: mu_down - mu_hat,
                        pull,
                        constraint,
                    })
                },
            )
            .collect::<Vec<_>>();

        // Collect results; warn about failures instead of silently dropping them.
        let mut ranking: Vec<RankingEntry> = Vec::with_capacity(entries.len());
        let mut n_failed = 0u32;
        for entry in entries {
            match entry {
                Ok(e) => ranking.push(e),
                Err(e) => {
                    n_failed += 1;
                    log::warn!("Ranking NP refit failed: {e}");
                }
            }
        }
        if n_failed > 0 {
            log::warn!(
                "Ranking: {n_failed} NP(s) omitted due to refit failures (see warnings above)"
            );
        }

        ranking.sort_by(|a, b| {
            let impact_a = a.delta_mu_up.abs().max(a.delta_mu_down.abs());
            let impact_b = b.delta_mu_up.abs().max(b.delta_mu_down.abs());
            impact_b
                .partial_cmp(&impact_a)
                .unwrap_or(std::cmp::Ordering::Equal)
                // Tie-break for deterministic ordering (important for ML artifacts/logging).
                .then_with(|| a.name.cmp(&b.name))
        });

        Ok(ranking)
    }

    /// Discovery-style test statistic `q0 = 2*(NLL(mu=0) - NLL_hat)` and its gradient
    /// w.r.t. the nominal yields of a single sample (main bins).
    ///
    /// The gradient is computed for the profiled objective via the envelope theorem:
    /// it is the partial derivative at the profiled optima.
    ///
    /// The returned gradient has length equal to the number of bins in the specified channel.
    pub fn q0_like_loss_and_grad_sample_nominal(
        &self,
        model: &HistFactoryModel,
        channel_idx: usize,
        sample_idx: usize,
    ) -> Result<(f64, Vec<f64>)> {
        let poi = model
            .poi_index()
            .ok_or_else(|| ns_core::Error::Validation("No POI defined".to_string()))?;

        // Validate nominal-override semantics for this sample.
        model.validate_sample_nominal_override_linear_safe(channel_idx, sample_idx)?;

        let mut tape = ns_ad::tape::Tape::with_capacity(model.n_params() * 20);
        let mut scratch = NllScratch::for_model(model);

        // Free fit: unconditional MLE.
        let free = self.fit_minimum_histfactory_with_tape_reuse(model, &mut tape, &mut scratch)?;
        if !free.converged {
            return Err(ns_core::Error::Validation(format!(
                "Free fit did not converge: {}",
                free.message
            )));
        }
        let mu_hat = free.parameters[poi];
        let nll_hat = free.fval;

        // Conditional fit at mu=0 (bounds clamping; no model clone).
        let base_bounds = model.parameter_bounds();
        let mut bounds0 = base_bounds.clone();
        bounds0[poi] = (0.0, 0.0);
        let mut warm0 = free.parameters.clone();
        warm0[poi] = 0.0;

        let fixed0 = self.fit_minimum_histfactory_from_with_bounds_reuse(
            model,
            &warm0,
            &bounds0,
            &mut tape,
            &mut scratch,
        )?;
        if !fixed0.converged {
            return Err(ns_core::Error::Validation(format!(
                "Fixed fit (mu=0) did not converge: {}",
                fixed0.message
            )));
        }

        let llr = 2.0 * (fixed0.fval - nll_hat);
        let q0 = llr.max(0.0);

        // One-sided discovery test: if mu_hat < 0, q0 = 0 and gradient is 0.
        if mu_hat < 0.0 || q0 == 0.0 {
            let n_bins = model.channel_bin_count(channel_idx)?;
            return Ok((0.0, vec![0.0; n_bins]));
        }

        let prepared = model.prepare();
        let n_bins = model.channel_bin_count(channel_idx)?;
        let mut grad_free = vec![0.0; n_bins];
        let mut grad_fixed = vec![0.0; n_bins];

        prepared.grad_nll_wrt_sample_nominal_main_reuse(
            &free.parameters,
            &mut scratch,
            channel_idx,
            sample_idx,
            &mut grad_free,
        )?;
        prepared.grad_nll_wrt_sample_nominal_main_reuse(
            &fixed0.parameters,
            &mut scratch,
            channel_idx,
            sample_idx,
            &mut grad_fixed,
        )?;

        let mut grad = vec![0.0; n_bins];
        for i in 0..n_bins {
            grad[i] = 2.0 * (grad_fixed[i] - grad_free[i]);
        }

        Ok((q0, grad))
    }

    /// Upper-limit-style test statistic `q_mu` (pyhf `qtilde`) and its gradient
    /// w.r.t. the nominal yields of a single sample (main bins).
    ///
    /// This follows the pyhf convention for `q_mu`:
    /// - `q_mu = 2*(NLL(mu_test) - NLL_hat)`, clipped at 0
    /// - if `mu_hat > mu_test`, `q_mu = 0` (one-sided)
    pub fn qmu_like_loss_and_grad_sample_nominal(
        &self,
        model: &HistFactoryModel,
        mu_test: f64,
        channel_idx: usize,
        sample_idx: usize,
    ) -> Result<(f64, Vec<f64>)> {
        let poi = model
            .poi_index()
            .ok_or_else(|| ns_core::Error::Validation("No POI defined".to_string()))?;

        model.validate_sample_nominal_override_linear_safe(channel_idx, sample_idx)?;

        let mut tape = ns_ad::tape::Tape::with_capacity(model.n_params() * 20);
        let mut scratch = NllScratch::for_model(model);

        let free = self.fit_minimum_histfactory_with_tape_reuse(model, &mut tape, &mut scratch)?;
        if !free.converged {
            return Err(ns_core::Error::Validation(format!(
                "Free fit did not converge: {}",
                free.message
            )));
        }
        let mu_hat = free.parameters[poi];
        let nll_hat = free.fval;

        let base_bounds = model.parameter_bounds();
        let mut bounds = base_bounds.clone();
        bounds[poi] = (mu_test, mu_test);
        let mut warm = free.parameters.clone();
        warm[poi] = mu_test;

        let fixed = self.fit_minimum_histfactory_from_with_bounds_reuse(
            model,
            &warm,
            &bounds,
            &mut tape,
            &mut scratch,
        )?;
        if !fixed.converged {
            return Err(ns_core::Error::Validation(format!(
                "Fixed fit (mu={}) did not converge: {}",
                mu_test, fixed.message
            )));
        }

        let llr = 2.0 * (fixed.fval - nll_hat);
        let q_mu = llr.max(0.0);
        if mu_hat > mu_test || q_mu == 0.0 {
            let n_bins = model.channel_bin_count(channel_idx)?;
            return Ok((0.0, vec![0.0; n_bins]));
        }

        let prepared = model.prepare();
        let n_bins = model.channel_bin_count(channel_idx)?;
        let mut grad_free = vec![0.0; n_bins];
        let mut grad_fixed = vec![0.0; n_bins];

        prepared.grad_nll_wrt_sample_nominal_main_reuse(
            &free.parameters,
            &mut scratch,
            channel_idx,
            sample_idx,
            &mut grad_free,
        )?;
        prepared.grad_nll_wrt_sample_nominal_main_reuse(
            &fixed.parameters,
            &mut scratch,
            channel_idx,
            sample_idx,
            &mut grad_fixed,
        )?;

        let mut grad = vec![0.0; n_bins];
        for i in 0..n_bins {
            grad[i] = 2.0 * (grad_fixed[i] - grad_free[i]);
        }

        Ok((q_mu, grad))
    }

    // --- GPU-accelerated methods (requires `cuda` feature) ---

    /// GPU-accelerated NLL minimization (no Hessian).
    ///
    /// Creates a `GpuSession` internally. For repeated fits on the same model
    /// structure (e.g. profile scan, ranking), prefer using `GpuSession` directly
    /// via [`gpu_session::CudaGpuSession`].
    #[cfg(feature = "cuda")]
    pub fn fit_minimum_gpu(
        &self,
        model: &HistFactoryModel,
    ) -> Result<crate::optimizer::OptimizationResult> {
        let session = crate::gpu_session::cuda_session(model)?;
        session.fit_minimum(model, &self.config)
    }

    /// GPU-accelerated NLL minimization from an explicit starting point.
    #[cfg(feature = "cuda")]
    pub fn fit_minimum_gpu_from(
        &self,
        model: &HistFactoryModel,
        initial_params: &[f64],
    ) -> Result<crate::optimizer::OptimizationResult> {
        let session = crate::gpu_session::cuda_session(model)?;
        session.fit_minimum_from(model, initial_params, &self.config)
    }

    /// GPU-accelerated NLL minimization with warm-start + custom bounds.
    ///
    /// Enables fixed-parameter fits without cloning the model: clamp the bounds
    /// of the fixed parameter to `(value, value)`.
    #[cfg(feature = "cuda")]
    pub fn fit_minimum_gpu_from_with_bounds(
        &self,
        model: &HistFactoryModel,
        initial_params: &[f64],
        bounds: &[(f64, f64)],
    ) -> Result<crate::optimizer::OptimizationResult> {
        let session = crate::gpu_session::cuda_session(model)?;
        session.fit_minimum_from_with_bounds(model, initial_params, bounds, &self.config)
    }

    /// GPU-accelerated full fit with Hessian and uncertainties.
    ///
    /// Uses GPU for NLL minimization, then falls back to CPU for Hessian
    /// computation (finite differences of GPU NLL+gradient).
    #[cfg(feature = "cuda")]
    pub fn fit_gpu(&self, model: &HistFactoryModel) -> Result<FitResult> {
        let session = crate::gpu_session::cuda_session(model)?;
        let result = session.fit_minimum(model, &self.config)?;
        let bounds = model.parameter_bounds();
        let diag = diagnostics_from_opt(&result, &bounds);

        // Compute Hessian via finite differences of GPU gradient
        let n = result.parameters.len();
        let hessian = self.compute_hessian_gpu(&session, &result.parameters)?;
        let diag_uncertainties = self.diagonal_uncertainties(&hessian, n);

        let fr = match self.invert_hessian(&hessian, n) {
            Some(covariance) => {
                let mut all_variances_ok = true;
                let mut uncertainties = Vec::with_capacity(n);
                for i in 0..n {
                    let var = covariance[(i, i)];
                    if var.is_finite() && var > 0.0 {
                        uncertainties.push(var.sqrt());
                    } else {
                        all_variances_ok = false;
                        uncertainties.push(diag_uncertainties[i]);
                    }
                }
                if all_variances_ok {
                    let cov_flat: Vec<f64> = covariance.iter().copied().collect();
                    FitResult::with_covariance(
                        result.parameters,
                        uncertainties,
                        cov_flat,
                        result.fval,
                        result.converged,
                        result.n_iter as usize,
                        result.n_fev,
                        result.n_gev,
                    )
                } else {
                    log::warn!("GPU fit: invalid covariance diagonal; omitting covariance matrix");
                    FitResult::new(
                        result.parameters,
                        uncertainties,
                        result.fval,
                        result.converged,
                        result.n_iter as usize,
                        result.n_fev,
                        result.n_gev,
                    )
                }
            }
            None => {
                log::warn!("GPU fit: Hessian inversion failed, using diagonal approximation");
                FitResult::new(
                    result.parameters,
                    diag_uncertainties,
                    result.fval,
                    result.converged,
                    result.n_iter as usize,
                    result.n_fev,
                    result.n_gev,
                )
            }
        };
        Ok(apply_diagnostics(fr, diag))
    }

    /// GPU-accelerated full fit from an explicit starting point (warm-start),
    /// with Hessian-based uncertainties.
    ///
    /// Uses GPU for NLL minimization, then falls back to CPU for Hessian
    /// computation (finite differences of GPU NLL+gradient).
    #[cfg(feature = "cuda")]
    pub fn fit_gpu_from(
        &self,
        model: &HistFactoryModel,
        initial_params: &[f64],
    ) -> Result<FitResult> {
        let session = crate::gpu_session::cuda_session(model)?;
        let result = session.fit_minimum_from(model, initial_params, &self.config)?;
        let bounds = model.parameter_bounds();
        let diag = diagnostics_from_opt(&result, &bounds);

        let n = result.parameters.len();
        let hessian = self.compute_hessian_gpu(&session, &result.parameters)?;
        let diag_uncertainties = self.diagonal_uncertainties(&hessian, n);

        let fr = match self.invert_hessian(&hessian, n) {
            Some(covariance) => {
                let mut all_variances_ok = true;
                let mut uncertainties = Vec::with_capacity(n);
                for i in 0..n {
                    let var = covariance[(i, i)];
                    if var.is_finite() && var > 0.0 {
                        uncertainties.push(var.sqrt());
                    } else {
                        all_variances_ok = false;
                        uncertainties.push(diag_uncertainties[i]);
                    }
                }
                if all_variances_ok {
                    let cov_flat: Vec<f64> = covariance.iter().copied().collect();
                    FitResult::with_covariance(
                        result.parameters,
                        uncertainties,
                        cov_flat,
                        result.fval,
                        result.converged,
                        result.n_iter as usize,
                        result.n_fev,
                        result.n_gev,
                    )
                } else {
                    log::warn!(
                        "GPU fit (warm-start): invalid covariance diagonal; omitting covariance matrix"
                    );
                    FitResult::new(
                        result.parameters,
                        uncertainties,
                        result.fval,
                        result.converged,
                        result.n_iter as usize,
                        result.n_fev,
                        result.n_gev,
                    )
                }
            }
            None => {
                log::warn!(
                    "GPU fit (warm-start): Hessian inversion failed, using diagonal approximation"
                );
                FitResult::new(
                    result.parameters,
                    diag_uncertainties,
                    result.fval,
                    result.converged,
                    result.n_iter as usize,
                    result.n_fev,
                    result.n_gev,
                )
            }
        };
        Ok(apply_diagnostics(fr, diag))
    }

    /// Compute Hessian via finite differences of GPU gradient.
    #[cfg(feature = "cuda")]
    fn compute_hessian_gpu(
        &self,
        session: &crate::gpu_session::CudaGpuSession,
        best_params: &[f64],
    ) -> Result<DMatrix<f64>> {
        let n = best_params.len();
        let (_, grad_center) = session.nll_grad(best_params)?;

        let mut hessian = DMatrix::zeros(n, n);

        for j in 0..n {
            let eps = 1e-4 * best_params[j].abs().max(1.0);

            let mut params_plus = best_params.to_vec();
            params_plus[j] += eps;
            let (_, grad_plus) = session.nll_grad(&params_plus)?;

            for i in 0..n {
                hessian[(i, j)] = (grad_plus[i] - grad_center[i]) / eps;
            }
        }

        // Symmetrise: H = (H + H^T) / 2
        let ht = hessian.transpose();
        hessian = (&hessian + &ht) * 0.5;

        Ok(hessian)
    }

    // --- Metal GPU-accelerated methods (requires `metal` feature) ---

    /// Metal GPU-accelerated NLL minimization (no Hessian).
    ///
    /// Convergence tolerance is clamped to at least `1e-3` for f32 precision.
    #[cfg(feature = "metal")]
    pub fn fit_minimum_metal(
        &self,
        model: &HistFactoryModel,
    ) -> Result<crate::optimizer::OptimizationResult> {
        let session = crate::gpu_session::metal_session(model)?;
        let mut config = self.config.clone();
        config.tol = config.tol.max(1e-3);
        session.fit_minimum(model, &config)
    }

    /// Metal GPU-accelerated NLL minimization from an explicit starting point.
    #[cfg(feature = "metal")]
    pub fn fit_minimum_metal_from(
        &self,
        model: &HistFactoryModel,
        initial_params: &[f64],
    ) -> Result<crate::optimizer::OptimizationResult> {
        let session = crate::gpu_session::metal_session(model)?;
        let mut config = self.config.clone();
        config.tol = config.tol.max(1e-3);
        session.fit_minimum_from(model, initial_params, &config)
    }

    /// Metal GPU-accelerated NLL minimization with warm-start + custom bounds.
    #[cfg(feature = "metal")]
    pub fn fit_minimum_metal_from_with_bounds(
        &self,
        model: &HistFactoryModel,
        initial_params: &[f64],
        bounds: &[(f64, f64)],
    ) -> Result<crate::optimizer::OptimizationResult> {
        let session = crate::gpu_session::metal_session(model)?;
        let mut config = self.config.clone();
        config.tol = config.tol.max(1e-3);
        session.fit_minimum_from_with_bounds(model, initial_params, bounds, &config)
    }

    /// Metal GPU-accelerated full fit with Hessian-based uncertainties.
    ///
    /// Uses Metal GPU for NLL minimization, then falls back to CPU for Hessian
    /// computation (finite differences of GPU gradient).
    #[cfg(feature = "metal")]
    pub fn fit_metal(&self, model: &HistFactoryModel) -> Result<FitResult> {
        let session = crate::gpu_session::metal_session(model)?;
        let mut config = self.config.clone();
        config.tol = config.tol.max(1e-3);

        let result = session.fit_minimum(model, &config)?;
        let bounds = model.parameter_bounds();
        let diag = diagnostics_from_opt(&result, &bounds);

        let n = result.parameters.len();
        let hessian = self.compute_hessian_metal(&session, &result.parameters)?;
        let diag_uncertainties = self.diagonal_uncertainties(&hessian, n);

        let fr = match self.invert_hessian(&hessian, n) {
            Some(covariance) => {
                let mut all_variances_ok = true;
                let mut uncertainties = Vec::with_capacity(n);
                for i in 0..n {
                    let var = covariance[(i, i)];
                    if var.is_finite() && var > 0.0 {
                        uncertainties.push(var.sqrt());
                    } else {
                        all_variances_ok = false;
                        uncertainties.push(diag_uncertainties[i]);
                    }
                }
                if all_variances_ok {
                    let cov_flat: Vec<f64> = covariance.iter().copied().collect();
                    FitResult::with_covariance(
                        result.parameters,
                        uncertainties,
                        cov_flat,
                        result.fval,
                        result.converged,
                        result.n_iter as usize,
                        result.n_fev,
                        result.n_gev,
                    )
                } else {
                    log::warn!(
                        "Metal fit: invalid covariance diagonal; omitting covariance matrix"
                    );
                    FitResult::new(
                        result.parameters,
                        uncertainties,
                        result.fval,
                        result.converged,
                        result.n_iter as usize,
                        result.n_fev,
                        result.n_gev,
                    )
                }
            }
            None => {
                log::warn!("Metal fit: Hessian inversion failed, using diagonal approximation");
                FitResult::new(
                    result.parameters,
                    diag_uncertainties,
                    result.fval,
                    result.converged,
                    result.n_iter as usize,
                    result.n_fev,
                    result.n_gev,
                )
            }
        };
        Ok(apply_diagnostics(fr, diag))
    }

    /// Metal GPU-accelerated full fit from an explicit starting point (warm-start),
    /// with Hessian-based uncertainties.
    #[cfg(feature = "metal")]
    pub fn fit_metal_from(
        &self,
        model: &HistFactoryModel,
        initial_params: &[f64],
    ) -> Result<FitResult> {
        let session = crate::gpu_session::metal_session(model)?;
        let mut config = self.config.clone();
        config.tol = config.tol.max(1e-3);

        let result = session.fit_minimum_from(model, initial_params, &config)?;
        let bounds = model.parameter_bounds();
        let diag = diagnostics_from_opt(&result, &bounds);

        let n = result.parameters.len();
        let hessian = self.compute_hessian_metal(&session, &result.parameters)?;
        let diag_uncertainties = self.diagonal_uncertainties(&hessian, n);

        let fr = match self.invert_hessian(&hessian, n) {
            Some(covariance) => {
                let mut all_variances_ok = true;
                let mut uncertainties = Vec::with_capacity(n);
                for i in 0..n {
                    let var = covariance[(i, i)];
                    if var.is_finite() && var > 0.0 {
                        uncertainties.push(var.sqrt());
                    } else {
                        all_variances_ok = false;
                        uncertainties.push(diag_uncertainties[i]);
                    }
                }
                if all_variances_ok {
                    let cov_flat: Vec<f64> = covariance.iter().copied().collect();
                    FitResult::with_covariance(
                        result.parameters,
                        uncertainties,
                        cov_flat,
                        result.fval,
                        result.converged,
                        result.n_iter as usize,
                        result.n_fev,
                        result.n_gev,
                    )
                } else {
                    log::warn!(
                        "Metal fit (warm-start): invalid covariance diagonal; omitting covariance matrix"
                    );
                    FitResult::new(
                        result.parameters,
                        uncertainties,
                        result.fval,
                        result.converged,
                        result.n_iter as usize,
                        result.n_fev,
                        result.n_gev,
                    )
                }
            }
            None => {
                log::warn!(
                    "Metal fit (warm-start): Hessian inversion failed, using diagonal approximation"
                );
                FitResult::new(
                    result.parameters,
                    diag_uncertainties,
                    result.fval,
                    result.converged,
                    result.n_iter as usize,
                    result.n_fev,
                    result.n_gev,
                )
            }
        };
        Ok(apply_diagnostics(fr, diag))
    }

    /// Compute Hessian via finite differences of Metal GPU gradient.
    #[cfg(feature = "metal")]
    fn compute_hessian_metal(
        &self,
        session: &crate::gpu_session::MetalGpuSession,
        best_params: &[f64],
    ) -> Result<DMatrix<f64>> {
        let n = best_params.len();
        let (_, grad_center) = session.nll_grad(best_params)?;

        let mut hessian = DMatrix::zeros(n, n);

        for j in 0..n {
            let eps = 1e-4 * best_params[j].abs().max(1.0);

            let mut params_plus = best_params.to_vec();
            params_plus[j] += eps;
            let (_, grad_plus) = session.nll_grad(&params_plus)?;

            for i in 0..n {
                hessian[(i, j)] = (grad_plus[i] - grad_center[i]) / eps;
            }
        }

        let ht = hessian.transpose();
        hessian = (&hessian + &ht) * 0.5;

        Ok(hessian)
    }
}

/// Entry in ranking plot: impact of a nuisance parameter on the POI.
#[must_use]
#[derive(Debug, Clone)]
pub struct RankingEntry {
    /// Parameter name
    pub name: String,
    /// POI shift when NP fixed at +1σ
    pub delta_mu_up: f64,
    /// POI shift when NP fixed at -1σ
    pub delta_mu_down: f64,
    /// Pull: (θ̂ − θ₀) / σ
    pub pull: f64,
    /// Constraint: σ̂ / σ
    pub constraint: f64,
}

impl Default for MaximumLikelihoodEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GPU-accelerated ranking
// ---------------------------------------------------------------------------

/// Compute NP ranking using GPU for per-NP refits.
///
/// Nominal fit uses CPU (needs Hessian for pull/constraint). Per-NP ±1σ refits
/// use a shared `GpuSession` with warm-start and bounds-clamping — sequential
/// but each refit is GPU-accelerated.
#[cfg(feature = "cuda")]
pub fn ranking_gpu(
    mle: &MaximumLikelihoodEstimator,
    model: &HistFactoryModel,
) -> Result<Vec<RankingEntry>> {
    use std::collections::HashSet;

    let poi_idx = model
        .poi_index()
        .ok_or_else(|| ns_core::Error::Validation("No POI defined".to_string()))?;

    // Nominal fit WITH Hessian (CPU — need uncertainties for constraint)
    let nominal_result = mle.fit_histfactory(model)?;
    let mu_hat = nominal_result.parameters[poi_idx];
    let base_bounds = model.parameter_bounds();

    // Ranking applies to nuisance parameters constrained by either:
    // - explicit Gaussian constraints (constraint_width.is_some()), OR
    // - auxiliary Poisson constraints (Barlow–Beeston ShapeSys).
    //
    // Keep this selection aligned with the CPU ranking implementation.
    let poisson_sigmas = model.poisson_constraint_sigmas();
    let mut np_set: HashSet<usize> = model
        .parameters()
        .iter()
        .enumerate()
        .filter(|(i, p)| *i != poi_idx && p.constraint_width.is_some())
        .map(|(i, _)| i)
        .collect();
    for &idx in poisson_sigmas.keys() {
        if idx != poi_idx {
            np_set.insert(idx);
        }
    }
    let mut np_indices: Vec<usize> = np_set.into_iter().collect();
    np_indices.sort_unstable();

    // Shared GPU session — model uploaded once
    let session = crate::gpu_session::cuda_session(model)?;
    let config = mle.config().clone();

    let mut entries = Vec::with_capacity(np_indices.len());

    for &np_idx in &np_indices {
        let param = &model.parameters()[np_idx];
        let center = param.constraint_center.unwrap_or(param.init);
        let sigma =
            param.constraint_width.or_else(|| poisson_sigmas.get(&np_idx).copied()).unwrap_or(0.1);

        // +1σ: fix NP via bounds-clamping
        let (b_lo, b_hi) = base_bounds[np_idx];
        let val_up = (center + sigma).min(b_hi);
        let mut bounds_up = base_bounds.clone();
        bounds_up[np_idx] = (val_up, val_up);
        let mut warm = nominal_result.parameters.clone();
        warm[np_idx] = val_up;

        let result_up = session.fit_minimum_from_with_bounds(model, &warm, &bounds_up, &config)?;
        let mu_up = result_up.parameters[poi_idx];

        // -1σ
        let val_down = (center - sigma).max(b_lo);
        let mut bounds_down = base_bounds.clone();
        bounds_down[np_idx] = (val_down, val_down);
        warm = nominal_result.parameters.clone();
        warm[np_idx] = val_down;

        let result_down =
            session.fit_minimum_from_with_bounds(model, &warm, &bounds_down, &config)?;
        let mu_down = result_down.parameters[poi_idx];

        // Pull and constraint
        let theta_hat = nominal_result.parameters[np_idx];
        let pull = (theta_hat - center) / sigma;
        let constraint = nominal_result.uncertainties[np_idx] / sigma;

        entries.push(RankingEntry {
            name: param.name.clone(),
            delta_mu_up: mu_up - mu_hat,
            delta_mu_down: mu_down - mu_hat,
            pull,
            constraint,
        });
    }

    // Sort by |impact| descending
    entries.sort_by(|a, b| {
        let impact_a = a.delta_mu_up.abs().max(a.delta_mu_down.abs());
        let impact_b = b.delta_mu_up.abs().max(b.delta_mu_down.abs());
        impact_b
            .partial_cmp(&impact_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.name.cmp(&b.name))
    });

    Ok(entries)
}

/// Compute NP ranking using Metal GPU for per-NP refits (hybrid CPU+Metal).
///
/// Nominal fit uses CPU (needs Hessian for pull/constraint). Per-NP ±1σ refits
/// use a shared `MetalGpuSession` with warm-start and bounds-clamping — sequential
/// but each refit is GPU-accelerated in f32.
#[cfg(feature = "metal")]
pub fn ranking_metal(
    mle: &MaximumLikelihoodEstimator,
    model: &HistFactoryModel,
) -> Result<Vec<RankingEntry>> {
    use std::collections::HashSet;

    let poi_idx = model
        .poi_index()
        .ok_or_else(|| ns_core::Error::Validation("No POI defined".to_string()))?;

    // Nominal fit WITH Hessian (CPU — need uncertainties for constraint)
    let nominal_result = mle.fit_histfactory(model)?;
    let mu_hat = nominal_result.parameters[poi_idx];
    let base_bounds = model.parameter_bounds();

    // Keep NP selection aligned with CPU ranking.
    let poisson_sigmas = model.poisson_constraint_sigmas();
    let mut np_set: HashSet<usize> = model
        .parameters()
        .iter()
        .enumerate()
        .filter(|(i, p)| *i != poi_idx && p.constraint_width.is_some())
        .map(|(i, _)| i)
        .collect();
    for &idx in poisson_sigmas.keys() {
        if idx != poi_idx {
            np_set.insert(idx);
        }
    }
    let mut np_indices: Vec<usize> = np_set.into_iter().collect();
    np_indices.sort_unstable();

    // Shared Metal GPU session — model uploaded once.
    let session = crate::gpu_session::metal_session(model)?;

    // f32 ranking refits: use slightly stricter tol than full Metal fit by default.
    // Warm-started, fixed-NP refits are typically easier than the nominal fit.
    let mut config = mle.config().clone();
    config.tol = config.tol.max(5e-4);

    let mut entries = Vec::with_capacity(np_indices.len());

    for &np_idx in &np_indices {
        let param = &model.parameters()[np_idx];
        let center = param.constraint_center.unwrap_or(param.init);
        let sigma =
            param.constraint_width.or_else(|| poisson_sigmas.get(&np_idx).copied()).unwrap_or(0.1);

        // +1σ: fix NP via bounds-clamping
        let (b_lo, b_hi) = base_bounds[np_idx];
        let val_up = (center + sigma).min(b_hi);
        let mut bounds_up = base_bounds.clone();
        bounds_up[np_idx] = (val_up, val_up);
        let mut warm = nominal_result.parameters.clone();
        warm[np_idx] = val_up;

        let result_up = session.fit_minimum_from_with_bounds(model, &warm, &bounds_up, &config)?;
        let mu_up = result_up.parameters[poi_idx];

        // -1σ
        let val_down = (center - sigma).max(b_lo);
        let mut bounds_down = base_bounds.clone();
        bounds_down[np_idx] = (val_down, val_down);
        warm = nominal_result.parameters.clone();
        warm[np_idx] = val_down;

        let result_down =
            session.fit_minimum_from_with_bounds(model, &warm, &bounds_down, &config)?;
        let mu_down = result_down.parameters[poi_idx];

        // Pull and constraint from CPU nominal fit.
        let theta_hat = nominal_result.parameters[np_idx];
        let pull = (theta_hat - center) / sigma;
        let constraint = nominal_result.uncertainties[np_idx] / sigma;

        entries.push(RankingEntry {
            name: param.name.clone(),
            delta_mu_up: mu_up - mu_hat,
            delta_mu_down: mu_down - mu_hat,
            pull,
            constraint,
        });
    }

    // Sort by |impact| descending (deterministic tie-break).
    entries.sort_by(|a, b| {
        let impact_a = a.delta_mu_up.abs().max(a.delta_mu_down.abs());
        let impact_b = b.delta_mu_up.abs().max(b.delta_mu_down.abs());
        impact_b
            .partial_cmp(&impact_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.name.cmp(&b.name))
    });

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ns_translate::pyhf::Workspace;

    fn load_simple_workspace() -> Workspace {
        let json = include_str!("../../../tests/fixtures/simple_workspace.json");
        serde_json::from_str(json).unwrap()
    }

    fn load_model_from_fixture(name: &str) -> HistFactoryModel {
        use std::path::PathBuf;
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures").join(name);
        let json = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()));
        let ws: Workspace = serde_json::from_str(&json).unwrap();
        HistFactoryModel::from_workspace(&ws).unwrap()
    }

    #[test]
    fn test_mle_fit_simple() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();

        let mle = MaximumLikelihoodEstimator::new();
        let result = mle.fit(&model).unwrap();

        println!("Fit result:");
        println!("  Parameters: {:?}", result.parameters);
        println!("  Uncertainties: {:?}", result.uncertainties);
        println!("  NLL: {:.6}", result.nll);
        println!("  Converged: {}", result.converged);
        println!("  Iterations: {}", result.n_iter);

        assert!(result.converged, "Fit should converge");

        let poi = result.parameters[0];
        assert!(poi > 0.0 && poi < 2.0, "POI should be reasonable: {}", poi);

        for (i, &unc) in result.uncertainties.iter().enumerate() {
            assert!(unc > 0.0, "Uncertainty[{}] should be positive: {}", i, unc);
            assert!(unc.is_finite(), "Uncertainty[{}] should be finite: {}", i, unc);
            assert!(
                unc < 1e5,
                "Uncertainty[{}] looks like a numerical fallback (too large): {}",
                i,
                unc
            );
        }

        assert!(result.nll > 0.0 && result.nll < 100.0);
    }

    #[test]
    fn test_mle_poi_extraction() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();

        let mle = MaximumLikelihoodEstimator::new();
        let result = mle.fit(&model).unwrap();

        let poi_idx = model.poi_index().expect("POI index should exist");
        assert_eq!(poi_idx, 0, "POI should be first parameter");

        let poi = result.parameters[poi_idx];
        let poi_unc = result.uncertainties[poi_idx];

        println!("Best-fit POI: {} ± {}", poi, poi_unc);

        assert!(poi > 0.0 && poi < 2.0);
        assert!(poi_unc > 0.0 && poi_unc < 1.0);
    }

    #[test]
    fn test_mle_covariance_matrix() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();

        let mle = MaximumLikelihoodEstimator::new();
        let result = mle.fit(&model).unwrap();

        let n = result.parameters.len();

        // Covariance matrix should be present
        assert!(result.covariance.is_some(), "Covariance matrix should be computed");
        let cov = result.covariance.as_ref().unwrap();
        assert_eq!(cov.len(), n * n, "Covariance should be N×N");

        println!("Covariance matrix ({}×{}):", n, n);
        for i in 0..n {
            let row: Vec<String> = (0..n).map(|j| format!("{:+.4e}", cov[i * n + j])).collect();
            println!("  [{}]", row.join(", "));
        }

        // Diagonal elements should be positive (variances)
        for i in 0..n {
            let var_i = cov[i * n + i];
            assert!(var_i > 0.0, "Variance[{}] should be positive: {}", i, var_i);
        }

        // Diagonal should match uncertainties^2
        for i in 0..n {
            let var_from_cov = cov[i * n + i];
            let unc_sq = result.uncertainties[i].powi(2);
            let rel_diff = ((var_from_cov - unc_sq) / unc_sq).abs();
            assert!(
                rel_diff < 1e-10,
                "Cov diagonal[{}] should match uncertainty²: cov={}, unc²={}",
                i,
                var_from_cov,
                unc_sq
            );
        }

        // Matrix should be approximately symmetric (it is by construction)
        for i in 0..n {
            for j in 0..n {
                let diff = (cov[i * n + j] - cov[j * n + i]).abs();
                assert!(diff < 1e-15, "Covariance should be symmetric: [{},{}]", i, j);
            }
        }
    }

    #[test]
    fn test_mle_correlations() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();

        let mle = MaximumLikelihoodEstimator::new();
        let result = mle.fit(&model).unwrap();

        let n = result.parameters.len();

        println!("Correlation matrix:");
        for i in 0..n {
            let row: Vec<String> = (0..n)
                .map(|j| {
                    let rho = result.correlation(i, j).unwrap();
                    format!("{:+.4}", rho)
                })
                .collect();
            println!("  [{}]", row.join(", "));
        }

        // Diagonal correlations should be 1.0
        for i in 0..n {
            let rho_ii = result.correlation(i, i).unwrap();
            assert!(
                (rho_ii - 1.0).abs() < 1e-10,
                "Diagonal correlation[{}] should be 1.0: {}",
                i,
                rho_ii
            );
        }

        // All correlations should be in [-1, 1]
        for i in 0..n {
            for j in 0..n {
                let rho = result.correlation(i, j).unwrap();
                assert!(
                    (-1.0 - 1e-10..=1.0 + 1e-10).contains(&rho),
                    "Correlation[{},{}] out of range: {}",
                    i,
                    j,
                    rho
                );
            }
        }
    }

    #[test]
    fn test_hessian_computation() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();

        let mle = MaximumLikelihoodEstimator::new();
        let result = mle.fit(&model).unwrap();

        // Compute Hessian at best-fit point
        let hessian = mle.compute_hessian(&model, &result.parameters).unwrap();
        let n = result.parameters.len();

        println!("Hessian matrix ({}×{}):", n, n);
        for i in 0..n {
            let row: Vec<String> = (0..n).map(|j| format!("{:+.4e}", hessian[(i, j)])).collect();
            println!("  [{}]", row.join(", "));
        }

        // Hessian should be symmetric
        for i in 0..n {
            for j in i + 1..n {
                let diff = (hessian[(i, j)] - hessian[(j, i)]).abs();
                let scale = hessian[(i, j)].abs().max(1e-10);
                assert!(
                    diff / scale < 1e-6,
                    "Hessian not symmetric: H[{},{}]={}, H[{},{}]={}",
                    i,
                    j,
                    hessian[(i, j)],
                    j,
                    i,
                    hessian[(j, i)]
                );
            }
        }

        // Diagonal should be positive (at a minimum, Hessian is positive definite)
        for i in 0..n {
            assert!(
                hessian[(i, i)] > 0.0,
                "Hessian diagonal[{}] should be positive: {}",
                i,
                hessian[(i, i)]
            );
        }
    }

    #[test]
    #[ignore = "slow; run with `cargo test -p ns-inference test_fit_toys -- --ignored`"]
    fn test_fit_toys() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();

        let mle = MaximumLikelihoodEstimator::new();
        let nominal = mle.fit(&model).unwrap();

        // Run 20 toy fits
        let n_toys = 20;
        let results = mle.fit_toys(&model, &nominal.parameters, n_toys, 42);

        assert_eq!(results.len(), n_toys);

        let mut converged = 0;
        let mut poi_values = Vec::new();
        for result in &results {
            match result {
                Ok(r) => {
                    if r.converged {
                        converged += 1;
                        poi_values.push(r.parameters[0]);
                    }
                }
                Err(e) => println!("Toy failed: {}", e),
            }
        }

        println!("Toys: {}/{} converged", converged, n_toys);
        println!("POI values: {:?}", poi_values);

        // Most toys should converge
        assert!(converged >= n_toys / 2, "Too few toys converged: {}/{}", converged, n_toys);

        // POI should scatter around nominal value
        let poi_mean: f64 = poi_values.iter().sum::<f64>() / poi_values.len() as f64;
        println!("Mean POI: {:.4} (nominal: {:.4})", poi_mean, nominal.parameters[0]);
    }

    #[test]
    #[ignore = "slow; run with `cargo test -p ns-inference test_fit_toys_reproducible -- --ignored`"]
    fn test_fit_toys_reproducible() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();
        let params: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();

        let mle = MaximumLikelihoodEstimator::new();

        // Same seed => same results
        let results1 = mle.fit_toys(&model, &params, 5, 123);
        let results2 = mle.fit_toys(&model, &params, 5, 123);

        for (r1, r2) in results1.iter().zip(results2.iter()) {
            if let (Ok(a), Ok(b)) = (r1, r2) {
                assert_eq!(a.parameters, b.parameters, "Toys should be reproducible");
            }
        }
    }

    #[test]
    fn test_fit_toys_smoke_fast_and_reproducible() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();
        let params: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();

        let mle = MaximumLikelihoodEstimator::new();

        // Keep this test fast: 2 toys only.
        let results1 = mle.fit_toys(&model, &params, 2, 123);
        let results2 = mle.fit_toys(&model, &params, 2, 123);

        assert_eq!(results1.len(), 2);
        assert_eq!(results2.len(), 2);

        for (r1, r2) in results1.iter().zip(results2.iter()) {
            match (r1, r2) {
                (Ok(a), Ok(b)) => {
                    assert_eq!(a.parameters, b.parameters, "Toy best-fits should be reproducible");
                    assert!(a.nll.is_finite(), "Toy NLL should be finite");
                    assert!(b.nll.is_finite(), "Toy NLL should be finite");
                }
                (Err(e1), Err(e2)) => {
                    // If both fail, at least ensure deterministic failure mode.
                    assert_eq!(e1.to_string(), e2.to_string());
                }
                _ => panic!("Toy results should be deterministically Ok/Err for a fixed seed"),
            }
        }
    }

    #[test]
    #[ignore = "very slow (~10min release, argmin+tape AD); run with `cargo test -p ns-inference --release test_fit_toys_pull_distribution -- --ignored`"]
    fn test_fit_toys_pull_distribution() {
        use rand::SeedableRng;
        use rand_distr::{Distribution, Poisson};

        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();

        let poi_idx = model.poi_index().expect("POI index should exist");
        let mu_true = 1.0;

        // Generate at POI = mu_true, nuisances at suggested init
        let mut truth: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();
        truth[poi_idx] = mu_true;

        let expected = model.expected_data(&truth).unwrap();

        // Cap iterations to bound runtime on pathological toys
        let config = OptimizerConfig { max_iter: 50, ..OptimizerConfig::default() };
        let mle = MaximumLikelihoodEstimator::with_config(config);
        let n_toys = 100;
        let seed = 0u64;

        let mut pulls = Vec::new();
        let mut n_converged = 0usize;
        let mut n_covered = 0usize;

        for toy_idx in 0..n_toys {
            let toy_seed = seed.wrapping_add(toy_idx as u64);
            let mut rng = rand::rngs::StdRng::seed_from_u64(toy_seed);

            let toy_data: Vec<f64> = expected
                .iter()
                .map(|&lam| {
                    let pois = Poisson::new(lam.max(1e-10)).unwrap();
                    pois.sample(&mut rng)
                })
                .collect();

            let toy_model = match model.with_observed_main(&toy_data) {
                Ok(m) => m,
                Err(_) => continue,
            };

            let fit = match mle.fit(&toy_model) {
                Ok(f) => f,
                Err(_) => continue,
            };

            if !fit.converged {
                continue;
            }
            n_converged += 1;
            let mu_hat = fit.parameters[poi_idx];
            let sigma_mu = fit.uncertainties[poi_idx];
            if sigma_mu <= 0.0 || !sigma_mu.is_finite() {
                continue;
            }
            let pull = (mu_hat - mu_true) / sigma_mu;
            pulls.push(pull);
            if pull.abs() <= 1.0 {
                n_covered += 1;
            }
        }

        let n = pulls.len() as f64;
        assert!(n >= 50.0, "Need at least 50 converged toys, got {}", n as usize);

        let mean: f64 = pulls.iter().sum::<f64>() / n;
        let var: f64 = pulls.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std = var.sqrt();
        let coverage = n_covered as f64 / pulls.len() as f64;

        // Print JSON summary for CI capture
        println!(
            "{{\"test\":\"pull_distribution\",\"n_toys\":{},\"n_converged\":{},\"n_pulls\":{},\
             \"pull_mean\":{:.4},\"pull_std\":{:.4},\"coverage_1sigma\":{:.4}}}",
            n_toys,
            n_converged,
            pulls.len(),
            mean,
            std,
            coverage
        );

        assert!(mean.abs() < 0.15, "Pull mean should be near 0: {:.4}", mean);
        assert!((std - 1.0).abs() < 0.15, "Pull std should be near 1: {:.4}", std);
        assert!((coverage - 0.68).abs() < 0.08, "1σ coverage should be near 68%: {:.4}", coverage);
    }

    #[test]
    fn test_ranking_contract() {
        let json = include_str!("../../../tests/fixtures/complex_workspace.json");
        let workspace: Workspace = serde_json::from_str(json).unwrap();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();
        let poi_idx = model.poi_index().unwrap();

        // Count expected: only constrained NPs (constraint_width.is_some()), skip POI
        let expected_np_names: Vec<&str> = model
            .parameters()
            .iter()
            .enumerate()
            .filter(|(i, p)| *i != poi_idx && p.constraint_width.is_some())
            .map(|(_, p)| p.name.as_str())
            .collect();

        let mle = MaximumLikelihoodEstimator::new();
        let ranking = mle.ranking(&model).unwrap();

        println!("Ranking ({} NPs):", ranking.len());
        for entry in &ranking {
            println!(
                "  {}: Δμ_up={:+.4}, Δμ_down={:+.4}, pull={:+.3}, constraint={:.3}",
                entry.name, entry.delta_mu_up, entry.delta_mu_down, entry.pull, entry.constraint
            );
        }

        // Contract 1: exactly one entry per constrained NP
        assert_eq!(
            ranking.len(),
            expected_np_names.len(),
            "ranking should have exactly one entry per constrained NP (expected {:?})",
            expected_np_names
        );

        // Contract 2: every entry name must be a constrained NP
        for entry in &ranking {
            assert!(
                expected_np_names.contains(&entry.name.as_str()),
                "ranking entry '{}' is not a constrained NP",
                entry.name
            );
        }

        // Contract 3: all values are finite and constraints positive
        for entry in &ranking {
            assert!(entry.delta_mu_up.is_finite(), "delta_mu_up should be finite");
            assert!(entry.delta_mu_down.is_finite(), "delta_mu_down should be finite");
            assert!(entry.pull.is_finite(), "pull should be finite");
            assert!(entry.constraint.is_finite(), "constraint should be finite");
            assert!(entry.constraint > 0.0, "constraint should be positive");
        }

        // Contract 4: sorted by descending |impact|
        for w in ranking.windows(2) {
            let impact_a = w[0].delta_mu_up.abs().max(w[0].delta_mu_down.abs());
            let impact_b = w[1].delta_mu_up.abs().max(w[1].delta_mu_down.abs());
            assert!(
                impact_a >= impact_b - 1e-15,
                "ranking should be sorted by |impact|: {} < {}",
                impact_a,
                impact_b
            );
        }
    }

    #[test]
    fn test_diagonal_uncertainties_abs_diag() {
        let mle = MaximumLikelihoodEstimator::new();
        let h = DMatrix::<f64>::from_diagonal(&nalgebra::DVector::from_vec(vec![-4.0, 9.0]));
        let u = mle.diagonal_uncertainties(&h, 2);
        assert!((u[0] - 0.5).abs() < 1e-12);
        assert!((u[1] - (1.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn test_fixed_poi_zero_converges_at_boundary() {
        // Fix POI=0 by setting bounds to (0,0).
        // The unconstrained MLE has mu > 0, so the optimizer must stop at the boundary
        // and report SolverConverged, not MaxIter.
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();
        let poi_idx = model.poi_index().expect("POI index should exist");

        let mle = MaximumLikelihoodEstimator::new();

        // Build bounds with POI clamped to 0
        let mut bounds: Vec<(f64, f64)> = model.parameter_bounds();
        bounds[poi_idx] = (0.0, 0.0);

        let mut init: Vec<f64> = model.parameter_init();
        init[poi_idx] = 0.0;

        let mut tape = ns_ad::tape::Tape::with_capacity(model.n_params() * 20);
        let result = mle
            .fit_minimum_histfactory_from_with_bounds_with_tape(&model, &init, &bounds, &mut tape)
            .unwrap();

        // POI must be exactly 0
        assert!(
            (result.parameters[poi_idx]).abs() < 1e-12,
            "POI should be fixed at 0, got {}",
            result.parameters[poi_idx]
        );

        // Must converge, not hit MaxIter
        assert!(
            result.converged,
            "Fixed-POI fit should converge at boundary, not hit MaxIter. Status: {}",
            result.message
        );

        // NLL at POI=0 should be worse (higher) than the unconstrained MLE
        let free_result = mle.fit_minimum_histfactory_with_tape(&model, &mut tape).unwrap();
        assert!(
            result.fval >= free_result.fval - 1e-10,
            "NLL at fixed POI=0 ({}) should be >= free MLE NLL ({})",
            result.fval,
            free_result.fval
        );

        // Nuisance parameters should still be reasonable
        for (i, &p) in result.parameters.iter().enumerate() {
            if i == poi_idx {
                continue;
            }
            assert!(p.is_finite(), "Param[{}] should be finite: {}", i, p);
        }
    }

    #[test]
    fn test_fit_diagnostics() {
        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();

        let mle = MaximumLikelihoodEstimator::new();
        let result = mle.fit(&model).unwrap();

        // termination_reason should be non-empty
        assert!(
            !result.termination_reason.is_empty(),
            "termination_reason should be set, got empty string"
        );

        // final_grad_norm should be finite (L-BFGS always has a gradient)
        assert!(
            result.final_grad_norm.is_finite(),
            "final_grad_norm should be finite, got {}",
            result.final_grad_norm
        );

        // initial_nll should be >= nll (optimizer should improve)
        assert!(
            result.initial_nll >= result.nll - 1e-10,
            "initial_nll ({}) should be >= nll ({})",
            result.initial_nll,
            result.nll
        );

        // n_active_bounds should be reasonable
        assert!(
            result.n_active_bounds <= result.parameters.len(),
            "n_active_bounds ({}) should be <= n_params ({})",
            result.n_active_bounds,
            result.parameters.len()
        );

        println!(
            "Diagnostics: reason='{}', grad_norm={:.2e}, initial_nll={:.4}, n_active_bounds={}",
            result.termination_reason,
            result.final_grad_norm,
            result.initial_nll,
            result.n_active_bounds
        );
    }

    #[test]
    fn test_q0_like_grad_nominal_finite_diff_simple() {
        let json = include_str!("../../../tests/fixtures/simple_workspace.json");
        let workspace: Workspace = serde_json::from_str(json).unwrap();
        let mut model = HistFactoryModel::from_workspace(&workspace).unwrap();

        let ch = model.channel_index("singlechannel").unwrap();
        let s = model.sample_index(ch, "signal").unwrap();
        model.validate_sample_nominal_override_linear_safe(ch, s).unwrap();

        let base = model.sample_nominal(ch, s).unwrap().to_vec();

        let mle = MaximumLikelihoodEstimator::with_config(OptimizerConfig {
            max_iter: 400,
            tol: 1e-6,
            m: 10,
            smooth_bounds: false,
        });

        let (q0, grad) = mle.q0_like_loss_and_grad_sample_nominal(&model, ch, s).unwrap();
        assert!(q0.is_finite());
        assert_eq!(grad.len(), base.len());

        // Pick a bin away from zero if possible to reduce clamp artifacts.
        let mut idx = 0usize;
        for (i, &v) in base.iter().enumerate() {
            if v > 0.5 {
                idx = i;
                break;
            }
        }

        let eps = 1e-3_f64.max(1e-3 * base[idx].abs());

        let mut plus = base.clone();
        plus[idx] += eps;
        model.set_sample_nominal(ch, s, &plus).unwrap();
        let (q0_p, _) = mle.q0_like_loss_and_grad_sample_nominal(&model, ch, s).unwrap();

        let mut minus = base.clone();
        minus[idx] -= eps;
        model.set_sample_nominal(ch, s, &minus).unwrap();
        let (q0_m, _) = mle.q0_like_loss_and_grad_sample_nominal(&model, ch, s).unwrap();

        model.set_sample_nominal(ch, s, &base).unwrap();

        let fd = (q0_p - q0_m) / (2.0 * eps);
        let g = grad[idx];
        let denom = fd.abs().max(g.abs()).max(1e-6);
        let rel = (fd - g).abs() / denom;
        assert!(
            rel < 5e-2,
            "finite-diff mismatch: idx={}, fd={}, grad={}, rel={}",
            idx,
            fd,
            g,
            rel
        );
    }

    #[test]
    fn test_qmu_like_grad_nominal_finite_diff_simple() {
        let json = include_str!("../../../tests/fixtures/simple_workspace.json");
        let workspace: Workspace = serde_json::from_str(json).unwrap();
        let mut model = HistFactoryModel::from_workspace(&workspace).unwrap();

        let ch = model.channel_index("singlechannel").unwrap();
        let s = model.sample_index(ch, "signal").unwrap();
        model.validate_sample_nominal_override_linear_safe(ch, s).unwrap();

        let base = model.sample_nominal(ch, s).unwrap().to_vec();

        let mle = MaximumLikelihoodEstimator::with_config(OptimizerConfig {
            max_iter: 400,
            tol: 1e-6,
            m: 10,
            smooth_bounds: false,
        });

        // Pick mu_test above typical mu_hat so we don't hit the one-sided clip.
        let mu_test = 5.0;

        let (q, grad) = mle.qmu_like_loss_and_grad_sample_nominal(&model, mu_test, ch, s).unwrap();
        assert!(q.is_finite());
        assert_eq!(grad.len(), base.len());

        let idx = 0usize;
        let eps = 1e-3_f64.max(1e-3 * base[idx].abs());

        let mut plus = base.clone();
        plus[idx] += eps;
        model.set_sample_nominal(ch, s, &plus).unwrap();
        let (q_p, _) = mle.qmu_like_loss_and_grad_sample_nominal(&model, mu_test, ch, s).unwrap();

        let mut minus = base.clone();
        minus[idx] -= eps;
        model.set_sample_nominal(ch, s, &minus).unwrap();
        let (q_m, _) = mle.qmu_like_loss_and_grad_sample_nominal(&model, mu_test, ch, s).unwrap();

        model.set_sample_nominal(ch, s, &base).unwrap();

        let fd = (q_p - q_m) / (2.0 * eps);
        let g = grad[idx];
        let denom = fd.abs().max(g.abs()).max(1e-6);
        let rel = (fd - g).abs() / denom;
        assert!(rel < 5e-2, "finite-diff mismatch: fd={}, grad={}, rel={}", fd, g, rel);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_ranking_matches_cpu_on_simple_fixture() {
        if !crate::gpu_session::is_metal_single_available() {
            eprintln!("Skipping: Metal not available");
            return;
        }

        let workspace = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&workspace).unwrap();
        let mle = MaximumLikelihoodEstimator::new();

        // CPU reference.
        let cpu = mle.ranking(&model).unwrap();
        // Metal hybrid (CPU nominal + Metal refits).
        let metal = ranking_metal(&mle, &model).unwrap();

        use std::collections::HashMap;
        let cpu_map: HashMap<&str, &RankingEntry> =
            cpu.iter().map(|e| (e.name.as_str(), e)).collect();
        let metal_map: HashMap<&str, &RankingEntry> =
            metal.iter().map(|e| (e.name.as_str(), e)).collect();

        // For meaningful impacts, require reasonable agreement.
        for (name, c) in cpu_map {
            let Some(m) = metal_map.get(name) else { continue };
            assert!(
                (m.pull - c.pull).abs() < 1e-10,
                "pull mismatch for {name}: metal={} cpu={}",
                m.pull,
                c.pull
            );
            assert!(
                (m.constraint - c.constraint).abs() < 1e-10,
                "constraint mismatch for {name}: metal={} cpu={}",
                m.constraint,
                c.constraint
            );

            let c_impact = c.delta_mu_up.abs().max(c.delta_mu_down.abs());
            if c_impact > 0.01 {
                assert!(
                    (m.delta_mu_up - c.delta_mu_up).abs() <= 1e-2,
                    "delta_mu_up mismatch for {name}: metal={} cpu={}",
                    m.delta_mu_up,
                    c.delta_mu_up
                );
                assert!(
                    (m.delta_mu_down - c.delta_mu_down).abs() <= 1e-2,
                    "delta_mu_down mismatch for {name}: metal={} cpu={}",
                    m.delta_mu_down,
                    c.delta_mu_down
                );
            }
        }
    }

    #[test]
    fn identifiability_warnings_well_identified() {
        // A well-conditioned Hessian should produce no warnings.
        let h = DMatrix::from_row_slice(2, 2, &[10.0, 0.1, 0.1, 8.0]);
        let names = vec!["a".into(), "b".into()];
        let unc = vec![0.316, 0.354]; // sqrt(1/10), sqrt(1/8) approx
        let w = super::identifiability_warnings(&h, 2, &names, &unc);
        assert!(w.is_empty(), "expected no warnings, got: {:?}", w);
    }

    #[test]
    fn identifiability_warnings_singular_hessian() {
        // Singular Hessian → warning.
        let h = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        let names = vec!["a".into(), "b".into()];
        let unc = vec![1.0, 1.0];
        let w = super::identifiability_warnings(&h, 2, &names, &unc);
        assert!(!w.is_empty(), "expected warnings for singular Hessian");
        assert!(w.iter().any(|s| s.contains("condition number") || s.contains("singular")));
    }

    #[test]
    fn identifiability_warnings_nan_uncertainty() {
        let h = DMatrix::from_row_slice(2, 2, &[10.0, 0.0, 0.0, 0.0]);
        let names = vec!["a".into(), "b".into()];
        let unc = vec![0.316, f64::NAN];
        let w = super::identifiability_warnings(&h, 2, &names, &unc);
        assert!(w.iter().any(|s| s.contains("'b'") && s.contains("NaN")));
    }

    #[test]
    fn identifiability_warnings_zero_diagonal() {
        let h = DMatrix::from_row_slice(2, 2, &[10.0, 0.0, 0.0, 1e-15]);
        let names = vec!["a".into(), "b".into()];
        let unc = vec![0.316, 1e6];
        let w = super::identifiability_warnings(&h, 2, &names, &unc);
        assert!(w.iter().any(|s| s.contains("'b'") && s.contains("not identifiable")));
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "slow; run manually to validate Metal ranking tolerances on large fixtures"]
    fn metal_ranking_smoke_thu_fixture() {
        if !crate::gpu_session::is_metal_single_available() {
            eprintln!("Skipping: Metal not available");
            return;
        }

        // Large fixture (O(100) NPs). This is for manual regression only.
        let model = load_model_from_fixture("workspace_tHu.json");
        let mle = MaximumLikelihoodEstimator::new();

        let ranking = ranking_metal(&mle, &model).unwrap();
        assert!(!ranking.is_empty());
        // Basic sanity: impacts should be finite.
        for e in ranking.iter().take(10) {
            assert!(e.delta_mu_up.is_finite());
            assert!(e.delta_mu_down.is_finite());
        }
    }
}
