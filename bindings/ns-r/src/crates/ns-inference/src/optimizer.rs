//! Optimization algorithms
//!
//! This module provides wrappers around argmin optimizers with a clean interface.

use argmin::core::{CostFunction, Executor, Gradient, State, TerminationReason, TerminationStatus};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use ns_core::Result;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone, Copy)]
enum BoundTransform {
    /// x = u
    Identity,
    /// x = lo + (hi - lo) * sigmoid(u)
    Finite { lo: f64, hi: f64 },
    /// x = lo + exp(u)
    Lower { lo: f64 },
    /// x = hi - exp(u)
    Upper { hi: f64 },
}

impl BoundTransform {
    fn forward(self, u: f64) -> f64 {
        match self {
            BoundTransform::Identity => u,
            BoundTransform::Finite { lo, hi } => {
                let s = sigmoid(u);
                lo + (hi - lo) * s
            }
            BoundTransform::Lower { lo } => lo + u.exp(),
            BoundTransform::Upper { hi } => hi - u.exp(),
        }
    }

    fn inverse(self, x: f64) -> f64 {
        const EPS: f64 = 1e-12;
        match self {
            BoundTransform::Identity => x,
            BoundTransform::Finite { lo, hi } => {
                let denom = (hi - lo).max(EPS);
                let mut t = (x - lo) / denom;
                t = t.clamp(EPS, 1.0 - EPS);
                logit(t)
            }
            BoundTransform::Lower { lo } => (x - lo).max(EPS).ln(),
            BoundTransform::Upper { hi } => (hi - x).max(EPS).ln(),
        }
    }

    /// dx/du at the given u (diagonal Jacobian).
    fn deriv(self, u: f64) -> f64 {
        match self {
            BoundTransform::Identity => 1.0,
            BoundTransform::Finite { lo, hi } => {
                let s = sigmoid(u);
                (hi - lo) * s * (1.0 - s)
            }
            BoundTransform::Lower { .. } => u.exp(),
            BoundTransform::Upper { .. } => -u.exp(),
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    // Numerically stable sigmoid.
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn logit(t: f64) -> f64 {
    (t / (1.0 - t)).ln()
}

fn build_bound_transforms(bounds: &[(f64, f64)]) -> Vec<BoundTransform> {
    bounds
        .iter()
        .map(|&(lo, hi)| {
            let lo_f = lo.is_finite();
            let hi_f = hi.is_finite();
            match (lo_f, hi_f) {
                (false, false) => BoundTransform::Identity,
                (true, false) => BoundTransform::Lower { lo },
                (false, true) => BoundTransform::Upper { hi },
                (true, true) => {
                    if (hi - lo).abs() <= 0.0 {
                        // Degenerate bounds are handled by ReducedObjective, but keep identity here.
                        BoundTransform::Identity
                    } else {
                        BoundTransform::Finite { lo, hi }
                    }
                }
            }
        })
        .collect()
}

struct TransformedObjective<'a> {
    objective: &'a dyn ObjectiveFunction,
    transforms: Vec<BoundTransform>,
}

impl<'a> TransformedObjective<'a> {
    fn to_x(&self, u: &[f64]) -> Vec<f64> {
        u.iter().zip(self.transforms.iter().copied()).map(|(&ui, t)| t.forward(ui)).collect()
    }

    fn to_u(&self, x: &[f64]) -> Vec<f64> {
        x.iter().zip(self.transforms.iter().copied()).map(|(&xi, t)| t.inverse(xi)).collect()
    }
}

impl<'a> ObjectiveFunction for TransformedObjective<'a> {
    fn eval(&self, params: &[f64]) -> Result<f64> {
        let x = self.to_x(params);
        self.objective.eval(&x)
    }

    fn gradient(&self, params: &[f64]) -> Result<Vec<f64>> {
        let x = self.to_x(params);
        let g_x = self.objective.gradient(&x)?;
        let mut g_u = Vec::with_capacity(g_x.len());
        for ((&ui, &gi), t) in params.iter().zip(g_x.iter()).zip(self.transforms.iter().copied()) {
            g_u.push(gi * t.deriv(ui));
        }
        Ok(g_u)
    }
}

/// Optimizer strategy presets matching common pyhf optimizer configurations.
///
/// These presets configure [`OptimizerConfig`] to approximate the behavior of
/// pyhf's supported optimizers (`scipy`, `minuit`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerStrategy {
    /// Default NextStat strategy: L-BFGS-B with hard boundary clamping.
    /// Fast, suitable for most HistFactory models. Equivalent to pyhf's
    /// `scipy` optimizer backend with `method="L-BFGS-B"`.
    Default,
    /// Minuit-like strategy: L-BFGS-B with smooth internal variable transforms
    /// (logistic/exp), mimicking Minuit2's MIGRAD internal parameterization.
    /// Better convergence near boundaries at the cost of slightly more evaluations.
    /// Use this for parity comparisons with `pyhf` using `iminuit` backend.
    MinuitLike,
    /// High-precision strategy: tighter tolerance, more L-BFGS corrections.
    /// For final results where robustness is preferred over speed.
    HighPrecision,
}

/// Configuration for L-BFGS-B optimizer
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Maximum number of iterations
    pub max_iter: u64,
    /// Convergence tolerance for gradient norm
    pub tol: f64,
    /// Number of corrections to approximate inverse Hessian.
    ///
    /// Set to [`Self::AUTO_M`] (0) to auto-select based on model size.
    /// Explicit values (e.g. 10, 20, 30) are used as-is.
    pub m: usize,
    /// If true, use a smooth bounds transform (logistic/exp) instead of hard clamping.
    ///
    /// This is closer in spirit to Minuit-style internal variable transforms, but can change
    /// behavior at exact boundaries. Keep false by default; enable only where parity needs it.
    pub smooth_bounds: bool,
}

impl OptimizerConfig {
    /// Sentinel value for auto-selecting L-BFGS-B history size based on model dimension.
    pub const AUTO_M: usize = 0;

    /// Returns the effective L-BFGS-B history size, auto-scaling for large models.
    ///
    /// When `m` is [`Self::AUTO_M`] (0), scales up for models with >50 parameters:
    ///   `effective_m = max(10, min(50, n_params / 5))`
    ///
    /// When `m` was explicitly set (e.g. via constructor or strategy preset), returns it unchanged.
    pub fn effective_m(&self, n_params: usize) -> usize {
        if self.m == Self::AUTO_M {
            (n_params / 5).max(10).min(50)
        } else {
            self.m
        }
    }

    /// Create a config from a named strategy preset.
    pub fn from_strategy(strategy: OptimizerStrategy) -> Self {
        match strategy {
            OptimizerStrategy::Default => Self::default(),
            OptimizerStrategy::MinuitLike => {
                Self { max_iter: 2000, tol: 1e-7, m: 20, smooth_bounds: true }
            }
            OptimizerStrategy::HighPrecision => {
                Self { max_iter: 5000, tol: 1e-9, m: 30, smooth_bounds: false }
            }
        }
    }
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self { max_iter: 1000, tol: 1e-6, m: Self::AUTO_M, smooth_bounds: false }
    }
}

/// Result of optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Best-fit parameters
    pub parameters: Vec<f64>,
    /// Function value at minimum
    pub fval: f64,
    /// Number of iterations
    pub n_iter: u64,
    /// Number of objective (cost) evaluations.
    pub n_fev: usize,
    /// Number of gradient evaluations.
    pub n_gev: usize,
    /// Convergence status
    pub converged: bool,
    /// Termination message
    pub message: String,
    /// Final gradient vector (None for gradient-free paths like 1D golden-section).
    pub final_gradient: Option<Vec<f64>>,
    /// Objective value at the initial point (before optimisation).
    pub initial_cost: f64,
}

impl fmt::Display for OptimizationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OptimizationResult(fval={:.6}, n_iter={}, n_fev={}, n_gev={}, converged={})",
            self.fval, self.n_iter, self.n_fev, self.n_gev, self.converged
        )
    }
}

/// Objective function trait for optimization
pub trait ObjectiveFunction: Send + Sync {
    /// Evaluate function at given parameters
    fn eval(&self, params: &[f64]) -> Result<f64>;

    /// Compute gradient at given parameters (numerical if not overridden)
    fn gradient(&self, params: &[f64]) -> Result<Vec<f64>> {
        // Default: central differences with adaptive step size
        let n = params.len();
        let mut grad = vec![0.0; n];

        for i in 0..n {
            // Adaptive step size: eps = sqrt(machine_epsilon) * max(|x_i|, 1)
            let eps = 1e-8 * params[i].abs().max(1.0);

            // Forward step
            let mut params_plus = params.to_vec();
            params_plus[i] += eps;
            let f_plus = self.eval(&params_plus)?;

            // Backward step
            let mut params_minus = params.to_vec();
            params_minus[i] -= eps;
            let f_minus = self.eval(&params_minus)?;

            // Central difference
            grad[i] = (f_plus - f_minus) / (2.0 * eps);
        }

        Ok(grad)
    }
}

/// Wrapper to make ObjectiveFunction compatible with argmin
struct ArgminProblem<'a> {
    objective: &'a dyn ObjectiveFunction,
    bounds: &'a [(f64, f64)],
    counts: Arc<FuncCounts>,
}

fn clamp_params(params: &[f64], bounds: &[(f64, f64)]) -> Vec<f64> {
    params.iter().zip(bounds.iter()).map(|(&v, &(lo, hi))| v.clamp(lo, hi)).collect()
}

/// Objective wrapper that removes fixed parameters (bounds where lo == hi) from the optimizer
/// parameter vector, while still evaluating the full objective in the original space.
///
/// This avoids L-BFGS coupling artifacts where a "fixed" dimension is repeatedly clamped,
/// subtly degrading convergence in the free subspace (important for profile scans where POI
/// is fixed via bounds).
struct ReducedObjective<'a> {
    objective: &'a dyn ObjectiveFunction,
    n_full: usize,
    free_idx: Vec<usize>,
    fixed: Vec<(usize, f64)>,
}

impl<'a> ReducedObjective<'a> {
    fn expand(&self, free_params: &[f64]) -> Vec<f64> {
        let mut full = vec![0.0; self.n_full];
        for &(i, v) in &self.fixed {
            full[i] = v;
        }
        for (k, &i) in self.free_idx.iter().enumerate() {
            full[i] = free_params[k];
        }
        full
    }

    fn expand_grad(&self, free_grad: &[f64]) -> Vec<f64> {
        let mut full = vec![0.0; self.n_full];
        for (k, &i) in self.free_idx.iter().enumerate() {
            full[i] = free_grad[k];
        }
        full
    }
}

impl<'a> ObjectiveFunction for ReducedObjective<'a> {
    fn eval(&self, params: &[f64]) -> Result<f64> {
        let full = self.expand(params);
        self.objective.eval(&full)
    }

    fn gradient(&self, params: &[f64]) -> Result<Vec<f64>> {
        let full = self.expand(params);
        let g_full = self.objective.gradient(&full)?;
        let mut g = Vec::with_capacity(self.free_idx.len());
        for &i in &self.free_idx {
            g.push(g_full[i]);
        }
        Ok(g)
    }
}

#[derive(Default)]
struct FuncCounts {
    cost: AtomicUsize,
    grad: AtomicUsize,
}

impl<'a> CostFunction for ArgminProblem<'a> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        self.counts.cost.fetch_add(1, Ordering::Relaxed);
        let clamped = clamp_params(params, self.bounds);
        self.objective.eval(&clamped).map_err(|e| argmin::core::Error::msg(e.to_string()))
    }
}

impl<'a> Gradient for ArgminProblem<'a> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(
        &self,
        params: &Self::Param,
    ) -> std::result::Result<Self::Gradient, argmin::core::Error> {
        self.counts.grad.fetch_add(1, Ordering::Relaxed);
        let clamped = clamp_params(params, self.bounds);
        let mut g = self
            .objective
            .gradient(&clamped)
            .map_err(|e| argmin::core::Error::msg(e.to_string()))?;

        // Projected-gradient heuristic: if we are at a bound and the gradient would push further
        // outside, zero that component. This matches the Phase 1 plan (“bounds via clamp”) and
        // prevents the line-search from repeatedly stepping into flat clamped regions.
        const EPS: f64 = 1e-12;
        for (i, (&x, &(lo, hi))) in clamped.iter().zip(self.bounds.iter()).enumerate() {
            if x <= lo + EPS && g[i] > 0.0 {
                g[i] = 0.0;
            }
            if x >= hi - EPS && g[i] < 0.0 {
                g[i] = 0.0;
            }
        }

        Ok(g)
    }
}

/// L-BFGS-B optimizer with box constraints
pub struct LbfgsbOptimizer {
    config: OptimizerConfig,
}

impl LbfgsbOptimizer {
    /// Create new L-BFGS-B optimizer with given configuration
    pub fn new(config: OptimizerConfig) -> Self {
        Self { config }
    }

    /// Minimize objective function with bounds
    ///
    /// # Arguments
    /// * `objective` - Objective function to minimize
    /// * `init_params` - Initial parameter values
    /// * `bounds` - Parameter bounds as (lower, upper) for each parameter
    ///
    /// # Returns
    /// Optimization result with best-fit parameters
    pub fn minimize(
        &self,
        objective: &dyn ObjectiveFunction,
        init_params: &[f64],
        bounds: &[(f64, f64)],
    ) -> Result<OptimizationResult> {
        if init_params.len() != bounds.len() {
            return Err(ns_core::Error::Validation(format!(
                "Parameter and bounds length mismatch: {} != {}",
                init_params.len(),
                bounds.len()
            )));
        }

        let init_clamped = clamp_params(init_params, bounds);
        let use_smooth_bounds = self.config.smooth_bounds;

        // Reduce any fixed dimensions (lo == hi) out of the optimizer space.
        //
        // This is especially important for profile scans which fix the POI by setting
        // bounds[poi] = (mu, mu). Keeping the fixed dimension in L-BFGS can introduce
        // coupling via the inverse-Hessian approximation and degrade convergence in
        // nuisance parameters, leading to small but systematic q(mu) mismatches.
        if bounds.len() > 1 {
            let mut free_idx = Vec::new();
            let mut fixed = Vec::new();
            for (i, &(lo, hi)) in bounds.iter().enumerate() {
                if (hi - lo).abs() <= 0.0 {
                    fixed.push((i, lo));
                } else {
                    free_idx.push(i);
                }
            }

            if !fixed.is_empty() {
                if free_idx.is_empty() {
                    let mut x = vec![0.0; bounds.len()];
                    for &(i, v) in &fixed {
                        x[i] = v;
                    }
                    let fval = objective.eval(&x)?;
                    return Ok(OptimizationResult {
                        parameters: x,
                        fval,
                        n_iter: 0,
                        n_fev: 1,
                        n_gev: 0,
                        converged: true,
                        message: "all bounds degenerate".to_string(),
                        final_gradient: None,
                        initial_cost: fval,
                    });
                }

                let init_free: Vec<f64> = free_idx.iter().map(|&i| init_clamped[i]).collect();
                let bounds_free: Vec<(f64, f64)> = free_idx.iter().map(|&i| bounds[i]).collect();
                let reduced = ReducedObjective { objective, n_full: bounds.len(), free_idx, fixed };

                let mut result = self.minimize(&reduced, &init_free, &bounds_free)?;
                result.parameters = reduced.expand(&result.parameters);
                result.final_gradient =
                    result.final_gradient.as_ref().map(|g| reduced.expand_grad(g));
                result.message = format!("fixed-dim reduction: {}", result.message);
                return Ok(result);
            }
        }

        // Special-case 1D problems: argmin's L-BFGS + clamping can behave poorly with
        // box constraints, even when the optimum is in the interior. For 1D likelihoods
        // (e.g. minimal examples), use a robust bracketed golden-section search instead.
        // Only use golden-section when bounds are finite; otherwise fall through to L-BFGS.
        if bounds.len() == 1
            && bounds[0].0.is_finite()
            && bounds[0].1.is_finite()
            && bounds[0].0 <= bounds[0].1
        {
            let (lo, hi) = bounds[0];
            if (hi - lo).abs() <= 0.0 {
                let x = lo;
                let fval = objective.eval(&[x])?;
                return Ok(OptimizationResult {
                    parameters: vec![x],
                    fval,
                    n_iter: 0,
                    n_fev: 1,
                    n_gev: 0,
                    converged: true,
                    message: "1D bounds degenerate".to_string(),
                    final_gradient: None,
                    initial_cost: fval,
                });
            }

            let initial_cost = objective.eval(&init_clamped)?;
            let mut n_fev = 1usize;
            let mut eval = |x: f64| -> Result<f64> {
                n_fev += 1;
                objective.eval(&[x])
            };

            // Golden-section search on [a, b]
            let mut a = lo;
            let mut b = hi;
            let phi = (5.0_f64.sqrt() - 1.0) / 2.0; // ~0.618
            let mut c = b - phi * (b - a);
            let mut d = a + phi * (b - a);

            // Respect an explicit initial value by shrinking the bracket if possible.
            let x0 = init_clamped[0];
            if x0 > a && x0 < b {
                // Keep a symmetric-ish bracket around x0, bounded by [lo, hi].
                let span = (b - a) * 0.5;
                a = (x0 - span).max(lo);
                b = (x0 + span).min(hi);
                c = b - phi * (b - a);
                d = a + phi * (b - a);
            }

            let mut fc = eval(c)?;
            let mut fd = eval(d)?;

            let tol_x = if self.config.tol > 0.0 { self.config.tol } else { 1e-8 };
            let mut it = 0u64;
            while it < self.config.max_iter {
                it += 1;
                if (b - a).abs() <= tol_x * (1.0 + 0.5 * (a.abs() + b.abs())) {
                    break;
                }
                if fc < fd {
                    b = d;
                    d = c;
                    fd = fc;
                    c = b - phi * (b - a);
                    fc = eval(c)?;
                } else {
                    a = c;
                    c = d;
                    fc = fd;
                    d = a + phi * (b - a);
                    fd = eval(d)?;
                }
            }

            let (x_best, f_best) = if fc < fd { (c, fc) } else { (d, fd) };
            return Ok(OptimizationResult {
                parameters: vec![x_best],
                fval: f_best,
                n_iter: it,
                n_fev,
                n_gev: 0,
                converged: it < self.config.max_iter,
                message: "1D golden-section search".to_string(),
                final_gradient: None,
                initial_cost,
            });
        }

        // Smooth bounds transform path (multi-dimensional): optimize in unconstrained space.
        //
        // This tends to be more robust than hard clamping for large constrained problems,
        // and closer in spirit to Minuit's internal variable transforms.
        if use_smooth_bounds && bounds.len() > 1 {
            let transforms = build_bound_transforms(bounds);
            let transformed = TransformedObjective { objective, transforms };
            let init_u = transformed.to_u(&init_clamped);

            let counts = Arc::new(FuncCounts::default());
            let unbounded: Vec<(f64, f64)> = vec![(f64::NEG_INFINITY, f64::INFINITY); bounds.len()];
            let problem = ArgminProblem {
                objective: &transformed,
                bounds: &unbounded,
                counts: counts.clone(),
            };

            let linesearch = MoreThuenteLineSearch::new();
            let tol_cost =
                if self.config.tol == 0.0 { 0.0 } else { (0.1 * self.config.tol).max(1e-12) };
            let effective_m = self.config.effective_m(bounds.len());
            let solver = LBFGS::new(linesearch, effective_m)
                .with_tolerance_grad(self.config.tol)
                .map_err(|e| {
                    ns_core::Error::Validation(format!(
                        "Invalid optimizer configuration (tol): {e}"
                    ))
                })?;
            let solver = solver.with_tolerance_cost(tol_cost).map_err(|e| {
                ns_core::Error::Validation(format!(
                    "Invalid optimizer configuration (tol_cost): {e}"
                ))
            })?;

            let initial_cost = objective.eval(&init_clamped)?;

            let res = Executor::new(problem, solver)
                .configure(|state| state.param(init_u).max_iters(self.config.max_iter))
                .run()
                .map_err(|e| ns_core::Error::Validation(format!("Optimization failed: {}", e)))?;

            let state = res.state();
            let best_u = state
                .get_best_param()
                .ok_or_else(|| ns_core::Error::Validation("No best parameters found".to_string()))?
                .clone();
            let best_x = transformed.to_x(&best_u);
            let fval = objective.eval(&best_x)?;
            let n_iter = state.get_iter();
            let n_fev = counts.cost.load(Ordering::Relaxed);
            let n_gev = counts.grad.load(Ordering::Relaxed);

            // Provide the gradient in the original x-space for diagnostics.
            let final_gradient = objective.gradient(&best_x).ok();

            let termination = state.get_termination_status();
            let converged = matches!(
                termination,
                TerminationStatus::Terminated(TerminationReason::SolverConverged)
                    | TerminationStatus::Terminated(TerminationReason::TargetCostReached)
            );
            let message = format!("smooth-bounds: {}", termination);

            return Ok(OptimizationResult {
                parameters: best_x,
                fval,
                n_iter,
                n_fev,
                n_gev,
                converged,
                message,
                final_gradient,
                initial_cost,
            });
        }

        let counts = Arc::new(FuncCounts::default());

        // Create argmin problem
        let problem = ArgminProblem { objective, bounds, counts: counts.clone() };

        // Create L-BFGS solver with line search
        let linesearch = MoreThuenteLineSearch::new();
        // Argmin's default cost tolerance is ~EPS, which is too strict for our NLL scales and
        // can lead to unnecessary max-iter terminations on larger HistFactory models.
        let tol_cost =
            if self.config.tol == 0.0 { 0.0 } else { (0.1 * self.config.tol).max(1e-12) };
        let effective_m = self.config.effective_m(bounds.len());
        let solver = LBFGS::new(linesearch, effective_m)
            .with_tolerance_grad(self.config.tol)
            .map_err(|e| {
                ns_core::Error::Validation(format!("Invalid optimizer configuration (tol): {e}"))
            })?;
        let solver = solver.with_tolerance_cost(tol_cost).map_err(|e| {
            ns_core::Error::Validation(format!("Invalid optimizer configuration (tol_cost): {e}"))
        })?;

        // Compute initial cost for diagnostics
        let initial_cost = objective.eval(&init_clamped)?;

        // Create executor
        let res = Executor::new(problem, solver)
            .configure(|state| state.param(init_clamped).max_iters(self.config.max_iter))
            .run()
            .map_err(|e| ns_core::Error::Validation(format!("Optimization failed: {}", e)))?;

        // Extract results
        let state = res.state();
        let best_params_unclamped = state
            .get_best_param()
            .ok_or_else(|| ns_core::Error::Validation("No best parameters found".to_string()))?
            .clone();
        let best_params = clamp_params(&best_params_unclamped, bounds);
        let fval = state.get_best_cost();
        let n_iter = state.get_iter();
        let n_fev = counts.cost.load(Ordering::Relaxed);
        let n_gev = counts.grad.load(Ordering::Relaxed);

        // Extract final gradient
        let final_gradient = state.get_gradient().cloned();

        // Check convergence
        let termination = state.get_termination_status();
        let converged = matches!(
            termination,
            TerminationStatus::Terminated(TerminationReason::SolverConverged)
                | TerminationStatus::Terminated(TerminationReason::TargetCostReached)
        );
        let message = termination.to_string();

        Ok(OptimizationResult {
            parameters: best_params,
            fval,
            n_iter,
            n_fev,
            n_gev,
            converged,
            message,
            final_gradient,
            initial_cost,
        })
    }
}

impl Default for LbfgsbOptimizer {
    fn default() -> Self {
        Self::new(OptimizerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Simple test function: f(x, y) = (x - 2)^2 + (y - 3)^2
    // Minimum at (2, 3) with f = 0
    struct QuadraticFunction;

    impl ObjectiveFunction for QuadraticFunction {
        fn eval(&self, params: &[f64]) -> Result<f64> {
            let x = params[0];
            let y = params[1];
            Ok((x - 2.0).powi(2) + (y - 3.0).powi(2))
        }

        fn gradient(&self, params: &[f64]) -> Result<Vec<f64>> {
            let x = params[0];
            let y = params[1];
            Ok(vec![2.0 * (x - 2.0), 2.0 * (y - 3.0)])
        }
    }

    #[test]
    fn test_optimizer_quadratic() {
        let config = OptimizerConfig { max_iter: 100, tol: 1e-6, m: 10, smooth_bounds: false };

        let optimizer = LbfgsbOptimizer::new(config);
        let objective = QuadraticFunction;

        let init = vec![0.0, 0.0];
        let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];

        let result = optimizer.minimize(&objective, &init, &bounds).unwrap();

        println!("{}", result);

        // Check convergence
        assert!(result.converged, "Optimizer should converge");

        // Check parameters
        assert_relative_eq!(result.parameters[0], 2.0, epsilon = 1e-4);
        assert_relative_eq!(result.parameters[1], 3.0, epsilon = 1e-4);

        // Check function value
        assert_relative_eq!(result.fval, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_optimizer_with_bounds() {
        let optimizer = LbfgsbOptimizer::default();
        let objective = QuadraticFunction;

        // Constrain to x in [3, 5], y in [1, 2]
        // Optimal within bounds: x=3, y=2
        let init = vec![4.0, 1.5];
        let bounds = vec![(3.0, 5.0), (1.0, 2.0)];

        let result = optimizer.minimize(&objective, &init, &bounds).unwrap();

        println!("{}", result);

        // Should find constrained optimum
        assert_relative_eq!(result.parameters[0], 3.0, epsilon = 1e-4);
        assert_relative_eq!(result.parameters[1], 2.0, epsilon = 1e-4);

        // Must converge properly, not hit MaxIter
        assert!(
            result.converged,
            "Optimizer should converge at constrained optimum, not MaxIter. Status: {}",
            result.message
        );
    }

    #[test]
    fn test_optimizer_handles_fixed_dimensions() {
        // Fix y=3 exactly via degenerate bounds and optimize x only. This should behave like
        // optimizing the reduced 1D objective (x - 2)^2, and must not degrade convergence.
        let config = OptimizerConfig { max_iter: 200, tol: 1e-8, m: 10, smooth_bounds: false };
        let optimizer = LbfgsbOptimizer::new(config);
        let objective = QuadraticFunction;

        let init = vec![0.0, 0.0];
        let bounds = vec![(-10.0, 10.0), (3.0, 3.0)];

        let result = optimizer.minimize(&objective, &init, &bounds).unwrap();
        assert!(result.converged, "fixed-dim problem should converge: {}", result.message);
        assert_relative_eq!(result.parameters[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(result.parameters[1], 3.0, epsilon = 1e-12);
        assert_relative_eq!(result.fval, 0.0, epsilon = 1e-8);
    }

    // Quadratic with negative offset: minimum is negative.
    struct QuadraticNegativeOffset;

    impl ObjectiveFunction for QuadraticNegativeOffset {
        fn eval(&self, params: &[f64]) -> Result<f64> {
            let x = params[0];
            Ok((x - 2.0).powi(2) - 5.0)
        }

        fn gradient(&self, params: &[f64]) -> Result<Vec<f64>> {
            let x = params[0];
            Ok(vec![2.0 * (x - 2.0)])
        }
    }

    #[test]
    fn test_optimizer_does_not_stop_at_negative_cost() {
        let config = OptimizerConfig { max_iter: 100, tol: 1e-6, m: 10, smooth_bounds: false };
        let optimizer = LbfgsbOptimizer::new(config);
        let objective = QuadraticNegativeOffset;

        // f(0) = -1 already < 0, but the true minimum is f(2) = -5.
        let init = vec![0.0];
        let bounds = vec![(-10.0, 10.0)];

        let result = optimizer.minimize(&objective, &init, &bounds).unwrap();

        assert!(result.converged, "Optimizer should converge");
        assert_relative_eq!(result.parameters[0], 2.0, epsilon = 1e-4);
        assert_relative_eq!(result.fval, -5.0, epsilon = 1e-6);
    }

    // Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2
    // Minimum at (a, a^2) with f = 0
    // Common test: a=1, b=100, min at (1, 1)
    struct RosenbrockFunction;

    impl ObjectiveFunction for RosenbrockFunction {
        fn eval(&self, params: &[f64]) -> Result<f64> {
            let x = params[0];
            let y = params[1];
            let a = 1.0;
            let b = 100.0;
            Ok((a - x).powi(2) + b * (y - x.powi(2)).powi(2))
        }
    }

    #[test]
    fn test_optimizer_rosenbrock() {
        let config = OptimizerConfig { max_iter: 1000, tol: 1e-6, m: 10, smooth_bounds: false };

        let optimizer = LbfgsbOptimizer::new(config);
        let objective = RosenbrockFunction;

        let init = vec![0.0, 0.0];
        let bounds = vec![(-10.0, 10.0), (-10.0, 10.0)];

        let result = optimizer.minimize(&objective, &init, &bounds).unwrap();

        println!("Rosenbrock: {}", result);

        // Rosenbrock is challenging, accept looser tolerance
        assert_relative_eq!(result.parameters[0], 1.0, epsilon = 1e-3);
        assert_relative_eq!(result.parameters[1], 1.0, epsilon = 1e-3);
        assert!(result.fval < 1e-4);
    }

    #[test]
    fn test_optimizer_converges_at_bound_when_minimum_outside() {
        // f(x,y) = (x+1)^2 + (y-3)^2  →  unconstrained min at (-1, 3)
        // Bounds: x in [0, 5], y in [0, 2]  →  constrained min at (0, 2)
        struct ShiftedQuadratic;

        impl ObjectiveFunction for ShiftedQuadratic {
            fn eval(&self, params: &[f64]) -> Result<f64> {
                let x = params[0];
                let y = params[1];
                Ok((x + 1.0).powi(2) + (y - 3.0).powi(2))
            }

            fn gradient(&self, params: &[f64]) -> Result<Vec<f64>> {
                let x = params[0];
                let y = params[1];
                Ok(vec![2.0 * (x + 1.0), 2.0 * (y - 3.0)])
            }
        }

        let config = OptimizerConfig { max_iter: 200, tol: 1e-6, m: 10, smooth_bounds: false };
        let optimizer = LbfgsbOptimizer::new(config);
        let objective = ShiftedQuadratic;

        let init = vec![3.0, 1.0];
        let bounds = vec![(0.0, 5.0), (0.0, 2.0)];

        let result = optimizer.minimize(&objective, &init, &bounds).unwrap();

        // Parameters should be at the boundary
        assert_relative_eq!(result.parameters[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(result.parameters[1], 2.0, epsilon = 1e-6);

        // f(0,2) = 1 + 1 = 2
        assert_relative_eq!(result.fval, 2.0, epsilon = 1e-6);

        // Must converge, not hit MaxIter
        assert!(
            result.converged,
            "Optimizer should converge at boundary, not hit MaxIter. Status: {}",
            result.message
        );
    }

    #[test]
    fn test_optimizer_1d_mu_stuck_at_zero() {
        // f(x) = (x + 1)^2  →  min at x = -1
        // Bound: x in [0, 10]  →  constrained min at x = 0
        // Mimics POI (mu) stuck at lower boundary.
        struct Quadratic1D;

        impl ObjectiveFunction for Quadratic1D {
            fn eval(&self, params: &[f64]) -> Result<f64> {
                let x = params[0];
                Ok((x + 1.0).powi(2))
            }

            fn gradient(&self, params: &[f64]) -> Result<Vec<f64>> {
                let x = params[0];
                Ok(vec![2.0 * (x + 1.0)])
            }
        }

        let config = OptimizerConfig { max_iter: 100, tol: 1e-8, m: 10, smooth_bounds: false };
        let optimizer = LbfgsbOptimizer::new(config);

        let init = vec![5.0];
        let bounds = vec![(0.0, 10.0)];

        let result = optimizer.minimize(&Quadratic1D, &init, &bounds).unwrap();

        // Should be pinned at 0 (golden-section tolerance is ~1e-8)
        assert_relative_eq!(result.parameters[0], 0.0, epsilon = 1e-6);

        // f(0) = 1
        assert_relative_eq!(result.fval, 1.0, epsilon = 1e-6);

        // Must converge (projected gradient = 0 at lower bound)
        assert!(
            result.converged,
            "1D mu-at-zero should converge, not hit MaxIter. Status: {}",
            result.message
        );

        // Should not need many iterations for a simple 1D case
        assert!(result.n_iter < 60, "Should converge quickly, used {} iterations", result.n_iter);
    }

    #[test]
    fn test_optimizer_1d_unbounded() {
        // f(x) = (x - 3)^2, min at x = 3
        // Bounds: (-inf, inf) — should fall through to L-BFGS
        struct Quad;
        impl ObjectiveFunction for Quad {
            fn eval(&self, p: &[f64]) -> Result<f64> {
                Ok((p[0] - 3.0).powi(2))
            }
            fn gradient(&self, p: &[f64]) -> Result<Vec<f64>> {
                Ok(vec![2.0 * (p[0] - 3.0)])
            }
        }
        let config = OptimizerConfig { max_iter: 100, tol: 1e-8, m: 10, smooth_bounds: false };
        let optimizer = LbfgsbOptimizer::new(config);
        let result =
            optimizer.minimize(&Quad, &[0.0], &[(f64::NEG_INFINITY, f64::INFINITY)]).unwrap();
        assert_relative_eq!(result.parameters[0], 3.0, epsilon = 1e-6);
        assert!(result.converged);
    }

    #[test]
    fn test_optimizer_1d_semi_bounded() {
        // f(x) = (x - 3)^2, min at x = 3
        // Bounds: (0, inf) — should fall through to L-BFGS
        struct Quad;
        impl ObjectiveFunction for Quad {
            fn eval(&self, p: &[f64]) -> Result<f64> {
                Ok((p[0] - 3.0).powi(2))
            }
            fn gradient(&self, p: &[f64]) -> Result<Vec<f64>> {
                Ok(vec![2.0 * (p[0] - 3.0)])
            }
        }
        let config = OptimizerConfig { max_iter: 100, tol: 1e-8, m: 10, smooth_bounds: false };
        let optimizer = LbfgsbOptimizer::new(config);
        let result = optimizer.minimize(&Quad, &[0.0], &[(0.0, f64::INFINITY)]).unwrap();
        assert_relative_eq!(result.parameters[0], 3.0, epsilon = 1e-6);
        assert!(result.converged);
    }
}
