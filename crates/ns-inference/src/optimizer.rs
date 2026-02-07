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

/// Configuration for L-BFGS-B optimizer
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Maximum number of iterations
    pub max_iter: u64,
    /// Convergence tolerance for gradient norm
    pub tol: f64,
    /// Number of corrections to approximate inverse Hessian
    pub m: usize,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self { max_iter: 1000, tol: 1e-6, m: 10 }
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

        let counts = Arc::new(FuncCounts::default());

        // Create argmin problem
        let problem = ArgminProblem { objective, bounds, counts: counts.clone() };

        // Create L-BFGS solver with line search
        let linesearch = MoreThuenteLineSearch::new();
        // Argmin's default cost tolerance is ~EPS, which is too strict for our NLL scales and
        // can lead to unnecessary max-iter terminations on larger HistFactory models.
        let tol_cost =
            if self.config.tol == 0.0 { 0.0 } else { (0.1 * self.config.tol).max(1e-12) };
        let solver = LBFGS::new(linesearch, self.config.m)
            .with_tolerance_grad(self.config.tol)
            .map_err(|e| {
                ns_core::Error::Validation(format!("Invalid optimizer configuration (tol): {e}"))
            })?;
        let solver = solver.with_tolerance_cost(tol_cost).map_err(|e| {
            ns_core::Error::Validation(format!("Invalid optimizer configuration (tol_cost): {e}"))
        })?;

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
        let config = OptimizerConfig { max_iter: 100, tol: 1e-6, m: 10 };

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
        let config = OptimizerConfig { max_iter: 100, tol: 1e-6, m: 10 };
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
        let config = OptimizerConfig { max_iter: 1000, tol: 1e-6, m: 10 };

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

        let config = OptimizerConfig { max_iter: 200, tol: 1e-6, m: 10 };
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

        let config = OptimizerConfig { max_iter: 100, tol: 1e-8, m: 10 };
        let optimizer = LbfgsbOptimizer::new(config);

        let init = vec![5.0];
        let bounds = vec![(0.0, 10.0)];

        let result = optimizer.minimize(&Quadratic1D, &init, &bounds).unwrap();

        // Should be pinned at 0
        assert_relative_eq!(result.parameters[0], 0.0, epsilon = 1e-10);

        // f(0) = 1
        assert_relative_eq!(result.fval, 1.0, epsilon = 1e-10);

        // Must converge (projected gradient = 0 at lower bound)
        assert!(
            result.converged,
            "1D mu-at-zero should converge, not hit MaxIter. Status: {}",
            result.message
        );

        // Should not need many iterations for a simple 1D case
        assert!(
            result.n_iter < 20,
            "Should converge quickly, used {} iterations",
            result.n_iter
        );
    }
}
