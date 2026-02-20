//! Posterior distribution for Bayesian inference.
//!
//! Wraps a [`HistFactoryModel`] and provides log-posterior density and
//! gradient in both constrained and unconstrained parameterizations.
//!
//! The model's NLL already includes Gaussian constraints. The [`Posterior`]
//! adds only user-specified additional priors (if any).

use crate::transforms::ParameterTransform;
use ns_core::Result;
use ns_core::traits::LogDensityModel;

/// Prior distribution for a single parameter.
#[derive(Debug, Clone)]
pub enum Prior {
    /// Flat (improper) prior — contributes 0 to log-posterior.
    Flat,
    /// Normal prior: `log p(theta) = -0.5 * ((theta - center) / width)^2 + const`.
    Normal {
        /// Center of the Gaussian prior.
        center: f64,
        /// Width (standard deviation) of the Gaussian prior.
        width: f64,
    },
}

/// Posterior wrapping a HistFactory model with transforms and priors.
///
/// - `logpdf(theta) = -model.nll(theta) + sum(prior_logpdf)`
/// - `logpdf_unconstrained(z) = logpdf(transform(z)) + log|J(z)|`
pub struct Posterior<'a, M: LogDensityModel + ?Sized> {
    model: &'a M,
    transform: ParameterTransform,
    priors: Vec<Prior>,
    /// True when all priors are `Prior::Flat`.
    all_flat_priors: bool,
}

impl<'a, M: LogDensityModel + ?Sized> Posterior<'a, M> {
    /// Create a new posterior with flat priors (relying on model's built-in constraints).
    pub fn new(model: &'a M) -> Self {
        let dim = model.dim();

        // Defensive: keep Posterior well-defined even if a model accidentally
        // returns bounds length != dim.
        let mut bounds: Vec<(f64, f64)> = model.parameter_bounds();
        if bounds.len() > dim {
            bounds.truncate(dim);
        } else if bounds.len() < dim {
            bounds.resize(dim, (f64::NEG_INFINITY, f64::INFINITY));
        }

        let transform = ParameterTransform::from_bounds(&bounds);
        let priors = vec![Prior::Flat; dim];
        Self { model, transform, priors, all_flat_priors: true }
    }

    /// Set priors (one per parameter). Length must match `model.dim()`.
    pub fn with_priors(mut self, priors: Vec<Prior>) -> Result<Self> {
        if priors.len() != self.model.dim() {
            return Err(ns_core::Error::Validation(format!(
                "priors length must match model.dim(): {} != {}",
                priors.len(),
                self.model.dim()
            )));
        }
        self.all_flat_priors = priors.iter().all(|p| matches!(p, Prior::Flat));
        self.priors = priors;
        Ok(self)
    }

    /// Number of parameters.
    pub fn dim(&self) -> usize {
        self.model.dim()
    }

    /// Reference to the underlying model.
    pub fn model(&self) -> &M {
        self.model
    }

    /// Reference to the parameter transform.
    pub fn transform(&self) -> &ParameterTransform {
        &self.transform
    }

    fn validate_theta_len(&self, theta: &[f64]) -> Result<()> {
        if theta.len() != self.dim() {
            return Err(ns_core::Error::Validation(format!(
                "expected theta length = model.dim() = {}, got {}",
                self.dim(),
                theta.len()
            )));
        }
        if theta.iter().any(|v| !v.is_finite()) {
            return Err(ns_core::Error::Validation(
                "theta must contain only finite values".to_string(),
            ));
        }
        Ok(())
    }

    fn validate_z_len(&self, z: &[f64]) -> Result<()> {
        if z.len() != self.dim() {
            return Err(ns_core::Error::Validation(format!(
                "expected z length = model.dim() = {}, got {}",
                self.dim(),
                z.len()
            )));
        }
        if z.iter().any(|v| !v.is_finite()) {
            return Err(ns_core::Error::Validation(
                "z must contain only finite values".to_string(),
            ));
        }
        Ok(())
    }

    fn is_nonfinite_validation(err: &ns_core::Error) -> bool {
        match err {
            ns_core::Error::Validation(msg) => msg.contains("must contain only finite values"),
            _ => false,
        }
    }

    /// Log-posterior in constrained space: `-model.nll(theta) + sum(prior_logpdf)`.
    pub fn logpdf(&self, theta: &[f64]) -> Result<f64> {
        self.validate_theta_len(theta)?;
        let nll = self.model.nll(theta)?;
        let mut lp = -nll;

        for (i, prior) in self.priors.iter().enumerate() {
            match prior {
                Prior::Flat => {}
                Prior::Normal { center, width } => {
                    if !width.is_finite() || *width <= 0.0 {
                        return Err(ns_core::Error::Validation(format!(
                            "Normal prior width must be finite and > 0, got {}",
                            width
                        )));
                    }
                    let pull = (theta[i] - center) / width;
                    lp -= 0.5 * pull * pull;
                }
            }
        }

        Ok(lp)
    }

    /// Gradient of log-posterior in constrained space.
    pub fn grad(&self, theta: &[f64]) -> Result<Vec<f64>> {
        self.validate_theta_len(theta)?;
        let mut g = self.model.grad_nll(theta)?;

        // grad(logpdf) = -grad(nll) + grad(prior)
        for gi in g.iter_mut() {
            *gi = -*gi;
        }

        for (i, prior) in self.priors.iter().enumerate() {
            match prior {
                Prior::Flat => {}
                Prior::Normal { center, width } => {
                    if !width.is_finite() || *width <= 0.0 {
                        return Err(ns_core::Error::Validation(format!(
                            "Normal prior width must be finite and > 0, got {}",
                            width
                        )));
                    }
                    g[i] -= (theta[i] - center) / (width * width);
                }
            }
        }

        Ok(g)
    }

    /// Log-posterior in unconstrained space: `logpdf(transform(z)) + log|J(z)|`.
    pub fn logpdf_unconstrained(&self, z: &[f64]) -> Result<f64> {
        if z.len() != self.dim() {
            return Err(ns_core::Error::Validation(format!(
                "expected z length = model.dim() = {}, got {}",
                self.dim(),
                z.len()
            )));
        }
        // HMC/NUTS can transiently propose non-finite values due to integrator instability.
        // Treat these as invalid states (logpdf = -inf) instead of hard-erroring.
        if z.iter().any(|v| !v.is_finite()) {
            return Ok(f64::NEG_INFINITY);
        }

        let theta = match self.to_constrained(z) {
            Ok(t) => t,
            Err(e) if Self::is_nonfinite_validation(&e) => return Ok(f64::NEG_INFINITY),
            Err(e) => return Err(e),
        };
        let lp = match self.logpdf(&theta) {
            Ok(v) => v,
            Err(e) if Self::is_nonfinite_validation(&e) => return Ok(f64::NEG_INFINITY),
            Err(e) => return Err(e),
        };
        let log_jac = self.transform.log_abs_det_jacobian(z);
        Ok(lp + log_jac)
    }

    /// Gradient of log-posterior in unconstrained space.
    ///
    /// Chain rule (diagonal Jacobian):
    /// `grad_z[i] = (dtheta_i/dz_i) * grad_theta[i] + d/dz_i log|J_i|`
    pub fn grad_unconstrained(&self, z: &[f64]) -> Result<Vec<f64>> {
        if z.len() != self.dim() {
            return Err(ns_core::Error::Validation(format!(
                "expected z length = model.dim() = {}, got {}",
                self.dim(),
                z.len()
            )));
        }
        // See `logpdf_unconstrained`: reject non-finite proposals gracefully.
        if z.iter().any(|v| !v.is_finite()) {
            return Ok(vec![0.0; self.dim()]);
        }

        let theta = match self.to_constrained(z) {
            Ok(t) => t,
            Err(e) if Self::is_nonfinite_validation(&e) => return Ok(vec![0.0; self.dim()]),
            Err(e) => return Err(e),
        };
        let grad_theta = match self.grad(&theta) {
            Ok(g) => g,
            Err(e) if Self::is_nonfinite_validation(&e) => return Ok(vec![0.0; self.dim()]),
            Err(e) => return Err(e),
        };
        let jac_diag = self.transform.jacobian_diag(z);
        let grad_log_jac = self.transform.grad_log_abs_det_jacobian(z);

        let grad_z: Vec<f64> = grad_theta
            .iter()
            .zip(jac_diag.iter())
            .zip(grad_log_jac.iter())
            .map(|((&gt, &jd), &glj)| gt * jd + glj)
            .collect();

        Ok(grad_z)
    }

    /// Fused log-posterior and gradient in unconstrained space.
    ///
    /// This avoids duplicate constrained transforms and allows models to use
    /// fused `nll+grad` implementations via `LogDensityModel::nll_grad_prepared`.
    pub fn logpdf_grad_unconstrained(&self, z: &[f64]) -> Result<(f64, Vec<f64>)> {
        if z.len() != self.dim() {
            return Err(ns_core::Error::Validation(format!(
                "expected z length = model.dim() = {}, got {}",
                self.dim(),
                z.len()
            )));
        }
        if z.iter().any(|v| !v.is_finite()) {
            return Ok((f64::NEG_INFINITY, vec![0.0; self.dim()]));
        }

        // Fast-path: when all transforms are identity (all params unbounded),
        // z IS theta, log|J| = 0, grad_log|J| = 0, jac_diag = 1.
        // Saves 4 Vec allocations (to_constrained, jac_diag, grad_log_jac, grad_z)
        // + all per-element transform math per leapfrog step.
        // Normal priors are fused in-place — no extra allocs.
        if self.transform.is_all_identity() {
            let prepared = self.model.prepared();
            let (nll, grad_nll) = match self.model.nll_grad_prepared(&prepared, z) {
                Ok(v) => v,
                Err(e) if Self::is_nonfinite_validation(&e) => {
                    return Ok((f64::NEG_INFINITY, vec![0.0; self.dim()]));
                }
                Err(e) => return Err(e),
            };
            let mut lp = -nll;
            let mut grad_z = grad_nll; // take ownership, no alloc
            for g in &mut grad_z { *g = -*g; }
            // Fuse prior contributions in-place (z == theta when identity)
            if !self.all_flat_priors {
                for (i, prior) in self.priors.iter().enumerate() {
                    match prior {
                        Prior::Flat => {}
                        Prior::Normal { center, width } => {
                            let pull = (z[i] - center) / width;
                            lp -= 0.5 * pull * pull;
                            grad_z[i] -= (z[i] - center) / (width * width);
                        }
                    }
                }
            }
            return Ok((lp, grad_z));
        }

        let theta = match self.to_constrained(z) {
            Ok(t) => t,
            Err(e) if Self::is_nonfinite_validation(&e) => {
                return Ok((f64::NEG_INFINITY, vec![0.0; self.dim()]));
            }
            Err(e) => return Err(e),
        };

        let prepared = self.model.prepared();
        let (nll, grad_nll) = match self.model.nll_grad_prepared(&prepared, &theta) {
            Ok(v) => v,
            Err(e) if Self::is_nonfinite_validation(&e) => {
                return Ok((f64::NEG_INFINITY, vec![0.0; self.dim()]));
            }
            Err(e) => return Err(e),
        };

        let mut lp = -nll;
        let mut grad_theta_logpdf = grad_nll; // take ownership, no alloc
        for g in &mut grad_theta_logpdf { *g = -*g; }

        for (i, prior) in self.priors.iter().enumerate() {
            match prior {
                Prior::Flat => {}
                Prior::Normal { center, width } => {
                    if !width.is_finite() || *width <= 0.0 {
                        return Err(ns_core::Error::Validation(format!(
                            "Normal prior width must be finite and > 0, got {}",
                            width
                        )));
                    }
                    let pull = (theta[i] - center) / width;
                    lp -= 0.5 * pull * pull;
                    grad_theta_logpdf[i] -= (theta[i] - center) / (width * width);
                }
            }
        }

        let log_jac = self.transform.log_abs_det_jacobian(z);
        let jac_diag = self.transform.jacobian_diag(z);
        let grad_log_jac = self.transform.grad_log_abs_det_jacobian(z);

        let grad_z: Vec<f64> = grad_theta_logpdf
            .iter()
            .zip(jac_diag.iter())
            .zip(grad_log_jac.iter())
            .map(|((&gt, &jd), &glj)| gt * jd + glj)
            .collect();

        Ok((lp + log_jac, grad_z))
    }

    /// Map constrained -> unconstrained.
    pub fn to_unconstrained(&self, theta: &[f64]) -> Result<Vec<f64>> {
        self.validate_theta_len(theta)?;
        Ok(self.transform.inverse(theta))
    }

    /// Map unconstrained -> constrained.
    pub fn to_constrained(&self, z: &[f64]) -> Result<Vec<f64>> {
        self.validate_z_len(z)?;
        Ok(self.transform.forward(z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regression::LogisticRegressionModel;
    use ns_translate::pyhf::HistFactoryModel;
    use ns_translate::pyhf::Workspace;

    fn load_simple_workspace() -> Workspace {
        let json = include_str!("../../../tests/fixtures/simple_workspace.json");
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn test_posterior_flat_prior_equals_neg_nll() {
        let ws = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();
        let posterior = Posterior::new(&model);

        let params = vec![1.0; model.n_params()];
        let lp = posterior.logpdf(&params).unwrap();
        let nll = model.nll(&params).unwrap();

        let diff = (lp + nll).abs();
        assert!(
            diff < 1e-12,
            "Flat prior: logpdf should equal -nll: lp={}, nll={}, diff={}",
            lp,
            nll,
            diff
        );
    }

    #[test]
    fn test_posterior_grad_vs_finite_diff() {
        let ws = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();
        let posterior = Posterior::new(&model);

        let theta = vec![1.2, 0.9, 1.1];
        let grad = posterior.grad(&theta).unwrap();

        let eps = 1e-7;
        for i in 0..theta.len() {
            let mut t_plus = theta.clone();
            t_plus[i] += eps;
            let mut t_minus = theta.clone();
            t_minus[i] -= eps;
            let g_fd = (posterior.logpdf(&t_plus).unwrap() - posterior.logpdf(&t_minus).unwrap())
                / (2.0 * eps);
            let diff = (grad[i] - g_fd).abs();
            let scale = grad[i].abs().max(1.0);
            assert!(
                diff / scale < 1e-5,
                "grad[{}]: analytical={}, fd={}, diff={}",
                i,
                grad[i],
                g_fd,
                diff
            );
        }
    }

    #[test]
    fn test_unconstrained_logpdf_includes_log_jac() {
        let ws = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();
        let posterior = Posterior::new(&model);

        let theta = vec![1.0; model.n_params()];
        let z = posterior.to_unconstrained(&theta).unwrap();
        let theta_back = posterior.to_constrained(&z).unwrap();

        // Roundtrip
        for (i, (&a, &b)) in theta.iter().zip(theta_back.iter()).enumerate() {
            assert!((a - b).abs() < 1e-10, "Roundtrip failed at [{}]: {} vs {}", i, a, b);
        }

        // logpdf_unconstrained = logpdf(theta) + log|J|
        let lp_constrained = posterior.logpdf(&theta).unwrap();
        let lp_unconstrained = posterior.logpdf_unconstrained(&z).unwrap();
        let log_jac = posterior.transform().log_abs_det_jacobian(&z);

        let diff = (lp_unconstrained - lp_constrained - log_jac).abs();
        assert!(
            diff < 1e-12,
            "lp_unconstrained={}, lp_constrained + log_jac={}, diff={}",
            lp_unconstrained,
            lp_constrained + log_jac,
            diff
        );
    }

    #[test]
    fn test_unconstrained_grad_vs_finite_diff() {
        let ws = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();
        let posterior = Posterior::new(&model);

        let theta = vec![1.2, 0.9, 1.1];
        let z = posterior.to_unconstrained(&theta).unwrap();
        let grad = posterior.grad_unconstrained(&z).unwrap();

        let eps = 1e-6;
        for i in 0..z.len() {
            let mut z_plus = z.clone();
            z_plus[i] += eps;
            let mut z_minus = z.clone();
            z_minus[i] -= eps;
            let g_fd = (posterior.logpdf_unconstrained(&z_plus).unwrap()
                - posterior.logpdf_unconstrained(&z_minus).unwrap())
                / (2.0 * eps);
            let diff = (grad[i] - g_fd).abs();
            let scale = grad[i].abs().max(1.0);
            assert!(
                diff / scale < 1e-4,
                "grad_unconstrained[{}]: analytical={}, fd={}, diff={}",
                i,
                grad[i],
                g_fd,
                diff
            );
        }
    }

    #[test]
    fn test_map_with_flat_prior_equals_mle() {
        let ws = load_simple_workspace();
        let model = HistFactoryModel::from_workspace(&ws).unwrap();

        // MLE fit
        let mle = crate::MaximumLikelihoodEstimator::new();
        let fit = mle.fit(&model).unwrap();

        // Posterior with flat priors: MAP should equal MLE
        let posterior = Posterior::new(&model);

        // At MLE, gradient of logpdf should be near zero (constrained)
        let grad = posterior.grad(&fit.parameters).unwrap();
        for (i, &g) in grad.iter().enumerate() {
            // Allow larger tolerance for bounded params (gradient may not be zero at boundary)
            let p = &model.parameters()[i];
            if (fit.parameters[i] - p.bounds.0).abs() < 1e-6
                || (fit.parameters[i] - p.bounds.1).abs() < 1e-6
            {
                continue;
            }
            assert!(g.abs() < 1e-3, "At MLE, grad[{}]={} should be ~0 (param={})", i, g, p.name);
        }
    }

    /// Test the identity-transform + Normal-priors fast-path in logpdf_grad_unconstrained.
    /// LogisticRegressionModel has all-unbounded params (identity transforms).
    /// Adding Normal priors must fuse correctly in the fast-path.
    #[test]
    fn test_identity_normal_prior_fused_grad_vs_finite_diff() {
        // Small logistic regression: 2 features + intercept = 3 params, all unbounded
        let x = vec![
            vec![1.0, 0.5],
            vec![-0.3, 1.2],
            vec![0.7, -0.8],
            vec![-1.1, 0.3],
            vec![0.2, 0.9],
        ];
        let y = vec![1u8, 0, 1, 0, 1];
        let model = LogisticRegressionModel::new(x, y, true).unwrap();

        // Verify identity transforms (all params unbounded)
        let posterior_flat = Posterior::new(&model);
        assert!(posterior_flat.transform().is_all_identity());

        // Add Normal priors: intercept ~ N(0, 5), beta1 ~ N(0, 2.5), beta2 ~ N(0, 2.5)
        let priors = vec![
            Prior::Normal { center: 0.0, width: 5.0 },
            Prior::Normal { center: 0.0, width: 2.5 },
            Prior::Normal { center: 0.0, width: 2.5 },
        ];
        let posterior = Posterior::new(&model).with_priors(priors).unwrap();
        assert!(posterior.transform().is_all_identity());
        assert!(!posterior.all_flat_priors);

        // Test point (z == theta for identity transforms)
        let z = vec![0.3, -0.5, 0.8];
        let (lp, grad) = posterior.logpdf_grad_unconstrained(&z).unwrap();

        // Verify logpdf value against logpdf_unconstrained (uses general path)
        let lp_ref = posterior.logpdf_unconstrained(&z).unwrap();
        assert!(
            (lp - lp_ref).abs() < 1e-12,
            "logpdf mismatch: fast={}, ref={}, diff={}",
            lp, lp_ref, (lp - lp_ref).abs()
        );

        // Verify gradient via finite differences
        let eps = 1e-7;
        for i in 0..z.len() {
            let mut z_plus = z.clone();
            z_plus[i] += eps;
            let mut z_minus = z.clone();
            z_minus[i] -= eps;
            let (lp_plus, _) = posterior.logpdf_grad_unconstrained(&z_plus).unwrap();
            let (lp_minus, _) = posterior.logpdf_grad_unconstrained(&z_minus).unwrap();
            let g_fd = (lp_plus - lp_minus) / (2.0 * eps);
            let diff = (grad[i] - g_fd).abs();
            let scale = grad[i].abs().max(1.0);
            assert!(
                diff / scale < 1e-5,
                "grad[{}]: analytical={}, fd={}, diff={}",
                i, grad[i], g_fd, diff
            );
        }

        // Verify prior actually changes the logpdf (not just flat)
        let lp_flat = posterior_flat.logpdf_grad_unconstrained(&z).unwrap().0;
        assert!(
            (lp - lp_flat).abs() > 1e-6,
            "Normal prior should change logpdf: with_prior={}, flat={}",
            lp, lp_flat
        );
    }
}

// ---------------------------------------------------------------------------
// MAP (Maximum A Posteriori) estimation
// ---------------------------------------------------------------------------

use crate::optimizer::{LbfgsbOptimizer, ObjectiveFunction, OptimizerConfig};

/// Result of MAP (Maximum A Posteriori) estimation.
#[derive(Debug, Clone)]
pub struct MapResult {
    /// MAP parameter estimates (mode of posterior).
    pub params: Vec<f64>,
    /// Standard errors from inverse observed Fisher information (Hessian-based).
    /// None if Hessian computation fails or is not requested.
    pub se: Option<Vec<f64>>,
    /// Negative log-posterior at MAP estimate.
    pub nll_posterior: f64,
    /// Negative log-likelihood at MAP estimate (without prior contribution).
    pub nll: f64,
    /// Log-prior at MAP estimate.
    pub log_prior: f64,
    /// Number of optimizer iterations.
    pub n_iter: usize,
    /// Whether the optimizer converged.
    pub converged: bool,
    /// Parameter names (if available from model).
    pub param_names: Option<Vec<String>>,
    /// Shrinkage: (MAP - MLE) / MLE for each parameter (measures prior influence).
    pub shrinkage: Option<Vec<f64>>,
}

/// Configuration for MAP estimation.
#[derive(Debug, Clone)]
pub struct MapConfig {
    /// Maximum optimizer iterations (default: 1000).
    pub max_iter: usize,
    /// Convergence tolerance (default: 1e-8).
    pub tol: f64,
    /// L-BFGS-B history size (default: 20).
    pub m: usize,
    /// Whether to compute Hessian-based standard errors (default: true).
    pub compute_se: bool,
    /// Whether to compute shrinkage vs MLE (default: false -- requires extra MLE fit).
    pub compute_shrinkage: bool,
    /// Initial parameter values (None = use model defaults).
    pub init: Option<Vec<f64>>,
}

impl Default for MapConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-8,
            m: 20,
            compute_se: true,
            compute_shrinkage: false,
            init: None,
        }
    }
}

/// Objective function for MAP: minimizes negative log-posterior.
///
/// `f(theta) = model.nll(theta) - log_prior(theta)`
/// `grad f(theta) = grad_nll(theta) - grad_log_prior(theta)`
struct MapObjective<'a, M: LogDensityModel + ?Sized> {
    posterior: &'a Posterior<'a, M>,
}

// SAFETY: L-BFGS-B optimizer is single-threaded within one minimize() call.
unsafe impl<M: LogDensityModel + ?Sized> Send for MapObjective<'_, M> {}
unsafe impl<M: LogDensityModel + ?Sized> Sync for MapObjective<'_, M> {}

impl<M: LogDensityModel + ?Sized> ObjectiveFunction for MapObjective<'_, M> {
    fn eval(&self, params: &[f64]) -> ns_core::Result<f64> {
        // negative log-posterior = -logpdf(theta) = nll(theta) - log_prior(theta)
        let lp = self.posterior.logpdf(params)?;
        Ok(-lp)
    }

    fn gradient(&self, params: &[f64]) -> ns_core::Result<Vec<f64>> {
        // grad(-logpdf) = -grad(logpdf)
        let mut g = self.posterior.grad(params)?;
        for gi in g.iter_mut() {
            *gi = -*gi;
        }
        Ok(g)
    }
}

/// Compute the log-prior contribution at the given parameters.
fn log_prior_at(priors: &[Prior], params: &[f64]) -> f64 {
    let mut lp = 0.0;
    for (i, prior) in priors.iter().enumerate() {
        match prior {
            Prior::Flat => {}
            Prior::Normal { center, width } => {
                if *width > 0.0 && width.is_finite() {
                    let pull = (params[i] - center) / width;
                    lp -= 0.5 * pull * pull;
                }
            }
        }
    }
    lp
}

/// Compute Hessian of the negative log-posterior via forward finite differences on the gradient.
///
/// H_{ij} = (grad_i(theta + h*e_j) - grad_i(theta)) / h, then symmetrized.
fn hessian_fd<M: LogDensityModel + ?Sized>(
    posterior: &Posterior<M>,
    params: &[f64],
    h: f64,
) -> ns_core::Result<Vec<Vec<f64>>> {
    let dim = params.len();
    let mut hess = vec![vec![0.0; dim]; dim];
    // grad of negative log-posterior at params
    let g0 = {
        let mut g = posterior.grad(params)?;
        for gi in g.iter_mut() {
            *gi = -*gi;
        }
        g
    };
    for i in 0..dim {
        let eps = h * params[i].abs().max(1.0);
        let mut p_plus = params.to_vec();
        p_plus[i] += eps;
        let g_plus = {
            let mut g = posterior.grad(&p_plus)?;
            for gi in g.iter_mut() {
                *gi = -*gi;
            }
            g
        };
        for j in 0..dim {
            hess[i][j] = (g_plus[j] - g0[j]) / eps;
        }
    }
    // Symmetrize
    for i in 0..dim {
        for j in (i + 1)..dim {
            let avg = 0.5 * (hess[i][j] + hess[j][i]);
            hess[i][j] = avg;
            hess[j][i] = avg;
        }
    }
    Ok(hess)
}

/// Invert a symmetric positive-definite matrix using Cholesky decomposition (nalgebra).
///
/// Returns `None` if the matrix is not positive definite (even after damping attempts).
fn invert_spd(mat: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    use nalgebra::DMatrix;

    let n = mat.len();
    if n == 0 {
        return Some(vec![]);
    }

    let mut h = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            h[(i, j)] = mat[i][j];
        }
    }

    let identity = DMatrix::identity(n, n);

    // Try Cholesky, with progressive damping if needed.
    let diag_scale = (0..n).map(|i| h[(i, i)].abs()).fold(0.0_f64, f64::max).max(1.0);
    let mut h_damped = h.clone();
    let mut damping = 0.0_f64;
    let max_attempts = 10;

    for attempt in 0..max_attempts {
        if let Some(chol) = nalgebra::linalg::Cholesky::new(h_damped.clone()) {
            let inv = chol.solve(&identity);
            // Convert back to Vec<Vec<f64>>
            let mut result = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    result[i][j] = inv[(i, j)];
                }
            }
            return Some(result);
        }
        if attempt + 1 == max_attempts {
            break;
        }
        let next_damping = if damping == 0.0 { diag_scale * 1e-9 } else { damping * 10.0 };
        let add = next_damping - damping;
        for i in 0..n {
            h_damped[(i, i)] += add;
        }
        damping = next_damping;
    }

    // Fall back to LU
    let inv = h_damped.lu().try_inverse()?;
    for i in 0..n {
        let v = inv[(i, i)];
        if !(v.is_finite() && v > 0.0) {
            return None;
        }
    }
    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            result[i][j] = inv[(i, j)];
        }
    }
    Some(result)
}

/// Compute MAP (Maximum A Posteriori) estimate for a model with priors.
///
/// This optimizes the posterior: `argmin_theta [ NLL(theta) - log_prior(theta) ]`
/// using L-BFGS-B with analytical gradients.
///
/// # Arguments
/// * `posterior` - Posterior object (model + priors)
/// * `config` - MAP configuration
///
/// # Returns
/// `MapResult` with parameter estimates, standard errors, and diagnostics.
pub fn map_estimate<M: LogDensityModel>(
    posterior: &Posterior<M>,
    config: &MapConfig,
) -> ns_core::Result<MapResult> {
    let dim = posterior.dim();

    // 1. Initial parameters
    let init = match &config.init {
        Some(v) => {
            if v.len() != dim {
                return Err(ns_core::Error::Validation(format!(
                    "map_estimate: init length {} != model.dim() {}",
                    v.len(),
                    dim
                )));
            }
            v.clone()
        }
        None => posterior.model().parameter_init(),
    };

    // 2. Bounds
    let bounds = posterior.model().parameter_bounds();

    // 3. Objective: minimize negative log-posterior
    let objective = MapObjective { posterior };

    let opt_config = OptimizerConfig {
        max_iter: config.max_iter as u64,
        tol: config.tol,
        m: config.m,
        smooth_bounds: false,
    };
    let optimizer = LbfgsbOptimizer::new(opt_config);
    let opt_result = optimizer.minimize(&objective, &init, &bounds)?;

    let map_params = opt_result.parameters;
    let converged = opt_result.converged;
    let n_iter = opt_result.n_iter as usize;
    let nll_posterior = opt_result.fval; // negative log-posterior

    // 4. Decompose: nll_data and log_prior at MAP
    let nll_data = posterior.model().nll(&map_params)?;
    let log_prior = log_prior_at(&posterior.priors, &map_params);

    // 5. Parameter names
    let param_names = {
        let names = posterior.model().parameter_names();
        if names.is_empty() { None } else { Some(names) }
    };

    // 6. Standard errors via Hessian
    let se = if config.compute_se {
        match hessian_fd(posterior, &map_params, 1e-4) {
            Ok(hess) => {
                match invert_spd(&hess) {
                    Some(cov) => {
                        let se_vec: Vec<f64> = (0..dim)
                            .map(|i| {
                                let v = cov[i][i];
                                if v.is_finite() && v > 0.0 { v.sqrt() } else { f64::NAN }
                            })
                            .collect();
                        Some(se_vec)
                    }
                    None => {
                        log::warn!("MAP: Hessian inversion failed, standard errors unavailable");
                        None
                    }
                }
            }
            Err(e) => {
                log::warn!("MAP: Hessian computation failed: {}", e);
                None
            }
        }
    } else {
        None
    };

    // 7. Shrinkage: run MLE without priors, compute (MAP - MLE) / MLE
    let shrinkage = if config.compute_shrinkage {
        let mle = crate::MaximumLikelihoodEstimator::new();
        match mle.fit(posterior.model()) {
            Ok(fit) => {
                let shrinkage_vec: Vec<f64> = map_params
                    .iter()
                    .zip(fit.parameters.iter())
                    .map(|(&map_val, &mle_val)| {
                        if mle_val.abs() < 1e-15 {
                            // Avoid division by zero; report absolute difference instead
                            map_val - mle_val
                        } else {
                            (map_val - mle_val) / mle_val
                        }
                    })
                    .collect();
                Some(shrinkage_vec)
            }
            Err(e) => {
                log::warn!("MAP: MLE fit for shrinkage failed: {}", e);
                None
            }
        }
    } else {
        None
    };

    Ok(MapResult {
        params: map_params,
        se,
        nll_posterior,
        nll: nll_data,
        log_prior,
        n_iter,
        converged,
        param_names,
        shrinkage,
    })
}

/// MAP estimation for individual PK from population priors.
///
/// Given population PK fit results (theta, omega), compute individual
/// MAP estimates for a single subject's data using population priors.
///
/// This creates Normal priors centered on `pop_theta` with standard deviations
/// from `pop_omega`, then calls [`map_estimate`].
///
/// # Arguments
/// * `model` - Individual PK model (e.g. `OneCompartmentOralPkModel`)
/// * `pop_theta` - Population fixed effects (used as prior means)
/// * `pop_omega` - Population BSV (used as prior SDs, log-normal)
/// * `config` - MAP configuration
///
/// # Returns
/// `MapResult` with individual MAP parameter estimates.
pub fn map_individual_pk<M: LogDensityModel>(
    model: &M,
    pop_theta: &[f64],
    pop_omega: &[f64],
    config: &MapConfig,
) -> ns_core::Result<MapResult> {
    let dim = model.dim();
    if pop_theta.len() != dim {
        return Err(ns_core::Error::Validation(format!(
            "map_individual_pk: pop_theta length {} != model.dim() {}",
            pop_theta.len(),
            dim
        )));
    }
    if pop_omega.len() != dim {
        return Err(ns_core::Error::Validation(format!(
            "map_individual_pk: pop_omega length {} != model.dim() {}",
            pop_omega.len(),
            dim
        )));
    }

    // Build Normal priors from population estimates
    let priors: Vec<Prior> = pop_theta
        .iter()
        .zip(pop_omega.iter())
        .map(|(&center, &width)| {
            if width > 0.0 && width.is_finite() {
                Prior::Normal { center, width }
            } else {
                Prior::Flat
            }
        })
        .collect();

    let posterior = Posterior::new(model).with_priors(priors)?;
    map_estimate(&posterior, config)
}

#[cfg(test)]
mod map_tests {
    use super::*;
    use ns_core::traits::{LogDensityModel, PreparedModelRef};

    /// Simple 1D Gaussian model: NLL = 0.5 * (x - data_mean)^2 / data_var
    /// This has MLE at x = data_mean.
    struct GaussianModel {
        data_mean: f64,
        data_var: f64,
    }

    impl LogDensityModel for GaussianModel {
        type Prepared<'a> = PreparedModelRef<'a, Self> where Self: 'a;

        fn dim(&self) -> usize {
            1
        }

        fn parameter_names(&self) -> Vec<String> {
            vec!["mu".to_string()]
        }

        fn parameter_bounds(&self) -> Vec<(f64, f64)> {
            vec![(f64::NEG_INFINITY, f64::INFINITY)]
        }

        fn parameter_init(&self) -> Vec<f64> {
            vec![0.0]
        }

        fn nll(&self, params: &[f64]) -> ns_core::Result<f64> {
            let x = params[0];
            Ok(0.5 * (x - self.data_mean).powi(2) / self.data_var)
        }

        fn grad_nll(&self, params: &[f64]) -> ns_core::Result<Vec<f64>> {
            let x = params[0];
            Ok(vec![(x - self.data_mean) / self.data_var])
        }

        fn prepared(&self) -> Self::Prepared<'_> {
            PreparedModelRef::new(self)
        }
    }

    /// Multi-dimensional Gaussian model: NLL = 0.5 * sum_i (x_i - mu_i)^2 / sigma_i^2
    struct MultiGaussianModel {
        means: Vec<f64>,
        variances: Vec<f64>,
    }

    impl LogDensityModel for MultiGaussianModel {
        type Prepared<'a> = PreparedModelRef<'a, Self> where Self: 'a;

        fn dim(&self) -> usize {
            self.means.len()
        }

        fn parameter_names(&self) -> Vec<String> {
            (0..self.means.len()).map(|i| format!("x_{}", i)).collect()
        }

        fn parameter_bounds(&self) -> Vec<(f64, f64)> {
            vec![(f64::NEG_INFINITY, f64::INFINITY); self.means.len()]
        }

        fn parameter_init(&self) -> Vec<f64> {
            vec![0.0; self.means.len()]
        }

        fn nll(&self, params: &[f64]) -> ns_core::Result<f64> {
            let mut nll = 0.0;
            for i in 0..self.means.len() {
                let pull = (params[i] - self.means[i]) / self.variances[i].sqrt();
                nll += 0.5 * pull * pull;
            }
            Ok(nll)
        }

        fn grad_nll(&self, params: &[f64]) -> ns_core::Result<Vec<f64>> {
            let mut g = vec![0.0; self.means.len()];
            for i in 0..self.means.len() {
                g[i] = (params[i] - self.means[i]) / self.variances[i];
            }
            Ok(g)
        }

        fn prepared(&self) -> Self::Prepared<'_> {
            PreparedModelRef::new(self)
        }
    }

    #[test]
    fn test_map_gaussian() {
        // NLL = 0.5*(x - 0)^2 / 1  => data likelihood is N(0, 1)
        // Prior: N(1, 1)            => prior is N(1, 1)
        // Posterior: proportional to exp(-0.5*x^2 - 0.5*(x-1)^2)
        //         = exp(-0.5*(2*x^2 - 2*x + 1))
        //         = exp(-0.5*2*(x^2 - x + 0.25) - 0.25)
        //         = exp(-(x - 0.5)^2 - 0.25)
        // MAP at x = 0.5
        let model = GaussianModel { data_mean: 0.0, data_var: 1.0 };
        let priors = vec![Prior::Normal { center: 1.0, width: 1.0 }];
        let posterior = Posterior::new(&model).with_priors(priors).unwrap();

        let config = MapConfig::default();
        let result = map_estimate(&posterior, &config).unwrap();

        assert!(result.converged, "MAP should converge");
        assert!(
            (result.params[0] - 0.5).abs() < 1e-6,
            "MAP should be at 0.5, got {}",
            result.params[0]
        );

        // NLL at MAP: 0.5 * (0.5 - 0)^2 / 1 = 0.125
        assert!(
            (result.nll - 0.125).abs() < 1e-6,
            "NLL at MAP should be 0.125, got {}",
            result.nll
        );

        // log_prior at MAP: -0.5 * (0.5 - 1)^2 / 1 = -0.125
        assert!(
            (result.log_prior - (-0.125)).abs() < 1e-6,
            "log_prior at MAP should be -0.125, got {}",
            result.log_prior
        );
    }

    #[test]
    fn test_map_se() {
        // NLL = 0.5*(x - 0)^2 / 1 with prior N(0, 2)
        // Negative log-posterior = 0.5*x^2 + 0.5*(x/2)^2 = 0.5*x^2*(1 + 1/4) = 0.625*x^2
        // Hessian of neg-log-posterior = 1.25
        // SE = 1/sqrt(1.25) = sqrt(0.8) ~ 0.8944
        let model = GaussianModel { data_mean: 0.0, data_var: 1.0 };
        let priors = vec![Prior::Normal { center: 0.0, width: 2.0 }];
        let posterior = Posterior::new(&model).with_priors(priors).unwrap();

        let config = MapConfig { compute_se: true, ..MapConfig::default() };
        let result = map_estimate(&posterior, &config).unwrap();

        assert!(result.converged);
        assert!(result.se.is_some(), "SE should be computed");

        let se = result.se.unwrap();
        let expected_se = (0.8_f64).sqrt(); // 1/sqrt(1.25) = sqrt(1/1.25) = sqrt(0.8)
        assert!(
            (se[0] - expected_se).abs() < 1e-4,
            "SE should be {}, got {}",
            expected_se,
            se[0]
        );
    }

    #[test]
    fn test_map_shrinkage() {
        // Model: NLL = 0.5*(x - 2)^2  => MLE at x = 2
        // Strong prior: N(0, 0.01)     => prior pulls strongly to 0
        // MAP should be close to 0, MLE at 2 => shrinkage ~ (0 - 2)/2 = -1
        let model = GaussianModel { data_mean: 2.0, data_var: 1.0 };
        let priors = vec![Prior::Normal { center: 0.0, width: 0.01 }];
        let posterior = Posterior::new(&model).with_priors(priors).unwrap();

        let config = MapConfig {
            compute_shrinkage: true,
            ..MapConfig::default()
        };
        let result = map_estimate(&posterior, &config).unwrap();

        assert!(result.converged);
        assert!(result.shrinkage.is_some(), "Shrinkage should be computed");

        let shrinkage = result.shrinkage.unwrap();
        // With very strong prior (width=0.01), MAP ~ 0, MLE = 2
        // shrinkage = (MAP - MLE) / MLE ~ (0 - 2) / 2 = -1
        assert!(
            shrinkage[0] < -0.9,
            "Strong prior should give shrinkage near -1, got {}",
            shrinkage[0]
        );

        // Weak prior test: N(0, 1e6) => nearly flat => MAP ~ MLE => shrinkage ~ 0
        let priors_weak = vec![Prior::Normal { center: 0.0, width: 1e6 }];
        let posterior_weak = Posterior::new(&model).with_priors(priors_weak).unwrap();

        let result_weak = map_estimate(&posterior_weak, &config).unwrap();
        let shrinkage_weak = result_weak.shrinkage.unwrap();
        assert!(
            shrinkage_weak[0].abs() < 0.01,
            "Weak prior should give shrinkage near 0, got {}",
            shrinkage_weak[0]
        );
    }

    #[test]
    fn test_map_no_prior_equals_mle() {
        // With flat priors, MAP = MLE
        let model = MultiGaussianModel {
            means: vec![1.0, -0.5, 3.0],
            variances: vec![1.0, 2.0, 0.5],
        };
        let posterior = Posterior::new(&model); // flat priors

        let config = MapConfig {
            compute_se: false,
            ..MapConfig::default()
        };
        let result = map_estimate(&posterior, &config).unwrap();

        assert!(result.converged, "MAP should converge");

        // MAP should equal MLE = data_means
        for (i, (&map_val, &expected)) in
            result.params.iter().zip(model.means.iter()).enumerate()
        {
            assert!(
                (map_val - expected).abs() < 1e-5,
                "MAP[{}] should equal MLE {}, got {}",
                i,
                expected,
                map_val
            );
        }
    }
}
