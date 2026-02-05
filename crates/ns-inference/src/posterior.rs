//! Posterior distribution for Bayesian inference.
//!
//! Wraps a [`HistFactoryModel`] and provides log-posterior density and
//! gradient in both constrained and unconstrained parameterizations.
//!
//! The model's NLL already includes Gaussian constraints. The [`Posterior`]
//! adds only user-specified additional priors (if any).

use crate::transforms::ParameterTransform;
use ns_core::Result;
use ns_translate::pyhf::HistFactoryModel;

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
pub struct Posterior<'a> {
    model: &'a HistFactoryModel,
    transform: ParameterTransform,
    priors: Vec<Prior>,
}

impl<'a> Posterior<'a> {
    /// Create a new posterior with flat priors (relying on model's built-in constraints).
    pub fn new(model: &'a HistFactoryModel) -> Self {
        let bounds: Vec<(f64, f64)> = model.parameters().iter().map(|p| p.bounds).collect();
        let transform = ParameterTransform::from_bounds(&bounds);
        let priors = vec![Prior::Flat; model.n_params()];
        Self { model, transform, priors }
    }

    /// Set priors (one per parameter). Length must match `model.n_params()`.
    pub fn with_priors(mut self, priors: Vec<Prior>) -> Self {
        assert_eq!(priors.len(), self.model.n_params());
        self.priors = priors;
        self
    }

    /// Number of parameters.
    pub fn dim(&self) -> usize {
        self.model.n_params()
    }

    /// Reference to the underlying model.
    pub fn model(&self) -> &HistFactoryModel {
        self.model
    }

    /// Reference to the parameter transform.
    pub fn transform(&self) -> &ParameterTransform {
        &self.transform
    }

    /// Log-posterior in constrained space: `-model.nll(theta) + sum(prior_logpdf)`.
    pub fn logpdf(&self, theta: &[f64]) -> Result<f64> {
        let nll = self.model.nll(theta)?;
        let mut lp = -nll;

        for (i, prior) in self.priors.iter().enumerate() {
            match prior {
                Prior::Flat => {}
                Prior::Normal { center, width } => {
                    let pull = (theta[i] - center) / width;
                    lp -= 0.5 * pull * pull;
                }
            }
        }

        Ok(lp)
    }

    /// Gradient of log-posterior in constrained space.
    pub fn grad(&self, theta: &[f64]) -> Result<Vec<f64>> {
        let mut g = self.model.gradient_reverse(theta)?;

        // grad(logpdf) = -grad(nll) + grad(prior)
        for gi in g.iter_mut() {
            *gi = -*gi;
        }

        for (i, prior) in self.priors.iter().enumerate() {
            match prior {
                Prior::Flat => {}
                Prior::Normal { center, width } => {
                    g[i] -= (theta[i] - center) / (width * width);
                }
            }
        }

        Ok(g)
    }

    /// Log-posterior in unconstrained space: `logpdf(transform(z)) + log|J(z)|`.
    pub fn logpdf_unconstrained(&self, z: &[f64]) -> Result<f64> {
        let theta = self.transform.forward(z);
        let lp = self.logpdf(&theta)?;
        let log_jac = self.transform.log_abs_det_jacobian(z);
        Ok(lp + log_jac)
    }

    /// Gradient of log-posterior in unconstrained space.
    ///
    /// Chain rule (diagonal Jacobian):
    /// `grad_z[i] = (dtheta_i/dz_i) * grad_theta[i] + d/dz_i log|J_i|`
    pub fn grad_unconstrained(&self, z: &[f64]) -> Result<Vec<f64>> {
        let theta = self.transform.forward(z);
        let grad_theta = self.grad(&theta)?;
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

    /// Map constrained -> unconstrained.
    pub fn to_unconstrained(&self, theta: &[f64]) -> Vec<f64> {
        self.transform.inverse(theta)
    }

    /// Map unconstrained -> constrained.
    pub fn to_constrained(&self, z: &[f64]) -> Vec<f64> {
        self.transform.forward(z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let z = posterior.to_unconstrained(&theta);
        let theta_back = posterior.to_constrained(&z);

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
        let z = posterior.to_unconstrained(&theta);
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
}
