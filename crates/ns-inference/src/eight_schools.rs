//! Eight Schools model (non-centered parameterization).
//!
//! A classic hierarchical model from Rubin (1981): J schools with treatment effects
//! `y_j ± sigma_j`. The non-centered parameterization avoids the funnel geometry
//! that plagues centered parameterizations in NUTS.
//!
//! Parameters: `[mu, tau, theta_raw_0, ..., theta_raw_{J-1}]`
//! - `mu` — overall mean (unbounded)
//! - `tau` — between-school std deviation (> 0)
//! - `theta_raw_j` — standardized school effects (unbounded)
//!
//! Derived: `theta_j = mu + tau * theta_raw_j`

use ns_core::Result;
use ns_core::traits::{LogDensityModel, PreparedModelRef};

/// Eight Schools hierarchical model (non-centered).
#[derive(Clone, Debug)]
pub struct EightSchoolsModel {
    /// Observed treatment effects per school.
    y: Vec<f64>,
    /// Known standard errors per school.
    sigma: Vec<f64>,
    /// Precomputed 1/sigma_j^2.
    inv_var: Vec<f64>,
    /// Prior std for mu: Normal(0, prior_mu_sigma).
    prior_mu_sigma: f64,
    /// Prior scale for tau: HalfCauchy(0, prior_tau_scale).
    prior_tau_scale: f64,
}

impl EightSchoolsModel {
    /// Create a new Eight Schools model.
    ///
    /// `y` and `sigma` must have the same length (J schools). `sigma` values must be positive.
    pub fn new(
        y: Vec<f64>,
        sigma: Vec<f64>,
        prior_mu_sigma: f64,
        prior_tau_scale: f64,
    ) -> Result<Self> {
        let j = y.len();
        if sigma.len() != j {
            return Err(ns_core::Error::Validation(format!(
                "y and sigma must have the same length, got {} and {}",
                j,
                sigma.len()
            )));
        }
        if j == 0 {
            return Err(ns_core::Error::Validation("at least one school required".into()));
        }
        for (i, &s) in sigma.iter().enumerate() {
            if !s.is_finite() || s <= 0.0 {
                return Err(ns_core::Error::Validation(format!(
                    "sigma[{}] must be finite and > 0, got {}",
                    i, s
                )));
            }
        }
        if !prior_mu_sigma.is_finite() || prior_mu_sigma <= 0.0 {
            return Err(ns_core::Error::Validation(format!(
                "prior_mu_sigma must be finite and > 0, got {}",
                prior_mu_sigma
            )));
        }
        if !prior_tau_scale.is_finite() || prior_tau_scale <= 0.0 {
            return Err(ns_core::Error::Validation(format!(
                "prior_tau_scale must be finite and > 0, got {}",
                prior_tau_scale
            )));
        }
        let inv_var: Vec<f64> = sigma.iter().map(|s| 1.0 / (s * s)).collect();
        Ok(Self { y, sigma, inv_var, prior_mu_sigma, prior_tau_scale })
    }

    /// Number of schools.
    pub fn n_schools(&self) -> usize {
        self.y.len()
    }
}

impl LogDensityModel for EightSchoolsModel {
    type Prepared<'a>
        = PreparedModelRef<'a, Self>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        // mu, tau, theta_raw_0..J-1
        2 + self.y.len()
    }

    fn parameter_names(&self) -> Vec<String> {
        let j = self.y.len();
        let mut names = Vec::with_capacity(2 + j);
        names.push("mu".into());
        names.push("tau".into());
        for i in 0..j {
            names.push(format!("theta_raw[{}]", i));
        }
        names
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        let j = self.y.len();
        let mut b = Vec::with_capacity(2 + j);
        b.push((f64::NEG_INFINITY, f64::INFINITY)); // mu
        b.push((0.0, f64::INFINITY)); // tau > 0
        for _ in 0..j {
            b.push((f64::NEG_INFINITY, f64::INFINITY)); // theta_raw_j
        }
        b
    }

    fn parameter_init(&self) -> Vec<f64> {
        let j = self.y.len();
        let mut p = Vec::with_capacity(2 + j);
        p.push(0.0); // mu
        p.push(1.0); // tau
        for _ in 0..j {
            p.push(0.0); // theta_raw_j
        }
        p
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        let j = self.y.len();
        if params.len() != 2 + j {
            return Err(ns_core::Error::Validation(format!(
                "expected {} parameters, got {}",
                2 + j,
                params.len()
            )));
        }

        let mu = params[0];
        let tau = params[1];

        if !tau.is_finite() || tau <= 0.0 {
            return Err(ns_core::Error::Validation(format!(
                "tau must be finite and > 0, got {}",
                tau
            )));
        }

        let mut nll = 0.0;

        // Likelihood: y_j ~ Normal(mu + tau * theta_raw_j, sigma_j)
        for i in 0..j {
            let theta_raw = params[2 + i];
            let theta = mu + tau * theta_raw;
            let r = self.y[i] - theta;
            nll += 0.5 * r * r * self.inv_var[i];
        }

        // Prior on theta_raw_j: Normal(0, 1)
        for i in 0..j {
            let theta_raw = params[2 + i];
            nll += 0.5 * theta_raw * theta_raw;
        }

        // Prior on mu: Normal(0, prior_mu_sigma)
        nll += 0.5 * (mu / self.prior_mu_sigma).powi(2);

        // Prior on tau: HalfCauchy(0, prior_tau_scale)
        // log-density: -log(pi/2 * scale * (1 + (tau/scale)^2))
        // NLL contribution: log(1 + (tau/scale)^2) + constants (dropped)
        let t = tau / self.prior_tau_scale;
        nll += (1.0 + t * t).ln();

        Ok(nll)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        let j = self.y.len();
        if params.len() != 2 + j {
            return Err(ns_core::Error::Validation(format!(
                "expected {} parameters, got {}",
                2 + j,
                params.len()
            )));
        }

        let mu = params[0];
        let tau = params[1];

        let mut grad = vec![0.0; 2 + j];

        // d/d(theta_raw_j):
        //   from likelihood: -(y_j - mu - tau*theta_raw_j) * tau / sigma_j^2
        //   from N(0,1) prior: theta_raw_j
        for i in 0..j {
            let theta_raw = params[2 + i];
            let theta = mu + tau * theta_raw;
            let r = self.y[i] - theta; // y_j - theta_j
            // d NLL / d theta_raw_j = -r * tau * inv_var_j + theta_raw_j
            grad[2 + i] = -r * tau * self.inv_var[i] + theta_raw;
        }

        // d/d(mu):
        //   from likelihood: -sum (y_j - theta_j) * inv_var_j
        //   from mu prior: mu / prior_mu_sigma^2
        let mut d_mu = 0.0;
        for i in 0..j {
            let theta_raw = params[2 + i];
            let theta = mu + tau * theta_raw;
            let r = self.y[i] - theta;
            d_mu -= r * self.inv_var[i];
        }
        d_mu += mu / (self.prior_mu_sigma * self.prior_mu_sigma);
        grad[0] = d_mu;

        // d/d(tau):
        //   from likelihood: -sum (y_j - theta_j) * theta_raw_j * inv_var_j
        //   from HalfCauchy: 2*tau / (prior_tau_scale^2 + tau^2)
        let mut d_tau = 0.0;
        for i in 0..j {
            let theta_raw = params[2 + i];
            let theta = mu + tau * theta_raw;
            let r = self.y[i] - theta;
            d_tau -= r * theta_raw * self.inv_var[i];
        }
        let s2 = self.prior_tau_scale * self.prior_tau_scale;
        d_tau += 2.0 * tau / (s2 + tau * tau);
        grad[1] = d_tau;

        Ok(grad)
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eight_schools_data() -> (Vec<f64>, Vec<f64>) {
        let y = vec![28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0];
        let sigma = vec![15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0];
        (y, sigma)
    }

    #[test]
    fn test_construction() {
        let (y, sigma) = eight_schools_data();
        let m = EightSchoolsModel::new(y, sigma, 5.0, 5.0).unwrap();
        assert_eq!(m.dim(), 10);
        assert_eq!(m.n_schools(), 8);
        assert_eq!(m.parameter_names().len(), 10);
        assert_eq!(m.parameter_names()[0], "mu");
        assert_eq!(m.parameter_names()[1], "tau");
        assert_eq!(m.parameter_names()[2], "theta_raw[0]");
    }

    #[test]
    fn test_nll_at_init() {
        let (y, sigma) = eight_schools_data();
        let m = EightSchoolsModel::new(y, sigma, 5.0, 5.0).unwrap();
        let init = m.parameter_init();
        let nll = m.nll(&init).unwrap();
        assert!(nll.is_finite());
    }

    #[test]
    fn test_gradient_finite_difference() {
        let (y, sigma) = eight_schools_data();
        let m = EightSchoolsModel::new(y, sigma, 5.0, 5.0).unwrap();

        // Evaluate at a non-trivial point
        let mut params = m.parameter_init();
        params[0] = 3.0; // mu
        params[1] = 2.0; // tau
        params[2] = 0.5; // theta_raw[0]
        params[3] = -0.3; // theta_raw[1]

        let grad = m.grad_nll(&params).unwrap();
        let eps = 1e-6;
        for i in 0..params.len() {
            let mut p_plus = params.clone();
            let mut p_minus = params.clone();
            p_plus[i] += eps;
            p_minus[i] -= eps;
            // Skip if lower bound violation
            if p_minus[i] <= 0.0 && i == 1 {
                p_minus[i] = eps;
                p_plus[i] = params[i] + eps;
                let fd =
                    (m.nll(&p_plus).unwrap() - m.nll(&p_minus).unwrap()) / (p_plus[i] - p_minus[i]);
                let err = (grad[i] - fd).abs();
                assert!(
                    err < 1e-4,
                    "gradient mismatch at param[{}]: analytic={:.8}, fd={:.8}, err={:.2e}",
                    i,
                    grad[i],
                    fd,
                    err
                );
                continue;
            }
            let fd = (m.nll(&p_plus).unwrap() - m.nll(&p_minus).unwrap()) / (2.0 * eps);
            let err = (grad[i] - fd).abs();
            assert!(
                err < 1e-4,
                "gradient mismatch at param[{}]: analytic={:.8}, fd={:.8}, err={:.2e}",
                i,
                grad[i],
                fd,
                err
            );
        }
    }

    #[test]
    fn test_validation_errors() {
        // Mismatched lengths
        assert!(EightSchoolsModel::new(vec![1.0], vec![1.0, 2.0], 5.0, 5.0).is_err());
        // Empty
        assert!(EightSchoolsModel::new(vec![], vec![], 5.0, 5.0).is_err());
        // Negative sigma
        assert!(EightSchoolsModel::new(vec![1.0], vec![-1.0], 5.0, 5.0).is_err());
        // Zero prior scale
        assert!(EightSchoolsModel::new(vec![1.0], vec![1.0], 0.0, 5.0).is_err());
    }

    #[test]
    fn test_nuts_sampling() {
        use crate::nuts::{NutsConfig, sample_nuts};
        let (y, sigma) = eight_schools_data();
        let m = EightSchoolsModel::new(y, sigma, 5.0, 5.0).unwrap();
        let config = NutsConfig { max_treedepth: 8, target_accept: 0.8, ..Default::default() };
        let chain = sample_nuts(&m, 200, 100, 42, config).unwrap();
        assert_eq!(chain.draws_constrained.len(), 100);
        assert_eq!(chain.draws_constrained[0].len(), 10);
    }
}
