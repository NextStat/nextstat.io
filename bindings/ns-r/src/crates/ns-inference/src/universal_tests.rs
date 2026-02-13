//! Universal (non-HEP) model tests for Phase 5 `LogDensityModel`.

use ns_core::Result;
use ns_core::traits::{LogDensityModel, PreparedModelRef};

/// 1D standard Normal negative log-likelihood: `0.5 * x^2 + const`.
///
/// This is a minimal "general statistics" model that is not tied to HistFactory.
struct StdNormal1D;

impl LogDensityModel for StdNormal1D {
    type Prepared<'a>
        = PreparedModelRef<'a, Self>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        1
    }

    fn parameter_names(&self) -> Vec<String> {
        vec!["x".to_string()]
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![(f64::NEG_INFINITY, f64::INFINITY)]
    }

    fn parameter_init(&self) -> Vec<f64> {
        vec![0.3]
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        let x = params[0];
        Ok(0.5 * x * x)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        let x = params[0];
        Ok(vec![x])
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

/// Simple 2D linear regression (Gaussian with sigma=1) with fixed data.
///
/// y_i ~ Normal(beta0 + beta1 * x_i, 1).
struct LinearRegression2D {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl LinearRegression2D {
    fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        debug_assert_eq!(x.len(), y.len());
        Self { x, y }
    }
}

impl LogDensityModel for LinearRegression2D {
    type Prepared<'a>
        = PreparedModelRef<'a, Self>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        2
    }

    fn parameter_names(&self) -> Vec<String> {
        vec!["beta0".to_string(), "beta1".to_string()]
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![(f64::NEG_INFINITY, f64::INFINITY), (f64::NEG_INFINITY, f64::INFINITY)]
    }

    fn parameter_init(&self) -> Vec<f64> {
        vec![0.0, 0.0]
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        let b0 = params[0];
        let b1 = params[1];
        let mut sse = 0.0;
        for (&x, &y) in self.x.iter().zip(self.y.iter()) {
            let r = y - (b0 + b1 * x);
            sse += r * r;
        }
        Ok(0.5 * sse)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        let b0 = params[0];
        let b1 = params[1];
        let mut g0 = 0.0;
        let mut g1 = 0.0;
        for (&x, &y) in self.x.iter().zip(self.y.iter()) {
            let r = y - (b0 + b1 * x);
            g0 -= r;
            g1 -= r * x;
        }
        Ok(vec![g0, g1])
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generic_mle_on_non_hep_model() {
        let m = StdNormal1D;
        let mle = crate::MaximumLikelihoodEstimator::new();
        let r = mle.fit(&m).unwrap();
        assert!(r.converged);
        assert!(r.parameters[0].abs() < 1e-6, "MLE should find x=0, got {}", r.parameters[0]);
    }

    #[test]
    fn test_generic_nuts_on_non_hep_model_smoke() {
        let m = StdNormal1D;
        let cfg = crate::NutsConfig {
            max_treedepth: 7,
            target_accept: 0.8,
            init_jitter: 0.0,
            init_jitter_rel: None,
            init_overdispersed_rel: None,
        };

        // Smoke: run a tiny chain and validate moments are sane.
        let chain = crate::sample_nuts(&m, 100, 200, 123, cfg).unwrap();
        let draws: Vec<f64> = chain.draws_constrained.iter().map(|d| d[0]).collect();
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        let var = draws.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / draws.len() as f64;
        assert!(mean.abs() < 0.25, "mean too far from 0: {}", mean);
        assert!(var > 0.4 && var < 1.6, "var out of range: {}", var);
    }

    #[test]
    fn test_generic_mle_on_linear_regression() {
        // Exact line y = 1 + 2x
        let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let y: Vec<f64> = x.iter().map(|&xi| 1.0 + 2.0 * xi).collect();
        let m = LinearRegression2D::new(x, y);

        let mle = crate::MaximumLikelihoodEstimator::new();
        let r = mle.fit(&m).unwrap();
        assert!(r.converged);
        assert!((r.parameters[0] - 1.0).abs() < 1e-6, "beta0 wrong: {:?}", r.parameters);
        assert!((r.parameters[1] - 2.0).abs() < 1e-6, "beta1 wrong: {:?}", r.parameters);
    }
}
