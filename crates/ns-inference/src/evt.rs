//! Extreme Value Theory models (Phase 9 Cross-Vertical).
//!
//! Two distribution families for modelling rare events and tail risk:
//!
//! - **GEV** (Generalized Extreme Value): fits block maxima.
//!   Three sub-families via shape parameter ξ:
//!   - ξ > 0: Fréchet (heavy tail) — finance, insurance
//!   - ξ = 0: Gumbel (light tail) — hydrology, wind speeds
//!   - ξ < 0: Weibull (bounded upper tail) — material strength
//!
//! - **GPD** (Generalized Pareto Distribution): fits exceedances over a threshold.
//!   Used in peaks-over-threshold (POT) analysis for reinsurance pricing,
//!   VaR/ES estimation, and flood frequency analysis.
//!
//! Both implement [`LogDensityModel`] for MLE via L-BFGS-B.
//!
//! ## Return levels
//!
//! The GEV model provides `return_level(T)` — the value exceeded on average
//! once every T blocks (e.g. 100-year flood level).

use ns_core::traits::{LogDensityModel, PreparedModelRef};
use ns_core::{Error, Result};
use ns_prob::math::exp_clamped;

// ---------------------------------------------------------------------------
// GEV (Generalized Extreme Value)
// ---------------------------------------------------------------------------

/// Generalized Extreme Value distribution model.
///
/// CDF: `G(x) = exp(-(1 + ξ·(x-μ)/σ)^(-1/ξ))` for ξ ≠ 0,
///       `G(x) = exp(-exp(-(x-μ)/σ))` for ξ = 0.
///
/// Parameters: `[mu, log_sigma, xi]`
/// - `mu` (location): unconstrained
/// - `log_sigma`: σ = exp(log_sigma) > 0
/// - `xi` (shape): unconstrained (Fréchet ξ>0, Gumbel ξ≈0, Weibull ξ<0)
#[derive(Debug, Clone)]
pub struct GevModel {
    data: Vec<f64>,
}

impl GevModel {
    /// Create a GEV model from observed block maxima.
    pub fn new(data: Vec<f64>) -> Result<Self> {
        if data.is_empty() {
            return Err(Error::Validation("data must be non-empty".to_string()));
        }
        if data.iter().any(|x| !x.is_finite()) {
            return Err(Error::Validation("data must be finite".to_string()));
        }
        Ok(Self { data })
    }

    /// Compute the T-block return level at the MLE parameters.
    ///
    /// The return level `z_T` is the value exceeded on average once every T blocks:
    /// `z_T = μ + (σ/ξ)·((-log(1-1/T))^(-ξ) - 1)` for ξ ≠ 0
    /// `z_T = μ - σ·log(-log(1-1/T))` for ξ ≈ 0
    pub fn return_level(params: &[f64], return_period: f64) -> Result<f64> {
        if params.len() != 3 {
            return Err(Error::Validation("params must have length 3".to_string()));
        }
        if return_period <= 1.0 {
            return Err(Error::Validation("return_period must be > 1".to_string()));
        }

        let mu = params[0];
        let sigma = exp_clamped(params[1]);
        let xi = params[2];
        let yp = -(-((return_period - 1.0) / return_period).ln()).ln();

        if xi.abs() < 1e-8 {
            // Gumbel limit
            Ok(mu + sigma * yp)
        } else {
            // General case: z_T = μ + (σ/ξ)·(yp^(-ξ) - 1)
            // where yp = -log(-log(1 - 1/T))
            let neg_log_p = -((return_period - 1.0) / return_period).ln();
            Ok(mu + sigma / xi * (neg_log_p.powf(-xi) - 1.0))
        }
    }
}

impl LogDensityModel for GevModel {
    type Prepared<'a>
        = PreparedModelRef<'a, Self>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        3 // mu, log_sigma, xi
    }

    fn parameter_names(&self) -> Vec<String> {
        vec!["mu".to_string(), "log_sigma".to_string(), "xi".to_string()]
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![
            (-1e6, 1e6),   // mu
            (-10.0, 10.0), // log_sigma
            (-5.0, 5.0),   // xi
        ]
    }

    fn parameter_init(&self) -> Vec<f64> {
        // Method of moments initialisation (Gumbel approximation):
        // mu ≈ mean - 0.5772·σ, σ ≈ std·√6/π
        let n = self.data.len() as f64;
        let mean = self.data.iter().sum::<f64>() / n;
        let var = self.data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = var.sqrt().max(1e-10);
        let sigma_init = std * (6.0_f64).sqrt() / std::f64::consts::PI;
        let mu_init = mean - 0.5772 * sigma_init;
        vec![mu_init, sigma_init.ln(), 0.1]
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        if params.len() != 3 {
            return Err(Error::Validation("params length mismatch".to_string()));
        }
        let mu = params[0];
        let log_sigma = params[1];
        let sigma = exp_clamped(log_sigma);
        let xi = params[2];

        if sigma <= 0.0 || !sigma.is_finite() {
            return Ok(f64::MAX);
        }

        let n = self.data.len() as f64;

        if xi.abs() < 1e-8 {
            // Gumbel case: ξ → 0
            // NLL = n·log(σ) + Σ[(x-μ)/σ + exp(-(x-μ)/σ)]
            let mut nll = n * log_sigma;
            for &x in &self.data {
                let z = (x - mu) / sigma;
                nll += z + (-z).exp();
            }
            Ok(nll)
        } else {
            // General case:
            // t_i = 1 + ξ·(x_i - μ)/σ
            // NLL = n·log(σ) + (1 + 1/ξ)·Σ log(t_i) + Σ t_i^(-1/ξ)
            // Constraint: t_i > 0 for all i.
            let mut nll = n * log_sigma;
            let inv_xi = 1.0 / xi;

            for &x in &self.data {
                let t = 1.0 + xi * (x - mu) / sigma;
                if t <= 0.0 {
                    return Ok(f64::MAX);
                }
                nll += (1.0 + inv_xi) * t.ln() + t.powf(-inv_xi);
            }
            Ok(nll)
        }
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        if params.len() != 3 {
            return Err(Error::Validation("params length mismatch".to_string()));
        }
        let mu = params[0];
        let log_sigma = params[1];
        let sigma = exp_clamped(log_sigma);
        let xi = params[2];

        if sigma <= 0.0 || !sigma.is_finite() {
            return Ok(vec![0.0; 3]);
        }

        let n = self.data.len() as f64;
        let mut g_mu = 0.0;
        let mut g_log_sigma = n; // from n·log(σ), ∂/∂log_σ = n (since σ = exp(log_σ))
        // Actually ∂(n·log_sigma)/∂log_sigma = n. But we also need the chain rule
        // for terms involving sigma.
        let mut g_xi = 0.0;

        if xi.abs() < 1e-8 {
            // Gumbel case
            // NLL = n·log_sigma + Σ[z_i + exp(-z_i)]  where z_i = (x_i - μ)/σ
            // ∂NLL/∂μ = Σ (-1/σ)·(1 - exp(-z_i))
            // ∂NLL/∂log_σ = n + Σ [-z_i + z_i·exp(-z_i)]
            //             = n + Σ z_i·(exp(-z_i) - 1)
            // ∂NLL/∂ξ: use limit as ξ→0 (handled by finite diff at the boundary)
            for &x in &self.data {
                let z = (x - mu) / sigma;
                let emz = (-z).exp();
                g_mu += (-1.0 / sigma) * (1.0 - emz);
                g_log_sigma += z * (emz - 1.0);
            }
            // For ξ gradient at ξ≈0, use numerical approximation
            let eps = 1e-5;
            let mut p_hi = params.to_vec();
            let mut p_lo = params.to_vec();
            p_hi[2] = eps;
            p_lo[2] = -eps;
            let f_hi = self.nll(&p_hi)?;
            let f_lo = self.nll(&p_lo)?;
            g_xi = (f_hi - f_lo) / (2.0 * eps);
        } else {
            // General case
            let inv_xi = 1.0 / xi;
            g_log_sigma = n; // reset; will add per-observation terms

            for &x in &self.data {
                let z = (x - mu) / sigma;
                let t = 1.0 + xi * z;
                if t <= 0.0 {
                    return Ok(vec![0.0; 3]);
                }
                let log_t = t.ln();
                let t_neg_inv_xi = t.powf(-inv_xi);

                // dt/dmu = -ξ/σ
                // dt/dlog_sigma = -ξ·z  (chain: ∂/∂log_σ of (ξ·(x-μ)/σ) = -ξ·z)
                // dt/dxi = z
                let dt_dmu = -xi / sigma;
                let dt_dlog_sigma = -xi * z;
                let dt_dxi = z;

                // ∂NLL_i/∂t = (1+1/ξ)/t + (-1/ξ)·t^(-1/ξ - 1) = (1+1/ξ)/t - t^(-1/ξ-1)/ξ
                let dnll_dt = (1.0 + inv_xi) / t - t_neg_inv_xi / (xi * t);

                g_mu += dnll_dt * dt_dmu;
                g_log_sigma += dnll_dt * dt_dlog_sigma;

                // ∂NLL_i/∂ξ has extra terms from (1+1/ξ)·log(t) and t^(-1/ξ)
                // ∂[(1+1/ξ)·log(t)]/∂ξ = -log(t)/ξ² + (1+1/ξ)·z/t
                // ∂[t^(-1/ξ)]/∂ξ = t^(-1/ξ)·[log(t)/ξ² - z/(ξ·t)]
                let term_1 = -log_t / (xi * xi) + (1.0 + inv_xi) * dt_dxi / t;
                let term_2 = t_neg_inv_xi * (log_t / (xi * xi) - dt_dxi / (xi * t));
                g_xi += term_1 + term_2;
            }
        }

        Ok(vec![g_mu, g_log_sigma, g_xi])
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

// ---------------------------------------------------------------------------
// GPD (Generalized Pareto Distribution)
// ---------------------------------------------------------------------------

/// Generalized Pareto Distribution model for exceedances over a threshold.
///
/// Given a threshold `u`, the excess `y = x - u` follows GPD:
/// - `P(Y > y) = (1 + ξ·y/σ)^(-1/ξ)` for ξ ≠ 0
/// - `P(Y > y) = exp(-y/σ)` for ξ = 0
///
/// Parameters: `[log_sigma, xi]`
/// - `log_sigma`: σ = exp(log_sigma) > 0
/// - `xi` (shape): unconstrained
///
/// The model is fitted to **exceedances** (values above the threshold),
/// i.e. `data[i] = x[i] - u` where `x[i] > u`.
#[derive(Debug, Clone)]
pub struct GpdModel {
    exceedances: Vec<f64>,
}

impl GpdModel {
    /// Create a GPD model from exceedances (must be strictly positive).
    ///
    /// `exceedances[i] = x[i] - threshold` for observations exceeding the threshold.
    pub fn new(exceedances: Vec<f64>) -> Result<Self> {
        if exceedances.is_empty() {
            return Err(Error::Validation("exceedances must be non-empty".to_string()));
        }
        if exceedances.iter().any(|&x| x <= 0.0 || !x.is_finite()) {
            return Err(Error::Validation(
                "exceedances must be strictly positive and finite".to_string(),
            ));
        }
        Ok(Self { exceedances })
    }

    /// Compute the excess quantile at probability level `p` (0 < p < 1).
    ///
    /// For risk management: `quantile(0.99)` gives the 99th percentile excess.
    pub fn quantile(params: &[f64], p: f64) -> Result<f64> {
        if params.len() != 2 {
            return Err(Error::Validation("params must have length 2".to_string()));
        }
        if !(p > 0.0 && p < 1.0) {
            return Err(Error::Validation("p must be in (0, 1)".to_string()));
        }

        let sigma = exp_clamped(params[0]);
        let xi = params[1];

        if xi.abs() < 1e-8 {
            // Exponential limit
            Ok(-sigma * (1.0 - p).ln())
        } else {
            Ok(sigma / xi * ((1.0 - p).powf(-xi) - 1.0))
        }
    }
}

impl LogDensityModel for GpdModel {
    type Prepared<'a>
        = PreparedModelRef<'a, Self>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        2 // log_sigma, xi
    }

    fn parameter_names(&self) -> Vec<String> {
        vec!["log_sigma".to_string(), "xi".to_string()]
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![
            (-10.0, 10.0), // log_sigma
            (-5.0, 5.0),   // xi
        ]
    }

    fn parameter_init(&self) -> Vec<f64> {
        // Method of moments: for GPD, E[Y] = σ/(1-ξ), Var[Y] = σ²/((1-ξ)²(1-2ξ))
        // Start with exponential (ξ=0): σ = mean(Y)
        let n = self.exceedances.len() as f64;
        let mean = self.exceedances.iter().sum::<f64>() / n;
        vec![mean.max(1e-10).ln(), 0.1]
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        if params.len() != 2 {
            return Err(Error::Validation("params length mismatch".to_string()));
        }
        let log_sigma = params[0];
        let sigma = exp_clamped(log_sigma);
        let xi = params[1];

        if sigma <= 0.0 || !sigma.is_finite() {
            return Ok(f64::MAX);
        }

        let n = self.exceedances.len() as f64;

        if xi.abs() < 1e-8 {
            // Exponential case: NLL = n·log(σ) + Σ y_i/σ
            let sum_y: f64 = self.exceedances.iter().sum();
            Ok(n * log_sigma + sum_y / sigma)
        } else {
            // General case:
            // NLL = n·log(σ) + (1 + 1/ξ)·Σ log(1 + ξ·y_i/σ)
            // Constraint: 1 + ξ·y_i/σ > 0 for all i.
            let inv_xi = 1.0 / xi;
            let mut nll = n * log_sigma;
            for &y in &self.exceedances {
                let t = 1.0 + xi * y / sigma;
                if t <= 0.0 {
                    return Ok(f64::MAX);
                }
                nll += (1.0 + inv_xi) * t.ln();
            }
            Ok(nll)
        }
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        if params.len() != 2 {
            return Err(Error::Validation("params length mismatch".to_string()));
        }
        let log_sigma = params[0];
        let sigma = exp_clamped(log_sigma);
        let xi = params[1];

        if sigma <= 0.0 || !sigma.is_finite() {
            return Ok(vec![0.0; 2]);
        }

        let n = self.exceedances.len() as f64;

        if xi.abs() < 1e-8 {
            // Exponential case
            // NLL = n·log_sigma + Σ y_i / exp(log_sigma)
            // ∂NLL/∂log_sigma = n - Σ y_i / σ
            let sum_y: f64 = self.exceedances.iter().sum();
            let g_log_sigma = n - sum_y / sigma;

            // ∂NLL/∂ξ at ξ≈0: numerical
            let eps = 1e-5;
            let mut p_hi = params.to_vec();
            let mut p_lo = params.to_vec();
            p_hi[1] = eps;
            p_lo[1] = -eps;
            let f_hi = self.nll(&p_hi)?;
            let f_lo = self.nll(&p_lo)?;
            let g_xi = (f_hi - f_lo) / (2.0 * eps);

            Ok(vec![g_log_sigma, g_xi])
        } else {
            let inv_xi = 1.0 / xi;
            let mut g_log_sigma = n;
            let mut g_xi = 0.0;

            for &y in &self.exceedances {
                let z = y / sigma;
                let t = 1.0 + xi * z;
                if t <= 0.0 {
                    return Ok(vec![0.0; 2]);
                }
                let log_t = t.ln();

                // dt/d(log_sigma) = -ξ·z  (chain rule: ∂(ξy/σ)/∂log_σ = -ξy/σ = -ξz)
                // dt/dξ = z
                let dt_dlog_sigma = -xi * z;
                let dt_dxi = z;

                // ∂NLL_i/∂t = (1+1/ξ)/t
                let dnll_dt = (1.0 + inv_xi) / t;

                g_log_sigma += dnll_dt * dt_dlog_sigma;

                // ∂NLL_i/∂ξ = -log(t)/ξ² + (1+1/ξ)·z/t
                g_xi += -log_t / (xi * xi) + (1.0 + inv_xi) * dt_dxi / t;
            }

            Ok(vec![g_log_sigma, g_xi])
        }
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn finite_diff_grad<M: LogDensityModel>(m: &M, params: &[f64], eps: f64) -> Vec<f64> {
        let mut g = vec![0.0; params.len()];
        for i in 0..params.len() {
            let mut p_hi = params.to_vec();
            let mut p_lo = params.to_vec();
            p_hi[i] += eps;
            p_lo[i] -= eps;
            let f_hi = m.nll(&p_hi).unwrap();
            let f_lo = m.nll(&p_lo).unwrap();
            g[i] = (f_hi - f_lo) / (2.0 * eps);
        }
        g
    }

    // Reference data: annual maximum temperatures (synthetic, Gumbel-like)
    fn gev_test_data() -> Vec<f64> {
        vec![
            34.2, 36.1, 33.8, 37.5, 35.0, 38.2, 34.9, 36.8, 35.5, 37.1, 39.0, 33.5, 36.5, 35.8,
            37.8, 34.5, 38.5, 36.2, 35.2, 37.3,
        ]
    }

    // ----- GEV tests -----

    #[test]
    fn gev_basic_properties() {
        let m = GevModel::new(gev_test_data()).unwrap();
        assert_eq!(m.dim(), 3);
        assert_eq!(m.parameter_names(), vec!["mu", "log_sigma", "xi"]);

        let init = m.parameter_init();
        assert_eq!(init.len(), 3);
        let nll = m.nll(&init).unwrap();
        assert!(nll.is_finite(), "NLL at init not finite: {nll}");
    }

    #[test]
    fn gev_rejects_empty_data() {
        assert!(GevModel::new(vec![]).is_err());
    }

    #[test]
    fn gev_rejects_non_finite() {
        assert!(GevModel::new(vec![1.0, f64::NAN]).is_err());
        assert!(GevModel::new(vec![1.0, f64::INFINITY]).is_err());
    }

    #[test]
    fn gev_nll_gumbel_case() {
        // ξ = 0 (Gumbel): NLL = n·log(σ) + Σ[z + exp(-z)]
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = GevModel::new(data.clone()).unwrap();

        let mu: f64 = 3.0;
        let sigma: f64 = 1.5;
        let params = [mu, sigma.ln(), 0.0];

        let nll = m.nll(&params).unwrap();
        let n = data.len() as f64;
        let expected: f64 = n * sigma.ln()
            + data
                .iter()
                .map(|&x| {
                    let z = (x - mu) / sigma;
                    z + (-z).exp()
                })
                .sum::<f64>();
        assert!((nll - expected).abs() < 1e-10, "Gumbel NLL: {nll} vs {expected}");
    }

    #[test]
    fn gev_nll_general_case() {
        let data = vec![2.0, 3.0, 5.0, 7.0, 10.0];
        let m = GevModel::new(data.clone()).unwrap();

        let mu: f64 = 3.0;
        let sigma: f64 = 2.0;
        let xi: f64 = 0.3;
        let params = [mu, sigma.ln(), xi];

        let nll = m.nll(&params).unwrap();
        assert!(nll.is_finite());

        // Verify manually
        let n = data.len() as f64;
        let mut expected = n * sigma.ln();
        for &x in &data {
            let t = 1.0 + xi * (x - mu) / sigma;
            assert!(t > 0.0);
            expected += (1.0 + 1.0 / xi) * t.ln() + t.powf(-1.0 / xi);
        }
        assert!((nll - expected).abs() < 1e-10);
    }

    #[test]
    fn gev_grad_matches_finite_diff_general() {
        let m = GevModel::new(gev_test_data()).unwrap();
        let params = [35.0, 1.0_f64.ln(), 0.2];
        let g = m.grad_nll(&params).unwrap();
        let g_fd = finite_diff_grad(&m, &params, 1e-6);

        for j in 0..3 {
            assert!(
                (g[j] - g_fd[j]).abs() < 1e-3,
                "GEV grad[{j}]: analytical={}, fd={}",
                g[j],
                g_fd[j]
            );
        }
    }

    #[test]
    fn gev_grad_matches_finite_diff_gumbel() {
        let m = GevModel::new(gev_test_data()).unwrap();
        let params = [35.0, 1.0_f64.ln(), 0.0]; // ξ = 0
        let g = m.grad_nll(&params).unwrap();
        let g_fd = finite_diff_grad(&m, &params, 1e-6);

        for j in 0..3 {
            assert!(
                (g[j] - g_fd[j]).abs() < 1e-3,
                "GEV Gumbel grad[{j}]: analytical={}, fd={}",
                g[j],
                g_fd[j]
            );
        }
    }

    #[test]
    fn gev_grad_negative_xi() {
        let m = GevModel::new(gev_test_data()).unwrap();
        let params = [36.0, 0.5_f64.ln(), -0.2]; // Weibull-type
        let g = m.grad_nll(&params).unwrap();
        let g_fd = finite_diff_grad(&m, &params, 1e-6);

        for j in 0..3 {
            assert!(
                (g[j] - g_fd[j]).abs() < 1e-3,
                "GEV ξ<0 grad[{j}]: analytical={}, fd={}",
                g[j],
                g_fd[j]
            );
        }
    }

    #[test]
    fn gev_return_level() {
        // For Gumbel (ξ→0): z_T = μ - σ·log(-log(1-1/T))
        let mu: f64 = 35.0;
        let sigma: f64 = 1.5;
        let params = [mu, sigma.ln(), 0.0];

        let z100 = GevModel::return_level(&params, 100.0).unwrap();
        let expected = mu - sigma * (-(1.0 - 1.0 / 100.0_f64).ln()).ln();
        assert!((z100 - expected).abs() < 1e-6);

        // Return period 1 should fail
        assert!(GevModel::return_level(&params, 1.0).is_err());
    }

    #[test]
    fn gev_infeasible_returns_max() {
        // If 1 + ξ·(x-μ)/σ ≤ 0, NLL should be MAX.
        // mu=25, sigma=1, xi=-0.5 → upper bound = mu - sigma/xi = 27.
        // For x=30: t = 1 + (-0.5)(30-25)/1 = -1.5 < 0 → infeasible.
        let data = vec![10.0, 20.0, 30.0];
        let m = GevModel::new(data).unwrap();
        let params = [25.0, 0.0, -0.5];
        let nll = m.nll(&params).unwrap();
        assert_eq!(nll, f64::MAX);
    }

    // ----- GPD tests -----

    fn gpd_test_data() -> Vec<f64> {
        // Synthetic exceedances (positive)
        vec![
            0.5, 1.2, 0.3, 2.1, 0.8, 1.5, 3.2, 0.6, 1.8, 0.4, 2.5, 0.9, 1.1, 0.7, 4.0, 1.3, 0.2,
            2.8, 1.6, 0.5,
        ]
    }

    #[test]
    fn gpd_basic_properties() {
        let m = GpdModel::new(gpd_test_data()).unwrap();
        assert_eq!(m.dim(), 2);
        assert_eq!(m.parameter_names(), vec!["log_sigma", "xi"]);

        let init = m.parameter_init();
        let nll = m.nll(&init).unwrap();
        assert!(nll.is_finite());
    }

    #[test]
    fn gpd_rejects_non_positive() {
        assert!(GpdModel::new(vec![1.0, 0.0, 2.0]).is_err());
        assert!(GpdModel::new(vec![1.0, -0.5]).is_err());
    }

    #[test]
    fn gpd_rejects_empty() {
        assert!(GpdModel::new(vec![]).is_err());
    }

    #[test]
    fn gpd_nll_exponential_case() {
        // ξ = 0 (exponential): NLL = n·log(σ) + Σ y/σ
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = GpdModel::new(data.clone()).unwrap();
        let sigma: f64 = 2.5;
        let params = [sigma.ln(), 0.0];
        let nll = m.nll(&params).unwrap();

        let n = data.len() as f64;
        let sum_y: f64 = data.iter().sum();
        let expected = n * sigma.ln() + sum_y / sigma;
        assert!((nll - expected).abs() < 1e-10);
    }

    #[test]
    fn gpd_grad_matches_finite_diff_general() {
        let m = GpdModel::new(gpd_test_data()).unwrap();
        let params = [0.5_f64.ln(), 0.3];
        let g = m.grad_nll(&params).unwrap();
        let g_fd = finite_diff_grad(&m, &params, 1e-6);

        for j in 0..2 {
            assert!(
                (g[j] - g_fd[j]).abs() < 1e-3,
                "GPD grad[{j}]: analytical={}, fd={}",
                g[j],
                g_fd[j]
            );
        }
    }

    #[test]
    fn gpd_grad_matches_finite_diff_exponential() {
        let m = GpdModel::new(gpd_test_data()).unwrap();
        let params = [1.0_f64.ln(), 0.0];
        let g = m.grad_nll(&params).unwrap();
        let g_fd = finite_diff_grad(&m, &params, 1e-6);

        for j in 0..2 {
            assert!(
                (g[j] - g_fd[j]).abs() < 1e-3,
                "GPD exp grad[{j}]: analytical={}, fd={}",
                g[j],
                g_fd[j]
            );
        }
    }

    #[test]
    fn gpd_grad_negative_xi() {
        let m = GpdModel::new(gpd_test_data()).unwrap();
        let params = [1.5_f64.ln(), -0.2];
        let g = m.grad_nll(&params).unwrap();
        let g_fd = finite_diff_grad(&m, &params, 1e-6);

        for j in 0..2 {
            assert!(
                (g[j] - g_fd[j]).abs() < 1e-3,
                "GPD ξ<0 grad[{j}]: analytical={}, fd={}",
                g[j],
                g_fd[j]
            );
        }
    }

    #[test]
    fn gpd_quantile() {
        // Exponential: Q(p) = -σ·log(1-p)
        let sigma: f64 = 2.0;
        let params = [sigma.ln(), 0.0];
        let q99 = GpdModel::quantile(&params, 0.99).unwrap();
        let expected = -sigma * (0.01_f64).ln();
        assert!((q99 - expected).abs() < 1e-6);

        assert!(GpdModel::quantile(&params, 0.0).is_err());
        assert!(GpdModel::quantile(&params, 1.0).is_err());
    }

    #[test]
    fn gpd_infeasible_returns_max() {
        let data = vec![1.0, 5.0, 10.0];
        let m = GpdModel::new(data).unwrap();
        // log_sigma = -1.0, sigma = exp(-1) ≈ 0.368, xi = -0.5
        // For y=10: t = 1 + (-0.5)*10/0.368 = 1 - 13.59 < 0 → infeasible.
        let params = [-1.0, -0.5];
        let nll = m.nll(&params).unwrap();
        assert_eq!(nll, f64::MAX);
    }
}
