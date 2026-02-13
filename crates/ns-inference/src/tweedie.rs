//! Gamma and Tweedie GLM families (Phase 9 Cross-Vertical).
//!
//! Two GLM families for non-negative continuous data with non-constant variance:
//!
//! - **Gamma GLM**: `Y ~ Gamma(shape=α, rate=α/μ)`, log link `log(μ) = Xβ`.
//!   Suitable for strictly positive continuous responses (insurance claim amounts,
//!   hospital costs, survival times).
//!
//! - **Tweedie GLM**: `Y ~ Tweedie(μ, φ, p)` with power parameter `p ∈ (1, 2)`,
//!   log link. Handles exact zeros (compound Poisson-Gamma).
//!   Suitable for insurance aggregate claims, rainfall data, zero-inflated positive
//!   continuous responses.
//!
//! Both implement [`LogDensityModel`] for use with `MaximumLikelihoodEstimator` and
//! `sample_nuts`.

use crate::regression::{DenseX, row_dot, validate_xy_dims};
use ns_core::traits::{LogDensityModel, PreparedModelRef};
use ns_core::{Error, Result};
use ns_prob::math::exp_clamped;
use statrs::function::gamma::{digamma, ln_gamma};

// ---------------------------------------------------------------------------
// Gamma GLM (log link)
// ---------------------------------------------------------------------------

/// Gamma GLM with log link.
///
/// Model:
/// - `Y_i ~ Gamma(shape = α, rate = α / μ_i)`
/// - `log(μ_i) = X_i β` (intercept column must be in X if desired)
///
/// Parameters: `[β_0, ..., β_{p-1}, log_alpha]`
///
/// The shape parameter `α = exp(log_alpha)` is shared across all observations.
/// Larger α → lower variance relative to the mean (coefficient of variation = 1/√α).
#[derive(Debug, Clone)]
pub struct GammaRegressionModel {
    x: DenseX,
    y: Vec<f64>,
    include_intercept: bool,
}

impl GammaRegressionModel {
    /// Create a Gamma GLM.
    ///
    /// - `x`: design matrix as Vec of row vectors (each row = one observation).
    /// - `y`: response values (must be strictly positive).
    /// - `include_intercept`: if true, prepends a column of ones to X.
    pub fn new(x: Vec<Vec<f64>>, y: Vec<f64>, include_intercept: bool) -> Result<Self> {
        let n = x.len();
        let p_raw = x.first().map(|r| r.len()).unwrap_or(0);

        if y.iter().any(|&v| v <= 0.0 || !v.is_finite()) {
            return Err(Error::Validation(
                "Gamma GLM: y must be strictly positive and finite".to_string(),
            ));
        }

        let x = if include_intercept {
            let mut augmented = Vec::with_capacity(n);
            for row in x {
                let mut new_row = Vec::with_capacity(row.len() + 1);
                new_row.push(1.0);
                new_row.extend(row);
                augmented.push(new_row);
            }
            augmented
        } else {
            x
        };

        let p = if include_intercept { p_raw + 1 } else { p_raw };
        validate_xy_dims(n, p, n * p, y.len())?;

        Ok(Self { x: DenseX::from_rows(x)?, y, include_intercept })
    }

    fn n_beta(&self) -> usize {
        self.x.p
    }
}

impl LogDensityModel for GammaRegressionModel {
    type Prepared<'a>
        = PreparedModelRef<'a, Self>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        self.n_beta() + 1 // β + log_alpha
    }

    fn parameter_names(&self) -> Vec<String> {
        let mut names: Vec<String> = if self.include_intercept {
            let mut v = vec!["intercept".to_string()];
            for j in 1..self.n_beta() {
                v.push(format!("beta_{}", j - 1));
            }
            v
        } else {
            (0..self.n_beta()).map(|j| format!("beta_{j}")).collect()
        };
        names.push("log_alpha".to_string());
        names
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        let mut bounds = vec![(-20.0, 20.0); self.n_beta()];
        bounds.push((-10.0, 10.0)); // log_alpha
        bounds
    }

    fn parameter_init(&self) -> Vec<f64> {
        let mut init = vec![0.0; self.n_beta()];
        // Initialize intercept to log(mean(y)) if intercept is included.
        if self.include_intercept {
            let mean_y = self.y.iter().sum::<f64>() / self.y.len() as f64;
            init[0] = mean_y.max(1e-10).ln();
        }
        init.push(0.0); // log_alpha = 0 → alpha = 1
        init
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        if params.len() != self.dim() {
            return Err(Error::Validation("params length mismatch".to_string()));
        }
        let beta = &params[..self.n_beta()];
        let log_alpha = params[self.n_beta()];
        let alpha = exp_clamped(log_alpha);

        if !alpha.is_finite() || alpha <= 0.0 {
            return Ok(f64::MAX);
        }

        let lg_alpha = ln_gamma(alpha);
        let n = self.x.n;
        let mut nll = 0.0;

        for i in 0..n {
            let eta = row_dot(self.x.row(i), beta);
            let log_mu = eta;
            let mu = exp_clamped(eta);
            if mu <= 0.0 || !mu.is_finite() {
                return Ok(f64::MAX);
            }
            let yi = self.y[i];

            // NLL_i = -[α·log(α/μ) + (α-1)·log(y) - α·y/μ - lgamma(α)]
            //       = -α·log(α) + α·log(μ) - (α-1)·log(y) + α·y/μ + lgamma(α)
            nll += -alpha * log_alpha + alpha * log_mu - (alpha - 1.0) * yi.ln()
                + alpha * yi / mu
                + lg_alpha;
        }

        Ok(nll)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        if params.len() != self.dim() {
            return Err(Error::Validation("params length mismatch".to_string()));
        }
        let beta = &params[..self.n_beta()];
        let log_alpha = params[self.n_beta()];
        let alpha = exp_clamped(log_alpha);

        if !alpha.is_finite() || alpha <= 0.0 {
            return Ok(vec![0.0; self.dim()]);
        }

        let psi_alpha = digamma(alpha);
        let n = self.x.n;
        let nb = self.n_beta();
        let mut grad = vec![0.0; self.dim()];

        for i in 0..n {
            let row = self.x.row(i);
            let eta = row_dot(row, beta);
            let mu = exp_clamped(eta);
            if mu <= 0.0 || !mu.is_finite() {
                continue;
            }
            let yi = self.y[i];

            // ∂NLL_i/∂β_j = α · (1 - y_i/μ_i) · x_{ij}
            let common_beta = alpha * (1.0 - yi / mu);
            for j in 0..nb {
                grad[j] += common_beta * row[j];
            }

            // ∂NLL_i/∂log_α = α · [-log(α) - 1 + log(μ) - log(y) + y/μ + ψ(α)]
            let grad_log_alpha = alpha * (-log_alpha - 1.0 + eta - yi.ln() + yi / mu + psi_alpha);
            grad[nb] += grad_log_alpha;
        }

        Ok(grad)
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

// ---------------------------------------------------------------------------
// Tweedie GLM (log link, power p ∈ (1, 2))
// ---------------------------------------------------------------------------

/// Tweedie GLM with log link.
///
/// Model:
/// - `Y_i ~ Tweedie(μ_i, φ, p)` with `Var(Y) = φ · μ^p`
/// - `log(μ_i) = X_i β`
/// - Power `p ∈ (1, 2)`: compound Poisson-Gamma, handles exact zeros
///
/// Parameters: `[β_0, ..., β_{k-1}, log_phi]` (p is fixed at construction)
///
/// NLL uses the saddle-point approximation (Dunn & Smyth 2005, Jørgensen 1997):
/// - For `y > 0`: `NLL_i = 0.5·log(2πφy^p) + d(y,μ)/(2φ)`
/// - For `y = 0`: `NLL_i = μ^(2-p) / (φ·(2-p))`
///
/// where `d(y, μ)` is the Tweedie unit deviance.
#[derive(Debug, Clone)]
pub struct TweedieRegressionModel {
    x: DenseX,
    y: Vec<f64>,
    include_intercept: bool,
    /// Fixed power parameter, must be in (1, 2).
    p: f64,
}

impl TweedieRegressionModel {
    /// Create a Tweedie GLM.
    ///
    /// - `x`: design matrix as Vec of row vectors.
    /// - `y`: response values (must be non-negative).
    /// - `include_intercept`: if true, prepends a column of ones.
    /// - `p`: Tweedie power parameter, must be in (1, 2) exclusive.
    ///   Common choices: 1.5 (insurance), 1.6-1.8 (rainfall).
    pub fn new(x: Vec<Vec<f64>>, y: Vec<f64>, include_intercept: bool, p: f64) -> Result<Self> {
        let n = x.len();
        let p_raw = x.first().map(|r| r.len()).unwrap_or(0);

        if !(p > 1.0 && p < 2.0) {
            return Err(Error::Validation("Tweedie power p must be in (1, 2)".to_string()));
        }
        if y.iter().any(|&v| v < 0.0 || !v.is_finite()) {
            return Err(Error::Validation(
                "Tweedie GLM: y must be non-negative and finite".to_string(),
            ));
        }

        let x = if include_intercept {
            let mut augmented = Vec::with_capacity(n);
            for row in x {
                let mut new_row = Vec::with_capacity(row.len() + 1);
                new_row.push(1.0);
                new_row.extend(row);
                augmented.push(new_row);
            }
            augmented
        } else {
            x
        };

        let p_cols = if include_intercept { p_raw + 1 } else { p_raw };
        validate_xy_dims(n, p_cols, n * p_cols, y.len())?;

        Ok(Self { x: DenseX::from_rows(x)?, y, include_intercept, p })
    }

    /// Tweedie power parameter.
    pub fn power(&self) -> f64 {
        self.p
    }

    fn n_beta(&self) -> usize {
        self.x.p
    }

    /// Tweedie unit deviance: d(y, μ) for p ∈ (1, 2).
    #[inline]
    fn unit_deviance(y: f64, mu: f64, p: f64) -> f64 {
        if y > 0.0 {
            2.0 * (y.powf(2.0 - p) / ((1.0 - p) * (2.0 - p)) - y * mu.powf(1.0 - p) / (1.0 - p)
                + mu.powf(2.0 - p) / (2.0 - p))
        } else {
            // y == 0
            2.0 * mu.powf(2.0 - p) / (2.0 - p)
        }
    }
}

impl LogDensityModel for TweedieRegressionModel {
    type Prepared<'a>
        = PreparedModelRef<'a, Self>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        self.n_beta() + 1 // β + log_phi
    }

    fn parameter_names(&self) -> Vec<String> {
        let mut names: Vec<String> = if self.include_intercept {
            let mut v = vec!["intercept".to_string()];
            for j in 1..self.n_beta() {
                v.push(format!("beta_{}", j - 1));
            }
            v
        } else {
            (0..self.n_beta()).map(|j| format!("beta_{j}")).collect()
        };
        names.push("log_phi".to_string());
        names
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        let mut bounds = vec![(-20.0, 20.0); self.n_beta()];
        bounds.push((-10.0, 10.0)); // log_phi
        bounds
    }

    fn parameter_init(&self) -> Vec<f64> {
        let mut init = vec![0.0; self.n_beta()];
        if self.include_intercept {
            let positive_y: Vec<f64> = self.y.iter().copied().filter(|&v| v > 0.0).collect();
            let mean_y = if positive_y.is_empty() {
                1.0
            } else {
                positive_y.iter().sum::<f64>() / positive_y.len() as f64
            };
            init[0] = mean_y.max(1e-10).ln();
        }
        init.push(0.0); // log_phi = 0 → phi = 1
        init
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        if params.len() != self.dim() {
            return Err(Error::Validation("params length mismatch".to_string()));
        }
        let beta = &params[..self.n_beta()];
        let log_phi = params[self.n_beta()];
        let phi = exp_clamped(log_phi);
        let p = self.p;

        if !phi.is_finite() || phi <= 0.0 {
            return Ok(f64::MAX);
        }

        let n = self.x.n;
        let mut nll = 0.0;
        let log_2pi = (2.0 * std::f64::consts::PI).ln();

        for i in 0..n {
            let eta = row_dot(self.x.row(i), beta);
            let mu = exp_clamped(eta);
            if mu <= 0.0 || !mu.is_finite() {
                return Ok(f64::MAX);
            }
            let yi = self.y[i];

            if yi > 0.0 {
                // Saddle-point NLL: 0.5·log(2πφy^p) + d(y,μ)/(2φ)
                let dev = Self::unit_deviance(yi, mu, p);
                nll += 0.5 * (log_2pi + log_phi + p * yi.ln()) + dev / (2.0 * phi);
            } else {
                // y == 0: -log P(Y=0) = μ^(2-p) / (φ·(2-p))
                nll += mu.powf(2.0 - p) / (phi * (2.0 - p));
            }
        }

        Ok(nll)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        if params.len() != self.dim() {
            return Err(Error::Validation("params length mismatch".to_string()));
        }
        let beta = &params[..self.n_beta()];
        let log_phi = params[self.n_beta()];
        let phi = exp_clamped(log_phi);
        let p = self.p;

        if !phi.is_finite() || phi <= 0.0 {
            return Ok(vec![0.0; self.dim()]);
        }

        let n = self.x.n;
        let nb = self.n_beta();
        let mut grad = vec![0.0; self.dim()];

        for i in 0..n {
            let row = self.x.row(i);
            let eta = row_dot(row, beta);
            let mu = exp_clamped(eta);
            if mu <= 0.0 || !mu.is_finite() {
                continue;
            }
            let yi = self.y[i];

            // ∂NLL_i/∂β_j = (1/φ) · μ^(1-p) · (μ - y) · x_j
            // This formula is the same for y > 0 and y = 0.
            let common_beta = mu.powf(1.0 - p) * (mu - yi) / phi;
            for j in 0..nb {
                grad[j] += common_beta * row[j];
            }

            // ∂NLL_i/∂log_φ:
            if yi > 0.0 {
                // = φ · [1/(2φ) - d/(2φ²)] = 0.5 - d/(2φ)
                let dev = Self::unit_deviance(yi, mu, p);
                grad[nb] += 0.5 - dev / (2.0 * phi);
            } else {
                // = φ · [-μ^(2-p) / (φ²·(2-p))] = -μ^(2-p) / (φ·(2-p))
                grad[nb] += -mu.powf(2.0 - p) / (phi * (2.0 - p));
            }
        }

        Ok(grad)
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

    // ----- Gamma GLM tests -----

    #[test]
    fn gamma_glm_basic_properties() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let y = vec![2.1, 4.3, 5.8, 8.1, 11.2];
        let m = GammaRegressionModel::new(x, y, true).unwrap();

        assert_eq!(m.dim(), 3); // intercept + beta_0 + log_alpha
        assert_eq!(m.parameter_names().len(), 3);
        assert_eq!(m.parameter_names()[0], "intercept");
        assert_eq!(m.parameter_names()[2], "log_alpha");

        let init = m.parameter_init();
        let nll = m.nll(&init).unwrap();
        assert!(nll.is_finite());
    }

    #[test]
    fn gamma_glm_rejects_non_positive_y() {
        let x = vec![vec![1.0], vec![2.0]];
        let y = vec![1.0, 0.0];
        assert!(GammaRegressionModel::new(x, y, false).is_err());

        let x = vec![vec![1.0], vec![2.0]];
        let y = vec![1.0, -1.0];
        assert!(GammaRegressionModel::new(x, y, false).is_err());
    }

    #[test]
    fn gamma_glm_grad_matches_finite_diff() {
        let x =
            vec![vec![1.2, 0.5], vec![0.8, 1.1], vec![2.0, 0.3], vec![1.5, 0.9], vec![0.7, 1.8]];
        let y = vec![3.1, 2.5, 4.2, 3.8, 2.9];
        let m = GammaRegressionModel::new(x, y, true).unwrap();

        let params = vec![0.5, 0.3, -0.1, 0.5]; // intercept, b0, b1, log_alpha
        let g = m.grad_nll(&params).unwrap();
        let g_fd = finite_diff_grad(&m, &params, 1e-6);

        for j in 0..params.len() {
            assert!(
                (g[j] - g_fd[j]).abs() < 1e-4,
                "Gamma grad[{j}]: analytical={}, fd={}",
                g[j],
                g_fd[j]
            );
        }
    }

    #[test]
    fn gamma_glm_nll_decreases_at_true_params() {
        // Simulate y ~ Gamma(shape=2, rate=2/μ) with μ = exp(1 + 0.5*x).
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        // At true params: μ = [e^1, e^1.5, e^2, e^2.5, e^3] = [2.72, 4.48, 7.39, 12.18, 20.09]
        // y values drawn from Gamma with these means and shape=2.
        let y = vec![2.5, 3.8, 8.1, 10.5, 22.3];
        let m = GammaRegressionModel::new(x, y, true).unwrap();

        let nll_init = m.nll(&m.parameter_init()).unwrap();
        // True params: intercept=1.0, beta=0.5, log_alpha=ln(2)
        let nll_true = m.nll(&[1.0, 0.5, 2.0_f64.ln()]).unwrap();
        assert!(
            nll_true < nll_init,
            "NLL at true params ({nll_true}) should be less than at init ({nll_init})"
        );
    }

    // ----- Tweedie GLM tests -----

    #[test]
    fn tweedie_glm_basic_properties() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let y = vec![0.0, 3.2, 0.0, 5.1]; // zeros allowed
        let m = TweedieRegressionModel::new(x, y, true, 1.5).unwrap();

        assert_eq!(m.dim(), 3); // intercept + beta_0 + log_phi
        assert_eq!(m.power(), 1.5);
        assert_eq!(m.parameter_names()[2], "log_phi");

        let init = m.parameter_init();
        let nll = m.nll(&init).unwrap();
        assert!(nll.is_finite());
    }

    #[test]
    fn tweedie_glm_rejects_invalid_p() {
        let x = vec![vec![1.0]];
        let y = vec![1.0];
        assert!(TweedieRegressionModel::new(x.clone(), y.clone(), false, 1.0).is_err());
        assert!(TweedieRegressionModel::new(x.clone(), y.clone(), false, 2.0).is_err());
        assert!(TweedieRegressionModel::new(x.clone(), y.clone(), false, 0.5).is_err());
    }

    #[test]
    fn tweedie_glm_rejects_negative_y() {
        let x = vec![vec![1.0], vec![2.0]];
        let y = vec![1.0, -0.5];
        assert!(TweedieRegressionModel::new(x, y, false, 1.5).is_err());
    }

    #[test]
    fn tweedie_glm_grad_matches_finite_diff() {
        let x = vec![
            vec![1.0, 0.5],
            vec![0.8, 1.1],
            vec![2.0, 0.3],
            vec![0.0, 1.5], // will produce y=0 case
            vec![1.5, 0.9],
        ];
        let y = vec![3.1, 0.0, 4.2, 0.0, 2.9]; // mix of zeros and positives
        let m = TweedieRegressionModel::new(x, y, true, 1.5).unwrap();

        let params = vec![0.5, 0.3, -0.1, 0.2]; // intercept, b0, b1, log_phi
        let g = m.grad_nll(&params).unwrap();
        let g_fd = finite_diff_grad(&m, &params, 1e-6);

        for j in 0..params.len() {
            assert!(
                (g[j] - g_fd[j]).abs() < 1e-4,
                "Tweedie grad[{j}]: analytical={}, fd={}",
                g[j],
                g_fd[j]
            );
        }
    }

    #[test]
    fn tweedie_glm_grad_all_zeros() {
        // Edge case: all y = 0.
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![0.0, 0.0, 0.0];
        let m = TweedieRegressionModel::new(x, y, true, 1.5).unwrap();

        let params = vec![0.5, 0.1, 0.0];
        let g = m.grad_nll(&params).unwrap();
        let g_fd = finite_diff_grad(&m, &params, 1e-6);

        for j in 0..params.len() {
            assert!(
                (g[j] - g_fd[j]).abs() < 1e-4,
                "Tweedie all-zeros grad[{j}]: analytical={}, fd={}",
                g[j],
                g_fd[j]
            );
        }
    }

    #[test]
    fn tweedie_glm_grad_all_positive() {
        // Edge case: no zeros.
        let x = vec![vec![0.5], vec![1.0], vec![1.5], vec![2.0]];
        let y = vec![1.2, 2.5, 3.8, 5.1];
        let m = TweedieRegressionModel::new(x, y, true, 1.7).unwrap();

        let params = vec![0.3, 0.4, -0.5];
        let g = m.grad_nll(&params).unwrap();
        let g_fd = finite_diff_grad(&m, &params, 1e-6);

        for j in 0..params.len() {
            assert!(
                (g[j] - g_fd[j]).abs() < 1e-4,
                "Tweedie all-positive grad[{j}]: analytical={}, fd={}",
                g[j],
                g_fd[j]
            );
        }
    }

    #[test]
    fn tweedie_unit_deviance_zero_y() {
        // d(0, μ) = 2·μ^(2-p)/(2-p)
        let mu = 3.0;
        let p = 1.5;
        let d = TweedieRegressionModel::unit_deviance(0.0, mu, p);
        let expected = 2.0 * mu.powf(2.0 - p) / (2.0 - p);
        assert!((d - expected).abs() < 1e-12);
    }

    #[test]
    fn tweedie_unit_deviance_y_equals_mu() {
        // d(μ, μ) = 0 for any μ > 0
        let mu = 5.0;
        let p = 1.5;
        let d = TweedieRegressionModel::unit_deviance(mu, mu, p);
        assert!(d.abs() < 1e-12, "d(μ,μ) should be 0, got {d}");
    }

    #[test]
    fn tweedie_unit_deviance_positive() {
        // d(y, μ) ≥ 0 for all y, μ > 0
        for &p in &[1.1, 1.3, 1.5, 1.7, 1.9] {
            for &y in &[0.1, 1.0, 5.0, 10.0] {
                for &mu in &[0.1, 1.0, 5.0, 10.0] {
                    let d = TweedieRegressionModel::unit_deviance(y, mu, p);
                    assert!(d >= -1e-12, "d({y}, {mu}) = {d} < 0 at p={p}");
                }
            }
        }
    }

    #[test]
    fn tweedie_different_p_values_compile_and_run() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![0.0, 2.5, 4.1];
        for &p in &[1.1, 1.3, 1.5, 1.7, 1.9] {
            let m = TweedieRegressionModel::new(x.clone(), y.clone(), true, p).unwrap();
            let nll = m.nll(&m.parameter_init()).unwrap();
            assert!(nll.is_finite(), "NLL not finite at p={p}");
        }
    }
}
