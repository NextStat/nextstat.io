//! Model builder (composition) MVP for general statistics.
//!
//! Goal: build common log-density models (GLM-style) from blocks without writing
//! bespoke `logpdf` each time.

use ns_core::traits::{LogDensityModel, PreparedModelRef};
use ns_core::{Error, Result};
use ns_prob::math::{exp_clamped, log1pexp, sigmoid};

#[inline]
fn validate_xy_rectangular(x: &[Vec<f64>]) -> Result<(usize, usize)> {
    let n = x.len();
    let p = x.first().map(|r| r.len()).unwrap_or(0);
    if n == 0 {
        return Err(Error::Validation("X must be non-empty (n>0)".to_string()));
    }
    if p == 0 {
        return Err(Error::Validation("X must have at least 1 feature column".to_string()));
    }
    for (i, row) in x.iter().enumerate() {
        if row.len() != p {
            return Err(Error::Validation(format!(
                "X must be rectangular: row {} has len {}, expected {}",
                i,
                row.len(),
                p
            )));
        }
        if row.iter().any(|v| !v.is_finite()) {
            return Err(Error::Validation("X must contain only finite values".to_string()));
        }
    }
    Ok((n, p))
}

#[inline]
fn row_dot(x_row: &[f64], beta: &[f64]) -> f64 {
    debug_assert_eq!(x_row.len(), beta.len());
    x_row.iter().zip(beta).map(|(&x, &b)| x * b).sum()
}

/// Dense row-major design matrix.
#[derive(Debug, Clone)]
struct DenseX {
    n: usize,
    p: usize,
    data: Vec<f64>, // length n*p, row-major
}

impl DenseX {
    fn from_rows(x: Vec<Vec<f64>>) -> Result<Self> {
        let (n, p) = validate_xy_rectangular(&x)?;
        let mut data = Vec::with_capacity(n * p);
        for row in x {
            data.extend_from_slice(&row);
        }
        Ok(Self { n, p, data })
    }

    #[inline]
    fn row(&self, i: usize) -> &[f64] {
        let start = i * self.p;
        &self.data[start..start + self.p]
    }
}

#[derive(Debug, Clone, Copy)]
struct NormalPrior {
    mu: f64,
    sigma: f64,
}

impl NormalPrior {
    fn validate(self) -> Result<Self> {
        if !self.mu.is_finite() {
            return Err(Error::Validation("prior mu must be finite".to_string()));
        }
        if !self.sigma.is_finite() || self.sigma <= 0.0 {
            return Err(Error::Validation("prior sigma must be finite and > 0".to_string()));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone)]
enum Likelihood {
    GaussianY { y: Vec<f64> },      // sigma fixed to 1
    BernoulliLogitY { y: Vec<u8> }, // y in {0,1}
    PoissonY { y: Vec<u64>, offset: Option<Vec<f64>> }, // y >= 0, log link
}

#[derive(Debug, Clone)]
struct RandomIntercept {
    n_groups: usize,
    group_idx: Vec<usize>, // length n
    non_centered: bool,
    // Parameter indices in the model parameter vector:
    mu_idx: usize,
    sigma_idx: usize,    // sigma_alpha > 0
    alpha_offset: usize, // alpha_j at params[alpha_offset + j]
    // Priors:
    mu_prior: NormalPrior,
    logsigma_prior_m: f64,
    logsigma_prior_s: f64,
}

impl RandomIntercept {
    fn validate(&self, n: usize) -> Result<()> {
        if self.n_groups == 0 {
            return Err(Error::Validation("n_groups must be > 0".to_string()));
        }
        if self.group_idx.len() != n {
            return Err(Error::Validation(format!(
                "group_idx length must match n: expected {}, got {}",
                n,
                self.group_idx.len()
            )));
        }
        if self.group_idx.iter().any(|&g| g >= self.n_groups) {
            return Err(Error::Validation("group_idx must be in [0, n_groups)".to_string()));
        }
        self.mu_prior.validate()?;
        if !self.logsigma_prior_m.is_finite() {
            return Err(Error::Validation("logsigma_prior_m must be finite".to_string()));
        }
        if !self.logsigma_prior_s.is_finite() || self.logsigma_prior_s <= 0.0 {
            return Err(Error::Validation("logsigma_prior_s must be finite and > 0".to_string()));
        }
        Ok(())
    }
}

/// A composed GLM-style model: linear predictor + likelihood + priors.
///
/// Supported:
/// - Linear regression: Gaussian likelihood with sigma fixed to 1
/// - Logistic regression: Bernoulli likelihood with logit link
/// - Optional Normal random intercept with (mu, sigma_alpha) hyperparameters
#[derive(Debug, Clone)]
pub struct ComposedGlmModel {
    x: DenseX,
    include_intercept: bool,
    likelihood: Likelihood,
    coef_prior: NormalPrior, // applied to intercept (if present) + all beta_j
    penalize_intercept: bool,
    random_intercept: Option<RandomIntercept>,
}

impl ComposedGlmModel {
    fn dim_internal(&self) -> usize {
        let base = self.x.p + if self.include_intercept { 1 } else { 0 };
        match &self.random_intercept {
            None => base,
            Some(ri) => base + 2 + ri.n_groups,
        }
    }

    fn validate(&self) -> Result<()> {
        self.coef_prior.validate()?;
        let n = self.x.n;
        match &self.likelihood {
            Likelihood::GaussianY { y } => {
                if y.len() != n {
                    return Err(Error::Validation(format!(
                        "y length must match n: expected {}, got {}",
                        n,
                        y.len()
                    )));
                }
                if y.iter().any(|v| !v.is_finite()) {
                    return Err(Error::Validation("y must contain only finite values".to_string()));
                }
            }
            Likelihood::BernoulliLogitY { y } => {
                if y.len() != n {
                    return Err(Error::Validation(format!(
                        "y length must match n: expected {}, got {}",
                        n,
                        y.len()
                    )));
                }
                if y.iter().any(|&v| v > 1) {
                    return Err(Error::Validation("y must contain only 0/1".to_string()));
                }
            }
            Likelihood::PoissonY { y, offset } => {
                if y.len() != n {
                    return Err(Error::Validation(format!(
                        "y length must match n: expected {}, got {}",
                        n,
                        y.len()
                    )));
                }
                if let Some(off) = offset {
                    if off.len() != n {
                        return Err(Error::Validation(format!(
                            "offset length must match n: expected {}, got {}",
                            n,
                            off.len()
                        )));
                    }
                    if off.iter().any(|v| !v.is_finite()) {
                        return Err(Error::Validation(
                            "offset must contain only finite values".to_string(),
                        ));
                    }
                }
            }
        }
        if let Some(ri) = &self.random_intercept {
            ri.validate(n)?;
        }
        Ok(())
    }

    #[inline]
    fn eta_components<'a>(&self, params: &'a [f64]) -> Result<(&'a [f64], f64)> {
        let dim = self.dim_internal();
        if params.len() != dim {
            return Err(Error::Validation(format!(
                "expected {} parameters, got {}",
                dim,
                params.len()
            )));
        }
        if params.iter().any(|v| !v.is_finite()) {
            return Err(Error::Validation("params must contain only finite values".to_string()));
        }
        let beta_offset = if self.include_intercept { 1 } else { 0 };
        let b0 = if self.include_intercept { params[0] } else { 0.0 };
        let beta = &params[beta_offset..beta_offset + self.x.p];
        Ok((beta, b0))
    }
}

impl LogDensityModel for ComposedGlmModel {
    type Prepared<'a>
        = PreparedModelRef<'a, Self>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        self.dim_internal()
    }

    fn parameter_names(&self) -> Vec<String> {
        let mut out = Vec::with_capacity(self.dim_internal());
        if self.include_intercept {
            out.push("intercept".to_string());
        }
        for j in 0..self.x.p {
            out.push(format!("beta{}", j + 1));
        }
        if let Some(ri) = &self.random_intercept {
            out.push("mu_alpha".to_string());
            out.push("sigma_alpha".to_string());
            for j in 0..ri.n_groups {
                if ri.non_centered {
                    out.push(format!("z_alpha{}", j + 1));
                } else {
                    out.push(format!("alpha{}", j + 1));
                }
            }
        }
        out
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        let mut out = vec![
            (f64::NEG_INFINITY, f64::INFINITY);
            self.x.p + if self.include_intercept { 1 } else { 0 }
        ];
        if let Some(ri) = &self.random_intercept {
            out.push((f64::NEG_INFINITY, f64::INFINITY)); // mu_alpha
            out.push((0.0, f64::INFINITY)); // sigma_alpha
            out.extend(vec![(f64::NEG_INFINITY, f64::INFINITY); ri.n_groups]);
        }
        out
    }

    fn parameter_init(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.x.p + if self.include_intercept { 1 } else { 0 }];
        if let Some(ri) = &self.random_intercept {
            out.push(0.0); // mu_alpha
            out.push(1.0); // sigma_alpha
            out.extend(vec![0.0; ri.n_groups]);
        }
        out
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        self.validate()?;
        let (beta, b0) = self.eta_components(params)?;

        let mut nll = 0.0;

        // Coefficient priors (independent Normal, up to additive constant).
        let coef_mu = self.coef_prior.mu;
        let coef_sig2 = self.coef_prior.sigma * self.coef_prior.sigma;
        if self.include_intercept && self.penalize_intercept {
            let v = params[0] - coef_mu;
            nll += 0.5 * v * v / coef_sig2;
        }
        let beta_offset = if self.include_intercept { 1 } else { 0 };
        for j in 0..self.x.p {
            let v = params[beta_offset + j] - coef_mu;
            nll += 0.5 * v * v / coef_sig2;
        }

        // Random intercept prior block (adds its own NLL terms).
        if let Some(ri) = &self.random_intercept {
            let mu = params[ri.mu_idx];
            let sigma = params[ri.sigma_idx];
            if !sigma.is_finite() || sigma <= 0.0 {
                return Err(Error::Validation(format!(
                    "sigma_alpha must be finite and > 0, got {}",
                    sigma
                )));
            }

            // mu prior: Normal(0, mu_prior.sigma)
            let mu_sig2 = ri.mu_prior.sigma * ri.mu_prior.sigma;
            let v = mu - ri.mu_prior.mu;
            nll += 0.5 * v * v / mu_sig2;

            // sigma prior: LogNormal(m, s) in terms of sigma (up to constant).
            let t = sigma.ln();
            let z = (t - ri.logsigma_prior_m) / ri.logsigma_prior_s;
            nll += 0.5 * z * z + t;

            if ri.non_centered {
                // Non-centered: z_j ~ Normal(0,1), alpha_j = mu + sigma * z_j.
                // Equivalent to centered alpha prior with Jacobian folded in.
                let mut sum_sq = 0.0;
                for j in 0..ri.n_groups {
                    let zj = params[ri.alpha_offset + j];
                    sum_sq += zj * zj;
                }
                nll += 0.5 * sum_sq;
            } else {
                // Centered: alpha_j | mu, sigma ~ Normal(mu, sigma).
                let mut sum_sq = 0.0;
                for j in 0..ri.n_groups {
                    let a = params[ri.alpha_offset + j];
                    let d = a - mu;
                    sum_sq += d * d;
                }
                nll += 0.5 * sum_sq / (sigma * sigma);
                nll += (ri.n_groups as f64) * sigma.ln();
            }
        }

        // Likelihood block.
        for i in 0..self.x.n {
            let mut eta = b0 + row_dot(self.x.row(i), beta);
            if let Some(ri) = &self.random_intercept {
                let g = ri.group_idx[i];
                if ri.non_centered {
                    let mu = params[ri.mu_idx];
                    let sigma = params[ri.sigma_idx];
                    let zj = params[ri.alpha_offset + g];
                    eta += mu + sigma * zj;
                } else {
                    eta += params[ri.alpha_offset + g];
                }
            }

            match &self.likelihood {
                Likelihood::GaussianY { y } => {
                    let r = y[i] - eta;
                    nll += 0.5 * r * r;
                }
                Likelihood::BernoulliLogitY { y } => {
                    let yi = y[i] as f64;
                    nll += log1pexp(eta) - yi * eta;
                }
                Likelihood::PoissonY { y, offset } => {
                    let mut eta2 = eta;
                    if let Some(off) = offset {
                        eta2 += off[i];
                    }
                    let mu = exp_clamped(eta2);
                    nll += mu - (y[i] as f64) * eta2;
                }
            }
        }

        Ok(nll)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        self.validate()?;
        let (beta, b0) = self.eta_components(params)?;

        let dim = self.dim_internal();
        let mut grad = vec![0.0; dim];

        // Coefficient priors.
        let coef_mu = self.coef_prior.mu;
        let coef_sig2 = self.coef_prior.sigma * self.coef_prior.sigma;
        if self.include_intercept && self.penalize_intercept {
            grad[0] += (params[0] - coef_mu) / coef_sig2;
        }
        let beta_offset = if self.include_intercept { 1 } else { 0 };
        for j in 0..self.x.p {
            grad[beta_offset + j] += (params[beta_offset + j] - coef_mu) / coef_sig2;
        }

        // Random intercept contributions.
        if let Some(ri) = &self.random_intercept {
            let mu = params[ri.mu_idx];
            let sigma = params[ri.sigma_idx];
            if !sigma.is_finite() || sigma <= 0.0 {
                return Err(Error::Validation(format!(
                    "sigma_alpha must be finite and > 0, got {}",
                    sigma
                )));
            }

            // mu prior: Normal(mu0, mu_sigma)
            let mu_sig2 = ri.mu_prior.sigma * ri.mu_prior.sigma;
            grad[ri.mu_idx] += (mu - ri.mu_prior.mu) / mu_sig2;

            // sigma prior: LogNormal(m,s): 0.5*((ln s - m)/s)^2 + ln s
            let t = sigma.ln();
            let z = (t - ri.logsigma_prior_m) / ri.logsigma_prior_s;
            let ddt = z / (ri.logsigma_prior_s) + 1.0;
            grad[ri.sigma_idx] += ddt * (1.0 / sigma);

            if ri.non_centered {
                // z_j ~ Normal(0,1)
                for j in 0..ri.n_groups {
                    let zj = params[ri.alpha_offset + j];
                    grad[ri.alpha_offset + j] += zj;
                }
            } else {
                // Centered alpha prior terms.
                let inv_var_a = 1.0 / (sigma * sigma);

                let mut sum_sq = 0.0;
                for j in 0..ri.n_groups {
                    let a = params[ri.alpha_offset + j];
                    let d = a - mu;
                    sum_sq += d * d;
                    grad[ri.alpha_offset + j] += d * inv_var_a;
                    grad[ri.mu_idx] += (mu - a) * inv_var_a;
                }
                // + n_groups * ln(sigma) term.
                grad[ri.sigma_idx] += (ri.n_groups as f64) / sigma;

                // derivative of 0.5 * sum_sq / sigma^2 wrt sigma:
                // d/dsigma [0.5 * sum_sq * sigma^{-2}] = -sum_sq * sigma^{-3}
                grad[ri.sigma_idx] += -sum_sq / (sigma * sigma * sigma);
            }
        }

        // Likelihood contributions.
        for i in 0..self.x.n {
            let mut eta = b0 + row_dot(self.x.row(i), beta);
            let ri_group = if let Some(ri) = &self.random_intercept {
                let g = ri.group_idx[i];
                if ri.non_centered {
                    let mu = params[ri.mu_idx];
                    let sigma = params[ri.sigma_idx];
                    let zj = params[ri.alpha_offset + g];
                    eta += mu + sigma * zj;
                } else {
                    eta += params[ri.alpha_offset + g];
                }
                Some(g)
            } else {
                None
            };

            let d_eta = match &self.likelihood {
                Likelihood::GaussianY { y } => eta - y[i],
                Likelihood::BernoulliLogitY { y } => sigmoid(eta) - (y[i] as f64),
                Likelihood::PoissonY { y, offset } => {
                    let mut eta2 = eta;
                    if let Some(off) = offset {
                        eta2 += off[i];
                    }
                    exp_clamped(eta2) - (y[i] as f64)
                }
            };

            if self.include_intercept {
                grad[0] += d_eta;
            }
            let row = self.x.row(i);
            for j in 0..self.x.p {
                grad[beta_offset + j] += d_eta * row[j];
            }
            if let (Some(ri), Some(g)) = (&self.random_intercept, ri_group) {
                if ri.non_centered {
                    let sigma = params[ri.sigma_idx];
                    let zj = params[ri.alpha_offset + g];
                    grad[ri.mu_idx] += d_eta;
                    grad[ri.sigma_idx] += d_eta * zj;
                    grad[ri.alpha_offset + g] += d_eta * sigma;
                } else {
                    grad[ri.alpha_offset + g] += d_eta;
                }
            }
        }

        Ok(grad)
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

/// Builder for [`ComposedGlmModel`].
#[derive(Debug, Clone)]
pub struct ModelBuilder {
    x: DenseX,
    include_intercept: bool,
    likelihood: Likelihood,
    coef_prior: NormalPrior,
    penalize_intercept: bool,
    random_intercept: Option<RandomIntercept>,
}

impl ModelBuilder {
    /// Start a Gaussian linear regression builder (sigma fixed to 1).
    pub fn linear_regression(
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        include_intercept: bool,
    ) -> Result<Self> {
        let x = DenseX::from_rows(x)?;
        if y.len() != x.n {
            return Err(Error::Validation(format!(
                "y length must match n: expected {}, got {}",
                x.n,
                y.len()
            )));
        }
        Ok(Self {
            x,
            include_intercept,
            likelihood: Likelihood::GaussianY { y },
            coef_prior: NormalPrior { mu: 0.0, sigma: 10.0 },
            penalize_intercept: false,
            random_intercept: None,
        })
    }

    /// Start a Bernoulli-logit (logistic regression) builder.
    pub fn logistic_regression(
        x: Vec<Vec<f64>>,
        y: Vec<u8>,
        include_intercept: bool,
    ) -> Result<Self> {
        let x = DenseX::from_rows(x)?;
        if y.len() != x.n {
            return Err(Error::Validation(format!(
                "y length must match n: expected {}, got {}",
                x.n,
                y.len()
            )));
        }
        if y.iter().any(|&v| v > 1) {
            return Err(Error::Validation("y must contain only 0/1".to_string()));
        }
        Ok(Self {
            x,
            include_intercept,
            likelihood: Likelihood::BernoulliLogitY { y },
            coef_prior: NormalPrior { mu: 0.0, sigma: 10.0 },
            penalize_intercept: false,
            random_intercept: None,
        })
    }

    /// Start a Poisson regression builder (log link) with optional offset.
    pub fn poisson_regression(
        x: Vec<Vec<f64>>,
        y: Vec<u64>,
        include_intercept: bool,
        offset: Option<Vec<f64>>,
    ) -> Result<Self> {
        let x = DenseX::from_rows(x)?;
        if y.len() != x.n {
            return Err(Error::Validation(format!(
                "y length must match n: expected {}, got {}",
                x.n,
                y.len()
            )));
        }
        if let Some(off) = &offset {
            if off.len() != x.n {
                return Err(Error::Validation(format!(
                    "offset length must match n: expected {}, got {}",
                    x.n,
                    off.len()
                )));
            }
            if off.iter().any(|v| !v.is_finite()) {
                return Err(Error::Validation("offset must contain only finite values".to_string()));
            }
        }
        Ok(Self {
            x,
            include_intercept,
            likelihood: Likelihood::PoissonY { y, offset },
            coef_prior: NormalPrior { mu: 0.0, sigma: 10.0 },
            penalize_intercept: false,
            random_intercept: None,
        })
    }

    /// Set a Normal prior applied to intercept (if present) and all beta coefficients.
    pub fn with_coef_prior_normal(mut self, mu: f64, sigma: f64) -> Result<Self> {
        self.coef_prior = NormalPrior { mu, sigma }.validate()?;
        Ok(self)
    }

    /// Whether to apply the coefficient prior to the intercept term.
    ///
    /// Default is `false` (common ridge convention).
    pub fn with_penalize_intercept(mut self, penalize_intercept: bool) -> Self {
        self.penalize_intercept = penalize_intercept;
        self
    }

    /// Add a Normal random intercept with hyperparameters (mu_alpha, sigma_alpha).
    ///
    /// `group_idx[i]` selects which intercept applies to observation `i`.
    pub fn with_random_intercept(mut self, group_idx: Vec<usize>, n_groups: usize) -> Result<Self> {
        let base = self.x.p + if self.include_intercept { 1 } else { 0 };
        let ri = RandomIntercept {
            n_groups,
            group_idx,
            non_centered: false,
            mu_idx: base,
            sigma_idx: base + 1,
            alpha_offset: base + 2,
            mu_prior: NormalPrior { mu: 0.0, sigma: 1.0 },
            logsigma_prior_m: 0.0,
            logsigma_prior_s: 0.5,
        };
        ri.validate(self.x.n)?;
        self.random_intercept = Some(ri);
        Ok(self)
    }

    /// Toggle a non-centered parameterization for the random intercept block.
    ///
    /// If enabled, the group parameters are `z_alpha[j] ~ Normal(0,1)` and the
    /// effective intercept is `alpha_j = mu_alpha + sigma_alpha * z_alpha[j]`.
    pub fn with_random_intercept_non_centered(mut self, non_centered: bool) -> Result<Self> {
        let Some(ri) = &mut self.random_intercept else {
            return Err(Error::Validation(
                "random intercept not enabled; call with_random_intercept first".to_string(),
            ));
        };
        ri.non_centered = non_centered;
        ri.validate(self.x.n)?;
        Ok(self)
    }

    /// Customize priors for the random intercept hyperparameters.
    pub fn with_random_intercept_priors(
        mut self,
        mu_prior: (f64, f64),
        logsigma_prior: (f64, f64),
    ) -> Result<Self> {
        let Some(ri) = &mut self.random_intercept else {
            return Err(Error::Validation(
                "random intercept not enabled; call with_random_intercept first".to_string(),
            ));
        };
        ri.mu_prior = NormalPrior { mu: mu_prior.0, sigma: mu_prior.1 }.validate()?;
        ri.logsigma_prior_m = logsigma_prior.0;
        ri.logsigma_prior_s = logsigma_prior.1;
        ri.validate(self.x.n)?;
        Ok(self)
    }

    /// Build the composed model.
    pub fn build(self) -> Result<ComposedGlmModel> {
        let m = ComposedGlmModel {
            x: self.x,
            include_intercept: self.include_intercept,
            likelihood: self.likelihood,
            coef_prior: self.coef_prior.validate()?,
            penalize_intercept: self.penalize_intercept,
            random_intercept: self.random_intercept,
        };
        m.validate()?;
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regression::{LinearRegressionModel, LogisticRegressionModel, PoissonRegressionModel};
    use serde::Deserialize;

    #[derive(Debug, Clone, Deserialize)]
    struct Fixture {
        kind: String,
        include_intercept: bool,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        offset: Option<Vec<f64>>,
        beta_hat: Vec<f64>,
        nll_at_hat: f64,
    }

    fn load_fixture(json: &'static str) -> Fixture {
        serde_json::from_str(json).unwrap()
    }

    fn inf_norm(v: &[f64]) -> f64 {
        v.iter().map(|x| x.abs()).fold(0.0, f64::max)
    }

    fn assert_vec_close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (ai - bi).abs();
            let scale = ai.abs().max(bi.abs()).max(1.0);
            assert!(
                diff / scale <= tol,
                "index {}: {} vs {} (diff={}, tol={})",
                i,
                ai,
                bi,
                diff,
                tol
            );
        }
    }

    #[test]
    fn test_builder_linear_regression_matches_explicit_model_at_hat() {
        let fx = load_fixture(include_str!("../../../tests/fixtures/regression/ols_small.json"));
        assert_eq!(fx.kind, "ols");
        let explicit =
            LinearRegressionModel::new(fx.x.clone(), fx.y.clone(), fx.include_intercept).unwrap();
        let built =
            ModelBuilder::linear_regression(fx.x.clone(), fx.y.clone(), fx.include_intercept)
                .unwrap()
                .with_coef_prior_normal(0.0, 1e9)
                .unwrap()
                .build()
                .unwrap();

        let nll_e = explicit.nll(&fx.beta_hat).unwrap();
        let nll_b = built.nll(&fx.beta_hat).unwrap();
        assert!((nll_e - nll_b).abs() < 1e-10);
        assert!((nll_b - fx.nll_at_hat).abs() < 1e-8);

        let g_e = explicit.grad_nll(&fx.beta_hat).unwrap();
        let g_b = built.grad_nll(&fx.beta_hat).unwrap();
        assert_vec_close(&g_e, &g_b, 1e-10);
        assert!(inf_norm(&g_b) < 1e-6);
    }

    #[test]
    fn test_builder_logistic_regression_matches_explicit_model_at_hat() {
        let fx =
            load_fixture(include_str!("../../../tests/fixtures/regression/logistic_small.json"));
        assert_eq!(fx.kind, "logistic");
        let y: Vec<u8> = fx.y.iter().map(|&v| if v >= 0.5 { 1 } else { 0 }).collect();

        let explicit =
            LogisticRegressionModel::new(fx.x.clone(), y.clone(), fx.include_intercept).unwrap();
        let built = ModelBuilder::logistic_regression(fx.x.clone(), y, fx.include_intercept)
            .unwrap()
            .with_coef_prior_normal(0.0, 1e9)
            .unwrap()
            .build()
            .unwrap();

        let nll_e = explicit.nll(&fx.beta_hat).unwrap();
        let nll_b = built.nll(&fx.beta_hat).unwrap();
        assert!((nll_e - nll_b).abs() < 1e-10);
        assert!((nll_b - fx.nll_at_hat).abs() < 1e-6);

        let g_e = explicit.grad_nll(&fx.beta_hat).unwrap();
        let g_b = built.grad_nll(&fx.beta_hat).unwrap();
        assert_vec_close(&g_e, &g_b, 1e-8);
        assert!(inf_norm(&g_b) < 1e-6);
    }

    #[test]
    fn test_builder_poisson_regression_matches_explicit_model_at_hat() {
        let fx = load_fixture(include_str!("../../../tests/fixtures/regression/poisson_small.json"));
        assert_eq!(fx.kind, "poisson");
        let y: Vec<u64> = fx.y.iter().map(|&v| v.round() as u64).collect();

        let explicit = PoissonRegressionModel::new(
            fx.x.clone(),
            y.clone(),
            fx.include_intercept,
            fx.offset.clone(),
        )
        .unwrap();

        let built = ModelBuilder::poisson_regression(
            fx.x.clone(),
            y,
            fx.include_intercept,
            fx.offset.clone(),
        )
        .unwrap()
        .with_coef_prior_normal(0.0, 1e9)
        .unwrap()
        .build()
        .unwrap();

        let nll_e = explicit.nll(&fx.beta_hat).unwrap();
        let nll_b = built.nll(&fx.beta_hat).unwrap();
        assert!((nll_e - nll_b).abs() < 1e-10);
        assert!((nll_b - fx.nll_at_hat).abs() < 1e-6);

        let g_e = explicit.grad_nll(&fx.beta_hat).unwrap();
        let g_b = built.grad_nll(&fx.beta_hat).unwrap();
        assert_vec_close(&g_e, &g_b, 1e-8);
        assert!(inf_norm(&g_b) < 1e-6);
    }

    fn finite_diff_grad<M: LogDensityModel>(m: &M, params: &[f64], eps: f64) -> Vec<f64> {
        let mut g = vec![0.0; params.len()];
        for i in 0..params.len() {
            let mut p1 = params.to_vec();
            let mut p2 = params.to_vec();
            p1[i] += eps;
            p2[i] -= eps;
            let f1 = m.nll(&p1).unwrap();
            let f2 = m.nll(&p2).unwrap();
            g[i] = (f1 - f2) / (2.0 * eps);
        }
        g
    }

    #[test]
    fn test_builder_random_intercept_grad_finite_diff_smoke() {
        let x = vec![vec![1.0, 0.5], vec![1.0, -0.3], vec![1.0, 1.2], vec![1.0, 0.1]];
        let y = vec![0u8, 1u8, 1u8, 0u8];
        let group_idx = vec![0usize, 1usize, 0usize, 1usize];

        let m = ModelBuilder::logistic_regression(x, y, false)
            .unwrap()
            .with_coef_prior_normal(0.0, 1e9)
            .unwrap()
            .with_random_intercept(group_idx, 2)
            .unwrap()
            .build()
            .unwrap();

        let mut p = m.parameter_init();
        // Make sigma_alpha slightly away from 0 to avoid singularities.
        let sigma_idx = m.parameter_names().iter().position(|s| s == "sigma_alpha").unwrap();
        p[sigma_idx] = 0.7;

        let g = m.grad_nll(&p).unwrap();
        let g_fd = finite_diff_grad(&m, &p, 1e-6);

        assert_vec_close(&g, &g_fd, 5e-4);
    }

    #[test]
    fn test_builder_random_intercept_non_centered_grad_finite_diff_smoke() {
        let x = vec![vec![1.0, 0.5], vec![1.0, -0.3], vec![1.0, 1.2], vec![1.0, 0.1]];
        let y = vec![0u8, 1u8, 1u8, 0u8];
        let group_idx = vec![0usize, 1usize, 0usize, 1usize];

        let m = ModelBuilder::logistic_regression(x, y, false)
            .unwrap()
            .with_coef_prior_normal(0.0, 1e9)
            .unwrap()
            .with_random_intercept(group_idx, 2)
            .unwrap()
            .with_random_intercept_non_centered(true)
            .unwrap()
            .build()
            .unwrap();

        let mut p = m.parameter_init();
        // Make sigma_alpha slightly away from 0 to avoid singularities.
        let sigma_idx = m.parameter_names().iter().position(|s| s == "sigma_alpha").unwrap();
        p[sigma_idx] = 0.7;

        let g = m.grad_nll(&p).unwrap();
        let g_fd = finite_diff_grad(&m, &p, 1e-6);

        assert_vec_close(&g, &g_fd, 5e-4);
    }
}
