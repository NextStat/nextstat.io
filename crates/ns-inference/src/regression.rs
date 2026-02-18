//! Regression models for general statistics use-cases (Phase 6).
//!
//! These models implement [`ns_core::traits::LogDensityModel`], so they can be used with:
//! - `MaximumLikelihoodEstimator` (MLE/MAP)
//! - `sample_nuts` / `sample_nuts_multichain` (Bayesian sampling)

use nalgebra::{DMatrix, DVector};
use ndarray::{ArrayView1, ArrayView2};
use ns_core::traits::{LogDensityModel, PreparedModelRef};
use ns_core::{Error, Result};
use ns_prob::math::{exp_clamped, log1pexp, sigmoid};
use statrs::function::gamma::{digamma, ln_gamma};

#[inline]
pub(crate) fn validate_xy_dims(n: usize, p: usize, x_len: usize, y_len: usize) -> Result<()> {
    if n == 0 {
        return Err(Error::Validation("X/y must be non-empty".to_string()));
    }
    if p == 0 {
        return Err(Error::Validation("X must have at least 1 feature column".to_string()));
    }
    if x_len != n * p {
        return Err(Error::Validation(format!(
            "X has wrong length: expected n*p={}, got {}",
            n * p,
            x_len
        )));
    }
    if y_len != n {
        return Err(Error::Validation(format!(
            "y has wrong length: expected n={}, got {}",
            n, y_len
        )));
    }
    Ok(())
}

#[inline]
pub(crate) fn row_dot(x_row: &[f64], beta: &[f64]) -> f64 {
    debug_assert_eq!(x_row.len(), beta.len());
    x_row.iter().zip(beta).map(|(&x, &b)| x * b).sum()
}

/// Dense row-major design matrix.
#[derive(Debug, Clone)]
pub(crate) struct DenseX {
    pub(crate) n: usize,
    pub(crate) p: usize,
    pub(crate) data: Vec<f64>, // length n*p, row-major
}

impl DenseX {
    pub(crate) fn from_rows(x: Vec<Vec<f64>>) -> Result<Self> {
        let n = x.len();
        let p = x.first().map(|r| r.len()).unwrap_or(0);
        if n == 0 || p == 0 {
            return Err(Error::Validation("X must be non-empty (n>0, p>0)".to_string()));
        }
        let mut data = Vec::with_capacity(n * p);
        for (i, row) in x.into_iter().enumerate() {
            if row.len() != p {
                return Err(Error::Validation(format!(
                    "X must be rectangular: row {} has len {}, expected {}",
                    i,
                    row.len(),
                    p
                )));
            }
            for v in row {
                if !v.is_finite() {
                    return Err(Error::Validation("X must contain only finite values".to_string()));
                }
                data.push(v);
            }
        }
        Ok(Self { n, p, data })
    }

    #[inline]
    pub(crate) fn row(&self, i: usize) -> &[f64] {
        let start = i * self.p;
        &self.data[start..start + self.p]
    }
}

/// Gaussian linear regression with fixed sigma=1.
///
/// Model:
/// `y_i ~ Normal(eta_i, 1)`, where `eta_i = intercept + X_i * beta`.
///
/// NLL (up to additive constant): `0.5 * sum_i (y_i - eta_i)^2`.
#[derive(Debug, Clone)]
pub struct LinearRegressionModel {
    x: DenseX,
    y: Vec<f64>,
    include_intercept: bool,
}

impl LinearRegressionModel {
    /// Create a new linear regression model from row-wise `X` and `y`.
    pub fn new(x: Vec<Vec<f64>>, y: Vec<f64>, include_intercept: bool) -> Result<Self> {
        let x = DenseX::from_rows(x)?;
        validate_xy_dims(x.n, x.p, x.data.len(), y.len())?;
        if y.iter().any(|v| !v.is_finite()) {
            return Err(Error::Validation("y must contain only finite values".to_string()));
        }
        Ok(Self { x, y, include_intercept })
    }

    #[inline]
    fn dim_internal(&self) -> usize {
        self.x.p + if self.include_intercept { 1 } else { 0 }
    }

    #[inline]
    fn eta(&self, i: usize, params: &[f64]) -> f64 {
        if self.include_intercept {
            let (b0, beta) = params.split_first().expect("params non-empty when include_intercept");
            *b0 + row_dot(self.x.row(i), beta)
        } else {
            row_dot(self.x.row(i), params)
        }
    }
}

impl LogDensityModel for LinearRegressionModel {
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
        out
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![(f64::NEG_INFINITY, f64::INFINITY); self.dim_internal()]
    }

    fn parameter_init(&self) -> Vec<f64> {
        vec![0.0; self.dim_internal()]
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        if params.len() != self.dim_internal() {
            return Err(Error::Validation(format!(
                "expected {} parameters, got {}",
                self.dim_internal(),
                params.len()
            )));
        }
        if params.iter().any(|v| !v.is_finite()) {
            return Err(Error::Validation("params must contain only finite values".to_string()));
        }

        let mut sse = 0.0;
        for i in 0..self.x.n {
            let r = self.y[i] - self.eta(i, params);
            sse += r * r;
        }
        Ok(0.5 * sse)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        if params.len() != self.dim_internal() {
            return Err(Error::Validation(format!(
                "expected {} parameters, got {}",
                self.dim_internal(),
                params.len()
            )));
        }
        if params.iter().any(|v| !v.is_finite()) {
            return Err(Error::Validation("params must contain only finite values".to_string()));
        }

        let mut grad = vec![0.0; self.dim_internal()];
        for i in 0..self.x.n {
            let eta = self.eta(i, params);
            let err = eta - self.y[i];
            if self.include_intercept {
                grad[0] += err;
                let row = self.x.row(i);
                for j in 0..self.x.p {
                    grad[1 + j] += err * row[j];
                }
            } else {
                let row = self.x.row(i);
                for j in 0..self.x.p {
                    grad[j] += err * row[j];
                }
            }
        }
        Ok(grad)
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

/// Closed-form OLS fit for [`LinearRegressionModel`] (sigma fixed to 1).
///
/// Solves the normal equations: `(X^T X) beta = X^T y`.
pub fn ols_fit(x: Vec<Vec<f64>>, y: Vec<f64>, include_intercept: bool) -> Result<Vec<f64>> {
    let m = LinearRegressionModel::new(x, y, include_intercept)?;
    let d = m.dim_internal();
    let p = m.x.p;

    // Accumulate XtX and Xty.
    let mut xtx = vec![0.0; d * d];
    let mut xty = vec![0.0; d];

    for i in 0..m.x.n {
        let yi = m.y[i];
        let row = m.x.row(i);

        if include_intercept {
            xty[0] += yi;
            xtx[0] += 1.0;
            for a in 0..p {
                let xa = row[a];
                xty[1 + a] += xa * yi;
                xtx[1 + a] += xa;
                xtx[(1 + a) * d] += xa;
                for b in 0..p {
                    xtx[(1 + a) * d + (1 + b)] += xa * row[b];
                }
            }
        } else {
            for a in 0..p {
                let xa = row[a];
                xty[a] += xa * yi;
                for b in 0..p {
                    xtx[a * d + b] += xa * row[b];
                }
            }
        }
    }

    let a = DMatrix::from_row_slice(d, d, &xtx);
    let b = DVector::from_vec(xty);
    let sol = a
        .lu()
        .solve(&b)
        .ok_or_else(|| Error::Computation("OLS solve failed (singular XtX)".to_string()))?;
    Ok(sol.iter().copied().collect())
}

/// Logistic regression (Bernoulli) with logit link.
///
/// Model:
/// `y_i ~ Bernoulli(sigmoid(eta_i))`, `eta_i = intercept + X_i * beta`
///
/// NLL: `sum_i log(1 + exp(eta_i)) - y_i * eta_i`
#[derive(Debug, Clone)]
pub struct LogisticRegressionModel {
    x: DenseX,
    y: Vec<u8>, // 0/1
    include_intercept: bool,
}

impl LogisticRegressionModel {
    const NDARRAY_FASTPATH_MIN_WORK: usize = 32_768;
    const NDARRAY_FASTPATH_MAX_MAMS_CHAINS: usize = 8;

    /// Create a new logistic regression model from row-wise `X` and binary `y`.
    pub fn new(x: Vec<Vec<f64>>, y: Vec<u8>, include_intercept: bool) -> Result<Self> {
        let x = DenseX::from_rows(x)?;
        validate_xy_dims(x.n, x.p, x.data.len(), y.len())?;
        if y.iter().any(|&v| v != 0 && v != 1) {
            return Err(Error::Validation("y must contain only 0/1 values".to_string()));
        }
        Ok(Self { x, y, include_intercept })
    }

    #[inline]
    fn dim_internal(&self) -> usize {
        self.x.p + if self.include_intercept { 1 } else { 0 }
    }

    /// Number of observations.
    pub fn n_obs(&self) -> usize {
        self.x.n
    }

    /// Number of feature columns (without intercept).
    pub fn n_features(&self) -> usize {
        self.x.p
    }

    #[inline]
    fn eta(&self, i: usize, params: &[f64]) -> f64 {
        if self.include_intercept {
            let (b0, beta) = params.split_first().expect("params non-empty when include_intercept");
            *b0 + row_dot(self.x.row(i), beta)
        } else {
            row_dot(self.x.row(i), params)
        }
    }

    #[inline]
    fn use_ndarray_fastpath(&self) -> bool {
        let chains_hint = crate::perf_hints::mams_chain_hint();
        let chain_ok = chains_hint == 0 || chains_hint <= Self::NDARRAY_FASTPATH_MAX_MAMS_CHAINS;
        chain_ok
            && self.x.n.saturating_mul(self.x.p) >= Self::NDARRAY_FASTPATH_MIN_WORK
            && std::env::var_os("NEXTSTAT_DISABLE_GLM_NDARRAY").is_none()
    }

    fn nll_grad_fused_scalar(&self, params: &[f64]) -> (f64, Vec<f64>) {
        let mut nll = 0.0;
        let mut grad = vec![0.0; self.dim_internal()];

        if self.include_intercept {
            let b0 = params[0];
            let beta = &params[1..];
            for i in 0..self.x.n {
                let row = self.x.row(i);
                let eta = b0 + row_dot(row, beta);
                let yi = self.y[i] as f64;
                nll += log1pexp(eta) - yi * eta;
                let err = sigmoid(eta) - yi;
                grad[0] += err;
                for j in 0..self.x.p {
                    grad[1 + j] += err * row[j];
                }
            }
        } else {
            for i in 0..self.x.n {
                let row = self.x.row(i);
                let eta = row_dot(row, params);
                let yi = self.y[i] as f64;
                nll += log1pexp(eta) - yi * eta;
                let err = sigmoid(eta) - yi;
                for j in 0..self.x.p {
                    grad[j] += err * row[j];
                }
            }
        }

        (nll, grad)
    }

    fn nll_grad_fused_ndarray(&self, params: &[f64]) -> (f64, Vec<f64>) {
        let x = ArrayView2::from_shape((self.x.n, self.x.p), &self.x.data)
            .expect("DenseX shape must match n*p");

        let (intercept, beta) =
            if self.include_intercept { (params[0], &params[1..]) } else { (0.0, params) };
        let beta_view = ArrayView1::from(beta);

        // eta = X @ beta (+ intercept if present)
        let mut eta = x.dot(&beta_view);
        if self.include_intercept {
            eta.mapv_inplace(|v| v + intercept);
        }

        let mut nll = 0.0;
        let mut diff = vec![0.0; self.x.n];
        for i in 0..self.x.n {
            let yi = self.y[i] as f64;
            let et = eta[i];
            nll += log1pexp(et) - yi * et;
            diff[i] = sigmoid(et) - yi;
        }

        let diff_view = ArrayView1::from(&diff);
        let grad_beta = x.t().dot(&diff_view);

        let mut grad = vec![0.0; self.dim_internal()];
        if self.include_intercept {
            grad[0] = diff.iter().sum();
            for j in 0..self.x.p {
                grad[1 + j] = grad_beta[j];
            }
        } else {
            for j in 0..self.x.p {
                grad[j] = grad_beta[j];
            }
        }

        (nll, grad)
    }
}

impl LogDensityModel for LogisticRegressionModel {
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
        out
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![(f64::NEG_INFINITY, f64::INFINITY); self.dim_internal()]
    }

    fn parameter_init(&self) -> Vec<f64> {
        vec![0.0; self.dim_internal()]
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        if params.len() != self.dim_internal() {
            return Err(Error::Validation(format!(
                "expected {} parameters, got {}",
                self.dim_internal(),
                params.len()
            )));
        }
        if params.iter().any(|v| !v.is_finite()) {
            return Err(Error::Validation("params must contain only finite values".to_string()));
        }

        let mut nll = 0.0;
        for i in 0..self.x.n {
            let eta = self.eta(i, params);
            nll += log1pexp(eta) - (self.y[i] as f64) * eta;
        }
        Ok(nll)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        if params.len() != self.dim_internal() {
            return Err(Error::Validation(format!(
                "expected {} parameters, got {}",
                self.dim_internal(),
                params.len()
            )));
        }
        if params.iter().any(|v| !v.is_finite()) {
            return Err(Error::Validation("params must contain only finite values".to_string()));
        }

        let mut grad = vec![0.0; self.dim_internal()];
        for i in 0..self.x.n {
            let eta = self.eta(i, params);
            let mu = sigmoid(eta);
            let err = mu - (self.y[i] as f64);
            if self.include_intercept {
                grad[0] += err;
                let row = self.x.row(i);
                for j in 0..self.x.p {
                    grad[1 + j] += err * row[j];
                }
            } else {
                let row = self.x.row(i);
                for j in 0..self.x.p {
                    grad[j] += err * row[j];
                }
            }
        }
        Ok(grad)
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }

    fn prefer_fused_eval_grad(&self) -> bool {
        true
    }

    fn nll_grad_prepared(
        &self,
        _prepared: &Self::Prepared<'_>,
        params: &[f64],
    ) -> Result<(f64, Vec<f64>)> {
        if params.len() != self.dim_internal() {
            return Err(Error::Validation(format!(
                "expected {} parameters, got {}",
                self.dim_internal(),
                params.len()
            )));
        }
        if params.iter().any(|v| !v.is_finite()) {
            return Err(Error::Validation("params must contain only finite values".to_string()));
        }
        let out = if self.use_ndarray_fastpath() {
            self.nll_grad_fused_ndarray(params)
        } else {
            self.nll_grad_fused_scalar(params)
        };
        Ok(out)
    }
}

/// Poisson regression with log link.
///
/// Model:
/// `y_i ~ Poisson(exp(eta_i))`, `eta_i = intercept + X_i * beta + offset_i`
///
/// NLL (up to additive constant): `sum_i exp(eta_i) - y_i * eta_i`
#[derive(Debug, Clone)]
pub struct PoissonRegressionModel {
    x: DenseX,
    y: Vec<u64>,
    include_intercept: bool,
    offset: Option<Vec<f64>>,
}

impl PoissonRegressionModel {
    /// Create a new Poisson regression model.
    ///
    /// Offset is optional; if provided, must have length `n` and all finite values.
    pub fn new(
        x: Vec<Vec<f64>>,
        y: Vec<u64>,
        include_intercept: bool,
        offset: Option<Vec<f64>>,
    ) -> Result<Self> {
        let x = DenseX::from_rows(x)?;
        validate_xy_dims(x.n, x.p, x.data.len(), y.len())?;
        if let Some(off) = &offset {
            if off.len() != x.n {
                return Err(Error::Validation(format!(
                    "offset has wrong length: expected n={}, got {}",
                    x.n,
                    off.len()
                )));
            }
            if off.iter().any(|v| !v.is_finite()) {
                return Err(Error::Validation(
                    "offset must contain only finite values".to_string(),
                ));
            }
        }
        Ok(Self { x, y, include_intercept, offset })
    }

    #[inline]
    fn dim_internal(&self) -> usize {
        self.x.p + if self.include_intercept { 1 } else { 0 }
    }

    #[inline]
    fn eta(&self, i: usize, params: &[f64]) -> f64 {
        let base = if self.include_intercept {
            let (b0, beta) = params.split_first().expect("params non-empty when include_intercept");
            *b0 + row_dot(self.x.row(i), beta)
        } else {
            row_dot(self.x.row(i), params)
        };
        if let Some(off) = &self.offset { base + off[i] } else { base }
    }
}

impl LogDensityModel for PoissonRegressionModel {
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
        out
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![(f64::NEG_INFINITY, f64::INFINITY); self.dim_internal()]
    }

    fn parameter_init(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.dim_internal()];
        // Moment-style init: start intercept at log(mean(y)) to keep the first line-search
        // step in a sane region for count data.
        if self.include_intercept {
            let n = self.y.len().max(1) as f64;
            let mean_y = self.y.iter().map(|&v| v as f64).sum::<f64>() / n;
            if mean_y.is_finite() && mean_y > 0.0 {
                out[0] = mean_y.max(1e-12).ln();
            }
        }
        out
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        if params.len() != self.dim_internal() {
            return Err(Error::Validation(format!(
                "expected {} parameters, got {}",
                self.dim_internal(),
                params.len()
            )));
        }
        if params.iter().any(|v| !v.is_finite()) {
            return Ok(1e300);
        }

        let mut nll = 0.0;
        for i in 0..self.x.n {
            let eta = self.eta(i, params);
            // Clamp eta more aggressively than `exp_clamped` to avoid generating enormous
            // mu/gradients during line search, which can overflow internal L-BFGS dot-products.
            let mu = (eta.clamp(-50.0, 50.0)).exp();
            // Use ln(mu) (which is clamped by `exp_clamped`) instead of raw `eta` to avoid
            // NaN/Inf during line search when proposals momentarily push eta out of range.
            // When eta is in range, `mu.ln() == eta` and this is exactly equivalent.
            nll += mu - (self.y[i] as f64) * mu.ln();
        }
        Ok(nll)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        if params.len() != self.dim_internal() {
            return Err(Error::Validation(format!(
                "expected {} parameters, got {}",
                self.dim_internal(),
                params.len()
            )));
        }
        if params.iter().any(|v| !v.is_finite()) {
            return Ok(vec![0.0; self.dim_internal()]);
        }

        let mut grad = vec![0.0; self.dim_internal()];
        for i in 0..self.x.n {
            let eta = self.eta(i, params);
            let mu = (eta.clamp(-50.0, 50.0)).exp();
            let err = mu - (self.y[i] as f64);
            if self.include_intercept {
                grad[0] += err;
                let row = self.x.row(i);
                for j in 0..self.x.p {
                    grad[1 + j] += err * row[j];
                }
            } else {
                let row = self.x.row(i);
                for j in 0..self.x.p {
                    grad[j] += err * row[j];
                }
            }
        }
        Ok(grad)
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

/// Negative binomial regression (NB2) with log link.
///
/// Model (GLM parameterization):
/// `y_i ~ NegBin(mean=mu_i, dispersion=alpha)`, `mu_i = exp(eta_i)`,
/// `eta_i = intercept + X_i * beta + offset_i`.
///
/// Dispersion `alpha > 0` implies: `Var(Y) = mu + alpha * mu^2`.
/// Internally we parameterize `alpha = exp(log_alpha)` to avoid bounds.
#[derive(Debug, Clone)]
pub struct NegativeBinomialRegressionModel {
    x: DenseX,
    y: Vec<u64>,
    include_intercept: bool,
    offset: Option<Vec<f64>>,
}

impl NegativeBinomialRegressionModel {
    /// Create a new negative binomial regression model.
    ///
    /// Offset is optional; if provided, must have length `n` and all finite values.
    pub fn new(
        x: Vec<Vec<f64>>,
        y: Vec<u64>,
        include_intercept: bool,
        offset: Option<Vec<f64>>,
    ) -> Result<Self> {
        let x = DenseX::from_rows(x)?;
        validate_xy_dims(x.n, x.p, x.data.len(), y.len())?;
        if let Some(off) = &offset {
            if off.len() != x.n {
                return Err(Error::Validation(format!(
                    "offset has wrong length: expected n={}, got {}",
                    x.n,
                    off.len()
                )));
            }
            if off.iter().any(|v| !v.is_finite()) {
                return Err(Error::Validation(
                    "offset must contain only finite values".to_string(),
                ));
            }
        }
        Ok(Self { x, y, include_intercept, offset })
    }

    #[inline]
    fn dim_internal(&self) -> usize {
        // beta + log_alpha
        (self.x.p + if self.include_intercept { 1 } else { 0 }) + 1
    }

    #[inline]
    fn split_params<'a>(&self, params: &'a [f64]) -> Result<(&'a [f64], f64)> {
        if params.len() != self.dim_internal() {
            return Err(Error::Validation(format!(
                "expected {} parameters, got {}",
                self.dim_internal(),
                params.len()
            )));
        }
        let (beta, log_alpha) = params.split_at(params.len() - 1);
        Ok((beta, log_alpha[0]))
    }

    #[inline]
    fn eta(&self, i: usize, beta: &[f64]) -> f64 {
        let base = if self.include_intercept {
            let (b0, b) = beta.split_first().expect("beta non-empty when include_intercept");
            *b0 + row_dot(self.x.row(i), b)
        } else {
            row_dot(self.x.row(i), beta)
        };
        if let Some(off) = &self.offset { base + off[i] } else { base }
    }
}

impl LogDensityModel for NegativeBinomialRegressionModel {
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
        out.push("log_alpha".to_string());
        out
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        // Bound log_alpha to avoid pathological theta=1/alpha behavior during line-search steps:
        // alpha -> +inf implies theta -> 0 (Gamma singularities), which can produce NaNs/Infs.
        // This keeps optimization stable while remaining effectively uninformative for real fits.
        let mut b = vec![(f64::NEG_INFINITY, f64::INFINITY); self.dim_internal()];
        // Avoid theta = 1/alpha becoming so small that Gamma(theta) overflows and we hit inf-inf
        // cancellation for y=0 terms during line search. This bound is still extremely wide in
        // practice (alpha up to ~3e3).
        // Also avoid alpha -> 0 (theta -> +inf) which can overflow ln_gamma(y+theta).
        b[self.dim_internal() - 1] = (-10.0, 8.0);
        b
    }

    fn parameter_init(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.dim_internal()];

        // Robust moment-style init so the very first line-search step doesn't walk into
        // invalid Gamma/digamma territory for NB dispersion.
        //
        // NB2: Var(Y) = mu + alpha * mu^2  =>  alpha ≈ (Var - mu) / mu^2
        let n = self.y.len().max(1) as f64;
        let mean_y = self.y.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var_y = self
            .y
            .iter()
            .map(|&v| {
                let d = v as f64 - mean_y;
                d * d
            })
            .sum::<f64>()
            / n;

        // Initialize intercept at log(mean_y) for a reasonable mu scale.
        if self.include_intercept && mean_y.is_finite() && mean_y > 0.0 {
            out[0] = mean_y.max(1e-12).ln();
        }

        let alpha = if mean_y.is_finite() && mean_y > 0.0 && var_y.is_finite() {
            ((var_y - mean_y) / (mean_y * mean_y)).max(1e-6)
        } else {
            0.2
        };
        out[self.dim_internal() - 1] = alpha.ln().clamp(-10.0, 8.0);

        out
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        if params.iter().any(|v| !v.is_finite()) {
            return Ok(1e300);
        }
        let (beta, log_alpha) = self.split_params(params)?;
        const LOG_ALPHA_LO: f64 = -10.0;
        const LOG_ALPHA_HI: f64 = 8.0;
        let log_alpha = log_alpha.clamp(LOG_ALPHA_LO, LOG_ALPHA_HI);
        let alpha = log_alpha.exp();
        if !alpha.is_finite() || alpha <= 0.0 {
            return Ok(1e300);
        }
        let theta = 1.0 / alpha;
        if !theta.is_finite() || theta <= 0.0 {
            return Ok(1e300);
        }

        let mut nll = 0.0;
        for i in 0..self.x.n {
            let eta = self.eta(i, beta);
            let mu = exp_clamped(eta);
            let y = self.y[i] as f64;

            // logpmf (NB2 mean/disp) ignoring constant -ln(y!)
            // log p = ln Gamma(y+theta) - ln Gamma(theta) + theta ln(theta/(theta+mu)) + y ln(mu/(theta+mu))
            let ln_p = ln_gamma(y + theta) - ln_gamma(theta)
                + theta * (theta.ln() - (theta + mu).ln())
                + y * (mu.ln() - (theta + mu).ln());
            nll -= ln_p;
        }
        Ok(if nll.is_finite() { nll } else { 1e300 })
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        if params.iter().any(|v| !v.is_finite()) {
            return Ok(vec![0.0; self.dim_internal()]);
        }
        let (beta, log_alpha_raw) = self.split_params(params)?;
        const LOG_ALPHA_LO: f64 = -10.0;
        const LOG_ALPHA_HI: f64 = 8.0;
        let log_alpha = log_alpha_raw.clamp(LOG_ALPHA_LO, LOG_ALPHA_HI);
        let alpha = log_alpha.exp();
        if !alpha.is_finite() || alpha <= 0.0 {
            return Ok(vec![0.0; self.dim_internal()]);
        }
        let theta = 1.0 / alpha;
        if !theta.is_finite() || theta <= 0.0 {
            return Ok(vec![0.0; self.dim_internal()]);
        }

        let n_beta = beta.len();
        let mut grad_beta = vec![0.0; n_beta];
        let mut grad_log_alpha = 0.0;

        for i in 0..self.x.n {
            let eta = self.eta(i, beta);
            let mu = exp_clamped(eta);
            let y_u = self.y[i];
            let y = y_u as f64;

            // d/deta nll = mu * (theta + y) / (theta + mu) - y
            let err = mu * (theta + y) / (theta + mu) - y;
            if self.include_intercept {
                grad_beta[0] += err;
                let row = self.x.row(i);
                for j in 0..self.x.p {
                    grad_beta[1 + j] += err * row[j];
                }
            } else {
                let row = self.x.row(i);
                for j in 0..self.x.p {
                    grad_beta[j] += err * row[j];
                }
            }

            // d/dtheta log p:
            // psi(y+theta) - psi(theta) + ln(theta) + 1 - ln(theta+mu) - (theta+y)/(theta+mu)
            let d_logp_d_theta = digamma(y + theta) - digamma(theta) + theta.ln() + 1.0
                - (theta + mu).ln()
                - (theta + y) / (theta + mu);
            if !d_logp_d_theta.is_finite() {
                continue;
            }
            let d_nll_d_theta = -d_logp_d_theta;
            // theta = 1/alpha, so d/d log_alpha = -theta * d/dtheta
            grad_log_alpha += -theta * d_nll_d_theta;
        }

        // If the optimizer proposes values outside the clamp range, the objective is
        // locally flat in log_alpha (projection), so report zero sensitivity.
        if log_alpha_raw <= LOG_ALPHA_LO || log_alpha_raw >= LOG_ALPHA_HI {
            grad_log_alpha = 0.0;
        }
        if !grad_log_alpha.is_finite() {
            grad_log_alpha = 0.0;
        }
        for g in &mut grad_beta {
            if !g.is_finite() {
                *g = 0.0;
            }
        }

        let mut out = Vec::with_capacity(self.dim_internal());
        out.extend_from_slice(&grad_beta);
        out.push(grad_log_alpha);
        Ok(out)
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Gamma, StandardNormal};
    use serde::Deserialize;

    #[derive(Debug, Clone, Deserialize)]
    struct Fixture {
        kind: String,
        include_intercept: bool,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        offset: Option<Vec<f64>>,
        beta_hat: Vec<f64>,
        log_alpha_hat: Option<f64>,
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
    fn test_ols_fit_matches_fixture_hat() {
        let fx = load_fixture(include_str!("../../../tests/fixtures/regression/ols_small.json"));
        assert_eq!(fx.kind, "ols");
        let beta = ols_fit(fx.x.clone(), fx.y.clone(), fx.include_intercept).unwrap();
        assert_vec_close(&beta, &fx.beta_hat, 1e-8);
    }

    #[test]
    fn test_linear_regression_nll_and_grad_at_fixture_hat() {
        let fx = load_fixture(include_str!("../../../tests/fixtures/regression/ols_small.json"));
        let m =
            LinearRegressionModel::new(fx.x.clone(), fx.y.clone(), fx.include_intercept).unwrap();
        let nll = m.nll(&fx.beta_hat).unwrap();
        assert!((nll - fx.nll_at_hat).abs() < 1e-8);
        let g = m.grad_nll(&fx.beta_hat).unwrap();
        assert!(inf_norm(&g) < 1e-6, "grad inf-norm too large: {}", inf_norm(&g));
    }

    #[test]
    fn test_logistic_regression_nll_and_grad_at_fixture_hat() {
        let fx =
            load_fixture(include_str!("../../../tests/fixtures/regression/logistic_small.json"));
        assert_eq!(fx.kind, "logistic");
        let y: Vec<u8> = fx.y.iter().map(|&v| if v >= 0.5 { 1u8 } else { 0u8 }).collect();
        let m = LogisticRegressionModel::new(fx.x.clone(), y, fx.include_intercept).unwrap();
        let nll = m.nll(&fx.beta_hat).unwrap();
        assert!((nll - fx.nll_at_hat).abs() < 1e-6);
        let g = m.grad_nll(&fx.beta_hat).unwrap();
        assert!(inf_norm(&g) < 1e-6, "grad inf-norm too large: {}", inf_norm(&g));
    }

    #[test]
    fn test_logistic_regression_fused_nll_grad_matches_separate() {
        let fx =
            load_fixture(include_str!("../../../tests/fixtures/regression/logistic_small.json"));
        let y: Vec<u8> = fx.y.iter().map(|&v| if v >= 0.5 { 1u8 } else { 0u8 }).collect();
        let m = LogisticRegressionModel::new(fx.x.clone(), y, fx.include_intercept).unwrap();

        let nll_ref = m.nll(&fx.beta_hat).unwrap();
        let grad_ref = m.grad_nll(&fx.beta_hat).unwrap();

        let prepared = m.prepared();
        let (nll_fused, grad_fused) = m.nll_grad_prepared(&prepared, &fx.beta_hat).unwrap();

        assert!((nll_ref - nll_fused).abs() < 1e-12);
        assert_vec_close(&grad_ref, &grad_fused, 1e-12);
    }

    #[test]
    fn test_logistic_regression_ndarray_fastpath_matches_scalar() {
        let n = 256usize;
        let p = 128usize; // n*p == fast-path threshold
        let mut x = vec![vec![0.0; p]; n];
        let mut y = vec![0u8; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..p {
                let v = ((i * 17 + j * 31) as f64 * 0.001).sin();
                x[i][j] = v;
                s += v * ((j as f64 + 1.0) * 0.003).cos();
            }
            y[i] = if s > 0.0 { 1 } else { 0 };
        }

        let m = LogisticRegressionModel::new(x, y, true).unwrap();
        let mut params = vec![0.0; p + 1];
        for (k, v) in params.iter_mut().enumerate() {
            *v = ((k as f64) * 0.01).cos() * 0.1;
        }

        let (nll_scalar, grad_scalar) = m.nll_grad_fused_scalar(&params);
        let (nll_nd, grad_nd) = m.nll_grad_fused_ndarray(&params);

        assert!((nll_scalar - nll_nd).abs() < 1e-10);
        assert_vec_close(&grad_scalar, &grad_nd, 1e-10);
    }

    #[test]
    fn test_logistic_regression_fastpath_respects_mams_chain_hint() {
        let n = 256usize;
        let p = 128usize;
        let x = vec![vec![0.0; p]; n];
        let y = vec![0u8; n];
        let m = LogisticRegressionModel::new(x, y, true).unwrap();

        crate::perf_hints::set_mams_chain_hint(4);
        assert!(m.use_ndarray_fastpath());

        crate::perf_hints::set_mams_chain_hint(64);
        assert!(!m.use_ndarray_fastpath());

        crate::perf_hints::clear_mams_chain_hint();
    }

    #[test]
    fn test_poisson_regression_nll_and_grad_at_fixture_hat() {
        let fx =
            load_fixture(include_str!("../../../tests/fixtures/regression/poisson_small.json"));
        assert_eq!(fx.kind, "poisson");
        let y: Vec<u64> = fx.y.iter().map(|&v| v.round() as u64).collect();
        let m =
            PoissonRegressionModel::new(fx.x.clone(), y, fx.include_intercept, fx.offset.clone())
                .unwrap();
        let nll = m.nll(&fx.beta_hat).unwrap();
        assert!((nll - fx.nll_at_hat).abs() < 1e-6);
        let g = m.grad_nll(&fx.beta_hat).unwrap();
        assert!(inf_norm(&g) < 1e-6, "grad inf-norm too large: {}", inf_norm(&g));
    }

    #[test]
    fn test_negbin_regression_nll_and_grad_at_fixture_hat() {
        let fx = load_fixture(include_str!("../../../tests/fixtures/regression/negbin_small.json"));
        assert_eq!(fx.kind, "negbin");
        let y: Vec<u64> = fx.y.iter().map(|&v| v.round() as u64).collect();
        let log_alpha = fx.log_alpha_hat.expect("negbin fixture must include log_alpha_hat");
        let mut params = fx.beta_hat.clone();
        params.push(log_alpha);
        let m = NegativeBinomialRegressionModel::new(
            fx.x.clone(),
            y,
            fx.include_intercept,
            fx.offset.clone(),
        )
        .unwrap();
        let nll = m.nll(&params).unwrap();
        assert!((nll - fx.nll_at_hat).abs() < 1e-6);
        let g = m.grad_nll(&params).unwrap();
        assert!(inf_norm(&g) < 1e-6, "grad inf-norm too large: {}", inf_norm(&g));
    }

    #[test]
    fn test_fixture_contract_shapes() {
        let fx =
            load_fixture(include_str!("../../../tests/fixtures/regression/logistic_small.json"));
        assert!(fx.include_intercept);
        assert!(!fx.x.is_empty());
        assert_eq!(fx.x.len(), fx.y.len());
        let p = fx.x[0].len();
        assert!(p > 0);
        assert_eq!(fx.beta_hat.len(), p + 1);
    }

    #[test]
    fn test_mle_linear_regression_recovers_fixture_hat() {
        let fx = load_fixture(include_str!("../../../tests/fixtures/regression/ols_small.json"));
        assert_eq!(fx.kind, "ols");
        let m =
            LinearRegressionModel::new(fx.x.clone(), fx.y.clone(), fx.include_intercept).unwrap();

        let mle = crate::mle::MaximumLikelihoodEstimator::new();
        let r = mle.fit(&m).unwrap();
        assert!(r.converged, "MLE should converge on OLS fixture");
        assert_vec_close(&r.parameters, &fx.beta_hat, 1e-6);
        assert!(
            (r.nll - fx.nll_at_hat).abs() < 1e-8,
            "nll mismatch: got {}, expected {}",
            r.nll,
            fx.nll_at_hat
        );
    }

    #[test]
    fn test_mle_logistic_regression_recovers_fixture_hat() {
        let fx =
            load_fixture(include_str!("../../../tests/fixtures/regression/logistic_small.json"));
        assert_eq!(fx.kind, "logistic");
        let y: Vec<u8> = fx.y.iter().map(|&v| if v >= 0.5 { 1 } else { 0 }).collect();
        let m = LogisticRegressionModel::new(fx.x.clone(), y, fx.include_intercept).unwrap();

        let mle = crate::mle::MaximumLikelihoodEstimator::new();
        let r = mle.fit(&m).unwrap();
        assert!(r.converged, "MLE should converge on logistic fixture");
        // Fixture hat is a numerical optimum; allow solver tolerance.
        assert_vec_close(&r.parameters, &fx.beta_hat, 1e-4);
        assert!(
            (r.nll - fx.nll_at_hat).abs() < 1e-6,
            "nll mismatch: got {}, expected {}",
            r.nll,
            fx.nll_at_hat
        );
    }

    #[test]
    fn test_mle_poisson_regression_recovers_fixture_hat() {
        let fx =
            load_fixture(include_str!("../../../tests/fixtures/regression/poisson_small.json"));
        assert_eq!(fx.kind, "poisson");
        let y: Vec<u64> = fx.y.iter().map(|&v| v.round() as u64).collect();
        let m =
            PoissonRegressionModel::new(fx.x.clone(), y, fx.include_intercept, fx.offset.clone())
                .unwrap();

        let mle = crate::mle::MaximumLikelihoodEstimator::new();
        let r = mle.fit(&m).unwrap();
        assert!(r.converged, "MLE should converge on poisson fixture");
        assert_vec_close(&r.parameters, &fx.beta_hat, 1e-4);
        assert!(
            (r.nll - fx.nll_at_hat).abs() < 1e-6,
            "nll mismatch: got {}, expected {}",
            r.nll,
            fx.nll_at_hat
        );
    }

    #[test]
    fn test_mle_negbin_regression_recovers_fixture_hat() {
        let fx = load_fixture(include_str!("../../../tests/fixtures/regression/negbin_small.json"));
        assert_eq!(fx.kind, "negbin");
        let y: Vec<u64> = fx.y.iter().map(|&v| v.round() as u64).collect();
        let log_alpha = fx.log_alpha_hat.expect("negbin fixture must include log_alpha_hat");
        let mut params_hat = fx.beta_hat.clone();
        params_hat.push(log_alpha);
        let m = NegativeBinomialRegressionModel::new(
            fx.x.clone(),
            y,
            fx.include_intercept,
            fx.offset.clone(),
        )
        .unwrap();

        let mle = crate::mle::MaximumLikelihoodEstimator::new();
        let r = mle.fit(&m).unwrap();
        assert!(r.converged, "MLE should converge on negbin fixture");
        assert_vec_close(&r.parameters, &params_hat, 1e-4);
        assert!(
            (r.nll - fx.nll_at_hat).abs() < 1e-6,
            "nll mismatch: got {}, expected {}",
            r.nll,
            fx.nll_at_hat
        );
    }

    fn make_negbin_benchmark_like_dataset(
        n: usize,
        p: usize,
        seed: u64,
    ) -> (Vec<Vec<f64>>, Vec<u64>, bool) {
        let mut rng = StdRng::seed_from_u64(seed);

        let mut x = vec![vec![0.0; p]; n];
        for row in &mut x {
            for v in row {
                *v = StandardNormal.sample(&mut rng);
            }
        }

        let mut beta = vec![0.0; p];
        if p == 1 {
            beta[0] = 0.5;
        } else {
            let denom = (p - 1) as f64;
            for (j, b) in beta.iter_mut().enumerate() {
                *b = 0.5 - (j as f64) * (1.0 / denom);
            }
        }
        let intercept = 0.5;
        let alpha = 1.0 / 5.0;
        let k = 1.0 / alpha;

        let mut y = Vec::with_capacity(n);
        for row in &x {
            let mut eta = intercept;
            for (xj, bj) in row.iter().zip(&beta) {
                eta += xj * bj;
            }
            let mu = eta.clamp(-10.0, 10.0).exp();
            let gamma = Gamma::new(k, alpha * mu).unwrap();
            let lambda = gamma.sample(&mut rng);
            let dist = rand_distr::Poisson::new(lambda.max(1e-12)).unwrap();
            y.push(dist.sample(&mut rng) as u64);
        }

        (x, y, true)
    }

    fn make_poisson_benchmark_like_dataset(
        n: usize,
        p: usize,
        seed: u64,
    ) -> (Vec<Vec<f64>>, Vec<u64>, bool) {
        let mut rng = StdRng::seed_from_u64(seed);

        let mut x = vec![vec![0.0; p]; n];
        for row in &mut x {
            for v in row {
                *v = StandardNormal.sample(&mut rng);
            }
        }

        let mut beta = vec![0.0; p];
        if p == 1 {
            beta[0] = 0.5;
        } else {
            let denom = (p - 1) as f64;
            for (j, b) in beta.iter_mut().enumerate() {
                *b = 0.5 - (j as f64) * (1.0 / denom);
            }
        }
        let intercept = 0.5;

        let mut y = Vec::with_capacity(n);
        for row in &x {
            let mut eta = intercept;
            for (xj, bj) in row.iter().zip(&beta) {
                eta += xj * bj;
            }
            let mu = eta.clamp(-10.0, 10.0).exp();
            let dist = rand_distr::Poisson::new(mu.max(1e-12)).unwrap();
            y.push(dist.sample(&mut rng) as u64);
        }

        (x, y, true)
    }

    #[test]
    fn test_mle_poisson_benchmark_like_dataset_converges() {
        let (x, y, include_intercept) = make_poisson_benchmark_like_dataset(1_000, 10, 42);
        let m = PoissonRegressionModel::new(x, y, include_intercept, None).unwrap();
        let mle = crate::mle::MaximumLikelihoodEstimator::new();
        let r = mle.fit(&m).unwrap();
        assert!(
            r.converged,
            "expected convergence on benchmark-like poisson data; reason={}",
            r.termination_reason
        );
        assert!(r.n_iter > 0, "optimizer should perform at least one iteration");
        assert!(r.nll.is_finite(), "nll should be finite");
        assert!(
            r.parameters.iter().all(|v| v.is_finite()),
            "all fitted parameters should be finite"
        );
    }

    #[test]
    fn test_mle_negbin_benchmark_like_dataset_converges() {
        let (x, y, include_intercept) = make_negbin_benchmark_like_dataset(1_000, 10, 42);
        let m = NegativeBinomialRegressionModel::new(x, y, include_intercept, None).unwrap();
        let mle = crate::mle::MaximumLikelihoodEstimator::new();
        let r = mle.fit(&m).unwrap();
        assert!(
            r.converged,
            "expected convergence on benchmark-like negbin data; reason={}",
            r.termination_reason
        );
        assert!(r.n_iter > 0, "optimizer should perform at least one iteration");
        assert!(r.nll.is_finite(), "nll should be finite");
        assert!(
            r.parameters.iter().all(|v| v.is_finite()),
            "all fitted parameters should be finite"
        );
    }
}
