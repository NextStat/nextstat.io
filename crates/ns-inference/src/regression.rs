//! Regression models for general statistics use-cases (Phase 6).
//!
//! These models implement [`ns_core::traits::LogDensityModel`], so they can be used with:
//! - `MaximumLikelihoodEstimator` (MLE/MAP)
//! - `sample_nuts` / `sample_nuts_multichain` (Bayesian sampling)

use nalgebra::{DMatrix, DVector};
use ndarray::{ArrayView1, ArrayView2};
use ns_core::traits::{LogDensityModel, PreparedModelRef};
use ns_core::{Error, Result};
use ns_prob::math::{exp_clamped, log1pexp, log1pexp_and_sigmoid, sigmoid};
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
    y: Vec<u8>,     // 0/1
    y_f64: Vec<f64>, // pre-cast y for hot loops (avoids u8→f64 per obs per step)
    include_intercept: bool,
}

impl LogisticRegressionModel {
    const NDARRAY_FASTPATH_MIN_WORK: usize = 100;
    const NDARRAY_FASTPATH_MAX_MAMS_CHAINS: usize = 8;

    /// Create a new logistic regression model from row-wise `X` and binary `y`.
    pub fn new(x: Vec<Vec<f64>>, y: Vec<u8>, include_intercept: bool) -> Result<Self> {
        let x = DenseX::from_rows(x)?;
        validate_xy_dims(x.n, x.p, x.data.len(), y.len())?;
        if y.iter().any(|&v| v != 0 && v != 1) {
            return Err(Error::Validation("y must contain only 0/1 values".to_string()));
        }
        let y_f64: Vec<f64> = y.iter().map(|&v| v as f64).collect();
        Ok(Self { x, y, y_f64, include_intercept })
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

    /// Minimum feature count for ndarray to beat scalar single-pass.
    ///
    /// Without BLAS, ndarray uses matrixmultiply which does 3 passes over the data
    /// (X@beta, element-wise, X.t()@diff) plus 2 temp allocations (eta + diff).
    /// For small p, LLVM auto-vectorizes the scalar loop and the single-pass
    /// approach avoids L2 cache thrashing.  Empirically, scalar wins for p < 64
    /// on EPYC 7502P without BLAS.
    const NDARRAY_FASTPATH_MIN_P: usize = 64;

    #[inline]
    fn use_ndarray_fastpath(&self) -> bool {
        let chains_hint = crate::perf_hints::mams_chain_hint();
        let chain_ok = chains_hint == 0 || chains_hint <= Self::NDARRAY_FASTPATH_MAX_MAMS_CHAINS;
        chain_ok
            && self.x.p >= Self::NDARRAY_FASTPATH_MIN_P
            && self.x.n.saturating_mul(self.x.p) >= Self::NDARRAY_FASTPATH_MIN_WORK
            && std::env::var_os("NEXTSTAT_DISABLE_GLM_NDARRAY").is_none()
    }

    fn nll_grad_fused_scalar(&self, params: &[f64]) -> (f64, Vec<f64>) {
        let dim = self.dim_internal();
        let mut nll = 0.0;
        let mut grad = vec![0.0; dim];
        let p = self.x.p;
        let n = self.x.n;
        let b0 = if self.include_intercept { params[0] } else { 0.0 };
        let beta = if self.include_intercept { &params[1..] } else { params };
        let g_off = if self.include_intercept { 1 } else { 0 };

        // 4x unrolled main loop: pipeline 4 independent exp() for ILP
        let n4 = n - n % 4;
        let mut i = 0;
        while i < n4 {
            let r0 = self.x.row(i);
            let r1 = self.x.row(i + 1);
            let r2 = self.x.row(i + 2);
            let r3 = self.x.row(i + 3);

            let eta0 = b0 + row_dot(r0, beta);
            let eta1 = b0 + row_dot(r1, beta);
            let eta2 = b0 + row_dot(r2, beta);
            let eta3 = b0 + row_dot(r3, beta);

            let (l0, m0) = log1pexp_and_sigmoid(eta0);
            let (l1, m1) = log1pexp_and_sigmoid(eta1);
            let (l2, m2) = log1pexp_and_sigmoid(eta2);
            let (l3, m3) = log1pexp_and_sigmoid(eta3);

            let y0 = self.y_f64[i];
            let y1 = self.y_f64[i + 1];
            let y2 = self.y_f64[i + 2];
            let y3 = self.y_f64[i + 3];

            nll += (l0 - y0 * eta0) + (l1 - y1 * eta1)
                 + (l2 - y2 * eta2) + (l3 - y3 * eta3);

            let e0 = m0 - y0;
            let e1 = m1 - y1;
            let e2 = m2 - y2;
            let e3 = m3 - y3;

            if self.include_intercept {
                grad[0] += e0 + e1 + e2 + e3;
            }
            for j in 0..p {
                grad[g_off + j] += e0 * r0[j] + e1 * r1[j]
                                 + e2 * r2[j] + e3 * r3[j];
            }
            i += 4;
        }

        // Remainder (0–3 observations)
        while i < n {
            let row = self.x.row(i);
            let eta = b0 + row_dot(row, beta);
            let yi = self.y_f64[i];
            let (log_term, mu) = log1pexp_and_sigmoid(eta);
            nll += log_term - yi * eta;
            let err = mu - yi;
            if self.include_intercept {
                grad[0] += err;
            }
            for j in 0..p {
                grad[g_off + j] += err * row[j];
            }
            i += 1;
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
            let yi = self.y_f64[i];
            let et = eta[i];
            let (log_term, mu) = log1pexp_and_sigmoid(et);
            nll += log_term - yi * et;
            diff[i] = mu - yi;
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
            nll += log1pexp(eta) - self.y_f64[i] * eta;
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
            let err = mu - self.y_f64[i];
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

    // ===================================================================
    // Finite-difference verification of CUDA NegBin kernel math
    // ===================================================================
    //
    // Reproduces EXACTLY the scalar kernel math from mams_leapfrog.cu
    // (nll_glm_negbin + grad_glm_negbin) and checks gradient via FD.

    /// CPU digamma matching device_digamma from mams_leapfrog.cu
    fn cuda_digamma(mut x: f64) -> f64 {
        let mut result = 0.0;
        while x < 8.0 {
            result -= 1.0 / x;
            x += 1.0;
        }
        let inv_x = 1.0 / x;
        let inv_x2 = inv_x * inv_x;
        result += x.ln() - 0.5 * inv_x
            - inv_x2
                * (1.0 / 12.0
                    - inv_x2
                        * (1.0 / 120.0
                            - inv_x2
                                * (1.0 / 252.0
                                    - inv_x2 * (1.0 / 240.0 - inv_x2 * (1.0 / 132.0)))));
        result
    }

    /// NLL matching nll_glm_negbin from mams_leapfrog.cu (including N(0,1) prior).
    /// x = [beta_0..beta_{p-1}, log_alpha], model_data row-major X (no intercept).
    fn cuda_kernel_negbin_nll(x: &[f64], x_mat: &[Vec<f64>], y: &[f64]) -> f64 {
        let n = y.len();
        let p = x.len() - 1;
        let beta = &x[..p];
        let log_alpha = x[p];
        let log_alpha_c = log_alpha.clamp(-10.0, 8.0);
        let alpha = log_alpha_c.exp();
        let theta = 1.0 / alpha;

        let mut nll = 0.0;
        // Prior
        for j in 0..p {
            nll += 0.5 * beta[j] * beta[j];
        }
        nll += 0.5 * log_alpha * log_alpha;

        for i in 0..n {
            let mut eta = 0.0;
            for j in 0..p {
                eta += x_mat[i][j] * beta[j];
            }
            let eta = eta.clamp(-50.0, 50.0);
            let mu = eta.exp();
            let yi = y[i];
            let denom = theta + mu;

            nll -= ln_gamma(yi + theta) - ln_gamma(theta)
                + theta * (theta / denom).ln()
                + yi * (mu / denom).ln();
        }
        nll
    }

    /// Gradient matching grad_glm_negbin from mams_leapfrog.cu (including N(0,1) prior).
    fn cuda_kernel_negbin_grad(x: &[f64], x_mat: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
        let n = y.len();
        let p = x.len() - 1;
        let beta = &x[..p];
        let log_alpha = x[p];
        let log_alpha_c = log_alpha.clamp(-10.0, 8.0);
        let alpha = log_alpha_c.exp();
        let theta = 1.0 / alpha;

        let mut grad = vec![0.0; x.len()];
        // Prior gradients
        for j in 0..p {
            grad[j] = beta[j];
        }
        grad[p] = log_alpha; // prior on log_alpha

        let mut d_log_alpha = 0.0;

        for i in 0..n {
            let mut eta = 0.0;
            for j in 0..p {
                eta += x_mat[i][j] * beta[j];
            }
            let eta = eta.clamp(-50.0, 50.0);
            let mu = eta.exp();
            let yi = y[i];
            let denom = theta + mu;

            // d(NLL_i)/d(eta_i) = mu*(theta+yi)/denom - yi
            let d_eta = mu * (theta + yi) / denom - yi;
            for j in 0..p {
                grad[j] += d_eta * x_mat[i][j];
            }

            // d(NLL_i)/d(log_alpha) via chain rule
            let psi_yi_theta = cuda_digamma(yi + theta);
            let psi_theta = cuda_digamma(theta);
            let d_theta =
                -(psi_yi_theta - psi_theta + (theta / denom).ln() + 1.0 - (theta + yi) / denom);
            d_log_alpha += d_theta * (-theta);
        }

        // Match CUDA fix: zero likelihood gradient when outside clamp range
        if log_alpha <= -10.0 || log_alpha >= 8.0 {
            d_log_alpha = 0.0;
        }

        grad[p] += d_log_alpha;
        grad
    }

    #[test]
    fn test_cuda_negbin_kernel_grad_matches_finite_diff() {
        // Generate small NegBin dataset (no intercept, matching LAPS convention)
        let (x_rows, y_u64, _include_intercept) = make_negbin_benchmark_like_dataset(50, 5, 42);
        // LAPS path: no intercept (user includes it in X if needed)
        // But the benchmark data has include_intercept=true, so p_data = 5 columns
        // and the generated data expects an intercept column.
        // For this test, just use the raw X without intercept handling.
        let y: Vec<f64> = y_u64.iter().map(|&v| v as f64).collect();

        // Test at several parameter values including near-zero and fitted
        let test_points = vec![
            vec![0.0; 6], // all zeros: beta[0..4]=0, log_alpha=0
            vec![0.1, -0.2, 0.3, -0.1, 0.05, 0.5],
            vec![0.5, 0.0, -0.5, 0.2, -0.3, -1.0],
            vec![0.01, 0.02, -0.01, 0.03, -0.02, 2.0],  // large log_alpha
            vec![0.01, 0.02, -0.01, 0.03, -0.02, -5.0],  // small log_alpha (large theta)
            vec![0.01, 0.02, -0.01, 0.03, -0.02, -11.0], // OUTSIDE clamp range (below -10)
            vec![0.01, 0.02, -0.01, 0.03, -0.02, 9.0],   // OUTSIDE clamp range (above 8)
        ];

        let h = 1e-7;

        for (test_idx, params) in test_points.iter().enumerate() {
            let nll = cuda_kernel_negbin_nll(params, &x_rows, &y);
            let grad = cuda_kernel_negbin_grad(params, &x_rows, &y);

            assert!(nll.is_finite(), "NLL not finite at test point {test_idx}: {nll}");

            // Finite-difference gradient check
            for d in 0..params.len() {
                let mut params_plus = params.clone();
                let mut params_minus = params.clone();
                params_plus[d] += h;
                params_minus[d] -= h;

                let nll_plus = cuda_kernel_negbin_nll(&params_plus, &x_rows, &y);
                let nll_minus = cuda_kernel_negbin_nll(&params_minus, &x_rows, &y);
                let fd_grad = (nll_plus - nll_minus) / (2.0 * h);

                let abs_err = (grad[d] - fd_grad).abs();
                let rel_denom = grad[d].abs().max(fd_grad.abs()).max(1e-8);
                let rel_err = abs_err / rel_denom;

                assert!(
                    rel_err < 1e-4 || abs_err < 1e-6,
                    "CUDA NegBin kernel gradient mismatch at test point {test_idx}, param {d} ({}): \
                     analytic={:.8e}, fd={:.8e}, abs_err={:.2e}, rel_err={:.2e}",
                    if d < params.len() - 1 {
                        format!("beta[{d}]")
                    } else {
                        "log_alpha".to_string()
                    },
                    grad[d],
                    fd_grad,
                    abs_err,
                    rel_err,
                );
            }
        }
    }

    /// Also verify that CUDA kernel NLL (minus prior) matches CPU reference NLL.
    #[test]
    fn test_cuda_negbin_nll_matches_cpu_reference() {
        let (x_rows, y_u64, _) = make_negbin_benchmark_like_dataset(50, 5, 42);
        let y_f64: Vec<f64> = y_u64.iter().map(|&v| v as f64).collect();

        // CPU reference model (no intercept to match LAPS convention)
        let m = NegativeBinomialRegressionModel::new(x_rows.clone(), y_u64, false, None).unwrap();

        let params = vec![0.1, -0.2, 0.3, -0.1, 0.05, 0.5]; // 5 beta + log_alpha

        // CPU NLL (likelihood only, no prior)
        let cpu_nll = m.nll(&params).unwrap();

        // CUDA NLL (with prior)
        let cuda_nll = cuda_kernel_negbin_nll(&params, &x_rows, &y_f64);

        // Compute prior
        let prior: f64 = params.iter().map(|&v| 0.5 * v * v).sum();

        let diff = (cuda_nll - cpu_nll - prior).abs();
        assert!(
            diff < 1e-8,
            "CUDA NLL - CPU NLL - prior = {:.2e} (CUDA={:.6}, CPU={:.6}, prior={:.6})",
            diff,
            cuda_nll,
            cpu_nll,
            prior,
        );
    }
}
