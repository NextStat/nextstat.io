//! Linear mixed models (LMM) with Gaussian outcomes (Phase 9 Pack B).
//!
//! This module provides an ML-style marginal likelihood for classic LMMs by
//! integrating out Normal random effects. For the Gaussian case this is exact
//! (often referred to as a "Laplace baseline" in mixed-model tooling).
//!
//! Baseline scope:
//! - right now: random intercept, or random intercept + one random slope
//! - independent (diagonal) random-effects covariance (no correlation)
//! - maximum-likelihood (no priors)

use ns_core::traits::{LogDensityModel, PreparedModelRef};
use ns_core::{Error, Result};

#[inline]
fn validate_xy_dims(n: usize, p: usize, x_len: usize, y_len: usize) -> Result<()> {
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
fn row_dot(x_row: &[f64], beta: &[f64]) -> f64 {
    debug_assert_eq!(x_row.len(), beta.len());
    x_row.iter().zip(beta).map(|(&x, &b)| x * b).sum()
}

/// Cholesky log-determinant of a symmetric positive-definite matrix (row-major flat).
/// Returns `Err` if the matrix is not positive definite.
fn cholesky_logdet(a: &[f64], n: usize) -> Result<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0;
            for k in 0..j {
                s += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let diag = a[i * n + i] - s;
                if diag <= 1e-300 {
                    return Err(Error::Computation(
                        "REML: X^T V^{-1} X is not positive definite".to_string(),
                    ));
                }
                l[i * n + j] = diag.sqrt();
            } else {
                l[i * n + j] = (a[i * n + j] - s) / l[j * n + j];
            }
        }
    }
    let mut logdet = 0.0;
    for i in 0..n {
        logdet += 2.0 * l[i * n + i].ln();
    }
    Ok(logdet)
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
    fn row(&self, i: usize) -> &[f64] {
        let start = i * self.p;
        &self.data[start..start + self.p]
    }
}

#[derive(Debug, Clone)]
struct GroupData {
    indices: Vec<usize>,
    // Sufficient stats for Z^T Z and for Z^T r:
    // intercept-only: s00 = m
    // intercept+slope: s00 = m, s01 = sum xk, s11 = sum xk^2
    s00: f64,
    s01: f64,
    s11: f64,
}

/// Random effects specification.
#[derive(Debug, Clone, Copy)]
pub enum RandomEffects {
    /// Random intercept only.
    Intercept,
    /// Random intercept + one random slope on `feature_idx` (0-based column index in `X`).
    InterceptSlope {
        /// 0-based column index in `X` used as the random-slope covariate.
        feature_idx: usize,
    },
}

/// Gaussian LMM with random intercept (and optional random slope), fit by ML.
///
/// Parameter vector:
/// - fixed effects: `intercept` (optional) + `beta[0..p)`
/// - `log_sigma_y`
/// - `log_tau_alpha`
/// - `log_tau_u` (only if random slope enabled)
#[derive(Debug, Clone)]
pub struct LmmMarginalModel {
    x: DenseX,
    y: Vec<f64>,
    include_intercept: bool,
    n_groups: usize,
    re: RandomEffects,
    groups: Vec<GroupData>,
    use_reml: bool,
}

impl LmmMarginalModel {
    /// Create a new Gaussian LMM with marginal likelihood (random effects integrated out).
    pub fn new(
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        include_intercept: bool,
        group_idx: Vec<usize>,
        n_groups: usize,
        re: RandomEffects,
    ) -> Result<Self> {
        let x = DenseX::from_rows(x)?;
        validate_xy_dims(x.n, x.p, x.data.len(), y.len())?;
        if y.iter().any(|v| !v.is_finite()) {
            return Err(Error::Validation("y must contain only finite values".to_string()));
        }
        if n_groups == 0 {
            return Err(Error::Validation("n_groups must be > 0".to_string()));
        }
        if group_idx.len() != x.n {
            return Err(Error::Validation(format!(
                "group_idx length must match n: expected {}, got {}",
                x.n,
                group_idx.len()
            )));
        }
        if group_idx.iter().any(|&g| g >= n_groups) {
            return Err(Error::Validation("group_idx must be in [0, n_groups)".to_string()));
        }
        if let RandomEffects::InterceptSlope { feature_idx } = re
            && feature_idx >= x.p
        {
            return Err(Error::Validation(format!(
                "feature_idx must be in [0, p), got {} (p={})",
                feature_idx, x.p
            )));
        }

        let mut groups: Vec<GroupData> = (0..n_groups)
            .map(|_| GroupData { indices: Vec::new(), s00: 0.0, s01: 0.0, s11: 0.0 })
            .collect();
        for i in 0..x.n {
            groups[group_idx[i]].indices.push(i);
        }
        for g in 0..n_groups {
            let idxs = groups[g].indices.clone();
            let m = idxs.len();
            if m == 0 {
                continue;
            }
            groups[g].s00 = m as f64;
            match re {
                RandomEffects::Intercept => {}
                RandomEffects::InterceptSlope { feature_idx } => {
                    let mut s01 = 0.0;
                    let mut s11 = 0.0;
                    for &i in &idxs {
                        let xk = x.row(i)[feature_idx];
                        s01 += xk;
                        s11 += xk * xk;
                    }
                    groups[g].s01 = s01;
                    groups[g].s11 = s11;
                }
            }
        }

        Ok(Self { x, y, include_intercept, n_groups, re, groups, use_reml: false })
    }

    /// Enable or disable REML (Restricted Maximum Likelihood) estimation.
    ///
    /// When `reml = true`, the marginal likelihood is corrected by
    /// `+0.5 * log|X^T V^{-1} X|`, which accounts for the loss of degrees of
    /// freedom from estimating the fixed effects β. This produces unbiased
    /// variance component estimates, especially important for small samples.
    ///
    /// Default is `false` (standard ML).
    pub fn with_reml(mut self, reml: bool) -> Self {
        self.use_reml = reml;
        self
    }

    /// Returns `true` if REML estimation is enabled.
    pub fn is_reml(&self) -> bool {
        self.use_reml
    }

    #[inline]
    fn beta_offset(&self) -> usize {
        if self.include_intercept { 1 } else { 0 }
    }

    #[inline]
    fn n_beta(&self) -> usize {
        self.x.p + self.beta_offset()
    }

    #[inline]
    fn unpack<'a>(&self, params: &'a [f64]) -> Result<(&'a [f64], f64, f64, f64, Option<f64>)> {
        let expect = self.dim();
        if params.len() != expect {
            return Err(Error::Validation(format!(
                "expected {} parameters, got {}",
                expect,
                params.len()
            )));
        }
        if params.iter().any(|v| !v.is_finite()) {
            return Err(Error::Validation("params must contain only finite values".to_string()));
        }
        let nb = self.n_beta();
        let beta = &params[..nb];
        let log_sigma_y = params[nb];
        let log_tau_alpha = params[nb + 1];
        let log_tau_u = match self.re {
            RandomEffects::Intercept => None,
            RandomEffects::InterceptSlope { .. } => Some(params[nb + 2]),
        };
        Ok((beta, log_sigma_y, log_tau_alpha, log_tau_u.unwrap_or(0.0), log_tau_u))
    }

    fn nll_internal(
        &self,
        beta: &[f64],
        sigma_y: f64,
        tau_alpha: f64,
        tau_u: Option<f64>,
    ) -> Result<f64> {
        if !sigma_y.is_finite() || sigma_y <= 0.0 {
            return Err(Error::Validation("sigma_y must be finite and > 0".to_string()));
        }
        if !tau_alpha.is_finite() || tau_alpha <= 0.0 {
            return Err(Error::Validation("tau_alpha must be finite and > 0".to_string()));
        }
        if let Some(tu) = tau_u
            && (!tu.is_finite() || tu <= 0.0)
        {
            return Err(Error::Validation("tau_u must be finite and > 0".to_string()));
        }

        let inv_a = 1.0 / (sigma_y * sigma_y);
        let log_a = (sigma_y * sigma_y).ln();
        let log_c =
            (tau_alpha * tau_alpha).ln() + if let Some(tu) = tau_u { (tu * tu).ln() } else { 0.0 };

        let nb = self.n_beta();
        let mut xtvinvx = if self.use_reml { vec![0.0; nb * nb] } else { vec![] };

        let mut nll = 0.0;
        for g in 0..self.n_groups {
            let gd = &self.groups[g];
            let m = gd.indices.len();
            if m == 0 {
                continue;
            }

            // Compute residuals and sufficient stats:
            let mut sum_r2 = 0.0;
            let mut t0 = 0.0;
            let mut t1 = 0.0;
            for &i in &gd.indices {
                let row = self.x.row(i);
                let eta = if self.include_intercept {
                    beta[0] + row_dot(row, &beta[1..])
                } else {
                    row_dot(row, beta)
                };
                let r = self.y[i] - eta;
                sum_r2 += r * r;
                t0 += r;
                if let (RandomEffects::InterceptSlope { feature_idx }, Some(_)) = (self.re, tau_u) {
                    let xk = row[feature_idx];
                    t1 += xk * r;
                }
            }

            let (log_det_m, quad_correction) = match (self.re, tau_u) {
                (RandomEffects::Intercept, _) => {
                    let inv_tau2 = 1.0 / (tau_alpha * tau_alpha);
                    let m11 = inv_tau2 + inv_a * gd.s00;
                    let log_det_m = m11.ln();
                    let u = (inv_a * t0) / m11;
                    (log_det_m, (inv_a * t0) * u)
                }
                (RandomEffects::InterceptSlope { .. }, Some(tu)) => {
                    let inv_tau_a2 = 1.0 / (tau_alpha * tau_alpha);
                    let inv_tau_u2 = 1.0 / (tu * tu);
                    let m00 = inv_tau_a2 + inv_a * gd.s00;
                    let m01 = inv_a * gd.s01;
                    let m11 = inv_tau_u2 + inv_a * gd.s11;
                    let det = m00 * m11 - m01 * m01;
                    if !det.is_finite() || det <= 0.0 {
                        return Err(Error::Validation(
                            "invalid mixed-model determinant".to_string(),
                        ));
                    }
                    let log_det_m = det.ln();
                    // u = M^{-1} (inv_a * t)
                    let b0 = inv_a * t0;
                    let b1 = inv_a * t1;
                    let u0 = (m11 * b0 - m01 * b1) / det;
                    let u1 = (-m01 * b0 + m00 * b1) / det;
                    let quad = b0 * u0 + b1 * u1;
                    (log_det_m, quad)
                }
                _ => {
                    return Err(Error::Validation(
                        "tau_u must be provided for InterceptSlope".to_string(),
                    ));
                }
            };

            let log_det_v = (m as f64) * log_a + log_c + log_det_m;
            let quad = inv_a * sum_r2 - quad_correction;

            nll += 0.5 * (log_det_v + quad);

            // REML: accumulate X_g^T V_g^{-1} X_g via Woodbury identity.
            if self.use_reml {
                let p = self.x.p;
                let mut xtx = vec![0.0; nb * nb];
                let mut xtz0 = vec![0.0; nb];
                let mut xtz1 = vec![0.0; nb];

                for &i in &gd.indices {
                    let row = self.x.row(i);
                    if self.include_intercept {
                        xtx[0] += 1.0;
                        xtz0[0] += 1.0;
                        for c in 0..p {
                            xtx[c + 1] += row[c];
                            xtx[(c + 1) * nb] += row[c];
                            xtz0[c + 1] += row[c];
                        }
                        for j in 0..p {
                            for c in 0..p {
                                xtx[(j + 1) * nb + (c + 1)] += row[j] * row[c];
                            }
                        }
                    } else {
                        for j in 0..p {
                            xtz0[j] += row[j];
                            for c in 0..p {
                                xtx[j * nb + c] += row[j] * row[c];
                            }
                        }
                    }
                    if let RandomEffects::InterceptSlope { feature_idx } = self.re {
                        let xk = row[feature_idx];
                        if self.include_intercept {
                            xtz1[0] += xk;
                            for c in 0..p {
                                xtz1[c + 1] += row[c] * xk;
                            }
                        } else {
                            for c in 0..p {
                                xtz1[c] += row[c] * xk;
                            }
                        }
                    }
                }

                for idx in 0..nb * nb {
                    xtvinvx[idx] += inv_a * xtx[idx];
                }

                match (self.re, tau_u) {
                    (RandomEffects::Intercept, _) => {
                        let inv_tau2 = 1.0 / (tau_alpha * tau_alpha);
                        let m_val = inv_tau2 + inv_a * gd.s00;
                        let coeff = inv_a * inv_a / m_val;
                        for j in 0..nb {
                            for k in 0..nb {
                                xtvinvx[j * nb + k] -= coeff * xtz0[j] * xtz0[k];
                            }
                        }
                    }
                    (RandomEffects::InterceptSlope { .. }, Some(tu)) => {
                        let inv_tau_a2 = 1.0 / (tau_alpha * tau_alpha);
                        let inv_tau_u2 = 1.0 / (tu * tu);
                        let a00 = inv_tau_a2 + inv_a * gd.s00;
                        let a01 = inv_a * gd.s01;
                        let a11 = inv_tau_u2 + inv_a * gd.s11;
                        let det = a00 * a11 - a01 * a01;
                        let coeff = inv_a * inv_a / det;
                        for j in 0..nb {
                            for k in 0..nb {
                                xtvinvx[j * nb + k] -= coeff
                                    * (a11 * xtz0[j] * xtz0[k]
                                        - a01 * (xtz0[j] * xtz1[k] + xtz1[j] * xtz0[k])
                                        + a00 * xtz1[j] * xtz1[k]);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        if self.use_reml {
            let logdet = cholesky_logdet(&xtvinvx, nb)?;
            nll += 0.5 * logdet;
        }

        Ok(nll)
    }
}

impl LogDensityModel for LmmMarginalModel {
    type Prepared<'a>
        = PreparedModelRef<'a, Self>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        self.n_beta()
            + 2
            + match self.re {
                RandomEffects::Intercept => 0,
                RandomEffects::InterceptSlope { .. } => 1,
            }
    }

    fn parameter_names(&self) -> Vec<String> {
        let mut out = Vec::with_capacity(self.dim());
        if self.include_intercept {
            out.push("intercept".to_string());
        }
        for j in 0..self.x.p {
            out.push(format!("beta{}", j + 1));
        }
        out.push("log_sigma_y".to_string());
        out.push("log_tau_alpha".to_string());
        if let RandomEffects::InterceptSlope { feature_idx } = self.re {
            out.push(format!("log_tau_u_beta{}", feature_idx + 1));
        }
        out
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![(f64::NEG_INFINITY, f64::INFINITY); self.dim()]
    }

    fn parameter_init(&self) -> Vec<f64> {
        // Heuristic initialization to avoid early-termination/local minima in
        // generic quasi-Newton solvers:
        // - intercept: mean(y)
        // - betas: per-feature univariate slope on centered data
        // - sigma_y: residual std under the heuristic beta
        // - tau_*: std of group-wise residual summaries
        let n = self.x.n;
        let p = self.x.p;
        let nb = self.n_beta();

        let y_mean = self.y.iter().sum::<f64>() / (n as f64);
        let mut x_mean = vec![0.0; p];
        for i in 0..n {
            let row = self.x.row(i);
            for j in 0..p {
                x_mean[j] += row[j];
            }
        }
        for j in 0..p {
            x_mean[j] /= n as f64;
        }

        let mut beta_init = vec![0.0; nb];
        if self.include_intercept {
            beta_init[0] = y_mean;
        }

        // Univariate slopes on centered x_j, centered y (rough, but stable).
        for j in 0..p {
            let mut num = 0.0;
            let mut den = 0.0;
            for i in 0..n {
                let xc = self.x.row(i)[j] - x_mean[j];
                let yc = self.y[i] - y_mean;
                num += xc * yc;
                den += xc * xc;
            }
            let b = if den > 0.0 { num / den } else { 0.0 };
            beta_init[self.beta_offset() + j] = b;
        }

        // Residuals under the heuristic beta.
        let mut resid = vec![0.0; n];
        let mut ss = 0.0;
        for i in 0..n {
            let row = self.x.row(i);
            let eta = if self.include_intercept {
                beta_init[0] + row_dot(row, &beta_init[1..])
            } else {
                row_dot(row, &beta_init)
            };
            let r = self.y[i] - eta;
            resid[i] = r;
            ss += r * r;
        }
        let sigma_y = (ss / (n as f64)).sqrt().max(1e-6);

        // Group-wise residual means approximate random intercepts.
        let mut alpha_hat = Vec::with_capacity(self.n_groups);
        for g in 0..self.n_groups {
            let idxs = &self.groups[g].indices;
            if idxs.is_empty() {
                continue;
            }
            let m = idxs.len() as f64;
            let mu = idxs.iter().map(|&i| resid[i]).sum::<f64>() / m;
            alpha_hat.push(mu);
        }
        let tau_alpha = if alpha_hat.len() >= 2 {
            let m = alpha_hat.len() as f64;
            let mean = alpha_hat.iter().sum::<f64>() / m;
            let var = alpha_hat.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / m;
            var.sqrt().max(1e-6)
        } else {
            1.0
        };

        // Group-wise slope residual summaries approximate random slopes.
        let tau_u = match self.re {
            RandomEffects::Intercept => None,
            RandomEffects::InterceptSlope { feature_idx } => {
                let mut u_hat = Vec::with_capacity(self.n_groups);
                for g in 0..self.n_groups {
                    let idxs = &self.groups[g].indices;
                    if idxs.len() < 2 {
                        continue;
                    }
                    let mut num = 0.0;
                    let mut den = 0.0;
                    for &i in idxs {
                        let xc = self.x.row(i)[feature_idx] - x_mean[feature_idx];
                        num += xc * resid[i];
                        den += xc * xc;
                    }
                    if den > 0.0 {
                        u_hat.push(num / den);
                    }
                }
                if u_hat.len() >= 2 {
                    let m = u_hat.len() as f64;
                    let mean = u_hat.iter().sum::<f64>() / m;
                    let var = u_hat.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / m;
                    Some(var.sqrt().max(1e-6))
                } else {
                    Some(1.0)
                }
            }
        };

        let mut init = vec![0.0; self.dim()];
        init[..nb].copy_from_slice(&beta_init);
        init[nb] = sigma_y.ln();
        init[nb + 1] = tau_alpha.ln();
        if let Some(tu) = tau_u {
            init[nb + 2] = tu.ln();
        }
        init
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        let (beta, log_sigma_y, log_tau_alpha, log_tau_u_val, log_tau_u_opt) =
            self.unpack(params)?;
        let sigma_y = log_sigma_y.exp();
        let tau_alpha = log_tau_alpha.exp();
        let tau_u = log_tau_u_opt.map(|_| log_tau_u_val.exp());
        self.nll_internal(beta, sigma_y, tau_alpha, tau_u)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        let (beta, log_sigma_y, log_tau_alpha, log_tau_u_val, log_tau_u_opt) =
            self.unpack(params)?;
        let sigma_y = log_sigma_y.exp();
        let tau_alpha = log_tau_alpha.exp();
        let tau_u = log_tau_u_opt.map(|_| log_tau_u_val.exp());

        let inv_a = 1.0 / (sigma_y * sigma_y);

        let nb = self.n_beta();
        let mut grad = vec![0.0; self.dim()];

        // Analytic gradient for beta (fixed effects): grad = -X^T V^{-1} r
        for g in 0..self.n_groups {
            let gd = &self.groups[g];
            let m = gd.indices.len();
            if m == 0 {
                continue;
            }

            // Residuals and Z^T r
            let mut t0 = 0.0;
            let mut t1 = 0.0;
            let mut r_cache: Vec<(usize, f64)> = Vec::with_capacity(m);
            for &i in &gd.indices {
                let row = self.x.row(i);
                let eta = if self.include_intercept {
                    beta[0] + row_dot(row, &beta[1..])
                } else {
                    row_dot(row, beta)
                };
                let r = self.y[i] - eta;
                t0 += r;
                if let (RandomEffects::InterceptSlope { feature_idx }, Some(_)) = (self.re, tau_u) {
                    let xk = row[feature_idx];
                    t1 += xk * r;
                }
                r_cache.push((i, r));
            }

            let (u0, u1) = match (self.re, tau_u) {
                (RandomEffects::Intercept, _) => {
                    let inv_tau2 = 1.0 / (tau_alpha * tau_alpha);
                    let m11 = inv_tau2 + inv_a * gd.s00;
                    let u = (inv_a * t0) / m11;
                    (u, 0.0)
                }
                (RandomEffects::InterceptSlope { .. }, Some(tu)) => {
                    let inv_tau_a2 = 1.0 / (tau_alpha * tau_alpha);
                    let inv_tau_u2 = 1.0 / (tu * tu);
                    let m00 = inv_tau_a2 + inv_a * gd.s00;
                    let m01 = inv_a * gd.s01;
                    let m11 = inv_tau_u2 + inv_a * gd.s11;
                    let det = m00 * m11 - m01 * m01;
                    if !det.is_finite() || det <= 0.0 {
                        return Err(Error::Validation(
                            "invalid mixed-model determinant".to_string(),
                        ));
                    }
                    let b0 = inv_a * t0;
                    let b1 = inv_a * t1;
                    let u0 = (m11 * b0 - m01 * b1) / det;
                    let u1 = (-m01 * b0 + m00 * b1) / det;
                    (u0, u1)
                }
                _ => {
                    return Err(Error::Validation(
                        "tau_u must be provided for InterceptSlope".to_string(),
                    ));
                }
            };

            for (i, r) in r_cache {
                let row = self.x.row(i);
                let z_dot_u = match self.re {
                    RandomEffects::Intercept => u0,
                    RandomEffects::InterceptSlope { feature_idx } => u0 + row[feature_idx] * u1,
                };
                let v = inv_a * r - inv_a * z_dot_u;
                if self.include_intercept {
                    grad[0] += -v;
                    for j in 0..self.x.p {
                        grad[1 + j] += -v * row[j];
                    }
                } else {
                    for j in 0..self.x.p {
                        grad[j] += -v * row[j];
                    }
                }
            }
        }

        // Finite-diff gradients for log-scale variance parameters (few dims).
        let eps = 1e-5;
        for k in nb..self.dim() {
            let mut p_hi = params.to_vec();
            let mut p_lo = params.to_vec();
            p_hi[k] += eps;
            p_lo[k] -= eps;
            let f_hi = self.nll(&p_hi)?;
            let f_lo = self.nll(&p_lo)?;
            grad[k] = (f_hi - f_lo) / (2.0 * eps);
        }

        Ok(grad)
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

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

    fn dense_group_nll(
        r: &[f64],
        sigma_y: f64,
        tau_alpha: f64,
        tau_u: Option<(f64, &[f64])>,
    ) -> f64 {
        let m = r.len();
        let sigma2 = sigma_y * sigma_y;
        let tau_a2 = tau_alpha * tau_alpha;

        let mut v = DMatrix::zeros(m, m);
        for i in 0..m {
            for j in 0..m {
                let mut val = tau_a2;
                if let Some((tu, z)) = tau_u {
                    val += (tu * tu) * z[i] * z[j];
                }
                if i == j {
                    val += sigma2;
                }
                v[(i, j)] = val;
            }
        }

        let chol = nalgebra::linalg::Cholesky::new(v).unwrap();
        let l = chol.l();
        let mut log_det = 0.0;
        for i in 0..m {
            log_det += 2.0 * l[(i, i)].ln();
        }

        let rv = DVector::from_row_slice(r);
        let x = chol.solve(&rv);
        let quad = rv.dot(&x);

        0.5 * (log_det + quad)
    }

    #[test]
    fn lmm_grad_matches_finite_diff_smoke_intercept() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![1.5], vec![2.5]];
        let y = vec![1.0, 2.1, 2.9, 4.2, 1.4, 2.7];
        let group_idx = vec![0usize, 0, 0, 1, 1, 1];
        let m = LmmMarginalModel::new(x, y, false, group_idx, 2, RandomEffects::Intercept).unwrap();

        let p = vec![0.1, 0.0, 0.0]; // beta1, log_sigma_y, log_tau_alpha
        let g = m.grad_nll(&p).unwrap();
        let g_fd = finite_diff_grad(&m, &p, 1e-6);
        for i in 0..p.len() {
            assert!((g[i] - g_fd[i]).abs() < 5e-4);
        }
    }

    #[test]
    fn lmm_nll_matches_dense_covariance_intercept() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![1.5], vec![2.5]];
        let y = vec![1.0, 2.1, 2.9, 4.2, 1.4, 2.7];
        let group_idx = vec![0usize, 0, 0, 1, 1, 1];
        let m = LmmMarginalModel::new(x, y, false, group_idx, 2, RandomEffects::Intercept).unwrap();

        // beta1, log_sigma_y, log_tau_alpha
        let params = vec![0.2, (0.5f64).ln(), (1.2f64).ln()];
        let nll = m.nll(&params).unwrap();

        let beta = params[0];
        let sigma_y = params[1].exp();
        let tau_alpha = params[2].exp();

        let mut nll_dense = 0.0;
        for gd in &m.groups {
            if gd.indices.is_empty() {
                continue;
            }
            let mut r = Vec::with_capacity(gd.indices.len());
            for &i in &gd.indices {
                let eta = row_dot(m.x.row(i), &[beta]);
                r.push(m.y[i] - eta);
            }
            nll_dense += dense_group_nll(&r, sigma_y, tau_alpha, None);
        }

        assert!((nll - nll_dense).abs() < 1e-10);
    }

    #[test]
    fn lmm_grad_matches_finite_diff_smoke_intercept_slope() {
        let x = vec![
            vec![1.0, 0.1],
            vec![1.0, 0.2],
            vec![1.0, 0.3],
            vec![1.0, 0.1],
            vec![1.0, 0.2],
            vec![1.0, 0.3],
        ];
        let y = vec![1.0, 1.1, 0.9, 1.4, 1.6, 1.3];
        let group_idx = vec![0usize, 0, 0, 1, 1, 1];
        let m = LmmMarginalModel::new(
            x,
            y,
            true,
            group_idx,
            2,
            RandomEffects::InterceptSlope { feature_idx: 1 },
        )
        .unwrap();

        // intercept, beta1, beta2, log_sigma_y, log_tau_alpha, log_tau_u
        let p = vec![0.2, 0.1, -0.3, 0.0, 0.0, 0.0];
        let g = m.grad_nll(&p).unwrap();
        let g_fd = finite_diff_grad(&m, &p, 1e-6);
        for i in 0..p.len() {
            assert!((g[i] - g_fd[i]).abs() < 5e-4);
        }
    }

    #[test]
    fn lmm_nll_matches_dense_covariance_intercept_slope() {
        let x = vec![
            vec![1.0, 0.1],
            vec![1.0, 0.2],
            vec![1.0, 0.3],
            vec![1.0, 0.1],
            vec![1.0, 0.2],
            vec![1.0, 0.3],
        ];
        let y = vec![1.0, 1.1, 0.9, 1.4, 1.6, 1.3];
        let group_idx = vec![0usize, 0, 0, 1, 1, 1];
        let m = LmmMarginalModel::new(
            x,
            y,
            true,
            group_idx,
            2,
            RandomEffects::InterceptSlope { feature_idx: 1 },
        )
        .unwrap();

        // intercept, beta1, beta2, log_sigma_y, log_tau_alpha, log_tau_u
        let params = vec![0.2, 0.1, -0.3, (0.7f64).ln(), (1.1f64).ln(), (0.6f64).ln()];
        let nll = m.nll(&params).unwrap();

        let sigma_y = params[3].exp();
        let tau_alpha = params[4].exp();
        let tau_u = params[5].exp();

        let mut nll_dense = 0.0;
        for gd in &m.groups {
            if gd.indices.is_empty() {
                continue;
            }
            let mut r = Vec::with_capacity(gd.indices.len());
            let mut z = Vec::with_capacity(gd.indices.len());
            for &i in &gd.indices {
                let row = m.x.row(i);
                let eta = params[0] + row_dot(row, &params[1..3]);
                r.push(m.y[i] - eta);
                z.push(row[1]);
            }
            nll_dense += dense_group_nll(&r, sigma_y, tau_alpha, Some((tau_u, &z)));
        }

        assert!((nll - nll_dense).abs() < 1e-10);
    }

    // ---- REML tests ----

    /// Dense (nalgebra) REML NLL reference: ML NLL + 0.5*log|X^T V^{-1} X|.
    fn dense_reml_correction(
        model: &LmmMarginalModel,
        _params: &[f64],
        sigma_y: f64,
        tau_alpha: f64,
        tau_u_val: Option<f64>,
    ) -> f64 {
        let n = model.x.n;
        let nb = model.n_beta();

        // Build full X (n × nb) with intercept column if needed
        let mut x_full = DMatrix::zeros(n, nb);
        for i in 0..n {
            let row = model.x.row(i);
            if model.include_intercept {
                x_full[(i, 0)] = 1.0;
                for j in 0..model.x.p {
                    x_full[(i, j + 1)] = row[j];
                }
            } else {
                for j in 0..model.x.p {
                    x_full[(i, j)] = row[j];
                }
            }
        }

        // Build block-diagonal V (n × n)
        let sigma2 = sigma_y * sigma_y;
        let tau_a2 = tau_alpha * tau_alpha;
        let mut v_mat = DMatrix::zeros(n, n);
        for gd in &model.groups {
            for &i in &gd.indices {
                for &j in &gd.indices {
                    let mut val = tau_a2;
                    if let (RandomEffects::InterceptSlope { feature_idx }, Some(tu)) =
                        (model.re, tau_u_val)
                    {
                        val +=
                            (tu * tu) * model.x.row(i)[feature_idx] * model.x.row(j)[feature_idx];
                    }
                    if i == j {
                        val += sigma2;
                    }
                    v_mat[(i, j)] = val;
                }
            }
        }

        // V^{-1}
        let chol_v = nalgebra::linalg::Cholesky::new(v_mat.clone()).unwrap();
        // X^T V^{-1} X
        let vinv_x = chol_v.solve(&x_full);
        let xtvinvx = x_full.transpose() * vinv_x;
        let chol_xtvinvx = nalgebra::linalg::Cholesky::new(xtvinvx).unwrap();
        let l = chol_xtvinvx.l();
        let mut logdet = 0.0;
        for i in 0..nb {
            logdet += 2.0 * l[(i, i)].ln();
        }
        0.5 * logdet
    }

    #[test]
    fn lmm_reml_nll_matches_dense_intercept() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![1.5], vec![2.5]];
        let y = vec![1.0, 2.1, 2.9, 4.2, 1.4, 2.7];
        let group_idx = vec![0usize, 0, 0, 1, 1, 1];

        let m_ml = LmmMarginalModel::new(
            x.clone(),
            y.clone(),
            false,
            group_idx.clone(),
            2,
            RandomEffects::Intercept,
        )
        .unwrap();
        let m_reml = LmmMarginalModel::new(x, y, false, group_idx, 2, RandomEffects::Intercept)
            .unwrap()
            .with_reml(true);

        let params = vec![0.2, (0.5f64).ln(), (1.2f64).ln()];
        let nll_ml = m_ml.nll(&params).unwrap();
        let nll_reml = m_reml.nll(&params).unwrap();

        let sigma_y = params[1].exp();
        let tau_alpha = params[2].exp();
        let correction = dense_reml_correction(&m_ml, &params, sigma_y, tau_alpha, None);

        // REML NLL = ML NLL + correction
        assert!(
            (nll_reml - (nll_ml + correction)).abs() < 1e-10,
            "REML mismatch: reml={}, ml+corr={}",
            nll_reml,
            nll_ml + correction
        );
        // REML NLL should be strictly larger than ML NLL
        assert!(nll_reml > nll_ml);
    }

    #[test]
    fn lmm_reml_nll_matches_dense_intercept_slope() {
        // Use non-collinear covariates (no column of ones — intercept is added automatically).
        let x = vec![
            vec![0.5, 0.1],
            vec![1.2, 0.2],
            vec![0.8, 0.3],
            vec![1.5, 0.1],
            vec![0.3, 0.2],
            vec![0.9, 0.3],
        ];
        let y = vec![1.0, 1.1, 0.9, 1.4, 1.6, 1.3];
        let group_idx = vec![0usize, 0, 0, 1, 1, 1];

        let m_ml = LmmMarginalModel::new(
            x.clone(),
            y.clone(),
            true,
            group_idx.clone(),
            2,
            RandomEffects::InterceptSlope { feature_idx: 1 },
        )
        .unwrap();
        let m_reml = LmmMarginalModel::new(
            x,
            y,
            true,
            group_idx,
            2,
            RandomEffects::InterceptSlope { feature_idx: 1 },
        )
        .unwrap()
        .with_reml(true);

        let params = vec![0.2, 0.1, -0.3, (0.7f64).ln(), (1.1f64).ln(), (0.6f64).ln()];
        let nll_ml = m_ml.nll(&params).unwrap();
        let nll_reml = m_reml.nll(&params).unwrap();

        let sigma_y = params[3].exp();
        let tau_alpha = params[4].exp();
        let tau_u = params[5].exp();
        let correction = dense_reml_correction(&m_ml, &params, sigma_y, tau_alpha, Some(tau_u));

        assert!(
            (nll_reml - (nll_ml + correction)).abs() < 1e-10,
            "REML mismatch: reml={}, ml+corr={}",
            nll_reml,
            nll_ml + correction
        );
        // REML NLL differs from ML NLL (correction can be positive or negative)
        assert!((nll_reml - nll_ml).abs() > 1e-15);
    }

    #[test]
    fn lmm_reml_grad_matches_finite_diff() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![1.5], vec![2.5]];
        let y = vec![1.0, 2.1, 2.9, 4.2, 1.4, 2.7];
        let group_idx = vec![0usize, 0, 0, 1, 1, 1];
        let m = LmmMarginalModel::new(x, y, false, group_idx, 2, RandomEffects::Intercept)
            .unwrap()
            .with_reml(true);

        let p = vec![0.1, 0.0, 0.0];
        let g = m.grad_nll(&p).unwrap();
        let g_fd = finite_diff_grad(&m, &p, 1e-6);
        for i in 0..p.len() {
            assert!(
                (g[i] - g_fd[i]).abs() < 5e-4,
                "REML grad[{}]: analytic={}, fd={}",
                i,
                g[i],
                g_fd[i]
            );
        }
    }

    #[test]
    fn lmm_reml_with_reml_false_equals_ml() {
        let x = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![1.5], vec![2.5]];
        let y = vec![1.0, 2.1, 2.9, 4.2, 1.4, 2.7];
        let group_idx = vec![0usize, 0, 0, 1, 1, 1];

        let m_ml = LmmMarginalModel::new(
            x.clone(),
            y.clone(),
            false,
            group_idx.clone(),
            2,
            RandomEffects::Intercept,
        )
        .unwrap();
        let m_reml_off = LmmMarginalModel::new(x, y, false, group_idx, 2, RandomEffects::Intercept)
            .unwrap()
            .with_reml(false);

        let params = vec![0.2, (0.5f64).ln(), (1.2f64).ln()];
        let nll_ml = m_ml.nll(&params).unwrap();
        let nll_off = m_reml_off.nll(&params).unwrap();

        assert!((nll_ml - nll_off).abs() < 1e-15, "with_reml(false) should equal ML");
        assert!(!m_ml.is_reml());
        assert!(!m_reml_off.is_reml());
    }
}
