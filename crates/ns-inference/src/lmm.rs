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
        if let RandomEffects::InterceptSlope { feature_idx } = re {
            if feature_idx >= x.p {
                return Err(Error::Validation(format!(
                    "feature_idx must be in [0, p), got {} (p={})",
                    feature_idx, x.p
                )));
            }
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

        Ok(Self {
            x,
            y,
            include_intercept,
            n_groups,
            re,
            groups,
        })
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

    fn nll_internal(&self, beta: &[f64], sigma_y: f64, tau_alpha: f64, tau_u: Option<f64>) -> Result<f64> {
        if !sigma_y.is_finite() || sigma_y <= 0.0 {
            return Err(Error::Validation("sigma_y must be finite and > 0".to_string()));
        }
        if !tau_alpha.is_finite() || tau_alpha <= 0.0 {
            return Err(Error::Validation("tau_alpha must be finite and > 0".to_string()));
        }
        if let Some(tu) = tau_u {
            if !tu.is_finite() || tu <= 0.0 {
                return Err(Error::Validation("tau_u must be finite and > 0".to_string()));
            }
        }

        let inv_a = 1.0 / (sigma_y * sigma_y);
        let log_a = (sigma_y * sigma_y).ln();
        let log_c = (tau_alpha * tau_alpha).ln()
            + if let Some(tu) = tau_u { (tu * tu).ln() } else { 0.0 };

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
                        return Err(Error::Validation("invalid mixed-model determinant".to_string()));
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
        }

        Ok(nll)
    }
}

impl LogDensityModel for LmmMarginalModel {
    type Prepared<'a> = PreparedModelRef<'a, Self> where Self: 'a;

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
        vec![0.0; self.dim()]
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        let (beta, log_sigma_y, log_tau_alpha, log_tau_u_val, log_tau_u_opt) = self.unpack(params)?;
        let sigma_y = log_sigma_y.exp();
        let tau_alpha = log_tau_alpha.exp();
        let tau_u = log_tau_u_opt.map(|_| log_tau_u_val.exp());
        self.nll_internal(beta, sigma_y, tau_alpha, tau_u)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        let (beta, log_sigma_y, log_tau_alpha, log_tau_u_val, log_tau_u_opt) = self.unpack(params)?;
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
                        return Err(Error::Validation("invalid mixed-model determinant".to_string()));
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
        let m = LmmMarginalModel::new(
            x,
            y,
            false,
            group_idx,
            2,
            RandomEffects::Intercept,
        )
        .unwrap();

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
        let m = LmmMarginalModel::new(
            x,
            y,
            false,
            group_idx,
            2,
            RandomEffects::Intercept,
        )
        .unwrap();

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
}
