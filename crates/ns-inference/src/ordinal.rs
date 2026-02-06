//! Ordinal regression models (Phase 9.C).
//!
//! Currently provides ordered logistic and ordered probit regression (proportional odds / latent
//! Gaussian threshold) models.

use ns_core::traits::{LogDensityModel, PreparedModelRef};
use ns_core::{Error, Result};
use ns_prob::math::{log_sigmoid, sigmoid, softplus};

use crate::regression::{DenseX, row_dot, validate_xy_dims};

const MIN_TAIL: f64 = 1e-300;

fn log1mexp(x: f64) -> f64 {
    // Stable log(1 - exp(x)) for x <= 0.
    //
    // Reference: https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    if !(x <= 0.0) {
        return f64::NEG_INFINITY;
    }
    // ln(0.5)
    const LN_HALF: f64 = -0.693_147_180_559_945_3;
    if x < LN_HALF {
        (1.0 - x.exp()).ln()
    } else {
        (-x.exp_m1()).ln()
    }
}

#[inline]
fn normal_phi(z: f64) -> f64 {
    // phi(z) = exp(-0.5*z^2)/sqrt(2*pi)
    const INV_SQRT_2PI: f64 = 0.3989422804014327;
    (-0.5 * z * z).exp() * INV_SQRT_2PI
}

#[inline]
fn normal_cdf(x: f64) -> f64 {
    // Use erfc for better tail behavior:
    // Phi(x) = 0.5 * erfc(-x / sqrt(2))
    0.5 * statrs::function::erf::erfc(-x / std::f64::consts::SQRT_2)
}

/// Ordered logistic regression (a.k.a. proportional odds model).
///
/// Observed outcome levels: `y_i ∈ {0, 1, ..., K-1}` with `K >= 2`.
///
/// Linear predictor:
/// `eta_i = X_i * beta`
///
/// Cutpoints `c_1 < ... < c_{K-1}` parameterize cumulative probabilities:
/// `P(y <= k | eta) = sigmoid(c_{k+1} - eta)` for `k=0..K-2`.
///
/// The model enforces ordering via an unconstrained raw parameterization:
/// - `c_1 = raw_1`
/// - `c_j = c_{j-1} + softplus(raw_j)` for `j=2..K-1`
///
/// Parameter vector layout:
/// - `beta[0..p)` (p slopes)
/// - `cut_raw[0..K-1)` (K-1 raw cutpoint params)
#[derive(Debug, Clone)]
pub struct OrderedLogitModel {
    x: DenseX,
    y: Vec<u8>,
    n_levels: usize,
}

impl OrderedLogitModel {
    /// Create an ordered logistic regression model.
    pub fn new(x: Vec<Vec<f64>>, y: Vec<u8>, n_levels: usize) -> Result<Self> {
        if n_levels < 2 {
            return Err(Error::Validation("n_levels must be >= 2".to_string()));
        }
        let x = DenseX::from_rows(x)?;
        validate_xy_dims(x.n, x.p, x.data.len(), y.len())?;
        if y.iter().any(|&k| (k as usize) >= n_levels) {
            return Err(Error::Validation(format!(
                "y contains a level >= n_levels (n_levels={})",
                n_levels
            )));
        }
        Ok(Self { x, y, n_levels })
    }

    #[inline]
    fn n_cuts(&self) -> usize {
        self.n_levels - 1
    }

    #[inline]
    fn dim_internal(&self) -> usize {
        self.x.p + self.n_cuts()
    }

    fn split_params<'a>(&self, params: &'a [f64]) -> Result<(&'a [f64], &'a [f64])> {
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
        Ok(params.split_at(self.x.p))
    }

    fn cutpoints_from_raw(&self, cut_raw: &[f64]) -> Vec<f64> {
        debug_assert_eq!(cut_raw.len(), self.n_cuts());
        let mut c = vec![0.0; cut_raw.len()];
        if c.is_empty() {
            return c;
        }
        c[0] = cut_raw[0];
        for j in 1..c.len() {
            // Positive increment enforces strict ordering.
            c[j] = c[j - 1] + softplus(cut_raw[j]);
        }
        c
    }

    #[inline]
    fn eta(&self, i: usize, beta: &[f64]) -> f64 {
        row_dot(self.x.row(i), beta)
    }
}

impl LogDensityModel for OrderedLogitModel {
    type Prepared<'a>
        = PreparedModelRef<'a, Self>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        self.dim_internal()
    }

    fn parameter_names(&self) -> Vec<String> {
        let mut out = Vec::with_capacity(self.dim_internal());
        for j in 0..self.x.p {
            out.push(format!("beta{}", j + 1));
        }
        for k in 0..self.n_cuts() {
            out.push(format!("cut_raw{}", k + 1));
        }
        out
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![(f64::NEG_INFINITY, f64::INFINITY); self.dim_internal()]
    }

    fn parameter_init(&self) -> Vec<f64> {
        // Betas start at 0; cutpoints spaced roughly by softplus(0)=~0.693.
        let mut out = vec![0.0; self.dim_internal()];
        let cuts_off = self.x.p;
        if self.n_cuts() > 0 {
            out[cuts_off] = -1.0;
            for j in 1..self.n_cuts() {
                out[cuts_off + j] = 0.0;
            }
        }
        out
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        let (beta, cut_raw) = self.split_params(params)?;
        let cuts = self.cutpoints_from_raw(cut_raw);

        let mut nll = 0.0;
        for i in 0..self.x.n {
            let eta = self.eta(i, beta);
            let yi = self.y[i] as usize;

            let logp = if yi == 0 {
                let u = cuts[0] - eta;
                log_sigmoid(u)
            } else if yi + 1 == self.n_levels {
                let u = eta - cuts[self.n_cuts() - 1];
                log_sigmoid(u)
            } else {
                // log(sigmoid(a) - sigmoid(b)), with a > b.
                let a = cuts[yi] - eta; // c_{yi+1} - eta
                let b = cuts[yi - 1] - eta; // c_{yi} - eta
                let la = log_sigmoid(a);
                let lb = log_sigmoid(b);
                la + log1mexp(lb - la)
            };

            if !logp.is_finite() {
                return Err(Error::Computation("ordered_logit produced non-finite logp".to_string()));
            }
            nll += -logp;
        }

        Ok(nll)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        let (beta, cut_raw) = self.split_params(params)?;
        let cuts = self.cutpoints_from_raw(cut_raw);

        let mut grad_beta = vec![0.0f64; self.x.p];
        let mut grad_c = vec![0.0f64; self.n_cuts()]; // grad wrt constrained cutpoints c_j

        for i in 0..self.x.n {
            let eta = self.eta(i, beta);
            let yi = self.y[i] as usize;

            // g_eta = d/deta NLL, and g_c[j] = d/dc_j NLL (constrained space).
            let (g_eta, g_ci, g_cip1) = if yi == 0 {
                let u = cuts[0] - eta;
                let s = sigmoid(u);
                // logp = log_sigmoid(u), u = c1 - eta
                let dlogp_deta = s - 1.0;
                let dlogp_dc1 = 1.0 - s;
                (-dlogp_deta, -dlogp_dc1, 0.0)
            } else if yi + 1 == self.n_levels {
                let u = eta - cuts[self.n_cuts() - 1];
                let s = sigmoid(u);
                // logp = log_sigmoid(u), u = eta - c_last
                let dlogp_deta = 1.0 - s;
                let dlogp_dclast = s - 1.0;
                (-dlogp_deta, 0.0, -dlogp_dclast)
            } else {
                // Interior: logp = log(sigmoid(a) - sigmoid(b))
                // a = c_{yi+1} - eta  -> cuts[yi]
                // b = c_{yi} - eta    -> cuts[yi-1]
                let a = cuts[yi] - eta;
                let b = cuts[yi - 1] - eta;
                let sa = sigmoid(a);
                let sb = sigmoid(b);
                let spa = sa * (1.0 - sa);
                let spb = sb * (1.0 - sb);
                let p = (sa - sb).max(MIN_TAIL);
                let dlogp_deta = (spb - spa) / p;
                let dlogp_dck = -spb / p; // c_{yi}
                let dlogp_dck1 = spa / p; // c_{yi+1}
                (-dlogp_deta, -dlogp_dck, -dlogp_dck1)
            };

            // Beta gradient: eta = x*beta
            let row = self.x.row(i);
            for j in 0..self.x.p {
                grad_beta[j] += g_eta * row[j];
            }

            // Cutpoint gradients (constrained space).
            if yi == 0 {
                grad_c[0] += g_ci;
            } else if yi + 1 == self.n_levels {
                grad_c[self.n_cuts() - 1] += g_cip1;
            } else {
                grad_c[yi - 1] += g_ci;
                grad_c[yi] += g_cip1;
            }
        }

        // Map constrained cutpoint gradients -> raw parameter gradients.
        let mut grad_cut_raw = vec![0.0f64; self.n_cuts()];
        if !grad_cut_raw.is_empty() {
            // raw[0] contributes to all cutpoints equally (c_j includes c1).
            grad_cut_raw[0] = grad_c.iter().sum::<f64>();

            // For j>=1, raw[j] contributes to c_j..c_last with factor sigmoid(raw[j]).
            // Use suffix sums to avoid O(K^2).
            let mut suffix = 0.0f64;
            for j in (1..self.n_cuts()).rev() {
                suffix += grad_c[j];
                grad_cut_raw[j] = sigmoid(cut_raw[j]) * suffix;
            }
        }

        let mut grad = Vec::with_capacity(self.dim_internal());
        grad.extend_from_slice(&grad_beta);
        grad.extend_from_slice(&grad_cut_raw);
        Ok(grad)
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

/// Ordered probit regression (latent Gaussian threshold model).
///
/// Observed outcome levels: `y_i ∈ {0, 1, ..., K-1}` with `K >= 2`.
///
/// Linear predictor:
/// `eta_i = X_i * beta`
///
/// Cutpoints `c_1 < ... < c_{K-1}` parameterize cumulative probabilities:
/// `P(y <= k | eta) = Phi(c_{k+1} - eta)` for `k=0..K-2`, where `Phi` is the
/// standard normal CDF.
///
/// Parameter vector layout:
/// - `beta[0..p)` (p slopes)
/// - `cut_raw[0..K-1)` (K-1 raw cutpoint params, mapped to ordered cutpoints via softplus)
#[derive(Debug, Clone)]
pub struct OrderedProbitModel {
    x: DenseX,
    y: Vec<u8>,
    n_levels: usize,
}

impl OrderedProbitModel {
    /// Create an ordered probit regression model.
    pub fn new(x: Vec<Vec<f64>>, y: Vec<u8>, n_levels: usize) -> Result<Self> {
        if n_levels < 2 {
            return Err(Error::Validation("n_levels must be >= 2".to_string()));
        }
        let x = DenseX::from_rows(x)?;
        validate_xy_dims(x.n, x.p, x.data.len(), y.len())?;
        if y.iter().any(|&k| (k as usize) >= n_levels) {
            return Err(Error::Validation(format!(
                "y contains a level >= n_levels (n_levels={})",
                n_levels
            )));
        }
        Ok(Self { x, y, n_levels })
    }

    #[inline]
    fn n_cuts(&self) -> usize {
        self.n_levels - 1
    }

    #[inline]
    fn dim_internal(&self) -> usize {
        self.x.p + self.n_cuts()
    }

    fn split_params<'a>(&self, params: &'a [f64]) -> Result<(&'a [f64], &'a [f64])> {
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
        Ok(params.split_at(self.x.p))
    }

    fn cutpoints_from_raw(&self, cut_raw: &[f64]) -> Vec<f64> {
        debug_assert_eq!(cut_raw.len(), self.n_cuts());
        let mut c = vec![0.0; cut_raw.len()];
        if c.is_empty() {
            return c;
        }
        c[0] = cut_raw[0];
        for j in 1..c.len() {
            c[j] = c[j - 1] + softplus(cut_raw[j]);
        }
        c
    }

    #[inline]
    fn eta(&self, i: usize, beta: &[f64]) -> f64 {
        row_dot(self.x.row(i), beta)
    }
}

impl LogDensityModel for OrderedProbitModel {
    type Prepared<'a>
        = PreparedModelRef<'a, Self>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        self.dim_internal()
    }

    fn parameter_names(&self) -> Vec<String> {
        let mut out = Vec::with_capacity(self.dim_internal());
        for j in 0..self.x.p {
            out.push(format!("beta{}", j + 1));
        }
        for k in 0..self.n_cuts() {
            out.push(format!("cut_raw{}", k + 1));
        }
        out
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![(f64::NEG_INFINITY, f64::INFINITY); self.dim_internal()]
    }

    fn parameter_init(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.dim_internal()];
        let cuts_off = self.x.p;
        if self.n_cuts() > 0 {
            out[cuts_off] = -1.0;
            for j in 1..self.n_cuts() {
                out[cuts_off + j] = 0.0;
            }
        }
        out
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        let (beta, cut_raw) = self.split_params(params)?;
        let cuts = self.cutpoints_from_raw(cut_raw);

        let mut nll = 0.0;
        for i in 0..self.x.n {
            let eta = self.eta(i, beta);
            let yi = self.y[i] as usize;

            let p = if yi == 0 {
                normal_cdf(cuts[0] - eta)
            } else if yi + 1 == self.n_levels {
                // 1 - Phi(c_last - eta) = Phi(eta - c_last)
                normal_cdf(eta - cuts[self.n_cuts() - 1])
            } else {
                let u_hi = cuts[yi] - eta;
                let u_lo = cuts[yi - 1] - eta;
                // Avoid cancellation in the upper tail when both CDFs are near 1.
                if u_lo > 0.0 {
                    (normal_cdf(-u_lo) - normal_cdf(-u_hi)).max(0.0)
                } else {
                    (normal_cdf(u_hi) - normal_cdf(u_lo)).max(0.0)
                }
            };

            nll -= p.max(MIN_TAIL).ln();
        }

        Ok(nll)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        let (beta, cut_raw) = self.split_params(params)?;
        let cuts = self.cutpoints_from_raw(cut_raw);

        let mut grad_beta = vec![0.0f64; self.x.p];
        let mut grad_c = vec![0.0f64; self.n_cuts()]; // constrained cutpoints c_j

        for i in 0..self.x.n {
            let eta = self.eta(i, beta);
            let yi = self.y[i] as usize;

            // g_eta = d/deta NLL, and g_c[j] = d/dc_j NLL (constrained space).
            let g_eta = if yi == 0 {
                let u = cuts[0] - eta;
                let f = normal_phi(u);
                let p0 = normal_cdf(u).max(MIN_TAIL);
                grad_c[0] += -f / p0;
                f / p0
            } else if yi + 1 == self.n_levels {
                let u = cuts[self.n_cuts() - 1] - eta;
                let f = normal_phi(u);
                let ps = normal_cdf(-u).max(MIN_TAIL);
                grad_c[self.n_cuts() - 1] += f / ps;
                -f / ps
            } else {
                let u_hi = cuts[yi] - eta;
                let u_lo = cuts[yi - 1] - eta;
                let f_hi = normal_phi(u_hi);
                let f_lo = normal_phi(u_lo);
                let p = if u_lo > 0.0 {
                    (normal_cdf(-u_lo) - normal_cdf(-u_hi)).max(MIN_TAIL)
                } else {
                    (normal_cdf(u_hi) - normal_cdf(u_lo)).max(MIN_TAIL)
                };

                grad_c[yi] += -f_hi / p;
                grad_c[yi - 1] += f_lo / p;
                (f_hi - f_lo) / p
            };

            let row = self.x.row(i);
            for j in 0..self.x.p {
                grad_beta[j] += g_eta * row[j];
            }
        }

        // Map constrained cutpoint gradients -> raw parameter gradients.
        let mut grad_cut_raw = vec![0.0f64; self.n_cuts()];
        if !grad_cut_raw.is_empty() {
            grad_cut_raw[0] = grad_c.iter().sum::<f64>();

            let mut suffix = 0.0f64;
            for j in (1..self.n_cuts()).rev() {
                suffix += grad_c[j];
                grad_cut_raw[j] = sigmoid(cut_raw[j]) * suffix;
            }
        }

        let mut grad = Vec::with_capacity(self.dim_internal());
        grad.extend_from_slice(&grad_beta);
        grad.extend_from_slice(&grad_cut_raw);
        Ok(grad)
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

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

    #[test]
    fn test_ordered_logit_shapes_and_finiteness_smoke() {
        let x = vec![vec![-2.0], vec![-1.0], vec![0.0], vec![1.0], vec![2.0]];
        let y = vec![0u8, 0u8, 1u8, 2u8, 2u8];
        let m = OrderedLogitModel::new(x, y, 3).unwrap();
        assert_eq!(m.dim(), 1 + 2);
        let p0 = m.parameter_init();
        let nll = m.nll(&p0).unwrap();
        let g = m.grad_nll(&p0).unwrap();
        assert!(nll.is_finite());
        assert_eq!(g.len(), m.dim());
        assert!(g.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ordered_logit_grad_matches_finite_diff_smoke() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let y = vec![0u8, 0u8, 1u8, 2u8, 2u8];
        let m = OrderedLogitModel::new(x, y, 3).unwrap();
        let p = vec![0.3, -0.2, 0.1];
        let g = m.grad_nll(&p).unwrap();
        let g_fd = finite_diff_grad(&m, &p, 1e-6);
        for i in 0..p.len() {
            assert!((g[i] - g_fd[i]).abs() < 5e-5);
        }
    }

    #[test]
    fn test_ordered_probit_shapes_and_finiteness_smoke() {
        let x = vec![vec![-2.0], vec![-1.0], vec![0.0], vec![1.0], vec![2.0]];
        let y = vec![0u8, 0u8, 1u8, 2u8, 2u8];
        let m = OrderedProbitModel::new(x, y, 3).unwrap();
        assert_eq!(m.dim(), 1 + 2);
        let p0 = m.parameter_init();
        let nll = m.nll(&p0).unwrap();
        let g = m.grad_nll(&p0).unwrap();
        assert!(nll.is_finite());
        assert_eq!(g.len(), m.dim());
        assert!(g.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ordered_probit_grad_matches_finite_diff_smoke() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let y = vec![0u8, 0u8, 1u8, 2u8, 2u8];
        let m = OrderedProbitModel::new(x, y, 3).unwrap();
        let p = vec![0.25, -0.1, 0.2];
        let g = m.grad_nll(&p).unwrap();
        let g_fd = finite_diff_grad(&m, &p, 1e-6);
        for i in 0..p.len() {
            assert!((g[i] - g_fd[i]).abs() < 5e-5);
        }
    }
}
