//! EM parameter estimation for linear-Gaussian state space models.
//!
//! Phase 8 goal: provide a simple, dependency-light baseline for fitting
//! process/observation noise covariances (Q/R) via maximum likelihood.

use nalgebra::{DMatrix, DVector};
use ns_core::{Error, Result};

use super::kalman::{KalmanFilterResult, KalmanModel, kalman_filter, rts_smoother, reduce_observation};

fn symmetrize(p: &DMatrix<f64>) -> DMatrix<f64> {
    0.5 * (p + p.transpose())
}

fn ensure_spd(mut a: DMatrix<f64>, min_diag: f64) -> Result<DMatrix<f64>> {
    a = symmetrize(&a);

    // Floor the diagonal to keep things numerically sane.
    let n = a.nrows().min(a.ncols());
    for i in 0..n {
        if !a[(i, i)].is_finite() {
            return Err(Error::Validation("covariance has non-finite diagonal".to_string()));
        }
        if a[(i, i)] < min_diag {
            a[(i, i)] = min_diag;
        }
    }

    // If still not SPD, add increasing jitter until Cholesky succeeds.
    if a.clone().cholesky().is_some() {
        return Ok(a);
    }

    let mut jitter = min_diag.max(1e-12);
    for _ in 0..20 {
        let j = DMatrix::<f64>::identity(a.nrows(), a.ncols()) * jitter;
        let candidate = symmetrize(&(a.clone() + j));
        if candidate.clone().cholesky().is_some() {
            return Ok(candidate);
        }
        jitter *= 10.0;
    }

    Err(Error::Computation(
        "failed to make covariance SPD (Cholesky never succeeded)".to_string(),
    ))
}

/// EM configuration.
#[derive(Debug, Clone)]
pub struct KalmanEmConfig {
    /// Maximum EM iterations.
    pub max_iter: usize,
    /// Relative tolerance on log-likelihood improvement.
    pub tol: f64,
    /// Whether to update Q.
    pub estimate_q: bool,
    /// Whether to update R.
    pub estimate_r: bool,
    /// Whether to update F (currently only supports 1D: n_state=1).
    pub estimate_f: bool,
    /// Whether to update H (currently only supports 1D: n_obs=1 and n_state=1).
    pub estimate_h: bool,
    /// Minimum diagonal value applied to Q/R to avoid degeneracy.
    pub min_diag: f64,
}

impl Default for KalmanEmConfig {
    fn default() -> Self {
        Self {
            max_iter: 50,
            tol: 1e-6,
            estimate_q: true,
            estimate_r: true,
            estimate_f: false,
            estimate_h: false,
            min_diag: 1e-12,
        }
    }
}

/// EM result.
#[derive(Debug, Clone)]
pub struct KalmanEmResult {
    /// Fitted model.
    pub model: KalmanModel,
    /// Log-likelihood per EM iteration (including iteration 0 at the initial params).
    pub loglik_trace: Vec<f64>,
    /// Whether the stopping criterion was met.
    pub converged: bool,
    /// Number of completed EM iterations.
    pub n_iter: usize,
}

#[derive(Debug, Clone)]
struct SmoothFull {
    m: Vec<DVector<f64>>,
    p: Vec<DMatrix<f64>>,
    // J_t for t=0..T-2
    gains: Vec<DMatrix<f64>>,
}

fn smoother_full(model: &KalmanModel, fr: &KalmanFilterResult) -> Result<SmoothFull> {
    // Use the existing smoother to get smoothed means/covs.
    let sr = rts_smoother(model, fr)?;

    let t_max = sr.smoothed_means.len();
    if t_max < 2 {
        return Err(Error::Validation("need at least 2 timesteps for EM".to_string()));
    }

    // Recompute/store smoother gains J_t = P_{t|t} F^T (P_{t+1|t})^{-1}.
    let mut gains = Vec::with_capacity(t_max - 1);
    for t in 0..t_max - 1 {
        let p_filt = &fr.filtered_covs[t];
        let p_pred_next = &fr.predicted_covs[t + 1];
        let chol = p_pred_next.clone().cholesky().ok_or_else(|| {
            Error::Computation("RTS smoother failed: predicted covariance not SPD".to_string())
        })?;
        let pf_ft = p_filt * model.f.transpose();
        let x = chol.solve(&pf_ft.transpose());
        let j = x.transpose();
        gains.push(j);
    }

    Ok(SmoothFull {
        m: sr.smoothed_means,
        p: sr.smoothed_covs,
        gains,
    })
}

fn e_xx(m: &DVector<f64>, p: &DMatrix<f64>) -> DMatrix<f64> {
    p + m * m.transpose()
}

fn e_xnext_x(m_next: &DVector<f64>, m: &DVector<f64>, p_next: &DMatrix<f64>, j: &DMatrix<f64>) -> DMatrix<f64> {
    // Approximate lag-one smoothed covariance:
    // Cov(x_{t+1}, x_t | Y) ~= P_{t+1|T} J_t^T
    //
    // This is sufficient for a baseline EM implementation and is validated by
    // synthetic-data tests in this module.
    let cov = p_next * j.transpose();
    cov + m_next * m.transpose()
}

/// Fit Q/R with EM while holding F/H/m0/P0 fixed.
pub fn kalman_em(model: &KalmanModel, ys: &[DVector<f64>], cfg: KalmanEmConfig) -> Result<KalmanEmResult> {
    if cfg.max_iter == 0 {
        return Err(Error::Validation("max_iter must be > 0".to_string()));
    }
    if !cfg.tol.is_finite() || cfg.tol <= 0.0 {
        return Err(Error::Validation("tol must be finite and > 0".to_string()));
    }
    if !cfg.min_diag.is_finite() || cfg.min_diag <= 0.0 {
        return Err(Error::Validation("min_diag must be finite and > 0".to_string()));
    }
    if ys.len() < 2 {
        return Err(Error::Validation("need at least 2 observations".to_string()));
    }

    let n = model.n_state();
    let m_obs = model.n_obs();

    // Current implementation of estimating F/H is scalar-only (1D).
    if (cfg.estimate_f || cfg.estimate_h) && !(n == 1 && m_obs == 1) {
        return Err(Error::Validation(
            "estimate_f/estimate_h currently require n_state=1 and n_obs=1".to_string(),
        ));
    }
    if cfg.estimate_h && cfg.estimate_q {
        // With free Q (state scale) and H (observation scale), the model becomes poorly
        // identifiable. Keep this baseline implementation explicit and safe.
        return Err(Error::Validation(
            "estimate_h requires estimate_q=false (fix Q to break scale degeneracy)".to_string(),
        ));
    }

    let mut cur = model.clone();
    let mut trace = Vec::with_capacity(cfg.max_iter + 1);
    let mut converged = false;

    let mut prev_ll: Option<f64> = None;
    let mut n_iter = 0usize;

    for iter in 0..=cfg.max_iter {
        let fr = kalman_filter(&cur, ys)?;
        let ll = fr.log_likelihood;
        trace.push(ll);

        if let Some(prev) = prev_ll {
            let denom = 1.0 + prev.abs();
            let rel = ((ll - prev).abs()) / denom;
            if rel <= cfg.tol {
                converged = true;
                n_iter = iter;
                break;
            }
        }

        // Last iteration in budget: stop after recording ll.
        if iter == cfg.max_iter {
            n_iter = iter;
            break;
        }

        let sf = smoother_full(&cur, &fr)?;

        // M-step updates.
        if cfg.estimate_f {
            // a = sum E[x_t x_{t-1}] / sum E[x_{t-1}^2]
            let mut num = 0.0;
            let mut den = 0.0;
            for t in 0..ys.len() - 1 {
                let e_xt_xt = e_xx(&sf.m[t], &sf.p[t])[(0, 0)];
                let e_xt1_xt = e_xnext_x(&sf.m[t + 1], &sf.m[t], &sf.p[t + 1], &sf.gains[t])[(0, 0)];
                den += e_xt_xt;
                num += e_xt1_xt;
            }
            if !den.is_finite() || den <= 0.0 {
                return Err(Error::Computation("EM estimate_f failed: invalid denominator".to_string()));
            }
            let a = num / den;
            if !a.is_finite() {
                return Err(Error::Computation("EM estimate_f failed: non-finite a".to_string()));
            }
            cur.f[(0, 0)] = a;
        }

        if cfg.estimate_h {
            // h = sum y_t E[x_t] / sum E[x_t^2] over observed y_t (skip NaN).
            let mut num = 0.0;
            let mut den = 0.0;
            for t in 0..ys.len() {
                let y = ys[t][0];
                if y.is_nan() {
                    continue;
                }
                let e_x = sf.m[t][0];
                let e_x2 = e_xx(&sf.m[t], &sf.p[t])[(0, 0)];
                num += y * e_x;
                den += e_x2;
            }
            if !den.is_finite() || den <= 0.0 {
                return Err(Error::Computation("EM estimate_h failed: invalid denominator".to_string()));
            }
            let h = num / den;
            if !h.is_finite() {
                return Err(Error::Computation("EM estimate_h failed: non-finite h".to_string()));
            }
            cur.h[(0, 0)] = h;
        }

        if cfg.estimate_q {
            let mut sum_q = DMatrix::<f64>::zeros(n, n);

            for t in 0..ys.len() - 1 {
                let e_x1x1 = e_xx(&sf.m[t + 1], &sf.p[t + 1]);
                let e_x0x0 = e_xx(&sf.m[t], &sf.p[t]);
                let e_x1x0 = e_xnext_x(&sf.m[t + 1], &sf.m[t], &sf.p[t + 1], &sf.gains[t]);
                let e_x0x1 = e_x1x0.transpose();

                let term =
                    &e_x1x1 - &cur.f * &e_x0x1 - &e_x1x0 * cur.f.transpose() + &cur.f * e_x0x0 * cur.f.transpose();
                sum_q += term;
            }

            let q_new = sum_q / ((ys.len() - 1) as f64);
            cur.q = ensure_spd(q_new, cfg.min_diag)?;
        }

        if cfg.estimate_r {
            // Update R using only observed dimensions (NaN means missing). For multivariate
            // observations, we accumulate contributions for (i,j) pairs only when both dims
            // are observed at time t.
            let mut sum_r = DMatrix::<f64>::zeros(m_obs, m_obs);
            let mut counts = DMatrix::<usize>::zeros(m_obs, m_obs);
            for t in 0..ys.len() {
                let y = &ys[t];

                let red = reduce_observation(&cur, y)?;
                let Some(red) = red else {
                    continue;
                };

                let hm = &red.h * &sf.m[t];
                let v = &red.y - hm;
                let term = &v * v.transpose() + &red.h * &sf.p[t] * red.h.transpose();

                for (ii, &i) in red.obs_idx.iter().enumerate() {
                    for (jj, &j) in red.obs_idx.iter().enumerate() {
                        sum_r[(i, j)] += term[(ii, jj)];
                        counts[(i, j)] += 1;
                    }
                }
            }

            let mut r_new = cur.r.clone();
            let mut any_used = false;
            for i in 0..m_obs {
                for j in 0..m_obs {
                    let c = counts[(i, j)];
                    if c > 0 {
                        r_new[(i, j)] = sum_r[(i, j)] / (c as f64);
                        any_used = true;
                    }
                }
            }

            if !any_used {
                return Err(Error::Validation(
                    "cannot estimate R: all observations are missing".to_string(),
                ));
            }

            cur.r = ensure_spd(r_new, cfg.min_diag)?;
        }

        prev_ll = Some(ll);
        n_iter = iter + 1;
    }

    Ok(KalmanEmResult {
        model: cur,
        loglik_trace: trace,
        converged,
        n_iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_gaussian(mean: f64, std: f64, u1: f64, u2: f64) -> f64 {
        // Box-Muller for deterministic-ish tests (inputs should be in (0,1)).
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        mean + std * r * theta.cos()
    }

    #[test]
    fn test_em_recovers_q_r_1d_local_level_smoke() {
        // 1D local level:
        // x_t = x_{t-1} + w, y_t = x_t + v
        // True q,r. Start EM from wrong values.
        let f = DMatrix::from_row_slice(1, 1, &[1.0]);
        let h = DMatrix::from_row_slice(1, 1, &[1.0]);
        let q_true = 0.05;
        let r_true = 0.20;
        let q = DMatrix::from_row_slice(1, 1, &[q_true]);
        let r = DMatrix::from_row_slice(1, 1, &[r_true]);
        let m0 = DVector::from_row_slice(&[0.0]);
        let p0 = DMatrix::from_row_slice(1, 1, &[1.0]);

        let true_model = KalmanModel::new(f.clone(), q, h.clone(), r, m0.clone(), p0.clone()).unwrap();

        // Deterministic pseudo-random sequence.
        let t_max = 200usize;
        let mut xs = vec![0.0f64; t_max];
        let mut ys = Vec::with_capacity(t_max);

        let mut u = 0.1234567f64;
        let mut v = 0.7654321f64;
        for t in 0..t_max {
            // LCG-ish update into (0,1)
            u = (u * 16807.0).fract();
            v = (v * 48271.0).fract();
            let w = sample_gaussian(0.0, q_true.sqrt(), u.max(1e-9), v.max(1e-9));
            let x = if t == 0 { w } else { xs[t - 1] + w };
            xs[t] = x;

            u = (u * 69621.0).fract();
            v = (v * 1013904223.0).fract();
            let e = sample_gaussian(0.0, r_true.sqrt(), u.max(1e-9), v.max(1e-9));
            ys.push(DVector::from_row_slice(&[x + e]));
        }

        let init_model = KalmanModel::new(
            f,
            DMatrix::from_row_slice(1, 1, &[0.5]),
            h,
            DMatrix::from_row_slice(1, 1, &[0.5]),
            m0,
            p0,
        )
        .unwrap();

        let res = kalman_em(
            &init_model,
            &ys,
            KalmanEmConfig {
                max_iter: 50,
                tol: 1e-7,
                estimate_q: true,
                estimate_r: true,
                estimate_f: false,
                estimate_h: false,
                min_diag: 1e-9,
            },
        )
        .unwrap();

        // Log-likelihood should not decrease (EM monotonic, up to tiny numerical issues).
        for w in res.loglik_trace.windows(2) {
            assert!(w[1] + 1e-6 >= w[0]);
        }

        let q_hat = res.model.q[(0, 0)];
        let r_hat = res.model.r[(0, 0)];

        // This is a smoke test: just ensure we're in the right ballpark.
        assert!((q_hat - q_true).abs() <= 0.05, "q_hat={} q_true={}", q_hat, q_true);
        assert!((r_hat - r_true).abs() <= 0.10, "r_hat={} r_true={}", r_hat, r_true);

        // Ensure the fitted model can score the data.
        let fr_true = kalman_filter(&true_model, &ys).unwrap();
        let fr_fit = kalman_filter(&res.model, &ys).unwrap();
        assert!(fr_fit.log_likelihood.is_finite());
        assert!(fr_true.log_likelihood.is_finite());
    }

    #[test]
    fn test_em_estimates_f_h_1d_ar1_smoke() {
        // Truth AR(1): x_t = a x_{t-1} + w; y_t = h x_t + v
        let a_true = 0.8;
        let h_true = 1.3;
        let q_true = 0.05;
        let r_true = 0.20;

        let _true_model = KalmanModel::new(
            DMatrix::from_row_slice(1, 1, &[a_true]),
            DMatrix::from_row_slice(1, 1, &[q_true]),
            DMatrix::from_row_slice(1, 1, &[h_true]),
            DMatrix::from_row_slice(1, 1, &[r_true]),
            DVector::from_row_slice(&[0.0]),
            DMatrix::from_row_slice(1, 1, &[1.0]),
        )
        .unwrap();

        // Simulate deterministically (Box-Muller with LCG-ish uniforms).
        let t_max = 300usize;
        let mut xs = vec![0.0f64; t_max];
        let mut ys = Vec::with_capacity(t_max);

        let mut u = 0.1234567f64;
        let mut v = 0.7654321f64;
        for t in 0..t_max {
            u = (u * 16807.0).fract();
            v = (v * 48271.0).fract();
            let w = sample_gaussian(0.0, q_true.sqrt(), u.max(1e-9), v.max(1e-9));
            let x = if t == 0 { w } else { a_true * xs[t - 1] + w };
            xs[t] = x;

            u = (u * 69621.0).fract();
            v = (v * 1013904223.0).fract();
            let e = sample_gaussian(0.0, r_true.sqrt(), u.max(1e-9), v.max(1e-9));
            ys.push(DVector::from_row_slice(&[h_true * x + e]));
        }

        // Init far from truth for f/h; keep q/r also wrong but positive.
        let init_model = KalmanModel::new(
            DMatrix::from_row_slice(1, 1, &[0.2]),
            DMatrix::from_row_slice(1, 1, &[q_true]),
            DMatrix::from_row_slice(1, 1, &[0.7]),
            DMatrix::from_row_slice(1, 1, &[0.5]),
            DVector::from_row_slice(&[0.0]),
            DMatrix::from_row_slice(1, 1, &[1.0]),
        )
        .unwrap();

        let res = kalman_em(
            &init_model,
            &ys,
            KalmanEmConfig {
                max_iter: 50,
                tol: 1e-7,
                estimate_q: false,
                estimate_r: true,
                estimate_f: true,
                estimate_h: true,
                min_diag: 1e-9,
            },
        )
        .unwrap();

        // Log-likelihood should be non-decreasing up to numerical jitter.
        for w in res.loglik_trace.windows(2) {
            assert!(w[1] + 1e-6 >= w[0]);
        }

        let a_hat = res.model.f[(0, 0)];
        let h_hat = res.model.h[(0, 0)];
        assert!(a_hat.is_finite());
        assert!(h_hat.is_finite());

        // Smoke-level closeness.
        assert!((a_hat - a_true).abs() <= 0.2, "a_hat={} a_true={}", a_hat, a_true);
        assert!((h_hat - h_true).abs() <= 0.3, "h_hat={} h_true={}", h_hat, h_true);

        let fr_fit = kalman_filter(&res.model, &ys).unwrap();
        assert!(fr_fit.log_likelihood.is_finite());
    }

    #[test]
    fn test_em_estimate_r_partial_missing_multivariate_does_not_drop_timesteps() {
        // Regression test: EM should handle partial missing observations the same way as
        // `kalman_filter` (NaN means missing per-dimension), i.e. it must not drop the entire
        // timestep just because one dimension is missing.
        //
        // This catches the failure mode where estimate_r would skip timesteps with ANY NaN,
        // effectively discarding information for multivariate series with partial missing.
        let f = DMatrix::from_row_slice(1, 1, &[1.0]);
        let q = DMatrix::from_row_slice(1, 1, &[0.1]);
        let h = DMatrix::from_row_slice(2, 1, &[1.0, 1.0]);
        let r = DMatrix::from_row_slice(2, 2, &[0.5, 0.0, 0.0, 0.5]);
        let m0 = DVector::from_row_slice(&[0.0]);
        let p0 = DMatrix::from_row_slice(1, 1, &[1.0]);

        let init_model = KalmanModel::new(f, q, h, r, m0, p0).unwrap();

        // Construct partial-missing observations:
        // - even t: y0 observed, y1 missing
        // - odd t:  y0 missing, y1 observed
        let t_max = 50usize;
        let mut ys: Vec<DVector<f64>> = Vec::with_capacity(t_max);
        for t in 0..t_max {
            if t % 2 == 0 {
                ys.push(DVector::from_row_slice(&[1.0, f64::NAN]));
            } else {
                ys.push(DVector::from_row_slice(&[f64::NAN, -1.0]));
            }
        }

        let res = kalman_em(
            &init_model,
            &ys,
            KalmanEmConfig {
                max_iter: 2,
                tol: 1e-9,
                estimate_q: false,
                estimate_r: true,
                estimate_f: false,
                estimate_h: false,
                min_diag: 1e-9,
            },
        )
        .unwrap();

        assert!(res.model.r[(0, 0)].is_finite() && res.model.r[(0, 0)] > 0.0);
        assert!(res.model.r[(1, 1)].is_finite() && res.model.r[(1, 1)] > 0.0);
    }
}
