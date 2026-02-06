//! Linear-Gaussian Kalman filter and RTS smoother.
//!
//! This module provides a small, dependency-light baseline for Phase 8.
//! It is intentionally focused on correctness and stable numerics.

use nalgebra::{DMatrix, DVector};
use ns_core::{Error, Result};

/// Time-invariant linear Gaussian state-space model.
///
/// State:
/// `x_t = F x_{t-1} + w_t`,  `w_t ~ N(0, Q)`
///
/// Observation:
/// `y_t = H x_t + v_t`,      `v_t ~ N(0, R)`
///
/// Initial prior:
/// `x_0 ~ N(m0, P0)`
#[derive(Debug, Clone)]
pub struct KalmanModel {
    /// State transition matrix `F` (n_state x n_state).
    pub f: DMatrix<f64>,
    /// Process noise covariance `Q` (n_state x n_state).
    pub q: DMatrix<f64>,
    /// Observation matrix `H` (n_obs x n_state).
    pub h: DMatrix<f64>,
    /// Observation noise covariance `R` (n_obs x n_obs).
    pub r: DMatrix<f64>,
    /// Initial state mean `m0` (n_state).
    pub m0: DVector<f64>,
    /// Initial state covariance `P0` (n_state x n_state).
    pub p0: DMatrix<f64>,
}

impl KalmanModel {
    /// Create a new model and validate matrix/vector dimensions.
    pub fn new(
        f: DMatrix<f64>,
        q: DMatrix<f64>,
        h: DMatrix<f64>,
        r: DMatrix<f64>,
        m0: DVector<f64>,
        p0: DMatrix<f64>,
    ) -> Result<Self> {
        let n = f.nrows();
        if n == 0 || f.ncols() != n {
            return Err(Error::Validation("F must be square with n_state>0".to_string()));
        }
        if q.nrows() != n || q.ncols() != n {
            return Err(Error::Validation("Q must be n_state x n_state".to_string()));
        }
        if p0.nrows() != n || p0.ncols() != n {
            return Err(Error::Validation("P0 must be n_state x n_state".to_string()));
        }
        if m0.len() != n {
            return Err(Error::Validation("m0 must have length n_state".to_string()));
        }
        let m = h.nrows();
        if m == 0 || h.ncols() != n {
            return Err(Error::Validation("H must be n_obs x n_state with n_obs>0".to_string()));
        }
        if r.nrows() != m || r.ncols() != m {
            return Err(Error::Validation("R must be n_obs x n_obs".to_string()));
        }

        if f.iter().any(|v| !v.is_finite())
            || q.iter().any(|v| !v.is_finite())
            || h.iter().any(|v| !v.is_finite())
            || r.iter().any(|v| !v.is_finite())
            || m0.iter().any(|v| !v.is_finite())
            || p0.iter().any(|v| !v.is_finite())
        {
            return Err(Error::Validation("model matrices/vectors must be finite".to_string()));
        }

        Ok(Self { f, q, h, r, m0, p0 })
    }

    /// Local level model (random walk) with 1D state and 1D observations.
    ///
    /// State:
    /// `x_t = x_{t-1} + w_t`, `w_t ~ N(0, q)`
    ///
    /// Observation:
    /// `y_t = x_t + v_t`, `v_t ~ N(0, r)`
    pub fn local_level(q: f64, r: f64, m0: f64, p0: f64) -> Result<Self> {
        if !q.is_finite() || q <= 0.0 {
            return Err(Error::Validation("q must be finite and > 0".to_string()));
        }
        if !r.is_finite() || r <= 0.0 {
            return Err(Error::Validation("r must be finite and > 0".to_string()));
        }
        if !m0.is_finite() {
            return Err(Error::Validation("m0 must be finite".to_string()));
        }
        if !p0.is_finite() || p0 <= 0.0 {
            return Err(Error::Validation("p0 must be finite and > 0".to_string()));
        }

        KalmanModel::new(
            DMatrix::from_row_slice(1, 1, &[1.0]),
            DMatrix::from_row_slice(1, 1, &[q]),
            DMatrix::from_row_slice(1, 1, &[1.0]),
            DMatrix::from_row_slice(1, 1, &[r]),
            DVector::from_row_slice(&[m0]),
            DMatrix::from_row_slice(1, 1, &[p0]),
        )
    }

    /// Local linear trend model (level + slope) with 2D state and 1D observations.
    ///
    /// State:
    /// `level_t = level_{t-1} + slope_{t-1} + w_level`
    /// `slope_t = slope_{t-1} + w_slope`
    ///
    /// Observation:
    /// `y_t = level_t + v_t`
    pub fn local_linear_trend(
        q_level: f64,
        q_slope: f64,
        r: f64,
        level0: f64,
        slope0: f64,
        p0_level: f64,
        p0_slope: f64,
    ) -> Result<Self> {
        if !q_level.is_finite() || q_level <= 0.0 {
            return Err(Error::Validation("q_level must be finite and > 0".to_string()));
        }
        if !q_slope.is_finite() || q_slope <= 0.0 {
            return Err(Error::Validation("q_slope must be finite and > 0".to_string()));
        }
        if !r.is_finite() || r <= 0.0 {
            return Err(Error::Validation("r must be finite and > 0".to_string()));
        }
        if !level0.is_finite() || !slope0.is_finite() {
            return Err(Error::Validation("initial state must be finite".to_string()));
        }
        if !p0_level.is_finite() || p0_level <= 0.0 {
            return Err(Error::Validation("p0_level must be finite and > 0".to_string()));
        }
        if !p0_slope.is_finite() || p0_slope <= 0.0 {
            return Err(Error::Validation("p0_slope must be finite and > 0".to_string()));
        }

        KalmanModel::new(
            DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 0.0, 1.0]),
            DMatrix::from_row_slice(2, 2, &[q_level, 0.0, 0.0, q_slope]),
            DMatrix::from_row_slice(1, 2, &[1.0, 0.0]),
            DMatrix::from_row_slice(1, 1, &[r]),
            DVector::from_row_slice(&[level0, slope0]),
            DMatrix::from_row_slice(2, 2, &[p0_level, 0.0, 0.0, p0_slope]),
        )
    }

    /// AR(1) state with 1D observations.
    ///
    /// State:
    /// `x_t = phi * x_{t-1} + w_t`, `w_t ~ N(0, q)`
    ///
    /// Observation:
    /// `y_t = x_t + v_t`, `v_t ~ N(0, r)`
    pub fn ar1(phi: f64, q: f64, r: f64, m0: f64, p0: f64) -> Result<Self> {
        if !phi.is_finite() {
            return Err(Error::Validation("phi must be finite".to_string()));
        }
        if !q.is_finite() || q <= 0.0 {
            return Err(Error::Validation("q must be finite and > 0".to_string()));
        }
        if !r.is_finite() || r <= 0.0 {
            return Err(Error::Validation("r must be finite and > 0".to_string()));
        }
        if !m0.is_finite() {
            return Err(Error::Validation("m0 must be finite".to_string()));
        }
        if !p0.is_finite() || p0 <= 0.0 {
            return Err(Error::Validation("p0 must be finite and > 0".to_string()));
        }

        KalmanModel::new(
            DMatrix::from_row_slice(1, 1, &[phi]),
            DMatrix::from_row_slice(1, 1, &[q]),
            DMatrix::from_row_slice(1, 1, &[1.0]),
            DMatrix::from_row_slice(1, 1, &[r]),
            DVector::from_row_slice(&[m0]),
            DMatrix::from_row_slice(1, 1, &[p0]),
        )
    }

    /// Local level + seasonal model (periodic component) with 1D observations.
    ///
    /// State is `[level, s1, s2, ..., s_{period-1}]` (dimension = `period`).
    ///
    /// Level:
    /// `level_t = level_{t-1} + w_level`, `w_level ~ N(0, q_level)`
    ///
    /// Seasonal (sum-to-zero constraint via (period-1)-dim representation):
    /// The seasonal block uses the standard dummy-seasonal transition:
    /// `s1_t = -sum(s_i_{t-1}) + w_season`
    /// `s(i+1)_t = s(i)_{t-1}`
    ///
    /// Observation:
    /// `y_t = level_t + s1_t + v_t`, `v_t ~ N(0, r)`
    pub fn local_level_seasonal(
        period: usize,
        q_level: f64,
        q_season: f64,
        r: f64,
        level0: f64,
        p0_level: f64,
        p0_season: f64,
    ) -> Result<Self> {
        if period < 2 {
            return Err(Error::Validation("period must be >= 2".to_string()));
        }
        if !q_level.is_finite() || q_level <= 0.0 {
            return Err(Error::Validation("q_level must be finite and > 0".to_string()));
        }
        if !q_season.is_finite() || q_season <= 0.0 {
            return Err(Error::Validation("q_season must be finite and > 0".to_string()));
        }
        if !r.is_finite() || r <= 0.0 {
            return Err(Error::Validation("r must be finite and > 0".to_string()));
        }
        if !level0.is_finite() {
            return Err(Error::Validation("level0 must be finite".to_string()));
        }
        if !p0_level.is_finite() || p0_level <= 0.0 {
            return Err(Error::Validation("p0_level must be finite and > 0".to_string()));
        }
        if !p0_season.is_finite() || p0_season <= 0.0 {
            return Err(Error::Validation("p0_season must be finite and > 0".to_string()));
        }

        let sdim = period - 1;
        let dim = 1 + sdim;

        // Build transition F.
        let mut f = DMatrix::<f64>::zeros(dim, dim);
        // level_t = level_{t-1}
        f[(0, 0)] = 1.0;

        // seasonal block
        // first row: [-1, -1, ..., -1]
        for j in 0..sdim {
            f[(1, 1 + j)] = -1.0;
        }
        // subdiagonal shift for s2..s_{sdim}
        for i in 1..sdim {
            f[(1 + i, 1 + (i - 1))] = 1.0;
        }

        // Q: diagonal for simplicity (keeps SPD for simulation).
        let mut q = DMatrix::<f64>::zeros(dim, dim);
        q[(0, 0)] = q_level;
        for j in 0..sdim {
            q[(1 + j, 1 + j)] = q_season;
        }

        // H: y = level + s1
        let mut h = DMatrix::<f64>::zeros(1, dim);
        h[(0, 0)] = 1.0;
        h[(0, 1)] = 1.0;

        let r = DMatrix::from_row_slice(1, 1, &[r]);

        let mut m0 = DVector::<f64>::zeros(dim);
        m0[0] = level0;

        let mut p0 = DMatrix::<f64>::zeros(dim, dim);
        p0[(0, 0)] = p0_level;
        for j in 0..sdim {
            p0[(1 + j, 1 + j)] = p0_season;
        }

        KalmanModel::new(f, q, h, r, m0, p0)
    }

    /// Local linear trend + seasonal model with 1D observations.
    ///
    /// State is `[level, slope, s1, s2, ..., s_{period-1}]`
    /// (dimension = `period + 1`).
    ///
    /// Level/slope follow the local linear trend model, and the seasonal block
    /// uses the same dummy-seasonal transition as [`local_level_seasonal`].
    pub fn local_linear_trend_seasonal(
        period: usize,
        q_level: f64,
        q_slope: f64,
        q_season: f64,
        r: f64,
        level0: f64,
        slope0: f64,
        p0_level: f64,
        p0_slope: f64,
        p0_season: f64,
    ) -> Result<Self> {
        if period < 2 {
            return Err(Error::Validation("period must be >= 2".to_string()));
        }
        if !q_level.is_finite() || q_level <= 0.0 {
            return Err(Error::Validation("q_level must be finite and > 0".to_string()));
        }
        if !q_slope.is_finite() || q_slope <= 0.0 {
            return Err(Error::Validation("q_slope must be finite and > 0".to_string()));
        }
        if !q_season.is_finite() || q_season <= 0.0 {
            return Err(Error::Validation("q_season must be finite and > 0".to_string()));
        }
        if !r.is_finite() || r <= 0.0 {
            return Err(Error::Validation("r must be finite and > 0".to_string()));
        }
        if !level0.is_finite() || !slope0.is_finite() {
            return Err(Error::Validation("initial state must be finite".to_string()));
        }
        if !p0_level.is_finite() || p0_level <= 0.0 {
            return Err(Error::Validation("p0_level must be finite and > 0".to_string()));
        }
        if !p0_slope.is_finite() || p0_slope <= 0.0 {
            return Err(Error::Validation("p0_slope must be finite and > 0".to_string()));
        }
        if !p0_season.is_finite() || p0_season <= 0.0 {
            return Err(Error::Validation("p0_season must be finite and > 0".to_string()));
        }

        let sdim = period - 1;
        let dim = 2 + sdim;

        let mut f = DMatrix::<f64>::zeros(dim, dim);
        // local linear trend block
        f[(0, 0)] = 1.0;
        f[(0, 1)] = 1.0;
        f[(1, 1)] = 1.0;

        // seasonal block starts at index 2
        for j in 0..sdim {
            f[(2, 2 + j)] = -1.0;
        }
        for i in 1..sdim {
            f[(2 + i, 2 + (i - 1))] = 1.0;
        }

        let mut q = DMatrix::<f64>::zeros(dim, dim);
        q[(0, 0)] = q_level;
        q[(1, 1)] = q_slope;
        for j in 0..sdim {
            q[(2 + j, 2 + j)] = q_season;
        }

        // y = level + s1
        let mut h = DMatrix::<f64>::zeros(1, dim);
        h[(0, 0)] = 1.0;
        h[(0, 2)] = 1.0;
        let r = DMatrix::from_row_slice(1, 1, &[r]);

        let mut m0 = DVector::<f64>::zeros(dim);
        m0[0] = level0;
        m0[1] = slope0;

        let mut p0 = DMatrix::<f64>::zeros(dim, dim);
        p0[(0, 0)] = p0_level;
        p0[(1, 1)] = p0_slope;
        for j in 0..sdim {
            p0[(2 + j, 2 + j)] = p0_season;
        }

        KalmanModel::new(f, q, h, r, m0, p0)
    }

    /// Number of latent state dimensions.
    pub fn n_state(&self) -> usize {
        self.f.nrows()
    }

    /// Number of observation dimensions.
    pub fn n_obs(&self) -> usize {
        self.h.nrows()
    }
}

/// Kalman filter output (per-time-step predicted and filtered states).
#[derive(Debug, Clone)]
pub struct KalmanFilterResult {
    /// Total log-likelihood `log p(y_0..y_{T-1})`.
    pub log_likelihood: f64,
    /// Prior means `m_{t|t-1}` for each observation time.
    pub predicted_means: Vec<DVector<f64>>,
    /// Prior covariances `P_{t|t-1}` for each observation time.
    pub predicted_covs: Vec<DMatrix<f64>>,
    /// Posterior means `m_{t|t}` for each observation time.
    pub filtered_means: Vec<DVector<f64>>,
    /// Posterior covariances `P_{t|t}` for each observation time.
    pub filtered_covs: Vec<DMatrix<f64>>,
}

/// RTS smoother output (smoothed states).
#[derive(Debug, Clone)]
pub struct KalmanSmootherResult {
    /// Smoothed means `m_{t|T}`.
    pub smoothed_means: Vec<DVector<f64>>,
    /// Smoothed covariances `P_{t|T}`.
    pub smoothed_covs: Vec<DMatrix<f64>>,
}

fn symmetrize(p: &DMatrix<f64>) -> DMatrix<f64> {
    0.5 * (p + p.transpose())
}

/// Run Kalman filtering on a full observation sequence.
///
/// Returns per-step predicted and filtered state distributions, plus the total log-likelihood.
pub fn kalman_filter(model: &KalmanModel, ys: &[DVector<f64>]) -> Result<KalmanFilterResult> {
    let n = model.n_state();
    let m = model.n_obs();
    if ys.is_empty() {
        return Err(Error::Validation("ys must be non-empty".to_string()));
    }
    for (t, y) in ys.iter().enumerate() {
        if y.len() != m {
            return Err(Error::Validation(format!(
                "y[{}] has wrong length: expected {}, got {}",
                t,
                m,
                y.len()
            )));
        }
        // Missing observations are represented as NaN. Reject infinities.
        if y.iter().any(|v| !v.is_finite() && !v.is_nan()) {
            return Err(Error::Validation(format!(
                "y[{}] must be finite or NaN (NaN means missing)",
                t
            )));
        }
    }

    let mut predicted_means = Vec::with_capacity(ys.len());
    let mut predicted_covs = Vec::with_capacity(ys.len());
    let mut filtered_means = Vec::with_capacity(ys.len());
    let mut filtered_covs = Vec::with_capacity(ys.len());

    let ln_2pi = (2.0 * std::f64::consts::PI).ln();

    // Prior for x_0.
    let mut m_pred = model.m0.clone();
    let mut p_pred = model.p0.clone();
    let mut loglik = 0.0f64;

    for y in ys {
        predicted_means.push(m_pred.clone());
        predicted_covs.push(p_pred.clone());

        // Select observed dimensions (NaN means missing).
        let mut obs_idx: Vec<usize> = Vec::new();
        for i in 0..m {
            if y[i].is_finite() {
                obs_idx.push(i);
            }
        }

        // If nothing is observed at this timestep: skip the update.
        if obs_idx.is_empty() {
            filtered_means.push(m_pred.clone());
            filtered_covs.push(p_pred.clone());

            // Predict next prior: (m_pred, p_pred) <- (F m, F P F^T + Q)
            m_pred = &model.f * &m_pred;
            p_pred = &model.f * &p_pred * model.f.transpose() + &model.q;
            p_pred = symmetrize(&p_pred);
            continue;
        }

        // Build reduced observation system for observed indices.
        let mo = obs_idx.len();
        let mut y_obs = DVector::<f64>::zeros(mo);
        let mut h_obs = DMatrix::<f64>::zeros(mo, n);
        let mut r_obs = DMatrix::<f64>::zeros(mo, mo);
        for (ii, &i) in obs_idx.iter().enumerate() {
            y_obs[ii] = y[i];
            for j in 0..n {
                h_obs[(ii, j)] = model.h[(i, j)];
            }
        }
        for (ii, &i) in obs_idx.iter().enumerate() {
            for (jj, &j) in obs_idx.iter().enumerate() {
                r_obs[(ii, jj)] = model.r[(i, j)];
            }
        }

        // Innovation: v = y_obs - H_obs m_pred
        let y_hat = &h_obs * &m_pred;
        let v = y_obs - y_hat;

        // Innovation covariance: S = H P_pred H^T + R
        let s = &h_obs * &p_pred * h_obs.transpose() + &r_obs;

        let chol = s.cholesky().ok_or_else(|| {
            Error::Computation("Kalman update failed: innovation covariance not SPD".to_string())
        })?;

        // quad = v^T S^{-1} v
        let s_inv_v = chol.solve(&v);
        let quad = v.dot(&s_inv_v);

        // logdet(S) = 2 * sum(log(diag(L)))
        let l = chol.l();
        let mut logdet = 0.0;
        for i in 0..mo {
            let d = l[(i, i)];
            if d <= 0.0 || !d.is_finite() {
                return Err(Error::Computation(
                    "Kalman update failed: invalid Cholesky diagonal".to_string(),
                ));
            }
            logdet += 2.0 * d.ln();
        }

        loglik += -0.5 * ((mo as f64) * ln_2pi + logdet + quad);

        // Kalman gain: K = P_pred H^T S^{-1}
        let ph_t = &p_pred * h_obs.transpose(); // n x mo
        let x = chol.solve(&ph_t.transpose()); // m x n
        let k = x.transpose(); // n x mo

        // Filtered mean: m = m_pred + K v
        let m_filt = &m_pred + &k * v;

        // Joseph form covariance update:
        // P = (I - K H) P_pred (I - K H)^T + K R K^T
        let i = DMatrix::<f64>::identity(n, n);
        let i_minus_kh = &i - &k * &h_obs;
        let p_filt = &i_minus_kh * &p_pred * i_minus_kh.transpose() + &k * &r_obs * k.transpose();
        let p_filt = symmetrize(&p_filt);

        filtered_means.push(m_filt.clone());
        filtered_covs.push(p_filt.clone());

        // Predict next prior: (m_pred, p_pred) <- (F m_filt, F P_filt F^T + Q)
        m_pred = &model.f * m_filt;
        p_pred = &model.f * p_filt * model.f.transpose() + &model.q;
        p_pred = symmetrize(&p_pred);
    }

    Ok(KalmanFilterResult {
        log_likelihood: loglik,
        predicted_means,
        predicted_covs,
        filtered_means,
        filtered_covs,
    })
}

/// Run RTS smoothing given a completed Kalman filter result.
pub fn rts_smoother(model: &KalmanModel, fr: &KalmanFilterResult) -> Result<KalmanSmootherResult> {
    let t_max = fr.filtered_means.len();
    if t_max == 0 {
        return Err(Error::Validation("filter result must be non-empty".to_string()));
    }
    if fr.predicted_means.len() != t_max
        || fr.predicted_covs.len() != t_max
        || fr.filtered_covs.len() != t_max
    {
        return Err(Error::Validation("filter result has inconsistent lengths".to_string()));
    }

    let n = model.n_state();

    let mut m_smooth = fr.filtered_means.clone();
    let mut p_smooth = fr.filtered_covs.clone();

    for t in (0..t_max - 1).rev() {
        // J_t = P_{t|t} F^T (P_{t+1|t})^{-1}
        let p_filt = &fr.filtered_covs[t];
        let p_pred_next = &fr.predicted_covs[t + 1];

        let chol = p_pred_next.clone().cholesky().ok_or_else(|| {
            Error::Computation("RTS smoother failed: predicted covariance not SPD".to_string())
        })?;

        let pf_ft = p_filt * model.f.transpose(); // n x n
        let x = chol.solve(&pf_ft.transpose()); // n x n
        let j = x.transpose(); // n x n

        // m_{t|T} = m_{t|t} + J (m_{t+1|T} - m_{t+1|t})
        let dm = &m_smooth[t + 1] - &fr.predicted_means[t + 1];
        m_smooth[t] = &fr.filtered_means[t] + &j * dm;

        // P_{t|T} = P_{t|t} + J (P_{t+1|T} - P_{t+1|t}) J^T
        let dp = &p_smooth[t + 1] - p_pred_next;
        let p = p_filt + &j * dp * j.transpose();
        p_smooth[t] = symmetrize(&p);

        // Sanity: keep dimensions stable.
        debug_assert_eq!(m_smooth[t].len(), n);
        debug_assert_eq!(p_smooth[t].nrows(), n);
        debug_assert_eq!(p_smooth[t].ncols(), n);
    }

    Ok(KalmanSmootherResult { smoothed_means: m_smooth, smoothed_covs: p_smooth })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_filter(
        y: &[f64],
        f: f64,
        q: f64,
        h: f64,
        r: f64,
        m0: f64,
        p0: f64,
    ) -> (Vec<f64>, Vec<f64>, f64) {
        let mut m_pred = m0;
        let mut p_pred = p0;
        let ln_2pi = (2.0 * std::f64::consts::PI).ln();

        let mut m_filt = Vec::with_capacity(y.len());
        let mut p_filt = Vec::with_capacity(y.len());
        let mut ll = 0.0;

        for &yt in y {
            // Missing observations are represented as NaN: skip update + no likelihood contribution.
            if yt.is_nan() {
                m_filt.push(m_pred);
                p_filt.push(p_pred);
                m_pred = f * m_pred;
                p_pred = f * f * p_pred + q;
                continue;
            }

            // v = y - h m
            let v = yt - h * m_pred;
            // s = h^2 p + r
            let s = h * h * p_pred + r;
            let k = (p_pred * h) / s;
            let m = m_pred + k * v;
            let p = (1.0 - k * h) * p_pred * (1.0 - k * h) + k * r * k;

            ll += -0.5 * (ln_2pi + s.ln() + (v * v) / s);

            m_filt.push(m);
            p_filt.push(p);

            m_pred = f * m;
            p_pred = f * f * p + q;
        }
        (m_filt, p_filt, ll)
    }

    fn assert_close(a: f64, b: f64, tol: f64) {
        let d = (a - b).abs();
        assert!(d <= tol, "a={} b={} |diff|={}", a, b, d);
    }

    #[test]
    fn test_kalman_filter_matches_scalar_reference() {
        // Local level model.
        let f = 1.0;
        let q = 0.1;
        let h = 1.0;
        let r = 0.2;
        let m0 = 0.0;
        let p0 = 1.0;

        let y = vec![0.9, 1.2, 0.8, 1.1];
        let (m_ref, p_ref, ll_ref) =
            scalar_filter(&y, f, q, h, r, m0, p0);

        let model = KalmanModel::new(
            DMatrix::from_row_slice(1, 1, &[f]),
            DMatrix::from_row_slice(1, 1, &[q]),
            DMatrix::from_row_slice(1, 1, &[h]),
            DMatrix::from_row_slice(1, 1, &[r]),
            DVector::from_row_slice(&[m0]),
            DMatrix::from_row_slice(1, 1, &[p0]),
        )
        .unwrap();

        let ys: Vec<DVector<f64>> = y.iter().map(|&v| DVector::from_row_slice(&[v])).collect();
        let fr = kalman_filter(&model, &ys).unwrap();

        assert_eq!(fr.filtered_means.len(), y.len());
        assert_eq!(fr.filtered_covs.len(), y.len());

        for t in 0..y.len() {
            assert_close(fr.filtered_means[t][0], m_ref[t], 1e-12);
            assert_close(fr.filtered_covs[t][0], p_ref[t], 1e-12);
        }
        assert_close(fr.log_likelihood, ll_ref, 1e-12);
    }

    #[test]
    fn test_rts_smoother_shapes_and_finiteness_smoke() {
        let model = KalmanModel::new(
            DMatrix::from_row_slice(1, 1, &[1.0]),
            DMatrix::from_row_slice(1, 1, &[0.1]),
            DMatrix::from_row_slice(1, 1, &[1.0]),
            DMatrix::from_row_slice(1, 1, &[0.2]),
            DVector::from_row_slice(&[0.0]),
            DMatrix::from_row_slice(1, 1, &[1.0]),
        )
        .unwrap();

        let y = vec![0.9, 1.2, 0.8, 1.1];
        let ys: Vec<DVector<f64>> = y.iter().map(|&v| DVector::from_row_slice(&[v])).collect();
        let fr = kalman_filter(&model, &ys).unwrap();
        let sr = rts_smoother(&model, &fr).unwrap();

        assert_eq!(sr.smoothed_means.len(), y.len());
        assert_eq!(sr.smoothed_covs.len(), y.len());
        for t in 0..y.len() {
            assert!(sr.smoothed_means[t][0].is_finite());
            assert!(sr.smoothed_covs[t][0].is_finite());
            assert!(sr.smoothed_covs[t][0] >= 0.0);
        }
    }

    #[test]
    fn test_kalman_filter_allows_missing_obs_as_nan() {
        // Same model as the scalar reference test, but with a missing y at t=1.
        let f = 1.0;
        let q = 0.1;
        let h = 1.0;
        let r = 0.2;
        let m0 = 0.0;
        let p0 = 1.0;

        let model = KalmanModel::new(
            DMatrix::from_row_slice(1, 1, &[f]),
            DMatrix::from_row_slice(1, 1, &[q]),
            DMatrix::from_row_slice(1, 1, &[h]),
            DMatrix::from_row_slice(1, 1, &[r]),
            DVector::from_row_slice(&[m0]),
            DMatrix::from_row_slice(1, 1, &[p0]),
        )
        .unwrap();

        let y = vec![0.9, f64::NAN, 0.8, 1.1];
        let (m_ref, p_ref, ll_ref) = scalar_filter(&y, f, q, h, r, m0, p0);

        let ys: Vec<DVector<f64>> = y.iter().map(|&v| DVector::from_row_slice(&[v])).collect();

        let fr = kalman_filter(&model, &ys).unwrap();
        assert!(fr.log_likelihood.is_finite());
        assert_eq!(fr.filtered_means.len(), y.len());
        assert_eq!(fr.filtered_covs.len(), y.len());

        for t in 0..y.len() {
            assert_close(fr.filtered_means[t][0], m_ref[t], 1e-12);
            assert_close(fr.filtered_covs[t][0], p_ref[t], 1e-12);
        }
        assert_close(fr.log_likelihood, ll_ref, 1e-12);
    }

    #[test]
    fn test_ar1_builder_matches_scalar_reference() {
        let phi = 0.9;
        let q = 0.1;
        let r = 0.2;
        let m0 = 0.0;
        let p0 = 1.0;

        let y = vec![0.9, 1.2, 0.8, 1.1];
        let (m_ref, p_ref, ll_ref) = scalar_filter(&y, phi, q, 1.0, r, m0, p0);

        let model = KalmanModel::ar1(phi, q, r, m0, p0).unwrap();
        let ys: Vec<DVector<f64>> = y.iter().map(|&v| DVector::from_row_slice(&[v])).collect();
        let fr = kalman_filter(&model, &ys).unwrap();

        for t in 0..y.len() {
            assert_close(fr.filtered_means[t][0], m_ref[t], 1e-12);
            assert_close(fr.filtered_covs[t][0], p_ref[t], 1e-12);
        }
        assert_close(fr.log_likelihood, ll_ref, 1e-12);
    }

    #[test]
    fn test_kalman_filter_partial_missing_multivariate_decoupled_matches_scalar_refs() {
        // Build a fully decoupled 2D model:
        // - state dims independent
        // - observation dims independent
        // This lets us validate partial-missing logic by comparing to two scalar filters.
        let f = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let q = DMatrix::from_row_slice(2, 2, &[0.1, 0.0, 0.0, 0.2]);
        let h = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        let r = DMatrix::from_row_slice(2, 2, &[0.3, 0.0, 0.0, 0.4]);
        let m0 = DVector::from_row_slice(&[0.0, 0.0]);
        let p0 = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);

        let model = KalmanModel::new(f, q, h, r, m0, p0).unwrap();

        // Missing is per-component.
        let y0 = vec![0.9, 1.0, 0.8, f64::NAN];
        let y1 = vec![1.1, f64::NAN, 0.95, 1.05];
        let ys: Vec<DVector<f64>> = (0..y0.len())
            .map(|t| DVector::from_row_slice(&[y0[t], y1[t]]))
            .collect();

        let fr = kalman_filter(&model, &ys).unwrap();

        let (m0_ref, p0_ref, ll0_ref) = scalar_filter(&y0, 1.0, 0.1, 1.0, 0.3, 0.0, 1.0);
        let (m1_ref, p1_ref, ll1_ref) = scalar_filter(&y1, 1.0, 0.2, 1.0, 0.4, 0.0, 1.0);

        for t in 0..y0.len() {
            assert_close(fr.filtered_means[t][0], m0_ref[t], 1e-12);
            assert_close(fr.filtered_covs[t][(0, 0)], p0_ref[t], 1e-12);
            assert_close(fr.filtered_means[t][1], m1_ref[t], 1e-12);
            assert_close(fr.filtered_covs[t][(1, 1)], p1_ref[t], 1e-12);

            // No cross-covariance should be introduced for a decoupled system.
            assert!(fr.filtered_covs[t][(0, 1)].abs() <= 1e-12);
            assert!(fr.filtered_covs[t][(1, 0)].abs() <= 1e-12);
        }

        assert_close(fr.log_likelihood, ll0_ref + ll1_ref, 1e-12);
    }

    #[test]
    fn test_local_level_seasonal_builder_smoke() {
        let model = KalmanModel::local_level_seasonal(4, 0.1, 0.2, 0.3, 0.0, 1.0, 1.0).unwrap();
        assert_eq!(model.n_state(), 4);
        assert_eq!(model.n_obs(), 1);

        let ys = vec![
            DVector::from_row_slice(&[0.9]),
            DVector::from_row_slice(&[1.2]),
            DVector::from_row_slice(&[0.8]),
            DVector::from_row_slice(&[1.1]),
        ];
        let fr = kalman_filter(&model, &ys).unwrap();
        assert!(fr.log_likelihood.is_finite());
    }

    #[test]
    fn test_local_linear_trend_seasonal_builder_smoke() {
        let model = KalmanModel::local_linear_trend_seasonal(
            4, 0.1, 0.05, 0.2, 0.3, 0.0, 0.0, 1.0, 1.0, 1.0,
        )
        .unwrap();
        assert_eq!(model.n_state(), 5);
        assert_eq!(model.n_obs(), 1);

        let ys = vec![
            DVector::from_row_slice(&[0.9]),
            DVector::from_row_slice(&[1.2]),
            DVector::from_row_slice(&[0.8]),
            DVector::from_row_slice(&[1.1]),
        ];
        let fr = kalman_filter(&model, &ys).unwrap();
        assert!(fr.log_likelihood.is_finite());
    }
}
