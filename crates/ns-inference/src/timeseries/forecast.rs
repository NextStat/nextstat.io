//! Forecasting utilities for linear-Gaussian state space models.
//!
//! Given a fitted/known model and a filtered state at time T-1, we can forecast:
//! - next latent state distribution x_{T+k}
//! - next observation distribution y_{T+k}

use nalgebra::{DMatrix, DVector};
use ns_core::{Error, Result};
use statrs::distribution::{ContinuousCDF, Normal};

use super::internal::symmetrize;
use super::kalman::{KalmanFilterResult, KalmanModel};

/// Forecast output.
#[derive(Debug, Clone)]
pub struct KalmanForecastResult {
    /// Predicted state means for steps 1..=K (each is n_state).
    pub state_means: Vec<DVector<f64>>,
    /// Predicted state covariances for steps 1..=K (each is n_state x n_state).
    pub state_covs: Vec<DMatrix<f64>>,
    /// Predicted observation means for steps 1..=K (each is n_obs).
    pub obs_means: Vec<DVector<f64>>,
    /// Predicted observation covariances for steps 1..=K (each is n_obs x n_obs).
    pub obs_covs: Vec<DMatrix<f64>>,
}

/// Observation prediction intervals (marginal, per observed dimension).
#[derive(Debug, Clone)]
pub struct KalmanForecastIntervals {
    /// Alpha for the two-sided interval (e.g. 0.05 means 95% interval).
    pub alpha: f64,
    /// Standard normal z-value for `1 - alpha/2`.
    pub z: f64,
    /// Lower bounds for steps 1..=K (each is n_obs).
    pub obs_lower: Vec<DVector<f64>>,
    /// Upper bounds for steps 1..=K (each is n_obs).
    pub obs_upper: Vec<DVector<f64>>,
}

/// Compute marginal normal prediction intervals for the observation forecasts in `fc`.
pub fn kalman_forecast_intervals(
    fc: &KalmanForecastResult,
    alpha: f64,
) -> Result<KalmanForecastIntervals> {
    if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
        return Err(Error::Validation("alpha must be in (0, 1)".to_string()));
    }
    if fc.obs_means.len() != fc.obs_covs.len() {
        return Err(Error::Validation("forecast result has inconsistent lengths".to_string()));
    }
    if fc.obs_means.is_empty() {
        return Err(Error::Validation("forecast result must be non-empty".to_string()));
    }

    let n_obs = fc.obs_means[0].len();
    for (k, (m, s)) in fc.obs_means.iter().zip(fc.obs_covs.iter()).enumerate() {
        if m.len() != n_obs {
            return Err(Error::Validation(format!("obs_means[{k}] has inconsistent length")));
        }
        if s.nrows() != n_obs || s.ncols() != n_obs {
            return Err(Error::Validation(format!("obs_covs[{k}] has wrong shape")));
        }
    }

    let normal = Normal::new(0.0, 1.0)
        .map_err(|e| Error::Validation(format!("failed to construct normal distribution: {e}")))?;
    let z = normal.inverse_cdf(1.0 - 0.5 * alpha);
    if !z.is_finite() || z <= 0.0 {
        return Err(Error::Computation("invalid z for alpha".to_string()));
    }

    let mut obs_lower = Vec::with_capacity(fc.obs_means.len());
    let mut obs_upper = Vec::with_capacity(fc.obs_means.len());

    for (m, s) in fc.obs_means.iter().zip(fc.obs_covs.iter()) {
        let mut lo = DVector::<f64>::zeros(n_obs);
        let mut hi = DVector::<f64>::zeros(n_obs);
        for i in 0..n_obs {
            let mu = m[i];
            let var = s[(i, i)];
            if !mu.is_finite() || !var.is_finite() {
                return Err(Error::Computation(
                    "forecast intervals failed: non-finite mean/variance".to_string(),
                ));
            }
            if var < 0.0 {
                return Err(Error::Computation(
                    "forecast intervals failed: negative marginal variance".to_string(),
                ));
            }
            let sd = var.sqrt();
            if !sd.is_finite() {
                return Err(Error::Computation(
                    "forecast intervals failed: non-finite marginal sd".to_string(),
                ));
            }
            lo[i] = mu - z * sd;
            hi[i] = mu + z * sd;
        }
        obs_lower.push(lo);
        obs_upper.push(hi);
    }

    Ok(KalmanForecastIntervals { alpha, z, obs_lower, obs_upper })
}

/// Forecast K steps ahead starting from a filtered state `(m_last, p_last)` at time T-1.
pub fn kalman_forecast_from_last(
    model: &KalmanModel,
    m_last: &DVector<f64>,
    p_last: &DMatrix<f64>,
    steps: usize,
) -> Result<KalmanForecastResult> {
    if steps == 0 {
        return Err(Error::Validation("steps must be > 0".to_string()));
    }
    let n = model.n_state();
    if m_last.len() != n {
        return Err(Error::Validation("m_last has wrong length".to_string()));
    }
    if p_last.nrows() != n || p_last.ncols() != n {
        return Err(Error::Validation("p_last has wrong shape".to_string()));
    }

    let mut m = m_last.clone();
    let mut p = p_last.clone();

    let mut state_means = Vec::with_capacity(steps);
    let mut state_covs = Vec::with_capacity(steps);
    let mut obs_means = Vec::with_capacity(steps);
    let mut obs_covs = Vec::with_capacity(steps);

    for _k in 0..steps {
        // Predict next state: x <- F x + w
        m = &model.f * &m;
        p = &model.f * &p * model.f.transpose() + &model.q;
        p = symmetrize(&p);

        // Predict observation: y <- H x + v
        let y_mean = &model.h * &m;
        let y_cov = &model.h * &p * model.h.transpose() + &model.r;

        state_means.push(m.clone());
        state_covs.push(p.clone());
        obs_means.push(y_mean);
        obs_covs.push(symmetrize(&y_cov));
    }

    Ok(KalmanForecastResult { state_means, state_covs, obs_means, obs_covs })
}

/// Forecast K steps ahead starting from the last filtered state in `fr`.
pub fn kalman_forecast(
    model: &KalmanModel,
    fr: &KalmanFilterResult,
    steps: usize,
) -> Result<KalmanForecastResult> {
    let t_max = fr.filtered_means.len();
    if t_max == 0 {
        return Err(Error::Validation("filter result must be non-empty".to_string()));
    }
    if fr.filtered_covs.len() != t_max {
        return Err(Error::Validation("filter result has inconsistent lengths".to_string()));
    }
    kalman_forecast_from_last(
        model,
        &fr.filtered_means[t_max - 1],
        &fr.filtered_covs[t_max - 1],
        steps,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_forecast_shapes_1d() {
        let model = KalmanModel::new(
            DMatrix::from_row_slice(1, 1, &[1.0]),
            DMatrix::from_row_slice(1, 1, &[0.1]),
            DMatrix::from_row_slice(1, 1, &[1.0]),
            DMatrix::from_row_slice(1, 1, &[0.2]),
            DVector::from_row_slice(&[0.0]),
            DMatrix::from_row_slice(1, 1, &[1.0]),
        )
        .unwrap();

        let m_last = DVector::from_row_slice(&[0.3]);
        let p_last = DMatrix::from_row_slice(1, 1, &[0.4]);
        let out = kalman_forecast_from_last(&model, &m_last, &p_last, 3).unwrap();

        assert_eq!(out.state_means.len(), 3);
        assert_eq!(out.state_covs.len(), 3);
        assert_eq!(out.obs_means.len(), 3);
        assert_eq!(out.obs_covs.len(), 3);

        for k in 0..3 {
            assert_eq!(out.state_means[k].len(), 1);
            assert_eq!(out.state_covs[k].nrows(), 1);
            assert_eq!(out.obs_means[k].len(), 1);
            assert_eq!(out.obs_covs[k].nrows(), 1);
        }
    }

    #[test]
    fn test_forecast_intervals_shapes_1d() {
        let model = KalmanModel::new(
            DMatrix::from_row_slice(1, 1, &[1.0]),
            DMatrix::from_row_slice(1, 1, &[0.1]),
            DMatrix::from_row_slice(1, 1, &[1.0]),
            DMatrix::from_row_slice(1, 1, &[0.2]),
            DVector::from_row_slice(&[0.0]),
            DMatrix::from_row_slice(1, 1, &[1.0]),
        )
        .unwrap();

        let m_last = DVector::from_row_slice(&[0.3]);
        let p_last = DMatrix::from_row_slice(1, 1, &[0.4]);
        let fc = kalman_forecast_from_last(&model, &m_last, &p_last, 2).unwrap();
        let iv = kalman_forecast_intervals(&fc, 0.05).unwrap();

        assert_eq!(iv.obs_lower.len(), 2);
        assert_eq!(iv.obs_upper.len(), 2);
        for k in 0..2 {
            assert_eq!(iv.obs_lower[k].len(), 1);
            assert_eq!(iv.obs_upper[k].len(), 1);
            assert!(iv.obs_lower[k][0].is_finite());
            assert!(iv.obs_upper[k][0].is_finite());
            assert!(iv.obs_lower[k][0] <= iv.obs_upper[k][0]);
        }
        assert!(iv.z.is_finite() && iv.z > 0.0);
    }
}
