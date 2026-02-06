//! Forecasting utilities for linear-Gaussian state space models.
//!
//! Given a fitted/known model and a filtered state at time T-1, we can forecast:
//! - next latent state distribution x_{T+k}
//! - next observation distribution y_{T+k}

use nalgebra::{DMatrix, DVector};
use ns_core::{Error, Result};

use super::kalman::{KalmanFilterResult, KalmanModel};

fn symmetrize(p: &DMatrix<f64>) -> DMatrix<f64> {
    0.5 * (p + p.transpose())
}

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

    Ok(KalmanForecastResult {
        state_means,
        state_covs,
        obs_means,
        obs_covs,
    })
}

/// Forecast K steps ahead starting from the last filtered state in `fr`.
pub fn kalman_forecast(model: &KalmanModel, fr: &KalmanFilterResult, steps: usize) -> Result<KalmanForecastResult> {
    let t_max = fr.filtered_means.len();
    if t_max == 0 {
        return Err(Error::Validation("filter result must be non-empty".to_string()));
    }
    if fr.filtered_covs.len() != t_max {
        return Err(Error::Validation("filter result has inconsistent lengths".to_string()));
    }
    kalman_forecast_from_last(model, &fr.filtered_means[t_max - 1], &fr.filtered_covs[t_max - 1], steps)
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
}

