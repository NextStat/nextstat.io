//! Simulation utilities for linear-Gaussian state space models.
//!
//! Generates a latent trajectory and corresponding observations:
//! - x_t = F x_{t-1} + w_t, w_t ~ N(0, Q)
//! - y_t = H x_t     + v_t, v_t ~ N(0, R)

use nalgebra::{DMatrix, DVector};
use ns_core::{Error, Result};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use super::kalman::KalmanModel;

/// Simulation output.
#[derive(Debug, Clone)]
pub struct KalmanSimResult {
    /// Latent states (length T, each is n_state).
    pub xs: Vec<DVector<f64>>,
    /// Observations (length T, each is n_obs).
    pub ys: Vec<DVector<f64>>,
}

fn sample_mvn_zero(rng: &mut StdRng, cov: &DMatrix<f64>) -> Result<DVector<f64>> {
    let n = cov.nrows();
    if cov.ncols() != n || n == 0 {
        return Err(Error::Validation("cov must be square and non-empty".to_string()));
    }

    let chol = cov.clone().cholesky().ok_or_else(|| {
        Error::Computation("covariance not SPD (Cholesky failed)".to_string())
    })?;
    let l = chol.l();

    let mut z = DVector::<f64>::zeros(n);
    for i in 0..n {
        z[i] = StandardNormal.sample(rng);
    }
    Ok(l * z)
}

/// Simulate T steps from the model starting at the initial prior mean `m0`.
///
/// Returns (xs, ys) of length T.
pub fn kalman_simulate(model: &KalmanModel, t_max: usize, seed: u64) -> Result<KalmanSimResult> {
    if t_max == 0 {
        return Err(Error::Validation("t_max must be > 0".to_string()));
    }

    let mut rng = StdRng::seed_from_u64(seed);

    let mut xs = Vec::with_capacity(t_max);
    let mut ys = Vec::with_capacity(t_max);

    let mut x = model.m0.clone();
    for _t in 0..t_max {
        // State evolution
        let w = sample_mvn_zero(&mut rng, &model.q)?;
        x = &model.f * x + w;

        // Observation
        let v = sample_mvn_zero(&mut rng, &model.r)?;
        let y = &model.h * &x + v;

        xs.push(x.clone());
        ys.push(y);
    }

    Ok(KalmanSimResult { xs, ys })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_shapes_smoke() {
        let model = KalmanModel::new(
            DMatrix::from_row_slice(1, 1, &[1.0]),
            DMatrix::from_row_slice(1, 1, &[0.1]),
            DMatrix::from_row_slice(1, 1, &[1.0]),
            DMatrix::from_row_slice(1, 1, &[0.2]),
            DVector::from_row_slice(&[0.0]),
            DMatrix::from_row_slice(1, 1, &[1.0]),
        )
        .unwrap();

        let sim = kalman_simulate(&model, 5, 123).unwrap();
        assert_eq!(sim.xs.len(), 5);
        assert_eq!(sim.ys.len(), 5);
        assert_eq!(sim.xs[0].len(), 1);
        assert_eq!(sim.ys[0].len(), 1);
    }
}

