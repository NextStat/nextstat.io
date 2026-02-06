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

use super::internal::symmetrize;
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

    // Fast path: SPD (Cholesky).
    if let Some(chol) = cov.clone().cholesky() {
        let l = chol.l();
        let mut z = DVector::<f64>::zeros(n);
        for i in 0..n {
            z[i] = StandardNormal.sample(rng);
        }
        return Ok(l * z);
    }

    // Fallback: PSD (e.g. ARMA innovations form can yield rank-deficient Q).
    let cov = symmetrize(cov);
    let eig = nalgebra::linalg::SymmetricEigen::new(cov);
    let mut max_ev = 0.0f64;
    let mut min_ev = f64::INFINITY;
    for &ev in eig.eigenvalues.iter() {
        if ev.is_finite() {
            max_ev = max_ev.max(ev.abs());
            min_ev = min_ev.min(ev);
        }
    }
    let tol = 1e-12 * max_ev.max(1.0);
    if min_ev < -tol {
        return Err(Error::Computation(
            "covariance not PSD (min eigenvalue < 0)".to_string(),
        ));
    }

    let mut a = eig.eigenvectors;
    for i in 0..n {
        let s = eig.eigenvalues[i].max(0.0).sqrt();
        for r in 0..n {
            a[(r, i)] *= s;
        }
    }

    let mut z = DVector::<f64>::zeros(n);
    for i in 0..n {
        z[i] = StandardNormal.sample(rng);
    }
    Ok(a * z)
}

/// Simulate T steps from the model, drawing `x_0 ~ N(m0, P0)` and then evolving.
///
/// Returns (xs, ys) of length T.
pub fn kalman_simulate(model: &KalmanModel, t_max: usize, seed: u64) -> Result<KalmanSimResult> {
    kalman_simulate_with_x0(model, t_max, seed, None)
}

/// Simulate T steps from the model with an explicit initial state `x0`.
///
/// - If `x0` is `None`, draws `x_0 ~ N(m0, P0)` (same as [`kalman_simulate`]).
/// - If `x0` is `Some`, uses it as the initial state deterministically.
pub fn kalman_simulate_with_x0(
    model: &KalmanModel,
    t_max: usize,
    seed: u64,
    x0: Option<DVector<f64>>,
) -> Result<KalmanSimResult> {
    if t_max == 0 {
        return Err(Error::Validation("t_max must be > 0".to_string()));
    }

    let mut rng = StdRng::seed_from_u64(seed);

    let mut xs = Vec::with_capacity(t_max);
    let mut ys = Vec::with_capacity(t_max);

    // Initial state: x0 ~ N(m0, P0)
    let mut x = if let Some(x0) = x0 {
        if x0.len() != model.n_state() {
            return Err(Error::Validation(format!(
                "x0 has wrong length: expected {}, got {}",
                model.n_state(),
                x0.len()
            )));
        }
        if x0.iter().any(|v| !v.is_finite()) {
            return Err(Error::Validation("x0 must contain only finite values".to_string()));
        }
        x0
    } else {
        &model.m0 + sample_mvn_zero(&mut rng, &model.p0)?
    };

    for t in 0..t_max {
        // State evolution for t>=1: x_t = F x_{t-1} + w_t
        if t > 0 {
            let w = sample_mvn_zero(&mut rng, &model.q)?;
            x = &model.f * x + w;
        }

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

    #[test]
    fn test_simulate_psd_q_smoke() {
        // Q is rank-deficient (PSD), which should still be simulatable.
        let model = KalmanModel::new(
            DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.0, 0.0]),
            DMatrix::from_row_slice(2, 2, &[0.3, 0.3, 0.3, 0.3]),
            DMatrix::from_row_slice(1, 2, &[1.0, 0.0]),
            DMatrix::from_row_slice(1, 1, &[1e-6]),
            DVector::from_row_slice(&[0.0, 0.0]),
            DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]),
        )
        .unwrap();

        let sim = kalman_simulate(&model, 5, 123).unwrap();
        assert_eq!(sim.xs.len(), 5);
        assert_eq!(sim.ys.len(), 5);
        assert_eq!(sim.xs[0].len(), 2);
        assert_eq!(sim.ys[0].len(), 1);
    }

    #[test]
    fn test_simulate_x0_depends_on_seed() {
        let model = KalmanModel::new(
            DMatrix::from_row_slice(1, 1, &[1.0]),
            DMatrix::from_row_slice(1, 1, &[0.1]),
            DMatrix::from_row_slice(1, 1, &[1.0]),
            DMatrix::from_row_slice(1, 1, &[0.2]),
            DVector::from_row_slice(&[0.0]),
            DMatrix::from_row_slice(1, 1, &[1.0]),
        )
        .unwrap();

        let a = kalman_simulate(&model, 3, 1).unwrap();
        let b = kalman_simulate(&model, 3, 2).unwrap();
        assert_ne!(a.xs[0][0], b.xs[0][0]);
    }

    #[test]
    fn test_simulate_can_fix_x0() {
        let model = KalmanModel::new(
            DMatrix::from_row_slice(1, 1, &[2.0]),
            DMatrix::from_row_slice(1, 1, &[0.1]),
            DMatrix::from_row_slice(1, 1, &[1.0]),
            DMatrix::from_row_slice(1, 1, &[0.2]),
            DVector::from_row_slice(&[10.0]),
            DMatrix::from_row_slice(1, 1, &[1.0]),
        )
        .unwrap();

        let a = kalman_simulate_with_x0(&model, 3, 1, Some(DVector::from_row_slice(&[123.0]))).unwrap();
        let b = kalman_simulate_with_x0(&model, 3, 2, Some(DVector::from_row_slice(&[123.0]))).unwrap();
        assert_eq!(a.xs[0][0], 123.0);
        assert_eq!(b.xs[0][0], 123.0);
    }

    #[test]
    fn test_simulate_x0_uses_p0_not_f_m0() {
        // With tiny Q/P0/R, the first state should be near m0, not near F*m0.
        let model = KalmanModel::new(
            DMatrix::from_row_slice(1, 1, &[2.0]),
            DMatrix::from_row_slice(1, 1, &[1e-12]),
            DMatrix::from_row_slice(1, 1, &[1.0]),
            DMatrix::from_row_slice(1, 1, &[1e-12]),
            DVector::from_row_slice(&[10.0]),
            DMatrix::from_row_slice(1, 1, &[1e-12]),
        )
        .unwrap();

        let sim = kalman_simulate(&model, 1, 123).unwrap();
        let x0 = sim.xs[0][0];
        assert!((x0 - 10.0).abs() < 1.0);
        assert!((x0 - 20.0).abs() > 5.0);
    }
}
