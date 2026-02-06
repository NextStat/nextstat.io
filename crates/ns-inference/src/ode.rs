//! Ordinary differential equation (ODE) solvers.
//!
//! Baseline deterministic integrator: classic fixed-step RK4.
//!
//! # Step policy (documented)
//! - Integrates forward in time only (`t1 >= t0`).
//! - Uses a fixed step size `dt` **except** for the final step, which is shortened so
//!   the solution lands exactly on `t1`.
//! - A hard `max_steps` guard prevents runaway loops for tiny `dt`.

use nalgebra::{DMatrix, DVector};
use ns_core::{Error, Result};

/// ODE solution as a time grid + states.
#[derive(Debug, Clone)]
pub struct OdeSolution {
    /// Time points (includes both `t0` and `t1`).
    pub t: Vec<f64>,
    /// State vectors aligned with `t`.
    pub y: Vec<Vec<f64>>,
}

impl OdeSolution {
    fn push(&mut self, t: f64, y: &DVector<f64>) {
        self.t.push(t);
        self.y.push(y.iter().copied().collect());
    }
}

/// Integrate a **linear** ODE system using RK4:
///
/// `dy/dt = A * y` with constant `A`.
pub fn rk4_linear(
    a: &DMatrix<f64>,
    y0: &[f64],
    t0: f64,
    t1: f64,
    dt: f64,
    max_steps: usize,
) -> Result<OdeSolution> {
    if !t0.is_finite() || !t1.is_finite() {
        return Err(Error::Validation("rk4_linear: t0/t1 must be finite".to_string()));
    }
    if t1 < t0 {
        return Err(Error::Validation("rk4_linear: requires t1 >= t0".to_string()));
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err(Error::Validation("rk4_linear: dt must be finite and > 0".to_string()));
    }
    if max_steps == 0 {
        return Err(Error::Validation("rk4_linear: max_steps must be > 0".to_string()));
    }
    if y0.is_empty() {
        return Err(Error::Validation("rk4_linear: y0 must be non-empty".to_string()));
    }
    if y0.iter().any(|v| !v.is_finite()) {
        return Err(Error::Validation("rk4_linear: y0 must be finite".to_string()));
    }
    let n = y0.len();
    if a.nrows() != n || a.ncols() != n {
        return Err(Error::Validation(format!(
            "rk4_linear: A must be square (n x n) with n=len(y0)={}, got {}x{}",
            n,
            a.nrows(),
            a.ncols()
        )));
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(Error::Validation("rk4_linear: A must be finite".to_string()));
    }

    let mut sol = OdeSolution { t: Vec::new(), y: Vec::new() };
    let mut t = t0;
    let mut y = DVector::from_column_slice(y0);
    sol.push(t, &y);

    if t1 == t0 {
        return Ok(sol);
    }

    for _step in 0..max_steps {
        if t >= t1 {
            break;
        }

        let h = (t1 - t).min(dt);

        // f(t, y) = A y  (time-independent)
        let k1 = a * &y;
        let k2 = a * (&y + k1.scale(0.5 * h));
        let k3 = a * (&y + k2.scale(0.5 * h));
        let k4 = a * (&y + k3.scale(h));

        y = y + (k1 + k2.scale(2.0) + k3.scale(2.0) + k4).scale(h / 6.0);
        t += h;

        sol.push(t, &y);

        if t == t1 {
            break;
        }
    }

    if *sol.t.last().unwrap_or(&t0) < t1 {
        return Err(Error::Validation(format!(
            "rk4_linear: exceeded max_steps={max_steps} before reaching t1={t1}"
        )));
    }

    Ok(sol)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[test]
    fn rk4_linear_exp_decay_matches_analytic() {
        // y' = -k y  =>  y(t) = y0 * exp(-k t)
        let k = 1.3_f64;
        let y0 = 2.0_f64;
        let a = DMatrix::from_row_slice(1, 1, &[-k]);
        let t0 = 0.0;
        let t1 = 1.0;
        let dt = 1e-2;

        let sol = rk4_linear(&a, &[y0], t0, t1, dt, 1_000_000).unwrap();
        let y1 = sol.y.last().unwrap()[0];
        let expected = y0 * (-k * t1).exp();

        // RK4 global error is O(dt^4); with dt=0.01 this should be very tight.
        // Keep a small buffer for floating-point accumulation differences across platforms.
        assert!((y1 - expected).abs() < 1e-9, "y1={y1} expected={expected}");
        assert_eq!(sol.t.first().copied().unwrap(), t0);
        assert_eq!(sol.t.last().copied().unwrap(), t1);
    }

    #[derive(Debug, Deserialize)]
    struct ExpDecayFixture {
        k: f64,
        y0: f64,
        t0: f64,
        t1: f64,
        dt: f64,
        y1_analytic: f64,
    }

    #[test]
    fn rk4_linear_matches_fixture_exp_decay() {
        let fx: ExpDecayFixture = serde_json::from_str(include_str!("../../../tests/fixtures/ode_rk4_exp_decay.json"))
            .unwrap();
        let a = DMatrix::from_row_slice(1, 1, &[-fx.k]);
        let sol = rk4_linear(&a, &[fx.y0], fx.t0, fx.t1, fx.dt, 1_000_000).unwrap();
        let y1 = sol.y.last().unwrap()[0];
        assert!((y1 - fx.y1_analytic).abs() < 1e-9);
    }
}
