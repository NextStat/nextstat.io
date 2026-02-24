//! Delay Differential Equation (DDE) solvers.
//!
//! Phase 4 (G12): fixed-delay DDE support via a method-of-steps integrator.
//!
//! Implemented model class:
//! - Linear fixed-delay system: `dy/dt = A*y(t) + B*y(t - tau)`
//!
//! Integrator:
//! - Fixed-step RK4 with linear interpolation of past states.
//! - Requires `dt <= tau` so RK stages only depend on already-known history.

use nalgebra::DMatrix;
use ns_core::{Error, Result};

use crate::ode::OdeSolution;

#[inline]
fn matvec(a: &DMatrix<f64>, x: &[f64], out: &mut [f64]) {
    let n = x.len();
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += a[(i, j)] * x[j];
        }
        out[i] = s;
    }
}

#[inline]
fn interp_history(
    times: &[f64],
    states: &[Vec<f64>],
    query_t: f64,
    t0: f64,
    y_history: &[f64],
) -> Vec<f64> {
    if query_t <= t0 || times.is_empty() {
        return y_history.to_vec();
    }
    if query_t <= times[0] {
        return states[0].clone();
    }
    if query_t >= *times.last().unwrap() {
        return states.last().unwrap().clone();
    }

    // Binary-search interval [lo, hi] containing query_t.
    let mut lo = 0usize;
    let mut hi = times.len() - 1;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if times[mid] <= query_t {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    let ta = times[lo];
    let tb = times[hi];
    let frac = if (tb - ta).abs() < 1e-30 { 0.0 } else { (query_t - ta) / (tb - ta) };
    let n = states[lo].len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        y[i] = states[lo][i] + frac * (states[hi][i] - states[lo][i]);
    }
    y
}

/// Integrate a linear fixed-delay DDE:
///
/// `dy/dt = A*y(t) + B*y(t - tau)`
///
/// with constant pre-history `y(t <= t0) = y_history`.
///
/// # Notes
/// - Method of steps with RK4 requires `dt <= tau`.
/// - Final step is shortened to end exactly at `t1`.
#[allow(clippy::too_many_arguments)]
pub fn rk4_linear_dde(
    a: &DMatrix<f64>,
    b: &DMatrix<f64>,
    y0: &[f64],
    y_history: &[f64],
    t0: f64,
    t1: f64,
    tau: f64,
    dt: f64,
    max_steps: usize,
) -> Result<OdeSolution> {
    if !t0.is_finite() || !t1.is_finite() {
        return Err(Error::Validation("rk4_linear_dde: t0/t1 must be finite".to_string()));
    }
    if t1 < t0 {
        return Err(Error::Validation("rk4_linear_dde: requires t1 >= t0".to_string()));
    }
    if !tau.is_finite() || tau <= 0.0 {
        return Err(Error::Validation("rk4_linear_dde: tau must be finite and > 0".to_string()));
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err(Error::Validation("rk4_linear_dde: dt must be finite and > 0".to_string()));
    }
    if dt > tau {
        return Err(Error::Validation(format!(
            "rk4_linear_dde: dt ({dt}) must be <= tau ({tau}) for explicit method-of-steps"
        )));
    }
    if max_steps == 0 {
        return Err(Error::Validation("rk4_linear_dde: max_steps must be > 0".to_string()));
    }
    if y0.is_empty() {
        return Err(Error::Validation("rk4_linear_dde: y0 must be non-empty".to_string()));
    }
    if y0.len() != y_history.len() {
        return Err(Error::Validation(format!(
            "rk4_linear_dde: y0.len()={} != y_history.len()={}",
            y0.len(),
            y_history.len()
        )));
    }
    if y0.iter().any(|v| !v.is_finite()) || y_history.iter().any(|v| !v.is_finite()) {
        return Err(Error::Validation(
            "rk4_linear_dde: y0/y_history must contain only finite values".to_string(),
        ));
    }

    let n = y0.len();
    if a.nrows() != n || a.ncols() != n || b.nrows() != n || b.ncols() != n {
        return Err(Error::Validation(format!(
            "rk4_linear_dde: A and B must be {}x{}, got A={}x{}, B={}x{}",
            n,
            n,
            a.nrows(),
            a.ncols(),
            b.nrows(),
            b.ncols()
        )));
    }
    if a.iter().any(|v| !v.is_finite()) || b.iter().any(|v| !v.is_finite()) {
        return Err(Error::Validation("rk4_linear_dde: A/B must be finite".to_string()));
    }

    let mut sol = OdeSolution { t: Vec::new(), y: Vec::new() };
    let mut t = t0;
    let mut y = y0.to_vec();
    sol.t.push(t);
    sol.y.push(y.clone());

    if t1 == t0 {
        return Ok(sol);
    }

    let mut ay = vec![0.0; n];
    let mut by = vec![0.0; n];
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut ytmp = vec![0.0; n];

    for _ in 0..max_steps {
        if t >= t1 {
            break;
        }
        let h = (t1 - t).min(dt);

        // k1 at t
        let ylag1 = interp_history(&sol.t, &sol.y, t - tau, t0, y_history);
        matvec(a, &y, &mut ay);
        matvec(b, &ylag1, &mut by);
        for i in 0..n {
            k1[i] = ay[i] + by[i];
            ytmp[i] = y[i] + 0.5 * h * k1[i];
        }

        // k2 at t + h/2
        let ylag2 = interp_history(&sol.t, &sol.y, t + 0.5 * h - tau, t0, y_history);
        matvec(a, &ytmp, &mut ay);
        matvec(b, &ylag2, &mut by);
        for i in 0..n {
            k2[i] = ay[i] + by[i];
            ytmp[i] = y[i] + 0.5 * h * k2[i];
        }

        // k3 at t + h/2
        let ylag3 = interp_history(&sol.t, &sol.y, t + 0.5 * h - tau, t0, y_history);
        matvec(a, &ytmp, &mut ay);
        matvec(b, &ylag3, &mut by);
        for i in 0..n {
            k3[i] = ay[i] + by[i];
            ytmp[i] = y[i] + h * k3[i];
        }

        // k4 at t + h
        let ylag4 = interp_history(&sol.t, &sol.y, t + h - tau, t0, y_history);
        matvec(a, &ytmp, &mut ay);
        matvec(b, &ylag4, &mut by);
        for i in 0..n {
            k4[i] = ay[i] + by[i];
        }

        for i in 0..n {
            y[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t += h;
        sol.t.push(t);
        sol.y.push(y.clone());
    }

    if *sol.t.last().unwrap_or(&t0) < t1 {
        return Err(Error::Validation(format!(
            "rk4_linear_dde: exceeded max_steps={max_steps} before reaching t1={t1}"
        )));
    }

    Ok(sol)
}

/// Scalar convenience wrapper for:
///
/// `dy/dt = a*y(t) + b*y(t - tau)`
#[allow(clippy::too_many_arguments)]
pub fn rk4_linear_dde_scalar(
    a: f64,
    b: f64,
    y0: f64,
    y_history: f64,
    t0: f64,
    t1: f64,
    tau: f64,
    dt: f64,
    max_steps: usize,
) -> Result<OdeSolution> {
    let aa = DMatrix::from_row_slice(1, 1, &[a]);
    let bb = DMatrix::from_row_slice(1, 1, &[b]);
    rk4_linear_dde(&aa, &bb, &[y0], &[y_history], t0, t1, tau, dt, max_steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dde_scalar_first_interval_matches_linear_piece() {
        // y'(t) = -y(t-1), history y=1 for t<=0, y(0)=1.
        // For t in [0,1], y(t-1)=1 => y'(t)=-1 => y(t)=1-t.
        let sol = rk4_linear_dde_scalar(-0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.01, 10000).unwrap();
        let y1 = sol.y.last().unwrap()[0];
        assert!((y1 - 0.0).abs() < 5e-3, "y(1)={}, expected ~0", y1);
    }

    #[test]
    fn dde_matrix_reduces_to_ode_when_b_zero() {
        // dy/dt = A y + 0*y(t-tau) -> standard ODE.
        let a = DMatrix::from_row_slice(2, 2, &[-1.0, 0.0, 0.0, -0.5]);
        let b = DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 0.0]);
        let y0 = vec![2.0, 3.0];
        let sol = rk4_linear_dde(&a, &b, &y0, &y0, 0.0, 2.0, 0.5, 0.01, 100000).unwrap();
        let y_end = sol.y.last().unwrap();
        let e0 = 2.0 * (-2.0_f64).exp();
        let e1 = 3.0 * (-1.0_f64).exp();
        assert!((y_end[0] - e0).abs() < 1e-4, "got {}, expected {}", y_end[0], e0);
        assert!((y_end[1] - e1).abs() < 1e-4, "got {}, expected {}", y_end[1], e1);
    }

    #[test]
    fn dde_rejects_dt_larger_than_delay() {
        let a = DMatrix::from_row_slice(1, 1, &[0.0]);
        let b = DMatrix::from_row_slice(1, 1, &[-1.0]);
        let err = rk4_linear_dde(&a, &b, &[1.0], &[1.0], 0.0, 1.0, 0.1, 0.2, 100)
            .expect_err("expected validation error for dt>tau");
        let msg = err.to_string();
        assert!(msg.contains("dt"));
        assert!(msg.contains("tau"));
    }
}
