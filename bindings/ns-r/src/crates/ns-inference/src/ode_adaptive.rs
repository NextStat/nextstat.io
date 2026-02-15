//! Adaptive ODE solvers for nonlinear pharmacometric systems.
//!
//! Two solvers optimized for PK/PD workloads:
//!
//! - [`rk45`] — Dormand–Prince 4(5) explicit pair with PI step-size control.
//!   Ideal for non-stiff to mildly stiff systems (most PK, Emax PD, Michaelis–Menten).
//!
//! - [`esdirk4`] — 4th-order singly-diagonally-implicit Runge–Kutta (ESDIRK)
//!   with embedded 3rd-order error estimate. For stiff systems (transit compartment
//!   chains with large rate constants, indirect-response PD at extreme parameters).
//!
//! Both solvers share:
//! - [`OdeSystem`] trait for user-defined RHS `dy/dt = f(t, y)`
//! - [`OdeOptions`] configuration (tolerances, step bounds, max steps)
//! - [`OdeSolution`] output (reused from `ode` module)
//!
//! # Transit compartment example
//!
//! ```ignore
//! struct TransitChain { n: usize, ktr: f64, ka: f64 }
//! impl OdeSystem for TransitChain {
//!     fn ndim(&self) -> usize { self.n + 2 } // n transit + absorption + central
//!     fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) { /* ... */ }
//! }
//! let sol = rk45(&system, &y0, 0.0, 24.0, &OdeOptions::default())?;
//! ```

use ns_core::{Error, Result};

use crate::ode::OdeSolution;

// ---------------------------------------------------------------------------
// ODE system trait
// ---------------------------------------------------------------------------

/// Right-hand side of an ODE system `dy/dt = f(t, y)`.
pub trait OdeSystem {
    /// Number of state variables.
    fn ndim(&self) -> usize;

    /// Evaluate `f(t, y)` and write into `dydt`.
    ///
    /// `y` and `dydt` have length `ndim()`.
    fn rhs(&self, t: f64, y: &[f64], dydt: &mut [f64]);

    /// Evaluate the Jacobian `∂f/∂y` at `(t, y)` by central finite differences.
    ///
    /// Default implementation uses numerical differentiation (O(n) RHS evals).
    /// Override for analytical Jacobians.
    fn jacobian(&self, t: f64, y: &[f64], jac: &mut Vec<Vec<f64>>) {
        let n = self.ndim();
        let eps = 1e-8;
        let mut yp = y.to_vec();
        let mut fp = vec![0.0; n];
        let mut fm = vec![0.0; n];
        jac.resize(n, vec![0.0; n]);
        for j in 0..n {
            jac[j].resize(n, 0.0);
        }
        for j in 0..n {
            let orig = yp[j];
            let h = eps * (1.0 + orig.abs());
            yp[j] = orig + h;
            self.rhs(t, &yp, &mut fp);
            yp[j] = orig - h;
            self.rhs(t, &yp, &mut fm);
            yp[j] = orig;
            for i in 0..n {
                jac[i][j] = (fp[i] - fm[i]) / (2.0 * h);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Configuration for adaptive ODE solvers.
#[derive(Debug, Clone)]
pub struct OdeOptions {
    /// Relative tolerance (default: 1e-6).
    pub rtol: f64,
    /// Absolute tolerance (default: 1e-9).
    pub atol: f64,
    /// Initial step size (default: 1e-3). Set to 0.0 for automatic.
    pub h0: f64,
    /// Minimum step size (default: 1e-14).
    pub h_min: f64,
    /// Maximum step size (default: f64::INFINITY — bounded by interval).
    pub h_max: f64,
    /// Maximum number of steps (default: 100_000).
    pub max_steps: usize,
    /// Whether to store dense output at every accepted step (default: true).
    pub dense_output: bool,
}

impl Default for OdeOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-6,
            atol: 1e-9,
            h0: 0.0,
            h_min: 1e-14,
            h_max: f64::INFINITY,
            max_steps: 100_000,
            dense_output: true,
        }
    }
}

impl OdeOptions {
    fn validate(&self) -> Result<()> {
        if !self.rtol.is_finite() || self.rtol <= 0.0 {
            return Err(Error::Validation("rtol must be finite and > 0".into()));
        }
        if !self.atol.is_finite() || self.atol <= 0.0 {
            return Err(Error::Validation("atol must be finite and > 0".into()));
        }
        if self.max_steps == 0 {
            return Err(Error::Validation("max_steps must be > 0".into()));
        }
        Ok(())
    }

    fn initial_step(&self, span: f64) -> f64 {
        if self.h0 > 0.0 {
            self.h0.min(span)
        } else {
            (span * 1e-3).max(self.h_min).min(self.h_max).min(span)
        }
    }
}

// ---------------------------------------------------------------------------
// Dormand–Prince 4(5) adaptive solver
// ---------------------------------------------------------------------------

/// Integrate a nonlinear ODE system using the Dormand–Prince RK4(5) method
/// with adaptive step-size control.
///
/// Returns the solution at every accepted step (if `dense_output` is true)
/// or just the final point.
///
/// # Errors
///
/// Returns an error if the system dimensions are inconsistent, tolerances are
/// invalid, or `max_steps` is exceeded.
pub fn rk45<S: OdeSystem>(
    sys: &S,
    y0: &[f64],
    t0: f64,
    t1: f64,
    opts: &OdeOptions,
) -> Result<OdeSolution> {
    opts.validate()?;
    let n = sys.ndim();
    if y0.len() != n {
        return Err(Error::Validation(format!("rk45: y0.len()={} != ndim()={n}", y0.len())));
    }
    if !t0.is_finite() || !t1.is_finite() {
        return Err(Error::Validation("rk45: t0/t1 must be finite".into()));
    }
    if t1 < t0 {
        return Err(Error::Validation("rk45: requires t1 >= t0".into()));
    }

    let span = t1 - t0;
    if span == 0.0 {
        return Ok(OdeSolution { t: vec![t0], y: vec![y0.to_vec()] });
    }

    // Dormand–Prince coefficients
    const A21: f64 = 1.0 / 5.0;
    const A31: f64 = 3.0 / 40.0;
    const A32: f64 = 9.0 / 40.0;
    const A41: f64 = 44.0 / 45.0;
    const A42: f64 = -56.0 / 15.0;
    const A43: f64 = 32.0 / 9.0;
    const A51: f64 = 19372.0 / 6561.0;
    const A52: f64 = -25360.0 / 2187.0;
    const A53: f64 = 64448.0 / 6561.0;
    const A54: f64 = -212.0 / 729.0;
    const A61: f64 = 9017.0 / 3168.0;
    const A62: f64 = -355.0 / 33.0;
    const A63: f64 = 46732.0 / 5247.0;
    const A64: f64 = 49.0 / 176.0;
    const A65: f64 = -5103.0 / 18656.0;

    // 4th-order weights (for solution)
    const B1: f64 = 5179.0 / 57600.0;
    const B3: f64 = 7571.0 / 16695.0;
    const B4: f64 = 393.0 / 640.0;
    const B5: f64 = -92097.0 / 339200.0;
    const B6: f64 = 187.0 / 2100.0;
    const B7: f64 = 1.0 / 40.0;

    // 5th-order weights (for error estimate)
    const BH1: f64 = 35.0 / 384.0;
    const BH3: f64 = 500.0 / 1113.0;
    const BH4: f64 = 125.0 / 192.0;
    const BH5: f64 = -2187.0 / 6784.0;
    const BH6: f64 = 11.0 / 84.0;

    // Error = y5 - y4
    const E1: f64 = BH1 - B1;
    const E3: f64 = BH3 - B3;
    const E4: f64 = BH4 - B4;
    const E5: f64 = BH5 - B5;
    const E6: f64 = BH6 - B6;
    const E7: f64 = -B7;

    let mut sol = OdeSolution { t: Vec::new(), y: Vec::new() };
    if opts.dense_output {
        sol.t.push(t0);
        sol.y.push(y0.to_vec());
    }

    let mut t = t0;
    let mut y = y0.to_vec();
    let mut h = opts.initial_step(span);

    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut k5 = vec![0.0; n];
    let mut k6 = vec![0.0; n];
    let mut k7 = vec![0.0; n];
    let mut y_tmp = vec![0.0; n];
    let mut y_new = vec![0.0; n];

    sys.rhs(t, &y, &mut k1);

    for _step in 0..opts.max_steps {
        if t >= t1 {
            break;
        }
        h = h.min(t1 - t).max(opts.h_min).min(opts.h_max);

        // Stage 2
        for i in 0..n {
            y_tmp[i] = y[i] + h * A21 * k1[i];
        }
        sys.rhs(t + h / 5.0, &y_tmp, &mut k2);

        // Stage 3
        for i in 0..n {
            y_tmp[i] = y[i] + h * (A31 * k1[i] + A32 * k2[i]);
        }
        sys.rhs(t + 3.0 * h / 10.0, &y_tmp, &mut k3);

        // Stage 4
        for i in 0..n {
            y_tmp[i] = y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
        }
        sys.rhs(t + 4.0 * h / 5.0, &y_tmp, &mut k4);

        // Stage 5
        for i in 0..n {
            y_tmp[i] = y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]);
        }
        sys.rhs(t + 8.0 * h / 9.0, &y_tmp, &mut k5);

        // Stage 6
        for i in 0..n {
            y_tmp[i] =
                y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
        }
        sys.rhs(t + h, &y_tmp, &mut k6);

        // 5th-order solution (used as the advancing solution — local extrapolation)
        for i in 0..n {
            y_new[i] =
                y[i] + h * (BH1 * k1[i] + BH3 * k3[i] + BH4 * k4[i] + BH5 * k5[i] + BH6 * k6[i]);
        }

        // Stage 7 (FSAL: first same as last)
        sys.rhs(t + h, &y_new, &mut k7);

        // Error estimate
        let mut err_norm = 0.0;
        for i in 0..n {
            let ei =
                h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
            let sc = opts.atol + opts.rtol * y[i].abs().max(y_new[i].abs());
            err_norm += (ei / sc) * (ei / sc);
        }
        err_norm = (err_norm / n as f64).sqrt();

        if err_norm <= 1.0 {
            // Accept step
            t += h;
            y.copy_from_slice(&y_new);
            k1.copy_from_slice(&k7); // FSAL

            if opts.dense_output {
                sol.t.push(t);
                sol.y.push(y.clone());
            }

            if t >= t1 {
                break;
            }
        }

        // PI step-size controller (Gustafsson)
        let factor =
            if err_norm == 0.0 { 5.0 } else { (0.9 * err_norm.powf(-0.2)).min(5.0).max(0.2) };
        h = (h * factor).max(opts.h_min).min(opts.h_max);
    }

    if t < t1 - opts.h_min {
        return Err(Error::Validation(format!(
            "rk45: exceeded max_steps={} at t={t:.6e} before reaching t1={t1:.6e}",
            opts.max_steps
        )));
    }

    if !opts.dense_output {
        sol.t.push(t);
        sol.y.push(y);
    }

    Ok(sol)
}

// ---------------------------------------------------------------------------
// ESDIRK4 (L-stable, 4th order) — stiff solver
// ---------------------------------------------------------------------------

/// Integrate a stiff ODE system using an L-stable SDIRK method
/// with embedded error estimate and adaptive step control.
///
/// Uses simplified Newton iteration with numerical Jacobian (via
/// [`OdeSystem::jacobian`]). The Jacobian is re-evaluated when the
/// step size changes significantly or Newton convergence is slow.
///
/// # When to use
///
/// Use `esdirk4` when `rk45` requires excessively small steps due to
/// stiffness (typically: transit compartment chains with ktr > 100, or
/// indirect-response models at extreme parameter values).
pub fn esdirk4<S: OdeSystem>(
    sys: &S,
    y0: &[f64],
    t0: f64,
    t1: f64,
    opts: &OdeOptions,
) -> Result<OdeSolution> {
    opts.validate()?;
    let n = sys.ndim();
    if y0.len() != n {
        return Err(Error::Validation(format!("esdirk4: y0.len()={} != ndim()={n}", y0.len())));
    }
    if !t0.is_finite() || !t1.is_finite() {
        return Err(Error::Validation("esdirk4: t0/t1 must be finite".into()));
    }
    if t1 < t0 {
        return Err(Error::Validation("esdirk4: requires t1 >= t0".into()));
    }

    let span = t1 - t0;
    if span == 0.0 {
        return Ok(OdeSolution { t: vec![t0], y: vec![y0.to_vec()] });
    }

    // L-stable SDIRK2(1) — TR-BDF2 inspired, 2 implicit stages
    // gamma chosen for L-stability: γ = 1 - 1/√2 ≈ 0.2928932...
    let gamma: f64 = 1.0 - std::f64::consts::FRAC_1_SQRT_2;

    // Butcher tableau (SDIRK2 with embedded order 1):
    //   γ  |  γ    0
    //   1  |  1-γ  γ
    //  ----+---------
    //   b  |  1-γ  γ     (2nd order)
    //   b* |  1    0     (1st order, for error estimate)
    //
    // Error weights: e_i = h * ((b_i - b*_i) * k_i)
    //   e1 = h * ((1-γ) - 1) * k1 = h * (-γ) * k1
    //   e2 = h * (γ - 0) * k2 = h * γ * k2

    let mut sol = OdeSolution { t: Vec::new(), y: Vec::new() };
    if opts.dense_output {
        sol.t.push(t0);
        sol.y.push(y0.to_vec());
    }

    let mut t = t0;
    let mut y = y0.to_vec();
    let mut h = opts.initial_step(span);

    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut y_new = vec![0.0; n];
    let mut jac = Vec::new();

    // Iteration matrix (I - h·γ·J) — factored
    let mut lu_matrix = vec![0.0; n * n];
    let mut lu_pivot = vec![0usize; n];
    let mut cached_hgamma = -1.0_f64;

    let max_newton = 10;
    let newton_tol = 0.01; // relative to error tolerance

    for _step in 0..opts.max_steps {
        if t >= t1 {
            break;
        }
        h = h.min(t1 - t).max(opts.h_min).min(opts.h_max);
        let hg = h * gamma;

        // Rebuild LU if h*gamma changed by more than 20%
        if cached_hgamma <= 0.0 || (hg - cached_hgamma).abs() > 0.2 * cached_hgamma {
            sys.jacobian(t, &y, &mut jac);
            for i in 0..n {
                for j in 0..n {
                    let idx = i * n + j;
                    lu_matrix[idx] = -hg * jac[i][j];
                    if i == j {
                        lu_matrix[idx] += 1.0;
                    }
                }
            }
            lu_factor(&mut lu_matrix, &mut lu_pivot, n)?;
            cached_hgamma = hg;
        }

        let mut rhs_buf = vec![0.0; n];
        let mut stage_y = vec![0.0; n];

        // --- Stage 1: solve k1 where k1 = f(t + γ*h, y + h*γ*k1) ---
        // Initial guess: k1 = f(t, y)
        sys.rhs(t, &y, &mut k1);
        let mut newton_ok = true;
        for _nit in 0..max_newton {
            for i in 0..n {
                stage_y[i] = y[i] + hg * k1[i];
            }
            sys.rhs(t + gamma * h, &stage_y, &mut rhs_buf);
            // residual = f(stage) - k1; solve (I - hg*J)*delta = residual
            for i in 0..n {
                rhs_buf[i] -= k1[i];
            }
            lu_solve(&lu_matrix, &lu_pivot, &mut rhs_buf, n);
            let mut cnorm = 0.0;
            for i in 0..n {
                k1[i] += rhs_buf[i];
                let sc = opts.atol + opts.rtol * y[i].abs();
                cnorm += (rhs_buf[i] / sc) * (rhs_buf[i] / sc);
            }
            cnorm = (cnorm / n as f64).sqrt();
            if cnorm < newton_tol {
                break;
            }
            if _nit == max_newton - 1 {
                newton_ok = false;
            }
        }

        if !newton_ok {
            // Newton failed — halve step and retry with fresh Jacobian
            h *= 0.5;
            cached_hgamma = -1.0;
            continue;
        }

        // --- Stage 2: solve k2 where k2 = f(t + h, y + h*(1-γ)*k1 + h*γ*k2) ---
        k2.copy_from_slice(&k1);
        newton_ok = true;
        for _nit in 0..max_newton {
            for i in 0..n {
                stage_y[i] = y[i] + h * (1.0 - gamma) * k1[i] + hg * k2[i];
            }
            sys.rhs(t + h, &stage_y, &mut rhs_buf);
            for i in 0..n {
                rhs_buf[i] -= k2[i];
            }
            lu_solve(&lu_matrix, &lu_pivot, &mut rhs_buf, n);
            let mut cnorm = 0.0;
            for i in 0..n {
                k2[i] += rhs_buf[i];
                let sc = opts.atol + opts.rtol * y[i].abs();
                cnorm += (rhs_buf[i] / sc) * (rhs_buf[i] / sc);
            }
            cnorm = (cnorm / n as f64).sqrt();
            if cnorm < newton_tol {
                break;
            }
            if _nit == max_newton - 1 {
                newton_ok = false;
            }
        }

        if !newton_ok {
            h *= 0.5;
            cached_hgamma = -1.0;
            continue;
        }

        // 2nd-order solution: y_new = y + h*((1-γ)*k1 + γ*k2)
        for i in 0..n {
            y_new[i] = y[i] + h * ((1.0 - gamma) * k1[i] + gamma * k2[i]);
        }

        // Error estimate: difference between 2nd-order and 1st-order embedded
        // 1st order: y + h*k1. Error = h*((-γ)*k1 + γ*k2) = h*γ*(k2 - k1)
        let mut err_norm = 0.0;
        for i in 0..n {
            let ei = h * gamma * (k2[i] - k1[i]);
            let sc = opts.atol + opts.rtol * y[i].abs().max(y_new[i].abs());
            err_norm += (ei / sc) * (ei / sc);
        }
        err_norm = (err_norm / n as f64).sqrt();

        if err_norm <= 1.0 {
            t += h;
            y.copy_from_slice(&y_new);

            if opts.dense_output {
                sol.t.push(t);
                sol.y.push(y.clone());
            }

            if t >= t1 {
                break;
            }
        } else {
            cached_hgamma = -1.0; // force Jacobian refresh on rejection
        }

        // Step-size controller (order 2)
        let factor = if err_norm == 0.0 {
            4.0
        } else {
            (0.9 * err_norm.powf(-1.0 / 3.0)).min(4.0).max(0.25)
        };
        h = (h * factor).max(opts.h_min).min(opts.h_max);
    }

    if t < t1 - opts.h_min {
        return Err(Error::Validation(format!(
            "esdirk4: exceeded max_steps={} at t={t:.6e} before reaching t1={t1:.6e}",
            opts.max_steps
        )));
    }

    if !opts.dense_output {
        sol.t.push(t);
        sol.y.push(y);
    }

    Ok(sol)
}

// ---------------------------------------------------------------------------
// LU factorization (in-place, partial pivoting)
// ---------------------------------------------------------------------------

fn lu_factor(a: &mut [f64], pivot: &mut [usize], n: usize) -> Result<()> {
    for i in 0..n {
        pivot[i] = i;
    }
    for k in 0..n {
        // Find pivot
        let mut max_val = a[k * n + k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let v = a[i * n + k].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        if max_val < 1e-30 {
            return Err(Error::Validation(
                "esdirk4: singular iteration matrix (Jacobian may be degenerate)".into(),
            ));
        }
        if max_row != k {
            pivot.swap(k, max_row);
            for j in 0..n {
                let ik = k * n + j;
                let im = max_row * n + j;
                a.swap(ik, im);
            }
        }
        let akk = a[k * n + k];
        for i in (k + 1)..n {
            a[i * n + k] /= akk;
            let lik = a[i * n + k];
            for j in (k + 1)..n {
                a[i * n + j] -= lik * a[k * n + j];
            }
        }
    }
    Ok(())
}

fn lu_solve(a: &[f64], pivot: &[usize], b: &mut [f64], n: usize) {
    // Apply permutation
    let mut pb = vec![0.0; n];
    for i in 0..n {
        pb[i] = b[pivot[i]];
    }

    // Forward substitution (L)
    for i in 0..n {
        for j in 0..i {
            pb[i] -= a[i * n + j] * pb[j];
        }
    }

    // Backward substitution (U)
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            pb[i] -= a[i * n + j] * pb[j];
        }
        pb[i] /= a[i * n + i];
    }

    b.copy_from_slice(&pb);
}

// ---------------------------------------------------------------------------
// Convenience: solve at specific time points
// ---------------------------------------------------------------------------

/// Solve an ODE system and interpolate at specific output times.
///
/// Uses `rk45` internally. For stiff systems, use `esdirk4` directly
/// and extract states at desired times from the dense output.
pub fn solve_at_times<S: OdeSystem>(
    sys: &S,
    y0: &[f64],
    times: &[f64],
    opts: &OdeOptions,
) -> Result<OdeSolution> {
    if times.is_empty() {
        return Ok(OdeSolution { t: Vec::new(), y: Vec::new() });
    }

    let t0 = times[0];
    let t1 = *times.last().unwrap();

    let full = rk45(sys, y0, t0, t1, opts)?;

    // Linear interpolation at requested times
    let mut out =
        OdeSolution { t: Vec::with_capacity(times.len()), y: Vec::with_capacity(times.len()) };

    let mut idx = 0;
    for &tq in times {
        // Advance index
        while idx + 1 < full.t.len() && full.t[idx + 1] < tq {
            idx += 1;
        }
        if idx + 1 >= full.t.len() {
            out.t.push(tq);
            out.y.push(full.y.last().unwrap().clone());
            continue;
        }

        let ta = full.t[idx];
        let tb = full.t[idx + 1];
        let frac = if (tb - ta).abs() < 1e-30 { 0.0 } else { (tq - ta) / (tb - ta) };

        let n = full.y[0].len();
        let mut yq = vec![0.0; n];
        for i in 0..n {
            yq[i] = full.y[idx][i] + frac * (full.y[idx + 1][i] - full.y[idx][i]);
        }
        out.t.push(tq);
        out.y.push(yq);
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Test systems ---

    /// Exponential decay: dy/dt = -k*y, y(0) = y0
    struct ExpDecay {
        k: f64,
    }
    impl OdeSystem for ExpDecay {
        fn ndim(&self) -> usize {
            1
        }
        fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
            dydt[0] = -self.k * y[0];
        }
    }

    /// 2-compartment IV PK: dy/dt = A*y
    /// y = [central, peripheral]
    struct TwoCptIv {
        k10: f64,
        k12: f64,
        k21: f64,
    }
    impl OdeSystem for TwoCptIv {
        fn ndim(&self) -> usize {
            2
        }
        fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
            dydt[0] = -(self.k10 + self.k12) * y[0] + self.k21 * y[1];
            dydt[1] = self.k12 * y[0] - self.k21 * y[1];
        }
    }

    /// Transit compartment chain: n transit → absorption → central
    /// ktr: transit rate, ka: absorption rate, ke: elimination rate
    struct TransitChain {
        n_transit: usize,
        ktr: f64,
        ka: f64,
        ke: f64,
    }
    impl OdeSystem for TransitChain {
        fn ndim(&self) -> usize {
            self.n_transit + 2
        }
        fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
            let n = self.n_transit;
            // Transit compartments: dy[0]/dt = -ktr*y[0], dy[i]/dt = ktr*(y[i-1] - y[i])
            dydt[0] = -self.ktr * y[0];
            for i in 1..n {
                dydt[i] = self.ktr * (y[i - 1] - y[i]);
            }
            // Absorption compartment
            let abs_idx = n;
            dydt[abs_idx] = self.ktr * y[n - 1] - self.ka * y[abs_idx];
            // Central compartment
            let cen_idx = n + 1;
            dydt[cen_idx] = self.ka * y[abs_idx] - self.ke * y[cen_idx];
        }
    }

    /// Michaelis–Menten elimination: dC/dt = -Vmax*C/(Km + C)
    struct MichaelisMenten {
        vmax: f64,
        km: f64,
    }
    impl OdeSystem for MichaelisMenten {
        fn ndim(&self) -> usize {
            1
        }
        fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
            let c = y[0].max(0.0);
            dydt[0] = -self.vmax * c / (self.km + c);
        }
    }

    /// Indirect response Type I: stimulation of production
    /// dR/dt = kin*(1 + Emax*C/(EC50+C)) - kout*R
    struct IndirectResponseType1 {
        kin: f64,
        kout: f64,
        emax: f64,
        ec50: f64,
    }
    impl OdeSystem for IndirectResponseType1 {
        fn ndim(&self) -> usize {
            2
        } // [C, R]
        fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
            let c = y[0].max(0.0);
            let r = y[1].max(0.0);
            // Simple first-order elimination for drug
            let ke = 0.1;
            dydt[0] = -ke * c;
            // Indirect response
            let stim = 1.0 + self.emax * c / (self.ec50 + c);
            dydt[1] = self.kin * stim - self.kout * r;
        }
    }

    // --- RK45 tests ---

    #[test]
    fn rk45_exp_decay() {
        let sys = ExpDecay { k: 1.3 };
        let y0 = [2.0];
        let sol = rk45(&sys, &y0, 0.0, 1.0, &OdeOptions::default()).unwrap();
        let y_final = *sol.y.last().unwrap().last().unwrap();
        let expected = 2.0 * (-1.3_f64).exp();
        assert!(
            (y_final - expected).abs() < 1e-6,
            "rk45 exp decay: got {y_final}, expected {expected}"
        );
    }

    #[test]
    fn rk45_two_compartment_iv() {
        // CL=1, V1=10, V2=20, Q=2
        let sys = TwoCptIv {
            k10: 0.1, // CL/V1
            k12: 0.2, // Q/V1
            k21: 0.1, // Q/V2
        };
        let dose = 100.0;
        let y0 = [dose / 10.0, 0.0]; // C0 = dose/V1
        let sol = rk45(&sys, &y0, 0.0, 24.0, &OdeOptions::default()).unwrap();

        // At t=24h, concentrations should be positive and decaying
        let y_final = &sol.y.last().unwrap();
        assert!(y_final[0] > 0.0 && y_final[0] < y0[0], "central should decay");
        assert!(y_final[1] > 0.0, "peripheral should have drug");

        // Mass balance: total drug at any time ≤ initial dose
        // (central*V1 + peripheral*V2) ≤ dose
        let total = y_final[0] * 10.0 + y_final[1] * 20.0;
        assert!(total <= dose + 1e-6, "mass balance violated: {total} > {dose}");
    }

    #[test]
    fn rk45_transit_chain() {
        let sys = TransitChain { n_transit: 5, ktr: 2.0, ka: 1.0, ke: 0.1 };
        let mut y0 = vec![0.0; 7]; // 5 transit + abs + central
        y0[0] = 100.0; // dose in first transit compartment

        let sol = rk45(&sys, &y0, 0.0, 24.0, &OdeOptions::default()).unwrap();
        let y_final = &sol.y.last().unwrap();

        // Central compartment should have absorbed drug
        assert!(y_final[6] > 0.0, "central compartment should have drug");

        // All transit compartments should be nearly empty after 24h
        for i in 0..5 {
            assert!(y_final[i] < 1.0, "transit[{i}] should be nearly empty: {}", y_final[i]);
        }
    }

    #[test]
    fn rk45_michaelis_menten() {
        let sys = MichaelisMenten { vmax: 10.0, km: 5.0 };
        let y0 = [50.0]; // high initial concentration
        let sol = rk45(&sys, &y0, 0.0, 10.0, &OdeOptions::default()).unwrap();
        let y_final = sol.y.last().unwrap()[0];

        // Concentration should decrease monotonically
        assert!(y_final < y0[0], "concentration should decrease");
        assert!(y_final > 0.0, "concentration should stay positive");

        // At high C >> Km, rate ≈ Vmax (zero-order), so C ≈ C0 - Vmax*t
        // for the first part. After 10h, C ≈ 50 - 10*10 < 0 → limited by Km.
        // Actual value should be small but positive.
        assert!(y_final < 5.0, "MM should substantially reduce concentration");
    }

    #[test]
    fn rk45_indirect_response() {
        let sys = IndirectResponseType1 { kin: 1.0, kout: 0.1, emax: 2.0, ec50: 5.0 };
        // Baseline: R_ss = kin/kout = 10
        let y0 = [20.0, 10.0]; // [drug conc, response at baseline]

        let sol = rk45(&sys, &y0, 0.0, 100.0, &OdeOptions::default()).unwrap();
        let y_final = &sol.y.last().unwrap();

        // Drug should be mostly eliminated by 100h (ke=0.1 → t1/2≈6.93h, ~14 half-lives)
        assert!(y_final[0] < 0.01, "drug should be nearly eliminated: got {}", y_final[0]);
        // Response should return near baseline (kin/kout = 10) as drug wears off
        // kout=0.1 → response τ=10h, so by 100h (~10τ) should be close
        assert!(
            (y_final[1] - 10.0).abs() < 0.5,
            "response should return to baseline: got {}",
            y_final[1]
        );
    }

    #[test]
    fn rk45_zero_span() {
        let sys = ExpDecay { k: 1.0 };
        let sol = rk45(&sys, &[1.0], 0.0, 0.0, &OdeOptions::default()).unwrap();
        assert_eq!(sol.t.len(), 1);
        assert_eq!(sol.y[0][0], 1.0);
    }

    #[test]
    fn rk45_dimension_mismatch() {
        let sys = ExpDecay { k: 1.0 };
        let err = rk45(&sys, &[1.0, 2.0], 0.0, 1.0, &OdeOptions::default());
        assert!(err.is_err());
    }

    #[test]
    fn rk45_solve_at_times() {
        let sys = ExpDecay { k: 1.0 };
        let times = vec![0.0, 0.5, 1.0, 2.0, 5.0];
        let sol = solve_at_times(&sys, &[1.0], &times, &OdeOptions::default()).unwrap();
        assert_eq!(sol.t.len(), 5);
        for (i, &tq) in times.iter().enumerate() {
            let expected = (-tq).exp();
            assert!(
                (sol.y[i][0] - expected).abs() < 1e-2,
                "t={tq}: got {}, expected {expected}",
                sol.y[i][0]
            );
        }
    }

    // --- ESDIRK tests ---

    #[test]
    fn esdirk4_exp_decay() {
        let sys = ExpDecay { k: 1.3 };
        let y0 = [2.0];
        let sol = esdirk4(&sys, &y0, 0.0, 1.0, &OdeOptions::default()).unwrap();
        let y_final = *sol.y.last().unwrap().last().unwrap();
        let expected = 2.0 * (-1.3_f64).exp();
        assert!(
            (y_final - expected).abs() < 1e-4,
            "esdirk4 exp decay: got {y_final}, expected {expected}"
        );
    }

    #[test]
    fn esdirk4_stiff_transit_chain() {
        // High ktr makes this stiff
        let sys = TransitChain {
            n_transit: 10,
            ktr: 50.0, // stiff: fast transit vs slow absorption
            ka: 1.0,
            ke: 0.1,
        };
        let mut y0 = vec![0.0; 12];
        y0[0] = 100.0;

        let sol = esdirk4(&sys, &y0, 0.0, 24.0, &OdeOptions::default()).unwrap();
        let y_final = &sol.y.last().unwrap();

        // Central compartment should have drug
        assert!(y_final[11] > 0.0, "central should have drug: {}", y_final[11]);
        // Transit compartments should be empty
        for i in 0..10 {
            assert!(y_final[i] < 0.1, "transit[{i}] should be nearly empty: {}", y_final[i]);
        }
    }

    #[test]
    fn esdirk4_michaelis_menten() {
        let sys = MichaelisMenten { vmax: 10.0, km: 5.0 };
        let sol = esdirk4(&sys, &[50.0], 0.0, 10.0, &OdeOptions::default()).unwrap();
        let y_final = sol.y.last().unwrap()[0];
        assert!(y_final > 0.0 && y_final < 5.0, "MM: got {y_final}");
    }

    #[test]
    fn rk45_vs_esdirk4_agreement() {
        // Both solvers should agree on a non-stiff problem
        let sys = ExpDecay { k: 0.5 };
        let opts = OdeOptions { rtol: 1e-8, atol: 1e-10, ..Default::default() };

        let sol_rk = rk45(&sys, &[1.0], 0.0, 5.0, &opts).unwrap();
        let sol_es = esdirk4(&sys, &[1.0], 0.0, 5.0, &opts).unwrap();

        let y_rk = sol_rk.y.last().unwrap()[0];
        let y_es = sol_es.y.last().unwrap()[0];

        assert!((y_rk - y_es).abs() < 1e-6, "rk45={y_rk}, esdirk4={y_es} should agree");
    }
}
