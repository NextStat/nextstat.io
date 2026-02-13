//! Interpolation utilities (HistFactory-style).

use ns_core::{Error, Result};

/// HistFactory-like interpolation codes for shape (template) modifiers.
///
/// This is a small subset needed for unbinned weight/template morphing:
/// - `Code0`: piecewise linear
/// - `Code4p`: smooth polynomial in [-1,1] with linear extrapolation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoSysInterpCode {
    /// Piecewise linear (InterpCode=0). pyhf default for HistoSys.
    Code0,
    /// Polynomial (InterpCode=4p). Smooth at alpha=0 for HistoSys.
    Code4p,
}

/// Interpolate a value between (down, nominal, up) and return `(value, d/dalpha value)`.
///
/// This matches the pyhf HistFactory interpolators:
/// - Code0: piecewise linear delta
/// - Code4p: smooth polynomial in [-1,1], linear extrapolation outside
pub fn histosys_interp(
    alpha: f64,
    down: f64,
    nominal: f64,
    up: f64,
    code: HistoSysInterpCode,
) -> Result<(f64, f64)> {
    if !alpha.is_finite() {
        return Err(Error::Validation(format!(
            "histosys interpolation requires finite alpha, got {alpha}"
        )));
    }
    if !(down.is_finite() && nominal.is_finite() && up.is_finite()) {
        return Err(Error::Validation(format!(
            "histosys interpolation requires finite templates, got down={down}, nominal={nominal}, up={up}"
        )));
    }

    match code {
        HistoSysInterpCode::Code0 => {
            if alpha >= 0.0 {
                let der = up - nominal;
                Ok((nominal + der * alpha, der))
            } else {
                let der = nominal - down;
                Ok((nominal + der * alpha, der))
            }
        }
        HistoSysInterpCode::Code4p => {
            let delta_up = up - nominal;
            let delta_dn = nominal - down;

            if alpha > 1.0 {
                // Linear extrapolation with +1 slope (delta_up).
                return Ok((nominal + delta_up * alpha, delta_up));
            }
            if alpha < -1.0 {
                // Linear extrapolation with -1 slope (delta_dn).
                return Ok((nominal + delta_dn * alpha, delta_dn));
            }

            // Smooth polynomial region.
            let s = 0.5 * (delta_up + delta_dn);
            let a = 0.0625 * (delta_up - delta_dn);

            // tmp3 = 3 α^6 - 10 α^4 + 15 α^2
            let a2 = alpha * alpha;
            let tmp3 = (3.0 * a2 * a2 * a2) - (10.0 * a2 * a2) + (15.0 * a2);

            // dtmp3/dα = 18 α^5 - 40 α^3 + 30 α
            let dtmp3 = (18.0 * alpha.powi(5)) - (40.0 * alpha.powi(3)) + (30.0 * alpha);

            let delta = alpha * s + tmp3 * a;
            let ddelta = s + dtmp3 * a;

            Ok((nominal + delta, ddelta))
        }
    }
}
