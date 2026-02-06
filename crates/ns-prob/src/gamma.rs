//! Gamma distribution utilities.

use ns_core::{Error, Result};
use statrs::function::gamma::ln_gamma;

/// Log-PDF of a Gamma distribution with `shape` and `rate` at `x`.
///
/// Parameterization:
/// - `shape > 0`
/// - `rate > 0` (inverse scale)
/// Support: `x >= 0`.
pub fn logpdf_shape_rate(x: f64, shape: f64, rate: f64) -> Result<f64> {
    if !shape.is_finite() || shape <= 0.0 {
        return Err(Error::Validation(format!(
            "shape must be finite and > 0, got {}",
            shape
        )));
    }
    if !rate.is_finite() || rate <= 0.0 {
        return Err(Error::Validation(format!("rate must be finite and > 0, got {}", rate)));
    }
    if x < 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if x == 0.0 {
        if shape < 1.0 {
            return Ok(f64::INFINITY);
        }
        if shape > 1.0 {
            return Ok(f64::NEG_INFINITY);
        }
        // shape == 1 => exponential
        return Ok(rate.ln());
    }

    let ln_norm = shape * rate.ln() - ln_gamma(shape);
    Ok(ln_norm + (shape - 1.0) * x.ln() - rate * x)
}

/// Log-PDF of a Gamma distribution with `shape` and `scale` at `x`.
///
/// Parameterization:
/// - `shape > 0`
/// - `scale > 0`
/// Support: `x >= 0`.
pub fn logpdf_shape_scale(x: f64, shape: f64, scale: f64) -> Result<f64> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::Validation(format!(
            "scale must be finite and > 0, got {}",
            scale
        )));
    }
    logpdf_shape_rate(x, shape, 1.0 / scale)
}

/// Negative log-likelihood for Gamma(shape, rate).
pub fn nll_shape_rate(x: f64, shape: f64, rate: f64) -> Result<f64> {
    Ok(-logpdf_shape_rate(x, shape, rate)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_one_matches_exponential() {
        let x = 0.7;
        let rate = 2.3;
        let lp_g = logpdf_shape_rate(x, 1.0, rate).unwrap();
        let lp_e = crate::exponential::logpdf(x, rate).unwrap();
        assert!((lp_g - lp_e).abs() < 1e-12);
    }

    #[test]
    fn test_out_of_support() {
        let lp = logpdf_shape_rate(-0.1, 2.0, 1.0).unwrap();
        assert!(lp.is_infinite() && lp.is_sign_negative());
    }

    #[test]
    fn test_invalid_params() {
        assert!(logpdf_shape_rate(1.0, 0.0, 1.0).is_err());
        assert!(logpdf_shape_rate(1.0, 1.0, 0.0).is_err());
    }
}

