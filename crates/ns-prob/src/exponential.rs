//! Exponential distribution utilities.

use ns_core::{Error, Result};

/// Log-PDF of an Exponential distribution at `x` with rate `rate`.
///
/// Support: `x >= 0`.
pub fn logpdf(x: f64, rate: f64) -> Result<f64> {
    if !rate.is_finite() || rate <= 0.0 {
        return Err(Error::Validation(format!(
            "rate must be finite and > 0, got {}",
            rate
        )));
    }
    if x < 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    Ok(rate.ln() - rate * x)
}

/// Negative log-likelihood of an Exponential distribution at `x`.
pub fn nll(x: f64, rate: f64) -> Result<f64> {
    Ok(-logpdf(x, rate)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_value() {
        let lp = logpdf(0.5, 2.0).unwrap();
        assert!((lp - (2.0f64.ln() - 1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_out_of_support() {
        let lp = logpdf(-0.1, 2.0).unwrap();
        assert!(lp.is_infinite() && lp.is_sign_negative());
    }

    #[test]
    fn test_invalid_rate() {
        assert!(logpdf(0.0, 0.0).is_err());
        assert!(logpdf(0.0, -1.0).is_err());
    }
}

