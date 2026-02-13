//! Beta distribution utilities.

use ns_core::{Error, Result};
use statrs::function::gamma::ln_gamma;

#[inline]
fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Log-PDF of a Beta(`a`, `b`) distribution at `x`.
///
/// Support: `0 <= x <= 1`.
pub fn logpdf(x: f64, a: f64, b: f64) -> Result<f64> {
    if !a.is_finite() || a <= 0.0 {
        return Err(Error::Validation(format!("a must be finite and > 0, got {}", a)));
    }
    if !b.is_finite() || b <= 0.0 {
        return Err(Error::Validation(format!("b must be finite and > 0, got {}", b)));
    }
    if !(0.0..=1.0).contains(&x) {
        return Ok(f64::NEG_INFINITY);
    }

    let ln_norm = -ln_beta(a, b);
    if x == 0.0 {
        if a < 1.0 {
            return Ok(f64::INFINITY);
        }
        if a > 1.0 {
            return Ok(f64::NEG_INFINITY);
        }
        // a == 1: x term is 0.
        return Ok(ln_norm);
    }
    if x == 1.0 {
        if b < 1.0 {
            return Ok(f64::INFINITY);
        }
        if b > 1.0 {
            return Ok(f64::NEG_INFINITY);
        }
        return Ok(ln_norm);
    }

    Ok(ln_norm + (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln())
}

/// Negative log-likelihood of a Beta distribution at `x`.
pub fn nll(x: f64, a: f64, b: f64) -> Result<f64> {
    Ok(-logpdf(x, a, b)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform() {
        for x in [0.0, 0.2, 0.5, 0.9, 1.0] {
            let lp = logpdf(x, 1.0, 1.0).unwrap();
            assert!((lp - 0.0).abs() < 1e-12, "x={}", x);
        }
    }

    #[test]
    fn test_symmetry_when_a_equals_b() {
        let a = 2.0;
        let b = 2.0;
        let lp1 = logpdf(0.2, a, b).unwrap();
        let lp2 = logpdf(0.8, a, b).unwrap();
        assert!((lp1 - lp2).abs() < 1e-12);
    }

    #[test]
    fn test_out_of_support() {
        let lp = logpdf(-0.1, 2.0, 3.0).unwrap();
        assert!(lp.is_infinite() && lp.is_sign_negative());
    }

    #[test]
    fn test_invalid_params() {
        assert!(logpdf(0.5, 0.0, 1.0).is_err());
        assert!(logpdf(0.5, 1.0, 0.0).is_err());
    }
}
