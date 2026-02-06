//! Weibull distribution utilities.

use ns_core::{Error, Result};

/// Log-PDF of a Weibull distribution at `x` with shape `k` and scale `lambda`.
///
/// Support: `x >= 0`.
pub fn logpdf(x: f64, k: f64, lambda: f64) -> Result<f64> {
    if !k.is_finite() || k <= 0.0 {
        return Err(Error::Validation(format!("k must be finite and > 0, got {}", k)));
    }
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(Error::Validation(format!(
            "lambda must be finite and > 0, got {}",
            lambda
        )));
    }
    if x < 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if x == 0.0 {
        if k < 1.0 {
            return Ok(f64::INFINITY);
        }
        if k > 1.0 {
            return Ok(f64::NEG_INFINITY);
        }
        // k == 1 => exponential with rate 1/lambda
        return Ok(-lambda.ln());
    }

    let x_over_l = x / lambda;
    Ok(k.ln() - lambda.ln() + (k - 1.0) * x_over_l.ln() - x_over_l.powf(k))
}

/// Negative log-likelihood for Weibull.
pub fn nll(x: f64, k: f64, lambda: f64) -> Result<f64> {
    Ok(-logpdf(x, k, lambda)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weibull_k1_matches_exponential() {
        let x = 0.7;
        let rate = 2.0;
        let lambda = 1.0 / rate;
        let lp_w = logpdf(x, 1.0, lambda).unwrap();
        let lp_e = crate::exponential::logpdf(x, rate).unwrap();
        assert!((lp_w - lp_e).abs() < 1e-12);
    }

    #[test]
    fn test_out_of_support() {
        let lp = logpdf(-0.1, 2.0, 1.0).unwrap();
        assert!(lp.is_infinite() && lp.is_sign_negative());
    }

    #[test]
    fn test_invalid_params() {
        assert!(logpdf(1.0, 0.0, 1.0).is_err());
        assert!(logpdf(1.0, 1.0, 0.0).is_err());
    }
}
