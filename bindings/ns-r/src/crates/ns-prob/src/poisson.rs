//! Poisson distribution utilities.

use ns_core::{Error, Result};
use statrs::function::gamma::ln_gamma;

#[inline]
fn ln_factorial(n: u64) -> f64 {
    ln_gamma(n as f64 + 1.0)
}

/// Log-PMF of a Poisson distribution with mean `lambda` at count `k`.
pub fn logpmf(k: u64, lambda: f64) -> Result<f64> {
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(Error::Validation(format!("lambda must be finite and >= 0, got {}", lambda)));
    }
    if lambda == 0.0 {
        return Ok(if k == 0 { 0.0 } else { f64::NEG_INFINITY });
    }

    let kf = k as f64;
    Ok(kf * lambda.ln() - lambda - ln_factorial(k))
}

/// Negative log-likelihood of a Poisson distribution at `k`.
pub fn nll(k: u64, lambda: f64) -> Result<f64> {
    Ok(-logpmf(k, lambda)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn factorial(n: u64) -> u64 {
        (1..=n).product()
    }

    #[test]
    fn test_basic_value_matches_naive() {
        let k = 3u64;
        let lambda: f64 = 2.5;
        let prob = lambda.powi(k as i32) * (-lambda).exp() / factorial(k) as f64;
        let lp_naive = prob.ln();
        let lp = logpmf(k, lambda).unwrap();
        assert!((lp - lp_naive).abs() < 1e-12);
    }

    #[test]
    fn test_lambda_zero() {
        assert_eq!(logpmf(0, 0.0).unwrap(), 0.0);
        assert!(logpmf(1, 0.0).unwrap().is_infinite());
    }

    #[test]
    fn test_invalid_lambda() {
        assert!(logpmf(0, -1.0).is_err());
    }
}
