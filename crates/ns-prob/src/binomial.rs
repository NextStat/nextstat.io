//! Binomial distribution utilities.

use ns_core::{Error, Result};
use statrs::function::gamma::ln_gamma;

#[inline]
fn ln_factorial(n: u64) -> f64 {
    ln_gamma(n as f64 + 1.0)
}

#[inline]
fn ln_choose(n: u64, k: u64) -> f64 {
    ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)
}

/// Log-PMF of a Binomial(`n`, `p`) at `k`.
pub fn logpmf(k: u64, n: u64, p: f64) -> Result<f64> {
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err(Error::Validation(format!("p must be finite and in [0,1], got {}", p)));
    }
    if k > n {
        return Err(Error::Validation(format!("k must be <= n, got k={} n={}", k, n)));
    }

    if p == 0.0 {
        return Ok(if k == 0 { 0.0 } else { f64::NEG_INFINITY });
    }
    if p == 1.0 {
        return Ok(if k == n { 0.0 } else { f64::NEG_INFINITY });
    }

    let kf = k as f64;
    let nf = n as f64;
    let ln_p = p.ln();
    let ln_q = (1.0 - p).ln();
    Ok(ln_choose(n, k) + kf * ln_p + (nf - kf) * ln_q)
}

/// Negative log-likelihood of a Binomial distribution at `k`.
pub fn nll(k: u64, n: u64, p: f64) -> Result<f64> {
    Ok(-logpmf(k, n, p)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn factorial(n: u64) -> u64 {
        (1..=n).product()
    }

    fn choose(n: u64, k: u64) -> u64 {
        factorial(n) / (factorial(k) * factorial(n - k))
    }

    #[test]
    fn test_basic_value_matches_naive() {
        let n = 10u64;
        let k = 3u64;
        let p: f64 = 0.2;
        let prob = choose(n, k) as f64 * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32);
        let lp_naive = prob.ln();
        let lp = logpmf(k, n, p).unwrap();
        assert!((lp - lp_naive).abs() < 1e-12);
    }

    #[test]
    fn test_edges_p0_p1() {
        assert_eq!(logpmf(0, 5, 0.0).unwrap(), 0.0);
        assert!(logpmf(1, 5, 0.0).unwrap().is_infinite());
        assert_eq!(logpmf(5, 5, 1.0).unwrap(), 0.0);
        assert!(logpmf(4, 5, 1.0).unwrap().is_infinite());
    }

    #[test]
    fn test_invalid_inputs() {
        assert!(logpmf(6, 5, 0.5).is_err());
        assert!(logpmf(0, 5, -0.1).is_err());
        assert!(logpmf(0, 5, 1.1).is_err());
    }
}
