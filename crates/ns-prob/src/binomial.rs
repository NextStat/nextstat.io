//! Binomial distribution utilities.

use crate::math::log_sigmoid;
use ns_core::{Error, Result};
use statrs::function::gamma::ln_gamma;

fn ln_choose(n: u64, k: u64) -> f64 {
    // ln(n choose k) = ln Γ(n+1) - ln Γ(k+1) - ln Γ(n-k+1)
    let n1 = (n as f64) + 1.0;
    let k1 = (k as f64) + 1.0;
    let nk1 = ((n - k) as f64) + 1.0;
    ln_gamma(n1) - ln_gamma(k1) - ln_gamma(nk1)
}

/// Log-PMF of a Binomial distribution `Binom(n, p)` at count `k`.
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
    Ok(ln_choose(n, k) + kf * p.ln() + (nf - kf) * (1.0 - p).ln())
}

/// Log-PMF of a Binomial distribution with probability in logit space.
pub fn logpmf_logit(k: u64, n: u64, logit_p: f64) -> Result<f64> {
    if k > n {
        return Err(Error::Validation(format!("k must be <= n, got k={} n={}", k, n)));
    }
    if !logit_p.is_finite() {
        return Err(Error::Validation(format!("logit_p must be finite, got {}", logit_p)));
    }
    let kf = k as f64;
    let nf = n as f64;
    Ok(ln_choose(n, k) + kf * log_sigmoid(logit_p) + (nf - kf) * log_sigmoid(-logit_p))
}

/// Negative log-likelihood for Binomial.
pub fn nll(k: u64, n: u64, p: f64) -> Result<f64> {
    Ok(-logpmf(k, n, p)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logpmf_logit_matches_prob() {
        let n = 12;
        let logit_p: f64 = -0.7;
        let p = 1.0 / (1.0 + (-logit_p).exp());
        for k in 0..=n {
            let a = logpmf(k, n, p).unwrap();
            let b = logpmf_logit(k, n, logit_p).unwrap();
            assert!((a - b).abs() < 1e-12);
        }
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
        assert!(logpmf(5, 4, 0.5).is_err());
        assert!(logpmf(2, 4, -0.1).is_err());
        assert!(logpmf_logit(2, 4, f64::INFINITY).is_err());
    }
}
