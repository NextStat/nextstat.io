//! Bernoulli distribution utilities.

use crate::math::log_sigmoid;
use ns_core::{Error, Result};

/// Log-PMF of a Bernoulli distribution at `x ∈ {0, 1}` with success probability `p`.
pub fn logpmf(x: u8, p: f64) -> Result<f64> {
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err(Error::Validation(format!("p must be finite and in [0,1], got {}", p)));
    }
    match x {
        0 => Ok((1.0 - p).ln()),
        1 => Ok(p.ln()),
        _ => Err(Error::Validation(format!("x must be 0 or 1, got {}", x))),
    }
}

/// Log-PMF of Bernoulli with probability in logit space.
///
/// Uses stable `log_sigmoid` helpers:
/// - `log p = log_sigmoid(logit_p)`
/// - `log(1-p) = log_sigmoid(-logit_p)`
pub fn logpmf_logit(x: u8, logit_p: f64) -> Result<f64> {
    if !logit_p.is_finite() {
        return Err(Error::Validation(format!("logit_p must be finite, got {}", logit_p)));
    }
    match x {
        0 => Ok(log_sigmoid(-logit_p)),
        1 => Ok(log_sigmoid(logit_p)),
        _ => Err(Error::Validation(format!("x must be 0 or 1, got {}", x))),
    }
}

/// Negative log-likelihood for Bernoulli.
pub fn nll(x: u8, p: f64) -> Result<f64> {
    Ok(-logpmf(x, p)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logpmf_basic() {
        let lp0 = logpmf(0, 0.2).unwrap();
        let lp1 = logpmf(1, 0.2).unwrap();
        assert!((lp0 - (0.8f64).ln()).abs() < 1e-12);
        assert!((lp1 - (0.2f64).ln()).abs() < 1e-12);
    }

    #[test]
    fn test_degenerate_probs() {
        assert_eq!(logpmf(0, 0.0).unwrap(), 0.0);
        assert!(
            logpmf(1, 0.0).unwrap().is_infinite() && logpmf(1, 0.0).unwrap().is_sign_negative()
        );
        assert_eq!(logpmf(1, 1.0).unwrap(), 0.0);
        assert!(
            logpmf(0, 1.0).unwrap().is_infinite() && logpmf(0, 1.0).unwrap().is_sign_negative()
        );
    }

    #[test]
    fn test_logpmf_logit_matches_prob() {
        let logit_p: f64 = 1.3;
        let p = 1.0 / (1.0 + (-logit_p).exp());
        let lp_p = logpmf(1, p).unwrap();
        let lp_l = logpmf_logit(1, logit_p).unwrap();
        assert!((lp_p - lp_l).abs() < 1e-12);
    }

    #[test]
    fn test_invalid_inputs() {
        assert!(logpmf(2, 0.5).is_err());
        assert!(logpmf(1, -0.1).is_err());
        assert!(logpmf_logit(0, f64::NAN).is_err());
    }
}
