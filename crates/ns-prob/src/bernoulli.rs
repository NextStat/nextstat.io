//! Bernoulli distribution utilities.

use ns_core::{Error, Result};

/// Log-PMF of a Bernoulli distribution at `k ∈ {0, 1}` with success probability `p`.
pub fn logpmf(k: u8, p: f64) -> Result<f64> {
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err(Error::Validation(format!("p must be finite and in [0,1], got {}", p)));
    }
    match k {
        0 => Ok((1.0 - p).ln()),
        1 => Ok(p.ln()),
        _ => Err(Error::Validation(format!("k must be 0 or 1, got {}", k))),
    }
}

/// Negative log-likelihood of a Bernoulli distribution at `k`.
pub fn nll(k: u8, p: f64) -> Result<f64> {
    Ok(-logpmf(k, p)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_values() {
        let p = 0.3;
        assert!((logpmf(1, p).unwrap() - p.ln()).abs() < 1e-12);
        assert!((logpmf(0, p).unwrap() - (1.0 - p).ln()).abs() < 1e-12);
    }

    #[test]
    fn test_invalid_inputs() {
        assert!(logpmf(2, 0.5).is_err());
        assert!(logpmf(0, -0.1).is_err());
        assert!(logpmf(0, 1.1).is_err());
    }

    #[test]
    fn test_degenerate_probs() {
        assert_eq!(logpmf(0, 0.0).unwrap(), 0.0);
        assert!(logpmf(1, 0.0).unwrap().is_infinite() && logpmf(1, 0.0).unwrap().is_sign_negative());
        assert_eq!(logpmf(1, 1.0).unwrap(), 0.0);
        assert!(logpmf(0, 1.0).unwrap().is_infinite() && logpmf(0, 1.0).unwrap().is_sign_negative());
    }
}

