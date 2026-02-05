//! Normal distribution utilities.

use ns_core::{Error, Result};

/// Natural log of `sqrt(2π)`.
///
/// `ln(sqrt(2π)) = 0.5*ln(2π)` (precomputed to keep this crate const-friendly).
const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_7;

/// Log-PDF of a Normal distribution `N(mu, sigma)` at `x`.
///
/// `log p(x) = -0.5 * ((x-mu)/sigma)^2 - ln(sigma) - ln(sqrt(2π))`
pub fn logpdf(x: f64, mu: f64, sigma: f64) -> Result<f64> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(Error::Validation(format!("sigma must be finite and > 0, got {}", sigma)));
    }
    let z = (x - mu) / sigma;
    Ok(-0.5 * z * z - sigma.ln() - LN_SQRT_2PI)
}

/// Negative log-likelihood for a Normal distribution `N(mu, sigma)` at `x`.
pub fn nll(x: f64, mu: f64, sigma: f64) -> Result<f64> {
    Ok(-logpdf(x, mu, sigma)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_at_zero() {
        let lp = logpdf(0.0, 0.0, 1.0).unwrap();
        assert!((lp + LN_SQRT_2PI).abs() < 1e-12);
    }

    #[test]
    fn test_symmetry() {
        let lp1 = logpdf(1.3, 0.0, 2.0).unwrap();
        let lp2 = logpdf(-1.3, 0.0, 2.0).unwrap();
        assert!((lp1 - lp2).abs() < 1e-12);
    }

    #[test]
    fn test_invalid_sigma() {
        assert!(logpdf(0.0, 0.0, 0.0).is_err());
        assert!(logpdf(0.0, 0.0, -1.0).is_err());
    }
}

