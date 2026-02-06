//! Student-t distribution utilities.

use ns_core::{Error, Result};
use statrs::function::gamma::ln_gamma;

/// Natural log of π.
const LN_PI: f64 = 1.144_729_885_849_400_2;

/// Log-PDF of a Student-t distribution at `x` with location `mu`, scale `sigma`, and dof `nu`.
pub fn logpdf(x: f64, mu: f64, sigma: f64, nu: f64) -> Result<f64> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(Error::Validation(format!("sigma must be finite and > 0, got {}", sigma)));
    }
    if !nu.is_finite() || nu <= 0.0 {
        return Err(Error::Validation(format!("nu must be finite and > 0, got {}", nu)));
    }

    let z = (x - mu) / sigma;
    let half_nu = 0.5 * nu;
    let a = ln_gamma(0.5 * (nu + 1.0)) - ln_gamma(half_nu);
    let b = -0.5 * (nu.ln() + LN_PI);
    let c = -sigma.ln();
    let d = -0.5 * (nu + 1.0) * (z * z / nu).ln_1p();
    Ok(a + b + c + d)
}

/// Negative log-likelihood of a Student-t distribution at `x`.
pub fn nll(x: f64, mu: f64, sigma: f64, nu: f64) -> Result<f64> {
    Ok(-logpdf(x, mu, sigma, nu)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cauchy_at_zero() {
        // nu=1 => Cauchy(0,1): pdf(0) = 1/pi
        let lp = logpdf(0.0, 0.0, 1.0, 1.0).unwrap();
        assert!((lp + std::f64::consts::PI.ln()).abs() < 1e-12);
    }

    #[test]
    fn test_symmetry() {
        let lp1 = logpdf(1.3, 0.0, 2.0, 5.0).unwrap();
        let lp2 = logpdf(-1.3, 0.0, 2.0, 5.0).unwrap();
        assert!((lp1 - lp2).abs() < 1e-12);
    }

    #[test]
    fn test_invalid_params() {
        assert!(logpdf(0.0, 0.0, 0.0, 5.0).is_err());
        assert!(logpdf(0.0, 0.0, 1.0, 0.0).is_err());
    }
}
