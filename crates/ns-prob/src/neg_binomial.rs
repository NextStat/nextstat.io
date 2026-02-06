//! Negative binomial distribution utilities.

use ns_core::{Error, Result};
use statrs::function::gamma::ln_gamma;

#[inline]
fn ln_factorial(n: u64) -> f64 {
    ln_gamma(n as f64 + 1.0)
}

/// Log-PMF of a Negative Binomial distribution (NB2) parameterized by mean `mu` and dispersion `alpha`.
///
/// This is the common GLM parameterization with:
/// - `mu > 0`
/// - `alpha > 0`
/// - `Var(Y) = mu + alpha * mu^2`
pub fn logpmf_mean_disp(k: u64, mu: f64, alpha: f64) -> Result<f64> {
    if !mu.is_finite() || mu <= 0.0 {
        return Err(Error::Validation(format!("mu must be finite and > 0, got {}", mu)));
    }
    if !alpha.is_finite() || alpha <= 0.0 {
        return Err(Error::Validation(format!(
            "alpha must be finite and > 0, got {}",
            alpha
        )));
    }

    let r = 1.0 / alpha;
    let p = r / (r + mu); // in (0,1)
    logpmf_r_p(k, r, p)
}

/// Log-PMF of a Negative Binomial distribution parameterized by `r` and `p`.
///
/// PMF (counting failures `k` before `r` successes):
/// `P(K=k) = C(k+r-1, k) * (1-p)^k * p^r`
///
/// - `r > 0`
/// - `0 < p < 1`
pub fn logpmf_r_p(k: u64, r: f64, p: f64) -> Result<f64> {
    if !r.is_finite() || r <= 0.0 {
        return Err(Error::Validation(format!("r must be finite and > 0, got {}", r)));
    }
    if !p.is_finite() || p <= 0.0 || p >= 1.0 {
        return Err(Error::Validation(format!("p must be finite and in (0,1), got {}", p)));
    }

    let kf = k as f64;
    let ln_coeff = ln_gamma(kf + r) - ln_gamma(r) - ln_factorial(k);
    Ok(ln_coeff + r * p.ln() + kf * (1.0 - p).ln())
}

/// Negative log-likelihood for NB2(mean, alpha).
pub fn nll_mean_disp(k: u64, mu: f64, alpha: f64) -> Result<f64> {
    Ok(-logpmf_mean_disp(k, mu, alpha)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_disp_matches_r_p() {
        let k = 3u64;
        let mu = 2.0;
        let alpha = 0.5;
        let r = 1.0 / alpha;
        let p = r / (r + mu);
        let lp1 = logpmf_mean_disp(k, mu, alpha).unwrap();
        let lp2 = logpmf_r_p(k, r, p).unwrap();
        assert!((lp1 - lp2).abs() < 1e-12);
    }

    #[test]
    fn test_invalid_params() {
        assert!(logpmf_mean_disp(0, 0.0, 1.0).is_err());
        assert!(logpmf_mean_disp(0, 1.0, 0.0).is_err());
        assert!(logpmf_r_p(0, 0.0, 0.5).is_err());
        assert!(logpmf_r_p(0, 1.0, 0.0).is_err());
        assert!(logpmf_r_p(0, 1.0, 1.0).is_err());
    }
}

