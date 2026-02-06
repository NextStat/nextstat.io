//! Scalar log-density helpers for common distributions.
//!
//! Design goal: tiny, dependency-light building blocks for model composition.
//! Parameterizations are explicit in each function name/docstring.

use ns_core::{Error, Result};

use crate::math::{log_sigmoid, sigmoid};

const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_7;

#[inline]
fn ln_gamma(x: f64) -> f64 {
    statrs::function::gamma::ln_gamma(x)
}

/// Log-PDF of Normal `N(mu, sigma)` at `x`.
pub fn normal_logpdf(x: f64, mu: f64, sigma: f64) -> Result<f64> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(Error::Validation(format!("sigma must be finite and > 0, got {}", sigma)));
    }
    let z = (x - mu) / sigma;
    Ok(-0.5 * z * z - sigma.ln() - LN_SQRT_2PI)
}

/// Log-PDF of Student-t with df `nu`, location `mu`, scale `sigma`.
pub fn student_t_logpdf(x: f64, nu: f64, mu: f64, sigma: f64) -> Result<f64> {
    if !nu.is_finite() || nu <= 0.0 {
        return Err(Error::Validation(format!("nu must be finite and > 0, got {}", nu)));
    }
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(Error::Validation(format!("sigma must be finite and > 0, got {}", sigma)));
    }
    let z = (x - mu) / sigma;
    let a = (nu + 1.0) * 0.5;
    let b = nu * 0.5;
    let norm = ln_gamma(a) - ln_gamma(b) - 0.5 * (nu * std::f64::consts::PI).ln() - sigma.ln();
    Ok(norm - a * (1.0 + (z * z) / nu).ln())
}

/// Log-PMF of Bernoulli(y) with logit parameter `eta = log(p/(1-p))`.
pub fn bernoulli_logpmf_logit(y: bool, eta: f64) -> Result<f64> {
    Ok(if y { log_sigmoid(eta) } else { log_sigmoid(-eta) })
}

/// Log-PMF of Binomial(n, k) with logit parameter `eta`.
pub fn binomial_logpmf_logit(k: u64, n: u64, eta: f64) -> Result<f64> {
    if k > n {
        return Err(Error::Validation(format!("k must be <= n, got k={} n={}", k, n)));
    }
    let ln_comb = ln_gamma((n + 1) as f64) - ln_gamma((k + 1) as f64) - ln_gamma((n - k + 1) as f64);
    Ok(ln_comb + (k as f64) * log_sigmoid(eta) + ((n - k) as f64) * log_sigmoid(-eta))
}

/// Log-PMF of Poisson(k | lambda).
pub fn poisson_logpmf(k: u64, lambda: f64) -> Result<f64> {
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(Error::Validation(format!(
            "lambda must be finite and > 0, got {}",
            lambda
        )));
    }
    let kf = k as f64;
    Ok(kf * lambda.ln() - lambda - ln_gamma(kf + 1.0))
}

/// Log-PMF of Negative Binomial with parameters `r` (shape) and `p` (success prob).
///
/// This is the distribution of the number of failures `k` before `r` successes:
/// `P(k) = C(k+r-1, k) * (1-p)^k * p^r`.
pub fn negbinom_logpmf(k: u64, r: f64, p: f64) -> Result<f64> {
    if !r.is_finite() || r <= 0.0 {
        return Err(Error::Validation(format!("r must be finite and > 0, got {}", r)));
    }
    if !p.is_finite() || p <= 0.0 || p >= 1.0 {
        return Err(Error::Validation(format!("p must be in (0,1), got {}", p)));
    }
    let kf = k as f64;
    let ln_coeff = ln_gamma(kf + r) - ln_gamma(r) - ln_gamma(kf + 1.0);
    Ok(ln_coeff + r * p.ln() + kf * (1.0 - p).ln())
}

/// Log-PDF of Gamma(shape=k, scale=theta) at `x`.
pub fn gamma_logpdf(x: f64, k: f64, theta: f64) -> Result<f64> {
    if !x.is_finite() || x <= 0.0 {
        return Err(Error::Validation(format!("x must be finite and > 0, got {}", x)));
    }
    if !k.is_finite() || k <= 0.0 {
        return Err(Error::Validation(format!("k must be finite and > 0, got {}", k)));
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(Error::Validation(format!("theta must be finite and > 0, got {}", theta)));
    }
    Ok((k - 1.0) * x.ln() - x / theta - k * theta.ln() - ln_gamma(k))
}

/// Log-PDF of LogNormal(mu, sigma) at `x`.
///
/// Defined as: `ln X ~ Normal(mu, sigma)`.
pub fn lognormal_logpdf(x: f64, mu: f64, sigma: f64) -> Result<f64> {
    if !x.is_finite() || x <= 0.0 {
        return Err(Error::Validation(format!("x must be finite and > 0, got {}", x)));
    }
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(Error::Validation(format!("sigma must be finite and > 0, got {}", sigma)));
    }
    let lx = x.ln();
    let z = (lx - mu) / sigma;
    Ok(-0.5 * z * z - sigma.ln() - LN_SQRT_2PI - lx)
}

/// Convert logit `eta` into probability `p`.
pub fn logit_to_prob(eta: f64) -> f64 {
    sigmoid(eta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_normal_standard_at_zero() {
        let lp = normal_logpdf(0.0, 0.0, 1.0).unwrap();
        assert_relative_eq!(lp, -LN_SQRT_2PI, epsilon = 1e-12);
    }

    #[test]
    fn test_student_t_symmetry_about_mu() {
        let lp1 = student_t_logpdf(1.3, 7.0, 0.0, 2.0).unwrap();
        let lp2 = student_t_logpdf(-1.3, 7.0, 0.0, 2.0).unwrap();
        assert_relative_eq!(lp1, lp2, epsilon = 1e-12);
    }

    #[test]
    fn test_bernoulli_logit_limits() {
        // eta -> +inf => y=true gets ~0, y=false gets -inf
        let lp_t = bernoulli_logpmf_logit(true, 100.0).unwrap();
        let lp_f = bernoulli_logpmf_logit(false, 100.0).unwrap();
        assert!(lp_t > -1e-6);
        assert!(lp_f < -50.0);
    }

    #[test]
    fn test_poisson_k0() {
        let lp = poisson_logpmf(0, 2.0).unwrap();
        assert_relative_eq!(lp, -2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_binomial_k0_n0() {
        let lp = binomial_logpmf_logit(0, 0, 0.0).unwrap();
        assert_relative_eq!(lp, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_negbinom_invalid_p() {
        assert!(negbinom_logpmf(3, 2.0, 0.0).is_err());
        assert!(negbinom_logpmf(3, 2.0, 1.0).is_err());
    }
}

