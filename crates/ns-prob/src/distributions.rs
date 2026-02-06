//! Scalar log-density helpers for common distributions.
//!
//! Design goal: tiny building blocks for model composition.
//!
//! This module intentionally provides "one-liner" wrappers in terms of the
//! canonical per-distribution modules in this crate.

use ns_core::Result;

use crate::math::sigmoid;

/// Log-PDF of Normal `N(mu, sigma)` at `x`.
pub fn normal_logpdf(x: f64, mu: f64, sigma: f64) -> Result<f64> {
    crate::normal::logpdf(x, mu, sigma)
}

/// Log-PDF of Student-t with df `nu`, location `mu`, scale `sigma`.
pub fn student_t_logpdf(x: f64, nu: f64, mu: f64, sigma: f64) -> Result<f64> {
    crate::student_t::logpdf(x, mu, sigma, nu)
}

/// Log-PMF of Bernoulli(y) with logit parameter `eta = log(p/(1-p))`.
pub fn bernoulli_logpmf_logit(y: bool, eta: f64) -> Result<f64> {
    crate::bernoulli::logpmf_logit(if y { 1 } else { 0 }, eta)
}

/// Log-PMF of Binomial(n, k) with logit parameter `eta`.
pub fn binomial_logpmf_logit(k: u64, n: u64, eta: f64) -> Result<f64> {
    crate::binomial::logpmf_logit(k, n, eta)
}

/// Log-PMF of Poisson(k | lambda).
pub fn poisson_logpmf(k: u64, lambda: f64) -> Result<f64> {
    crate::poisson::logpmf(k, lambda)
}

/// Log-PMF of Negative Binomial with parameters `r` (shape) and `p` (success prob).
///
/// This is the distribution of the number of failures `k` before `r` successes:
/// `P(k) = C(k+r-1, k) * (1-p)^k * p^r`.
pub fn negbinom_logpmf(k: u64, r: f64, p: f64) -> Result<f64> {
    crate::neg_binomial::logpmf_r_p(k, r, p)
}

/// Log-PDF of Gamma(shape=k, scale=theta) at `x`.
pub fn gamma_logpdf(x: f64, k: f64, theta: f64) -> Result<f64> {
    crate::gamma::logpdf_shape_scale(x, k, theta)
}

/// Log-PDF of LogNormal(mu, sigma) at `x`.
///
/// Defined as: `ln X ~ Normal(mu, sigma)`.
pub fn lognormal_logpdf(x: f64, mu: f64, sigma: f64) -> Result<f64> {
    if !x.is_finite() || x <= 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    let lx = x.ln();
    let lp = crate::normal::logpdf(lx, mu, sigma)?;
    Ok(lp - lx)
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
        let lp2 = crate::normal::logpdf(0.0, 0.0, 1.0).unwrap();
        assert_relative_eq!(lp, lp2, epsilon = 1e-12);
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
