//! Bootstrap confidence interval utilities.
//!
//! This module provides:
//! - percentile intervals,
//! - BCa (bias-corrected and accelerated) intervals,
//! - reusable diagnostics (`z0`, `a`, adjusted alphas).
//!
//! BCa references:
//! - Efron (1987), "Better Bootstrap Confidence Intervals"

use ns_core::{Error, Result};
use statrs::distribution::{ContinuousCDF, Normal};

const PROB_EPS: f64 = 1e-12;

/// Bootstrap CI method selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootstrapCiMethod {
    /// Percentile interval from bootstrap quantiles.
    Percentile,
    /// Bias-corrected and accelerated interval (BCa).
    Bca,
}

/// BCa diagnostics for observability and debugging.
#[derive(Debug, Clone)]
pub struct BcaDiagnostics {
    /// Bias-correction constant.
    pub z0: f64,
    /// Acceleration constant.
    pub acceleration: f64,
    /// Requested lower alpha (`(1-conf_level)/2`).
    pub alpha_low: f64,
    /// Requested upper alpha (`1 - (1-conf_level)/2`).
    pub alpha_high: f64,
    /// BCa-adjusted lower alpha.
    pub alpha_low_adj: f64,
    /// BCa-adjusted upper alpha.
    pub alpha_high_adj: f64,
    /// Number of bootstrap samples used.
    pub n_bootstrap: usize,
    /// Number of jackknife estimates used.
    pub n_jackknife: usize,
}

#[inline]
fn standard_normal() -> Normal {
    // Safe by construction for mean=0, sigma=1.
    Normal::new(0.0, 1.0).expect("standard normal should be constructible")
}

#[inline]
fn clip_prob(p: f64) -> f64 {
    p.clamp(PROB_EPS, 1.0 - PROB_EPS)
}

#[inline]
fn inv_norm_cdf(p: f64) -> f64 {
    standard_normal().inverse_cdf(clip_prob(p))
}

#[inline]
fn norm_cdf(z: f64) -> f64 {
    standard_normal().cdf(z)
}

/// Quantile for sorted data via linear interpolation.
///
/// - `q=0` returns min
/// - `q=1` returns max
/// - empty input returns `NaN`
pub fn quantile_linear_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let q = q.clamp(0.0, 1.0);
    let pos = q * (sorted.len() - 1) as f64;
    let i = pos.floor() as usize;
    let j = pos.ceil() as usize;
    if i == j {
        return sorted[i];
    }
    let t = pos - i as f64;
    (1.0 - t) * sorted[i] + t * sorted[j]
}

/// Quantile via sorting + linear interpolation.
pub fn quantile_linear(data: &[f64], q: f64) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let mut v = data.to_vec();
    v.sort_by(f64::total_cmp);
    quantile_linear_sorted(&v, q)
}

/// Percentile bootstrap interval.
pub fn percentile_interval(samples: &[f64], conf_level: f64) -> Result<(f64, f64)> {
    if samples.len() < 2 {
        return Err(Error::Validation(
            "percentile_interval requires at least 2 samples".to_string(),
        ));
    }
    if !(conf_level.is_finite() && conf_level > 0.0 && conf_level < 1.0) {
        return Err(Error::Validation(format!("conf_level must be in (0,1), got {conf_level}")));
    }

    let alpha = (1.0 - conf_level) / 2.0;
    let lo = quantile_linear(samples, alpha);
    let hi = quantile_linear(samples, 1.0 - alpha);
    Ok((lo.min(hi), lo.max(hi)))
}

/// Estimate BCa bias correction constant `z0` from bootstrap samples.
///
/// Uses mid-rank ties: `p = (n_lt + 0.5*n_eq) / n`.
pub fn estimate_bias_correction_z0(theta_hat: f64, bootstrap_samples: &[f64]) -> Result<f64> {
    if bootstrap_samples.is_empty() {
        return Err(Error::Validation(
            "estimate_bias_correction_z0 requires non-empty bootstrap_samples".to_string(),
        ));
    }
    if !theta_hat.is_finite() {
        return Err(Error::Validation("theta_hat must be finite".to_string()));
    }

    let mut n_lt = 0usize;
    let mut n_eq = 0usize;
    for &x in bootstrap_samples {
        if !x.is_finite() {
            return Err(Error::Validation(
                "bootstrap_samples must contain only finite values".to_string(),
            ));
        }
        if x < theta_hat {
            n_lt += 1;
        } else if x == theta_hat {
            n_eq += 1;
        }
    }

    let p = (n_lt as f64 + 0.5 * n_eq as f64) / bootstrap_samples.len() as f64;
    Ok(inv_norm_cdf(p))
}

/// Estimate BCa acceleration from jackknife leave-one-out estimates.
///
/// Formula:
/// `a = sum((mean_jk - jk_i)^3) / (6 * (sum((mean_jk - jk_i)^2))^(3/2))`
pub fn estimate_acceleration_from_jackknife(jackknife_estimates: &[f64]) -> Result<f64> {
    if jackknife_estimates.len() < 3 {
        return Err(Error::Validation(
            "estimate_acceleration_from_jackknife requires at least 3 jackknife estimates"
                .to_string(),
        ));
    }
    if jackknife_estimates.iter().any(|v| !v.is_finite()) {
        return Err(Error::Validation(
            "jackknife_estimates must contain only finite values".to_string(),
        ));
    }

    let mean = jackknife_estimates.iter().sum::<f64>() / jackknife_estimates.len() as f64;
    let mut sum2 = 0.0;
    let mut sum3 = 0.0;
    for &v in jackknife_estimates {
        let d = mean - v;
        sum2 += d * d;
        sum3 += d * d * d;
    }

    if !(sum2.is_finite() && sum2 > 0.0) {
        // Degenerate case: no variation across jackknife estimates.
        return Ok(0.0);
    }
    let denom = 6.0 * sum2.powf(1.5);
    let a = sum3 / denom;
    if !a.is_finite() {
        return Err(Error::Computation("acceleration estimate is non-finite".to_string()));
    }
    Ok(a)
}

/// BCa-adjusted alpha from raw alpha, `z0`, and acceleration `a`.
pub fn bca_adjusted_alpha(alpha: f64, z0: f64, acceleration: f64) -> f64 {
    let z_alpha = inv_norm_cdf(alpha);
    let denom = 1.0 - acceleration * (z0 + z_alpha);
    if !denom.is_finite() || denom.abs() < 1e-12 {
        // Conservative clipping on numerical boundary.
        return if denom.is_sign_negative() { PROB_EPS } else { 1.0 - PROB_EPS };
    }
    clip_prob(norm_cdf(z0 + (z0 + z_alpha) / denom))
}

/// Compute BCa interval from bootstrap + jackknife estimates.
pub fn bca_interval(
    theta_hat: f64,
    bootstrap_samples: &[f64],
    jackknife_estimates: &[f64],
    conf_level: f64,
) -> Result<((f64, f64), BcaDiagnostics)> {
    if !(conf_level.is_finite() && conf_level > 0.0 && conf_level < 1.0) {
        return Err(Error::Validation(format!("conf_level must be in (0,1), got {conf_level}")));
    }
    if bootstrap_samples.len() < 2 {
        return Err(Error::Validation(
            "bca_interval requires at least 2 bootstrap samples".to_string(),
        ));
    }

    let z0 = estimate_bias_correction_z0(theta_hat, bootstrap_samples)?;
    let acceleration = estimate_acceleration_from_jackknife(jackknife_estimates)?;

    let alpha_low = (1.0 - conf_level) / 2.0;
    let alpha_high = 1.0 - alpha_low;
    let alpha_low_adj = bca_adjusted_alpha(alpha_low, z0, acceleration);
    let alpha_high_adj = bca_adjusted_alpha(alpha_high, z0, acceleration);

    let lo = quantile_linear(bootstrap_samples, alpha_low_adj);
    let hi = quantile_linear(bootstrap_samples, alpha_high_adj);
    let (lower, upper) = (lo.min(hi), lo.max(hi));

    let diag = BcaDiagnostics {
        z0,
        acceleration,
        alpha_low,
        alpha_high,
        alpha_low_adj,
        alpha_high_adj,
        n_bootstrap: bootstrap_samples.len(),
        n_jackknife: jackknife_estimates.len(),
    };

    Ok(((lower, upper), diag))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_linear_sorted_edges() {
        let s = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((quantile_linear_sorted(&s, 0.0) - 1.0).abs() < 1e-12);
        assert!((quantile_linear_sorted(&s, 1.0) - 5.0).abs() < 1e-12);
        assert!((quantile_linear_sorted(&s, 0.5) - 3.0).abs() < 1e-12);
        assert!((quantile_linear_sorted(&s, 0.25) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn percentile_interval_smoke() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (lo, hi) = percentile_interval(&xs, 0.8).unwrap();
        assert!(lo.is_finite() && hi.is_finite());
        assert!(lo < hi);
    }

    #[test]
    fn z0_midrank_tie_is_zero_at_half() {
        let theta_hat = 0.0;
        let bs = [-1.0, 0.0, 1.0, 2.0];
        let z0 = estimate_bias_correction_z0(theta_hat, &bs).unwrap();
        // p = (1 + 0.5*1) / 4 = 0.375 => z0 < 0 (sanity check)
        assert!(z0 < 0.0);
    }

    #[test]
    fn acceleration_zero_for_degenerate_jackknife() {
        let jk = [2.0, 2.0, 2.0, 2.0];
        let a = estimate_acceleration_from_jackknife(&jk).unwrap();
        assert!(a.abs() < 1e-15);
    }

    #[test]
    fn bca_adjusted_alpha_reduces_to_identity_when_z0_a_zero() {
        let alphas = [0.025, 0.1, 0.5, 0.9, 0.975];
        for &a in &alphas {
            let adj = bca_adjusted_alpha(a, 0.0, 0.0);
            assert!((adj - a).abs() < 1e-10, "alpha={a} adj={adj}");
        }
    }

    #[test]
    fn bca_interval_smoke() {
        // Slightly right-skewed bootstrap sample.
        let bootstrap = [0.2, 0.3, 0.31, 0.35, 0.4, 0.45, 0.6, 0.9, 1.1, 1.2];
        // Leave-one-out estimates for acceleration.
        let jackknife = [0.41, 0.43, 0.39, 0.42, 0.40, 0.44, 0.38, 0.41, 0.43, 0.40];
        let theta_hat = 0.4;

        let ((lo, hi), diag) = bca_interval(theta_hat, &bootstrap, &jackknife, 0.95).unwrap();
        assert!(lo.is_finite() && hi.is_finite());
        assert!(lo < hi);
        assert!(diag.z0.is_finite());
        assert!(diag.acceleration.is_finite());
    }
}
