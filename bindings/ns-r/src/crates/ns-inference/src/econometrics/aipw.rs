//! Doubly-robust AIPW estimator and Rosenbaum sensitivity analysis.
//!
//! Implements the Augmented Inverse Probability Weighting (AIPW) estimator
//! for the Average Treatment Effect (ATE), which is doubly robust: consistent
//! if either the propensity score model or the outcome model is correctly
//! specified.
//!
//! Also provides Rosenbaum bounds for sensitivity analysis of the ATE
//! to unobserved confounding.
//!
//! # References
//!
//! - Robins, Rotnitzky & Zhao (1994), "Estimation of regression coefficients
//!   when some regressors are not always observed."
//! - Rosenbaum (2002), *Observational Studies*, Ch. 4.

use ns_core::{Error, Result};

/// Result of the AIPW estimator.
#[derive(Debug, Clone)]
pub struct AipwResult {
    /// Estimated ATE (Average Treatment Effect).
    pub ate: f64,
    /// Standard error of the ATE (influence-function based).
    pub se: f64,
    /// 95% confidence interval lower bound.
    pub ci_lower: f64,
    /// 95% confidence interval upper bound.
    pub ci_upper: f64,
    /// Number of treated observations.
    pub n_treated: usize,
    /// Number of control observations.
    pub n_control: usize,
    /// Mean propensity score.
    pub mean_propensity: f64,
}

/// AIPW estimator for the Average Treatment Effect.
///
/// The AIPW estimator combines an outcome model (mu) and a propensity score
/// model (e) for double robustness:
///
/// `ATE = (1/n) Σ [ μ₁(Xᵢ) - μ₀(Xᵢ)
///                  + Dᵢ(Yᵢ - μ₁(Xᵢ))/e(Xᵢ)
///                  - (1-Dᵢ)(Yᵢ - μ₀(Xᵢ))/(1-e(Xᵢ)) ]`
///
/// # Arguments
///
/// - `y` — observed outcome (length n).
/// - `treat` — treatment indicator: 1 = treated, 0 = control (length n).
/// - `propensity` — estimated propensity scores P(D=1|X) ∈ (0,1) (length n).
/// - `mu1` — predicted outcome under treatment E[Y(1)|X] (length n).
/// - `mu0` — predicted outcome under control E[Y(0)|X] (length n).
/// - `trim` — propensity score trimming threshold. Observations with
///   `e < trim` or `e > 1-trim` are excluded. Default: 0.01.
pub fn aipw_ate(
    y: &[f64],
    treat: &[u8],
    propensity: &[f64],
    mu1: &[f64],
    mu0: &[f64],
    trim: f64,
) -> Result<AipwResult> {
    let n = y.len();
    if n == 0 {
        return Err(Error::Validation("y must be non-empty".into()));
    }
    if treat.len() != n || propensity.len() != n || mu1.len() != n || mu0.len() != n {
        return Err(Error::Validation("All input arrays must have the same length".into()));
    }
    if trim < 0.0 || trim >= 0.5 {
        return Err(Error::Validation("trim must be in [0, 0.5)".into()));
    }

    // Compute influence function values
    let mut psi_values = Vec::with_capacity(n);
    let mut n_treated = 0usize;
    let mut n_control = 0usize;
    let mut sum_prop = 0.0;
    let mut n_used = 0usize;

    for i in 0..n {
        let d = treat[i];
        if d != 0 && d != 1 {
            return Err(Error::Validation("treat must be 0 or 1".into()));
        }

        let e = propensity[i];
        // Trim
        if e < trim || e > 1.0 - trim {
            continue;
        }

        let d_f = d as f64;
        sum_prop += e;
        n_used += 1;

        if d == 1 {
            n_treated += 1;
        } else {
            n_control += 1;
        }

        // AIPW influence function for observation i:
        // ψᵢ = μ₁(Xᵢ) - μ₀(Xᵢ) + D(Y - μ₁)/e - (1-D)(Y - μ₀)/(1-e)
        let psi = (mu1[i] - mu0[i]) + d_f * (y[i] - mu1[i]) / e
            - (1.0 - d_f) * (y[i] - mu0[i]) / (1.0 - e);

        psi_values.push(psi);
    }

    if n_used == 0 {
        return Err(Error::Computation(
            "No observations left after propensity score trimming".into(),
        ));
    }
    if n_treated == 0 || n_control == 0 {
        return Err(Error::Computation(
            "Need both treated and control observations after trimming".into(),
        ));
    }

    let n_f = n_used as f64;
    let ate: f64 = psi_values.iter().sum::<f64>() / n_f;

    // Standard error via influence function variance
    let var: f64 = psi_values.iter().map(|&psi| (psi - ate).powi(2)).sum::<f64>() / (n_f * n_f);
    let se = var.max(0.0).sqrt();

    let ci_lower = ate - 1.96 * se;
    let ci_upper = ate + 1.96 * se;
    let mean_propensity = sum_prop / n_f;

    Ok(AipwResult { ate, se, ci_lower, ci_upper, n_treated, n_control, mean_propensity })
}

/// Result of Rosenbaum sensitivity analysis.
#[derive(Debug, Clone)]
pub struct RosenbaumResult {
    /// Gamma values tested (sensitivity parameter Γ ≥ 1).
    pub gammas: Vec<f64>,
    /// Upper bound on p-value for each Γ.
    pub p_upper: Vec<f64>,
    /// Lower bound on p-value for each Γ.
    pub p_lower: Vec<f64>,
    /// Critical Γ at which the result becomes insignificant (p > 0.05).
    /// `None` if significant at all tested Γ values.
    pub gamma_critical: Option<f64>,
}

/// Rosenbaum bounds for sensitivity to unobserved confounding.
///
/// Tests how robust the treatment effect is to a hypothetical unobserved
/// confounder that changes the odds of treatment by a factor of Γ.
///
/// # Arguments
///
/// - `y_treated` — outcomes for matched treated units.
/// - `y_control` — outcomes for matched control units (same length, paired).
/// - `gammas` — sensitivity parameter values to test (Γ ≥ 1).
///
/// Uses the Wilcoxon signed-rank test statistic framework.
pub fn rosenbaum_bounds(
    y_treated: &[f64],
    y_control: &[f64],
    gammas: &[f64],
) -> Result<RosenbaumResult> {
    let n = y_treated.len();
    if n == 0 {
        return Err(Error::Validation("Matched pairs must be non-empty".into()));
    }
    if y_control.len() != n {
        return Err(Error::Validation("y_treated and y_control must have the same length".into()));
    }
    if gammas.is_empty() {
        return Err(Error::Validation("gammas must be non-empty".into()));
    }
    for &g in gammas {
        if g < 1.0 {
            return Err(Error::Validation("All gamma values must be >= 1.0".into()));
        }
    }

    // Compute paired differences and their absolute ranks
    let mut diffs: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let d = y_treated[i] - y_control[i];
        if d.abs() > 0.0 {
            diffs.push(d);
        }
    }

    let m = diffs.len();
    if m == 0 {
        return Ok(RosenbaumResult {
            gammas: gammas.to_vec(),
            p_upper: vec![1.0; gammas.len()],
            p_lower: vec![1.0; gammas.len()],
            gamma_critical: Some(1.0),
        });
    }

    // Rank by absolute value
    let mut indices: Vec<usize> = (0..m).collect();
    indices.sort_by(|&a, &b| diffs[a].abs().partial_cmp(&diffs[b].abs()).unwrap());
    let mut ranks = vec![0.0_f64; m];
    for (rank, &idx) in indices.iter().enumerate() {
        ranks[idx] = (rank + 1) as f64;
    }

    // Observed test statistic: sum of ranks for positive differences
    let t_obs: f64 = diffs.iter().zip(&ranks).filter(|(d, _)| **d > 0.0).map(|(_, r)| *r).sum();

    let sum_ranks: f64 = ranks.iter().sum();

    let mut p_upper_vec = Vec::with_capacity(gammas.len());
    let mut p_lower_vec = Vec::with_capacity(gammas.len());
    let mut gamma_critical = None;

    for &gamma in gammas {
        // Under Γ, the probability of a positive difference ranges
        // from 1/(1+Γ) to Γ/(1+Γ)
        let p_plus_upper = gamma / (1.0 + gamma);
        let p_plus_lower = 1.0 / (1.0 + gamma);

        // Expected value and variance of T under each bound
        let e_upper = sum_ranks * p_plus_upper;
        let e_lower = sum_ranks * p_plus_lower;

        // Variance: Σ r_i² * p * (1-p)
        let sum_ranks_sq: f64 = ranks.iter().map(|r| r * r).sum();
        let var_upper = sum_ranks_sq * p_plus_upper * (1.0 - p_plus_upper);
        let var_lower = sum_ranks_sq * p_plus_lower * (1.0 - p_plus_lower);

        // Normal approximation for p-values
        let z_upper = if var_upper > 0.0 { (t_obs - e_upper) / var_upper.sqrt() } else { 0.0 };
        let z_lower = if var_lower > 0.0 { (t_obs - e_lower) / var_lower.sqrt() } else { 0.0 };

        // One-sided p-values using normal approximation
        let p_upper = normal_cdf_complement(z_upper);
        let p_lower = normal_cdf_complement(z_lower);

        p_upper_vec.push(p_upper);
        p_lower_vec.push(p_lower);

        if gamma_critical.is_none() && p_upper > 0.05 {
            gamma_critical = Some(gamma);
        }
    }

    Ok(RosenbaumResult {
        gammas: gammas.to_vec(),
        p_upper: p_upper_vec,
        p_lower: p_lower_vec,
        gamma_critical,
    })
}

/// Standard normal CDF complement: P(Z > z) using rational approximation.
fn normal_cdf_complement(z: f64) -> f64 {
    // Abramowitz & Stegun approximation 26.2.17
    if z > 8.0 {
        return 0.0;
    }
    if z < -8.0 {
        return 1.0;
    }
    let x = z.abs();
    let b1 = 0.319381530;
    let b2 = -0.356563782;
    let b3 = 1.781477937;
    let b4 = -1.821255978;
    let b5 = 1.330274429;
    let p = 0.2316419;
    let t = 1.0 / (1.0 + p * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    let phi = (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf_complement = phi * (b1 * t + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5);

    if z >= 0.0 { cdf_complement } else { 1.0 - cdf_complement }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aipw_no_treatment_effect() {
        // mu1 = mu0 = y for all, propensity = 0.5 → ATE should be ~0
        let n = 100;
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut treat = vec![0u8; n];
        for i in 0..n / 2 {
            treat[i] = 1;
        }
        let propensity = vec![0.5; n];
        let mu1 = y.clone();
        let mu0 = y.clone();

        let res = aipw_ate(&y, &treat, &propensity, &mu1, &mu0, 0.01).unwrap();
        assert!(res.ate.abs() < 1e-10, "ATE={}, expected ~0", res.ate);
        assert_eq!(res.n_treated, 50);
        assert_eq!(res.n_control, 50);
    }

    #[test]
    fn test_aipw_constant_effect() {
        // True ATE = 5. Perfect outcome model: mu1 = mu0 + 5
        let n = 200;
        let mut y = Vec::with_capacity(n);
        let mut treat = Vec::with_capacity(n);
        let mut propensity = Vec::with_capacity(n);
        let mut mu1 = Vec::with_capacity(n);
        let mut mu0 = Vec::with_capacity(n);

        for i in 0..n {
            let d = if i < n / 2 { 1u8 } else { 0u8 };
            let base = (i as f64) * 0.1;
            let yi = base + if d == 1 { 5.0 } else { 0.0 };

            y.push(yi);
            treat.push(d);
            propensity.push(0.5);
            mu1.push(base + 5.0);
            mu0.push(base);
        }

        let res = aipw_ate(&y, &treat, &propensity, &mu1, &mu0, 0.01).unwrap();
        assert!((res.ate - 5.0).abs() < 1e-10, "ATE={}, expected 5.0", res.ate);
    }

    #[test]
    fn test_aipw_validation() {
        assert!(aipw_ate(&[], &[], &[], &[], &[], 0.01).is_err());
        // Invalid trim
        assert!(aipw_ate(&[1.0], &[1], &[0.5], &[1.0], &[1.0], 0.6).is_err());
    }

    #[test]
    fn test_rosenbaum_bounds_basic() {
        // Clear treatment effect: all treated > control
        let y_treated = vec![10.0, 12.0, 15.0, 11.0, 13.0];
        let y_control = vec![5.0, 6.0, 7.0, 5.5, 6.5];
        let gammas = vec![1.0, 1.5, 2.0, 3.0, 5.0];

        let res = rosenbaum_bounds(&y_treated, &y_control, &gammas).unwrap();
        assert_eq!(res.gammas.len(), 5);
        assert_eq!(res.p_upper.len(), 5);
        assert_eq!(res.p_lower.len(), 5);
        // At Γ=1, p should be small (significant)
        assert!(res.p_upper[0] < 0.05 || res.p_lower[0] < 0.05);
    }

    #[test]
    fn test_rosenbaum_validation() {
        assert!(rosenbaum_bounds(&[], &[], &[1.0]).is_err());
        assert!(rosenbaum_bounds(&[1.0], &[2.0], &[0.5]).is_err()); // gamma < 1
    }

    #[test]
    fn test_normal_cdf_complement() {
        // P(Z > 0) ≈ 0.5
        assert!((normal_cdf_complement(0.0) - 0.5).abs() < 0.001);
        // P(Z > 1.96) ≈ 0.025
        assert!((normal_cdf_complement(1.96) - 0.025).abs() < 0.002);
        // P(Z > -1.96) ≈ 0.975
        assert!((normal_cdf_complement(-1.96) - 0.975).abs() < 0.002);
    }
}
