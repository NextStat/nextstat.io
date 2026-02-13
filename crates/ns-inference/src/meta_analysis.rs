//! Meta-analysis: fixed-effects and random-effects pooling (Phase 9 Cross-Vertical).
//!
//! Combines effect sizes from multiple independent studies to produce a single
//! pooled estimate with confidence interval. Used in:
//! - **Pharma**: pooling treatment effects across clinical trials
//! - **Epidemiology**: pooling relative risks, odds ratios
//! - **Social science**: pooling standardized mean differences
//!
//! ## Methods
//!
//! - **Fixed-effects** (inverse-variance): assumes a single true effect.
//!   Weight_i = 1/SE_i². Pooled = Σ(w_i·θ_i) / Σ(w_i).
//!
//! - **Random-effects** (DerSimonian–Laird 1986): assumes effects drawn from
//!   a distribution with between-study variance τ².
//!   Weight_i = 1/(SE_i² + τ²). Estimate of τ² from Cochran's Q.
//!
//! ## Heterogeneity
//!
//! - **Q** (Cochran): weighted sum of squared deviations. Under H₀ (homogeneity), Q ~ χ²(k-1).
//! - **I²** (Higgins & Thompson 2002): percentage of total variability due to heterogeneity.
//!   I² = max(0, (Q - df) / Q) × 100%. Benchmarks: 25% low, 50% moderate, 75% high.
//! - **H²**: ratio of Q to its df. H² = Q / (k-1).
//!
//! ## Forest plot data
//!
//! The results include per-study data suitable for rendering a forest plot
//! (study label, effect, CI, weight).

use ns_core::{Error, Result};

/// Input for one study in a meta-analysis.
#[derive(Debug, Clone)]
pub struct StudyEffect {
    /// Study label (e.g. "Smith 2024").
    pub label: String,
    /// Point estimate (e.g. log-odds ratio, mean difference).
    pub estimate: f64,
    /// Standard error of the estimate (must be > 0).
    pub se: f64,
}

/// A single row in the forest plot output.
#[derive(Debug, Clone)]
pub struct ForestRow {
    /// Study label.
    pub label: String,
    /// Point estimate.
    pub estimate: f64,
    /// Standard error.
    pub se: f64,
    /// Lower bound of the confidence interval.
    pub ci_lower: f64,
    /// Upper bound of the confidence interval.
    pub ci_upper: f64,
    /// Weight assigned to this study (as a fraction of total weight, 0–1).
    pub weight: f64,
}

/// Heterogeneity statistics.
#[derive(Debug, Clone)]
pub struct Heterogeneity {
    /// Cochran's Q statistic.
    pub q: f64,
    /// Degrees of freedom (k - 1).
    pub df: usize,
    /// p-value for Q (chi-squared test).
    pub p_value: f64,
    /// I² percentage (0–100). Proportion of variability due to heterogeneity.
    pub i_squared: f64,
    /// H² = Q / df. Ratio of observed to expected dispersion.
    pub h_squared: f64,
    /// Estimated between-study variance (τ²). Zero for fixed-effects.
    pub tau_squared: f64,
}

/// Result of a meta-analysis.
#[derive(Debug, Clone)]
pub struct MetaAnalysisResult {
    /// Pooled effect estimate.
    pub estimate: f64,
    /// Standard error of the pooled estimate.
    pub se: f64,
    /// Lower bound of the confidence interval.
    pub ci_lower: f64,
    /// Upper bound of the confidence interval.
    pub ci_upper: f64,
    /// Z-statistic for the pooled estimate (estimate / se).
    pub z: f64,
    /// Two-sided p-value for the pooled estimate.
    pub p_value: f64,
    /// Method used: "fixed" or "random".
    pub method: &'static str,
    /// Confidence level (e.g. 0.95).
    pub conf_level: f64,
    /// Number of studies.
    pub k: usize,
    /// Heterogeneity statistics.
    pub heterogeneity: Heterogeneity,
    /// Per-study forest plot data.
    pub forest: Vec<ForestRow>,
}

/// Run a fixed-effects meta-analysis (inverse-variance method).
///
/// All studies are assumed to share the same true effect size.
///
/// # Arguments
/// - `studies`: effect sizes with standard errors.
/// - `conf_level`: confidence level for intervals (e.g. 0.95).
pub fn meta_fixed(studies: &[StudyEffect], conf_level: f64) -> Result<MetaAnalysisResult> {
    validate_inputs(studies, conf_level)?;

    let k = studies.len();
    let z_alpha = normal_quantile((1.0 + conf_level) / 2.0);

    // Inverse-variance weights: w_i = 1 / se_i²
    let weights: Vec<f64> = studies.iter().map(|s| 1.0 / (s.se * s.se)).collect();
    let w_sum: f64 = weights.iter().sum();

    // Pooled estimate
    let estimate: f64 =
        weights.iter().zip(studies.iter()).map(|(w, s)| w * s.estimate).sum::<f64>() / w_sum;

    let se = (1.0 / w_sum).sqrt();
    let z_stat = estimate / se;
    let p_val = 2.0 * (1.0 - normal_cdf(z_stat.abs()));

    // Heterogeneity
    let het = compute_heterogeneity(studies, &weights, estimate);

    // Forest plot data
    let forest = build_forest(studies, &weights, w_sum, conf_level, z_alpha);

    Ok(MetaAnalysisResult {
        estimate,
        se,
        ci_lower: estimate - z_alpha * se,
        ci_upper: estimate + z_alpha * se,
        z: z_stat,
        p_value: p_val,
        method: "fixed",
        conf_level,
        k,
        heterogeneity: het,
        forest,
    })
}

/// Run a random-effects meta-analysis (DerSimonian–Laird method).
///
/// Effects are assumed drawn from a normal distribution with unknown
/// between-study variance τ².
///
/// # Arguments
/// - `studies`: effect sizes with standard errors.
/// - `conf_level`: confidence level for intervals (e.g. 0.95).
pub fn meta_random(studies: &[StudyEffect], conf_level: f64) -> Result<MetaAnalysisResult> {
    validate_inputs(studies, conf_level)?;

    let k = studies.len();
    let z_alpha = normal_quantile((1.0 + conf_level) / 2.0);

    // Step 1: fixed-effects weights and pooled estimate for Q calculation
    let w_fixed: Vec<f64> = studies.iter().map(|s| 1.0 / (s.se * s.se)).collect();
    let w_fixed_sum: f64 = w_fixed.iter().sum();
    let theta_fixed: f64 =
        w_fixed.iter().zip(studies.iter()).map(|(w, s)| w * s.estimate).sum::<f64>() / w_fixed_sum;

    // Step 2: Cochran's Q
    let q: f64 = w_fixed
        .iter()
        .zip(studies.iter())
        .map(|(w, s)| w * (s.estimate - theta_fixed).powi(2))
        .sum();

    // Step 3: DerSimonian-Laird estimate of τ²
    let df = (k - 1) as f64;
    let c = w_fixed_sum - w_fixed.iter().map(|w| w * w).sum::<f64>() / w_fixed_sum;
    let tau_sq = ((q - df) / c).max(0.0);

    // Step 4: Random-effects weights
    let w_random: Vec<f64> = studies.iter().map(|s| 1.0 / (s.se * s.se + tau_sq)).collect();
    let w_random_sum: f64 = w_random.iter().sum();

    // Pooled estimate
    let estimate: f64 =
        w_random.iter().zip(studies.iter()).map(|(w, s)| w * s.estimate).sum::<f64>()
            / w_random_sum;

    let se = (1.0 / w_random_sum).sqrt();
    let z_stat = estimate / se;
    let p_val = 2.0 * (1.0 - normal_cdf(z_stat.abs()));

    // Heterogeneity (using fixed-effects weights for Q, but reporting tau_sq)
    let mut het = compute_heterogeneity(studies, &w_fixed, theta_fixed);
    het.tau_squared = tau_sq;

    // Forest plot data (with random-effects weights)
    let forest = build_forest(studies, &w_random, w_random_sum, conf_level, z_alpha);

    Ok(MetaAnalysisResult {
        estimate,
        se,
        ci_lower: estimate - z_alpha * se,
        ci_upper: estimate + z_alpha * se,
        z: z_stat,
        p_value: p_val,
        method: "random",
        conf_level,
        k,
        heterogeneity: het,
        forest,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn validate_inputs(studies: &[StudyEffect], conf_level: f64) -> Result<()> {
    if studies.len() < 2 {
        return Err(Error::Validation("meta-analysis requires at least 2 studies".to_string()));
    }
    if !(conf_level > 0.0 && conf_level < 1.0) {
        return Err(Error::Validation("conf_level must be in (0, 1)".to_string()));
    }
    for (i, s) in studies.iter().enumerate() {
        if !s.estimate.is_finite() {
            return Err(Error::Validation(format!("study {i}: estimate must be finite")));
        }
        if !s.se.is_finite() || s.se <= 0.0 {
            return Err(Error::Validation(format!("study {i}: se must be positive and finite")));
        }
    }
    Ok(())
}

fn compute_heterogeneity(studies: &[StudyEffect], weights: &[f64], theta: f64) -> Heterogeneity {
    let k = studies.len();
    let df = k - 1;

    let q: f64 =
        weights.iter().zip(studies.iter()).map(|(w, s)| w * (s.estimate - theta).powi(2)).sum();

    let p_value = 1.0 - chi_squared_cdf(q, df as f64);
    let i_squared = if q > df as f64 { ((q - df as f64) / q) * 100.0 } else { 0.0 };
    let h_squared = if df > 0 { q / df as f64 } else { 1.0 };

    Heterogeneity {
        q,
        df,
        p_value,
        i_squared,
        h_squared,
        tau_squared: 0.0, // caller may override for random-effects
    }
}

fn build_forest(
    studies: &[StudyEffect],
    weights: &[f64],
    w_sum: f64,
    _conf_level: f64,
    z_alpha: f64,
) -> Vec<ForestRow> {
    studies
        .iter()
        .zip(weights.iter())
        .map(|(s, &w)| ForestRow {
            label: s.label.clone(),
            estimate: s.estimate,
            se: s.se,
            ci_lower: s.estimate - z_alpha * s.se,
            ci_upper: s.estimate + z_alpha * s.se,
            weight: w / w_sum,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Statistical functions (self-contained to avoid cross-module deps)
// ---------------------------------------------------------------------------

/// Standard normal CDF via erfc.
#[inline]
fn normal_cdf(x: f64) -> f64 {
    0.5 * statrs::function::erf::erfc(-x / std::f64::consts::SQRT_2)
}

/// Standard normal quantile (inverse CDF) via statrs.
fn normal_quantile(p: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};
    Normal::new(0.0, 1.0).unwrap().inverse_cdf(p)
}

/// Chi-squared CDF via regularized lower incomplete gamma.
fn chi_squared_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    statrs::function::gamma::gamma_lr(df / 2.0, x / 2.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn example_studies() -> Vec<StudyEffect> {
        // 5 studies with known effects (log-OR style)
        vec![
            StudyEffect { label: "Study A".into(), estimate: 0.50, se: 0.20 },
            StudyEffect { label: "Study B".into(), estimate: 0.30, se: 0.15 },
            StudyEffect { label: "Study C".into(), estimate: 0.60, se: 0.25 },
            StudyEffect { label: "Study D".into(), estimate: 0.40, se: 0.10 },
            StudyEffect { label: "Study E".into(), estimate: 0.55, se: 0.30 },
        ]
    }

    // ----- Fixed-effects tests -----

    #[test]
    fn fixed_basic_properties() {
        let res = meta_fixed(&example_studies(), 0.95).unwrap();
        assert_eq!(res.method, "fixed");
        assert_eq!(res.k, 5);
        assert_eq!(res.conf_level, 0.95);
        assert!(res.estimate.is_finite());
        assert!(res.se > 0.0);
        assert!(res.ci_lower < res.estimate);
        assert!(res.ci_upper > res.estimate);
        assert!(res.p_value >= 0.0 && res.p_value <= 1.0);
        assert_eq!(res.forest.len(), 5);
    }

    #[test]
    fn fixed_pooled_estimate_is_weighted_mean() {
        let studies = example_studies();
        let res = meta_fixed(&studies, 0.95).unwrap();

        // Manual calculation
        let weights: Vec<f64> = studies.iter().map(|s| 1.0 / (s.se * s.se)).collect();
        let w_sum: f64 = weights.iter().sum();
        let expected: f64 =
            weights.iter().zip(studies.iter()).map(|(w, s)| w * s.estimate).sum::<f64>() / w_sum;

        assert!(
            (res.estimate - expected).abs() < 1e-10,
            "pooled: {} vs expected: {}",
            res.estimate,
            expected
        );
    }

    #[test]
    fn fixed_weights_sum_to_one() {
        let res = meta_fixed(&example_studies(), 0.95).unwrap();
        let sum: f64 = res.forest.iter().map(|r| r.weight).sum();
        assert!((sum - 1.0).abs() < 1e-10, "weights sum = {sum}");
    }

    #[test]
    fn fixed_largest_weight_smallest_se() {
        let res = meta_fixed(&example_studies(), 0.95).unwrap();
        // Study D (se=0.10) should have the largest weight.
        let max_weight_study =
            res.forest.iter().max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap()).unwrap();
        assert_eq!(max_weight_study.label, "Study D");
    }

    #[test]
    fn fixed_heterogeneity() {
        let res = meta_fixed(&example_studies(), 0.95).unwrap();
        let het = &res.heterogeneity;
        assert_eq!(het.df, 4);
        assert!(het.q >= 0.0);
        assert!(het.i_squared >= 0.0 && het.i_squared <= 100.0);
        assert!(het.h_squared >= 0.0);
        assert!(het.p_value >= 0.0 && het.p_value <= 1.0);
        assert_eq!(het.tau_squared, 0.0); // fixed-effects: no tau²
    }

    // ----- Random-effects tests -----

    #[test]
    fn random_basic_properties() {
        let res = meta_random(&example_studies(), 0.95).unwrap();
        assert_eq!(res.method, "random");
        assert_eq!(res.k, 5);
        assert!(res.se > 0.0);
        assert!(res.ci_lower < res.ci_upper);
        assert!(res.heterogeneity.tau_squared >= 0.0);
    }

    #[test]
    fn random_se_at_least_as_large_as_fixed() {
        let studies = example_studies();
        let fixed = meta_fixed(&studies, 0.95).unwrap();
        let random = meta_random(&studies, 0.95).unwrap();
        // Random-effects SE should be ≥ fixed-effects SE (wider CIs).
        assert!(random.se >= fixed.se - 1e-12, "RE SE ({}) < FE SE ({})", random.se, fixed.se);
    }

    #[test]
    fn random_ci_wider_than_fixed() {
        let studies = example_studies();
        let fixed = meta_fixed(&studies, 0.95).unwrap();
        let random = meta_random(&studies, 0.95).unwrap();
        let fixed_width = fixed.ci_upper - fixed.ci_lower;
        let random_width = random.ci_upper - random.ci_lower;
        assert!(
            random_width >= fixed_width - 1e-12,
            "RE CI width ({random_width}) < FE CI width ({fixed_width})"
        );
    }

    #[test]
    fn random_homogeneous_studies_gives_tau_zero() {
        // When all studies have the same estimate, Q ≈ 0, τ² = 0.
        let studies = vec![
            StudyEffect { label: "A".into(), estimate: 0.5, se: 0.1 },
            StudyEffect { label: "B".into(), estimate: 0.5, se: 0.2 },
            StudyEffect { label: "C".into(), estimate: 0.5, se: 0.15 },
        ];
        let res = meta_random(&studies, 0.95).unwrap();
        assert!(
            res.heterogeneity.tau_squared < 1e-10,
            "tau² = {} for homogeneous studies",
            res.heterogeneity.tau_squared
        );
        assert!(res.heterogeneity.i_squared < 1e-5);
    }

    #[test]
    fn random_heterogeneous_studies_gives_positive_tau() {
        // Very different effect sizes → τ² > 0.
        let studies = vec![
            StudyEffect { label: "A".into(), estimate: -1.0, se: 0.1 },
            StudyEffect { label: "B".into(), estimate: 0.0, se: 0.1 },
            StudyEffect { label: "C".into(), estimate: 1.0, se: 0.1 },
            StudyEffect { label: "D".into(), estimate: 2.0, se: 0.1 },
        ];
        let res = meta_random(&studies, 0.95).unwrap();
        assert!(
            res.heterogeneity.tau_squared > 0.1,
            "tau² = {} for heterogeneous studies",
            res.heterogeneity.tau_squared
        );
        assert!(res.heterogeneity.i_squared > 50.0);
    }

    // ----- Validation tests -----

    #[test]
    fn rejects_fewer_than_two_studies() {
        let one = vec![StudyEffect { label: "A".into(), estimate: 0.5, se: 0.1 }];
        assert!(meta_fixed(&one, 0.95).is_err());
        assert!(meta_random(&one, 0.95).is_err());
    }

    #[test]
    fn rejects_invalid_conf_level() {
        let s = example_studies();
        assert!(meta_fixed(&s, 0.0).is_err());
        assert!(meta_fixed(&s, 1.0).is_err());
        assert!(meta_random(&s, -0.1).is_err());
    }

    #[test]
    fn rejects_non_positive_se() {
        let bad = vec![
            StudyEffect { label: "A".into(), estimate: 0.5, se: 0.1 },
            StudyEffect { label: "B".into(), estimate: 0.3, se: 0.0 },
        ];
        assert!(meta_fixed(&bad, 0.95).is_err());
    }

    #[test]
    fn rejects_non_finite_estimate() {
        let bad = vec![
            StudyEffect { label: "A".into(), estimate: f64::NAN, se: 0.1 },
            StudyEffect { label: "B".into(), estimate: 0.3, se: 0.1 },
        ];
        assert!(meta_fixed(&bad, 0.95).is_err());
    }

    // ----- Reference value test (manual computation) -----

    #[test]
    fn fixed_two_studies_manual() {
        // Study 1: θ=1.0, se=0.5 → w=4.0
        // Study 2: θ=2.0, se=1.0 → w=1.0
        // Pooled = (4*1 + 1*2) / (4+1) = 6/5 = 1.2
        // SE = 1/√5 ≈ 0.4472
        let studies = vec![
            StudyEffect { label: "S1".into(), estimate: 1.0, se: 0.5 },
            StudyEffect { label: "S2".into(), estimate: 2.0, se: 1.0 },
        ];
        let res = meta_fixed(&studies, 0.95).unwrap();
        assert!((res.estimate - 1.2).abs() < 1e-10, "pooled = {}", res.estimate);
        assert!((res.se - (1.0_f64 / 5.0).sqrt()).abs() < 1e-10, "se = {}", res.se);
    }

    #[test]
    fn normal_quantile_known_values() {
        assert!((normal_quantile(0.5) - 0.0).abs() < 1e-8);
        assert!((normal_quantile(0.975) - 1.959964).abs() < 1e-4);
        assert!((normal_quantile(0.025) + 1.959964).abs() < 1e-4);
        assert!((normal_quantile(0.995) - 2.5758).abs() < 1e-3);
    }

    #[test]
    fn forest_ci_symmetry() {
        let res = meta_fixed(&example_studies(), 0.95).unwrap();
        for row in &res.forest {
            let half_width = (row.ci_upper - row.ci_lower) / 2.0;
            let center = (row.ci_upper + row.ci_lower) / 2.0;
            assert!((center - row.estimate).abs() < 1e-10, "CI not centered for {}", row.label);
            assert!(
                (half_width - 1.96 * row.se).abs() < 0.01 * row.se,
                "CI half-width not ≈ 1.96·SE for {}",
                row.label
            );
        }
    }
}
