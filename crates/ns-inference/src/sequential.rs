//! Group sequential testing: boundaries and alpha-spending (Phase 9 Cross-Vertical).
//!
//! Implements group sequential designs for clinical trials and A/B testing,
//! allowing data to be examined at pre-planned interim analyses while controlling
//! the overall Type I error rate.
//!
//! ## Methods
//!
//! - **O'Brien–Fleming** (1979): conservative early boundaries that become less
//!   extreme at later looks. Spends very little α early, preserving most power
//!   for the final analysis. Critical values: `c_k = C / √(t_k)`.
//!
//! - **Pocock** (1977): constant critical values across all looks. Spends α
//!   more evenly, allowing easier early stopping. Critical values: `c_k = C`.
//!
//! - **Lan–DeMets alpha-spending** (1983): flexible approach that defines a
//!   spending function α*(t) where t is the information fraction. The
//!   incremental alpha at each look is α*(t_k) - α*(t_{k-1}).
//!   Supports O'Brien–Fleming-like and Pocock-like spending functions, plus
//!   the Hwang–Shih–DeCani power family.
//!
//! ## Vertical applications
//!
//! - **Pharma / clinical trials**: interim analyses with DSMB (Data Safety Monitoring Board)
//! - **A/B testing**: continuous monitoring with valid stopping rules
//! - **Adaptive trials**: flexible designs with pre-specified spending
//!
//! ## References
//!
//! - O'Brien PC, Fleming TR (1979). A multiple testing procedure for clinical trials.
//!   *Biometrics* 35:549–556.
//! - Pocock SJ (1977). Group sequential methods in the design and analysis of
//!   clinical trials. *Biometrika* 64:191–199.
//! - Lan KKG, DeMets DL (1983). Discrete sequential boundaries for clinical trials.
//!   *Biometrika* 70:659–663.

use ns_core::{Error, Result};

// ---------------------------------------------------------------------------
// Boundary types
// ---------------------------------------------------------------------------

/// Type of group sequential boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryType {
    /// O'Brien-Fleming: c_k = C / √(t_k). Conservative early, liberal late.
    OBrienFleming,
    /// Pocock: constant c_k = C. Equal spending across looks.
    Pocock,
}

/// Alpha-spending function family.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpendingFunction {
    /// O'Brien-Fleming-like: α*(t) = 2 - 2·Φ(z_{α/2} / √t).
    OBrienFlemingLike,
    /// Pocock-like: α*(t) = α · ln(1 + (e-1)·t).
    PocockLike,
    /// Hwang-Shih-DeCani power family: α*(t) = α · (1 - e^{-γt}) / (1 - e^{-γ}).
    /// γ = -4 ≈ OF-like, γ = 1 ≈ Pocock-like, γ = 0 → linear spending.
    HwangShihDeCani(f64),
}

// ---------------------------------------------------------------------------
// Group sequential design result
// ---------------------------------------------------------------------------

/// A single analysis (look) in a group sequential design.
#[derive(Debug, Clone)]
pub struct SequentialLook {
    /// Analysis number (1-indexed).
    pub look: usize,
    /// Information fraction at this look (0, 1].
    pub info_fraction: f64,
    /// Nominal (unadjusted) significance level spent at this look.
    pub nominal_alpha: f64,
    /// Cumulative alpha spent up to and including this look.
    pub cumulative_alpha: f64,
    /// Critical value (z-scale, two-sided upper).
    pub critical_value: f64,
}

/// Result of a group sequential design computation.
#[derive(Debug, Clone)]
pub struct SequentialDesign {
    /// Overall significance level.
    pub alpha: f64,
    /// Number of planned looks.
    pub n_looks: usize,
    /// Information fractions at each look.
    pub info_fractions: Vec<f64>,
    /// Details per look.
    pub looks: Vec<SequentialLook>,
    /// Whether the design is two-sided.
    pub two_sided: bool,
}

/// Result of applying a sequential test to observed data.
#[derive(Debug, Clone)]
pub struct SequentialTestResult {
    /// The look number at which the decision was made (1-indexed), or None if not yet stopped.
    pub stopped_at: Option<usize>,
    /// Whether the null hypothesis was rejected at any look.
    pub rejected: bool,
    /// The test statistic at the current look.
    pub z_statistic: f64,
    /// The critical value at the current look.
    pub critical_value: f64,
    /// Adjusted p-value accounting for multiple looks.
    pub adjusted_p_value: f64,
    /// All previous z-statistics (one per completed look).
    pub z_history: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Design computation
// ---------------------------------------------------------------------------

/// Compute group sequential boundaries using the classical approach.
///
/// # Arguments
/// - `n_looks`: number of planned interim + final analyses (must be >= 1).
/// - `alpha`: overall Type I error rate (two-sided).
/// - `boundary_type`: O'Brien-Fleming or Pocock.
/// - `info_fractions`: optional custom information fractions. If `None`, uses
///   equally spaced fractions `[1/K, 2/K, ..., 1]`.
///
/// # Returns
/// A `SequentialDesign` with critical values and nominal alpha at each look.
pub fn group_sequential_design(
    n_looks: usize,
    alpha: f64,
    boundary_type: BoundaryType,
    info_fractions: Option<Vec<f64>>,
) -> Result<SequentialDesign> {
    if n_looks == 0 {
        return Err(Error::Validation("n_looks must be >= 1".to_string()));
    }
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(Error::Validation("alpha must be in (0, 1)".to_string()));
    }

    let fracs = match info_fractions {
        Some(f) => {
            if f.len() != n_looks {
                return Err(Error::Validation(format!(
                    "info_fractions length {} must equal n_looks {}",
                    f.len(),
                    n_looks
                )));
            }
            for (i, &v) in f.iter().enumerate() {
                if !(v > 0.0 && v <= 1.0) {
                    return Err(Error::Validation(format!(
                        "info_fractions[{}] = {} must be in (0, 1]",
                        i, v
                    )));
                }
            }
            for w in f.windows(2) {
                if w[1] <= w[0] {
                    return Err(Error::Validation(
                        "info_fractions must be strictly increasing".to_string(),
                    ));
                }
            }
            if (f[n_looks - 1] - 1.0).abs() > 1e-10 {
                return Err(Error::Validation("last info_fraction must be 1.0".to_string()));
            }
            f
        }
        None => (1..=n_looks).map(|k| k as f64 / n_looks as f64).collect(),
    };

    // Find the constant C such that the overall Type I error = alpha.
    // For both OF and Pocock, we search for C via bisection on the spending.
    let c_star = find_boundary_constant(alpha, &fracs, boundary_type)?;

    let mut looks = Vec::with_capacity(n_looks);
    let mut cum_alpha = 0.0;

    for (k, &t) in fracs.iter().enumerate() {
        let cv = match boundary_type {
            BoundaryType::OBrienFleming => c_star / t.sqrt(),
            BoundaryType::Pocock => c_star,
        };
        // Nominal alpha at this look: P(|Z| > cv) = 2*(1 - Phi(cv)).
        let nom_alpha = 2.0 * normal_sf(cv);
        cum_alpha += nom_alpha;

        looks.push(SequentialLook {
            look: k + 1,
            info_fraction: t,
            nominal_alpha: nom_alpha,
            cumulative_alpha: cum_alpha.min(alpha),
            critical_value: cv,
        });
    }

    Ok(SequentialDesign { alpha, n_looks, info_fractions: fracs, looks, two_sided: true })
}

/// Compute group sequential boundaries using the Lan-DeMets alpha-spending approach.
///
/// # Arguments
/// - `info_fractions`: information fractions at each look (must end at 1.0).
/// - `alpha`: overall Type I error rate (two-sided).
/// - `spending`: the spending function to use.
///
/// # Returns
/// A `SequentialDesign` with critical values derived from incremental spending.
pub fn alpha_spending_design(
    info_fractions: &[f64],
    alpha: f64,
    spending: SpendingFunction,
) -> Result<SequentialDesign> {
    let n_looks = info_fractions.len();
    if n_looks == 0 {
        return Err(Error::Validation("info_fractions must be non-empty".to_string()));
    }
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(Error::Validation("alpha must be in (0, 1)".to_string()));
    }
    for (i, &v) in info_fractions.iter().enumerate() {
        if !(v > 0.0 && v <= 1.0) {
            return Err(Error::Validation(format!(
                "info_fractions[{}] = {} must be in (0, 1]",
                i, v
            )));
        }
    }
    for w in info_fractions.windows(2) {
        if w[1] <= w[0] {
            return Err(Error::Validation(
                "info_fractions must be strictly increasing".to_string(),
            ));
        }
    }
    if (info_fractions[n_looks - 1] - 1.0).abs() > 1e-10 {
        return Err(Error::Validation("last info_fraction must be 1.0".to_string()));
    }

    let mut looks = Vec::with_capacity(n_looks);
    let mut prev_spent = 0.0_f64;

    for (k, &t) in info_fractions.iter().enumerate() {
        let cum_spent = evaluate_spending(spending, t, alpha);
        let increment = (cum_spent - prev_spent).max(0.0);
        // Convert incremental alpha to critical value: z such that P(|Z| > z) = increment.
        // z = Phi^{-1}(1 - increment/2).
        let cv = if increment >= 1.0 {
            0.0
        } else if increment <= 0.0 {
            8.0 // effectively infinite
        } else {
            normal_quantile(1.0 - increment / 2.0)
        };

        looks.push(SequentialLook {
            look: k + 1,
            info_fraction: t,
            nominal_alpha: increment,
            cumulative_alpha: cum_spent.min(alpha),
            critical_value: cv,
        });

        prev_spent = cum_spent;
    }

    Ok(SequentialDesign {
        alpha,
        n_looks,
        info_fractions: info_fractions.to_vec(),
        looks,
        two_sided: true,
    })
}

/// Evaluate the test at a given look against a pre-computed design.
///
/// # Arguments
/// - `design`: the sequential design (from `group_sequential_design` or `alpha_spending_design`).
/// - `z_statistics`: observed z-statistics at each completed look so far.
///
/// # Returns
/// A `SequentialTestResult` indicating whether to stop and reject.
pub fn sequential_test(
    design: &SequentialDesign,
    z_statistics: &[f64],
) -> Result<SequentialTestResult> {
    if z_statistics.is_empty() {
        return Err(Error::Validation("z_statistics must be non-empty".to_string()));
    }
    if z_statistics.len() > design.n_looks {
        return Err(Error::Validation(format!(
            "z_statistics length {} exceeds n_looks {}",
            z_statistics.len(),
            design.n_looks
        )));
    }

    let current_look = z_statistics.len();
    let z_current = z_statistics[current_look - 1];

    // Check all looks up to current for rejection.
    for (k, &z) in z_statistics.iter().enumerate() {
        let cv = design.looks[k].critical_value;
        if z.abs() >= cv {
            // Compute adjusted p-value: smallest alpha at which we would reject.
            let adj_p = compute_adjusted_p_value(design, z_statistics, k);
            return Ok(SequentialTestResult {
                stopped_at: Some(k + 1),
                rejected: true,
                z_statistic: z,
                critical_value: cv,
                adjusted_p_value: adj_p,
                z_history: z_statistics.to_vec(),
            });
        }
    }

    let cv_current = design.looks[current_look - 1].critical_value;
    // Not rejected at any look so far.
    let adj_p = compute_adjusted_p_value(design, z_statistics, current_look - 1);
    Ok(SequentialTestResult {
        stopped_at: None,
        rejected: false,
        z_statistic: z_current,
        critical_value: cv_current,
        adjusted_p_value: adj_p.min(1.0),
        z_history: z_statistics.to_vec(),
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Evaluate a spending function at information fraction t.
fn evaluate_spending(spending: SpendingFunction, t: f64, alpha: f64) -> f64 {
    match spending {
        SpendingFunction::OBrienFlemingLike => {
            // α*(t) = 2 - 2·Φ(z_{α/2} / √t)
            let z_half = normal_quantile(1.0 - alpha / 2.0);
            2.0 * (1.0 - normal_cdf(z_half / t.sqrt()))
        }
        SpendingFunction::PocockLike => {
            // α*(t) = α · ln(1 + (e-1)·t)
            alpha * (1.0 + (std::f64::consts::E - 1.0) * t).ln()
        }
        SpendingFunction::HwangShihDeCani(gamma) => {
            if gamma.abs() < 1e-10 {
                // Linear spending: α*(t) = α·t
                alpha * t
            } else {
                // α*(t) = α · (1 - e^{-γt}) / (1 - e^{-γ})
                alpha * (1.0 - (-gamma * t).exp()) / (1.0 - (-gamma).exp())
            }
        }
    }
}

/// Find the boundary constant C for OF or Pocock design via bisection.
///
/// We search for C such that the sum of incremental alphas across K looks
/// equals the target alpha. This is an approximation using independent
/// increments (Jennison & Turnbull approach).
fn find_boundary_constant(alpha: f64, fracs: &[f64], boundary_type: BoundaryType) -> Result<f64> {
    let target = alpha;

    let overall_alpha = |c: f64| -> f64 {
        let mut total = 0.0_f64;
        for &t in fracs {
            let cv = match boundary_type {
                BoundaryType::OBrienFleming => c / t.sqrt(),
                BoundaryType::Pocock => c,
            };
            total += 2.0 * normal_sf(cv);
        }
        total
    };

    // Bisection: find C in [0.5, 10].
    let mut lo = 0.5_f64;
    let mut hi = 10.0_f64;

    if overall_alpha(lo) < target {
        return Err(Error::Computation(
            "cannot find boundary constant: alpha too large for this design".to_string(),
        ));
    }

    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let a = overall_alpha(mid);
        if (a - target).abs() < 1e-12 {
            return Ok(mid);
        }
        if a > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    Ok(0.5 * (lo + hi))
}

/// Compute an adjusted p-value for the sequential test.
///
/// Simplified approach: the adjusted p-value at look k is the minimum nominal
/// p-value across all looks up to k, adjusted by the number of looks.
fn compute_adjusted_p_value(design: &SequentialDesign, z_stats: &[f64], look_idx: usize) -> f64 {
    let mut min_stage_p = f64::INFINITY;
    for k in 0..=look_idx {
        if k < z_stats.len() {
            let nominal_p = 2.0 * normal_sf(z_stats[k].abs());
            let cum_alpha = design.looks[k].cumulative_alpha;
            // Stagewise ordering: p_adj = cum_alpha if z crosses boundary at this stage.
            // Otherwise, scale the nominal p by the ratio of cumulative to nominal alpha.
            let ratio = if design.looks[k].nominal_alpha > 0.0 {
                cum_alpha / design.looks[k].nominal_alpha
            } else {
                1.0
            };
            let stage_p = (nominal_p * ratio).min(1.0);
            if stage_p < min_stage_p {
                min_stage_p = stage_p;
            }
        }
    }
    min_stage_p.min(1.0).max(0.0)
}

/// Standard normal CDF.
fn normal_cdf(x: f64) -> f64 {
    0.5 * statrs::function::erf::erfc(-x / std::f64::consts::SQRT_2)
}

/// Standard normal survival function P(Z > z).
fn normal_sf(z: f64) -> f64 {
    0.5 * statrs::function::erf::erfc(z / std::f64::consts::SQRT_2)
}

/// Standard normal quantile (inverse CDF).
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    let (sign, pp) = if p < 0.5 { (-1.0, 1.0 - p) } else { (1.0, p) };
    let t = (-2.0 * (1.0 - pp).ln()).sqrt();

    const C0: f64 = 2.515_517;
    const C1: f64 = 0.802_853;
    const C2: f64 = 0.010_328;
    const D1: f64 = 1.432_788;
    const D2: f64 = 0.189_269;
    const D3: f64 = 0.001_308;

    let num = C0 + t * (C1 + t * C2);
    let den = 1.0 + t * (D1 + t * (D2 + t * D3));
    sign * (t - num / den)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obrien_fleming_5_looks() {
        let design = group_sequential_design(5, 0.05, BoundaryType::OBrienFleming, None).unwrap();
        assert_eq!(design.n_looks, 5);
        assert_eq!(design.looks.len(), 5);
        // OF boundaries should decrease over looks.
        for w in design.looks.windows(2) {
            assert!(
                w[0].critical_value > w[1].critical_value,
                "OF boundaries should decrease: {} > {}",
                w[0].critical_value,
                w[1].critical_value
            );
        }
        // First boundary should be very conservative.
        assert!(design.looks[0].critical_value > 3.0);
        // Last boundary should be close to 2.0 (normal z_{0.025}).
        assert!(design.looks[4].critical_value < 2.5);
        assert!(design.looks[4].critical_value > 1.5);
    }

    #[test]
    fn pocock_5_looks() {
        let design = group_sequential_design(5, 0.05, BoundaryType::Pocock, None).unwrap();
        assert_eq!(design.n_looks, 5);
        // Pocock: all critical values should be equal.
        let cv0 = design.looks[0].critical_value;
        for look in &design.looks {
            assert!(
                (look.critical_value - cv0).abs() < 1e-10,
                "Pocock boundaries should be constant"
            );
        }
        // Pocock boundary should be > z_{0.025} = 1.96 (more conservative per look).
        assert!(cv0 > 1.96, "Pocock cv = {}", cv0);
    }

    #[test]
    fn single_look_equals_fixed() {
        // With 1 look, both OF and Pocock should give z_{α/2}.
        let of = group_sequential_design(1, 0.05, BoundaryType::OBrienFleming, None).unwrap();
        let pk = group_sequential_design(1, 0.05, BoundaryType::Pocock, None).unwrap();
        let z_crit = normal_quantile(0.975);
        assert!(
            (of.looks[0].critical_value - z_crit).abs() < 0.1,
            "OF single look: {} vs {}",
            of.looks[0].critical_value,
            z_crit
        );
        assert!(
            (pk.looks[0].critical_value - z_crit).abs() < 0.1,
            "Pocock single look: {} vs {}",
            pk.looks[0].critical_value,
            z_crit
        );
    }

    #[test]
    fn alpha_spending_of_like() {
        let fracs: Vec<f64> = (1..=4).map(|k| k as f64 / 4.0).collect();
        let design =
            alpha_spending_design(&fracs, 0.05, SpendingFunction::OBrienFlemingLike).unwrap();
        assert_eq!(design.n_looks, 4);
        // OF-like spending: very little alpha spent early.
        assert!(
            design.looks[0].nominal_alpha < 0.01,
            "OF-like should spend little early: {}",
            design.looks[0].nominal_alpha
        );
        // Cumulative at the end should be close to alpha.
        let last = &design.looks[3];
        assert!(
            (last.cumulative_alpha - 0.05).abs() < 0.01,
            "cumulative alpha = {}",
            last.cumulative_alpha
        );
    }

    #[test]
    fn alpha_spending_pocock_like() {
        let fracs: Vec<f64> = (1..=4).map(|k| k as f64 / 4.0).collect();
        let design = alpha_spending_design(&fracs, 0.05, SpendingFunction::PocockLike).unwrap();
        assert_eq!(design.n_looks, 4);
        // Pocock-like spends more evenly.
        let first_frac = design.looks[0].nominal_alpha / 0.05;
        assert!(
            first_frac > 0.05,
            "Pocock-like should spend non-trivially early: frac = {}",
            first_frac
        );
    }

    #[test]
    fn alpha_spending_hsd() {
        let fracs: Vec<f64> = (1..=3).map(|k| k as f64 / 3.0).collect();
        // gamma = -4 is OF-like.
        let design_of =
            alpha_spending_design(&fracs, 0.05, SpendingFunction::HwangShihDeCani(-4.0)).unwrap();
        // gamma = 1 is Pocock-like.
        let design_pk =
            alpha_spending_design(&fracs, 0.05, SpendingFunction::HwangShihDeCani(1.0)).unwrap();
        // OF-like should spend less early than Pocock-like.
        assert!(
            design_of.looks[0].nominal_alpha < design_pk.looks[0].nominal_alpha,
            "OF-like early spend {} should be < Pocock-like {}",
            design_of.looks[0].nominal_alpha,
            design_pk.looks[0].nominal_alpha
        );
    }

    #[test]
    fn alpha_spending_linear() {
        let fracs: Vec<f64> = (1..=4).map(|k| k as f64 / 4.0).collect();
        // gamma ≈ 0 is linear spending.
        let design =
            alpha_spending_design(&fracs, 0.05, SpendingFunction::HwangShihDeCani(0.0)).unwrap();
        // Each look should spend approximately alpha/K = 0.0125.
        for look in &design.looks {
            assert!(
                (look.nominal_alpha - 0.0125).abs() < 0.005,
                "linear spending: nominal_alpha = {}",
                look.nominal_alpha
            );
        }
    }

    #[test]
    fn sequential_test_reject_early() {
        let design = group_sequential_design(3, 0.05, BoundaryType::OBrienFleming, None).unwrap();
        // Large z at first look should reject.
        let cv1 = design.looks[0].critical_value;
        let result = sequential_test(&design, &[cv1 + 0.1]).unwrap();
        assert!(result.rejected);
        assert_eq!(result.stopped_at, Some(1));
    }

    #[test]
    fn sequential_test_no_reject() {
        let design = group_sequential_design(3, 0.05, BoundaryType::OBrienFleming, None).unwrap();
        // Small z at all looks should not reject.
        let result = sequential_test(&design, &[0.5, 0.8, 1.0]).unwrap();
        assert!(!result.rejected);
        assert_eq!(result.stopped_at, None);
    }

    #[test]
    fn sequential_test_reject_at_final() {
        let design = group_sequential_design(3, 0.05, BoundaryType::OBrienFleming, None).unwrap();
        let cv3 = design.looks[2].critical_value;
        let result = sequential_test(&design, &[0.5, 0.8, cv3 + 0.1]).unwrap();
        assert!(result.rejected);
        assert_eq!(result.stopped_at, Some(3));
    }

    #[test]
    fn validation_errors() {
        assert!(group_sequential_design(0, 0.05, BoundaryType::OBrienFleming, None).is_err());
        assert!(group_sequential_design(3, 0.0, BoundaryType::OBrienFleming, None).is_err());
        assert!(group_sequential_design(3, 1.0, BoundaryType::OBrienFleming, None).is_err());
        assert!(
            group_sequential_design(3, 0.05, BoundaryType::OBrienFleming, Some(vec![0.5, 0.8]))
                .is_err()
        );
        // Non-increasing fractions.
        assert!(
            group_sequential_design(
                3,
                0.05,
                BoundaryType::OBrienFleming,
                Some(vec![0.5, 0.3, 1.0])
            )
            .is_err()
        );
    }

    #[test]
    fn custom_info_fractions() {
        let fracs = vec![0.25, 0.5, 1.0];
        let design =
            group_sequential_design(3, 0.05, BoundaryType::OBrienFleming, Some(fracs.clone()))
                .unwrap();
        assert_eq!(design.info_fractions, fracs);
        assert_eq!(design.looks.len(), 3);
    }
}
