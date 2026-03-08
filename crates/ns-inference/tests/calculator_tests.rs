//! Integration tests for the sample-size calculator.

use ns_inference::calculator::{
    CalculatorConfig, SampleSizeResult, calculate_mde_curve, calculate_power_curve,
    calculate_sample_size, calculate_sensitivity_breakdown, calculate_sequential_schedule,
};

/// Helper: build a default two-sided config.
fn base_config() -> CalculatorConfig {
    CalculatorConfig {
        baseline_rate: 0.02,
        mde_relative: 0.10,
        alpha: 0.05,
        power: 0.80,
        two_sided: true,
        num_looks: 1,
        spending_function: "none".to_string(),
        overdispersion_rho: 0.0,
        measurement_sigma: 0.0,
        cuped_rho_squared: 0.0,
        expected_r_squared_from_covariates: 0.0,
        expected_num_covariates: 0,
        selected_covariates: Vec::new(),
        pre_treatment_covariates_only: true,
        delay_lambda: None,
        delay_window_days: None,
        daily_traffic_per_arm: None,
    }
}

// ---------------------------------------------------------------------------
// Fixed-horizon baseline tests
// ---------------------------------------------------------------------------

#[test]
fn fixed_horizon_low_baseline() {
    // 2% baseline, 10% relative MDE (absolute delta = 0.002) → ~80K per arm.
    // With p1=0.02, p2=0.022 the absolute effect is very small, so ~80K is correct.
    let cfg = base_config();
    let res = calculate_sample_size(&cfg).unwrap();

    assert!(
        res.n_per_arm >= 78_000 && res.n_per_arm <= 83_000,
        "expected ~80K per arm, got {}",
        res.n_per_arm,
    );
    assert_eq!(res.n_total, res.n_per_arm * 2);
    assert_eq!(res.n_fixed, res.n_per_arm, "no adjustments -> n_fixed == n_per_arm");
    assert!((res.inflation_sequential - 1.0).abs() < 1e-10);
    assert!((res.inflation_overdispersion - 1.0).abs() < 1e-10);
    assert!((res.reduction_cuped - 1.0).abs() < 1e-10);
    assert_eq!(res.variance_reduction_method, ns_inference::VarianceReductionMethod::None);
    assert!((res.delay_loss_fraction - 0.0).abs() < 1e-10);
}

#[test]
fn fixed_horizon_high_baseline() {
    // 10% baseline, 20% relative MDE → ~4K per arm.
    let mut cfg = base_config();
    cfg.baseline_rate = 0.10;
    cfg.mde_relative = 0.20;
    let res = calculate_sample_size(&cfg).unwrap();

    assert!(
        res.n_per_arm >= 3_000 && res.n_per_arm <= 5_000,
        "expected ~4K per arm, got {}",
        res.n_per_arm,
    );
}

// ---------------------------------------------------------------------------
// One-sided vs two-sided
// ---------------------------------------------------------------------------

#[test]
fn one_sided_needs_fewer_samples() {
    let two_sided = calculate_sample_size(&base_config()).unwrap();

    let mut cfg = base_config();
    cfg.two_sided = false;
    let one_sided = calculate_sample_size(&cfg).unwrap();

    assert!(
        one_sided.n_per_arm < two_sided.n_per_arm,
        "one-sided ({}) should need fewer samples than two-sided ({})",
        one_sided.n_per_arm,
        two_sided.n_per_arm,
    );
}

// ---------------------------------------------------------------------------
// Sequential inflation
// ---------------------------------------------------------------------------

#[test]
fn sequential_inflation_increases_n() {
    let fixed = calculate_sample_size(&base_config()).unwrap();

    let mut cfg = base_config();
    cfg.num_looks = 3;
    cfg.spending_function = "obrien_fleming".to_string();
    let seq = calculate_sample_size(&cfg).unwrap();

    assert!(
        seq.inflation_sequential > 1.0,
        "sequential inflation should be > 1.0, got {}",
        seq.inflation_sequential,
    );
    assert!(
        seq.n_per_arm > fixed.n_per_arm,
        "sequential n ({}) should exceed fixed n ({})",
        seq.n_per_arm,
        fixed.n_per_arm,
    );
}

#[test]
fn sequential_pocock_inflates_more_than_obf() {
    let mut cfg_obf = base_config();
    cfg_obf.num_looks = 5;
    cfg_obf.spending_function = "obrien_fleming".to_string();
    let obf = calculate_sample_size(&cfg_obf).unwrap();

    let mut cfg_pocock = base_config();
    cfg_pocock.num_looks = 5;
    cfg_pocock.spending_function = "pocock".to_string();
    let pocock = calculate_sample_size(&cfg_pocock).unwrap();

    assert!(
        pocock.inflation_sequential > obf.inflation_sequential,
        "Pocock ({:.4}) should inflate more than OBF ({:.4})",
        pocock.inflation_sequential,
        obf.inflation_sequential,
    );
}

#[test]
fn single_look_no_inflation() {
    let mut cfg = base_config();
    cfg.num_looks = 1;
    let res = calculate_sample_size(&cfg).unwrap();
    assert!(
        (res.inflation_sequential - 1.0).abs() < 1e-10,
        "1 look should produce inflation=1.0, got {}",
        res.inflation_sequential,
    );
}

// ---------------------------------------------------------------------------
// Overdispersion
// ---------------------------------------------------------------------------

#[test]
fn overdispersion_increases_n() {
    let fixed = calculate_sample_size(&base_config()).unwrap();

    let mut cfg = base_config();
    cfg.overdispersion_rho = 0.02;
    let res = calculate_sample_size(&cfg).unwrap();

    let expected_vif = 1.0 / (1.0 - 0.02);
    assert!((res.inflation_overdispersion - expected_vif).abs() < 1e-6);
    assert!(
        res.n_per_arm > fixed.n_per_arm,
        "overdispersion n ({}) should exceed fixed n ({})",
        res.n_per_arm,
        fixed.n_per_arm,
    );
}

#[test]
fn overdispersion_zero_is_noop() {
    let mut cfg = base_config();
    cfg.overdispersion_rho = 0.0;
    let res = calculate_sample_size(&cfg).unwrap();
    assert!((res.inflation_overdispersion - 1.0).abs() < 1e-10);
}

#[test]
fn overdispersion_invalid_rho() {
    let mut cfg = base_config();
    cfg.overdispersion_rho = 1.0;
    assert!(calculate_sample_size(&cfg).is_err());

    cfg.overdispersion_rho = -0.1;
    assert!(calculate_sample_size(&cfg).is_err());
}

// ---------------------------------------------------------------------------
// CUPED variance reduction
// ---------------------------------------------------------------------------

#[test]
fn cuped_reduces_n() {
    let fixed = calculate_sample_size(&base_config()).unwrap();

    let mut cfg = base_config();
    cfg.cuped_rho_squared = 0.30;
    let res = calculate_sample_size(&cfg).unwrap();

    assert!((res.reduction_cuped - 0.70).abs() < 1e-10);
    assert!(
        res.n_per_arm < fixed.n_per_arm,
        "CUPED n ({}) should be less than fixed n ({})",
        res.n_per_arm,
        fixed.n_per_arm,
    );
    assert_eq!(res.variance_reduction_method, ns_inference::VarianceReductionMethod::Cuped);
}

#[test]
fn cure_expected_r_squared_reduces_n() {
    let fixed = calculate_sample_size(&base_config()).unwrap();

    let mut cfg = base_config();
    cfg.expected_r_squared_from_covariates = 0.45;
    cfg.expected_num_covariates = 3;
    cfg.selected_covariates =
        vec!["pre_clicks".to_string(), "pre_spend".to_string(), "pre_geo_mix".to_string()];
    let res = calculate_sample_size(&cfg).unwrap();

    assert!((res.variance_reduction_factor - 0.55).abs() < 1e-10);
    assert_eq!(res.reduction_cuped, res.variance_reduction_factor);
    assert_eq!(res.expected_r_squared_from_covariates, 0.45);
    assert_eq!(res.num_covariates, 3);
    assert_eq!(res.selected_covariates.len(), 3);
    assert_eq!(res.variance_reduction_method, ns_inference::VarianceReductionMethod::Cure);
    assert!(res.n_per_arm < fixed.n_per_arm);
}

#[test]
fn cuped_zero_is_noop() {
    let mut cfg = base_config();
    cfg.cuped_rho_squared = 0.0;
    let res = calculate_sample_size(&cfg).unwrap();
    assert!((res.reduction_cuped - 1.0).abs() < 1e-10);
}

#[test]
fn cuped_invalid_values() {
    let mut cfg = base_config();
    cfg.cuped_rho_squared = 1.0;
    assert!(calculate_sample_size(&cfg).is_err());

    cfg.cuped_rho_squared = -0.1;
    assert!(calculate_sample_size(&cfg).is_err());
}

#[test]
fn conflicting_variance_reduction_inputs_error() {
    let mut cfg = base_config();
    cfg.cuped_rho_squared = 0.20;
    cfg.expected_r_squared_from_covariates = 0.30;
    assert!(calculate_sample_size(&cfg).is_err());
}

#[test]
fn variance_reduction_requires_pre_treatment_flag() {
    let mut cfg = base_config();
    cfg.expected_r_squared_from_covariates = 0.25;
    cfg.pre_treatment_covariates_only = false;
    assert!(calculate_sample_size(&cfg).is_err());
}

// ---------------------------------------------------------------------------
// Delay correction
// ---------------------------------------------------------------------------

#[test]
fn delay_correction_increases_n() {
    let fixed = calculate_sample_size(&base_config()).unwrap();

    let mut cfg = base_config();
    cfg.delay_lambda = Some(0.1);
    cfg.delay_window_days = Some(7.0);
    let res = calculate_sample_size(&cfg).unwrap();

    assert!(res.delay_loss_fraction > 0.0, "delay loss fraction should be > 0");
    assert!(res.delay_loss_fraction < 1.0, "delay loss fraction should be < 1");
    assert!(
        res.n_per_arm > fixed.n_per_arm,
        "delay-corrected n ({}) should exceed fixed n ({})",
        res.n_per_arm,
        fixed.n_per_arm,
    );
}

#[test]
fn delay_loss_fraction_semantics() {
    // delay_loss_fraction should be 1 - observed_fraction = exp(-lambda * window)
    let mut cfg = base_config();
    cfg.delay_lambda = Some(0.1);
    cfg.delay_window_days = Some(7.0);
    let res = calculate_sample_size(&cfg).unwrap();

    let expected_loss = (-0.1_f64 * 7.0).exp(); // exp(-0.7)
    assert!(
        (res.delay_loss_fraction - expected_loss).abs() < 1e-10,
        "delay_loss_fraction should be exp(-lambda*window) = {:.6}, got {:.6}",
        expected_loss,
        res.delay_loss_fraction,
    );
}

#[test]
fn delay_partial_config_error() {
    let mut cfg = base_config();
    cfg.delay_lambda = Some(0.1);
    // delay_window_days not set
    assert!(calculate_sample_size(&cfg).is_err());
}

// ---------------------------------------------------------------------------
// Measurement systematics
// ---------------------------------------------------------------------------

#[test]
fn systematics_increases_n() {
    let fixed = calculate_sample_size(&base_config()).unwrap();

    let mut cfg = base_config();
    cfg.measurement_sigma = 0.005;
    let res = calculate_sample_size(&cfg).unwrap();

    assert!(res.inflation_systematics >= 1.0);
    assert!(
        res.n_per_arm >= fixed.n_per_arm,
        "systematics n ({}) should be >= fixed n ({})",
        res.n_per_arm,
        fixed.n_per_arm,
    );
}

// ---------------------------------------------------------------------------
// Combined adjustments
// ---------------------------------------------------------------------------

#[test]
fn combined_adjustments_are_multiplicative() {
    let mut cfg = base_config();
    cfg.num_looks = 3;
    cfg.spending_function = "obrien_fleming".to_string();
    cfg.overdispersion_rho = 0.02;
    cfg.expected_r_squared_from_covariates = 0.20;
    cfg.expected_num_covariates = 2;
    let res = calculate_sample_size(&cfg).unwrap();

    // All factors should be populated.
    assert!(res.inflation_sequential > 1.0);
    assert!(res.inflation_overdispersion > 1.0);
    assert!(res.reduction_cuped < 1.0);
    assert!((res.delay_loss_fraction - 0.0).abs() < 1e-10);

    // The adjusted n should be approximately n_fixed * product of factors.
    let product = res.inflation_sequential * res.inflation_overdispersion * res.reduction_cuped;
    let expected = (res.n_fixed as f64 * product).ceil() as u64;
    // Allow 1 unit difference due to ceiling rounding at different stages.
    assert!(
        (res.n_per_arm as i64 - expected as i64).unsigned_abs() <= 1,
        "combined n ({}) ≈ n_fixed * factors ({})",
        res.n_per_arm,
        expected,
    );
}

// ---------------------------------------------------------------------------
// Daily traffic → duration_days
// ---------------------------------------------------------------------------

#[test]
fn daily_traffic_produces_duration_days() {
    let mut cfg = base_config();
    cfg.daily_traffic_per_arm = Some(1000.0);
    let res = calculate_sample_size(&cfg).unwrap();

    assert!(
        res.duration_days.is_some(),
        "duration_days should be Some when daily_traffic_per_arm is set"
    );
    let duration = res.duration_days.unwrap();
    let expected = res.n_per_arm as f64 / 1000.0;
    assert!(
        (duration - expected).abs() < 1e-6,
        "duration_days ({:.2}) should be n_per_arm / daily_traffic ({:.2})",
        duration,
        expected,
    );
}

#[test]
fn no_daily_traffic_no_duration() {
    let cfg = base_config();
    let res = calculate_sample_size(&cfg).unwrap();
    assert!(
        res.duration_days.is_none(),
        "duration_days should be None when daily_traffic_per_arm is not set"
    );
}

#[test]
fn zero_daily_traffic_no_duration() {
    let mut cfg = base_config();
    cfg.daily_traffic_per_arm = Some(0.0);
    let res = calculate_sample_size(&cfg).unwrap();
    assert!(
        res.duration_days.is_none(),
        "duration_days should be None when daily_traffic_per_arm is 0"
    );
}

// ---------------------------------------------------------------------------
// Assumptions
// ---------------------------------------------------------------------------

#[test]
fn assumptions_nonempty_with_adjustments() {
    let mut cfg = base_config();
    cfg.overdispersion_rho = 0.02;
    cfg.expected_r_squared_from_covariates = 0.30;
    cfg.expected_num_covariates = 2;
    cfg.selected_covariates = vec!["pre_clicks".to_string(), "pre_spend".to_string()];
    let res = calculate_sample_size(&cfg).unwrap();

    // Should contain at least the test type + overdispersion + CURE
    assert!(
        res.assumptions.len() >= 3,
        "expected at least 3 assumptions, got {}: {:?}",
        res.assumptions.len(),
        res.assumptions,
    );
    let joined = res.assumptions.join(" ");
    assert!(joined.contains("overdispersion"), "assumptions should mention overdispersion");
    assert!(joined.contains("CURE"), "assumptions should mention CURE");
}

#[test]
fn assumptions_base_only_has_test_type() {
    let cfg = base_config();
    let res = calculate_sample_size(&cfg).unwrap();
    assert!(!res.assumptions.is_empty(), "assumptions should at least contain test type",);
    assert!(
        res.assumptions[0].contains("Two-sided"),
        "first assumption should mention two-sided test",
    );
}

// ---------------------------------------------------------------------------
// Mode
// ---------------------------------------------------------------------------

#[test]
fn mode_quick_for_base() {
    let cfg = base_config();
    let res = calculate_sample_size(&cfg).unwrap();
    assert_eq!(res.mode, "quick");
}

#[test]
fn mode_sequential_for_multi_look() {
    let mut cfg = base_config();
    cfg.num_looks = 3;
    cfg.spending_function = "obrien_fleming".to_string();
    let res = calculate_sample_size(&cfg).unwrap();
    assert_eq!(res.mode, "sequential");
}

#[test]
fn mode_real_world_for_adjustments() {
    let mut cfg = base_config();
    cfg.overdispersion_rho = 0.02;
    let res = calculate_sample_size(&cfg).unwrap();
    assert_eq!(res.mode, "real_world");
}

#[test]
fn mode_real_world_overrides_sequential() {
    // When both sequential and real-world adjustments are active, mode should be "real_world".
    let mut cfg = base_config();
    cfg.num_looks = 3;
    cfg.spending_function = "obrien_fleming".to_string();
    cfg.overdispersion_rho = 0.02;
    let res = calculate_sample_size(&cfg).unwrap();
    assert_eq!(res.mode, "real_world");
}

// ---------------------------------------------------------------------------
// Serde roundtrip
// ---------------------------------------------------------------------------

#[test]
fn serde_roundtrip_config() {
    let cfg = base_config();
    let json = serde_json::to_string_pretty(&cfg).unwrap();
    let cfg2: CalculatorConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(cfg, cfg2);
}

#[test]
fn serde_roundtrip_result() {
    let res = calculate_sample_size(&base_config()).unwrap();
    let json = serde_json::to_string_pretty(&res).unwrap();
    let res2: SampleSizeResult = serde_json::from_str(&json).unwrap();
    // Integer fields must match exactly; floats may have representation noise.
    assert_eq!(res.n_per_arm, res2.n_per_arm);
    assert_eq!(res.n_total, res2.n_total);
    assert_eq!(res.n_fixed, res2.n_fixed);
    assert!((res.inflation_sequential - res2.inflation_sequential).abs() < 1e-12);
}

#[test]
fn serde_defaults_work() {
    // Only required fields; all others should get defaults.
    let json = r#"{"baseline_rate": 0.05, "mde_relative": 0.15}"#;
    let cfg: CalculatorConfig = serde_json::from_str(json).unwrap();
    assert!((cfg.alpha - 0.05).abs() < 1e-10);
    assert!((cfg.power - 0.80).abs() < 1e-10);
    assert!(cfg.two_sided);
    assert_eq!(cfg.num_looks, 1);
    assert_eq!(cfg.spending_function, "none");
    assert!((cfg.overdispersion_rho - 0.0).abs() < 1e-15);
    assert!((cfg.measurement_sigma - 0.0).abs() < 1e-15);
    assert!((cfg.cuped_rho_squared - 0.0).abs() < 1e-15);
    assert!((cfg.expected_r_squared_from_covariates - 0.0).abs() < 1e-15);
    assert_eq!(cfg.expected_num_covariates, 0);
    assert!(cfg.selected_covariates.is_empty());
    assert!(cfg.pre_treatment_covariates_only);
}

// ---------------------------------------------------------------------------
// Power curve tests
// ---------------------------------------------------------------------------

#[test]
fn test_power_curve_monotonic() {
    let cfg = base_config();
    let points = calculate_power_curve(&cfg, 50).unwrap();

    for i in 1..points.len() {
        assert!(
            points[i].power >= points[i - 1].power - 1e-10,
            "power must increase monotonically with n: point[{}].power={:.6} < point[{}].power={:.6}",
            i,
            points[i].power,
            i - 1,
            points[i - 1].power,
        );
    }
}

#[test]
fn test_power_curve_reaches_target() {
    let cfg = base_config();
    let result = calculate_sample_size(&cfg).unwrap();
    let n_target = result.n_per_arm as i64;

    let points = calculate_power_curve(&cfg, 100).unwrap();

    // Find the point closest to n_target.
    let closest = points.iter().min_by_key(|p| (p.n_per_arm - n_target).abs()).unwrap();

    assert!(
        (closest.power - 0.80).abs() < 0.05,
        "at n_target={}, power should be ~0.80, got {:.4} (n_per_arm={})",
        n_target,
        closest.power,
        closest.n_per_arm,
    );
}

#[test]
fn test_power_curve_length() {
    let cfg = base_config();
    for num_points in [1, 10, 25, 100] {
        let points = calculate_power_curve(&cfg, num_points).unwrap();
        assert_eq!(
            points.len(),
            num_points,
            "expected {} points, got {}",
            num_points,
            points.len(),
        );
    }
}

// ---------------------------------------------------------------------------
// MDE curve tests
// ---------------------------------------------------------------------------

#[test]
fn test_mde_curve_monotonic_decreasing() {
    let cfg = base_config();
    let points = calculate_mde_curve(&cfg, 50).unwrap();

    for i in 1..points.len() {
        assert!(
            points[i].mde_relative <= points[i - 1].mde_relative + 1e-10,
            "MDE must decrease as n increases: point[{}].mde={:.6} > point[{}].mde={:.6}",
            i,
            points[i].mde_relative,
            i - 1,
            points[i - 1].mde_relative,
        );
    }
}

#[test]
fn test_mde_curve_length() {
    let cfg = base_config();
    for num_points in [1, 10, 25, 100] {
        let points = calculate_mde_curve(&cfg, num_points).unwrap();
        assert_eq!(
            points.len(),
            num_points,
            "expected {} points, got {}",
            num_points,
            points.len(),
        );
    }
}

// ---------------------------------------------------------------------------
// Sequential schedule tests
// ---------------------------------------------------------------------------

#[test]
fn test_sequential_schedule_obf_3_looks() {
    let mut cfg = base_config();
    cfg.num_looks = 3;
    cfg.spending_function = "obrien_fleming".to_string();

    let schedule = calculate_sequential_schedule(&cfg).unwrap();

    assert_eq!(schedule.looks.len(), 3, "should have 3 looks");

    // OBF boundaries decrease (early looks are conservative, later are liberal).
    for i in 1..schedule.looks.len() {
        assert!(
            schedule.looks[i].critical_value < schedule.looks[i - 1].critical_value,
            "OBF critical values should decrease: look[{}]={:.4} >= look[{}]={:.4}",
            i,
            schedule.looks[i].critical_value,
            i - 1,
            schedule.looks[i - 1].critical_value,
        );
    }

    // Info fractions should be increasing.
    for i in 1..schedule.looks.len() {
        assert!(
            schedule.looks[i].info_fraction > schedule.looks[i - 1].info_fraction,
            "info fractions should increase",
        );
    }

    // n_per_arm_at_look should increase with info_fraction.
    for i in 1..schedule.looks.len() {
        assert!(
            schedule.looks[i].n_per_arm_at_look >= schedule.looks[i - 1].n_per_arm_at_look,
            "n_per_arm_at_look should increase: look[{}]={} < look[{}]={}",
            i,
            schedule.looks[i].n_per_arm_at_look,
            i - 1,
            schedule.looks[i - 1].n_per_arm_at_look,
        );
    }

    // Last look should have info_fraction = 1.0.
    let last = schedule.looks.last().unwrap();
    assert!(
        (last.info_fraction - 1.0).abs() < 1e-10,
        "last info_fraction should be 1.0, got {}",
        last.info_fraction,
    );
}

#[test]
fn test_sequential_schedule_requires_2_plus_looks() {
    let mut cfg = base_config();
    cfg.num_looks = 1;
    let result = calculate_sequential_schedule(&cfg);
    assert!(result.is_err(), "sequential schedule with num_looks=1 should return an error",);
}

#[test]
fn test_sequential_comparison_fixed_lt_obf_lt_pocock() {
    let mut cfg = base_config();
    cfg.num_looks = 5;
    cfg.spending_function = "obrien_fleming".to_string();

    let schedule = calculate_sequential_schedule(&cfg).unwrap();

    assert!(
        schedule.comparison.fixed_n < schedule.comparison.obf_n,
        "fixed_n ({}) should be < obf_n ({})",
        schedule.comparison.fixed_n,
        schedule.comparison.obf_n,
    );
    assert!(
        schedule.comparison.obf_n < schedule.comparison.pocock_n,
        "obf_n ({}) should be < pocock_n ({})",
        schedule.comparison.obf_n,
        schedule.comparison.pocock_n,
    );
}

// ---------------------------------------------------------------------------
// Sensitivity breakdown tests
// ---------------------------------------------------------------------------

#[test]
fn test_sensitivity_breakdown_sums_to_one() {
    let mut cfg = base_config();
    cfg.num_looks = 3;
    cfg.spending_function = "obrien_fleming".to_string();
    cfg.overdispersion_rho = 0.02;
    cfg.measurement_sigma = 0.005;
    cfg.expected_r_squared_from_covariates = 0.10;
    cfg.expected_num_covariates = 2;
    cfg.delay_lambda = Some(0.1);
    cfg.delay_window_days = Some(7.0);

    let breakdown = calculate_sensitivity_breakdown(&cfg).unwrap();

    let sum = breakdown.base_fraction
        + breakdown.sequential_fraction
        + breakdown.overdispersion_fraction
        + breakdown.systematics_fraction
        + breakdown.variance_reduction_fraction
        + breakdown.delay_fraction;

    assert!(
        (sum - 1.0).abs() < 0.03,
        "all fractions should sum to ~1.0, got {:.6} (base={:.4}, seq={:.4}, od={:.4}, sys={:.4}, vr={:.4}, cuped={:.4}, delay={:.4})",
        sum,
        breakdown.base_fraction,
        breakdown.sequential_fraction,
        breakdown.overdispersion_fraction,
        breakdown.systematics_fraction,
        breakdown.variance_reduction_fraction,
        breakdown.cuped_fraction,
        breakdown.delay_fraction,
    );
    assert!(
        (breakdown.cuped_fraction - breakdown.variance_reduction_fraction).abs() < 1e-12,
        "legacy cuped_fraction should mirror variance_reduction_fraction",
    );
}

#[test]
fn test_sensitivity_breakdown_base_only() {
    // With no adjustments active, base_fraction should be ~1.0
    // and all other fractions should be ~0.0.
    let cfg = base_config();
    let breakdown = calculate_sensitivity_breakdown(&cfg).unwrap();

    assert!(
        (breakdown.base_fraction - 1.0).abs() < 0.02,
        "base_fraction should be ~1.0 with no adjustments, got {:.6}",
        breakdown.base_fraction,
    );
    assert!(
        breakdown.sequential_fraction.abs() < 0.01,
        "sequential_fraction should be ~0.0 with no adjustments, got {:.6}",
        breakdown.sequential_fraction,
    );
    assert!(
        breakdown.overdispersion_fraction.abs() < 0.01,
        "overdispersion_fraction should be ~0.0, got {:.6}",
        breakdown.overdispersion_fraction,
    );
    assert!(
        breakdown.systematics_fraction.abs() < 0.01,
        "systematics_fraction should be ~0.0, got {:.6}",
        breakdown.systematics_fraction,
    );
    assert!(
        breakdown.cuped_fraction.abs() < 0.01,
        "cuped_fraction should be ~0.0, got {:.6}",
        breakdown.cuped_fraction,
    );
    assert!(
        breakdown.variance_reduction_fraction.abs() < 0.01,
        "variance_reduction_fraction should be ~0.0, got {:.6}",
        breakdown.variance_reduction_fraction,
    );
    assert!(
        breakdown.delay_fraction.abs() < 0.01,
        "delay_fraction should be ~0.0, got {:.6}",
        breakdown.delay_fraction,
    );
    assert_eq!(
        breakdown.n_final, breakdown.n_base,
        "n_final should equal n_base with no adjustments",
    );
}
