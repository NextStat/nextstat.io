//! A/B test sample-size calculator with optional adjustments.
//!
//! Implements fixed-horizon Z-test for two proportions with plug-in adjustments for:
//! - **Sequential testing** (group sequential inflation via O'Brien-Fleming / Pocock)
//! - **Overdispersion** (Beta-Binomial variance inflation)
//! - **Measurement systematics** (systematic uncertainty inflation)
//! - **CUPED/CURE** (variance reduction via pre-treatment covariates)
//! - **Delay correction** (exposure window ramp-up)

use ns_core::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::ads_artifacts::{
    AdsArtifactAssumption, AdsArtifactSemanticContext, AdsSystematicsProfile,
};
use crate::sequential::{BoundaryType, group_sequential_design};
use crate::variance_reduction::VarianceReductionMethod;

// ---------------------------------------------------------------------------
// Private normal helpers (each module keeps its own copies per crate convention)
// ---------------------------------------------------------------------------

/// Standard normal CDF: P(Z <= x).
fn normal_cdf(x: f64) -> f64 {
    0.5 * statrs::function::erf::erfc(-x / std::f64::consts::SQRT_2)
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

    let numer = C0 + t * (C1 + t * C2);
    let denom = 1.0 + t * (D1 + t * (D2 + t * D3));
    let approx = sign * (t - numer / denom);

    // One Newton refinement for higher accuracy.
    let phi = normal_cdf(approx);
    let pdf = (-0.5 * approx * approx).exp() / (2.0 * std::f64::consts::PI).sqrt();
    if pdf.abs() < 1e-300 {
        return approx;
    }
    approx - (phi - p) / pdf
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the sample-size calculator.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalculatorConfig {
    /// Baseline conversion rate (control arm), e.g. 0.02 for 2%.
    pub baseline_rate: f64,
    /// Minimum detectable effect as a relative lift, e.g. 0.10 for +10%.
    pub mde_relative: f64,
    /// Significance level (two-sided by default), e.g. 0.05.
    #[serde(default = "default_alpha")]
    pub alpha: f64,
    /// Statistical power, e.g. 0.80.
    #[serde(default = "default_power")]
    pub power: f64,
    /// Use two-sided test (default: true).
    #[serde(default = "default_two_sided")]
    pub two_sided: bool,

    // -- Optional adjustments -----------------------------------------------
    /// Number of interim looks for group sequential design (1 = fixed horizon, >1 = sequential).
    #[serde(default = "default_num_looks")]
    pub num_looks: usize,
    /// Spending function for sequential inflation: "none", "obrien_fleming", "pocock".
    #[serde(default = "default_spending_function")]
    pub spending_function: String,
    /// Beta-Binomial overdispersion parameter rho in [0, 1).
    #[serde(default)]
    pub overdispersion_rho: f64,
    /// Systematic measurement uncertainty (measurement_sigma).
    #[serde(default)]
    pub measurement_sigma: f64,
    /// Legacy CUPED single-covariate rho^2 input, kept for backward compatibility.
    #[serde(default)]
    pub cuped_rho_squared: f64,
    /// Expected pooled R^2 from the selected pre-treatment covariates, in [0, 1).
    #[serde(default)]
    pub expected_r_squared_from_covariates: f64,
    /// Expected number of selected pre-treatment covariates.
    #[serde(default)]
    pub expected_num_covariates: usize,
    /// Names of the selected pre-treatment covariates used to form the expected R^2.
    #[serde(default)]
    pub selected_covariates: Vec<String>,
    /// Must remain true: variance reduction only allows pre-treatment covariates.
    #[serde(default = "default_pre_treatment_covariates_only")]
    pub pre_treatment_covariates_only: bool,
    /// Delay correction: exponential ramp-up rate (lambda).
    #[serde(default)]
    pub delay_lambda: Option<f64>,
    /// Delay correction: observation window in days.
    #[serde(default)]
    pub delay_window_days: Option<f64>,
    /// Daily traffic per arm, used to compute duration_days.
    #[serde(default)]
    pub daily_traffic_per_arm: Option<f64>,
}

fn default_alpha() -> f64 {
    0.05
}
fn default_power() -> f64 {
    0.80
}
fn default_two_sided() -> bool {
    true
}
fn default_num_looks() -> usize {
    1
}
fn default_spending_function() -> String {
    "none".to_string()
}
fn default_pre_treatment_covariates_only() -> bool {
    true
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Typed planner summary for CUPED/CURE assumptions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VarianceReductionPlan {
    /// Variance-reduction method implied by the planner assumptions.
    pub method: VarianceReductionMethod,
    /// Expected pooled R^2 from the selected pre-treatment covariates.
    pub expected_r_squared: f64,
    /// Multiplicative variance-reduction factor applied to the sample-size baseline.
    pub variance_reduction_factor: f64,
    /// Effective sample multiplier implied by the variance reduction factor.
    pub effective_sample_multiplier: f64,
    /// Number of selected pre-treatment covariates.
    pub num_covariates: usize,
    /// Selected pre-treatment covariates echoed for logging/artifact surfaces.
    pub selected_covariates: Vec<String>,
    /// Must remain true: variance reduction only allows pre-treatment covariates.
    pub pre_treatment_covariates_only: bool,
}

/// Result of a sample-size calculation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SampleSizeResult {
    /// Required sample size per arm (after all adjustments, ceiling).
    pub n_per_arm: u64,
    /// Total required sample size (both arms).
    pub n_total: u64,
    /// Baseline (fixed-horizon) sample size per arm before adjustments.
    pub n_fixed: u64,
    /// Inflation factor from group sequential design (1.0 if none).
    pub inflation_sequential: f64,
    /// Variance inflation factor from overdispersion (1.0 if none).
    pub inflation_overdispersion: f64,
    /// Inflation factor from measurement systematics (1.0 if none).
    pub inflation_systematics: f64,
    /// Legacy field name preserved for CUPED/CURE backward compatibility.
    pub reduction_cuped: f64,
    /// Variance-reduction method that was applied.
    pub variance_reduction_method: VarianceReductionMethod,
    /// Variance reduction factor from CUPED/CURE (1.0 if none).
    pub variance_reduction_factor: f64,
    /// Expected R^2 from the selected pre-treatment covariates.
    pub expected_r_squared_from_covariates: f64,
    /// Typed variance-reduction plan metadata for planner/artifact consumers.
    pub variance_reduction: VarianceReductionPlan,
    /// Effective sample multiplier implied by the variance reduction factor.
    pub effective_sample_multiplier: f64,
    /// Number of selected pre-treatment covariates.
    pub num_covariates: usize,
    /// Selected pre-treatment covariates echoed for logging/artifact surfaces.
    pub selected_covariates: Vec<String>,
    /// Must remain true: variance reduction only allows pre-treatment covariates.
    pub pre_treatment_covariates_only: bool,
    /// Fraction of conversions lost due to delay (0.0 if none).
    pub delay_loss_fraction: f64,
    /// Estimated experiment duration in days (when daily_traffic_per_arm is provided).
    pub duration_days: Option<f64>,
    /// Human-readable descriptions of active assumptions.
    pub assumptions: Vec<String>,
    /// Shared artifact semantics used by calculator/WASM/reporting consumers.
    pub artifact_semantics: AdsArtifactSemanticContext,
    /// Calculation mode: "quick", "sequential", or "real_world".
    pub mode: String,
}

// ---------------------------------------------------------------------------
// Core computation
// ---------------------------------------------------------------------------

/// Calculate the required sample size per arm for a two-proportion Z-test,
/// with optional adjustments for sequential testing, overdispersion,
/// measurement systematics, CUPED/CURE variance reduction, and delay correction.
pub fn calculate_sample_size(config: &CalculatorConfig) -> Result<SampleSizeResult> {
    // ---- Validate inputs --------------------------------------------------
    validate_config(config)?;

    let p1 = config.baseline_rate;
    let p2 = p1 * (1.0 + config.mde_relative);
    let delta = p2 - p1;

    if delta.abs() < 1e-15 {
        return Err(Error::Validation("mde_relative produces zero absolute effect".to_string()));
    }
    if p2 <= 0.0 || p2 >= 1.0 {
        return Err(Error::Validation(format!("treatment rate p2 = {p2:.6} out of (0, 1)")));
    }

    // ---- Z-quantiles ------------------------------------------------------
    let z_alpha = if config.two_sided {
        normal_quantile(1.0 - config.alpha / 2.0)
    } else {
        normal_quantile(1.0 - config.alpha)
    };
    let z_beta = normal_quantile(config.power);

    // ---- Fixed-horizon sample size ----------------------------------------
    let p_bar = (p1 + p2) / 2.0;
    let numerator = z_alpha * (2.0 * p_bar * (1.0 - p_bar)).sqrt()
        + z_beta * (p1 * (1.0 - p1) + p2 * (1.0 - p2)).sqrt();
    let n_fixed_raw = (numerator * numerator) / (delta * delta);
    let n_fixed = n_fixed_raw.ceil() as u64;

    // ---- Sequential inflation ---------------------------------------------
    let inflation_sequential = compute_sequential_inflation(config, z_alpha)?;

    // ---- Overdispersion (Beta-Binomial) -----------------------------------
    let rho = config.overdispersion_rho;
    let inflation_overdispersion = if rho.abs() > 1e-15 {
        if !(0.0..1.0).contains(&rho) {
            return Err(Error::Validation("overdispersion_rho must be in [0, 1)".to_string()));
        }
        1.0 / (1.0 - rho)
    } else {
        if rho < 0.0 {
            return Err(Error::Validation("overdispersion_rho must be in [0, 1)".to_string()));
        }
        1.0
    };

    // ---- Measurement systematics ------------------------------------------
    // measurement_sigma is a relative uncertainty on observed rates (e.g. 0.15
    // means ±15% due to viewability, fraud, cross-device).  This adds an
    // irreducible variance floor that does NOT shrink with more data:
    //
    //   Var(effect) = v_stat/n  +  v_sys
    //   v_sys = 2 * (sigma * p_bar)^2
    //
    // Required n with systematics:
    //   n = (z_a + z_b)^2 * v_stat / (delta^2 - (z_a + z_b)^2 * v_sys)
    //
    // inflation = n_sys / n_base = delta^2 / (delta^2 - (z_a+z_b)^2 * v_sys)
    //           = 1 / (1 - (z_a+z_b)^2 * v_sys / delta^2)
    //
    // If denominator <= 0 the effect is undetectable under current systematics.
    let sigma = config.measurement_sigma;
    let inflation_systematics = if sigma.abs() > 1e-15 {
        if sigma < 0.0 {
            return Err(Error::Validation("measurement_sigma must be >= 0".to_string()));
        }
        let v_sys = 2.0 * (sigma * p_bar) * (sigma * p_bar);
        let z_sum = z_alpha + z_beta;
        let denom = delta * delta - z_sum * z_sum * v_sys;
        if denom <= 0.0 {
            return Err(Error::Validation(format!(
                "measurement_sigma={sigma:.4} makes the effect undetectable — \
                 systematic uncertainty dominates the signal (delta={delta:.6}, \
                 sigma_sys={:.6}). Reduce measurement_sigma or increase MDE.",
                (v_sys.sqrt())
            )));
        }
        let raw = (delta * delta) / denom;
        raw.max(1.0)
    } else {
        if sigma < 0.0 {
            return Err(Error::Validation("measurement_sigma must be >= 0".to_string()));
        }
        1.0
    };

    // ---- CUPED/CURE variance reduction ------------------------------------
    let variance_reduction = resolve_variance_reduction_plan(config)?;
    let expected_r_squared = variance_reduction.expected_r_squared;
    let selected_covariates = variance_reduction.selected_covariates.clone();
    let num_covariates = variance_reduction.num_covariates;
    let variance_reduction_method = variance_reduction.method;
    let variance_reduction_factor = variance_reduction.variance_reduction_factor;
    let reduction_cuped = variance_reduction_factor;
    let effective_sample_multiplier = variance_reduction.effective_sample_multiplier;

    // ---- Delay correction -------------------------------------------------
    let (delay_inflation, delay_loss_fraction) = compute_delay_correction(config)?;

    // ---- Combine adjustments ----------------------------------------------
    let n_adjusted = n_fixed_raw
        * inflation_sequential
        * inflation_overdispersion
        * inflation_systematics
        * variance_reduction_factor
        * delay_inflation;

    let n_per_arm = (n_adjusted.ceil() as u64).max(1);

    // ---- Duration ---------------------------------------------------------
    let duration_days = config
        .daily_traffic_per_arm
        .and_then(|d| if d > 0.0 { Some(n_per_arm as f64 / d) } else { None });

    // ---- Assumptions ------------------------------------------------------
    let mut assumptions = Vec::new();
    if config.two_sided {
        assumptions.push("Two-sided test".to_string());
    } else {
        assumptions.push("One-sided test".to_string());
    }
    if config.num_looks > 1 {
        assumptions.push(format!(
            "Group sequential design with {} looks ({})",
            config.num_looks, config.spending_function
        ));
    }
    if config.overdispersion_rho.abs() > 1e-15 {
        assumptions
            .push(format!("Beta-Binomial overdispersion (rho = {:.4})", config.overdispersion_rho));
    }
    if config.measurement_sigma.abs() > 1e-15 {
        assumptions
            .push(format!("Measurement systematics (sigma = {:.4})", config.measurement_sigma));
    }
    if expected_r_squared.abs() > 1e-15 {
        let method_label = match variance_reduction_method {
            VarianceReductionMethod::Cuped => "CUPED",
            VarianceReductionMethod::Cure => "CURE",
            VarianceReductionMethod::None => "variance reduction",
        };
        let covariate_suffix = if !selected_covariates.is_empty() {
            format!("; covariates = {}", selected_covariates.join(", "))
        } else if num_covariates > 0 {
            format!("; covariates = {}", num_covariates)
        } else {
            String::new()
        };
        assumptions.push(format!(
            "{method_label} variance reduction (expected R^2 = {:.4}; pre-treatment only = {}{covariate_suffix})",
            expected_r_squared,
            config.pre_treatment_covariates_only
        ));
    }
    if let (Some(lambda), Some(window)) = (config.delay_lambda, config.delay_window_days) {
        assumptions
            .push(format!("Delay correction (lambda = {lambda:.4}, window = {window:.1} days)",));
    }

    // ---- Mode -------------------------------------------------------------
    let has_real_world = config.overdispersion_rho.abs() > 1e-15
        || config.measurement_sigma.abs() > 1e-15
        || expected_r_squared.abs() > 1e-15
        || (config.delay_lambda.is_some() && config.delay_window_days.is_some());
    let mode = if has_real_world {
        "real_world".to_string()
    } else if config.num_looks > 1 {
        "sequential".to_string()
    } else {
        "quick".to_string()
    };
    let artifact_semantics = planner_artifact_semantics(config, &variance_reduction);

    Ok(SampleSizeResult {
        n_per_arm,
        n_total: n_per_arm * 2,
        n_fixed,
        inflation_sequential,
        inflation_overdispersion,
        inflation_systematics,
        reduction_cuped,
        variance_reduction_method,
        variance_reduction_factor,
        expected_r_squared_from_covariates: expected_r_squared,
        variance_reduction,
        effective_sample_multiplier,
        num_covariates,
        selected_covariates,
        pre_treatment_covariates_only: config.pre_treatment_covariates_only,
        delay_loss_fraction,
        duration_days,
        assumptions,
        artifact_semantics,
        mode,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_config(config: &CalculatorConfig) -> Result<()> {
    if config.baseline_rate <= 0.0 || config.baseline_rate >= 1.0 {
        return Err(Error::Validation("baseline_rate must be in (0, 1)".to_string()));
    }
    if config.mde_relative == 0.0 {
        return Err(Error::Validation("mde_relative must be non-zero".to_string()));
    }
    if config.alpha <= 0.0 || config.alpha >= 1.0 {
        return Err(Error::Validation("alpha must be in (0, 1)".to_string()));
    }
    if config.power <= 0.0 || config.power >= 1.0 {
        return Err(Error::Validation("power must be in (0, 1)".to_string()));
    }
    Ok(())
}

fn resolve_expected_r_squared(config: &CalculatorConfig) -> Result<f64> {
    let legacy = config.cuped_rho_squared;
    let general = config.expected_r_squared_from_covariates;

    if legacy.abs() > 1e-15 && general.abs() > 1e-15 && (legacy - general).abs() > 1e-12 {
        return Err(Error::Validation(
            "cuped_rho_squared and expected_r_squared_from_covariates disagree; provide only one value or make them equal"
                .to_string(),
        ));
    }

    let r_squared = if general.abs() > 1e-15 { general } else { legacy };
    if !(0.0..1.0).contains(&r_squared) {
        return Err(Error::Validation(
            "expected_r_squared_from_covariates must be in [0, 1)".to_string(),
        ));
    }
    if r_squared.abs() > 1e-15 && !config.pre_treatment_covariates_only {
        return Err(Error::Validation(
            "variance reduction requires pre_treatment_covariates_only=true".to_string(),
        ));
    }
    Ok(r_squared)
}

fn resolve_selected_covariates(config: &CalculatorConfig) -> Result<Vec<String>> {
    let selected = config.selected_covariates.clone();
    let expected_num_covariates = config.expected_num_covariates;
    if expected_num_covariates > 0
        && !selected.is_empty()
        && expected_num_covariates != selected.len()
    {
        return Err(Error::Validation(format!(
            "expected_num_covariates ({}) must match selected_covariates length ({})",
            expected_num_covariates,
            selected.len()
        )));
    }
    Ok(selected)
}

fn resolve_num_covariates(config: &CalculatorConfig, expected_r_squared: f64) -> Result<usize> {
    let selected_covariates = resolve_selected_covariates(config)?;
    let num_covariates = if !selected_covariates.is_empty() {
        selected_covariates.len()
    } else if config.expected_num_covariates > 0 {
        config.expected_num_covariates
    } else if config.expected_r_squared_from_covariates.abs() > 1e-15 {
        2
    } else if config.cuped_rho_squared.abs() > 1e-15 {
        1
    } else if expected_r_squared.abs() > 1e-15 {
        1
    } else {
        0
    };
    Ok(num_covariates)
}

fn resolve_variance_reduction_method(
    config: &CalculatorConfig,
    expected_r_squared: f64,
    num_covariates: usize,
) -> VarianceReductionMethod {
    if expected_r_squared.abs() <= 1e-15 {
        VarianceReductionMethod::None
    } else if config.expected_r_squared_from_covariates.abs() > 1e-15 {
        if num_covariates <= 1 {
            VarianceReductionMethod::Cuped
        } else {
            VarianceReductionMethod::Cure
        }
    } else if num_covariates > 1 {
        VarianceReductionMethod::Cure
    } else {
        VarianceReductionMethod::Cuped
    }
}

fn resolve_variance_reduction_plan(config: &CalculatorConfig) -> Result<VarianceReductionPlan> {
    let expected_r_squared = resolve_expected_r_squared(config)?;
    let selected_covariates = resolve_selected_covariates(config)?;
    let num_covariates = resolve_num_covariates(config, expected_r_squared)?;
    let method = resolve_variance_reduction_method(config, expected_r_squared, num_covariates);
    let variance_reduction_factor = 1.0 - expected_r_squared;
    let effective_sample_multiplier = effective_sample_multiplier(variance_reduction_factor);
    Ok(VarianceReductionPlan {
        method,
        expected_r_squared,
        variance_reduction_factor,
        effective_sample_multiplier,
        num_covariates,
        selected_covariates,
        pre_treatment_covariates_only: config.pre_treatment_covariates_only,
    })
}

fn effective_sample_multiplier(variance_reduction_factor: f64) -> f64 {
    if variance_reduction_factor > 1e-15 { 1.0 / variance_reduction_factor } else { f64::INFINITY }
}

fn planner_systematics_profile(config: &CalculatorConfig) -> AdsSystematicsProfile {
    let mut profile = AdsSystematicsProfile::from_defaults();
    let delay_enabled = config.delay_lambda.is_some() && config.delay_window_days.is_some();
    let measurement_enabled = config.measurement_sigma.abs() > 1e-15;

    for entry in &mut profile.entries {
        entry.enabled = match entry.systematic_id.as_str() {
            "conversion_lag_curve" => delay_enabled,
            "viewability_baseline" | "cross_device_partial_rate" | "residual_fraud_rate" => {
                measurement_enabled
            }
            "organic_cannibalization" | "experiment_interference" => false,
            _ => entry.enabled,
        };
    }

    profile
}

fn planner_typed_assumptions(
    config: &CalculatorConfig,
    variance_reduction: &VarianceReductionPlan,
) -> Vec<AdsArtifactAssumption> {
    let mut assumptions = vec![
        AdsArtifactAssumption {
            key: "analysis_surface".to_string(),
            value: "sample_size_planner".to_string(),
        },
        AdsArtifactAssumption {
            key: "test_sidedness".to_string(),
            value: if config.two_sided { "two_sided" } else { "one_sided" }.to_string(),
        },
    ];

    if config.num_looks > 1 {
        assumptions.push(AdsArtifactAssumption {
            key: "planned_looks".to_string(),
            value: config.num_looks.to_string(),
        });
        assumptions.push(AdsArtifactAssumption {
            key: "spending_function".to_string(),
            value: config.spending_function.clone(),
        });
    }

    if config.overdispersion_rho.abs() > 1e-15 {
        assumptions.push(AdsArtifactAssumption {
            key: "overdispersion_rho".to_string(),
            value: format!("{:.6}", config.overdispersion_rho),
        });
    }

    if config.measurement_sigma.abs() > 1e-15 {
        assumptions.push(AdsArtifactAssumption {
            key: "measurement_sigma".to_string(),
            value: format!("{:.6}", config.measurement_sigma),
        });
        assumptions.push(AdsArtifactAssumption {
            key: "measurement_sigma_scope".to_string(),
            value: "aggregate_measurement_floor".to_string(),
        });
    }

    if variance_reduction.expected_r_squared.abs() > 1e-15 {
        assumptions.push(AdsArtifactAssumption {
            key: "variance_reduction_method".to_string(),
            value: match variance_reduction.method {
                VarianceReductionMethod::Cuped => "cuped",
                VarianceReductionMethod::Cure => "cure",
                VarianceReductionMethod::None => "none",
            }
            .to_string(),
        });
        assumptions.push(AdsArtifactAssumption {
            key: "expected_r_squared_from_covariates".to_string(),
            value: format!("{:.6}", variance_reduction.expected_r_squared),
        });
        assumptions.push(AdsArtifactAssumption {
            key: "pre_treatment_covariates_only".to_string(),
            value: variance_reduction.pre_treatment_covariates_only.to_string(),
        });
        if !variance_reduction.selected_covariates.is_empty() {
            assumptions.push(AdsArtifactAssumption {
                key: "selected_covariates".to_string(),
                value: variance_reduction.selected_covariates.join(","),
            });
        }
    }

    if let (Some(lambda), Some(window_days)) = (config.delay_lambda, config.delay_window_days) {
        assumptions.push(AdsArtifactAssumption {
            key: "delay_lambda".to_string(),
            value: format!("{lambda:.6}"),
        });
        assumptions.push(AdsArtifactAssumption {
            key: "delay_window_days".to_string(),
            value: format!("{window_days:.3}"),
        });
    }

    assumptions
}

fn planner_artifact_semantics(
    config: &CalculatorConfig,
    variance_reduction: &VarianceReductionPlan,
) -> AdsArtifactSemanticContext {
    AdsArtifactSemanticContext::new(
        planner_systematics_profile(config),
        planner_typed_assumptions(config, variance_reduction),
    )
}

/// Compute sequential inflation factor using group_sequential_design.
fn compute_sequential_inflation(config: &CalculatorConfig, z_fixed: f64) -> Result<f64> {
    if config.num_looks <= 1 {
        return Ok(1.0);
    }

    let boundary_type = match config.spending_function.as_str() {
        "pocock" => BoundaryType::Pocock,
        "obrien_fleming" => BoundaryType::OBrienFleming,
        "none" => BoundaryType::OBrienFleming, // default fallback
        other => {
            return Err(Error::Validation(format!(
                "unknown spending_function: '{}'; expected 'none', 'obrien_fleming', or 'pocock'",
                other
            )));
        }
    };

    // Use the sequential module to build the design; fall back to a conservative
    // heuristic if the computation fails (e.g. extreme parameters).
    //
    // group_sequential_design() expects overall two-sided alpha and returns per-side
    // critical values.  For one-sided tests we pass 2*alpha so each side gets the
    // full one-sided alpha budget; for two-sided we pass alpha directly.
    let alpha_seq = if config.two_sided { config.alpha } else { 2.0 * config.alpha };

    match group_sequential_design(config.num_looks, alpha_seq, boundary_type, None) {
        Ok(design) => {
            // The last look's critical value relative to the fixed-horizon z gives
            // the sample-size inflation factor.
            if let Some(last) = design.looks.last() {
                let ratio = last.critical_value / z_fixed;
                Ok((ratio * ratio).max(1.0))
            } else {
                // Should not happen, but defend.
                Ok(1.0)
            }
        }
        Err(_) => {
            // Fallback: Pocock-style inflation ≈ 1 + 0.2*(looks-1) is a rough upper bound.
            let fallback = 1.0 + 0.2 * (config.num_looks as f64 - 1.0);
            Ok(fallback.max(1.0))
        }
    }
}

/// Compute delay correction: returns (inflation_factor, loss_fraction).
fn compute_delay_correction(config: &CalculatorConfig) -> Result<(f64, f64)> {
    let (lambda, window) = match (config.delay_lambda, config.delay_window_days) {
        (Some(l), Some(w)) => (l, w),
        (None, None) => return Ok((1.0, 0.0)),
        _ => {
            return Err(Error::Validation(
                "delay_lambda and delay_window_days must both be set or both be None".to_string(),
            ));
        }
    };

    if lambda <= 0.0 {
        return Err(Error::Validation("delay_lambda must be > 0".to_string()));
    }
    if window <= 0.0 {
        return Err(Error::Validation("delay_window_days must be > 0".to_string()));
    }

    let observed_fraction = 1.0 - (-lambda * window).exp();
    if observed_fraction <= 0.0 || !observed_fraction.is_finite() {
        return Err(Error::Computation(
            "delay correction produced invalid observed fraction".to_string(),
        ));
    }

    let loss_fraction = 1.0 - observed_fraction;
    let inflation = 1.0 / observed_fraction;
    Ok((inflation, loss_fraction))
}

/// Compute the combined multiplicative variance scaling factor
/// (sequential x overdispersion x CUPED/CURE x delay).
///
/// Does NOT include measurement systematics, which is additive (irreducible
/// variance floor), not multiplicative.  Callers handle v_sys separately.
fn compute_multiplicative_factor(config: &CalculatorConfig, z_alpha: f64) -> Result<f64> {
    let inflation_sequential = compute_sequential_inflation(config, z_alpha)?;

    let rho = config.overdispersion_rho;
    let inflation_overdispersion =
        if rho.abs() > 1e-15 && rho < 1.0 { 1.0 / (1.0 - rho) } else { 1.0 };

    let variance_reduction_factor = 1.0 - resolve_expected_r_squared(config)?;

    let (delay_inflation, _) = compute_delay_correction(config)?;

    Ok(inflation_sequential
        * inflation_overdispersion
        * variance_reduction_factor
        * delay_inflation)
}

// ---------------------------------------------------------------------------
// Power curve & MDE curve
// ---------------------------------------------------------------------------

/// A single point on the power-vs-sample-size curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerCurvePoint {
    pub n_per_arm: i64,
    pub power: f64,
}

/// A single point on the MDE-vs-sample-size curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MdeCurvePoint {
    pub n_per_arm: i64,
    pub mde_relative: f64,
}

/// Generate a power curve: power vs sample size at fixed MDE.
///
/// Each point computes power using the full adjusted variance model:
///
///   Var(effect) = v_stat * mult_factor / n  +  v_sys
///
/// where `mult_factor` captures sequential, overdispersion, CUPED/CURE, and delay
/// adjustments, and `v_sys = 2*(sigma*p_bar)^2` is the irreducible systematic
/// variance floor.
pub fn calculate_power_curve(
    config: &CalculatorConfig,
    num_points: usize,
) -> Result<Vec<PowerCurvePoint>> {
    let result = calculate_sample_size(config)?;
    let n_target = result.n_per_arm as f64;

    let n_min = (0.1 * n_target).max(1.0);
    let n_max = 3.0 * n_target;

    let p1 = config.baseline_rate;
    let p2 = p1 * (1.0 + config.mde_relative);
    let delta = (p2 - p1).abs();
    let p_bar = (p1 + p2) / 2.0;

    let z_alpha = if config.two_sided {
        normal_quantile(1.0 - config.alpha / 2.0)
    } else {
        normal_quantile(1.0 - config.alpha)
    };

    let mult = compute_multiplicative_factor(config, z_alpha)?;
    let v_stat = p1 * (1.0 - p1) + p2 * (1.0 - p2);
    let sigma = config.measurement_sigma;
    let v_sys = if sigma.abs() > 1e-15 { 2.0 * (sigma * p_bar) * (sigma * p_bar) } else { 0.0 };

    let mut points = Vec::with_capacity(num_points);
    for i in 0..num_points {
        let frac = if num_points <= 1 { 0.5 } else { i as f64 / (num_points - 1) as f64 };
        let n = n_min + frac * (n_max - n_min);
        let n_int = n.round().max(1.0) as i64;

        let var_n = v_stat * mult / n_int as f64 + v_sys;
        let power = if var_n > 0.0 { normal_cdf(delta / var_n.sqrt() - z_alpha) } else { 1.0 };

        points.push(PowerCurvePoint { n_per_arm: n_int, power });
    }

    Ok(points)
}

/// Generate an MDE curve: detectable relative effect vs sample size at fixed power.
///
/// Each point computes MDE using the full adjusted variance model:
///
///   Var(effect) = v_stat_h0 * mult_factor / n  +  v_sys_h0
///
/// Under H0, `p_bar ≈ p1` (treatment rate unknown), so `v_sys_h0 = 2*(sigma*p1)^2`.
pub fn calculate_mde_curve(
    config: &CalculatorConfig,
    num_points: usize,
) -> Result<Vec<MdeCurvePoint>> {
    let result = calculate_sample_size(config)?;
    let n_target = result.n_per_arm as f64;

    let n_min = (0.1 * n_target).max(1.0);
    let n_max = 3.0 * n_target;

    let p1 = config.baseline_rate;

    let z_alpha = if config.two_sided {
        normal_quantile(1.0 - config.alpha / 2.0)
    } else {
        normal_quantile(1.0 - config.alpha)
    };
    let z_beta = normal_quantile(config.power);

    let mult = compute_multiplicative_factor(config, z_alpha)?;
    let v_stat_h0 = 2.0 * p1 * (1.0 - p1);
    let sigma = config.measurement_sigma;
    let v_sys_h0 = if sigma.abs() > 1e-15 { 2.0 * (sigma * p1) * (sigma * p1) } else { 0.0 };

    let mut points = Vec::with_capacity(num_points);
    for i in 0..num_points {
        let frac = if num_points <= 1 { 0.5 } else { i as f64 / (num_points - 1) as f64 };
        let n = n_min + frac * (n_max - n_min);
        let n_int = n.round().max(1.0) as i64;

        let var_n = v_stat_h0 * mult / n_int as f64 + v_sys_h0;
        let delta = (z_alpha + z_beta) * var_n.sqrt();
        let mde_relative = delta / p1;

        points.push(MdeCurvePoint { n_per_arm: n_int, mde_relative });
    }

    Ok(points)
}

// ---------------------------------------------------------------------------
// Sequential schedule
// ---------------------------------------------------------------------------

/// Per-look information for a sequential schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequentialLookInfo {
    pub look_number: usize,
    pub info_fraction: f64,
    pub critical_value: f64,
    pub nominal_alpha: f64,
    pub cumulative_alpha: f64,
    pub n_per_arm_at_look: u64,
}

/// Comparison of max sample sizes across fixed, OBF, and Pocock designs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequentialComparison {
    pub fixed_n: u64,
    pub obf_n: u64,
    pub pocock_n: u64,
}

/// Result of computing a sequential analysis schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequentialScheduleResult {
    pub looks: Vec<SequentialLookInfo>,
    pub spending_function: String,
    pub total_alpha: f64,
    pub inflation_factor: f64,
    pub comparison: SequentialComparison,
    pub artifact_semantics: AdsArtifactSemanticContext,
}

/// Breakdown of how each adjustment contributes to the final sample size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityBreakdown {
    pub n_final: u64,
    pub n_base: u64,
    pub base_fraction: f64,
    pub sequential_fraction: f64,
    pub overdispersion_fraction: f64,
    pub systematics_fraction: f64,
    /// Legacy field name preserved for backward compatibility.
    pub cuped_fraction: f64,
    pub variance_reduction_fraction: f64,
    pub variance_reduction_method: VarianceReductionMethod,
    pub variance_reduction: VarianceReductionPlan,
    pub num_covariates: usize,
    pub selected_covariates: Vec<String>,
    pub delay_fraction: f64,
    pub artifact_semantics: AdsArtifactSemanticContext,
}

/// Compute the per-look sequential schedule with boundaries and a comparison table.
///
/// Requires `config.num_looks >= 2`; returns an error otherwise.
pub fn calculate_sequential_schedule(
    config: &CalculatorConfig,
) -> Result<SequentialScheduleResult> {
    if config.num_looks < 2 {
        return Err(Error::Validation("sequential schedule requires num_looks >= 2".to_string()));
    }

    // Get the final n_per_arm from the full calculator.
    let result = calculate_sample_size(config)?;
    let n_per_arm = result.n_per_arm;

    // Parse boundary type (same logic as compute_sequential_inflation).
    let boundary_type = match config.spending_function.as_str() {
        "pocock" => BoundaryType::Pocock,
        "obrien_fleming" => BoundaryType::OBrienFleming,
        "none" => BoundaryType::OBrienFleming,
        other => {
            return Err(Error::Validation(format!(
                "unknown spending_function: '{}'; expected 'none', 'obrien_fleming', or 'pocock'",
                other
            )));
        }
    };

    // Build the sequential design.
    let alpha_seq = if config.two_sided { config.alpha } else { 2.0 * config.alpha };
    let design = group_sequential_design(config.num_looks, alpha_seq, boundary_type, None)?;

    // Map each SequentialLook to SequentialLookInfo.
    let looks: Vec<SequentialLookInfo> = design
        .looks
        .iter()
        .map(|look| SequentialLookInfo {
            look_number: look.look,
            info_fraction: look.info_fraction,
            critical_value: look.critical_value,
            nominal_alpha: look.nominal_alpha,
            cumulative_alpha: look.cumulative_alpha,
            n_per_arm_at_look: (n_per_arm as f64 * look.info_fraction).ceil() as u64,
        })
        .collect();

    let total_alpha = design.looks.last().map(|l| l.cumulative_alpha).unwrap_or(config.alpha);

    // Compute comparison: fixed, OBF, Pocock max n.
    let mut fixed_cfg = config.clone();
    fixed_cfg.num_looks = 1;
    fixed_cfg.spending_function = "none".to_string();
    let fixed_n = calculate_sample_size(&fixed_cfg)?.n_per_arm;

    let mut obf_cfg = config.clone();
    obf_cfg.num_looks = config.num_looks;
    obf_cfg.spending_function = "obrien_fleming".to_string();
    let obf_n = calculate_sample_size(&obf_cfg)?.n_per_arm;

    let mut pocock_cfg = config.clone();
    pocock_cfg.num_looks = config.num_looks;
    pocock_cfg.spending_function = "pocock".to_string();
    let pocock_n = calculate_sample_size(&pocock_cfg)?.n_per_arm;

    Ok(SequentialScheduleResult {
        looks,
        spending_function: config.spending_function.clone(),
        total_alpha,
        inflation_factor: result.inflation_sequential,
        comparison: SequentialComparison { fixed_n, obf_n, pocock_n },
        artifact_semantics: result.artifact_semantics,
    })
}

/// Compute a sensitivity breakdown showing how each adjustment contributes
/// to the final sample size as a fraction of n_final.
///
/// Adjustments are applied progressively in this order:
/// base -> sequential -> overdispersion -> systematics -> variance reduction -> delay.
pub fn calculate_sensitivity_breakdown(config: &CalculatorConfig) -> Result<SensitivityBreakdown> {
    // 1. n_base: no adjustments at all.
    let mut base_cfg = config.clone();
    base_cfg.num_looks = 1;
    base_cfg.spending_function = "none".to_string();
    base_cfg.overdispersion_rho = 0.0;
    base_cfg.measurement_sigma = 0.0;
    base_cfg.cuped_rho_squared = 0.0;
    base_cfg.expected_r_squared_from_covariates = 0.0;
    base_cfg.expected_num_covariates = 0;
    base_cfg.selected_covariates.clear();
    base_cfg.delay_lambda = None;
    base_cfg.delay_window_days = None;
    let n_base = calculate_sample_size(&base_cfg)?.n_per_arm;

    // 2. n_after_seq: + sequential.
    let mut seq_cfg = base_cfg.clone();
    seq_cfg.num_looks = config.num_looks;
    seq_cfg.spending_function = config.spending_function.clone();
    let n_after_seq = calculate_sample_size(&seq_cfg)?.n_per_arm;

    // 3. n_after_od: + overdispersion.
    let mut od_cfg = seq_cfg.clone();
    od_cfg.overdispersion_rho = config.overdispersion_rho;
    let n_after_od = calculate_sample_size(&od_cfg)?.n_per_arm;

    // 4. n_after_sys: + systematics.
    let mut sys_cfg = od_cfg.clone();
    sys_cfg.measurement_sigma = config.measurement_sigma;
    let n_after_sys = calculate_sample_size(&sys_cfg)?.n_per_arm;

    // 5. n_after_variance_reduction: + CUPED/CURE.
    let mut variance_cfg = sys_cfg.clone();
    variance_cfg.cuped_rho_squared = config.cuped_rho_squared;
    variance_cfg.expected_r_squared_from_covariates = config.expected_r_squared_from_covariates;
    variance_cfg.expected_num_covariates = config.expected_num_covariates;
    variance_cfg.selected_covariates = config.selected_covariates.clone();
    variance_cfg.pre_treatment_covariates_only = config.pre_treatment_covariates_only;
    let n_after_variance_reduction = calculate_sample_size(&variance_cfg)?.n_per_arm;

    // 6. n_final: + delay.
    let mut final_cfg = variance_cfg.clone();
    final_cfg.delay_lambda = config.delay_lambda;
    final_cfg.delay_window_days = config.delay_window_days;
    let n_final = calculate_sample_size(&final_cfg)?.n_per_arm;

    // Compute fractions. Guard against n_final == 0.
    let nf = n_final.max(1) as f64;
    let base_fraction = n_base as f64 / nf;
    let sequential_fraction = (n_after_seq as f64 - n_base as f64) / nf;
    let overdispersion_fraction = (n_after_od as f64 - n_after_seq as f64) / nf;
    let systematics_fraction = (n_after_sys as f64 - n_after_od as f64) / nf;
    let variance_reduction_fraction = (n_after_variance_reduction as f64 - n_after_sys as f64) / nf;
    let cuped_fraction = variance_reduction_fraction;
    let delay_fraction = (n_final as f64 - n_after_variance_reduction as f64) / nf;
    let variance_reduction = resolve_variance_reduction_plan(config)?;
    let selected_covariates = variance_reduction.selected_covariates.clone();
    let artifact_semantics = planner_artifact_semantics(config, &variance_reduction);

    Ok(SensitivityBreakdown {
        n_final,
        n_base,
        base_fraction,
        sequential_fraction,
        overdispersion_fraction,
        systematics_fraction,
        cuped_fraction,
        variance_reduction_fraction,
        variance_reduction_method: variance_reduction.method,
        variance_reduction: variance_reduction.clone(),
        num_covariates: variance_reduction.num_covariates,
        selected_covariates,
        delay_fraction,
        artifact_semantics,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build a default config.
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

    #[test]
    fn normal_helpers_known_values() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-10);
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-4);
        assert!((normal_quantile(0.975) - 1.96).abs() < 1e-3);
        assert!((normal_quantile(0.5)).abs() < 1e-10);
    }

    #[test]
    fn validation_errors() {
        let mut cfg = base_config();
        cfg.baseline_rate = 0.0;
        assert!(calculate_sample_size(&cfg).is_err());

        cfg = base_config();
        cfg.mde_relative = 0.0;
        assert!(calculate_sample_size(&cfg).is_err());

        cfg = base_config();
        cfg.alpha = 1.5;
        assert!(calculate_sample_size(&cfg).is_err());

        cfg = base_config();
        cfg.power = 0.0;
        assert!(calculate_sample_size(&cfg).is_err());
    }

    #[test]
    fn serde_roundtrip() {
        let cfg = base_config();
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: CalculatorConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, cfg2);

        let result = calculate_sample_size(&cfg).unwrap();
        let json = serde_json::to_string(&result).unwrap();
        let result2: SampleSizeResult = serde_json::from_str(&json).unwrap();
        // Integer fields must match exactly; floats may round-trip with representation noise
        // so we compare structurally.
        assert_eq!(result.n_per_arm, result2.n_per_arm);
        assert_eq!(result.n_total, result2.n_total);
        assert_eq!(result.n_fixed, result2.n_fixed);
        assert!((result.inflation_sequential - result2.inflation_sequential).abs() < 1e-12);
        assert!((result.inflation_overdispersion - result2.inflation_overdispersion).abs() < 1e-12);
        assert!((result.reduction_cuped - result2.reduction_cuped).abs() < 1e-12);
    }

    #[test]
    fn sample_size_result_exposes_shared_artifact_semantics() {
        let mut cfg = base_config();
        cfg.measurement_sigma = 0.005;
        cfg.expected_r_squared_from_covariates = 0.12;
        cfg.expected_num_covariates = 2;
        cfg.selected_covariates = vec!["pre_ctr".into(), "pre_cvr".into()];
        cfg.delay_lambda = Some(0.15);
        cfg.delay_window_days = Some(7.0);

        let result = calculate_sample_size(&cfg).unwrap();

        assert_eq!(
            result.artifact_semantics.contract.artifact_type,
            crate::ads_artifacts::ADS_STATISTICAL_ARTIFACT_TYPE
        );
        assert_eq!(
            result.artifact_semantics.contract.systematics_registry_version,
            crate::ads_artifacts::ADS_SYSTEMATICS_REGISTRY_VERSION
        );
        assert_eq!(result.artifact_semantics.systematics_profile.entries.len(), 6);
        assert!(
            result
                .artifact_semantics
                .assumptions
                .iter()
                .any(|item| item.key == "variance_reduction_method" && item.value == "cure")
        );
        assert!(
            result
                .artifact_semantics
                .assumptions
                .iter()
                .any(|item| item.key == "selected_covariates" && item.value == "pre_ctr,pre_cvr")
        );
    }

    #[test]
    fn sequential_and_sensitivity_surfaces_share_artifact_semantics() {
        let mut cfg = base_config();
        cfg.num_looks = 3;
        cfg.spending_function = "obrien_fleming".into();
        cfg.measurement_sigma = 0.003;

        let schedule = calculate_sequential_schedule(&cfg).unwrap();
        let breakdown = calculate_sensitivity_breakdown(&cfg).unwrap();

        assert_eq!(
            schedule.artifact_semantics.contract.artifact_type,
            crate::ads_artifacts::ADS_STATISTICAL_ARTIFACT_TYPE
        );
        assert_eq!(
            breakdown.artifact_semantics.contract.version,
            crate::ads_artifacts::ADS_STATISTICAL_ARTIFACT_VERSION
        );
        assert!(
            schedule
                .artifact_semantics
                .assumptions
                .iter()
                .any(|item| item.key == "planned_looks" && item.value == "3")
        );
        let measurement_entry = breakdown
            .artifact_semantics
            .systematics_profile
            .entries
            .iter()
            .find(|entry| entry.systematic_id == "viewability_baseline")
            .expect("viewability_baseline must exist");
        assert!(measurement_entry.enabled);
    }
}
