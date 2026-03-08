use ns_inference::{
    CovariateProvenance, CupedArmData, MultiCovariateArmData, VarianceReductionMethod,
    VarianceReductionSolver, cuped_adjust, cure_adjust,
};
use serde::Deserialize;

const CUPED_BINARY_FIXTURE: &str =
    include_str!("../../../tests/fixtures/variance_reduction/cuped_binary.json");
const CURE_REVENUE_FIXTURE: &str =
    include_str!("../../../tests/fixtures/variance_reduction/cure_revenue.json");
const CURE_RATIO_STYLE_FIXTURE: &str =
    include_str!("../../../tests/fixtures/variance_reduction/cure_ratio_style.json");
const CURE_LOW_CONVERSION_FIXTURE: &str =
    include_str!("../../../tests/fixtures/variance_reduction/cure_low_conversion.json");
const CURE_MULTI_CHANNEL_FIXTURE: &str =
    include_str!("../../../tests/fixtures/variance_reduction/cure_multi_channel.json");
const CURE_COLLINEAR_RIDGE_FIXTURE: &str =
    include_str!("../../../tests/fixtures/variance_reduction/cure_collinear_ridge.json");

#[derive(Debug, Deserialize)]
struct VarianceReductionFixture {
    schema_version: String,
    metric_type: String,
    method: VarianceReductionMethod,
    control_outcomes: Vec<f64>,
    variant_outcomes: Vec<f64>,
    control_covariates: Vec<Vec<f64>>,
    variant_covariates: Vec<Vec<f64>>,
    selected_covariates: Vec<String>,
    covariate_provenance: Vec<CovariateProvenance>,
    pre_treatment_only: bool,
    expected: ExpectedVarianceReductionResult,
}

#[derive(Debug, Deserialize)]
struct ExpectedVarianceReductionResult {
    method: VarianceReductionMethod,
    adjusted_mean_control: f64,
    adjusted_mean_variant: f64,
    effect: f64,
    theta: Vec<f64>,
    #[serde(default)]
    rho: Option<f64>,
    r_squared: f64,
    variance_reduction_factor: f64,
    effective_sample_multiplier: f64,
    solver: VarianceReductionSolver,
    regression_rank: usize,
    condition_number: Option<f64>,
    ridge_lambda: Option<f64>,
}

fn load_fixture(json: &'static str) -> VarianceReductionFixture {
    serde_json::from_str(json).expect("variance reduction fixture should deserialize")
}

fn assert_close(label: &str, got: f64, want: f64, tol: f64) {
    let diff = (got - want).abs();
    assert!(
        diff <= tol,
        "{label} mismatch: got {got:.12}, want {want:.12}, diff {diff:.12}, tol {tol:.12}"
    );
}

fn assert_option_close(label: &str, got: Option<f64>, want: Option<f64>, tol: f64) {
    match (got, want) {
        (Some(got), Some(want)) => assert_close(label, got, want, tol),
        (None, None) => {}
        (got_opt, want_opt) => panic!("{label} mismatch: got {got_opt:?}, want {want_opt:?}"),
    }
}

#[allow(clippy::too_many_arguments)]
fn assert_common_cure_fields(
    fixture: &VarianceReductionFixture,
    got_method: VarianceReductionMethod,
    got_adjusted_mean_control: f64,
    got_adjusted_mean_variant: f64,
    got_effect: f64,
    got_theta: &[f64],
    got_r_squared: f64,
    got_variance_reduction_factor: f64,
    got_effective_sample_multiplier: f64,
    got_solver: VarianceReductionSolver,
    got_regression_rank: usize,
    got_condition_number: Option<f64>,
    got_ridge_lambda: Option<f64>,
    got_selected_covariates: &[String],
    got_covariate_provenance: &[CovariateProvenance],
    got_pre_treatment_only: bool,
    got_provenance_validated: bool,
) {
    assert_eq!(fixture.schema_version, "nextstat.variance_reduction_fixture.v1");
    assert!(!fixture.metric_type.is_empty());
    assert_eq!(got_method, fixture.method);
    assert_eq!(got_method, fixture.expected.method);
    assert_eq!(got_selected_covariates, fixture.selected_covariates);
    assert_eq!(got_covariate_provenance, fixture.covariate_provenance);
    assert_eq!(got_pre_treatment_only, fixture.pre_treatment_only);
    assert!(got_provenance_validated);

    assert_close(
        "adjusted_mean_control",
        got_adjusted_mean_control,
        fixture.expected.adjusted_mean_control,
        1e-9,
    );
    assert_close(
        "adjusted_mean_variant",
        got_adjusted_mean_variant,
        fixture.expected.adjusted_mean_variant,
        1e-9,
    );
    assert_close("effect", got_effect, fixture.expected.effect, 1e-9);
    assert_eq!(got_theta.len(), fixture.expected.theta.len());
    for (idx, (got, want)) in got_theta.iter().zip(fixture.expected.theta.iter()).enumerate() {
        assert_close(&format!("theta[{idx}]"), *got, *want, 1e-9);
    }
    assert_close("r_squared", got_r_squared, fixture.expected.r_squared, 1e-9);
    assert_close(
        "variance_reduction_factor",
        got_variance_reduction_factor,
        fixture.expected.variance_reduction_factor,
        1e-9,
    );
    assert_close(
        "effective_sample_multiplier",
        got_effective_sample_multiplier,
        fixture.expected.effective_sample_multiplier,
        1e-9,
    );
    assert_eq!(got_solver, fixture.expected.solver);
    assert_eq!(got_regression_rank, fixture.expected.regression_rank);
    assert_option_close(
        "condition_number",
        got_condition_number,
        fixture.expected.condition_number,
        1e-9,
    );
    assert_option_close("ridge_lambda", got_ridge_lambda, fixture.expected.ridge_lambda, 1e-12);
}

#[test]
fn cuped_binary_fixture_matches_reference_outputs() {
    let fixture = load_fixture(CUPED_BINARY_FIXTURE);
    let control = CupedArmData {
        outcomes: fixture.control_outcomes.clone(),
        covariates: fixture.control_covariates.iter().map(|row| row[0]).collect(),
        covariate_name: fixture.selected_covariates.first().cloned(),
        covariate_provenance: fixture.covariate_provenance.first().cloned(),
        pre_treatment_only: fixture.pre_treatment_only,
    };
    let variant = CupedArmData {
        outcomes: fixture.variant_outcomes.clone(),
        covariates: fixture.variant_covariates.iter().map(|row| row[0]).collect(),
        covariate_name: fixture.selected_covariates.first().cloned(),
        covariate_provenance: fixture.covariate_provenance.first().cloned(),
        pre_treatment_only: fixture.pre_treatment_only,
    };

    let result = cuped_adjust(&control, &variant).expect("cuped fixture should succeed");
    assert_common_cure_fields(
        &fixture,
        result.method,
        result.adjusted_mean_control,
        result.adjusted_mean_variant,
        result.effect,
        &[result.theta],
        result.r_squared,
        result.variance_reduction_factor,
        result.effective_sample_multiplier,
        result.solver,
        result.regression_rank,
        result.condition_number,
        result.ridge_lambda,
        &result.selected_covariates,
        &result.covariate_provenance,
        result.pre_treatment_only,
        result.provenance_validated,
    );
    assert_close("rho", result.rho, fixture.expected.rho.expect("rho required"), 1e-9);
}

fn assert_cure_fixture(json: &'static str) {
    let fixture = load_fixture(json);
    let control = MultiCovariateArmData {
        outcomes: fixture.control_outcomes.clone(),
        covariates: fixture.control_covariates.clone(),
        covariate_names: fixture.selected_covariates.clone(),
        covariate_provenance: fixture.covariate_provenance.clone(),
        pre_treatment_only: fixture.pre_treatment_only,
    };
    let variant = MultiCovariateArmData {
        outcomes: fixture.variant_outcomes.clone(),
        covariates: fixture.variant_covariates.clone(),
        covariate_names: fixture.selected_covariates.clone(),
        covariate_provenance: fixture.covariate_provenance.clone(),
        pre_treatment_only: fixture.pre_treatment_only,
    };

    let result = cure_adjust(&control, &variant).expect("cure fixture should succeed");
    assert_common_cure_fields(
        &fixture,
        result.method,
        result.adjusted_mean_control,
        result.adjusted_mean_variant,
        result.effect,
        &result.theta,
        result.r_squared,
        result.variance_reduction_factor,
        result.effective_sample_multiplier,
        result.solver,
        result.regression_rank,
        result.condition_number,
        result.ridge_lambda,
        &result.selected_covariates,
        &result.covariate_provenance,
        result.pre_treatment_only,
        result.provenance_validated,
    );
}

#[test]
fn cure_revenue_fixture_matches_reference_outputs() {
    assert_cure_fixture(CURE_REVENUE_FIXTURE);
}

#[test]
fn cure_ratio_style_fixture_matches_reference_outputs() {
    assert_cure_fixture(CURE_RATIO_STYLE_FIXTURE);
}

#[test]
fn cure_low_conversion_fixture_matches_reference_outputs() {
    assert_cure_fixture(CURE_LOW_CONVERSION_FIXTURE);
}

#[test]
fn cure_multi_channel_fixture_matches_reference_outputs() {
    assert_cure_fixture(CURE_MULTI_CHANNEL_FIXTURE);
}

#[test]
fn cure_collinear_ridge_fixture_matches_reference_outputs() {
    assert_cure_fixture(CURE_COLLINEAR_RIDGE_FIXTURE);
}
