use std::collections::{BTreeMap, BTreeSet};
use std::ops::{Deref, Range};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::optimizer::{LbfgsbOptimizer, ObjectiveFunction, OptimizationResult, OptimizerConfig};
use csv::Trim;
use nalgebra::{DMatrix, DVector};
use ns_core::{Error, Result};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

pub const MEASUREMENT_COMBINATION_SCHEMA_V0: &str = "nextstat_measurement_combination_v0";
pub const MEASUREMENT_COMBINATION_MANIFEST_SCHEMA_V0: &str =
    "nextstat_measurement_combination_manifest_v0";
pub const MEASUREMENT_COMBINATION_RESULT_SCHEMA_V0: &str =
    "nextstat_measurement_combination_result_v0";
pub const MEASUREMENT_COMBINATION_CALIBRATION_SCHEMA_V0: &str =
    "nextstat_measurement_combination_calibration_v0";
pub const MEASUREMENT_COMBINATION_CALIBRATION_STUDY_SCHEMA_V0: &str =
    "nextstat_measurement_combination_calibration_study_v0";
pub const MEASUREMENT_COMBINATION_SCENARIO_STUDY_SCHEMA_V0: &str =
    "nextstat_measurement_combination_scenario_study_v0";
pub const MEASUREMENT_COMBINATION_SCENARIO_STUDY_SOLVER_PARITY_SCHEMA_V0: &str =
    "nextstat_measurement_combination_scenario_study_solver_parity_v0";
pub const MEASUREMENT_COMBINATION_SCENARIO_STUDY_SOLVER_PARITY_DIGEST_SCHEMA_V0: &str =
    "nextstat_measurement_combination_scenario_study_solver_parity_digest_v0";
pub const MEASUREMENT_COMBINATION_SCENARIO_CONFIG_SCHEMA_V0: &str =
    "nextstat_measurement_combination_scenarios_v0";
pub const MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_SCHEMA_V0: &str =
    "nextstat_measurement_combination_calibration_campaign_v0";
pub const MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_SOLVER_PARITY_SCHEMA_V0: &str =
    "nextstat_measurement_combination_calibration_campaign_solver_parity_v0";
pub const MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_SOLVER_PARITY_DIGEST_SCHEMA_V0: &str =
    "nextstat_measurement_combination_calibration_campaign_solver_parity_digest_v0";
pub const MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_SUMMARY_SCHEMA_V0: &str =
    "nextstat_measurement_combination_calibration_campaign_summary_v0";
pub const MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_BRIEF_SCHEMA_V0: &str =
    "nextstat_measurement_combination_calibration_campaign_brief_v0";
pub const MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_FAMILY_REPORT_SCHEMA_V0: &str =
    "nextstat_measurement_combination_calibration_campaign_family_report_v0";
pub const MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_FAMILY_MATRIX_SCHEMA_V0: &str =
    "nextstat_measurement_combination_calibration_campaign_family_matrix_v0";
pub const MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_PORTFOLIO_SCHEMA_V0: &str =
    "nextstat_measurement_combination_calibration_campaign_portfolio_v0";
pub const MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_PORTFOLIO_STABILITY_SCHEMA_V0: &str =
    "nextstat_measurement_combination_calibration_campaign_portfolio_stability_v0";

const SYMM_TOL: f64 = 1e-9;
const PSD_TOL: f64 = 1e-10;
const CHOLESKY_FAST_PATH_MIN_DIAG_RATIO: f64 = 1e-4;
const TAU_MIN: f64 = 1e-6;
const TAU_MAX: f64 = 1e6;
const THETA_BOUND: f64 = 50.0;
const GVM_STABILITY_STABLE: &str = "stable";
const GVM_STABILITY_RESEARCH_GRADE: &str = "research-grade";
const ANALYTIC_FAST_PATH_NM_THRESHOLD: usize = 128;
const BARTLETT_FAST_PATH_NM_THRESHOLD: usize = 128;
const NUMERICAL_PAPER_ANALYTIC_WARM_START_NM_THRESHOLD: usize = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementCombinationSolver {
    Numerical,
    NumericalPaper,
    AnalyticPerturbative,
    Auto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationSpec {
    pub schema_version: String,
    pub poi: String,
    pub measurements: Vec<MeasurementInput>,
    pub stat_covariance: Vec<Vec<f64>>,
    #[serde(default)]
    pub systematics: Vec<SystematicSource>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationManifest {
    pub schema_version: String,
    #[serde(default = "default_measurement_combination_manifest_poi")]
    pub poi: String,
    pub measurements_table: String,
    pub stat_covariance_table: String,
    #[serde(default)]
    pub systematics_table: Option<String>,
    #[serde(default)]
    pub correlations_table: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementInput {
    pub name: String,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystematicSource {
    pub name: String,
    pub magnitudes: Vec<f64>,
    pub corr: Vec<Vec<f64>>,
    #[serde(default)]
    pub error_on_error: f64,
    #[serde(default)]
    pub aux_mean: f64,
}

fn default_measurement_combination_manifest_poi() -> String {
    "mu".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationResult {
    pub schema_version: String,
    pub poi: String,
    pub mu_hat: f64,
    pub confidence_interval: ConfidenceInterval,
    pub goodness_of_fit: GoodnessOfFit,
    pub converged: bool,
    pub stability: String,
    pub optimizer: OptimizerDiagnostics,
    pub diagnostics: ResearchDiagnostics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationReport {
    pub schema_version: String,
    pub poi: String,
    pub ci_level: f64,
    pub n_toys: usize,
    pub seed: u64,
    pub stability: String,
    pub reference: MeasurementCombinationResult,
    pub summary: MeasurementCombinationCalibrationSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationSummary {
    pub df: usize,
    pub mean_q: f64,
    pub mean_q_star: f64,
    pub sd_q: f64,
    pub sd_q_star: f64,
    pub sem_q: f64,
    pub sem_q_star: f64,
    pub mean_q_abs_error_to_df: f64,
    pub mean_q_star_abs_error_to_df: f64,
    pub bartlett_improves_mean_q: bool,
    pub mean_sigma: f64,
    pub mean_sigma_star: f64,
    pub mean_sigma_star_to_sigma_ratio: f64,
    pub sigma_star_ge_sigma_fraction: f64,
    pub toy_generation_method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationStudyReport {
    pub schema_version: String,
    pub poi: String,
    pub ci_level: f64,
    pub n_toys: usize,
    pub seeds: Vec<u64>,
    pub stability: String,
    pub reference: MeasurementCombinationResult,
    pub per_seed: Vec<MeasurementCombinationCalibrationSeedReport>,
    pub aggregate: MeasurementCombinationCalibrationStudySummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationSeedReport {
    pub seed: u64,
    pub summary: MeasurementCombinationCalibrationSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationStudySummary {
    pub n_runs: usize,
    pub bartlett_improves_mean_q_fraction: f64,
    pub mean_of_mean_q: f64,
    pub mean_of_mean_q_star: f64,
    pub mean_of_mean_sigma: f64,
    pub mean_of_mean_sigma_star: f64,
    pub min_mean_sigma: f64,
    pub max_mean_sigma: f64,
    pub min_mean_sigma_star: f64,
    pub max_mean_sigma_star: f64,
    pub min_mean_sigma_star_to_sigma_ratio: f64,
    pub max_mean_sigma_star_to_sigma_ratio: f64,
    pub max_abs_mean_sigma_star_to_sigma_ratio_delta_from_reference: f64,
    pub min_sigma_star_ge_sigma_fraction: f64,
    pub max_sigma_star_ge_sigma_fraction: f64,
    pub toy_generation_method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationScenarioStudySpec {
    pub schema_version: String,
    pub scenarios: Vec<MeasurementCombinationScenarioSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationScenarioSpec {
    pub name: String,
    #[serde(default)]
    pub error_on_error: Vec<ScenarioErrorOnErrorAssignment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioErrorOnErrorAssignment {
    pub systematic: String,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationScenarioStudyReport {
    pub schema_version: String,
    pub poi: String,
    pub ci_level: f64,
    pub stability: String,
    pub baseline: MeasurementCombinationResult,
    pub scenarios: Vec<MeasurementCombinationScenarioResult>,
    pub aggregate: MeasurementCombinationScenarioStudySummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationScenarioResult {
    pub name: String,
    pub assignments: Vec<ScenarioErrorOnErrorAssignment>,
    pub result: MeasurementCombinationResult,
    pub comparison: MeasurementCombinationScenarioComparison,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationScenarioComparison {
    pub mu_shift_from_baseline: f64,
    pub abs_mu_shift_from_baseline: f64,
    pub sigma_ratio_to_baseline: f64,
    pub interval_width_ratio_to_baseline: f64,
    pub chi2_delta_from_baseline: f64,
    pub max_perturbative_condition: Option<f64>,
    pub all_perturbative_within_threshold: bool,
    pub supported_systematics: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationScenarioStudySummary {
    pub n_scenarios: usize,
    pub all_converged: bool,
    pub all_perturbative_within_threshold: bool,
    pub min_sigma_ratio_to_baseline: f64,
    pub max_sigma_ratio_to_baseline: f64,
    pub largest_abs_mu_shift_scenario: String,
    pub largest_abs_mu_shift: f64,
    pub widest_interval_scenario: String,
    pub widest_interval_ratio_to_baseline: f64,
    pub max_supported_systematics: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationScenarioStudySolverParityReport {
    pub schema_version: String,
    pub poi: String,
    pub ci_level: f64,
    pub lhs_solver: String,
    pub rhs_solver: String,
    pub stability: String,
    pub baseline: MeasurementCombinationSolverParityBaseline,
    pub scenarios: Vec<MeasurementCombinationScenarioSolverParityEntry>,
    pub aggregate: MeasurementCombinationScenarioSolverParitySummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationSolverParityBaseline {
    pub lhs_optimizer_method: String,
    pub rhs_optimizer_method: String,
    pub mu_abs_diff: f64,
    pub sigma_abs_diff: f64,
    pub sigma_rel_diff: f64,
    pub chi2_abs_diff: f64,
    pub q_star_abs_diff: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationScenarioSolverParityEntry {
    pub name: String,
    pub assignments: Vec<ScenarioErrorOnErrorAssignment>,
    pub lhs_optimizer_method: String,
    pub rhs_optimizer_method: String,
    pub mu_abs_diff: f64,
    pub sigma_abs_diff: f64,
    pub sigma_rel_diff: f64,
    pub chi2_abs_diff: f64,
    pub q_star_abs_diff: Option<f64>,
    pub same_supported_systematics: bool,
    pub lhs_supported_systematics: Vec<String>,
    pub rhs_supported_systematics: Vec<String>,
    pub lhs_all_perturbative_within_threshold: bool,
    pub rhs_all_perturbative_within_threshold: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationScenarioSolverParitySummary {
    pub n_scenarios: usize,
    pub max_mu_abs_diff: f64,
    pub max_mu_abs_diff_scenario: String,
    pub max_sigma_rel_diff: f64,
    pub max_sigma_rel_diff_scenario: String,
    pub max_q_star_abs_diff: f64,
    pub max_q_star_abs_diff_scenario: String,
    pub all_scenarios_converged: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationScenarioStudySolverParityDigest {
    pub schema_version: String,
    pub source_schema_version: String,
    pub poi: String,
    pub ci_level: f64,
    pub lhs_solver: String,
    pub rhs_solver: String,
    pub stability: String,
    pub baseline: MeasurementCombinationSolverParityBaseline,
    pub dominant_mu_gap_scenario: String,
    pub dominant_sigma_gap_scenario: String,
    pub dominant_q_star_gap_scenario: String,
    pub aggregate: MeasurementCombinationScenarioStudySolverParityDigestSummary,
    pub scenarios: Vec<MeasurementCombinationScenarioStudySolverParityDigestEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationScenarioStudySolverParityDigestSummary {
    pub n_scenarios: usize,
    pub max_mu_abs_diff: f64,
    pub max_sigma_rel_diff: f64,
    pub max_q_star_abs_diff: f64,
    pub n_supported_systematics_mismatch_scenarios: usize,
    pub n_perturbative_overlap_failure_scenarios: usize,
    pub supported_systematics_mismatch_scenarios: Vec<String>,
    pub perturbative_overlap_failure_scenarios: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationScenarioStudySolverParityDigestEntry {
    pub name: String,
    pub n_assignments: usize,
    pub mu_gap_rank: usize,
    pub sigma_gap_rank: usize,
    pub q_star_gap_rank: usize,
    pub mu_abs_diff: f64,
    pub sigma_rel_diff: f64,
    pub q_star_abs_diff: Option<f64>,
    pub same_supported_systematics: bool,
    pub both_perturbative_within_threshold: bool,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignReport {
    pub schema_version: String,
    pub poi: String,
    pub ci_level: f64,
    pub n_toys: usize,
    pub seeds: Vec<u64>,
    pub stability: String,
    pub baseline: MeasurementCombinationResult,
    pub scenarios: Vec<MeasurementCombinationCalibrationCampaignScenario>,
    pub aggregate: MeasurementCombinationCalibrationCampaignSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignScenario {
    pub name: String,
    pub assignments: Vec<ScenarioErrorOnErrorAssignment>,
    pub fit: MeasurementCombinationResult,
    pub calibration: MeasurementCombinationCalibrationStudySummary,
    pub comparison: MeasurementCombinationCalibrationCampaignComparison,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignComparison {
    pub fit_sigma_ratio_to_baseline: f64,
    pub fit_interval_width_ratio_to_baseline: f64,
    pub calibration_min_mean_sigma_star_to_sigma_ratio: f64,
    pub calibration_max_mean_sigma_star_to_sigma_ratio: f64,
    pub calibration_max_abs_ratio_delta_from_reference: f64,
    pub calibration_min_sigma_star_ge_sigma_fraction: f64,
    pub bartlett_improves_mean_q_fraction: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignSummary {
    pub n_scenarios: usize,
    pub all_converged: bool,
    pub all_calibration_sigma_star_ge_sigma_fraction_ge_0_99: bool,
    pub max_fit_sigma_ratio_to_baseline: f64,
    pub min_fit_sigma_ratio_to_baseline: f64,
    pub max_calibration_mean_sigma_star_to_sigma_ratio: f64,
    pub min_calibration_mean_sigma_star_to_sigma_ratio: f64,
    pub widest_fit_interval_scenario: String,
    pub highest_calibration_sigma_ratio_scenario: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignSolverParityReport {
    pub schema_version: String,
    pub poi: String,
    pub ci_level: f64,
    pub n_toys: usize,
    pub seeds: Vec<u64>,
    pub lhs_solver: String,
    pub rhs_solver: String,
    pub stability: String,
    pub baseline: MeasurementCombinationSolverParityBaseline,
    pub scenarios: Vec<MeasurementCombinationCalibrationCampaignSolverParityEntry>,
    pub aggregate: MeasurementCombinationCalibrationCampaignSolverParitySummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignSolverParityEntry {
    pub name: String,
    pub assignments: Vec<ScenarioErrorOnErrorAssignment>,
    pub lhs_fit_optimizer_method: String,
    pub rhs_fit_optimizer_method: String,
    pub fit_mu_abs_diff: f64,
    pub fit_sigma_abs_diff: f64,
    pub fit_sigma_rel_diff: f64,
    pub fit_q_star_abs_diff: Option<f64>,
    pub mean_sigma_star_to_sigma_ratio_center_abs_diff: f64,
    pub sigma_star_ge_sigma_fraction_abs_diff: f64,
    pub bartlett_improves_mean_q_fraction_abs_diff: f64,
    pub lhs_toy_generation_method: String,
    pub rhs_toy_generation_method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignSolverParitySummary {
    pub n_scenarios: usize,
    pub max_fit_mu_abs_diff: f64,
    pub max_fit_mu_abs_diff_scenario: String,
    pub max_fit_sigma_rel_diff: f64,
    pub max_fit_sigma_rel_diff_scenario: String,
    pub max_fit_q_star_abs_diff: f64,
    pub max_fit_q_star_abs_diff_scenario: String,
    pub max_calibration_ratio_center_abs_diff: f64,
    pub max_calibration_ratio_center_abs_diff_scenario: String,
    pub all_scenarios_converged: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignSolverParityDigest {
    pub schema_version: String,
    pub source_schema_version: String,
    pub poi: String,
    pub ci_level: f64,
    pub lhs_solver: String,
    pub rhs_solver: String,
    pub stability: String,
    pub n_toys: usize,
    pub seeds: Vec<u64>,
    pub baseline: MeasurementCombinationSolverParityBaseline,
    pub dominant_fit_gap_scenario: String,
    pub dominant_calibration_gap_scenario: String,
    pub aggregate: MeasurementCombinationCalibrationCampaignSolverParityDigestSummary,
    pub scenarios: Vec<MeasurementCombinationCalibrationCampaignSolverParityDigestEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignSolverParityDigestSummary {
    pub n_scenarios: usize,
    pub max_fit_mu_abs_diff: f64,
    pub max_fit_sigma_rel_diff: f64,
    pub max_fit_q_star_abs_diff: f64,
    pub max_calibration_ratio_center_abs_diff: f64,
    pub n_toy_generation_method_mismatch_scenarios: usize,
    pub toy_generation_method_mismatch_scenarios: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignSolverParityDigestEntry {
    pub name: String,
    pub n_assignments: usize,
    pub fit_gap_rank: usize,
    pub calibration_gap_rank: usize,
    pub fit_mu_abs_diff: f64,
    pub fit_sigma_rel_diff: f64,
    pub fit_q_star_abs_diff: Option<f64>,
    pub calibration_ratio_center_abs_diff: f64,
    pub sigma_star_ge_sigma_fraction_abs_diff: f64,
    pub bartlett_improves_mean_q_fraction_abs_diff: f64,
    pub same_toy_generation_method: bool,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignDigest {
    pub schema_version: String,
    pub source_schema_version: String,
    pub poi: String,
    pub ci_level: f64,
    pub stability: String,
    pub baseline_mu_hat: f64,
    pub baseline_sigma: f64,
    pub dominant_fit_scenario: String,
    pub dominant_calibration_scenario: String,
    pub aggregate: MeasurementCombinationCalibrationCampaignDigestSummary,
    pub scenarios: Vec<MeasurementCombinationCalibrationCampaignDigestScenario>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignDigestSummary {
    pub n_scenarios: usize,
    pub n_calibration_neutral_scenarios: usize,
    pub max_fit_sigma_ratio_to_baseline: f64,
    pub max_calibration_mean_sigma_star_to_sigma_ratio: f64,
    pub near_neutral_calibration_scenarios: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignDigestScenario {
    pub name: String,
    pub n_assignments: usize,
    pub fit_rank_by_sigma: usize,
    pub calibration_rank_by_sigma_ratio: usize,
    pub fit_sigma_ratio_to_baseline: f64,
    pub fit_sigma_delta_from_baseline: f64,
    pub fit_interval_width_ratio_to_baseline: f64,
    pub calibration_mean_sigma_star_to_sigma_ratio_center: f64,
    pub calibration_mean_sigma_star_to_sigma_ratio_span: f64,
    pub calibration_min_sigma_star_ge_sigma_fraction: f64,
    pub bartlett_improves_mean_q_fraction: f64,
    pub supported_systematics: Vec<String>,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignBrief {
    pub schema_version: String,
    pub source_schema_version: String,
    pub stability: String,
    pub shared_poi: Option<String>,
    pub pois: Vec<String>,
    pub entries: Vec<MeasurementCombinationCalibrationCampaignBriefEntry>,
    pub aggregate: MeasurementCombinationCalibrationCampaignBriefSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignBriefEntry {
    pub label: String,
    pub poi: String,
    pub dominant_fit_scenario: String,
    pub dominant_calibration_scenario: String,
    pub max_fit_sigma_ratio_to_baseline: f64,
    pub max_calibration_mean_sigma_star_to_sigma_ratio: f64,
    pub n_near_neutral_calibration_scenarios: usize,
    pub near_neutral_calibration_scenarios: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignBriefSummary {
    pub n_artifacts: usize,
    pub labels: Vec<String>,
    pub highest_fit_inflation_label: String,
    pub highest_calibration_inflation_label: String,
    pub labels_with_near_neutral_calibration: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignFamilyReport {
    pub schema_version: String,
    pub source_schema_version: String,
    pub stability: String,
    pub shared_poi: Option<String>,
    pub pois: Vec<String>,
    pub families: Vec<MeasurementCombinationCalibrationCampaignFamilyReportEntry>,
    pub aggregate: MeasurementCombinationCalibrationCampaignFamilyReportSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignFamilyReportEntry {
    pub label: String,
    pub shared_poi: Option<String>,
    pub pois: Vec<String>,
    pub n_artifacts: usize,
    pub artifact_labels: Vec<String>,
    pub highest_fit_inflation_artifact: String,
    pub highest_fit_inflation_value: f64,
    pub highest_calibration_inflation_artifact: String,
    pub highest_calibration_inflation_value: f64,
    pub labels_with_near_neutral_calibration: Vec<String>,
    pub has_mixed_pois: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignFamilyReportSummary {
    pub n_families: usize,
    pub n_total_artifacts: usize,
    pub family_labels: Vec<String>,
    pub family_with_highest_fit_inflation: String,
    pub family_with_highest_calibration_inflation: String,
    pub families_with_mixed_pois: Vec<String>,
    pub families_with_near_neutral_calibration: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignFamilyMatrix {
    pub schema_version: String,
    pub source_schema_version: String,
    pub stability: String,
    pub shared_poi: Option<String>,
    pub pois: Vec<String>,
    pub families: Vec<MeasurementCombinationCalibrationCampaignFamilyMatrixEntry>,
    pub pairwise: Vec<MeasurementCombinationCalibrationCampaignFamilyPairwiseRelation>,
    pub aggregate: MeasurementCombinationCalibrationCampaignFamilyMatrixSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignFamilyMatrixEntry {
    pub label: String,
    pub highest_fit_inflation_value: f64,
    pub highest_calibration_inflation_value: f64,
    pub joint_severity_score: f64,
    pub fit_rank: usize,
    pub calibration_rank: usize,
    pub joint_rank: usize,
    pub n_artifacts: usize,
    pub has_mixed_pois: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignFamilyPairwiseRelation {
    pub lhs: String,
    pub rhs: String,
    pub lhs_fit_minus_rhs: f64,
    pub lhs_calibration_minus_rhs: f64,
    pub lhs_joint_minus_rhs: f64,
    pub fit_dominance: String,
    pub calibration_dominance: String,
    pub joint_dominance: String,
    pub same_poi_coverage: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignFamilyMatrixSummary {
    pub n_families: usize,
    pub fit_order: Vec<String>,
    pub calibration_order: Vec<String>,
    pub joint_order: Vec<String>,
    pub family_with_highest_joint_severity: String,
    pub families_with_mixed_pois: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignPortfolioReport {
    pub schema_version: String,
    pub source_schema_version: String,
    pub stability: String,
    pub shared_poi: Option<String>,
    pub pois: Vec<String>,
    pub entries: Vec<MeasurementCombinationCalibrationCampaignPortfolioEntry>,
    pub pairwise: Vec<MeasurementCombinationCalibrationCampaignPortfolioPairwiseRelation>,
    pub aggregate: MeasurementCombinationCalibrationCampaignPortfolioSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignPortfolioEntry {
    pub label: String,
    pub shared_poi: Option<String>,
    pub pois: Vec<String>,
    pub n_families: usize,
    pub family_leader: String,
    pub max_fit_inflation: f64,
    pub max_calibration_inflation: f64,
    pub max_joint_severity: f64,
    pub fit_order: Vec<String>,
    pub calibration_order: Vec<String>,
    pub joint_order: Vec<String>,
    pub has_mixed_pois: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignPortfolioPairwiseRelation {
    pub lhs: String,
    pub rhs: String,
    pub lhs_fit_minus_rhs: f64,
    pub lhs_calibration_minus_rhs: f64,
    pub lhs_joint_minus_rhs: f64,
    pub fit_dominance: String,
    pub calibration_dominance: String,
    pub joint_dominance: String,
    pub same_poi_coverage: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignPortfolioSummary {
    pub n_portfolios: usize,
    pub portfolio_labels: Vec<String>,
    pub portfolio_with_highest_fit_inflation: String,
    pub portfolio_with_highest_calibration_inflation: String,
    pub portfolio_with_highest_joint_severity: String,
    pub portfolios_with_mixed_pois: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignPortfolioStabilityReport {
    pub schema_version: String,
    pub source_schema_version: String,
    pub stability: String,
    pub shared_poi: Option<String>,
    pub pois: Vec<String>,
    pub runs: Vec<MeasurementCombinationCalibrationCampaignPortfolioStabilityRun>,
    pub pairwise: Vec<MeasurementCombinationCalibrationCampaignPortfolioStabilityPairwiseRelation>,
    pub aggregate: MeasurementCombinationCalibrationCampaignPortfolioStabilitySummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignPortfolioStabilityRun {
    pub label: String,
    pub shared_poi: Option<String>,
    pub pois: Vec<String>,
    pub n_portfolios: usize,
    pub fit_leader: String,
    pub calibration_leader: String,
    pub joint_leader: String,
    pub fit_order: Vec<String>,
    pub calibration_order: Vec<String>,
    pub joint_order: Vec<String>,
    pub has_mixed_pois: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignPortfolioStabilityPairwiseRelation {
    pub lhs: String,
    pub rhs: String,
    pub same_poi_coverage: bool,
    pub same_portfolio_labels: bool,
    pub same_fit_leader: bool,
    pub same_calibration_leader: bool,
    pub same_joint_leader: bool,
    pub same_fit_order: bool,
    pub same_calibration_order: bool,
    pub same_joint_order: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCombinationCalibrationCampaignPortfolioStabilitySummary {
    pub n_runs: usize,
    pub run_labels: Vec<String>,
    pub reference_run: String,
    pub stable_fit_leader: bool,
    pub stable_calibration_leader: bool,
    pub stable_joint_leader: bool,
    pub stable_fit_order: bool,
    pub stable_calibration_order: bool,
    pub stable_joint_order: bool,
    pub runs_with_mixed_pois: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub ci_level: f64,
    pub lower: f64,
    pub upper: f64,
    pub sigma: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoodnessOfFit {
    pub chi2: f64,
    pub df: usize,
    pub p_value: Option<f64>,
    pub method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerDiagnostics {
    pub method: String,
    pub n_iter: usize,
    pub n_fev: usize,
    pub n_gev: usize,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchDiagnostics {
    pub input_schema_version: String,
    pub n_measurements: usize,
    pub n_systematics: usize,
    pub requested_error_on_error: bool,
    pub supports_error_on_error: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_solver: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effective_solver: Option<String>,
    pub corr_regularization_deltas: Vec<f64>,
    pub profiled_variance_scales: Vec<f64>,
    pub theta_l2_norms: Vec<f64>,
    pub perturbative_validity: PerturbativeValidityDiagnostics,
    pub bartlett: BartlettDiagnostics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BartlettDiagnostics {
    pub supported: bool,
    pub method: String,
    pub unsupported_reason: Option<String>,
    pub supported_systematics: Vec<String>,
    pub b_mu_theta: Option<f64>,
    pub b_tilde_theta: Option<f64>,
    pub b_mu: Option<f64>,
    pub b_q: Option<f64>,
    pub w_mu_scale: Option<f64>,
    pub q_scale: Option<f64>,
    pub q_star: Option<f64>,
    pub p_value_star: Option<f64>,
    pub sigma_scale: Option<f64>,
    pub sigma_star: Option<f64>,
    pub sigma2_unbiased_estimates: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbativeValidityDiagnostics {
    pub threshold: f64,
    pub systematic_names: Vec<String>,
    pub condition_values: Vec<f64>,
    pub within_threshold: Vec<bool>,
}

#[derive(Clone)]
struct SystematicPrepared {
    name: String,
    magnitudes: DVector<f64>,
    corr_cov: DMatrix<f64>,
    corr_reg: DMatrix<f64>,
    corr_factor: DMatrix<f64>,
    corr_precision: DMatrix<f64>,
    effect_matrix: DMatrix<f64>,
    regularization_delta: f64,
    error_on_error: f64,
    aux_mean: f64,
}

#[derive(Clone)]
struct Layout {
    theta_ranges: Vec<Range<usize>>,
    tau_indices: Vec<Option<usize>>,
    n_params: usize,
}

struct PreparedSpecShared {
    poi: String,
    ones: DVector<f64>,
    base_cov: DMatrix<f64>,
    base_precision: DMatrix<f64>,
    systematics: Vec<SystematicPrepared>,
    corr_regularization_deltas: Vec<f64>,
    n_systematics_total: usize,
    layout: Layout,
    mu_bounds: (f64, f64),
    fixed_sigma_guess: f64,
}

#[derive(Clone)]
struct PreparedSpec {
    shared: Arc<PreparedSpecShared>,
    y: DVector<f64>,
}

impl Deref for PreparedSpec {
    type Target = PreparedSpecShared;

    fn deref(&self) -> &Self::Target {
        &self.shared
    }
}

struct MeasurementCombineObjective<'a> {
    prep: &'a PreparedSpec,
    cache: Mutex<Option<ObjectiveEvalCache>>,
}

#[derive(Clone)]
struct ProfiledGvmState {
    theta_original: Vec<DVector<f64>>,
    profiled_variance_scales: Vec<f64>,
    theta_l2_norms: Vec<f64>,
}

struct NumericalGvmFit {
    fit: OptimizationResult,
    state: ProfiledGvmState,
}

struct AnalyticProfilePoint {
    mu: f64,
    nll: f64,
    state: ProfiledGvmState,
}

// `order` counts refinement steps in the epsilon^2 expansion. In particular,
// `order=1` means one self-consistent perturbative refinement step, which
// includes the leading O(epsilon^2) corrections; it does not mean O(epsilon).
struct AnalyticPerturbativeSolver<'a> {
    prep: &'a PreparedSpec,
    order: usize,
}

#[derive(Clone)]
struct PaperLayout {
    theta_ranges: Vec<Range<usize>>,
    tau_indices: Vec<usize>,
    n_params: usize,
}

struct PaperMeasurementCombineObjective<'a> {
    prep: &'a PreparedSpec,
    layout: PaperLayout,
    cache: Mutex<Option<ObjectiveEvalCache>>,
}

#[derive(Clone)]
struct ObjectiveEvalCache {
    params: Vec<f64>,
    cost: f64,
    gradient: Vec<f64>,
}

struct PaperNumericalGvmFit {
    fit: OptimizationResult,
    state: ProfiledGvmState,
}

#[derive(Debug, Clone, Copy, Default)]
struct ProfileBoundWorkload {
    n_profile_fits: usize,
    bracket_fits: usize,
    bisect_fits: usize,
    total_n_iter: u64,
    total_n_fev: usize,
    total_n_gev: usize,
}

impl ProfileBoundWorkload {
    fn record(&mut self, prof: &OptimizationResult, phase: ProfileBoundPhase) {
        self.n_profile_fits += 1;
        match phase {
            ProfileBoundPhase::Bracket => self.bracket_fits += 1,
            ProfileBoundPhase::Bisect => self.bisect_fits += 1,
        }
        self.total_n_iter += prof.n_iter;
        self.total_n_fev += prof.n_fev;
        self.total_n_gev += prof.n_gev;
    }
}

#[derive(Debug, Clone, Copy)]
enum ProfileBoundPhase {
    Bracket,
    Bisect,
}

#[derive(Debug, Clone, Copy, Default)]
struct ProfileBoundResult {
    mu: f64,
    workload: ProfileBoundWorkload,
}

#[derive(Debug, Clone, Copy, Default)]
struct ProfileCiWorkload {
    lower: ProfileBoundWorkload,
    upper: ProfileBoundWorkload,
}

fn paper_profile_scan_optimizer_config() -> OptimizerConfig {
    // Profile scans repeatedly solve nuisance-only fits while only needing stable q(mu)
    // ordering and interval crossings, so we can relax the MLE-level HighPrecision
    // tolerance slightly without changing the public API surface.
    OptimizerConfig { max_iter: 2000, tol: 1e-7, m: 10, smooth_bounds: false }
}

fn paper_profile_bracket_optimizer_config() -> OptimizerConfig {
    // The initial bracket fit only needs to establish a target crossing, not the final
    // interval endpoint. A lighter L-BFGS-B setup materially reduces the dominant first
    // profiled fit cost on large original-theta problems while preserving the crossing.
    OptimizerConfig { max_iter: 500, tol: 1e-5, m: 5, smooth_bounds: false }
}

fn default_numerical_paper_warm_start(
    objective: &PaperMeasurementCombineObjective,
    ci_level: f64,
) -> Option<PaperWarmStartGuide> {
    let nm = objective.prep.y.len().saturating_mul(objective.prep.n_systematics_total);
    // On large original-theta problems the analytic warm-start guide itself can dominate total
    // runtime; beyond this threshold, a direct numerical-paper fit is cheaper overall.
    if nm >= NUMERICAL_PAPER_ANALYTIC_WARM_START_NM_THRESHOLD {
        None
    } else {
        objective.analytic_warm_start_guide(ci_level)
    }
}

#[derive(Clone)]
struct PaperProfileBoundHint {
    mu: f64,
    params: Vec<f64>,
}

#[derive(Clone)]
struct PaperWarmStartGuide {
    mle_params: Vec<f64>,
    lower_hint: Option<PaperProfileBoundHint>,
    upper_hint: Option<PaperProfileBoundHint>,
}

enum AnalyticPerturbativeAttempt {
    Completed(Box<MeasurementCombinationResult>),
    FallbackWarmStart(Option<PaperWarmStartGuide>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BartlettWorkspaceBuildPath {
    Reference,
    Fast,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AnalyticLinearSolvePath {
    Reference,
    Fast,
}

struct BartlettWorkspace {
    n_per_source: usize,
    theta_hat_original: Vec<DVector<f64>>,
    j_blocks: Vec<DMatrix<f64>>,
    j_tilde_blocks: Vec<DMatrix<f64>>,
    build_path: BartlettWorkspaceBuildPath,
}

#[derive(Debug, Deserialize)]
struct MeasurementTableRow {
    name: String,
    value: f64,
}

#[derive(Debug, Deserialize)]
struct SystematicTableRow {
    systematic: String,
    measurement: String,
    magnitude: f64,
    #[serde(default)]
    error_on_error: Option<f64>,
    #[serde(default)]
    aux_mean: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct CorrelationTableRow {
    systematic: String,
    row_measurement: String,
    col_measurement: String,
    corr: f64,
}

#[derive(Debug, Clone)]
struct SystematicTableAccumulator {
    magnitudes: BTreeMap<String, f64>,
    error_on_error: Option<f64>,
    aux_mean: Option<f64>,
}

impl MeasurementCombinationSpec {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != MEASUREMENT_COMBINATION_SCHEMA_V0 {
            return Err(Error::Validation(format!(
                "unsupported schema_version for measurement combination: got={} expected={}",
                self.schema_version, MEASUREMENT_COMBINATION_SCHEMA_V0
            )));
        }
        if self.poi.trim().is_empty() {
            return Err(Error::Validation("poi must be non-empty".to_string()));
        }
        if self.measurements.is_empty() {
            return Err(Error::Validation(
                "measurement combination requires at least one measurement".to_string(),
            ));
        }
        let n = self.measurements.len();
        validate_square_symmetric("stat_covariance", &self.stat_covariance, n, false, true)?;

        for (idx, syst) in self.systematics.iter().enumerate() {
            if syst.name.trim().is_empty() {
                return Err(Error::Validation(format!(
                    "systematics[{idx}].name must be non-empty"
                )));
            }
            if syst.magnitudes.len() != n {
                return Err(Error::Validation(format!(
                    "systematics[{idx}].magnitudes length mismatch: got={} expected={}",
                    syst.magnitudes.len(),
                    n
                )));
            }
            if syst.error_on_error < 0.0 {
                return Err(Error::Validation(format!(
                    "systematics[{idx}].error_on_error must be >= 0"
                )));
            }
            validate_square_symmetric(
                &format!("systematics[{idx}].corr"),
                &syst.corr,
                n,
                true,
                false,
            )?;
        }
        Ok(())
    }
}

impl MeasurementCombinationManifest {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version.trim() != MEASUREMENT_COMBINATION_MANIFEST_SCHEMA_V0 {
            return Err(Error::Validation(format!(
                "schema_version must be '{}'",
                MEASUREMENT_COMBINATION_MANIFEST_SCHEMA_V0
            )));
        }
        if self.poi.trim().is_empty() {
            return Err(Error::Validation("poi must be non-empty".to_string()));
        }
        if self.measurements_table.trim().is_empty() {
            return Err(Error::Validation("measurements_table must be non-empty".to_string()));
        }
        if self.stat_covariance_table.trim().is_empty() {
            return Err(Error::Validation("stat_covariance_table must be non-empty".to_string()));
        }
        if let Some(systematics) = &self.systematics_table
            && systematics.trim().is_empty()
        {
            return Err(Error::Validation(
                "systematics_table must be non-empty when provided".to_string(),
            ));
        }
        if let Some(correlations) = &self.correlations_table
            && correlations.trim().is_empty()
        {
            return Err(Error::Validation(
                "correlations_table must be non-empty when provided".to_string(),
            ));
        }
        Ok(())
    }
}

pub fn build_measurement_combination_spec_from_tables(
    poi: &str,
    measurements_table: &str,
    stat_covariance_table: &str,
    systematics_table: Option<&str>,
    correlations_table: Option<&str>,
) -> Result<MeasurementCombinationSpec> {
    if poi.trim().is_empty() {
        return Err(Error::Validation("poi must be non-empty".to_string()));
    }

    let measurements = parse_measurement_rows(measurements_table)?;
    let measurement_names: Vec<String> = measurements.iter().map(|m| m.name.clone()).collect();
    let stat_covariance = parse_named_square_matrix_table(
        stat_covariance_table,
        &measurement_names,
        "stat_covariance",
    )?;
    let systematics =
        parse_systematics_tables(systematics_table, correlations_table, &measurement_names)?;

    let spec = MeasurementCombinationSpec {
        schema_version: MEASUREMENT_COMBINATION_SCHEMA_V0.to_string(),
        poi: poi.trim().to_string(),
        measurements,
        stat_covariance,
        systematics,
    };
    spec.validate()?;
    Ok(spec)
}

pub fn build_measurement_combination_spec_from_manifest_path(
    manifest_path: &Path,
) -> Result<MeasurementCombinationSpec> {
    let manifest_text = std::fs::read_to_string(manifest_path).map_err(|e| {
        Error::Validation(format!(
            "failed to read measurement-combination manifest '{}': {e}",
            manifest_path.display()
        ))
    })?;
    let manifest = parse_measurement_combination_manifest(manifest_path, &manifest_text)?;
    let base_dir =
        manifest_path.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
    let measurements_path =
        resolve_measurement_combination_manifest_path(&base_dir, &manifest.measurements_table)?;
    let stat_covariance_path =
        resolve_measurement_combination_manifest_path(&base_dir, &manifest.stat_covariance_table)?;
    let systematics_path = manifest
        .systematics_table
        .as_deref()
        .map(|path| resolve_measurement_combination_manifest_path(&base_dir, path))
        .transpose()?;
    let correlations_path = manifest
        .correlations_table
        .as_deref()
        .map(|path| resolve_measurement_combination_manifest_path(&base_dir, path))
        .transpose()?;

    let measurements_text = std::fs::read_to_string(&measurements_path).map_err(|e| {
        Error::Validation(format!(
            "failed to read manifest measurements table '{}': {e}",
            measurements_path.display()
        ))
    })?;
    let stat_covariance_text = std::fs::read_to_string(&stat_covariance_path).map_err(|e| {
        Error::Validation(format!(
            "failed to read manifest stat_covariance table '{}': {e}",
            stat_covariance_path.display()
        ))
    })?;
    let systematics_text = systematics_path
        .as_ref()
        .map(|path| {
            std::fs::read_to_string(path).map_err(|e| {
                Error::Validation(format!(
                    "failed to read manifest systematics table '{}': {e}",
                    path.display()
                ))
            })
        })
        .transpose()?;
    let correlations_text = correlations_path
        .as_ref()
        .map(|path| {
            std::fs::read_to_string(path).map_err(|e| {
                Error::Validation(format!(
                    "failed to read manifest correlations table '{}': {e}",
                    path.display()
                ))
            })
        })
        .transpose()?;

    build_measurement_combination_spec_from_tables(
        &manifest.poi,
        &measurements_text,
        &stat_covariance_text,
        systematics_text.as_deref(),
        correlations_text.as_deref(),
    )
}

fn parse_measurement_combination_manifest(
    manifest_path: &Path,
    manifest_text: &str,
) -> Result<MeasurementCombinationManifest> {
    let extension =
        manifest_path.extension().and_then(|ext| ext.to_str()).map(|ext| ext.to_ascii_lowercase());
    let manifest = match extension.as_deref() {
        Some("json") => serde_json::from_str::<MeasurementCombinationManifest>(manifest_text)
            .map_err(|e| {
                Error::Validation(format!(
                    "invalid measurement-combination manifest JSON '{}': {e}",
                    manifest_path.display()
                ))
            })?,
        _ => serde_yaml_ng::from_str::<MeasurementCombinationManifest>(manifest_text)
            .or_else(|yaml_err| {
                serde_json::from_str::<MeasurementCombinationManifest>(manifest_text).map_err(
                    |json_err| {
                        Error::Validation(format!(
                            "invalid measurement-combination manifest '{}': yaml={yaml_err}; json={json_err}",
                            manifest_path.display()
                        ))
                    },
                )
            })?,
    };
    manifest.validate()?;
    Ok(manifest)
}

fn resolve_measurement_combination_manifest_path(
    base_dir: &Path,
    raw_path: &str,
) -> Result<PathBuf> {
    let trimmed = raw_path.trim();
    if trimmed.is_empty() {
        return Err(Error::Validation(
            "measurement-combination manifest path entries must be non-empty".to_string(),
        ));
    }
    let path = Path::new(trimmed);
    Ok(if path.is_absolute() { path.to_path_buf() } else { base_dir.join(path) })
}

fn infer_table_delimiter(table: &str) -> u8 {
    let first = table.lines().map(str::trim).find(|line| !line.is_empty()).unwrap_or("");
    if first.contains('\t') && !first.contains(',') { b'\t' } else { b',' }
}

fn parse_measurement_rows(table: &str) -> Result<Vec<MeasurementInput>> {
    let delimiter = infer_table_delimiter(table);
    let mut reader = csv::ReaderBuilder::new()
        .trim(Trim::All)
        .delimiter(delimiter)
        .from_reader(table.as_bytes());
    let mut seen = BTreeSet::new();
    let mut rows = Vec::new();
    for row in reader.deserialize::<MeasurementTableRow>() {
        let row = row.map_err(|e| Error::Validation(format!("invalid measurements table: {e}")))?;
        if row.name.trim().is_empty() {
            return Err(Error::Validation(
                "measurements table contains an empty measurement name".to_string(),
            ));
        }
        if !seen.insert(row.name.clone()) {
            return Err(Error::Validation(format!(
                "measurements table contains duplicate measurement '{}'",
                row.name
            )));
        }
        rows.push(MeasurementInput { name: row.name, value: row.value });
    }
    if rows.is_empty() {
        return Err(Error::Validation(
            "measurements table must contain at least one row".to_string(),
        ));
    }
    Ok(rows)
}

fn parse_named_square_matrix_table(
    table: &str,
    expected_names: &[String],
    label: &str,
) -> Result<Vec<Vec<f64>>> {
    let delimiter = infer_table_delimiter(table);
    let mut reader = csv::ReaderBuilder::new()
        .trim(Trim::All)
        .has_headers(false)
        .delimiter(delimiter)
        .from_reader(table.as_bytes());
    let mut records = reader.records();
    let header = records
        .next()
        .ok_or_else(|| Error::Validation(format!("{label} table is empty")))?
        .map_err(|e| Error::Validation(format!("invalid {label} table: {e}")))?;
    if header.len() < 2 {
        return Err(Error::Validation(format!(
            "{label} table must have a leading row-name column and at least one data column"
        )));
    }
    let col_names: Vec<String> = header.iter().skip(1).map(|v| v.trim().to_string()).collect();
    if col_names.iter().any(|name| name.is_empty()) {
        return Err(Error::Validation(format!(
            "{label} header contains an empty measurement name"
        )));
    }
    let expected: BTreeSet<_> = expected_names.iter().cloned().collect();
    let actual: BTreeSet<_> = col_names.iter().cloned().collect();
    if actual != expected {
        return Err(Error::Validation(format!(
            "{label} columns must match measurement names exactly"
        )));
    }
    let col_positions: BTreeMap<_, _> =
        col_names.iter().enumerate().map(|(idx, name)| (name.clone(), idx)).collect();
    let mut row_map: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for record in records {
        let record =
            record.map_err(|e| Error::Validation(format!("invalid {label} table: {e}")))?;
        if record.len() != col_names.len() + 1 {
            return Err(Error::Validation(format!(
                "{label} row has {} columns; expected {}",
                record.len(),
                col_names.len() + 1
            )));
        }
        let row_name = record
            .get(0)
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| Error::Validation(format!("{label} row has empty row name")))?;
        if !expected.contains(row_name) {
            return Err(Error::Validation(format!(
                "{label} row '{row_name}' does not match any measurement name"
            )));
        }
        if row_map.contains_key(row_name) {
            return Err(Error::Validation(format!(
                "{label} row '{row_name}' appears more than once"
            )));
        }
        let values = record
            .iter()
            .skip(1)
            .map(|value| {
                value.parse::<f64>().map_err(|e| {
                    Error::Validation(format!("{label} contains a non-numeric entry: {e}"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        row_map.insert(row_name.to_string(), values);
    }
    if row_map.len() != expected_names.len() {
        return Err(Error::Validation(format!(
            "{label} must contain exactly one row for each measurement"
        )));
    }
    let mut matrix = Vec::with_capacity(expected_names.len());
    for row_name in expected_names {
        let row_values = row_map
            .get(row_name)
            .ok_or_else(|| Error::Validation(format!("{label} missing row '{row_name}'")))?;
        let mut row = Vec::with_capacity(expected_names.len());
        for col_name in expected_names {
            let idx = *col_positions.get(col_name).expect("column position should exist");
            row.push(row_values[idx]);
        }
        matrix.push(row);
    }
    Ok(matrix)
}

fn parse_systematics_tables(
    systematics_table: Option<&str>,
    correlations_table: Option<&str>,
    measurement_names: &[String],
) -> Result<Vec<SystematicSource>> {
    let Some(systematics_table) = systematics_table else {
        if correlations_table.is_some() {
            return Err(Error::Validation(
                "correlations table requires a systematics table".to_string(),
            ));
        }
        return Ok(Vec::new());
    };

    let delimiter = infer_table_delimiter(systematics_table);
    let mut reader = csv::ReaderBuilder::new()
        .trim(Trim::All)
        .delimiter(delimiter)
        .from_reader(systematics_table.as_bytes());
    let measurement_set: BTreeSet<_> = measurement_names.iter().cloned().collect();
    let mut systematics: BTreeMap<String, SystematicTableAccumulator> = BTreeMap::new();
    for row in reader.deserialize::<SystematicTableRow>() {
        let row = row.map_err(|e| Error::Validation(format!("invalid systematics table: {e}")))?;
        if row.systematic.trim().is_empty() {
            return Err(Error::Validation(
                "systematics table contains an empty systematic name".to_string(),
            ));
        }
        if !measurement_set.contains(&row.measurement) {
            return Err(Error::Validation(format!(
                "systematics table references unknown measurement '{}'",
                row.measurement
            )));
        }
        let entry = systematics.entry(row.systematic.clone()).or_insert_with(|| {
            SystematicTableAccumulator {
                magnitudes: BTreeMap::new(),
                error_on_error: None,
                aux_mean: None,
            }
        });
        if entry.magnitudes.insert(row.measurement.clone(), row.magnitude).is_some() {
            return Err(Error::Validation(format!(
                "systematics table repeats systematic '{}' for measurement '{}'",
                row.systematic, row.measurement
            )));
        }
        if let Some(value) = row.error_on_error {
            match entry.error_on_error {
                Some(current) if (current - value).abs() > 1e-12 => {
                    return Err(Error::Validation(format!(
                        "systematics table has inconsistent error_on_error for systematic '{}'",
                        row.systematic
                    )));
                }
                None => entry.error_on_error = Some(value),
                _ => {}
            }
        }
        if let Some(value) = row.aux_mean {
            match entry.aux_mean {
                Some(current) if (current - value).abs() > 1e-12 => {
                    return Err(Error::Validation(format!(
                        "systematics table has inconsistent aux_mean for systematic '{}'",
                        row.systematic
                    )));
                }
                None => entry.aux_mean = Some(value),
                _ => {}
            }
        }
    }

    if systematics.is_empty() {
        return Ok(Vec::new());
    }

    for (name, entry) in &systematics {
        if entry.magnitudes.len() != measurement_names.len() {
            return Err(Error::Validation(format!(
                "systematics table must define a magnitude for every measurement in systematic '{name}'"
            )));
        }
    }

    let corr_rows = parse_correlation_rows(correlations_table)?;
    let mut out = Vec::with_capacity(systematics.len());
    for (name, entry) in systematics {
        let corr = build_correlation_matrix_for_systematic(&name, &corr_rows, measurement_names)?;
        let magnitudes = measurement_names
            .iter()
            .map(|measurement| *entry.magnitudes.get(measurement).expect("magnitude should exist"))
            .collect();
        out.push(SystematicSource {
            name,
            magnitudes,
            corr,
            error_on_error: entry.error_on_error.unwrap_or(0.0),
            aux_mean: entry.aux_mean.unwrap_or(0.0),
        });
    }
    Ok(out)
}

fn parse_correlation_rows(
    correlations_table: Option<&str>,
) -> Result<BTreeMap<String, Vec<CorrelationTableRow>>> {
    let Some(correlations_table) = correlations_table else {
        return Ok(BTreeMap::new());
    };
    let delimiter = infer_table_delimiter(correlations_table);
    let mut reader = csv::ReaderBuilder::new()
        .trim(Trim::All)
        .delimiter(delimiter)
        .from_reader(correlations_table.as_bytes());
    let mut rows: BTreeMap<String, Vec<CorrelationTableRow>> = BTreeMap::new();
    for row in reader.deserialize::<CorrelationTableRow>() {
        let row = row.map_err(|e| Error::Validation(format!("invalid correlations table: {e}")))?;
        if row.systematic.trim().is_empty()
            || row.row_measurement.trim().is_empty()
            || row.col_measurement.trim().is_empty()
        {
            return Err(Error::Validation(
                "correlations table contains an empty systematic or measurement name".to_string(),
            ));
        }
        rows.entry(row.systematic.clone()).or_default().push(row);
    }
    Ok(rows)
}

fn build_correlation_matrix_for_systematic(
    systematic: &str,
    corr_rows: &BTreeMap<String, Vec<CorrelationTableRow>>,
    measurement_names: &[String],
) -> Result<Vec<Vec<f64>>> {
    let n = measurement_names.len();
    let index: BTreeMap<_, _> =
        measurement_names.iter().enumerate().map(|(i, name)| (name.clone(), i)).collect();
    let mut matrix: Vec<Vec<Option<f64>>> = vec![vec![None; n]; n];
    if let Some(rows) = corr_rows.get(systematic) {
        for row in rows {
            let i = *index.get(&row.row_measurement).ok_or_else(|| {
                Error::Validation(format!(
                    "correlations table references unknown measurement '{}' in systematic '{}'",
                    row.row_measurement, systematic
                ))
            })?;
            let j = *index.get(&row.col_measurement).ok_or_else(|| {
                Error::Validation(format!(
                    "correlations table references unknown measurement '{}' in systematic '{}'",
                    row.col_measurement, systematic
                ))
            })?;
            if let Some(existing) = matrix[i][j] {
                if (existing - row.corr).abs() > 1e-12 {
                    return Err(Error::Validation(format!(
                        "correlations table contains conflicting entries for systematic '{}' pair ('{}', '{}')",
                        systematic, row.row_measurement, row.col_measurement
                    )));
                }
            } else {
                matrix[i][j] = Some(row.corr);
            }
        }
        for i in 0..n {
            matrix[i][i] = Some(matrix[i][i].unwrap_or(1.0));
            for j in (i + 1)..n {
                match (matrix[i][j], matrix[j][i]) {
                    (Some(a), Some(b)) if (a - b).abs() > 1e-12 => {
                        return Err(Error::Validation(format!(
                            "correlations table contains asymmetric entries for systematic '{}' pair ('{}', '{}')",
                            systematic, measurement_names[i], measurement_names[j]
                        )));
                    }
                    (Some(a), Some(_)) => {
                        matrix[i][j] = Some(a);
                        matrix[j][i] = Some(a);
                    }
                    (Some(a), None) => {
                        matrix[j][i] = Some(a);
                    }
                    (None, Some(b)) => {
                        matrix[i][j] = Some(b);
                    }
                    (None, None) => {
                        return Err(Error::Validation(format!(
                            "correlations table is incomplete for systematic '{}' pair ('{}', '{}')",
                            systematic, measurement_names[i], measurement_names[j]
                        )));
                    }
                }
            }
        }
    } else {
        for (i, row) in matrix.iter_mut().enumerate() {
            row[i] = Some(1.0);
        }
    }
    Ok(matrix
        .into_iter()
        .map(|row| row.into_iter().map(|value| value.unwrap_or(0.0)).collect())
        .collect())
}

impl PreparedSpec {
    fn from_spec(spec: &MeasurementCombinationSpec) -> Result<Self> {
        let n = spec.measurements.len();
        let y = DVector::from_iterator(n, spec.measurements.iter().map(|m| m.value));
        let ones = DVector::from_element(n, 1.0);
        let mut base_cov = matrix_from_rows(&spec.stat_covariance, n, n)?;

        let mut theta_ranges = Vec::with_capacity(spec.systematics.len());
        let mut tau_indices = Vec::with_capacity(spec.systematics.len());
        let mut corr_regularization_deltas = Vec::with_capacity(spec.systematics.len());
        let mut systematics = Vec::new();
        let mut cursor = 1usize;
        for syst in &spec.systematics {
            let raw_corr = matrix_from_rows(&syst.corr, n, n)?;
            let regularization_delta = corr_regularization_delta(&raw_corr)?;
            corr_regularization_deltas.push(regularization_delta);
            if syst.error_on_error > 0.0 {
                let (regularized_corr, _) = regularize_corr_for_precision(&raw_corr)?;
                let factor = factorize_corr_for_nuisance(&regularized_corr)?;
                let corr_precision = symmetric_pseudoinverse(&regularized_corr)?;
                let k = factor.ncols();
                let theta_range = cursor..(cursor + k);
                cursor += k;
                let tau_index = Some(cursor);
                cursor += 1;
                theta_ranges.push(theta_range);
                tau_indices.push(tau_index);
                let mags_diag = DMatrix::from_diagonal(&DVector::from_vec(syst.magnitudes.clone()));
                systematics.push(SystematicPrepared {
                    name: syst.name.clone(),
                    magnitudes: DVector::from_vec(syst.magnitudes.clone()),
                    corr_cov: raw_corr,
                    corr_reg: regularized_corr,
                    corr_factor: factor.clone(),
                    corr_precision,
                    effect_matrix: mags_diag * factor,
                    regularization_delta,
                    error_on_error: syst.error_on_error,
                    aux_mean: syst.aux_mean,
                });
            } else {
                for i in 0..n {
                    for j in 0..n {
                        base_cov[(i, j)] +=
                            syst.magnitudes[i] * raw_corr[(i, j)] * syst.magnitudes[j];
                    }
                }
            }
        }
        let layout = Layout { theta_ranges, tau_indices, n_params: cursor };
        let base_precision = symmetric_pseudoinverse(&base_cov)?;

        let total_cov = total_covariance(spec)?;
        let total_precision = symmetric_pseudoinverse(&total_cov)?;
        let denom = ones.dot(&(&total_precision * &ones));
        if !denom.is_finite() || denom <= 0.0 {
            return Err(Error::Computation(format!(
                "invalid combination denominator from total covariance: {denom}"
            )));
        }
        let fixed_sigma_guess = (1.0 / denom).sqrt();
        let ymin = spec.measurements.iter().map(|m| m.value).fold(f64::INFINITY, f64::min);
        let ymax = spec.measurements.iter().map(|m| m.value).fold(f64::NEG_INFINITY, f64::max);
        let span = (ymax - ymin).abs().max(fixed_sigma_guess).max(1.0);
        let mu_bounds = (ymin - 50.0 * span, ymax + 50.0 * span);

        Ok(Self {
            shared: Arc::new(PreparedSpecShared {
                poi: spec.poi.clone(),
                ones,
                base_cov,
                base_precision,
                systematics,
                corr_regularization_deltas,
                n_systematics_total: spec.systematics.len(),
                layout,
                mu_bounds,
                fixed_sigma_guess,
            }),
            y,
        })
    }

    fn init_params(&self) -> Result<Vec<f64>> {
        let base = fixed_variance_solution(self)?;
        let mut params = vec![0.0; self.layout.n_params];
        params[0] = base.mu_hat;
        for (i, syst) in self.systematics.iter().enumerate() {
            let range = self.layout.theta_ranges[i].clone();
            for j in range {
                params[j] = 0.0;
            }
            if let Some(idx) = self.layout.tau_indices[i] {
                params[idx] = 0.0;
            }
            if syst.error_on_error == 0.0 {
                continue;
            }
        }
        Ok(params)
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        let mut bounds = vec![self.mu_bounds; self.layout.n_params];
        bounds[0] = self.mu_bounds;
        for theta_range in &self.layout.theta_ranges {
            for idx in theta_range.clone() {
                bounds[idx] = (-THETA_BOUND, THETA_BOUND);
            }
        }
        for tau_idx in self.layout.tau_indices.iter().flatten() {
            bounds[*tau_idx] = (TAU_MIN.ln(), TAU_MAX.ln());
        }
        bounds
    }

    fn clone_with_y(&self, y: DVector<f64>) -> Self {
        Self { shared: Arc::clone(&self.shared), y }
    }
}

impl<'a> MeasurementCombineObjective<'a> {
    fn new(prep: &'a PreparedSpec) -> Self {
        Self { prep, cache: Mutex::new(None) }
    }

    fn tau_value(&self, params: &[f64], syst_idx: usize) -> f64 {
        self.prep.layout.tau_indices[syst_idx].map(|idx| params[idx].exp()).unwrap_or(1.0)
    }

    fn theta_slice<'b>(&self, params: &'b [f64], syst_idx: usize) -> &'b [f64] {
        let range = self.prep.layout.theta_ranges[syst_idx].clone();
        &params[range]
    }

    fn compute_value_and_gradient(&self, params: &[f64]) -> Result<ObjectiveEvalCache> {
        if params.len() != self.prep.layout.n_params {
            return Err(Error::Validation(format!(
                "objective parameter length mismatch: got={} expected={}",
                params.len(),
                self.prep.layout.n_params
            )));
        }

        let mu = params[0];
        let mut pred = self.prep.ones.clone() * mu;
        let mut theta_blocks = Vec::with_capacity(self.prep.systematics.len());
        for (s_idx, syst) in self.prep.systematics.iter().enumerate() {
            let theta = DVector::from_column_slice(self.theta_slice(params, s_idx));
            pred += &syst.effect_matrix * &theta;
            theta_blocks.push(theta);
        }
        let resid = &self.prep.y - pred;
        let weighted_resid = &self.prep.base_precision * &resid;

        let mut nll = 0.5 * resid.dot(&weighted_resid);
        let mut grad = vec![0.0; self.prep.layout.n_params];
        grad[0] = -self.prep.ones.dot(&weighted_resid);

        for (s_idx, syst) in self.prep.systematics.iter().enumerate() {
            let theta = &theta_blocks[s_idx];
            let theta_grad = -(&syst.effect_matrix.transpose() * &weighted_resid)
                + theta.scale(1.0 / self.tau_value(params, s_idx));
            let range = self.prep.layout.theta_ranges[s_idx].clone();
            for (offset, idx) in range.enumerate() {
                grad[idx] = theta_grad[offset];
            }

            let tau = self.tau_value(params, s_idx);
            let delta = theta - DVector::from_element(theta.len(), syst.aux_mean);
            let quad = delta.dot(&delta);
            if syst.error_on_error > 0.0 {
                let eps2 = syst.error_on_error * syst.error_on_error;
                let k = theta.len() as f64;
                nll += 0.5 * quad / tau
                    + 0.5 * (k + 1.0 / (2.0 * eps2)) * tau.ln()
                    + 1.0 / (2.0 * eps2 * tau);
                let tau_grad =
                    0.5 * (k + 1.0 / (2.0 * eps2)) - 0.5 * quad / tau - 1.0 / (2.0 * eps2 * tau);
                if let Some(idx) = self.prep.layout.tau_indices[s_idx] {
                    grad[idx] = tau_grad;
                }
            }
        }

        Ok(ObjectiveEvalCache { params: params.to_vec(), cost: nll, gradient: grad })
    }

    fn cached_eval(&self, params: &[f64]) -> Result<ObjectiveEvalCache> {
        if let Some(cache) = self.cache.lock().unwrap().as_ref()
            && cache.params == params
        {
            return Ok(cache.clone());
        }
        let cache = self.compute_value_and_gradient(params)?;
        *self.cache.lock().unwrap() = Some(cache.clone());
        Ok(cache)
    }

    fn cached_cost(&self, params: &[f64]) -> Result<f64> {
        if let Some(cache) = self.cache.lock().unwrap().as_ref()
            && cache.params == params
        {
            return Ok(cache.cost);
        }
        let cache = self.compute_value_and_gradient(params)?;
        let cost = cache.cost;
        *self.cache.lock().unwrap() = Some(cache);
        Ok(cost)
    }

    fn cached_gradient(&self, params: &[f64]) -> Result<Vec<f64>> {
        if let Some(cache) = self.cache.lock().unwrap().as_ref()
            && cache.params == params
        {
            return Ok(cache.gradient.clone());
        }
        let cache = self.compute_value_and_gradient(params)?;
        let gradient = cache.gradient.clone();
        *self.cache.lock().unwrap() = Some(cache);
        Ok(gradient)
    }
}

impl ObjectiveFunction for MeasurementCombineObjective<'_> {
    fn eval(&self, params: &[f64]) -> Result<f64> {
        self.cached_cost(params)
    }

    fn gradient(&self, params: &[f64]) -> Result<Vec<f64>> {
        self.cached_gradient(params)
    }
}

impl PaperLayout {
    fn for_prep(prep: &PreparedSpec) -> Self {
        let mut theta_ranges = Vec::with_capacity(prep.systematics.len());
        let mut tau_indices = Vec::with_capacity(prep.systematics.len());
        let mut cursor = 1usize;
        let n = prep.y.len();
        for _ in &prep.systematics {
            theta_ranges.push(cursor..(cursor + n));
            cursor += n;
            tau_indices.push(cursor);
            cursor += 1;
        }
        Self { theta_ranges, tau_indices, n_params: cursor }
    }
}

impl<'a> PaperMeasurementCombineObjective<'a> {
    fn new(prep: &'a PreparedSpec) -> Self {
        let layout = PaperLayout::for_prep(prep);
        Self { prep, layout, cache: Mutex::new(None) }
    }

    fn init_params(&self) -> Result<Vec<f64>> {
        let base = fixed_variance_solution(self.prep)?;
        let mut params = vec![0.0; self.layout.n_params];
        params[0] = base.mu_hat;
        Ok(params)
    }

    fn init_params_from_profiled_state(
        &self,
        mu: f64,
        state: &ProfiledGvmState,
    ) -> Result<Vec<f64>> {
        if state.theta_original.len() != self.prep.systematics.len()
            || state.profiled_variance_scales.len() != self.prep.systematics.len()
        {
            return Err(Error::Validation(
                "paper warm-start state does not match the prepared systematics layout".to_string(),
            ));
        }
        let mut params = vec![0.0; self.layout.n_params];
        params[0] = mu.clamp(self.prep.mu_bounds.0, self.prep.mu_bounds.1);
        for (s_idx, syst) in self.prep.systematics.iter().enumerate() {
            let theta = &state.theta_original[s_idx];
            if theta.len() != self.prep.y.len() {
                return Err(Error::Validation(format!(
                    "paper warm-start theta length mismatch for systematic '{}': got={} expected={}",
                    syst.name,
                    theta.len(),
                    self.prep.y.len()
                )));
            }
            let range = self.layout.theta_ranges[s_idx].clone();
            for (offset, idx) in range.enumerate() {
                params[idx] = theta[offset].clamp(-THETA_BOUND, THETA_BOUND);
            }
            let tau = state.profiled_variance_scales[s_idx].clamp(TAU_MIN, TAU_MAX).max(1e-12);
            params[self.layout.tau_indices[s_idx]] = tau.ln();
        }
        Ok(params)
    }

    fn analytic_warm_start_params(&self) -> Option<Vec<f64>> {
        let solver = AnalyticPerturbativeSolver::new(self.prep, 1);
        let point = minimize_analytic_profile(&solver, self.prep.mu_bounds).ok()?;
        self.init_params_from_profiled_state(point.mu, &point.state).ok()
    }

    fn analytic_warm_start_guide(&self, ci_level: f64) -> Option<PaperWarmStartGuide> {
        if !(0.0 < ci_level && ci_level < 1.0) {
            return None;
        }
        let solver = AnalyticPerturbativeSolver::new(self.prep, 1);
        let mle = minimize_analytic_profile(&solver, self.prep.mu_bounds).ok()?;
        let mle_params = self.init_params_from_profiled_state(mle.mu, &mle.state).ok()?;
        let mut guide = PaperWarmStartGuide { mle_params, lower_hint: None, upper_hint: None };
        if ensure_perturbative_validity(self.prep, &mle.state).is_err() {
            return Some(guide);
        }

        let chi2_level = ChiSquared::new(1.0).ok()?.inverse_cdf(ci_level);
        let target = mle.nll + 0.5 * chi2_level;
        let sigma_guess = self.prep.fixed_sigma_guess.max(1e-3);

        if let Ok(lower_mu) =
            find_analytic_profile_bound(&solver, mle.mu, -1.0, sigma_guess, target)
            && let Ok(lower_point) = solver.profile_at_mu_raw(lower_mu)
            && let Ok(params) = self.init_params_from_profiled_state(lower_mu, &lower_point.state)
        {
            guide.lower_hint = Some(PaperProfileBoundHint { mu: lower_mu, params });
        }

        if let Ok(upper_mu) = find_analytic_profile_bound(&solver, mle.mu, 1.0, sigma_guess, target)
            && let Ok(upper_point) = solver.profile_at_mu_raw(upper_mu)
            && let Ok(params) = self.init_params_from_profiled_state(upper_mu, &upper_point.state)
        {
            guide.upper_hint = Some(PaperProfileBoundHint { mu: upper_mu, params });
        }

        Some(guide)
    }

    fn bounds(&self) -> Vec<(f64, f64)> {
        let mut bounds = vec![self.prep.mu_bounds; self.layout.n_params];
        for theta_range in &self.layout.theta_ranges {
            for idx in theta_range.clone() {
                bounds[idx] = (-THETA_BOUND, THETA_BOUND);
            }
        }
        for tau_idx in &self.layout.tau_indices {
            bounds[*tau_idx] = (TAU_MIN.ln(), TAU_MAX.ln());
        }
        bounds
    }

    fn theta_slice<'b>(&self, params: &'b [f64], syst_idx: usize) -> &'b [f64] {
        let range = self.layout.theta_ranges[syst_idx].clone();
        &params[range]
    }

    fn tau_value(&self, params: &[f64], syst_idx: usize) -> f64 {
        params[self.layout.tau_indices[syst_idx]].exp()
    }

    fn compute_value_and_gradient(&self, params: &[f64]) -> Result<ObjectiveEvalCache> {
        if params.len() != self.layout.n_params {
            return Err(Error::Validation(format!(
                "paper objective parameter length mismatch: got={} expected={}",
                params.len(),
                self.layout.n_params
            )));
        }

        let n = self.prep.y.len();
        let mu = params[0];
        let mut resid = self.prep.y.as_slice().iter().map(|&value| value - mu).collect::<Vec<_>>();
        for (s_idx, syst) in self.prep.systematics.iter().enumerate() {
            let theta = self.theta_slice(params, s_idx);
            let magnitudes = syst.magnitudes.as_slice();
            for i in 0..n {
                resid[i] -= magnitudes[i] * theta[i];
            }
        }

        let mut weighted_resid = vec![0.0; n];
        for i in 0..n {
            let mut row_sum = 0.0;
            for j in 0..n {
                row_sum += self.prep.base_precision[(i, j)] * resid[j];
            }
            weighted_resid[i] = row_sum;
        }

        let mut nll =
            0.5 * resid.iter().zip(weighted_resid.iter()).map(|(lhs, rhs)| lhs * rhs).sum::<f64>();
        let mut grad = vec![0.0; self.layout.n_params];
        grad[0] = -weighted_resid.iter().sum::<f64>();

        let mut delta = vec![0.0; n];
        let mut corr_delta = vec![0.0; n];

        for (s_idx, syst) in self.prep.systematics.iter().enumerate() {
            let range = self.layout.theta_ranges[s_idx].clone();
            let theta = self.theta_slice(params, s_idx);
            for i in 0..n {
                delta[i] = theta[i] - syst.aux_mean;
            }

            for i in 0..n {
                let mut row_sum = 0.0;
                for j in 0..n {
                    row_sum += syst.corr_precision[(i, j)] * delta[j];
                }
                corr_delta[i] = row_sum;
            }

            let tau = self.tau_value(params, s_idx);
            let inv_tau = 1.0 / tau;
            let magnitudes = syst.magnitudes.as_slice();
            let mut quad = 0.0;
            for i in 0..n {
                quad += delta[i] * corr_delta[i];
                grad[range.start + i] =
                    -magnitudes[i] * weighted_resid[i] + corr_delta[i] * inv_tau;
            }

            let eps2 = syst.error_on_error * syst.error_on_error;
            let n_f = n as f64;
            nll += 0.5 * quad / tau
                + 0.5 * (n_f + 1.0 / (2.0 * eps2)) * tau.ln()
                + 1.0 / (2.0 * eps2 * tau);
            grad[self.layout.tau_indices[s_idx]] =
                0.5 * (n_f + 1.0 / (2.0 * eps2)) - 0.5 * quad / tau - 1.0 / (2.0 * eps2 * tau);
        }

        Ok(ObjectiveEvalCache { params: params.to_vec(), cost: nll, gradient: grad })
    }

    fn cached_eval(&self, params: &[f64]) -> Result<ObjectiveEvalCache> {
        if let Some(cache) = self.cache.lock().unwrap().as_ref()
            && cache.params == params
        {
            return Ok(cache.clone());
        }
        let cache = self.compute_value_and_gradient(params)?;
        *self.cache.lock().unwrap() = Some(cache.clone());
        Ok(cache)
    }

    fn cached_cost(&self, params: &[f64]) -> Result<f64> {
        if let Some(cache) = self.cache.lock().unwrap().as_ref()
            && cache.params == params
        {
            return Ok(cache.cost);
        }
        let cache = self.compute_value_and_gradient(params)?;
        let cost = cache.cost;
        *self.cache.lock().unwrap() = Some(cache);
        Ok(cost)
    }

    fn cached_gradient(&self, params: &[f64]) -> Result<Vec<f64>> {
        if let Some(cache) = self.cache.lock().unwrap().as_ref()
            && cache.params == params
        {
            return Ok(cache.gradient.clone());
        }
        let cache = self.compute_value_and_gradient(params)?;
        let gradient = cache.gradient.clone();
        *self.cache.lock().unwrap() = Some(cache);
        Ok(gradient)
    }
}

impl ObjectiveFunction for PaperMeasurementCombineObjective<'_> {
    fn eval(&self, params: &[f64]) -> Result<f64> {
        self.cached_cost(params)
    }

    fn gradient(&self, params: &[f64]) -> Result<Vec<f64>> {
        self.cached_gradient(params)
    }
}

impl<'a> AnalyticPerturbativeSolver<'a> {
    fn new(prep: &'a PreparedSpec, order: usize) -> Self {
        Self { prep, order }
    }

    fn profile_at_mu(&self, mu: f64) -> Result<AnalyticProfilePoint> {
        let point = self.profile_at_mu_raw(mu)?;
        ensure_perturbative_validity(self.prep, &point.state)?;
        Ok(point)
    }

    fn profile_at_mu_raw(&self, mu: f64) -> Result<AnalyticProfilePoint> {
        self.profile_at_mu_raw_with_threshold(mu, ANALYTIC_FAST_PATH_NM_THRESHOLD)
    }

    fn profile_at_mu_raw_with_threshold(
        &self,
        mu: f64,
        nm_threshold: usize,
    ) -> Result<AnalyticProfilePoint> {
        let n = self.prep.y.len();
        let s_count = self.prep.systematics.len();
        let dim = n * s_count;
        if dim == 0 {
            return Err(Error::Validation(
                "analytic perturbative solver requires at least one uncertain systematic"
                    .to_string(),
            ));
        }

        let residual_mu = &self.prep.y - self.prep.ones.clone().scale(mu);
        let (mut theta, _) =
            self.solve_linear_system_with_threshold(&residual_mu, None, None, nm_threshold)?;
        let mut sigma2 = self.compute_sigma2(&theta);

        for _ in 1..=self.order {
            let predicted = self.predicted_shift(&theta);
            let residual = &residual_mu - predicted;
            let (delta, _) = self.solve_linear_system_with_threshold(
                &residual,
                Some(&theta),
                Some(&sigma2),
                nm_threshold,
            )?;
            for (theta_s, delta_s) in theta.iter_mut().zip(delta) {
                *theta_s += delta_s;
            }
            sigma2 = self.compute_sigma2(&theta);
        }

        let state = ProfiledGvmState {
            theta_l2_norms: theta.iter().map(|theta_s| theta_s.norm()).collect(),
            theta_original: theta.clone(),
            profiled_variance_scales: sigma2.clone(),
        };
        let nll = analytic_profile_nll(self.prep, mu, &theta, &sigma2)?;
        Ok(AnalyticProfilePoint { mu, nll, state })
    }

    fn solve_linear_system_with_threshold(
        &self,
        residual: &DVector<f64>,
        theta: Option<&[DVector<f64>]>,
        sigma2: Option<&[f64]>,
        nm_threshold: usize,
    ) -> Result<(Vec<DVector<f64>>, AnalyticLinearSolvePath)> {
        let nm = self.prep.y.len().saturating_mul(self.prep.systematics.len());
        if nm > nm_threshold
            && let Some(theta_fast) = self.solve_linear_system_fast(residual, theta, sigma2)?
        {
            return Ok((theta_fast, AnalyticLinearSolvePath::Fast));
        }
        let theta_ref = self.solve_linear_system_reference(residual, theta, sigma2)?;
        Ok((theta_ref, AnalyticLinearSolvePath::Reference))
    }

    fn solve_linear_system_reference(
        &self,
        residual: &DVector<f64>,
        theta: Option<&[DVector<f64>]>,
        sigma2: Option<&[f64]>,
    ) -> Result<Vec<DVector<f64>>> {
        let c = self.build_system_matrix(sigma2)?;
        let rhs = self.build_rhs(residual, theta, sigma2);
        let c_inv = symmetric_pseudoinverse(&c)?;
        let theta_flat = &c_inv * rhs;
        Ok(self.unflatten_theta(&theta_flat))
    }

    fn solve_linear_system_fast(
        &self,
        residual: &DVector<f64>,
        theta: Option<&[DVector<f64>]>,
        sigma2: Option<&[f64]>,
    ) -> Result<Option<Vec<DVector<f64>>>> {
        let n = self.prep.y.len();
        let mut w = self.prep.base_cov.clone();
        let v_inv_residual = &self.prep.base_precision * residual;
        let mut rhs = DVector::zeros(n);
        let mut centers = Vec::with_capacity(self.prep.systematics.len());
        let mut u_blocks = Vec::with_capacity(self.prep.systematics.len());

        for (s_idx, syst) in self.prep.systematics.iter().enumerate() {
            if syst.corr_factor.ncols() != n {
                return Ok(None);
            }
            let scale = sigma2.map(|values| values[s_idx]).unwrap_or(1.0).max(1e-12);
            let u_block = syst.corr_reg.clone().scale(scale);
            let center = if let Some(theta_all) = theta {
                DVector::from_element(n, syst.aux_mean) - &theta_all[s_idx]
            } else {
                DVector::from_element(n, syst.aux_mean)
            };
            rhs += syst.magnitudes.component_mul(&center);

            let design_v_inv_residual = syst.magnitudes.component_mul(&v_inv_residual);
            let u_design_v_inv_residual = &u_block * design_v_inv_residual;
            rhs += syst.magnitudes.component_mul(&u_design_v_inv_residual);

            for i in 0..n {
                for j in 0..n {
                    w[(i, j)] += syst.magnitudes[i] * u_block[(i, j)] * syst.magnitudes[j];
                }
            }

            centers.push(center);
            u_blocks.push(u_block);
        }

        let w = symmetrize_matrix(w);
        let z = solve_symmetric_linear_system(&w, &rhs)?;
        let correction_driver = &v_inv_residual - &z;
        let mut out = Vec::with_capacity(self.prep.systematics.len());
        for (s_idx, syst) in self.prep.systematics.iter().enumerate() {
            let design_driver = syst.magnitudes.component_mul(&correction_driver);
            let correction = &u_blocks[s_idx] * design_driver;
            out.push(&centers[s_idx] + correction);
        }
        Ok(Some(out))
    }

    fn build_system_matrix(&self, sigma2: Option<&[f64]>) -> Result<DMatrix<f64>> {
        let n = self.prep.y.len();
        let s_count = self.prep.systematics.len();
        let dim = n * s_count;
        let mut out = DMatrix::zeros(dim, dim);
        for (s_idx, syst_s) in self.prep.systematics.iter().enumerate() {
            let d_s = DMatrix::from_diagonal(&syst_s.magnitudes);
            for (p_idx, syst_p) in self.prep.systematics.iter().enumerate() {
                let d_p = DMatrix::from_diagonal(&syst_p.magnitudes);
                let mut block = &d_s * &self.prep.base_precision * &d_p;
                if s_idx == p_idx {
                    let denom = sigma2.map(|values| values[s_idx]).unwrap_or(1.0).max(1e-12);
                    block += syst_s.corr_precision.clone().scale(1.0 / denom);
                }
                self.write_block(&mut out, s_idx, p_idx, &block);
            }
        }
        Ok(out)
    }

    fn build_rhs(
        &self,
        residual: &DVector<f64>,
        theta: Option<&[DVector<f64>]>,
        sigma2: Option<&[f64]>,
    ) -> DVector<f64> {
        let n = self.prep.y.len();
        let s_count = self.prep.systematics.len();
        let mut out = DVector::zeros(n * s_count);
        for (s_idx, syst) in self.prep.systematics.iter().enumerate() {
            let d_s = DMatrix::from_diagonal(&syst.magnitudes);
            let mut rhs_s = &d_s * &self.prep.base_precision * residual;
            let u_vec = DVector::from_element(n, syst.aux_mean);
            let denom = sigma2.map(|values| values[s_idx]).unwrap_or(1.0).max(1e-12);
            let nuisance_centered = if let Some(theta_all) = theta {
                &u_vec - &theta_all[s_idx]
            } else {
                u_vec.clone()
            };
            rhs_s += (&syst.corr_precision * &nuisance_centered).scale(1.0 / denom);
            self.write_vec_block(&mut out, s_idx, &rhs_s);
        }
        out
    }

    fn compute_sigma2(&self, theta: &[DVector<f64>]) -> Vec<f64> {
        let n = self.prep.y.len() as f64;
        self.prep
            .systematics
            .iter()
            .zip(theta)
            .map(|(syst, theta_s)| {
                let eps2 = syst.error_on_error * syst.error_on_error;
                let delta = theta_s - DVector::from_element(theta_s.len(), syst.aux_mean);
                let quad = (&delta.transpose() * &syst.corr_precision * &delta)[(0, 0)];
                ((1.0 + 2.0 * eps2 * quad) / (1.0 + 2.0 * n * eps2)).max(1e-12)
            })
            .collect()
    }

    fn predicted_shift(&self, theta: &[DVector<f64>]) -> DVector<f64> {
        let mut out = DVector::zeros(self.prep.y.len());
        for (syst, theta_s) in self.prep.systematics.iter().zip(theta) {
            out += syst.magnitudes.component_mul(theta_s);
        }
        out
    }

    fn unflatten_theta(&self, flat: &DVector<f64>) -> Vec<DVector<f64>> {
        let n = self.prep.y.len();
        self.prep
            .systematics
            .iter()
            .enumerate()
            .map(|(s_idx, _)| flat.rows(s_idx * n, n).into_owned())
            .collect()
    }

    fn write_block(
        &self,
        out: &mut DMatrix<f64>,
        row_s: usize,
        col_s: usize,
        block: &DMatrix<f64>,
    ) {
        let n = self.prep.y.len();
        let row0 = row_s * n;
        let col0 = col_s * n;
        for i in 0..n {
            for j in 0..n {
                out[(row0 + i, col0 + j)] = block[(i, j)];
            }
        }
    }

    fn write_vec_block(&self, out: &mut DVector<f64>, s_idx: usize, block: &DVector<f64>) {
        let n = self.prep.y.len();
        let row0 = s_idx * n;
        for i in 0..n {
            out[row0 + i] = block[i];
        }
    }
}

fn analytic_profile_nll(
    prep: &PreparedSpec,
    mu: f64,
    theta: &[DVector<f64>],
    sigma2: &[f64],
) -> Result<f64> {
    let mut pred = prep.ones.clone() * mu;
    for (syst, theta_s) in prep.systematics.iter().zip(theta) {
        pred += syst.magnitudes.component_mul(theta_s);
    }
    let resid = &prep.y - pred;
    let mut nll = 0.5 * quad_form(&prep.base_precision, &resid);
    let n = prep.y.len() as f64;
    for ((syst, theta_s), sigma2_s) in prep.systematics.iter().zip(theta).zip(sigma2) {
        let eps2 = syst.error_on_error * syst.error_on_error;
        let delta = theta_s - DVector::from_element(theta_s.len(), syst.aux_mean);
        let quad = (&delta.transpose() * &syst.corr_precision * &delta)[(0, 0)];
        let tau = sigma2_s.max(1e-12);
        let scaled = 1.0 + 2.0 * eps2 * quad;
        if !tau.is_finite() || !scaled.is_finite() || scaled <= 0.0 {
            return Err(Error::Computation(
                "analytic perturbative profile produced a non-finite variance term".to_string(),
            ));
        }
        nll += 0.5 * (n + 1.0 / (2.0 * eps2)) * scaled.ln();
    }
    Ok(nll)
}

fn solve_symmetric_linear_system(m: &DMatrix<f64>, rhs: &DVector<f64>) -> Result<DVector<f64>> {
    if m.nrows() != m.ncols() {
        return Err(Error::Validation(
            "symmetric linear solve requires a square matrix".to_string(),
        ));
    }
    if m.nrows() != rhs.len() {
        return Err(Error::Validation(format!(
            "symmetric linear solve rhs length mismatch: got={} expected={}",
            rhs.len(),
            m.nrows()
        )));
    }
    if let Some(cholesky) = m.clone().cholesky() {
        let lower = cholesky.l();
        let mut min_diag = f64::INFINITY;
        let mut max_diag = 0.0_f64;
        for i in 0..lower.nrows() {
            let diag = lower[(i, i)].abs();
            min_diag = min_diag.min(diag);
            max_diag = max_diag.max(diag);
        }
        let diag_ratio = if max_diag > 0.0 { min_diag / max_diag } else { 0.0 };
        if diag_ratio.is_finite() && diag_ratio >= CHOLESKY_FAST_PATH_MIN_DIAG_RATIO {
            return Ok(cholesky.solve(rhs));
        }
    }
    let pinv = symmetric_pseudoinverse(m)?;
    Ok(pinv * rhs)
}

fn minimize_analytic_profile(
    solver: &AnalyticPerturbativeSolver<'_>,
    bounds: (f64, f64),
) -> Result<AnalyticProfilePoint> {
    let base = fixed_variance_solution(solver.prep)?;
    let span = (6.0 * base.sigma.max(solver.prep.fixed_sigma_guess).max(1e-3)).max(1e-2);
    let (mut lo, mut hi) = (
        (base.mu_hat - span).clamp(bounds.0, bounds.1),
        (base.mu_hat + span).clamp(bounds.0, bounds.1),
    );
    if !lo.is_finite() || !hi.is_finite() || lo >= hi {
        return Err(Error::Validation(
            "invalid mu bounds for analytic perturbative solve".to_string(),
        ));
    }

    let n_scan = 25usize;
    let mut scan = Vec::with_capacity(n_scan);
    for idx in 0..n_scan {
        let frac = idx as f64 / (n_scan as f64 - 1.0);
        let mu = lo + frac * (hi - lo);
        let prof = solver.profile_at_mu_raw(mu)?;
        scan.push(prof);
    }
    let (best_idx, _) =
        scan.iter().enumerate().min_by(|(_, a), (_, b)| a.nll.total_cmp(&b.nll)).ok_or_else(
            || Error::Computation("analytic perturbative scan produced no points".to_string()),
        )?;
    if best_idx > 0 {
        lo = scan[best_idx - 1].mu;
    }
    if best_idx + 1 < scan.len() {
        hi = scan[best_idx + 1].mu;
    }

    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let inv_phi = 1.0 / phi;

    let mut c = hi - (hi - lo) * inv_phi;
    let mut d = lo + (hi - lo) * inv_phi;
    let mut fc = solver.profile_at_mu_raw(c)?;
    let mut fd = solver.profile_at_mu_raw(d)?;
    for _ in 0..96 {
        if (hi - lo).abs() < 1e-6 {
            break;
        }
        if fc.nll <= fd.nll {
            hi = d;
            d = c;
            fd = fc;
            c = hi - (hi - lo) * inv_phi;
            fc = solver.profile_at_mu_raw(c)?;
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + (hi - lo) * inv_phi;
            fd = solver.profile_at_mu_raw(d)?;
        }
    }
    if fc.nll <= fd.nll { Ok(fc) } else { Ok(fd) }
}

fn ensure_perturbative_validity(prep: &PreparedSpec, state: &ProfiledGvmState) -> Result<()> {
    let validity = compute_perturbative_validity(prep, state);
    if validity.within_threshold.iter().all(|value| *value) {
        Ok(())
    } else {
        Err(perturbative_validity_error())
    }
}

fn perturbative_validity_error() -> Error {
    Error::Computation(
        "analytic perturbative path is outside the Eq. (29)/(60) validity radius".to_string(),
    )
}

fn is_perturbative_validity_error(err: &Error) -> bool {
    matches!(err, Error::Computation(message) if message == "analytic perturbative path is outside the Eq. (29)/(60) validity radius")
}

fn paper_warm_start_from_profiled_state(
    prep: &PreparedSpec,
    mu: f64,
    state: &ProfiledGvmState,
) -> Option<PaperWarmStartGuide> {
    let objective = PaperMeasurementCombineObjective::new(prep);
    let mle_params = objective.init_params_from_profiled_state(mu, state).ok()?;
    Some(PaperWarmStartGuide { mle_params, lower_hint: None, upper_hint: None })
}

pub fn combine_measurements(
    spec: &MeasurementCombinationSpec,
    ci_level: f64,
) -> Result<MeasurementCombinationResult> {
    combine_measurements_with_solver(spec, ci_level, MeasurementCombinationSolver::Auto)
}

pub fn combine_measurements_with_solver(
    spec: &MeasurementCombinationSpec,
    ci_level: f64,
    solver: MeasurementCombinationSolver,
) -> Result<MeasurementCombinationResult> {
    spec.validate()?;
    if !(0.0 < ci_level && ci_level < 1.0) {
        return Err(Error::Validation(format!(
            "ci_level must satisfy 0 < ci_level < 1, got={ci_level}"
        )));
    }
    let prep = PreparedSpec::from_spec(spec)?;
    let requested_error_on_error = spec.systematics.iter().any(|s| s.error_on_error > 0.0);
    if requested_error_on_error {
        match solver {
            MeasurementCombinationSolver::Numerical => numerical_gvm_result(prep, ci_level),
            MeasurementCombinationSolver::NumericalPaper => {
                numerical_paper_gvm_result(prep, ci_level)
            }
            MeasurementCombinationSolver::AnalyticPerturbative => {
                analytic_perturbative_gvm_result(prep, ci_level, 1)
            }
            MeasurementCombinationSolver::Auto => {
                match analytic_perturbative_attempt(prep.clone(), ci_level, 1)? {
                    AnalyticPerturbativeAttempt::Completed(mut result) => {
                        let (requested_solver, effective_solver) = maybe_solver_dispatch(
                            MeasurementCombinationSolver::Auto,
                            MeasurementCombinationSolver::AnalyticPerturbative,
                        );
                        result.diagnostics.requested_solver = requested_solver;
                        result.diagnostics.effective_solver = effective_solver;
                        Ok(*result)
                    }
                    AnalyticPerturbativeAttempt::FallbackWarmStart(warm_start) => {
                        let mut result = numerical_paper_gvm_result_with_warm_start(
                            prep,
                            ci_level,
                            warm_start.as_ref(),
                        )?;
                        let (requested_solver, effective_solver) = maybe_solver_dispatch(
                            MeasurementCombinationSolver::Auto,
                            MeasurementCombinationSolver::NumericalPaper,
                        );
                        result.diagnostics.requested_solver = requested_solver;
                        result.diagnostics.effective_solver = effective_solver;
                        Ok(result)
                    }
                }
            }
        }
    } else {
        fixed_variance_result(prep, ci_level)
    }
}

pub fn calibrate_measurements_toys(
    spec: &MeasurementCombinationSpec,
    ci_level: f64,
    n_toys: usize,
    seed: u64,
) -> Result<MeasurementCombinationCalibrationReport> {
    calibrate_measurements_toys_with_solver(
        spec,
        ci_level,
        MeasurementCombinationSolver::Auto,
        n_toys,
        seed,
    )
}

pub fn calibrate_measurements_toys_with_solver(
    spec: &MeasurementCombinationSpec,
    ci_level: f64,
    solver: MeasurementCombinationSolver,
    n_toys: usize,
    seed: u64,
) -> Result<MeasurementCombinationCalibrationReport> {
    if n_toys == 0 {
        return Err(Error::Validation("n_toys must be >= 1".to_string()));
    }
    let reference = combine_measurements_with_solver(spec, ci_level, solver)?;
    if !reference.diagnostics.requested_error_on_error {
        return Err(Error::Validation(
            "toy calibration requires at least one systematic with error_on_error > 0".to_string(),
        ));
    }
    calibrate_measurements_toys_with_reference(spec, &reference, ci_level, solver, n_toys, seed)
}

pub fn calibrate_measurements_toys_study(
    spec: &MeasurementCombinationSpec,
    ci_level: f64,
    n_toys: usize,
    seeds: &[u64],
) -> Result<MeasurementCombinationCalibrationStudyReport> {
    calibrate_measurements_toys_study_with_solver(
        spec,
        ci_level,
        MeasurementCombinationSolver::Auto,
        n_toys,
        seeds,
    )
}

pub fn calibrate_measurements_toys_study_with_solver(
    spec: &MeasurementCombinationSpec,
    ci_level: f64,
    solver: MeasurementCombinationSolver,
    n_toys: usize,
    seeds: &[u64],
) -> Result<MeasurementCombinationCalibrationStudyReport> {
    if seeds.is_empty() {
        return Err(Error::Validation("calibration study requires at least one seed".to_string()));
    }
    if n_toys == 0 {
        return Err(Error::Validation("n_toys must be >= 1".to_string()));
    }
    let reference = combine_measurements_with_solver(spec, ci_level, solver)?;
    if !reference.diagnostics.requested_error_on_error {
        return Err(Error::Validation(
            "toy calibration requires at least one systematic with error_on_error > 0".to_string(),
        ));
    }

    let can_par = can_parallelize_measurement_combine_outer(seeds.len());
    let per_seed_reports = if can_par {
        seeds
            .par_iter()
            .map(|&seed| {
                let report = calibrate_measurements_toys_with_reference(
                    spec, &reference, ci_level, solver, n_toys, seed,
                )?;
                Ok::<MeasurementCombinationCalibrationSeedReport, Error>(
                    MeasurementCombinationCalibrationSeedReport { seed, summary: report.summary },
                )
            })
            .collect::<Vec<_>>()
    } else {
        seeds
            .iter()
            .map(|&seed| {
                let report = calibrate_measurements_toys_with_reference(
                    spec, &reference, ci_level, solver, n_toys, seed,
                )?;
                Ok::<MeasurementCombinationCalibrationSeedReport, Error>(
                    MeasurementCombinationCalibrationSeedReport { seed, summary: report.summary },
                )
            })
            .collect::<Vec<_>>()
    };

    let mut per_seed = Vec::with_capacity(seeds.len());
    let mut mean_q = Vec::with_capacity(seeds.len());
    let mut mean_q_star = Vec::with_capacity(seeds.len());
    let mut mean_sigma = Vec::with_capacity(seeds.len());
    let mut mean_sigma_star = Vec::with_capacity(seeds.len());
    let mut ratio = Vec::with_capacity(seeds.len());
    let mut sigma_star_fraction = Vec::with_capacity(seeds.len());
    let mut bartlett_improves = 0usize;
    let mut toy_generation_method = None;
    let reference_sigma_scale = reference.diagnostics.bartlett.sigma_scale.unwrap_or(1.0);

    for report in per_seed_reports {
        let report = report?;
        if report.summary.bartlett_improves_mean_q {
            bartlett_improves += 1;
        }
        if let Some(existing) = toy_generation_method.as_ref() {
            if existing != &report.summary.toy_generation_method {
                return Err(Error::Computation(
                    "inconsistent toy_generation_method across calibration study seeds".to_string(),
                ));
            }
        } else {
            toy_generation_method = Some(report.summary.toy_generation_method.clone());
        }
        mean_q.push(report.summary.mean_q);
        mean_q_star.push(report.summary.mean_q_star);
        mean_sigma.push(report.summary.mean_sigma);
        mean_sigma_star.push(report.summary.mean_sigma_star);
        ratio.push(report.summary.mean_sigma_star_to_sigma_ratio);
        sigma_star_fraction.push(report.summary.sigma_star_ge_sigma_fraction);
        per_seed.push(report);
    }

    let min_mean_sigma = mean_sigma.iter().copied().fold(f64::INFINITY, f64::min);
    let max_mean_sigma = mean_sigma.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_mean_sigma_star = mean_sigma_star.iter().copied().fold(f64::INFINITY, f64::min);
    let max_mean_sigma_star = mean_sigma_star.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_ratio = ratio.iter().copied().fold(f64::INFINITY, f64::min);
    let max_ratio = ratio.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_sigma_star_fraction = sigma_star_fraction.iter().copied().fold(f64::INFINITY, f64::min);
    let max_sigma_star_fraction =
        sigma_star_fraction.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let max_abs_ratio_delta_from_reference =
        ratio.iter().map(|value| (*value - reference_sigma_scale).abs()).fold(0.0, f64::max);

    Ok(MeasurementCombinationCalibrationStudyReport {
        schema_version: MEASUREMENT_COMBINATION_CALIBRATION_STUDY_SCHEMA_V0.to_string(),
        poi: spec.poi.clone(),
        ci_level,
        n_toys,
        seeds: seeds.to_vec(),
        stability: GVM_STABILITY_STABLE.to_string(),
        reference,
        per_seed,
        aggregate: MeasurementCombinationCalibrationStudySummary {
            n_runs: seeds.len(),
            bartlett_improves_mean_q_fraction: bartlett_improves as f64 / seeds.len() as f64,
            mean_of_mean_q: mean(&mean_q),
            mean_of_mean_q_star: mean(&mean_q_star),
            mean_of_mean_sigma: mean(&mean_sigma),
            mean_of_mean_sigma_star: mean(&mean_sigma_star),
            min_mean_sigma,
            max_mean_sigma,
            min_mean_sigma_star,
            max_mean_sigma_star,
            min_mean_sigma_star_to_sigma_ratio: min_ratio,
            max_mean_sigma_star_to_sigma_ratio: max_ratio,
            max_abs_mean_sigma_star_to_sigma_ratio_delta_from_reference:
                max_abs_ratio_delta_from_reference,
            min_sigma_star_ge_sigma_fraction: min_sigma_star_fraction,
            max_sigma_star_ge_sigma_fraction: max_sigma_star_fraction,
            toy_generation_method: toy_generation_method.unwrap_or_else(|| {
                "measurement_side_gvm_unbiased_sigma2_star_normalized_spec".to_string()
            }),
        },
    })
}

pub fn study_measurement_combination_scenarios(
    spec: &MeasurementCombinationSpec,
    scenario_study: &MeasurementCombinationScenarioStudySpec,
    ci_level: f64,
) -> Result<MeasurementCombinationScenarioStudyReport> {
    study_measurement_combination_scenarios_with_solver(
        spec,
        scenario_study,
        ci_level,
        MeasurementCombinationSolver::Auto,
    )
}

pub fn study_measurement_combination_scenarios_with_solver(
    spec: &MeasurementCombinationSpec,
    scenario_study: &MeasurementCombinationScenarioStudySpec,
    ci_level: f64,
    solver: MeasurementCombinationSolver,
) -> Result<MeasurementCombinationScenarioStudyReport> {
    validate_scenario_study_spec(spec, scenario_study)?;
    let baseline = combine_measurements_with_solver(spec, ci_level, solver)?;
    let baseline_width = baseline.confidence_interval.upper - baseline.confidence_interval.lower;
    let baseline_sigma = baseline.confidence_interval.sigma.max(1e-12);

    let can_par = can_parallelize_measurement_combine_outer(scenario_study.scenarios.len());
    let scenario_results = if can_par {
        scenario_study
            .scenarios
            .par_iter()
            .map(|scenario| {
                build_measurement_combination_scenario_result(
                    spec,
                    scenario,
                    &baseline,
                    baseline_width,
                    baseline_sigma,
                    ci_level,
                    solver,
                )
            })
            .collect::<Vec<_>>()
    } else {
        scenario_study
            .scenarios
            .iter()
            .map(|scenario| {
                build_measurement_combination_scenario_result(
                    spec,
                    scenario,
                    &baseline,
                    baseline_width,
                    baseline_sigma,
                    ci_level,
                    solver,
                )
            })
            .collect::<Vec<_>>()
    };

    let mut scenarios = Vec::with_capacity(scenario_study.scenarios.len());
    let mut all_converged = true;
    let mut all_perturbative_within_threshold = true;
    let mut min_sigma_ratio = f64::INFINITY;
    let mut max_sigma_ratio = f64::NEG_INFINITY;
    let mut widest_interval_ratio = f64::NEG_INFINITY;
    let mut widest_interval_scenario = String::new();
    let mut largest_abs_mu_shift = f64::NEG_INFINITY;
    let mut largest_abs_mu_shift_scenario = String::new();
    let mut max_supported_systematics = 0usize;

    for scenario in scenario_results {
        let scenario = scenario?;
        all_converged &= scenario.result.converged;
        all_perturbative_within_threshold &= scenario.comparison.all_perturbative_within_threshold;
        min_sigma_ratio = min_sigma_ratio.min(scenario.comparison.sigma_ratio_to_baseline);
        max_sigma_ratio = max_sigma_ratio.max(scenario.comparison.sigma_ratio_to_baseline);
        max_supported_systematics =
            max_supported_systematics.max(scenario.comparison.supported_systematics.len());

        if scenario.comparison.interval_width_ratio_to_baseline > widest_interval_ratio {
            widest_interval_ratio = scenario.comparison.interval_width_ratio_to_baseline;
            widest_interval_scenario = scenario.name.clone();
        }
        if scenario.comparison.abs_mu_shift_from_baseline > largest_abs_mu_shift {
            largest_abs_mu_shift = scenario.comparison.abs_mu_shift_from_baseline;
            largest_abs_mu_shift_scenario = scenario.name.clone();
        }

        scenarios.push(scenario);
    }

    Ok(MeasurementCombinationScenarioStudyReport {
        schema_version: MEASUREMENT_COMBINATION_SCENARIO_STUDY_SCHEMA_V0.to_string(),
        poi: spec.poi.clone(),
        ci_level,
        stability: GVM_STABILITY_RESEARCH_GRADE.to_string(),
        baseline,
        scenarios,
        aggregate: MeasurementCombinationScenarioStudySummary {
            n_scenarios: scenario_study.scenarios.len(),
            all_converged,
            all_perturbative_within_threshold,
            min_sigma_ratio_to_baseline: min_sigma_ratio,
            max_sigma_ratio_to_baseline: max_sigma_ratio,
            largest_abs_mu_shift_scenario,
            largest_abs_mu_shift,
            widest_interval_scenario,
            widest_interval_ratio_to_baseline: widest_interval_ratio,
            max_supported_systematics,
        },
    })
}

pub fn run_measurement_combination_calibration_campaign(
    spec: &MeasurementCombinationSpec,
    scenario_study: &MeasurementCombinationScenarioStudySpec,
    ci_level: f64,
    n_toys: usize,
    seeds: &[u64],
) -> Result<MeasurementCombinationCalibrationCampaignReport> {
    run_measurement_combination_calibration_campaign_with_solver(
        spec,
        scenario_study,
        ci_level,
        MeasurementCombinationSolver::Auto,
        n_toys,
        seeds,
    )
}

pub fn run_measurement_combination_calibration_campaign_with_solver(
    spec: &MeasurementCombinationSpec,
    scenario_study: &MeasurementCombinationScenarioStudySpec,
    ci_level: f64,
    solver: MeasurementCombinationSolver,
    n_toys: usize,
    seeds: &[u64],
) -> Result<MeasurementCombinationCalibrationCampaignReport> {
    if seeds.is_empty() {
        return Err(Error::Validation(
            "calibration campaign requires at least one seed".to_string(),
        ));
    }
    let scenario_report = study_measurement_combination_scenarios_with_solver(
        spec,
        scenario_study,
        ci_level,
        solver,
    )?;
    let baseline_width = scenario_report.baseline.confidence_interval.upper
        - scenario_report.baseline.confidence_interval.lower;
    let baseline_sigma = scenario_report.baseline.confidence_interval.sigma.max(1e-12);

    let can_par = can_parallelize_measurement_combine_outer(scenario_study.scenarios.len());
    let campaign_scenarios = if can_par {
        scenario_study
            .scenarios
            .par_iter()
            .zip(scenario_report.scenarios.par_iter())
            .map(|(scenario_spec, scenario_fit)| {
                build_measurement_combination_calibration_campaign_scenario(
                    spec,
                    scenario_spec,
                    scenario_fit,
                    baseline_width,
                    baseline_sigma,
                    ci_level,
                    solver,
                    n_toys,
                    seeds,
                )
            })
            .collect::<Vec<_>>()
    } else {
        scenario_study
            .scenarios
            .iter()
            .zip(&scenario_report.scenarios)
            .map(|(scenario_spec, scenario_fit)| {
                build_measurement_combination_calibration_campaign_scenario(
                    spec,
                    scenario_spec,
                    scenario_fit,
                    baseline_width,
                    baseline_sigma,
                    ci_level,
                    solver,
                    n_toys,
                    seeds,
                )
            })
            .collect::<Vec<_>>()
    };

    let mut scenarios = Vec::with_capacity(scenario_study.scenarios.len());
    let mut all_converged = true;
    let mut all_sigma_star_fraction_ge_099 = true;
    let mut min_fit_sigma_ratio = f64::INFINITY;
    let mut max_fit_sigma_ratio = f64::NEG_INFINITY;
    let mut min_calibration_ratio = f64::INFINITY;
    let mut max_calibration_ratio = f64::NEG_INFINITY;
    let mut widest_fit_interval_ratio = f64::NEG_INFINITY;
    let mut widest_fit_interval_scenario = String::new();
    let mut highest_calibration_sigma_ratio = f64::NEG_INFINITY;
    let mut highest_calibration_sigma_ratio_scenario = String::new();

    for scenario in campaign_scenarios {
        let scenario = scenario?;
        all_converged &= scenario.fit.converged;
        all_sigma_star_fraction_ge_099 &=
            scenario.comparison.calibration_min_sigma_star_ge_sigma_fraction >= 0.99;
        min_fit_sigma_ratio =
            min_fit_sigma_ratio.min(scenario.comparison.fit_sigma_ratio_to_baseline);
        max_fit_sigma_ratio =
            max_fit_sigma_ratio.max(scenario.comparison.fit_sigma_ratio_to_baseline);
        min_calibration_ratio = min_calibration_ratio
            .min(scenario.comparison.calibration_min_mean_sigma_star_to_sigma_ratio);
        max_calibration_ratio = max_calibration_ratio
            .max(scenario.comparison.calibration_max_mean_sigma_star_to_sigma_ratio);

        if scenario.comparison.fit_interval_width_ratio_to_baseline > widest_fit_interval_ratio {
            widest_fit_interval_ratio = scenario.comparison.fit_interval_width_ratio_to_baseline;
            widest_fit_interval_scenario = scenario.name.clone();
        }
        if scenario.comparison.calibration_max_mean_sigma_star_to_sigma_ratio
            > highest_calibration_sigma_ratio
        {
            highest_calibration_sigma_ratio =
                scenario.comparison.calibration_max_mean_sigma_star_to_sigma_ratio;
            highest_calibration_sigma_ratio_scenario = scenario.name.clone();
        }

        scenarios.push(scenario);
    }

    Ok(MeasurementCombinationCalibrationCampaignReport {
        schema_version: MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_SCHEMA_V0.to_string(),
        poi: spec.poi.clone(),
        ci_level,
        n_toys,
        seeds: seeds.to_vec(),
        stability: GVM_STABILITY_RESEARCH_GRADE.to_string(),
        baseline: scenario_report.baseline,
        scenarios,
        aggregate: MeasurementCombinationCalibrationCampaignSummary {
            n_scenarios: scenario_study.scenarios.len(),
            all_converged,
            all_calibration_sigma_star_ge_sigma_fraction_ge_0_99: all_sigma_star_fraction_ge_099,
            max_fit_sigma_ratio_to_baseline: max_fit_sigma_ratio,
            min_fit_sigma_ratio_to_baseline: min_fit_sigma_ratio,
            max_calibration_mean_sigma_star_to_sigma_ratio: max_calibration_ratio,
            min_calibration_mean_sigma_star_to_sigma_ratio: min_calibration_ratio,
            widest_fit_interval_scenario,
            highest_calibration_sigma_ratio_scenario,
        },
    })
}

pub fn compare_measurement_combination_scenario_study_solvers(
    spec: &MeasurementCombinationSpec,
    scenario_study: &MeasurementCombinationScenarioStudySpec,
    ci_level: f64,
    lhs_solver: MeasurementCombinationSolver,
    rhs_solver: MeasurementCombinationSolver,
) -> Result<MeasurementCombinationScenarioStudySolverParityReport> {
    let lhs = study_measurement_combination_scenarios_with_solver(
        spec,
        scenario_study,
        ci_level,
        lhs_solver,
    )?;
    let rhs = study_measurement_combination_scenarios_with_solver(
        spec,
        scenario_study,
        ci_level,
        rhs_solver,
    )?;

    compare_measurement_combination_scenario_study_reports(
        &lhs,
        &rhs,
        solver_label(lhs_solver),
        solver_label(rhs_solver),
    )
}

pub fn compare_measurement_combination_scenario_study_reports(
    lhs: &MeasurementCombinationScenarioStudyReport,
    rhs: &MeasurementCombinationScenarioStudyReport,
    lhs_solver: &str,
    rhs_solver: &str,
) -> Result<MeasurementCombinationScenarioStudySolverParityReport> {
    if lhs.schema_version != MEASUREMENT_COMBINATION_SCENARIO_STUDY_SCHEMA_V0 {
        return Err(Error::Validation(format!(
            "unsupported lhs scenario study schema_version '{}'",
            lhs.schema_version
        )));
    }
    if rhs.schema_version != MEASUREMENT_COMBINATION_SCENARIO_STUDY_SCHEMA_V0 {
        return Err(Error::Validation(format!(
            "unsupported rhs scenario study schema_version '{}'",
            rhs.schema_version
        )));
    }
    if lhs.poi != rhs.poi {
        return Err(Error::Computation(format!(
            "scenario solver parity requires matching POIs: lhs='{}' rhs='{}'",
            lhs.poi, rhs.poi
        )));
    }
    if (lhs.ci_level - rhs.ci_level).abs() > 1e-12 {
        return Err(Error::Computation(format!(
            "scenario solver parity requires matching ci_level: lhs={} rhs={}",
            lhs.ci_level, rhs.ci_level
        )));
    }
    if lhs.scenarios.len() != rhs.scenarios.len() {
        return Err(Error::Computation(
            "scenario solver parity reports produced different scenario counts".to_string(),
        ));
    }

    let baseline = MeasurementCombinationSolverParityBaseline {
        lhs_optimizer_method: lhs.baseline.optimizer.method.clone(),
        rhs_optimizer_method: rhs.baseline.optimizer.method.clone(),
        mu_abs_diff: (lhs.baseline.mu_hat - rhs.baseline.mu_hat).abs(),
        sigma_abs_diff: (lhs.baseline.confidence_interval.sigma
            - rhs.baseline.confidence_interval.sigma)
            .abs(),
        sigma_rel_diff: (lhs.baseline.confidence_interval.sigma
            - rhs.baseline.confidence_interval.sigma)
            .abs()
            / lhs
                .baseline
                .confidence_interval
                .sigma
                .abs()
                .max(rhs.baseline.confidence_interval.sigma.abs())
                .max(1e-12),
        chi2_abs_diff: (lhs.baseline.goodness_of_fit.chi2 - rhs.baseline.goodness_of_fit.chi2)
            .abs(),
        q_star_abs_diff: optional_abs_diff(
            lhs.baseline.diagnostics.bartlett.q_star,
            rhs.baseline.diagnostics.bartlett.q_star,
        ),
    };

    let mut scenarios = Vec::with_capacity(lhs.scenarios.len());
    let mut max_mu_abs_diff = baseline.mu_abs_diff;
    let mut max_mu_abs_diff_scenario = "baseline".to_string();
    let mut max_sigma_rel_diff = baseline.sigma_rel_diff;
    let mut max_sigma_rel_diff_scenario = "baseline".to_string();
    let mut max_q_star_abs_diff = baseline.q_star_abs_diff.unwrap_or(0.0);
    let mut max_q_star_abs_diff_scenario = "baseline".to_string();
    let mut all_scenarios_converged = lhs.baseline.converged && rhs.baseline.converged;

    for (lhs_scenario, rhs_scenario) in lhs.scenarios.iter().zip(&rhs.scenarios) {
        if lhs_scenario.name != rhs_scenario.name {
            return Err(Error::Computation(format!(
                "scenario solver parity mismatch: lhs='{}' rhs='{}'",
                lhs_scenario.name, rhs_scenario.name
            )));
        }
        let mu_abs_diff = (lhs_scenario.result.mu_hat - rhs_scenario.result.mu_hat).abs();
        let sigma_abs_diff = (lhs_scenario.result.confidence_interval.sigma
            - rhs_scenario.result.confidence_interval.sigma)
            .abs();
        let sigma_rel_diff = sigma_abs_diff
            / lhs_scenario
                .result
                .confidence_interval
                .sigma
                .abs()
                .max(rhs_scenario.result.confidence_interval.sigma.abs())
                .max(1e-12);
        let q_star_abs_diff = optional_abs_diff(
            lhs_scenario.result.diagnostics.bartlett.q_star,
            rhs_scenario.result.diagnostics.bartlett.q_star,
        );
        if mu_abs_diff > max_mu_abs_diff {
            max_mu_abs_diff = mu_abs_diff;
            max_mu_abs_diff_scenario = lhs_scenario.name.clone();
        }
        if sigma_rel_diff > max_sigma_rel_diff {
            max_sigma_rel_diff = sigma_rel_diff;
            max_sigma_rel_diff_scenario = lhs_scenario.name.clone();
        }
        if q_star_abs_diff.unwrap_or(0.0) > max_q_star_abs_diff {
            max_q_star_abs_diff = q_star_abs_diff.unwrap_or(0.0);
            max_q_star_abs_diff_scenario = lhs_scenario.name.clone();
        }
        all_scenarios_converged &= lhs_scenario.result.converged && rhs_scenario.result.converged;
        scenarios.push(MeasurementCombinationScenarioSolverParityEntry {
            name: lhs_scenario.name.clone(),
            assignments: lhs_scenario.assignments.clone(),
            lhs_optimizer_method: lhs_scenario.result.optimizer.method.clone(),
            rhs_optimizer_method: rhs_scenario.result.optimizer.method.clone(),
            mu_abs_diff,
            sigma_abs_diff,
            sigma_rel_diff,
            chi2_abs_diff: (lhs_scenario.result.goodness_of_fit.chi2
                - rhs_scenario.result.goodness_of_fit.chi2)
                .abs(),
            q_star_abs_diff,
            same_supported_systematics: lhs_scenario.comparison.supported_systematics
                == rhs_scenario.comparison.supported_systematics,
            lhs_supported_systematics: lhs_scenario.comparison.supported_systematics.clone(),
            rhs_supported_systematics: rhs_scenario.comparison.supported_systematics.clone(),
            lhs_all_perturbative_within_threshold: lhs_scenario
                .comparison
                .all_perturbative_within_threshold,
            rhs_all_perturbative_within_threshold: rhs_scenario
                .comparison
                .all_perturbative_within_threshold,
        });
    }

    Ok(MeasurementCombinationScenarioStudySolverParityReport {
        schema_version: MEASUREMENT_COMBINATION_SCENARIO_STUDY_SOLVER_PARITY_SCHEMA_V0.to_string(),
        poi: lhs.poi.clone(),
        ci_level: lhs.ci_level,
        lhs_solver: lhs_solver.to_string(),
        rhs_solver: rhs_solver.to_string(),
        stability: GVM_STABILITY_RESEARCH_GRADE.to_string(),
        baseline,
        scenarios,
        aggregate: MeasurementCombinationScenarioSolverParitySummary {
            n_scenarios: lhs.scenarios.len(),
            max_mu_abs_diff,
            max_mu_abs_diff_scenario,
            max_sigma_rel_diff,
            max_sigma_rel_diff_scenario,
            max_q_star_abs_diff,
            max_q_star_abs_diff_scenario,
            all_scenarios_converged,
        },
    })
}

pub fn compare_measurement_combination_calibration_campaign_solvers(
    spec: &MeasurementCombinationSpec,
    scenario_study: &MeasurementCombinationScenarioStudySpec,
    ci_level: f64,
    lhs_solver: MeasurementCombinationSolver,
    rhs_solver: MeasurementCombinationSolver,
    n_toys: usize,
    seeds: &[u64],
) -> Result<MeasurementCombinationCalibrationCampaignSolverParityReport> {
    let lhs = run_measurement_combination_calibration_campaign_with_solver(
        spec,
        scenario_study,
        ci_level,
        lhs_solver,
        n_toys,
        seeds,
    )?;
    let rhs = run_measurement_combination_calibration_campaign_with_solver(
        spec,
        scenario_study,
        ci_level,
        rhs_solver,
        n_toys,
        seeds,
    )?;

    compare_measurement_combination_calibration_campaign_reports(
        &lhs,
        &rhs,
        solver_label(lhs_solver),
        solver_label(rhs_solver),
    )
}

pub fn compare_measurement_combination_calibration_campaign_reports(
    lhs: &MeasurementCombinationCalibrationCampaignReport,
    rhs: &MeasurementCombinationCalibrationCampaignReport,
    lhs_solver: &str,
    rhs_solver: &str,
) -> Result<MeasurementCombinationCalibrationCampaignSolverParityReport> {
    if lhs.schema_version != MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_SCHEMA_V0 {
        return Err(Error::Validation(format!(
            "unsupported lhs calibration campaign schema_version '{}'",
            lhs.schema_version
        )));
    }
    if rhs.schema_version != MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_SCHEMA_V0 {
        return Err(Error::Validation(format!(
            "unsupported rhs calibration campaign schema_version '{}'",
            rhs.schema_version
        )));
    }
    if lhs.poi != rhs.poi {
        return Err(Error::Computation(format!(
            "campaign solver parity requires matching POIs: lhs='{}' rhs='{}'",
            lhs.poi, rhs.poi
        )));
    }
    if (lhs.ci_level - rhs.ci_level).abs() > 1e-12 {
        return Err(Error::Computation(format!(
            "campaign solver parity requires matching ci_level: lhs={} rhs={}",
            lhs.ci_level, rhs.ci_level
        )));
    }
    if lhs.n_toys != rhs.n_toys {
        return Err(Error::Computation(format!(
            "campaign solver parity requires matching n_toys: lhs={} rhs={}",
            lhs.n_toys, rhs.n_toys
        )));
    }
    if lhs.seeds != rhs.seeds {
        return Err(Error::Computation(
            "campaign solver parity requires matching seeds".to_string(),
        ));
    }
    if lhs.scenarios.len() != rhs.scenarios.len() {
        return Err(Error::Computation(
            "campaign solver parity reports produced different scenario counts".to_string(),
        ));
    }

    let baseline = MeasurementCombinationSolverParityBaseline {
        lhs_optimizer_method: lhs.baseline.optimizer.method.clone(),
        rhs_optimizer_method: rhs.baseline.optimizer.method.clone(),
        mu_abs_diff: (lhs.baseline.mu_hat - rhs.baseline.mu_hat).abs(),
        sigma_abs_diff: (lhs.baseline.confidence_interval.sigma
            - rhs.baseline.confidence_interval.sigma)
            .abs(),
        sigma_rel_diff: (lhs.baseline.confidence_interval.sigma
            - rhs.baseline.confidence_interval.sigma)
            .abs()
            / lhs
                .baseline
                .confidence_interval
                .sigma
                .abs()
                .max(rhs.baseline.confidence_interval.sigma.abs())
                .max(1e-12),
        chi2_abs_diff: (lhs.baseline.goodness_of_fit.chi2 - rhs.baseline.goodness_of_fit.chi2)
            .abs(),
        q_star_abs_diff: optional_abs_diff(
            lhs.baseline.diagnostics.bartlett.q_star,
            rhs.baseline.diagnostics.bartlett.q_star,
        ),
    };

    let mut scenarios = Vec::with_capacity(lhs.scenarios.len());
    let mut max_fit_mu_abs_diff = baseline.mu_abs_diff;
    let mut max_fit_mu_abs_diff_scenario = "baseline".to_string();
    let mut max_fit_sigma_rel_diff = baseline.sigma_rel_diff;
    let mut max_fit_sigma_rel_diff_scenario = "baseline".to_string();
    let mut max_fit_q_star_abs_diff = baseline.q_star_abs_diff.unwrap_or(0.0);
    let mut max_fit_q_star_abs_diff_scenario = "baseline".to_string();
    let mut max_calibration_ratio_center_abs_diff = 0.0;
    let mut max_calibration_ratio_center_abs_diff_scenario = String::new();
    let mut all_scenarios_converged = lhs.baseline.converged && rhs.baseline.converged;

    for (lhs_scenario, rhs_scenario) in lhs.scenarios.iter().zip(&rhs.scenarios) {
        if lhs_scenario.name != rhs_scenario.name {
            return Err(Error::Computation(format!(
                "campaign solver parity mismatch: lhs='{}' rhs='{}'",
                lhs_scenario.name, rhs_scenario.name
            )));
        }
        let fit_mu_abs_diff = (lhs_scenario.fit.mu_hat - rhs_scenario.fit.mu_hat).abs();
        let fit_sigma_abs_diff = (lhs_scenario.fit.confidence_interval.sigma
            - rhs_scenario.fit.confidence_interval.sigma)
            .abs();
        let fit_sigma_rel_diff = fit_sigma_abs_diff
            / lhs_scenario
                .fit
                .confidence_interval
                .sigma
                .abs()
                .max(rhs_scenario.fit.confidence_interval.sigma.abs())
                .max(1e-12);
        let fit_q_star_abs_diff = optional_abs_diff(
            lhs_scenario.fit.diagnostics.bartlett.q_star,
            rhs_scenario.fit.diagnostics.bartlett.q_star,
        );
        let lhs_ratio_center = 0.5
            * (lhs_scenario.calibration.min_mean_sigma_star_to_sigma_ratio
                + lhs_scenario.calibration.max_mean_sigma_star_to_sigma_ratio);
        let rhs_ratio_center = 0.5
            * (rhs_scenario.calibration.min_mean_sigma_star_to_sigma_ratio
                + rhs_scenario.calibration.max_mean_sigma_star_to_sigma_ratio);
        let ratio_center_abs_diff = (lhs_ratio_center - rhs_ratio_center).abs();
        if fit_mu_abs_diff > max_fit_mu_abs_diff {
            max_fit_mu_abs_diff = fit_mu_abs_diff;
            max_fit_mu_abs_diff_scenario = lhs_scenario.name.clone();
        }
        if fit_sigma_rel_diff > max_fit_sigma_rel_diff {
            max_fit_sigma_rel_diff = fit_sigma_rel_diff;
            max_fit_sigma_rel_diff_scenario = lhs_scenario.name.clone();
        }
        if fit_q_star_abs_diff.unwrap_or(0.0) > max_fit_q_star_abs_diff {
            max_fit_q_star_abs_diff = fit_q_star_abs_diff.unwrap_or(0.0);
            max_fit_q_star_abs_diff_scenario = lhs_scenario.name.clone();
        }
        if ratio_center_abs_diff > max_calibration_ratio_center_abs_diff {
            max_calibration_ratio_center_abs_diff = ratio_center_abs_diff;
            max_calibration_ratio_center_abs_diff_scenario = lhs_scenario.name.clone();
        }
        all_scenarios_converged &= lhs_scenario.fit.converged && rhs_scenario.fit.converged;
        scenarios.push(MeasurementCombinationCalibrationCampaignSolverParityEntry {
            name: lhs_scenario.name.clone(),
            assignments: lhs_scenario.assignments.clone(),
            lhs_fit_optimizer_method: lhs_scenario.fit.optimizer.method.clone(),
            rhs_fit_optimizer_method: rhs_scenario.fit.optimizer.method.clone(),
            fit_mu_abs_diff,
            fit_sigma_abs_diff,
            fit_sigma_rel_diff,
            fit_q_star_abs_diff,
            mean_sigma_star_to_sigma_ratio_center_abs_diff: ratio_center_abs_diff,
            sigma_star_ge_sigma_fraction_abs_diff: (lhs_scenario
                .calibration
                .min_sigma_star_ge_sigma_fraction
                - rhs_scenario.calibration.min_sigma_star_ge_sigma_fraction)
                .abs(),
            bartlett_improves_mean_q_fraction_abs_diff: (lhs_scenario
                .calibration
                .bartlett_improves_mean_q_fraction
                - rhs_scenario.calibration.bartlett_improves_mean_q_fraction)
                .abs(),
            lhs_toy_generation_method: lhs_scenario.calibration.toy_generation_method.clone(),
            rhs_toy_generation_method: rhs_scenario.calibration.toy_generation_method.clone(),
        });
    }

    Ok(MeasurementCombinationCalibrationCampaignSolverParityReport {
        schema_version: MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_SOLVER_PARITY_SCHEMA_V0
            .to_string(),
        poi: lhs.poi.clone(),
        ci_level: lhs.ci_level,
        n_toys: lhs.n_toys,
        seeds: lhs.seeds.clone(),
        lhs_solver: lhs_solver.to_string(),
        rhs_solver: rhs_solver.to_string(),
        stability: GVM_STABILITY_RESEARCH_GRADE.to_string(),
        baseline,
        scenarios,
        aggregate: MeasurementCombinationCalibrationCampaignSolverParitySummary {
            n_scenarios: lhs.scenarios.len(),
            max_fit_mu_abs_diff,
            max_fit_mu_abs_diff_scenario,
            max_fit_sigma_rel_diff,
            max_fit_sigma_rel_diff_scenario,
            max_fit_q_star_abs_diff,
            max_fit_q_star_abs_diff_scenario,
            max_calibration_ratio_center_abs_diff,
            max_calibration_ratio_center_abs_diff_scenario,
            all_scenarios_converged,
        },
    })
}

pub fn render_measurement_combination_scenario_study_solver_parity_markdown(
    report: &MeasurementCombinationScenarioStudySolverParityReport,
) -> Result<String> {
    if report.schema_version != MEASUREMENT_COMBINATION_SCENARIO_STUDY_SOLVER_PARITY_SCHEMA_V0 {
        return Err(Error::Validation(format!(
            "unsupported scenario-study solver parity schema_version '{}'",
            report.schema_version
        )));
    }

    let mut out = String::new();
    out.push_str("# Measurement Combination Scenario Study Solver Parity\n\n");
    out.push_str(&format!("- POI: `{}`\n", report.poi));
    out.push_str(&format!("- Solver pair: `{}` vs `{}`\n", report.lhs_solver, report.rhs_solver));
    out.push_str(&format!(
        "- Baseline mu abs diff: `{}`\n",
        fmt_digest_number(report.baseline.mu_abs_diff)
    ));
    out.push_str(&format!(
        "- Baseline sigma rel diff: `{}`\n\n",
        fmt_digest_number(report.baseline.sigma_rel_diff)
    ));
    out.push_str(
        "| Scenario | LHS method | RHS method | Mu abs diff | Sigma abs diff | Sigma rel diff | q* abs diff | Same supported systs |\n",
    );
    out.push_str("| --- | --- | --- | ---: | ---: | ---: | ---: | --- |\n");
    for scenario in &report.scenarios {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
            scenario.name,
            scenario.lhs_optimizer_method,
            scenario.rhs_optimizer_method,
            fmt_digest_number(scenario.mu_abs_diff),
            fmt_digest_number(scenario.sigma_abs_diff),
            fmt_digest_number(scenario.sigma_rel_diff),
            scenario.q_star_abs_diff.map(fmt_digest_number).unwrap_or_else(|| "-".to_string()),
            if scenario.same_supported_systematics { "yes" } else { "no" }
        ));
    }
    Ok(out)
}

pub fn render_measurement_combination_calibration_campaign_solver_parity_markdown(
    report: &MeasurementCombinationCalibrationCampaignSolverParityReport,
) -> Result<String> {
    if report.schema_version != MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_SOLVER_PARITY_SCHEMA_V0
    {
        return Err(Error::Validation(format!(
            "unsupported calibration-campaign solver parity schema_version '{}'",
            report.schema_version
        )));
    }

    let mut out = String::new();
    out.push_str("# Measurement Combination Calibration Campaign Solver Parity\n\n");
    out.push_str(&format!("- POI: `{}`\n", report.poi));
    out.push_str(&format!("- Solver pair: `{}` vs `{}`\n", report.lhs_solver, report.rhs_solver));
    out.push_str(&format!(
        "- Seeds: `{}`\n",
        report.seeds.iter().map(|seed| seed.to_string()).collect::<Vec<_>>().join("`, `")
    ));
    out.push_str(&format!(
        "- Baseline mu abs diff: `{}`\n",
        fmt_digest_number(report.baseline.mu_abs_diff)
    ));
    out.push_str(&format!(
        "- Baseline sigma rel diff: `{}`\n\n",
        fmt_digest_number(report.baseline.sigma_rel_diff)
    ));
    out.push_str(
        "| Scenario | LHS fit method | RHS fit method | Fit mu abs diff | Fit sigma rel diff | Fit q* abs diff | Cal ratio center abs diff | LHS toy method | RHS toy method |\n",
    );
    out.push_str("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |\n");
    for scenario in &report.scenarios {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            scenario.name,
            scenario.lhs_fit_optimizer_method,
            scenario.rhs_fit_optimizer_method,
            fmt_digest_number(scenario.fit_mu_abs_diff),
            fmt_digest_number(scenario.fit_sigma_rel_diff),
            scenario.fit_q_star_abs_diff.map(fmt_digest_number).unwrap_or_else(|| "-".to_string()),
            fmt_digest_number(scenario.mean_sigma_star_to_sigma_ratio_center_abs_diff),
            scenario.lhs_toy_generation_method,
            scenario.rhs_toy_generation_method,
        ));
    }
    Ok(out)
}

pub fn summarize_measurement_combination_scenario_study_solver_parity(
    report: &MeasurementCombinationScenarioStudySolverParityReport,
) -> Result<MeasurementCombinationScenarioStudySolverParityDigest> {
    if report.schema_version != MEASUREMENT_COMBINATION_SCENARIO_STUDY_SOLVER_PARITY_SCHEMA_V0 {
        return Err(Error::Validation(format!(
            "unsupported scenario-study solver parity schema_version '{}'",
            report.schema_version
        )));
    }
    if report.scenarios.is_empty() {
        return Err(Error::Validation(
            "scenario-study solver parity digest requires at least one scenario".to_string(),
        ));
    }

    let mu_ranks = descending_ranks(
        &report.scenarios,
        |scenario| &scenario.name,
        |scenario| scenario.mu_abs_diff,
    );
    let sigma_ranks = descending_ranks(
        &report.scenarios,
        |scenario| &scenario.name,
        |scenario| scenario.sigma_rel_diff,
    );
    let q_star_ranks = descending_ranks(
        &report.scenarios,
        |scenario| &scenario.name,
        |scenario| scenario.q_star_abs_diff.unwrap_or(0.0),
    );

    let mut supported_systematics_mismatch_scenarios = Vec::new();
    let mut perturbative_overlap_failure_scenarios = Vec::new();
    let mut scenarios = Vec::with_capacity(report.scenarios.len());
    for scenario in &report.scenarios {
        let both_perturbative_within_threshold = scenario.lhs_all_perturbative_within_threshold
            && scenario.rhs_all_perturbative_within_threshold;
        if !scenario.same_supported_systematics {
            supported_systematics_mismatch_scenarios.push(scenario.name.clone());
        }
        if !both_perturbative_within_threshold {
            perturbative_overlap_failure_scenarios.push(scenario.name.clone());
        }
        let label = classify_scenario_solver_parity_entry(
            &scenario.name,
            scenario.same_supported_systematics,
            both_perturbative_within_threshold,
            &report.aggregate.max_mu_abs_diff_scenario,
            &report.aggregate.max_sigma_rel_diff_scenario,
            &report.aggregate.max_q_star_abs_diff_scenario,
        );
        scenarios.push(MeasurementCombinationScenarioStudySolverParityDigestEntry {
            name: scenario.name.clone(),
            n_assignments: scenario.assignments.len(),
            mu_gap_rank: *mu_ranks.get(&scenario.name).expect("mu rank should exist"),
            sigma_gap_rank: *sigma_ranks.get(&scenario.name).expect("sigma rank should exist"),
            q_star_gap_rank: *q_star_ranks.get(&scenario.name).expect("q* rank should exist"),
            mu_abs_diff: scenario.mu_abs_diff,
            sigma_rel_diff: scenario.sigma_rel_diff,
            q_star_abs_diff: scenario.q_star_abs_diff,
            same_supported_systematics: scenario.same_supported_systematics,
            both_perturbative_within_threshold,
            label,
        });
    }

    Ok(MeasurementCombinationScenarioStudySolverParityDigest {
        schema_version: MEASUREMENT_COMBINATION_SCENARIO_STUDY_SOLVER_PARITY_DIGEST_SCHEMA_V0
            .to_string(),
        source_schema_version: report.schema_version.clone(),
        poi: report.poi.clone(),
        ci_level: report.ci_level,
        lhs_solver: report.lhs_solver.clone(),
        rhs_solver: report.rhs_solver.clone(),
        stability: report.stability.clone(),
        baseline: report.baseline.clone(),
        dominant_mu_gap_scenario: report.aggregate.max_mu_abs_diff_scenario.clone(),
        dominant_sigma_gap_scenario: report.aggregate.max_sigma_rel_diff_scenario.clone(),
        dominant_q_star_gap_scenario: report.aggregate.max_q_star_abs_diff_scenario.clone(),
        aggregate: MeasurementCombinationScenarioStudySolverParityDigestSummary {
            n_scenarios: report.aggregate.n_scenarios,
            max_mu_abs_diff: report.aggregate.max_mu_abs_diff,
            max_sigma_rel_diff: report.aggregate.max_sigma_rel_diff,
            max_q_star_abs_diff: report.aggregate.max_q_star_abs_diff,
            n_supported_systematics_mismatch_scenarios: supported_systematics_mismatch_scenarios
                .len(),
            n_perturbative_overlap_failure_scenarios: perturbative_overlap_failure_scenarios.len(),
            supported_systematics_mismatch_scenarios,
            perturbative_overlap_failure_scenarios,
        },
        scenarios,
    })
}

pub fn render_measurement_combination_scenario_study_solver_parity_digest_markdown(
    digest: &MeasurementCombinationScenarioStudySolverParityDigest,
) -> Result<String> {
    if digest.schema_version
        != MEASUREMENT_COMBINATION_SCENARIO_STUDY_SOLVER_PARITY_DIGEST_SCHEMA_V0
    {
        return Err(Error::Validation(format!(
            "unsupported scenario-study solver parity digest schema_version '{}'",
            digest.schema_version
        )));
    }

    let mut out = String::new();
    out.push_str("# Measurement Combination Scenario Study Solver Parity Digest\n\n");
    out.push_str(&format!("- POI: `{}`\n", digest.poi));
    out.push_str(&format!("- Solver pair: `{}` vs `{}`\n", digest.lhs_solver, digest.rhs_solver));
    out.push_str(&format!("- Dominant mu gap scenario: `{}`\n", digest.dominant_mu_gap_scenario));
    out.push_str(&format!(
        "- Dominant sigma gap scenario: `{}`\n",
        digest.dominant_sigma_gap_scenario
    ));
    out.push_str(&format!(
        "- Dominant q* gap scenario: `{}`\n",
        digest.dominant_q_star_gap_scenario
    ));
    out.push_str(&format!(
        "- Supported-systematics mismatches: `{}`\n",
        fmt_digest_label_list(&digest.aggregate.supported_systematics_mismatch_scenarios)
    ));
    out.push_str(&format!(
        "- Perturbative overlap failures: `{}`\n\n",
        fmt_digest_label_list(&digest.aggregate.perturbative_overlap_failure_scenarios)
    ));
    out.push_str(
        "| Scenario | Label | Assignments | Mu rank | Sigma rank | q* rank | Mu abs diff | Sigma rel diff | q* abs diff | Same supported systs | Both perturbative |\n",
    );
    out.push_str("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n");
    for scenario in &digest.scenarios {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            scenario.name,
            scenario.label,
            scenario.n_assignments,
            scenario.mu_gap_rank,
            scenario.sigma_gap_rank,
            scenario.q_star_gap_rank,
            fmt_digest_number(scenario.mu_abs_diff),
            fmt_digest_number(scenario.sigma_rel_diff),
            scenario.q_star_abs_diff.map(fmt_digest_number).unwrap_or_else(|| "-".to_string()),
            if scenario.same_supported_systematics { "yes" } else { "no" },
            if scenario.both_perturbative_within_threshold { "yes" } else { "no" }
        ));
    }
    Ok(out)
}

pub fn summarize_measurement_combination_calibration_campaign_solver_parity(
    report: &MeasurementCombinationCalibrationCampaignSolverParityReport,
) -> Result<MeasurementCombinationCalibrationCampaignSolverParityDigest> {
    if report.schema_version != MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_SOLVER_PARITY_SCHEMA_V0
    {
        return Err(Error::Validation(format!(
            "unsupported calibration-campaign solver parity schema_version '{}'",
            report.schema_version
        )));
    }
    if report.scenarios.is_empty() {
        return Err(Error::Validation(
            "calibration-campaign solver parity digest requires at least one scenario".to_string(),
        ));
    }

    let fit_ranks = descending_ranks(
        &report.scenarios,
        |scenario| &scenario.name,
        |scenario| scenario.fit_sigma_rel_diff,
    );
    let calibration_ranks = descending_ranks(
        &report.scenarios,
        |scenario| &scenario.name,
        |scenario| scenario.mean_sigma_star_to_sigma_ratio_center_abs_diff,
    );
    let mut toy_generation_method_mismatch_scenarios = Vec::new();
    let mut scenarios = Vec::with_capacity(report.scenarios.len());
    for scenario in &report.scenarios {
        let same_toy_generation_method =
            scenario.lhs_toy_generation_method == scenario.rhs_toy_generation_method;
        if !same_toy_generation_method {
            toy_generation_method_mismatch_scenarios.push(scenario.name.clone());
        }
        let label = classify_calibration_solver_parity_entry(
            &scenario.name,
            same_toy_generation_method,
            &report.aggregate.max_fit_sigma_rel_diff_scenario,
            &report.aggregate.max_calibration_ratio_center_abs_diff_scenario,
        );
        scenarios.push(MeasurementCombinationCalibrationCampaignSolverParityDigestEntry {
            name: scenario.name.clone(),
            n_assignments: scenario.assignments.len(),
            fit_gap_rank: *fit_ranks.get(&scenario.name).expect("fit rank should exist"),
            calibration_gap_rank: *calibration_ranks
                .get(&scenario.name)
                .expect("calibration rank should exist"),
            fit_mu_abs_diff: scenario.fit_mu_abs_diff,
            fit_sigma_rel_diff: scenario.fit_sigma_rel_diff,
            fit_q_star_abs_diff: scenario.fit_q_star_abs_diff,
            calibration_ratio_center_abs_diff: scenario
                .mean_sigma_star_to_sigma_ratio_center_abs_diff,
            sigma_star_ge_sigma_fraction_abs_diff: scenario.sigma_star_ge_sigma_fraction_abs_diff,
            bartlett_improves_mean_q_fraction_abs_diff: scenario
                .bartlett_improves_mean_q_fraction_abs_diff,
            same_toy_generation_method,
            label,
        });
    }

    Ok(MeasurementCombinationCalibrationCampaignSolverParityDigest {
        schema_version: MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_SOLVER_PARITY_DIGEST_SCHEMA_V0
            .to_string(),
        source_schema_version: report.schema_version.clone(),
        poi: report.poi.clone(),
        ci_level: report.ci_level,
        lhs_solver: report.lhs_solver.clone(),
        rhs_solver: report.rhs_solver.clone(),
        stability: report.stability.clone(),
        n_toys: report.n_toys,
        seeds: report.seeds.clone(),
        baseline: report.baseline.clone(),
        dominant_fit_gap_scenario: report.aggregate.max_fit_sigma_rel_diff_scenario.clone(),
        dominant_calibration_gap_scenario: report
            .aggregate
            .max_calibration_ratio_center_abs_diff_scenario
            .clone(),
        aggregate: MeasurementCombinationCalibrationCampaignSolverParityDigestSummary {
            n_scenarios: report.aggregate.n_scenarios,
            max_fit_mu_abs_diff: report.aggregate.max_fit_mu_abs_diff,
            max_fit_sigma_rel_diff: report.aggregate.max_fit_sigma_rel_diff,
            max_fit_q_star_abs_diff: report.aggregate.max_fit_q_star_abs_diff,
            max_calibration_ratio_center_abs_diff: report
                .aggregate
                .max_calibration_ratio_center_abs_diff,
            n_toy_generation_method_mismatch_scenarios: toy_generation_method_mismatch_scenarios
                .len(),
            toy_generation_method_mismatch_scenarios,
        },
        scenarios,
    })
}

pub fn render_measurement_combination_calibration_campaign_solver_parity_digest_markdown(
    digest: &MeasurementCombinationCalibrationCampaignSolverParityDigest,
) -> Result<String> {
    if digest.schema_version
        != MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_SOLVER_PARITY_DIGEST_SCHEMA_V0
    {
        return Err(Error::Validation(format!(
            "unsupported calibration-campaign solver parity digest schema_version '{}'",
            digest.schema_version
        )));
    }

    let mut out = String::new();
    out.push_str("# Measurement Combination Calibration Campaign Solver Parity Digest\n\n");
    out.push_str(&format!("- POI: `{}`\n", digest.poi));
    out.push_str(&format!("- Solver pair: `{}` vs `{}`\n", digest.lhs_solver, digest.rhs_solver));
    out.push_str(&format!("- Dominant fit gap scenario: `{}`\n", digest.dominant_fit_gap_scenario));
    out.push_str(&format!(
        "- Dominant calibration gap scenario: `{}`\n",
        digest.dominant_calibration_gap_scenario
    ));
    out.push_str(&format!(
        "- Toy-method mismatches: `{}`\n\n",
        fmt_digest_label_list(&digest.aggregate.toy_generation_method_mismatch_scenarios)
    ));
    out.push_str(
        "| Scenario | Label | Assignments | Fit rank | Calibration rank | Fit mu abs diff | Fit sigma rel diff | Fit q* abs diff | Cal ratio center abs diff | Sigma* frac abs diff | Bartlett improve frac abs diff | Same toy method |\n",
    );
    out.push_str(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n",
    );
    for scenario in &digest.scenarios {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            scenario.name,
            scenario.label,
            scenario.n_assignments,
            scenario.fit_gap_rank,
            scenario.calibration_gap_rank,
            fmt_digest_number(scenario.fit_mu_abs_diff),
            fmt_digest_number(scenario.fit_sigma_rel_diff),
            scenario.fit_q_star_abs_diff.map(fmt_digest_number).unwrap_or_else(|| "-".to_string()),
            fmt_digest_number(scenario.calibration_ratio_center_abs_diff),
            fmt_digest_number(scenario.sigma_star_ge_sigma_fraction_abs_diff),
            fmt_digest_number(scenario.bartlett_improves_mean_q_fraction_abs_diff),
            if scenario.same_toy_generation_method { "yes" } else { "no" }
        ));
    }
    Ok(out)
}

pub fn summarize_measurement_combination_calibration_campaign(
    report: &MeasurementCombinationCalibrationCampaignReport,
) -> Result<MeasurementCombinationCalibrationCampaignDigest> {
    if report.schema_version != MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_SCHEMA_V0 {
        return Err(Error::Validation(format!(
            "unsupported calibration campaign schema_version '{}'",
            report.schema_version
        )));
    }
    if report.scenarios.is_empty() {
        return Err(Error::Validation(
            "calibration campaign summary requires at least one scenario".to_string(),
        ));
    }

    let fit_ranks = descending_ranks(
        &report.scenarios,
        |scenario| &scenario.name,
        |scenario| scenario.comparison.fit_sigma_ratio_to_baseline,
    );
    let calibration_ranks = descending_ranks(
        &report.scenarios,
        |scenario| &scenario.name,
        |scenario| scenario.comparison.calibration_max_mean_sigma_star_to_sigma_ratio,
    );

    let mut near_neutral = Vec::new();
    let mut scenarios = Vec::with_capacity(report.scenarios.len());
    for scenario in &report.scenarios {
        let center = 0.5
            * (scenario.comparison.calibration_min_mean_sigma_star_to_sigma_ratio
                + scenario.comparison.calibration_max_mean_sigma_star_to_sigma_ratio);
        let span = scenario.comparison.calibration_max_mean_sigma_star_to_sigma_ratio
            - scenario.comparison.calibration_min_mean_sigma_star_to_sigma_ratio;
        let fit_rank = *fit_ranks.get(&scenario.name).unwrap_or(&0);
        let calibration_rank = *calibration_ranks.get(&scenario.name).unwrap_or(&0);
        let label = classify_calibration_campaign_scenario(
            &scenario.name,
            fit_rank,
            calibration_rank,
            center,
        );
        if (center - 1.0).abs() <= 1e-3 {
            near_neutral.push(scenario.name.clone());
        }

        scenarios.push(MeasurementCombinationCalibrationCampaignDigestScenario {
            name: scenario.name.clone(),
            n_assignments: scenario.assignments.len(),
            fit_rank_by_sigma: fit_rank,
            calibration_rank_by_sigma_ratio: calibration_rank,
            fit_sigma_ratio_to_baseline: scenario.comparison.fit_sigma_ratio_to_baseline,
            fit_sigma_delta_from_baseline: scenario.comparison.fit_sigma_ratio_to_baseline - 1.0,
            fit_interval_width_ratio_to_baseline: scenario
                .comparison
                .fit_interval_width_ratio_to_baseline,
            calibration_mean_sigma_star_to_sigma_ratio_center: center,
            calibration_mean_sigma_star_to_sigma_ratio_span: span,
            calibration_min_sigma_star_ge_sigma_fraction: scenario
                .comparison
                .calibration_min_sigma_star_ge_sigma_fraction,
            bartlett_improves_mean_q_fraction: scenario
                .comparison
                .bartlett_improves_mean_q_fraction,
            supported_systematics: scenario.fit.diagnostics.bartlett.supported_systematics.clone(),
            label,
        });
    }

    Ok(MeasurementCombinationCalibrationCampaignDigest {
        schema_version: MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_SUMMARY_SCHEMA_V0.to_string(),
        source_schema_version: report.schema_version.clone(),
        poi: report.poi.clone(),
        ci_level: report.ci_level,
        stability: report.stability.clone(),
        baseline_mu_hat: report.baseline.mu_hat,
        baseline_sigma: report.baseline.confidence_interval.sigma,
        dominant_fit_scenario: report.aggregate.widest_fit_interval_scenario.clone(),
        dominant_calibration_scenario: report
            .aggregate
            .highest_calibration_sigma_ratio_scenario
            .clone(),
        aggregate: MeasurementCombinationCalibrationCampaignDigestSummary {
            n_scenarios: report.aggregate.n_scenarios,
            n_calibration_neutral_scenarios: near_neutral.len(),
            max_fit_sigma_ratio_to_baseline: report.aggregate.max_fit_sigma_ratio_to_baseline,
            max_calibration_mean_sigma_star_to_sigma_ratio: report
                .aggregate
                .max_calibration_mean_sigma_star_to_sigma_ratio,
            near_neutral_calibration_scenarios: near_neutral,
        },
        scenarios,
    })
}

pub fn render_measurement_combination_calibration_campaign_digest_markdown(
    digest: &MeasurementCombinationCalibrationCampaignDigest,
) -> Result<String> {
    if digest.schema_version != MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_SUMMARY_SCHEMA_V0 {
        return Err(Error::Validation(format!(
            "unsupported calibration campaign summary schema_version '{}'",
            digest.schema_version
        )));
    }

    let mut out = String::new();
    out.push_str("# Measurement Combination Calibration Campaign Digest\n\n");
    out.push_str(&format!("- POI: `{}`\n", digest.poi));
    out.push_str(&format!(
        "- Baseline: `mu_hat = {}` and `sigma = {}`\n",
        fmt_digest_number(digest.baseline_mu_hat),
        fmt_digest_number(digest.baseline_sigma)
    ));
    out.push_str(&format!("- Dominant fit scenario: `{}`\n", digest.dominant_fit_scenario));
    out.push_str(&format!(
        "- Dominant calibration scenario: `{}`\n",
        digest.dominant_calibration_scenario
    ));
    if digest.aggregate.near_neutral_calibration_scenarios.is_empty() {
        out.push_str("- Near-neutral calibration scenarios: none\n\n");
    } else {
        out.push_str(&format!(
            "- Near-neutral calibration scenarios: `{}`\n\n",
            digest.aggregate.near_neutral_calibration_scenarios.join("`, `")
        ));
    }

    out.push_str(
        "| Scenario | Label | Fit Rank | Cal Rank | Fit sigma ratio | Fit delta | Cal center | Cal span | Min sigma*>=sigma frac | Bartlett improve frac | Supported systs |\n",
    );
    out.push_str("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n");
    for scenario in &digest.scenarios {
        let supported = if scenario.supported_systematics.is_empty() {
            "-".to_string()
        } else {
            scenario.supported_systematics.join(", ")
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            scenario.name,
            scenario.label,
            scenario.fit_rank_by_sigma,
            scenario.calibration_rank_by_sigma_ratio,
            fmt_digest_number(scenario.fit_sigma_ratio_to_baseline),
            fmt_digest_number(scenario.fit_sigma_delta_from_baseline),
            fmt_digest_number(scenario.calibration_mean_sigma_star_to_sigma_ratio_center),
            fmt_digest_number(scenario.calibration_mean_sigma_star_to_sigma_ratio_span),
            fmt_digest_number(scenario.calibration_min_sigma_star_ge_sigma_fraction),
            fmt_digest_number(scenario.bartlett_improves_mean_q_fraction),
            supported
        ));
    }
    Ok(out)
}

pub fn build_measurement_combination_calibration_campaign_brief(
    digests: &[(String, MeasurementCombinationCalibrationCampaignDigest)],
) -> Result<MeasurementCombinationCalibrationCampaignBrief> {
    if digests.is_empty() {
        return Err(Error::Validation(
            "calibration campaign brief requires at least one digest".to_string(),
        ));
    }

    let mut pois = BTreeSet::new();
    for (label, digest) in digests {
        if digest.schema_version != MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_SUMMARY_SCHEMA_V0 {
            return Err(Error::Validation(format!(
                "unsupported calibration campaign summary schema_version '{}' for '{}'",
                digest.schema_version, label
            )));
        }
        pois.insert(digest.poi.clone());
    }

    let mut entries = Vec::with_capacity(digests.len());
    let mut labels = Vec::with_capacity(digests.len());
    let mut labels_with_neutral = Vec::new();

    for (label, digest) in digests {
        labels.push(label.clone());
        if digest.aggregate.n_calibration_neutral_scenarios > 0 {
            labels_with_neutral.push(label.clone());
        }
        entries.push(MeasurementCombinationCalibrationCampaignBriefEntry {
            label: label.clone(),
            poi: digest.poi.clone(),
            dominant_fit_scenario: digest.dominant_fit_scenario.clone(),
            dominant_calibration_scenario: digest.dominant_calibration_scenario.clone(),
            max_fit_sigma_ratio_to_baseline: digest.aggregate.max_fit_sigma_ratio_to_baseline,
            max_calibration_mean_sigma_star_to_sigma_ratio: digest
                .aggregate
                .max_calibration_mean_sigma_star_to_sigma_ratio,
            n_near_neutral_calibration_scenarios: digest.aggregate.n_calibration_neutral_scenarios,
            near_neutral_calibration_scenarios: digest
                .aggregate
                .near_neutral_calibration_scenarios
                .clone(),
        });
    }

    let mut fit_scores: Vec<(String, f64)> = entries
        .iter()
        .map(|entry| (entry.label.clone(), entry.max_fit_sigma_ratio_to_baseline))
        .collect();
    sort_scored_labels_desc(&mut fit_scores);
    let highest_fit_label =
        top_scored_label(&fit_scores).expect("brief entries should contain a fit leader");

    let mut cal_scores: Vec<(String, f64)> = entries
        .iter()
        .map(|entry| (entry.label.clone(), entry.max_calibration_mean_sigma_star_to_sigma_ratio))
        .collect();
    sort_scored_labels_desc(&mut cal_scores);
    let highest_cal_label =
        top_scored_label(&cal_scores).expect("brief entries should contain a calibration leader");

    Ok(MeasurementCombinationCalibrationCampaignBrief {
        schema_version: MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_BRIEF_SCHEMA_V0.to_string(),
        source_schema_version: MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_SUMMARY_SCHEMA_V0
            .to_string(),
        stability: GVM_STABILITY_RESEARCH_GRADE.to_string(),
        shared_poi: if pois.len() == 1 { pois.iter().next().cloned() } else { None },
        pois: pois.into_iter().collect(),
        entries,
        aggregate: MeasurementCombinationCalibrationCampaignBriefSummary {
            n_artifacts: digests.len(),
            labels,
            highest_fit_inflation_label: highest_fit_label,
            highest_calibration_inflation_label: highest_cal_label,
            labels_with_near_neutral_calibration: labels_with_neutral,
        },
    })
}

pub fn render_measurement_combination_calibration_campaign_brief_markdown(
    brief: &MeasurementCombinationCalibrationCampaignBrief,
) -> Result<String> {
    if brief.schema_version != MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_BRIEF_SCHEMA_V0 {
        return Err(Error::Validation(format!(
            "unsupported calibration campaign brief schema_version '{}'",
            brief.schema_version
        )));
    }

    let mut out = String::new();
    out.push_str("# Measurement Combination Calibration Campaign Brief\n\n");
    if let Some(poi) = &brief.shared_poi {
        out.push_str(&format!("- POI: `{}`\n", poi));
    } else {
        out.push_str(&format!("- POIs: `{}`\n", brief.pois.join("`, `")));
    }
    out.push_str(&format!(
        "- Highest fit inflation artifact: `{}`\n",
        brief.aggregate.highest_fit_inflation_label
    ));
    out.push_str(&format!(
        "- Highest calibration inflation artifact: `{}`\n",
        brief.aggregate.highest_calibration_inflation_label
    ));
    if brief.aggregate.labels_with_near_neutral_calibration.is_empty() {
        out.push_str("- Artifacts with near-neutral calibration: none\n\n");
    } else {
        out.push_str(&format!(
            "- Artifacts with near-neutral calibration: `{}`\n\n",
            brief.aggregate.labels_with_near_neutral_calibration.join("`, `")
        ));
    }
    out.push_str(
        "| Label | POI | Dominant fit | Dominant calibration | Max fit sigma ratio | Max calibration ratio | Near-neutral count | Near-neutral scenarios |\n",
    );
    out.push_str("| --- | --- | --- | --- | ---: | ---: | ---: | --- |\n");
    for entry in &brief.entries {
        let neutral = if entry.near_neutral_calibration_scenarios.is_empty() {
            "-".to_string()
        } else {
            entry.near_neutral_calibration_scenarios.join(", ")
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
            entry.label,
            entry.poi,
            entry.dominant_fit_scenario,
            entry.dominant_calibration_scenario,
            fmt_digest_number(entry.max_fit_sigma_ratio_to_baseline),
            fmt_digest_number(entry.max_calibration_mean_sigma_star_to_sigma_ratio),
            entry.n_near_neutral_calibration_scenarios,
            neutral
        ));
    }
    Ok(out)
}

pub fn build_measurement_combination_calibration_campaign_family_report(
    briefs: &[(String, MeasurementCombinationCalibrationCampaignBrief)],
) -> Result<MeasurementCombinationCalibrationCampaignFamilyReport> {
    if briefs.is_empty() {
        return Err(Error::Validation(
            "calibration campaign family report requires at least one brief".to_string(),
        ));
    }

    let mut pois = BTreeSet::new();
    let mut families = Vec::with_capacity(briefs.len());
    let mut family_labels = Vec::with_capacity(briefs.len());
    let mut families_with_mixed_pois = Vec::new();
    let mut families_with_neutral = Vec::new();
    let mut total_artifacts = 0usize;

    for (label, brief) in briefs {
        if brief.schema_version != MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_BRIEF_SCHEMA_V0 {
            return Err(Error::Validation(format!(
                "unsupported calibration campaign brief schema_version '{}' for '{}'",
                brief.schema_version, label
            )));
        }
        for poi in &brief.pois {
            pois.insert(poi.clone());
        }
        family_labels.push(label.clone());
        total_artifacts += brief.aggregate.n_artifacts;

        if brief.shared_poi.is_none() && brief.pois.len() > 1 {
            families_with_mixed_pois.push(label.clone());
        }
        if !brief.aggregate.labels_with_near_neutral_calibration.is_empty() {
            families_with_neutral.push(label.clone());
        }
        let fit_value = brief
            .entries
            .iter()
            .map(|entry| entry.max_fit_sigma_ratio_to_baseline)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut fit_artifact_scores: Vec<(String, f64)> = brief
            .entries
            .iter()
            .map(|entry| (entry.label.clone(), entry.max_fit_sigma_ratio_to_baseline))
            .collect();
        sort_scored_labels_desc(&mut fit_artifact_scores);
        let fit_artifact = top_scored_label(&fit_artifact_scores)
            .unwrap_or_else(|| brief.aggregate.highest_fit_inflation_label.clone());
        let cal_value = brief
            .entries
            .iter()
            .map(|entry| entry.max_calibration_mean_sigma_star_to_sigma_ratio)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut cal_artifact_scores: Vec<(String, f64)> = brief
            .entries
            .iter()
            .map(|entry| {
                (entry.label.clone(), entry.max_calibration_mean_sigma_star_to_sigma_ratio)
            })
            .collect();
        sort_scored_labels_desc(&mut cal_artifact_scores);
        let cal_artifact = top_scored_label(&cal_artifact_scores)
            .unwrap_or_else(|| brief.aggregate.highest_calibration_inflation_label.clone());

        families.push(MeasurementCombinationCalibrationCampaignFamilyReportEntry {
            label: label.clone(),
            shared_poi: brief.shared_poi.clone(),
            pois: brief.pois.clone(),
            n_artifacts: brief.aggregate.n_artifacts,
            artifact_labels: brief.aggregate.labels.clone(),
            highest_fit_inflation_artifact: fit_artifact,
            highest_fit_inflation_value: fit_value,
            highest_calibration_inflation_artifact: cal_artifact,
            highest_calibration_inflation_value: cal_value,
            labels_with_near_neutral_calibration: brief
                .aggregate
                .labels_with_near_neutral_calibration
                .clone(),
            has_mixed_pois: brief.shared_poi.is_none() && brief.pois.len() > 1,
        });
    }

    let mut fit_family_scores: Vec<(String, f64)> = families
        .iter()
        .map(|family| (family.label.clone(), family.highest_fit_inflation_value))
        .collect();
    sort_scored_labels_desc(&mut fit_family_scores);
    let highest_fit_family =
        top_scored_label(&fit_family_scores).expect("family reports should contain a fit leader");

    let mut cal_family_scores: Vec<(String, f64)> = families
        .iter()
        .map(|family| (family.label.clone(), family.highest_calibration_inflation_value))
        .collect();
    sort_scored_labels_desc(&mut cal_family_scores);
    let highest_cal_family = top_scored_label(&cal_family_scores)
        .expect("family reports should contain a calibration leader");

    Ok(MeasurementCombinationCalibrationCampaignFamilyReport {
        schema_version: MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_FAMILY_REPORT_SCHEMA_V0
            .to_string(),
        source_schema_version: MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_BRIEF_SCHEMA_V0
            .to_string(),
        stability: GVM_STABILITY_RESEARCH_GRADE.to_string(),
        shared_poi: if pois.len() == 1 { pois.iter().next().cloned() } else { None },
        pois: pois.into_iter().collect(),
        families,
        aggregate: MeasurementCombinationCalibrationCampaignFamilyReportSummary {
            n_families: briefs.len(),
            n_total_artifacts: total_artifacts,
            family_labels,
            family_with_highest_fit_inflation: highest_fit_family,
            family_with_highest_calibration_inflation: highest_cal_family,
            families_with_mixed_pois,
            families_with_near_neutral_calibration: families_with_neutral,
        },
    })
}

pub fn render_measurement_combination_calibration_campaign_family_report_markdown(
    report: &MeasurementCombinationCalibrationCampaignFamilyReport,
) -> Result<String> {
    if report.schema_version != MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_FAMILY_REPORT_SCHEMA_V0
    {
        return Err(Error::Validation(format!(
            "unsupported calibration campaign family report schema_version '{}'",
            report.schema_version
        )));
    }

    let mut out = String::new();
    out.push_str("# Measurement Combination Calibration Campaign Family Report\n\n");
    if let Some(poi) = &report.shared_poi {
        out.push_str(&format!("- POI: `{}`\n", poi));
    } else {
        out.push_str(&format!("- POIs: `{}`\n", report.pois.join("`, `")));
    }
    out.push_str(&format!(
        "- Family with highest fit inflation: `{}`\n",
        report.aggregate.family_with_highest_fit_inflation
    ));
    out.push_str(&format!(
        "- Family with highest calibration inflation: `{}`\n",
        report.aggregate.family_with_highest_calibration_inflation
    ));
    if report.aggregate.families_with_mixed_pois.is_empty() {
        out.push_str("- Families with mixed POIs: none\n");
    } else {
        out.push_str(&format!(
            "- Families with mixed POIs: `{}`\n",
            report.aggregate.families_with_mixed_pois.join("`, `")
        ));
    }
    if report.aggregate.families_with_near_neutral_calibration.is_empty() {
        out.push_str("- Families with near-neutral calibration: none\n\n");
    } else {
        out.push_str(&format!(
            "- Families with near-neutral calibration: `{}`\n\n",
            report.aggregate.families_with_near_neutral_calibration.join("`, `")
        ));
    }
    out.push_str(
        "| Family | POIs | Artifacts | Highest fit artifact | Max fit sigma ratio | Highest calibration artifact | Max calibration ratio | Mixed POIs | Near-neutral artifacts |\n",
    );
    out.push_str("| --- | --- | --- | --- | ---: | --- | ---: | --- | --- |\n");
    for family in &report.families {
        let pois =
            if let Some(poi) = &family.shared_poi { poi.clone() } else { family.pois.join(", ") };
        let neutral = if family.labels_with_near_neutral_calibration.is_empty() {
            "-".to_string()
        } else {
            family.labels_with_near_neutral_calibration.join(", ")
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            family.label,
            pois,
            family.artifact_labels.join(", "),
            family.highest_fit_inflation_artifact,
            fmt_digest_number(family.highest_fit_inflation_value),
            family.highest_calibration_inflation_artifact,
            fmt_digest_number(family.highest_calibration_inflation_value),
            if family.has_mixed_pois { "yes" } else { "no" },
            neutral
        ));
    }
    Ok(out)
}

pub fn build_measurement_combination_calibration_campaign_family_matrix(
    report: &MeasurementCombinationCalibrationCampaignFamilyReport,
) -> Result<MeasurementCombinationCalibrationCampaignFamilyMatrix> {
    if report.schema_version != MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_FAMILY_REPORT_SCHEMA_V0
    {
        return Err(Error::Validation(format!(
            "unsupported calibration campaign family report schema_version '{}'",
            report.schema_version
        )));
    }

    let mut fit_scores: Vec<(String, f64)> = report
        .families
        .iter()
        .map(|family| (family.label.clone(), family.highest_fit_inflation_value))
        .collect();
    sort_scored_labels_desc(&mut fit_scores);

    let mut cal_scores: Vec<(String, f64)> = report
        .families
        .iter()
        .map(|family| (family.label.clone(), family.highest_calibration_inflation_value))
        .collect();
    sort_scored_labels_desc(&mut cal_scores);

    let mut joint_scores: Vec<(String, f64)> = report
        .families
        .iter()
        .map(|family| {
            (
                family.label.clone(),
                family.highest_fit_inflation_value * family.highest_calibration_inflation_value,
            )
        })
        .collect();
    sort_scored_labels_desc(&mut joint_scores);

    let fit_ranks: BTreeMap<String, usize> =
        fit_scores.iter().enumerate().map(|(idx, (label, _))| (label.clone(), idx + 1)).collect();
    let cal_ranks: BTreeMap<String, usize> =
        cal_scores.iter().enumerate().map(|(idx, (label, _))| (label.clone(), idx + 1)).collect();
    let joint_ranks: BTreeMap<String, usize> =
        joint_scores.iter().enumerate().map(|(idx, (label, _))| (label.clone(), idx + 1)).collect();

    let families: Vec<MeasurementCombinationCalibrationCampaignFamilyMatrixEntry> = report
        .families
        .iter()
        .map(|family| MeasurementCombinationCalibrationCampaignFamilyMatrixEntry {
            label: family.label.clone(),
            highest_fit_inflation_value: family.highest_fit_inflation_value,
            highest_calibration_inflation_value: family.highest_calibration_inflation_value,
            joint_severity_score: family.highest_fit_inflation_value
                * family.highest_calibration_inflation_value,
            fit_rank: *fit_ranks.get(&family.label).expect("family should have fit rank"),
            calibration_rank: *cal_ranks
                .get(&family.label)
                .expect("family should have calibration rank"),
            joint_rank: *joint_ranks.get(&family.label).expect("family should have joint rank"),
            n_artifacts: family.n_artifacts,
            has_mixed_pois: family.has_mixed_pois,
        })
        .collect();

    let mut pairwise = Vec::new();
    for (idx, lhs) in report.families.iter().enumerate() {
        for rhs in report.families.iter().skip(idx + 1) {
            let lhs_joint =
                lhs.highest_fit_inflation_value * lhs.highest_calibration_inflation_value;
            let rhs_joint =
                rhs.highest_fit_inflation_value * rhs.highest_calibration_inflation_value;
            pairwise.push(MeasurementCombinationCalibrationCampaignFamilyPairwiseRelation {
                lhs: lhs.label.clone(),
                rhs: rhs.label.clone(),
                lhs_fit_minus_rhs: lhs.highest_fit_inflation_value
                    - rhs.highest_fit_inflation_value,
                lhs_calibration_minus_rhs: lhs.highest_calibration_inflation_value
                    - rhs.highest_calibration_inflation_value,
                lhs_joint_minus_rhs: lhs_joint - rhs_joint,
                fit_dominance: dominance_label(
                    lhs.highest_fit_inflation_value,
                    rhs.highest_fit_inflation_value,
                ),
                calibration_dominance: dominance_label(
                    lhs.highest_calibration_inflation_value,
                    rhs.highest_calibration_inflation_value,
                ),
                joint_dominance: dominance_label(lhs_joint, rhs_joint),
                same_poi_coverage: lhs.pois == rhs.pois,
            });
        }
    }

    Ok(MeasurementCombinationCalibrationCampaignFamilyMatrix {
        schema_version: MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_FAMILY_MATRIX_SCHEMA_V0
            .to_string(),
        source_schema_version: MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_FAMILY_REPORT_SCHEMA_V0
            .to_string(),
        stability: GVM_STABILITY_RESEARCH_GRADE.to_string(),
        shared_poi: report.shared_poi.clone(),
        pois: report.pois.clone(),
        families,
        pairwise,
        aggregate: MeasurementCombinationCalibrationCampaignFamilyMatrixSummary {
            n_families: report.aggregate.n_families,
            fit_order: fit_scores.into_iter().map(|(label, _)| label).collect(),
            calibration_order: cal_scores.into_iter().map(|(label, _)| label).collect(),
            joint_order: joint_scores.into_iter().map(|(label, _)| label).collect(),
            family_with_highest_joint_severity: joint_ranks
                .iter()
                .find(|(_, rank)| **rank == 1)
                .map(|(label, _)| label.clone())
                .unwrap_or_default(),
            families_with_mixed_pois: report.aggregate.families_with_mixed_pois.clone(),
        },
    })
}

pub fn render_measurement_combination_calibration_campaign_family_matrix_markdown(
    matrix: &MeasurementCombinationCalibrationCampaignFamilyMatrix,
) -> Result<String> {
    if matrix.schema_version != MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_FAMILY_MATRIX_SCHEMA_V0
    {
        return Err(Error::Validation(format!(
            "unsupported calibration campaign family matrix schema_version '{}'",
            matrix.schema_version
        )));
    }

    let mut out = String::new();
    out.push_str("# Measurement Combination Calibration Campaign Family Matrix\n\n");
    if let Some(poi) = &matrix.shared_poi {
        out.push_str(&format!("- POI: `{}`\n", poi));
    } else {
        out.push_str(&format!("- POIs: `{}`\n", matrix.pois.join("`, `")));
    }
    out.push_str(&format!("- Fit order: `{}`\n", matrix.aggregate.fit_order.join("`, `")));
    out.push_str(&format!(
        "- Calibration order: `{}`\n",
        matrix.aggregate.calibration_order.join("`, `")
    ));
    out.push_str(&format!(
        "- Joint severity leader: `{}`\n\n",
        matrix.aggregate.family_with_highest_joint_severity
    ));
    out.push_str(
        "| Family | Fit rank | Calibration rank | Joint rank | Fit sigma ratio | Calibration ratio | Joint severity | Artifacts | Mixed POIs |\n",
    );
    out.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n");
    for family in &matrix.families {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            family.label,
            family.fit_rank,
            family.calibration_rank,
            family.joint_rank,
            fmt_digest_number(family.highest_fit_inflation_value),
            fmt_digest_number(family.highest_calibration_inflation_value),
            fmt_digest_number(family.joint_severity_score),
            family.n_artifacts,
            if family.has_mixed_pois { "yes" } else { "no" }
        ));
    }
    if !matrix.pairwise.is_empty() {
        out.push_str("\n| LHS | RHS | Fit dominance | Calibration dominance | Joint dominance | LHS fit minus RHS | LHS calibration minus RHS | LHS joint minus RHS | Same POI coverage |\n");
        out.push_str("| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |\n");
        for relation in &matrix.pairwise {
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                relation.lhs,
                relation.rhs,
                relation.fit_dominance,
                relation.calibration_dominance,
                relation.joint_dominance,
                fmt_digest_number(relation.lhs_fit_minus_rhs),
                fmt_digest_number(relation.lhs_calibration_minus_rhs),
                fmt_digest_number(relation.lhs_joint_minus_rhs),
                if relation.same_poi_coverage { "yes" } else { "no" }
            ));
        }
    }
    Ok(out)
}

pub fn build_measurement_combination_calibration_campaign_portfolio_report(
    matrices: &[(String, MeasurementCombinationCalibrationCampaignFamilyMatrix)],
) -> Result<MeasurementCombinationCalibrationCampaignPortfolioReport> {
    if matrices.is_empty() {
        return Err(Error::Validation(
            "calibration campaign portfolio report requires at least one family matrix".to_string(),
        ));
    }

    let mut pois = BTreeSet::new();
    let mut entries = Vec::with_capacity(matrices.len());
    let mut portfolio_labels = Vec::with_capacity(matrices.len());
    let mut mixed_poi_portfolios = Vec::new();

    for (label, matrix) in matrices {
        if matrix.schema_version
            != MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_FAMILY_MATRIX_SCHEMA_V0
        {
            return Err(Error::Validation(format!(
                "unsupported calibration campaign family matrix schema_version '{}' for '{}'",
                matrix.schema_version, label
            )));
        }
        for poi in &matrix.pois {
            pois.insert(poi.clone());
        }
        portfolio_labels.push(label.clone());

        let max_fit = matrix
            .families
            .iter()
            .map(|family| family.highest_fit_inflation_value)
            .fold(f64::NEG_INFINITY, f64::max);
        let max_cal = matrix
            .families
            .iter()
            .map(|family| family.highest_calibration_inflation_value)
            .fold(f64::NEG_INFINITY, f64::max);
        let max_joint = matrix
            .families
            .iter()
            .map(|family| family.joint_severity_score)
            .fold(f64::NEG_INFINITY, f64::max);

        let has_mixed_pois = matrix.shared_poi.is_none() && matrix.pois.len() > 1;
        if has_mixed_pois {
            mixed_poi_portfolios.push(label.clone());
        }
        entries.push(MeasurementCombinationCalibrationCampaignPortfolioEntry {
            label: label.clone(),
            shared_poi: matrix.shared_poi.clone(),
            pois: matrix.pois.clone(),
            n_families: matrix.aggregate.n_families,
            family_leader: matrix.aggregate.family_with_highest_joint_severity.clone(),
            max_fit_inflation: max_fit,
            max_calibration_inflation: max_cal,
            max_joint_severity: max_joint,
            fit_order: matrix.aggregate.fit_order.clone(),
            calibration_order: matrix.aggregate.calibration_order.clone(),
            joint_order: matrix.aggregate.joint_order.clone(),
            has_mixed_pois,
        });
    }

    let mut fit_scores: Vec<(String, f64)> =
        entries.iter().map(|entry| (entry.label.clone(), entry.max_fit_inflation)).collect();
    sort_scored_labels_desc(&mut fit_scores);
    let highest_fit_label =
        top_scored_label(&fit_scores).expect("portfolio entries should contain a fit leader");

    let mut cal_scores: Vec<(String, f64)> = entries
        .iter()
        .map(|entry| (entry.label.clone(), entry.max_calibration_inflation))
        .collect();
    sort_scored_labels_desc(&mut cal_scores);
    let highest_cal_label = top_scored_label(&cal_scores)
        .expect("portfolio entries should contain a calibration leader");

    let mut joint_scores: Vec<(String, f64)> =
        entries.iter().map(|entry| (entry.label.clone(), entry.max_joint_severity)).collect();
    sort_scored_labels_desc(&mut joint_scores);
    let highest_joint_label =
        top_scored_label(&joint_scores).expect("portfolio entries should contain a joint leader");

    let mut pairwise = Vec::new();
    for (idx, lhs) in entries.iter().enumerate() {
        for rhs in entries.iter().skip(idx + 1) {
            pairwise.push(MeasurementCombinationCalibrationCampaignPortfolioPairwiseRelation {
                lhs: lhs.label.clone(),
                rhs: rhs.label.clone(),
                lhs_fit_minus_rhs: lhs.max_fit_inflation - rhs.max_fit_inflation,
                lhs_calibration_minus_rhs: lhs.max_calibration_inflation
                    - rhs.max_calibration_inflation,
                lhs_joint_minus_rhs: lhs.max_joint_severity - rhs.max_joint_severity,
                fit_dominance: dominance_label(lhs.max_fit_inflation, rhs.max_fit_inflation),
                calibration_dominance: dominance_label(
                    lhs.max_calibration_inflation,
                    rhs.max_calibration_inflation,
                ),
                joint_dominance: dominance_label(lhs.max_joint_severity, rhs.max_joint_severity),
                same_poi_coverage: lhs.pois == rhs.pois,
            });
        }
    }

    Ok(MeasurementCombinationCalibrationCampaignPortfolioReport {
        schema_version: MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_PORTFOLIO_SCHEMA_V0
            .to_string(),
        source_schema_version: MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_FAMILY_MATRIX_SCHEMA_V0
            .to_string(),
        stability: GVM_STABILITY_RESEARCH_GRADE.to_string(),
        shared_poi: if pois.len() == 1 { pois.iter().next().cloned() } else { None },
        pois: pois.into_iter().collect(),
        entries,
        pairwise,
        aggregate: MeasurementCombinationCalibrationCampaignPortfolioSummary {
            n_portfolios: matrices.len(),
            portfolio_labels,
            portfolio_with_highest_fit_inflation: highest_fit_label,
            portfolio_with_highest_calibration_inflation: highest_cal_label,
            portfolio_with_highest_joint_severity: highest_joint_label,
            portfolios_with_mixed_pois: mixed_poi_portfolios,
        },
    })
}

pub fn render_measurement_combination_calibration_campaign_portfolio_markdown(
    report: &MeasurementCombinationCalibrationCampaignPortfolioReport,
) -> Result<String> {
    if report.schema_version != MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_PORTFOLIO_SCHEMA_V0 {
        return Err(Error::Validation(format!(
            "unsupported calibration campaign portfolio schema_version '{}'",
            report.schema_version
        )));
    }

    let mut out = String::new();
    out.push_str("# Measurement Combination Calibration Campaign Portfolio Report\n\n");
    if let Some(poi) = &report.shared_poi {
        out.push_str(&format!("- POI: `{}`\n", poi));
    } else {
        out.push_str(&format!("- POIs: `{}`\n", report.pois.join("`, `")));
    }
    out.push_str(&format!(
        "- Portfolio with highest fit inflation: `{}`\n",
        report.aggregate.portfolio_with_highest_fit_inflation
    ));
    out.push_str(&format!(
        "- Portfolio with highest calibration inflation: `{}`\n",
        report.aggregate.portfolio_with_highest_calibration_inflation
    ));
    out.push_str(&format!(
        "- Portfolio with highest joint severity: `{}`\n\n",
        report.aggregate.portfolio_with_highest_joint_severity
    ));
    out.push_str(
        "| Portfolio | POIs | Families | Joint leader | Max fit inflation | Max calibration inflation | Max joint severity | Mixed POIs |\n",
    );
    out.push_str("| --- | --- | ---: | --- | ---: | ---: | ---: | --- |\n");
    for entry in &report.entries {
        let pois =
            if let Some(poi) = &entry.shared_poi { poi.clone() } else { entry.pois.join(", ") };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
            entry.label,
            pois,
            entry.n_families,
            entry.family_leader,
            fmt_digest_number(entry.max_fit_inflation),
            fmt_digest_number(entry.max_calibration_inflation),
            fmt_digest_number(entry.max_joint_severity),
            if entry.has_mixed_pois { "yes" } else { "no" }
        ));
    }
    if !report.pairwise.is_empty() {
        out.push_str("\n| LHS | RHS | Fit dominance | Calibration dominance | Joint dominance | LHS fit minus RHS | LHS calibration minus RHS | LHS joint minus RHS | Same POI coverage |\n");
        out.push_str("| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |\n");
        for relation in &report.pairwise {
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                relation.lhs,
                relation.rhs,
                relation.fit_dominance,
                relation.calibration_dominance,
                relation.joint_dominance,
                fmt_digest_number(relation.lhs_fit_minus_rhs),
                fmt_digest_number(relation.lhs_calibration_minus_rhs),
                fmt_digest_number(relation.lhs_joint_minus_rhs),
                if relation.same_poi_coverage { "yes" } else { "no" }
            ));
        }
    }
    Ok(out)
}

pub fn build_measurement_combination_calibration_campaign_portfolio_stability_report(
    reports: &[(String, MeasurementCombinationCalibrationCampaignPortfolioReport)],
) -> Result<MeasurementCombinationCalibrationCampaignPortfolioStabilityReport> {
    if reports.is_empty() {
        return Err(Error::Validation(
            "calibration campaign portfolio stability report requires at least one portfolio"
                .to_string(),
        ));
    }

    let mut pois = BTreeSet::new();
    let mut runs = Vec::with_capacity(reports.len());
    let mut pairwise = Vec::new();
    let mut run_labels = Vec::with_capacity(reports.len());
    let mut runs_with_mixed_pois = Vec::new();

    for (label, report) in reports {
        if report.schema_version != MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_PORTFOLIO_SCHEMA_V0
        {
            return Err(Error::Validation(format!(
                "unsupported calibration campaign portfolio schema_version '{}' for '{}'",
                report.schema_version, label
            )));
        }
        for poi in &report.pois {
            pois.insert(poi.clone());
        }
        run_labels.push(label.clone());
        let mut fit_order: Vec<(String, f64)> = report
            .entries
            .iter()
            .map(|entry| (entry.label.clone(), entry.max_fit_inflation))
            .collect();
        sort_scored_labels_desc(&mut fit_order);
        let mut calibration_order: Vec<(String, f64)> = report
            .entries
            .iter()
            .map(|entry| (entry.label.clone(), entry.max_calibration_inflation))
            .collect();
        sort_scored_labels_desc(&mut calibration_order);
        let mut joint_order: Vec<(String, f64)> = report
            .entries
            .iter()
            .map(|entry| (entry.label.clone(), entry.max_joint_severity))
            .collect();
        sort_scored_labels_desc(&mut joint_order);
        let has_mixed_pois = report.shared_poi.is_none() && report.pois.len() > 1;
        if has_mixed_pois {
            runs_with_mixed_pois.push(label.clone());
        }
        runs.push(MeasurementCombinationCalibrationCampaignPortfolioStabilityRun {
            label: label.clone(),
            shared_poi: report.shared_poi.clone(),
            pois: report.pois.clone(),
            n_portfolios: report.aggregate.n_portfolios,
            fit_leader: report.aggregate.portfolio_with_highest_fit_inflation.clone(),
            calibration_leader: report
                .aggregate
                .portfolio_with_highest_calibration_inflation
                .clone(),
            joint_leader: report.aggregate.portfolio_with_highest_joint_severity.clone(),
            fit_order: fit_order.into_iter().map(|(label, _)| label).collect::<Vec<_>>(),
            calibration_order: calibration_order
                .into_iter()
                .map(|(label, _)| label)
                .collect::<Vec<_>>(),
            joint_order: joint_order.into_iter().map(|(label, _)| label).collect::<Vec<_>>(),
            has_mixed_pois,
        });
    }

    for (idx, lhs) in runs.iter().enumerate() {
        for rhs in runs.iter().skip(idx + 1) {
            let lhs_labels: BTreeSet<String> = lhs.fit_order.iter().cloned().collect();
            let rhs_labels: BTreeSet<String> = rhs.fit_order.iter().cloned().collect();
            pairwise.push(
                MeasurementCombinationCalibrationCampaignPortfolioStabilityPairwiseRelation {
                    lhs: lhs.label.clone(),
                    rhs: rhs.label.clone(),
                    same_poi_coverage: lhs.pois == rhs.pois,
                    same_portfolio_labels: lhs_labels == rhs_labels,
                    same_fit_leader: lhs.fit_leader == rhs.fit_leader,
                    same_calibration_leader: lhs.calibration_leader == rhs.calibration_leader,
                    same_joint_leader: lhs.joint_leader == rhs.joint_leader,
                    same_fit_order: lhs.fit_order == rhs.fit_order,
                    same_calibration_order: lhs.calibration_order == rhs.calibration_order,
                    same_joint_order: lhs.joint_order == rhs.joint_order,
                },
            );
        }
    }

    let reference = runs.first().cloned().expect("at least one run should exist for stability");
    let stable_fit_leader = runs.iter().all(|run| run.fit_leader == reference.fit_leader);
    let stable_calibration_leader =
        runs.iter().all(|run| run.calibration_leader == reference.calibration_leader);
    let stable_joint_leader = runs.iter().all(|run| run.joint_leader == reference.joint_leader);
    let stable_fit_order = pairwise.iter().all(|rel| rel.same_fit_order);
    let stable_calibration_order = pairwise.iter().all(|rel| rel.same_calibration_order);
    let stable_joint_order = pairwise.iter().all(|rel| rel.same_joint_order);
    Ok(MeasurementCombinationCalibrationCampaignPortfolioStabilityReport {
        schema_version: MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_PORTFOLIO_STABILITY_SCHEMA_V0
            .to_string(),
        source_schema_version: MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_PORTFOLIO_SCHEMA_V0
            .to_string(),
        stability: GVM_STABILITY_RESEARCH_GRADE.to_string(),
        shared_poi: if pois.len() == 1 { pois.iter().next().cloned() } else { None },
        pois: pois.into_iter().collect(),
        runs,
        pairwise,
        aggregate: MeasurementCombinationCalibrationCampaignPortfolioStabilitySummary {
            n_runs: reports.len(),
            run_labels,
            reference_run: reference.label.clone(),
            stable_fit_leader,
            stable_calibration_leader,
            stable_joint_leader,
            stable_fit_order,
            stable_calibration_order,
            stable_joint_order,
            runs_with_mixed_pois,
        },
    })
}

pub fn render_measurement_combination_calibration_campaign_portfolio_stability_markdown(
    report: &MeasurementCombinationCalibrationCampaignPortfolioStabilityReport,
) -> Result<String> {
    if report.schema_version
        != MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_PORTFOLIO_STABILITY_SCHEMA_V0
    {
        return Err(Error::Validation(format!(
            "unsupported calibration campaign portfolio stability schema_version '{}'",
            report.schema_version
        )));
    }

    let mut out = String::new();
    out.push_str("# Measurement Combination Calibration Campaign Portfolio Stability\n\n");
    if let Some(poi) = &report.shared_poi {
        out.push_str(&format!("- POI: `{}`\n", poi));
    } else {
        out.push_str(&format!("- POIs: `{}`\n", report.pois.join("`, `")));
    }
    out.push_str(&format!(
        "- Stable fit leader: `{}`\n",
        if report.aggregate.stable_fit_leader { "yes" } else { "no" }
    ));
    out.push_str(&format!(
        "- Stable calibration leader: `{}`\n",
        if report.aggregate.stable_calibration_leader { "yes" } else { "no" }
    ));
    out.push_str(&format!(
        "- Stable joint order: `{}`\n\n",
        if report.aggregate.stable_joint_order { "yes" } else { "no" }
    ));
    out.push_str(
        "| Run | POIs | Portfolios | Fit leader | Calibration leader | Joint leader | Mixed POIs |\n",
    );
    out.push_str("| --- | --- | ---: | --- | --- | --- | --- |\n");
    for run in &report.runs {
        let pois = if let Some(poi) = &run.shared_poi { poi.clone() } else { run.pois.join(", ") };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} |\n",
            run.label,
            pois,
            run.n_portfolios,
            run.fit_leader,
            run.calibration_leader,
            run.joint_leader,
            if run.has_mixed_pois { "yes" } else { "no" }
        ));
    }
    if !report.pairwise.is_empty() {
        out.push_str("\n| LHS | RHS | Same POIs | Same labels | Same fit leader | Same calibration leader | Same joint leader | Same fit order | Same calibration order | Same joint order |\n");
        out.push_str("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
        for rel in &report.pairwise {
            out.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                rel.lhs,
                rel.rhs,
                if rel.same_poi_coverage { "yes" } else { "no" },
                if rel.same_portfolio_labels { "yes" } else { "no" },
                if rel.same_fit_leader { "yes" } else { "no" },
                if rel.same_calibration_leader { "yes" } else { "no" },
                if rel.same_joint_leader { "yes" } else { "no" },
                if rel.same_fit_order { "yes" } else { "no" },
                if rel.same_calibration_order { "yes" } else { "no" },
                if rel.same_joint_order { "yes" } else { "no" }
            ));
        }
    }
    Ok(out)
}

fn descending_ranks<T, N, F>(items: &[T], name: N, score: F) -> BTreeMap<String, usize>
where
    N: Fn(&T) -> &str,
    F: Fn(&T) -> f64,
{
    let mut scored: Vec<(String, f64)> =
        items.iter().map(|item| (name(item).to_string(), score(item))).collect();
    sort_scored_labels_desc(&mut scored);
    scored.into_iter().enumerate().map(|(idx, (name, _))| (name, idx + 1)).collect()
}

fn score_label_desc_cmp(
    label_a: &str,
    score_a: f64,
    label_b: &str,
    score_b: f64,
) -> std::cmp::Ordering {
    score_b.total_cmp(&score_a).then_with(|| label_a.cmp(label_b))
}

fn sort_scored_labels_desc(scored: &mut [(String, f64)]) {
    scored.sort_by(|(label_a, score_a), (label_b, score_b)| {
        score_label_desc_cmp(label_a, *score_a, label_b, *score_b)
    });
}

fn top_scored_label(scored: &[(String, f64)]) -> Option<String> {
    scored.first().map(|(label, _)| label.clone())
}

fn classify_calibration_campaign_scenario(
    name: &str,
    fit_rank: usize,
    calibration_rank: usize,
    calibration_center: f64,
) -> String {
    if fit_rank == 1 && calibration_rank == 1 {
        return "dominant-fit-and-calibration".to_string();
    }
    if fit_rank == 1 {
        return "dominant-fit".to_string();
    }
    if calibration_rank == 1 {
        return "dominant-calibration".to_string();
    }
    if (calibration_center - 1.0).abs() <= 1e-3 {
        return "calibration-neutral".to_string();
    }
    format!("secondary:{name}")
}

fn classify_scenario_solver_parity_entry(
    name: &str,
    same_supported_systematics: bool,
    both_perturbative_within_threshold: bool,
    dominant_mu_gap_scenario: &str,
    dominant_sigma_gap_scenario: &str,
    dominant_q_star_gap_scenario: &str,
) -> String {
    if !same_supported_systematics {
        return "supported-systematics-mismatch".to_string();
    }
    if !both_perturbative_within_threshold {
        return "perturbative-overlap-failure".to_string();
    }
    if name == dominant_mu_gap_scenario
        || name == dominant_sigma_gap_scenario
        || name == dominant_q_star_gap_scenario
    {
        return "dominant-gap".to_string();
    }
    "tight".to_string()
}

fn classify_calibration_solver_parity_entry(
    name: &str,
    same_toy_generation_method: bool,
    dominant_fit_gap_scenario: &str,
    dominant_calibration_gap_scenario: &str,
) -> String {
    if !same_toy_generation_method {
        return "toy-method-mismatch".to_string();
    }
    if name == dominant_fit_gap_scenario && name == dominant_calibration_gap_scenario {
        return "dominant-fit-and-calibration-gap".to_string();
    }
    if name == dominant_fit_gap_scenario {
        return "dominant-fit-gap".to_string();
    }
    if name == dominant_calibration_gap_scenario {
        return "dominant-calibration-gap".to_string();
    }
    "tight".to_string()
}

fn dominance_label(lhs: f64, rhs: f64) -> String {
    if (lhs - rhs).abs() <= 1e-12 {
        "tie".to_string()
    } else if lhs > rhs {
        "lhs_higher".to_string()
    } else {
        "rhs_higher".to_string()
    }
}

fn fmt_digest_number(value: f64) -> String {
    format!("{value:.6}")
}

fn fmt_digest_label_list(labels: &[String]) -> String {
    if labels.is_empty() { "none".to_string() } else { labels.join("`, `") }
}

fn validate_scenario_study_spec(
    spec: &MeasurementCombinationSpec,
    scenario_study: &MeasurementCombinationScenarioStudySpec,
) -> Result<()> {
    if scenario_study.schema_version != MEASUREMENT_COMBINATION_SCENARIO_CONFIG_SCHEMA_V0 {
        return Err(Error::Validation(format!(
            "unsupported scenario study schema_version '{}'",
            scenario_study.schema_version
        )));
    }
    if scenario_study.scenarios.is_empty() {
        return Err(Error::Validation("scenario study requires at least one scenario".to_string()));
    }
    let known_systematics: BTreeSet<&str> =
        spec.systematics.iter().map(|s| s.name.as_str()).collect();
    let mut seen_names = BTreeSet::new();
    for scenario in &scenario_study.scenarios {
        if scenario.name.trim().is_empty() {
            return Err(Error::Validation(
                "scenario study contains an empty scenario name".to_string(),
            ));
        }
        if !seen_names.insert(scenario.name.as_str()) {
            return Err(Error::Validation(format!("duplicate scenario name '{}'", scenario.name)));
        }
        let mut seen_systematics = BTreeSet::new();
        for assignment in &scenario.error_on_error {
            if assignment.value < 0.0 {
                return Err(Error::Validation(format!(
                    "scenario '{}' has negative error_on_error for systematic '{}'",
                    scenario.name, assignment.systematic
                )));
            }
            if !known_systematics.contains(assignment.systematic.as_str()) {
                return Err(Error::Validation(format!(
                    "scenario '{}' references unknown systematic '{}'",
                    scenario.name, assignment.systematic
                )));
            }
            if !seen_systematics.insert(assignment.systematic.as_str()) {
                return Err(Error::Validation(format!(
                    "scenario '{}' contains duplicate systematic assignment '{}'",
                    scenario.name, assignment.systematic
                )));
            }
        }
    }
    Ok(())
}

fn apply_scenario_to_spec(
    spec: &MeasurementCombinationSpec,
    scenario: &MeasurementCombinationScenarioSpec,
) -> Result<MeasurementCombinationSpec> {
    let mut out = spec.clone();
    for systematic in &mut out.systematics {
        systematic.error_on_error = 0.0;
    }
    for assignment in &scenario.error_on_error {
        let systematic =
            out.systematics.iter_mut().find(|s| s.name == assignment.systematic).ok_or_else(
                || {
                    Error::Validation(format!(
                        "scenario '{}' references unknown systematic '{}'",
                        scenario.name, assignment.systematic
                    ))
                },
            )?;
        systematic.error_on_error = assignment.value;
    }
    Ok(out)
}

fn solver_label(solver: MeasurementCombinationSolver) -> &'static str {
    match solver {
        MeasurementCombinationSolver::Numerical => "numerical",
        MeasurementCombinationSolver::NumericalPaper => "numerical-paper",
        MeasurementCombinationSolver::AnalyticPerturbative => "analytic-perturbative",
        MeasurementCombinationSolver::Auto => "auto",
    }
}

fn maybe_solver_dispatch(
    requested: MeasurementCombinationSolver,
    effective: MeasurementCombinationSolver,
) -> (Option<String>, Option<String>) {
    if requested == effective {
        (None, None)
    } else {
        (Some(solver_label(requested).to_string()), Some(solver_label(effective).to_string()))
    }
}

fn optional_abs_diff(lhs: Option<f64>, rhs: Option<f64>) -> Option<f64> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => Some((lhs - rhs).abs()),
        _ => None,
    }
}

fn calibration_fit_solver(solver: MeasurementCombinationSolver) -> MeasurementCombinationSolver {
    solver
}

fn calibration_toy_generation_solver(
    solver: MeasurementCombinationSolver,
) -> MeasurementCombinationSolver {
    match solver {
        MeasurementCombinationSolver::Numerical => MeasurementCombinationSolver::Numerical,
        MeasurementCombinationSolver::NumericalPaper
        | MeasurementCombinationSolver::AnalyticPerturbative
        | MeasurementCombinationSolver::Auto => MeasurementCombinationSolver::NumericalPaper,
    }
}

fn toy_generation_method_label(solver: MeasurementCombinationSolver) -> &'static str {
    match solver {
        MeasurementCombinationSolver::Numerical => {
            "measurement_side_gvm_unbiased_sigma2_star_normalized_spec"
        }
        MeasurementCombinationSolver::NumericalPaper => {
            "measurement_side_gvm_unbiased_sigma2_star_normalized_spec_original_theta"
        }
        MeasurementCombinationSolver::AnalyticPerturbative => {
            "measurement_side_gvm_unbiased_sigma2_star_normalized_spec_original_theta_paper_reference"
        }
        MeasurementCombinationSolver::Auto => {
            "measurement_side_gvm_unbiased_sigma2_star_normalized_spec_auto"
        }
    }
}

fn can_parallelize_measurement_combine_outer(work_items: usize) -> bool {
    work_items > 1 && rayon::current_thread_index().is_none() && rayon::current_num_threads() > 1
}

fn build_measurement_combination_scenario_result(
    spec: &MeasurementCombinationSpec,
    scenario: &MeasurementCombinationScenarioSpec,
    baseline: &MeasurementCombinationResult,
    baseline_width: f64,
    baseline_sigma: f64,
    ci_level: f64,
    solver: MeasurementCombinationSolver,
) -> Result<MeasurementCombinationScenarioResult> {
    let scenario_spec = apply_scenario_to_spec(spec, scenario)?;
    let result = combine_measurements_with_solver(&scenario_spec, ci_level, solver)?;
    let width = result.confidence_interval.upper - result.confidence_interval.lower;
    let sigma_ratio = result.confidence_interval.sigma / baseline_sigma;
    let width_ratio = width / baseline_width.max(1e-12);
    let mu_shift = result.mu_hat - baseline.mu_hat;
    let abs_mu_shift = mu_shift.abs();
    let chi2_delta = result.goodness_of_fit.chi2 - baseline.goodness_of_fit.chi2;
    let validity = &result.diagnostics.perturbative_validity;
    let max_condition = validity.condition_values.iter().copied().reduce(f64::max);
    let all_within = validity.within_threshold.iter().all(|flag| *flag);
    let supported_systematics = result.diagnostics.bartlett.supported_systematics.clone();

    Ok(MeasurementCombinationScenarioResult {
        name: scenario.name.clone(),
        assignments: scenario.error_on_error.clone(),
        result,
        comparison: MeasurementCombinationScenarioComparison {
            mu_shift_from_baseline: mu_shift,
            abs_mu_shift_from_baseline: abs_mu_shift,
            sigma_ratio_to_baseline: sigma_ratio,
            interval_width_ratio_to_baseline: width_ratio,
            chi2_delta_from_baseline: chi2_delta,
            max_perturbative_condition: max_condition,
            all_perturbative_within_threshold: all_within,
            supported_systematics,
        },
    })
}

fn build_measurement_combination_calibration_campaign_scenario(
    spec: &MeasurementCombinationSpec,
    scenario_spec: &MeasurementCombinationScenarioSpec,
    scenario_fit: &MeasurementCombinationScenarioResult,
    baseline_width: f64,
    baseline_sigma: f64,
    ci_level: f64,
    solver: MeasurementCombinationSolver,
    n_toys: usize,
    seeds: &[u64],
) -> Result<MeasurementCombinationCalibrationCampaignScenario> {
    let scenario_measurement_spec = apply_scenario_to_spec(spec, scenario_spec)?;
    let calibration = calibrate_measurements_toys_study_with_solver(
        &scenario_measurement_spec,
        ci_level,
        solver,
        n_toys,
        seeds,
    )?;
    let fit_width = scenario_fit.result.confidence_interval.upper
        - scenario_fit.result.confidence_interval.lower;
    let fit_sigma_ratio = scenario_fit.result.confidence_interval.sigma / baseline_sigma;
    let fit_width_ratio = fit_width / baseline_width.max(1e-12);
    let calibration_min_ratio = calibration.aggregate.min_mean_sigma_star_to_sigma_ratio;
    let calibration_max_ratio = calibration.aggregate.max_mean_sigma_star_to_sigma_ratio;

    Ok(MeasurementCombinationCalibrationCampaignScenario {
        name: scenario_spec.name.clone(),
        assignments: scenario_spec.error_on_error.clone(),
        fit: scenario_fit.result.clone(),
        calibration: calibration.aggregate.clone(),
        comparison: MeasurementCombinationCalibrationCampaignComparison {
            fit_sigma_ratio_to_baseline: fit_sigma_ratio,
            fit_interval_width_ratio_to_baseline: fit_width_ratio,
            calibration_min_mean_sigma_star_to_sigma_ratio: calibration_min_ratio,
            calibration_max_mean_sigma_star_to_sigma_ratio: calibration_max_ratio,
            calibration_max_abs_ratio_delta_from_reference: calibration
                .aggregate
                .max_abs_mean_sigma_star_to_sigma_ratio_delta_from_reference,
            calibration_min_sigma_star_ge_sigma_fraction: calibration
                .aggregate
                .min_sigma_star_ge_sigma_fraction,
            bartlett_improves_mean_q_fraction: calibration
                .aggregate
                .bartlett_improves_mean_q_fraction,
        },
    })
}

fn calibrate_measurements_toys_with_reference(
    spec: &MeasurementCombinationSpec,
    reference: &MeasurementCombinationResult,
    ci_level: f64,
    solver: MeasurementCombinationSolver,
    n_toys: usize,
    seed: u64,
) -> Result<MeasurementCombinationCalibrationReport> {
    let fit_solver = calibration_fit_solver(solver);
    let toy_solver = calibration_toy_generation_solver(solver);
    let toy_generation = build_measurement_toy_generation_context(spec, toy_solver)?;
    let can_par = can_parallelize_measurement_combine_outer(n_toys);
    let toy_results = if can_par {
        (0..n_toys)
            .into_par_iter()
            .map(|toy_idx| {
                run_measurement_calibration_toy(
                    &toy_generation,
                    ci_level,
                    fit_solver,
                    seed,
                    toy_idx,
                )
            })
            .collect::<Vec<_>>()
    } else {
        (0..n_toys)
            .map(|toy_idx| {
                run_measurement_calibration_toy(
                    &toy_generation,
                    ci_level,
                    fit_solver,
                    seed,
                    toy_idx,
                )
            })
            .collect::<Vec<_>>()
    };

    let mut q_values = Vec::with_capacity(n_toys);
    let mut q_star_values = Vec::with_capacity(n_toys);
    let mut sigma_values = Vec::with_capacity(n_toys);
    let mut sigma_star_values = Vec::with_capacity(n_toys);
    for toy in toy_results {
        let toy = toy?;
        q_values.push(toy.q);
        q_star_values.push(toy.q_star);
        sigma_values.push(toy.sigma);
        sigma_star_values.push(toy.sigma_star);
    }

    let df = reference.goodness_of_fit.df;
    let mean_q = mean(&q_values);
    let mean_q_star = mean(&q_star_values);
    let sd_q = sample_sd(&q_values, mean_q);
    let sd_q_star = sample_sd(&q_star_values, mean_q_star);
    let sem_q = sd_q / (n_toys as f64).sqrt();
    let sem_q_star = sd_q_star / (n_toys as f64).sqrt();
    let df_f = df as f64;
    let mean_q_abs_error_to_df = (mean_q - df_f).abs();
    let mean_q_star_abs_error_to_df = (mean_q_star - df_f).abs();
    let mean_sigma = mean(&sigma_values);
    let mean_sigma_star = mean(&sigma_star_values);
    let mean_sigma_star_to_sigma_ratio = mean_sigma_star / mean_sigma.max(1e-12);
    let sigma_star_ge_sigma_fraction = sigma_values
        .iter()
        .zip(&sigma_star_values)
        .filter(|(sigma, sigma_star)| **sigma_star >= **sigma)
        .count() as f64
        / n_toys as f64;

    Ok(MeasurementCombinationCalibrationReport {
        schema_version: MEASUREMENT_COMBINATION_CALIBRATION_SCHEMA_V0.to_string(),
        poi: spec.poi.clone(),
        ci_level,
        n_toys,
        seed,
        stability: GVM_STABILITY_STABLE.to_string(),
        reference: reference.clone(),
        summary: MeasurementCombinationCalibrationSummary {
            df,
            mean_q,
            mean_q_star,
            sd_q,
            sd_q_star,
            sem_q,
            sem_q_star,
            mean_q_abs_error_to_df,
            mean_q_star_abs_error_to_df,
            bartlett_improves_mean_q: mean_q_star_abs_error_to_df <= mean_q_abs_error_to_df,
            mean_sigma,
            mean_sigma_star,
            mean_sigma_star_to_sigma_ratio,
            sigma_star_ge_sigma_fraction,
            toy_generation_method: toy_generation_method_label(toy_solver).to_string(),
        },
    })
}

#[derive(Debug, Clone, Copy)]
struct MeasurementCalibrationToyResult {
    q: f64,
    q_star: f64,
    sigma: f64,
    sigma_star: f64,
}

#[derive(Clone)]
struct MeasurementToyGenerationContext {
    template_spec: MeasurementCombinationSpec,
    prepared_template: PreparedSpec,
    mean: DVector<f64>,
    base_factor: DMatrix<f64>,
    systematic_effect_factors: Vec<DMatrix<f64>>,
    paper_warm_start: Option<PaperWarmStartGuide>,
}

fn calibration_toy_result_from_full_result(
    out: &MeasurementCombinationResult,
) -> Result<MeasurementCalibrationToyResult> {
    let q_star = out.diagnostics.bartlett.q_star.ok_or_else(|| {
        Error::Computation("missing q_star in toy calibration output".to_string())
    })?;
    let sigma_star = out.diagnostics.bartlett.sigma_star.ok_or_else(|| {
        Error::Computation("missing sigma_star in toy calibration output".to_string())
    })?;
    Ok(MeasurementCalibrationToyResult {
        q: out.goodness_of_fit.chi2,
        q_star,
        sigma: out.confidence_interval.sigma,
        sigma_star,
    })
}

fn calibration_toy_result_from_fit(
    prep: &PreparedSpec,
    state: &ProfiledGvmState,
    sigma: f64,
    gof_stat: f64,
    df: usize,
) -> Result<MeasurementCalibrationToyResult> {
    let bartlett = compute_bartlett_diagnostics(prep, state, sigma, gof_stat, df);
    let q_star = bartlett.q_star.ok_or_else(|| {
        Error::Computation("missing q_star in toy calibration output".to_string())
    })?;
    let sigma_star = bartlett.sigma_star.ok_or_else(|| {
        Error::Computation("missing sigma_star in toy calibration output".to_string())
    })?;
    Ok(MeasurementCalibrationToyResult { q: gof_stat, q_star, sigma, sigma_star })
}

fn build_measurement_toy_generation_context(
    spec: &MeasurementCombinationSpec,
    solver: MeasurementCombinationSolver,
) -> Result<MeasurementToyGenerationContext> {
    let prep = PreparedSpec::from_spec(spec)?;
    let (mu_hat, state) = match solver {
        MeasurementCombinationSolver::Numerical => {
            let objective = MeasurementCombineObjective::new(&prep);
            let bounds = prep.bounds();
            let fit = numerical_gvm_fit(&prep, &objective, &bounds)?;
            (fit.fit.parameters[0], fit.state)
        }
        MeasurementCombinationSolver::NumericalPaper
        | MeasurementCombinationSolver::AnalyticPerturbative
        | MeasurementCombinationSolver::Auto => {
            let objective = PaperMeasurementCombineObjective::new(&prep);
            let bounds = objective.bounds();
            let fit = numerical_paper_gvm_fit(&prep, &objective, &bounds, None)?;
            (fit.fit.parameters[0], fit.state)
        }
    };
    let paper_warm_start = match solver {
        MeasurementCombinationSolver::Numerical => None,
        MeasurementCombinationSolver::NumericalPaper
        | MeasurementCombinationSolver::AnalyticPerturbative
        | MeasurementCombinationSolver::Auto => {
            paper_warm_start_from_profiled_state(&prep, mu_hat, &state)
        }
    };
    let workspace = build_bartlett_workspace(&prep, &state)?;
    let sigma2_unbiased = compute_unbiased_sigma2_estimates(&prep, &state, &workspace);
    let base_factor = covariance_sampling_factor(&prep.base_cov);
    let systematic_effect_factors = prep
        .systematics
        .iter()
        .zip(sigma2_unbiased.iter())
        .map(|(syst, sigma2)| {
            let factor = covariance_sampling_factor(&syst.corr_reg.scale(*sigma2));
            DMatrix::from_diagonal(&syst.magnitudes) * factor
        })
        .collect();

    Ok(MeasurementToyGenerationContext {
        template_spec: spec.clone(),
        prepared_template: prep.clone(),
        mean: prep.ones.clone() * mu_hat,
        base_factor,
        systematic_effect_factors,
        paper_warm_start,
    })
}

fn sample_gvm_toy_values_from_context(
    context: &MeasurementToyGenerationContext,
    seed: u64,
) -> DVector<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut y = context.mean.clone();
    y += sample_mvn_zero_with_factor(&mut rng, &context.base_factor);
    for factor in &context.systematic_effect_factors {
        y += sample_mvn_zero_with_factor(&mut rng, factor);
    }
    y
}

fn simulate_gvm_toy_spec_from_context(
    context: &MeasurementToyGenerationContext,
    seed: u64,
) -> MeasurementCombinationSpec {
    let y = sample_gvm_toy_values_from_context(context, seed);
    let mut toy = context.template_spec.clone();
    for (measurement, value) in toy.measurements.iter_mut().zip(y.iter()) {
        measurement.value = *value;
    }
    toy
}

fn simulate_gvm_toy_prepared_spec_from_context(
    context: &MeasurementToyGenerationContext,
    seed: u64,
) -> PreparedSpec {
    context.prepared_template.clone_with_y(sample_gvm_toy_values_from_context(context, seed))
}

fn numerical_gvm_calibration_toy_result(
    prep: PreparedSpec,
    ci_level: f64,
) -> Result<MeasurementCalibrationToyResult> {
    let objective = MeasurementCombineObjective::new(&prep);
    let bounds = prep.bounds();
    let gvm_fit = numerical_gvm_fit(&prep, &objective, &bounds)?;
    let fit = &gvm_fit.fit;
    let chi2_level = ChiSquared::new(1.0)
        .map_err(|e| {
            Error::Computation(format!("failed to construct chi-squared distribution: {e}"))
        })?
        .inverse_cdf(ci_level);
    let (ci_lo, ci_hi) =
        profile_ci_mu(&objective, &prep, &bounds, &fit.parameters, fit.fval, chi2_level)?;
    let sigma = ((ci_hi - ci_lo) / 2.0).abs();
    let df = prep.y.len().saturating_sub(1);
    let gof_stat = 2.0 * fit.fval.max(0.0);
    calibration_toy_result_from_fit(&prep, &gvm_fit.state, sigma, gof_stat, df)
}

fn numerical_paper_gvm_calibration_toy_result_with_warm_start(
    prep: PreparedSpec,
    ci_level: f64,
    warm_start: Option<&PaperWarmStartGuide>,
) -> Result<MeasurementCalibrationToyResult> {
    let objective = PaperMeasurementCombineObjective::new(&prep);
    let bounds = objective.bounds();
    let owned_warm_start;
    let warm_start = if let Some(warm_start) = warm_start {
        Some(warm_start)
    } else {
        owned_warm_start = default_numerical_paper_warm_start(&objective, ci_level);
        owned_warm_start.as_ref()
    };
    let gvm_fit = numerical_paper_gvm_fit(&prep, &objective, &bounds, warm_start)?;
    let fit = &gvm_fit.fit;
    let chi2_level = ChiSquared::new(1.0)
        .map_err(|e| {
            Error::Computation(format!("failed to construct chi-squared distribution: {e}"))
        })?
        .inverse_cdf(ci_level);
    let (ci_lo, ci_hi) = profile_ci_mu_paper(
        &objective,
        &bounds,
        &fit.parameters,
        fit.fval,
        chi2_level,
        warm_start,
    )?;
    let sigma = ((ci_hi - ci_lo) / 2.0).abs();
    let df = prep.y.len().saturating_sub(1);
    let gof_stat = 2.0 * fit.fval.max(0.0);
    calibration_toy_result_from_fit(&prep, &gvm_fit.state, sigma, gof_stat, df)
}

fn combine_measurements_calibration_toy_result_with_prepared_solver(
    prep: PreparedSpec,
    ci_level: f64,
    solver: MeasurementCombinationSolver,
) -> Result<MeasurementCalibrationToyResult> {
    combine_measurements_calibration_toy_result_with_prepared_solver_and_warm_start(
        prep, ci_level, solver, None,
    )
}

fn combine_measurements_calibration_toy_result_with_prepared_solver_and_warm_start(
    prep: PreparedSpec,
    ci_level: f64,
    solver: MeasurementCombinationSolver,
    preferred_paper_warm_start: Option<&PaperWarmStartGuide>,
) -> Result<MeasurementCalibrationToyResult> {
    if !(0.0 < ci_level && ci_level < 1.0) {
        return Err(Error::Validation(format!(
            "ci_level must satisfy 0 < ci_level < 1, got={ci_level}"
        )));
    }
    let requested_error_on_error = prep.systematics.iter().any(|s| s.error_on_error > 0.0);
    if !requested_error_on_error {
        let out = fixed_variance_result(prep, ci_level)?;
        return calibration_toy_result_from_full_result(&out);
    }

    match solver {
        MeasurementCombinationSolver::Numerical => {
            numerical_gvm_calibration_toy_result(prep, ci_level)
        }
        MeasurementCombinationSolver::NumericalPaper => {
            numerical_paper_gvm_calibration_toy_result_with_warm_start(
                prep,
                ci_level,
                preferred_paper_warm_start,
            )
        }
        MeasurementCombinationSolver::AnalyticPerturbative => {
            let out = analytic_perturbative_gvm_result(prep, ci_level, 1)?;
            calibration_toy_result_from_full_result(&out)
        }
        MeasurementCombinationSolver::Auto => {
            match analytic_perturbative_attempt(prep.clone(), ci_level, 1)? {
                AnalyticPerturbativeAttempt::Completed(result) => {
                    calibration_toy_result_from_full_result(&result)
                }
                AnalyticPerturbativeAttempt::FallbackWarmStart(warm_start) => {
                    let warm_start = warm_start.as_ref().or(preferred_paper_warm_start);
                    numerical_paper_gvm_calibration_toy_result_with_warm_start(
                        prep, ci_level, warm_start,
                    )
                }
            }
        }
    }
}

fn combine_measurements_calibration_toy_result_with_solver(
    spec: &MeasurementCombinationSpec,
    ci_level: f64,
    solver: MeasurementCombinationSolver,
) -> Result<MeasurementCalibrationToyResult> {
    spec.validate()?;
    let prep = PreparedSpec::from_spec(spec)?;
    combine_measurements_calibration_toy_result_with_prepared_solver(prep, ci_level, solver)
}

fn run_measurement_calibration_toy(
    toy_generation: &MeasurementToyGenerationContext,
    ci_level: f64,
    fit_solver: MeasurementCombinationSolver,
    seed: u64,
    toy_idx: usize,
) -> Result<MeasurementCalibrationToyResult> {
    let prep = simulate_gvm_toy_prepared_spec_from_context(
        toy_generation,
        seed.wrapping_add(toy_idx as u64),
    );
    combine_measurements_calibration_toy_result_with_prepared_solver_and_warm_start(
        prep,
        ci_level,
        fit_solver,
        toy_generation.paper_warm_start.as_ref(),
    )
}

fn fixed_variance_result(
    prep: PreparedSpec,
    ci_level: f64,
) -> Result<MeasurementCombinationResult> {
    let base = fixed_variance_solution(&prep)?;
    let z = standard_normal()?.inverse_cdf(0.5 + 0.5 * ci_level);
    let df = prep.y.len().saturating_sub(1);
    let p_value = if df > 0 { Some(1.0 - chi_squared_cdf(base.chi2, df as f64)) } else { None };
    Ok(MeasurementCombinationResult {
        schema_version: MEASUREMENT_COMBINATION_RESULT_SCHEMA_V0.to_string(),
        poi: prep.poi.clone(),
        mu_hat: base.mu_hat,
        confidence_interval: ConfidenceInterval {
            ci_level,
            lower: base.mu_hat - z * base.sigma,
            upper: base.mu_hat + z * base.sigma,
            sigma: base.sigma,
        },
        goodness_of_fit: GoodnessOfFit {
            chi2: base.chi2,
            df,
            p_value,
            method: "fixed_variance_gaussian".to_string(),
        },
        converged: true,
        stability: GVM_STABILITY_STABLE.to_string(),
        optimizer: OptimizerDiagnostics {
            method: "closed_form_blue".to_string(),
            n_iter: 1,
            n_fev: 1,
            n_gev: 0,
            message: "closed form".to_string(),
        },
        diagnostics: ResearchDiagnostics {
            input_schema_version: MEASUREMENT_COMBINATION_SCHEMA_V0.to_string(),
            n_measurements: prep.y.len(),
            n_systematics: prep.n_systematics_total,
            requested_error_on_error: false,
            supports_error_on_error: true,
            requested_solver: None,
            effective_solver: None,
            corr_regularization_deltas: prep.corr_regularization_deltas.clone(),
            profiled_variance_scales: vec![1.0; prep.n_systematics_total],
            theta_l2_norms: vec![0.0; prep.n_systematics_total],
            perturbative_validity: PerturbativeValidityDiagnostics {
                threshold: 1.0,
                systematic_names: Vec::new(),
                condition_values: Vec::new(),
                within_threshold: Vec::new(),
            },
            bartlett: unsupported_bartlett_diagnostics(
                "Bartlett diagnostics require at least one systematic with error_on_error > 0",
            ),
        },
    })
}

fn numerical_gvm_result(prep: PreparedSpec, ci_level: f64) -> Result<MeasurementCombinationResult> {
    let objective = MeasurementCombineObjective::new(&prep);
    let bounds = prep.bounds();
    let gvm_fit = numerical_gvm_fit(&prep, &objective, &bounds)?;
    let fit = &gvm_fit.fit;
    let chi2_level = ChiSquared::new(1.0)
        .map_err(|e| {
            Error::Computation(format!("failed to construct chi-squared distribution: {e}"))
        })?
        .inverse_cdf(ci_level);
    let (ci_lo, ci_hi) =
        profile_ci_mu(&objective, &prep, &bounds, &fit.parameters, fit.fval, chi2_level)?;
    let sigma = ((ci_hi - ci_lo) / 2.0).abs();
    let df = prep.y.len().saturating_sub(1);
    let gof_stat = 2.0 * fit.fval.max(0.0);
    let p_value = if df > 0 { Some(1.0 - chi_squared_cdf(gof_stat, df as f64)) } else { None };
    let perturbative_validity = compute_perturbative_validity(&prep, &gvm_fit.state);
    let bartlett = compute_bartlett_diagnostics(&prep, &gvm_fit.state, sigma, gof_stat, df);

    Ok(MeasurementCombinationResult {
        schema_version: MEASUREMENT_COMBINATION_RESULT_SCHEMA_V0.to_string(),
        poi: prep.poi.clone(),
        mu_hat: fit.parameters[0],
        confidence_interval: ConfidenceInterval { ci_level, lower: ci_lo, upper: ci_hi, sigma },
        goodness_of_fit: GoodnessOfFit {
            chi2: gof_stat,
            df,
            p_value,
            method: "profile_penalty_proxy".to_string(),
        },
        converged: fit.converged,
        stability: GVM_STABILITY_STABLE.to_string(),
        optimizer: OptimizerDiagnostics {
            method: "numerical_profile_gvm".to_string(),
            n_iter: fit.n_iter as usize,
            n_fev: fit.n_fev,
            n_gev: fit.n_gev,
            message: fit.message.clone(),
        },
        diagnostics: ResearchDiagnostics {
            input_schema_version: MEASUREMENT_COMBINATION_SCHEMA_V0.to_string(),
            n_measurements: prep.y.len(),
            n_systematics: prep.n_systematics_total,
            requested_error_on_error: true,
            supports_error_on_error: true,
            requested_solver: None,
            effective_solver: None,
            corr_regularization_deltas: prep.corr_regularization_deltas.clone(),
            profiled_variance_scales: gvm_fit.state.profiled_variance_scales,
            theta_l2_norms: gvm_fit.state.theta_l2_norms,
            perturbative_validity,
            bartlett,
        },
    })
}

fn numerical_paper_gvm_result(
    prep: PreparedSpec,
    ci_level: f64,
) -> Result<MeasurementCombinationResult> {
    numerical_paper_gvm_result_with_warm_start(prep, ci_level, None)
}

fn numerical_paper_gvm_result_with_warm_start(
    prep: PreparedSpec,
    ci_level: f64,
    warm_start: Option<&PaperWarmStartGuide>,
) -> Result<MeasurementCombinationResult> {
    let objective = PaperMeasurementCombineObjective::new(&prep);
    let bounds = objective.bounds();
    let owned_warm_start;
    let warm_start = if let Some(warm_start) = warm_start {
        Some(warm_start)
    } else {
        owned_warm_start = default_numerical_paper_warm_start(&objective, ci_level);
        owned_warm_start.as_ref()
    };
    let gvm_fit = numerical_paper_gvm_fit(&prep, &objective, &bounds, warm_start)?;
    let fit = &gvm_fit.fit;
    let chi2_level = ChiSquared::new(1.0)
        .map_err(|e| {
            Error::Computation(format!("failed to construct chi-squared distribution: {e}"))
        })?
        .inverse_cdf(ci_level);
    let (ci_lo, ci_hi) = profile_ci_mu_paper(
        &objective,
        &bounds,
        &fit.parameters,
        fit.fval,
        chi2_level,
        warm_start,
    )?;
    let sigma = ((ci_hi - ci_lo) / 2.0).abs();
    let df = prep.y.len().saturating_sub(1);
    let gof_stat = 2.0 * fit.fval.max(0.0);
    let p_value = if df > 0 { Some(1.0 - chi_squared_cdf(gof_stat, df as f64)) } else { None };
    let perturbative_validity = compute_perturbative_validity(&prep, &gvm_fit.state);
    let bartlett = compute_bartlett_diagnostics(&prep, &gvm_fit.state, sigma, gof_stat, df);

    Ok(MeasurementCombinationResult {
        schema_version: MEASUREMENT_COMBINATION_RESULT_SCHEMA_V0.to_string(),
        poi: prep.poi.clone(),
        mu_hat: fit.parameters[0],
        confidence_interval: ConfidenceInterval { ci_level, lower: ci_lo, upper: ci_hi, sigma },
        goodness_of_fit: GoodnessOfFit {
            chi2: gof_stat,
            df,
            p_value,
            method: "profile_penalty_proxy_paper_theta".to_string(),
        },
        converged: fit.converged,
        stability: GVM_STABILITY_STABLE.to_string(),
        optimizer: OptimizerDiagnostics {
            method: "numerical_profile_gvm_original_theta".to_string(),
            n_iter: fit.n_iter as usize,
            n_fev: fit.n_fev,
            n_gev: fit.n_gev,
            message: fit.message.clone(),
        },
        diagnostics: ResearchDiagnostics {
            input_schema_version: MEASUREMENT_COMBINATION_SCHEMA_V0.to_string(),
            n_measurements: prep.y.len(),
            n_systematics: prep.n_systematics_total,
            requested_error_on_error: true,
            supports_error_on_error: true,
            requested_solver: None,
            effective_solver: None,
            corr_regularization_deltas: prep.corr_regularization_deltas.clone(),
            profiled_variance_scales: gvm_fit.state.profiled_variance_scales,
            theta_l2_norms: gvm_fit.state.theta_l2_norms,
            perturbative_validity,
            bartlett,
        },
    })
}

fn analytic_perturbative_gvm_result(
    prep: PreparedSpec,
    ci_level: f64,
    order: usize,
) -> Result<MeasurementCombinationResult> {
    match analytic_perturbative_attempt(prep, ci_level, order)? {
        AnalyticPerturbativeAttempt::Completed(result) => Ok(*result),
        AnalyticPerturbativeAttempt::FallbackWarmStart(_) => Err(perturbative_validity_error()),
    }
}

fn analytic_perturbative_attempt(
    prep: PreparedSpec,
    ci_level: f64,
    order: usize,
) -> Result<AnalyticPerturbativeAttempt> {
    let solver = AnalyticPerturbativeSolver::new(&prep, order);
    let mle = match minimize_analytic_profile(&solver, prep.mu_bounds) {
        Ok(mle) => mle,
        Err(err) if is_perturbative_validity_error(&err) => {
            return Ok(AnalyticPerturbativeAttempt::FallbackWarmStart(None));
        }
        Err(err) => return Err(err),
    };
    let fallback_warm_start = paper_warm_start_from_profiled_state(&prep, mle.mu, &mle.state);
    if ensure_perturbative_validity(&prep, &mle.state).is_err() {
        return Ok(AnalyticPerturbativeAttempt::FallbackWarmStart(fallback_warm_start));
    }
    let chi2_level = ChiSquared::new(1.0)
        .map_err(|e| {
            Error::Computation(format!("failed to construct chi-squared distribution: {e}"))
        })?
        .inverse_cdf(ci_level);
    let target = mle.nll + 0.5 * chi2_level;
    let sigma_guess = prep.fixed_sigma_guess.max(1e-3);
    let (ci_lo, ci_hi) = rayon::join(
        || find_analytic_profile_bound(&solver, mle.mu, -1.0, sigma_guess, target),
        || find_analytic_profile_bound(&solver, mle.mu, 1.0, sigma_guess, target),
    );
    let ci_lo = match ci_lo {
        Ok(ci_lo) => ci_lo,
        Err(err) if is_perturbative_validity_error(&err) => {
            return Ok(AnalyticPerturbativeAttempt::FallbackWarmStart(fallback_warm_start));
        }
        Err(err) => return Err(err),
    };
    let ci_hi = match ci_hi {
        Ok(ci_hi) => ci_hi,
        Err(err) if is_perturbative_validity_error(&err) => {
            return Ok(AnalyticPerturbativeAttempt::FallbackWarmStart(fallback_warm_start));
        }
        Err(err) => return Err(err),
    };
    match solver.profile_at_mu(ci_lo) {
        Ok(_) => {}
        Err(err) if is_perturbative_validity_error(&err) => {
            return Ok(AnalyticPerturbativeAttempt::FallbackWarmStart(fallback_warm_start));
        }
        Err(err) => return Err(err),
    }
    match solver.profile_at_mu(ci_hi) {
        Ok(_) => {}
        Err(err) if is_perturbative_validity_error(&err) => {
            return Ok(AnalyticPerturbativeAttempt::FallbackWarmStart(fallback_warm_start));
        }
        Err(err) => return Err(err),
    }
    let sigma = ((ci_hi - ci_lo) / 2.0).abs();
    let df = prep.y.len().saturating_sub(1);
    let gof_stat = 2.0 * mle.nll.max(0.0);
    let p_value = if df > 0 { Some(1.0 - chi_squared_cdf(gof_stat, df as f64)) } else { None };
    let perturbative_validity = compute_perturbative_validity(&prep, &mle.state);
    let bartlett = compute_bartlett_diagnostics(&prep, &mle.state, sigma, gof_stat, df);

    Ok(AnalyticPerturbativeAttempt::Completed(Box::new(MeasurementCombinationResult {
        schema_version: MEASUREMENT_COMBINATION_RESULT_SCHEMA_V0.to_string(),
        poi: prep.poi.clone(),
        mu_hat: mle.mu,
        confidence_interval: ConfidenceInterval { ci_level, lower: ci_lo, upper: ci_hi, sigma },
        goodness_of_fit: GoodnessOfFit {
            chi2: gof_stat,
            df,
            p_value,
            method: "analytic_perturbative_profile".to_string(),
        },
        converged: true,
        stability: GVM_STABILITY_STABLE.to_string(),
        optimizer: OptimizerDiagnostics {
            method: "analytic_perturbative_order_eps2".to_string(),
            n_iter: order.max(1),
            n_fev: 0,
            n_gev: 0,
            message: "perturbative profile in original theta basis".to_string(),
        },
        diagnostics: ResearchDiagnostics {
            input_schema_version: MEASUREMENT_COMBINATION_SCHEMA_V0.to_string(),
            n_measurements: prep.y.len(),
            n_systematics: prep.n_systematics_total,
            requested_error_on_error: true,
            supports_error_on_error: true,
            requested_solver: None,
            effective_solver: None,
            corr_regularization_deltas: prep.corr_regularization_deltas.clone(),
            profiled_variance_scales: mle.state.profiled_variance_scales,
            theta_l2_norms: mle.state.theta_l2_norms,
            perturbative_validity,
            bartlett,
        },
    })))
}

fn numerical_gvm_fit(
    prep: &PreparedSpec,
    objective: &MeasurementCombineObjective,
    bounds: &[(f64, f64)],
) -> Result<NumericalGvmFit> {
    let init = prep.init_params()?;
    let optimizer = LbfgsbOptimizer::new(OptimizerConfig::from_strategy(
        crate::optimizer::OptimizerStrategy::HighPrecision,
    ));
    let fit = optimizer.minimize(objective, &init, bounds)?;
    let profiled_variance_scales = prep
        .systematics
        .iter()
        .enumerate()
        .map(|(i, _)| objective.tau_value(&fit.parameters, i))
        .collect::<Vec<_>>();
    let theta_original = prep
        .systematics
        .iter()
        .enumerate()
        .map(|(i, syst)| {
            let theta = DVector::from_column_slice(objective.theta_slice(&fit.parameters, i));
            &syst.corr_factor * theta
        })
        .collect::<Vec<_>>();
    let theta_l2_norms = theta_original.iter().map(|theta| theta.norm()).collect::<Vec<_>>();
    Ok(NumericalGvmFit {
        fit,
        state: ProfiledGvmState { theta_original, profiled_variance_scales, theta_l2_norms },
    })
}

fn numerical_paper_gvm_fit(
    prep: &PreparedSpec,
    objective: &PaperMeasurementCombineObjective,
    bounds: &[(f64, f64)],
    warm_start: Option<&PaperWarmStartGuide>,
) -> Result<PaperNumericalGvmFit> {
    let init = warm_start
        .map(|guide| guide.mle_params.clone())
        .or_else(|| objective.analytic_warm_start_params())
        .unwrap_or(objective.init_params()?);
    let optimizer = LbfgsbOptimizer::new(OptimizerConfig::from_strategy(
        crate::optimizer::OptimizerStrategy::HighPrecision,
    ));
    let fit = optimizer.minimize(objective, &init, bounds)?;
    let profiled_variance_scales = prep
        .systematics
        .iter()
        .enumerate()
        .map(|(i, _)| objective.tau_value(&fit.parameters, i))
        .collect::<Vec<_>>();
    let theta_original = prep
        .systematics
        .iter()
        .enumerate()
        .map(|(i, _)| DVector::from_column_slice(objective.theta_slice(&fit.parameters, i)))
        .collect::<Vec<_>>();
    let theta_l2_norms = theta_original.iter().map(|theta| theta.norm()).collect::<Vec<_>>();
    Ok(PaperNumericalGvmFit {
        fit,
        state: ProfiledGvmState { theta_original, profiled_variance_scales, theta_l2_norms },
    })
}

fn profile_ci_mu(
    objective: &MeasurementCombineObjective,
    prep: &PreparedSpec,
    bounds: &[(f64, f64)],
    mle_params: &[f64],
    mle_nll: f64,
    chi2_level: f64,
) -> Result<(f64, f64)> {
    let target = mle_nll + 0.5 * chi2_level;
    let sigma_guess = prep.fixed_sigma_guess.max(1e-3);
    let optimizer = LbfgsbOptimizer::new(OptimizerConfig::from_strategy(
        crate::optimizer::OptimizerStrategy::HighPrecision,
    ));
    let (lo, hi) = rayon::join(
        || {
            let mut lower_start = mle_params.to_vec();
            find_profile_bound(
                objective,
                bounds,
                &optimizer,
                &mut lower_start,
                mle_params[0],
                mle_nll,
                -1.0,
                sigma_guess,
                target,
            )
        },
        || {
            let mut upper_start = mle_params.to_vec();
            find_profile_bound(
                objective,
                bounds,
                &optimizer,
                &mut upper_start,
                mle_params[0],
                mle_nll,
                1.0,
                sigma_guess,
                target,
            )
        },
    );
    let lo = lo?;
    let hi = hi?;
    Ok((lo, hi))
}

fn find_profile_bound(
    objective: &MeasurementCombineObjective,
    bounds: &[(f64, f64)],
    optimizer: &LbfgsbOptimizer,
    start_params: &mut [f64],
    mu_hat: f64,
    mu_hat_fval: f64,
    direction: f64,
    sigma_guess: f64,
    target: f64,
) -> Result<f64> {
    let mu_bounds = bounds[0];
    let mut inner_mu = mu_hat;
    let mut inner_fval = mu_hat_fval;
    let mut inner_params = start_params.to_vec();
    let mut outer_mu = mu_hat;
    let mut outer_fval = mu_hat_fval;
    let mut outer_params = start_params.to_vec();
    let mut step = initial_profile_bracket_step(sigma_guess, target, mu_hat_fval);
    let max_error_on_error =
        objective.prep.systematics.iter().map(|syst| syst.error_on_error).fold(0.0_f64, f64::max);

    for _ in 0..40 {
        let candidate = (mu_hat + direction * step).clamp(mu_bounds.0, mu_bounds.1);
        let prof =
            profile_at_mu_with_optimizer(objective, bounds, &outer_params, candidate, optimizer)?;
        if prof.fval >= target || (candidate - outer_mu).abs() < 1e-12 {
            outer_mu = candidate;
            outer_fval = prof.fval;
            outer_params = prof.parameters;
            break;
        }
        inner_mu = candidate;
        inner_fval = prof.fval;
        inner_params = prof.parameters.clone();
        outer_mu = candidate;
        outer_fval = prof.fval;
        outer_params = prof.parameters;
        step *= 2.0;
    }

    if outer_mu == inner_mu {
        return Ok(outer_mu);
    }

    for _ in 0..80 {
        let mid = select_profile_bound_candidate(
            inner_mu,
            inner_fval,
            outer_mu,
            outer_fval,
            target,
            mu_hat_fval,
        );
        let seed = interpolate_profile_seed(
            mid,
            inner_mu,
            &inner_params,
            outer_mu,
            &outer_params,
            sigma_guess,
            max_error_on_error,
        );
        let prof = profile_at_mu_with_optimizer(objective, bounds, &seed, mid, optimizer)?;
        if (prof.fval - target).abs() < 1e-7 || (outer_mu - inner_mu).abs() < 1e-6 {
            return Ok(mid);
        }
        if prof.fval < target {
            inner_mu = mid;
            inner_fval = prof.fval;
            inner_params = prof.parameters;
        } else {
            outer_mu = mid;
            outer_fval = prof.fval;
            outer_params = prof.parameters;
        }
    }
    Ok(0.5 * (inner_mu + outer_mu))
}

fn find_analytic_profile_bound(
    solver: &AnalyticPerturbativeSolver<'_>,
    mu_hat: f64,
    direction: f64,
    sigma_guess: f64,
    target: f64,
) -> Result<f64> {
    let mu_bounds = solver.prep.mu_bounds;
    let mut inner_mu = mu_hat;
    let mu_hat_nll = solver.profile_at_mu_raw(mu_hat)?.nll;
    let mut inner_nll = mu_hat_nll;
    let mut outer_mu = mu_hat;
    let mut outer_nll = inner_nll;
    let mut step = initial_profile_bracket_step(sigma_guess, target, mu_hat_nll);

    for _ in 0..40 {
        let candidate = (mu_hat + direction * step).clamp(mu_bounds.0, mu_bounds.1);
        let prof = solver.profile_at_mu_raw(candidate)?;
        if prof.nll >= target || (candidate - outer_mu).abs() < 1e-12 {
            outer_mu = candidate;
            outer_nll = prof.nll;
            break;
        }
        inner_mu = candidate;
        inner_nll = prof.nll;
        outer_mu = candidate;
        outer_nll = prof.nll;
        step *= 2.0;
    }

    if outer_mu == inner_mu {
        return Ok(outer_mu);
    }

    for _ in 0..80 {
        let mid = select_profile_bound_candidate(
            inner_mu, inner_nll, outer_mu, outer_nll, target, mu_hat_nll,
        );
        let prof = solver.profile_at_mu_raw(mid)?;
        if (prof.nll - target).abs() < 1e-7 || (outer_mu - inner_mu).abs() < 1e-6 {
            return Ok(mid);
        }
        if prof.nll < target {
            inner_mu = mid;
            inner_nll = prof.nll;
        } else {
            outer_mu = mid;
            outer_nll = prof.nll;
        }
        let _ = (inner_nll, outer_nll);
    }
    Ok(0.5 * (inner_mu + outer_mu))
}

fn profile_at_mu_with_optimizer(
    objective: &dyn ObjectiveFunction,
    bounds: &[(f64, f64)],
    init: &[f64],
    mu: f64,
    optimizer: &LbfgsbOptimizer,
) -> Result<OptimizationResult> {
    let mut prof_bounds = bounds.to_vec();
    prof_bounds[0] = (mu, mu);
    let mut prof_init = init.to_vec();
    prof_init[0] = mu;
    optimizer.minimize(objective, &prof_init, &prof_bounds)
}

fn profile_ci_mu_paper(
    objective: &PaperMeasurementCombineObjective,
    bounds: &[(f64, f64)],
    mle_params: &[f64],
    mle_nll: f64,
    chi2_level: f64,
    warm_start: Option<&PaperWarmStartGuide>,
) -> Result<(f64, f64)> {
    let (interval, _) = profile_ci_mu_paper_with_config_and_workload(
        objective,
        bounds,
        mle_params,
        mle_nll,
        chi2_level,
        warm_start,
        paper_profile_scan_optimizer_config(),
    )?;
    Ok(interval)
}

fn profile_ci_mu_paper_with_workload(
    objective: &PaperMeasurementCombineObjective,
    bounds: &[(f64, f64)],
    mle_params: &[f64],
    mle_nll: f64,
    chi2_level: f64,
    warm_start: Option<&PaperWarmStartGuide>,
) -> Result<((f64, f64), ProfileCiWorkload)> {
    profile_ci_mu_paper_with_config_and_workload(
        objective,
        bounds,
        mle_params,
        mle_nll,
        chi2_level,
        warm_start,
        OptimizerConfig::from_strategy(crate::optimizer::OptimizerStrategy::HighPrecision),
    )
}

fn profile_ci_mu_paper_with_config_and_workload(
    objective: &PaperMeasurementCombineObjective,
    bounds: &[(f64, f64)],
    mle_params: &[f64],
    mle_nll: f64,
    chi2_level: f64,
    warm_start: Option<&PaperWarmStartGuide>,
    optimizer_config: OptimizerConfig,
) -> Result<((f64, f64), ProfileCiWorkload)> {
    let target = mle_nll + 0.5 * chi2_level;
    let sigma_guess = objective.prep.fixed_sigma_guess.max(1e-3);
    let bracket_optimizer =
        LbfgsbOptimizer::new_relaxed_profile_bracket(paper_profile_bracket_optimizer_config());
    let bisect_optimizer = LbfgsbOptimizer::new_bounded_profile_bisect(optimizer_config);
    let lower_hint = warm_start.and_then(|guide| guide.lower_hint.as_ref());
    let upper_hint = warm_start.and_then(|guide| guide.upper_hint.as_ref());
    let (lo, hi) = rayon::join(
        || {
            let mut lower_start = mle_params.to_vec();
            find_profile_bound_paper(
                objective,
                bounds,
                &bracket_optimizer,
                &bisect_optimizer,
                &mut lower_start,
                mle_params[0],
                mle_nll,
                -1.0,
                sigma_guess,
                target,
                lower_hint,
            )
        },
        || {
            let mut upper_start = mle_params.to_vec();
            find_profile_bound_paper(
                objective,
                bounds,
                &bracket_optimizer,
                &bisect_optimizer,
                &mut upper_start,
                mle_params[0],
                mle_nll,
                1.0,
                sigma_guess,
                target,
                upper_hint,
            )
        },
    );
    let lo = lo?;
    let hi = hi?;
    Ok(((lo.mu, hi.mu), ProfileCiWorkload { lower: lo.workload, upper: hi.workload }))
}

fn find_profile_bound_paper(
    objective: &PaperMeasurementCombineObjective,
    bounds: &[(f64, f64)],
    bracket_optimizer: &LbfgsbOptimizer,
    bisect_optimizer: &LbfgsbOptimizer,
    start_params: &mut [f64],
    mu_hat: f64,
    mu_hat_fval: f64,
    direction: f64,
    sigma_guess: f64,
    target: f64,
    hint: Option<&PaperProfileBoundHint>,
) -> Result<ProfileBoundResult> {
    let mu_bounds = bounds[0];
    let mut inner_mu = mu_hat;
    let mut inner_fval = mu_hat_fval;
    let mut inner_params = start_params.to_vec();
    let mut outer_mu = mu_hat;
    let mut outer_fval = mu_hat_fval;
    let mut outer_params = start_params.to_vec();
    let mut step = initial_profile_bracket_step(sigma_guess, target, mu_hat_fval);
    let mut bracketed = false;
    let mut workload = ProfileBoundWorkload::default();
    let max_error_on_error =
        objective.prep.systematics.iter().map(|syst| syst.error_on_error).fold(0.0_f64, f64::max);

    if let Some(hint) = hint {
        let hint_mu = hint.mu.clamp(mu_bounds.0, mu_bounds.1);
        if direction * (hint_mu - mu_hat) > 1e-12 {
            let prof = profile_at_mu_with_optimizer(
                objective,
                bounds,
                &hint.params,
                hint_mu,
                bracket_optimizer,
            )?;
            workload.record(&prof, ProfileBoundPhase::Bracket);
            if prof.fval >= target {
                outer_mu = hint_mu;
                outer_fval = prof.fval;
                outer_params = prof.parameters;
                bracketed = true;
            } else {
                inner_mu = hint_mu;
                inner_fval = prof.fval;
                inner_params = prof.parameters.clone();
                outer_mu = hint_mu;
                outer_fval = prof.fval;
                outer_params = prof.parameters;
                step = (hint_mu - mu_hat).abs().max(sigma_guess.max(1e-3));
            }
        }
    }

    if !bracketed {
        for _ in 0..40 {
            let candidate = (outer_mu + direction * step).clamp(mu_bounds.0, mu_bounds.1);
            let prof = profile_at_mu_with_optimizer(
                objective,
                bounds,
                &outer_params,
                candidate,
                bracket_optimizer,
            )?;
            workload.record(&prof, ProfileBoundPhase::Bracket);
            if prof.fval >= target || (candidate - outer_mu).abs() < 1e-12 {
                outer_mu = candidate;
                outer_fval = prof.fval;
                outer_params = prof.parameters;
                break;
            }
            inner_mu = candidate;
            inner_fval = prof.fval;
            inner_params = prof.parameters.clone();
            outer_mu = candidate;
            outer_fval = prof.fval;
            outer_params = prof.parameters;
            step *= 2.0;
        }
    }

    if outer_mu == inner_mu {
        return Ok(ProfileBoundResult { mu: outer_mu, workload });
    }

    for _ in 0..80 {
        let mid = select_profile_bound_candidate(
            inner_mu,
            inner_fval,
            outer_mu,
            outer_fval,
            target,
            mu_hat_fval,
        );
        let seed = interpolate_profile_seed(
            mid,
            inner_mu,
            &inner_params,
            outer_mu,
            &outer_params,
            sigma_guess,
            max_error_on_error,
        );
        let prof = profile_at_mu_with_optimizer(objective, bounds, &seed, mid, bisect_optimizer)?;
        workload.record(&prof, ProfileBoundPhase::Bisect);
        if (prof.fval - target).abs() < 1e-7 || (outer_mu - inner_mu).abs() < 1e-6 {
            return Ok(ProfileBoundResult { mu: mid, workload });
        }
        if prof.fval < target {
            inner_mu = mid;
            inner_fval = prof.fval;
            inner_params = prof.parameters;
        } else {
            outer_mu = mid;
            outer_fval = prof.fval;
            outer_params = prof.parameters;
        }
    }
    Ok(ProfileBoundResult { mu: 0.5 * (inner_mu + outer_mu), workload })
}

fn interpolate_profile_seed(
    target_mu: f64,
    inner_mu: f64,
    inner_params: &[f64],
    outer_mu: f64,
    outer_params: &[f64],
    sigma_guess: f64,
    max_error_on_error: f64,
) -> Vec<f64> {
    let span = outer_mu - inner_mu;
    if span.abs() < 1e-12 || inner_params.len() != outer_params.len() {
        return inner_params.to_vec();
    }
    let frac = (target_mu - inner_mu) / span;
    if !frac.is_finite() {
        return inner_params.to_vec();
    }
    let inner_dist = (target_mu - inner_mu).abs();
    let outer_dist = (outer_mu - target_mu).abs();
    if frac <= 0.1 || inner_dist > span.abs() || outer_dist > span.abs() {
        return inner_params.to_vec();
    }
    if frac >= 0.9 {
        return outer_params.to_vec();
    }
    let interpolation_span_limit =
        (3.0 - max_error_on_error.clamp(0.0, 1.0)).max(1.5) * sigma_guess.max(1e-3);
    if span.abs() > interpolation_span_limit {
        return if inner_dist <= outer_dist {
            inner_params.to_vec()
        } else {
            outer_params.to_vec()
        };
    }
    inner_params
        .iter()
        .zip(outer_params.iter())
        .map(|(&inner, &outer)| inner + frac * (outer - inner))
        .collect()
}

fn select_profile_bound_candidate(
    inner_mu: f64,
    inner_fval: f64,
    outer_mu: f64,
    outer_fval: f64,
    target: f64,
    mle_fval: f64,
) -> f64 {
    let span = outer_mu - inner_mu;
    if !span.is_finite() || span.abs() < 1e-12 {
        return inner_mu;
    }

    let y_inner = (inner_fval - mle_fval).max(0.0).sqrt();
    let y_outer = (outer_fval - mle_fval).max(0.0).sqrt();
    let y_target = (target - mle_fval).max(0.0).sqrt();
    let denom = y_outer - y_inner;
    if !denom.is_finite() || denom.abs() < 1e-12 {
        return 0.5 * (inner_mu + outer_mu);
    }

    let frac = ((y_target - y_inner) / denom).clamp(0.05, 0.95);
    let candidate = inner_mu + frac * span;
    if candidate.is_finite() { candidate } else { 0.5 * (inner_mu + outer_mu) }
}

fn initial_profile_bracket_step(sigma_guess: f64, target: f64, mle_fval: f64) -> f64 {
    let target_delta = (target - mle_fval).max(0.0);
    let z_score = (2.0 * target_delta).sqrt();
    let expected_dist = z_score * sigma_guess.max(1e-3);
    (1.5 * expected_dist).max(sigma_guess.max(1e-3))
}

struct FixedVarianceSolution {
    mu_hat: f64,
    sigma: f64,
    chi2: f64,
}

fn fixed_variance_solution(prep: &PreparedSpec) -> Result<FixedVarianceSolution> {
    let mut total_cov = symmetric_pseudoinverse(&prep.base_precision)?;
    for syst in &prep.systematics {
        for i in 0..prep.y.len() {
            for j in 0..prep.y.len() {
                total_cov[(i, j)] +=
                    syst.magnitudes[i] * syst.corr_cov[(i, j)] * syst.magnitudes[j];
            }
        }
    }
    let total_precision = symmetric_pseudoinverse(&total_cov)?;
    let denom = prep.ones.dot(&(&total_precision * &prep.ones));
    if !denom.is_finite() || denom <= 0.0 {
        return Err(Error::Computation(format!("invalid combination denominator: {denom}")));
    }
    let mu_hat = prep.ones.dot(&(&total_precision * &prep.y)) / denom;
    let sigma = (1.0 / denom).sqrt();
    let resid = &prep.y - prep.ones.clone().scale(mu_hat);
    let chi2 = quad_form(&total_precision, &resid);
    Ok(FixedVarianceSolution { mu_hat, sigma, chi2 })
}

fn total_covariance(spec: &MeasurementCombinationSpec) -> Result<DMatrix<f64>> {
    let n = spec.measurements.len();
    let mut cov = matrix_from_rows(&spec.stat_covariance, n, n)?;
    for syst in &spec.systematics {
        for i in 0..n {
            for j in 0..n {
                cov[(i, j)] += syst.magnitudes[i] * syst.corr[i][j] * syst.magnitudes[j];
            }
        }
    }
    Ok(cov)
}

fn quad_form(matrix: &DMatrix<f64>, v: &DVector<f64>) -> f64 {
    (v.transpose() * matrix * v)[(0, 0)]
}

fn standard_normal() -> Result<Normal> {
    Normal::new(0.0, 1.0)
        .map_err(|e| Error::Computation(format!("failed to construct normal distribution: {e}")))
}

fn matrix_from_rows(rows: &[Vec<f64>], nrows: usize, ncols: usize) -> Result<DMatrix<f64>> {
    let mut data = Vec::with_capacity(nrows * ncols);
    for row in rows {
        if row.len() != ncols {
            return Err(Error::Validation(format!(
                "matrix column mismatch: got={} expected={ncols}",
                row.len()
            )));
        }
        data.extend(row.iter().copied());
    }
    Ok(DMatrix::from_row_slice(nrows, ncols, &data))
}

fn validate_square_symmetric(
    label: &str,
    rows: &[Vec<f64>],
    n: usize,
    require_corr_diag: bool,
    require_psd: bool,
) -> Result<()> {
    if rows.len() != n {
        return Err(Error::Validation(format!(
            "{label} row count mismatch: got={} expected={n}",
            rows.len()
        )));
    }
    let m = matrix_from_rows(rows, n, n)?;
    for i in 0..n {
        for j in 0..n {
            let a = m[(i, j)];
            let b = m[(j, i)];
            if !a.is_finite() || !b.is_finite() {
                return Err(Error::Validation(format!("{label} contains non-finite values")));
            }
            if (a - b).abs() > SYMM_TOL {
                return Err(Error::Validation(format!("{label} must be symmetric")));
            }
        }
        if require_corr_diag && (m[(i, i)] - 1.0).abs() > SYMM_TOL {
            return Err(Error::Validation(format!(
                "{label} diagonal must equal 1 within tolerance"
            )));
        }
    }
    if require_psd {
        let eigen = m.symmetric_eigen();
        if eigen.eigenvalues.iter().any(|v| *v < -PSD_TOL) {
            return Err(Error::Validation(format!("{label} must be positive semidefinite")));
        }
    }
    Ok(())
}

fn regularize_corr_for_precision(m: &DMatrix<f64>) -> Result<(DMatrix<f64>, f64)> {
    if m.nrows() != m.ncols() {
        return Err(Error::Validation(
            "correlation matrix must be square for regularization".to_string(),
        ));
    }
    let min_eigenvalue = corr_regularization_delta(m)?;
    if min_eigenvalue == 0.0 {
        return Ok((m.clone(), 0.0));
    }

    let delta = min_eigenvalue;
    let mut regularized = m.clone();
    for i in 0..regularized.nrows() {
        regularized[(i, i)] += delta;
    }
    Ok((regularized, delta))
}

fn corr_regularization_delta(m: &DMatrix<f64>) -> Result<f64> {
    if m.nrows() != m.ncols() {
        return Err(Error::Validation(
            "correlation matrix must be square for regularization".to_string(),
        ));
    }
    let eig = m.clone().symmetric_eigen();
    let min_eigenvalue = eig.eigenvalues.iter().fold(f64::INFINITY, |acc, v| acc.min(*v));
    if min_eigenvalue >= -PSD_TOL {
        return Ok(0.0);
    }
    Ok(-min_eigenvalue)
}

fn factorize_corr_for_nuisance(m: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    if m.nrows() != m.ncols() {
        return Err(Error::Validation(
            "correlation matrix must be square for factorization".to_string(),
        ));
    }
    let eig = m.clone().symmetric_eigen();
    let cols = eig.eigenvalues.iter().filter(|v| **v > PSD_TOL).count();
    if cols == 0 {
        return Ok(DMatrix::zeros(m.nrows(), 0));
    }
    let mut out = DMatrix::zeros(m.nrows(), cols);
    let mut c = 0usize;
    for (i, ev) in eig.eigenvalues.iter().enumerate() {
        if *ev > PSD_TOL {
            let scale = ev.sqrt();
            for r in 0..m.nrows() {
                out[(r, c)] = eig.eigenvectors[(r, i)] * scale;
            }
            c += 1;
        }
    }
    Ok(out)
}

fn symmetric_pseudoinverse(m: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    if m.nrows() != m.ncols() {
        return Err(Error::Validation("matrix must be square".to_string()));
    }
    if let Some(cholesky) = m.clone().cholesky() {
        let lower = cholesky.l();
        let mut min_diag = f64::INFINITY;
        let mut max_diag = 0.0_f64;
        for i in 0..lower.nrows() {
            let diag = lower[(i, i)].abs();
            min_diag = min_diag.min(diag);
            max_diag = max_diag.max(diag);
        }
        let diag_ratio = if max_diag > 0.0 { min_diag / max_diag } else { 0.0 };
        if diag_ratio.is_finite() && diag_ratio >= CHOLESKY_FAST_PATH_MIN_DIAG_RATIO {
            return Ok(cholesky.inverse());
        }
    }
    let eig = m.clone().symmetric_eigen();
    let mut inv_diag = DMatrix::zeros(m.nrows(), m.ncols());
    for i in 0..eig.eigenvalues.len() {
        let ev = eig.eigenvalues[i];
        if ev < -PSD_TOL {
            return Err(Error::Validation("matrix must be positive semidefinite".to_string()));
        }
        if ev > PSD_TOL {
            inv_diag[(i, i)] = 1.0 / ev;
        }
    }
    Ok(&eig.eigenvectors * inv_diag * eig.eigenvectors.transpose())
}

fn symmetrize_matrix(m: DMatrix<f64>) -> DMatrix<f64> {
    let transpose = m.transpose();
    (m + transpose).scale(0.5)
}

fn trace_product(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>) -> f64 {
    debug_assert_eq!(lhs.nrows(), rhs.nrows());
    debug_assert_eq!(lhs.ncols(), rhs.ncols());
    let mut trace = 0.0;
    for i in 0..lhs.nrows() {
        for j in 0..lhs.ncols() {
            trace += lhs[(i, j)] * rhs[(j, i)];
        }
    }
    trace
}

fn trace_square_of_product(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>) -> f64 {
    let product = lhs * rhs;
    let mut trace = 0.0;
    for i in 0..product.nrows() {
        for j in 0..product.ncols() {
            trace += product[(i, j)] * product[(j, i)];
        }
    }
    trace
}

fn chi_squared_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    statrs::function::gamma::gamma_lr(df / 2.0, x / 2.0)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_sd(values: &[f64], mean: f64) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let var = values
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f64>()
        / (values.len() as f64 - 1.0);
    var.sqrt()
}

fn unsupported_bartlett_diagnostics(reason: impl Into<String>) -> BartlettDiagnostics {
    BartlettDiagnostics {
        supported: false,
        method: "lawley_order_eps2".to_string(),
        unsupported_reason: Some(reason.into()),
        supported_systematics: Vec::new(),
        b_mu_theta: None,
        b_tilde_theta: None,
        b_mu: None,
        b_q: None,
        w_mu_scale: None,
        q_scale: None,
        q_star: None,
        p_value_star: None,
        sigma_scale: None,
        sigma_star: None,
        sigma2_unbiased_estimates: Vec::new(),
    }
}

fn compute_perturbative_validity(
    prep: &PreparedSpec,
    state: &ProfiledGvmState,
) -> PerturbativeValidityDiagnostics {
    let mut condition_values = Vec::with_capacity(prep.systematics.len());
    let mut within_threshold = Vec::with_capacity(prep.systematics.len());
    let mut systematic_names = Vec::with_capacity(prep.systematics.len());

    for (s_idx, syst) in prep.systematics.iter().enumerate() {
        let theta_original = &state.theta_original[s_idx];
        let quad = (&theta_original.transpose() * &syst.corr_precision * theta_original)[(0, 0)];
        let value = 2.0 * syst.error_on_error * syst.error_on_error * quad;
        systematic_names.push(syst.name.clone());
        condition_values.push(value);
        within_threshold.push(value < 1.0);
    }

    PerturbativeValidityDiagnostics {
        threshold: 1.0,
        systematic_names,
        condition_values,
        within_threshold,
    }
}

fn compute_bartlett_diagnostics(
    prep: &PreparedSpec,
    state: &ProfiledGvmState,
    sigma: f64,
    gof_stat: f64,
    df: usize,
) -> BartlettDiagnostics {
    if prep.systematics.is_empty() {
        return unsupported_bartlett_diagnostics(
            "Bartlett diagnostics require at least one systematic with error_on_error > 0",
        );
    }

    match compute_supported_bartlett_diagnostics(prep, state, sigma, gof_stat, df) {
        Ok(bartlett) => bartlett,
        Err(err) => unsupported_bartlett_diagnostics(format!(
            "Bartlett diagnostics failed for an otherwise supported case: {err}"
        )),
    }
}

fn compute_supported_bartlett_diagnostics(
    prep: &PreparedSpec,
    state: &ProfiledGvmState,
    sigma: f64,
    gof_stat: f64,
    df: usize,
) -> Result<BartlettDiagnostics> {
    let workspace = build_bartlett_workspace(prep, state)?;
    let n = workspace.n_per_source;
    let sigma2_unbiased_estimates = compute_unbiased_sigma2_estimates(prep, state, &workspace);

    let mut b_mu_theta = 0.0;
    let mut b_tilde_theta = 0.0;
    for (s_idx, syst) in prep.systematics.iter().enumerate() {
        let sigma2 = state.profiled_variance_scales[s_idx];
        let eps2 = syst.error_on_error * syst.error_on_error;
        let j_block = &workspace.j_blocks[s_idx];
        let j_tilde_block = &workspace.j_tilde_blocks[s_idx];
        let rho_inv = &syst.corr_precision;

        let t1 = trace_product(j_block, rho_inv);
        let t2 = trace_square_of_product(j_block, rho_inv);
        let t1_tilde = trace_product(j_tilde_block, rho_inv);
        let t2_tilde = trace_square_of_product(j_tilde_block, rho_inv);

        b_mu_theta += (4.0 * t1 / sigma2 + (t1 * t1 - 2.0 * t2) / (sigma2 * sigma2)) * eps2;
        b_tilde_theta += (4.0 * t1_tilde / sigma2
            + (t1_tilde * t1_tilde - 2.0 * t2_tilde) / (sigma2 * sigma2))
            * eps2;
    }

    let b_mu = b_mu_theta - b_tilde_theta;
    let w_mu_scale = 1.0 + b_mu;
    if !w_mu_scale.is_finite() || w_mu_scale <= 0.0 {
        return Err(Error::Computation(format!(
            "non-positive Bartlett profile scale: {w_mu_scale}"
        )));
    }

    let eps2_sum =
        prep.systematics.iter().map(|syst| syst.error_on_error * syst.error_on_error).sum::<f64>();
    let b_q = (2.0 * n as f64 + (n * n) as f64) * eps2_sum - b_mu_theta;
    let (q_scale, q_star, p_value_star) = if df > 0 {
        let scale = 1.0 + b_q / df as f64;
        if !scale.is_finite() || scale <= 0.0 {
            return Err(Error::Computation(format!("non-positive Bartlett GOF scale: {scale}")));
        }
        let q_star = gof_stat / scale;
        let p_value_star = Some(1.0 - chi_squared_cdf(q_star, df as f64));
        (Some(scale), Some(q_star), p_value_star)
    } else {
        (None, None, None)
    };

    Ok(BartlettDiagnostics {
        supported: true,
        method: "lawley_order_eps2_general".to_string(),
        unsupported_reason: None,
        supported_systematics: prep.systematics.iter().map(|s| s.name.clone()).collect(),
        b_mu_theta: Some(b_mu_theta),
        b_tilde_theta: Some(b_tilde_theta),
        b_mu: Some(b_mu),
        b_q: Some(b_q),
        w_mu_scale: Some(w_mu_scale),
        q_scale,
        q_star,
        p_value_star,
        sigma_scale: Some(w_mu_scale.sqrt()),
        sigma_star: Some(sigma * w_mu_scale.sqrt()),
        sigma2_unbiased_estimates,
    })
}

fn compute_unbiased_sigma2_estimates(
    prep: &PreparedSpec,
    state: &ProfiledGvmState,
    workspace: &BartlettWorkspace,
) -> Vec<f64> {
    let n = workspace.n_per_source as f64;
    prep.systematics
        .iter()
        .enumerate()
        .map(|(s_idx, syst)| {
            let eps2 = syst.error_on_error * syst.error_on_error;
            let j_block = &workspace.j_blocks[s_idx];
            let rho_inv = &syst.corr_precision;
            let t1 = trace_product(j_block, rho_inv);
            let quad = (&workspace.theta_hat_original[s_idx].transpose()
                * rho_inv
                * &workspace.theta_hat_original[s_idx])[(0, 0)];
            let sigma2 = (1.0 + 2.0 * eps2 * quad + 2.0 * eps2 * t1) / (1.0 + 2.0 * n * eps2);
            sigma2.max(1e-12).min(state.profiled_variance_scales[s_idx].max(1e-12) * 1e6)
        })
        .collect()
}

fn build_bartlett_workspace(
    prep: &PreparedSpec,
    state: &ProfiledGvmState,
) -> Result<BartlettWorkspace> {
    build_bartlett_workspace_with_threshold(prep, state, BARTLETT_FAST_PATH_NM_THRESHOLD)
}

fn build_bartlett_workspace_with_threshold(
    prep: &PreparedSpec,
    state: &ProfiledGvmState,
    nm_threshold: usize,
) -> Result<BartlettWorkspace> {
    let nm = prep.y.len().saturating_mul(prep.systematics.len());
    if nm > nm_threshold
        && let Some(workspace) = build_bartlett_workspace_fast(prep, state)?
    {
        return Ok(workspace);
    }
    build_bartlett_workspace_reference(prep, state)
}

fn build_bartlett_workspace_reference(
    prep: &PreparedSpec,
    state: &ProfiledGvmState,
) -> Result<BartlettWorkspace> {
    let n = prep.y.len();
    let n_sources = prep.systematics.len();
    let mut theta_offsets = Vec::with_capacity(n_sources);
    let mut theta_cursor = 0usize;
    for _ in &prep.systematics {
        theta_offsets.push(theta_cursor);
        theta_cursor += n;
    }

    let a_ones = &prep.base_precision * &prep.ones;
    let h_mu_mu = prep.ones.dot(&a_ones);
    let mut h_theta_theta = DMatrix::zeros(theta_cursor, theta_cursor);
    let mut h_mu_theta = DVector::zeros(theta_cursor);
    let design_diags = prep
        .systematics
        .iter()
        .map(|syst| DMatrix::from_diagonal(&syst.magnitudes))
        .collect::<Vec<_>>();
    let mut theta_hat_original = Vec::with_capacity(n_sources);

    for (s_idx, syst) in prep.systematics.iter().enumerate() {
        let offset = theta_offsets[s_idx];
        let sigma2 = state.profiled_variance_scales[s_idx];
        if !sigma2.is_finite() || sigma2 <= 0.0 {
            return Err(Error::Computation(format!(
                "invalid profiled variance scale for Bartlett diagnostics: systematic={} tau={sigma2}",
                syst.name
            )));
        }
        theta_hat_original.push(state.theta_original[s_idx].clone());

        let column = &design_diags[s_idx] * &a_ones;
        for i in 0..n {
            h_mu_theta[offset + i] = column[i];
        }

        for p_idx in 0..prep.systematics.len() {
            let p_offset = theta_offsets[p_idx];
            let cross = &design_diags[s_idx] * &prep.base_precision * &design_diags[p_idx];
            for i in 0..n {
                for j in 0..n {
                    h_theta_theta[(offset + i, p_offset + j)] = cross[(i, j)];
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                h_theta_theta[(offset + i, offset + j)] += syst.corr_precision[(i, j)] / sigma2;
            }
        }
    }

    let mut h = DMatrix::zeros(1 + theta_cursor, 1 + theta_cursor);
    h[(0, 0)] = h_mu_mu;
    for i in 0..theta_cursor {
        h[(0, 1 + i)] = h_mu_theta[i];
        h[(1 + i, 0)] = h_mu_theta[i];
    }
    for i in 0..theta_cursor {
        for j in 0..theta_cursor {
            h[(1 + i, 1 + j)] = h_theta_theta[(i, j)];
        }
    }

    let j_inv = symmetric_pseudoinverse(&h)?;
    let j_tilde_inv = symmetric_pseudoinverse(&h_theta_theta)?;
    let j_blocks = theta_offsets
        .iter()
        .map(|offset| j_inv.view((1 + offset, 1 + offset), (n, n)).into_owned())
        .collect();
    let j_tilde_blocks = theta_offsets
        .iter()
        .map(|offset| j_tilde_inv.view((*offset, *offset), (n, n)).into_owned())
        .collect();
    Ok(BartlettWorkspace {
        n_per_source: n,
        theta_hat_original,
        j_blocks,
        j_tilde_blocks,
        build_path: BartlettWorkspaceBuildPath::Reference,
    })
}

fn build_bartlett_workspace_fast(
    prep: &PreparedSpec,
    state: &ProfiledGvmState,
) -> Result<Option<BartlettWorkspace>> {
    let n = prep.y.len();
    let mut theta_hat_original = Vec::with_capacity(prep.systematics.len());
    let mut u_blocks = Vec::with_capacity(prep.systematics.len());
    let mut left_blocks = Vec::with_capacity(prep.systematics.len());
    let mut right_blocks = Vec::with_capacity(prep.systematics.len());
    let mut s_matrix = prep.base_cov.clone();

    for (s_idx, syst) in prep.systematics.iter().enumerate() {
        let sigma2 = state.profiled_variance_scales[s_idx];
        if !sigma2.is_finite() || sigma2 <= 0.0 {
            return Err(Error::Computation(format!(
                "invalid profiled variance scale for Bartlett diagnostics: systematic={} tau={sigma2}",
                syst.name
            )));
        }
        if syst.corr_factor.ncols() != n {
            return Ok(None);
        }

        let design = DMatrix::from_diagonal(&syst.magnitudes);
        let regularized_corr = &syst.corr_factor * syst.corr_factor.transpose();
        let u_block = regularized_corr.scale(sigma2);
        let left = &design * &u_block;
        let right = &u_block * &design;
        let s_contrib = &left * &design;

        s_matrix += s_contrib;
        theta_hat_original.push(state.theta_original[s_idx].clone());
        u_blocks.push(u_block);
        left_blocks.push(left);
        right_blocks.push(right);
    }

    let s_matrix = symmetrize_matrix(s_matrix);
    let Some(cholesky) = s_matrix.cholesky() else {
        return Ok(None);
    };

    let s_inv_ones = cholesky.solve(&prep.ones);
    let schur = prep.ones.dot(&s_inv_ones);
    if !schur.is_finite() || schur <= 0.0 {
        return Ok(None);
    }
    let mu_variance = 1.0 / schur;
    let mut j_tilde_blocks = Vec::with_capacity(prep.systematics.len());
    let mut x_blocks = Vec::with_capacity(prep.systematics.len());

    for s_idx in 0..prep.systematics.len() {
        let s_inv_left = cholesky.solve(&left_blocks[s_idx]);
        let j_tilde = symmetrize_matrix(&u_blocks[s_idx] - &right_blocks[s_idx] * &s_inv_left);
        let x = &right_blocks[s_idx] * &s_inv_ones;
        j_tilde_blocks.push(j_tilde);
        x_blocks.push(x);
    }
    let mut j_blocks = Vec::with_capacity(prep.systematics.len());
    for s_idx in 0..prep.systematics.len() {
        let correction = (&x_blocks[s_idx] * x_blocks[s_idx].transpose()).scale(mu_variance);
        let j_block = symmetrize_matrix(&j_tilde_blocks[s_idx] + correction);
        j_blocks.push(j_block);
    }

    Ok(Some(BartlettWorkspace {
        n_per_source: n,
        theta_hat_original,
        j_blocks,
        j_tilde_blocks,
        build_path: BartlettWorkspaceBuildPath::Fast,
    }))
}

fn covariance_sampling_factor(cov: &DMatrix<f64>) -> DMatrix<f64> {
    let eig = cov.clone().symmetric_eigen();
    let mut sqrt_diag = DMatrix::zeros(cov.nrows(), cov.ncols());
    for i in 0..cov.nrows() {
        sqrt_diag[(i, i)] = eig.eigenvalues[i].max(0.0).sqrt();
    }
    eig.eigenvectors * sqrt_diag
}

fn sample_mvn_zero_with_factor(
    rng: &mut rand::rngs::StdRng,
    factor: &DMatrix<f64>,
) -> DVector<f64> {
    let mut z = DVector::<f64>::zeros(factor.ncols());
    for i in 0..factor.ncols() {
        z[i] = StandardNormal.sample(rng);
    }
    factor * z
}

fn sample_mvn_zero(rng: &mut rand::rngs::StdRng, cov: &DMatrix<f64>) -> DVector<f64> {
    let factor = covariance_sampling_factor(cov);
    sample_mvn_zero_with_factor(rng, &factor)
}

fn simulate_gvm_toy_spec(
    spec: &MeasurementCombinationSpec,
    seed: u64,
) -> Result<MeasurementCombinationSpec> {
    simulate_gvm_toy_spec_with_solver(spec, MeasurementCombinationSolver::Numerical, seed)
}

fn simulate_gvm_toy_spec_with_solver(
    spec: &MeasurementCombinationSpec,
    solver: MeasurementCombinationSolver,
    seed: u64,
) -> Result<MeasurementCombinationSpec> {
    let context = build_measurement_toy_generation_context(spec, solver)?;
    Ok(simulate_gvm_toy_spec_from_context(&context, seed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::path::{Path, PathBuf};

    fn assert_json_close(actual: &serde_json::Value, expected: &serde_json::Value, path: &str) {
        match (actual, expected) {
            (serde_json::Value::Object(a), serde_json::Value::Object(e)) => {
                assert_eq!(a.len(), e.len(), "object key count mismatch at {path}");
                for (key, expected_value) in e {
                    let child = format!("{path}.{key}");
                    let actual_value =
                        a.get(key).unwrap_or_else(|| panic!("missing key at {child}"));
                    assert_json_close(actual_value, expected_value, &child);
                }
            }
            (serde_json::Value::Array(a), serde_json::Value::Array(e)) => {
                assert_eq!(a.len(), e.len(), "array length mismatch at {path}");
                for (idx, (actual_value, expected_value)) in a.iter().zip(e.iter()).enumerate() {
                    let child = format!("{path}[{idx}]");
                    assert_json_close(actual_value, expected_value, &child);
                }
            }
            (serde_json::Value::Number(a), serde_json::Value::Number(e)) => {
                let actual_value = a.as_f64().expect("actual number should convert to f64");
                let expected_value = e.as_f64().expect("expected number should convert to f64");
                let tol = 1e-4_f64.max(1e-4_f64 * actual_value.abs().max(expected_value.abs()));
                assert!(
                    (actual_value - expected_value).abs() <= tol,
                    "numeric mismatch at {path}: actual={actual_value} expected={expected_value} tol={tol}"
                );
            }
            _ => {
                assert_eq!(actual, expected, "value mismatch at {path}");
            }
        }
    }

    fn assert_json_matches_fixture(actual: &impl Serialize, fixture: &Path) {
        let actual: serde_json::Value =
            serde_json::to_value(actual).expect("actual value should serialize to JSON");
        let expected: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(fixture).expect("fixture should exist"))
                .expect("fixture should be valid JSON");
        assert_json_close(&actual, &expected, "$");
    }

    fn simple_spec() -> MeasurementCombinationSpec {
        MeasurementCombinationSpec {
            schema_version: MEASUREMENT_COMBINATION_SCHEMA_V0.to_string(),
            poi: "mu".to_string(),
            measurements: vec![
                MeasurementInput { name: "m1".to_string(), value: 1.0 },
                MeasurementInput { name: "m2".to_string(), value: 3.0 },
            ],
            stat_covariance: vec![vec![1.0, 0.0], vec![0.0, 4.0]],
            systematics: vec![],
        }
    }

    fn outlier_spec(epsilon: f64) -> MeasurementCombinationSpec {
        MeasurementCombinationSpec {
            schema_version: MEASUREMENT_COMBINATION_SCHEMA_V0.to_string(),
            poi: "mu".to_string(),
            measurements: vec![
                MeasurementInput { name: "m1".to_string(), value: 0.0 },
                MeasurementInput { name: "m2".to_string(), value: 0.1 },
                MeasurementInput { name: "m3".to_string(), value: 3.0 },
            ],
            stat_covariance: vec![
                vec![0.2 * 0.2, 0.0, 0.0],
                vec![0.0, 0.2 * 0.2, 0.0],
                vec![0.0, 0.0, 0.2 * 0.2],
            ],
            systematics: vec![SystematicSource {
                name: "new".to_string(),
                magnitudes: vec![0.0, 0.0, 0.2],
                corr: vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]],
                error_on_error: epsilon,
                aux_mean: 0.0,
            }],
        }
    }

    fn outlier_full_gvm_spec() -> MeasurementCombinationSpec {
        outlier_spec(0.30)
    }

    fn simulate_gvm_toy_spec_with_solver_uncached_reference(
        spec: &MeasurementCombinationSpec,
        solver: MeasurementCombinationSolver,
        seed: u64,
    ) -> Result<MeasurementCombinationSpec> {
        let prep = PreparedSpec::from_spec(spec)?;
        let (mu_hat, state) = match solver {
            MeasurementCombinationSolver::Numerical => {
                let objective = MeasurementCombineObjective::new(&prep);
                let bounds = prep.bounds();
                let fit = numerical_gvm_fit(&prep, &objective, &bounds)?;
                (fit.fit.parameters[0], fit.state)
            }
            MeasurementCombinationSolver::NumericalPaper
            | MeasurementCombinationSolver::AnalyticPerturbative
            | MeasurementCombinationSolver::Auto => {
                let objective = PaperMeasurementCombineObjective::new(&prep);
                let bounds = objective.bounds();
                let fit = numerical_paper_gvm_fit(&prep, &objective, &bounds, None)?;
                (fit.fit.parameters[0], fit.state)
            }
        };
        let workspace = build_bartlett_workspace(&prep, &state)?;
        let sigma2_unbiased = compute_unbiased_sigma2_estimates(&prep, &state, &workspace);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut y = prep.ones.clone() * mu_hat;
        y += sample_mvn_zero(&mut rng, &prep.base_cov);

        for (s_idx, syst) in prep.systematics.iter().enumerate() {
            let cov = syst.corr_reg.scale(sigma2_unbiased[s_idx]);
            let theta_original = sample_mvn_zero(&mut rng, &cov);
            let diag = DMatrix::from_diagonal(&syst.magnitudes);
            y += diag * theta_original;
        }

        let mut toy = spec.clone();
        for (measurement, value) in toy.measurements.iter_mut().zip(y.iter()) {
            measurement.value = *value;
        }
        Ok(toy)
    }

    fn trivial_rank1_spec(epsilon: f64) -> MeasurementCombinationSpec {
        MeasurementCombinationSpec {
            schema_version: MEASUREMENT_COMBINATION_SCHEMA_V0.to_string(),
            poi: "mu".to_string(),
            measurements: vec![
                MeasurementInput { name: "m1".to_string(), value: 1.0 },
                MeasurementInput { name: "m2".to_string(), value: 1.3 },
            ],
            stat_covariance: vec![vec![0.2 * 0.2, 0.0], vec![0.0, 0.25 * 0.25]],
            systematics: vec![SystematicSource {
                name: "scale".to_string(),
                magnitudes: vec![0.3, 0.3],
                corr: vec![vec![1.0, 1.0], vec![1.0, 1.0]],
                error_on_error: epsilon,
                aux_mean: 0.0,
            }],
        }
    }

    fn finite_difference_gradient<O: ObjectiveFunction>(
        objective: &O,
        params: &[f64],
    ) -> Result<Vec<f64>> {
        let mut grad = vec![0.0; params.len()];
        for i in 0..params.len() {
            let eps = 1e-7 * params[i].abs().max(1.0);
            let mut plus = params.to_vec();
            plus[i] += eps;
            let mut minus = params.to_vec();
            minus[i] -= eps;
            let f_plus = objective.eval(&plus)?;
            let f_minus = objective.eval(&minus)?;
            grad[i] = (f_plus - f_minus) / (2.0 * eps);
        }
        Ok(grad)
    }

    fn synthetic_fast_path_spec(
        n_measurements: usize,
        n_systematics: usize,
        epsilon: f64,
    ) -> MeasurementCombinationSpec {
        let measurements = (0..n_measurements)
            .map(|idx| MeasurementInput {
                name: format!("m{}", idx + 1),
                value: 170.0 + idx as f64 * 0.05,
            })
            .collect::<Vec<_>>();
        let stat_covariance = (0..n_measurements)
            .map(|i| {
                (0..n_measurements)
                    .map(|j| {
                        if i == j {
                            let sigma = 0.20 + 0.01 * i as f64;
                            sigma * sigma
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let systematics = (0..n_systematics)
            .map(|s_idx| SystematicSource {
                name: format!("sys_{}", s_idx + 1),
                magnitudes: (0..n_measurements)
                    .map(|m_idx| 0.02 + 0.002 * s_idx as f64 + 0.001 * m_idx as f64)
                    .collect::<Vec<_>>(),
                corr: (0..n_measurements)
                    .map(|i| {
                        (0..n_measurements)
                            .map(|j| if i == j { 1.0 } else { 0.0 })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>(),
                error_on_error: epsilon,
                aux_mean: 0.0,
            })
            .collect::<Vec<_>>();

        MeasurementCombinationSpec {
            schema_version: MEASUREMENT_COMBINATION_SCHEMA_V0.to_string(),
            poi: "mu".to_string(),
            measurements,
            stat_covariance,
            systematics,
        }
    }

    fn synthetic_bartlett_state(prep: &PreparedSpec, sigma2: f64) -> ProfiledGvmState {
        ProfiledGvmState {
            theta_original: prep.systematics.iter().map(|_| DVector::zeros(prep.y.len())).collect(),
            profiled_variance_scales: vec![sigma2; prep.systematics.len()],
            theta_l2_norms: vec![0.0; prep.systematics.len()],
        }
    }

    fn bartlett_workspace_scalars(
        prep: &PreparedSpec,
        state: &ProfiledGvmState,
        workspace: &BartlettWorkspace,
    ) -> (f64, f64, Vec<f64>) {
        let sigma2_unbiased = compute_unbiased_sigma2_estimates(prep, state, workspace);
        let mut b_mu_theta = 0.0;
        let mut b_tilde_theta = 0.0;
        for (s_idx, syst) in prep.systematics.iter().enumerate() {
            let sigma2 = state.profiled_variance_scales[s_idx];
            let eps2 = syst.error_on_error * syst.error_on_error;
            let j_block = &workspace.j_blocks[s_idx];
            let j_tilde_block = &workspace.j_tilde_blocks[s_idx];
            let rho_inv = &syst.corr_precision;

            let t1 = trace_product(j_block, rho_inv);
            let t2 = trace_square_of_product(j_block, rho_inv);
            let t1_tilde = trace_product(j_tilde_block, rho_inv);
            let t2_tilde = trace_square_of_product(j_tilde_block, rho_inv);

            b_mu_theta += (4.0 * t1 / sigma2 + (t1 * t1 - 2.0 * t2) / (sigma2 * sigma2)) * eps2;
            b_tilde_theta += (4.0 * t1_tilde / sigma2
                + (t1_tilde * t1_tilde - 2.0 * t2_tilde) / (sigma2 * sigma2))
                * eps2;
        }
        (b_mu_theta, b_tilde_theta, sigma2_unbiased)
    }

    fn reduced_literature_spec(error_on_error: f64) -> MeasurementCombinationSpec {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_gvm_literature_reduced.json");
        let mut spec: MeasurementCombinationSpec =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        spec.systematics[0].error_on_error = error_on_error;
        spec
    }

    fn full_literature_spec(error_on_error_for: Option<(&str, f64)>) -> MeasurementCombinationSpec {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_gvm_topmass_full.json");
        let mut spec: MeasurementCombinationSpec =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        if let Some((name, epsilon)) = error_on_error_for {
            let syst = spec
                .systematics
                .iter_mut()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("missing systematic {name}"));
            syst.error_on_error = epsilon;
        }
        spec
    }

    fn calibration_outlier_fixture_spec() -> MeasurementCombinationSpec {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_calibration_outlier_input.json");
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn outlier_scenario_study_spec() -> MeasurementCombinationScenarioStudySpec {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_outlier_scenarios.json");
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn full_literature_scenario_study_spec() -> MeasurementCombinationScenarioStudySpec {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_topmass_full_scenarios.json");
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn full_literature_scenario_subset(names: &[&str]) -> MeasurementCombinationScenarioStudySpec {
        let mut spec = full_literature_scenario_study_spec();
        spec.scenarios.retain(|scenario| names.iter().any(|name| *name == scenario.name));
        assert_eq!(
            spec.scenarios.len(),
            names.len(),
            "expected requested full-literature scenario subset to exist"
        );
        spec
    }

    fn build_full_literature_portfolio_from_campaign_reports(
        multi_report: &MeasurementCombinationCalibrationCampaignReport,
        bjes_report: &MeasurementCombinationCalibrationCampaignReport,
    ) -> MeasurementCombinationCalibrationCampaignPortfolioReport {
        let multi_digest = summarize_measurement_combination_calibration_campaign(multi_report)
            .expect("full literature multi-scenario digest should build");
        let bjes_digest = summarize_measurement_combination_calibration_campaign(bjes_report)
            .expect("full literature bjes-only digest should build");

        let combo_brief = build_measurement_combination_calibration_campaign_brief(&[
            ("full_multi".to_string(), multi_digest.clone()),
            ("bjes_only".to_string(), bjes_digest.clone()),
        ])
        .expect("combo brief should build");
        let multi_only_brief = build_measurement_combination_calibration_campaign_brief(&[(
            "full_multi".to_string(),
            multi_digest,
        )])
        .expect("multi-only brief should build");
        let bjes_only_brief = build_measurement_combination_calibration_campaign_brief(&[(
            "bjes_only".to_string(),
            bjes_digest,
        )])
        .expect("bjes-only brief should build");

        let combo_family_report =
            build_measurement_combination_calibration_campaign_family_report(&[
                ("full_vs_bjes".to_string(), combo_brief),
                ("multi_only".to_string(), multi_only_brief),
            ])
            .expect("combo family report should build");
        let bjes_family_report = build_measurement_combination_calibration_campaign_family_report(
            &[("bjes_only_family".to_string(), bjes_only_brief)],
        )
        .expect("bjes family report should build");

        let combo_matrix =
            build_measurement_combination_calibration_campaign_family_matrix(&combo_family_report)
                .expect("combo family matrix should build");
        let bjes_matrix =
            build_measurement_combination_calibration_campaign_family_matrix(&bjes_family_report)
                .expect("bjes family matrix should build");

        build_measurement_combination_calibration_campaign_portfolio_report(&[
            ("combo_portfolio".to_string(), combo_matrix),
            ("bjes_only_portfolio".to_string(), bjes_matrix),
        ])
        .expect("portfolio should build")
    }

    fn full_literature_calibration_report_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_calibration_topmass_full_bjes_report.json");
        path
    }

    fn full_literature_calibration_campaign_report_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push(
            "tests/fixtures/measurement_combine_calibration_topmass_full_campaign_report.json",
        );
        path
    }

    fn calibration_outlier_study_report_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_calibration_outlier_study_report.json");
        path
    }

    fn scenario_outlier_study_report_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_outlier_scenario_study_report.json");
        path
    }

    fn scenario_outlier_calibration_campaign_report_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_outlier_calibration_campaign_report.json");
        path
    }

    fn scenario_outlier_calibration_campaign_summary_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_outlier_calibration_campaign_summary.json");
        path
    }

    fn full_literature_calibration_campaign_summary_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push(
            "tests/fixtures/measurement_combine_calibration_topmass_full_campaign_summary.json",
        );
        path
    }

    fn scenario_outlier_calibration_campaign_summary_markdown_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_outlier_calibration_campaign_summary.md");
        path
    }

    fn full_literature_calibration_campaign_summary_markdown_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push(
            "tests/fixtures/measurement_combine_calibration_topmass_full_campaign_summary.md",
        );
        path
    }

    fn scenario_solver_parity_report_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_solver_parity_scenario_study_report.json");
        path
    }

    fn scenario_solver_parity_markdown_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_solver_parity_scenario_study_report.md");
        path
    }

    fn scenario_solver_parity_digest_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_solver_parity_scenario_study_digest.json");
        path
    }

    fn scenario_solver_parity_digest_markdown_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_solver_parity_scenario_study_digest.md");
        path
    }

    fn calibration_campaign_solver_parity_report_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push(
            "tests/fixtures/measurement_combine_solver_parity_calibration_campaign_report.json",
        );
        path
    }

    fn calibration_campaign_solver_parity_markdown_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push(
            "tests/fixtures/measurement_combine_solver_parity_calibration_campaign_report.md",
        );
        path
    }

    fn calibration_campaign_solver_parity_digest_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push(
            "tests/fixtures/measurement_combine_solver_parity_calibration_campaign_digest.json",
        );
        path
    }

    fn calibration_campaign_solver_parity_digest_markdown_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push(
            "tests/fixtures/measurement_combine_solver_parity_calibration_campaign_digest.md",
        );
        path
    }

    fn calibration_campaign_brief_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_calibration_campaign_brief.json");
        path
    }

    fn calibration_campaign_brief_markdown_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_calibration_campaign_brief.md");
        path
    }

    fn calibration_campaign_topmass_only_brief_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_calibration_topmass_only_brief.json");
        path
    }

    fn calibration_campaign_family_report_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_calibration_campaign_family_report.json");
        path
    }

    fn calibration_campaign_family_report_markdown_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_calibration_campaign_family_report.md");
        path
    }

    fn calibration_campaign_family_matrix_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_calibration_campaign_family_matrix.json");
        path
    }

    fn calibration_campaign_family_matrix_markdown_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_calibration_campaign_family_matrix.md");
        path
    }

    fn calibration_campaign_topmass_only_family_report_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_calibration_topmass_only_family_report.json");
        path
    }

    fn calibration_campaign_topmass_only_family_matrix_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_calibration_topmass_only_family_matrix.json");
        path
    }

    fn calibration_campaign_portfolio_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_calibration_campaign_portfolio.json");
        path
    }

    fn calibration_campaign_portfolio_markdown_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_calibration_campaign_portfolio.md");
        path
    }

    fn calibration_campaign_portfolio_stability_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push(
            "tests/fixtures/measurement_combine_calibration_campaign_portfolio_stability.json",
        );
        path
    }

    fn calibration_campaign_portfolio_stability_markdown_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_calibration_campaign_portfolio_stability.md");
        path
    }

    fn numerical_paper_multistart_family_report_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push(
            "tests/fixtures/measurement_combine_numerical_paper_multistart_family_report.json",
        );
        path
    }

    fn numerical_paper_multistart_family_report_markdown_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_numerical_paper_multistart_family_report.md");
        path
    }

    fn numerical_paper_multistart_mixed_family_report_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push(
            "tests/fixtures/measurement_combine_numerical_paper_multistart_mixed_family_report.json",
        );
        path
    }

    fn numerical_paper_multistart_mixed_family_report_markdown_fixture() -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push(
            "tests/fixtures/measurement_combine_numerical_paper_multistart_mixed_family_report.md",
        );
        path
    }

    fn load_calibration_campaign_report(
        path: &PathBuf,
    ) -> MeasurementCombinationCalibrationCampaignReport {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn load_calibration_campaign_summary(
        path: &PathBuf,
    ) -> MeasurementCombinationCalibrationCampaignDigest {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn load_calibration_campaign_brief(
        path: &PathBuf,
    ) -> MeasurementCombinationCalibrationCampaignBrief {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn load_calibration_campaign_family_report(
        path: &PathBuf,
    ) -> MeasurementCombinationCalibrationCampaignFamilyReport {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn load_calibration_campaign_family_matrix(
        path: &PathBuf,
    ) -> MeasurementCombinationCalibrationCampaignFamilyMatrix {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn load_calibration_campaign_portfolio(
        path: &PathBuf,
    ) -> MeasurementCombinationCalibrationCampaignPortfolioReport {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct NumericalPaperMultiStartReportEntry {
        label: String,
        mu_shift: f64,
        nuisance_scale: f64,
        phase: f64,
        mu_abs_diff: f64,
        fval_abs_diff: f64,
        ci_lower_abs_diff: f64,
        ci_upper_abs_diff: f64,
        max_ci_abs_diff: f64,
        within_tolerance: bool,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct NumericalPaperMultiStartTierReport {
        label: String,
        n_measurements: usize,
        n_systematics: usize,
        n_starts: usize,
        ci_level: f64,
        epsilon: f64,
        baseline_mu_hat: f64,
        baseline_fval: f64,
        baseline_ci_lower: f64,
        baseline_ci_upper: f64,
        mu_tol: f64,
        fval_tol: f64,
        ci_tol: f64,
        max_mu_abs_diff: f64,
        max_fval_abs_diff: f64,
        max_ci_abs_diff: f64,
        worst_start_label: String,
        all_within_tolerance: bool,
        starts: Vec<NumericalPaperMultiStartReportEntry>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct NumericalPaperMultiStartFamilyReportAggregate {
        n_tiers: usize,
        tier_labels: Vec<String>,
        all_tiers_within_tolerance: bool,
        worst_mu_tier: String,
        worst_mu_abs_diff: f64,
        worst_fval_tier: String,
        worst_fval_abs_diff: f64,
        worst_ci_tier: String,
        worst_ci_abs_diff: f64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct NumericalPaperMultiStartFamilyReport {
        schema_version: String,
        source_solver: String,
        ci_level: f64,
        epsilon: f64,
        stability: String,
        tiers: Vec<NumericalPaperMultiStartTierReport>,
        aggregate: NumericalPaperMultiStartFamilyReportAggregate,
    }

    fn build_numerical_paper_multistart_report_for_spec(
        label: impl Into<String>,
        spec: MeasurementCombinationSpec,
        n_starts: usize,
        ci_level: f64,
        epsilon: f64,
        mu_tol: f64,
        fval_tol: f64,
        ci_tol: f64,
    ) -> NumericalPaperMultiStartTierReport {
        let n_measurements = spec.measurements.len();
        let n_systematics = spec.systematics.len();
        let prep = PreparedSpec::from_spec(&spec).unwrap();
        let objective = PaperMeasurementCombineObjective::new(&prep);
        let bounds = objective.bounds();
        let baseline = numerical_paper_gvm_fit(&prep, &objective, &bounds, None).unwrap();
        let chi2_level = ChiSquared::new(1.0).unwrap().inverse_cdf(ci_level);
        let (baseline_lo, baseline_hi) = profile_ci_mu_paper(
            &objective,
            &bounds,
            &baseline.fit.parameters,
            baseline.fit.fval,
            chi2_level,
            None,
        )
        .unwrap();

        let cold = objective.init_params().unwrap();
        let sigma_guess = prep.fixed_sigma_guess;
        let mut starts = default_multistart_offsets(sigma_guess);
        starts.truncate(n_starts);

        let mut entries = Vec::with_capacity(starts.len());
        for (idx, &(mu_shift, nuisance_scale, phase)) in starts.iter().enumerate() {
            let init = deterministic_paper_multistart_init(
                &cold,
                &bounds,
                mu_shift,
                nuisance_scale,
                phase,
            );
            let fit = fit_numerical_paper_from_init(&objective, &bounds, &init).unwrap();
            let (lo, hi) = profile_ci_mu_paper(
                &objective,
                &bounds,
                &fit.parameters,
                fit.fval,
                chi2_level,
                None,
            )
            .unwrap();
            let mu_abs_diff = (fit.parameters[0] - baseline.fit.parameters[0]).abs();
            let fval_abs_diff = (fit.fval - baseline.fit.fval).abs();
            let ci_lower_abs_diff = (lo - baseline_lo).abs();
            let ci_upper_abs_diff = (hi - baseline_hi).abs();
            let max_ci_abs_diff = ci_lower_abs_diff.max(ci_upper_abs_diff);
            entries.push(NumericalPaperMultiStartReportEntry {
                label: format!("start_{}", idx + 1),
                mu_shift,
                nuisance_scale,
                phase,
                mu_abs_diff,
                fval_abs_diff,
                ci_lower_abs_diff,
                ci_upper_abs_diff,
                max_ci_abs_diff,
                within_tolerance: mu_abs_diff <= mu_tol
                    && fval_abs_diff <= fval_tol
                    && ci_lower_abs_diff <= ci_tol
                    && ci_upper_abs_diff <= ci_tol,
            });
        }

        let max_mu_abs_diff = entries.iter().map(|entry| entry.mu_abs_diff).fold(0.0_f64, f64::max);
        let max_fval_abs_diff =
            entries.iter().map(|entry| entry.fval_abs_diff).fold(0.0_f64, f64::max);
        let max_ci_abs_diff =
            entries.iter().map(|entry| entry.max_ci_abs_diff).fold(0.0_f64, f64::max);
        let worst_start = entries
            .iter()
            .max_by(|lhs, rhs| {
                let lhs_score = (lhs.mu_abs_diff / mu_tol.max(1e-30))
                    .max(lhs.fval_abs_diff / fval_tol.max(1e-30))
                    .max(lhs.max_ci_abs_diff / ci_tol.max(1e-30));
                let rhs_score = (rhs.mu_abs_diff / mu_tol.max(1e-30))
                    .max(rhs.fval_abs_diff / fval_tol.max(1e-30))
                    .max(rhs.max_ci_abs_diff / ci_tol.max(1e-30));
                lhs_score.partial_cmp(&rhs_score).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|entry| entry.label.clone())
            .unwrap_or_else(|| "none".to_string());

        NumericalPaperMultiStartTierReport {
            label: label.into(),
            n_measurements,
            n_systematics,
            n_starts,
            ci_level,
            epsilon,
            baseline_mu_hat: baseline.fit.parameters[0],
            baseline_fval: baseline.fit.fval,
            baseline_ci_lower: baseline_lo,
            baseline_ci_upper: baseline_hi,
            mu_tol,
            fval_tol,
            ci_tol,
            max_mu_abs_diff,
            max_fval_abs_diff,
            max_ci_abs_diff,
            worst_start_label: worst_start,
            all_within_tolerance: entries.iter().all(|entry| entry.within_tolerance),
            starts: entries,
        }
    }

    fn build_numerical_paper_multistart_tier_report(
        n_measurements: usize,
        n_systematics: usize,
        n_starts: usize,
        ci_level: f64,
        epsilon: f64,
        mu_tol: f64,
        fval_tol: f64,
        ci_tol: f64,
    ) -> NumericalPaperMultiStartTierReport {
        build_numerical_paper_multistart_report_for_spec(
            format!("synthetic_{}x{}", n_measurements, n_systematics),
            synthetic_fast_path_spec(n_measurements, n_systematics, epsilon),
            n_starts,
            ci_level,
            epsilon,
            mu_tol,
            fval_tol,
            ci_tol,
        )
    }

    fn build_numerical_paper_multistart_family_report() -> NumericalPaperMultiStartFamilyReport {
        let tiers = vec![
            build_numerical_paper_multistart_tier_report(32, 24, 3, 0.68, 0.05, 3e-7, 1e-8, 1e-6),
            build_numerical_paper_multistart_tier_report(64, 48, 2, 0.68, 0.05, 1e-6, 1e-7, 5e-6),
            build_numerical_paper_multistart_tier_report(96, 64, 1, 0.68, 0.05, 3e-6, 3e-7, 1e-5),
        ];

        let worst_mu = tiers
            .iter()
            .max_by(|lhs, rhs| {
                lhs.max_mu_abs_diff
                    .partial_cmp(&rhs.max_mu_abs_diff)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        let worst_fval = tiers
            .iter()
            .max_by(|lhs, rhs| {
                lhs.max_fval_abs_diff
                    .partial_cmp(&rhs.max_fval_abs_diff)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        let worst_ci = tiers
            .iter()
            .max_by(|lhs, rhs| {
                lhs.max_ci_abs_diff
                    .partial_cmp(&rhs.max_ci_abs_diff)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        NumericalPaperMultiStartFamilyReport {
            schema_version:
                "nextstat_measurement_combination_numerical_paper_multistart_family_report_v0"
                    .to_string(),
            source_solver: "numerical-paper".to_string(),
            ci_level: 0.68,
            epsilon: 0.05,
            stability: GVM_STABILITY_RESEARCH_GRADE.to_string(),
            aggregate: NumericalPaperMultiStartFamilyReportAggregate {
                n_tiers: tiers.len(),
                tier_labels: tiers.iter().map(|tier| tier.label.clone()).collect(),
                all_tiers_within_tolerance: tiers.iter().all(|tier| tier.all_within_tolerance),
                worst_mu_tier: worst_mu.label.clone(),
                worst_mu_abs_diff: worst_mu.max_mu_abs_diff,
                worst_fval_tier: worst_fval.label.clone(),
                worst_fval_abs_diff: worst_fval.max_fval_abs_diff,
                worst_ci_tier: worst_ci.label.clone(),
                worst_ci_abs_diff: worst_ci.max_ci_abs_diff,
            },
            tiers,
        }
    }

    fn build_numerical_paper_multistart_mixed_family_report() -> NumericalPaperMultiStartFamilyReport
    {
        let tiers = vec![
            build_numerical_paper_multistart_report_for_spec(
                "literature_topmass_bjes_0p05",
                full_literature_spec(Some(("b-JES", 0.05))),
                3,
                0.68,
                0.05,
                5e-7,
                1e-8,
                5e-6,
            ),
            build_numerical_paper_multistart_tier_report(32, 24, 3, 0.68, 0.05, 3e-7, 1e-8, 1e-6),
            build_numerical_paper_multistart_tier_report(64, 48, 2, 0.68, 0.05, 1e-6, 1e-7, 5e-6),
            build_numerical_paper_multistart_tier_report(96, 64, 1, 0.68, 0.05, 3e-6, 3e-7, 1e-5),
            build_numerical_paper_multistart_tier_report(128, 96, 1, 0.68, 0.05, 1e-5, 1e-6, 3e-5),
        ];

        let worst_mu = tiers
            .iter()
            .max_by(|lhs, rhs| {
                lhs.max_mu_abs_diff
                    .partial_cmp(&rhs.max_mu_abs_diff)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        let worst_fval = tiers
            .iter()
            .max_by(|lhs, rhs| {
                lhs.max_fval_abs_diff
                    .partial_cmp(&rhs.max_fval_abs_diff)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        let worst_ci = tiers
            .iter()
            .max_by(|lhs, rhs| {
                lhs.max_ci_abs_diff
                    .partial_cmp(&rhs.max_ci_abs_diff)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        NumericalPaperMultiStartFamilyReport {
            schema_version:
                "nextstat_measurement_combination_numerical_paper_multistart_family_report_v0"
                    .to_string(),
            source_solver: "numerical-paper".to_string(),
            ci_level: 0.68,
            epsilon: 0.05,
            stability: GVM_STABILITY_RESEARCH_GRADE.to_string(),
            aggregate: NumericalPaperMultiStartFamilyReportAggregate {
                n_tiers: tiers.len(),
                tier_labels: tiers.iter().map(|tier| tier.label.clone()).collect(),
                all_tiers_within_tolerance: tiers.iter().all(|tier| tier.all_within_tolerance),
                worst_mu_tier: worst_mu.label.clone(),
                worst_mu_abs_diff: worst_mu.max_mu_abs_diff,
                worst_fval_tier: worst_fval.label.clone(),
                worst_fval_abs_diff: worst_fval.max_fval_abs_diff,
                worst_ci_tier: worst_ci.label.clone(),
                worst_ci_abs_diff: worst_ci.max_ci_abs_diff,
            },
            tiers,
        }
    }

    fn render_numerical_paper_multistart_family_report_markdown(
        report: &NumericalPaperMultiStartFamilyReport,
    ) -> String {
        let mut out = String::new();
        out.push_str("# NumericalPaper Multi-Start Family Stability\n\n");
        out.push_str(&format!(
            "- Solver: `{}`\n- CI level: `{:.2}`\n- Epsilon: `{:.2}`\n- All tiers within tolerance: `{}`\n\n",
            report.source_solver,
            report.ci_level,
            report.epsilon,
            if report.aggregate.all_tiers_within_tolerance { "yes" } else { "no" }
        ));
        out.push_str("| Tier | Starts | Max |mu| drift | Max fval drift | Max CI drift | Worst start | Within tolerance |\n");
        out.push_str("| --- | ---: | ---: | ---: | ---: | --- | --- |\n");
        for tier in &report.tiers {
            out.push_str(&format!(
                "| {} | {} | {:.9e} | {:.9e} | {:.9e} | {} | {} |\n",
                tier.label,
                tier.n_starts,
                tier.max_mu_abs_diff,
                tier.max_fval_abs_diff,
                tier.max_ci_abs_diff,
                tier.worst_start_label,
                if tier.all_within_tolerance { "yes" } else { "no" }
            ));
        }
        out.push_str("\n## Worst Starts\n\n");
        out.push_str("| Tier | Start | mu_shift | nuisance_scale | phase | |mu| drift | fval drift | max CI drift |\n");
        out.push_str("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for tier in &report.tiers {
            let worst = tier
                .starts
                .iter()
                .find(|entry| entry.label == tier.worst_start_label)
                .expect("worst start should exist");
            out.push_str(&format!(
                "| {} | {} | {:.6} | {:.6} | {:.6} | {:.9e} | {:.9e} | {:.9e} |\n",
                tier.label,
                worst.label,
                worst.mu_shift,
                worst.nuisance_scale,
                worst.phase,
                worst.mu_abs_diff,
                worst.fval_abs_diff,
                worst.max_ci_abs_diff
            ));
        }
        out
    }

    #[test]
    fn validate_rejects_negative_error_on_error() {
        let mut spec = simple_spec();
        spec.systematics.push(SystematicSource {
            name: "s1".to_string(),
            magnitudes: vec![0.1, 0.2],
            corr: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            error_on_error: -0.1,
            aux_mean: 0.0,
        });
        let err = spec.validate().unwrap_err().to_string();
        assert!(err.contains("error_on_error"));
    }

    #[test]
    fn validate_rejects_non_symmetric_corr() {
        let mut spec = simple_spec();
        spec.systematics.push(SystematicSource {
            name: "s1".to_string(),
            magnitudes: vec![0.1, 0.2],
            corr: vec![vec![1.0, 0.2], vec![0.0, 1.0]],
            error_on_error: 0.0,
            aux_mean: 0.0,
        });
        let err = spec.validate().unwrap_err().to_string();
        assert!(err.contains("symmetric"));
    }

    #[test]
    fn validate_rejects_non_symmetric_stat_covariance() {
        let mut spec = simple_spec();
        spec.stat_covariance = vec![vec![1.0, 0.2], vec![0.0, 4.0]];
        let err = spec.validate().unwrap_err().to_string();
        assert!(err.contains("stat_covariance"));
        assert!(err.contains("symmetric"));
    }

    #[test]
    fn validate_rejects_mismatched_systematic_magnitude_length() {
        let mut spec = simple_spec();
        spec.systematics.push(SystematicSource {
            name: "s1".to_string(),
            magnitudes: vec![0.1],
            corr: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            error_on_error: 0.0,
            aux_mean: 0.0,
        });
        let err = spec.validate().unwrap_err().to_string();
        assert!(err.contains("magnitudes length mismatch"));
    }

    #[test]
    fn validate_rejects_non_unit_corr_diagonal() {
        let mut spec = simple_spec();
        spec.systematics.push(SystematicSource {
            name: "s1".to_string(),
            magnitudes: vec![0.1, 0.2],
            corr: vec![vec![0.9, 0.0], vec![0.0, 1.0]],
            error_on_error: 0.0,
            aux_mean: 0.0,
        });
        let err = spec.validate().unwrap_err().to_string();
        assert!(err.contains("diagonal must equal 1"));
    }

    #[test]
    fn combine_single_measurement_returns_measurement_value() {
        let spec = MeasurementCombinationSpec {
            schema_version: MEASUREMENT_COMBINATION_SCHEMA_V0.to_string(),
            poi: "mu".to_string(),
            measurements: vec![MeasurementInput { name: "m1".to_string(), value: 2.5 }],
            stat_covariance: vec![vec![0.09]],
            systematics: vec![],
        };
        let out = combine_measurements(&spec, 0.68).unwrap();
        assert_relative_eq!(out.mu_hat, 2.5, epsilon = 1e-12);
        assert_eq!(out.goodness_of_fit.df, 0);
        assert!(out.goodness_of_fit.p_value.is_none());
    }

    #[test]
    fn combine_independent_gaussians_matches_inverse_variance_pooling() {
        let out = combine_measurements(&simple_spec(), 0.68).unwrap();
        assert_relative_eq!(out.mu_hat, 1.4, epsilon = 1e-12);
        assert_relative_eq!(
            out.confidence_interval.sigma,
            (1.0_f64 / 1.25_f64).sqrt(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn gvm_with_zero_error_on_error_matches_closed_form_blue_with_same_covariance() {
        let mut spec = simple_spec();
        spec.systematics.push(SystematicSource {
            name: "s1".to_string(),
            magnitudes: vec![0.3, 0.5],
            corr: vec![vec![1.0, 0.2], vec![0.2, 1.0]],
            error_on_error: 0.0,
            aux_mean: 0.0,
        });

        let out = combine_measurements(&spec, 0.68).unwrap();
        let prep = PreparedSpec::from_spec(&spec).unwrap();
        let expected = fixed_variance_solution(&prep).unwrap();

        assert_relative_eq!(out.mu_hat, expected.mu_hat, epsilon = 1e-12);
        assert_relative_eq!(out.confidence_interval.sigma, expected.sigma, epsilon = 1e-12);
        assert_relative_eq!(out.goodness_of_fit.chi2, expected.chi2, epsilon = 1e-12);
    }

    #[test]
    fn zero_magnitude_systematic_has_no_effect() {
        let baseline = combine_measurements(&simple_spec(), 0.68).unwrap();
        let mut spec = simple_spec();
        spec.systematics.push(SystematicSource {
            name: "zero".to_string(),
            magnitudes: vec![0.0, 0.0],
            corr: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            error_on_error: 0.0,
            aux_mean: 0.0,
        });
        let out = combine_measurements(&spec, 0.68).unwrap();

        assert_relative_eq!(out.mu_hat, baseline.mu_hat, epsilon = 1e-12);
        assert_relative_eq!(
            out.confidence_interval.sigma,
            baseline.confidence_interval.sigma,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            out.goodness_of_fit.chi2,
            baseline.goodness_of_fit.chi2,
            epsilon = 1e-12
        );
    }

    #[test]
    fn cross_source_decoupling_matches_effective_variance_inflation() {
        let spec = MeasurementCombinationSpec {
            schema_version: MEASUREMENT_COMBINATION_SCHEMA_V0.to_string(),
            poi: "mu".to_string(),
            measurements: vec![
                MeasurementInput { name: "m1".to_string(), value: 1.0 },
                MeasurementInput { name: "m2".to_string(), value: 3.0 },
            ],
            stat_covariance: vec![vec![1.0, 0.0], vec![0.0, 4.0]],
            systematics: vec![
                SystematicSource {
                    name: "s1".to_string(),
                    magnitudes: vec![0.3, 0.0],
                    corr: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                    error_on_error: 0.0,
                    aux_mean: 0.0,
                },
                SystematicSource {
                    name: "s2".to_string(),
                    magnitudes: vec![0.0, 0.4],
                    corr: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                    error_on_error: 0.0,
                    aux_mean: 0.0,
                },
            ],
        };
        let equivalent = MeasurementCombinationSpec {
            schema_version: MEASUREMENT_COMBINATION_SCHEMA_V0.to_string(),
            poi: "mu".to_string(),
            measurements: spec.measurements.clone(),
            stat_covariance: vec![vec![1.0 + 0.3 * 0.3, 0.0], vec![0.0, 4.0 + 0.4 * 0.4]],
            systematics: vec![],
        };

        let out = combine_measurements(&spec, 0.68).unwrap();
        let expected = combine_measurements(&equivalent, 0.68).unwrap();

        assert_relative_eq!(out.mu_hat, expected.mu_hat, epsilon = 1e-12);
        assert_relative_eq!(
            out.confidence_interval.sigma,
            expected.confidence_interval.sigma,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            out.goodness_of_fit.chi2,
            expected.goodness_of_fit.chi2,
            epsilon = 1e-12
        );
    }

    #[test]
    fn combine_with_positive_error_on_error_returns_finite_result() {
        let out = combine_measurements(&outlier_spec(0.30), 0.68).unwrap();
        assert!(out.mu_hat.is_finite());
        assert!(out.converged);
        assert_eq!(out.diagnostics.profiled_variance_scales.len(), 1);
        assert!(out.diagnostics.profiled_variance_scales[0].is_finite());
        assert!(out.diagnostics.profiled_variance_scales[0] > 0.0);
        assert!(out.diagnostics.bartlett.supported);
        assert!(out.diagnostics.bartlett.b_mu.unwrap().is_finite());
    }

    #[test]
    fn increasing_error_on_error_widens_interval_and_reduces_outlier_pull() {
        let fixed = combine_measurements(&outlier_spec(0.0), 0.68).unwrap();
        let gvm = combine_measurements(&outlier_spec(0.50), 0.68).unwrap();
        let fixed_width = fixed.confidence_interval.upper - fixed.confidence_interval.lower;
        let gvm_width = gvm.confidence_interval.upper - gvm.confidence_interval.lower;
        assert!(
            gvm_width > fixed_width,
            "expected wider interval: fixed={fixed_width} gvm={gvm_width}"
        );
        assert!(
            gvm.mu_hat.abs() < fixed.mu_hat.abs(),
            "expected GVM estimate to move toward the inlier cluster: fixed={} gvm={}",
            fixed.mu_hat,
            gvm.mu_hat
        );
    }

    #[test]
    fn trivial_correlation_limits_match_closed_form_blue() {
        for corr in [
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
            vec![vec![1.0, -1.0], vec![-1.0, 1.0]],
        ] {
            let spec = MeasurementCombinationSpec {
                schema_version: MEASUREMENT_COMBINATION_SCHEMA_V0.to_string(),
                poi: "mu".to_string(),
                measurements: vec![
                    MeasurementInput { name: "m1".to_string(), value: 1.0 },
                    MeasurementInput { name: "m2".to_string(), value: 1.3 },
                ],
                stat_covariance: vec![vec![0.2 * 0.2, 0.0], vec![0.0, 0.25 * 0.25]],
                systematics: vec![SystematicSource {
                    name: "scale".to_string(),
                    magnitudes: vec![0.3, 0.3],
                    corr,
                    error_on_error: 0.0,
                    aux_mean: 0.0,
                }],
            };

            let out = combine_measurements(&spec, 0.68).unwrap();
            let prep = PreparedSpec::from_spec(&spec).unwrap();
            let expected = fixed_variance_solution(&prep).unwrap();

            assert_relative_eq!(out.mu_hat, expected.mu_hat, epsilon = 1e-12);
            assert_relative_eq!(out.confidence_interval.sigma, expected.sigma, epsilon = 1e-12);
            assert_relative_eq!(out.goodness_of_fit.chi2, expected.chi2, epsilon = 1e-12);
        }
    }

    #[test]
    fn epsilon_sweep_preserves_locked_trend_artifact() {
        let epsilons = [0.0, 0.1, 0.3, 0.5];
        let results = epsilons
            .iter()
            .map(|epsilon| combine_measurements(&outlier_spec(*epsilon), 0.68).unwrap())
            .collect::<Vec<_>>();
        let widths = results
            .iter()
            .map(|result| result.confidence_interval.upper - result.confidence_interval.lower)
            .collect::<Vec<_>>();
        let outlier_distances =
            results.iter().map(|result| result.mu_hat.abs()).collect::<Vec<_>>();

        assert!(widths[1] > widths[0], "expected initial interval widening: {widths:?}");
        assert!(widths[2] > widths[1], "expected continued interval widening: {widths:?}");
        assert!(
            widths[3] > widths[0],
            "expected final interval to remain inflated relative to epsilon=0: {widths:?}"
        );
        assert!(
            (widths[3] - widths[2]).abs() <= 2e-3,
            "expected high-epsilon interval change to stay within the locked artifact tolerance: {widths:?}"
        );
        for pair in outlier_distances.windows(2) {
            assert!(
                pair[1] <= pair[0],
                "expected monotonic movement toward the inlier cluster: {outlier_distances:?}"
            );
        }
    }

    #[test]
    fn literature_reduced_fixture_matches_printed_blue_result_with_rounded_input_tolerance() {
        let out = combine_measurements(&reduced_literature_spec(0.0), 0.683).unwrap();
        let half_width = (out.confidence_interval.upper - out.confidence_interval.lower) / 2.0;
        assert!((out.mu_hat - 172.91).abs() <= 2e-2, "unexpected mu_hat: {}", out.mu_hat);
        assert!((half_width - 0.29).abs() <= 1e-2, "unexpected half-width: {half_width}");
    }

    #[test]
    fn literature_reduced_fixture_shifts_toward_baseline_and_widens_with_error_on_error() {
        let baseline = 172.51;
        let fixed = combine_measurements(&reduced_literature_spec(0.0), 0.68).unwrap();
        let eps_small = combine_measurements(&reduced_literature_spec(0.2), 0.68).unwrap();
        let eps_large = combine_measurements(&reduced_literature_spec(0.5), 0.68).unwrap();

        let fixed_width = fixed.confidence_interval.upper - fixed.confidence_interval.lower;
        let small_width = eps_small.confidence_interval.upper - eps_small.confidence_interval.lower;
        let large_width = eps_large.confidence_interval.upper - eps_large.confidence_interval.lower;

        assert!(small_width > fixed_width, "expected epsilon=0.2 to widen the interval");
        assert!(large_width >= small_width, "expected monotonic interval widening");
        assert!(
            (eps_small.mu_hat - baseline).abs() < (fixed.mu_hat - baseline).abs(),
            "expected epsilon=0.2 to move mu toward the published baseline"
        );
        assert!(
            (eps_large.mu_hat - baseline).abs() <= (eps_small.mu_hat - baseline).abs(),
            "expected epsilon=0.5 to move mu no farther from the baseline than epsilon=0.2"
        );
    }

    #[test]
    fn trivial_rank1_gvm_surfaces_bartlett_diagnostics() {
        let out = combine_measurements(&trivial_rank1_spec(0.25), 0.68).unwrap();
        let bartlett = &out.diagnostics.bartlett;

        assert!(bartlett.supported, "expected Bartlett diagnostics to be supported");
        assert_eq!(bartlett.method, "lawley_order_eps2_general");
        assert_eq!(bartlett.supported_systematics, vec!["scale".to_string()]);
        assert!(bartlett.unsupported_reason.is_none());
        assert!(bartlett.b_mu_theta.unwrap().is_finite());
        assert!(bartlett.b_tilde_theta.unwrap().is_finite());
        assert!(bartlett.b_mu.unwrap().is_finite());
        assert!(bartlett.b_q.unwrap().is_finite());
        assert!(bartlett.w_mu_scale.unwrap() > 0.0);
        assert!(bartlett.sigma_scale.unwrap() > 0.0);
        assert!(bartlett.sigma_star.unwrap().is_finite());
        assert!(bartlett.q_scale.unwrap() > 0.0);
        assert!(bartlett.q_star.unwrap().is_finite());
        assert!(bartlett.p_value_star.unwrap().is_finite());
    }

    #[test]
    fn trivial_rank1_gvm_surfaces_perturbative_validity_diagnostics() {
        let out = combine_measurements(&trivial_rank1_spec(0.25), 0.68).unwrap();
        let validity = &out.diagnostics.perturbative_validity;

        assert_eq!(validity.threshold, 1.0);
        assert_eq!(validity.systematic_names, vec!["scale".to_string()]);
        assert_eq!(validity.condition_values.len(), 1);
        assert_eq!(validity.within_threshold.len(), 1);
        assert!(validity.condition_values[0].is_finite());
    }

    #[test]
    fn full_literature_fixture_matches_printed_blue_result() {
        let out = combine_measurements(&full_literature_spec(None), 0.683).unwrap();
        let half_width = (out.confidence_interval.upper - out.confidence_interval.lower) / 2.0;
        assert!((out.mu_hat - 172.51).abs() <= 5e-3, "unexpected mu_hat: {}", out.mu_hat);
        assert!((half_width - 0.33).abs() <= 1e-2, "unexpected half-width: {half_width}");
    }

    #[test]
    fn full_literature_fixture_accepts_raw_non_psd_source_matrices_and_surfaces_regularization() {
        let spec = full_literature_spec(None);
        spec.validate().unwrap();
        let prep = PreparedSpec::from_spec(&spec).unwrap();
        let fixed = combine_measurements(&spec, 0.68).unwrap();

        assert!(
            prep.corr_regularization_deltas.iter().any(|delta| *delta > 0.0),
            "expected published non-PSD source matrices to be detected during preparation"
        );
        assert!(
            fixed.diagnostics.corr_regularization_deltas.iter().any(|delta| *delta > 0.0),
            "expected regularization diagnostics to be exposed in the public result"
        );
    }

    #[test]
    fn full_literature_nontrivial_gvm_surfaces_bartlett_diagnostics() {
        let out = combine_measurements(&full_literature_spec(Some(("b-JES", 0.5))), 0.68).unwrap();
        let bartlett = &out.diagnostics.bartlett;

        assert!(bartlett.supported, "expected Bartlett diagnostics to be supported");
        assert_eq!(bartlett.method, "lawley_order_eps2_general");
        assert!(bartlett.unsupported_reason.is_none());
        assert_eq!(bartlett.supported_systematics, vec!["b-JES".to_string()]);
        assert!(bartlett.b_mu_theta.unwrap().is_finite());
        assert!(bartlett.b_tilde_theta.unwrap().is_finite());
        assert!(bartlett.b_mu.unwrap().is_finite());
        assert!(bartlett.b_q.unwrap().is_finite());
        assert!(bartlett.w_mu_scale.unwrap() > 0.0);
        assert!(bartlett.q_scale.unwrap() > 0.0);
        assert!(bartlett.q_star.unwrap().is_finite());
        assert!(bartlett.p_value_star.unwrap().is_finite());
        assert_eq!(bartlett.sigma2_unbiased_estimates.len(), 1);
        assert!(bartlett.sigma2_unbiased_estimates[0].is_finite());
    }

    #[test]
    fn full_literature_nontrivial_gvm_surfaces_perturbative_validity_diagnostics() {
        let out = combine_measurements(&full_literature_spec(Some(("b-JES", 0.5))), 0.68).unwrap();
        let validity = &out.diagnostics.perturbative_validity;

        assert_eq!(validity.threshold, 1.0);
        assert_eq!(validity.systematic_names, vec!["b-JES".to_string()]);
        assert_eq!(validity.condition_values.len(), 1);
        assert_eq!(validity.within_threshold.len(), 1);
        assert!(validity.condition_values[0].is_finite());
    }

    #[test]
    fn bartlett_workspace_fast_path_matches_reference_blocks_on_large_supported_case() {
        let spec = synthetic_fast_path_spec(10, 13, 0.20);
        let prep = PreparedSpec::from_spec(&spec).unwrap();
        let state = synthetic_bartlett_state(&prep, 1.15);
        let reference = build_bartlett_workspace_with_threshold(&prep, &state, usize::MAX).unwrap();
        let fast = build_bartlett_workspace_with_threshold(&prep, &state, 0).unwrap();

        assert_eq!(reference.build_path, BartlettWorkspaceBuildPath::Reference);
        assert_eq!(fast.build_path, BartlettWorkspaceBuildPath::Fast);
        assert_eq!(reference.j_blocks.len(), fast.j_blocks.len());
        assert_eq!(reference.j_tilde_blocks.len(), fast.j_tilde_blocks.len());

        let mut max_j_diff = 0.0_f64;
        for (lhs, rhs) in reference.j_blocks.iter().zip(&fast.j_blocks) {
            for i in 0..lhs.nrows() {
                for j in 0..lhs.ncols() {
                    max_j_diff = max_j_diff.max((lhs[(i, j)] - rhs[(i, j)]).abs());
                }
            }
        }
        let mut max_j_tilde_diff = 0.0_f64;
        for (lhs, rhs) in reference.j_tilde_blocks.iter().zip(&fast.j_tilde_blocks) {
            for i in 0..lhs.nrows() {
                for j in 0..lhs.ncols() {
                    max_j_tilde_diff = max_j_tilde_diff.max((lhs[(i, j)] - rhs[(i, j)]).abs());
                }
            }
        }

        let (reference_b_mu_theta, reference_b_tilde_theta, reference_sigma2) =
            bartlett_workspace_scalars(&prep, &state, &reference);
        let (fast_b_mu_theta, fast_b_tilde_theta, fast_sigma2) =
            bartlett_workspace_scalars(&prep, &state, &fast);

        assert!(max_j_diff < 1e-12, "unexpected J-block gap: {max_j_diff}");
        assert!(max_j_tilde_diff < 1e-12, "unexpected J_tilde gap: {max_j_tilde_diff}");
        assert_relative_eq!(
            reference_b_mu_theta,
            fast_b_mu_theta,
            epsilon = 1e-12,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            reference_b_tilde_theta,
            fast_b_tilde_theta,
            epsilon = 1e-12,
            max_relative = 1e-12
        );
        assert_eq!(reference_sigma2.len(), fast_sigma2.len());
        for (lhs, rhs) in reference_sigma2.iter().zip(&fast_sigma2) {
            assert_relative_eq!(lhs, rhs, epsilon = 1e-12, max_relative = 1e-12);
        }
    }

    #[test]
    fn bartlett_workspace_guarded_dispatch_uses_fast_path_above_threshold() {
        let spec = synthetic_fast_path_spec(10, 13, 0.20);
        let prep = PreparedSpec::from_spec(&spec).unwrap();
        let state = synthetic_bartlett_state(&prep, 1.15);

        assert!(prep.y.len() * prep.systematics.len() > BARTLETT_FAST_PATH_NM_THRESHOLD);
        let workspace = build_bartlett_workspace(&prep, &state).unwrap();
        assert_eq!(workspace.build_path, BartlettWorkspaceBuildPath::Fast);
    }

    #[test]
    fn bartlett_workspace_fast_path_falls_back_for_rank_deficient_corr_blocks() {
        let spec = trivial_rank1_spec(0.25);
        let prep = PreparedSpec::from_spec(&spec).unwrap();
        let state = synthetic_bartlett_state(&prep, 1.0);

        let workspace = build_bartlett_workspace_with_threshold(&prep, &state, 0).unwrap();
        assert_eq!(workspace.build_path, BartlettWorkspaceBuildPath::Reference);
    }

    #[test]
    fn analytic_linear_solve_fast_path_matches_dense_reference_on_large_supported_case() {
        let spec = synthetic_fast_path_spec(10, 13, 0.20);
        let prep = PreparedSpec::from_spec(&spec).unwrap();
        let solver = AnalyticPerturbativeSolver::new(&prep, 1);
        let mu = 170.35;
        let residual_mu = &prep.y - prep.ones.clone().scale(mu);

        let (theta_reference, path_reference) = solver
            .solve_linear_system_with_threshold(&residual_mu, None, None, usize::MAX)
            .unwrap();
        let (theta_fast, path_fast) =
            solver.solve_linear_system_with_threshold(&residual_mu, None, None, 0).unwrap();

        assert_eq!(path_reference, AnalyticLinearSolvePath::Reference);
        assert_eq!(path_fast, AnalyticLinearSolvePath::Fast);
        assert_eq!(theta_reference.len(), theta_fast.len());
        for (lhs, rhs) in theta_reference.iter().zip(&theta_fast) {
            for i in 0..lhs.len() {
                assert_relative_eq!(lhs[i], rhs[i], epsilon = 1e-12, max_relative = 1e-12);
            }
        }

        let sigma2 = solver.compute_sigma2(&theta_reference);
        let residual_refined = &residual_mu - solver.predicted_shift(&theta_reference);
        let (delta_reference, path_reference) = solver
            .solve_linear_system_with_threshold(
                &residual_refined,
                Some(&theta_reference),
                Some(&sigma2),
                usize::MAX,
            )
            .unwrap();
        let (delta_fast, path_fast) = solver
            .solve_linear_system_with_threshold(
                &residual_refined,
                Some(&theta_reference),
                Some(&sigma2),
                0,
            )
            .unwrap();

        assert_eq!(path_reference, AnalyticLinearSolvePath::Reference);
        assert_eq!(path_fast, AnalyticLinearSolvePath::Fast);
        assert_eq!(delta_reference.len(), delta_fast.len());
        for (lhs, rhs) in delta_reference.iter().zip(&delta_fast) {
            for i in 0..lhs.len() {
                assert_relative_eq!(lhs[i], rhs[i], epsilon = 1e-12, max_relative = 1e-12);
            }
        }

        let point_reference = solver.profile_at_mu_raw_with_threshold(mu, usize::MAX).unwrap();
        let point_fast = solver.profile_at_mu_raw_with_threshold(mu, 0).unwrap();
        assert_relative_eq!(
            point_reference.nll,
            point_fast.nll,
            epsilon = 1e-12,
            max_relative = 1e-12
        );
        assert_eq!(
            point_reference.state.profiled_variance_scales.len(),
            point_fast.state.profiled_variance_scales.len()
        );
        for (lhs, rhs) in point_reference
            .state
            .profiled_variance_scales
            .iter()
            .zip(&point_fast.state.profiled_variance_scales)
        {
            assert_relative_eq!(lhs, rhs, epsilon = 1e-12, max_relative = 1e-12);
        }
    }

    #[test]
    fn analytic_linear_solve_fast_path_falls_back_for_rank_deficient_corr_blocks() {
        let spec = trivial_rank1_spec(0.25);
        let prep = PreparedSpec::from_spec(&spec).unwrap();
        let solver = AnalyticPerturbativeSolver::new(&prep, 1);
        let residual_mu = &prep.y - prep.ones.clone().scale(1.1);

        let (_, path) =
            solver.solve_linear_system_with_threshold(&residual_mu, None, None, 0).unwrap();
        assert_eq!(path, AnalyticLinearSolvePath::Reference);
    }

    #[test]
    fn analytic_perturbative_trivial_rank1_matches_numerical_reference() {
        let spec = trivial_rank1_spec(0.02);
        let numerical = combine_measurements_with_solver(
            &spec,
            0.68,
            MeasurementCombinationSolver::NumericalPaper,
        )
        .unwrap();
        let analytic = combine_measurements_with_solver(
            &spec,
            0.68,
            MeasurementCombinationSolver::AnalyticPerturbative,
        )
        .unwrap();

        assert_eq!(analytic.optimizer.method, "analytic_perturbative_order_eps2");
        assert!((analytic.mu_hat - numerical.mu_hat).abs() < 2e-3);
        assert!(analytic.diagnostics.perturbative_validity.within_threshold.iter().all(|v| *v));
    }

    #[test]
    fn analytic_perturbative_full_literature_bjes_matches_numerical_reference() {
        let spec = full_literature_spec(Some(("b-JES", 0.05)));
        let numerical = combine_measurements_with_solver(
            &spec,
            0.68,
            MeasurementCombinationSolver::NumericalPaper,
        )
        .unwrap();
        let analytic = combine_measurements_with_solver(
            &spec,
            0.68,
            MeasurementCombinationSolver::AnalyticPerturbative,
        )
        .unwrap();

        assert_eq!(analytic.optimizer.method, "analytic_perturbative_order_eps2");
        assert!((analytic.mu_hat - numerical.mu_hat).abs() < 1e-2);
        assert!(
            (analytic.confidence_interval.sigma - numerical.confidence_interval.sigma).abs() < 1e-2
        );
        assert!(analytic.diagnostics.perturbative_validity.within_threshold.iter().all(|v| *v));
    }

    #[test]
    fn analytic_perturbative_rejects_outside_validity_radius() {
        let spec = outlier_spec(1.5);
        let err = combine_measurements_with_solver(
            &spec,
            0.68,
            MeasurementCombinationSolver::AnalyticPerturbative,
        )
        .unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("perturbative") || message.contains("validity"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn analytic_perturbative_attempt_returns_fallback_warm_start_on_invalid_case() {
        let prep = PreparedSpec::from_spec(&outlier_spec(1.5)).unwrap();
        let attempt = analytic_perturbative_attempt(prep, 0.68, 1).unwrap();

        match attempt {
            AnalyticPerturbativeAttempt::FallbackWarmStart(Some(guide)) => {
                assert!(!guide.mle_params.is_empty());
                assert!(guide.lower_hint.is_none());
                assert!(guide.upper_hint.is_none());
            }
            _ => panic!("expected invalid analytic attempt to yield fallback warm start"),
        }
    }

    #[test]
    fn combine_measurements_default_uses_auto_solver_on_valid_low_epsilon_case() {
        let spec = full_literature_spec(Some(("b-JES", 0.05)));
        let out = combine_measurements(&spec, 0.68).unwrap();

        assert_eq!(out.optimizer.method, "analytic_perturbative_order_eps2");
        assert!(out.diagnostics.perturbative_validity.within_threshold.iter().all(|v| *v));
        assert_eq!(out.diagnostics.requested_solver.as_deref(), Some("auto"));
        assert_eq!(out.diagnostics.effective_solver.as_deref(), Some("analytic-perturbative"));
    }

    #[test]
    fn combine_measurements_default_falls_back_to_numerical_paper_outside_validity_radius() {
        let spec = outlier_spec(1.5);
        let out = combine_measurements(&spec, 0.68).unwrap();
        let paper = combine_measurements_with_solver(
            &spec,
            0.68,
            MeasurementCombinationSolver::NumericalPaper,
        )
        .unwrap();

        assert_eq!(out.optimizer.method, "numerical_profile_gvm_original_theta");
        assert_eq!(out.diagnostics.requested_solver.as_deref(), Some("auto"));
        assert_eq!(out.diagnostics.effective_solver.as_deref(), Some("numerical-paper"));
        assert_relative_eq!(out.mu_hat, paper.mu_hat, epsilon = 1e-12);
        assert_relative_eq!(
            out.confidence_interval.sigma,
            paper.confidence_interval.sigma,
            epsilon = 1e-12
        );
        assert_relative_eq!(out.goodness_of_fit.chi2, paper.goodness_of_fit.chi2, epsilon = 1e-12);
    }

    #[test]
    #[ignore = "research-grade slow gate; run explicitly"]
    fn full_literature_bjes_gvm_fit_moves_toward_published_blue_baseline() {
        let fixed_spec = full_literature_spec(None);
        let fixed = combine_measurements(&fixed_spec, 0.68).unwrap();

        let gvm_spec = full_literature_spec(Some(("b-JES", 0.5)));
        let prep = PreparedSpec::from_spec(&gvm_spec).unwrap();
        let objective = MeasurementCombineObjective::new(&prep);
        let bounds = prep.bounds();
        let gvm_fit = numerical_gvm_fit(&prep, &objective, &bounds).unwrap();

        assert!(gvm_fit.fit.converged, "expected converged full published b-JES GVM fit");
        assert!(
            gvm_fit.state.profiled_variance_scales.iter().any(|tau| *tau > 1.0),
            "expected at least one profiled variance scale inflation"
        );
        assert!(
            (gvm_fit.fit.parameters[0] - fixed.mu_hat).abs() < 0.1,
            "expected full-combination central-value shift to stay within the published 0.1 GeV scale"
        );
        let gvm = combine_measurements(&gvm_spec, 0.68).unwrap();
        let fixed_width = fixed.confidence_interval.upper - fixed.confidence_interval.lower;
        let gvm_width = gvm.confidence_interval.upper - gvm.confidence_interval.lower;
        assert!(gvm.converged, "expected converged full published b-JES GVM result");
        assert!(gvm_width > fixed_width, "expected b-JES epsilon to widen the interval");
    }

    #[test]
    #[ignore = "research-grade slow gate; run explicitly"]
    fn bartlett_q_star_mean_is_closer_to_df_than_raw_q_in_toy_mc() {
        let spec = trivial_rank1_spec(0.25);
        let n_toys = 128usize;
        let mut sum_q = 0.0;
        let mut sum_q_star = 0.0;
        let mut used = 0usize;

        for toy_idx in 0..n_toys {
            let toy = simulate_gvm_toy_spec(&spec, 10_000 + toy_idx as u64).unwrap();
            let out = combine_measurements(&toy, 0.68).unwrap();
            sum_q += out.goodness_of_fit.chi2;
            sum_q_star += out.diagnostics.bartlett.q_star.unwrap();
            used += 1;
        }

        let mean_q = sum_q / used as f64;
        let mean_q_star = sum_q_star / used as f64;
        let df = (spec.measurements.len() - 1) as f64;
        assert!(
            (mean_q_star - df).abs() < (mean_q - df).abs(),
            "expected Bartlett-corrected GOF mean to move closer to df: mean_q={mean_q} mean_q_star={mean_q_star} df={df}"
        );
    }

    #[test]
    fn calibrate_measurements_toys_returns_research_grade_report() {
        let report = calibrate_measurements_toys(&outlier_full_gvm_spec(), 0.68, 16, 123).unwrap();
        assert_eq!(report.schema_version, "nextstat_measurement_combination_calibration_v0");
        assert_eq!(report.stability, "stable");
        assert_eq!(report.n_toys, 16);
        assert_eq!(report.seed, 123);
        assert_eq!(report.reference.poi, "mu");
        assert_eq!(report.summary.df, 2);
        assert!(report.summary.mean_q.is_finite());
        assert!(report.summary.mean_q_star.is_finite());
        assert!(report.summary.mean_q_abs_error_to_df.is_finite());
        assert!(report.summary.mean_q_star_abs_error_to_df.is_finite());
        assert!(report.summary.mean_sigma.is_finite());
        assert!(report.summary.mean_sigma_star.is_finite());
        assert!(report.summary.mean_sigma_star_to_sigma_ratio.is_finite());
        assert!(report.summary.sigma_star_ge_sigma_fraction.is_finite());
        assert!(report.summary.sigma_star_ge_sigma_fraction >= 0.0);
        assert!(report.summary.sigma_star_ge_sigma_fraction <= 1.0);
        assert_eq!(
            report.summary.bartlett_improves_mean_q,
            report.summary.mean_q_star_abs_error_to_df <= report.summary.mean_q_abs_error_to_df
        );
        assert!(
            report.summary.mean_sigma_star_to_sigma_ratio > 0.0,
            "expected positive sigma ratio"
        );
    }

    #[test]
    fn calibrate_measurements_toys_default_uses_auto_fit_and_paper_toy_generation() {
        let report = calibrate_measurements_toys(
            &full_literature_spec(Some(("b-JES", 0.05))),
            0.68,
            8,
            2026,
        )
        .unwrap();

        assert_eq!(report.reference.optimizer.method, "analytic_perturbative_order_eps2");
        assert_eq!(
            report.summary.toy_generation_method,
            "measurement_side_gvm_unbiased_sigma2_star_normalized_spec_original_theta"
        );
    }

    #[test]
    fn calibration_toy_fast_path_matches_full_numerical_paper_result() {
        let toy_spec = simulate_gvm_toy_spec_with_solver(
            &outlier_full_gvm_spec(),
            MeasurementCombinationSolver::NumericalPaper,
            123,
        )
        .unwrap();
        let fast = combine_measurements_calibration_toy_result_with_solver(
            &toy_spec,
            0.68,
            MeasurementCombinationSolver::NumericalPaper,
        )
        .unwrap();
        let full = combine_measurements_with_solver(
            &toy_spec,
            0.68,
            MeasurementCombinationSolver::NumericalPaper,
        )
        .unwrap();
        let expected = calibration_toy_result_from_full_result(&full).unwrap();

        assert_relative_eq!(fast.q, expected.q, epsilon = 1e-12);
        assert_relative_eq!(fast.q_star, expected.q_star, epsilon = 1e-12);
        assert_relative_eq!(fast.sigma, expected.sigma, epsilon = 1e-12);
        assert_relative_eq!(fast.sigma_star, expected.sigma_star, epsilon = 1e-12);
    }

    #[test]
    fn calibration_toy_fast_path_matches_full_auto_fallback_result() {
        let toy_spec = simulate_gvm_toy_spec_with_solver(
            &outlier_full_gvm_spec(),
            MeasurementCombinationSolver::NumericalPaper,
            123u64.wrapping_add(14),
        )
        .unwrap();
        let fast = combine_measurements_calibration_toy_result_with_solver(
            &toy_spec,
            0.68,
            MeasurementCombinationSolver::Auto,
        )
        .unwrap();
        let full =
            combine_measurements_with_solver(&toy_spec, 0.68, MeasurementCombinationSolver::Auto)
                .unwrap();
        let expected = calibration_toy_result_from_full_result(&full).unwrap();

        assert_relative_eq!(fast.q, expected.q, epsilon = 1e-12);
        assert_relative_eq!(fast.q_star, expected.q_star, epsilon = 1e-12);
        assert_relative_eq!(fast.sigma, expected.sigma, epsilon = 1e-12);
        assert_relative_eq!(fast.sigma_star, expected.sigma_star, epsilon = 1e-12);
    }

    #[test]
    fn cached_toy_generation_context_matches_uncached_reference_path() {
        let spec = outlier_full_gvm_spec();
        let context = build_measurement_toy_generation_context(
            &spec,
            MeasurementCombinationSolver::NumericalPaper,
        )
        .unwrap();

        for seed in [123_u64, 124_u64, 125_u64] {
            let cached = simulate_gvm_toy_spec_from_context(&context, seed);
            let uncached = simulate_gvm_toy_spec_with_solver_uncached_reference(
                &spec,
                MeasurementCombinationSolver::NumericalPaper,
                seed,
            )
            .unwrap();

            for (cached_measurement, uncached_measurement) in
                cached.measurements.iter().zip(uncached.measurements.iter())
            {
                assert_relative_eq!(
                    cached_measurement.value,
                    uncached_measurement.value,
                    epsilon = 1e-12
                );
            }
        }
    }

    #[test]
    fn paper_toy_generation_context_caches_template_warm_start() {
        let spec = outlier_full_gvm_spec();
        let context = build_measurement_toy_generation_context(
            &spec,
            MeasurementCombinationSolver::NumericalPaper,
        )
        .unwrap();

        let warm_start = context
            .paper_warm_start
            .as_ref()
            .expect("paper toy generation context should cache a template paper warm-start");
        assert!(!warm_start.mle_params.is_empty());
        assert!(warm_start.lower_hint.is_none());
        assert!(warm_start.upper_hint.is_none());
    }

    fn template_paper_warm_start_preserves_numerical_paper_toy_result() {
        let spec = outlier_full_gvm_spec();
        let context = build_measurement_toy_generation_context(
            &spec,
            MeasurementCombinationSolver::NumericalPaper,
        )
        .unwrap();

        let prep = simulate_gvm_toy_prepared_spec_from_context(&context, 123);
        let expected = combine_measurements_calibration_toy_result_with_prepared_solver(
            prep.clone(),
            0.68,
            MeasurementCombinationSolver::NumericalPaper,
        )
        .unwrap();
        let actual =
            combine_measurements_calibration_toy_result_with_prepared_solver_and_warm_start(
                prep,
                0.68,
                MeasurementCombinationSolver::NumericalPaper,
                context.paper_warm_start.as_ref(),
            )
            .unwrap();

        assert_relative_eq!(actual.q, expected.q, epsilon = 1e-9);
        assert_relative_eq!(actual.q_star, expected.q_star, epsilon = 1e-8);
        assert_relative_eq!(actual.sigma, expected.sigma, epsilon = 1e-8);
        assert_relative_eq!(actual.sigma_star, expected.sigma_star, epsilon = 1e-8);
    }

    #[test]
    fn template_paper_warm_start_preserves_auto_fallback_toy_result() {
        let spec = outlier_full_gvm_spec();
        let context = build_measurement_toy_generation_context(
            &spec,
            MeasurementCombinationSolver::NumericalPaper,
        )
        .unwrap();

        let prep = simulate_gvm_toy_prepared_spec_from_context(&context, 123u64.wrapping_add(14));
        let expected = combine_measurements_calibration_toy_result_with_prepared_solver(
            prep.clone(),
            0.68,
            MeasurementCombinationSolver::Auto,
        )
        .unwrap();
        let actual =
            combine_measurements_calibration_toy_result_with_prepared_solver_and_warm_start(
                prep,
                0.68,
                MeasurementCombinationSolver::Auto,
                context.paper_warm_start.as_ref(),
            )
            .unwrap();

        assert_relative_eq!(actual.q, expected.q, epsilon = 1e-12);
        assert_relative_eq!(actual.q_star, expected.q_star, epsilon = 1e-8);
        assert_relative_eq!(actual.sigma, expected.sigma, epsilon = 1e-12);
        assert_relative_eq!(actual.sigma_star, expected.sigma_star, epsilon = 1e-8);
    }

    #[test]
    fn reduced_basis_objective_gradient_matches_finite_difference() {
        let spec = outlier_full_gvm_spec();
        let prep = PreparedSpec::from_spec(&spec).unwrap();
        let objective = MeasurementCombineObjective::new(&prep);
        let params = prep.init_params().unwrap();
        let analytic = objective.gradient(&params).unwrap();
        let numeric = finite_difference_gradient(&objective, &params).unwrap();

        for (idx, (lhs, rhs)) in analytic.iter().zip(numeric.iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() < 1e-5,
                "reduced-basis gradient mismatch at idx={idx}: analytic={lhs} numeric={rhs}"
            );
        }
    }

    #[test]
    fn paper_objective_gradient_matches_finite_difference() {
        let spec = outlier_full_gvm_spec();
        let prep = PreparedSpec::from_spec(&spec).unwrap();
        let objective = PaperMeasurementCombineObjective::new(&prep);
        let params = objective
            .analytic_warm_start_params()
            .unwrap_or_else(|| objective.init_params().unwrap());
        let analytic = objective.gradient(&params).unwrap();
        let numeric = finite_difference_gradient(&objective, &params).unwrap();

        for (idx, (lhs, rhs)) in analytic.iter().zip(numeric.iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() < 1e-5,
                "paper gradient mismatch at idx={idx}: analytic={lhs} numeric={rhs}"
            );
        }
    }

    #[test]
    fn prepared_toy_context_path_matches_spec_roundtrip_reference() {
        let spec = outlier_full_gvm_spec();
        let context = build_measurement_toy_generation_context(
            &spec,
            MeasurementCombinationSolver::NumericalPaper,
        )
        .unwrap();

        for (seed, solver) in [
            (123_u64, MeasurementCombinationSolver::NumericalPaper),
            (123_u64.wrapping_add(14), MeasurementCombinationSolver::Auto),
        ] {
            let toy_spec = simulate_gvm_toy_spec_from_context(&context, seed);
            let expected =
                combine_measurements_calibration_toy_result_with_solver(&toy_spec, 0.68, solver)
                    .unwrap();

            let prep = simulate_gvm_toy_prepared_spec_from_context(&context, seed);
            let actual = combine_measurements_calibration_toy_result_with_prepared_solver(
                prep, 0.68, solver,
            )
            .unwrap();

            assert_relative_eq!(actual.q, expected.q, epsilon = 1e-12);
            assert_relative_eq!(actual.q_star, expected.q_star, epsilon = 1e-12);
            assert_relative_eq!(actual.sigma, expected.sigma, epsilon = 1e-12);
            assert_relative_eq!(actual.sigma_star, expected.sigma_star, epsilon = 1e-12);
        }
    }

    #[test]
    fn generated_calibration_toy_specs_still_support_auto_solver_fallback_contract() {
        let spec = outlier_full_gvm_spec();
        for toy_idx in 0..16 {
            let toy_spec = simulate_gvm_toy_spec_with_solver(
                &spec,
                MeasurementCombinationSolver::NumericalPaper,
                123u64.wrapping_add(toy_idx as u64),
            )
            .unwrap();
            let out = combine_measurements_with_solver(
                &toy_spec,
                0.68,
                MeasurementCombinationSolver::Auto,
            )
            .unwrap_or_else(|err| {
                panic!("toy_idx={toy_idx} should not leak auto fallback error: {err}")
            });

            assert!(out.mu_hat.is_finite(), "toy_idx={toy_idx}");
            assert!(out.confidence_interval.sigma.is_finite(), "toy_idx={toy_idx}");
            assert!(out.goodness_of_fit.chi2.is_finite(), "toy_idx={toy_idx}");
        }
    }

    #[test]
    fn calibrate_measurements_toys_study_returns_research_grade_report() {
        let report =
            calibrate_measurements_toys_study(&outlier_full_gvm_spec(), 0.68, 16, &[123, 124, 125])
                .unwrap();
        assert_eq!(report.schema_version, "nextstat_measurement_combination_calibration_study_v0");
        assert_eq!(report.stability, "stable");
        assert_eq!(report.seeds, vec![123, 124, 125]);
        assert_eq!(report.per_seed.len(), 3);
        assert_eq!(report.aggregate.n_runs, 3);
        assert!(report.aggregate.mean_of_mean_q.is_finite());
        assert!(report.aggregate.mean_of_mean_q_star.is_finite());
        assert!(report.aggregate.mean_of_mean_sigma.is_finite());
        assert!(report.aggregate.mean_of_mean_sigma_star.is_finite());
        assert!(report.aggregate.min_mean_sigma_star_to_sigma_ratio.is_finite());
        assert!(report.aggregate.max_mean_sigma_star_to_sigma_ratio.is_finite());
        assert!(
            report
                .aggregate
                .max_abs_mean_sigma_star_to_sigma_ratio_delta_from_reference
                .is_finite()
        );
        assert!(report.aggregate.max_sigma_star_ge_sigma_fraction <= 1.0);
        assert!(report.aggregate.min_sigma_star_ge_sigma_fraction >= 0.0);
    }

    #[test]
    fn calibrate_measurements_toys_study_parallel_outer_loop_matches_sequential_reference() {
        let spec = outlier_full_gvm_spec();
        let sequential = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| calibrate_measurements_toys_study(&spec, 0.68, 8, &[123, 124, 125]))
            .unwrap();
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| calibrate_measurements_toys_study(&spec, 0.68, 8, &[123, 124, 125]))
            .unwrap();

        assert_eq!(
            serde_json::to_value(&parallel).unwrap(),
            serde_json::to_value(&sequential).unwrap()
        );
    }

    #[test]
    fn scenario_study_returns_research_grade_report() {
        let report = study_measurement_combination_scenarios(
            &calibration_outlier_fixture_spec(),
            &outlier_scenario_study_spec(),
            0.68,
        )
        .unwrap();
        assert_eq!(report.schema_version, "nextstat_measurement_combination_scenario_study_v0");
        assert_eq!(report.stability, "research-grade");
        assert_eq!(report.scenarios.len(), 3);
        assert_eq!(report.aggregate.n_scenarios, 3);
        assert!(report.aggregate.all_converged);
        assert!(report.aggregate.max_sigma_ratio_to_baseline.is_finite());
        assert!(report.aggregate.min_sigma_ratio_to_baseline.is_finite());
        assert!(report.aggregate.largest_abs_mu_shift.is_finite());
        assert!(!report.aggregate.widest_interval_scenario.is_empty());
    }

    #[test]
    fn scenario_study_parallel_outer_loop_matches_sequential_reference() {
        let spec = calibration_outlier_fixture_spec();
        let scenarios = outlier_scenario_study_spec();
        let sequential = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| study_measurement_combination_scenarios(&spec, &scenarios, 0.68))
            .unwrap();
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| study_measurement_combination_scenarios(&spec, &scenarios, 0.68))
            .unwrap();

        assert_eq!(
            serde_json::to_value(&parallel).unwrap(),
            serde_json::to_value(&sequential).unwrap()
        );
    }

    #[test]
    fn scenario_study_default_uses_auto_solver_contract() {
        let scenario_study = MeasurementCombinationScenarioStudySpec {
            schema_version: MEASUREMENT_COMBINATION_SCENARIO_CONFIG_SCHEMA_V0.to_string(),
            scenarios: vec![MeasurementCombinationScenarioSpec {
                name: "bjes_0p05".to_string(),
                error_on_error: vec![ScenarioErrorOnErrorAssignment {
                    systematic: "b-JES".to_string(),
                    value: 0.05,
                }],
            }],
        };
        let report = study_measurement_combination_scenarios(
            &full_literature_spec(None),
            &scenario_study,
            0.68,
        )
        .unwrap();

        assert_eq!(report.baseline.optimizer.method, "closed_form_blue");
        assert_eq!(report.scenarios[0].result.optimizer.method, "analytic_perturbative_order_eps2");
        assert_eq!(report.baseline.diagnostics.requested_solver, None);
        assert_eq!(report.baseline.diagnostics.effective_solver, None);
        assert_eq!(
            report.scenarios[0].result.diagnostics.requested_solver.as_deref(),
            Some("auto")
        );
        assert_eq!(
            report.scenarios[0].result.diagnostics.effective_solver.as_deref(),
            Some("analytic-perturbative")
        );
    }

    #[test]
    fn calibration_campaign_returns_research_grade_report() {
        let report = run_measurement_combination_calibration_campaign(
            &calibration_outlier_fixture_spec(),
            &outlier_scenario_study_spec(),
            0.68,
            16,
            &[123, 124, 125],
        )
        .unwrap();
        assert_eq!(
            report.schema_version,
            "nextstat_measurement_combination_calibration_campaign_v0"
        );
        assert_eq!(report.stability, "research-grade");
        assert_eq!(report.seeds, vec![123, 124, 125]);
        assert_eq!(report.scenarios.len(), 3);
        assert_eq!(report.aggregate.n_scenarios, 3);
        assert!(report.aggregate.max_fit_sigma_ratio_to_baseline.is_finite());
        assert!(report.aggregate.max_calibration_mean_sigma_star_to_sigma_ratio.is_finite());
        assert!(!report.aggregate.highest_calibration_sigma_ratio_scenario.is_empty());
    }

    #[test]
    fn calibration_campaign_parallel_outer_loop_matches_sequential_reference() {
        let spec = calibration_outlier_fixture_spec();
        let scenarios = outlier_scenario_study_spec();
        let sequential = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                run_measurement_combination_calibration_campaign(
                    &spec,
                    &scenarios,
                    0.68,
                    8,
                    &[123, 124, 125],
                )
            })
            .unwrap();
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                run_measurement_combination_calibration_campaign(
                    &spec,
                    &scenarios,
                    0.68,
                    8,
                    &[123, 124, 125],
                )
            })
            .unwrap();

        assert_eq!(
            serde_json::to_value(&parallel).unwrap(),
            serde_json::to_value(&sequential).unwrap()
        );
    }

    #[test]
    fn calibration_campaign_default_uses_auto_fit_and_paper_toy_generation_contract() {
        let scenario_study = MeasurementCombinationScenarioStudySpec {
            schema_version: MEASUREMENT_COMBINATION_SCENARIO_CONFIG_SCHEMA_V0.to_string(),
            scenarios: vec![MeasurementCombinationScenarioSpec {
                name: "bjes_0p05".to_string(),
                error_on_error: vec![ScenarioErrorOnErrorAssignment {
                    systematic: "b-JES".to_string(),
                    value: 0.05,
                }],
            }],
        };
        let report = run_measurement_combination_calibration_campaign(
            &full_literature_spec(None),
            &scenario_study,
            0.68,
            8,
            &[2026, 2027],
        )
        .unwrap();

        assert_eq!(report.baseline.optimizer.method, "closed_form_blue");
        assert_eq!(report.scenarios[0].fit.optimizer.method, "analytic_perturbative_order_eps2");
        assert_eq!(report.baseline.diagnostics.requested_solver, None);
        assert_eq!(report.baseline.diagnostics.effective_solver, None);
        assert_eq!(report.scenarios[0].fit.diagnostics.requested_solver.as_deref(), Some("auto"));
        assert_eq!(
            report.scenarios[0].fit.diagnostics.effective_solver.as_deref(),
            Some("analytic-perturbative")
        );
        assert_eq!(
            report.scenarios[0].calibration.toy_generation_method,
            "measurement_side_gvm_unbiased_sigma2_star_normalized_spec_original_theta"
        );
    }

    #[test]
    fn calibration_with_solver_uses_paper_reference_contract() {
        let spec = full_literature_spec(Some(("b-JES", 0.05)));
        let report = calibrate_measurements_toys_with_solver(
            &spec,
            0.68,
            MeasurementCombinationSolver::AnalyticPerturbative,
            8,
            2026,
        )
        .unwrap();

        assert_eq!(report.reference.optimizer.method, "analytic_perturbative_order_eps2");
        assert_eq!(
            report.summary.toy_generation_method,
            "measurement_side_gvm_unbiased_sigma2_star_normalized_spec_original_theta"
        );
    }

    #[test]
    fn scenario_study_with_solver_propagates_requested_optimizer() {
        let scenario_study = MeasurementCombinationScenarioStudySpec {
            schema_version: MEASUREMENT_COMBINATION_SCENARIO_CONFIG_SCHEMA_V0.to_string(),
            scenarios: vec![MeasurementCombinationScenarioSpec {
                name: "bjes_0p05".to_string(),
                error_on_error: vec![ScenarioErrorOnErrorAssignment {
                    systematic: "b-JES".to_string(),
                    value: 0.05,
                }],
            }],
        };
        let report = study_measurement_combination_scenarios_with_solver(
            &full_literature_spec(None),
            &scenario_study,
            0.68,
            MeasurementCombinationSolver::AnalyticPerturbative,
        )
        .unwrap();

        assert_eq!(report.baseline.optimizer.method, "closed_form_blue");
        assert_eq!(report.scenarios[0].result.optimizer.method, "analytic_perturbative_order_eps2");
    }

    #[test]
    fn calibration_campaign_with_solver_propagates_fit_and_toy_solver_contract() {
        let scenario_study = MeasurementCombinationScenarioStudySpec {
            schema_version: MEASUREMENT_COMBINATION_SCENARIO_CONFIG_SCHEMA_V0.to_string(),
            scenarios: vec![MeasurementCombinationScenarioSpec {
                name: "bjes_0p05".to_string(),
                error_on_error: vec![ScenarioErrorOnErrorAssignment {
                    systematic: "b-JES".to_string(),
                    value: 0.05,
                }],
            }],
        };
        let report = run_measurement_combination_calibration_campaign_with_solver(
            &full_literature_spec(None),
            &scenario_study,
            0.68,
            MeasurementCombinationSolver::AnalyticPerturbative,
            8,
            &[2026, 2027],
        )
        .unwrap();

        assert_eq!(report.baseline.optimizer.method, "closed_form_blue");
        assert_eq!(report.scenarios[0].fit.optimizer.method, "analytic_perturbative_order_eps2");
        assert_eq!(
            report.scenarios[0].calibration.toy_generation_method,
            "measurement_side_gvm_unbiased_sigma2_star_normalized_spec_original_theta"
        );
    }

    #[test]
    fn paper_analytic_warm_start_is_nontrivial_for_full_literature_low_epsilon_case() {
        let prep = PreparedSpec::from_spec(&full_literature_spec(Some(("b-JES", 0.05)))).unwrap();
        let objective = PaperMeasurementCombineObjective::new(&prep);
        let cold = objective.init_params().unwrap();
        let warm = objective.analytic_warm_start_params().unwrap();

        assert_eq!(cold.len(), warm.len());
        assert_ne!(warm, cold);
        assert!(warm[0].is_finite());
        assert!(warm[0] >= objective.prep.mu_bounds.0 && warm[0] <= objective.prep.mu_bounds.1);
        assert!(warm.iter().skip(1).any(|value| value.abs() > 1e-10));
        assert!(warm.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn paper_init_params_from_profiled_state_roundtrips_analytic_state() {
        let prep = PreparedSpec::from_spec(&full_literature_spec(Some(("b-JES", 0.05)))).unwrap();
        let objective = PaperMeasurementCombineObjective::new(&prep);
        let solver = AnalyticPerturbativeSolver::new(&prep, 1);
        let point = minimize_analytic_profile(&solver, prep.mu_bounds).unwrap();
        let init = objective.init_params_from_profiled_state(point.mu, &point.state).unwrap();

        assert_eq!(init.len(), objective.layout.n_params);
        assert!((init[0] - point.mu).abs() < 1e-12);
        for (s_idx, theta) in point.state.theta_original.iter().enumerate() {
            let range = objective.layout.theta_ranges[s_idx].clone();
            for (offset, idx) in range.enumerate() {
                assert!((init[idx] - theta[offset]).abs() < 1e-12);
            }
            let tau_idx = objective.layout.tau_indices[s_idx];
            let expected_tau =
                point.state.profiled_variance_scales[s_idx].clamp(TAU_MIN, TAU_MAX).max(1e-12).ln();
            assert!((init[tau_idx] - expected_tau).abs() < 1e-12);
        }
    }

    #[test]
    fn paper_analytic_warm_start_guide_supplies_profile_hints_for_low_epsilon_case() {
        let prep = PreparedSpec::from_spec(&full_literature_spec(Some(("b-JES", 0.05)))).unwrap();
        let objective = PaperMeasurementCombineObjective::new(&prep);
        let guide = objective.analytic_warm_start_guide(0.68).unwrap();

        assert_eq!(guide.mle_params.len(), objective.layout.n_params);
        let lower = guide.lower_hint.expect("expected lower profile hint");
        let upper = guide.upper_hint.expect("expected upper profile hint");
        assert!(lower.mu < guide.mle_params[0]);
        assert!(upper.mu > guide.mle_params[0]);
        assert_eq!(lower.params.len(), objective.layout.n_params);
        assert_eq!(upper.params.len(), objective.layout.n_params);
        assert!(lower.params.iter().all(|value| value.is_finite()));
        assert!(upper.params.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn numerical_paper_default_warm_start_threshold_prefers_no_guide_for_large_original_theta_cases()
     {
        let small_spec = synthetic_fast_path_spec(32, 24, 0.05);
        let small_prep = PreparedSpec::from_spec(&small_spec).unwrap();
        let small_objective = PaperMeasurementCombineObjective::new(&small_prep);
        assert!(
            default_numerical_paper_warm_start(&small_objective, 0.68).is_some(),
            "moderate original-theta problems should still use analytic guide"
        );

        let large_spec = synthetic_fast_path_spec(64, 48, 0.05);
        let large_prep = PreparedSpec::from_spec(&large_spec).unwrap();
        let large_objective = PaperMeasurementCombineObjective::new(&large_prep);
        assert!(
            default_numerical_paper_warm_start(&large_objective, 0.68).is_none(),
            "large original-theta problems should bypass expensive analytic guide"
        );
    }

    #[test]
    fn numerical_paper_large_problem_default_result_matches_explicit_no_hint_path() {
        let spec = synthetic_fast_path_spec(64, 48, 0.05);
        let prep = PreparedSpec::from_spec(&spec).unwrap();
        let objective = PaperMeasurementCombineObjective::new(&prep);
        let bounds = objective.bounds();
        let fit_no_hint = numerical_paper_gvm_fit(&prep, &objective, &bounds, None).unwrap();
        let chi2_level = ChiSquared::new(1.0).unwrap().inverse_cdf(0.68);
        let (ci_lo, ci_hi) = profile_ci_mu_paper(
            &objective,
            &bounds,
            &fit_no_hint.fit.parameters,
            fit_no_hint.fit.fval,
            chi2_level,
            None,
        )
        .unwrap();
        let sigma = ((ci_hi - ci_lo) / 2.0).abs();

        let result = numerical_paper_gvm_result(prep.clone(), 0.68).unwrap();
        assert_relative_eq!(result.mu_hat, fit_no_hint.fit.parameters[0], epsilon = 1e-10);
        assert_relative_eq!(result.confidence_interval.lower, ci_lo, epsilon = 1e-10);
        assert_relative_eq!(result.confidence_interval.upper, ci_hi, epsilon = 1e-10);
        assert_relative_eq!(result.confidence_interval.sigma, sigma, epsilon = 1e-10);
    }

    fn clamp_test_param_to_bounds(value: f64, bounds: (f64, f64)) -> f64 {
        let (lo, hi) = bounds;
        let mut out = value;
        if lo.is_finite() {
            out = out.max(lo);
        }
        if hi.is_finite() {
            out = out.min(hi);
        }
        out
    }

    fn deterministic_paper_multistart_init(
        base: &[f64],
        bounds: &[(f64, f64)],
        mu_shift: f64,
        nuisance_scale: f64,
        phase: f64,
    ) -> Vec<f64> {
        let mut start = base.to_vec();
        start[0] = clamp_test_param_to_bounds(start[0] + mu_shift, bounds[0]);
        for i in 1..start.len() {
            let pattern = (((i + 1) as f64) * 1.618_033_988_75 + phase).sin();
            let span = if bounds[i].0.is_finite() && bounds[i].1.is_finite() {
                (bounds[i].1 - bounds[i].0).abs().min(1.0)
            } else {
                1.0 + base[i].abs()
            };
            let delta = nuisance_scale * span * pattern;
            start[i] = clamp_test_param_to_bounds(start[i] + delta, bounds[i]);
        }
        start
    }

    fn fit_numerical_paper_from_init(
        objective: &PaperMeasurementCombineObjective,
        bounds: &[(f64, f64)],
        init: &[f64],
    ) -> Result<crate::optimizer::OptimizationResult> {
        let optimizer = LbfgsbOptimizer::new(OptimizerConfig::from_strategy(
            crate::optimizer::OptimizerStrategy::HighPrecision,
        ));
        optimizer.minimize(objective, init, bounds)
    }

    fn assert_numerical_paper_multistart_converges_to_baseline(
        spec: &MeasurementCombinationSpec,
        ci_level: f64,
        starts: &[(f64, f64, f64)],
        mu_tol: f64,
        fval_tol: f64,
        ci_tol: f64,
    ) {
        let prep = PreparedSpec::from_spec(spec).unwrap();
        let objective = PaperMeasurementCombineObjective::new(&prep);
        let bounds = objective.bounds();
        let baseline = numerical_paper_gvm_fit(&prep, &objective, &bounds, None).unwrap();
        let chi2_level = ChiSquared::new(1.0).unwrap().inverse_cdf(ci_level);
        let (baseline_lo, baseline_hi) = profile_ci_mu_paper(
            &objective,
            &bounds,
            &baseline.fit.parameters,
            baseline.fit.fval,
            chi2_level,
            None,
        )
        .unwrap();

        let cold = objective.init_params().unwrap();
        for &(mu_shift, nuisance_scale, phase) in starts {
            let init = deterministic_paper_multistart_init(
                &cold,
                &bounds,
                mu_shift,
                nuisance_scale,
                phase,
            );
            let fit = fit_numerical_paper_from_init(&objective, &bounds, &init).unwrap();
            let (lo, hi) = profile_ci_mu_paper(
                &objective,
                &bounds,
                &fit.parameters,
                fit.fval,
                chi2_level,
                None,
            )
            .unwrap();

            assert!(
                (fit.fval - baseline.fit.fval).abs() <= fval_tol,
                "multi-start fval drift too large: actual={} baseline={} tol={}",
                fit.fval,
                baseline.fit.fval,
                fval_tol
            );
            assert!(
                (fit.parameters[0] - baseline.fit.parameters[0]).abs() <= mu_tol,
                "multi-start mu drift too large: actual={} baseline={} tol={}",
                fit.parameters[0],
                baseline.fit.parameters[0],
                mu_tol
            );
            assert!(
                (lo - baseline_lo).abs() <= ci_tol,
                "multi-start CI lower drift too large: actual={} baseline={} tol={}",
                lo,
                baseline_lo,
                ci_tol
            );
            assert!(
                (hi - baseline_hi).abs() <= ci_tol,
                "multi-start CI upper drift too large: actual={} baseline={} tol={}",
                hi,
                baseline_hi,
                ci_tol
            );
        }
    }

    fn default_multistart_offsets(sigma_guess: f64) -> Vec<(f64, f64, f64)> {
        vec![
            (1.5 * sigma_guess, 0.20, 0.0),
            (-1.5 * sigma_guess, 0.20, 1.0),
            (2.5 * sigma_guess, 0.30, 2.0),
        ]
    }

    #[test]
    fn numerical_paper_multistart_matches_baseline_on_medium_low_epsilon_case() {
        let spec = synthetic_fast_path_spec(32, 24, 0.05);
        let sigma_guess = PreparedSpec::from_spec(&spec).unwrap().fixed_sigma_guess;
        let starts = default_multistart_offsets(sigma_guess);
        assert_numerical_paper_multistart_converges_to_baseline(
            &spec, 0.68, &starts, 3e-7, 1e-8, 1e-6,
        );
    }

    #[test]
    #[ignore = "research-grade slow gate; run explicitly"]
    fn numerical_paper_multistart_matches_baseline_on_large_low_epsilon_case() {
        let spec = synthetic_fast_path_spec(64, 48, 0.05);
        let sigma_guess = PreparedSpec::from_spec(&spec).unwrap().fixed_sigma_guess;
        let mut starts = default_multistart_offsets(sigma_guess);
        starts.truncate(2);
        assert_numerical_paper_multistart_converges_to_baseline(
            &spec, 0.68, &starts, 1e-6, 1e-7, 5e-6,
        );
    }

    #[test]
    #[ignore = "research-grade slow gate; run explicitly"]
    fn numerical_paper_multistart_family_stays_stable_across_synthetic_tiers() {
        let report = build_numerical_paper_multistart_family_report();
        assert!(report.aggregate.all_tiers_within_tolerance);
        assert_json_matches_fixture(&report, &numerical_paper_multistart_family_report_fixture());
    }

    #[test]
    fn numerical_paper_multistart_family_report_markdown_matches_committed_artifact() {
        let report: NumericalPaperMultiStartFamilyReport = serde_json::from_str(
            &std::fs::read_to_string(numerical_paper_multistart_family_report_fixture()).unwrap(),
        )
        .unwrap();
        let actual = render_numerical_paper_multistart_family_report_markdown(&report);
        let expected =
            std::fs::read_to_string(numerical_paper_multistart_family_report_markdown_fixture())
                .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    #[ignore = "research-grade slow gate; run explicitly"]
    fn numerical_paper_multistart_mixed_family_stays_stable_across_literature_and_synthetic_tiers()
    {
        let report = build_numerical_paper_multistart_mixed_family_report();
        assert!(report.aggregate.all_tiers_within_tolerance);
        assert_json_matches_fixture(
            &report,
            &numerical_paper_multistart_mixed_family_report_fixture(),
        );
    }

    #[test]
    fn numerical_paper_multistart_mixed_family_report_markdown_matches_committed_artifact() {
        let report: NumericalPaperMultiStartFamilyReport = serde_json::from_str(
            &std::fs::read_to_string(numerical_paper_multistart_mixed_family_report_fixture())
                .unwrap(),
        )
        .unwrap();
        let actual = render_numerical_paper_multistart_family_report_markdown(&report);
        let expected = std::fs::read_to_string(
            numerical_paper_multistart_mixed_family_report_markdown_fixture(),
        )
        .unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn interpolate_profile_seed_blends_endpoint_vectors() {
        let inner = vec![1.0, 2.0];
        let outer = vec![5.0, 10.0];
        let seed = interpolate_profile_seed(0.25, 0.0, &inner, 1.0, &outer, 1.0, 0.0);

        assert_eq!(seed.len(), 2);
        assert!((seed[0] - 2.0).abs() < 1e-12);
        assert!((seed[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn interpolate_profile_seed_falls_back_when_span_is_zero() {
        let inner = vec![1.0, 2.0];
        let outer = vec![3.0, 4.0];
        let seed = interpolate_profile_seed(0.8, 1.0, &inner, 1.0, &outer, 1.0, 0.0);
        assert_eq!(seed, inner);
    }

    #[test]
    fn interpolate_profile_seed_falls_back_to_nearest_endpoint_near_edges() {
        let inner = vec![1.0, 2.0];
        let outer = vec![5.0, 6.0];
        assert_eq!(interpolate_profile_seed(0.05, 0.0, &inner, 1.0, &outer, 1.0, 0.0), inner);
        assert_eq!(interpolate_profile_seed(0.95, 0.0, &inner, 1.0, &outer, 1.0, 0.0), outer);
    }

    #[test]
    fn interpolate_profile_seed_allows_wider_span_for_low_error_on_error() {
        let inner = vec![0.0, 1.0];
        let outer = vec![10.0, 11.0];
        let low_eps_seed = interpolate_profile_seed(1.5, 0.0, &inner, 2.5, &outer, 1.0, 0.05);
        let high_eps_seed = interpolate_profile_seed(1.5, 0.0, &inner, 2.5, &outer, 1.0, 0.9);
        assert_eq!(low_eps_seed, vec![6.0, 7.0]);
        assert_eq!(high_eps_seed, outer);
    }

    #[test]
    fn select_profile_bound_candidate_uses_sqrt_space_on_quadratic_profile() {
        let candidate = select_profile_bound_candidate(1.0, 1.0, 3.0, 9.0, 4.0, 0.0);
        assert!((candidate - 2.0).abs() < 1e-12);
    }

    #[test]
    fn select_profile_bound_candidate_falls_back_to_midpoint_when_endpoint_values_match() {
        let candidate = select_profile_bound_candidate(0.0, 5.0, 10.0, 5.0, 5.0, 0.0);
        assert!((candidate - 5.0).abs() < 1e-12);
    }

    #[test]
    fn initial_profile_bracket_step_inflates_sigma_guess() {
        assert!((initial_profile_bracket_step(2.0, 0.5, 0.0) - 3.0).abs() < 1e-12);
        assert!((initial_profile_bracket_step(1e-6, 0.5, 0.0) - 1.5e-3).abs() < 1e-12);
    }

    #[test]
    fn profile_ci_mu_paper_workload_helper_matches_plain_interval_and_reports_effort() {
        let spec = synthetic_fast_path_spec(32, 24, 0.05);
        let prep = PreparedSpec::from_spec(&spec).unwrap();
        let objective = PaperMeasurementCombineObjective::new(&prep);
        let bounds = objective.bounds();
        let fit = numerical_paper_gvm_fit(&prep, &objective, &bounds, None).unwrap();
        let chi2_level = ChiSquared::new(1.0).unwrap().inverse_cdf(0.68);

        let plain = profile_ci_mu_paper(
            &objective,
            &bounds,
            &fit.fit.parameters,
            fit.fit.fval,
            chi2_level,
            None,
        )
        .unwrap();
        let (instrumented, workload) = profile_ci_mu_paper_with_config_and_workload(
            &objective,
            &bounds,
            &fit.fit.parameters,
            fit.fit.fval,
            chi2_level,
            None,
            paper_profile_scan_optimizer_config(),
        )
        .unwrap();

        assert_relative_eq!(plain.0, instrumented.0, epsilon = 1e-12, max_relative = 1e-12);
        assert_relative_eq!(plain.1, instrumented.1, epsilon = 1e-12, max_relative = 1e-12);
        assert!(workload.lower.n_profile_fits > 0);
        assert!(workload.upper.n_profile_fits > 0);
        assert_eq!(
            workload.lower.n_profile_fits,
            workload.lower.bracket_fits + workload.lower.bisect_fits
        );
        assert_eq!(
            workload.upper.n_profile_fits,
            workload.upper.bracket_fits + workload.upper.bisect_fits
        );
        assert!(workload.lower.total_n_fev > 0);
        assert!(workload.upper.total_n_fev > 0);
    }

    #[test]
    fn relaxed_paper_profile_scan_matches_high_precision_intervals_on_representative_cases() {
        for spec in
            [synthetic_fast_path_spec(32, 24, 0.05), full_literature_spec(Some(("b-JES", 0.05)))]
        {
            let prep = PreparedSpec::from_spec(&spec).unwrap();
            let objective = PaperMeasurementCombineObjective::new(&prep);
            let bounds = objective.bounds();
            let warm_start = objective.analytic_warm_start_guide(0.68);
            let fit =
                numerical_paper_gvm_fit(&prep, &objective, &bounds, warm_start.as_ref()).unwrap();
            let chi2_level = ChiSquared::new(1.0).unwrap().inverse_cdf(0.68);

            let (reference_interval, _) = profile_ci_mu_paper_with_config_and_workload(
                &objective,
                &bounds,
                &fit.fit.parameters,
                fit.fit.fval,
                chi2_level,
                warm_start.as_ref(),
                OptimizerConfig::from_strategy(crate::optimizer::OptimizerStrategy::HighPrecision),
            )
            .unwrap();
            let (relaxed_interval, _) = profile_ci_mu_paper_with_config_and_workload(
                &objective,
                &bounds,
                &fit.fit.parameters,
                fit.fit.fval,
                chi2_level,
                warm_start.as_ref(),
                paper_profile_scan_optimizer_config(),
            )
            .unwrap();

            assert_relative_eq!(
                reference_interval.0,
                relaxed_interval.0,
                epsilon = 1e-6,
                max_relative = 1e-6
            );
            assert_relative_eq!(
                reference_interval.1,
                relaxed_interval.1,
                epsilon = 1e-6,
                max_relative = 1e-6
            );
        }
    }

    #[test]
    #[ignore = "research-grade workload report; run explicitly"]
    fn profile_ci_mu_paper_workload_report_for_large_synthetic_cases() {
        for (label, n_measurements, n_systematics, epsilon) in
            [("32x24", 32usize, 24usize, 0.05), ("64x48", 64usize, 48usize, 0.05)]
        {
            let spec = synthetic_fast_path_spec(n_measurements, n_systematics, epsilon);
            let prep = PreparedSpec::from_spec(&spec).unwrap();
            let objective = PaperMeasurementCombineObjective::new(&prep);
            let bounds = objective.bounds();
            let fit = numerical_paper_gvm_fit(&prep, &objective, &bounds, None).unwrap();
            let chi2_level = ChiSquared::new(1.0).unwrap().inverse_cdf(0.68);
            let ((lo, hi), workload) = profile_ci_mu_paper_with_workload(
                &objective,
                &bounds,
                &fit.fit.parameters,
                fit.fit.fval,
                chi2_level,
                None,
            )
            .unwrap();

            eprintln!(
                "profile-ci workload {label}: interval=({lo:.6},{hi:.6}) lower={{fits:{}, bracket:{}, bisect:{}, n_iter:{}, n_fev:{}, n_gev:{}}} upper={{fits:{}, bracket:{}, bisect:{}, n_iter:{}, n_fev:{}, n_gev:{}}}",
                workload.lower.n_profile_fits,
                workload.lower.bracket_fits,
                workload.lower.bisect_fits,
                workload.lower.total_n_iter,
                workload.lower.total_n_fev,
                workload.lower.total_n_gev,
                workload.upper.n_profile_fits,
                workload.upper.bracket_fits,
                workload.upper.bisect_fits,
                workload.upper.total_n_iter,
                workload.upper.total_n_fev,
                workload.upper.total_n_gev,
            );
        }
    }

    #[test]
    #[ignore = "research-grade warm-start workload report; run explicitly"]
    fn profile_ci_mu_paper_warm_start_guide_workload_report_for_large_synthetic_case() {
        let spec = synthetic_fast_path_spec(64, 48, 0.05);
        let prep = PreparedSpec::from_spec(&spec).unwrap();
        let objective = PaperMeasurementCombineObjective::new(&prep);
        let bounds = objective.bounds();
        let fit_no_hint = numerical_paper_gvm_fit(&prep, &objective, &bounds, None).unwrap();
        let chi2_level = ChiSquared::new(1.0).unwrap().inverse_cdf(0.68);
        let (_, workload_no_hint) = profile_ci_mu_paper_with_workload(
            &objective,
            &bounds,
            &fit_no_hint.fit.parameters,
            fit_no_hint.fit.fval,
            chi2_level,
            None,
        )
        .unwrap();

        let warm_start = objective
            .analytic_warm_start_guide(0.68)
            .expect("expected analytic warm start guide for large synthetic case");
        let fit_with_hint =
            numerical_paper_gvm_fit(&prep, &objective, &bounds, Some(&warm_start)).unwrap();
        let (_, workload_with_hint) = profile_ci_mu_paper_with_workload(
            &objective,
            &bounds,
            &fit_with_hint.fit.parameters,
            fit_with_hint.fit.fval,
            chi2_level,
            Some(&warm_start),
        )
        .unwrap();

        eprintln!(
            "profile-ci warm-start workload 64x48: no_hint lower={{fits:{}, bracket:{}, bisect:{}, n_iter:{}, n_fev:{}, n_gev:{}}} upper={{fits:{}, bracket:{}, bisect:{}, n_iter:{}, n_fev:{}, n_gev:{}}}; with_hint lower={{fits:{}, bracket:{}, bisect:{}, n_iter:{}, n_fev:{}, n_gev:{}}} upper={{fits:{}, bracket:{}, bisect:{}, n_iter:{}, n_fev:{}, n_gev:{}}}",
            workload_no_hint.lower.n_profile_fits,
            workload_no_hint.lower.bracket_fits,
            workload_no_hint.lower.bisect_fits,
            workload_no_hint.lower.total_n_iter,
            workload_no_hint.lower.total_n_fev,
            workload_no_hint.lower.total_n_gev,
            workload_no_hint.upper.n_profile_fits,
            workload_no_hint.upper.bracket_fits,
            workload_no_hint.upper.bisect_fits,
            workload_no_hint.upper.total_n_iter,
            workload_no_hint.upper.total_n_fev,
            workload_no_hint.upper.total_n_gev,
            workload_with_hint.lower.n_profile_fits,
            workload_with_hint.lower.bracket_fits,
            workload_with_hint.lower.bisect_fits,
            workload_with_hint.lower.total_n_iter,
            workload_with_hint.lower.total_n_fev,
            workload_with_hint.lower.total_n_gev,
            workload_with_hint.upper.n_profile_fits,
            workload_with_hint.upper.bracket_fits,
            workload_with_hint.upper.bisect_fits,
            workload_with_hint.upper.total_n_iter,
            workload_with_hint.upper.total_n_fev,
            workload_with_hint.upper.total_n_gev,
        );
    }

    #[test]
    #[ignore = "research-grade toy warm-start workload report; run explicitly"]
    fn profile_ci_mu_paper_template_warm_start_workload_report_for_large_synthetic_toy_case() {
        let spec = synthetic_fast_path_spec(64, 48, 0.05);
        let context = build_measurement_toy_generation_context(
            &spec,
            MeasurementCombinationSolver::NumericalPaper,
        )
        .unwrap();
        let prep = simulate_gvm_toy_prepared_spec_from_context(&context, 2026);
        let objective = PaperMeasurementCombineObjective::new(&prep);
        let bounds = objective.bounds();
        let chi2_level = ChiSquared::new(1.0).unwrap().inverse_cdf(0.68);

        let fit_no_hint = numerical_paper_gvm_fit(&prep, &objective, &bounds, None).unwrap();
        let (_, workload_no_hint) = profile_ci_mu_paper_with_workload(
            &objective,
            &bounds,
            &fit_no_hint.fit.parameters,
            fit_no_hint.fit.fval,
            chi2_level,
            None,
        )
        .unwrap();

        let preferred = context
            .paper_warm_start
            .as_ref()
            .expect("expected template warm-start for numerical-paper toy context");
        let fit_with_hint =
            numerical_paper_gvm_fit(&prep, &objective, &bounds, Some(preferred)).unwrap();
        let (_, workload_with_hint) = profile_ci_mu_paper_with_workload(
            &objective,
            &bounds,
            &fit_with_hint.fit.parameters,
            fit_with_hint.fit.fval,
            chi2_level,
            Some(preferred),
        )
        .unwrap();

        eprintln!(
            "profile-ci template warm-start workload 64x48 toy: no_hint lower={{fits:{}, bracket:{}, bisect:{}, n_iter:{}, n_fev:{}, n_gev:{}}} upper={{fits:{}, bracket:{}, bisect:{}, n_iter:{}, n_fev:{}, n_gev:{}}}; with_hint lower={{fits:{}, bracket:{}, bisect:{}, n_iter:{}, n_fev:{}, n_gev:{}}} upper={{fits:{}, bracket:{}, bisect:{}, n_iter:{}, n_fev:{}, n_gev:{}}}",
            workload_no_hint.lower.n_profile_fits,
            workload_no_hint.lower.bracket_fits,
            workload_no_hint.lower.bisect_fits,
            workload_no_hint.lower.total_n_iter,
            workload_no_hint.lower.total_n_fev,
            workload_no_hint.lower.total_n_gev,
            workload_no_hint.upper.n_profile_fits,
            workload_no_hint.upper.bracket_fits,
            workload_no_hint.upper.bisect_fits,
            workload_no_hint.upper.total_n_iter,
            workload_no_hint.upper.total_n_fev,
            workload_no_hint.upper.total_n_gev,
            workload_with_hint.lower.n_profile_fits,
            workload_with_hint.lower.bracket_fits,
            workload_with_hint.lower.bisect_fits,
            workload_with_hint.lower.total_n_iter,
            workload_with_hint.lower.total_n_fev,
            workload_with_hint.lower.total_n_gev,
            workload_with_hint.upper.n_profile_fits,
            workload_with_hint.upper.bracket_fits,
            workload_with_hint.upper.bisect_fits,
            workload_with_hint.upper.total_n_iter,
            workload_with_hint.upper.total_n_fev,
            workload_with_hint.upper.total_n_gev,
        );
    }

    #[test]
    #[ignore = "research-grade toy warm-start workload report; run explicitly"]
    fn profile_ci_mu_paper_upper_only_template_warm_start_workload_report_for_large_synthetic_toy_case()
     {
        let spec = synthetic_fast_path_spec(64, 48, 0.05);
        let context = build_measurement_toy_generation_context(
            &spec,
            MeasurementCombinationSolver::NumericalPaper,
        )
        .unwrap();
        let prep = simulate_gvm_toy_prepared_spec_from_context(&context, 2026);
        let objective = PaperMeasurementCombineObjective::new(&prep);
        let bounds = objective.bounds();
        let chi2_level = ChiSquared::new(1.0).unwrap().inverse_cdf(0.68);

        let fit_no_hint = numerical_paper_gvm_fit(&prep, &objective, &bounds, None).unwrap();
        let (_, workload_no_hint) = profile_ci_mu_paper_with_workload(
            &objective,
            &bounds,
            &fit_no_hint.fit.parameters,
            fit_no_hint.fit.fval,
            chi2_level,
            None,
        )
        .unwrap();

        let full_guide = context
            .paper_warm_start
            .as_ref()
            .expect("expected template warm-start for numerical-paper toy context");
        let upper_only = PaperWarmStartGuide {
            mle_params: full_guide.mle_params.clone(),
            lower_hint: None,
            upper_hint: full_guide.upper_hint.clone(),
        };
        let fit_upper_only =
            numerical_paper_gvm_fit(&prep, &objective, &bounds, Some(&upper_only)).unwrap();
        let (_, workload_upper_only) = profile_ci_mu_paper_with_workload(
            &objective,
            &bounds,
            &fit_upper_only.fit.parameters,
            fit_upper_only.fit.fval,
            chi2_level,
            Some(&upper_only),
        )
        .unwrap();

        eprintln!(
            "profile-ci upper-only template warm-start workload 64x48 toy: no_hint lower={{fits:{}, bracket:{}, bisect:{}, n_iter:{}, n_fev:{}, n_gev:{}}} upper={{fits:{}, bracket:{}, bisect:{}, n_iter:{}, n_fev:{}, n_gev:{}}}; upper_only lower={{fits:{}, bracket:{}, bisect:{}, n_iter:{}, n_fev:{}, n_gev:{}}} upper={{fits:{}, bracket:{}, bisect:{}, n_iter:{}, n_fev:{}, n_gev:{}}}",
            workload_no_hint.lower.n_profile_fits,
            workload_no_hint.lower.bracket_fits,
            workload_no_hint.lower.bisect_fits,
            workload_no_hint.lower.total_n_iter,
            workload_no_hint.lower.total_n_fev,
            workload_no_hint.lower.total_n_gev,
            workload_no_hint.upper.n_profile_fits,
            workload_no_hint.upper.bracket_fits,
            workload_no_hint.upper.bisect_fits,
            workload_no_hint.upper.total_n_iter,
            workload_no_hint.upper.total_n_fev,
            workload_no_hint.upper.total_n_gev,
            workload_upper_only.lower.n_profile_fits,
            workload_upper_only.lower.bracket_fits,
            workload_upper_only.lower.bisect_fits,
            workload_upper_only.lower.total_n_iter,
            workload_upper_only.lower.total_n_fev,
            workload_upper_only.lower.total_n_gev,
            workload_upper_only.upper.n_profile_fits,
            workload_upper_only.upper.bracket_fits,
            workload_upper_only.upper.bisect_fits,
            workload_upper_only.upper.total_n_iter,
            workload_upper_only.upper.total_n_fev,
            workload_upper_only.upper.total_n_gev,
        );
    }

    #[test]
    #[ignore = "research-grade toy timing report; run explicitly"]
    fn numerical_paper_toy_timing_report_with_and_without_template_warm_start() {
        let spec = synthetic_fast_path_spec(64, 48, 0.05);
        let context = build_measurement_toy_generation_context(
            &spec,
            MeasurementCombinationSolver::NumericalPaper,
        )
        .unwrap();
        let prep = simulate_gvm_toy_prepared_spec_from_context(&context, 2026);
        let full_guide = context
            .paper_warm_start
            .as_ref()
            .expect("expected template warm-start for numerical-paper toy context");
        let upper_only = PaperWarmStartGuide {
            mle_params: full_guide.mle_params.clone(),
            lower_hint: None,
            upper_hint: full_guide.upper_hint.clone(),
        };

        let mut no_hint_runs = Vec::new();
        let mut full_hint_runs = Vec::new();
        let mut upper_only_runs = Vec::new();
        for _ in 0..3 {
            let started = std::time::Instant::now();
            let out = numerical_paper_gvm_calibration_toy_result_with_warm_start(
                prep.clone(),
                0.68,
                None,
            )
            .unwrap();
            no_hint_runs.push((started.elapsed(), out.sigma_star));

            let started = std::time::Instant::now();
            let out = numerical_paper_gvm_calibration_toy_result_with_warm_start(
                prep.clone(),
                0.68,
                Some(full_guide),
            )
            .unwrap();
            full_hint_runs.push((started.elapsed(), out.sigma_star));

            let started = std::time::Instant::now();
            let out = numerical_paper_gvm_calibration_toy_result_with_warm_start(
                prep.clone(),
                0.68,
                Some(&upper_only),
            )
            .unwrap();
            upper_only_runs.push((started.elapsed(), out.sigma_star));
        }

        eprintln!(
            "numerical-paper toy timing 64x48: no_hint={:?}; full_hint={:?}; upper_only={:?}",
            no_hint_runs, full_hint_runs, upper_only_runs
        );
    }

    #[test]
    #[ignore = "research-grade fit workload report; run explicitly"]
    fn numerical_paper_fit_workload_report_with_and_without_template_warm_start() {
        let spec = synthetic_fast_path_spec(64, 48, 0.05);
        let context = build_measurement_toy_generation_context(
            &spec,
            MeasurementCombinationSolver::NumericalPaper,
        )
        .unwrap();
        let prep = simulate_gvm_toy_prepared_spec_from_context(&context, 2026);
        let objective = PaperMeasurementCombineObjective::new(&prep);
        let bounds = objective.bounds();
        let full_guide = context
            .paper_warm_start
            .as_ref()
            .expect("expected template warm-start for numerical-paper toy context");
        let upper_only = PaperWarmStartGuide {
            mle_params: full_guide.mle_params.clone(),
            lower_hint: None,
            upper_hint: full_guide.upper_hint.clone(),
        };

        let fit_no_hint = numerical_paper_gvm_fit(&prep, &objective, &bounds, None).unwrap();
        let fit_full_hint =
            numerical_paper_gvm_fit(&prep, &objective, &bounds, Some(full_guide)).unwrap();
        let fit_upper_only =
            numerical_paper_gvm_fit(&prep, &objective, &bounds, Some(&upper_only)).unwrap();

        eprintln!(
            "numerical-paper fit workload 64x48 toy: no_hint={{n_iter:{}, n_fev:{}, n_gev:{}, fval:{:.9}}}; full_hint={{n_iter:{}, n_fev:{}, n_gev:{}, fval:{:.9}}}; upper_only={{n_iter:{}, n_fev:{}, n_gev:{}, fval:{:.9}}}",
            fit_no_hint.fit.n_iter,
            fit_no_hint.fit.n_fev,
            fit_no_hint.fit.n_gev,
            fit_no_hint.fit.fval,
            fit_full_hint.fit.n_iter,
            fit_full_hint.fit.n_fev,
            fit_full_hint.fit.n_gev,
            fit_full_hint.fit.fval,
            fit_upper_only.fit.n_iter,
            fit_upper_only.fit.n_fev,
            fit_upper_only.fit.n_gev,
            fit_upper_only.fit.fval,
        );
    }

    #[test]
    #[ignore = "research-grade direct timing report; run explicitly"]
    fn numerical_paper_direct_timing_report_with_and_without_analytic_warm_start() {
        let spec = synthetic_fast_path_spec(64, 48, 0.05);
        let prep = PreparedSpec::from_spec(&spec).unwrap();
        let objective = PaperMeasurementCombineObjective::new(&prep);
        let bounds = objective.bounds();
        let chi2_level = ChiSquared::new(1.0).unwrap().inverse_cdf(0.68);

        let started = std::time::Instant::now();
        let current_default =
            numerical_paper_gvm_result_with_warm_start(prep.clone(), 0.68, None).unwrap();
        let current_default_elapsed = started.elapsed();

        let started = std::time::Instant::now();
        let fit_no_hint = numerical_paper_gvm_fit(&prep, &objective, &bounds, None).unwrap();
        let (_, workload_no_hint) = profile_ci_mu_paper_with_workload(
            &objective,
            &bounds,
            &fit_no_hint.fit.parameters,
            fit_no_hint.fit.fval,
            chi2_level,
            None,
        )
        .unwrap();
        let no_hint_elapsed = started.elapsed();

        let started = std::time::Instant::now();
        let guide = objective
            .analytic_warm_start_guide(0.68)
            .expect("expected analytic warm-start guide for large synthetic case");
        let guide_elapsed = started.elapsed();
        let started = std::time::Instant::now();
        let fit_with_hint =
            numerical_paper_gvm_fit(&prep, &objective, &bounds, Some(&guide)).unwrap();
        let (_, workload_with_hint) = profile_ci_mu_paper_with_workload(
            &objective,
            &bounds,
            &fit_with_hint.fit.parameters,
            fit_with_hint.fit.fval,
            chi2_level,
            Some(&guide),
        )
        .unwrap();
        let with_hint_elapsed = started.elapsed();

        eprintln!(
            "numerical-paper direct timing 64x48: current_default={{elapsed:{:?}, sigma_star:{:.12}}}; no_hint_fit_ci={{elapsed:{:?}, fit_n_iter:{}, fit_n_fev:{}, fit_n_gev:{}, lower_n_gev:{}, upper_n_gev:{}}}; analytic_guide={{elapsed:{:?}}}; guided_fit_ci={{elapsed:{:?}, fit_n_iter:{}, fit_n_fev:{}, fit_n_gev:{}, lower_n_gev:{}, upper_n_gev:{}}}",
            current_default_elapsed,
            current_default.diagnostics.bartlett.sigma_star.unwrap(),
            no_hint_elapsed,
            fit_no_hint.fit.n_iter,
            fit_no_hint.fit.n_fev,
            fit_no_hint.fit.n_gev,
            workload_no_hint.lower.total_n_gev,
            workload_no_hint.upper.total_n_gev,
            guide_elapsed,
            with_hint_elapsed,
            fit_with_hint.fit.n_iter,
            fit_with_hint.fit.n_fev,
            fit_with_hint.fit.n_gev,
            workload_with_hint.lower.total_n_gev,
            workload_with_hint.upper.total_n_gev,
        );
    }

    #[test]
    fn symmetric_pseudoinverse_matches_inverse_for_positive_definite_matrix() {
        let matrix = DMatrix::from_row_slice(
            3,
            3,
            &[
                4.0, 1.0, 0.5, //
                1.0, 3.0, 0.25, //
                0.5, 0.25, 2.0,
            ],
        );
        let inv = symmetric_pseudoinverse(&matrix).unwrap();
        let ident = &matrix * inv;
        for i in 0..ident.nrows() {
            for j in 0..ident.ncols() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(ident[(i, j)], expected, epsilon = 1e-10, max_relative = 1e-9);
            }
        }
    }

    #[test]
    fn symmetric_pseudoinverse_retains_zero_modes_for_semidefinite_matrix() {
        let matrix = DMatrix::from_row_slice(
            2,
            2,
            &[
                1.0, 1.0, //
                1.0, 1.0,
            ],
        );
        let pinv = symmetric_pseudoinverse(&matrix).unwrap();
        assert_relative_eq!(pinv[(0, 0)], 0.25, epsilon = 1e-12);
        assert_relative_eq!(pinv[(0, 1)], 0.25, epsilon = 1e-12);
        assert_relative_eq!(pinv[(1, 0)], 0.25, epsilon = 1e-12);
        assert_relative_eq!(pinv[(1, 1)], 0.25, epsilon = 1e-12);
    }

    #[test]
    fn symmetric_pseudoinverse_handles_near_singular_positive_definite_matrix() {
        let eps = 1e-8;
        let matrix = DMatrix::from_row_slice(
            2,
            2,
            &[
                1.0,
                1.0 - eps, //
                1.0 - eps,
                1.0,
            ],
        );
        let inv = symmetric_pseudoinverse(&matrix).unwrap();
        let ident = &matrix * inv;
        for i in 0..ident.nrows() {
            for j in 0..ident.ncols() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(ident[(i, j)], expected, epsilon = 1e-6, max_relative = 1e-6);
            }
        }
    }

    #[test]
    fn scenario_study_solver_parity_is_tight_for_full_literature_low_epsilon_case() {
        let scenario_study = MeasurementCombinationScenarioStudySpec {
            schema_version: MEASUREMENT_COMBINATION_SCENARIO_CONFIG_SCHEMA_V0.to_string(),
            scenarios: vec![MeasurementCombinationScenarioSpec {
                name: "bjes_0p05".to_string(),
                error_on_error: vec![ScenarioErrorOnErrorAssignment {
                    systematic: "b-JES".to_string(),
                    value: 0.05,
                }],
            }],
        };
        let report = compare_measurement_combination_scenario_study_solvers(
            &full_literature_spec(None),
            &scenario_study,
            0.68,
            MeasurementCombinationSolver::NumericalPaper,
            MeasurementCombinationSolver::AnalyticPerturbative,
        )
        .unwrap();

        assert_eq!(
            report.schema_version,
            MEASUREMENT_COMBINATION_SCENARIO_STUDY_SOLVER_PARITY_SCHEMA_V0
        );
        assert_eq!(report.lhs_solver, "numerical-paper");
        assert_eq!(report.rhs_solver, "analytic-perturbative");
        assert_eq!(report.aggregate.n_scenarios, 1);
        assert!(report.aggregate.all_scenarios_converged);
        assert!(report.aggregate.max_mu_abs_diff < 1e-2);
        assert!(report.aggregate.max_sigma_rel_diff < 5e-2);
    }

    #[test]
    fn calibration_campaign_solver_parity_is_tight_for_full_literature_low_epsilon_case() {
        let scenario_study = MeasurementCombinationScenarioStudySpec {
            schema_version: MEASUREMENT_COMBINATION_SCENARIO_CONFIG_SCHEMA_V0.to_string(),
            scenarios: vec![MeasurementCombinationScenarioSpec {
                name: "bjes_0p05".to_string(),
                error_on_error: vec![ScenarioErrorOnErrorAssignment {
                    systematic: "b-JES".to_string(),
                    value: 0.05,
                }],
            }],
        };
        let report = compare_measurement_combination_calibration_campaign_solvers(
            &full_literature_spec(None),
            &scenario_study,
            0.68,
            MeasurementCombinationSolver::NumericalPaper,
            MeasurementCombinationSolver::AnalyticPerturbative,
            8,
            &[2026, 2027],
        )
        .unwrap();

        assert_eq!(
            report.schema_version,
            MEASUREMENT_COMBINATION_CALIBRATION_CAMPAIGN_SOLVER_PARITY_SCHEMA_V0
        );
        assert_eq!(report.lhs_solver, "numerical-paper");
        assert_eq!(report.rhs_solver, "analytic-perturbative");
        assert_eq!(report.aggregate.n_scenarios, 1);
        assert!(report.aggregate.all_scenarios_converged);
        assert!(report.aggregate.max_fit_mu_abs_diff < 1e-2);
        assert!(report.aggregate.max_fit_sigma_rel_diff < 5e-2);
        assert!(report.aggregate.max_calibration_ratio_center_abs_diff < 5e-2);
    }

    #[test]
    fn scenario_study_solver_parity_fixture_matches_committed_artifact() {
        let scenario_study = MeasurementCombinationScenarioStudySpec {
            schema_version: MEASUREMENT_COMBINATION_SCENARIO_CONFIG_SCHEMA_V0.to_string(),
            scenarios: vec![MeasurementCombinationScenarioSpec {
                name: "bjes_0p05".to_string(),
                error_on_error: vec![ScenarioErrorOnErrorAssignment {
                    systematic: "b-JES".to_string(),
                    value: 0.05,
                }],
            }],
        };
        let report = compare_measurement_combination_scenario_study_solvers(
            &full_literature_spec(None),
            &scenario_study,
            0.68,
            MeasurementCombinationSolver::NumericalPaper,
            MeasurementCombinationSolver::AnalyticPerturbative,
        )
        .unwrap();
        assert_json_matches_fixture(&report, &scenario_solver_parity_report_fixture());
    }

    #[test]
    fn scenario_study_solver_parity_reports_match_direct_compare_path() {
        let scenario_study = MeasurementCombinationScenarioStudySpec {
            schema_version: MEASUREMENT_COMBINATION_SCENARIO_CONFIG_SCHEMA_V0.to_string(),
            scenarios: vec![MeasurementCombinationScenarioSpec {
                name: "bjes_0p05".to_string(),
                error_on_error: vec![ScenarioErrorOnErrorAssignment {
                    systematic: "b-JES".to_string(),
                    value: 0.05,
                }],
            }],
        };
        let lhs = study_measurement_combination_scenarios_with_solver(
            &full_literature_spec(None),
            &scenario_study,
            0.68,
            MeasurementCombinationSolver::NumericalPaper,
        )
        .unwrap();
        let rhs = study_measurement_combination_scenarios_with_solver(
            &full_literature_spec(None),
            &scenario_study,
            0.68,
            MeasurementCombinationSolver::AnalyticPerturbative,
        )
        .unwrap();
        let from_reports = compare_measurement_combination_scenario_study_reports(
            &lhs,
            &rhs,
            "numerical-paper",
            "analytic-perturbative",
        )
        .unwrap();
        let direct = compare_measurement_combination_scenario_study_solvers(
            &full_literature_spec(None),
            &scenario_study,
            0.68,
            MeasurementCombinationSolver::NumericalPaper,
            MeasurementCombinationSolver::AnalyticPerturbative,
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(&from_reports).unwrap(),
            serde_json::to_value(&direct).unwrap()
        );
    }

    #[test]
    fn scenario_study_solver_parity_markdown_matches_committed_artifact() {
        let report: MeasurementCombinationScenarioStudySolverParityReport = serde_json::from_str(
            &std::fs::read_to_string(scenario_solver_parity_report_fixture()).unwrap(),
        )
        .unwrap();
        let markdown =
            render_measurement_combination_scenario_study_solver_parity_markdown(&report).unwrap();
        let expected = std::fs::read_to_string(scenario_solver_parity_markdown_fixture()).unwrap();
        assert_eq!(markdown, expected);
    }

    #[test]
    fn scenario_study_solver_parity_digest_matches_committed_artifact() {
        let report: MeasurementCombinationScenarioStudySolverParityReport = serde_json::from_str(
            &std::fs::read_to_string(scenario_solver_parity_report_fixture()).unwrap(),
        )
        .unwrap();
        let digest =
            summarize_measurement_combination_scenario_study_solver_parity(&report).unwrap();
        assert_json_matches_fixture(&digest, &scenario_solver_parity_digest_fixture());
    }

    #[test]
    fn scenario_study_solver_parity_digest_markdown_matches_committed_artifact() {
        let digest: MeasurementCombinationScenarioStudySolverParityDigest = serde_json::from_str(
            &std::fs::read_to_string(scenario_solver_parity_digest_fixture()).unwrap(),
        )
        .unwrap();
        let markdown =
            render_measurement_combination_scenario_study_solver_parity_digest_markdown(&digest)
                .unwrap();
        let expected =
            std::fs::read_to_string(scenario_solver_parity_digest_markdown_fixture()).unwrap();
        assert_eq!(markdown, expected);
    }

    #[test]
    fn calibration_campaign_solver_parity_fixture_matches_committed_artifact() {
        let scenario_study = MeasurementCombinationScenarioStudySpec {
            schema_version: MEASUREMENT_COMBINATION_SCENARIO_CONFIG_SCHEMA_V0.to_string(),
            scenarios: vec![MeasurementCombinationScenarioSpec {
                name: "bjes_0p05".to_string(),
                error_on_error: vec![ScenarioErrorOnErrorAssignment {
                    systematic: "b-JES".to_string(),
                    value: 0.05,
                }],
            }],
        };
        let report = compare_measurement_combination_calibration_campaign_solvers(
            &full_literature_spec(None),
            &scenario_study,
            0.68,
            MeasurementCombinationSolver::NumericalPaper,
            MeasurementCombinationSolver::AnalyticPerturbative,
            8,
            &[2026, 2027],
        )
        .unwrap();
        assert_json_matches_fixture(&report, &calibration_campaign_solver_parity_report_fixture());
    }

    #[test]
    fn calibration_campaign_solver_parity_reports_match_direct_compare_path() {
        let scenario_study = MeasurementCombinationScenarioStudySpec {
            schema_version: MEASUREMENT_COMBINATION_SCENARIO_CONFIG_SCHEMA_V0.to_string(),
            scenarios: vec![MeasurementCombinationScenarioSpec {
                name: "bjes_0p05".to_string(),
                error_on_error: vec![ScenarioErrorOnErrorAssignment {
                    systematic: "b-JES".to_string(),
                    value: 0.05,
                }],
            }],
        };
        let lhs = run_measurement_combination_calibration_campaign_with_solver(
            &full_literature_spec(None),
            &scenario_study,
            0.68,
            MeasurementCombinationSolver::NumericalPaper,
            8,
            &[2026, 2027],
        )
        .unwrap();
        let rhs = run_measurement_combination_calibration_campaign_with_solver(
            &full_literature_spec(None),
            &scenario_study,
            0.68,
            MeasurementCombinationSolver::AnalyticPerturbative,
            8,
            &[2026, 2027],
        )
        .unwrap();
        let from_reports = compare_measurement_combination_calibration_campaign_reports(
            &lhs,
            &rhs,
            "numerical-paper",
            "analytic-perturbative",
        )
        .unwrap();
        let direct = compare_measurement_combination_calibration_campaign_solvers(
            &full_literature_spec(None),
            &scenario_study,
            0.68,
            MeasurementCombinationSolver::NumericalPaper,
            MeasurementCombinationSolver::AnalyticPerturbative,
            8,
            &[2026, 2027],
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(&from_reports).unwrap(),
            serde_json::to_value(&direct).unwrap()
        );
    }

    #[test]
    fn calibration_campaign_solver_parity_markdown_matches_committed_artifact() {
        let report: MeasurementCombinationCalibrationCampaignSolverParityReport =
            serde_json::from_str(
                &std::fs::read_to_string(calibration_campaign_solver_parity_report_fixture())
                    .unwrap(),
            )
            .unwrap();
        let markdown =
            render_measurement_combination_calibration_campaign_solver_parity_markdown(&report)
                .unwrap();
        let expected =
            std::fs::read_to_string(calibration_campaign_solver_parity_markdown_fixture()).unwrap();
        assert_eq!(markdown, expected);
    }

    #[test]
    fn calibration_campaign_solver_parity_digest_matches_committed_artifact() {
        let report: MeasurementCombinationCalibrationCampaignSolverParityReport =
            serde_json::from_str(
                &std::fs::read_to_string(calibration_campaign_solver_parity_report_fixture())
                    .unwrap(),
            )
            .unwrap();
        let digest =
            summarize_measurement_combination_calibration_campaign_solver_parity(&report).unwrap();
        assert_json_matches_fixture(&digest, &calibration_campaign_solver_parity_digest_fixture());
    }

    #[test]
    fn calibration_campaign_solver_parity_digest_markdown_matches_committed_artifact() {
        let digest: MeasurementCombinationCalibrationCampaignSolverParityDigest =
            serde_json::from_str(
                &std::fs::read_to_string(calibration_campaign_solver_parity_digest_fixture())
                    .unwrap(),
            )
            .unwrap();
        let markdown =
            render_measurement_combination_calibration_campaign_solver_parity_digest_markdown(
                &digest,
            )
            .unwrap();
        let expected =
            std::fs::read_to_string(calibration_campaign_solver_parity_digest_markdown_fixture())
                .unwrap();
        assert_eq!(markdown, expected);
    }

    #[test]
    fn calibration_campaign_summary_returns_research_grade_report() {
        let report = load_calibration_campaign_report(
            &scenario_outlier_calibration_campaign_report_fixture(),
        );
        let summary = summarize_measurement_combination_calibration_campaign(&report).unwrap();

        assert_eq!(
            summary.schema_version,
            "nextstat_measurement_combination_calibration_campaign_summary_v0"
        );
        assert_eq!(summary.source_schema_version, report.schema_version);
        assert_eq!(summary.stability, "research-grade");
        assert_eq!(summary.aggregate.n_scenarios, 3);
        assert_eq!(summary.scenarios.len(), 3);
        assert_eq!(summary.dominant_calibration_scenario, "new_0p5");
        assert!(summary.baseline_sigma.is_finite());
        assert!(summary.aggregate.max_calibration_mean_sigma_star_to_sigma_ratio.is_finite());
    }

    #[test]
    fn calibration_fixture_matches_committed_artifact() {
        let spec = calibration_outlier_fixture_spec();
        let report = calibrate_measurements_toys(&spec, 0.68, 16, 123).unwrap();

        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.pop();
        path.pop();
        path.push("tests/fixtures/measurement_combine_calibration_outlier_report.json");
        assert_json_matches_fixture(&report, &path);
    }

    #[test]
    fn calibration_study_fixture_matches_committed_artifact() {
        let report = calibrate_measurements_toys_study(
            &calibration_outlier_fixture_spec(),
            0.68,
            16,
            &[123, 124, 125],
        )
        .unwrap();
        assert_json_matches_fixture(&report, &calibration_outlier_study_report_fixture());
    }

    #[test]
    fn scenario_study_fixture_matches_committed_artifact() {
        let report = study_measurement_combination_scenarios(
            &calibration_outlier_fixture_spec(),
            &outlier_scenario_study_spec(),
            0.68,
        )
        .unwrap();
        assert_json_matches_fixture(&report, &scenario_outlier_study_report_fixture());
    }

    #[test]
    fn calibration_campaign_fixture_matches_committed_artifact() {
        let report = run_measurement_combination_calibration_campaign(
            &calibration_outlier_fixture_spec(),
            &outlier_scenario_study_spec(),
            0.68,
            16,
            &[123, 124, 125],
        )
        .unwrap();
        assert_json_matches_fixture(
            &report,
            &scenario_outlier_calibration_campaign_report_fixture(),
        );
    }

    #[test]
    fn calibration_campaign_summary_fixture_matches_committed_artifact() {
        let report = load_calibration_campaign_report(
            &scenario_outlier_calibration_campaign_report_fixture(),
        );
        let summary = summarize_measurement_combination_calibration_campaign(&report).unwrap();
        assert_json_matches_fixture(
            &summary,
            &scenario_outlier_calibration_campaign_summary_fixture(),
        );
    }

    #[test]
    fn calibration_campaign_summary_markdown_matches_committed_artifact() {
        let report = load_calibration_campaign_report(
            &scenario_outlier_calibration_campaign_report_fixture(),
        );
        let summary = summarize_measurement_combination_calibration_campaign(&report).unwrap();
        let markdown =
            render_measurement_combination_calibration_campaign_digest_markdown(&summary).unwrap();
        let expected = std::fs::read_to_string(
            scenario_outlier_calibration_campaign_summary_markdown_fixture(),
        )
        .unwrap();
        assert_eq!(markdown, expected);
    }

    #[test]
    #[ignore = "research-grade slow gate; run explicitly"]
    fn full_literature_bjes_calibration_report_has_stable_research_metrics() {
        let report = calibrate_measurements_toys(
            &full_literature_spec(Some(("b-JES", 0.5))),
            0.68,
            32,
            2026,
        )
        .unwrap();
        assert_eq!(report.summary.df, 14);
        assert!(report.summary.mean_q.is_finite());
        assert!(report.summary.mean_q_star.is_finite());
        assert!(report.summary.mean_sigma.is_finite());
        assert!(report.summary.mean_sigma_star.is_finite());
        assert!(report.summary.mean_sigma_star_to_sigma_ratio > 0.0);
        assert!(report.summary.sigma_star_ge_sigma_fraction >= 0.0);
        assert!(report.summary.sigma_star_ge_sigma_fraction <= 1.0);
        assert!(report.reference.diagnostics.bartlett.supported);
        assert!(report.summary.mean_sigma_star > report.summary.mean_sigma);
        assert!(report.summary.mean_sigma_star_to_sigma_ratio > 1.01);
        assert!(report.summary.mean_sigma_star_to_sigma_ratio < 1.013);
        assert!(report.summary.sigma_star_ge_sigma_fraction >= 0.99);
    }

    #[test]
    #[ignore = "research-grade slow gate; run explicitly"]
    fn full_literature_bjes_calibration_fixture_matches_committed_artifact() {
        let report = calibrate_measurements_toys(
            &full_literature_spec(Some(("b-JES", 0.5))),
            0.68,
            32,
            2026,
        )
        .unwrap();
        assert_json_matches_fixture(&report, &full_literature_calibration_report_fixture());
    }

    #[test]
    #[ignore = "research-grade slow gate; run explicitly"]
    fn full_literature_bjes_calibration_sigma_star_stability_holds_across_seeds() {
        let study = calibrate_measurements_toys_study(
            &full_literature_spec(Some(("b-JES", 0.5))),
            0.68,
            32,
            &[2026, 2027, 2028, 2029, 2030, 2031],
        )
        .unwrap();

        assert!(study.reference.diagnostics.bartlett.supported);
        assert!(study.aggregate.min_sigma_star_ge_sigma_fraction >= 0.99);
        assert!(study.aggregate.min_mean_sigma_star_to_sigma_ratio > 1.01);
        assert!(study.aggregate.max_mean_sigma_star_to_sigma_ratio < 1.013);
        assert!(
            study.aggregate.max_abs_mean_sigma_star_to_sigma_ratio_delta_from_reference <= 1.5e-4,
            "expected calibration sigma inflation to track reference Bartlett sigma_scale"
        );
        assert!(
            study.aggregate.max_mean_sigma - study.aggregate.min_mean_sigma <= 8e-4,
            "unexpected mean_sigma drift across seeds"
        );
        assert!(
            study.aggregate.max_mean_sigma_star - study.aggregate.min_mean_sigma_star <= 9e-4,
            "unexpected mean_sigma_star drift across seeds"
        );
        assert!(
            study.aggregate.max_mean_sigma_star_to_sigma_ratio
                - study.aggregate.min_mean_sigma_star_to_sigma_ratio
                <= 1e-4,
            "unexpected sigma inflation ratio drift across seeds"
        );
    }

    #[test]
    #[ignore = "research-grade slow gate; run explicitly"]
    fn full_literature_multi_systematic_scenario_study_has_stable_ordering() {
        let report = study_measurement_combination_scenarios(
            &full_literature_spec(None),
            &full_literature_scenario_study_spec(),
            0.68,
        )
        .unwrap();

        assert_eq!(report.aggregate.n_scenarios, 3);
        assert!(report.aggregate.all_converged);
        assert!(report.aggregate.all_perturbative_within_threshold);
        assert_eq!(report.aggregate.widest_interval_scenario, "bjes_0p5");
        assert_eq!(report.aggregate.max_supported_systematics, 7);
        assert!(report.aggregate.min_sigma_ratio_to_baseline > 1.0);
        assert!(report.aggregate.max_sigma_ratio_to_baseline < 1.05);

        let theory = report
            .scenarios
            .iter()
            .find(|scenario| scenario.name == "theory_core_0p5")
            .expect("theory scenario should be present");
        let jes = report
            .scenarios
            .iter()
            .find(|scenario| scenario.name == "jes_family_0p5")
            .expect("jes scenario should be present");
        assert_eq!(theory.comparison.supported_systematics.len(), 4);
        assert_eq!(jes.comparison.supported_systematics.len(), 7);
        assert!(theory.comparison.max_perturbative_condition.unwrap() < 0.3);
        assert!(jes.comparison.max_perturbative_condition.unwrap() < 0.5);
    }

    #[test]
    #[ignore = "research-grade slow gate; run explicitly"]
    fn full_literature_multi_systematic_calibration_campaign_has_stable_sigma_envelope() {
        let report = run_measurement_combination_calibration_campaign(
            &full_literature_spec(None),
            &full_literature_scenario_study_spec(),
            0.68,
            8,
            &[2026, 2027],
        )
        .unwrap();

        assert_eq!(report.aggregate.n_scenarios, 3);
        assert!(report.aggregate.all_converged);
        assert!(!report.aggregate.all_calibration_sigma_star_ge_sigma_fraction_ge_0_99);
        assert!(report.aggregate.min_fit_sigma_ratio_to_baseline > 1.0);
        assert!(report.aggregate.max_fit_sigma_ratio_to_baseline < 1.05);
        assert!(report.aggregate.min_calibration_mean_sigma_star_to_sigma_ratio > 0.999);
        assert!(report.aggregate.max_calibration_mean_sigma_star_to_sigma_ratio < 1.02);
        assert_eq!(report.aggregate.widest_fit_interval_scenario, "bjes_0p5");
        assert_eq!(report.aggregate.highest_calibration_sigma_ratio_scenario, "bjes_0p5");

        let bjes = report
            .scenarios
            .iter()
            .find(|scenario| scenario.name == "bjes_0p5")
            .expect("bjes scenario should be present");
        let theory = report
            .scenarios
            .iter()
            .find(|scenario| scenario.name == "theory_core_0p5")
            .expect("theory scenario should be present");
        let jes = report
            .scenarios
            .iter()
            .find(|scenario| scenario.name == "jes_family_0p5")
            .expect("jes scenario should be present");
        assert!(
            bjes.comparison.fit_sigma_ratio_to_baseline
                > theory.comparison.fit_sigma_ratio_to_baseline
        );
        assert!(
            theory.comparison.fit_sigma_ratio_to_baseline
                > jes.comparison.fit_sigma_ratio_to_baseline
        );
        assert!(bjes.comparison.calibration_min_sigma_star_ge_sigma_fraction >= 0.99);
        assert!(theory.comparison.calibration_min_sigma_star_ge_sigma_fraction >= 0.75);
        assert!(jes.comparison.calibration_min_sigma_star_ge_sigma_fraction >= 0.375);
        assert!(bjes.comparison.calibration_min_mean_sigma_star_to_sigma_ratio > 1.011);
        assert!(bjes.comparison.calibration_max_mean_sigma_star_to_sigma_ratio < 1.012);
        assert!(theory.comparison.calibration_max_abs_ratio_delta_from_reference < 1e-8);
        assert!(jes.comparison.calibration_max_abs_ratio_delta_from_reference < 1e-8);
    }

    #[test]
    #[ignore = "research-grade slow gate; run explicitly"]
    fn full_literature_multi_systematic_calibration_campaign_fixture_matches_committed_artifact() {
        let report = run_measurement_combination_calibration_campaign(
            &full_literature_spec(None),
            &full_literature_scenario_study_spec(),
            0.68,
            8,
            &[2026, 2027],
        )
        .unwrap();

        assert_json_matches_fixture(
            &report,
            &full_literature_calibration_campaign_report_fixture(),
        );
    }

    #[test]
    fn full_literature_multi_systematic_calibration_campaign_summary_fixture_matches_committed_artifact()
     {
        let report = load_calibration_campaign_report(
            &full_literature_calibration_campaign_report_fixture(),
        );
        let summary = summarize_measurement_combination_calibration_campaign(&report).unwrap();
        assert_json_matches_fixture(
            &summary,
            &full_literature_calibration_campaign_summary_fixture(),
        );
    }

    #[test]
    fn full_literature_multi_systematic_calibration_campaign_summary_markdown_matches_committed_artifact()
     {
        let report = load_calibration_campaign_report(
            &full_literature_calibration_campaign_report_fixture(),
        );
        let summary = summarize_measurement_combination_calibration_campaign(&report).unwrap();
        let markdown =
            render_measurement_combination_calibration_campaign_digest_markdown(&summary).unwrap();
        let expected = std::fs::read_to_string(
            full_literature_calibration_campaign_summary_markdown_fixture(),
        )
        .unwrap();
        assert_eq!(markdown, expected);
    }

    #[test]
    fn calibration_campaign_brief_fixture_matches_committed_artifact() {
        let outlier = load_calibration_campaign_summary(
            &scenario_outlier_calibration_campaign_summary_fixture(),
        );
        let full = load_calibration_campaign_summary(
            &full_literature_calibration_campaign_summary_fixture(),
        );
        let brief = build_measurement_combination_calibration_campaign_brief(&[
            ("outlier".to_string(), outlier),
            ("topmass_full".to_string(), full),
        ])
        .unwrap();
        assert_json_matches_fixture(&brief, &calibration_campaign_brief_fixture());
    }

    #[test]
    fn calibration_campaign_brief_markdown_matches_committed_artifact() {
        let outlier = load_calibration_campaign_summary(
            &scenario_outlier_calibration_campaign_summary_fixture(),
        );
        let full = load_calibration_campaign_summary(
            &full_literature_calibration_campaign_summary_fixture(),
        );
        let brief = build_measurement_combination_calibration_campaign_brief(&[
            ("outlier".to_string(), outlier),
            ("topmass_full".to_string(), full),
        ])
        .unwrap();
        let markdown =
            render_measurement_combination_calibration_campaign_brief_markdown(&brief).unwrap();
        let expected =
            std::fs::read_to_string(calibration_campaign_brief_markdown_fixture()).unwrap();
        assert_eq!(markdown, expected);
    }

    #[test]
    fn calibration_campaign_topmass_only_brief_fixture_matches_committed_artifact() {
        let full = load_calibration_campaign_summary(
            &full_literature_calibration_campaign_summary_fixture(),
        );
        let brief = build_measurement_combination_calibration_campaign_brief(&[(
            "topmass_full".to_string(),
            full,
        )])
        .unwrap();
        assert_json_matches_fixture(&brief, &calibration_campaign_topmass_only_brief_fixture());
    }

    #[test]
    fn calibration_campaign_family_report_fixture_matches_committed_artifact() {
        let cross_fixture = load_calibration_campaign_brief(&calibration_campaign_brief_fixture());
        let topmass_only =
            load_calibration_campaign_brief(&calibration_campaign_topmass_only_brief_fixture());
        let report = build_measurement_combination_calibration_campaign_family_report(&[
            ("cross_fixture".to_string(), cross_fixture),
            ("topmass_only".to_string(), topmass_only),
        ])
        .unwrap();
        assert_json_matches_fixture(&report, &calibration_campaign_family_report_fixture());
    }

    #[test]
    fn calibration_campaign_family_report_markdown_matches_committed_artifact() {
        let cross_fixture = load_calibration_campaign_brief(&calibration_campaign_brief_fixture());
        let topmass_only =
            load_calibration_campaign_brief(&calibration_campaign_topmass_only_brief_fixture());
        let report = build_measurement_combination_calibration_campaign_family_report(&[
            ("cross_fixture".to_string(), cross_fixture),
            ("topmass_only".to_string(), topmass_only),
        ])
        .unwrap();
        let markdown =
            render_measurement_combination_calibration_campaign_family_report_markdown(&report)
                .unwrap();
        let expected =
            std::fs::read_to_string(calibration_campaign_family_report_markdown_fixture()).unwrap();
        assert_eq!(markdown, expected);
    }

    #[test]
    fn calibration_campaign_family_matrix_fixture_matches_committed_artifact() {
        let report =
            load_calibration_campaign_family_report(&calibration_campaign_family_report_fixture());
        let matrix =
            build_measurement_combination_calibration_campaign_family_matrix(&report).unwrap();
        assert_json_matches_fixture(&matrix, &calibration_campaign_family_matrix_fixture());
    }

    #[test]
    fn calibration_campaign_family_matrix_markdown_matches_committed_artifact() {
        let report =
            load_calibration_campaign_family_report(&calibration_campaign_family_report_fixture());
        let matrix =
            build_measurement_combination_calibration_campaign_family_matrix(&report).unwrap();
        let markdown =
            render_measurement_combination_calibration_campaign_family_matrix_markdown(&matrix)
                .unwrap();
        let expected =
            std::fs::read_to_string(calibration_campaign_family_matrix_markdown_fixture()).unwrap();
        assert_eq!(markdown, expected);
    }

    #[test]
    fn calibration_campaign_topmass_only_family_report_fixture_matches_committed_artifact() {
        let brief =
            load_calibration_campaign_brief(&calibration_campaign_topmass_only_brief_fixture());
        let report = build_measurement_combination_calibration_campaign_family_report(&[(
            "topmass_only".to_string(),
            brief,
        )])
        .unwrap();
        assert_json_matches_fixture(
            &report,
            &calibration_campaign_topmass_only_family_report_fixture(),
        );
    }

    #[test]
    fn calibration_campaign_topmass_only_family_matrix_fixture_matches_committed_artifact() {
        let report = load_calibration_campaign_family_report(
            &calibration_campaign_topmass_only_family_report_fixture(),
        );
        let matrix =
            build_measurement_combination_calibration_campaign_family_matrix(&report).unwrap();
        assert_json_matches_fixture(
            &matrix,
            &calibration_campaign_topmass_only_family_matrix_fixture(),
        );
    }

    #[test]
    fn calibration_campaign_portfolio_fixture_matches_committed_artifact() {
        let cross =
            load_calibration_campaign_family_matrix(&calibration_campaign_family_matrix_fixture());
        let topmass_only = load_calibration_campaign_family_matrix(
            &calibration_campaign_topmass_only_family_matrix_fixture(),
        );
        let report = build_measurement_combination_calibration_campaign_portfolio_report(&[
            ("cross_portfolio".to_string(), cross),
            ("topmass_only_portfolio".to_string(), topmass_only),
        ])
        .unwrap();
        assert_json_matches_fixture(&report, &calibration_campaign_portfolio_fixture());
    }

    #[test]
    fn calibration_campaign_portfolio_markdown_matches_committed_artifact() {
        let cross =
            load_calibration_campaign_family_matrix(&calibration_campaign_family_matrix_fixture());
        let topmass_only = load_calibration_campaign_family_matrix(
            &calibration_campaign_topmass_only_family_matrix_fixture(),
        );
        let report = build_measurement_combination_calibration_campaign_portfolio_report(&[
            ("cross_portfolio".to_string(), cross),
            ("topmass_only_portfolio".to_string(), topmass_only),
        ])
        .unwrap();
        let markdown =
            render_measurement_combination_calibration_campaign_portfolio_markdown(&report)
                .unwrap();
        let expected =
            std::fs::read_to_string(calibration_campaign_portfolio_markdown_fixture()).unwrap();
        assert_eq!(markdown, expected);
    }

    #[test]
    fn calibration_campaign_portfolio_stability_fixture_matches_committed_artifact() {
        let portfolio =
            load_calibration_campaign_portfolio(&calibration_campaign_portfolio_fixture());
        let report =
            build_measurement_combination_calibration_campaign_portfolio_stability_report(&[
                ("seedgrid_a".to_string(), portfolio.clone()),
                ("seedgrid_b".to_string(), portfolio),
            ])
            .unwrap();
        assert_json_matches_fixture(&report, &calibration_campaign_portfolio_stability_fixture());
    }

    #[test]
    fn calibration_campaign_portfolio_stability_markdown_matches_committed_artifact() {
        let portfolio =
            load_calibration_campaign_portfolio(&calibration_campaign_portfolio_fixture());
        let report =
            build_measurement_combination_calibration_campaign_portfolio_stability_report(&[
                ("seedgrid_a".to_string(), portfolio.clone()),
                ("seedgrid_b".to_string(), portfolio),
            ])
            .unwrap();
        let markdown =
            render_measurement_combination_calibration_campaign_portfolio_stability_markdown(
                &report,
            )
            .unwrap();
        let expected =
            std::fs::read_to_string(calibration_campaign_portfolio_stability_markdown_fixture())
                .unwrap();
        assert_eq!(markdown, expected);
    }

    #[test]
    fn build_measurement_combination_spec_from_tables_parses_csv_bundle() {
        let measurements = "name,value\nm1,1.0\nm2,3.0\n";
        let stat_covariance = "measurement,m2,m1\nm2,4.0,0.0\nm1,0.0,1.0\n";
        let systematics = "\
systematic,measurement,magnitude,error_on_error,aux_mean
s1,m1,0.2,0.1,0.0
s1,m2,0.3,0.1,0.0
";
        let correlations = "\
systematic,row_measurement,col_measurement,corr
s1,m1,m2,0.5
";

        let spec = build_measurement_combination_spec_from_tables(
            "mu",
            measurements,
            stat_covariance,
            Some(systematics),
            Some(correlations),
        )
        .unwrap();

        assert_eq!(spec.poi, "mu");
        assert_eq!(spec.measurements.len(), 2);
        assert_eq!(spec.measurements[0].name, "m1");
        assert_eq!(spec.measurements[1].name, "m2");
        assert_eq!(spec.stat_covariance, vec![vec![1.0, 0.0], vec![0.0, 4.0]]);
        assert_eq!(spec.systematics.len(), 1);
        assert_eq!(spec.systematics[0].name, "s1");
        assert_eq!(spec.systematics[0].magnitudes, vec![0.2, 0.3]);
        assert_eq!(spec.systematics[0].error_on_error, 0.1);
        assert_eq!(spec.systematics[0].corr, vec![vec![1.0, 0.5], vec![0.5, 1.0]]);
    }

    #[test]
    fn build_measurement_combination_spec_from_tables_defaults_identity_correlation_and_accepts_tsv()
     {
        let measurements = "name\tvalue\nm1\t1.0\nm2\t2.0\n";
        let stat_covariance = "measurement\tm1\tm2\nm1\t0.04\t0.01\nm2\t0.01\t0.09\n";
        let systematics = "\
systematic\tmeasurement\tmagnitude\terror_on_error\taux_mean
luminosity\tm1\t0.2\t\t
luminosity\tm2\t0.3\t\t
";

        let spec = build_measurement_combination_spec_from_tables(
            "signal_strength",
            measurements,
            stat_covariance,
            Some(systematics),
            None,
        )
        .unwrap();

        assert_eq!(spec.poi, "signal_strength");
        assert_eq!(spec.systematics.len(), 1);
        assert_eq!(spec.systematics[0].corr, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert_eq!(spec.systematics[0].error_on_error, 0.0);
        assert_eq!(spec.systematics[0].aux_mean, 0.0);
    }

    #[test]
    fn build_measurement_combination_spec_from_manifest_path_matches_committed_example() {
        let mut manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        manifest.pop();
        manifest.pop();
        manifest.push("docs/examples/gvm-stable-first/manifest.yaml");

        let mut spec_fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        spec_fixture.pop();
        spec_fixture.pop();
        spec_fixture.push("docs/examples/gvm-stable-first/spec.json");

        let spec = build_measurement_combination_spec_from_manifest_path(&manifest).unwrap();
        let expected: MeasurementCombinationSpec =
            serde_json::from_str(&std::fs::read_to_string(spec_fixture).unwrap()).unwrap();
        assert_eq!(serde_json::to_value(&spec).unwrap(), serde_json::to_value(&expected).unwrap());
    }

    #[test]
    fn build_measurement_combination_spec_from_manifest_path_resolves_relative_bundle_paths() {
        let unique = format!(
            "nextstat-gvm-manifest-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        );
        let bundle_dir = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&bundle_dir).unwrap();

        std::fs::write(bundle_dir.join("measurements.tsv"), "name\tvalue\nm1\t1.0\nm2\t2.0\n")
            .unwrap();
        std::fs::write(
            bundle_dir.join("stat.tsv"),
            "measurement\tm1\tm2\nm1\t0.04\t0.01\nm2\t0.01\t0.09\n",
        )
        .unwrap();
        std::fs::write(
            bundle_dir.join("systematics.tsv"),
            "systematic\tmeasurement\tmagnitude\terror_on_error\taux_mean\nluminosity\tm1\t0.2\t0.1\t0.0\nluminosity\tm2\t0.3\t0.1\t0.0\n",
        )
        .unwrap();
        std::fs::write(
            bundle_dir.join("manifest.yaml"),
            "\
schema_version: nextstat_measurement_combination_manifest_v0
poi: mu
measurements_table: measurements.tsv
stat_covariance_table: stat.tsv
systematics_table: systematics.tsv
",
        )
        .unwrap();

        let spec = build_measurement_combination_spec_from_manifest_path(
            &bundle_dir.join("manifest.yaml"),
        )
        .unwrap();
        assert_eq!(spec.poi, "mu");
        assert_eq!(spec.measurements.len(), 2);
        assert_eq!(spec.systematics.len(), 1);
        assert_eq!(spec.systematics[0].corr, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);

        std::fs::remove_dir_all(bundle_dir).unwrap();
    }

    #[test]
    #[ignore = "research-grade slow gate; run explicitly"]
    fn full_literature_portfolio_stability_holds_across_seed_grids() {
        let multi_scenarios = full_literature_scenario_study_spec();
        let bjes_only_scenarios = full_literature_scenario_subset(&["bjes_0p5"]);

        let multi_a = run_measurement_combination_calibration_campaign(
            &full_literature_spec(None),
            &multi_scenarios,
            0.68,
            2,
            &[2026],
        )
        .unwrap();
        let bjes_a = run_measurement_combination_calibration_campaign(
            &full_literature_spec(None),
            &bjes_only_scenarios,
            0.68,
            2,
            &[2026],
        )
        .unwrap();
        let portfolio_a = build_full_literature_portfolio_from_campaign_reports(&multi_a, &bjes_a);

        let multi_b = run_measurement_combination_calibration_campaign(
            &full_literature_spec(None),
            &multi_scenarios,
            0.68,
            2,
            &[2030],
        )
        .unwrap();
        let bjes_b = run_measurement_combination_calibration_campaign(
            &full_literature_spec(None),
            &bjes_only_scenarios,
            0.68,
            2,
            &[2030],
        )
        .unwrap();
        let portfolio_b = build_full_literature_portfolio_from_campaign_reports(&multi_b, &bjes_b);

        let stability =
            build_measurement_combination_calibration_campaign_portfolio_stability_report(&[
                ("seedgrid_a".to_string(), portfolio_a),
                ("seedgrid_b".to_string(), portfolio_b),
            ])
            .unwrap();

        assert_eq!(stability.aggregate.n_runs, 2);
        assert!(stability.aggregate.stable_fit_leader);
        assert!(stability.aggregate.stable_calibration_leader);
        assert!(stability.aggregate.stable_joint_leader);
        assert!(stability.aggregate.stable_fit_order);
        assert!(stability.aggregate.stable_calibration_order);
        assert!(stability.aggregate.stable_joint_order);
        assert_eq!(stability.aggregate.reference_run, "seedgrid_a");

        let relation = stability
            .pairwise
            .iter()
            .find(|relation| relation.lhs == "seedgrid_a" && relation.rhs == "seedgrid_b")
            .expect("pairwise stability relation should exist");
        assert!(relation.same_poi_coverage);
        assert!(relation.same_portfolio_labels);
        assert!(relation.same_fit_leader);
        assert!(relation.same_calibration_leader);
        assert!(relation.same_joint_leader);
        assert!(relation.same_fit_order);
        assert!(relation.same_calibration_order);
        assert!(relation.same_joint_order);

        let run = stability
            .runs
            .iter()
            .find(|run| run.label == "seedgrid_a")
            .expect("seedgrid_a run should exist");
        assert_eq!(run.fit_leader, "bjes_only_portfolio");
        assert_eq!(run.calibration_leader, "bjes_only_portfolio");
        assert_eq!(run.joint_leader, "bjes_only_portfolio");
        assert_eq!(
            run.fit_order,
            vec!["bjes_only_portfolio".to_string(), "combo_portfolio".to_string()]
        );
        assert_eq!(
            run.calibration_order,
            vec!["bjes_only_portfolio".to_string(), "combo_portfolio".to_string()]
        );
        assert_eq!(
            run.joint_order,
            vec!["bjes_only_portfolio".to_string(), "combo_portfolio".to_string()]
        );
    }
}
