//! Bayesian beta-binomial clinical trial design helpers.

use std::collections::HashSet;
use std::fmt::Write as _;

use ns_core::{Error, Result};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Beta as RandBeta, Binomial, Distribution, Normal as RandNormal};
use serde::{Deserialize, Serialize};
use statrs::distribution::{Beta, ContinuousCDF, Normal as StatNormal};

pub const BETA_BINOMIAL_DESIGN_SCHEMA_V0: &str = "nextstat_beta_binomial_design_v0";
pub const BETA_BINOMIAL_DESIGN_ANALYSIS_SCHEMA_V0: &str =
    "nextstat_beta_binomial_design_analysis_v0";
pub const BETA_BINOMIAL_OPERATING_CHARACTERISTICS_SCHEMA_V0: &str =
    "nextstat_beta_binomial_operating_characteristics_v0";
pub const BETA_BINOMIAL_POSTERIOR_PREDICTIVE_SCHEMA_V0: &str =
    "nextstat_beta_binomial_posterior_predictive_v0";
pub const BETA_BINOMIAL_PRIOR_SENSITIVITY_CAMPAIGN_SCHEMA_V0: &str =
    "nextstat_beta_binomial_prior_sensitivity_campaign_v0";
pub const BETA_BINOMIAL_PRIOR_SENSITIVITY_REPORT_SCHEMA_V0: &str =
    "nextstat_beta_binomial_prior_sensitivity_report_v0";
pub const BETA_BINOMIAL_DESIGN_REPORT_SCHEMA_V0: &str = "nextstat_beta_binomial_design_report_v0";
pub const NORMAL_NORMAL_DESIGN_SCHEMA_V0: &str = "nextstat_normal_normal_design_v0";
pub const NORMAL_NORMAL_DESIGN_ANALYSIS_SCHEMA_V0: &str =
    "nextstat_normal_normal_design_analysis_v0";
pub const NORMAL_NORMAL_OPERATING_CHARACTERISTICS_SCHEMA_V0: &str =
    "nextstat_normal_normal_operating_characteristics_v0";
pub const NORMAL_NORMAL_POSTERIOR_PREDICTIVE_SCHEMA_V0: &str =
    "nextstat_normal_normal_posterior_predictive_v0";
pub const NORMAL_NORMAL_PRIOR_SENSITIVITY_CAMPAIGN_SCHEMA_V0: &str =
    "nextstat_normal_normal_prior_sensitivity_campaign_v0";
pub const NORMAL_NORMAL_PRIOR_SENSITIVITY_REPORT_SCHEMA_V0: &str =
    "nextstat_normal_normal_prior_sensitivity_report_v0";
pub const NORMAL_NORMAL_DESIGN_REPORT_SCHEMA_V0: &str = "nextstat_normal_normal_design_report_v0";

const SUPERIORITY_PROBABILITY_GRID_SIZE: usize = 4096;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BetaPrior {
    pub alpha: f64,
    pub beta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BetaBinomialLook {
    pub id: String,
    pub n_control: u64,
    pub n_treatment: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BetaBinomialDecisionRule {
    pub posterior_probability_threshold: f64,
    pub treatment_effect_margin: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BetaBinomialDecisionRules {
    pub success: BetaBinomialDecisionRule,
    pub futility: BetaBinomialDecisionRule,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BetaBinomialAnalysisConfig {
    pub credible_interval_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BetaBinomialScenario {
    pub id: String,
    pub p_control: f64,
    pub p_treatment: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BetaBinomialSimulationConfig {
    pub n_replicates: usize,
    pub seed: u64,
    pub scenarios: Vec<BetaBinomialScenario>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BetaBinomialDesignSpec {
    pub schema_version: String,
    pub design_id: String,
    pub control_prior: BetaPrior,
    pub treatment_prior: BetaPrior,
    pub looks: Vec<BetaBinomialLook>,
    pub decision_rules: BetaBinomialDecisionRules,
    pub analysis: BetaBinomialAnalysisConfig,
    pub simulation: Option<BetaBinomialSimulationConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BetaBinomialObservedData {
    pub look_id: String,
    pub control_successes: u64,
    pub treatment_successes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BetaPosteriorSummary {
    pub alpha: f64,
    pub beta: f64,
    pub mean: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BetaBinomialEffectDifferenceSummary {
    pub margin: f64,
    pub posterior_mean: f64,
    pub posterior_probability_gt_margin: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BetaBinomialPosteriorSummary {
    pub control: BetaPosteriorSummary,
    pub treatment: BetaPosteriorSummary,
    pub effect_difference: BetaBinomialEffectDifferenceSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BetaBinomialRecommendedAction {
    Continue,
    StopForSuccess,
    StopForFutility,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BetaBinomialDecisionSummary {
    pub success: bool,
    pub futility: bool,
    pub recommended_action: BetaBinomialRecommendedAction,
    pub posterior_probability_gt_margin: f64,
    pub success_threshold: f64,
    pub futility_threshold: f64,
    pub margin: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BetaBinomialDesignAnalysisResult {
    pub schema_version: String,
    pub stability: String,
    pub design_id: String,
    pub look: BetaBinomialLook,
    pub observed: BetaBinomialObservedData,
    pub posterior: BetaBinomialPosteriorSummary,
    pub decision: BetaBinomialDecisionSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BetaBinomialLookOperatingCharacteristics {
    pub look_id: String,
    pub stop_probability: f64,
    pub success_probability: f64,
    pub futility_probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BetaBinomialScenarioOperatingCharacteristics {
    pub scenario_id: String,
    pub p_control: f64,
    pub p_treatment: f64,
    pub success_rate: f64,
    pub futility_rate: f64,
    pub no_decision_rate: f64,
    pub expected_total_sample_size: f64,
    pub look_summaries: Vec<BetaBinomialLookOperatingCharacteristics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BetaBinomialOperatingCharacteristicsResult {
    pub schema_version: String,
    pub stability: String,
    pub design_id: String,
    pub n_replicates: usize,
    pub seed: u64,
    pub scenarios: Vec<BetaBinomialScenarioOperatingCharacteristics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BetaBinomialPosteriorPredictiveLookSummary {
    pub look_id: String,
    pub conditional_stop_probability: f64,
    pub conditional_success_probability: f64,
    pub conditional_futility_probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BetaBinomialPosteriorPredictiveResult {
    pub schema_version: String,
    pub stability: String,
    pub design_id: String,
    pub current_analysis: BetaBinomialDesignAnalysisResult,
    pub n_replicates: usize,
    pub seed: u64,
    pub current_total_sample_size: f64,
    pub expected_total_sample_size: f64,
    pub expected_remaining_sample_size: f64,
    pub eventual_success_probability: f64,
    pub eventual_futility_probability: f64,
    pub eventual_no_decision_probability: f64,
    pub future_look_summaries: Vec<BetaBinomialPosteriorPredictiveLookSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BetaBinomialPriorSensitivityVariant {
    pub id: String,
    pub control_prior: BetaPrior,
    pub treatment_prior: BetaPrior,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BetaBinomialPriorSensitivityCampaign {
    pub schema_version: String,
    pub variants: Vec<BetaBinomialPriorSensitivityVariant>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BetaBinomialPriorSensitivityVariantResult {
    pub variant_id: String,
    pub is_baseline: bool,
    pub control_prior: BetaPrior,
    pub treatment_prior: BetaPrior,
    pub posterior_mean: f64,
    pub posterior_probability_gt_margin: f64,
    pub recommended_action: BetaBinomialRecommendedAction,
    pub eventual_success_probability: f64,
    pub eventual_futility_probability: f64,
    pub eventual_no_decision_probability: f64,
    pub expected_total_sample_size: f64,
    pub expected_remaining_sample_size: f64,
    pub future_look_summaries: Vec<BetaBinomialPosteriorPredictiveLookSummary>,
    pub posterior_probability_delta_vs_baseline: f64,
    pub eventual_success_probability_delta_vs_baseline: f64,
    pub expected_total_sample_size_delta_vs_baseline: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BetaBinomialPriorSensitivityReport {
    pub schema_version: String,
    pub stability: String,
    pub design_id: String,
    pub look: BetaBinomialLook,
    pub observed: BetaBinomialObservedData,
    pub n_replicates: usize,
    pub seed: u64,
    pub variants: Vec<BetaBinomialPriorSensitivityVariantResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BayesianDesignReportProvenance {
    pub software_name: String,
    pub software_version: String,
    pub design_schema_version: String,
    pub analysis_schema_version: String,
    pub operating_characteristics_schema_version: String,
    pub posterior_predictive_schema_version: String,
    pub prior_sensitivity_campaign_schema_version: String,
    pub prior_sensitivity_report_schema_version: String,
    pub simulation_seed: u64,
    pub n_replicates: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BetaBinomialDesignReport {
    pub schema_version: String,
    pub stability: String,
    pub design_family: String,
    pub design_spec: BetaBinomialDesignSpec,
    pub current_analysis: BetaBinomialDesignAnalysisResult,
    pub operating_characteristics: BetaBinomialOperatingCharacteristicsResult,
    pub posterior_predictive: BetaBinomialPosteriorPredictiveResult,
    pub prior_sensitivity: BetaBinomialPriorSensitivityReport,
    pub provenance: BayesianDesignReportProvenance,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct BetaPosteriorParams {
    alpha: f64,
    beta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NormalPrior {
    pub mean: f64,
    pub sd: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NormalNormalLikelihood {
    pub known_sd_control: f64,
    pub known_sd_treatment: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NormalNormalLook {
    pub id: String,
    pub n_control: u64,
    pub n_treatment: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NormalNormalDecisionRule {
    pub posterior_probability_threshold: f64,
    pub treatment_effect_margin: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NormalNormalDecisionRules {
    pub success: NormalNormalDecisionRule,
    pub futility: NormalNormalDecisionRule,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NormalNormalAnalysisConfig {
    pub credible_interval_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NormalNormalScenario {
    pub id: String,
    pub mean_control: f64,
    pub mean_treatment: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NormalNormalSimulationConfig {
    pub n_replicates: usize,
    pub seed: u64,
    pub scenarios: Vec<NormalNormalScenario>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NormalNormalDesignSpec {
    pub schema_version: String,
    pub design_id: String,
    pub control_prior: NormalPrior,
    pub treatment_prior: NormalPrior,
    pub likelihood: NormalNormalLikelihood,
    pub looks: Vec<NormalNormalLook>,
    pub decision_rules: NormalNormalDecisionRules,
    pub analysis: NormalNormalAnalysisConfig,
    pub simulation: Option<NormalNormalSimulationConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NormalNormalObservedData {
    pub look_id: String,
    pub control_sample_mean: f64,
    pub treatment_sample_mean: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalPosteriorSummary {
    pub posterior_mean: f64,
    pub posterior_sd: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalNormalEffectDifferenceSummary {
    pub margin: f64,
    pub posterior_mean: f64,
    pub posterior_sd: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub posterior_probability_gt_margin: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalNormalPosteriorSummary {
    pub control: NormalPosteriorSummary,
    pub treatment: NormalPosteriorSummary,
    pub effect_difference: NormalNormalEffectDifferenceSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NormalNormalRecommendedAction {
    Continue,
    StopForSuccess,
    StopForFutility,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalNormalDecisionSummary {
    pub success: bool,
    pub futility: bool,
    pub recommended_action: NormalNormalRecommendedAction,
    pub posterior_probability_gt_margin: f64,
    pub success_threshold: f64,
    pub futility_threshold: f64,
    pub margin: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalNormalDesignAnalysisResult {
    pub schema_version: String,
    pub stability: String,
    pub design_id: String,
    pub look: NormalNormalLook,
    pub observed: NormalNormalObservedData,
    pub posterior: NormalNormalPosteriorSummary,
    pub decision: NormalNormalDecisionSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalNormalLookOperatingCharacteristics {
    pub look_id: String,
    pub stop_probability: f64,
    pub success_probability: f64,
    pub futility_probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalNormalScenarioOperatingCharacteristics {
    pub scenario_id: String,
    pub mean_control: f64,
    pub mean_treatment: f64,
    pub success_rate: f64,
    pub futility_rate: f64,
    pub no_decision_rate: f64,
    pub expected_total_sample_size: f64,
    pub look_summaries: Vec<NormalNormalLookOperatingCharacteristics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalNormalOperatingCharacteristicsResult {
    pub schema_version: String,
    pub stability: String,
    pub design_id: String,
    pub n_replicates: usize,
    pub seed: u64,
    pub scenarios: Vec<NormalNormalScenarioOperatingCharacteristics>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalNormalPosteriorPredictiveLookSummary {
    pub look_id: String,
    pub conditional_stop_probability: f64,
    pub conditional_success_probability: f64,
    pub conditional_futility_probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalNormalPosteriorPredictiveResult {
    pub schema_version: String,
    pub stability: String,
    pub design_id: String,
    pub current_analysis: NormalNormalDesignAnalysisResult,
    pub n_replicates: usize,
    pub seed: u64,
    pub current_total_sample_size: f64,
    pub expected_total_sample_size: f64,
    pub expected_remaining_sample_size: f64,
    pub eventual_success_probability: f64,
    pub eventual_futility_probability: f64,
    pub eventual_no_decision_probability: f64,
    pub future_look_summaries: Vec<NormalNormalPosteriorPredictiveLookSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NormalNormalPriorSensitivityVariant {
    pub id: String,
    pub control_prior: NormalPrior,
    pub treatment_prior: NormalPrior,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NormalNormalPriorSensitivityCampaign {
    pub schema_version: String,
    pub variants: Vec<NormalNormalPriorSensitivityVariant>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalNormalPriorSensitivityVariantResult {
    pub variant_id: String,
    pub is_baseline: bool,
    pub control_prior: NormalPrior,
    pub treatment_prior: NormalPrior,
    pub posterior_mean: f64,
    pub posterior_probability_gt_margin: f64,
    pub recommended_action: NormalNormalRecommendedAction,
    pub eventual_success_probability: f64,
    pub eventual_futility_probability: f64,
    pub eventual_no_decision_probability: f64,
    pub expected_total_sample_size: f64,
    pub expected_remaining_sample_size: f64,
    pub future_look_summaries: Vec<NormalNormalPosteriorPredictiveLookSummary>,
    pub posterior_probability_delta_vs_baseline: f64,
    pub eventual_success_probability_delta_vs_baseline: f64,
    pub expected_total_sample_size_delta_vs_baseline: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NormalNormalPriorSensitivityReport {
    pub schema_version: String,
    pub stability: String,
    pub design_id: String,
    pub look: NormalNormalLook,
    pub observed: NormalNormalObservedData,
    pub n_replicates: usize,
    pub seed: u64,
    pub variants: Vec<NormalNormalPriorSensitivityVariantResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NormalNormalDesignReport {
    pub schema_version: String,
    pub stability: String,
    pub design_family: String,
    pub design_spec: NormalNormalDesignSpec,
    pub current_analysis: NormalNormalDesignAnalysisResult,
    pub operating_characteristics: NormalNormalOperatingCharacteristicsResult,
    pub posterior_predictive: NormalNormalPosteriorPredictiveResult,
    pub prior_sensitivity: NormalNormalPriorSensitivityReport,
    pub provenance: BayesianDesignReportProvenance,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct NormalPosteriorParams {
    mean: f64,
    variance: f64,
}

impl BetaBinomialDesignSpec {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != BETA_BINOMIAL_DESIGN_SCHEMA_V0 {
            return Err(Error::Validation(format!(
                "schema_version must be '{}', got '{}'",
                BETA_BINOMIAL_DESIGN_SCHEMA_V0, self.schema_version
            )));
        }
        if self.design_id.trim().is_empty() {
            return Err(Error::Validation("design_id must be non-empty".to_string()));
        }
        validate_prior(&self.control_prior, "control_prior")?;
        validate_prior(&self.treatment_prior, "treatment_prior")?;
        if self.looks.is_empty() {
            return Err(Error::Validation(
                "beta-binomial design requires at least one look".to_string(),
            ));
        }
        validate_probability_unit_interval(
            self.decision_rules.success.posterior_probability_threshold,
            "decision_rules.success.posterior_probability_threshold",
        )?;
        validate_probability_unit_interval(
            self.decision_rules.futility.posterior_probability_threshold,
            "decision_rules.futility.posterior_probability_threshold",
        )?;
        if self.decision_rules.success.posterior_probability_threshold
            <= self.decision_rules.futility.posterior_probability_threshold
        {
            return Err(Error::Validation(
                "success threshold must be strictly greater than futility threshold".to_string(),
            ));
        }
        if !self.decision_rules.success.treatment_effect_margin.is_finite() {
            return Err(Error::Validation(
                "decision_rules.success.treatment_effect_margin must be finite".to_string(),
            ));
        }
        if !self.decision_rules.futility.treatment_effect_margin.is_finite() {
            return Err(Error::Validation(
                "decision_rules.futility.treatment_effect_margin must be finite".to_string(),
            ));
        }
        if (self.decision_rules.success.treatment_effect_margin
            - self.decision_rules.futility.treatment_effect_margin)
            .abs()
            > 1e-12
        {
            return Err(Error::Validation(
                "success and futility rules must share the same treatment_effect_margin in v0"
                    .to_string(),
            ));
        }
        let ci = self.analysis.credible_interval_level;
        if !ci.is_finite() || !(0.0..1.0).contains(&ci) {
            return Err(Error::Validation(
                "analysis.credible_interval_level must be finite and in (0,1)".to_string(),
            ));
        }

        let mut seen_look_ids = HashSet::with_capacity(self.looks.len());
        let mut prev_n_control = 0_u64;
        let mut prev_n_treatment = 0_u64;
        for look in &self.looks {
            if look.id.trim().is_empty() {
                return Err(Error::Validation("look id must be non-empty".to_string()));
            }
            if !seen_look_ids.insert(look.id.clone()) {
                return Err(Error::Validation(format!("duplicate look id '{}'", look.id)));
            }
            if look.n_control == 0 {
                return Err(Error::Validation(format!(
                    "look '{}' must have n_control > 0",
                    look.id
                )));
            }
            if look.n_treatment == 0 {
                return Err(Error::Validation(format!(
                    "look '{}' must have n_treatment > 0",
                    look.id
                )));
            }
            if look.n_control < prev_n_control {
                return Err(Error::Validation(
                    "looks must be cumulative and non-decreasing for n_control".to_string(),
                ));
            }
            if look.n_treatment < prev_n_treatment {
                return Err(Error::Validation(
                    "looks must be cumulative and non-decreasing for n_treatment".to_string(),
                ));
            }
            prev_n_control = look.n_control;
            prev_n_treatment = look.n_treatment;
        }

        if let Some(sim) = &self.simulation {
            if sim.n_replicates == 0 {
                return Err(Error::Validation("simulation.n_replicates must be >= 1".to_string()));
            }
            if sim.scenarios.is_empty() {
                return Err(Error::Validation(
                    "simulation.scenarios must contain at least one scenario".to_string(),
                ));
            }
            let mut seen_scenario_ids = HashSet::with_capacity(sim.scenarios.len());
            for scenario in &sim.scenarios {
                if scenario.id.trim().is_empty() {
                    return Err(Error::Validation(
                        "simulation scenario id must be non-empty".to_string(),
                    ));
                }
                if !seen_scenario_ids.insert(scenario.id.clone()) {
                    return Err(Error::Validation(format!(
                        "duplicate simulation scenario id '{}'",
                        scenario.id
                    )));
                }
                validate_probability_unit_interval(
                    scenario.p_control,
                    &format!("simulation scenario '{}' p_control", scenario.id),
                )?;
                validate_probability_unit_interval(
                    scenario.p_treatment,
                    &format!("simulation scenario '{}' p_treatment", scenario.id),
                )?;
            }
        }

        Ok(())
    }
}

impl BetaBinomialPriorSensitivityCampaign {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != BETA_BINOMIAL_PRIOR_SENSITIVITY_CAMPAIGN_SCHEMA_V0 {
            return Err(Error::Validation(format!(
                "schema_version must be '{}', got '{}'",
                BETA_BINOMIAL_PRIOR_SENSITIVITY_CAMPAIGN_SCHEMA_V0, self.schema_version
            )));
        }
        if self.variants.is_empty() {
            return Err(Error::Validation(
                "prior sensitivity campaign requires at least one variant".to_string(),
            ));
        }
        let mut seen_variant_ids = HashSet::with_capacity(self.variants.len());
        for variant in &self.variants {
            if variant.id.trim().is_empty() {
                return Err(Error::Validation(
                    "prior sensitivity variant id must be non-empty".to_string(),
                ));
            }
            if variant.id == "baseline" {
                return Err(Error::Validation(
                    "prior sensitivity variant id 'baseline' is reserved".to_string(),
                ));
            }
            if !seen_variant_ids.insert(variant.id.clone()) {
                return Err(Error::Validation(format!(
                    "duplicate prior sensitivity variant id '{}'",
                    variant.id
                )));
            }
            validate_prior(&variant.control_prior, "variant.control_prior")?;
            validate_prior(&variant.treatment_prior, "variant.treatment_prior")?;
        }
        Ok(())
    }
}

pub fn analyze_beta_binomial_design(
    spec: &BetaBinomialDesignSpec,
    observed: &BetaBinomialObservedData,
) -> Result<BetaBinomialDesignAnalysisResult> {
    spec.validate()?;
    let look = find_look(spec, &observed.look_id)?;
    validate_observed_counts(look, observed)?;
    let margin = spec.decision_rules.success.treatment_effect_margin;
    build_analysis_result(spec, look, observed.clone(), margin)
}

pub fn beta_binomial_operating_characteristics(
    spec: &BetaBinomialDesignSpec,
) -> Result<BetaBinomialOperatingCharacteristicsResult> {
    spec.validate()?;
    let sim = spec.simulation.as_ref().ok_or_else(|| {
        Error::Validation(
            "simulation block is required for beta-binomial operating characteristics".to_string(),
        )
    })?;
    let mut rng = StdRng::seed_from_u64(sim.seed);
    let final_look = spec.looks.last().expect("validated non-empty looks");

    let mut scenarios = Vec::with_capacity(sim.scenarios.len());
    for scenario in &sim.scenarios {
        let mut success_count = 0usize;
        let mut futility_count = 0usize;
        let mut no_decision_count = 0usize;
        let mut total_sample_size_sum = 0.0;
        let mut look_stop_counts = vec![0usize; spec.looks.len()];
        let mut look_success_counts = vec![0usize; spec.looks.len()];
        let mut look_futility_counts = vec![0usize; spec.looks.len()];

        for _ in 0..sim.n_replicates {
            let mut prev_n_control = 0_u64;
            let mut prev_n_treatment = 0_u64;
            let mut control_successes = 0_u64;
            let mut treatment_successes = 0_u64;
            let mut stopped = false;

            for (look_idx, look) in spec.looks.iter().enumerate() {
                let inc_control = look.n_control - prev_n_control;
                let inc_treatment = look.n_treatment - prev_n_treatment;
                let control_draw = Binomial::new(inc_control, scenario.p_control).map_err(|e| {
                    Error::Computation(format!(
                        "failed to build control binomial for scenario '{}': {}",
                        scenario.id, e
                    ))
                })?;
                let treatment_draw =
                    Binomial::new(inc_treatment, scenario.p_treatment).map_err(|e| {
                        Error::Computation(format!(
                            "failed to build treatment binomial for scenario '{}': {}",
                            scenario.id, e
                        ))
                    })?;
                control_successes += control_draw.sample(&mut rng);
                treatment_successes += treatment_draw.sample(&mut rng);

                let observed = BetaBinomialObservedData {
                    look_id: look.id.clone(),
                    control_successes,
                    treatment_successes,
                };
                let analysis = build_analysis_result(
                    spec,
                    look,
                    observed,
                    spec.decision_rules.success.treatment_effect_margin,
                )?;
                let total_n = (look.n_control + look.n_treatment) as f64;

                if analysis.decision.success {
                    success_count += 1;
                    look_stop_counts[look_idx] += 1;
                    look_success_counts[look_idx] += 1;
                    total_sample_size_sum += total_n;
                    stopped = true;
                    break;
                }
                if analysis.decision.futility {
                    futility_count += 1;
                    look_stop_counts[look_idx] += 1;
                    look_futility_counts[look_idx] += 1;
                    total_sample_size_sum += total_n;
                    stopped = true;
                    break;
                }

                prev_n_control = look.n_control;
                prev_n_treatment = look.n_treatment;
            }

            if !stopped {
                no_decision_count += 1;
                total_sample_size_sum += (final_look.n_control + final_look.n_treatment) as f64;
            }
        }

        let n_reps = sim.n_replicates as f64;
        let look_summaries = spec
            .looks
            .iter()
            .enumerate()
            .map(|(idx, look)| BetaBinomialLookOperatingCharacteristics {
                look_id: look.id.clone(),
                stop_probability: look_stop_counts[idx] as f64 / n_reps,
                success_probability: look_success_counts[idx] as f64 / n_reps,
                futility_probability: look_futility_counts[idx] as f64 / n_reps,
            })
            .collect();

        scenarios.push(BetaBinomialScenarioOperatingCharacteristics {
            scenario_id: scenario.id.clone(),
            p_control: scenario.p_control,
            p_treatment: scenario.p_treatment,
            success_rate: success_count as f64 / n_reps,
            futility_rate: futility_count as f64 / n_reps,
            no_decision_rate: no_decision_count as f64 / n_reps,
            expected_total_sample_size: total_sample_size_sum / n_reps,
            look_summaries,
        });
    }

    Ok(BetaBinomialOperatingCharacteristicsResult {
        schema_version: BETA_BINOMIAL_OPERATING_CHARACTERISTICS_SCHEMA_V0.to_string(),
        stability: "research-grade".to_string(),
        design_id: spec.design_id.clone(),
        n_replicates: sim.n_replicates,
        seed: sim.seed,
        scenarios,
    })
}

pub fn beta_binomial_posterior_predictive(
    spec: &BetaBinomialDesignSpec,
    observed: &BetaBinomialObservedData,
) -> Result<BetaBinomialPosteriorPredictiveResult> {
    spec.validate()?;
    let sim = spec.simulation.as_ref().ok_or_else(|| {
        Error::Validation(
            "simulation block is required for beta-binomial posterior predictive forecast"
                .to_string(),
        )
    })?;
    let look_idx =
        spec.looks.iter().position(|look| look.id == observed.look_id).ok_or_else(|| {
            Error::Validation(format!("look_id '{}' was not found in the design", observed.look_id))
        })?;
    let look = &spec.looks[look_idx];
    validate_observed_counts(look, observed)?;
    let margin = spec.decision_rules.success.treatment_effect_margin;
    let current_analysis = build_analysis_result(spec, look, observed.clone(), margin)?;
    let current_total_sample_size = (look.n_control + look.n_treatment) as f64;

    if current_analysis.decision.success
        || current_analysis.decision.futility
        || look_idx + 1 == spec.looks.len()
    {
        return Ok(beta_terminal_posterior_predictive_result(
            spec,
            current_analysis,
            sim.seed,
            sim.n_replicates,
        ));
    }

    let control_posterior =
        posterior_from_counts(&spec.control_prior, observed.control_successes, look.n_control)?;
    let treatment_posterior = posterior_from_counts(
        &spec.treatment_prior,
        observed.treatment_successes,
        look.n_treatment,
    )?;
    let future_looks = &spec.looks[(look_idx + 1)..];
    let final_look = spec.looks.last().expect("validated non-empty looks");

    let mut rng = StdRng::seed_from_u64(sim.seed);
    let mut success_count = 0usize;
    let mut futility_count = 0usize;
    let mut no_decision_count = 0usize;
    let mut total_sample_size_sum = 0.0;
    let mut look_stop_counts = vec![0usize; future_looks.len()];
    let mut look_success_counts = vec![0usize; future_looks.len()];
    let mut look_futility_counts = vec![0usize; future_looks.len()];

    for _ in 0..sim.n_replicates {
        let latent_control_p = sample_beta_probability(control_posterior, &mut rng)?;
        let latent_treatment_p = sample_beta_probability(treatment_posterior, &mut rng)?;
        let mut control_successes = observed.control_successes;
        let mut treatment_successes = observed.treatment_successes;
        let mut prev_n_control = look.n_control;
        let mut prev_n_treatment = look.n_treatment;
        let mut stopped = false;

        for (future_idx, future_look) in future_looks.iter().enumerate() {
            let inc_control = future_look.n_control - prev_n_control;
            let inc_treatment = future_look.n_treatment - prev_n_treatment;
            if inc_control > 0 {
                let dist = Binomial::new(inc_control, latent_control_p).map_err(|e| {
                    Error::Computation(format!(
                        "failed to construct posterior predictive control binomial: {}",
                        e
                    ))
                })?;
                control_successes += dist.sample(&mut rng);
            }
            if inc_treatment > 0 {
                let dist = Binomial::new(inc_treatment, latent_treatment_p).map_err(|e| {
                    Error::Computation(format!(
                        "failed to construct posterior predictive treatment binomial: {}",
                        e
                    ))
                })?;
                treatment_successes += dist.sample(&mut rng);
            }

            let future_observed = BetaBinomialObservedData {
                look_id: future_look.id.clone(),
                control_successes,
                treatment_successes,
            };
            let analysis = build_analysis_result(spec, future_look, future_observed, margin)?;
            let total_n = (future_look.n_control + future_look.n_treatment) as f64;

            if analysis.decision.success {
                success_count += 1;
                look_stop_counts[future_idx] += 1;
                look_success_counts[future_idx] += 1;
                total_sample_size_sum += total_n;
                stopped = true;
                break;
            }
            if analysis.decision.futility {
                futility_count += 1;
                look_stop_counts[future_idx] += 1;
                look_futility_counts[future_idx] += 1;
                total_sample_size_sum += total_n;
                stopped = true;
                break;
            }

            prev_n_control = future_look.n_control;
            prev_n_treatment = future_look.n_treatment;
        }

        if !stopped {
            no_decision_count += 1;
            total_sample_size_sum += (final_look.n_control + final_look.n_treatment) as f64;
        }
    }

    let n_reps = sim.n_replicates as f64;
    let future_look_summaries = future_looks
        .iter()
        .enumerate()
        .map(|(idx, future_look)| BetaBinomialPosteriorPredictiveLookSummary {
            look_id: future_look.id.clone(),
            conditional_stop_probability: look_stop_counts[idx] as f64 / n_reps,
            conditional_success_probability: look_success_counts[idx] as f64 / n_reps,
            conditional_futility_probability: look_futility_counts[idx] as f64 / n_reps,
        })
        .collect();
    let expected_total_sample_size = total_sample_size_sum / n_reps;

    Ok(BetaBinomialPosteriorPredictiveResult {
        schema_version: BETA_BINOMIAL_POSTERIOR_PREDICTIVE_SCHEMA_V0.to_string(),
        stability: "research-grade".to_string(),
        design_id: spec.design_id.clone(),
        current_analysis,
        n_replicates: sim.n_replicates,
        seed: sim.seed,
        current_total_sample_size,
        expected_total_sample_size,
        expected_remaining_sample_size: (expected_total_sample_size - current_total_sample_size)
            .max(0.0),
        eventual_success_probability: success_count as f64 / n_reps,
        eventual_futility_probability: futility_count as f64 / n_reps,
        eventual_no_decision_probability: no_decision_count as f64 / n_reps,
        future_look_summaries,
    })
}

pub fn beta_binomial_prior_sensitivity(
    spec: &BetaBinomialDesignSpec,
    observed: &BetaBinomialObservedData,
    campaign: &BetaBinomialPriorSensitivityCampaign,
) -> Result<BetaBinomialPriorSensitivityReport> {
    spec.validate()?;
    campaign.validate()?;
    let baseline_forecast = beta_binomial_posterior_predictive(spec, observed)?;
    let baseline_variant = beta_prior_sensitivity_variant_result(
        "baseline",
        true,
        spec.control_prior.clone(),
        spec.treatment_prior.clone(),
        &baseline_forecast,
        &baseline_forecast,
    );

    let mut variants = Vec::with_capacity(campaign.variants.len() + 1);
    variants.push(baseline_variant);

    for variant in &campaign.variants {
        let mut variant_spec = spec.clone();
        variant_spec.control_prior = variant.control_prior.clone();
        variant_spec.treatment_prior = variant.treatment_prior.clone();
        let forecast = beta_binomial_posterior_predictive(&variant_spec, observed)?;
        variants.push(beta_prior_sensitivity_variant_result(
            &variant.id,
            false,
            variant.control_prior.clone(),
            variant.treatment_prior.clone(),
            &forecast,
            &baseline_forecast,
        ));
    }

    Ok(BetaBinomialPriorSensitivityReport {
        schema_version: BETA_BINOMIAL_PRIOR_SENSITIVITY_REPORT_SCHEMA_V0.to_string(),
        stability: "research-grade".to_string(),
        design_id: spec.design_id.clone(),
        look: baseline_forecast.current_analysis.look.clone(),
        observed: observed.clone(),
        n_replicates: baseline_forecast.n_replicates,
        seed: baseline_forecast.seed,
        variants,
    })
}

pub fn beta_binomial_design_report(
    spec: &BetaBinomialDesignSpec,
    observed: &BetaBinomialObservedData,
    campaign: &BetaBinomialPriorSensitivityCampaign,
) -> Result<BetaBinomialDesignReport> {
    spec.validate()?;
    campaign.validate()?;

    let posterior_predictive = beta_binomial_posterior_predictive(spec, observed)?;
    let operating_characteristics = beta_binomial_operating_characteristics(spec)?;
    let prior_sensitivity = beta_binomial_prior_sensitivity(spec, observed, campaign)?;
    let sim = spec.simulation.as_ref().ok_or_else(|| {
        Error::Validation(
            "simulation block is required for beta-binomial design report".to_string(),
        )
    })?;

    Ok(BetaBinomialDesignReport {
        schema_version: BETA_BINOMIAL_DESIGN_REPORT_SCHEMA_V0.to_string(),
        stability: "research-grade".to_string(),
        design_family: "beta_binomial".to_string(),
        design_spec: spec.clone(),
        current_analysis: posterior_predictive.current_analysis.clone(),
        operating_characteristics,
        posterior_predictive,
        prior_sensitivity,
        provenance: design_report_provenance(
            BETA_BINOMIAL_DESIGN_SCHEMA_V0,
            BETA_BINOMIAL_DESIGN_ANALYSIS_SCHEMA_V0,
            BETA_BINOMIAL_OPERATING_CHARACTERISTICS_SCHEMA_V0,
            BETA_BINOMIAL_POSTERIOR_PREDICTIVE_SCHEMA_V0,
            BETA_BINOMIAL_PRIOR_SENSITIVITY_CAMPAIGN_SCHEMA_V0,
            BETA_BINOMIAL_PRIOR_SENSITIVITY_REPORT_SCHEMA_V0,
            sim.seed,
            sim.n_replicates,
        ),
    })
}

pub fn render_beta_binomial_design_report_markdown(
    report: &BetaBinomialDesignReport,
) -> Result<String> {
    if report.schema_version != BETA_BINOMIAL_DESIGN_REPORT_SCHEMA_V0 {
        return Err(Error::Validation(format!(
            "schema_version must be '{}', got '{}'",
            BETA_BINOMIAL_DESIGN_REPORT_SCHEMA_V0, report.schema_version
        )));
    }

    let spec = &report.design_spec;
    spec.validate()?;
    let mut markdown = String::new();
    writeln!(&mut markdown, "# Bayesian Trial Design Report").unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(&mut markdown, "- Family: beta_binomial").unwrap();
    writeln!(&mut markdown, "- Design ID: {}", report.current_analysis.design_id).unwrap();
    writeln!(&mut markdown, "- Stability: {}", report.stability).unwrap();
    writeln!(
        &mut markdown,
        "- Software: {} {}",
        report.provenance.software_name, report.provenance.software_version
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "- Simulation: {} replicates, seed {}",
        report.provenance.n_replicates, report.provenance.simulation_seed
    )
    .unwrap();
    writeln!(&mut markdown).unwrap();

    writeln!(&mut markdown, "## Provenance").unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(&mut markdown, "- Report schema: {}", report.schema_version).unwrap();
    writeln!(&mut markdown, "- Design schema: {}", report.provenance.design_schema_version)
        .unwrap();
    writeln!(&mut markdown, "- Analysis schema: {}", report.provenance.analysis_schema_version)
        .unwrap();
    writeln!(
        &mut markdown,
        "- Operating characteristics schema: {}",
        report.provenance.operating_characteristics_schema_version
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "- Posterior predictive schema: {}",
        report.provenance.posterior_predictive_schema_version
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "- Prior sensitivity campaign schema: {}",
        report.provenance.prior_sensitivity_campaign_schema_version
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "- Prior sensitivity report schema: {}",
        report.provenance.prior_sensitivity_report_schema_version
    )
    .unwrap();
    writeln!(&mut markdown).unwrap();

    writeln!(&mut markdown, "## Design Spec").unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(&mut markdown, "### Priors").unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(&mut markdown, "| Arm | Prior | Prior mean |").unwrap();
    writeln!(&mut markdown, "| --- | --- | ---: |").unwrap();
    writeln!(
        &mut markdown,
        "| Control | Beta({}, {}) | {} |",
        format_decimal(spec.control_prior.alpha),
        format_decimal(spec.control_prior.beta),
        format_decimal(
            spec.control_prior.alpha / (spec.control_prior.alpha + spec.control_prior.beta)
        )
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "| Treatment | Beta({}, {}) | {} |",
        format_decimal(spec.treatment_prior.alpha),
        format_decimal(spec.treatment_prior.beta),
        format_decimal(
            spec.treatment_prior.alpha / (spec.treatment_prior.alpha + spec.treatment_prior.beta)
        )
    )
    .unwrap();
    writeln!(&mut markdown).unwrap();

    writeln!(&mut markdown, "### Looks").unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(&mut markdown, "| Look | N control | N treatment |").unwrap();
    writeln!(&mut markdown, "| --- | ---: | ---: |").unwrap();
    for look in &spec.looks {
        writeln!(&mut markdown, "| {} | {} | {} |", look.id, look.n_control, look.n_treatment)
            .unwrap();
    }
    writeln!(&mut markdown).unwrap();

    writeln!(&mut markdown, "### Decision Criteria").unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(&mut markdown, "| Rule | Posterior threshold | Margin | Credible interval level |")
        .unwrap();
    writeln!(&mut markdown, "| --- | ---: | ---: | ---: |").unwrap();
    writeln!(
        &mut markdown,
        "| Success | {} | {} | {} |",
        format_decimal(spec.decision_rules.success.posterior_probability_threshold),
        format_decimal(spec.decision_rules.success.treatment_effect_margin),
        format_decimal(spec.analysis.credible_interval_level)
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "| Futility | {} | {} | {} |",
        format_decimal(spec.decision_rules.futility.posterior_probability_threshold),
        format_decimal(spec.decision_rules.futility.treatment_effect_margin),
        format_decimal(spec.analysis.credible_interval_level)
    )
    .unwrap();
    writeln!(&mut markdown).unwrap();

    writeln!(&mut markdown, "### Simulation Scenarios").unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(&mut markdown, "| Scenario | Control rate | Treatment rate |").unwrap();
    writeln!(&mut markdown, "| --- | ---: | ---: |").unwrap();
    for scenario in &report.operating_characteristics.scenarios {
        writeln!(
            &mut markdown,
            "| {} | {} | {} |",
            scenario.scenario_id,
            format_decimal(scenario.p_control),
            format_decimal(scenario.p_treatment)
        )
        .unwrap();
    }
    writeln!(&mut markdown).unwrap();

    writeln!(&mut markdown, "## Current Analysis").unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(
        &mut markdown,
        "- Look: {} (N control = {}, N treatment = {})",
        report.current_analysis.look.id,
        report.current_analysis.look.n_control,
        report.current_analysis.look.n_treatment
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "- Observed successes: control = {}, treatment = {}",
        report.current_analysis.observed.control_successes,
        report.current_analysis.observed.treatment_successes
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "- Recommended action: {}",
        format_beta_action(&report.current_analysis.decision.recommended_action)
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "- Posterior mean treatment effect: {}",
        format_decimal(report.current_analysis.posterior.effect_difference.posterior_mean)
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "- Posterior Pr(effect > margin): {}",
        format_decimal(report.current_analysis.decision.posterior_probability_gt_margin)
    )
    .unwrap();
    writeln!(&mut markdown).unwrap();

    writeln!(&mut markdown, "## Operating Characteristics").unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(&mut markdown, "| Scenario | Success | Futility | No decision | Expected total N |")
        .unwrap();
    writeln!(&mut markdown, "| --- | ---: | ---: | ---: | ---: |").unwrap();
    for scenario in &report.operating_characteristics.scenarios {
        writeln!(
            &mut markdown,
            "| {} | {} | {} | {} | {} |",
            scenario.scenario_id,
            format_decimal(scenario.success_rate),
            format_decimal(scenario.futility_rate),
            format_decimal(scenario.no_decision_rate),
            format_decimal(scenario.expected_total_sample_size)
        )
        .unwrap();
    }
    writeln!(&mut markdown).unwrap();

    writeln!(&mut markdown, "## Posterior Predictive Forecast").unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(
        &mut markdown,
        "| Eventual success | Eventual futility | Eventual no decision | Expected total N | Expected remaining N |"
    )
    .unwrap();
    writeln!(&mut markdown, "| ---: | ---: | ---: | ---: | ---: |").unwrap();
    writeln!(
        &mut markdown,
        "| {} | {} | {} | {} | {} |",
        format_decimal(report.posterior_predictive.eventual_success_probability),
        format_decimal(report.posterior_predictive.eventual_futility_probability),
        format_decimal(report.posterior_predictive.eventual_no_decision_probability),
        format_decimal(report.posterior_predictive.expected_total_sample_size),
        format_decimal(report.posterior_predictive.expected_remaining_sample_size)
    )
    .unwrap();
    writeln!(&mut markdown).unwrap();

    writeln!(&mut markdown, "### Future Looks").unwrap();
    writeln!(&mut markdown).unwrap();
    if report.posterior_predictive.future_look_summaries.is_empty() {
        writeln!(&mut markdown, "_No future looks remain._").unwrap();
    } else {
        writeln!(&mut markdown, "| Future look | Stop | Success | Futility |").unwrap();
        writeln!(&mut markdown, "| --- | ---: | ---: | ---: |").unwrap();
        for look in &report.posterior_predictive.future_look_summaries {
            writeln!(
                &mut markdown,
                "| {} | {} | {} | {} |",
                look.look_id,
                format_decimal(look.conditional_stop_probability),
                format_decimal(look.conditional_success_probability),
                format_decimal(look.conditional_futility_probability)
            )
            .unwrap();
        }
    }
    writeln!(&mut markdown).unwrap();

    writeln!(&mut markdown, "## Prior Sensitivity").unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(
        &mut markdown,
        "| Variant | Baseline | Action | Posterior Pr(effect > margin) | Delta vs baseline | Eventual success | Delta vs baseline | Expected total N | Delta vs baseline |"
    )
    .unwrap();
    writeln!(&mut markdown, "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        .unwrap();
    for variant in &report.prior_sensitivity.variants {
        writeln!(
            &mut markdown,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            variant.variant_id,
            if variant.is_baseline { "yes" } else { "no" },
            format_beta_action(&variant.recommended_action),
            format_decimal(variant.posterior_probability_gt_margin),
            format_signed_decimal(variant.posterior_probability_delta_vs_baseline),
            format_decimal(variant.eventual_success_probability),
            format_signed_decimal(variant.eventual_success_probability_delta_vs_baseline),
            format_decimal(variant.expected_total_sample_size),
            format_signed_decimal(variant.expected_total_sample_size_delta_vs_baseline)
        )
        .unwrap();
    }

    Ok(markdown)
}

fn beta_terminal_posterior_predictive_result(
    spec: &BetaBinomialDesignSpec,
    current_analysis: BetaBinomialDesignAnalysisResult,
    seed: u64,
    n_replicates: usize,
) -> BetaBinomialPosteriorPredictiveResult {
    let current_total_sample_size =
        (current_analysis.look.n_control + current_analysis.look.n_treatment) as f64;
    let (
        eventual_success_probability,
        eventual_futility_probability,
        eventual_no_decision_probability,
    ) = if current_analysis.decision.success {
        (1.0, 0.0, 0.0)
    } else if current_analysis.decision.futility {
        (0.0, 1.0, 0.0)
    } else {
        (0.0, 0.0, 1.0)
    };
    BetaBinomialPosteriorPredictiveResult {
        schema_version: BETA_BINOMIAL_POSTERIOR_PREDICTIVE_SCHEMA_V0.to_string(),
        stability: "research-grade".to_string(),
        design_id: spec.design_id.clone(),
        current_analysis,
        n_replicates,
        seed,
        current_total_sample_size,
        expected_total_sample_size: current_total_sample_size,
        expected_remaining_sample_size: 0.0,
        eventual_success_probability,
        eventual_futility_probability,
        eventual_no_decision_probability,
        future_look_summaries: Vec::new(),
    }
}

fn beta_prior_sensitivity_variant_result(
    variant_id: &str,
    is_baseline: bool,
    control_prior: BetaPrior,
    treatment_prior: BetaPrior,
    forecast: &BetaBinomialPosteriorPredictiveResult,
    baseline: &BetaBinomialPosteriorPredictiveResult,
) -> BetaBinomialPriorSensitivityVariantResult {
    BetaBinomialPriorSensitivityVariantResult {
        variant_id: variant_id.to_string(),
        is_baseline,
        control_prior,
        treatment_prior,
        posterior_mean: forecast.current_analysis.posterior.effect_difference.posterior_mean,
        posterior_probability_gt_margin: forecast
            .current_analysis
            .posterior
            .effect_difference
            .posterior_probability_gt_margin,
        recommended_action: forecast.current_analysis.decision.recommended_action.clone(),
        eventual_success_probability: forecast.eventual_success_probability,
        eventual_futility_probability: forecast.eventual_futility_probability,
        eventual_no_decision_probability: forecast.eventual_no_decision_probability,
        expected_total_sample_size: forecast.expected_total_sample_size,
        expected_remaining_sample_size: forecast.expected_remaining_sample_size,
        future_look_summaries: forecast.future_look_summaries.clone(),
        posterior_probability_delta_vs_baseline: forecast
            .current_analysis
            .posterior
            .effect_difference
            .posterior_probability_gt_margin
            - baseline.current_analysis.posterior.effect_difference.posterior_probability_gt_margin,
        eventual_success_probability_delta_vs_baseline: forecast.eventual_success_probability
            - baseline.eventual_success_probability,
        expected_total_sample_size_delta_vs_baseline: forecast.expected_total_sample_size
            - baseline.expected_total_sample_size,
    }
}

fn validate_prior(prior: &BetaPrior, name: &str) -> Result<()> {
    if !prior.alpha.is_finite() || prior.alpha <= 0.0 {
        return Err(Error::Validation(format!("{}.alpha must be finite and > 0", name)));
    }
    if !prior.beta.is_finite() || prior.beta <= 0.0 {
        return Err(Error::Validation(format!("{}.beta must be finite and > 0", name)));
    }
    Ok(())
}

fn validate_probability_unit_interval(value: f64, label: &str) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(Error::Validation(format!("{} must be finite and in [0,1]", label)));
    }
    Ok(())
}

fn find_look<'a>(spec: &'a BetaBinomialDesignSpec, look_id: &str) -> Result<&'a BetaBinomialLook> {
    spec.looks.iter().find(|look| look.id == look_id).ok_or_else(|| {
        Error::Validation(format!("look_id '{}' was not found in the design", look_id))
    })
}

fn validate_observed_counts(
    look: &BetaBinomialLook,
    observed: &BetaBinomialObservedData,
) -> Result<()> {
    if observed.control_successes > look.n_control {
        return Err(Error::Validation(format!(
            "control_successes must be <= n_control at look '{}'",
            look.id
        )));
    }
    if observed.treatment_successes > look.n_treatment {
        return Err(Error::Validation(format!(
            "treatment_successes must be <= n_treatment at look '{}'",
            look.id
        )));
    }
    Ok(())
}

fn build_analysis_result(
    spec: &BetaBinomialDesignSpec,
    look: &BetaBinomialLook,
    observed: BetaBinomialObservedData,
    margin: f64,
) -> Result<BetaBinomialDesignAnalysisResult> {
    let control_posterior =
        posterior_from_counts(&spec.control_prior, observed.control_successes, look.n_control)?;
    let treatment_posterior = posterior_from_counts(
        &spec.treatment_prior,
        observed.treatment_successes,
        look.n_treatment,
    )?;
    let control_summary =
        summarize_posterior(control_posterior, spec.analysis.credible_interval_level)?;
    let treatment_summary =
        summarize_posterior(treatment_posterior, spec.analysis.credible_interval_level)?;
    let probability_gt_margin =
        posterior_probability_gt_margin(control_posterior, treatment_posterior, margin)?;
    let posterior_mean = treatment_summary.mean - control_summary.mean;
    let success =
        probability_gt_margin >= spec.decision_rules.success.posterior_probability_threshold;
    let futility =
        probability_gt_margin <= spec.decision_rules.futility.posterior_probability_threshold;
    let recommended_action = if success {
        BetaBinomialRecommendedAction::StopForSuccess
    } else if futility {
        BetaBinomialRecommendedAction::StopForFutility
    } else {
        BetaBinomialRecommendedAction::Continue
    };

    Ok(BetaBinomialDesignAnalysisResult {
        schema_version: BETA_BINOMIAL_DESIGN_ANALYSIS_SCHEMA_V0.to_string(),
        stability: "research-grade".to_string(),
        design_id: spec.design_id.clone(),
        look: look.clone(),
        observed,
        posterior: BetaBinomialPosteriorSummary {
            control: control_summary,
            treatment: treatment_summary,
            effect_difference: BetaBinomialEffectDifferenceSummary {
                margin,
                posterior_mean,
                posterior_probability_gt_margin: probability_gt_margin,
            },
        },
        decision: BetaBinomialDecisionSummary {
            success,
            futility,
            recommended_action,
            posterior_probability_gt_margin: probability_gt_margin,
            success_threshold: spec.decision_rules.success.posterior_probability_threshold,
            futility_threshold: spec.decision_rules.futility.posterior_probability_threshold,
            margin,
        },
    })
}

fn posterior_from_counts(
    prior: &BetaPrior,
    successes: u64,
    n_total: u64,
) -> Result<BetaPosteriorParams> {
    if successes > n_total {
        return Err(Error::Validation(format!(
            "successes must be <= n_total, got {} > {}",
            successes, n_total
        )));
    }
    Ok(BetaPosteriorParams {
        alpha: prior.alpha + successes as f64,
        beta: prior.beta + (n_total - successes) as f64,
    })
}

fn summarize_posterior(
    posterior: BetaPosteriorParams,
    credible_interval_level: f64,
) -> Result<BetaPosteriorSummary> {
    let dist = beta_distribution(posterior)?;
    let tail = (1.0 - credible_interval_level) / 2.0;
    Ok(BetaPosteriorSummary {
        alpha: posterior.alpha,
        beta: posterior.beta,
        mean: posterior.alpha / (posterior.alpha + posterior.beta),
        ci_lower: dist.inverse_cdf(tail),
        ci_upper: dist.inverse_cdf(1.0 - tail),
    })
}

fn beta_distribution(posterior: BetaPosteriorParams) -> Result<Beta> {
    Beta::new(posterior.alpha, posterior.beta).map_err(|e| {
        Error::Computation(format!(
            "failed to construct Beta({}, {}): {}",
            posterior.alpha, posterior.beta, e
        ))
    })
}

fn sample_beta_probability(posterior: BetaPosteriorParams, rng: &mut StdRng) -> Result<f64> {
    let dist = RandBeta::new(posterior.alpha, posterior.beta).map_err(|e| {
        Error::Computation(format!(
            "failed to construct posterior predictive Beta({}, {}): {}",
            posterior.alpha, posterior.beta, e
        ))
    })?;
    Ok(dist.sample(rng))
}

fn posterior_probability_gt_margin(
    control: BetaPosteriorParams,
    treatment: BetaPosteriorParams,
    margin: f64,
) -> Result<f64> {
    if !margin.is_finite() {
        return Err(Error::Validation("margin must be finite".to_string()));
    }
    if margin <= -1.0 {
        return Ok(1.0);
    }
    if margin >= 1.0 {
        return Ok(0.0);
    }

    let control_dist = beta_distribution(control)?;
    let treatment_dist = beta_distribution(treatment)?;
    let n_grid = SUPERIORITY_PROBABILITY_GRID_SIZE as f64;
    let mut sum = 0.0;
    for idx in 0..SUPERIORITY_PROBABILITY_GRID_SIZE {
        let u = (idx as f64 + 0.5) / n_grid;
        let treatment_quantile = treatment_dist.inverse_cdf(u);
        let shifted = treatment_quantile - margin;
        let cdf = if shifted <= 0.0 {
            0.0
        } else if shifted >= 1.0 {
            1.0
        } else {
            control_dist.cdf(shifted)
        };
        sum += cdf;
    }
    Ok((sum / n_grid).clamp(0.0, 1.0))
}

impl NormalNormalDesignSpec {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != NORMAL_NORMAL_DESIGN_SCHEMA_V0 {
            return Err(Error::Validation(format!(
                "schema_version must be '{}', got '{}'",
                NORMAL_NORMAL_DESIGN_SCHEMA_V0, self.schema_version
            )));
        }
        if self.design_id.trim().is_empty() {
            return Err(Error::Validation("design_id must be non-empty".to_string()));
        }
        validate_normal_prior(&self.control_prior, "control_prior")?;
        validate_normal_prior(&self.treatment_prior, "treatment_prior")?;
        validate_positive_sd(self.likelihood.known_sd_control, "likelihood.known_sd_control")?;
        validate_positive_sd(self.likelihood.known_sd_treatment, "likelihood.known_sd_treatment")?;
        if self.looks.is_empty() {
            return Err(Error::Validation(
                "normal-normal design requires at least one look".to_string(),
            ));
        }
        validate_probability_unit_interval(
            self.decision_rules.success.posterior_probability_threshold,
            "decision_rules.success.posterior_probability_threshold",
        )?;
        validate_probability_unit_interval(
            self.decision_rules.futility.posterior_probability_threshold,
            "decision_rules.futility.posterior_probability_threshold",
        )?;
        if self.decision_rules.success.posterior_probability_threshold
            <= self.decision_rules.futility.posterior_probability_threshold
        {
            return Err(Error::Validation(
                "success threshold must be strictly greater than futility threshold".to_string(),
            ));
        }
        if !self.decision_rules.success.treatment_effect_margin.is_finite() {
            return Err(Error::Validation(
                "decision_rules.success.treatment_effect_margin must be finite".to_string(),
            ));
        }
        if !self.decision_rules.futility.treatment_effect_margin.is_finite() {
            return Err(Error::Validation(
                "decision_rules.futility.treatment_effect_margin must be finite".to_string(),
            ));
        }
        if (self.decision_rules.success.treatment_effect_margin
            - self.decision_rules.futility.treatment_effect_margin)
            .abs()
            > 1e-12
        {
            return Err(Error::Validation(
                "success and futility rules must share the same treatment_effect_margin in v0"
                    .to_string(),
            ));
        }
        let ci = self.analysis.credible_interval_level;
        if !ci.is_finite() || !(0.0..1.0).contains(&ci) {
            return Err(Error::Validation(
                "analysis.credible_interval_level must be finite and in (0,1)".to_string(),
            ));
        }

        let mut seen_look_ids = HashSet::with_capacity(self.looks.len());
        let mut prev_n_control = 0_u64;
        let mut prev_n_treatment = 0_u64;
        for look in &self.looks {
            if look.id.trim().is_empty() {
                return Err(Error::Validation("look id must be non-empty".to_string()));
            }
            if !seen_look_ids.insert(look.id.clone()) {
                return Err(Error::Validation(format!("duplicate look id '{}'", look.id)));
            }
            if look.n_control == 0 {
                return Err(Error::Validation(format!(
                    "look '{}' must have n_control > 0",
                    look.id
                )));
            }
            if look.n_treatment == 0 {
                return Err(Error::Validation(format!(
                    "look '{}' must have n_treatment > 0",
                    look.id
                )));
            }
            if look.n_control < prev_n_control {
                return Err(Error::Validation(
                    "looks must be cumulative and non-decreasing for n_control".to_string(),
                ));
            }
            if look.n_treatment < prev_n_treatment {
                return Err(Error::Validation(
                    "looks must be cumulative and non-decreasing for n_treatment".to_string(),
                ));
            }
            prev_n_control = look.n_control;
            prev_n_treatment = look.n_treatment;
        }

        if let Some(sim) = &self.simulation {
            if sim.n_replicates == 0 {
                return Err(Error::Validation("simulation.n_replicates must be >= 1".to_string()));
            }
            if sim.scenarios.is_empty() {
                return Err(Error::Validation(
                    "simulation.scenarios must contain at least one scenario".to_string(),
                ));
            }
            let mut seen_scenario_ids = HashSet::with_capacity(sim.scenarios.len());
            for scenario in &sim.scenarios {
                if scenario.id.trim().is_empty() {
                    return Err(Error::Validation(
                        "simulation scenario id must be non-empty".to_string(),
                    ));
                }
                if !seen_scenario_ids.insert(scenario.id.clone()) {
                    return Err(Error::Validation(format!(
                        "duplicate simulation scenario id '{}'",
                        scenario.id
                    )));
                }
                if !scenario.mean_control.is_finite() {
                    return Err(Error::Validation(format!(
                        "simulation scenario '{}' mean_control must be finite",
                        scenario.id
                    )));
                }
                if !scenario.mean_treatment.is_finite() {
                    return Err(Error::Validation(format!(
                        "simulation scenario '{}' mean_treatment must be finite",
                        scenario.id
                    )));
                }
            }
        }

        Ok(())
    }
}

impl NormalNormalPriorSensitivityCampaign {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != NORMAL_NORMAL_PRIOR_SENSITIVITY_CAMPAIGN_SCHEMA_V0 {
            return Err(Error::Validation(format!(
                "schema_version must be '{}', got '{}'",
                NORMAL_NORMAL_PRIOR_SENSITIVITY_CAMPAIGN_SCHEMA_V0, self.schema_version
            )));
        }
        if self.variants.is_empty() {
            return Err(Error::Validation(
                "prior sensitivity campaign requires at least one variant".to_string(),
            ));
        }
        let mut seen_variant_ids = HashSet::with_capacity(self.variants.len());
        for variant in &self.variants {
            if variant.id.trim().is_empty() {
                return Err(Error::Validation(
                    "prior sensitivity variant id must be non-empty".to_string(),
                ));
            }
            if variant.id == "baseline" {
                return Err(Error::Validation(
                    "prior sensitivity variant id 'baseline' is reserved".to_string(),
                ));
            }
            if !seen_variant_ids.insert(variant.id.clone()) {
                return Err(Error::Validation(format!(
                    "duplicate prior sensitivity variant id '{}'",
                    variant.id
                )));
            }
            validate_normal_prior(&variant.control_prior, "variant.control_prior")?;
            validate_normal_prior(&variant.treatment_prior, "variant.treatment_prior")?;
        }
        Ok(())
    }
}

pub fn analyze_normal_normal_design(
    spec: &NormalNormalDesignSpec,
    observed: &NormalNormalObservedData,
) -> Result<NormalNormalDesignAnalysisResult> {
    spec.validate()?;
    let look = find_normal_look(spec, &observed.look_id)?;
    validate_normal_observed(look, observed)?;
    let margin = spec.decision_rules.success.treatment_effect_margin;
    build_normal_normal_analysis_result(spec, look, observed.clone(), margin)
}

pub fn normal_normal_operating_characteristics(
    spec: &NormalNormalDesignSpec,
) -> Result<NormalNormalOperatingCharacteristicsResult> {
    spec.validate()?;
    let sim = spec.simulation.as_ref().ok_or_else(|| {
        Error::Validation(
            "simulation block is required for normal-normal operating characteristics".to_string(),
        )
    })?;
    let mut rng = StdRng::seed_from_u64(sim.seed);
    let final_look = spec.looks.last().expect("validated non-empty looks");

    let mut scenarios = Vec::with_capacity(sim.scenarios.len());
    for scenario in &sim.scenarios {
        let mut success_count = 0usize;
        let mut futility_count = 0usize;
        let mut no_decision_count = 0usize;
        let mut total_sample_size_sum = 0.0;
        let mut look_stop_counts = vec![0usize; spec.looks.len()];
        let mut look_success_counts = vec![0usize; spec.looks.len()];
        let mut look_futility_counts = vec![0usize; spec.looks.len()];

        for _ in 0..sim.n_replicates {
            let mut prev_n_control = 0_u64;
            let mut prev_n_treatment = 0_u64;
            let mut control_sum = 0.0;
            let mut treatment_sum = 0.0;
            let mut stopped = false;

            for (look_idx, look) in spec.looks.iter().enumerate() {
                let inc_control = look.n_control - prev_n_control;
                let inc_treatment = look.n_treatment - prev_n_treatment;
                control_sum += sample_normal_sum(
                    inc_control,
                    scenario.mean_control,
                    spec.likelihood.known_sd_control,
                    &mut rng,
                )?;
                treatment_sum += sample_normal_sum(
                    inc_treatment,
                    scenario.mean_treatment,
                    spec.likelihood.known_sd_treatment,
                    &mut rng,
                )?;

                let observed = NormalNormalObservedData {
                    look_id: look.id.clone(),
                    control_sample_mean: control_sum / look.n_control as f64,
                    treatment_sample_mean: treatment_sum / look.n_treatment as f64,
                };
                let analysis = build_normal_normal_analysis_result(
                    spec,
                    look,
                    observed,
                    spec.decision_rules.success.treatment_effect_margin,
                )?;
                let total_n = (look.n_control + look.n_treatment) as f64;

                if analysis.decision.success {
                    success_count += 1;
                    look_stop_counts[look_idx] += 1;
                    look_success_counts[look_idx] += 1;
                    total_sample_size_sum += total_n;
                    stopped = true;
                    break;
                }
                if analysis.decision.futility {
                    futility_count += 1;
                    look_stop_counts[look_idx] += 1;
                    look_futility_counts[look_idx] += 1;
                    total_sample_size_sum += total_n;
                    stopped = true;
                    break;
                }

                prev_n_control = look.n_control;
                prev_n_treatment = look.n_treatment;
            }

            if !stopped {
                no_decision_count += 1;
                total_sample_size_sum += (final_look.n_control + final_look.n_treatment) as f64;
            }
        }

        let n_reps = sim.n_replicates as f64;
        let look_summaries = spec
            .looks
            .iter()
            .enumerate()
            .map(|(idx, look)| NormalNormalLookOperatingCharacteristics {
                look_id: look.id.clone(),
                stop_probability: look_stop_counts[idx] as f64 / n_reps,
                success_probability: look_success_counts[idx] as f64 / n_reps,
                futility_probability: look_futility_counts[idx] as f64 / n_reps,
            })
            .collect();

        scenarios.push(NormalNormalScenarioOperatingCharacteristics {
            scenario_id: scenario.id.clone(),
            mean_control: scenario.mean_control,
            mean_treatment: scenario.mean_treatment,
            success_rate: success_count as f64 / n_reps,
            futility_rate: futility_count as f64 / n_reps,
            no_decision_rate: no_decision_count as f64 / n_reps,
            expected_total_sample_size: total_sample_size_sum / n_reps,
            look_summaries,
        });
    }

    Ok(NormalNormalOperatingCharacteristicsResult {
        schema_version: NORMAL_NORMAL_OPERATING_CHARACTERISTICS_SCHEMA_V0.to_string(),
        stability: "research-grade".to_string(),
        design_id: spec.design_id.clone(),
        n_replicates: sim.n_replicates,
        seed: sim.seed,
        scenarios,
    })
}

pub fn normal_normal_posterior_predictive(
    spec: &NormalNormalDesignSpec,
    observed: &NormalNormalObservedData,
) -> Result<NormalNormalPosteriorPredictiveResult> {
    spec.validate()?;
    let sim = spec.simulation.as_ref().ok_or_else(|| {
        Error::Validation(
            "simulation block is required for normal-normal posterior predictive forecast"
                .to_string(),
        )
    })?;
    let look_idx =
        spec.looks.iter().position(|look| look.id == observed.look_id).ok_or_else(|| {
            Error::Validation(format!("look_id '{}' was not found in the design", observed.look_id))
        })?;
    let look = &spec.looks[look_idx];
    validate_normal_observed(look, observed)?;
    let margin = spec.decision_rules.success.treatment_effect_margin;
    let current_analysis =
        build_normal_normal_analysis_result(spec, look, observed.clone(), margin)?;
    let current_total_sample_size = (look.n_control + look.n_treatment) as f64;

    if current_analysis.decision.success
        || current_analysis.decision.futility
        || look_idx + 1 == spec.looks.len()
    {
        return Ok(normal_terminal_posterior_predictive_result(
            spec,
            current_analysis,
            sim.seed,
            sim.n_replicates,
        ));
    }

    let control_posterior = posterior_from_normal_sample_mean(
        &spec.control_prior,
        spec.likelihood.known_sd_control,
        look.n_control,
        observed.control_sample_mean,
    )?;
    let treatment_posterior = posterior_from_normal_sample_mean(
        &spec.treatment_prior,
        spec.likelihood.known_sd_treatment,
        look.n_treatment,
        observed.treatment_sample_mean,
    )?;
    let future_looks = &spec.looks[(look_idx + 1)..];
    let final_look = spec.looks.last().expect("validated non-empty looks");

    let mut rng = StdRng::seed_from_u64(sim.seed);
    let mut success_count = 0usize;
    let mut futility_count = 0usize;
    let mut no_decision_count = 0usize;
    let mut total_sample_size_sum = 0.0;
    let mut look_stop_counts = vec![0usize; future_looks.len()];
    let mut look_success_counts = vec![0usize; future_looks.len()];
    let mut look_futility_counts = vec![0usize; future_looks.len()];

    for _ in 0..sim.n_replicates {
        let latent_control_mean = sample_normal_mean(control_posterior, &mut rng)?;
        let latent_treatment_mean = sample_normal_mean(treatment_posterior, &mut rng)?;
        let mut prev_n_control = look.n_control;
        let mut prev_n_treatment = look.n_treatment;
        let mut control_sum = observed.control_sample_mean * look.n_control as f64;
        let mut treatment_sum = observed.treatment_sample_mean * look.n_treatment as f64;
        let mut stopped = false;

        for (future_idx, future_look) in future_looks.iter().enumerate() {
            let inc_control = future_look.n_control - prev_n_control;
            let inc_treatment = future_look.n_treatment - prev_n_treatment;
            control_sum += sample_normal_sum(
                inc_control,
                latent_control_mean,
                spec.likelihood.known_sd_control,
                &mut rng,
            )?;
            treatment_sum += sample_normal_sum(
                inc_treatment,
                latent_treatment_mean,
                spec.likelihood.known_sd_treatment,
                &mut rng,
            )?;

            let future_observed = NormalNormalObservedData {
                look_id: future_look.id.clone(),
                control_sample_mean: control_sum / future_look.n_control as f64,
                treatment_sample_mean: treatment_sum / future_look.n_treatment as f64,
            };
            let analysis =
                build_normal_normal_analysis_result(spec, future_look, future_observed, margin)?;
            let total_n = (future_look.n_control + future_look.n_treatment) as f64;

            if analysis.decision.success {
                success_count += 1;
                look_stop_counts[future_idx] += 1;
                look_success_counts[future_idx] += 1;
                total_sample_size_sum += total_n;
                stopped = true;
                break;
            }
            if analysis.decision.futility {
                futility_count += 1;
                look_stop_counts[future_idx] += 1;
                look_futility_counts[future_idx] += 1;
                total_sample_size_sum += total_n;
                stopped = true;
                break;
            }

            prev_n_control = future_look.n_control;
            prev_n_treatment = future_look.n_treatment;
        }

        if !stopped {
            no_decision_count += 1;
            total_sample_size_sum += (final_look.n_control + final_look.n_treatment) as f64;
        }
    }

    let n_reps = sim.n_replicates as f64;
    let future_look_summaries = future_looks
        .iter()
        .enumerate()
        .map(|(idx, future_look)| NormalNormalPosteriorPredictiveLookSummary {
            look_id: future_look.id.clone(),
            conditional_stop_probability: look_stop_counts[idx] as f64 / n_reps,
            conditional_success_probability: look_success_counts[idx] as f64 / n_reps,
            conditional_futility_probability: look_futility_counts[idx] as f64 / n_reps,
        })
        .collect();
    let expected_total_sample_size = total_sample_size_sum / n_reps;

    Ok(NormalNormalPosteriorPredictiveResult {
        schema_version: NORMAL_NORMAL_POSTERIOR_PREDICTIVE_SCHEMA_V0.to_string(),
        stability: "research-grade".to_string(),
        design_id: spec.design_id.clone(),
        current_analysis,
        n_replicates: sim.n_replicates,
        seed: sim.seed,
        current_total_sample_size,
        expected_total_sample_size,
        expected_remaining_sample_size: (expected_total_sample_size - current_total_sample_size)
            .max(0.0),
        eventual_success_probability: success_count as f64 / n_reps,
        eventual_futility_probability: futility_count as f64 / n_reps,
        eventual_no_decision_probability: no_decision_count as f64 / n_reps,
        future_look_summaries,
    })
}

pub fn normal_normal_prior_sensitivity(
    spec: &NormalNormalDesignSpec,
    observed: &NormalNormalObservedData,
    campaign: &NormalNormalPriorSensitivityCampaign,
) -> Result<NormalNormalPriorSensitivityReport> {
    spec.validate()?;
    campaign.validate()?;
    let baseline_forecast = normal_normal_posterior_predictive(spec, observed)?;
    let baseline_variant = normal_prior_sensitivity_variant_result(
        "baseline",
        true,
        spec.control_prior.clone(),
        spec.treatment_prior.clone(),
        &baseline_forecast,
        &baseline_forecast,
    );

    let mut variants = Vec::with_capacity(campaign.variants.len() + 1);
    variants.push(baseline_variant);

    for variant in &campaign.variants {
        let mut variant_spec = spec.clone();
        variant_spec.control_prior = variant.control_prior.clone();
        variant_spec.treatment_prior = variant.treatment_prior.clone();
        let forecast = normal_normal_posterior_predictive(&variant_spec, observed)?;
        variants.push(normal_prior_sensitivity_variant_result(
            &variant.id,
            false,
            variant.control_prior.clone(),
            variant.treatment_prior.clone(),
            &forecast,
            &baseline_forecast,
        ));
    }

    Ok(NormalNormalPriorSensitivityReport {
        schema_version: NORMAL_NORMAL_PRIOR_SENSITIVITY_REPORT_SCHEMA_V0.to_string(),
        stability: "research-grade".to_string(),
        design_id: spec.design_id.clone(),
        look: baseline_forecast.current_analysis.look.clone(),
        observed: observed.clone(),
        n_replicates: baseline_forecast.n_replicates,
        seed: baseline_forecast.seed,
        variants,
    })
}

pub fn normal_normal_design_report(
    spec: &NormalNormalDesignSpec,
    observed: &NormalNormalObservedData,
    campaign: &NormalNormalPriorSensitivityCampaign,
) -> Result<NormalNormalDesignReport> {
    spec.validate()?;
    campaign.validate()?;

    let posterior_predictive = normal_normal_posterior_predictive(spec, observed)?;
    let operating_characteristics = normal_normal_operating_characteristics(spec)?;
    let prior_sensitivity = normal_normal_prior_sensitivity(spec, observed, campaign)?;
    let sim = spec.simulation.as_ref().ok_or_else(|| {
        Error::Validation(
            "simulation block is required for normal-normal design report".to_string(),
        )
    })?;

    Ok(NormalNormalDesignReport {
        schema_version: NORMAL_NORMAL_DESIGN_REPORT_SCHEMA_V0.to_string(),
        stability: "research-grade".to_string(),
        design_family: "normal_normal".to_string(),
        design_spec: spec.clone(),
        current_analysis: posterior_predictive.current_analysis.clone(),
        operating_characteristics,
        posterior_predictive,
        prior_sensitivity,
        provenance: design_report_provenance(
            NORMAL_NORMAL_DESIGN_SCHEMA_V0,
            NORMAL_NORMAL_DESIGN_ANALYSIS_SCHEMA_V0,
            NORMAL_NORMAL_OPERATING_CHARACTERISTICS_SCHEMA_V0,
            NORMAL_NORMAL_POSTERIOR_PREDICTIVE_SCHEMA_V0,
            NORMAL_NORMAL_PRIOR_SENSITIVITY_CAMPAIGN_SCHEMA_V0,
            NORMAL_NORMAL_PRIOR_SENSITIVITY_REPORT_SCHEMA_V0,
            sim.seed,
            sim.n_replicates,
        ),
    })
}

pub fn render_normal_normal_design_report_markdown(
    report: &NormalNormalDesignReport,
) -> Result<String> {
    if report.schema_version != NORMAL_NORMAL_DESIGN_REPORT_SCHEMA_V0 {
        return Err(Error::Validation(format!(
            "schema_version must be '{}', got '{}'",
            NORMAL_NORMAL_DESIGN_REPORT_SCHEMA_V0, report.schema_version
        )));
    }

    let spec = &report.design_spec;
    spec.validate()?;
    let mut markdown = String::new();
    writeln!(&mut markdown, "# Bayesian Trial Design Report").unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(&mut markdown, "- Family: normal_normal").unwrap();
    writeln!(&mut markdown, "- Design ID: {}", report.current_analysis.design_id).unwrap();
    writeln!(&mut markdown, "- Stability: {}", report.stability).unwrap();
    writeln!(
        &mut markdown,
        "- Software: {} {}",
        report.provenance.software_name, report.provenance.software_version
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "- Simulation: {} replicates, seed {}",
        report.provenance.n_replicates, report.provenance.simulation_seed
    )
    .unwrap();
    writeln!(&mut markdown).unwrap();

    writeln!(&mut markdown, "## Provenance").unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(&mut markdown, "- Report schema: {}", report.schema_version).unwrap();
    writeln!(&mut markdown, "- Design schema: {}", report.provenance.design_schema_version)
        .unwrap();
    writeln!(&mut markdown, "- Analysis schema: {}", report.provenance.analysis_schema_version)
        .unwrap();
    writeln!(
        &mut markdown,
        "- Operating characteristics schema: {}",
        report.provenance.operating_characteristics_schema_version
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "- Posterior predictive schema: {}",
        report.provenance.posterior_predictive_schema_version
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "- Prior sensitivity campaign schema: {}",
        report.provenance.prior_sensitivity_campaign_schema_version
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "- Prior sensitivity report schema: {}",
        report.provenance.prior_sensitivity_report_schema_version
    )
    .unwrap();
    writeln!(&mut markdown).unwrap();

    writeln!(&mut markdown, "## Design Spec").unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(&mut markdown, "### Priors").unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(&mut markdown, "| Arm | Prior |").unwrap();
    writeln!(&mut markdown, "| --- | --- |").unwrap();
    writeln!(
        &mut markdown,
        "| Control | Normal(mean={}, sd={}) |",
        format_decimal(spec.control_prior.mean),
        format_decimal(spec.control_prior.sd)
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "| Treatment | Normal(mean={}, sd={}) |",
        format_decimal(spec.treatment_prior.mean),
        format_decimal(spec.treatment_prior.sd)
    )
    .unwrap();
    writeln!(&mut markdown).unwrap();

    writeln!(&mut markdown, "### Likelihood").unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(&mut markdown, "| Control known sd | Treatment known sd | Credible interval level |")
        .unwrap();
    writeln!(&mut markdown, "| ---: | ---: | ---: |").unwrap();
    writeln!(
        &mut markdown,
        "| {} | {} | {} |",
        format_decimal(spec.likelihood.known_sd_control),
        format_decimal(spec.likelihood.known_sd_treatment),
        format_decimal(spec.analysis.credible_interval_level)
    )
    .unwrap();
    writeln!(&mut markdown).unwrap();

    writeln!(&mut markdown, "### Looks").unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(&mut markdown, "| Look | N control | N treatment |").unwrap();
    writeln!(&mut markdown, "| --- | ---: | ---: |").unwrap();
    for look in &spec.looks {
        writeln!(&mut markdown, "| {} | {} | {} |", look.id, look.n_control, look.n_treatment)
            .unwrap();
    }
    writeln!(&mut markdown).unwrap();

    writeln!(&mut markdown, "### Decision Criteria").unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(&mut markdown, "| Rule | Posterior threshold | Margin |").unwrap();
    writeln!(&mut markdown, "| --- | ---: | ---: |").unwrap();
    writeln!(
        &mut markdown,
        "| Success | {} | {} |",
        format_decimal(spec.decision_rules.success.posterior_probability_threshold),
        format_decimal(spec.decision_rules.success.treatment_effect_margin)
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "| Futility | {} | {} |",
        format_decimal(spec.decision_rules.futility.posterior_probability_threshold),
        format_decimal(spec.decision_rules.futility.treatment_effect_margin)
    )
    .unwrap();
    writeln!(&mut markdown).unwrap();

    writeln!(&mut markdown, "### Simulation Scenarios").unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(&mut markdown, "| Scenario | Control mean | Treatment mean |").unwrap();
    writeln!(&mut markdown, "| --- | ---: | ---: |").unwrap();
    for scenario in &report.operating_characteristics.scenarios {
        writeln!(
            &mut markdown,
            "| {} | {} | {} |",
            scenario.scenario_id,
            format_decimal(scenario.mean_control),
            format_decimal(scenario.mean_treatment)
        )
        .unwrap();
    }
    writeln!(&mut markdown).unwrap();

    writeln!(&mut markdown, "## Current Analysis").unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(
        &mut markdown,
        "- Look: {} (N control = {}, N treatment = {})",
        report.current_analysis.look.id,
        report.current_analysis.look.n_control,
        report.current_analysis.look.n_treatment
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "- Observed means: control = {}, treatment = {}",
        format_decimal(report.current_analysis.observed.control_sample_mean),
        format_decimal(report.current_analysis.observed.treatment_sample_mean)
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "- Recommended action: {}",
        format_normal_action(&report.current_analysis.decision.recommended_action)
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "- Posterior mean treatment effect: {}",
        format_decimal(report.current_analysis.posterior.effect_difference.posterior_mean)
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "- Posterior Pr(effect > margin): {}",
        format_decimal(report.current_analysis.decision.posterior_probability_gt_margin)
    )
    .unwrap();
    writeln!(&mut markdown).unwrap();

    writeln!(&mut markdown, "## Operating Characteristics").unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(&mut markdown, "| Scenario | Success | Futility | No decision | Expected total N |")
        .unwrap();
    writeln!(&mut markdown, "| --- | ---: | ---: | ---: | ---: |").unwrap();
    for scenario in &report.operating_characteristics.scenarios {
        writeln!(
            &mut markdown,
            "| {} | {} | {} | {} | {} |",
            scenario.scenario_id,
            format_decimal(scenario.success_rate),
            format_decimal(scenario.futility_rate),
            format_decimal(scenario.no_decision_rate),
            format_decimal(scenario.expected_total_sample_size)
        )
        .unwrap();
    }
    writeln!(&mut markdown).unwrap();

    writeln!(&mut markdown, "## Posterior Predictive Forecast").unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(
        &mut markdown,
        "| Eventual success | Eventual futility | Eventual no decision | Expected total N | Expected remaining N |"
    )
    .unwrap();
    writeln!(&mut markdown, "| ---: | ---: | ---: | ---: | ---: |").unwrap();
    writeln!(
        &mut markdown,
        "| {} | {} | {} | {} | {} |",
        format_decimal(report.posterior_predictive.eventual_success_probability),
        format_decimal(report.posterior_predictive.eventual_futility_probability),
        format_decimal(report.posterior_predictive.eventual_no_decision_probability),
        format_decimal(report.posterior_predictive.expected_total_sample_size),
        format_decimal(report.posterior_predictive.expected_remaining_sample_size)
    )
    .unwrap();
    writeln!(&mut markdown).unwrap();

    writeln!(&mut markdown, "### Future Looks").unwrap();
    writeln!(&mut markdown).unwrap();
    if report.posterior_predictive.future_look_summaries.is_empty() {
        writeln!(&mut markdown, "_No future looks remain._").unwrap();
    } else {
        writeln!(&mut markdown, "| Future look | Stop | Success | Futility |").unwrap();
        writeln!(&mut markdown, "| --- | ---: | ---: | ---: |").unwrap();
        for look in &report.posterior_predictive.future_look_summaries {
            writeln!(
                &mut markdown,
                "| {} | {} | {} | {} |",
                look.look_id,
                format_decimal(look.conditional_stop_probability),
                format_decimal(look.conditional_success_probability),
                format_decimal(look.conditional_futility_probability)
            )
            .unwrap();
        }
    }
    writeln!(&mut markdown).unwrap();

    writeln!(&mut markdown, "## Prior Sensitivity").unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(
        &mut markdown,
        "| Variant | Baseline | Action | Posterior Pr(effect > margin) | Delta vs baseline | Eventual success | Delta vs baseline | Expected total N | Delta vs baseline |"
    )
    .unwrap();
    writeln!(&mut markdown, "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        .unwrap();
    for variant in &report.prior_sensitivity.variants {
        writeln!(
            &mut markdown,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            variant.variant_id,
            if variant.is_baseline { "yes" } else { "no" },
            format_normal_action(&variant.recommended_action),
            format_decimal(variant.posterior_probability_gt_margin),
            format_signed_decimal(variant.posterior_probability_delta_vs_baseline),
            format_decimal(variant.eventual_success_probability),
            format_signed_decimal(variant.eventual_success_probability_delta_vs_baseline),
            format_decimal(variant.expected_total_sample_size),
            format_signed_decimal(variant.expected_total_sample_size_delta_vs_baseline)
        )
        .unwrap();
    }

    Ok(markdown)
}

fn normal_terminal_posterior_predictive_result(
    spec: &NormalNormalDesignSpec,
    current_analysis: NormalNormalDesignAnalysisResult,
    seed: u64,
    n_replicates: usize,
) -> NormalNormalPosteriorPredictiveResult {
    let current_total_sample_size =
        (current_analysis.look.n_control + current_analysis.look.n_treatment) as f64;
    let (
        eventual_success_probability,
        eventual_futility_probability,
        eventual_no_decision_probability,
    ) = if current_analysis.decision.success {
        (1.0, 0.0, 0.0)
    } else if current_analysis.decision.futility {
        (0.0, 1.0, 0.0)
    } else {
        (0.0, 0.0, 1.0)
    };
    NormalNormalPosteriorPredictiveResult {
        schema_version: NORMAL_NORMAL_POSTERIOR_PREDICTIVE_SCHEMA_V0.to_string(),
        stability: "research-grade".to_string(),
        design_id: spec.design_id.clone(),
        current_analysis,
        n_replicates,
        seed,
        current_total_sample_size,
        expected_total_sample_size: current_total_sample_size,
        expected_remaining_sample_size: 0.0,
        eventual_success_probability,
        eventual_futility_probability,
        eventual_no_decision_probability,
        future_look_summaries: Vec::new(),
    }
}

fn normal_prior_sensitivity_variant_result(
    variant_id: &str,
    is_baseline: bool,
    control_prior: NormalPrior,
    treatment_prior: NormalPrior,
    forecast: &NormalNormalPosteriorPredictiveResult,
    baseline: &NormalNormalPosteriorPredictiveResult,
) -> NormalNormalPriorSensitivityVariantResult {
    NormalNormalPriorSensitivityVariantResult {
        variant_id: variant_id.to_string(),
        is_baseline,
        control_prior,
        treatment_prior,
        posterior_mean: forecast.current_analysis.posterior.effect_difference.posterior_mean,
        posterior_probability_gt_margin: forecast
            .current_analysis
            .posterior
            .effect_difference
            .posterior_probability_gt_margin,
        recommended_action: forecast.current_analysis.decision.recommended_action.clone(),
        eventual_success_probability: forecast.eventual_success_probability,
        eventual_futility_probability: forecast.eventual_futility_probability,
        eventual_no_decision_probability: forecast.eventual_no_decision_probability,
        expected_total_sample_size: forecast.expected_total_sample_size,
        expected_remaining_sample_size: forecast.expected_remaining_sample_size,
        future_look_summaries: forecast.future_look_summaries.clone(),
        posterior_probability_delta_vs_baseline: forecast
            .current_analysis
            .posterior
            .effect_difference
            .posterior_probability_gt_margin
            - baseline.current_analysis.posterior.effect_difference.posterior_probability_gt_margin,
        eventual_success_probability_delta_vs_baseline: forecast.eventual_success_probability
            - baseline.eventual_success_probability,
        expected_total_sample_size_delta_vs_baseline: forecast.expected_total_sample_size
            - baseline.expected_total_sample_size,
    }
}

fn validate_normal_prior(prior: &NormalPrior, name: &str) -> Result<()> {
    if !prior.mean.is_finite() {
        return Err(Error::Validation(format!("{}.mean must be finite", name)));
    }
    validate_positive_sd(prior.sd, &format!("{}.sd", name))
}

fn validate_positive_sd(value: f64, label: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(Error::Validation(format!("{} must be finite and > 0", label)));
    }
    Ok(())
}

fn find_normal_look<'a>(
    spec: &'a NormalNormalDesignSpec,
    look_id: &str,
) -> Result<&'a NormalNormalLook> {
    spec.looks.iter().find(|look| look.id == look_id).ok_or_else(|| {
        Error::Validation(format!("look_id '{}' was not found in the design", look_id))
    })
}

fn validate_normal_observed(
    look: &NormalNormalLook,
    observed: &NormalNormalObservedData,
) -> Result<()> {
    if !observed.control_sample_mean.is_finite() {
        return Err(Error::Validation(format!(
            "look '{}' requires finite control_sample_mean",
            look.id
        )));
    }
    if !observed.treatment_sample_mean.is_finite() {
        return Err(Error::Validation(format!(
            "look '{}' requires finite treatment_sample_mean",
            look.id
        )));
    }
    Ok(())
}

fn build_normal_normal_analysis_result(
    spec: &NormalNormalDesignSpec,
    look: &NormalNormalLook,
    observed: NormalNormalObservedData,
    margin: f64,
) -> Result<NormalNormalDesignAnalysisResult> {
    let control_posterior = posterior_from_normal_sample_mean(
        &spec.control_prior,
        spec.likelihood.known_sd_control,
        look.n_control,
        observed.control_sample_mean,
    )?;
    let treatment_posterior = posterior_from_normal_sample_mean(
        &spec.treatment_prior,
        spec.likelihood.known_sd_treatment,
        look.n_treatment,
        observed.treatment_sample_mean,
    )?;
    let control_summary =
        summarize_normal_posterior(control_posterior, spec.analysis.credible_interval_level)?;
    let treatment_summary =
        summarize_normal_posterior(treatment_posterior, spec.analysis.credible_interval_level)?;
    let diff_mean = treatment_posterior.mean - control_posterior.mean;
    let diff_variance = treatment_posterior.variance + control_posterior.variance;
    let diff_sd = diff_variance.sqrt();
    let diff_dist = normal_distribution(diff_mean, diff_sd)?;
    let tail = (1.0 - spec.analysis.credible_interval_level) / 2.0;
    let probability_gt_margin = 1.0 - diff_dist.cdf(margin);
    let success =
        probability_gt_margin >= spec.decision_rules.success.posterior_probability_threshold;
    let futility =
        probability_gt_margin <= spec.decision_rules.futility.posterior_probability_threshold;
    let recommended_action = if success {
        NormalNormalRecommendedAction::StopForSuccess
    } else if futility {
        NormalNormalRecommendedAction::StopForFutility
    } else {
        NormalNormalRecommendedAction::Continue
    };

    Ok(NormalNormalDesignAnalysisResult {
        schema_version: NORMAL_NORMAL_DESIGN_ANALYSIS_SCHEMA_V0.to_string(),
        stability: "research-grade".to_string(),
        design_id: spec.design_id.clone(),
        look: look.clone(),
        observed,
        posterior: NormalNormalPosteriorSummary {
            control: control_summary,
            treatment: treatment_summary,
            effect_difference: NormalNormalEffectDifferenceSummary {
                margin,
                posterior_mean: diff_mean,
                posterior_sd: diff_sd,
                ci_lower: diff_dist.inverse_cdf(tail),
                ci_upper: diff_dist.inverse_cdf(1.0 - tail),
                posterior_probability_gt_margin: probability_gt_margin,
            },
        },
        decision: NormalNormalDecisionSummary {
            success,
            futility,
            recommended_action,
            posterior_probability_gt_margin: probability_gt_margin,
            success_threshold: spec.decision_rules.success.posterior_probability_threshold,
            futility_threshold: spec.decision_rules.futility.posterior_probability_threshold,
            margin,
        },
    })
}

fn posterior_from_normal_sample_mean(
    prior: &NormalPrior,
    known_sd: f64,
    n_total: u64,
    sample_mean: f64,
) -> Result<NormalPosteriorParams> {
    validate_positive_sd(known_sd, "known_sd")?;
    if n_total == 0 {
        return Err(Error::Validation("n_total must be > 0".to_string()));
    }
    if !sample_mean.is_finite() {
        return Err(Error::Validation("sample_mean must be finite".to_string()));
    }

    let prior_variance = prior.sd * prior.sd;
    let known_variance = known_sd * known_sd;
    let posterior_precision = 1.0 / prior_variance + n_total as f64 / known_variance;
    let posterior_variance = 1.0 / posterior_precision;
    let posterior_mean = posterior_variance
        * ((prior.mean / prior_variance) + (n_total as f64 * sample_mean / known_variance));

    Ok(NormalPosteriorParams { mean: posterior_mean, variance: posterior_variance })
}

fn summarize_normal_posterior(
    posterior: NormalPosteriorParams,
    credible_interval_level: f64,
) -> Result<NormalPosteriorSummary> {
    let dist = normal_distribution(posterior.mean, posterior.variance.sqrt())?;
    let tail = (1.0 - credible_interval_level) / 2.0;
    Ok(NormalPosteriorSummary {
        posterior_mean: posterior.mean,
        posterior_sd: posterior.variance.sqrt(),
        ci_lower: dist.inverse_cdf(tail),
        ci_upper: dist.inverse_cdf(1.0 - tail),
    })
}

fn normal_distribution(mean: f64, sd: f64) -> Result<StatNormal> {
    StatNormal::new(mean, sd).map_err(|e| {
        Error::Computation(format!("failed to construct Normal({}, {}): {}", mean, sd, e))
    })
}

fn sample_normal_mean(posterior: NormalPosteriorParams, rng: &mut StdRng) -> Result<f64> {
    let dist = RandNormal::new(posterior.mean, posterior.variance.sqrt()).map_err(|e| {
        Error::Computation(format!(
            "failed to construct posterior predictive Normal({}, {}): {}",
            posterior.mean,
            posterior.variance.sqrt(),
            e
        ))
    })?;
    Ok(dist.sample(rng))
}

fn sample_normal_sum(n_total: u64, mean: f64, known_sd: f64, rng: &mut StdRng) -> Result<f64> {
    if n_total == 0 {
        return Ok(0.0);
    }
    let sd_sum = known_sd * (n_total as f64).sqrt();
    let dist = RandNormal::new(mean * n_total as f64, sd_sum).map_err(|e| {
        Error::Computation(format!(
            "failed to construct simulation normal(mean={}, sd={}): {}",
            mean * n_total as f64,
            sd_sum,
            e
        ))
    })?;
    Ok(dist.sample(rng))
}

fn design_report_provenance(
    design_schema_version: &str,
    analysis_schema_version: &str,
    operating_characteristics_schema_version: &str,
    posterior_predictive_schema_version: &str,
    prior_sensitivity_campaign_schema_version: &str,
    prior_sensitivity_report_schema_version: &str,
    simulation_seed: u64,
    n_replicates: usize,
) -> BayesianDesignReportProvenance {
    BayesianDesignReportProvenance {
        software_name: "nextstat".to_string(),
        software_version: ns_core::VERSION.to_string(),
        design_schema_version: design_schema_version.to_string(),
        analysis_schema_version: analysis_schema_version.to_string(),
        operating_characteristics_schema_version: operating_characteristics_schema_version
            .to_string(),
        posterior_predictive_schema_version: posterior_predictive_schema_version.to_string(),
        prior_sensitivity_campaign_schema_version: prior_sensitivity_campaign_schema_version
            .to_string(),
        prior_sensitivity_report_schema_version: prior_sensitivity_report_schema_version
            .to_string(),
        simulation_seed,
        n_replicates,
    }
}

fn format_decimal(value: f64) -> String {
    format!("{value:.3}")
}

fn format_signed_decimal(value: f64) -> String {
    format!("{value:+.3}")
}

fn format_beta_action(action: &BetaBinomialRecommendedAction) -> &'static str {
    match action {
        BetaBinomialRecommendedAction::Continue => "continue",
        BetaBinomialRecommendedAction::StopForSuccess => "stop_for_success",
        BetaBinomialRecommendedAction::StopForFutility => "stop_for_futility",
    }
}

fn format_normal_action(action: &NormalNormalRecommendedAction) -> &'static str {
    match action {
        NormalNormalRecommendedAction::Continue => "continue",
        NormalNormalRecommendedAction::StopForSuccess => "stop_for_success",
        NormalNormalRecommendedAction::StopForFutility => "stop_for_futility",
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    fn demo_spec() -> BetaBinomialDesignSpec {
        BetaBinomialDesignSpec {
            schema_version: BETA_BINOMIAL_DESIGN_SCHEMA_V0.to_string(),
            design_id: "binary_superiority_demo".to_string(),
            control_prior: BetaPrior { alpha: 1.0, beta: 1.0 },
            treatment_prior: BetaPrior { alpha: 1.0, beta: 1.0 },
            looks: vec![
                BetaBinomialLook { id: "interim".to_string(), n_control: 20, n_treatment: 20 },
                BetaBinomialLook { id: "final".to_string(), n_control: 40, n_treatment: 40 },
            ],
            decision_rules: BetaBinomialDecisionRules {
                success: BetaBinomialDecisionRule {
                    posterior_probability_threshold: 0.95,
                    treatment_effect_margin: 0.0,
                },
                futility: BetaBinomialDecisionRule {
                    posterior_probability_threshold: 0.20,
                    treatment_effect_margin: 0.0,
                },
            },
            analysis: BetaBinomialAnalysisConfig { credible_interval_level: 0.95 },
            simulation: Some(BetaBinomialSimulationConfig {
                n_replicates: 32,
                seed: 123,
                scenarios: vec![
                    BetaBinomialScenario {
                        id: "null".to_string(),
                        p_control: 0.4,
                        p_treatment: 0.4,
                    },
                    BetaBinomialScenario {
                        id: "alt".to_string(),
                        p_control: 0.4,
                        p_treatment: 0.6,
                    },
                ],
            }),
        }
    }

    fn normal_demo_spec() -> NormalNormalDesignSpec {
        NormalNormalDesignSpec {
            schema_version: NORMAL_NORMAL_DESIGN_SCHEMA_V0.to_string(),
            design_id: "continuous_superiority_demo".to_string(),
            control_prior: NormalPrior { mean: 0.0, sd: 10.0 },
            treatment_prior: NormalPrior { mean: 0.0, sd: 10.0 },
            likelihood: NormalNormalLikelihood { known_sd_control: 1.0, known_sd_treatment: 1.0 },
            looks: vec![
                NormalNormalLook { id: "interim".to_string(), n_control: 10, n_treatment: 10 },
                NormalNormalLook { id: "final".to_string(), n_control: 20, n_treatment: 20 },
            ],
            decision_rules: NormalNormalDecisionRules {
                success: NormalNormalDecisionRule {
                    posterior_probability_threshold: 0.975,
                    treatment_effect_margin: 0.0,
                },
                futility: NormalNormalDecisionRule {
                    posterior_probability_threshold: 0.10,
                    treatment_effect_margin: 0.0,
                },
            },
            analysis: NormalNormalAnalysisConfig { credible_interval_level: 0.95 },
            simulation: Some(NormalNormalSimulationConfig {
                n_replicates: 32,
                seed: 456,
                scenarios: vec![
                    NormalNormalScenario {
                        id: "null".to_string(),
                        mean_control: 0.0,
                        mean_treatment: 0.0,
                    },
                    NormalNormalScenario {
                        id: "alt".to_string(),
                        mean_control: 0.0,
                        mean_treatment: 0.75,
                    },
                ],
            }),
        }
    }

    fn beta_prior_campaign() -> BetaBinomialPriorSensitivityCampaign {
        BetaBinomialPriorSensitivityCampaign {
            schema_version: BETA_BINOMIAL_PRIOR_SENSITIVITY_CAMPAIGN_SCHEMA_V0.to_string(),
            variants: vec![
                BetaBinomialPriorSensitivityVariant {
                    id: "skeptical".to_string(),
                    control_prior: BetaPrior { alpha: 1.0, beta: 1.0 },
                    treatment_prior: BetaPrior { alpha: 1.0, beta: 8.0 },
                },
                BetaBinomialPriorSensitivityVariant {
                    id: "enthusiastic".to_string(),
                    control_prior: BetaPrior { alpha: 1.0, beta: 1.0 },
                    treatment_prior: BetaPrior { alpha: 8.0, beta: 1.0 },
                },
            ],
        }
    }

    fn normal_prior_campaign() -> NormalNormalPriorSensitivityCampaign {
        NormalNormalPriorSensitivityCampaign {
            schema_version: NORMAL_NORMAL_PRIOR_SENSITIVITY_CAMPAIGN_SCHEMA_V0.to_string(),
            variants: vec![
                NormalNormalPriorSensitivityVariant {
                    id: "skeptical".to_string(),
                    control_prior: NormalPrior { mean: 0.0, sd: 10.0 },
                    treatment_prior: NormalPrior { mean: -1.0, sd: 0.2 },
                },
                NormalNormalPriorSensitivityVariant {
                    id: "enthusiastic".to_string(),
                    control_prior: NormalPrior { mean: 0.0, sd: 10.0 },
                    treatment_prior: NormalPrior { mean: 1.0, sd: 0.2 },
                },
            ],
        }
    }

    #[test]
    fn test_validate_rejects_non_monotone_looks() {
        let mut spec = demo_spec();
        spec.looks[1].n_control = 10;
        let err = spec.validate().unwrap_err();
        assert!(
            err.to_string().contains("non-decreasing for n_control"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_identical_posteriors_imply_half_superiority_probability() {
        let posterior = BetaPosteriorParams { alpha: 9.0, beta: 13.0 };
        let prob = posterior_probability_gt_margin(posterior, posterior, 0.0).unwrap();
        assert_relative_eq!(prob, 0.5, epsilon = 5e-4);
    }

    #[test]
    fn test_analysis_updates_posterior_exactly() {
        let spec = demo_spec();
        let observed = BetaBinomialObservedData {
            look_id: "interim".to_string(),
            control_successes: 8,
            treatment_successes: 14,
        };
        let result = analyze_beta_binomial_design(&spec, &observed).unwrap();

        assert_relative_eq!(result.posterior.control.alpha, 9.0, epsilon = 1e-12);
        assert_relative_eq!(result.posterior.control.beta, 13.0, epsilon = 1e-12);
        assert_relative_eq!(result.posterior.control.mean, 9.0 / 22.0, epsilon = 1e-12);
        assert_relative_eq!(result.posterior.treatment.alpha, 15.0, epsilon = 1e-12);
        assert_relative_eq!(result.posterior.treatment.beta, 7.0, epsilon = 1e-12);
        assert_relative_eq!(
            result.posterior.effect_difference.posterior_mean,
            6.0 / 22.0,
            epsilon = 1e-12
        );
        assert!(result.posterior.effect_difference.posterior_probability_gt_margin > 0.95);
        assert_eq!(
            result.decision.recommended_action,
            BetaBinomialRecommendedAction::StopForSuccess
        );
    }

    #[test]
    fn test_operating_characteristics_are_deterministic() {
        let spec = demo_spec();
        let first = beta_binomial_operating_characteristics(&spec).unwrap();
        let second = beta_binomial_operating_characteristics(&spec).unwrap();

        assert_eq!(first, second);

        let null = first.scenarios.iter().find(|item| item.scenario_id == "null").unwrap();
        let alt = first.scenarios.iter().find(|item| item.scenario_id == "alt").unwrap();

        assert_relative_eq!(
            null.success_rate + null.futility_rate + null.no_decision_rate,
            1.0,
            epsilon = 1e-12
        );
        assert!(alt.success_rate > null.success_rate);
    }

    #[test]
    fn test_beta_posterior_predictive_is_deterministic() {
        let spec = demo_spec();
        let observed = BetaBinomialObservedData {
            look_id: "interim".to_string(),
            control_successes: 8,
            treatment_successes: 9,
        };
        let first = beta_binomial_posterior_predictive(&spec, &observed).unwrap();
        let second = beta_binomial_posterior_predictive(&spec, &observed).unwrap();

        assert_eq!(first, second);
        assert_relative_eq!(
            first.eventual_success_probability
                + first.eventual_futility_probability
                + first.eventual_no_decision_probability,
            1.0,
            epsilon = 1e-12
        );
        assert_eq!(first.future_look_summaries.len(), 1);
        assert_relative_eq!(
            first.eventual_success_probability,
            first.future_look_summaries[0].conditional_success_probability,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_beta_prior_sensitivity_orders_variants() {
        let spec = demo_spec();
        let observed = BetaBinomialObservedData {
            look_id: "interim".to_string(),
            control_successes: 8,
            treatment_successes: 9,
        };
        let report =
            beta_binomial_prior_sensitivity(&spec, &observed, &beta_prior_campaign()).unwrap();
        let baseline = report.variants.iter().find(|item| item.variant_id == "baseline").unwrap();
        let skeptical = report.variants.iter().find(|item| item.variant_id == "skeptical").unwrap();
        let enthusiastic =
            report.variants.iter().find(|item| item.variant_id == "enthusiastic").unwrap();

        assert_relative_eq!(baseline.posterior_probability_delta_vs_baseline, 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            baseline.eventual_success_probability_delta_vs_baseline,
            0.0,
            epsilon = 1e-12
        );
        assert!(
            skeptical.posterior_probability_gt_margin < baseline.posterior_probability_gt_margin
        );
        assert!(skeptical.eventual_success_probability < baseline.eventual_success_probability);
        assert!(enthusiastic.eventual_success_probability > baseline.eventual_success_probability);
    }

    #[test]
    fn test_normal_normal_validate_rejects_non_monotone_looks() {
        let mut spec = normal_demo_spec();
        spec.looks[1].n_treatment = 5;
        let err = spec.validate().unwrap_err();
        assert!(
            err.to_string().contains("non-decreasing for n_treatment"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_normal_normal_identical_posteriors_imply_half_superiority_probability() {
        let posterior = NormalPosteriorParams { mean: 0.5, variance: 0.25 };
        let dist = normal_distribution(0.0, (2.0 * posterior.variance).sqrt()).unwrap();
        let prob = 1.0 - dist.cdf(0.0);
        assert_relative_eq!(prob, 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_normal_normal_analysis_updates_posterior_exactly() {
        let spec = normal_demo_spec();
        let observed = NormalNormalObservedData {
            look_id: "interim".to_string(),
            control_sample_mean: 0.1,
            treatment_sample_mean: 1.0,
        };
        let result = analyze_normal_normal_design(&spec, &observed).unwrap();

        let posterior_variance: f64 = 1.0_f64 / (1.0_f64 / 100.0_f64 + 10.0_f64);
        let posterior_sd = posterior_variance.sqrt();
        let control_mean = posterior_variance * 10.0 * 0.1;
        let treatment_mean = posterior_variance * 10.0 * 1.0;
        let diff_mean = treatment_mean - control_mean;
        let diff_sd = (2.0_f64 * posterior_variance).sqrt();

        assert_relative_eq!(result.posterior.control.posterior_mean, control_mean, epsilon = 1e-12);
        assert_relative_eq!(result.posterior.control.posterior_sd, posterior_sd, epsilon = 1e-12);
        assert_relative_eq!(
            result.posterior.treatment.posterior_mean,
            treatment_mean,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            result.posterior.effect_difference.posterior_mean,
            diff_mean,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            result.posterior.effect_difference.posterior_sd,
            diff_sd,
            epsilon = 1e-12
        );
        assert!(result.posterior.effect_difference.posterior_probability_gt_margin > 0.975);
        assert_eq!(
            result.decision.recommended_action,
            NormalNormalRecommendedAction::StopForSuccess
        );
    }

    #[test]
    fn test_normal_normal_operating_characteristics_are_deterministic() {
        let spec = normal_demo_spec();
        let first = normal_normal_operating_characteristics(&spec).unwrap();
        let second = normal_normal_operating_characteristics(&spec).unwrap();

        assert_eq!(first, second);

        let null = first.scenarios.iter().find(|item| item.scenario_id == "null").unwrap();
        let alt = first.scenarios.iter().find(|item| item.scenario_id == "alt").unwrap();

        assert_relative_eq!(
            null.success_rate + null.futility_rate + null.no_decision_rate,
            1.0,
            epsilon = 1e-12
        );
        assert!(alt.success_rate > null.success_rate);
    }

    #[test]
    fn test_normal_normal_posterior_predictive_is_deterministic() {
        let spec = normal_demo_spec();
        let observed = NormalNormalObservedData {
            look_id: "interim".to_string(),
            control_sample_mean: 0.1,
            treatment_sample_mean: 0.3,
        };
        let first = normal_normal_posterior_predictive(&spec, &observed).unwrap();
        let second = normal_normal_posterior_predictive(&spec, &observed).unwrap();

        assert_eq!(first, second);
        assert_relative_eq!(
            first.eventual_success_probability
                + first.eventual_futility_probability
                + first.eventual_no_decision_probability,
            1.0,
            epsilon = 1e-12
        );
        assert_eq!(first.future_look_summaries.len(), 1);
        assert_relative_eq!(
            first.eventual_success_probability,
            first.future_look_summaries[0].conditional_success_probability,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_normal_normal_prior_sensitivity_orders_variants() {
        let spec = normal_demo_spec();
        let observed = NormalNormalObservedData {
            look_id: "interim".to_string(),
            control_sample_mean: 0.1,
            treatment_sample_mean: 0.3,
        };
        let report =
            normal_normal_prior_sensitivity(&spec, &observed, &normal_prior_campaign()).unwrap();
        let baseline = report.variants.iter().find(|item| item.variant_id == "baseline").unwrap();
        let skeptical = report.variants.iter().find(|item| item.variant_id == "skeptical").unwrap();
        let enthusiastic =
            report.variants.iter().find(|item| item.variant_id == "enthusiastic").unwrap();

        assert_relative_eq!(baseline.posterior_probability_delta_vs_baseline, 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            baseline.eventual_success_probability_delta_vs_baseline,
            0.0,
            epsilon = 1e-12
        );
        assert!(
            skeptical.posterior_probability_gt_margin < baseline.posterior_probability_gt_margin
        );
        assert!(skeptical.eventual_success_probability < baseline.eventual_success_probability);
        assert!(enthusiastic.eventual_success_probability > baseline.eventual_success_probability);
    }
}
