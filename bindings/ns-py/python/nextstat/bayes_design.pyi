from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

class BetaPosteriorSummary(TypedDict):
    alpha: float
    beta: float
    mean: float
    ci_lower: float
    ci_upper: float

class BetaBinomialEffectDifferenceSummary(TypedDict):
    margin: float
    posterior_mean: float
    posterior_probability_gt_margin: float

class BetaBinomialPosteriorSummary(TypedDict):
    control: BetaPosteriorSummary
    treatment: BetaPosteriorSummary
    effect_difference: BetaBinomialEffectDifferenceSummary

class BetaBinomialLookSummary(TypedDict):
    id: str
    n_control: int
    n_treatment: int

class BetaBinomialObservedData(TypedDict):
    look_id: str
    control_successes: int
    treatment_successes: int

class BetaBinomialDecisionSummary(TypedDict):
    success: bool
    futility: bool
    recommended_action: Literal["continue", "stop_for_success", "stop_for_futility"]
    posterior_probability_gt_margin: float
    success_threshold: float
    futility_threshold: float
    margin: float

class BetaBinomialAnalysisResult(TypedDict):
    schema_version: str
    stability: str
    design_id: str
    look: BetaBinomialLookSummary
    observed: BetaBinomialObservedData
    posterior: BetaBinomialPosteriorSummary
    decision: BetaBinomialDecisionSummary

class BetaBinomialLookOperatingCharacteristics(TypedDict):
    look_id: str
    stop_probability: float
    success_probability: float
    futility_probability: float

class BetaBinomialScenarioOperatingCharacteristics(TypedDict):
    scenario_id: str
    p_control: float
    p_treatment: float
    success_rate: float
    futility_rate: float
    no_decision_rate: float
    expected_total_sample_size: float
    look_summaries: list[BetaBinomialLookOperatingCharacteristics]

class BetaBinomialOperatingCharacteristicsResult(TypedDict):
    schema_version: str
    stability: str
    design_id: str
    n_replicates: int
    seed: int
    scenarios: list[BetaBinomialScenarioOperatingCharacteristics]

class BetaBinomialPosteriorPredictiveLookSummary(TypedDict):
    look_id: str
    conditional_stop_probability: float
    conditional_success_probability: float
    conditional_futility_probability: float

class BetaBinomialPosteriorPredictiveResult(TypedDict):
    schema_version: str
    stability: str
    design_id: str
    current_analysis: BetaBinomialAnalysisResult
    n_replicates: int
    seed: int
    current_total_sample_size: float
    expected_total_sample_size: float
    expected_remaining_sample_size: float
    eventual_success_probability: float
    eventual_futility_probability: float
    eventual_no_decision_probability: float
    future_look_summaries: list[BetaBinomialPosteriorPredictiveLookSummary]

class BetaPriorSensitivityVariantResult(TypedDict):
    variant_id: str
    is_baseline: bool
    control_prior: dict[str, float]
    treatment_prior: dict[str, float]
    posterior_mean: float
    posterior_probability_gt_margin: float
    recommended_action: Literal["continue", "stop_for_success", "stop_for_futility"]
    eventual_success_probability: float
    eventual_futility_probability: float
    eventual_no_decision_probability: float
    expected_total_sample_size: float
    expected_remaining_sample_size: float
    future_look_summaries: list[BetaBinomialPosteriorPredictiveLookSummary]
    posterior_probability_delta_vs_baseline: float
    eventual_success_probability_delta_vs_baseline: float
    expected_total_sample_size_delta_vs_baseline: float

class BetaPriorSensitivityReport(TypedDict):
    schema_version: str
    stability: str
    design_id: str
    look: BetaBinomialLookSummary
    observed: BetaBinomialObservedData
    n_replicates: int
    seed: int
    variants: list[BetaPriorSensitivityVariantResult]

class BayesianDesignReportProvenance(TypedDict):
    software_name: str
    software_version: str
    design_schema_version: str
    analysis_schema_version: str
    operating_characteristics_schema_version: str
    posterior_predictive_schema_version: str
    prior_sensitivity_campaign_schema_version: str
    prior_sensitivity_report_schema_version: str
    simulation_seed: int
    n_replicates: int

class BayesianDesignReportBundleArtifactPaths(TypedDict):
    run_bundle_meta: str
    run_bundle_manifest: str
    frozen_report_json: str
    design_report_markdown: str
    design_spec: str
    current_analysis: str
    operating_characteristics: str
    posterior_predictive: str
    prior_sensitivity: str
    provenance: str

class BayesianDesignReportBundleSummary(TypedDict):
    schema_version: str
    stability: str
    design_family: Literal["beta_binomial", "normal_normal"]
    report_schema_version: str
    deterministic: bool
    artifact_paths: BayesianDesignReportBundleArtifactPaths

class BayesianDesignRegulatoryAppendix(TypedDict):
    schema_version: str
    stability: str
    appendix_id: str
    design_family: Literal["beta_binomial", "normal_normal"]
    design_id: str
    source_report_schema_version: str
    generated_from_frozen_report: bool
    required_sections: list[str]
    section_order: list[str]
    sections: dict[str, Any]

class BayesianPriorConflictThresholds(TypedDict):
    high_recommended_action_flip_count_threshold: int
    moderate_decision_margin_ratio_threshold: float
    moderate_eventual_success_probability_range_threshold: float
    moderate_expected_total_sample_size_range_fraction_threshold: float

class BayesianPriorConflictMetrics(TypedDict):
    baseline_posterior_probability_gt_margin: float
    success_threshold: float
    futility_threshold: float
    nearest_decision_threshold_margin: float
    decision_margin_ratio: float | None
    posterior_probability_range: float
    eventual_success_probability_range: float
    expected_total_sample_size_range: float
    expected_total_sample_size_range_fraction_of_plan: float
    max_abs_posterior_probability_delta_vs_baseline: float
    max_abs_eventual_success_probability_delta_vs_baseline: float
    max_abs_expected_total_sample_size_delta_vs_baseline: float
    recommended_action_flip_count: int
    recommended_action_flip_variant_ids: list[str]

class BayesianPriorConflictVariantSummary(TypedDict):
    variant_id: str
    is_baseline: bool
    recommended_action: Literal["continue", "stop_for_success", "stop_for_futility"]
    posterior_probability_gt_margin: float
    eventual_success_probability: float
    expected_total_sample_size: float
    posterior_probability_delta_vs_baseline: float
    eventual_success_probability_delta_vs_baseline: float
    expected_total_sample_size_delta_vs_baseline: float

class BayesianPriorConflictDiagnostic(TypedDict):
    schema_version: str
    stability: str
    diagnostic_id: str
    design_family: Literal["beta_binomial", "normal_normal"]
    design_id: str
    source_report_schema_version: str
    source_prior_sensitivity_schema_version: str
    generated_from_frozen_report: bool
    baseline_variant_id: str
    baseline_recommended_action: Literal[
        "continue", "stop_for_success", "stop_for_futility"
    ]
    reported_variant_count: int
    conflict_severity: Literal["low", "moderate", "high"]
    decision_instability: bool
    thresholds: BayesianPriorConflictThresholds
    metrics: BayesianPriorConflictMetrics
    rationale: list[str]
    variant_summaries: list[BayesianPriorConflictVariantSummary]

class BayesianHistoricalControlBorrowingSource(TypedDict):
    source_id: str
    source_role: Literal["external_control_arm", "legacy_internal_control"]
    planned_control_sample_size: float
    exchangeability_assessment: Literal["low", "moderate", "high"]
    data_cut_label: str

class BayesianHistoricalControlBorrowingStrength(TypedDict):
    full_borrowing_fraction: float
    tapered_borrowing_fraction: float
    suspended_borrowing_fraction: float

class BayesianHistoricalControlBorrowingGating(TypedDict):
    meets_minimum_current_control_sample_size: bool
    meets_minimum_control_information_fraction: bool
    passes_action_flip_gate: bool
    within_full_eventual_success_probability_range: bool
    within_full_expected_total_sample_size_range_fraction: bool

class BayesianHistoricalControlBorrowingDiagnostics(TypedDict):
    prior_conflict_severity: Literal["low", "moderate", "high"]
    decision_instability: bool
    recommended_action_flip_count: int
    current_control_sample_size: float
    planned_control_sample_size: float
    current_control_information_fraction: float
    historical_source_count: int
    total_planned_historical_control_sample_size: float
    eventual_success_probability_range: float
    expected_total_sample_size_range_fraction_of_plan: float

class BayesianHistoricalControlBorrowingReview(TypedDict):
    schema_version: str
    stability: str
    review_id: str
    design_family: Literal["beta_binomial", "normal_normal"]
    design_id: str
    policy_id: str
    borrowing_model: Literal["power_prior", "commensurate"]
    source_report_schema_version: str
    source_policy_schema_version: str
    source_prior_conflict_schema_version: str
    generated_from_frozen_report: bool
    recommended_borrowing_state: Literal["retain", "taper", "suspend"]
    borrowing_eligible: bool
    current_effective_borrowing_fraction: float
    current_effective_historical_control_sample_size: float
    borrowing_strength: BayesianHistoricalControlBorrowingStrength
    gating: BayesianHistoricalControlBorrowingGating
    diagnostics: BayesianHistoricalControlBorrowingDiagnostics
    historical_sources: list[BayesianHistoricalControlBorrowingSource]
    rationale: list[str]

class BayesianHistoricalControlBorrowingOperatingCharacteristicsLookSummary(TypedDict):
    look_id: str
    review_probability: float
    retain_probability: float
    taper_probability: float
    suspend_probability: float
    borrowing_eligible_probability: float
    decision_instability_probability: float
    high_conflict_probability: float
    mean_effective_borrowing_fraction_when_reviewed: float
    mean_effective_historical_control_sample_size_when_reviewed: float

class BayesianHistoricalControlBorrowingOperatingCharacteristicsScenario(TypedDict):
    scenario_id: str
    p_control: NotRequired[float]
    p_treatment: NotRequired[float]
    mean_control: NotRequired[float]
    mean_treatment: NotRequired[float]
    retain_rate: float
    taper_rate: float
    suspend_rate: float
    borrowing_eligible_rate: float
    decision_instability_rate: float
    high_conflict_rate: float
    mean_terminal_effective_borrowing_fraction: float
    mean_terminal_effective_historical_control_sample_size: float
    look_summaries: list[
        BayesianHistoricalControlBorrowingOperatingCharacteristicsLookSummary
    ]

class BayesianHistoricalControlBorrowingOperatingCharacteristics(TypedDict):
    schema_version: str
    stability: str
    design_family: Literal["beta_binomial", "normal_normal"]
    design_id: str
    policy_id: str
    source_design_schema_version: str
    source_campaign_schema_version: str
    source_policy_schema_version: str
    derived_review_schema_version: str
    generated_from_seeded_simulation: bool
    n_replicates: int
    seed: int
    scenarios: list[BayesianHistoricalControlBorrowingOperatingCharacteristicsScenario]

class BayesianRobustMixturePriorComponentWeight(TypedDict):
    component_id: str
    component_role: Literal["informative", "weak_reference"]
    base_weight: float
    effective_weight: float
    prior: dict[str, float]

class BayesianRobustMixturePriorGating(TypedDict):
    meets_minimum_information_fraction: bool
    passes_action_flip_gate: bool
    within_retain_eventual_success_probability_range: bool
    within_retain_expected_total_sample_size_range_fraction: bool

class BayesianRobustMixturePriorDiagnostics(TypedDict):
    prior_conflict_severity: Literal["low", "moderate", "high"]
    decision_instability: bool
    recommended_action_flip_count: int
    current_target_sample_size: float
    planned_target_sample_size: float
    current_information_fraction: float
    posterior_probability_range: float
    eventual_success_probability_range: float
    expected_total_sample_size_range_fraction_of_plan: float

class BayesianRobustMixturePriorReview(TypedDict):
    schema_version: str
    stability: str
    review_id: str
    design_family: Literal["beta_binomial", "normal_normal"]
    design_id: str
    policy_id: str
    mixture_model: Literal["robust_mixture_beta", "robust_mixture_normal"]
    prior_target: Literal["control_prior", "treatment_prior"]
    source_report_schema_version: str
    source_policy_schema_version: str
    source_prior_conflict_schema_version: str
    generated_from_frozen_report: bool
    recommended_mixture_state: Literal["retain", "taper", "fallback_to_weak"]
    mixture_eligible: bool
    current_informative_weight: float
    effective_component_weights: list[BayesianRobustMixturePriorComponentWeight]
    gating: BayesianRobustMixturePriorGating
    diagnostics: BayesianRobustMixturePriorDiagnostics
    rationale: list[str]

class BayesianRobustMixturePriorOperatingCharacteristicsLookSummary(TypedDict):
    look_id: str
    review_probability: float
    retain_probability: float
    taper_probability: float
    fallback_to_weak_probability: float
    mixture_eligible_probability: float
    decision_instability_probability: float
    high_conflict_probability: float
    mean_informative_weight_when_reviewed: float

class BayesianRobustMixturePriorOperatingCharacteristicsScenario(TypedDict):
    scenario_id: str
    p_control: NotRequired[float]
    p_treatment: NotRequired[float]
    mean_control: NotRequired[float]
    mean_treatment: NotRequired[float]
    retain_rate: float
    taper_rate: float
    fallback_to_weak_rate: float
    mixture_eligible_rate: float
    decision_instability_rate: float
    high_conflict_rate: float
    mean_terminal_informative_weight: float
    look_summaries: list[
        BayesianRobustMixturePriorOperatingCharacteristicsLookSummary
    ]

class BayesianRobustMixturePriorOperatingCharacteristics(TypedDict):
    schema_version: str
    stability: str
    design_family: Literal["beta_binomial", "normal_normal"]
    design_id: str
    policy_id: str
    source_design_schema_version: str
    source_campaign_schema_version: str
    source_policy_schema_version: str
    derived_review_schema_version: str
    generated_from_seeded_simulation: bool
    n_replicates: int
    seed: int
    scenarios: list[BayesianRobustMixturePriorOperatingCharacteristicsScenario]

class BetaBinomialDesignReport(TypedDict):
    schema_version: str
    stability: str
    design_family: Literal["beta_binomial"]
    design_spec: dict[str, Any]
    current_analysis: BetaBinomialAnalysisResult
    operating_characteristics: BetaBinomialOperatingCharacteristicsResult
    posterior_predictive: BetaBinomialPosteriorPredictiveResult
    prior_sensitivity: BetaPriorSensitivityReport
    provenance: BayesianDesignReportProvenance

def analyze_beta_binomial_design(
    spec_or_path: dict[str, Any] | str | Path,
    observed_or_path: dict[str, Any] | str | Path,
) -> BetaBinomialAnalysisResult: ...
def simulate_beta_binomial_design(
    spec_or_path: dict[str, Any] | str | Path,
) -> BetaBinomialOperatingCharacteristicsResult: ...
def forecast_beta_binomial_design(
    spec_or_path: dict[str, Any] | str | Path,
    observed_or_path: dict[str, Any] | str | Path,
) -> BetaBinomialPosteriorPredictiveResult: ...
def analyze_beta_binomial_prior_sensitivity(
    spec_or_path: dict[str, Any] | str | Path,
    observed_or_path: dict[str, Any] | str | Path,
    campaign_or_path: dict[str, Any] | str | Path,
) -> BetaPriorSensitivityReport: ...
def build_beta_binomial_design_report(
    spec_or_path: dict[str, Any] | str | Path,
    observed_or_path: dict[str, Any] | str | Path,
    campaign_or_path: dict[str, Any] | str | Path,
) -> BetaBinomialDesignReport: ...
def render_beta_binomial_design_report(
    report_or_path: dict[str, Any] | str | Path,
) -> str: ...
def write_beta_binomial_design_report_bundle(
    bundle_dir: str | Path,
    report_or_path: dict[str, Any] | str | Path,
) -> BayesianDesignReportBundleSummary: ...
def build_beta_binomial_regulatory_appendix(
    report_or_path: dict[str, Any] | str | Path,
) -> BayesianDesignRegulatoryAppendix: ...
def build_beta_binomial_prior_conflict_diagnostic(
    report_or_path: dict[str, Any] | str | Path,
) -> BayesianPriorConflictDiagnostic: ...
def build_beta_binomial_historical_control_borrowing_review(
    report_or_path: dict[str, Any] | str | Path,
    policy_or_path: dict[str, Any] | str | Path,
) -> BayesianHistoricalControlBorrowingReview: ...
def simulate_beta_binomial_historical_control_borrowing_operating_characteristics(
    spec_or_path: dict[str, Any] | str | Path,
    campaign_or_path: dict[str, Any] | str | Path,
    policy_or_path: dict[str, Any] | str | Path,
) -> BayesianHistoricalControlBorrowingOperatingCharacteristics: ...
def build_beta_binomial_robust_mixture_prior_review(
    report_or_path: dict[str, Any] | str | Path,
    policy_or_path: dict[str, Any] | str | Path,
) -> BayesianRobustMixturePriorReview: ...
def simulate_beta_binomial_robust_mixture_prior_operating_characteristics(
    spec_or_path: dict[str, Any] | str | Path,
    campaign_or_path: dict[str, Any] | str | Path,
    policy_or_path: dict[str, Any] | str | Path,
) -> BayesianRobustMixturePriorOperatingCharacteristics: ...
def render_bayesian_regulatory_appendix_markdown(
    appendix_or_path: dict[str, Any] | str | Path,
) -> str: ...
def write_bayesian_regulatory_appendix_pdf(
    pdf_path: str | Path,
    appendix_or_path: dict[str, Any] | str | Path,
) -> None: ...

class NormalPosteriorSummary(TypedDict):
    posterior_mean: float
    posterior_sd: float
    ci_lower: float
    ci_upper: float

class NormalNormalEffectDifferenceSummary(TypedDict):
    margin: float
    posterior_mean: float
    posterior_sd: float
    ci_lower: float
    ci_upper: float
    posterior_probability_gt_margin: float

class NormalNormalPosteriorSummary(TypedDict):
    control: NormalPosteriorSummary
    treatment: NormalPosteriorSummary
    effect_difference: NormalNormalEffectDifferenceSummary

class NormalNormalLookSummary(TypedDict):
    id: str
    n_control: int
    n_treatment: int

class NormalNormalObservedData(TypedDict):
    look_id: str
    control_sample_mean: float
    treatment_sample_mean: float

class NormalNormalDecisionSummary(TypedDict):
    success: bool
    futility: bool
    recommended_action: Literal["continue", "stop_for_success", "stop_for_futility"]
    posterior_probability_gt_margin: float
    success_threshold: float
    futility_threshold: float
    margin: float

class NormalNormalAnalysisResult(TypedDict):
    schema_version: str
    stability: str
    design_id: str
    look: NormalNormalLookSummary
    observed: NormalNormalObservedData
    posterior: NormalNormalPosteriorSummary
    decision: NormalNormalDecisionSummary

class NormalNormalLookOperatingCharacteristics(TypedDict):
    look_id: str
    stop_probability: float
    success_probability: float
    futility_probability: float

class NormalNormalScenarioOperatingCharacteristics(TypedDict):
    scenario_id: str
    mean_control: float
    mean_treatment: float
    success_rate: float
    futility_rate: float
    no_decision_rate: float
    expected_total_sample_size: float
    look_summaries: list[NormalNormalLookOperatingCharacteristics]

class NormalNormalOperatingCharacteristicsResult(TypedDict):
    schema_version: str
    stability: str
    design_id: str
    n_replicates: int
    seed: int
    scenarios: list[NormalNormalScenarioOperatingCharacteristics]

class NormalNormalPosteriorPredictiveLookSummary(TypedDict):
    look_id: str
    conditional_stop_probability: float
    conditional_success_probability: float
    conditional_futility_probability: float

class NormalNormalPosteriorPredictiveResult(TypedDict):
    schema_version: str
    stability: str
    design_id: str
    current_analysis: NormalNormalAnalysisResult
    n_replicates: int
    seed: int
    current_total_sample_size: float
    expected_total_sample_size: float
    expected_remaining_sample_size: float
    eventual_success_probability: float
    eventual_futility_probability: float
    eventual_no_decision_probability: float
    future_look_summaries: list[NormalNormalPosteriorPredictiveLookSummary]

class NormalPriorSensitivityVariantResult(TypedDict):
    variant_id: str
    is_baseline: bool
    control_prior: dict[str, float]
    treatment_prior: dict[str, float]
    posterior_mean: float
    posterior_probability_gt_margin: float
    recommended_action: Literal["continue", "stop_for_success", "stop_for_futility"]
    eventual_success_probability: float
    eventual_futility_probability: float
    eventual_no_decision_probability: float
    expected_total_sample_size: float
    expected_remaining_sample_size: float
    future_look_summaries: list[NormalNormalPosteriorPredictiveLookSummary]
    posterior_probability_delta_vs_baseline: float
    eventual_success_probability_delta_vs_baseline: float
    expected_total_sample_size_delta_vs_baseline: float

class NormalPriorSensitivityReport(TypedDict):
    schema_version: str
    stability: str
    design_id: str
    look: NormalNormalLookSummary
    observed: NormalNormalObservedData
    n_replicates: int
    seed: int
    variants: list[NormalPriorSensitivityVariantResult]

class NormalNormalDesignReport(TypedDict):
    schema_version: str
    stability: str
    design_family: Literal["normal_normal"]
    design_spec: dict[str, Any]
    current_analysis: NormalNormalAnalysisResult
    operating_characteristics: NormalNormalOperatingCharacteristicsResult
    posterior_predictive: NormalNormalPosteriorPredictiveResult
    prior_sensitivity: NormalPriorSensitivityReport
    provenance: BayesianDesignReportProvenance

def analyze_normal_normal_design(
    spec_or_path: dict[str, Any] | str | Path,
    observed_or_path: dict[str, Any] | str | Path,
) -> NormalNormalAnalysisResult: ...
def simulate_normal_normal_design(
    spec_or_path: dict[str, Any] | str | Path,
) -> NormalNormalOperatingCharacteristicsResult: ...
def forecast_normal_normal_design(
    spec_or_path: dict[str, Any] | str | Path,
    observed_or_path: dict[str, Any] | str | Path,
) -> NormalNormalPosteriorPredictiveResult: ...
def analyze_normal_normal_prior_sensitivity(
    spec_or_path: dict[str, Any] | str | Path,
    observed_or_path: dict[str, Any] | str | Path,
    campaign_or_path: dict[str, Any] | str | Path,
) -> NormalPriorSensitivityReport: ...
def build_normal_normal_design_report(
    spec_or_path: dict[str, Any] | str | Path,
    observed_or_path: dict[str, Any] | str | Path,
    campaign_or_path: dict[str, Any] | str | Path,
) -> NormalNormalDesignReport: ...
def render_normal_normal_design_report(
    report_or_path: dict[str, Any] | str | Path,
) -> str: ...
def write_normal_normal_design_report_bundle(
    bundle_dir: str | Path,
    report_or_path: dict[str, Any] | str | Path,
) -> BayesianDesignReportBundleSummary: ...
def build_normal_normal_regulatory_appendix(
    report_or_path: dict[str, Any] | str | Path,
) -> BayesianDesignRegulatoryAppendix: ...
def build_normal_normal_prior_conflict_diagnostic(
    report_or_path: dict[str, Any] | str | Path,
) -> BayesianPriorConflictDiagnostic: ...
def build_normal_normal_historical_control_borrowing_review(
    report_or_path: dict[str, Any] | str | Path,
    policy_or_path: dict[str, Any] | str | Path,
) -> BayesianHistoricalControlBorrowingReview: ...
def simulate_normal_normal_historical_control_borrowing_operating_characteristics(
    spec_or_path: dict[str, Any] | str | Path,
    campaign_or_path: dict[str, Any] | str | Path,
    policy_or_path: dict[str, Any] | str | Path,
) -> BayesianHistoricalControlBorrowingOperatingCharacteristics: ...
def build_normal_normal_robust_mixture_prior_review(
    report_or_path: dict[str, Any] | str | Path,
    policy_or_path: dict[str, Any] | str | Path,
) -> BayesianRobustMixturePriorReview: ...
def simulate_normal_normal_robust_mixture_prior_operating_characteristics(
    spec_or_path: dict[str, Any] | str | Path,
    campaign_or_path: dict[str, Any] | str | Path,
    policy_or_path: dict[str, Any] | str | Path,
) -> BayesianRobustMixturePriorOperatingCharacteristics: ...
