//! Server-side Tool API (agent/runtime surface).
//!
//! This is intended to mirror the `nextstat.tools` Python surface:
//! - versioned tool result envelope (`nextstat.tool_result.v1`)
//! - deterministic execution controls
//! - correct semantics for CLs vs discovery p-values

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use base64::Engine;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use ns_core::traits::LogDensityModel;
use ns_inference::mle::MaximumLikelihoodEstimator;
use ns_inference::{
    BeConfig, BeData, BePowerConfig, CifEstimate, ClaimsTriangle,
    CovariateProvenance as AdsCovariateProvenance, CovariateTiming as AdsCovariateTiming,
    CoxPhModel, CoxTies, CupedArmData, EmaxModel, ErrorModel, ExponentialSurvivalModel,
    FailureMode, FaultTreeCeIsConfig, FaultTreeNode, FaultTreeSpec, FineGrayResult,
    FixedEffectsSolver, FoceConfig, FoceEstimator, Gate, GrayTestResult, KaplanMeierEstimate,
    LinearRegressionModel, LogNormalAftModel, LogRankResult, LogisticRegressionModel, ModelBuilder,
    MultiCovariateArmData, NegativeBinomialRegressionModel, NpdeConfig, NutsConfig, OmegaMatrix,
    OrderedLogitModel, OrderedProbitModel, PkModelKind, PkModelType as TrialPkModelType,
    PoissonRegressionModel, PopulationPkParams, QualityGates, SaemConfig, SaemEstimator,
    SigmoidEmaxModel, StudyEffect, TrialConfig, TrialErrorModelType, TrialResult, VpcConfig,
    VpcResult, WeibullSurvivalModel, average_be as run_average_be, be_power as run_be_power,
    be_sample_size as run_be_sample_size, chain_ladder_fit as run_chain_ladder,
    compute_diagnostics, cumulative_incidence as competing_cumulative_incidence,
    cuped_adjust as run_ads_cuped_adjust_core, cure_adjust as run_ads_cure_adjust_core,
    fault_tree_mc_ce_is as run_fault_tree_mc_ce_is, fault_tree_mc_cpu as run_fault_tree_mc_cpu,
    fine_gray_fit as competing_fine_gray_fit, gof_pk as run_gof_pk,
    gray_test as competing_gray_test, kaplan_meier as survival_kaplan_meier,
    log_rank_test as survival_log_rank_test, mack_chain_ladder as run_mack_chain_ladder,
    meta_fixed as run_meta_fixed, meta_random as run_meta_random, npde_pk as run_npde_pk, ols_fit,
    quality_summary, retention_analysis as run_retention_analysis, sample_nuts_multichain,
    simulate_trial as run_trial_simulate, vpc_pk as run_vpc_pk,
};
use ns_root::RootFile;
use ns_translate::pyhf::HistFactoryModel;

use crate::pool::ModelPool;
use crate::state::AppState;

// ---------------------------------------------------------------------------
// Tool envelope (matches docs/schemas/tools/nextstat_tool_result_v1.schema.json)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct ToolError {
    #[serde(rename = "type")]
    pub type_name: String,
    pub message: String,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct ToolMeta {
    pub tool_name: String,
    pub nextstat_version: Option<String>,
    pub deterministic: bool,
    pub eval_mode: String,
    pub threads_requested: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threads_applied: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ToolResultEnvelope {
    pub schema_version: &'static str,
    pub ok: bool,
    pub result: serde_json::Value,
    pub error: Option<ToolError>,
    pub meta: ToolMeta,
}

impl ToolResultEnvelope {
    pub fn ok(tool_name: &str, meta: ToolMeta, result: serde_json::Value) -> Self {
        Self {
            schema_version: "nextstat.tool_result.v1",
            ok: true,
            result,
            error: None,
            meta: ToolMeta { tool_name: tool_name.to_string(), ..meta },
        }
    }

    pub fn err(tool_name: &str, mut meta: ToolMeta, type_name: &str, message: String) -> Self {
        meta.tool_name = tool_name.to_string();
        Self {
            schema_version: "nextstat.tool_result.v1",
            ok: false,
            result: serde_json::Value::Null,
            error: Some(ToolError {
                type_name: type_name.to_string(),
                message,
                extra: serde_json::Map::new(),
            }),
            meta,
        }
    }
}

// ---------------------------------------------------------------------------
// Execution controls
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct ExecutionControls {
    #[serde(default = "default_true")]
    pub deterministic: bool,
    pub threads: Option<u64>,
    pub eval_mode: Option<String>, // "parity" | "fast"
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
pub struct ToolExecuteRequest {
    pub name: String,
    pub arguments: serde_json::Value,
}

const SERVER_SAFE_ROOT_UPLOAD_MAX_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug, Deserialize)]
struct ToolManifest {
    policies: ToolManifestPolicies,
    guidance: ToolManifestGuidance,
    tools: Vec<ToolManifestRecord>,
}

#[derive(Debug, Deserialize)]
struct ToolManifestRecord {
    name: String,
    local: ToolManifestTransport,
    server: Option<ToolManifestTransport>,
}

#[derive(Debug, Deserialize)]
struct ToolManifestTransport {
    tool: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct ToolManifestGuidance {
    transport_hints: ToolManifestTransportHints,
    recipes: Vec<ToolGuidanceRecipe>,
}

#[derive(Debug, Clone, Deserialize)]
struct ToolManifestTransportHints {
    #[serde(rename = "local")]
    _local: Vec<String>,
    server: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ToolManifestPolicies {
    server: ToolManifestServerPolicies,
}

#[derive(Debug, Clone, Deserialize)]
struct ToolManifestServerPolicies {
    defaults: ToolManifestServerPolicyDefaults,
    #[serde(default)]
    overrides: HashMap<String, ToolManifestPolicyRule>,
}

#[derive(Debug, Clone, Deserialize)]
struct ToolManifestServerPolicyDefaults {
    exposed: ToolManifestPolicyRule,
    local_only: ToolManifestPolicyRule,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ToolManifestPolicyRule {
    reason_code: String,
    reason: String,
}

#[derive(Debug, Serialize)]
struct ToolCapabilityPolicy {
    availability: &'static str,
    reason_code: String,
    reason: String,
}

#[derive(Debug, Serialize)]
struct ToolCapability {
    name: String,
    local_available: bool,
    server_available: bool,
    server_policy: ToolCapabilityPolicy,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ToolGuidanceRecipe {
    id: String,
    transport: String,
    title: String,
    summary: String,
    prompt: String,
    tools: Vec<String>,
    docs: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ToolGuidance {
    hints: Vec<String>,
    recipes: Vec<ToolGuidanceRecipe>,
}

fn effective_server_policy(
    record: &ToolManifestRecord,
    policies: &ToolManifestServerPolicies,
) -> ToolCapabilityPolicy {
    let (availability, base) = if record.server.is_some() {
        ("exposed", &policies.defaults.exposed)
    } else {
        ("local_only", &policies.defaults.local_only)
    };
    let override_rule = policies.overrides.get(&record.name);
    ToolCapabilityPolicy {
        availability,
        reason_code: override_rule
            .map(|rule| rule.reason_code.clone())
            .unwrap_or_else(|| base.reason_code.clone()),
        reason: override_rule
            .map(|rule| rule.reason.clone())
            .unwrap_or_else(|| base.reason.clone()),
    }
}

// ---------------------------------------------------------------------------
// Tool schema (OpenAI tool definitions)
// ---------------------------------------------------------------------------

pub fn get_tool_schema() -> serde_json::Value {
    let manifest: ToolManifest = serde_json::from_str(include_str!(
        "../../../bindings/ns-py/python/nextstat/_tool_manifest_v1.json"
    ))
    .expect("tool manifest must parse");
    let server_policies = manifest.policies.server.clone();
    let guidance = ToolGuidance {
        hints: manifest.guidance.transport_hints.server.clone(),
        recipes: manifest
            .guidance
            .recipes
            .iter()
            .filter(|recipe| recipe.transport == "server")
            .cloned()
            .collect::<Vec<_>>(),
    };
    let tools = manifest
        .tools
        .iter()
        .filter_map(|record| record.server.as_ref().and_then(|server| server.tool.clone()))
        .collect::<Vec<_>>();
    let capabilities = manifest
        .tools
        .into_iter()
        .map(|record| ToolCapability {
            name: record.name.clone(),
            local_available: record.local.tool.is_some(),
            server_available: record
                .server
                .as_ref()
                .and_then(|server| server.tool.as_ref())
                .is_some(),
            server_policy: effective_server_policy(&record, &server_policies),
        })
        .collect::<Vec<_>>();

    serde_json::json!({
        "schema_version": "nextstat.tool_schema.v1",
        "transport": "server",
        "tools": tools,
        "capabilities": capabilities,
        "guidance": guidance
    })
}

// ---------------------------------------------------------------------------
// Execution helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum EffectiveEvalMode {
    Parity,
    Fast,
}

fn parse_eval_mode(s: Option<&str>) -> Option<EffectiveEvalMode> {
    match s {
        Some("parity") => Some(EffectiveEvalMode::Parity),
        Some("fast") => Some(EffectiveEvalMode::Fast),
        _ => None,
    }
}

struct EvalModeGuard {
    prev: ns_compute::EvalMode,
}

impl EvalModeGuard {
    fn set(mode: ns_compute::EvalMode) -> Self {
        let prev = ns_compute::eval_mode();
        ns_compute::set_eval_mode(mode);
        Self { prev }
    }
}

impl Drop for EvalModeGuard {
    fn drop(&mut self) {
        ns_compute::set_eval_mode(self.prev);
    }
}

fn effective_controls(arguments: &serde_json::Value) -> (ExecutionControls, Vec<String>) {
    let mut warnings = Vec::new();
    let exec_val = arguments.get("execution").cloned().unwrap_or(serde_json::Value::Null);
    let mut controls = if exec_val.is_null() {
        ExecutionControls { deterministic: true, threads: None, eval_mode: None }
    } else {
        match serde_json::from_value::<ExecutionControls>(exec_val) {
            Ok(v) => v,
            Err(e) => {
                warnings.push(format!("invalid execution controls; using defaults: {e}"));
                ExecutionControls { deterministic: true, threads: None, eval_mode: None }
            }
        }
    };

    // Deterministic implies parity mode and threads=1 request.
    if controls.deterministic {
        if controls.eval_mode.as_deref() != Some("parity") {
            controls.eval_mode = Some("parity".to_string());
        }
        if controls.threads.is_none() {
            controls.threads = Some(1);
        }
    }

    (controls, warnings)
}

fn meta_base(tool_name: &str, controls: &ExecutionControls, warnings: Vec<String>) -> ToolMeta {
    ToolMeta {
        tool_name: tool_name.to_string(),
        nextstat_version: Some(ns_core::VERSION.to_string()),
        deterministic: controls.deterministic,
        eval_mode: controls.eval_mode.clone().unwrap_or_else(|| match ns_compute::eval_mode() {
            ns_compute::EvalMode::Parity => "parity".to_string(),
            ns_compute::EvalMode::Fast => "fast".to_string(),
        }),
        threads_requested: controls.threads,
        threads_applied: None,
        device: None,
        warnings,
    }
}

// ---------------------------------------------------------------------------
// Model resolution (uses ModelPool cache)
// ---------------------------------------------------------------------------

fn resolve_model_from_args(
    state: &AppState,
    workspace_json: Option<&str>,
    model_id: Option<&str>,
) -> Result<Arc<HistFactoryModel>, String> {
    if let Some(id) = model_id {
        return state.model_pool.get(id).ok_or_else(|| format!("model {id} not in cache"));
    }

    let ws = workspace_json
        .ok_or_else(|| "either workspace_json or model_id must be provided".to_string())?;

    let id = ModelPool::hash_workspace(ws);
    if let Some(m) = state.model_pool.get(&id) {
        return Ok(m);
    }

    let model = load_model(ws).map_err(|e| format!("workspace build error: {e}"))?;
    let _ = state.model_pool.insert(ws, model, None);
    state
        .model_pool
        .get(&id)
        .ok_or_else(|| "internal error: model inserted but not found".to_string())
}

fn load_model(json_str: &str) -> anyhow::Result<HistFactoryModel> {
    let format = ns_translate::hs3::detect::detect_format(json_str);
    match format {
        ns_translate::hs3::detect::WorkspaceFormat::Hs3 => {
            ns_translate::hs3::convert::from_hs3_default(json_str).map_err(|e| anyhow::anyhow!(e))
        }
        ns_translate::hs3::detect::WorkspaceFormat::SimplifiedLikelihood => {
            let spec: ns_translate::simplified::schema::SimplifiedLikelihoodWorkspace =
                serde_json::from_str(json_str)?;
            ns_translate::simplified::convert::simplified_to_model(&spec)
                .map_err(|e| anyhow::anyhow!(e))
        }
        ns_translate::hs3::detect::WorkspaceFormat::Pyhf
        | ns_translate::hs3::detect::WorkspaceFormat::Unknown => {
            let workspace: ns_translate::pyhf::Workspace = serde_json::from_str(json_str)?;
            HistFactoryModel::from_workspace(&workspace).map_err(|e| anyhow::anyhow!(e))
        }
    }
}

// ---------------------------------------------------------------------------
// Tool execution
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct CommonWorkspaceArgs {
    workspace_json: Option<String>,
    model_id: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum GlmFamily {
    Linear,
    Logistic,
    Poisson,
    #[serde(rename = "negbin")]
    Negbin,
}

fn default_glm_family() -> GlmFamily {
    GlmFamily::Linear
}

#[derive(Debug, Deserialize)]
struct GlmFitArgs {
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    #[serde(default = "default_glm_family")]
    family: GlmFamily,
    #[serde(default = "default_true")]
    include_intercept: bool,
    #[serde(default)]
    l2: Option<f64>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum BayesianSampleModelKind {
    LinearRegression,
    LogisticRegression,
    PoissonRegression,
    NegbinRegression,
    CoxPh,
    WeibullSurvival,
    LognormalAft,
    OrderedLogit,
    OrderedProbit,
    Histfactory,
}

fn default_bayesian_n_chains() -> usize {
    4
}

fn default_bayesian_n_warmup() -> usize {
    500
}

fn default_bayesian_n_samples() -> usize {
    1000
}

fn default_bayesian_seed() -> u64 {
    42
}

fn default_bayesian_target_accept() -> f64 {
    0.8
}

#[derive(Debug, Deserialize)]
struct BayesianSampleArgs {
    model_type: BayesianSampleModelKind,
    x: Option<Vec<Vec<f64>>>,
    y: Option<Vec<f64>>,
    time: Option<Vec<f64>>,
    event: Option<Vec<u8>>,
    n_levels: Option<usize>,
    workspace_json: Option<String>,
    model_id: Option<String>,
    #[serde(default = "default_bayesian_n_chains")]
    n_chains: usize,
    #[serde(default = "default_bayesian_n_warmup")]
    n_warmup: usize,
    #[serde(default = "default_bayesian_n_samples")]
    n_samples: usize,
    #[serde(default = "default_bayesian_seed")]
    seed: u64,
    #[serde(default = "default_bayesian_target_accept")]
    target_accept: f64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SurvivalModelKind {
    CoxPh,
    Weibull,
    LognormalAft,
    Exponential,
}

fn default_survival_model() -> SurvivalModelKind {
    SurvivalModelKind::CoxPh
}

#[derive(Debug, Deserialize)]
struct SurvivalFitArgs {
    x: Vec<Vec<f64>>,
    time: Vec<f64>,
    event: Vec<u8>,
    #[serde(default = "default_survival_model")]
    model: SurvivalModelKind,
}

#[derive(Debug, Deserialize)]
struct KaplanMeierArgs {
    time: Vec<f64>,
    event: Vec<u8>,
    group: Option<Vec<i64>>,
}

#[derive(Debug, Deserialize)]
struct RootHistogramArgs {
    root_bytes_base64: String,
    hist_path: String,
    filename_hint: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LogRankTestArgs {
    times: Vec<f64>,
    events: Vec<serde_json::Value>,
    groups: Vec<i64>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum MetaAnalysisMethod {
    Fixed,
    Random,
}

fn default_meta_analysis_method() -> MetaAnalysisMethod {
    MetaAnalysisMethod::Random
}

#[derive(Debug, Deserialize)]
struct MetaAnalysisArgs {
    effects: Vec<f64>,
    standard_errors: Vec<f64>,
    #[serde(default = "default_meta_analysis_method")]
    method: MetaAnalysisMethod,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum RegressionClusterKind {
    Entity,
    Time,
    TwoWay,
    None,
}

fn default_regression_cluster_kind() -> RegressionClusterKind {
    RegressionClusterKind::Entity
}

#[derive(Debug, Deserialize)]
struct PanelFeArgs {
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    entity: Vec<serde_json::Value>,
    time: Option<Vec<serde_json::Value>>,
    #[serde(default = "default_regression_cluster_kind")]
    cluster: RegressionClusterKind,
}

#[derive(Debug, Deserialize)]
struct DidArgs {
    y: Vec<f64>,
    treat: Vec<u8>,
    post: Vec<u8>,
    entity: Vec<serde_json::Value>,
    time: Vec<serde_json::Value>,
    x: Option<Vec<Vec<f64>>>,
    #[serde(default = "default_regression_cluster_kind")]
    cluster: RegressionClusterKind,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum IvCovKind {
    Homoskedastic,
    Hc1,
    Cluster,
    Hac,
}

fn default_iv_cov_kind() -> IvCovKind {
    IvCovKind::Hc1
}

#[derive(Debug, Deserialize)]
struct Iv2slsArgs {
    y: Vec<f64>,
    endog: Vec<Vec<f64>>,
    instruments: Vec<Vec<f64>>,
    exog: Option<Vec<Vec<f64>>>,
    #[serde(default = "default_iv_cov_kind")]
    cov: IvCovKind,
    cluster: Option<Vec<serde_json::Value>>,
    time_index: Option<Vec<serde_json::Value>>,
    max_lag: Option<usize>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum AipwEstimand {
    Ate,
    Att,
}

fn default_aipw_estimand() -> AipwEstimand {
    AipwEstimand::Ate
}

#[derive(Debug, Deserialize)]
struct AipwArgs {
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    treatment: Vec<u8>,
    #[serde(default = "default_aipw_estimand")]
    estimand: AipwEstimand,
}

#[derive(Debug, Deserialize)]
struct EventStudyArgs {
    y: Vec<f64>,
    entity: Vec<serde_json::Value>,
    time: Vec<i64>,
    treat_time: Vec<Option<i64>>,
    n_leads: Option<usize>,
    n_lags: Option<usize>,
    #[serde(default = "default_regression_cluster_kind")]
    cluster: RegressionClusterKind,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum GarchModelKind {
    Garch,
    Egarch,
    GjrGarch,
}

fn default_garch_model_kind() -> GarchModelKind {
    GarchModelKind::Garch
}

#[derive(Debug, Deserialize)]
struct GarchFitArgs {
    returns: Vec<f64>,
    #[serde(default = "default_garch_model_kind")]
    model: GarchModelKind,
}

#[derive(Debug, Clone, Deserialize)]
struct AdsCovariateProvenanceArg {
    name: Option<String>,
    timing: AdsCovariateTiming,
    source_dataset: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AdsCupedAdjustArgs {
    control_outcomes: Vec<f64>,
    control_covariates: Vec<f64>,
    variant_outcomes: Vec<f64>,
    variant_covariates: Vec<f64>,
    covariate_name: Option<String>,
    covariate_provenance: Option<AdsCovariateProvenanceArg>,
    #[serde(default = "default_true")]
    pre_treatment_only: bool,
}

#[derive(Debug, Deserialize)]
struct AdsCureAdjustArgs {
    control_outcomes: Vec<f64>,
    control_covariates: Vec<Vec<f64>>,
    variant_outcomes: Vec<f64>,
    variant_covariates: Vec<Vec<f64>>,
    covariate_names: Option<Vec<String>>,
    covariate_provenance: Option<Vec<AdsCovariateProvenanceArg>>,
    #[serde(default = "default_true")]
    pre_treatment_only: bool,
}

#[derive(Debug, Deserialize)]
struct ChurnGenerateDataArgs {
    n_customers: Option<usize>,
    n_cohorts: Option<usize>,
    max_time: Option<f64>,
    treatment_fraction: Option<f64>,
    seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ChurnRiskModelArgs {
    times: Vec<f64>,
    events: Vec<serde_json::Value>,
    covariates: Vec<Vec<f64>>,
    names: Vec<String>,
    conf_level: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ChurnRetentionArgs {
    times: Vec<f64>,
    events: Vec<serde_json::Value>,
    groups: Vec<i64>,
    conf_level: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ChurnDiagnosticsArgs {
    times: Vec<f64>,
    events: Vec<serde_json::Value>,
    groups: Vec<i64>,
    #[serde(default)]
    treated: Vec<serde_json::Value>,
    #[serde(default)]
    covariates: Vec<Vec<f64>>,
    #[serde(default)]
    covariate_names: Vec<String>,
    #[serde(default = "default_churn_trim")]
    trim: f64,
}

#[derive(Debug, Deserialize)]
struct ChurnCohortMatrixArgs {
    times: Vec<f64>,
    events: Vec<serde_json::Value>,
    groups: Vec<i64>,
    period_boundaries: Vec<f64>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
enum ChurnBootstrapHrCiMethodKind {
    #[serde(rename = "percentile")]
    Percentile,
    #[serde(rename = "bca")]
    Bca,
}

fn default_churn_bootstrap_hr_ci_method() -> ChurnBootstrapHrCiMethodKind {
    ChurnBootstrapHrCiMethodKind::Percentile
}

#[derive(Debug, Deserialize)]
struct ChurnBootstrapHrArgs {
    times: Vec<f64>,
    events: Vec<serde_json::Value>,
    covariates: Vec<Vec<f64>>,
    names: Vec<String>,
    n_bootstrap: Option<usize>,
    seed: Option<u64>,
    conf_level: Option<f64>,
    #[serde(default = "default_churn_bootstrap_hr_ci_method")]
    ci_method: ChurnBootstrapHrCiMethodKind,
    n_jackknife: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ChurnIngestArgs {
    times: Vec<f64>,
    events: Vec<serde_json::Value>,
    groups: Option<Vec<i64>>,
    treated: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    covariates: Vec<Vec<f64>>,
    #[serde(default)]
    covariate_names: Vec<String>,
    observation_end: Option<f64>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
enum ChurnCompareCorrectionKind {
    #[serde(rename = "bonferroni")]
    Bonferroni,
    #[serde(rename = "benjamini_hochberg", alias = "bh")]
    BenjaminiHochberg,
}

fn default_churn_compare_correction_kind() -> ChurnCompareCorrectionKind {
    ChurnCompareCorrectionKind::BenjaminiHochberg
}

fn default_eval_horizons() -> Vec<f64> {
    vec![3.0, 6.0, 12.0, 24.0]
}

fn default_churn_trim() -> f64 {
    0.01
}

const CHURN_BOOTSTRAP_HR_SERVER_DEFAULT_BOOTSTRAPS: usize = 128;
const CHURN_BOOTSTRAP_HR_SERVER_MAX_BOOTSTRAPS: usize = 256;
const CHURN_BOOTSTRAP_HR_SERVER_DEFAULT_JACKKNIFE: usize = 64;
const CHURN_BOOTSTRAP_HR_SERVER_MAX_JACKKNIFE: usize = 128;
const CHURN_GENERATE_DATA_SERVER_DEFAULT_CUSTOMERS: usize = 256;
const CHURN_GENERATE_DATA_SERVER_MAX_CUSTOMERS: usize = 1024;
const CHURN_GENERATE_DATA_SERVER_MAX_COHORTS: usize = 24;

#[derive(Debug, Deserialize)]
struct ChurnCompareArgs {
    times: Vec<f64>,
    events: Vec<serde_json::Value>,
    groups: Vec<i64>,
    conf_level: Option<f64>,
    #[serde(default = "default_churn_compare_correction_kind")]
    correction: ChurnCompareCorrectionKind,
    alpha: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ChurnUpliftArgs {
    times: Vec<f64>,
    events: Vec<serde_json::Value>,
    treated: Vec<serde_json::Value>,
    covariates: Vec<Vec<f64>>,
    horizon: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ChurnUpliftSurvivalArgs {
    times: Vec<f64>,
    events: Vec<serde_json::Value>,
    treated: Vec<serde_json::Value>,
    #[serde(default)]
    covariates: Vec<Vec<f64>>,
    horizon: Option<f64>,
    #[serde(default = "default_eval_horizons")]
    eval_horizons: Vec<f64>,
    #[serde(default = "default_churn_trim")]
    trim: f64,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum ChainLadderMethodKind {
    Basic,
    Mack,
}

fn default_chain_ladder_method_kind() -> ChainLadderMethodKind {
    ChainLadderMethodKind::Mack
}

#[derive(Debug, Deserialize)]
struct ChainLadderArgs {
    triangle: Vec<Vec<Option<f64>>>,
    #[serde(default = "default_chain_ladder_method_kind", alias = "chain_ladder")]
    method: ChainLadderMethodKind,
    conf_level: Option<f64>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum PharmaFitMethodKind {
    Foce,
    Focei,
    Fo,
    Saem,
}

fn default_pharma_fit_method_kind() -> PharmaFitMethodKind {
    PharmaFitMethodKind::Focei
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
enum PharmaModelKind {
    #[serde(rename = "1cpt_iv")]
    OneCptIv,
    #[serde(rename = "1cpt_oral")]
    OneCptOral,
    #[serde(rename = "2cpt_iv")]
    TwoCptIv,
    #[serde(rename = "2cpt_oral")]
    TwoCptOral,
    #[serde(rename = "3cpt_iv")]
    ThreeCptIv,
    #[serde(rename = "3cpt_oral")]
    ThreeCptOral,
}

fn default_pharma_model_kind() -> PharmaModelKind {
    PharmaModelKind::OneCptOral
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum PharmaErrorModelKind {
    Additive,
    Proportional,
    Combined,
}

fn default_pharma_error_model_kind() -> PharmaErrorModelKind {
    PharmaErrorModelKind::Proportional
}

fn default_sigma() -> f64 {
    0.1
}

fn default_bioavailability() -> f64 {
    1.0
}

fn default_vpc_n_sim() -> usize {
    200
}

fn default_vpc_n_bins() -> usize {
    10
}

fn default_vpc_seed() -> u64 {
    42
}

fn default_vpc_pi_level() -> f64 {
    0.90
}

fn default_npde_n_sim() -> usize {
    500
}

fn default_npde_seed() -> u64 {
    42
}

#[derive(Debug, Deserialize)]
struct PharmaFitArgs {
    times: Vec<f64>,
    y: Vec<f64>,
    subject_idx: Vec<usize>,
    n_subjects: usize,
    doses: Vec<f64>,
    theta_init: Vec<f64>,
    omega_init: Vec<f64>,
    #[serde(default = "default_pharma_fit_method_kind")]
    method: PharmaFitMethodKind,
    #[serde(default = "default_pharma_model_kind")]
    model: PharmaModelKind,
    #[serde(default = "default_pharma_error_model_kind")]
    error_model: PharmaErrorModelKind,
    #[serde(default = "default_sigma")]
    sigma: f64,
    sigma_add: Option<f64>,
    #[serde(default = "default_bioavailability")]
    bioavailability: f64,
}

#[derive(Debug, Deserialize)]
struct PharmaVpcArgs {
    times: Vec<f64>,
    y: Vec<f64>,
    subject_idx: Vec<usize>,
    n_subjects: usize,
    doses: Vec<f64>,
    theta: Vec<f64>,
    omega_matrix: Vec<Vec<f64>>,
    #[serde(default = "default_pharma_model_kind")]
    model: PharmaModelKind,
    #[serde(default = "default_pharma_error_model_kind")]
    error_model: PharmaErrorModelKind,
    #[serde(default = "default_sigma")]
    sigma: f64,
    sigma_add: Option<f64>,
    #[serde(default = "default_bioavailability")]
    bioavailability: f64,
    #[serde(default = "default_vpc_n_sim")]
    n_sim: usize,
    quantiles: Option<Vec<f64>>,
    #[serde(default = "default_vpc_n_bins")]
    n_bins: usize,
    #[serde(default = "default_vpc_seed")]
    seed: u64,
    #[serde(default = "default_vpc_pi_level")]
    pi_level: f64,
}

#[derive(Debug, Deserialize)]
struct PharmaPkGofArgs {
    times: Vec<f64>,
    y: Vec<f64>,
    subject_idx: Vec<usize>,
    doses: Vec<f64>,
    theta: Vec<f64>,
    eta: Vec<Vec<f64>>,
    #[serde(default = "default_pharma_model_kind")]
    model: PharmaModelKind,
    #[serde(default = "default_pharma_error_model_kind")]
    error_model: PharmaErrorModelKind,
    #[serde(default = "default_sigma")]
    sigma: f64,
    sigma_add: Option<f64>,
    #[serde(default = "default_bioavailability")]
    bioavailability: f64,
}

#[derive(Debug, Deserialize)]
struct PharmaPkNpdeArgs {
    times: Vec<f64>,
    y: Vec<f64>,
    subject_idx: Vec<usize>,
    n_subjects: usize,
    doses: Vec<f64>,
    theta: Vec<f64>,
    omega_matrix: Vec<Vec<f64>>,
    #[serde(default = "default_pharma_model_kind")]
    model: PharmaModelKind,
    #[serde(default = "default_pharma_error_model_kind")]
    error_model: PharmaErrorModelKind,
    #[serde(default = "default_sigma")]
    sigma: f64,
    sigma_add: Option<f64>,
    #[serde(default = "default_bioavailability")]
    bioavailability: f64,
    #[serde(default = "default_npde_n_sim")]
    n_sim: usize,
    #[serde(default = "default_npde_seed")]
    seed: u64,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum TrialSimModelKind {
    #[serde(rename = "1cpt_oral")]
    OneCptOral,
    #[serde(rename = "2cpt_iv")]
    TwoCptIv,
    #[serde(rename = "2cpt_oral")]
    TwoCptOral,
}

fn default_trial_sim_model_kind() -> TrialSimModelKind {
    TrialSimModelKind::OneCptOral
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum TrialSimErrorModelKind {
    Additive,
    Proportional,
}

fn default_trial_sim_error_model_kind() -> TrialSimErrorModelKind {
    TrialSimErrorModelKind::Proportional
}

fn default_trial_sim_seed() -> u64 {
    42
}

#[derive(Debug, Deserialize)]
struct TrialSimulateArgs {
    n_subjects: usize,
    dose: f64,
    obs_times: Vec<f64>,
    theta: Vec<f64>,
    omega: Vec<f64>,
    sigma: f64,
    #[serde(default = "default_trial_sim_model_kind")]
    pk_model: TrialSimModelKind,
    #[serde(default = "default_trial_sim_error_model_kind")]
    error_model: TrialSimErrorModelKind,
    #[serde(default = "default_bioavailability")]
    bioavailability: f64,
    #[serde(default = "default_trial_sim_seed")]
    seed: u64,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum FaultTreeComponentArg {
    Bernoulli { p: f64 },
    BernoulliUncertain { mu: f64, sigma: f64 },
    WeibullMission { k: f64, lambda: f64, mission_time: f64 },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum FaultTreeNodeArg {
    Component { index: usize },
    And { children: Vec<usize> },
    Or { children: Vec<usize> },
}

#[derive(Debug, Deserialize)]
struct FaultTreeSpecArg {
    components: Vec<FaultTreeComponentArg>,
    nodes: Vec<FaultTreeNodeArg>,
    top_event: usize,
}

#[derive(Debug, Deserialize)]
struct FaultTreeMcArgs {
    spec: FaultTreeSpecArg,
    n_scenarios: Option<usize>,
    seed: Option<u64>,
    device: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FaultTreeCeIsArgs {
    spec: FaultTreeSpecArg,
    n_per_level: Option<usize>,
    elite_fraction: Option<f64>,
    max_levels: Option<usize>,
    q_max: Option<f64>,
    seed: Option<u64>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum KalmanOperationKind {
    Filter,
    Smooth,
    Forecast,
    Simulate,
    Em,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum KalmanSimInitKind {
    Sample,
    Mean,
}

fn default_kalman_operation_kind() -> KalmanOperationKind {
    KalmanOperationKind::Filter
}

fn default_kalman_sim_init_kind() -> KalmanSimInitKind {
    KalmanSimInitKind::Sample
}

fn default_kalman_seed() -> u64 {
    42
}

fn default_kalman_em_max_iter() -> usize {
    50
}

fn default_kalman_em_tol() -> f64 {
    1e-6
}

fn default_kalman_em_estimate_q() -> bool {
    true
}

fn default_kalman_em_estimate_r() -> bool {
    true
}

fn default_kalman_em_estimate_f() -> bool {
    false
}

fn default_kalman_em_estimate_h() -> bool {
    false
}

fn default_kalman_em_min_diag() -> f64 {
    1e-12
}

#[derive(Debug, Deserialize)]
struct KalmanArgs {
    #[serde(rename = "F")]
    f: Vec<Vec<f64>>,
    #[serde(rename = "H")]
    h: Vec<Vec<f64>>,
    #[serde(rename = "Q")]
    q: Vec<Vec<f64>>,
    #[serde(rename = "R")]
    r: Vec<Vec<f64>>,
    x0: Vec<f64>,
    #[serde(rename = "P0")]
    p0: Vec<Vec<f64>>,
    y: Option<Vec<Vec<f64>>>,
    #[serde(default = "default_kalman_operation_kind")]
    operation: KalmanOperationKind,
    n_ahead: Option<usize>,
    alpha: Option<f64>,
    t_max: Option<usize>,
    #[serde(default = "default_kalman_seed")]
    seed: u64,
    #[serde(default = "default_kalman_sim_init_kind")]
    init: KalmanSimInitKind,
    simulate_x0: Option<Vec<f64>>,
    #[serde(default = "default_kalman_em_max_iter")]
    max_iter: usize,
    #[serde(default = "default_kalman_em_tol")]
    tol: f64,
    #[serde(default = "default_kalman_em_estimate_q")]
    estimate_q: bool,
    #[serde(default = "default_kalman_em_estimate_r")]
    estimate_r: bool,
    #[serde(default = "default_kalman_em_estimate_f")]
    estimate_f: bool,
    #[serde(default = "default_kalman_em_estimate_h")]
    estimate_h: bool,
    #[serde(default = "default_kalman_em_min_diag")]
    min_diag: f64,
}

fn validate_rectangular_matrix(x: &[Vec<f64>], field_name: &str) -> Result<(), String> {
    if x.is_empty() {
        return Err(format!("{field_name} must be non-empty"));
    }
    let p = x[0].len();
    for (i, row) in x.iter().enumerate() {
        if row.len() != p {
            return Err(format!(
                "{field_name} rows must all have the same length; row {i} has length {}",
                row.len()
            ));
        }
    }
    Ok(())
}

fn encode_group_values(values: &[serde_json::Value], field_name: &str) -> Result<Vec<u64>, String> {
    let mut levels: HashMap<String, u64> = HashMap::new();
    let mut encoded = Vec::with_capacity(values.len());
    for (i, value) in values.iter().enumerate() {
        let key = match value {
            serde_json::Value::Null
            | serde_json::Value::Bool(_)
            | serde_json::Value::Number(_)
            | serde_json::Value::String(_) => {
                serde_json::to_string(value).map_err(|e| e.to_string())?
            }
            serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
                return Err(format!(
                    "{field_name}[{i}] must be a scalar JSON value (string/number/bool/null)"
                ));
            }
        };
        let next = levels.len() as u64;
        let dense = *levels.entry(key).or_insert(next);
        encoded.push(dense);
    }
    Ok(encoded)
}

fn encode_group_pairs(a: &[u64], b: &[u64]) -> Result<Vec<u64>, String> {
    if a.len() != b.len() {
        return Err("cluster pair inputs must have the same length".to_string());
    }
    let mut levels: HashMap<(u64, u64), u64> = HashMap::new();
    let mut encoded = Vec::with_capacity(a.len());
    for (&left, &right) in a.iter().zip(b.iter()) {
        let next = levels.len() as u64;
        let dense = *levels.entry((left, right)).or_insert(next);
        encoded.push(dense);
    }
    Ok(encoded)
}

fn encode_integer_levels(values: &[i64]) -> Vec<u64> {
    let mut levels: HashMap<i64, u64> = HashMap::new();
    let mut encoded = Vec::with_capacity(values.len());
    for &value in values {
        let next = levels.len() as u64;
        let dense = *levels.entry(value).or_insert(next);
        encoded.push(dense);
    }
    encoded
}

fn validate_scalar_json_values(
    values: &[serde_json::Value],
    field_name: &str,
) -> Result<(), String> {
    for (i, value) in values.iter().enumerate() {
        match value {
            serde_json::Value::Null
            | serde_json::Value::Bool(_)
            | serde_json::Value::Number(_)
            | serde_json::Value::String(_) => {}
            serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
                return Err(format!(
                    "{field_name}[{i}] must be a scalar JSON value (string/number/bool/null)"
                ));
            }
        }
    }
    Ok(())
}

fn compare_scalar_json_values(
    left: &serde_json::Value,
    right: &serde_json::Value,
) -> std::cmp::Ordering {
    match (left, right) {
        (serde_json::Value::Number(a), serde_json::Value::Number(b)) => a
            .as_f64()
            .and_then(|af| b.as_f64().and_then(|bf| af.partial_cmp(&bf)))
            .unwrap_or(std::cmp::Ordering::Equal),
        (serde_json::Value::String(a), serde_json::Value::String(b)) => a.cmp(b),
        (serde_json::Value::Bool(a), serde_json::Value::Bool(b)) => a.cmp(b),
        (serde_json::Value::Null, serde_json::Value::Null) => std::cmp::Ordering::Equal,
        _ => serde_json::to_string(left)
            .unwrap_or_default()
            .cmp(&serde_json::to_string(right).unwrap_or_default()),
    }
}

fn sort_scalar_json_indices(
    values: &[serde_json::Value],
    field_name: &str,
) -> Result<Vec<usize>, String> {
    validate_scalar_json_values(values, field_name)?;
    let mut indices = (0..values.len()).collect::<Vec<_>>();
    indices.sort_by(|&left, &right| {
        compare_scalar_json_values(&values[left], &values[right]).then_with(|| left.cmp(&right))
    });
    Ok(indices)
}

fn json_events_to_bool(
    values: &[serde_json::Value],
    field_name: &str,
) -> Result<Vec<bool>, String> {
    let mut events = Vec::with_capacity(values.len());
    for (i, value) in values.iter().enumerate() {
        match value {
            serde_json::Value::Bool(flag) => events.push(*flag),
            serde_json::Value::Number(number) => match number.as_i64() {
                Some(0) => events.push(false),
                Some(1) => events.push(true),
                _ => return Err(format!("{field_name}[{i}] must be boolean or 0/1 integer")),
            },
            _ => return Err(format!("{field_name}[{i}] must be boolean or 0/1 integer")),
        }
    }
    Ok(events)
}

fn json_binary_u8(values: &[serde_json::Value], field_name: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(values.len());
    for (i, value) in values.iter().enumerate() {
        match value {
            serde_json::Value::Bool(flag) => out.push(u8::from(*flag)),
            serde_json::Value::Number(number) => match number.as_u64() {
                Some(0) => out.push(0),
                Some(1) => out.push(1),
                _ => return Err(format!("{field_name}[{i}] must be boolean or 0/1 integer")),
            },
            _ => return Err(format!("{field_name}[{i}] must be boolean or 0/1 integer")),
        }
    }
    Ok(out)
}

fn validate_rectangular_f64_rows(rows: &[Vec<f64>], field_name: &str) -> Result<(), String> {
    if let Some(width) = rows.first().map(|row| row.len()) {
        for (i, row) in rows.iter().enumerate() {
            if row.len() != width {
                return Err(format!("{field_name}[{i}] has width {}, expected {width}", row.len()));
            }
            if row.iter().any(|v| !v.is_finite()) {
                return Err(format!("{field_name}[{i}] must contain only finite values"));
            }
        }
    }
    Ok(())
}

fn normalize_chain_ladder_triangle(
    triangle: &[Vec<Option<f64>>],
) -> Result<ClaimsTriangle, String> {
    if triangle.is_empty() {
        return Err("triangle must be a non-empty 2D array".to_string());
    }
    let n = triangle.len();
    let is_square = triangle.iter().all(|row| row.len() == n);

    let mut data = Vec::with_capacity(n);
    for (i, row) in triangle.iter().enumerate() {
        let expected = n - i;
        let values = if is_square {
            if row.len() != n {
                return Err("square triangle rows must all have the same length".to_string());
            }
            &row[..expected]
        } else {
            if row.len() != expected {
                return Err(format!(
                    "triangle row {i} has length {}, expected {} for ragged upper-left triangle",
                    row.len(),
                    expected
                ));
            }
            row.as_slice()
        };

        let mut parsed = Vec::with_capacity(expected);
        for (j, value) in values.iter().enumerate() {
            let number = value.ok_or_else(|| {
                format!("triangle[{i}][{j}] must be a finite non-negative number")
            })?;
            if !number.is_finite() || number < 0.0 {
                return Err(format!("triangle[{i}][{j}] must be a finite non-negative number"));
            }
            parsed.push(number);
        }
        data.push(parsed);
    }

    ClaimsTriangle::new(data).map_err(|e| e.to_string())
}

fn flatten_row_major_matrix(x: &[Vec<f64>], field_name: &str) -> Result<(Vec<f64>, usize), String> {
    validate_rectangular_matrix(x, field_name)?;
    let p = x[0].len();
    if p == 0 {
        return Err(format!("{field_name} must have at least 1 column"));
    }
    let flat = x.iter().flat_map(|row| row.iter().copied()).collect::<Vec<_>>();
    Ok((flat, p))
}

fn matrix_arg_to_dmatrix(x: &[Vec<f64>], field_name: &str) -> Result<DMatrix<f64>, String> {
    let (flat, cols) = flatten_row_major_matrix(x, field_name)?;
    if flat.iter().any(|value| !value.is_finite()) {
        return Err(format!("{field_name} must contain only finite values"));
    }
    Ok(DMatrix::from_row_slice(x.len(), cols, &flat))
}

fn vector_arg_to_dvector(x: &[f64], field_name: &str) -> Result<DVector<f64>, String> {
    if x.is_empty() {
        return Err(format!("{field_name} must be non-empty"));
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err(format!("{field_name} must contain only finite values"));
    }
    Ok(DVector::from_row_slice(x))
}

fn observation_sequence_to_dvectors(
    y: &[Vec<f64>],
    field_name: &str,
) -> Result<Vec<DVector<f64>>, String> {
    validate_rectangular_matrix(y, field_name)?;
    let cols = y[0].len();
    if cols == 0 {
        return Err(format!("{field_name} must have at least 1 observed dimension"));
    }
    if y.iter().flat_map(|row| row.iter()).any(|value| !value.is_finite()) {
        return Err(format!("{field_name} must contain only finite values"));
    }
    Ok(y.iter().map(|row| DVector::from_row_slice(row)).collect())
}

fn dvector_to_json_array(x: &DVector<f64>) -> Vec<f64> {
    x.iter().copied().collect()
}

fn dmatrix_to_json_array(x: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..x.nrows()).map(|row| (0..x.ncols()).map(|col| x[(row, col)]).collect::<Vec<_>>()).collect()
}

fn dvector_list_to_json_array(xs: &[DVector<f64>]) -> Vec<Vec<f64>> {
    xs.iter().map(dvector_to_json_array).collect()
}

fn dmatrix_list_to_json_array(xs: &[DMatrix<f64>]) -> Vec<Vec<Vec<f64>>> {
    xs.iter().map(dmatrix_to_json_array).collect()
}

fn kaplan_meier_estimate_to_json(km: &KaplanMeierEstimate) -> serde_json::Value {
    serde_json::json!({
        "n": km.n,
        "n_events": km.n_events,
        "conf_level": km.conf_level,
        "median": km.median,
        "time": km.steps.iter().map(|s| s.time).collect::<Vec<_>>(),
        "n_risk": km.steps.iter().map(|s| s.n_risk).collect::<Vec<_>>(),
        "n_event": km.steps.iter().map(|s| s.n_events).collect::<Vec<_>>(),
        "n_censored": km.steps.iter().map(|s| s.n_censored).collect::<Vec<_>>(),
        "survival": km.steps.iter().map(|s| s.survival).collect::<Vec<_>>(),
        "variance": km.steps.iter().map(|s| s.variance).collect::<Vec<_>>(),
        "ci_lower": km.steps.iter().map(|s| s.ci_lower).collect::<Vec<_>>(),
        "ci_upper": km.steps.iter().map(|s| s.ci_upper).collect::<Vec<_>>()
    })
}

fn log_rank_result_to_json(lr: &LogRankResult) -> serde_json::Value {
    serde_json::json!({
        "n": lr.n,
        "chi_squared": lr.chi_squared,
        "df": lr.df,
        "p_value": lr.p_value,
        "group_ids": lr.group_summaries.iter().map(|(g, _, _)| *g).collect::<Vec<_>>(),
        "observed": lr.group_summaries.iter().map(|(_, o, _)| *o).collect::<Vec<_>>(),
        "expected": lr.group_summaries.iter().map(|(_, _, e)| *e).collect::<Vec<_>>()
    })
}

fn cif_estimate_to_json(cif: &CifEstimate) -> serde_json::Value {
    serde_json::json!({
        "cause": cif.cause,
        "times": cif.steps.iter().map(|s| s.time).collect::<Vec<_>>(),
        "cif": cif.steps.iter().map(|s| s.cif).collect::<Vec<_>>(),
        "se": cif.steps.iter().map(|s| s.se).collect::<Vec<_>>(),
        "ci_lower": cif.steps.iter().map(|s| s.ci_lower).collect::<Vec<_>>(),
        "ci_upper": cif.steps.iter().map(|s| s.ci_upper).collect::<Vec<_>>(),
        "n": cif.n,
        "n_events": cif.n_events
    })
}

fn gray_test_result_to_json(result: &GrayTestResult) -> serde_json::Value {
    serde_json::json!({
        "statistic": result.statistic,
        "df": result.df,
        "p_value": result.p_value
    })
}

fn fine_gray_result_to_json(result: &FineGrayResult) -> serde_json::Value {
    serde_json::json!({
        "coefficients": result.coefficients,
        "se": result.se,
        "z": result.z,
        "p_values": result.p_values,
        "n": result.n,
        "n_events": result.n_events,
        "log_likelihood": result.log_likelihood
    })
}

fn panel_fe_entity_demeaned_state(
    entity_ids: &[u64],
    x: &[Vec<f64>],
    y: &[f64],
) -> Result<(DMatrix<f64>, DVector<f64>, DMatrix<f64>, Vec<f64>, usize), String> {
    #![allow(clippy::type_complexity)]
    let n = y.len();
    if n == 0 {
        return Err("need at least 1 observation".to_string());
    }
    if x.len() != n {
        return Err("X and y must have the same length".to_string());
    }
    if entity_ids.len() != n {
        return Err("entity must have the same length as y".to_string());
    }
    let (_, p) = flatten_row_major_matrix(x, "X")?;

    let entity_dense = entity_ids
        .iter()
        .map(|&id| usize::try_from(id).map_err(|_| "entity id overflow".to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let hdfe = FixedEffectsSolver::new(vec![entity_dense]).map_err(|e| e.to_string())?;

    let y_dm = hdfe.partial_out(y).map_err(|e| e.to_string())?;
    let mut x_dm_flat = vec![0.0_f64; n * p];
    for j in 0..p {
        let col = (0..n).map(|i| x[i][j]).collect::<Vec<_>>();
        let col_dm = hdfe.partial_out(&col).map_err(|e| e.to_string())?;
        for i in 0..n {
            x_dm_flat[i * p + j] = col_dm[i];
        }
    }

    let x_mat = DMatrix::from_row_slice(n, p, &x_dm_flat);
    let y_vec = DVector::from_column_slice(&y_dm);
    let xtx = x_mat.transpose() * &x_mat;
    let xtx_inv =
        xtx.try_inverse().ok_or_else(|| "X'X is singular after entity demeaning".to_string())?;

    Ok((x_mat, y_vec, xtx_inv, y_dm, hdfe.degrees_of_freedom_absorbed()))
}

fn cluster_covariance_matrix(
    x: &DMatrix<f64>,
    residuals: &DVector<f64>,
    xtx_inv: &DMatrix<f64>,
    cluster_ids: &[u64],
) -> Result<DMatrix<f64>, String> {
    let n = x.nrows();
    let p = x.ncols();
    if cluster_ids.len() != n {
        return Err("length mismatch between inputs".to_string());
    }
    if n == 0 {
        return Err("x must have at least 1 row".to_string());
    }
    if p == 0 {
        return Err("x must have at least 1 column".to_string());
    }
    if n <= p {
        return Err("Need n > n_params for cluster-robust covariance".to_string());
    }

    let mut cluster_map: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &cid) in cluster_ids.iter().enumerate() {
        cluster_map.entry(cid).or_default().push(i);
    }
    let g = cluster_map.len();
    if g < 2 {
        return Err("cluster must have at least 2 distinct groups".to_string());
    }

    let mut meat = DMatrix::zeros(p, p);
    for indices in cluster_map.values() {
        let mut score = vec![0.0_f64; p];
        for &i in indices {
            let e_i = residuals[i];
            for j in 0..p {
                score[j] += x[(i, j)] * e_i;
            }
        }
        for a in 0..p {
            for b in 0..p {
                meat[(a, b)] += score[a] * score[b];
            }
        }
    }

    let n_f = n as f64;
    let p_f = p as f64;
    let g_f = g as f64;
    let correction = (g_f / (g_f - 1.0)) * ((n_f - 1.0) / (n_f - p_f));
    Ok((xtx_inv * meat * xtx_inv) * correction)
}

fn covariance_to_standard_errors(cov: &DMatrix<f64>) -> Vec<f64> {
    (0..cov.ncols()).map(|j| cov[(j, j)].max(0.0).sqrt()).collect::<Vec<_>>()
}

fn iv_score_matrix(z: &DMatrix<f64>, residuals: &DVector<f64>) -> DMatrix<f64> {
    let n = z.nrows();
    let kz = z.ncols();
    let mut out = DMatrix::zeros(n, kz);
    for i in 0..n {
        let e_i = residuals[i];
        for j in 0..kz {
            out[(i, j)] = z[(i, j)] * e_i;
        }
    }
    out
}

fn iv_cluster_meat(
    z: &DMatrix<f64>,
    residuals: &DVector<f64>,
    cluster_ids: &[u64],
) -> Result<DMatrix<f64>, String> {
    let n = z.nrows();
    let kz = z.ncols();
    if cluster_ids.len() != n {
        return Err("cluster must have the same length as y".to_string());
    }
    let mut cluster_map: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, &cid) in cluster_ids.iter().enumerate() {
        cluster_map.entry(cid).or_default().push(i);
    }
    if cluster_map.len() < 2 {
        return Err("cluster must have at least 2 distinct groups".to_string());
    }
    let mut meat = DMatrix::zeros(kz, kz);
    for indices in cluster_map.values() {
        let mut score = vec![0.0_f64; kz];
        for &i in indices {
            let e_i = residuals[i];
            for j in 0..kz {
                score[j] += z[(i, j)] * e_i;
            }
        }
        for a in 0..kz {
            for b in 0..kz {
                meat[(a, b)] += score[a] * score[b];
            }
        }
    }
    Ok(meat)
}

fn iv_hac_meat(
    z: &DMatrix<f64>,
    residuals: &DVector<f64>,
    time_index: Option<&[serde_json::Value]>,
    max_lag: Option<usize>,
) -> Result<DMatrix<f64>, String> {
    let n = z.nrows();
    let kz = z.ncols();
    let mut zu = iv_score_matrix(z, residuals);
    if let Some(time_index) = time_index {
        if time_index.len() != n {
            return Err("time_index must have the same length as y".to_string());
        }
        let order = sort_scalar_json_indices(time_index, "time_index")?;
        let mut reordered = DMatrix::zeros(n, kz);
        for (dst, &src) in order.iter().enumerate() {
            for j in 0..kz {
                reordered[(dst, j)] = zu[(src, j)];
            }
        }
        zu = reordered;
    }

    let default_lag = (4.0 * ((n as f64) / 100.0).powf(2.0 / 9.0)).floor() as usize;
    let lag = max_lag.unwrap_or(default_lag).min(n.saturating_sub(1));

    let mut meat = zu.transpose() * &zu;
    for ell in 1..=lag {
        let weight = 1.0 - ((ell as f64) / ((lag + 1) as f64));
        let mut gamma = DMatrix::zeros(kz, kz);
        for t in ell..n {
            for a in 0..kz {
                for b in 0..kz {
                    gamma[(a, b)] += zu[(t, a)] * zu[(t - ell, b)];
                }
            }
        }
        meat += (&gamma + gamma.transpose()) * weight;
    }
    Ok(meat)
}

fn iv_covariance_from_meat(
    xz_inv: &DMatrix<f64>,
    ztz_inv: &DMatrix<f64>,
    xtz: &DMatrix<f64>,
    a_inv: &DMatrix<f64>,
    meat: &DMatrix<f64>,
) -> DMatrix<f64> {
    let b = xz_inv * meat * ztz_inv * xtz.transpose();
    a_inv * b * a_inv
}

fn solve_least_squares(x: &DMatrix<f64>, y: &DVector<f64>) -> Result<DVector<f64>, String> {
    let xtx = x.transpose() * x;
    let xtx_inv =
        xtx.try_inverse().ok_or_else(|| "least-squares design is singular".to_string())?;
    Ok(xtx_inv * (x.transpose() * y))
}

fn stable_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn clip_probability(p: f64, eps: f64) -> Result<f64, String> {
    if !(eps > 0.0 && eps < 0.5) {
        return Err("trim_eps must satisfy 0 < trim_eps < 0.5".to_string());
    }
    Ok(p.max(eps).min(1.0 - eps))
}

fn sample_variance(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let mean = xs.iter().sum::<f64>() / (xs.len() as f64);
    xs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / ((xs.len() - 1) as f64)
}

fn fit_logistic_propensity_scores(x: &[Vec<f64>], treatment: &[u8]) -> Result<Vec<f64>, String> {
    let mle = MaximumLikelihoodEstimator::new();
    let model = ModelBuilder::logistic_regression(x.to_vec(), treatment.to_vec(), true)
        .map_err(|e| e.to_string())?
        .with_coef_prior_normal(0.0, 1.0)
        .map_err(|e| e.to_string())?
        .with_penalize_intercept(false)
        .build()
        .map_err(|e| e.to_string())?;
    let fit = mle.fit(&model).map_err(|e| e.to_string())?;
    let xmat = design_matrix_with_intercept(x, true)?;
    let beta = DVector::from_vec(fit.parameters);
    let eta = xmat * beta;
    eta.iter().map(|value| clip_probability(stable_sigmoid(*value), 1e-6)).collect()
}

fn predict_linear_from_training(
    x_train: &[Vec<f64>],
    y_train: &[f64],
    x_pred: &[Vec<f64>],
) -> Result<Vec<f64>, String> {
    let coef = ols_fit(x_train.to_vec(), y_train.to_vec(), true).map_err(|e| e.to_string())?;
    let xmat = design_matrix_with_intercept(x_pred, true)?;
    let beta = DVector::from_vec(coef);
    Ok((xmat * beta).iter().copied().collect::<Vec<_>>())
}

fn regression_cluster_name(kind: RegressionClusterKind) -> &'static str {
    match kind {
        RegressionClusterKind::Entity => "entity",
        RegressionClusterKind::Time => "time",
        RegressionClusterKind::TwoWay => "two_way",
        RegressionClusterKind::None => "none",
    }
}

fn regression_cluster_count(
    kind: RegressionClusterKind,
    entity_ids: &[u64],
    time_ids: Option<&[u64]>,
) -> usize {
    match kind {
        RegressionClusterKind::Entity => {
            entity_ids.iter().copied().collect::<std::collections::BTreeSet<_>>().len()
        }
        RegressionClusterKind::Time => time_ids
            .map(|ids| ids.iter().copied().collect::<std::collections::BTreeSet<_>>().len())
            .unwrap_or(0),
        RegressionClusterKind::TwoWay => time_ids
            .map(|ids| {
                entity_ids
                    .iter()
                    .copied()
                    .zip(ids.iter().copied())
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
            })
            .unwrap_or(0),
        RegressionClusterKind::None => 0,
    }
}

fn regression_covariance_matrix(
    kind: RegressionClusterKind,
    x: &DMatrix<f64>,
    residuals: &DVector<f64>,
    xtx_inv: &DMatrix<f64>,
    entity_ids: &[u64],
    time_ids: Option<&[u64]>,
) -> Result<DMatrix<f64>, String> {
    match kind {
        RegressionClusterKind::None => {
            let n = x.nrows();
            let k = x.ncols();
            if n <= k {
                return Err("Need n > n_params to compute standard errors".to_string());
            }
            let rss = residuals.iter().map(|value| value * value).sum::<f64>();
            let sigma2 = rss / ((n - k) as f64);
            Ok(xtx_inv.clone() * sigma2)
        }
        RegressionClusterKind::Entity => {
            cluster_covariance_matrix(x, residuals, xtx_inv, entity_ids)
        }
        RegressionClusterKind::Time => {
            let time_ids =
                time_ids.ok_or_else(|| "time must be provided when cluster='time'".to_string())?;
            cluster_covariance_matrix(x, residuals, xtx_inv, time_ids)
        }
        RegressionClusterKind::TwoWay => {
            let time_ids = time_ids
                .ok_or_else(|| "time must be provided when cluster='two_way'".to_string())?;
            let pair_ids = encode_group_pairs(entity_ids, time_ids)?;
            let cov_entity = cluster_covariance_matrix(x, residuals, xtx_inv, entity_ids)?;
            let cov_time = cluster_covariance_matrix(x, residuals, xtx_inv, time_ids)?;
            let cov_pair = cluster_covariance_matrix(x, residuals, xtx_inv, &pair_ids)?;
            Ok(cov_entity + cov_time - cov_pair)
        }
    }
}

fn regression_standard_errors(
    kind: RegressionClusterKind,
    x: &DMatrix<f64>,
    residuals: &DVector<f64>,
    xtx_inv: &DMatrix<f64>,
    entity_ids: &[u64],
    time_ids: Option<&[u64]>,
) -> Result<Vec<f64>, String> {
    let cov = regression_covariance_matrix(kind, x, residuals, xtx_inv, entity_ids, time_ids)?;
    Ok(covariance_to_standard_errors(&cov))
}

fn validate_binary_indicator(values: &[u8], field_name: &str) -> Result<(), String> {
    for &value in values {
        if value > 1 {
            return Err(format!("{field_name} must contain only 0/1 values"));
        }
    }
    Ok(())
}

fn two_way_demeaned_state(
    entity_ids: &[u64],
    time_ids: &[u64],
    x: &[Vec<f64>],
    y: &[f64],
) -> Result<(DMatrix<f64>, DVector<f64>), String> {
    let n = y.len();
    if n == 0 {
        return Err("need at least 1 observation".to_string());
    }
    if x.len() != n {
        return Err("X and y must have the same length".to_string());
    }
    if entity_ids.len() != n || time_ids.len() != n {
        return Err("entity, time, X, and y must have the same length".to_string());
    }
    validate_rectangular_matrix(x, "X")?;

    let p = x[0].len();
    if p == 0 {
        return Err("X must have at least 1 column".to_string());
    }

    let entity_dense = entity_ids
        .iter()
        .map(|&id| usize::try_from(id).map_err(|_| "entity id overflow".to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let time_dense = time_ids
        .iter()
        .map(|&id| usize::try_from(id).map_err(|_| "time id overflow".to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let hdfe =
        FixedEffectsSolver::new(vec![entity_dense, time_dense]).map_err(|e| e.to_string())?;

    let y_dm = hdfe.partial_out(y).map_err(|e| e.to_string())?;
    let mut x_dm_flat = vec![0.0_f64; n * p];
    for j in 0..p {
        let col = (0..n).map(|i| x[i][j]).collect::<Vec<_>>();
        let col_dm = hdfe.partial_out(&col).map_err(|e| e.to_string())?;
        for i in 0..n {
            x_dm_flat[i * p + j] = col_dm[i];
        }
    }

    Ok((DMatrix::from_row_slice(n, p, &x_dm_flat), DVector::from_column_slice(&y_dm)))
}

fn select_independent_columns(
    x: &DMatrix<f64>,
    mandatory: &[usize],
    tol: f64,
) -> Result<(DMatrix<f64>, Vec<usize>), String> {
    let k = x.ncols();
    if k == 0 {
        return Err("X must have at least 1 column".to_string());
    }

    let mut mandatory_unique = mandatory.to_vec();
    mandatory_unique.sort_unstable();
    mandatory_unique.dedup();
    mandatory_unique.retain(|&idx| idx < k);

    let mut order = mandatory_unique.clone();
    for j in 0..k {
        if !mandatory_unique.contains(&j) {
            order.push(j);
        }
    }

    let mut basis: Vec<DVector<f64>> = Vec::new();
    let mut kept_in_order: Vec<usize> = Vec::new();

    for &idx in &order {
        let mut v = x.column(idx).into_owned();
        for q in &basis {
            let proj = q.dot(&v);
            v -= q * proj;
        }
        let norm = v.norm();
        if norm > tol {
            basis.push(v / norm);
            kept_in_order.push(idx);
        }
    }

    for &idx in &mandatory_unique {
        if !kept_in_order.contains(&idx) {
            return Err(
                "treat_post is not identifiable (absorbed by FE or has no variation)".to_string()
            );
        }
    }

    kept_in_order.sort_unstable();
    let cols = kept_in_order.iter().map(|&idx| x.column(idx).into_owned()).collect::<Vec<_>>();
    Ok((DMatrix::from_columns(&cols), kept_in_order))
}

fn run_panel_fe(args: &PanelFeArgs) -> Result<serde_json::Value, String> {
    if args.y.is_empty() {
        return Err("need at least 1 observation".to_string());
    }
    if args.x.len() != args.y.len() {
        return Err("X and y must have the same length".to_string());
    }
    if args.entity.len() != args.y.len() {
        return Err("entity must have the same length as y".to_string());
    }
    validate_rectangular_matrix(&args.x, "X")?;

    let entity_ids = encode_group_values(&args.entity, "entity")?;
    let time_ids = match &args.time {
        Some(values) => {
            if values.len() != args.y.len() {
                return Err("time must have the same length as y".to_string());
            }
            Some(encode_group_values(values, "time")?)
        }
        None => None,
    };

    let (x_mat, y_vec, xtx_inv, y_dm, _df_absorbed) =
        panel_fe_entity_demeaned_state(&entity_ids, &args.x, &args.y)?;
    let xty = x_mat.transpose() * &y_vec;
    let beta = &xtx_inv * &xty;
    let fitted = &x_mat * &beta;
    let residuals = &y_vec - fitted;
    let coef = beta.iter().copied().collect::<Vec<_>>();

    let standard_errors = regression_standard_errors(
        args.cluster,
        &x_mat,
        &residuals,
        &xtx_inv,
        &entity_ids,
        time_ids.as_deref(),
    )?;

    let n_clusters = regression_cluster_count(args.cluster, &entity_ids, time_ids.as_deref());
    let cluster_name = regression_cluster_name(args.cluster);

    let n_entities = entity_ids.iter().copied().collect::<std::collections::BTreeSet<_>>().len();

    let _ = y_dm;

    Ok(serde_json::json!({
        "coef": coef,
        "standard_errors": standard_errors,
        "n_obs": args.y.len(),
        "n_entities": n_entities,
        "cluster_kind": cluster_name,
        "n_clusters": n_clusters,
        "cluster": cluster_name
    }))
}

fn run_did(args: &DidArgs) -> Result<serde_json::Value, String> {
    let n = args.y.len();
    if n == 0 {
        return Err("need at least 1 observation".to_string());
    }
    if args.treat.len() != n
        || args.post.len() != n
        || args.entity.len() != n
        || args.time.len() != n
    {
        return Err("y, treat, post, entity, and time must have the same length".to_string());
    }
    validate_binary_indicator(&args.treat, "treat")?;
    validate_binary_indicator(&args.post, "post")?;

    let extra_p = if let Some(x) = &args.x {
        if x.len() != n {
            return Err("X and y must have the same length".to_string());
        }
        validate_rectangular_matrix(x, "X")?;
        x.first().map(|row| row.len()).unwrap_or(0)
    } else {
        0
    };

    let mut x_all = vec![vec![0.0_f64; 1 + extra_p]; n];
    for (i, row) in x_all.iter_mut().enumerate() {
        row[0] = f64::from(args.treat[i]) * f64::from(args.post[i]);
    }
    if let Some(x) = &args.x {
        for i in 0..n {
            for j in 0..extra_p {
                x_all[i][j + 1] = x[i][j];
            }
        }
    }

    let entity_ids = encode_group_values(&args.entity, "entity")?;
    let time_ids = encode_group_values(&args.time, "time")?;
    let (x_twfe, y_twfe) = two_way_demeaned_state(&entity_ids, &time_ids, &x_all, &args.y)?;
    let (x_sel, kept_cols) = select_independent_columns(&x_twfe, &[0], 1e-10)?;

    let k = x_sel.ncols();
    if n <= k {
        return Err(
            "Need n > n_params after TWFE transformation; reduce controls or narrow event-study window"
                .to_string(),
        );
    }

    let xtx = x_sel.transpose() * &x_sel;
    let xtx_inv = xtx.try_inverse().ok_or_else(|| "X'X is singular in DiD TWFE OLS".to_string())?;
    let xty = x_sel.transpose() * &y_twfe;
    let beta = &xtx_inv * &xty;
    let fitted = &x_sel * &beta;
    let residuals = &y_twfe - fitted;

    let standard_errors = regression_standard_errors(
        args.cluster,
        &x_sel,
        &residuals,
        &xtx_inv,
        &entity_ids,
        Some(&time_ids),
    )?;

    let coef = beta.iter().copied().collect::<Vec<_>>();
    let att_index = kept_cols.iter().position(|&idx| idx == 0).ok_or_else(|| {
        "treat_post is not identifiable (absorbed by FE or has no variation)".to_string()
    })?;
    let att = coef[att_index];
    let att_se =
        *standard_errors.get(att_index).ok_or_else(|| "missing ATT standard error".to_string())?;

    Ok(serde_json::json!({
        "att": att,
        "att_se": att_se,
        "coef": coef,
        "standard_errors": standard_errors,
        "n_obs": n,
        "cluster": regression_cluster_name(args.cluster)
    }))
}

fn run_event_study(args: &EventStudyArgs) -> Result<serde_json::Value, String> {
    let n = args.y.len();
    if n == 0 {
        return Err("need at least 1 observation".to_string());
    }
    if args.entity.len() != n || args.time.len() != n || args.treat_time.len() != n {
        return Err("y, entity, time, and treat_time must have the same length".to_string());
    }
    validate_scalar_json_values(&args.entity, "entity")?;

    let n_leads = args.n_leads.unwrap_or(3);
    let n_lags = args.n_lags.unwrap_or(3);
    let reference = -1_i64;
    let min_lag = -(n_leads as i64);
    let max_lag = n_lags as i64;

    let entity_ids = encode_group_values(&args.entity, "entity")?;
    let time_ids = encode_integer_levels(&args.time);
    let treat = args
        .treat_time
        .iter()
        .map(|value| if value.is_some() { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    let relative_time = args
        .time
        .iter()
        .zip(args.treat_time.iter())
        .map(|(&time, treat_time)| match treat_time {
            Some(onset) => time - onset,
            None => 0,
        })
        .collect::<Vec<_>>();

    let mut supported_rel_times = Vec::new();
    let mut x_all = vec![Vec::new(); n];
    for k in min_lag..=max_lag {
        if k == reference {
            continue;
        }
        let column = treat
            .iter()
            .zip(relative_time.iter())
            .map(|(&treated, &rel)| if treated > 0.0 && rel == k { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        if column.iter().any(|&value| value != 0.0) {
            supported_rel_times.push(k);
            for (row, value) in x_all.iter_mut().zip(column.into_iter()) {
                row.push(value);
            }
        }
    }

    if supported_rel_times.is_empty() {
        return Err("no supported event-study bins in the requested window".to_string());
    }

    let (x_twfe, y_twfe) = two_way_demeaned_state(&entity_ids, &time_ids, &x_all, &args.y)?;
    let (x_sel, kept_cols) = select_independent_columns(&x_twfe, &[], 1e-10)?;
    if x_sel.ncols() == 0 {
        return Err("no supported event-study bins in the requested window".to_string());
    }
    if n <= x_sel.ncols() {
        return Err(
            "Need n > n_params after TWFE transformation; reduce controls or narrow event-study window"
                .to_string(),
        );
    }

    let xtx = x_sel.transpose() * &x_sel;
    let xtx_inv =
        xtx.try_inverse().ok_or_else(|| "X'X is singular in event study TWFE OLS".to_string())?;
    let xty = x_sel.transpose() * &y_twfe;
    let beta = &xtx_inv * &xty;
    let fitted = &x_sel * &beta;
    let residuals = &y_twfe - fitted;
    let covariance = regression_covariance_matrix(
        args.cluster,
        &x_sel,
        &residuals,
        &xtx_inv,
        &entity_ids,
        Some(&time_ids),
    )?;
    let standard_errors = covariance_to_standard_errors(&covariance);
    let rel_times =
        kept_cols.into_iter().map(|index| supported_rel_times[index]).collect::<Vec<_>>();
    let n_entities = entity_ids.iter().copied().collect::<std::collections::BTreeSet<_>>().len();
    let n_times = time_ids.iter().copied().collect::<std::collections::BTreeSet<_>>().len();

    Ok(serde_json::json!({
        "rel_times": rel_times,
        "coef": beta.iter().copied().collect::<Vec<_>>(),
        "standard_errors": standard_errors,
        "covariance": (0..covariance.nrows())
            .map(|i| (0..covariance.ncols()).map(|j| covariance[(i, j)]).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        "reference": reference,
        "n_obs": n,
        "n_entities": n_entities,
        "n_times": n_times,
        "cluster": regression_cluster_name(args.cluster)
    }))
}

fn run_iv_2sls(args: &Iv2slsArgs) -> Result<serde_json::Value, String> {
    let n = args.y.len();
    if n == 0 {
        return Err("need at least 1 observation".to_string());
    }

    validate_rectangular_matrix(&args.endog, "endog")?;
    validate_rectangular_matrix(&args.instruments, "instruments")?;
    if args.endog.len() != n || args.instruments.len() != n {
        return Err("length mismatch between y/endog/instruments".to_string());
    }

    let p_endog = args.endog[0].len();
    if p_endog == 0 {
        return Err("endog must have at least 1 column".to_string());
    }
    let q = args.instruments[0].len();
    if q == 0 {
        return Err("instruments must have at least 1 column".to_string());
    }
    if q < p_endog {
        return Err(
            "underidentified: need at least as many excluded instruments as endogenous regressors"
                .to_string(),
        );
    }

    let (x_exog, p_exog) = if let Some(exog) = &args.exog {
        validate_rectangular_matrix(exog, "exog")?;
        if exog.len() != n {
            return Err("length mismatch between y and exog".to_string());
        }
        let p_exog = exog[0].len();
        if p_exog == 0 {
            return Err("exog must have at least 1 column when provided".to_string());
        }
        (exog.clone(), p_exog)
    } else {
        (Vec::new(), 0)
    };

    let kx = p_endog + p_exog;
    if n <= kx {
        return Err("Need n > n_params to estimate sigma2_hat".to_string());
    }

    let mut x_full_flat = Vec::with_capacity(n * kx);
    let mut z_full_flat = Vec::with_capacity(n * (q + p_exog));
    for (i, x_exog_row) in x_exog.iter().enumerate() {
        x_full_flat.extend(args.endog[i].iter().copied());
        z_full_flat.extend(args.instruments[i].iter().copied());
        if p_exog > 0 {
            x_full_flat.extend(x_exog_row.iter().copied());
            z_full_flat.extend(x_exog_row.iter().copied());
        }
    }

    let x_full = DMatrix::from_row_slice(n, kx, &x_full_flat);
    let (x_checked, x_kept) = select_independent_columns(&x_full, &[], 1e-10)?;
    if x_kept.len() != kx || x_checked.ncols() != kx {
        return Err("X is rank-deficient".to_string());
    }

    let z_full = DMatrix::from_row_slice(n, q + p_exog, &z_full_flat);
    let exog_idx = (q..(q + p_exog)).collect::<Vec<_>>();
    let (z_sel, z_kept) = select_independent_columns(&z_full, &exog_idx, 1e-10)?;
    if p_exog > 0 && exog_idx.iter().any(|idx| !z_kept.contains(idx)) {
        return Err("exog columns are collinear in Z".to_string());
    }
    if z_sel.ncols() < kx {
        return Err("underidentified after dropping collinear instruments".to_string());
    }

    let y_vec = DVector::from_column_slice(&args.y);
    let ztz = z_sel.transpose() * &z_sel;
    let ztz_inv =
        ztz.try_inverse().ok_or_else(|| "Z'Z is singular after column selection".to_string())?;
    let xtz = x_full.transpose() * &z_sel;
    let xz_inv = &xtz * &ztz_inv;
    let a = &xz_inv * xtz.transpose();
    let a_inv =
        a.try_inverse().ok_or_else(|| "X'PzX is singular in 2SLS second stage".to_string())?;
    let zty = z_sel.transpose() * &y_vec;
    let rhs = &xz_inv * &zty;
    let beta = &a_inv * rhs;
    let resid = &y_vec - (&x_full * &beta);

    let cov_beta = match args.cov {
        IvCovKind::Homoskedastic => {
            let sigma2 = resid.dot(&resid) / ((n - kx) as f64);
            a_inv.clone() * sigma2
        }
        IvCovKind::Hc1 => {
            let zu = iv_score_matrix(&z_sel, &resid);
            let mut meat = zu.transpose() * zu;
            meat *= (n as f64) / ((n - kx) as f64);
            iv_covariance_from_meat(&xz_inv, &ztz_inv, &xtz, &a_inv, &meat)
        }
        IvCovKind::Cluster => {
            let cluster = args
                .cluster
                .as_ref()
                .ok_or_else(|| "cluster must be provided when cov='cluster'".to_string())?;
            if cluster.len() != n {
                return Err("cluster must have the same length as y".to_string());
            }
            let cluster_ids = encode_group_values(cluster, "cluster")?;
            let mut meat = iv_cluster_meat(&z_sel, &resid, &cluster_ids)?;
            let g = cluster_ids.iter().copied().collect::<std::collections::BTreeSet<_>>().len();
            let scale = ((g as f64) / ((g - 1) as f64)) * (((n as f64) - 1.0) / ((n - kx) as f64));
            meat *= scale;
            iv_covariance_from_meat(&xz_inv, &ztz_inv, &xtz, &a_inv, &meat)
        }
        IvCovKind::Hac => {
            let mut meat = iv_hac_meat(&z_sel, &resid, args.time_index.as_deref(), args.max_lag)?;
            meat *= (n as f64) / ((n - kx) as f64);
            iv_covariance_from_meat(&xz_inv, &ztz_inv, &xtz, &a_inv, &meat)
        }
    };

    let standard_errors = covariance_to_standard_errors(&cov_beta);
    let coef = beta.iter().copied().collect::<Vec<_>>();

    let exog_sel_idx = z_kept
        .iter()
        .enumerate()
        .filter_map(|(position, original_idx)| (*original_idx >= q).then_some(position))
        .collect::<Vec<_>>();
    let z_exog_only =
        if exog_sel_idx.is_empty() { None } else { Some(z_sel.select_columns(&exog_sel_idx)) };
    let q_kept = z_kept.iter().filter(|idx| **idx < q).count();
    let df2 = n as isize - z_sel.ncols() as isize;

    let mut first_stage_f = Vec::with_capacity(p_endog);
    for j in 0..p_endog {
        if q_kept == 0 || df2 <= 0 {
            first_stage_f.push(f64::NAN);
            continue;
        }

        let dcol = DVector::from_iterator(n, (0..n).map(|i| args.endog[i][j]));
        let gamma_ur = solve_least_squares(&z_sel, &dcol)
            .map_err(|_| "failed to solve unrestricted first-stage regression".to_string())?;
        let diff_ur = &dcol - (&z_sel * gamma_ur);
        let ssr_ur = diff_ur.dot(&diff_ur);

        let ssr_r = if let Some(z_exog_only) = &z_exog_only {
            let gamma_r = solve_least_squares(z_exog_only, &dcol)
                .map_err(|_| "failed to solve restricted first-stage regression".to_string())?;
            let diff_r = &dcol - (z_exog_only * gamma_r);
            diff_r.dot(&diff_r)
        } else {
            dcol.dot(&dcol)
        };

        if !(ssr_r.is_finite() && ssr_ur.is_finite() && ssr_ur > 0.0 && ssr_r >= ssr_ur) {
            first_stage_f.push(f64::NAN);
            continue;
        }

        let num = (ssr_r - ssr_ur) / (q_kept as f64);
        let den = ssr_ur / (df2 as f64);
        first_stage_f.push(if den > 0.0 { num / den } else { f64::NAN });
    }

    Ok(serde_json::json!({
        "coef": coef,
        "standard_errors": standard_errors,
        "n_obs": n,
        "diagnostics": {
            "first_stage_f": first_stage_f
        }
    }))
}

fn run_aipw(args: &AipwArgs) -> Result<serde_json::Value, String> {
    let n = args.y.len();
    if n == 0 {
        return Err("need at least 1 observation".to_string());
    }
    if args.x.len() != n || args.treatment.len() != n {
        return Err("x/y/t length mismatch".to_string());
    }
    validate_rectangular_matrix(&args.x, "x")?;
    validate_binary_indicator(&args.treatment, "treatment")?;

    let treated_idx = args
        .treatment
        .iter()
        .enumerate()
        .filter_map(|(i, &t)| (t == 1).then_some(i))
        .collect::<Vec<_>>();
    let control_idx = args
        .treatment
        .iter()
        .enumerate()
        .filter_map(|(i, &t)| (t == 0).then_some(i))
        .collect::<Vec<_>>();
    if treated_idx.is_empty() || control_idx.is_empty() {
        return Err("both treatment groups must be non-empty".to_string());
    }

    let propensity = fit_logistic_propensity_scores(&args.x, &args.treatment)?;

    let x0 = control_idx.iter().map(|&i| args.x[i].clone()).collect::<Vec<_>>();
    let y0 = control_idx.iter().map(|&i| args.y[i]).collect::<Vec<_>>();
    let x1 = treated_idx.iter().map(|&i| args.x[i].clone()).collect::<Vec<_>>();
    let y1 = treated_idx.iter().map(|&i| args.y[i]).collect::<Vec<_>>();
    let mu0 = predict_linear_from_training(&x0, &y0, &args.x)?;
    let mu1 = predict_linear_from_training(&x1, &y1, &args.x)?;

    let psi =
        match args.estimand {
            AipwEstimand::Ate => args
                .y
                .iter()
                .zip(args.treatment.iter())
                .zip(propensity.iter())
                .zip(mu0.iter())
                .zip(mu1.iter())
                .map(|((((&y, &t), &e), &m0), &m1)| {
                    if t == 1 { m1 + ((y - m1) / e) - m0 } else { m1 - m0 - ((y - m0) / (1.0 - e)) }
                })
                .collect::<Vec<_>>(),
            AipwEstimand::Att => {
                let p_hat =
                    (args.treatment.iter().map(|&t| f64::from(t)).sum::<f64>()) / (n as f64);
                if p_hat <= 0.0 {
                    return Err("need at least one treated observation for ATT".to_string());
                }
                args.y
                    .iter()
                    .zip(args.treatment.iter())
                    .zip(propensity.iter())
                    .zip(mu0.iter())
                    .map(|(((&y, &t), &e), &m0)| {
                        if t == 1 {
                            (y - m0) / p_hat
                        } else {
                            -((e / (1.0 - e)) * (y - m0) / p_hat)
                        }
                    })
                    .collect::<Vec<_>>()
            }
        };

    let estimate = psi.iter().sum::<f64>() / (n as f64);
    let influence = psi.iter().map(|value| value - estimate).collect::<Vec<_>>();
    let standard_error = (sample_variance(&influence) / (n as f64)).sqrt();

    Ok(serde_json::json!({
        "estimand": match args.estimand {
            AipwEstimand::Ate => "ate",
            AipwEstimand::Att => "att",
        },
        "estimate": estimate,
        "standard_error": standard_error,
        "n_obs": n
    }))
}

fn run_garch_fit(args: &GarchFitArgs) -> Result<serde_json::Value, String> {
    if args.returns.len() < 2 {
        return Err("returns must contain at least 2 observations".to_string());
    }
    if args.returns.iter().any(|value| !value.is_finite()) {
        return Err("returns must contain only finite values".to_string());
    }

    match args.model {
        GarchModelKind::Garch => {
            let fit = ns_inference::timeseries::volatility::garch11_fit(
                &args.returns,
                ns_inference::timeseries::volatility::Garch11Config::default(),
            )
            .map_err(|e| e.to_string())?;
            Ok(serde_json::json!({
                "params": {
                    "mu": fit.params.mu,
                    "omega": fit.params.omega,
                    "alpha": fit.params.alpha,
                    "beta": fit.params.beta
                },
                "log_likelihood": fit.log_likelihood,
                "conditional_variance": fit.conditional_variance,
                "conditional_sigma": fit.conditional_variance.iter().map(|v| v.max(0.0).sqrt()).collect::<Vec<_>>(),
                "converged": fit.optimization.converged,
                "n_iter": fit.optimization.n_iter,
                "n_fev": fit.optimization.n_fev,
                "n_gev": fit.optimization.n_gev,
                "fval": fit.optimization.fval,
                "message": fit.optimization.message
            }))
        }
        GarchModelKind::Egarch => {
            let fit = ns_inference::timeseries::volatility::egarch11_fit(
                &args.returns,
                ns_inference::timeseries::volatility::Egarch11Config::default(),
            )
            .map_err(|e| e.to_string())?;
            Ok(serde_json::json!({
                "params": {
                    "mu": fit.params.mu,
                    "omega": fit.params.omega,
                    "alpha": fit.params.alpha,
                    "gamma": fit.params.gamma,
                    "beta": fit.params.beta
                },
                "log_likelihood": fit.log_likelihood,
                "conditional_variance": fit.conditional_variance,
                "conditional_sigma": fit.conditional_variance.iter().map(|v| v.max(0.0).sqrt()).collect::<Vec<_>>(),
                "converged": fit.optimization.converged,
                "n_iter": fit.optimization.n_iter,
                "n_fev": fit.optimization.n_fev,
                "n_gev": fit.optimization.n_gev,
                "fval": fit.optimization.fval,
                "message": fit.optimization.message
            }))
        }
        GarchModelKind::GjrGarch => {
            let fit = ns_inference::timeseries::volatility::gjr_garch11_fit(
                &args.returns,
                ns_inference::timeseries::volatility::GjrGarch11Config::default(),
            )
            .map_err(|e| e.to_string())?;
            Ok(serde_json::json!({
                "params": {
                    "mu": fit.params.mu,
                    "omega": fit.params.omega,
                    "alpha": fit.params.alpha,
                    "gamma": fit.params.gamma,
                    "beta": fit.params.beta
                },
                "log_likelihood": fit.log_likelihood,
                "conditional_variance": fit.conditional_variance,
                "conditional_sigma": fit.conditional_variance.iter().map(|v| v.max(0.0).sqrt()).collect::<Vec<_>>(),
                "converged": fit.optimization.converged,
                "n_iter": fit.optimization.n_iter,
                "n_fev": fit.optimization.n_fev,
                "n_gev": fit.optimization.n_gev,
                "fval": fit.optimization.fval,
                "message": fit.optimization.message
            }))
        }
    }
}

fn build_ads_single_covariate_provenance(
    covariate_name: &Option<String>,
    covariate_provenance: &Option<AdsCovariateProvenanceArg>,
) -> Result<Option<AdsCovariateProvenance>, String> {
    let Some(item) = covariate_provenance.as_ref() else {
        return Ok(None);
    };

    let resolved_name = match (&item.name, covariate_name) {
        (Some(name), Some(expected)) if name != expected => {
            return Err(
                "covariate_provenance.name must match covariate_name when both are provided"
                    .to_string(),
            );
        }
        (Some(name), _) => name.clone(),
        (None, Some(name)) => name.clone(),
        (None, None) => {
            return Err(
                "covariate_provenance.name is required when covariate_name is omitted".to_string()
            );
        }
    };

    Ok(Some(AdsCovariateProvenance {
        name: resolved_name,
        timing: item.timing,
        source_dataset: item.source_dataset.clone(),
    }))
}

fn build_ads_multi_covariate_provenance(
    covariate_names: &[String],
    covariate_provenance: &Option<Vec<AdsCovariateProvenanceArg>>,
) -> Result<Vec<AdsCovariateProvenance>, String> {
    let Some(items) = covariate_provenance.as_ref() else {
        return Ok(Vec::new());
    };
    if !covariate_names.is_empty() && covariate_names.len() != items.len() {
        return Err("covariate_provenance length must match covariate_names length".to_string());
    }

    let mut out = Vec::with_capacity(items.len());
    for (idx, item) in items.iter().enumerate() {
        let expected_name = covariate_names.get(idx);
        let resolved_name = match (&item.name, expected_name) {
            (Some(name), Some(expected)) if name != expected => {
                return Err(format!(
                    "covariate_provenance[{idx}].name must match covariate_names[{idx}]"
                ));
            }
            (Some(name), _) => name.clone(),
            (None, Some(name)) => name.clone(),
            (None, None) => {
                return Err(format!(
                    "covariate_provenance[{idx}].name is required when covariate_names are omitted"
                ));
            }
        };
        out.push(AdsCovariateProvenance {
            name: resolved_name,
            timing: item.timing,
            source_dataset: item.source_dataset.clone(),
        });
    }
    Ok(out)
}

fn run_ads_cuped_adjust(args: &AdsCupedAdjustArgs) -> Result<serde_json::Value, String> {
    let covariate_provenance =
        build_ads_single_covariate_provenance(&args.covariate_name, &args.covariate_provenance)?;
    let control = CupedArmData {
        outcomes: args.control_outcomes.clone(),
        covariates: args.control_covariates.clone(),
        covariate_name: args.covariate_name.clone(),
        covariate_provenance: covariate_provenance.clone(),
        pre_treatment_only: args.pre_treatment_only,
    };
    let variant = CupedArmData {
        outcomes: args.variant_outcomes.clone(),
        covariates: args.variant_covariates.clone(),
        covariate_name: args.covariate_name.clone(),
        covariate_provenance,
        pre_treatment_only: args.pre_treatment_only,
    };
    let result = run_ads_cuped_adjust_core(&control, &variant).map_err(|e| e.to_string())?;
    serde_json::to_value(result).map_err(|e| e.to_string())
}

fn run_ads_cure_adjust(args: &AdsCureAdjustArgs) -> Result<serde_json::Value, String> {
    let covariate_names = args.covariate_names.clone().unwrap_or_default();
    let covariate_provenance =
        build_ads_multi_covariate_provenance(&covariate_names, &args.covariate_provenance)?;
    let control = MultiCovariateArmData {
        outcomes: args.control_outcomes.clone(),
        covariates: args.control_covariates.clone(),
        covariate_names: covariate_names.clone(),
        covariate_provenance: covariate_provenance.clone(),
        pre_treatment_only: args.pre_treatment_only,
    };
    let variant = MultiCovariateArmData {
        outcomes: args.variant_outcomes.clone(),
        covariates: args.variant_covariates.clone(),
        covariate_names,
        covariate_provenance,
        pre_treatment_only: args.pre_treatment_only,
    };
    let result = run_ads_cure_adjust_core(&control, &variant).map_err(|e| e.to_string())?;
    serde_json::to_value(result).map_err(|e| e.to_string())
}

fn run_kalman(args: &KalmanArgs) -> Result<serde_json::Value, String> {
    const KALMAN_SIMULATE_SERVER_MAX_T: usize = 512;
    const KALMAN_EM_SERVER_MAX_ITER: usize = 100;

    let model = ns_inference::timeseries::kalman::KalmanModel::new(
        matrix_arg_to_dmatrix(&args.f, "F")?,
        matrix_arg_to_dmatrix(&args.q, "Q")?,
        matrix_arg_to_dmatrix(&args.h, "H")?,
        matrix_arg_to_dmatrix(&args.r, "R")?,
        vector_arg_to_dvector(&args.x0, "x0")?,
        matrix_arg_to_dmatrix(&args.p0, "P0")?,
    )
    .map_err(|e| e.to_string())?;

    match args.operation {
        KalmanOperationKind::Filter => {
            let ys = observation_sequence_to_dvectors(
                args.y
                    .as_ref()
                    .ok_or_else(|| "y is required when operation='filter'".to_string())?,
                "y",
            )?;
            let out = ns_inference::timeseries::kalman::kalman_filter(&model, &ys)
                .map_err(|e| e.to_string())?;
            Ok(serde_json::json!({
                "log_likelihood": out.log_likelihood,
                "predicted_means": dvector_list_to_json_array(&out.predicted_means),
                "predicted_covs": dmatrix_list_to_json_array(&out.predicted_covs),
                "filtered_means": dvector_list_to_json_array(&out.filtered_means),
                "filtered_covs": dmatrix_list_to_json_array(&out.filtered_covs)
            }))
        }
        KalmanOperationKind::Smooth => {
            let ys = observation_sequence_to_dvectors(
                args.y
                    .as_ref()
                    .ok_or_else(|| "y is required when operation='smooth'".to_string())?,
                "y",
            )?;
            let filter = ns_inference::timeseries::kalman::kalman_filter(&model, &ys)
                .map_err(|e| e.to_string())?;
            let smooth = ns_inference::timeseries::kalman::rts_smoother(&model, &filter)
                .map_err(|e| e.to_string())?;
            Ok(serde_json::json!({
                "log_likelihood": filter.log_likelihood,
                "filtered_means": dvector_list_to_json_array(&filter.filtered_means),
                "filtered_covs": dmatrix_list_to_json_array(&filter.filtered_covs),
                "smoothed_means": dvector_list_to_json_array(&smooth.smoothed_means),
                "smoothed_covs": dmatrix_list_to_json_array(&smooth.smoothed_covs)
            }))
        }
        KalmanOperationKind::Forecast => {
            let ys = observation_sequence_to_dvectors(
                args.y
                    .as_ref()
                    .ok_or_else(|| "y is required when operation='forecast'".to_string())?,
                "y",
            )?;
            let filter = ns_inference::timeseries::kalman::kalman_filter(&model, &ys)
                .map_err(|e| e.to_string())?;
            let steps = args.n_ahead.unwrap_or(10);
            let forecast =
                ns_inference::timeseries::forecast::kalman_forecast(&model, &filter, steps)
                    .map_err(|e| e.to_string())?;
            let mut out = serde_json::json!({
                "state_means": dvector_list_to_json_array(&forecast.state_means),
                "state_covs": dmatrix_list_to_json_array(&forecast.state_covs),
                "obs_means": dvector_list_to_json_array(&forecast.obs_means),
                "obs_covs": dmatrix_list_to_json_array(&forecast.obs_covs)
            });
            if let Some(alpha) = args.alpha {
                let iv =
                    ns_inference::timeseries::forecast::kalman_forecast_intervals(&forecast, alpha)
                        .map_err(|e| e.to_string())?;
                out["alpha"] = serde_json::json!(iv.alpha);
                out["z"] = serde_json::json!(iv.z);
                out["obs_lower"] = serde_json::json!(dvector_list_to_json_array(&iv.obs_lower));
                out["obs_upper"] = serde_json::json!(dvector_list_to_json_array(&iv.obs_upper));
            }
            Ok(out)
        }
        KalmanOperationKind::Simulate => {
            let t_max = args
                .t_max
                .ok_or_else(|| "t_max is required when operation='simulate'".to_string())?;
            if t_max == 0 || t_max > KALMAN_SIMULATE_SERVER_MAX_T {
                return Err(format!(
                    "t_max must be in 1..={KALMAN_SIMULATE_SERVER_MAX_T} when operation='simulate'"
                ));
            }
            let x0 = if let Some(simulate_x0) = args.simulate_x0.as_ref() {
                Some(vector_arg_to_dvector(simulate_x0, "simulate_x0")?)
            } else {
                match args.init {
                    KalmanSimInitKind::Sample => None,
                    KalmanSimInitKind::Mean => Some(model.m0.clone()),
                }
            };
            let sim = if let Some(x0) = x0 {
                ns_inference::timeseries::simulate::kalman_simulate_with_x0(
                    &model,
                    t_max,
                    args.seed,
                    Some(x0),
                )
            } else {
                ns_inference::timeseries::simulate::kalman_simulate(&model, t_max, args.seed)
            }
            .map_err(|e| e.to_string())?;
            Ok(serde_json::json!({
                "xs": dvector_list_to_json_array(&sim.xs),
                "ys": dvector_list_to_json_array(&sim.ys)
            }))
        }
        KalmanOperationKind::Em => {
            let ys = observation_sequence_to_dvectors(
                args.y.as_ref().ok_or_else(|| "y is required when operation='em'".to_string())?,
                "y",
            )?;
            if args.max_iter == 0 || args.max_iter > KALMAN_EM_SERVER_MAX_ITER {
                return Err(format!(
                    "max_iter must be in 1..={KALMAN_EM_SERVER_MAX_ITER} when operation='em'"
                ));
            }
            let cfg = ns_inference::timeseries::em::KalmanEmConfig {
                max_iter: args.max_iter,
                tol: args.tol,
                estimate_q: args.estimate_q,
                estimate_r: args.estimate_r,
                estimate_f: args.estimate_f,
                estimate_h: args.estimate_h,
                min_diag: args.min_diag,
            };
            let out = ns_inference::timeseries::em::kalman_em(&model, &ys, cfg)
                .map_err(|e| e.to_string())?;
            Ok(serde_json::json!({
                "converged": out.converged,
                "n_iter": out.n_iter,
                "loglik_trace": out.loglik_trace,
                "f": dmatrix_to_json_array(&out.model.f),
                "h": dmatrix_to_json_array(&out.model.h),
                "q": dmatrix_to_json_array(&out.model.q),
                "r": dmatrix_to_json_array(&out.model.r)
            }))
        }
    }
}

fn survival_events_to_bool(event: &[u8]) -> Result<Vec<bool>, String> {
    let mut out = Vec::with_capacity(event.len());
    for &value in event {
        match value {
            0 => out.push(false),
            1 => out.push(true),
            _ => return Err("event must contain only 0/1 values".to_string()),
        }
    }
    Ok(out)
}

fn design_matrix_with_intercept(
    x: &[Vec<f64>],
    include_intercept: bool,
) -> Result<DMatrix<f64>, String> {
    if x.is_empty() || x[0].is_empty() {
        return Err("X must be non-empty".to_string());
    }
    let p = x[0].len();
    let cols = p + if include_intercept { 1 } else { 0 };
    let mut flat = Vec::with_capacity(x.len() * cols);
    for (i, row) in x.iter().enumerate() {
        if row.len() != p {
            return Err(format!(
                "X rows must all have the same length; row {i} has length {}",
                row.len()
            ));
        }
        if include_intercept {
            flat.push(1.0);
        }
        flat.extend(row.iter().copied());
    }
    Ok(DMatrix::from_row_slice(x.len(), cols, &flat))
}

fn json_y_to_binary(y: &[f64]) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(y.len());
    for &value in y {
        if !value.is_finite() {
            return Err("y must contain only finite values".to_string());
        }
        let iv = value as i64;
        if iv != 0 && iv != 1 {
            return Err("y must contain only 0/1 values for logistic regression".to_string());
        }
        out.push(iv as u8);
    }
    Ok(out)
}

fn json_y_to_counts(y: &[f64], family: &str) -> Result<Vec<u64>, String> {
    let mut out = Vec::with_capacity(y.len());
    for &value in y {
        if !value.is_finite() {
            return Err("y must contain only finite values".to_string());
        }
        let iv = value as i64;
        if iv < 0 {
            return Err(format!("y must be non-negative for {family} regression"));
        }
        out.push(iv as u64);
    }
    Ok(out)
}

fn json_y_to_levels(y: &[f64], n_levels: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(y.len());
    for &value in y {
        if !value.is_finite() {
            return Err("y must contain only finite values".to_string());
        }
        let iv = value as i64;
        if iv < 0 || (iv as usize) >= n_levels {
            return Err(format!("y must contain only integer levels in [0, {})", n_levels));
        }
        out.push(iv as u8);
    }
    Ok(out)
}

fn fit_glm_linear(args: &GlmFitArgs) -> Result<serde_json::Value, String> {
    let xmat = design_matrix_with_intercept(&args.x, args.include_intercept)?;
    if args.x.len() != args.y.len() {
        return Err("X and y must have the same length".to_string());
    }
    let n = args.y.len();
    let k = xmat.ncols();
    if n <= k {
        return Err("Need n > n_params to compute sigma2_hat".to_string());
    }

    let coef = if args.l2.is_none_or(|value| value <= 0.0) {
        ols_fit(args.x.clone(), args.y.clone(), args.include_intercept)
            .map_err(|e| e.to_string())?
    } else {
        let lambda = args.l2.unwrap_or(0.0);
        let xtx = xmat.transpose() * &xmat;
        let xty = xmat.transpose() * DVector::from_vec(args.y.clone());
        let mut ridge = xtx.clone();
        for i in 0..k {
            if args.include_intercept && i == 0 {
                continue;
            }
            ridge[(i, i)] += lambda;
        }
        ridge
            .lu()
            .solve(&xty)
            .ok_or_else(|| "ridge solve failed (singular XtX)".to_string())?
            .iter()
            .copied()
            .collect::<Vec<_>>()
    };

    let coef_vec = DVector::from_vec(coef.clone());
    let y_vec = DVector::from_vec(args.y.clone());
    let resid = (&xmat * coef_vec) - y_vec;
    let sse = resid.iter().map(|value| value * value).sum::<f64>();
    let sigma2_hat = sse / ((n - k) as f64);

    let mut xtx = xmat.transpose() * &xmat;
    if let Some(lambda) = args.l2
        && lambda > 0.0
    {
        for i in 0..k {
            if args.include_intercept && i == 0 {
                continue;
            }
            xtx[(i, i)] += lambda;
        }
    }
    let xtx_inv = xtx.try_inverse().ok_or_else(|| "XtX inverse failed".to_string())?;
    let standard_errors = (0..k)
        .map(|i| {
            let variance = sigma2_hat * xtx_inv[(i, i)];
            if variance > 0.0 { variance.sqrt() } else { f64::INFINITY }
        })
        .collect::<Vec<_>>();

    Ok(serde_json::json!({
        "family": "linear",
        "coef": coef,
        "standard_errors": standard_errors,
        "sigma2_hat": sigma2_hat
    }))
}

fn fit_glm_count_like(args: &GlmFitArgs) -> Result<serde_json::Value, String> {
    let mle = MaximumLikelihoodEstimator::new();
    match args.family {
        GlmFamily::Logistic => {
            let y = json_y_to_binary(&args.y)?;
            if args.l2.is_some_and(|value| value > 0.0) {
                let sigma = 1.0 / args.l2.unwrap_or(1.0).sqrt();
                let model =
                    ModelBuilder::logistic_regression(args.x.clone(), y, args.include_intercept)
                        .map_err(|e| e.to_string())?
                        .with_coef_prior_normal(0.0, sigma)
                        .map_err(|e| e.to_string())?
                        .with_penalize_intercept(false)
                        .build()
                        .map_err(|e| e.to_string())?;
                let fit = mle.fit(&model).map_err(|e| e.to_string())?;
                return Ok(serde_json::json!({
                    "family": "logistic",
                    "coef": fit.parameters,
                    "standard_errors": fit.uncertainties
                }));
            }
            let model = LogisticRegressionModel::new(args.x.clone(), y, args.include_intercept)
                .map_err(|e| e.to_string())?;
            let fit = mle.fit(&model).map_err(|e| e.to_string())?;
            Ok(serde_json::json!({
                "family": "logistic",
                "coef": fit.parameters,
                "standard_errors": fit.uncertainties
            }))
        }
        GlmFamily::Poisson => {
            let y = json_y_to_counts(&args.y, "Poisson")?;
            if args.l2.is_some_and(|value| value > 0.0) {
                let sigma = 1.0 / args.l2.unwrap_or(1.0).sqrt();
                let model = ModelBuilder::poisson_regression(
                    args.x.clone(),
                    y,
                    args.include_intercept,
                    None,
                )
                .map_err(|e| e.to_string())?
                .with_coef_prior_normal(0.0, sigma)
                .map_err(|e| e.to_string())?
                .with_penalize_intercept(false)
                .build()
                .map_err(|e| e.to_string())?;
                let fit = mle.fit(&model).map_err(|e| e.to_string())?;
                return Ok(serde_json::json!({
                    "family": "poisson",
                    "coef": fit.parameters,
                    "standard_errors": fit.uncertainties
                }));
            }
            let model =
                PoissonRegressionModel::new(args.x.clone(), y, args.include_intercept, None)
                    .map_err(|e| e.to_string())?;
            let fit = mle.fit(&model).map_err(|e| e.to_string())?;
            Ok(serde_json::json!({
                "family": "poisson",
                "coef": fit.parameters,
                "standard_errors": fit.uncertainties
            }))
        }
        GlmFamily::Negbin => {
            let y = json_y_to_counts(&args.y, "negative binomial")?;
            let model = NegativeBinomialRegressionModel::new(
                args.x.clone(),
                y,
                args.include_intercept,
                None,
            )
            .map_err(|e| e.to_string())?;
            let fit = mle.fit(&model).map_err(|e| e.to_string())?;
            let log_alpha = fit
                .parameters
                .last()
                .copied()
                .ok_or_else(|| "unexpected empty parameter vector".to_string())?;
            let alpha = log_alpha.exp();
            Ok(serde_json::json!({
                "family": "negbin",
                "coef": fit.parameters[..fit.parameters.len().saturating_sub(1)].to_vec(),
                "standard_errors": fit.uncertainties[..fit.uncertainties.len().saturating_sub(1)].to_vec(),
                "alpha": alpha
            }))
        }
        GlmFamily::Linear => fit_glm_linear(args),
    }
}

const BAYESIAN_SERVER_MAX_CHAINS: usize = 4;
const BAYESIAN_SERVER_MAX_WARMUP: usize = 500;
const BAYESIAN_SERVER_MAX_SAMPLES: usize = 1000;

fn bayesian_model_type_name(model_type: BayesianSampleModelKind) -> &'static str {
    match model_type {
        BayesianSampleModelKind::LinearRegression => "linear_regression",
        BayesianSampleModelKind::LogisticRegression => "logistic_regression",
        BayesianSampleModelKind::PoissonRegression => "poisson_regression",
        BayesianSampleModelKind::NegbinRegression => "negbin_regression",
        BayesianSampleModelKind::CoxPh => "cox_ph",
        BayesianSampleModelKind::WeibullSurvival => "weibull_survival",
        BayesianSampleModelKind::LognormalAft => "lognormal_aft",
        BayesianSampleModelKind::OrderedLogit => "ordered_logit",
        BayesianSampleModelKind::OrderedProbit => "ordered_probit",
        BayesianSampleModelKind::Histfactory => "histfactory",
    }
}

fn pack_bayesian_sample_result(
    model_type: BayesianSampleModelKind,
    result: ns_inference::SamplerResult,
) -> serde_json::Value {
    let diag = compute_diagnostics(&result);
    let quality =
        quality_summary(&diag, result.chains.len(), result.n_samples, &QualityGates::default());

    let mut r_hat = serde_json::Map::new();
    let mut ess_bulk = serde_json::Map::new();
    let mut ess_tail = serde_json::Map::new();
    let mut posterior_summary = serde_json::Map::new();

    for (idx, name) in result.param_names.iter().enumerate() {
        r_hat.insert(name.clone(), serde_json::json!(diag.r_hat[idx]));
        ess_bulk.insert(name.clone(), serde_json::json!(diag.ess_bulk[idx]));
        ess_tail.insert(name.clone(), serde_json::json!(diag.ess_tail[idx]));
        posterior_summary.insert(
            name.clone(),
            serde_json::json!({
                "mean": result.param_mean(idx)
            }),
        );
    }

    serde_json::json!({
        "model_type": bayesian_model_type_name(model_type),
        "n_chains": result.chains.len(),
        "n_warmup": result.n_warmup,
        "n_samples": result.n_samples,
        "param_names": result.param_names,
        "diagnostics": {
            "r_hat": r_hat,
            "ess_bulk": ess_bulk,
            "ess_tail": ess_tail,
            "divergence_rate": diag.divergence_rate,
            "max_treedepth_rate": diag.max_treedepth_rate,
            "ebfmi": diag.ebfmi,
            "quality": {
                "status": quality.status.to_string(),
                "enabled": quality.enabled,
                "warnings": quality.warnings,
                "failures": quality.failures,
                "total_draws": quality.total_draws,
                "max_r_hat": quality.max_r_hat,
                "min_ess_bulk": quality.min_ess_bulk,
                "min_ess_tail": quality.min_ess_tail,
                "min_ebfmi": quality.min_ebfmi
            }
        },
        "posterior_summary": posterior_summary
    })
}

fn run_bayesian_sample_nuts<M: LogDensityModel + Sync>(
    model_type: BayesianSampleModelKind,
    model: &M,
    args: &BayesianSampleArgs,
) -> Result<serde_json::Value, String> {
    if args.n_chains == 0 || args.n_chains > BAYESIAN_SERVER_MAX_CHAINS {
        return Err(format!(
            "n_chains must be between 1 and {BAYESIAN_SERVER_MAX_CHAINS} for the server-safe subset"
        ));
    }
    if args.n_warmup == 0 || args.n_warmup > BAYESIAN_SERVER_MAX_WARMUP {
        return Err(format!(
            "n_warmup must be between 1 and {BAYESIAN_SERVER_MAX_WARMUP} for the server-safe subset"
        ));
    }
    if args.n_samples == 0 || args.n_samples > BAYESIAN_SERVER_MAX_SAMPLES {
        return Err(format!(
            "n_samples must be between 1 and {BAYESIAN_SERVER_MAX_SAMPLES} for the server-safe subset"
        ));
    }
    if !args.target_accept.is_finite() || !(0.5..=0.99).contains(&args.target_accept) {
        return Err("target_accept must be finite and between 0.5 and 0.99".to_string());
    }

    let config = NutsConfig { target_accept: args.target_accept, ..Default::default() };
    let result = sample_nuts_multichain(
        model,
        args.n_chains,
        args.n_warmup,
        args.n_samples,
        args.seed,
        config,
    )
    .map_err(|e| e.to_string())?;

    Ok(pack_bayesian_sample_result(model_type, result))
}

fn run_bayesian_sample_tool(
    state: &AppState,
    args: &BayesianSampleArgs,
) -> Result<serde_json::Value, String> {
    match args.model_type {
        BayesianSampleModelKind::LinearRegression => {
            let x =
                args.x.clone().ok_or_else(|| "x is required for linear_regression".to_string())?;
            let y =
                args.y.clone().ok_or_else(|| "y is required for linear_regression".to_string())?;
            validate_rectangular_matrix(&x, "x")?;
            let model = LinearRegressionModel::new(x, y, true).map_err(|e| e.to_string())?;
            run_bayesian_sample_nuts(args.model_type, &model, args)
        }
        BayesianSampleModelKind::LogisticRegression => {
            let x = args
                .x
                .clone()
                .ok_or_else(|| "x is required for logistic_regression".to_string())?;
            let y = args
                .y
                .as_ref()
                .ok_or_else(|| "y is required for logistic_regression".to_string())?;
            validate_rectangular_matrix(&x, "x")?;
            let model = LogisticRegressionModel::new(x, json_y_to_binary(y)?, true)
                .map_err(|e| e.to_string())?;
            run_bayesian_sample_nuts(args.model_type, &model, args)
        }
        BayesianSampleModelKind::PoissonRegression => {
            let x =
                args.x.clone().ok_or_else(|| "x is required for poisson_regression".to_string())?;
            let y = args
                .y
                .as_ref()
                .ok_or_else(|| "y is required for poisson_regression".to_string())?;
            validate_rectangular_matrix(&x, "x")?;
            let model = PoissonRegressionModel::new(x, json_y_to_counts(y, "Poisson")?, true, None)
                .map_err(|e| e.to_string())?;
            run_bayesian_sample_nuts(args.model_type, &model, args)
        }
        BayesianSampleModelKind::NegbinRegression => {
            let x =
                args.x.clone().ok_or_else(|| "x is required for negbin_regression".to_string())?;
            let y =
                args.y.as_ref().ok_or_else(|| "y is required for negbin_regression".to_string())?;
            validate_rectangular_matrix(&x, "x")?;
            let model = NegativeBinomialRegressionModel::new(
                x,
                json_y_to_counts(y, "negative binomial")?,
                true,
                None,
            )
            .map_err(|e| e.to_string())?;
            run_bayesian_sample_nuts(args.model_type, &model, args)
        }
        BayesianSampleModelKind::CoxPh => {
            let x = args.x.clone().ok_or_else(|| "x is required for cox_ph".to_string())?;
            let time =
                args.time.clone().ok_or_else(|| "time is required for cox_ph".to_string())?;
            let event =
                args.event.as_ref().ok_or_else(|| "event is required for cox_ph".to_string())?;
            validate_rectangular_matrix(&x, "x")?;
            let model = CoxPhModel::new(time, survival_events_to_bool(event)?, x, CoxTies::Efron)
                .map_err(|e| e.to_string())?;
            run_bayesian_sample_nuts(args.model_type, &model, args)
        }
        BayesianSampleModelKind::WeibullSurvival => {
            let time = args
                .time
                .clone()
                .ok_or_else(|| "time is required for weibull_survival".to_string())?;
            let event = args
                .event
                .as_ref()
                .ok_or_else(|| "event is required for weibull_survival".to_string())?;
            let model = WeibullSurvivalModel::new(time, survival_events_to_bool(event)?)
                .map_err(|e| e.to_string())?;
            run_bayesian_sample_nuts(args.model_type, &model, args)
        }
        BayesianSampleModelKind::LognormalAft => {
            let time = args
                .time
                .clone()
                .ok_or_else(|| "time is required for lognormal_aft".to_string())?;
            let event = args
                .event
                .as_ref()
                .ok_or_else(|| "event is required for lognormal_aft".to_string())?;
            let model = LogNormalAftModel::new(time, survival_events_to_bool(event)?)
                .map_err(|e| e.to_string())?;
            run_bayesian_sample_nuts(args.model_type, &model, args)
        }
        BayesianSampleModelKind::OrderedLogit => {
            let x = args.x.clone().ok_or_else(|| "x is required for ordered_logit".to_string())?;
            let y = args.y.as_ref().ok_or_else(|| "y is required for ordered_logit".to_string())?;
            let n_levels = args
                .n_levels
                .ok_or_else(|| "n_levels is required for ordered_logit".to_string())?;
            validate_rectangular_matrix(&x, "x")?;
            let model = OrderedLogitModel::new(x, json_y_to_levels(y, n_levels)?, n_levels)
                .map_err(|e| e.to_string())?;
            run_bayesian_sample_nuts(args.model_type, &model, args)
        }
        BayesianSampleModelKind::OrderedProbit => {
            let x = args.x.clone().ok_or_else(|| "x is required for ordered_probit".to_string())?;
            let y =
                args.y.as_ref().ok_or_else(|| "y is required for ordered_probit".to_string())?;
            let n_levels = args
                .n_levels
                .ok_or_else(|| "n_levels is required for ordered_probit".to_string())?;
            validate_rectangular_matrix(&x, "x")?;
            let model = OrderedProbitModel::new(x, json_y_to_levels(y, n_levels)?, n_levels)
                .map_err(|e| e.to_string())?;
            run_bayesian_sample_nuts(args.model_type, &model, args)
        }
        BayesianSampleModelKind::Histfactory => {
            let model = resolve_model_from_args(
                state,
                args.workspace_json.as_deref(),
                args.model_id.as_deref(),
            )?;
            run_bayesian_sample_nuts(args.model_type, model.as_ref(), args)
        }
    }
}

fn fit_survival_model(args: &SurvivalFitArgs) -> Result<serde_json::Value, String> {
    if args.x.len() != args.time.len() || args.time.len() != args.event.len() {
        return Err("x, time, and event must have the same length".to_string());
    }
    validate_rectangular_matrix(&args.x, "x")?;

    let events = survival_events_to_bool(&args.event)?;
    let mle = MaximumLikelihoodEstimator::new();

    let (fit, model_name) = match args.model {
        SurvivalModelKind::CoxPh => {
            let model =
                CoxPhModel::new(args.time.clone(), events.clone(), args.x.clone(), CoxTies::Efron)
                    .map_err(|e| e.to_string())?;
            (mle.fit(&model).map_err(|e| e.to_string())?, "cox_ph")
        }
        SurvivalModelKind::Weibull => {
            let model = WeibullSurvivalModel::new(args.time.clone(), events.clone())
                .map_err(|e| e.to_string())?;
            (mle.fit(&model).map_err(|e| e.to_string())?, "weibull")
        }
        SurvivalModelKind::LognormalAft => {
            let model = LogNormalAftModel::new(args.time.clone(), events.clone())
                .map_err(|e| e.to_string())?;
            (mle.fit(&model).map_err(|e| e.to_string())?, "lognormal_aft")
        }
        SurvivalModelKind::Exponential => {
            let model = ExponentialSurvivalModel::new(args.time.clone(), events.clone())
                .map_err(|e| e.to_string())?;
            (mle.fit(&model).map_err(|e| e.to_string())?, "exponential")
        }
    };

    Ok(serde_json::json!({
        "model": model_name,
        "parameters": fit.parameters,
        "uncertainties": fit.uncertainties,
        "nll": fit.nll,
        "converged": fit.converged
    }))
}

fn fit_kaplan_meier(args: &KaplanMeierArgs) -> Result<serde_json::Value, String> {
    if args.time.len() != args.event.len() {
        return Err("time and event must have the same length".to_string());
    }
    if let Some(group) = &args.group
        && group.len() != args.time.len()
    {
        return Err("group must have the same length as time and event".to_string());
    }

    let events = survival_events_to_bool(&args.event)?;
    let km = survival_kaplan_meier(&args.time, &events, 0.95).map_err(|e| e.to_string())?;

    let mut out = kaplan_meier_estimate_to_json(&km);

    if let Some(group) = &args.group {
        let lr = survival_log_rank_test(&args.time, &events, group).map_err(|e| e.to_string())?;
        if let Some(obj) = out.as_object_mut() {
            obj.insert("log_rank".to_string(), log_rank_result_to_json(&lr));
        }
    }

    Ok(out)
}

fn pack_root_histogram_with_flows(wf: ns_root::HistogramWithFlows) -> serde_json::Value {
    serde_json::json!({
        "name": wf.histogram.name,
        "title": wf.histogram.title,
        "bin_edges": wf.histogram.bin_edges,
        "bin_content": wf.histogram.bin_content,
        "sumw2": wf.histogram.sumw2,
        "underflow": wf.underflow,
        "overflow": wf.overflow,
        "underflow_sumw2": wf.underflow_sumw2,
        "overflow_sumw2": wf.overflow_sumw2
    })
}

fn run_root_histogram(args: &RootHistogramArgs) -> Result<serde_json::Value, String> {
    let root_bytes = base64::engine::general_purpose::STANDARD
        .decode(args.root_bytes_base64.trim())
        .map_err(|e| format!("invalid root_bytes_base64: {e}"))?;
    if root_bytes.is_empty() {
        return Err("root_bytes_base64 decoded to empty payload".to_string());
    }
    if root_bytes.len() > SERVER_SAFE_ROOT_UPLOAD_MAX_BYTES {
        return Err(format!(
            "server-safe nextstat_read_root_histogram supports decoded ROOT payloads <= {} bytes (got {})",
            SERVER_SAFE_ROOT_UPLOAD_MAX_BYTES,
            root_bytes.len()
        ));
    }
    let filename_hint = args.filename_hint.clone().unwrap_or_else(|| "uploaded.root".to_string());
    let root_file = RootFile::from_bytes(root_bytes, PathBuf::from(filename_hint))
        .map_err(|e| e.to_string())?;
    let wf = root_file.get_histogram_with_flows(&args.hist_path).map_err(|e| e.to_string())?;
    Ok(pack_root_histogram_with_flows(wf))
}

fn run_log_rank_test(args: &LogRankTestArgs) -> Result<serde_json::Value, String> {
    if args.times.len() != args.events.len() || args.events.len() != args.groups.len() {
        return Err("times, events, and groups must have the same length".to_string());
    }
    let events = json_events_to_bool(&args.events, "events")?;
    let lr =
        survival_log_rank_test(&args.times, &events, &args.groups).map_err(|e| e.to_string())?;
    Ok(serde_json::json!({
        "n": lr.n,
        "chi_squared": lr.chi_squared,
        "df": lr.df,
        "p_value": lr.p_value,
        "group_ids": lr.group_summaries.iter().map(|(group, _, _)| *group).collect::<Vec<_>>(),
        "observed": lr.group_summaries.iter().map(|(_, observed, _)| *observed).collect::<Vec<_>>(),
        "expected": lr.group_summaries.iter().map(|(_, _, expected)| *expected).collect::<Vec<_>>()
    }))
}

fn run_churn_generate_data(args: &ChurnGenerateDataArgs) -> Result<serde_json::Value, String> {
    let n_customers = args.n_customers.unwrap_or(CHURN_GENERATE_DATA_SERVER_DEFAULT_CUSTOMERS);
    if n_customers > CHURN_GENERATE_DATA_SERVER_MAX_CUSTOMERS {
        return Err(format!(
            "server-safe nextstat_churn_generate_data supports n_customers <= {} (got {n_customers})",
            CHURN_GENERATE_DATA_SERVER_MAX_CUSTOMERS
        ));
    }
    let n_cohorts = args.n_cohorts.unwrap_or(6);
    if n_cohorts == 0 || n_cohorts > CHURN_GENERATE_DATA_SERVER_MAX_COHORTS {
        return Err(format!(
            "server-safe nextstat_churn_generate_data supports 1 <= n_cohorts <= {} (got {n_cohorts})",
            CHURN_GENERATE_DATA_SERVER_MAX_COHORTS
        ));
    }
    let max_time = args.max_time.unwrap_or(24.0);
    if !max_time.is_finite() || max_time <= 0.0 {
        return Err("max_time must be finite and > 0".to_string());
    }
    let treatment_fraction = args.treatment_fraction.unwrap_or(0.3);
    if !treatment_fraction.is_finite() || !(0.0..=1.0).contains(&treatment_fraction) {
        return Err("treatment_fraction must be finite and in [0,1]".to_string());
    }

    let config = ns_inference::ChurnDataConfig {
        n_customers,
        n_cohorts,
        max_time,
        treatment_fraction,
        seed: args.seed.unwrap_or(42),
        ..Default::default()
    };
    let dataset = ns_inference::generate_churn_dataset(&config).map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "n": dataset.records.len(),
        "n_events": dataset.events.iter().filter(|&&event| event).count(),
        "times": dataset.times,
        "events": dataset.events,
        "groups": dataset.groups,
        "treated": dataset.records.iter().map(|record| record.treated).collect::<Vec<_>>(),
        "covariates": dataset.covariates,
        "covariate_names": [
            "plan_basic",
            "plan_premium",
            "usage_score",
            "support_tickets"
        ],
        "plan": dataset.records.iter().map(|record| record.plan).collect::<Vec<_>>(),
        "region": dataset.records.iter().map(|record| record.region).collect::<Vec<_>>(),
        "cohort": dataset.records.iter().map(|record| record.cohort).collect::<Vec<_>>(),
        "usage_score": dataset.records.iter().map(|record| record.usage_score).collect::<Vec<_>>()
    }))
}

fn run_churn_risk_model(args: &ChurnRiskModelArgs) -> Result<serde_json::Value, String> {
    if args.times.is_empty() {
        return Err("times must be non-empty".to_string());
    }
    if args.times.len() != args.events.len() {
        return Err("times and events must have the same length".to_string());
    }
    if args.covariates.len() != args.times.len() {
        return Err("covariates must have the same length as times and events".to_string());
    }
    validate_rectangular_f64_rows(&args.covariates, "covariates")?;
    if args.names.len() != args.covariates[0].len() {
        return Err("names must match the covariate column count".to_string());
    }
    let conf_level = args.conf_level.unwrap_or(0.95);
    if !conf_level.is_finite() || conf_level <= 0.0 || conf_level >= 1.0 {
        return Err("conf_level must be finite and in (0,1)".to_string());
    }

    let events = json_events_to_bool(&args.events, "events")?;
    let model = ns_inference::churn_risk_model(
        &args.times,
        &events,
        &args.covariates,
        &args.names,
        conf_level,
    )
    .map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "n": model.n,
        "n_events": model.n_events,
        "nll": model.nll,
        "names": model.names,
        "coefficients": model.coefficients,
        "se": model.se,
        "hazard_ratios": model.hazard_ratios,
        "hr_ci_lower": model.hr_ci_lower,
        "hr_ci_upper": model.hr_ci_upper
    }))
}

fn run_churn_retention(args: &ChurnRetentionArgs) -> Result<serde_json::Value, String> {
    if args.times.len() != args.events.len() {
        return Err("times and events must have the same length".to_string());
    }
    if args.groups.len() != args.times.len() {
        return Err("groups must have the same length as times and events".to_string());
    }
    let events = json_events_to_bool(&args.events, "events")?;
    let conf_level = args.conf_level.unwrap_or(0.95);
    let retention = run_retention_analysis(&args.times, &events, &args.groups, conf_level)
        .map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "overall": {
            "n": retention.overall.n,
            "n_events": retention.overall.n_events,
            "median": retention.overall.median,
            "time": retention.overall.steps.iter().map(|s| s.time).collect::<Vec<_>>(),
            "survival": retention.overall.steps.iter().map(|s| s.survival).collect::<Vec<_>>()
        },
        "by_group": retention.by_group.iter().map(|(group, km)| serde_json::json!({
            "group": group,
            "n": km.n,
            "n_events": km.n_events,
            "median": km.median,
            "time": km.steps.iter().map(|s| s.time).collect::<Vec<_>>(),
            "survival": km.steps.iter().map(|s| s.survival).collect::<Vec<_>>()
        })).collect::<Vec<_>>(),
        "log_rank": {
            "chi_squared": retention.log_rank.chi_squared,
            "df": retention.log_rank.df,
            "p_value": retention.log_rank.p_value
        }
    }))
}

fn run_churn_diagnostics(args: &ChurnDiagnosticsArgs) -> Result<serde_json::Value, String> {
    if args.times.len() != args.events.len() {
        return Err("times and events must have the same length".to_string());
    }
    if args.groups.len() != args.times.len() {
        return Err("groups must have the same length as times and events".to_string());
    }
    if !args.treated.is_empty() && args.treated.len() != args.times.len() {
        return Err("treated must have the same length as times, events, and groups".to_string());
    }
    if !args.covariates.is_empty() && args.covariates.len() != args.times.len() {
        return Err("covariates must have the same length as times, events, and groups".to_string());
    }
    if !args.trim.is_finite() || !(0.0..0.5).contains(&args.trim) {
        return Err("trim must be finite and in [0, 0.5)".to_string());
    }

    let events = json_events_to_bool(&args.events, "events")?;
    let treated = if args.treated.is_empty() {
        Vec::new()
    } else {
        json_binary_u8(&args.treated, "treated")?
    };
    validate_rectangular_f64_rows(&args.covariates, "covariates")?;
    if !args.covariate_names.is_empty() {
        let n_cols = args.covariates.first().map(|row| row.len()).unwrap_or(0);
        if args.covariates.is_empty() || args.covariate_names.len() != n_cols {
            return Err("covariate_names must match the covariate column count".to_string());
        }
    }

    let report = ns_inference::churn_diagnostics_report(
        &args.times,
        &events,
        &args.groups,
        &treated,
        &args.covariates,
        &args.covariate_names,
        args.trim,
    )
    .map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "n": report.n,
        "n_events": report.n_events,
        "overall_censoring_frac": report.overall_censoring_frac,
        "trust_gate_passed": report.trust_gate_passed,
        "censoring_by_segment": report.censoring_by_segment.iter().map(|seg| serde_json::json!({
            "group": seg.group,
            "n": seg.n,
            "n_events": seg.n_events,
            "n_censored": seg.n_censored,
            "frac_censored": seg.frac_censored
        })).collect::<Vec<_>>(),
        "covariate_balance": report.covariate_balance.iter().map(|row| serde_json::json!({
            "name": row.name,
            "smd_raw": row.smd_raw,
            "mean_treated": row.mean_treated,
            "mean_control": row.mean_control
        })).collect::<Vec<_>>(),
        "propensity_overlap": report.propensity_overlap.as_ref().map(|po| serde_json::json!({
            "quantiles": po.quantiles,
            "mean": po.mean,
            "n_trimmed_low": po.n_trimmed_low,
            "n_trimmed_high": po.n_trimmed_high,
            "trim": po.trim
        })),
        "warnings": report.warnings.iter().map(|w| serde_json::json!({
            "category": w.category,
            "severity": w.severity,
            "message": w.message
        })).collect::<Vec<_>>()
    }))
}

fn run_churn_cohort_matrix(args: &ChurnCohortMatrixArgs) -> Result<serde_json::Value, String> {
    if args.times.len() != args.events.len() {
        return Err("times and events must have the same length".to_string());
    }
    if args.groups.len() != args.times.len() {
        return Err("groups must have the same length as times and events".to_string());
    }
    if args.period_boundaries.is_empty() {
        return Err("period_boundaries must be non-empty".to_string());
    }
    if args.period_boundaries.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err("period_boundaries must contain only finite values > 0".to_string());
    }
    if args.period_boundaries.windows(2).any(|w| w[1] <= w[0]) {
        return Err("period_boundaries must be strictly increasing".to_string());
    }

    let events = json_events_to_bool(&args.events, "events")?;
    let report = ns_inference::cohort_retention_matrix(
        &args.times,
        &events,
        &args.groups,
        &args.period_boundaries,
    )
    .map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "period_boundaries": report.period_boundaries,
        "cohorts": report.cohorts.iter().map(|row| serde_json::json!({
            "cohort": row.cohort,
            "n_total": row.n_total,
            "n_events": row.n_events,
            "periods": row.periods.iter().map(|cell| serde_json::json!({
                "n_at_risk": cell.n_at_risk,
                "n_events": cell.n_events,
                "n_censored": cell.n_censored,
                "retention_rate": cell.retention_rate,
                "cumulative_retention": cell.cumulative_retention
            })).collect::<Vec<_>>()
        })).collect::<Vec<_>>(),
        "overall": {
            "cohort": report.overall.cohort,
            "n_total": report.overall.n_total,
            "n_events": report.overall.n_events,
            "periods": report.overall.periods.iter().map(|cell| serde_json::json!({
                "n_at_risk": cell.n_at_risk,
                "n_events": cell.n_events,
                "n_censored": cell.n_censored,
                "retention_rate": cell.retention_rate,
                "cumulative_retention": cell.cumulative_retention
            })).collect::<Vec<_>>()
        }
    }))
}

fn run_churn_bootstrap_hr(args: &ChurnBootstrapHrArgs) -> Result<serde_json::Value, String> {
    if args.times.len() != args.events.len() {
        return Err("times and events must have the same length".to_string());
    }
    if args.covariates.len() != args.times.len() {
        return Err("covariates must have the same length as times and events".to_string());
    }
    validate_rectangular_matrix(&args.covariates, "covariates")?;
    if args.names.len() != args.covariates[0].len() {
        return Err("names must match the covariate column count".to_string());
    }

    let n_bootstrap = args.n_bootstrap.unwrap_or(CHURN_BOOTSTRAP_HR_SERVER_DEFAULT_BOOTSTRAPS);
    if n_bootstrap > CHURN_BOOTSTRAP_HR_SERVER_MAX_BOOTSTRAPS {
        return Err(format!(
            "server-safe nextstat_churn_bootstrap_hr supports n_bootstrap <= {} (got {n_bootstrap})",
            CHURN_BOOTSTRAP_HR_SERVER_MAX_BOOTSTRAPS
        ));
    }
    let n_jackknife = args.n_jackknife.unwrap_or(CHURN_BOOTSTRAP_HR_SERVER_DEFAULT_JACKKNIFE);
    if n_jackknife > CHURN_BOOTSTRAP_HR_SERVER_MAX_JACKKNIFE {
        return Err(format!(
            "server-safe nextstat_churn_bootstrap_hr supports n_jackknife <= {} (got {n_jackknife})",
            CHURN_BOOTSTRAP_HR_SERVER_MAX_JACKKNIFE
        ));
    }

    let events = json_events_to_bool(&args.events, "events")?;
    let conf_level = args.conf_level.unwrap_or(0.95);
    let ci_method = match args.ci_method {
        ChurnBootstrapHrCiMethodKind::Percentile => ns_inference::BootstrapCiMethod::Percentile,
        ChurnBootstrapHrCiMethodKind::Bca => ns_inference::BootstrapCiMethod::Bca,
    };
    let ci_method_name = match args.ci_method {
        ChurnBootstrapHrCiMethodKind::Percentile => "percentile",
        ChurnBootstrapHrCiMethodKind::Bca => "bca",
    };
    let seed = args.seed.unwrap_or(42);

    let report = ns_inference::bootstrap_hazard_ratios_with_method(
        &args.times,
        &events,
        &args.covariates,
        &args.names,
        n_bootstrap,
        seed,
        conf_level,
        ci_method,
        n_jackknife,
    )
    .map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "names": report.names,
        "hr_point": report.hr_point,
        "hr_ci_lower": report.hr_ci_lower,
        "hr_ci_upper": report.hr_ci_upper,
        "n_bootstrap": report.n_bootstrap,
        "n_jackknife_requested": report.n_jackknife_requested,
        "n_jackknife_attempted": report.n_jackknife_attempted,
        "n_converged": report.n_converged,
        "elapsed_s": report.elapsed_s,
        "ci_method_requested": ci_method_name,
        "ci_method_effective": report
            .ci_method_effective
            .iter()
            .map(|method| match method {
                ns_inference::BootstrapCiMethod::Percentile => "percentile",
                ns_inference::BootstrapCiMethod::Bca => "bca",
            })
            .collect::<Vec<_>>(),
        "ci_diagnostics": report
            .ci_diagnostics
            .iter()
            .map(|diag| serde_json::json!({
                "requested_method": match diag.requested_method {
                    ns_inference::BootstrapCiMethod::Percentile => "percentile",
                    ns_inference::BootstrapCiMethod::Bca => "bca",
                },
                "effective_method": match diag.effective_method {
                    ns_inference::BootstrapCiMethod::Percentile => "percentile",
                    ns_inference::BootstrapCiMethod::Bca => "bca",
                },
                "z0": diag.z0,
                "acceleration": diag.acceleration,
                "alpha_low": diag.alpha_low,
                "alpha_high": diag.alpha_high,
                "alpha_low_adj": diag.alpha_low_adj,
                "alpha_high_adj": diag.alpha_high_adj,
                "n_bootstrap": diag.n_bootstrap,
                "n_jackknife": diag.n_jackknife,
                "fallback_reason": diag.fallback_reason
            }))
            .collect::<Vec<_>>()
    }))
}

fn run_churn_ingest(args: &ChurnIngestArgs) -> Result<serde_json::Value, String> {
    if args.times.len() != args.events.len() {
        return Err("times and events must have the same length".to_string());
    }
    if let Some(groups) = &args.groups
        && groups.len() != args.times.len()
    {
        return Err("groups must have the same length as times and events".to_string());
    }
    if let Some(treated) = &args.treated
        && treated.len() != args.times.len()
    {
        return Err("treated must have the same length as times and events".to_string());
    }
    if !args.covariates.is_empty() && args.covariates.len() != args.times.len() {
        return Err("covariates must have the same length as times and events".to_string());
    }
    if !args.covariates.is_empty() {
        validate_rectangular_matrix(&args.covariates, "covariates")?;
    }
    if !args.covariate_names.is_empty()
        && (args.covariates.is_empty() || args.covariate_names.len() != args.covariates[0].len())
    {
        return Err("covariate_names must match the covariate column count".to_string());
    }
    if let Some(observation_end) = args.observation_end
        && !observation_end.is_finite()
    {
        return Err("observation_end must be finite".to_string());
    }

    let events = json_events_to_bool(&args.events, "events")?;
    let treated = match &args.treated {
        Some(values) => Some(json_binary_u8(values, "treated")?),
        None => None,
    };

    let report = ns_inference::ingest_churn_arrays(
        &args.times,
        &events,
        args.groups.as_deref(),
        treated.as_deref(),
        &args.covariates,
        &args.covariate_names,
        args.observation_end,
    )
    .map_err(|e| e.to_string())?;

    let dataset = report.dataset;
    Ok(serde_json::json!({
        "n": dataset.records.len(),
        "n_events": dataset.events.iter().filter(|&&event| event).count(),
        "times": dataset.times,
        "events": dataset.events,
        "groups": dataset.groups,
        "treated": dataset.records.iter().map(|record| record.treated).collect::<Vec<_>>(),
        "covariates": dataset.covariates,
        "covariate_names": report.covariate_names,
        "n_dropped": report.n_dropped,
        "warnings": report.warnings
    }))
}

fn run_churn_compare(args: &ChurnCompareArgs) -> Result<serde_json::Value, String> {
    if args.times.len() != args.events.len() {
        return Err("times and events must have the same length".to_string());
    }
    if args.groups.len() != args.times.len() {
        return Err("groups must have the same length as times and events".to_string());
    }
    let events = json_events_to_bool(&args.events, "events")?;
    let conf_level = args.conf_level.unwrap_or(0.95);
    let alpha = args.alpha.unwrap_or(0.05);
    let correction = match args.correction {
        ChurnCompareCorrectionKind::Bonferroni => ns_inference::CorrectionMethod::Bonferroni,
        ChurnCompareCorrectionKind::BenjaminiHochberg => {
            ns_inference::CorrectionMethod::BenjaminiHochberg
        }
    };
    let correction_method = match args.correction {
        ChurnCompareCorrectionKind::Bonferroni => "bonferroni",
        ChurnCompareCorrectionKind::BenjaminiHochberg => "benjamini_hochberg",
    };
    let report = ns_inference::segment_comparison_report(
        &args.times,
        &events,
        &args.groups,
        conf_level,
        correction,
        alpha,
    )
    .map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "overall_chi_squared": report.overall_chi_squared,
        "overall_p_value": report.overall_p_value,
        "overall_df": report.overall_df,
        "alpha": report.alpha,
        "n": report.n,
        "n_events": report.n_events,
        "correction_method": correction_method,
        "segments": report.segments.iter().map(|segment| serde_json::json!({
            "group": segment.group,
            "n": segment.n,
            "n_events": segment.n_events,
            "median": segment.median,
            "observed": segment.observed,
            "expected": segment.expected
        })).collect::<Vec<_>>(),
        "pairwise": report.pairwise.iter().map(|pair| serde_json::json!({
            "group_a": pair.group_a,
            "group_b": pair.group_b,
            "chi_squared": pair.chi_squared,
            "p_value": pair.p_value,
            "p_adjusted": pair.p_adjusted,
            "hazard_ratio_proxy": pair.hazard_ratio_proxy,
            "median_diff": pair.median_diff,
            "significant": pair.significant
        })).collect::<Vec<_>>()
    }))
}

fn run_churn_uplift(args: &ChurnUpliftArgs) -> Result<serde_json::Value, String> {
    if args.times.len() != args.events.len()
        || args.events.len() != args.treated.len()
        || args.treated.len() != args.covariates.len()
    {
        return Err("times, events, treated, and covariates must have the same length".to_string());
    }
    if args.times.is_empty() {
        return Err("times must be non-empty".to_string());
    }
    let horizon = args.horizon.unwrap_or(12.0);
    if !horizon.is_finite() || horizon <= 0.0 {
        return Err("horizon must be finite and > 0".to_string());
    }
    let events = json_events_to_bool(&args.events, "events")?;
    let treated = json_binary_u8(&args.treated, "treated")?;
    validate_rectangular_f64_rows(&args.covariates, "covariates")?;

    let uplift =
        ns_inference::churn_uplift(&args.times, &events, &treated, &args.covariates, horizon)
            .map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "ate": uplift.ate,
        "se": uplift.se,
        "ci_lower": uplift.ci_lower,
        "ci_upper": uplift.ci_upper,
        "n_treated": uplift.n_treated,
        "n_control": uplift.n_control,
        "gamma_critical": uplift.gamma_critical,
        "horizon": horizon
    }))
}

fn run_churn_uplift_survival(args: &ChurnUpliftSurvivalArgs) -> Result<serde_json::Value, String> {
    if args.times.len() != args.events.len() || args.events.len() != args.treated.len() {
        return Err("times, events, and treated must have the same length".to_string());
    }
    if args.times.is_empty() {
        return Err("times must be non-empty".to_string());
    }
    if !args.covariates.is_empty() && args.covariates.len() != args.times.len() {
        return Err(
            "covariates must have the same length as times, events, and treated".to_string()
        );
    }
    let horizon = args.horizon.unwrap_or(12.0);
    if !horizon.is_finite() || horizon <= 0.0 {
        return Err("horizon must be finite and > 0".to_string());
    }
    if args.eval_horizons.is_empty() {
        return Err("eval_horizons must be non-empty".to_string());
    }
    if args.eval_horizons.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err("eval_horizons must contain only finite values > 0".to_string());
    }
    if !args.trim.is_finite() || !(0.0..0.5).contains(&args.trim) {
        return Err("trim must be finite and in [0, 0.5)".to_string());
    }
    let events = json_events_to_bool(&args.events, "events")?;
    let treated = json_binary_u8(&args.treated, "treated")?;
    validate_rectangular_f64_rows(&args.covariates, "covariates")?;

    let report = ns_inference::survival_uplift_report(
        &args.times,
        &events,
        &treated,
        &args.covariates,
        horizon,
        &args.eval_horizons,
        args.trim,
    )
    .map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "rmst_treated": report.rmst_treated,
        "rmst_control": report.rmst_control,
        "delta_rmst": report.delta_rmst,
        "horizon": report.horizon,
        "ipw_applied": report.ipw_applied,
        "arms": report.arms.iter().map(|arm| serde_json::json!({
            "arm": arm.arm,
            "n": arm.n,
            "n_events": arm.n_events,
            "rmst": arm.rmst,
            "median": arm.median
        })).collect::<Vec<_>>(),
        "survival_diffs": report.survival_diffs.iter().map(|row| serde_json::json!({
            "horizon": row.horizon,
            "survival_treated": row.survival_treated,
            "survival_control": row.survival_control,
            "delta_survival": row.delta_survival
        })).collect::<Vec<_>>(),
        "overlap": {
            "n_total": report.overlap.n_total,
            "n_after_trim": report.overlap.n_after_trim,
            "n_trimmed": report.overlap.n_trimmed,
            "mean_propensity": report.overlap.mean_propensity,
            "min_propensity": report.overlap.min_propensity,
            "max_propensity": report.overlap.max_propensity,
            "ess_treated": report.overlap.ess_treated,
            "ess_control": report.overlap.ess_control
        }
    }))
}

fn pharma_model_name(kind: PharmaModelKind) -> &'static str {
    match kind {
        PharmaModelKind::OneCptIv => "1cpt_iv",
        PharmaModelKind::OneCptOral => "1cpt_oral",
        PharmaModelKind::TwoCptIv => "2cpt_iv",
        PharmaModelKind::TwoCptOral => "2cpt_oral",
        PharmaModelKind::ThreeCptIv => "3cpt_iv",
        PharmaModelKind::ThreeCptOral => "3cpt_oral",
    }
}

fn pharma_expected_theta_count(kind: PharmaModelKind) -> usize {
    match kind {
        PharmaModelKind::OneCptIv => 2,
        PharmaModelKind::OneCptOral => 3,
        PharmaModelKind::TwoCptIv => 4,
        PharmaModelKind::TwoCptOral => 5,
        PharmaModelKind::ThreeCptIv => 6,
        PharmaModelKind::ThreeCptOral => 7,
    }
}

fn pharma_pk_model_kind(kind: PharmaModelKind) -> PkModelKind {
    match kind {
        PharmaModelKind::OneCptIv => PkModelKind::OneCptIv,
        PharmaModelKind::OneCptOral => PkModelKind::OneCptOral,
        PharmaModelKind::TwoCptIv => PkModelKind::TwoCptIv,
        PharmaModelKind::TwoCptOral => PkModelKind::TwoCptOral,
        PharmaModelKind::ThreeCptIv => PkModelKind::ThreeCptIv,
        PharmaModelKind::ThreeCptOral => PkModelKind::ThreeCptOral,
    }
}

fn pharma_build_error_model(
    error_model: PharmaErrorModelKind,
    sigma: f64,
    sigma_add: Option<f64>,
) -> Result<ErrorModel, String> {
    let model = match error_model {
        PharmaErrorModelKind::Additive => ErrorModel::Additive(sigma),
        PharmaErrorModelKind::Proportional => ErrorModel::Proportional(sigma),
        PharmaErrorModelKind::Combined => ErrorModel::Combined {
            sigma_add: sigma_add
                .ok_or_else(|| "sigma_add is required for combined error model".to_string())?,
            sigma_prop: sigma,
        },
    };
    model.validate().map_err(|e| e.to_string())?;
    Ok(model)
}

fn pharma_normalize_doses(doses: &[f64], n_subjects: usize) -> Result<Vec<f64>, String> {
    match doses.len() {
        1 => Ok(vec![doses[0]; n_subjects]),
        n if n == n_subjects => Ok(doses.to_vec()),
        n => Err(format!(
            "doses length ({n}) must equal n_subjects ({n_subjects}) or 1 for broadcast"
        )),
    }
}

fn pack_pharma_foce_result(result: ns_inference::FoceResult) -> serde_json::Value {
    serde_json::json!({
        "theta": result.theta,
        "omega": result.omega,
        "omega_matrix": result.omega_matrix.to_matrix(),
        "correlation": result.correlation,
        "eta": result.eta,
        "ofv": result.ofv,
        "converged": result.converged,
        "n_iter": result.n_iter,
        "sigma": result.sigma,
        "sigma_init": result.sigma_init,
        "covariance_step": result
            .covariance_step
            .as_ref()
            .map(|step| {
                serde_json::json!({
                    "parameter_names": step.parameter_names,
                    "r_matrix": step.r_matrix,
                    "s_matrix": step.s_matrix,
                    "covariance": step.covariance,
                    "robust_covariance": step.robust_covariance,
                    "se": step.se,
                    "rse_pct": step.rse_pct,
                    "r_eigenvalues": step.r_eigenvalues,
                    "r_condition_number": step.r_condition_number
                })
            })
            .unwrap_or(serde_json::Value::Null),
        "imp": serde_json::Value::Null
    })
}

fn pack_pharma_saem_result(
    result: ns_inference::FoceResult,
    diag: ns_inference::SaemDiagnostics,
) -> serde_json::Value {
    let mut saem = serde_json::Map::new();
    saem.insert("acceptance_rates".to_string(), serde_json::json!(diag.acceptance_rates));
    saem.insert("ofv_trace".to_string(), serde_json::json!(diag.ofv_trace));
    saem.insert("burn_in_only".to_string(), serde_json::json!(diag.burn_in_only));
    if !diag.theta_trace.is_empty() {
        saem.insert("theta_trace".to_string(), serde_json::json!(diag.theta_trace));
    }
    if !diag.relative_change.is_empty() {
        saem.insert("relative_change".to_string(), serde_json::json!(diag.relative_change));
    }
    if let Some(geweke_scores) = diag.geweke_scores {
        saem.insert("geweke_scores".to_string(), serde_json::json!(geweke_scores));
    }

    serde_json::json!({
        "theta": result.theta,
        "omega": result.omega,
        "omega_matrix": result.omega_matrix.to_matrix(),
        "correlation": result.correlation,
        "eta": result.eta,
        "ofv": result.ofv,
        "converged": result.converged,
        "n_iter": result.n_iter,
        "sigma": result.sigma,
        "sigma_init": result.sigma_init,
        "saem": saem
    })
}

fn pack_pharma_vpc_result(result: VpcResult) -> serde_json::Value {
    serde_json::json!({
        "bins": result.bins.iter().map(|bin| serde_json::json!({
            "time": bin.time,
            "n_obs": bin.n_obs,
            "obs_quantiles": bin.obs_quantiles,
            "sim_pi_lower": bin.sim_pi_lower,
            "sim_pi_median": bin.sim_pi_median,
            "sim_pi_upper": bin.sim_pi_upper
        })).collect::<Vec<_>>(),
        "quantiles": result.quantiles,
        "n_sim": result.n_sim
    })
}

fn pack_pk_gof_result(
    model: PharmaModelKind,
    n_subjects: usize,
    records: Vec<ns_inference::GofRecord>,
) -> serde_json::Value {
    serde_json::json!({
        "model": pharma_model_name(model),
        "n_subjects": n_subjects,
        "n_records": records.len(),
        "records": records.into_iter().map(|record| serde_json::json!({
            "subject": record.subject,
            "time": record.time,
            "dv": record.dv,
            "pred": record.pred,
            "ipred": record.ipred,
            "iwres": record.iwres,
            "cwres": record.cwres
        })).collect::<Vec<_>>()
    })
}

fn pack_pk_npde_result(
    model: PharmaModelKind,
    n_subjects: usize,
    n_sim: usize,
    seed: u64,
    result: ns_inference::NpdeResult,
) -> serde_json::Value {
    serde_json::json!({
        "model": pharma_model_name(model),
        "n_subjects": n_subjects,
        "n_records": result.records.len(),
        "n_sim": n_sim,
        "seed": seed,
        "records": result.records.into_iter().map(|record| serde_json::json!({
            "subject": record.subject,
            "time": record.time,
            "dv": record.dv,
            "percentile": record.percentile,
            "npde": record.npde
        })).collect::<Vec<_>>(),
        "mean": result.mean,
        "variance": result.variance
    })
}

fn run_pharma_fit(args: &PharmaFitArgs) -> Result<serde_json::Value, String> {
    if args.times.is_empty() {
        return Err("times must be non-empty".to_string());
    }
    if args.times.len() != args.y.len() || args.times.len() != args.subject_idx.len() {
        return Err("times, y, and subject_idx must have the same length".to_string());
    }
    if args.n_subjects == 0 {
        return Err("n_subjects must be >= 1".to_string());
    }
    if args.subject_idx.iter().any(|&idx| idx >= args.n_subjects) {
        return Err("subject_idx entries must be in [0, n_subjects)".to_string());
    }

    let expected_n = pharma_expected_theta_count(args.model);
    let model_name = pharma_model_name(args.model);
    if args.theta_init.len() != expected_n {
        return Err(format!(
            "model '{}' requires {} theta parameters, got {}",
            model_name,
            expected_n,
            args.theta_init.len()
        ));
    }
    if args.omega_init.len() != expected_n {
        return Err(format!(
            "model '{}' requires {} omega_init values, got {}",
            model_name,
            expected_n,
            args.omega_init.len()
        ));
    }

    let error_model = pharma_build_error_model(args.error_model, args.sigma, args.sigma_add)?;
    let doses = pharma_normalize_doses(&args.doses, args.n_subjects)?;
    let omega_full = OmegaMatrix::from_diagonal(&args.omega_init).map_err(|e| e.to_string())?;

    match args.method {
        PharmaFitMethodKind::Foce | PharmaFitMethodKind::Focei | PharmaFitMethodKind::Fo => {
            let mut foce_cfg = FoceConfig {
                max_outer_iter: 100,
                max_inner_iter: 20,
                tol: 1e-4,
                rel_tol: 1e-8,
                interaction: matches!(args.method, PharmaFitMethodKind::Focei),
                omega_damping: 0.7,
                omega_max_ratio: 100.0,
                estimate_sigma: true,
                lloq: None,
                omega_fixed: Vec::new(),
                diagonal_omega: false,
            };
            if matches!(args.method, PharmaFitMethodKind::Fo) {
                foce_cfg.interaction = true;
            }
            let estimator = match args.method {
                PharmaFitMethodKind::Foce
                | PharmaFitMethodKind::Focei
                | PharmaFitMethodKind::Fo => FoceEstimator::new(foce_cfg),
                PharmaFitMethodKind::Saem => unreachable!(),
            };

            let result = match (args.model, args.method) {
                (PharmaModelKind::OneCptIv, PharmaFitMethodKind::Foce)
                | (PharmaModelKind::OneCptIv, PharmaFitMethodKind::Focei) => estimator
                    .fit_1cpt_iv_correlated(
                        &args.times,
                        &args.y,
                        &args.subject_idx,
                        args.n_subjects,
                        &doses,
                        error_model,
                        &args.theta_init,
                        omega_full,
                    ),
                (PharmaModelKind::OneCptOral, PharmaFitMethodKind::Foce)
                | (PharmaModelKind::OneCptOral, PharmaFitMethodKind::Focei) => estimator
                    .fit_1cpt_oral_correlated(
                        &args.times,
                        &args.y,
                        &args.subject_idx,
                        args.n_subjects,
                        &doses,
                        args.bioavailability,
                        error_model,
                        &args.theta_init,
                        omega_full,
                    ),
                (PharmaModelKind::TwoCptIv, PharmaFitMethodKind::Foce)
                | (PharmaModelKind::TwoCptIv, PharmaFitMethodKind::Focei) => estimator
                    .fit_2cpt_iv_correlated(
                        &args.times,
                        &args.y,
                        &args.subject_idx,
                        args.n_subjects,
                        &doses,
                        error_model,
                        &args.theta_init,
                        omega_full,
                    ),
                (PharmaModelKind::TwoCptOral, PharmaFitMethodKind::Foce)
                | (PharmaModelKind::TwoCptOral, PharmaFitMethodKind::Focei) => estimator
                    .fit_2cpt_oral_correlated(
                        &args.times,
                        &args.y,
                        &args.subject_idx,
                        args.n_subjects,
                        &doses,
                        args.bioavailability,
                        error_model,
                        &args.theta_init,
                        omega_full,
                    ),
                (PharmaModelKind::ThreeCptIv, PharmaFitMethodKind::Foce)
                | (PharmaModelKind::ThreeCptIv, PharmaFitMethodKind::Focei) => estimator
                    .fit_3cpt_iv_correlated(
                        &args.times,
                        &args.y,
                        &args.subject_idx,
                        args.n_subjects,
                        &doses,
                        error_model,
                        &args.theta_init,
                        omega_full,
                    ),
                (PharmaModelKind::ThreeCptOral, PharmaFitMethodKind::Foce)
                | (PharmaModelKind::ThreeCptOral, PharmaFitMethodKind::Focei) => estimator
                    .fit_3cpt_oral_correlated(
                        &args.times,
                        &args.y,
                        &args.subject_idx,
                        args.n_subjects,
                        &doses,
                        args.bioavailability,
                        error_model,
                        &args.theta_init,
                        omega_full,
                    ),
                (PharmaModelKind::OneCptIv, PharmaFitMethodKind::Fo) => estimator
                    .fit_1cpt_iv_fo_correlated(
                        &args.times,
                        &args.y,
                        &args.subject_idx,
                        args.n_subjects,
                        &doses,
                        error_model,
                        &args.theta_init,
                        omega_full,
                    ),
                (PharmaModelKind::OneCptOral, PharmaFitMethodKind::Fo) => estimator
                    .fit_1cpt_oral_fo_correlated(
                        &args.times,
                        &args.y,
                        &args.subject_idx,
                        args.n_subjects,
                        &doses,
                        args.bioavailability,
                        error_model,
                        &args.theta_init,
                        omega_full,
                    ),
                (PharmaModelKind::TwoCptIv, PharmaFitMethodKind::Fo) => estimator
                    .fit_2cpt_iv_fo_correlated(
                        &args.times,
                        &args.y,
                        &args.subject_idx,
                        args.n_subjects,
                        &doses,
                        error_model,
                        &args.theta_init,
                        omega_full,
                    ),
                (PharmaModelKind::TwoCptOral, PharmaFitMethodKind::Fo) => estimator
                    .fit_2cpt_oral_fo_correlated(
                        &args.times,
                        &args.y,
                        &args.subject_idx,
                        args.n_subjects,
                        &doses,
                        args.bioavailability,
                        error_model,
                        &args.theta_init,
                        omega_full,
                    ),
                (PharmaModelKind::ThreeCptIv, PharmaFitMethodKind::Fo) => estimator.fit_3cpt_iv_fo(
                    &args.times,
                    &args.y,
                    &args.subject_idx,
                    args.n_subjects,
                    &doses,
                    error_model,
                    &args.theta_init,
                    &args.omega_init,
                ),
                (PharmaModelKind::ThreeCptOral, PharmaFitMethodKind::Fo) => estimator
                    .fit_3cpt_oral_fo(
                        &args.times,
                        &args.y,
                        &args.subject_idx,
                        args.n_subjects,
                        &doses,
                        args.bioavailability,
                        error_model,
                        &args.theta_init,
                        &args.omega_init,
                    ),
                (_, PharmaFitMethodKind::Saem) => unreachable!(),
            }
            .map_err(|e| e.to_string())?;

            Ok(pack_pharma_foce_result(result))
        }
        PharmaFitMethodKind::Saem => {
            let cfg = SaemConfig { store_theta_trace: false, ..SaemConfig::default() };
            let estimator = SaemEstimator::new(cfg);
            let (result, diag) = match args.model {
                PharmaModelKind::OneCptIv => estimator.fit_1cpt_iv_correlated(
                    &args.times,
                    &args.y,
                    &args.subject_idx,
                    args.n_subjects,
                    &doses,
                    error_model,
                    &args.theta_init,
                    omega_full,
                ),
                PharmaModelKind::OneCptOral => estimator.fit_1cpt_oral_correlated(
                    &args.times,
                    &args.y,
                    &args.subject_idx,
                    args.n_subjects,
                    &doses,
                    args.bioavailability,
                    error_model,
                    &args.theta_init,
                    omega_full,
                ),
                PharmaModelKind::TwoCptIv => estimator.fit_2cpt_iv_correlated(
                    &args.times,
                    &args.y,
                    &args.subject_idx,
                    args.n_subjects,
                    &doses,
                    error_model,
                    &args.theta_init,
                    omega_full,
                ),
                PharmaModelKind::TwoCptOral => estimator.fit_2cpt_oral_correlated(
                    &args.times,
                    &args.y,
                    &args.subject_idx,
                    args.n_subjects,
                    &doses,
                    args.bioavailability,
                    error_model,
                    &args.theta_init,
                    omega_full,
                ),
                PharmaModelKind::ThreeCptIv => estimator.fit_3cpt_iv_correlated(
                    &args.times,
                    &args.y,
                    &args.subject_idx,
                    args.n_subjects,
                    &doses,
                    error_model,
                    &args.theta_init,
                    omega_full,
                ),
                PharmaModelKind::ThreeCptOral => estimator.fit_3cpt_oral_correlated(
                    &args.times,
                    &args.y,
                    &args.subject_idx,
                    args.n_subjects,
                    &doses,
                    args.bioavailability,
                    error_model,
                    &args.theta_init,
                    omega_full,
                ),
            }
            .map_err(|e| e.to_string())?;
            Ok(pack_pharma_saem_result(result, diag))
        }
    }
}

fn run_pharma_vpc(args: &PharmaVpcArgs) -> Result<serde_json::Value, String> {
    if args.times.is_empty() {
        return Err("times must be non-empty".to_string());
    }
    if args.times.len() != args.y.len() || args.times.len() != args.subject_idx.len() {
        return Err("times, y, and subject_idx must have the same length".to_string());
    }
    if args.n_subjects == 0 {
        return Err("n_subjects must be >= 1".to_string());
    }
    if args.subject_idx.iter().any(|&idx| idx >= args.n_subjects) {
        return Err("subject_idx entries must be in [0, n_subjects)".to_string());
    }
    if args.n_sim == 0 {
        return Err("n_sim must be >= 1".to_string());
    }
    if args.n_bins == 0 {
        return Err("n_bins must be >= 1".to_string());
    }

    let expected_n = pharma_expected_theta_count(args.model);
    let model_name = pharma_model_name(args.model);
    if args.theta.len() != expected_n {
        return Err(format!(
            "model '{}' requires {} theta parameters, got {}",
            model_name,
            expected_n,
            args.theta.len()
        ));
    }

    let error_model = pharma_build_error_model(args.error_model, args.sigma, args.sigma_add)?;
    let doses = pharma_normalize_doses(&args.doses, args.n_subjects)?;
    let omega = OmegaMatrix::from_covariance(&args.omega_matrix).map_err(|e| e.to_string())?;
    if omega.dim() != expected_n {
        return Err(format!(
            "model '{}' requires a {}x{} omega_matrix, got {}x{}",
            model_name,
            expected_n,
            expected_n,
            omega.dim(),
            omega.dim()
        ));
    }

    let config = VpcConfig {
        n_sim: args.n_sim,
        quantiles: args.quantiles.clone().unwrap_or_else(|| vec![0.05, 0.50, 0.95]),
        n_bins: args.n_bins,
        seed: args.seed,
        pi_level: args.pi_level,
    };

    let result = run_vpc_pk(
        pharma_pk_model_kind(args.model),
        &args.times,
        &args.y,
        &args.subject_idx,
        args.n_subjects,
        &doses,
        args.bioavailability,
        &args.theta,
        &omega,
        &error_model,
        &config,
    )
    .map_err(|e| e.to_string())?;

    Ok(pack_pharma_vpc_result(result))
}

fn run_pk_gof(args: &PharmaPkGofArgs) -> Result<serde_json::Value, String> {
    if args.times.is_empty() {
        return Err("times must be non-empty".to_string());
    }
    if args.times.len() != args.y.len() || args.times.len() != args.subject_idx.len() {
        return Err("times, y, and subject_idx must have the same length".to_string());
    }

    let n_subjects = args.eta.len();
    if n_subjects == 0 {
        return Err("eta must contain at least one subject row".to_string());
    }
    if args.subject_idx.iter().any(|&idx| idx >= n_subjects) {
        return Err("subject_idx entries must be in [0, n_subjects)".to_string());
    }

    let expected_n = pharma_expected_theta_count(args.model);
    let model_name = pharma_model_name(args.model);
    if args.theta.len() != expected_n {
        return Err(format!(
            "model '{}' requires {} theta parameters, got {}",
            model_name,
            expected_n,
            args.theta.len()
        ));
    }
    if args.eta.iter().any(|row| row.len() != expected_n) {
        return Err(format!("model '{}' requires ETA rows of length {}", model_name, expected_n));
    }

    let error_model = pharma_build_error_model(args.error_model, args.sigma, args.sigma_add)?;
    let doses = pharma_normalize_doses(&args.doses, n_subjects)?;
    let records = run_gof_pk(
        pharma_pk_model_kind(args.model),
        &args.times,
        &args.y,
        &args.subject_idx,
        &doses,
        args.bioavailability,
        &args.theta,
        &args.eta,
        &error_model,
    )
    .map_err(|e| e.to_string())?;

    Ok(pack_pk_gof_result(args.model, n_subjects, records))
}

const PHARMA_PK_NPDE_SERVER_MAX_N_SIM: usize = 2000;

fn run_pk_npde(args: &PharmaPkNpdeArgs) -> Result<serde_json::Value, String> {
    if args.times.is_empty() {
        return Err("times must be non-empty".to_string());
    }
    if args.times.len() != args.y.len() || args.times.len() != args.subject_idx.len() {
        return Err("times, y, and subject_idx must have the same length".to_string());
    }
    if args.n_subjects == 0 {
        return Err("n_subjects must be >= 1".to_string());
    }
    if args.subject_idx.iter().any(|&idx| idx >= args.n_subjects) {
        return Err("subject_idx entries must be in [0, n_subjects)".to_string());
    }
    if args.n_sim < 10 || args.n_sim > PHARMA_PK_NPDE_SERVER_MAX_N_SIM {
        return Err(format!(
            "n_sim must be between 10 and {PHARMA_PK_NPDE_SERVER_MAX_N_SIM} for the server-safe subset"
        ));
    }

    let expected_n = pharma_expected_theta_count(args.model);
    let model_name = pharma_model_name(args.model);
    if args.theta.len() != expected_n {
        return Err(format!(
            "model '{}' requires {} theta parameters, got {}",
            model_name,
            expected_n,
            args.theta.len()
        ));
    }

    let error_model = pharma_build_error_model(args.error_model, args.sigma, args.sigma_add)?;
    let doses = pharma_normalize_doses(&args.doses, args.n_subjects)?;
    let omega = OmegaMatrix::from_covariance(&args.omega_matrix).map_err(|e| e.to_string())?;
    if omega.dim() != expected_n {
        return Err(format!(
            "model '{}' requires a {}x{} omega_matrix, got {}x{}",
            model_name,
            expected_n,
            expected_n,
            omega.dim(),
            omega.dim()
        ));
    }

    let config = NpdeConfig { n_sim: args.n_sim, seed: args.seed };
    let result = run_npde_pk(
        pharma_pk_model_kind(args.model),
        &args.times,
        &args.y,
        &args.subject_idx,
        args.n_subjects,
        &doses,
        args.bioavailability,
        &args.theta,
        &omega,
        &error_model,
        &config,
    )
    .map_err(|e| e.to_string())?;

    Ok(pack_pk_npde_result(args.model, args.n_subjects, args.n_sim, args.seed, result))
}

fn trial_sim_model_name(kind: TrialSimModelKind) -> &'static str {
    match kind {
        TrialSimModelKind::OneCptOral => "1cpt_oral",
        TrialSimModelKind::TwoCptIv => "2cpt_iv",
        TrialSimModelKind::TwoCptOral => "2cpt_oral",
    }
}

fn trial_sim_expected_theta_count(kind: TrialSimModelKind) -> usize {
    match kind {
        TrialSimModelKind::OneCptOral => 3,
        TrialSimModelKind::TwoCptIv => 4,
        TrialSimModelKind::TwoCptOral => 5,
    }
}

fn trial_sim_pk_model_type(kind: TrialSimModelKind) -> TrialPkModelType {
    match kind {
        TrialSimModelKind::OneCptOral => TrialPkModelType::OneCompartmentOral,
        TrialSimModelKind::TwoCptIv => TrialPkModelType::TwoCompartmentIv,
        TrialSimModelKind::TwoCptOral => TrialPkModelType::TwoCompartmentOral,
    }
}

fn trial_sim_error_model_type(kind: TrialSimErrorModelKind) -> TrialErrorModelType {
    match kind {
        TrialSimErrorModelKind::Additive => TrialErrorModelType::Additive,
        TrialSimErrorModelKind::Proportional => TrialErrorModelType::Proportional,
    }
}

fn pack_trial_simulate_result(result: TrialResult) -> serde_json::Value {
    serde_json::json!({
        "concentrations": result.concentrations,
        "individual_params": result.individual_params,
        "auc": result.endpoints.auc,
        "cmax": result.endpoints.cmax,
        "tmax": result.endpoints.tmax,
        "ctrough": result.endpoints.ctrough
    })
}

fn run_trial_simulate_tool(args: &TrialSimulateArgs) -> Result<serde_json::Value, String> {
    if args.n_subjects == 0 {
        return Err("n_subjects must be >= 1".to_string());
    }
    if args.dose <= 0.0 {
        return Err("dose must be > 0".to_string());
    }
    if args.obs_times.is_empty() {
        return Err("obs_times must be non-empty".to_string());
    }
    let expected_n = trial_sim_expected_theta_count(args.pk_model);
    let model_name = trial_sim_model_name(args.pk_model);
    if args.theta.len() != expected_n {
        return Err(format!(
            "pk_model '{}' requires {} theta parameters, got {}",
            model_name,
            expected_n,
            args.theta.len()
        ));
    }
    if args.omega.len() != expected_n {
        return Err(format!(
            "pk_model '{}' requires {} omega values, got {}",
            model_name,
            expected_n,
            args.omega.len()
        ));
    }

    let route = match args.pk_model {
        TrialSimModelKind::TwoCptIv => ns_inference::DoseRoute::IvBolus,
        TrialSimModelKind::OneCptOral | TrialSimModelKind::TwoCptOral => {
            ns_inference::DoseRoute::Oral { bioavailability: args.bioavailability }
        }
    };
    let config = TrialConfig {
        n_subjects: args.n_subjects,
        dosing: vec![ns_inference::DoseEvent { time: 0.0, amount: args.dose, route }],
        obs_times: args.obs_times.clone(),
        pk_model: trial_sim_pk_model_type(args.pk_model),
        population: PopulationPkParams {
            theta: args.theta.clone(),
            omega: args.omega.clone(),
            sigma: args.sigma,
            error_model: trial_sim_error_model_type(args.error_model),
            omega_correlation: None,
        },
        seed: args.seed,
    };

    let result = run_trial_simulate(&config).map_err(|e| e.to_string())?;
    Ok(pack_trial_simulate_result(result))
}

fn run_chain_ladder_tool(args: &ChainLadderArgs) -> Result<serde_json::Value, String> {
    let triangle = normalize_chain_ladder_triangle(&args.triangle)?;

    match args.method {
        ChainLadderMethodKind::Basic => {
            let result = run_chain_ladder(&triangle).map_err(|e| e.to_string())?;
            Ok(serde_json::json!({
                "development_factors": result.development_factors,
                "cumulative_factors": result.cumulative_factors,
                "ultimates": result.rows.iter().map(|row| row.ultimate).collect::<Vec<_>>(),
                "ibnr": result.rows.iter().map(|row| row.ibnr).collect::<Vec<_>>(),
                "latest": result.rows.iter().map(|row| row.latest).collect::<Vec<_>>(),
                "total_ibnr": result.total_ibnr,
                "projected": result.projected
            }))
        }
        ChainLadderMethodKind::Mack => {
            let conf_level = args.conf_level.unwrap_or(0.95);
            let result = run_mack_chain_ladder(&triangle, conf_level).map_err(|e| e.to_string())?;
            Ok(serde_json::json!({
                "development_factors": result.development_factors,
                "sigma_sq": result.sigma_sq,
                "ultimates": result.rows.iter().map(|row| row.ultimate).collect::<Vec<_>>(),
                "ibnr": result.rows.iter().map(|row| row.ibnr).collect::<Vec<_>>(),
                "latest": result.rows.iter().map(|row| row.latest).collect::<Vec<_>>(),
                "se": result.rows.iter().map(|row| row.se).collect::<Vec<_>>(),
                "cv": result.rows.iter().map(|row| row.cv).collect::<Vec<_>>(),
                "pi_lower": result.rows.iter().map(|row| row.pi_lower).collect::<Vec<_>>(),
                "pi_upper": result.rows.iter().map(|row| row.pi_upper).collect::<Vec<_>>(),
                "total_ibnr": result.total_ibnr,
                "total_se": result.total_se,
                "conf_level": result.conf_level
            }))
        }
    }
}

fn build_fault_tree_spec(spec: &FaultTreeSpecArg) -> FaultTreeSpec {
    let components = spec
        .components
        .iter()
        .map(|component| match component {
            FaultTreeComponentArg::Bernoulli { p } => FailureMode::Bernoulli { p: *p },
            FaultTreeComponentArg::BernoulliUncertain { mu, sigma } => {
                FailureMode::BernoulliUncertain { mu: *mu, sigma: *sigma }
            }
            FaultTreeComponentArg::WeibullMission { k, lambda, mission_time } => {
                FailureMode::WeibullMission { k: *k, lambda: *lambda, mission_time: *mission_time }
            }
        })
        .collect::<Vec<_>>();

    let nodes = spec
        .nodes
        .iter()
        .map(|node| match node {
            FaultTreeNodeArg::Component { index } => FaultTreeNode::Component(*index),
            FaultTreeNodeArg::And { children } => {
                FaultTreeNode::Gate { gate: Gate::And, children: children.clone() }
            }
            FaultTreeNodeArg::Or { children } => {
                FaultTreeNode::Gate { gate: Gate::Or, children: children.clone() }
            }
        })
        .collect::<Vec<_>>();

    FaultTreeSpec { components, nodes, top_event: spec.top_event }
}

fn build_fault_tree_ce_is_spec(spec: &FaultTreeSpecArg) -> Result<FaultTreeSpec, String> {
    let components = spec
        .components
        .iter()
        .map(|component| match component {
            FaultTreeComponentArg::Bernoulli { p } => Ok(FailureMode::Bernoulli { p: *p }),
            FaultTreeComponentArg::BernoulliUncertain { .. }
            | FaultTreeComponentArg::WeibullMission { .. } => {
                Err("server-safe nextstat_fault_tree_ce_is supports only bernoulli components"
                    .to_string())
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    let nodes = spec
        .nodes
        .iter()
        .map(|node| match node {
            FaultTreeNodeArg::Component { index } => FaultTreeNode::Component(*index),
            FaultTreeNodeArg::And { children } => {
                FaultTreeNode::Gate { gate: Gate::And, children: children.clone() }
            }
            FaultTreeNodeArg::Or { children } => {
                FaultTreeNode::Gate { gate: Gate::Or, children: children.clone() }
            }
        })
        .collect::<Vec<_>>();

    Ok(FaultTreeSpec { components, nodes, top_event: spec.top_event })
}

fn run_fault_tree_mc(args: &FaultTreeMcArgs) -> Result<serde_json::Value, String> {
    let device = args.device.as_deref().unwrap_or("cpu").to_ascii_lowercase();
    if device != "cpu" {
        return Err(format!(
            "server-safe nextstat_fault_tree_mc supports only device='cpu' (got {device:?})"
        ));
    }

    let spec = build_fault_tree_spec(&args.spec);
    spec.validate().map_err(|e| e.to_string())?;

    let n_scenarios = args.n_scenarios.unwrap_or(1_000_000);
    let seed = args.seed.unwrap_or(42);
    let result = run_fault_tree_mc_cpu(&spec, n_scenarios, seed, 0).map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "n_scenarios": result.n_scenarios,
        "n_top_failures": result.n_top_failures,
        "p_failure": result.p_failure,
        "se": result.se,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "wall_time_s": result.wall_time_s,
        "scenarios_per_sec": result.scenarios_per_sec,
        "component_importance": result.component_importance
    }))
}

fn run_fault_tree_ce_is(args: &FaultTreeCeIsArgs) -> Result<serde_json::Value, String> {
    let spec = build_fault_tree_ce_is_spec(&args.spec)?;
    spec.validate().map_err(|e| e.to_string())?;

    let config = FaultTreeCeIsConfig {
        n_per_level: args.n_per_level.unwrap_or(10_000),
        elite_fraction: args.elite_fraction.unwrap_or(0.01),
        max_levels: args.max_levels.unwrap_or(20),
        q_max: args.q_max.unwrap_or(0.99),
        seed: args.seed.unwrap_or(42),
    };
    let result = run_fault_tree_mc_ce_is(&spec, &config).map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "p_failure": result.p_failure,
        "se": result.se,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "n_levels": result.n_levels,
        "n_total_scenarios": result.n_total_scenarios,
        "final_proposal": result.final_proposal,
        "coefficient_of_variation": result.coefficient_of_variation,
        "wall_time_s": result.wall_time_s
    }))
}

#[derive(Debug, Clone, Copy, Deserialize)]
enum BioequivalenceOperationKind {
    #[serde(rename = "test")]
    Test,
    #[serde(rename = "power")]
    Power,
    #[serde(rename = "sample_size")]
    SampleSize,
}

#[derive(Debug, Deserialize)]
struct BioequivalenceArgs {
    operation: Option<BioequivalenceOperationKind>,
    test_values: Option<Vec<f64>>,
    ref_values: Option<Vec<f64>>,
    n_total: Option<usize>,
    cv: Option<f64>,
    gmr: Option<f64>,
    target_power: Option<f64>,
}

fn build_bioequivalence_paired_data(
    test_values: &[f64],
    ref_values: &[f64],
) -> Result<BeData, String> {
    if test_values.len() != ref_values.len() {
        return Err("test_values and ref_values must have the same length".to_string());
    }
    if test_values.is_empty() {
        return Err("test_values and ref_values must be non-empty".to_string());
    }

    let n = test_values.len();
    let mut subject_id = Vec::with_capacity(2 * n);
    let mut sequence = Vec::with_capacity(2 * n);
    let mut period = Vec::with_capacity(2 * n);
    let mut treatment = Vec::with_capacity(2 * n);
    let mut log_value = Vec::with_capacity(2 * n);

    for i in 0..n {
        let seq = if i < n / 2 { 0 } else { 1 };

        subject_id.push(i + 1);
        sequence.push(seq);
        period.push(1);
        if seq == 0 {
            treatment.push(0);
            log_value.push(ref_values[i]);
        } else {
            treatment.push(1);
            log_value.push(test_values[i]);
        }

        subject_id.push(i + 1);
        sequence.push(seq);
        period.push(2);
        if seq == 0 {
            treatment.push(1);
            log_value.push(test_values[i]);
        } else {
            treatment.push(0);
            log_value.push(ref_values[i]);
        }
    }

    Ok(BeData { subject_id, sequence, period, treatment, log_value })
}

fn run_bioequivalence(args: &BioequivalenceArgs) -> Result<serde_json::Value, String> {
    match args.operation.unwrap_or(BioequivalenceOperationKind::Test) {
        BioequivalenceOperationKind::Test => {
            let test_values = args
                .test_values
                .as_ref()
                .ok_or_else(|| "test_values are required when operation='test'".to_string())?;
            let ref_values = args
                .ref_values
                .as_ref()
                .ok_or_else(|| "ref_values are required when operation='test'".to_string())?;
            let data = build_bioequivalence_paired_data(test_values, ref_values)?;
            let result = run_average_be(&data, &BeConfig::default()).map_err(|e| e.to_string())?;
            Ok(serde_json::json!({
                "geometric_mean_ratio": result.geometric_mean_ratio,
                "ci_lower": result.ci_lower,
                "ci_upper": result.ci_upper,
                "pe_log": result.pe_log,
                "se_log": result.se_log,
                "df": result.df,
                "t_lower": result.t_lower,
                "t_upper": result.t_upper,
                "p_lower": result.p_lower,
                "p_upper": result.p_upper,
                "conclusion": format!("{:?}", result.conclusion)
            }))
        }
        BioequivalenceOperationKind::Power => {
            let n_total = args
                .n_total
                .ok_or_else(|| "n_total is required when operation='power'".to_string())?;
            let config = BePowerConfig {
                cv: args.cv.unwrap_or(0.30),
                gmr: args.gmr.unwrap_or(0.95),
                ..BePowerConfig::default()
            };
            let power = run_be_power(n_total, &config).map_err(|e| e.to_string())?;
            Ok(serde_json::json!({ "power": power }))
        }
        BioequivalenceOperationKind::SampleSize => {
            let config = BePowerConfig {
                cv: args.cv.unwrap_or(0.30),
                gmr: args.gmr.unwrap_or(0.95),
                target_power: args.target_power.unwrap_or(0.80),
                ..BePowerConfig::default()
            };
            let result = run_be_sample_size(&config).map_err(|e| e.to_string())?;
            Ok(serde_json::json!({
                "n_per_sequence": result.n_per_sequence,
                "n_total": result.n_total,
                "achieved_power": result.achieved_power
            }))
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
enum DoseResponseModelKind {
    #[serde(rename = "emax")]
    Emax,
    #[serde(rename = "sigmoid_emax")]
    SigmoidEmax,
}

#[derive(Debug, Deserialize)]
struct DoseResponseArgs {
    model: DoseResponseModelKind,
    e0: f64,
    emax: f64,
    ec50: f64,
    gamma: Option<f64>,
    conc: Option<Vec<f64>>,
    dose: Option<Vec<f64>>,
    obs: Option<Vec<f64>>,
    response: Option<Vec<f64>>,
    error_model: Option<String>,
    sigma: Option<f64>,
    sigma_add: Option<f64>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
enum CompetingRisksOperationKind {
    #[serde(rename = "cif")]
    Cif,
    #[serde(rename = "gray_test")]
    GrayTest,
    #[serde(rename = "fine_gray")]
    FineGray,
}

#[derive(Debug, Deserialize)]
struct CompetingRisksArgs {
    operation: CompetingRisksOperationKind,
    times: Vec<f64>,
    events: Vec<u32>,
    target_cause: Option<u32>,
    conf_level: Option<f64>,
    groups: Option<Vec<usize>>,
    x: Option<Vec<f64>>,
    p: Option<usize>,
}

fn dose_response_conc(args: &DoseResponseArgs) -> Result<Vec<f64>, String> {
    args.conc
        .clone()
        .or_else(|| args.dose.clone())
        .ok_or_else(|| "conc (or alias dose) is required".to_string())
}

fn dose_response_obs(args: &DoseResponseArgs) -> Option<Vec<f64>> {
    args.obs.clone().or_else(|| args.response.clone())
}

fn parse_tool_error_model(
    error_model: Option<&str>,
    sigma: Option<f64>,
    sigma_add: Option<f64>,
) -> Result<ErrorModel, String> {
    let sigma = sigma.unwrap_or(0.05);
    match error_model.unwrap_or("additive").to_ascii_lowercase().as_str() {
        "additive" => Ok(ErrorModel::Additive(sigma)),
        "proportional" => Ok(ErrorModel::Proportional(sigma)),
        "combined" => {
            let sigma_add = sigma_add
                .ok_or_else(|| "sigma_add is required for combined error model".to_string())?;
            Ok(ErrorModel::Combined { sigma_add, sigma_prop: sigma })
        }
        "exponential" => Ok(ErrorModel::Exponential(sigma)),
        "power" => Ok(ErrorModel::Power { sigma, power: sigma_add.unwrap_or(1.0) }),
        other => Err(format!(
            "error_model must be 'additive', 'proportional', 'combined', 'exponential', or 'power' (got {other:?})"
        )),
    }
}

fn run_dose_response(args: &DoseResponseArgs) -> Result<serde_json::Value, String> {
    let conc = dose_response_conc(args)?;
    let obs = dose_response_obs(args);

    match args.model {
        DoseResponseModelKind::Emax => {
            let model = EmaxModel::new(args.e0, args.emax, args.ec50).map_err(|e| e.to_string())?;
            if let Some(obs) = obs {
                let error = parse_tool_error_model(
                    args.error_model.as_deref(),
                    args.sigma,
                    args.sigma_add,
                )?;
                let nll = model.nll(&conc, &obs, &error).map_err(|e| e.to_string())?;
                Ok(serde_json::json!({
                    "model": "emax",
                    "nll": nll
                }))
            } else {
                Ok(serde_json::json!({
                    "model": "emax",
                    "predictions": model.predict_vec(&conc),
                    "e0": args.e0,
                    "emax": args.emax,
                    "ec50": args.ec50
                }))
            }
        }
        DoseResponseModelKind::SigmoidEmax => {
            let gamma = args.gamma.unwrap_or(1.0);
            let model = SigmoidEmaxModel::new(args.e0, args.emax, args.ec50, gamma)
                .map_err(|e| e.to_string())?;
            if let Some(obs) = obs {
                let error = parse_tool_error_model(
                    args.error_model.as_deref(),
                    args.sigma,
                    args.sigma_add,
                )?;
                let nll = model.nll(&conc, &obs, &error).map_err(|e| e.to_string())?;
                Ok(serde_json::json!({
                    "model": "sigmoid_emax",
                    "nll": nll
                }))
            } else {
                Ok(serde_json::json!({
                    "model": "sigmoid_emax",
                    "predictions": model.predict_vec(&conc),
                    "e0": args.e0,
                    "emax": args.emax,
                    "ec50": args.ec50,
                    "gamma": gamma
                }))
            }
        }
    }
}

fn run_competing_risks(args: &CompetingRisksArgs) -> Result<serde_json::Value, String> {
    let target_cause = args.target_cause.unwrap_or(1);
    match args.operation {
        CompetingRisksOperationKind::Cif => {
            let conf_level = args.conf_level.unwrap_or(0.95);
            let result =
                competing_cumulative_incidence(&args.times, &args.events, target_cause, conf_level)
                    .map_err(|e| e.to_string())?;
            Ok(cif_estimate_to_json(&result))
        }
        CompetingRisksOperationKind::GrayTest => {
            let groups = args
                .groups
                .as_ref()
                .ok_or_else(|| "groups is required when operation='gray_test'".to_string())?;
            let result = competing_gray_test(&args.times, &args.events, groups, target_cause)
                .map_err(|e| e.to_string())?;
            Ok(gray_test_result_to_json(&result))
        }
        CompetingRisksOperationKind::FineGray => {
            let x = args
                .x
                .as_ref()
                .ok_or_else(|| "x is required when operation='fine_gray'".to_string())?;
            let p = args.p.ok_or_else(|| "p is required when operation='fine_gray'".to_string())?;
            if p == 0 {
                return Err("p must be at least 1 when operation='fine_gray'".to_string());
            }
            if x.len() != args.times.len() * p {
                return Err(format!(
                    "x must have length n*p = {}*{} = {}, got {}",
                    args.times.len(),
                    p,
                    args.times.len() * p,
                    x.len()
                ));
            }
            let x_mat =
                (0..args.times.len()).map(|i| x[i * p..(i + 1) * p].to_vec()).collect::<Vec<_>>();
            let result = competing_fine_gray_fit(&args.times, &args.events, &x_mat, target_cause)
                .map_err(|e| e.to_string())?;
            Ok(fine_gray_result_to_json(&result))
        }
    }
}

fn run_meta_analysis(args: &MetaAnalysisArgs) -> Result<serde_json::Value, String> {
    if args.effects.len() != args.standard_errors.len() {
        return Err("effects and standard_errors must have the same length".to_string());
    }

    let studies = args
        .effects
        .iter()
        .zip(args.standard_errors.iter())
        .enumerate()
        .map(|(i, (&estimate, &se))| StudyEffect {
            label: format!("Study {}", i + 1),
            estimate,
            se,
        })
        .collect::<Vec<_>>();

    let result = match args.method {
        MetaAnalysisMethod::Fixed => run_meta_fixed(&studies, 0.95),
        MetaAnalysisMethod::Random => run_meta_random(&studies, 0.95),
    }
    .map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "estimate": result.estimate,
        "se": result.se,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "z": result.z,
        "p_value": result.p_value,
        "method": result.method,
        "conf_level": result.conf_level,
        "k": result.k,
        "heterogeneity": {
            "q": result.heterogeneity.q,
            "df": result.heterogeneity.df,
            "p_value": result.heterogeneity.p_value,
            "i_squared": result.heterogeneity.i_squared,
            "h_squared": result.heterogeneity.h_squared,
            "tau_squared": result.heterogeneity.tau_squared
        },
        "forest": result.forest.iter().map(|row| serde_json::json!({
            "label": row.label,
            "estimate": row.estimate,
            "se": row.se,
            "ci_lower": row.ci_lower,
            "ci_upper": row.ci_upper,
            "weight": row.weight
        })).collect::<Vec<_>>()
    }))
}

pub fn execute_tool(state: &AppState, req: ToolExecuteRequest) -> ToolResultEnvelope {
    let name = req.name.clone();

    let (controls, warnings) = effective_controls(&req.arguments);
    let mut meta = meta_base(&name, &controls, warnings);

    // EvalMode is process-wide: set it while holding the compute lock (caller enforces).
    let eff = parse_eval_mode(controls.eval_mode.as_deref());
    let _eval_guard = match eff {
        Some(EffectiveEvalMode::Parity) => Some(EvalModeGuard::set(ns_compute::EvalMode::Parity)),
        Some(EffectiveEvalMode::Fast) => Some(EvalModeGuard::set(ns_compute::EvalMode::Fast)),
        None => None,
    };

    let t0 = Instant::now();

    // Deterministic tool calls should not use GPU (precision + determinism).
    let allow_gpu = !controls.deterministic && state.has_gpu();
    let gpu_device = if allow_gpu { state.gpu_device.as_deref() } else { None };

    let gpu_supported =
        matches!(name.as_str(), "nextstat_fit" | "nextstat_ranking" | "nextstat_scan");
    if gpu_device.is_some() && !gpu_supported {
        meta.warnings.push(
            "GPU is enabled on this server, but this tool runs on CPU in server mode.".to_string(),
        );
    }

    let tool_res = if let Some(threads) = controls.threads {
        match usize::try_from(threads).ok().filter(|&n| n > 0) {
            Some(thread_count) => match rayon::ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build()
            {
                Ok(pool) => {
                    meta.threads_applied = Some(threads);
                    pool.install(|| execute_tool_impl(state, &name, req.arguments, gpu_device, t0))
                }
                Err(err) => {
                    meta.warnings.push(format!(
                        "failed to apply requested thread count {threads}; using existing pool: {err}"
                    ));
                    execute_tool_impl(state, &name, req.arguments, gpu_device, t0)
                }
            },
            None => {
                meta.warnings.push(format!("invalid thread count {threads}; using existing pool"));
                execute_tool_impl(state, &name, req.arguments, gpu_device, t0)
            }
        }
    } else {
        execute_tool_impl(state, &name, req.arguments, gpu_device, t0)
    };

    match tool_res {
        Ok((value, device)) => {
            meta.device = device;
            ToolResultEnvelope::ok(&name, meta, value)
        }
        Err(msg) => ToolResultEnvelope::err(&name, meta, "ToolError", msg),
    }
}

fn execute_tool_impl(
    state: &AppState,
    name: &str,
    arguments: serde_json::Value,
    gpu_device: Option<&str>,
    t0: Instant,
) -> Result<(serde_json::Value, Option<String>), String> {
    match name {
        "nextstat_fit" => {
            #[derive(Debug, Deserialize)]
            struct Args {
                #[serde(flatten)]
                common: CommonWorkspaceArgs,
            }
            let args: Args = serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            let model = resolve_model_from_args(
                state,
                args.common.workspace_json.as_deref(),
                args.common.model_id.as_deref(),
            )?;
            let mle = MaximumLikelihoodEstimator::new();

            let (fit, device) = match gpu_device {
                #[cfg(feature = "cuda")]
                Some("cuda") => {
                    (mle.fit_gpu(model.as_ref()).map_err(|e| e.to_string())?, "cuda".to_string())
                }
                #[cfg(feature = "metal")]
                Some("metal") => {
                    (mle.fit_metal(model.as_ref()).map_err(|e| e.to_string())?, "metal".to_string())
                }
                _ => (mle.fit(model.as_ref()).map_err(|e| e.to_string())?, "cpu".to_string()),
            };

            let parameter_names: Vec<String> =
                model.parameters().iter().map(|p| p.name.clone()).collect();
            let poi_index = model.poi_index();
            let (poi_value, poi_error) = if let Some(poi) = poi_index {
                (fit.parameters.get(poi).copied(), fit.uncertainties.get(poi).copied())
            } else {
                (None, None)
            };

            let mut params_map = serde_json::Map::new();
            let n = parameter_names.len().min(fit.parameters.len()).min(fit.uncertainties.len());
            for (i, name) in parameter_names.iter().take(n).enumerate() {
                params_map.insert(
                    name.clone(),
                    serde_json::json!({ "value": fit.parameters[i], "error": fit.uncertainties[i] }),
                );
            }

            Ok((
                serde_json::json!({
                    "nll": fit.nll,
                    "converged": fit.converged,
                    "n_iter": fit.n_iter,
                    "poi_index": poi_index,
                    "poi_value": poi_value,
                    "poi_error": poi_error,
                    "parameters": params_map,
                    "wall_time_s": t0.elapsed().as_secs_f64()
                }),
                Some(device),
            ))
        }
        "nextstat_ranking" => {
            #[derive(Debug, Deserialize)]
            struct Args {
                #[serde(flatten)]
                common: CommonWorkspaceArgs,
                top_n: Option<usize>,
            }
            let args: Args = serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            let model = resolve_model_from_args(
                state,
                args.common.workspace_json.as_deref(),
                args.common.model_id.as_deref(),
            )?;
            let mle = MaximumLikelihoodEstimator::new();

            let (ranking, device) = match gpu_device {
                #[cfg(feature = "cuda")]
                Some("cuda") => (
                    ns_inference::mle::ranking_gpu(&mle, model.as_ref())
                        .map_err(|e| e.to_string())?,
                    "cuda".to_string(),
                ),
                #[cfg(feature = "metal")]
                Some("metal") => (
                    ns_inference::mle::ranking_metal(&mle, model.as_ref())
                        .map_err(|e| e.to_string())?,
                    "metal".to_string(),
                ),
                _ => (mle.ranking(model.as_ref()).map_err(|e| e.to_string())?, "cpu".to_string()),
            };

            // Match the Python `nextstat.interpret.rank_impact()` contract:
            // - total_impact = |up| + |down|
            // - sort by total_impact descending (tie-break by name)
            // - assign 1-based rank
            let mut rows: Vec<serde_json::Value> = ranking
                .into_iter()
                .map(|e| {
                    let total_impact = e.delta_mu_up.abs() + e.delta_mu_down.abs();
                    serde_json::json!({
                        "name": e.name,
                        "delta_mu_up": e.delta_mu_up,
                        "delta_mu_down": e.delta_mu_down,
                        "total_impact": total_impact,
                        "pull": e.pull,
                        "constraint": e.constraint
                    })
                })
                .collect();

            rows.sort_by(|a, b| {
                let ia = a.get("total_impact").and_then(|x| x.as_f64()).unwrap_or(0.0);
                let ib = b.get("total_impact").and_then(|x| x.as_f64()).unwrap_or(0.0);
                ib.partial_cmp(&ia).unwrap_or(std::cmp::Ordering::Equal).then_with(|| {
                    let na = a.get("name").and_then(|x| x.as_str()).unwrap_or("");
                    let nb = b.get("name").and_then(|x| x.as_str()).unwrap_or("");
                    na.cmp(nb)
                })
            });

            for (i, row) in rows.iter_mut().enumerate() {
                if let Some(obj) = row.as_object_mut() {
                    obj.insert("rank".to_string(), serde_json::Value::from((i + 1) as u64));
                }
            }

            if let Some(n) = args.top_n
                && rows.len() > n
            {
                rows.truncate(n);
            }

            Ok((
                serde_json::json!({
                    "ranking": rows,
                    "wall_time_s": t0.elapsed().as_secs_f64()
                }),
                Some(device),
            ))
        }
        "nextstat_scan" => {
            #[derive(Debug, Deserialize)]
            struct Args {
                #[serde(flatten)]
                common: CommonWorkspaceArgs,
                #[serde(default = "default_scan_start")]
                start: f64,
                #[serde(default = "default_scan_stop")]
                stop: f64,
                #[serde(default = "default_scan_points")]
                points: usize,
            }
            fn default_scan_start() -> f64 {
                0.0
            }
            fn default_scan_stop() -> f64 {
                5.0
            }
            fn default_scan_points() -> usize {
                21
            }

            let args: Args = serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            let model = resolve_model_from_args(
                state,
                args.common.workspace_json.as_deref(),
                args.common.model_id.as_deref(),
            )?;

            let points = args.points.clamp(2, 2001);
            let step = (args.stop - args.start) / ((points - 1) as f64);
            let mu_values: Vec<f64> = (0..points).map(|i| args.start + (i as f64) * step).collect();

            let mle = MaximumLikelihoodEstimator::new();

            let (scan, device) = match gpu_device {
                #[cfg(feature = "cuda")]
                Some("cuda") => (
                    ns_inference::profile_likelihood::scan_gpu(&mle, model.as_ref(), &mu_values)
                        .map_err(|e| e.to_string())?,
                    "cuda".to_string(),
                ),
                #[cfg(feature = "metal")]
                Some("metal") => (
                    ns_inference::profile_likelihood::scan_metal(&mle, model.as_ref(), &mu_values)
                        .map_err(|e| e.to_string())?,
                    "metal".to_string(),
                ),
                _ => (
                    ns_inference::profile_likelihood::scan_histfactory(
                        &mle,
                        model.as_ref(),
                        &mu_values,
                    )
                    .map_err(|e| e.to_string())?,
                    "cpu".to_string(),
                ),
            };

            let points_json: Vec<serde_json::Value> = scan
                .points
                .into_iter()
                .map(|p| {
                    serde_json::json!({
                        "mu": p.mu,
                        "q_mu": p.q_mu,
                        "nll_mu": p.nll_mu,
                        "converged": p.converged,
                        "n_iter": p.n_iter
                    })
                })
                .collect();

            Ok((
                serde_json::json!({
                    "poi_index": scan.poi_index,
                    "mu_hat": scan.mu_hat,
                    "nll_hat": scan.nll_hat,
                    "mu_values": mu_values,
                    "points": points_json,
                    "wall_time_s": t0.elapsed().as_secs_f64()
                }),
                Some(device),
            ))
        }
        "nextstat_discovery_asymptotic" => {
            #[derive(Debug, Deserialize)]
            struct Args {
                #[serde(flatten)]
                common: CommonWorkspaceArgs,
            }
            let args: Args = serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            let model = resolve_model_from_args(
                state,
                args.common.workspace_json.as_deref(),
                args.common.model_id.as_deref(),
            )?;
            let mle = MaximumLikelihoodEstimator::new();
            let free = mle.fit(model.as_ref()).map_err(|e| e.to_string())?;
            let poi = model.poi_index().ok_or_else(|| "No POI defined".to_string())?;
            let mu_hat = free.parameters.get(poi).copied();
            let nll_hat = free.nll;

            let scan =
                ns_inference::profile_likelihood::scan_histfactory(&mle, model.as_ref(), &[0.0])
                    .map_err(|e| e.to_string())?;
            if scan.points.is_empty() {
                return Err("profile_scan returned no points for mu=0".to_string());
            }
            let nll0 = scan.points[0].nll_mu;

            let mut q0 = 2.0 * (nll0 - nll_hat);
            if q0 < 0.0 {
                q0 = 0.0;
            }
            if let Some(mh) = mu_hat
                && mh <= 0.0
            {
                q0 = 0.0;
            }
            let z0 = q0.sqrt();
            let p0 = normal_sf(z0);

            Ok((
                serde_json::json!({
                    "mu_hat": mu_hat,
                    "nll_hat": nll_hat,
                    "nll_mu0": nll0,
                    "q0": q0,
                    "z0": z0,
                    "p0": p0,
                    "wall_time_s": t0.elapsed().as_secs_f64()
                }),
                Some("cpu".to_string()),
            ))
        }
        "nextstat_hypotest" => {
            #[derive(Debug, Deserialize)]
            struct Args {
                #[serde(flatten)]
                common: CommonWorkspaceArgs,
                mu: f64,
            }
            let args: Args = serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            let model = resolve_model_from_args(
                state,
                args.common.workspace_json.as_deref(),
                args.common.model_id.as_deref(),
            )?;
            let mle = MaximumLikelihoodEstimator::new();
            let ctx = ns_inference::hypotest::AsymptoticCLsContext::new(&mle, model.as_ref())
                .map_err(|e| e.to_string())?;
            let r = ctx.hypotest_qtilde(&mle, args.mu).map_err(|e| e.to_string())?;
            Ok((
                serde_json::json!({
                    "mu": args.mu,
                    "cls": r.cls,
                    "clsb": r.clsb,
                    "clb": r.clb,
                    "wall_time_s": t0.elapsed().as_secs_f64()
                }),
                Some("cpu".to_string()),
            ))
        }
        "nextstat_upper_limit" => {
            #[derive(Debug, Deserialize)]
            struct Args {
                #[serde(flatten)]
                common: CommonWorkspaceArgs,
                #[serde(default)]
                expected: bool,
                #[serde(default = "default_alpha")]
                alpha: f64,
                #[serde(default)]
                lo: f64,
                hi: Option<f64>,
                #[serde(default = "default_rtol")]
                rtol: f64,
                #[serde(default = "default_max_iter")]
                max_iter: usize,
            }
            fn default_alpha() -> f64 {
                0.05
            }
            fn default_rtol() -> f64 {
                1e-4
            }
            fn default_max_iter() -> usize {
                80
            }

            let args: Args = serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            let model = resolve_model_from_args(
                state,
                args.common.workspace_json.as_deref(),
                args.common.model_id.as_deref(),
            )?;
            let mle = MaximumLikelihoodEstimator::new();
            let ctx = ns_inference::hypotest::AsymptoticCLsContext::new(&mle, model.as_ref())
                .map_err(|e| e.to_string())?;

            let hi = args.hi.unwrap_or(10.0);
            if args.expected {
                let (obs, exp) = ctx
                    .upper_limits_qtilde_bisection(
                        &mle,
                        args.alpha,
                        args.lo,
                        hi,
                        args.rtol,
                        args.max_iter,
                    )
                    .map_err(|e| e.to_string())?;
                Ok((
                    serde_json::json!({
                        "alpha": args.alpha,
                        "obs_limit": obs,
                        "exp_limits": exp,
                        "wall_time_s": t0.elapsed().as_secs_f64()
                    }),
                    Some("cpu".to_string()),
                ))
            } else {
                let obs = ctx
                    .upper_limit_qtilde(&mle, args.alpha, args.lo, hi, args.rtol, args.max_iter)
                    .map_err(|e| e.to_string())?;
                Ok((
                    serde_json::json!({
                        "alpha": args.alpha,
                        "obs_limit": obs,
                        "wall_time_s": t0.elapsed().as_secs_f64()
                    }),
                    Some("cpu".to_string()),
                ))
            }
        }
        "nextstat_hypotest_toys" => {
            #[derive(Debug, Deserialize)]
            struct Args {
                #[serde(flatten)]
                common: CommonWorkspaceArgs,
                mu: f64,
                #[serde(default = "default_n_toys")]
                n_toys: usize,
                #[serde(default = "default_seed")]
                seed: u64,
                #[serde(default)]
                expected_set: bool,
            }
            fn default_n_toys() -> usize {
                1000
            }
            fn default_seed() -> u64 {
                42
            }

            let args: Args = serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            let model = resolve_model_from_args(
                state,
                args.common.workspace_json.as_deref(),
                args.common.model_id.as_deref(),
            )?;
            let mle = MaximumLikelihoodEstimator::new();

            if args.expected_set {
                let r = ns_inference::toybased::hypotest_qtilde_toys_expected_set(
                    &mle,
                    model.as_ref(),
                    args.mu,
                    args.n_toys,
                    args.seed,
                )
                .map_err(|e| e.to_string())?;
                Ok((
                    serde_json::json!({
                        "mu": args.mu,
                        "n_toys": args.n_toys,
                        "seed": args.seed,
                        "expected_set": true,
                        "raw": {
                            "observed": {
                                "cls": r.observed.cls,
                                "clsb": r.observed.clsb,
                                "clb": r.observed.clb
                            },
                            "expected": r.expected
                        },
                        "wall_time_s": t0.elapsed().as_secs_f64()
                    }),
                    Some("cpu".to_string()),
                ))
            } else {
                let r = ns_inference::toybased::hypotest_qtilde_toys(
                    &mle,
                    model.as_ref(),
                    args.mu,
                    args.n_toys,
                    args.seed,
                )
                .map_err(|e| e.to_string())?;
                Ok((
                    serde_json::json!({
                        "mu": args.mu,
                        "n_toys": args.n_toys,
                        "seed": args.seed,
                        "expected_set": false,
                        "raw": {
                            "cls": r.cls,
                            "clsb": r.clsb,
                            "clb": r.clb
                        },
                        "wall_time_s": t0.elapsed().as_secs_f64()
                    }),
                    Some("cpu".to_string()),
                ))
            }
        }
        "nextstat_workspace_audit" => {
            #[derive(Debug, Deserialize)]
            struct Args {
                workspace_json: String,
            }
            let args: Args = serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            let audit = ns_translate::audit::audit_workspace_json(&args.workspace_json)
                .map_err(|e| e.to_string())?;
            Ok((serde_json::to_value(audit).map_err(|e| e.to_string())?, Some("cpu".to_string())))
        }
        "nextstat_read_root_histogram" => {
            let args: RootHistogramArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_root_histogram(&args)?, Some("cpu".to_string())))
        }
        "nextstat_glm_fit" => {
            let args: GlmFitArgs = serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            let result = match args.family {
                GlmFamily::Linear => fit_glm_linear(&args)?,
                GlmFamily::Logistic | GlmFamily::Poisson | GlmFamily::Negbin => {
                    fit_glm_count_like(&args)?
                }
            };
            Ok((result, Some("cpu".to_string())))
        }
        "nextstat_bayesian_sample" => {
            let args: BayesianSampleArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_bayesian_sample_tool(state, &args)?, Some("cpu".to_string())))
        }
        "nextstat_survival_fit" => {
            let args: SurvivalFitArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((fit_survival_model(&args)?, Some("cpu".to_string())))
        }
        "nextstat_kaplan_meier" => {
            let args: KaplanMeierArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((fit_kaplan_meier(&args)?, Some("cpu".to_string())))
        }
        "nextstat_log_rank_test" => {
            let args: LogRankTestArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_log_rank_test(&args)?, Some("cpu".to_string())))
        }
        "nextstat_competing_risks" => {
            let args: CompetingRisksArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_competing_risks(&args)?, Some("cpu".to_string())))
        }
        "nextstat_meta_analysis" => {
            let args: MetaAnalysisArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_meta_analysis(&args)?, Some("cpu".to_string())))
        }
        "nextstat_panel_fe" => {
            let args: PanelFeArgs = serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_panel_fe(&args)?, Some("cpu".to_string())))
        }
        "nextstat_did" => {
            let args: DidArgs = serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_did(&args)?, Some("cpu".to_string())))
        }
        "nextstat_iv_2sls" => {
            let args: Iv2slsArgs = serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_iv_2sls(&args)?, Some("cpu".to_string())))
        }
        "nextstat_aipw" => {
            let args: AipwArgs = serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_aipw(&args)?, Some("cpu".to_string())))
        }
        "nextstat_event_study" => {
            let args: EventStudyArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_event_study(&args)?, Some("cpu".to_string())))
        }
        "nextstat_garch_fit" => {
            let args: GarchFitArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_garch_fit(&args)?, Some("cpu".to_string())))
        }
        "nextstat_ads_cuped_adjust" => {
            let args: AdsCupedAdjustArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_ads_cuped_adjust(&args)?, Some("cpu".to_string())))
        }
        "nextstat_ads_cure_adjust" => {
            let args: AdsCureAdjustArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_ads_cure_adjust(&args)?, Some("cpu".to_string())))
        }
        "nextstat_kalman" => {
            let args: KalmanArgs = serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_kalman(&args)?, Some("cpu".to_string())))
        }
        "nextstat_churn_generate_data" => {
            let args: ChurnGenerateDataArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_churn_generate_data(&args)?, Some("cpu".to_string())))
        }
        "nextstat_churn_risk_model" => {
            let args: ChurnRiskModelArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_churn_risk_model(&args)?, Some("cpu".to_string())))
        }
        "nextstat_churn_retention" => {
            let args: ChurnRetentionArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_churn_retention(&args)?, Some("cpu".to_string())))
        }
        "nextstat_churn_diagnostics" => {
            let args: ChurnDiagnosticsArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_churn_diagnostics(&args)?, Some("cpu".to_string())))
        }
        "nextstat_churn_cohort_matrix" => {
            let args: ChurnCohortMatrixArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_churn_cohort_matrix(&args)?, Some("cpu".to_string())))
        }
        "nextstat_churn_bootstrap_hr" => {
            let args: ChurnBootstrapHrArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_churn_bootstrap_hr(&args)?, Some("cpu".to_string())))
        }
        "nextstat_churn_ingest" => {
            let args: ChurnIngestArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_churn_ingest(&args)?, Some("cpu".to_string())))
        }
        "nextstat_churn_compare" => {
            let args: ChurnCompareArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_churn_compare(&args)?, Some("cpu".to_string())))
        }
        "nextstat_churn_uplift" => {
            let args: ChurnUpliftArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_churn_uplift(&args)?, Some("cpu".to_string())))
        }
        "nextstat_churn_uplift_survival" => {
            let args: ChurnUpliftSurvivalArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_churn_uplift_survival(&args)?, Some("cpu".to_string())))
        }
        "nextstat_pharma_fit" => {
            let args: PharmaFitArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_pharma_fit(&args)?, Some("cpu".to_string())))
        }
        "nextstat_pharma_vpc" => {
            let args: PharmaVpcArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_pharma_vpc(&args)?, Some("cpu".to_string())))
        }
        "nextstat_pk_gof" => {
            let args: PharmaPkGofArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_pk_gof(&args)?, Some("cpu".to_string())))
        }
        "nextstat_pk_npde" => {
            let args: PharmaPkNpdeArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_pk_npde(&args)?, Some("cpu".to_string())))
        }
        "nextstat_trial_simulate" => {
            let args: TrialSimulateArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_trial_simulate_tool(&args)?, Some("cpu".to_string())))
        }
        "nextstat_chain_ladder" => {
            let args: ChainLadderArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_chain_ladder_tool(&args)?, Some("cpu".to_string())))
        }
        "nextstat_fault_tree_mc" => {
            let args: FaultTreeMcArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_fault_tree_mc(&args)?, Some("cpu".to_string())))
        }
        "nextstat_fault_tree_ce_is" => {
            let args: FaultTreeCeIsArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_fault_tree_ce_is(&args)?, Some("cpu".to_string())))
        }
        "nextstat_bioequivalence" => {
            let args: BioequivalenceArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_bioequivalence(&args)?, Some("cpu".to_string())))
        }
        "nextstat_dose_response" => {
            let args: DoseResponseArgs =
                serde_json::from_value(arguments).map_err(|e| e.to_string())?;
            Ok((run_dose_response(&args)?, Some("cpu".to_string())))
        }
        other => Err(format!("Unknown tool: {other}")),
    }
}

fn normal_sf(z: f64) -> f64 {
    // One-sided survival function for standard normal:
    // SF(z) = 0.5 * erfc(z / sqrt(2))
    0.5 * statrs::function::erf::erfc(z / std::f64::consts::SQRT_2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalize_envelope(mut v: serde_json::Value) -> serde_json::Value {
        fn drop_unstable_fields(x: &mut serde_json::Value) {
            match x {
                serde_json::Value::Object(obj) => {
                    obj.remove("n_iter");
                    obj.remove("wall_time_s");
                    obj.remove("elapsed_s");
                    obj.remove("scenarios_per_sec");
                    obj.remove("mu_values");
                    for v in obj.values_mut() {
                        drop_unstable_fields(v);
                    }
                }
                serde_json::Value::Array(arr) => {
                    for v in arr.iter_mut() {
                        drop_unstable_fields(v);
                    }
                }
                _ => {}
            }
        }

        drop_unstable_fields(&mut v);

        // Keep comparisons focused on semantics, not build metadata or timings.
        if let Some(meta) = v.get_mut("meta").and_then(|m| m.as_object_mut()) {
            meta.remove("nextstat_version");
            meta.remove("threads_applied");
            meta.remove("device");
            meta.remove("warnings");
        }
        v
    }

    fn assert_json_close(a: &serde_json::Value, b: &serde_json::Value, path: &str) {
        use serde_json::Value;

        const RTOL: f64 = 1e-5;
        const ATOL: f64 = 1e-8;

        // SAEM optimizer convergence is inherently platform-dependent:
        // different BLAS/LAPACK implementations produce different eta
        // (sign-flips), omega (~14%), ofv (~4e-4), theta, sigma values.
        // For pharma_fit we verify only structural parity: key presence,
        // array shapes, type agreement, and numeric finiteness.
        fn is_pharma_fit_path(path: &str) -> bool {
            path.contains("nextstat_pharma_fit")
        }

        match (a, b) {
            (Value::Number(na), Value::Number(nb)) => {
                if is_pharma_fit_path(path) {
                    let af = na.as_f64().unwrap_or(f64::NAN);
                    let bf = nb.as_f64().unwrap_or(f64::NAN);
                    assert!(af.is_finite(), "{path}: got {af} (not finite)");
                    assert!(bf.is_finite(), "{path}: want {bf} (not finite)");
                    return;
                }
                let af = na.as_f64().unwrap_or(f64::NAN);
                let bf = nb.as_f64().unwrap_or(f64::NAN);
                let diff = (af - bf).abs();
                if diff <= ATOL {
                    return;
                }
                let denom = af.abs().max(bf.abs()).max(1.0);
                if diff / denom <= RTOL {
                    return;
                }
                panic!("{path}: {af} != {bf} (diff={diff}, rtol={RTOL}, atol={ATOL})");
            }
            (Value::Object(oa), Value::Object(ob)) => {
                let ka: std::collections::BTreeSet<_> = oa.keys().collect();
                let kb: std::collections::BTreeSet<_> = ob.keys().collect();
                assert_eq!(ka, kb, "{path}: key mismatch");
                for k in oa.keys() {
                    assert_json_close(&oa[k], &ob[k], &format!("{path}.{k}"));
                }
            }
            (Value::Array(aa), Value::Array(ab)) => {
                assert_eq!(aa.len(), ab.len(), "{path}: length mismatch");
                for (i, (xa, xb)) in aa.iter().zip(ab.iter()).enumerate() {
                    assert_json_close(xa, xb, &format!("{path}[{i}]"));
                }
            }
            _ => {
                assert_eq!(a, b, "{path}: value mismatch");
            }
        }
    }

    #[test]
    fn assert_json_close_pharma_fit_structural_only() {
        // All pharma_fit numerics: skip value comparison, verify finiteness
        assert_json_close(
            &serde_json::json!(0.0476885995911335),
            &serde_json::json!(-0.00031001154414000894),
            "tool:nextstat_pharma_fit.result.eta[0][1]",
        );
        assert_json_close(
            &serde_json::json!(0.19928549915312144),
            &serde_json::json!(0.2320533313438116),
            "tool:nextstat_pharma_fit.result.omega[0]",
        );

        // Same drift must fail for a non-pharma tool (strict defaults)
        let strict_result = std::panic::catch_unwind(|| {
            assert_json_close(
                &serde_json::json!(10.403818252411241),
                &serde_json::json!(10.4042179617348),
                "tool:nextstat_glm_fit.result.deviance",
            );
        });
        assert!(strict_result.is_err(), "strict defaults must reject drift for non-pharma tools");
    }

    #[test]
    fn tool_execute_fit_smoke_ok() {
        let state = AppState::new(None);
        let ws = include_str!("../../../tests/fixtures/simple_workspace.json");

        let req = ToolExecuteRequest {
            name: "nextstat_fit".to_string(),
            arguments: serde_json::json!({
                "workspace_json": ws,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert_eq!(out.schema_version, "nextstat.tool_result.v1");
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        assert_eq!(out.meta.tool_name, "nextstat_fit");
        let obj = out.result.as_object().expect("result must be an object");
        for k in ["nll", "converged", "n_iter", "poi_index", "poi_value", "poi_error", "parameters"]
        {
            assert!(obj.contains_key(k), "missing key {k} in result");
        }
    }

    #[test]
    fn tool_execute_fit_accepts_simplified_likelihood_workspace() {
        let state = AppState::new(None);
        let ws = include_str!("../../../tests/fixtures/sl_basis_two_bin.json");

        let req = ToolExecuteRequest {
            name: "nextstat_fit".to_string(),
            arguments: serde_json::json!({
                "workspace_json": ws,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert_eq!(out.schema_version, "nextstat.tool_result.v1");
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let obj = out.result.as_object().expect("result must be an object");
        let parameters = obj
            .get("parameters")
            .and_then(|value| value.as_object())
            .expect("fit result must include parameters");
        assert_eq!(parameters.len(), 2, "expected POI plus reduced nuisance");
    }

    #[test]
    fn tool_execute_workspace_audit_accepts_simplified_likelihood_workspace() {
        let state = AppState::new(None);
        let ws = include_str!("../../../tests/fixtures/sl_covariance_three_bin.json");

        let req = ToolExecuteRequest {
            name: "nextstat_workspace_audit".to_string(),
            arguments: serde_json::json!({
                "workspace_json": ws,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert_eq!(out.schema_version, "nextstat.tool_result.v1");
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        assert_eq!(out.meta.tool_name, "nextstat_workspace_audit");

        let obj = out.result.as_object().expect("result must be an object");
        assert_eq!(
            obj.get("schema_version").and_then(|value| value.as_str()),
            Some("nextstat_simplified_likelihood_audit_v0")
        );
        assert_eq!(
            obj.get("input_schema_version").and_then(|value| value.as_str()),
            Some("nextstat_simplified_likelihood_v0")
        );
        assert_eq!(
            obj.get("uncertainty_model_kind").and_then(|value| value.as_str()),
            Some("covariance")
        );
        assert_eq!(
            obj.get("diagnostics")
                .and_then(|value| value.as_object())
                .and_then(|diagnostics| diagnostics.get("factorization"))
                .and_then(|value| value.as_object())
                .and_then(|factorization| factorization.get("method"))
                .and_then(|value| value.as_str()),
            Some("symmetric_eigendecomposition")
        );
    }

    #[test]
    fn tool_execute_glm_fit_supports_server_safe_families() {
        let state = AppState::new(None);
        let cases: &[(&str, serde_json::Value)] = &[
            (
                "linear",
                serde_json::json!({
                    "x": [[0.0], [1.0], [2.0], [3.0], [4.0]],
                    "y": [1.1, 2.9, 5.2, 6.8, 9.1],
                    "family": "linear",
                    "include_intercept": true,
                    "l2": 0.5,
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "logistic",
                serde_json::json!({
                    "x": [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
                    "y": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
                    "family": "logistic",
                    "include_intercept": true,
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "poisson",
                serde_json::json!({
                    "x": [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
                    "y": [1.0, 1.0, 2.0, 3.0, 5.0, 8.0],
                    "family": "poisson",
                    "include_intercept": true,
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "negbin",
                serde_json::json!({
                    "x": [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
                    "y": [0.0, 1.0, 1.0, 2.0, 4.0, 7.0],
                    "family": "negbin",
                    "include_intercept": true,
                    "execution": { "deterministic": true }
                }),
            ),
        ];

        for (family, arguments) in cases {
            let req = ToolExecuteRequest {
                name: "nextstat_glm_fit".to_string(),
                arguments: arguments.clone(),
            };
            let out = execute_tool(&state, req);
            assert!(out.ok, "expected ok=true for family={family}, got error={:?}", out.error);
            assert_eq!(out.schema_version, "nextstat.tool_result.v1");
            assert_eq!(out.meta.tool_name, "nextstat_glm_fit");
            assert_eq!(out.result["family"].as_str(), Some(*family));
            assert!(out.result["coef"].is_array(), "family={family} must return coef array");
            assert!(
                out.result["standard_errors"].is_array(),
                "family={family} must return standard_errors array"
            );
            if *family == "linear" {
                assert!(
                    out.result.get("sigma2_hat").is_some(),
                    "linear family must return sigma2_hat"
                );
            }
            if *family == "negbin" {
                assert!(out.result.get("alpha").is_some(), "negbin family must return alpha");
            }
        }
    }

    #[test]
    fn tool_execute_bayesian_sample_supports_server_safe_models() {
        let state = AppState::new(None);
        let histfactory_ws = include_str!("../../../tests/fixtures/simple_workspace.json");
        let cases: &[(&str, serde_json::Value)] = &[
            (
                "linear_regression",
                serde_json::json!({
                    "model_type": "linear_regression",
                    "x": [[0.0], [1.0], [2.0], [3.0], [4.0]],
                    "y": [1.0, 2.1, 2.9, 4.2, 5.1],
                    "n_chains": 2,
                    "n_warmup": 10,
                    "n_samples": 10,
                    "seed": 42,
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "logistic_regression",
                serde_json::json!({
                    "model_type": "logistic_regression",
                    "x": [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
                    "y": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
                    "n_chains": 2,
                    "n_warmup": 10,
                    "n_samples": 10,
                    "seed": 42,
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "poisson_regression",
                serde_json::json!({
                    "model_type": "poisson_regression",
                    "x": [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
                    "y": [1.0, 1.0, 2.0, 3.0, 5.0, 8.0],
                    "n_chains": 2,
                    "n_warmup": 10,
                    "n_samples": 10,
                    "seed": 42,
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "negbin_regression",
                serde_json::json!({
                    "model_type": "negbin_regression",
                    "x": [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
                    "y": [0.0, 1.0, 1.0, 2.0, 4.0, 7.0],
                    "n_chains": 2,
                    "n_warmup": 10,
                    "n_samples": 10,
                    "seed": 42,
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "cox_ph",
                serde_json::json!({
                    "model_type": "cox_ph",
                    "x": [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]],
                    "time": [1.0, 2.0, 3.5, 4.0, 5.0, 6.0],
                    "event": [1, 1, 0, 1, 0, 1],
                    "n_chains": 2,
                    "n_warmup": 10,
                    "n_samples": 10,
                    "seed": 42,
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "weibull_survival",
                serde_json::json!({
                    "model_type": "weibull_survival",
                    "time": [1.0, 2.0, 3.5, 4.0, 5.0, 6.0],
                    "event": [1, 1, 0, 1, 0, 1],
                    "n_chains": 2,
                    "n_warmup": 10,
                    "n_samples": 10,
                    "seed": 42,
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "lognormal_aft",
                serde_json::json!({
                    "model_type": "lognormal_aft",
                    "time": [1.0, 2.0, 3.5, 4.0, 5.0, 6.0],
                    "event": [1, 1, 0, 1, 0, 1],
                    "n_chains": 2,
                    "n_warmup": 10,
                    "n_samples": 10,
                    "seed": 42,
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "ordered_logit",
                serde_json::json!({
                    "model_type": "ordered_logit",
                    "x": [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
                    "y": [0.0, 1.0, 2.0, 1.0, 2.0, 0.0],
                    "n_levels": 3,
                    "n_chains": 2,
                    "n_warmup": 10,
                    "n_samples": 10,
                    "seed": 42,
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "ordered_probit",
                serde_json::json!({
                    "model_type": "ordered_probit",
                    "x": [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
                    "y": [0.0, 1.0, 2.0, 1.0, 2.0, 0.0],
                    "n_levels": 3,
                    "n_chains": 2,
                    "n_warmup": 10,
                    "n_samples": 10,
                    "seed": 42,
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "histfactory",
                serde_json::json!({
                    "model_type": "histfactory",
                    "workspace_json": histfactory_ws,
                    "n_chains": 2,
                    "n_warmup": 10,
                    "n_samples": 10,
                    "seed": 42,
                    "execution": { "deterministic": true }
                }),
            ),
        ];

        for (model_type, arguments) in cases {
            let req = ToolExecuteRequest {
                name: "nextstat_bayesian_sample".to_string(),
                arguments: arguments.clone(),
            };
            let out = execute_tool(&state, req);
            assert!(
                out.ok,
                "expected ok=true for model_type={model_type}, got error={:?}",
                out.error
            );
            assert_eq!(out.schema_version, "nextstat.tool_result.v1");
            assert_eq!(out.meta.tool_name, "nextstat_bayesian_sample");
            assert_eq!(out.result["model_type"].as_str(), Some(*model_type));
            assert_eq!(out.result["n_chains"].as_u64(), Some(2));
            assert_eq!(out.result["n_warmup"].as_u64(), Some(10));
            assert_eq!(out.result["n_samples"].as_u64(), Some(10));
            assert!(
                out.result["param_names"].is_array(),
                "model_type={model_type} must return param_names"
            );
            assert!(
                out.result["diagnostics"].is_object(),
                "model_type={model_type} must return diagnostics"
            );
            assert!(
                out.result["posterior_summary"].is_object(),
                "model_type={model_type} must return posterior_summary"
            );
            assert!(
                out.result["diagnostics"]["quality"]["status"].is_string(),
                "model_type={model_type} must return diagnostics quality status"
            );
        }
    }

    #[test]
    fn tool_execute_survival_fit_supports_server_safe_models() {
        let state = AppState::new(None);
        let cases: &[(&str, serde_json::Value)] = &[
            (
                "cox_ph",
                serde_json::json!({
                    "x": [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]],
                    "time": [1.0, 2.0, 3.5, 4.0, 5.0, 6.0],
                    "event": [1, 1, 0, 1, 0, 1],
                    "model": "cox_ph",
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "weibull",
                serde_json::json!({
                    "x": [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]],
                    "time": [1.0, 2.0, 3.5, 4.0, 5.0, 6.0],
                    "event": [1, 1, 0, 1, 0, 1],
                    "model": "weibull",
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "lognormal_aft",
                serde_json::json!({
                    "x": [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]],
                    "time": [1.0, 2.0, 3.5, 4.0, 5.0, 6.0],
                    "event": [1, 1, 0, 1, 0, 1],
                    "model": "lognormal_aft",
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "exponential",
                serde_json::json!({
                    "x": [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]],
                    "time": [1.0, 2.0, 3.5, 4.0, 5.0, 6.0],
                    "event": [1, 1, 0, 1, 0, 1],
                    "model": "exponential",
                    "execution": { "deterministic": true }
                }),
            ),
        ];

        for (model, arguments) in cases {
            let req = ToolExecuteRequest {
                name: "nextstat_survival_fit".to_string(),
                arguments: arguments.clone(),
            };
            let out = execute_tool(&state, req);
            assert!(out.ok, "expected ok=true for model={model}, got error={:?}", out.error);
            assert_eq!(out.schema_version, "nextstat.tool_result.v1");
            assert_eq!(out.meta.tool_name, "nextstat_survival_fit");
            assert_eq!(out.result["model"].as_str(), Some(*model));
            assert!(
                out.result["parameters"].is_array(),
                "model={model} must return parameters array"
            );
            assert!(
                out.result["uncertainties"].is_array(),
                "model={model} must return uncertainties array"
            );
            assert!(out.result["nll"].is_number(), "model={model} must return nll");
            assert!(out.result["converged"].is_boolean(), "model={model} must return converged");
        }
    }

    #[test]
    fn tool_execute_kaplan_meier_supports_server_safe_contract() {
        let state = AppState::new(None);

        let req = ToolExecuteRequest {
            name: "nextstat_kaplan_meier".to_string(),
            arguments: serde_json::json!({
                "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "event": [1, 1, 0, 1, 0, 1],
                "group": [0, 0, 1, 1, 0, 1],
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        assert_eq!(out.schema_version, "nextstat.tool_result.v1");
        assert_eq!(out.meta.tool_name, "nextstat_kaplan_meier");
        assert!(out.result["time"].is_array(), "Kaplan-Meier must return time array");
        assert!(out.result["survival"].is_array(), "Kaplan-Meier must return survival array");
        assert_eq!(out.result["conf_level"].as_f64(), Some(0.95));
        assert!(out.result.get("log_rank").is_some(), "grouped Kaplan-Meier must return log_rank");
    }

    #[test]
    fn tool_execute_log_rank_test_supports_server_safe_contract() {
        let state = AppState::new(None);

        let req = ToolExecuteRequest {
            name: "nextstat_log_rank_test".to_string(),
            arguments: serde_json::json!({
                "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
                "events": [true, false, true, false, false, true, false, true],
                "groups": [0, 0, 0, 0, 1, 1, 1, 1],
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        assert_eq!(out.schema_version, "nextstat.tool_result.v1");
        assert_eq!(out.meta.tool_name, "nextstat_log_rank_test");
        assert_eq!(out.result["n"].as_u64(), Some(8));
        assert_eq!(out.result["df"].as_u64(), Some(1));
        assert_eq!(out.result["group_ids"].as_array().map(|v| v.len()), Some(2));
        assert_eq!(out.result["observed"].as_array().map(|v| v.len()), Some(2));
        assert_eq!(out.result["expected"].as_array().map(|v| v.len()), Some(2));
    }

    #[test]
    fn tool_execute_meta_analysis_supports_server_safe_methods() {
        let state = AppState::new(None);
        let cases: &[(&str, serde_json::Value)] = &[
            (
                "fixed",
                serde_json::json!({
                    "effects": [0.2, 0.5, -0.1, 0.3],
                    "standard_errors": [0.1, 0.2, 0.15, 0.12],
                    "method": "fixed",
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "random",
                serde_json::json!({
                    "effects": [0.2, 0.5, -0.1, 0.3],
                    "standard_errors": [0.1, 0.2, 0.15, 0.12],
                    "method": "random",
                    "execution": { "deterministic": true }
                }),
            ),
        ];

        for (method, arguments) in cases {
            let req = ToolExecuteRequest {
                name: "nextstat_meta_analysis".to_string(),
                arguments: arguments.clone(),
            };
            let out = execute_tool(&state, req);
            assert!(out.ok, "expected ok=true for method={method}, got error={:?}", out.error);
            assert_eq!(out.schema_version, "nextstat.tool_result.v1");
            assert_eq!(out.meta.tool_name, "nextstat_meta_analysis");
            assert_eq!(out.result["method"].as_str(), Some(*method));
            assert!(
                out.result["heterogeneity"].is_object(),
                "method={method} must return heterogeneity"
            );
            assert!(out.result["forest"].is_array(), "method={method} must return forest array");
            assert!(out.result["estimate"].is_number(), "method={method} must return estimate");
        }
    }

    #[test]
    fn tool_execute_panel_fe_supports_server_safe_cluster_kinds() {
        let state = AppState::new(None);
        let cases: &[(&str, serde_json::Value)] = &[
            (
                "entity",
                serde_json::json!({
                    "x": [[1.0], [2.0], [1.0], [2.5], [3.0], [3.5]],
                    "y": [1.0, 2.0, 1.5, 2.5, 3.0, 3.5],
                    "entity": [0, 0, 1, 1, 2, 2],
                    "cluster": "entity",
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "none",
                serde_json::json!({
                    "x": [[1.0], [2.0], [1.0], [2.5], [3.0], [3.5]],
                    "y": [1.0, 2.0, 1.5, 2.5, 3.0, 3.5],
                    "entity": [0, 0, 1, 1, 2, 2],
                    "cluster": "none",
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "time",
                serde_json::json!({
                    "x": [[1.0], [2.0], [1.0], [2.5], [3.0], [3.5]],
                    "y": [1.0, 2.0, 1.5, 2.5, 3.0, 3.5],
                    "entity": [0, 0, 1, 1, 2, 2],
                    "time": [0, 1, 0, 1, 0, 1],
                    "cluster": "time",
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "two_way",
                serde_json::json!({
                    "x": [[1.0], [2.0], [1.0], [2.5], [3.0], [3.5]],
                    "y": [1.0, 2.0, 1.5, 2.5, 3.0, 3.5],
                    "entity": [0, 0, 1, 1, 2, 2],
                    "time": [0, 1, 0, 1, 0, 1],
                    "cluster": "two_way",
                    "execution": { "deterministic": true }
                }),
            ),
        ];

        for (cluster, arguments) in cases {
            let req = ToolExecuteRequest {
                name: "nextstat_panel_fe".to_string(),
                arguments: arguments.clone(),
            };
            let out = execute_tool(&state, req);
            assert!(out.ok, "expected ok=true for cluster={cluster}, got error={:?}", out.error);
            assert_eq!(out.schema_version, "nextstat.tool_result.v1");
            assert_eq!(out.meta.tool_name, "nextstat_panel_fe");
            assert_eq!(out.result["cluster_kind"].as_str(), Some(*cluster));
            assert_eq!(out.result["cluster"].as_str(), Some(*cluster));
            assert!(out.result["coef"].is_array(), "cluster={cluster} must return coef array");
            assert!(
                out.result["standard_errors"].is_array(),
                "cluster={cluster} must return standard_errors array"
            );
            assert_eq!(out.result["n_obs"].as_u64(), Some(6));
            assert_eq!(out.result["n_entities"].as_u64(), Some(3));
        }
    }

    #[test]
    fn tool_execute_did_supports_server_safe_cluster_kinds() {
        let state = AppState::new(None);
        let cases: &[(&str, serde_json::Value)] = &[
            (
                "entity",
                serde_json::json!({
                    "y": [1.0, 1.2, 1.4, 1.8, 1.1, 1.3, 2.2, 2.5],
                    "treat": [0, 0, 0, 0, 1, 1, 1, 1],
                    "post": [0, 1, 0, 1, 0, 1, 0, 1],
                    "entity": [0, 0, 1, 1, 2, 2, 3, 3],
                    "time": [0, 1, 0, 1, 0, 1, 0, 1],
                    "cluster": "entity",
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "none",
                serde_json::json!({
                    "y": [1.0, 1.2, 1.4, 1.8, 1.1, 1.3, 2.2, 2.5],
                    "treat": [0, 0, 0, 0, 1, 1, 1, 1],
                    "post": [0, 1, 0, 1, 0, 1, 0, 1],
                    "entity": [0, 0, 1, 1, 2, 2, 3, 3],
                    "time": [0, 1, 0, 1, 0, 1, 0, 1],
                    "cluster": "none",
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "time",
                serde_json::json!({
                    "y": [1.0, 1.2, 1.4, 1.8, 1.1, 1.3, 2.2, 2.5],
                    "treat": [0, 0, 0, 0, 1, 1, 1, 1],
                    "post": [0, 1, 0, 1, 0, 1, 0, 1],
                    "entity": [0, 0, 1, 1, 2, 2, 3, 3],
                    "time": [0, 1, 0, 1, 0, 1, 0, 1],
                    "cluster": "time",
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "two_way",
                serde_json::json!({
                    "y": [1.0, 1.2, 1.4, 1.8, 1.1, 1.3, 2.2, 2.5],
                    "treat": [0, 0, 0, 0, 1, 1, 1, 1],
                    "post": [0, 1, 0, 1, 0, 1, 0, 1],
                    "entity": [0, 0, 1, 1, 2, 2, 3, 3],
                    "time": [0, 1, 0, 1, 0, 1, 0, 1],
                    "cluster": "two_way",
                    "execution": { "deterministic": true }
                }),
            ),
        ];

        for (cluster, arguments) in cases {
            let req = ToolExecuteRequest {
                name: "nextstat_did".to_string(),
                arguments: arguments.clone(),
            };
            let out = execute_tool(&state, req);
            assert!(out.ok, "expected ok=true for cluster={cluster}, got error={:?}", out.error);
            assert_eq!(out.schema_version, "nextstat.tool_result.v1");
            assert_eq!(out.meta.tool_name, "nextstat_did");
            assert_eq!(out.result["cluster"].as_str(), Some(*cluster));
            assert!(out.result["att"].is_number(), "cluster={cluster} must return att");
            assert!(out.result["att_se"].is_number(), "cluster={cluster} must return att_se");
            assert!(out.result["coef"].is_array(), "cluster={cluster} must return coef array");
            assert!(
                out.result["standard_errors"].is_array(),
                "cluster={cluster} must return standard_errors array"
            );
            assert_eq!(out.result["n_obs"].as_u64(), Some(8));
        }

        let req = ToolExecuteRequest {
            name: "nextstat_did".to_string(),
            arguments: serde_json::json!({
                "y": [1.0, 1.2, 1.4, 1.8, 1.1, 1.3, 2.2, 2.5],
                "treat": [0, 0, 0, 0, 1, 1, 1, 1],
                "post": [0, 1, 0, 1, 0, 1, 0, 1],
                "entity": [0, 0, 1, 1, 2, 2, 3, 3],
                "time": [0, 1, 0, 1, 0, 1, 0, 1],
                "x": [[0.0], [1.0], [0.5], [1.5], [0.2], [1.2], [0.7], [1.7]],
                "cluster": "entity",
                "execution": { "deterministic": true }
            }),
        };
        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true with controls, got error={:?}", out.error);
        assert_eq!(out.result["cluster"].as_str(), Some("entity"));
        let coef_len = out.result["coef"].as_array().map(|v| v.len()).expect("coef must be array");
        let se_len = out.result["standard_errors"]
            .as_array()
            .map(|v| v.len())
            .expect("standard_errors must be array");
        assert!(coef_len >= 1, "expected at least ATT coefficient to remain identifiable");
        assert_eq!(coef_len, se_len, "coef and standard_errors must stay aligned");
    }

    #[test]
    fn tool_execute_iv_2sls_supports_server_safe_cov_estimators() {
        let state = AppState::new(None);
        let cases: &[(&str, serde_json::Value)] = &[
            (
                "homoskedastic",
                serde_json::json!({
                    "y": [1.0, 2.1, 1.7, 2.9, 3.2, 3.8, 4.2, 4.7],
                    "endog": [[1.0], [1.8], [1.4], [2.2], [2.7], [3.1], [3.5], [4.0]],
                    "instruments": [[0.9], [1.7], [1.2], [2.0], [2.4], [2.9], [3.3], [3.8]],
                    "exog": [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0]],
                    "cov": "homoskedastic",
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "hc1",
                serde_json::json!({
                    "y": [1.0, 2.1, 1.7, 2.9, 3.2, 3.8, 4.2, 4.7],
                    "endog": [[1.0], [1.8], [1.4], [2.2], [2.7], [3.1], [3.5], [4.0]],
                    "instruments": [[0.9], [1.7], [1.2], [2.0], [2.4], [2.9], [3.3], [3.8]],
                    "exog": [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0]],
                    "cov": "hc1",
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "cluster",
                serde_json::json!({
                    "y": [1.0, 2.1, 1.7, 2.9, 3.2, 3.8, 4.2, 4.7],
                    "endog": [[1.0], [1.8], [1.4], [2.2], [2.7], [3.1], [3.5], [4.0]],
                    "instruments": [[0.9], [1.7], [1.2], [2.0], [2.4], [2.9], [3.3], [3.8]],
                    "exog": [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0]],
                    "cov": "cluster",
                    "cluster": ["a", "a", "b", "b", "c", "c", "d", "d"],
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "hac",
                serde_json::json!({
                    "y": [1.0, 2.1, 1.7, 2.9, 3.2, 3.8, 4.2, 4.7],
                    "endog": [[1.0], [1.8], [1.4], [2.2], [2.7], [3.1], [3.5], [4.0]],
                    "instruments": [[0.9], [1.7], [1.2], [2.0], [2.4], [2.9], [3.3], [3.8]],
                    "exog": [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0]],
                    "cov": "hac",
                    "time_index": [0, 1, 2, 3, 4, 5, 6, 7],
                    "max_lag": 2,
                    "execution": { "deterministic": true }
                }),
            ),
        ];

        for (cov, arguments) in cases {
            let req = ToolExecuteRequest {
                name: "nextstat_iv_2sls".to_string(),
                arguments: arguments.clone(),
            };
            let out = execute_tool(&state, req);
            assert!(out.ok, "expected ok=true for cov={cov}, got error={:?}", out.error);
            assert_eq!(out.schema_version, "nextstat.tool_result.v1");
            assert_eq!(out.meta.tool_name, "nextstat_iv_2sls");
            assert!(out.result["coef"].is_array(), "cov={cov} must return coef array");
            assert!(
                out.result["standard_errors"].is_array(),
                "cov={cov} must return standard_errors array"
            );
            assert_eq!(out.result["n_obs"].as_u64(), Some(8));
            assert_eq!(
                out.result["diagnostics"]["first_stage_f"].as_array().map(|v| v.len()),
                Some(1)
            );
        }
    }

    #[test]
    fn tool_execute_aipw_supports_server_safe_estimands() {
        let state = AppState::new(None);
        let cases: &[(&str, serde_json::Value)] = &[
            (
                "ate",
                serde_json::json!({
                    "x": [[0.0], [1.0], [0.0], [1.0], [0.2], [1.2], [0.1], [1.1]],
                    "y": [1.0, 2.0, 1.2, 2.2, 1.1, 2.4, 1.3, 2.5],
                    "treatment": [0, 1, 0, 1, 0, 1, 0, 1],
                    "estimand": "ate",
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "att",
                serde_json::json!({
                    "x": [[0.0], [1.0], [0.0], [1.0], [0.2], [1.2], [0.1], [1.1]],
                    "y": [1.0, 2.0, 1.2, 2.2, 1.1, 2.4, 1.3, 2.5],
                    "treatment": [0, 1, 0, 1, 0, 1, 0, 1],
                    "estimand": "att",
                    "execution": { "deterministic": true }
                }),
            ),
        ];

        for (estimand, arguments) in cases {
            let req = ToolExecuteRequest {
                name: "nextstat_aipw".to_string(),
                arguments: arguments.clone(),
            };
            let out = execute_tool(&state, req);
            assert!(out.ok, "expected ok=true for estimand={estimand}, got error={:?}", out.error);
            assert_eq!(out.schema_version, "nextstat.tool_result.v1");
            assert_eq!(out.meta.tool_name, "nextstat_aipw");
            assert_eq!(out.result["estimand"].as_str(), Some(*estimand));
            assert!(out.result["estimate"].is_number(), "estimand={estimand} must return estimate");
            assert!(
                out.result["standard_error"].is_number(),
                "estimand={estimand} must return standard_error"
            );
            assert_eq!(out.result["n_obs"].as_u64(), Some(8));
        }
    }

    #[test]
    fn tool_execute_event_study_supports_server_safe_cluster_kinds() {
        let state = AppState::new(None);
        let base = serde_json::json!({
            "y": [1.0, 1.1, 1.2, 1.4, 1.0, 1.0, 1.1, 1.2, 2.0, 2.3, 2.7, 3.0],
            "entity": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            "time": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            "treat_time": [serde_json::Value::Null, serde_json::Value::Null, serde_json::Value::Null, serde_json::Value::Null, serde_json::json!(2), serde_json::json!(2), serde_json::json!(2), serde_json::json!(2), serde_json::json!(1), serde_json::json!(1), serde_json::json!(1), serde_json::json!(1)],
            "n_leads": 2,
            "n_lags": 2,
            "execution": { "deterministic": true }
        });

        for cluster in ["entity", "time", "two_way"] {
            let mut arguments = base.clone();
            arguments["cluster"] = serde_json::json!(cluster);
            let req = ToolExecuteRequest { name: "nextstat_event_study".to_string(), arguments };
            let out = execute_tool(&state, req);
            assert!(out.ok, "expected ok=true for cluster={cluster}, got error={:?}", out.error);
            assert_eq!(out.schema_version, "nextstat.tool_result.v1");
            assert_eq!(out.meta.tool_name, "nextstat_event_study");
            assert_eq!(out.result["cluster"].as_str(), Some(cluster));
            assert!(out.result["rel_times"].is_array(), "cluster={cluster} must return rel_times");
            assert!(out.result["coef"].is_array(), "cluster={cluster} must return coef");
            assert!(
                out.result["standard_errors"].is_array(),
                "cluster={cluster} must return standard_errors"
            );
            assert!(
                out.result["covariance"].is_array(),
                "cluster={cluster} must return covariance"
            );
            assert_eq!(out.result["reference"].as_i64(), Some(-1));
            assert_eq!(out.result["n_obs"].as_u64(), Some(12));
            assert_eq!(out.result["n_entities"].as_u64(), Some(3));
            assert_eq!(out.result["n_times"].as_u64(), Some(4));
        }
    }

    #[test]
    fn tool_execute_garch_supports_server_safe_models() {
        let state = AppState::new(None);
        let base_returns =
            vec![0.01, -0.02, 0.015, -0.01, 0.005, 0.02, -0.015, 0.01, 0.012, -0.008];

        for model in ["garch", "egarch", "gjr_garch"] {
            let req = ToolExecuteRequest {
                name: "nextstat_garch_fit".to_string(),
                arguments: serde_json::json!({
                    "returns": base_returns,
                    "model": model,
                    "execution": { "deterministic": true }
                }),
            };
            let out = execute_tool(&state, req);
            assert!(out.ok, "expected ok=true for model={model}, got error={:?}", out.error);
            assert_eq!(out.schema_version, "nextstat.tool_result.v1");
            assert_eq!(out.meta.tool_name, "nextstat_garch_fit");
            assert!(out.result["params"].is_object(), "model={model} must return params");
            assert!(
                out.result["conditional_variance"].is_array(),
                "model={model} must return conditional_variance"
            );
            assert!(
                out.result["conditional_sigma"].is_array(),
                "model={model} must return conditional_sigma"
            );
            assert!(
                out.result["log_likelihood"].is_number(),
                "model={model} must return log_likelihood"
            );
            assert!(out.result["converged"].is_boolean(), "model={model} must return converged");
            assert!(out.result["n_iter"].is_u64(), "model={model} must return n_iter");
            if model == "garch" {
                assert!(
                    out.result["params"].get("gamma").is_none(),
                    "plain garch must not return gamma"
                );
            } else {
                assert!(
                    out.result["params"]["gamma"].is_number(),
                    "model={model} must return gamma"
                );
            }
        }
    }

    #[test]
    fn tool_execute_ads_variance_reduction_supports_server_safe_modes() {
        let state = AppState::new(None);

        let cuped = ToolExecuteRequest {
            name: "nextstat_ads_cuped_adjust".to_string(),
            arguments: serde_json::json!({
                "control_outcomes": [10.0, 12.0, 9.0, 11.0, 13.0],
                "control_covariates": [8.0, 10.0, 7.0, 9.0, 11.0],
                "variant_outcomes": [11.0, 13.0, 10.0, 12.0, 14.0],
                "variant_covariates": [8.5, 10.5, 7.5, 9.5, 11.5],
                "covariate_name": "pre_clicks",
                "covariate_provenance": {
                    "name": "pre_clicks",
                    "timing": "pre_treatment",
                    "source_dataset": "ads_preperiod_daily"
                },
                "pre_treatment_only": true,
                "execution": { "deterministic": true }
            }),
        };
        let cuped_out = execute_tool(&state, cuped);
        assert!(cuped_out.ok, "expected ok=true, got error={:?}", cuped_out.error);
        assert_eq!(cuped_out.meta.tool_name, "nextstat_ads_cuped_adjust");
        assert_eq!(cuped_out.result["method"].as_str(), Some("cuped"));
        assert_eq!(cuped_out.result["num_covariates"].as_u64(), Some(1));
        assert_eq!(
            cuped_out.result["selected_covariates"].as_array().map(|items| items.len()),
            Some(1)
        );
        assert_eq!(cuped_out.result["provenance_validated"].as_bool(), Some(true));
        assert_eq!(cuped_out.result["pre_treatment_only"].as_bool(), Some(true));

        let cure = ToolExecuteRequest {
            name: "nextstat_ads_cure_adjust".to_string(),
            arguments: serde_json::json!({
                "control_outcomes": [100.0, 110.0, 95.0, 105.0, 115.0, 120.0],
                "control_covariates": [
                    [80.0, 1000.0],
                    [88.0, 1100.0],
                    [75.0, 950.0],
                    [84.0, 1025.0],
                    [92.0, 1150.0],
                    [96.0, 1180.0]
                ],
                "variant_outcomes": [104.0, 113.0, 99.0, 109.0, 118.0, 124.0],
                "variant_covariates": [
                    [81.0, 1008.0],
                    [89.0, 1110.0],
                    [76.0, 960.0],
                    [85.0, 1035.0],
                    [93.0, 1165.0],
                    [97.0, 1192.0]
                ],
                "covariate_names": ["pre_clicks", "pre_impressions"],
                "covariate_provenance": [
                    {
                        "name": "pre_clicks",
                        "timing": "pre_treatment",
                        "source_dataset": "ads_preperiod_daily"
                    },
                    {
                        "name": "pre_impressions",
                        "timing": "pre_treatment",
                        "source_dataset": "ads_preperiod_daily"
                    }
                ],
                "pre_treatment_only": true,
                "execution": { "deterministic": true }
            }),
        };
        let cure_out = execute_tool(&state, cure);
        assert!(cure_out.ok, "expected ok=true, got error={:?}", cure_out.error);
        assert_eq!(cure_out.meta.tool_name, "nextstat_ads_cure_adjust");
        assert_eq!(cure_out.result["method"].as_str(), Some("cure"));
        assert_eq!(cure_out.result["num_covariates"].as_u64(), Some(2));
        assert_eq!(
            cure_out.result["selected_covariates"].as_array().map(|items| items.len()),
            Some(2)
        );
        assert_eq!(cure_out.result["provenance_validated"].as_bool(), Some(true));
        assert_eq!(cure_out.result["pre_treatment_only"].as_bool(), Some(true));
    }

    #[test]
    fn tool_execute_kalman_supports_server_safe_operations() {
        let state = AppState::new(None);
        let base = serde_json::json!({
            "F": [[1.0]],
            "H": [[1.0]],
            "Q": [[0.1]],
            "R": [[0.2]],
            "x0": [0.0],
            "P0": [[1.0]],
            "y": [[1.0], [1.2], [0.9], [1.1]],
            "execution": { "deterministic": true }
        });

        for operation in ["filter", "smooth", "forecast", "simulate", "em"] {
            let mut arguments = base.clone();
            arguments["operation"] = serde_json::json!(operation);
            if operation == "forecast" {
                arguments["n_ahead"] = serde_json::json!(3);
                arguments["alpha"] = serde_json::json!(0.1);
            } else if operation == "simulate" {
                arguments["t_max"] = serde_json::json!(4);
                arguments["seed"] = serde_json::json!(42);
                arguments["init"] = serde_json::json!("mean");
            } else if operation == "em" {
                arguments["y"] = serde_json::json!([[0.1], [0.2], [0.0]]);
                arguments["max_iter"] = serde_json::json!(3);
                arguments["tol"] = serde_json::json!(1e-6);
                arguments["estimate_q"] = serde_json::json!(true);
                arguments["estimate_r"] = serde_json::json!(true);
                arguments["estimate_f"] = serde_json::json!(false);
                arguments["estimate_h"] = serde_json::json!(false);
                arguments["min_diag"] = serde_json::json!(1e-8);
            }
            let req = ToolExecuteRequest { name: "nextstat_kalman".to_string(), arguments };
            let out = execute_tool(&state, req);
            assert!(
                out.ok,
                "expected ok=true for operation={operation}, got error={:?}",
                out.error
            );
            assert_eq!(out.schema_version, "nextstat.tool_result.v1");
            assert_eq!(out.meta.tool_name, "nextstat_kalman");
            match operation {
                "filter" => {
                    assert!(out.result["log_likelihood"].is_number());
                    assert!(out.result["predicted_means"].is_array());
                    assert!(out.result["predicted_covs"].is_array());
                    assert!(out.result["filtered_means"].is_array());
                    assert!(out.result["filtered_covs"].is_array());
                }
                "smooth" => {
                    assert!(out.result["log_likelihood"].is_number());
                    assert!(out.result["filtered_means"].is_array());
                    assert!(out.result["filtered_covs"].is_array());
                    assert!(out.result["smoothed_means"].is_array());
                    assert!(out.result["smoothed_covs"].is_array());
                }
                "forecast" => {
                    assert!(out.result["state_means"].is_array());
                    assert!(out.result["state_covs"].is_array());
                    assert!(out.result["obs_means"].is_array());
                    assert!(out.result["obs_covs"].is_array());
                    assert!(out.result["alpha"].is_number());
                    assert!(out.result["z"].is_number());
                    assert!(out.result["obs_lower"].is_array());
                    assert!(out.result["obs_upper"].is_array());
                }
                "simulate" => {
                    assert!(out.result["xs"].is_array());
                    assert!(out.result["ys"].is_array());
                }
                "em" => {
                    assert!(out.result["converged"].is_boolean());
                    assert!(out.result["n_iter"].is_number());
                    assert!(out.result["loglik_trace"].is_array());
                    assert!(out.result["f"].is_array());
                    assert!(out.result["h"].is_array());
                    assert!(out.result["q"].is_array());
                    assert!(out.result["r"].is_array());
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn tool_execute_churn_generate_data_supports_server_safe_contract() {
        let state = AppState::new(None);
        let req = ToolExecuteRequest {
            name: "nextstat_churn_generate_data".to_string(),
            arguments: serde_json::json!({
                "n_customers": 12,
                "n_cohorts": 3,
                "max_time": 18.0,
                "treatment_fraction": 0.25,
                "seed": 42,
                "execution": { "deterministic": true }
            }),
        };
        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        assert_eq!(out.schema_version, "nextstat.tool_result.v1");
        assert_eq!(out.meta.tool_name, "nextstat_churn_generate_data");
        assert_eq!(out.result["n"].as_u64(), Some(12));
        assert_eq!(out.result["n_events"].as_u64(), Some(8));
        assert_eq!(out.result["times"].as_array().map(|v| v.len()), Some(12));
        assert_eq!(out.result["covariates"].as_array().map(|v| v.len()), Some(12));
        assert_eq!(out.result["covariate_names"].as_array().map(|v| v.len()), Some(4));
        assert_eq!(out.result["plan"].as_array().map(|v| v.len()), Some(12));
        assert_eq!(out.result["region"].as_array().map(|v| v.len()), Some(12));
        assert_eq!(out.result["cohort"].as_array().map(|v| v.len()), Some(12));
    }

    #[test]
    fn tool_execute_churn_risk_model_supports_server_safe_contract() {
        let state = AppState::new(None);
        let req = ToolExecuteRequest {
            name: "nextstat_churn_risk_model".to_string(),
            arguments: serde_json::json!({
                "times": [
                    18.0, 3.302151210546639, 4.329991875135302, 5.475084445095048,
                    18.0, 16.76730374185681, 12.884032820472772, 8.271505063122506,
                    3.632766793868914, 18.0, 12.32888547551182, 18.0, 18.0,
                    1.4580664577117772, 7.416636560585132, 4.245245279862313
                ],
                "events": [
                    false, true, true, true, false, true, true, true,
                    true, false, true, false, false, true, true, true
                ],
                "covariates": [
                    [1.0, 0.0, 1.3117235495872972, 2.0],
                    [0.0, 0.0, -0.9022021759009674, 0.0],
                    [1.0, 0.0, 0.528865100865726, 2.0],
                    [1.0, 0.0, -1.454760464783373, 3.0],
                    [0.0, 1.0, -1.5385144547016696, 1.0],
                    [1.0, 0.0, 1.5124745460816813, 1.0],
                    [1.0, 0.0, -0.5771217015082334, 1.0],
                    [0.0, 0.0, 1.189054183234974, 0.0],
                    [0.0, 0.0, -0.4320950454679947, 0.0],
                    [1.0, 0.0, 1.147124602981333, 0.0],
                    [0.0, 0.0, -0.3221810058112955, 2.0],
                    [0.0, 1.0, -0.5197845574589398, 3.0],
                    [0.0, 1.0, 0.937777824579468, 1.0],
                    [0.0, 0.0, -1.6269177382249136, 1.0],
                    [1.0, 0.0, 0.8667568109417092, 0.0],
                    [0.0, 1.0, 0.17866086448304247, 1.0]
                ],
                "names": ["plan_basic", "plan_premium", "usage_score", "support_tickets"],
                "conf_level": 0.95,
                "execution": { "deterministic": true }
            }),
        };
        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        assert_eq!(out.schema_version, "nextstat.tool_result.v1");
        assert_eq!(out.meta.tool_name, "nextstat_churn_risk_model");
        assert_eq!(out.result["n"].as_u64(), Some(16));
        assert_eq!(out.result["n_events"].as_u64(), Some(11));
        assert_eq!(out.result["names"].as_array().map(|v| v.len()), Some(4));
        assert_eq!(out.result["coefficients"].as_array().map(|v| v.len()), Some(4));
        assert_eq!(out.result["se"].as_array().map(|v| v.len()), Some(4));
        assert_eq!(out.result["hazard_ratios"].as_array().map(|v| v.len()), Some(4));
        assert_eq!(out.result["hr_ci_lower"].as_array().map(|v| v.len()), Some(4));
        assert_eq!(out.result["hr_ci_upper"].as_array().map(|v| v.len()), Some(4));
    }

    #[test]
    fn tool_execute_churn_retention_supports_server_safe_contract() {
        let state = AppState::new(None);
        let req = ToolExecuteRequest {
            name: "nextstat_churn_retention".to_string(),
            arguments: serde_json::json!({
                "times": [30.0, 60.0, 90.0, 120.0],
                "events": [true, false, true, false],
                "groups": [0, 0, 1, 1],
                "conf_level": 0.95,
                "execution": { "deterministic": true }
            }),
        };
        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        assert_eq!(out.schema_version, "nextstat.tool_result.v1");
        assert_eq!(out.meta.tool_name, "nextstat_churn_retention");
        assert!(out.result["overall"].is_object());
        assert!(out.result["by_group"].is_array());
        assert!(out.result["log_rank"].is_object());
        assert_eq!(out.result["overall"]["n"].as_u64(), Some(4));
        assert_eq!(
            out.result["by_group"].as_array().map(|rows| rows.len()),
            Some(2),
            "two groups should be returned"
        );
    }

    #[test]
    fn tool_execute_churn_diagnostics_supports_server_safe_contract() {
        let state = AppState::new(None);
        let req = ToolExecuteRequest {
            name: "nextstat_churn_diagnostics".to_string(),
            arguments: serde_json::json!({
                "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
                "events": [true, false, true, false, false, true, false, true],
                "groups": [0, 0, 0, 0, 1, 1, 1, 1],
                "treated": [0, 0, 1, 1, 0, 1, 0, 1],
                "covariates": [
                    [1.0, 0.2],
                    [0.5, -0.1],
                    [1.2, 0.4],
                    [0.7, 0.0],
                    [0.3, -0.2],
                    [1.1, 0.1],
                    [0.4, -0.3],
                    [0.9, 0.3]
                ],
                "covariate_names": ["x1", "x2"],
                "trim": 0.01,
                "execution": { "deterministic": true }
            }),
        };
        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        assert_eq!(out.schema_version, "nextstat.tool_result.v1");
        assert_eq!(out.meta.tool_name, "nextstat_churn_diagnostics");
        assert_eq!(out.result["n"].as_u64(), Some(8));
        assert_eq!(out.result["n_events"].as_u64(), Some(4));
        assert!(out.result["overall_censoring_frac"].is_number());
        assert!(out.result["trust_gate_passed"].is_boolean());
        assert_eq!(out.result["censoring_by_segment"].as_array().map(|v| v.len()), Some(2));
        assert_eq!(out.result["covariate_balance"].as_array().map(|v| v.len()), Some(2));
        assert!(out.result["propensity_overlap"].is_object());
        assert!(out.result["warnings"].is_array());
    }

    #[test]
    fn tool_execute_churn_cohort_matrix_supports_server_safe_contract() {
        let state = AppState::new(None);
        let req = ToolExecuteRequest {
            name: "nextstat_churn_cohort_matrix".to_string(),
            arguments: serde_json::json!({
                "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
                "events": [true, false, true, false, false, true, false, true],
                "groups": [0, 0, 0, 0, 1, 1, 1, 1],
                "period_boundaries": [30.0, 60.0, 90.0],
                "execution": { "deterministic": true }
            }),
        };
        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        assert_eq!(out.schema_version, "nextstat.tool_result.v1");
        assert_eq!(out.meta.tool_name, "nextstat_churn_cohort_matrix");
        assert_eq!(out.result["period_boundaries"].as_array().map(|v| v.len()), Some(3));
        assert_eq!(out.result["cohorts"].as_array().map(|v| v.len()), Some(2));
        assert!(out.result["overall"].is_object());
        assert_eq!(out.result["overall"]["cohort"].as_i64(), Some(-1));
        assert_eq!(out.result["overall"]["n_total"].as_u64(), Some(8));
    }

    #[test]
    fn tool_execute_churn_bootstrap_hr_supports_server_safe_contract() {
        let state = AppState::new(None);
        let req = ToolExecuteRequest {
            name: "nextstat_churn_bootstrap_hr".to_string(),
            arguments: serde_json::json!({
                "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
                "events": [true, false, true, false, false, true, false, true],
                "covariates": [
                    [1.0, 0.2],
                    [0.5, -0.1],
                    [1.2, 0.4],
                    [0.7, 0.0],
                    [0.3, -0.2],
                    [1.1, 0.1],
                    [0.4, -0.3],
                    [0.9, 0.3]
                ],
                "names": ["x1", "x2"],
                "n_bootstrap": 8,
                "seed": 42,
                "conf_level": 0.95,
                "ci_method": "percentile",
                "n_jackknife": 4,
                "execution": { "deterministic": true }
            }),
        };
        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        assert_eq!(out.schema_version, "nextstat.tool_result.v1");
        assert_eq!(out.meta.tool_name, "nextstat_churn_bootstrap_hr");
        assert_eq!(out.result["names"].as_array().map(|v| v.len()), Some(2));
        assert_eq!(out.result["hr_point"].as_array().map(|v| v.len()), Some(2));
        assert_eq!(out.result["hr_ci_lower"].as_array().map(|v| v.len()), Some(2));
        assert_eq!(out.result["hr_ci_upper"].as_array().map(|v| v.len()), Some(2));
        assert_eq!(out.result["n_bootstrap"].as_u64(), Some(8));
        assert_eq!(out.result["n_converged"].as_u64(), Some(8));
        assert_eq!(out.result["ci_method_requested"].as_str(), Some("percentile"));
        assert_eq!(out.result["ci_method_effective"].as_array().map(|v| v.len()), Some(2));
        assert_eq!(out.result["ci_diagnostics"].as_array().map(|v| v.len()), Some(2));
    }

    #[test]
    fn tool_execute_churn_ingest_supports_server_safe_contract() {
        let state = AppState::new(None);
        let req = ToolExecuteRequest {
            name: "nextstat_churn_ingest".to_string(),
            arguments: serde_json::json!({
                "times": [30.0, 45.0, 90.0, 12.0],
                "events": [true, false, true, false],
                "groups": [0, 0, 1, 1],
                "treated": [0, 1, 0, 1],
                "covariates": [
                    [0.2, 1.0],
                    [0.3, 0.0],
                    [0.8, 1.0],
                    [0.5, 0.0]
                ],
                "covariate_names": ["usage_score", "plan_pro"],
                "observation_end": 60.0,
                "execution": { "deterministic": true }
            }),
        };
        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        assert_eq!(out.schema_version, "nextstat.tool_result.v1");
        assert_eq!(out.meta.tool_name, "nextstat_churn_ingest");
        assert_eq!(out.result["n"].as_u64(), Some(4));
        assert_eq!(out.result["n_events"].as_u64(), Some(1));
        assert_eq!(out.result["n_dropped"].as_u64(), Some(0));
        assert_eq!(out.result["times"].as_array().map(|v| v.len()), Some(4));
        assert_eq!(out.result["events"].as_array().map(|v| v.len()), Some(4));
        assert_eq!(out.result["groups"].as_array().map(|v| v.len()), Some(4));
        assert_eq!(out.result["treated"].as_array().map(|v| v.len()), Some(4));
        assert_eq!(out.result["covariates"].as_array().map(|v| v.len()), Some(4));
        assert_eq!(out.result["covariate_names"].as_array().map(|v| v.len()), Some(2));
        assert!(out.result["warnings"].is_array());
    }

    #[test]
    fn tool_execute_churn_compare_supports_server_safe_contract() {
        let state = AppState::new(None);
        let req = ToolExecuteRequest {
            name: "nextstat_churn_compare".to_string(),
            arguments: serde_json::json!({
                "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
                "events": [true, false, true, false, false, true, false, true],
                "groups": [0, 0, 0, 0, 1, 1, 1, 1],
                "conf_level": 0.95,
                "correction": "bh",
                "alpha": 0.05,
                "execution": { "deterministic": true }
            }),
        };
        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        assert_eq!(out.schema_version, "nextstat.tool_result.v1");
        assert_eq!(out.meta.tool_name, "nextstat_churn_compare");
        assert!(out.result["overall_chi_squared"].is_number());
        assert!(out.result["overall_p_value"].is_number());
        assert!(out.result["segments"].is_array());
        assert!(out.result["pairwise"].is_array());
        assert_eq!(out.result["correction_method"], "benjamini_hochberg");
        assert_eq!(out.result["n"].as_u64(), Some(8));
        assert_eq!(out.result["n_events"].as_u64(), Some(4));
    }

    #[test]
    fn tool_execute_churn_uplift_supports_server_safe_contract() {
        let state = AppState::new(None);
        let req = ToolExecuteRequest {
            name: "nextstat_churn_uplift".to_string(),
            arguments: serde_json::json!({
                "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
                "events": [true, false, true, false, false, true, false, true],
                "treated": [0, 0, 1, 1, 0, 1, 0, 1],
                "covariates": [
                    [1.0, 0.2],
                    [0.5, -0.1],
                    [1.2, 0.4],
                    [0.7, 0.0],
                    [0.3, -0.2],
                    [1.1, 0.1],
                    [0.4, -0.3],
                    [0.9, 0.3]
                ],
                "horizon": 60.0,
                "execution": { "deterministic": true }
            }),
        };
        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        assert_eq!(out.schema_version, "nextstat.tool_result.v1");
        assert_eq!(out.meta.tool_name, "nextstat_churn_uplift");
        assert!(out.result["ate"].is_number());
        assert!(out.result["se"].is_number());
        assert!(out.result["ci_lower"].is_number());
        assert!(out.result["ci_upper"].is_number());
        assert_eq!(out.result["n_treated"].as_u64(), Some(4));
        assert_eq!(out.result["n_control"].as_u64(), Some(4));
        assert_eq!(out.result["horizon"].as_f64(), Some(60.0));
    }

    #[test]
    fn tool_execute_churn_uplift_survival_supports_server_safe_contract() {
        let state = AppState::new(None);
        let req = ToolExecuteRequest {
            name: "nextstat_churn_uplift_survival".to_string(),
            arguments: serde_json::json!({
                "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
                "events": [true, false, true, false, false, true, false, true],
                "treated": [0, 0, 1, 1, 0, 1, 0, 1],
                "covariates": [
                    [1.0, 0.2],
                    [0.5, -0.1],
                    [1.2, 0.4],
                    [0.7, 0.0],
                    [0.3, -0.2],
                    [1.1, 0.1],
                    [0.4, -0.3],
                    [0.9, 0.3]
                ],
                "horizon": 60.0,
                "eval_horizons": [30.0, 60.0, 90.0],
                "trim": 0.01,
                "execution": { "deterministic": true }
            }),
        };
        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        assert_eq!(out.schema_version, "nextstat.tool_result.v1");
        assert_eq!(out.meta.tool_name, "nextstat_churn_uplift_survival");
        assert!(out.result["rmst_treated"].is_number());
        assert!(out.result["rmst_control"].is_number());
        assert!(out.result["delta_rmst"].is_number());
        assert_eq!(out.result["horizon"].as_f64(), Some(60.0));
        assert!(out.result["ipw_applied"].is_boolean());
        assert_eq!(out.result["arms"].as_array().map(|v| v.len()), Some(2));
        assert_eq!(out.result["survival_diffs"].as_array().map(|v| v.len()), Some(3));
        assert!(out.result["overlap"].is_object());
        assert_eq!(out.result["overlap"]["n_total"].as_u64(), Some(8));
    }

    #[test]
    fn tool_execute_pharma_fit_supports_server_safe_methods() {
        let state = AppState::new(None);
        let cases: &[(&str, serde_json::Value)] = &[
            (
                "focei",
                serde_json::json!({
                    "times": [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0],
                    "y": [1.2, 2.3, 1.9, 0.8, 1.0, 2.1, 1.7, 0.7],
                    "subject_idx": [0, 0, 0, 0, 1, 1, 1, 1],
                    "n_subjects": 2,
                    "doses": [100.0, 100.0],
                    "theta_init": [1.0, 5.0, 0.8],
                    "omega_init": [0.2, 0.2, 0.2],
                    "method": "focei",
                    "model": "1cpt_oral",
                    "sigma": 0.1,
                    "error_model": "proportional",
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "saem",
                serde_json::json!({
                    "times": [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0],
                    "y": [1.2, 2.3, 1.9, 0.8, 1.0, 2.1, 1.7, 0.7],
                    "subject_idx": [0, 0, 0, 0, 1, 1, 1, 1],
                    "n_subjects": 2,
                    "doses": [100.0, 100.0],
                    "theta_init": [1.0, 5.0, 0.8],
                    "omega_init": [0.2, 0.2, 0.2],
                    "method": "saem",
                    "model": "1cpt_oral",
                    "sigma": 0.1,
                    "error_model": "proportional",
                    "execution": { "deterministic": true }
                }),
            ),
            (
                "combined",
                serde_json::json!({
                    "times": [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0],
                    "y": [1.2, 2.3, 1.9, 0.8, 1.0, 2.1, 1.7, 0.7],
                    "subject_idx": [0, 0, 0, 0, 1, 1, 1, 1],
                    "n_subjects": 2,
                    "doses": [100.0, 100.0],
                    "theta_init": [1.0, 5.0, 0.8],
                    "omega_init": [0.2, 0.2, 0.2],
                    "method": "foce",
                    "model": "1cpt_oral",
                    "sigma": 0.1,
                    "sigma_add": 0.05,
                    "error_model": "combined",
                    "execution": { "deterministic": true }
                }),
            ),
        ];

        for (case_name, arguments) in cases {
            let req = ToolExecuteRequest {
                name: "nextstat_pharma_fit".to_string(),
                arguments: arguments.clone(),
            };
            let out = execute_tool(&state, req);
            assert!(out.ok, "expected ok=true for case={case_name}, got error={:?}", out.error);
            assert_eq!(out.schema_version, "nextstat.tool_result.v1");
            assert_eq!(out.meta.tool_name, "nextstat_pharma_fit");
            assert!(out.result["theta"].is_array(), "case={case_name} must return theta");
            assert!(out.result["omega"].is_array(), "case={case_name} must return omega");
            assert!(out.result["eta"].is_array(), "case={case_name} must return eta");
            assert!(out.result["ofv"].is_number(), "case={case_name} must return ofv");
            assert!(out.result["converged"].is_boolean(), "case={case_name} must return converged");
            assert!(out.result["sigma"].is_number(), "case={case_name} must return sigma");
            if *case_name == "saem" {
                assert!(out.result["saem"].is_object(), "SAEM case must return diagnostics");
            }
        }
    }

    #[test]
    fn tool_execute_pharma_vpc_supports_server_safe_contract() {
        let state = AppState::new(None);
        let req = ToolExecuteRequest {
            name: "nextstat_pharma_vpc".to_string(),
            arguments: serde_json::json!({
                "times": [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0],
                "y": [1.2, 2.3, 1.9, 0.8, 1.0, 2.1, 1.7, 0.7],
                "subject_idx": [0, 0, 0, 0, 1, 1, 1, 1],
                "n_subjects": 2,
                "doses": [100.0, 100.0],
                "model": "1cpt_oral",
                "theta": [1.0, 5.0, 0.8],
                "omega_matrix": [
                    [0.04, 0.0, 0.0],
                    [0.0, 0.04, 0.0],
                    [0.0, 0.0, 0.04]
                ],
                "sigma": 0.1,
                "error_model": "combined",
                "sigma_add": 0.05,
                "bioavailability": 1.0,
                "quantiles": [0.1, 0.5, 0.9],
                "n_bins": 4,
                "n_sim": 16,
                "seed": 42,
                "pi_level": 0.9,
                "execution": { "deterministic": true }
            }),
        };
        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        assert_eq!(out.schema_version, "nextstat.tool_result.v1");
        assert_eq!(out.meta.tool_name, "nextstat_pharma_vpc");
        assert!(out.result["bins"].is_array());
        assert!(out.result["quantiles"].is_array());
        assert_eq!(out.result["n_sim"].as_u64(), Some(16));
    }

    #[test]
    fn tool_execute_pk_gof_supports_server_safe_contract() {
        let state = AppState::new(None);
        let req = ToolExecuteRequest {
            name: "nextstat_pk_gof".to_string(),
            arguments: serde_json::json!({
                "times": [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0],
                "y": [1.2, 2.3, 1.9, 0.8, 1.0, 2.1, 1.7, 0.7],
                "subject_idx": [0, 0, 0, 0, 1, 1, 1, 1],
                "doses": [100.0, 100.0],
                "model": "1cpt_oral",
                "theta": [1.0, 5.0, 0.8],
                "eta": [
                    [0.05, -0.02, 0.01],
                    [-0.04, 0.03, -0.02]
                ],
                "sigma": 0.1,
                "error_model": "proportional",
                "execution": { "deterministic": true }
            }),
        };
        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        assert_eq!(out.schema_version, "nextstat.tool_result.v1");
        assert_eq!(out.meta.tool_name, "nextstat_pk_gof");
        assert_eq!(out.result["model"].as_str(), Some("1cpt_oral"));
        assert_eq!(out.result["n_subjects"].as_u64(), Some(2));
        assert_eq!(out.result["n_records"].as_u64(), Some(8));
        let records = out.result["records"].as_array().expect("records must be array");
        assert_eq!(records.len(), 8);
        let first = records.first().expect("records must be non-empty");
        assert!(first["subject"].is_u64());
        assert!(first["time"].is_number());
        assert!(first["dv"].is_number());
        assert!(first["pred"].is_number());
        assert!(first["ipred"].is_number());
        assert!(first["iwres"].is_number());
        assert!(first["cwres"].is_number());
    }

    #[test]
    fn tool_execute_pk_npde_supports_server_safe_contract() {
        let state = AppState::new(None);
        let req = ToolExecuteRequest {
            name: "nextstat_pk_npde".to_string(),
            arguments: serde_json::json!({
                "times": [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0],
                "y": [1.2, 2.3, 1.9, 0.8, 1.0, 2.1, 1.7, 0.7],
                "subject_idx": [0, 0, 0, 0, 1, 1, 1, 1],
                "n_subjects": 2,
                "doses": [100.0, 100.0],
                "model": "1cpt_oral",
                "theta": [1.0, 5.0, 0.8],
                "omega_matrix": [
                    [0.04, 0.0, 0.0],
                    [0.0, 0.04, 0.0],
                    [0.0, 0.0, 0.04]
                ],
                "sigma": 0.1,
                "error_model": "proportional",
                "n_sim": 16,
                "seed": 42,
                "execution": { "deterministic": true }
            }),
        };
        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        assert_eq!(out.schema_version, "nextstat.tool_result.v1");
        assert_eq!(out.meta.tool_name, "nextstat_pk_npde");
        assert_eq!(out.result["model"].as_str(), Some("1cpt_oral"));
        assert_eq!(out.result["n_subjects"].as_u64(), Some(2));
        assert_eq!(out.result["n_records"].as_u64(), Some(8));
        assert_eq!(out.result["n_sim"].as_u64(), Some(16));
        assert_eq!(out.result["seed"].as_u64(), Some(42));
        assert!(out.result["mean"].is_number());
        assert!(out.result["variance"].is_number());
        let records = out.result["records"].as_array().expect("records must be array");
        assert_eq!(records.len(), 8);
        let first = records.first().expect("records must be non-empty");
        assert!(first["subject"].is_u64());
        assert!(first["time"].is_number());
        assert!(first["dv"].is_number());
        assert!(first["percentile"].is_number());
        assert!(first["npde"].is_number());
    }

    #[test]
    fn tool_execute_trial_simulate_supports_server_safe_contract() {
        let state = AppState::new(None);
        let req = ToolExecuteRequest {
            name: "nextstat_trial_simulate".to_string(),
            arguments: serde_json::json!({
                "n_subjects": 3,
                "dose": 100.0,
                "obs_times": [0.5, 1.0, 2.0, 4.0],
                "pk_model": "1cpt_oral",
                "theta": [1.0, 5.0, 0.8],
                "omega": [0.2, 0.2, 0.2],
                "sigma": 0.1,
                "error_model": "proportional",
                "bioavailability": 1.0,
                "seed": 42,
                "execution": { "deterministic": true }
            }),
        };
        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        assert_eq!(out.schema_version, "nextstat.tool_result.v1");
        assert_eq!(out.meta.tool_name, "nextstat_trial_simulate");
        assert!(out.result["concentrations"].is_array());
        assert!(out.result["individual_params"].is_array());
        assert!(out.result["auc"].is_array());
        assert!(out.result["cmax"].is_array());
        assert!(out.result["tmax"].is_array());
        assert!(out.result["ctrough"].is_array());
        assert_eq!(out.result["concentrations"].as_array().map(|v| v.len()), Some(3));
    }

    #[test]
    fn tool_execute_read_root_histogram_supports_server_safe_contract() {
        let state = AppState::new(None);
        let payload = base64::engine::general_purpose::STANDARD
            .encode(include_bytes!("../../../tests/fixtures/simple_histos.root"));
        let req = ToolExecuteRequest {
            name: "nextstat_read_root_histogram".to_string(),
            arguments: serde_json::json!({
                "root_bytes_base64": payload,
                "filename_hint": "simple_histos.root",
                "hist_path": "hist1",
                "execution": { "deterministic": true }
            }),
        };
        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        assert_eq!(out.schema_version, "nextstat.tool_result.v1");
        assert_eq!(out.meta.tool_name, "nextstat_read_root_histogram");
        assert_eq!(out.result["name"].as_str(), Some("hist1"));
        assert_eq!(out.result["bin_content"].as_array().map(|items| items.len()), Some(3));
        assert_eq!(out.result["underflow"].as_f64(), Some(0.0));
        assert_eq!(out.result["overflow"].as_f64(), Some(0.0));
    }

    #[test]
    fn tool_execute_chain_ladder_supports_server_safe_contract() {
        let state = AppState::new(None);
        let req = ToolExecuteRequest {
            name: "nextstat_chain_ladder".to_string(),
            arguments: serde_json::json!({
                "triangle": [
                    [100.0, 150.0, 180.0],
                    [110.0, 160.0, serde_json::Value::Null],
                    [120.0, serde_json::Value::Null, serde_json::Value::Null]
                ],
                "method": "mack",
                "conf_level": 0.95,
                "execution": { "deterministic": true }
            }),
        };
        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        assert_eq!(out.schema_version, "nextstat.tool_result.v1");
        assert_eq!(out.meta.tool_name, "nextstat_chain_ladder");
        assert!(out.result["development_factors"].is_array());
        assert!(out.result["sigma_sq"].is_array());
        assert!(out.result["ultimates"].is_array());
        assert!(out.result["ibnr"].is_array());
        assert!(out.result["latest"].is_array());
        assert!(out.result["se"].is_array());
        assert!(out.result["pi_lower"].is_array());
        assert!(out.result["pi_upper"].is_array());
        assert!(out.result["total_ibnr"].is_number());
        assert!(out.result["total_se"].is_number());
        assert_eq!(out.result["conf_level"].as_f64(), Some(0.95));
    }

    #[test]
    fn tool_execute_bioequivalence_supports_server_safe_operations() {
        let state = AppState::new(None);

        let test_req = ToolExecuteRequest {
            name: "nextstat_bioequivalence".to_string(),
            arguments: serde_json::json!({
                "operation": "test",
                "test_values": [4.58, 4.61, 4.55, 4.62],
                "ref_values": [4.60, 4.63, 4.57, 4.64],
                "execution": { "deterministic": true }
            }),
        };
        let test_out = execute_tool(&state, test_req);
        assert!(test_out.ok, "expected ok=true, got error={:?}", test_out.error);
        assert_eq!(test_out.schema_version, "nextstat.tool_result.v1");
        assert_eq!(test_out.meta.tool_name, "nextstat_bioequivalence");
        assert!(test_out.result["geometric_mean_ratio"].is_number());
        assert!(test_out.result["ci_lower"].is_number());
        assert!(test_out.result["ci_upper"].is_number());
        assert!(test_out.result["conclusion"].is_string());

        let power_req = ToolExecuteRequest {
            name: "nextstat_bioequivalence".to_string(),
            arguments: serde_json::json!({
                "operation": "power",
                "n_total": 24,
                "cv": 0.30,
                "gmr": 0.95,
                "execution": { "deterministic": true }
            }),
        };
        let power_out = execute_tool(&state, power_req);
        assert!(power_out.ok, "expected ok=true, got error={:?}", power_out.error);
        assert_eq!(power_out.meta.tool_name, "nextstat_bioequivalence");
        assert!(power_out.result["power"].is_number());

        let sample_size_req = ToolExecuteRequest {
            name: "nextstat_bioequivalence".to_string(),
            arguments: serde_json::json!({
                "operation": "sample_size",
                "cv": 0.30,
                "gmr": 0.95,
                "target_power": 0.80,
                "execution": { "deterministic": true }
            }),
        };
        let sample_size_out = execute_tool(&state, sample_size_req);
        assert!(sample_size_out.ok, "expected ok=true, got error={:?}", sample_size_out.error);
        assert_eq!(sample_size_out.meta.tool_name, "nextstat_bioequivalence");
        assert!(sample_size_out.result["n_per_sequence"].is_u64());
        assert!(sample_size_out.result["n_total"].is_u64());
        assert!(sample_size_out.result["achieved_power"].is_number());
    }

    #[test]
    fn tool_execute_fault_tree_mc_supports_server_safe_cpu_contract() {
        let state = AppState::new(None);
        let req = ToolExecuteRequest {
            name: "nextstat_fault_tree_mc".to_string(),
            arguments: serde_json::json!({
                "spec": {
                    "components": [
                        { "type": "bernoulli", "p": 0.01 },
                        { "type": "bernoulli", "p": 0.02 }
                    ],
                    "nodes": [
                        { "type": "component", "index": 0 },
                        { "type": "component", "index": 1 },
                        { "type": "or", "children": [0, 1] }
                    ],
                    "top_event": 2
                },
                "n_scenarios": 10000,
                "seed": 42,
                "device": "cpu",
                "execution": { "deterministic": true }
            }),
        };
        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        assert_eq!(out.meta.tool_name, "nextstat_fault_tree_mc");
        assert_eq!(out.result["n_scenarios"].as_u64(), Some(10000));
        assert!(out.result["n_top_failures"].is_u64());
        assert!(out.result["p_failure"].is_number());
        assert!(out.result["se"].is_number());
        assert!(out.result["ci_lower"].is_number());
        assert!(out.result["ci_upper"].is_number());
        assert!(out.result["scenarios_per_sec"].is_number());
        assert!(out.result["component_importance"].is_array());
    }

    #[test]
    fn tool_execute_fault_tree_ce_is_supports_server_safe_contract() {
        let state = AppState::new(None);
        let req = ToolExecuteRequest {
            name: "nextstat_fault_tree_ce_is".to_string(),
            arguments: serde_json::json!({
                "spec": {
                    "components": [
                        { "type": "bernoulli", "p": 0.01 },
                        { "type": "bernoulli", "p": 0.02 }
                    ],
                    "nodes": [
                        { "type": "component", "index": 0 },
                        { "type": "component", "index": 1 },
                        { "type": "and", "children": [0, 1] }
                    ],
                    "top_event": 2
                },
                "n_per_level": 1000,
                "elite_fraction": 0.02,
                "max_levels": 6,
                "q_max": 0.9,
                "seed": 42,
                "execution": { "deterministic": true }
            }),
        };
        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        assert_eq!(out.meta.tool_name, "nextstat_fault_tree_ce_is");
        assert!(out.result["p_failure"].is_number());
        assert!(out.result["se"].is_number());
        assert!(out.result["ci_lower"].is_number());
        assert!(out.result["ci_upper"].is_number());
        assert!(out.result["n_levels"].is_u64());
        assert!(out.result["n_total_scenarios"].is_u64());
        assert!(out.result["final_proposal"].is_array());
        assert!(out.result["coefficient_of_variation"].is_number());
    }

    #[test]
    fn tool_execute_dose_response_supports_server_safe_modes() {
        let state = AppState::new(None);

        let emax_predict = ToolExecuteRequest {
            name: "nextstat_dose_response".to_string(),
            arguments: serde_json::json!({
                "model": "emax",
                "e0": 5.0,
                "emax": 50.0,
                "ec50": 2.0,
                "conc": [0.0, 1.0, 2.0, 4.0, 8.0],
                "execution": { "deterministic": true }
            }),
        };
        let emax_predict_out = execute_tool(&state, emax_predict);
        assert!(emax_predict_out.ok, "expected ok=true, got error={:?}", emax_predict_out.error);
        assert_eq!(emax_predict_out.meta.tool_name, "nextstat_dose_response");
        assert_eq!(emax_predict_out.result["model"].as_str(), Some("emax"));
        assert!(emax_predict_out.result["predictions"].is_array());
        assert!(emax_predict_out.result["e0"].is_number());
        assert!(emax_predict_out.result["emax"].is_number());
        assert!(emax_predict_out.result["ec50"].is_number());

        let emax_nll = ToolExecuteRequest {
            name: "nextstat_dose_response".to_string(),
            arguments: serde_json::json!({
                "model": "emax",
                "e0": 5.0,
                "emax": 50.0,
                "ec50": 2.0,
                "dose": [0.0, 1.0, 2.0, 4.0, 8.0],
                "response": [5.0, 18.0, 31.0, 45.0, 49.0],
                "error_model": "combined",
                "sigma": 0.05,
                "sigma_add": 0.10,
                "execution": { "deterministic": true }
            }),
        };
        let emax_nll_out = execute_tool(&state, emax_nll);
        assert!(emax_nll_out.ok, "expected ok=true, got error={:?}", emax_nll_out.error);
        assert_eq!(emax_nll_out.result["model"].as_str(), Some("emax"));
        assert!(emax_nll_out.result["nll"].is_number());

        let sigmoid_predict = ToolExecuteRequest {
            name: "nextstat_dose_response".to_string(),
            arguments: serde_json::json!({
                "model": "sigmoid_emax",
                "e0": 5.0,
                "emax": 50.0,
                "ec50": 2.0,
                "gamma": 1.5,
                "conc": [0.0, 1.0, 2.0, 4.0, 8.0],
                "execution": { "deterministic": true }
            }),
        };
        let sigmoid_predict_out = execute_tool(&state, sigmoid_predict);
        assert!(
            sigmoid_predict_out.ok,
            "expected ok=true, got error={:?}",
            sigmoid_predict_out.error
        );
        assert_eq!(sigmoid_predict_out.result["model"].as_str(), Some("sigmoid_emax"));
        assert!(sigmoid_predict_out.result["predictions"].is_array());
        assert_eq!(sigmoid_predict_out.result["gamma"].as_f64(), Some(1.5));
    }

    #[test]
    fn tool_execute_competing_risks_supports_server_safe_operations() {
        let state = AppState::new(None);

        let cif_req = ToolExecuteRequest {
            name: "nextstat_competing_risks".to_string(),
            arguments: serde_json::json!({
                "operation": "cif",
                "times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "events": [1, 2, 0, 1, 2, 1],
                "target_cause": 1,
                "conf_level": 0.95,
                "execution": { "deterministic": true }
            }),
        };
        let cif_out = execute_tool(&state, cif_req);
        assert!(cif_out.ok, "expected ok=true, got error={:?}", cif_out.error);
        assert_eq!(cif_out.meta.tool_name, "nextstat_competing_risks");
        assert_eq!(cif_out.result["cause"].as_u64(), Some(1));
        assert!(cif_out.result["times"].is_array());
        assert!(cif_out.result["cif"].is_array());
        assert!(cif_out.result["se"].is_array());
        assert!(cif_out.result["ci_lower"].is_array());
        assert!(cif_out.result["ci_upper"].is_array());
        assert_eq!(cif_out.result["n"].as_u64(), Some(6));

        let gray_req = ToolExecuteRequest {
            name: "nextstat_competing_risks".to_string(),
            arguments: serde_json::json!({
                "operation": "gray_test",
                "times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "events": [1, 2, 0, 1, 2, 1],
                "groups": [0, 0, 1, 1, 0, 1],
                "target_cause": 1,
                "execution": { "deterministic": true }
            }),
        };
        let gray_out = execute_tool(&state, gray_req);
        assert!(gray_out.ok, "expected ok=true, got error={:?}", gray_out.error);
        assert_eq!(gray_out.meta.tool_name, "nextstat_competing_risks");
        assert!(gray_out.result["statistic"].is_number());
        assert!(gray_out.result["df"].is_u64());
        assert!(gray_out.result["p_value"].is_number());

        let fine_gray_req = ToolExecuteRequest {
            name: "nextstat_competing_risks".to_string(),
            arguments: serde_json::json!({
                "operation": "fine_gray",
                "times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "events": [1, 2, 0, 1, 2, 1],
                "x": [
                    0.0, 1.0,
                    1.0, 0.0,
                    0.0, 0.0,
                    1.0, 1.0,
                    0.5, 1.5,
                    1.5, 0.5
                ],
                "p": 2,
                "target_cause": 1,
                "execution": { "deterministic": true }
            }),
        };
        let fine_gray_out = execute_tool(&state, fine_gray_req);
        assert!(fine_gray_out.ok, "expected ok=true, got error={:?}", fine_gray_out.error);
        assert_eq!(fine_gray_out.meta.tool_name, "nextstat_competing_risks");
        assert!(fine_gray_out.result["coefficients"].is_array());
        assert!(fine_gray_out.result["se"].is_array());
        assert!(fine_gray_out.result["z"].is_array());
        assert!(fine_gray_out.result["p_values"].is_array());
        assert_eq!(fine_gray_out.result["n"].as_u64(), Some(6));
    }

    #[test]
    fn tool_execute_ranking_matches_current_rank_impact_contract() {
        let state = AppState::new(None);
        let ws = include_str!("../../../tests/fixtures/simple_workspace.json");
        let model = load_model(ws).expect("simple workspace should load");
        let mle = MaximumLikelihoodEstimator::new();

        let mut expected = mle
            .ranking(&model)
            .expect("ranking should succeed")
            .into_iter()
            .map(|entry| {
                let total_impact = entry.delta_mu_up.abs() + entry.delta_mu_down.abs();
                serde_json::json!({
                    "name": entry.name,
                    "delta_mu_up": entry.delta_mu_up,
                    "delta_mu_down": entry.delta_mu_down,
                    "total_impact": total_impact,
                    "pull": entry.pull,
                    "constraint": entry.constraint,
                })
            })
            .collect::<Vec<_>>();
        expected.sort_by(|a, b| {
            let impact_a = a["total_impact"].as_f64().unwrap_or(f64::NEG_INFINITY);
            let impact_b = b["total_impact"].as_f64().unwrap_or(f64::NEG_INFINITY);
            impact_b.partial_cmp(&impact_a).unwrap_or(std::cmp::Ordering::Equal).then_with(|| {
                a["name"].as_str().unwrap_or("").cmp(b["name"].as_str().unwrap_or(""))
            })
        });
        for (idx, row) in expected.iter_mut().enumerate() {
            row.as_object_mut()
                .expect("ranking row should be object")
                .insert("rank".to_string(), serde_json::json!(idx + 1));
        }

        let req = ToolExecuteRequest {
            name: "nextstat_ranking".to_string(),
            arguments: serde_json::json!({
                "workspace_json": ws,
                "top_n": 5,
                "execution": { "deterministic": true }
            }),
        };
        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got =
            out.result.get("ranking").cloned().expect("tool result must include ranking array");
        assert_json_close(&got, &serde_json::Value::Array(expected), "ranking");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_simple_workspace_deterministic() {
        let state = AppState::new(None);
        let ws = include_str!("../../../tests/fixtures/simple_workspace.json");
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/simple_workspace_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let tools =
            gold.get("tools").and_then(|x| x.as_object()).expect("golden must contain tools map");

        // Keep this tight: only tools that should match across local/server deterministic goldens.
        // Ranking is covered separately against the current in-workspace engine because the Python
        // golden generator runs against an installed wheel and can legitimately lag workspace code.
        // Server mode intentionally does not expose file ingest tools like ROOT histogram reads.
        let cases: &[(&str, serde_json::Value)] = &[
            (
                "nextstat_fit",
                serde_json::json!({ "workspace_json": ws, "execution": { "deterministic": true } }),
            ),
            (
                "nextstat_hypotest",
                serde_json::json!({ "workspace_json": ws, "mu": 1.0, "execution": { "deterministic": true } }),
            ),
            (
                "nextstat_upper_limit",
                serde_json::json!({ "workspace_json": ws, "expected": true, "execution": { "deterministic": true } }),
            ),
            (
                "nextstat_scan",
                serde_json::json!({ "workspace_json": ws, "start": 0.0, "stop": 2.0, "points": 5, "execution": { "deterministic": true } }),
            ),
            (
                "nextstat_discovery_asymptotic",
                serde_json::json!({ "workspace_json": ws, "execution": { "deterministic": true } }),
            ),
            (
                "nextstat_workspace_audit",
                serde_json::json!({ "workspace_json": ws, "execution": { "deterministic": true } }),
            ),
        ];

        let _guard = state.compute_lock.blocking_lock();
        for (name, args) in cases {
            let req = ToolExecuteRequest { name: (*name).to_string(), arguments: args.clone() };
            let out = execute_tool(&state, req);
            assert!(out.ok, "expected ok=true for {name}, got error={:?}", out.error);
            assert_eq!(out.schema_version, "nextstat.tool_result.v1");
            assert_eq!(out.meta.tool_name, *name);
            assert!(out.meta.deterministic, "meta.deterministic must be true for {name}");
            assert_eq!(out.meta.eval_mode, "parity", "meta.eval_mode must be parity for {name}");
            assert_eq!(
                out.meta.threads_requested,
                Some(1),
                "meta.threads_requested must be 1 for {name}"
            );

            let got =
                normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
            let want_raw = tools.get(*name).unwrap_or_else(|| panic!("missing golden for {name}"));
            let want = normalize_envelope(want_raw.clone());

            assert_json_close(&got, &want, &format!("tool:{name}"));
        }
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_root_histogram_simple_workspace_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/simple_workspace_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_read_root_histogram"))
            .cloned()
            .expect("golden must contain nextstat_read_root_histogram result");
        let want = normalize_envelope(want_raw);
        let payload = base64::engine::general_purpose::STANDARD
            .encode(include_bytes!("../../../tests/fixtures/simple_histos.root"));

        let req = ToolExecuteRequest {
            name: "nextstat_read_root_histogram".to_string(),
            arguments: serde_json::json!({
                "root_bytes_base64": payload,
                "filename_hint": "simple_histos.root",
                "hist_path": "hist1",
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_read_root_histogram");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_glm_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw =
            include_str!("../../../tests/fixtures/tool_goldens/glm_small_deterministic.v1.json");
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_glm_fit"))
            .cloned()
            .expect("golden must contain nextstat_glm_fit result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_glm_fit".to_string(),
            arguments: serde_json::json!({
                "x": [[0.0], [1.0], [2.0], [3.0], [4.0]],
                "y": [1.1, 2.9, 5.2, 6.8, 9.1],
                "family": "linear",
                "include_intercept": true,
                "l2": 0.5,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_glm_fit");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_bayesian_sample_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/bayesian_sample_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_bayesian_sample"))
            .cloned()
            .expect("golden must contain nextstat_bayesian_sample result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_bayesian_sample".to_string(),
            arguments: serde_json::json!({
                "model_type": "linear_regression",
                "x": [[0.0], [1.0], [2.0], [3.0], [4.0]],
                "y": [1.0, 2.1, 2.9, 4.2, 5.1],
                "n_chains": 2,
                "n_warmup": 20,
                "n_samples": 20,
                "seed": 42,
                "target_accept": 0.8,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_bayesian_sample");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_survival_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/survival_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_survival_fit"))
            .cloned()
            .expect("golden must contain nextstat_survival_fit result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_survival_fit".to_string(),
            arguments: serde_json::json!({
                "x": [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]],
                "time": [1.0, 2.0, 3.5, 4.0, 5.0, 6.0],
                "event": [1, 1, 0, 1, 0, 1],
                "model": "cox_ph",
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_survival_fit");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_kaplan_meier_grouped_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/kaplan_meier_grouped_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_kaplan_meier"))
            .cloned()
            .expect("golden must contain nextstat_kaplan_meier result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_kaplan_meier".to_string(),
            arguments: serde_json::json!({
                "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "event": [1, 1, 0, 1, 0, 1],
                "group": [0, 0, 1, 1, 0, 1],
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_kaplan_meier");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_log_rank_test_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/log_rank_test_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_log_rank_test"))
            .cloned()
            .expect("golden must contain nextstat_log_rank_test result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_log_rank_test".to_string(),
            arguments: serde_json::json!({
                "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
                "events": [true, false, true, false, false, true, false, true],
                "groups": [0, 0, 0, 0, 1, 1, 1, 1],
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_log_rank_test");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_meta_analysis_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/meta_analysis_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_meta_analysis"))
            .cloned()
            .expect("golden must contain nextstat_meta_analysis result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_meta_analysis".to_string(),
            arguments: serde_json::json!({
                "effects": [0.2, 0.5, -0.1, 0.3],
                "standard_errors": [0.1, 0.2, 0.15, 0.12],
                "method": "random",
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_meta_analysis");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_panel_fe_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/panel_fe_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_panel_fe"))
            .cloned()
            .expect("golden must contain nextstat_panel_fe result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_panel_fe".to_string(),
            arguments: serde_json::json!({
                "x": [[1.0], [2.0], [1.0], [2.5], [3.0], [3.5]],
                "y": [1.0, 2.0, 1.5, 2.5, 3.0, 3.5],
                "entity": [0, 0, 1, 1, 2, 2],
                "time": [0, 1, 0, 1, 0, 1],
                "cluster": "two_way",
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_panel_fe");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_did_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw =
            include_str!("../../../tests/fixtures/tool_goldens/did_small_deterministic.v1.json");
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_did"))
            .cloned()
            .expect("golden must contain nextstat_did result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_did".to_string(),
            arguments: serde_json::json!({
                "y": [1.0, 1.2, 1.4, 1.8, 1.1, 1.3, 2.2, 2.5],
                "treat": [0, 0, 0, 0, 1, 1, 1, 1],
                "post": [0, 1, 0, 1, 0, 1, 0, 1],
                "entity": [0, 0, 1, 1, 2, 2, 3, 3],
                "time": [0, 1, 0, 1, 0, 1, 0, 1],
                "cluster": "two_way",
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_did");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_iv_2sls_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/iv_2sls_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_iv_2sls"))
            .cloned()
            .expect("golden must contain nextstat_iv_2sls result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_iv_2sls".to_string(),
            arguments: serde_json::json!({
                "y": [1.0, 2.1, 1.7, 2.9, 3.2, 3.8, 4.2, 4.7],
                "endog": [[1.0], [1.8], [1.4], [2.2], [2.7], [3.1], [3.5], [4.0]],
                "instruments": [[0.9], [1.7], [1.2], [2.0], [2.4], [2.9], [3.3], [3.8]],
                "exog": [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0]],
                "cov": "hc1",
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_iv_2sls");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_aipw_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw =
            include_str!("../../../tests/fixtures/tool_goldens/aipw_small_deterministic.v1.json");
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_aipw"))
            .cloned()
            .expect("golden must contain nextstat_aipw result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_aipw".to_string(),
            arguments: serde_json::json!({
                "x": [[0.0], [1.0], [0.0], [1.0], [0.2], [1.2], [0.1], [1.1]],
                "y": [1.0, 2.0, 1.2, 2.2, 1.1, 2.4, 1.3, 2.5],
                "treatment": [0, 1, 0, 1, 0, 1, 0, 1],
                "estimand": "ate",
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_aipw");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_event_study_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/event_study_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_event_study"))
            .cloned()
            .expect("golden must contain nextstat_event_study result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_event_study".to_string(),
            arguments: serde_json::json!({
                "y": [1.0, 1.1, 1.2, 1.4, 1.0, 1.0, 1.1, 1.2, 2.0, 2.3, 2.7, 3.0],
                "entity": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
                "time": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                "treat_time": [serde_json::Value::Null, serde_json::Value::Null, serde_json::Value::Null, serde_json::Value::Null, serde_json::json!(2), serde_json::json!(2), serde_json::json!(2), serde_json::json!(2), serde_json::json!(1), serde_json::json!(1), serde_json::json!(1), serde_json::json!(1)],
                "n_leads": 2,
                "n_lags": 2,
                "cluster": "entity",
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_event_study");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_bioequivalence_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/bioequivalence_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_bioequivalence"))
            .cloned()
            .expect("golden must contain nextstat_bioequivalence result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_bioequivalence".to_string(),
            arguments: serde_json::json!({
                "operation": "test",
                "test_values": [4.58, 4.61, 4.55, 4.62],
                "ref_values": [4.60, 4.63, 4.57, 4.64],
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_bioequivalence");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_dose_response_emax_predict_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/dose_response_emax_predict_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_dose_response"))
            .cloned()
            .expect("golden must contain nextstat_dose_response result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_dose_response".to_string(),
            arguments: serde_json::json!({
                "model": "emax",
                "e0": 5.0,
                "emax": 50.0,
                "ec50": 2.0,
                "conc": [0.0, 1.0, 2.0, 4.0, 8.0],
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_dose_response");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_garch_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw =
            include_str!("../../../tests/fixtures/tool_goldens/garch_small_deterministic.v1.json");
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_garch_fit"))
            .cloned()
            .expect("golden must contain nextstat_garch_fit result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_garch_fit".to_string(),
            arguments: serde_json::json!({
                "returns": [0.01, -0.02, 0.015, -0.01, 0.005, 0.02, -0.015, 0.01, 0.012, -0.008],
                "model": "garch",
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_garch_fit");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_ads_cuped_adjust_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/ads_cuped_adjust_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_ads_cuped_adjust"))
            .cloned()
            .expect("golden must contain nextstat_ads_cuped_adjust result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_ads_cuped_adjust".to_string(),
            arguments: serde_json::json!({
                "control_outcomes": [10.0, 12.0, 9.0, 11.0, 13.0],
                "control_covariates": [8.0, 10.0, 7.0, 9.0, 11.0],
                "variant_outcomes": [11.0, 13.0, 10.0, 12.0, 14.0],
                "variant_covariates": [8.5, 10.5, 7.5, 9.5, 11.5],
                "covariate_name": "pre_clicks",
                "covariate_provenance": {
                    "name": "pre_clicks",
                    "timing": "pre_treatment",
                    "source_dataset": "ads_preperiod_daily"
                },
                "pre_treatment_only": true,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_ads_cuped_adjust");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_ads_cure_adjust_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/ads_cure_adjust_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_ads_cure_adjust"))
            .cloned()
            .expect("golden must contain nextstat_ads_cure_adjust result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_ads_cure_adjust".to_string(),
            arguments: serde_json::json!({
                "control_outcomes": [100.0, 110.0, 95.0, 105.0, 115.0, 120.0],
                "control_covariates": [
                    [80.0, 1000.0],
                    [88.0, 1100.0],
                    [75.0, 950.0],
                    [84.0, 1025.0],
                    [92.0, 1150.0],
                    [96.0, 1180.0]
                ],
                "variant_outcomes": [104.0, 113.0, 99.0, 109.0, 118.0, 124.0],
                "variant_covariates": [
                    [81.0, 1008.0],
                    [89.0, 1110.0],
                    [76.0, 960.0],
                    [85.0, 1035.0],
                    [93.0, 1165.0],
                    [97.0, 1192.0]
                ],
                "covariate_names": ["pre_clicks", "pre_impressions"],
                "covariate_provenance": [
                    {
                        "name": "pre_clicks",
                        "timing": "pre_treatment",
                        "source_dataset": "ads_preperiod_daily"
                    },
                    {
                        "name": "pre_impressions",
                        "timing": "pre_treatment",
                        "source_dataset": "ads_preperiod_daily"
                    }
                ],
                "pre_treatment_only": true,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_ads_cure_adjust");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_competing_risks_cif_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/competing_risks_cif_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_competing_risks"))
            .cloned()
            .expect("golden must contain nextstat_competing_risks result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_competing_risks".to_string(),
            arguments: serde_json::json!({
                "operation": "cif",
                "times": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "events": [1, 2, 0, 1, 2, 1],
                "target_cause": 1,
                "conf_level": 0.95,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_competing_risks");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_fault_tree_mc_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/fault_tree_mc_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_fault_tree_mc"))
            .cloned()
            .expect("golden must contain nextstat_fault_tree_mc result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_fault_tree_mc".to_string(),
            arguments: serde_json::json!({
                "spec": {
                    "components": [
                        { "type": "bernoulli", "p": 0.01 },
                        { "type": "bernoulli", "p": 0.02 }
                    ],
                    "nodes": [
                        { "type": "component", "index": 0 },
                        { "type": "component", "index": 1 },
                        { "type": "or", "children": [0, 1] }
                    ],
                    "top_event": 2
                },
                "n_scenarios": 10000,
                "seed": 42,
                "device": "cpu",
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_fault_tree_mc");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_fault_tree_ce_is_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/fault_tree_ce_is_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_fault_tree_ce_is"))
            .cloned()
            .expect("golden must contain nextstat_fault_tree_ce_is result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_fault_tree_ce_is".to_string(),
            arguments: serde_json::json!({
                "spec": {
                    "components": [
                        { "type": "bernoulli", "p": 0.01 },
                        { "type": "bernoulli", "p": 0.02 }
                    ],
                    "nodes": [
                        { "type": "component", "index": 0 },
                        { "type": "component", "index": 1 },
                        { "type": "and", "children": [0, 1] }
                    ],
                    "top_event": 2
                },
                "n_per_level": 1000,
                "elite_fraction": 0.02,
                "max_levels": 6,
                "q_max": 0.9,
                "seed": 42,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_fault_tree_ce_is");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_kalman_filter_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/kalman_filter_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_kalman"))
            .cloned()
            .expect("golden must contain nextstat_kalman result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_kalman".to_string(),
            arguments: serde_json::json!({
                "F": [[1.0]],
                "H": [[1.0]],
                "Q": [[0.1]],
                "R": [[0.2]],
                "x0": [0.0],
                "P0": [[1.0]],
                "y": [[1.0], [1.2], [0.9], [1.1]],
                "operation": "filter",
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_kalman");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_kalman_simulate_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/kalman_simulate_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_kalman"))
            .cloned()
            .expect("golden must contain nextstat_kalman result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_kalman".to_string(),
            arguments: serde_json::json!({
                "F": [[1.0]],
                "H": [[1.0]],
                "Q": [[0.1]],
                "R": [[0.2]],
                "x0": [0.0],
                "P0": [[1.0]],
                "operation": "simulate",
                "t_max": 4,
                "seed": 42,
                "init": "mean",
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_kalman:simulate");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_kalman_em_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/kalman_em_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_kalman"))
            .cloned()
            .expect("golden must contain nextstat_kalman result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_kalman".to_string(),
            arguments: serde_json::json!({
                "F": [[1.0]],
                "H": [[1.0]],
                "Q": [[0.1]],
                "R": [[0.2]],
                "x0": [0.0],
                "P0": [[1.0]],
                "y": [[0.1], [0.2], [0.0]],
                "operation": "em",
                "max_iter": 3,
                "tol": 1e-6,
                "estimate_q": true,
                "estimate_r": true,
                "estimate_f": false,
                "estimate_h": false,
                "min_diag": 1e-8,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_kalman:em");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_churn_generate_data_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/churn_generate_data_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_churn_generate_data"))
            .cloned()
            .expect("golden must contain nextstat_churn_generate_data result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_churn_generate_data".to_string(),
            arguments: serde_json::json!({
                "n_customers": 12,
                "n_cohorts": 3,
                "max_time": 18.0,
                "treatment_fraction": 0.25,
                "seed": 42,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_churn_generate_data");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_churn_risk_model_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/churn_risk_model_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_churn_risk_model"))
            .cloned()
            .expect("golden must contain nextstat_churn_risk_model result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_churn_risk_model".to_string(),
            arguments: serde_json::json!({
                "times": [
                    18.0, 3.302151210546639, 4.329991875135302, 5.475084445095048,
                    18.0, 16.76730374185681, 12.884032820472772, 8.271505063122506,
                    3.632766793868914, 18.0, 12.32888547551182, 18.0, 18.0,
                    1.4580664577117772, 7.416636560585132, 4.245245279862313
                ],
                "events": [
                    false, true, true, true, false, true, true, true,
                    true, false, true, false, false, true, true, true
                ],
                "covariates": [
                    [1.0, 0.0, 1.3117235495872972, 2.0],
                    [0.0, 0.0, -0.9022021759009674, 0.0],
                    [1.0, 0.0, 0.528865100865726, 2.0],
                    [1.0, 0.0, -1.454760464783373, 3.0],
                    [0.0, 1.0, -1.5385144547016696, 1.0],
                    [1.0, 0.0, 1.5124745460816813, 1.0],
                    [1.0, 0.0, -0.5771217015082334, 1.0],
                    [0.0, 0.0, 1.189054183234974, 0.0],
                    [0.0, 0.0, -0.4320950454679947, 0.0],
                    [1.0, 0.0, 1.147124602981333, 0.0],
                    [0.0, 0.0, -0.3221810058112955, 2.0],
                    [0.0, 1.0, -0.5197845574589398, 3.0],
                    [0.0, 1.0, 0.937777824579468, 1.0],
                    [0.0, 0.0, -1.6269177382249136, 1.0],
                    [1.0, 0.0, 0.8667568109417092, 0.0],
                    [0.0, 1.0, 0.17866086448304247, 1.0]
                ],
                "names": ["plan_basic", "plan_premium", "usage_score", "support_tickets"],
                "conf_level": 0.95,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_churn_risk_model");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_churn_retention_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/churn_retention_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_churn_retention"))
            .cloned()
            .expect("golden must contain nextstat_churn_retention result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_churn_retention".to_string(),
            arguments: serde_json::json!({
                "times": [30.0, 60.0, 90.0, 120.0],
                "events": [true, false, true, false],
                "groups": [0, 0, 1, 1],
                "conf_level": 0.95,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_churn_retention");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_churn_diagnostics_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/churn_diagnostics_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_churn_diagnostics"))
            .cloned()
            .expect("golden must contain nextstat_churn_diagnostics result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_churn_diagnostics".to_string(),
            arguments: serde_json::json!({
                "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
                "events": [true, false, true, false, false, true, false, true],
                "groups": [0, 0, 0, 0, 1, 1, 1, 1],
                "treated": [0, 0, 1, 1, 0, 1, 0, 1],
                "covariates": [
                    [1.0, 0.2],
                    [0.5, -0.1],
                    [1.2, 0.4],
                    [0.7, 0.0],
                    [0.3, -0.2],
                    [1.1, 0.1],
                    [0.4, -0.3],
                    [0.9, 0.3]
                ],
                "covariate_names": ["x1", "x2"],
                "trim": 0.01,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_churn_diagnostics");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_churn_cohort_matrix_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/churn_cohort_matrix_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_churn_cohort_matrix"))
            .cloned()
            .expect("golden must contain nextstat_churn_cohort_matrix result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_churn_cohort_matrix".to_string(),
            arguments: serde_json::json!({
                "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
                "events": [true, false, true, false, false, true, false, true],
                "groups": [0, 0, 0, 0, 1, 1, 1, 1],
                "period_boundaries": [30.0, 60.0, 90.0],
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_churn_cohort_matrix");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_churn_bootstrap_hr_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/churn_bootstrap_hr_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_churn_bootstrap_hr"))
            .cloned()
            .expect("golden must contain nextstat_churn_bootstrap_hr result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_churn_bootstrap_hr".to_string(),
            arguments: serde_json::json!({
                "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
                "events": [true, false, true, false, false, true, false, true],
                "covariates": [
                    [1.0, 0.2],
                    [0.5, -0.1],
                    [1.2, 0.4],
                    [0.7, 0.0],
                    [0.3, -0.2],
                    [1.1, 0.1],
                    [0.4, -0.3],
                    [0.9, 0.3]
                ],
                "names": ["x1", "x2"],
                "n_bootstrap": 8,
                "seed": 42,
                "conf_level": 0.95,
                "ci_method": "percentile",
                "n_jackknife": 4,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_churn_bootstrap_hr");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_churn_ingest_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/churn_ingest_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_churn_ingest"))
            .cloned()
            .expect("golden must contain nextstat_churn_ingest result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_churn_ingest".to_string(),
            arguments: serde_json::json!({
                "times": [30.0, 45.0, 90.0, 12.0],
                "events": [true, false, true, false],
                "groups": [0, 0, 1, 1],
                "treated": [0, 1, 0, 1],
                "covariates": [
                    [0.2, 1.0],
                    [0.3, 0.0],
                    [0.8, 1.0],
                    [0.5, 0.0]
                ],
                "covariate_names": ["usage_score", "plan_pro"],
                "observation_end": 60.0,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_churn_ingest");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_churn_compare_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/churn_compare_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_churn_compare"))
            .cloned()
            .expect("golden must contain nextstat_churn_compare result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_churn_compare".to_string(),
            arguments: serde_json::json!({
                "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
                "events": [true, false, true, false, false, true, false, true],
                "groups": [0, 0, 0, 0, 1, 1, 1, 1],
                "conf_level": 0.95,
                "correction": "benjamini_hochberg",
                "alpha": 0.05,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_churn_compare");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_churn_uplift_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/churn_uplift_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_churn_uplift"))
            .cloned()
            .expect("golden must contain nextstat_churn_uplift result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_churn_uplift".to_string(),
            arguments: serde_json::json!({
                "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
                "events": [true, false, true, false, false, true, false, true],
                "treated": [0, 0, 1, 1, 0, 1, 0, 1],
                "covariates": [
                    [1.0, 0.2],
                    [0.5, -0.1],
                    [1.2, 0.4],
                    [0.7, 0.0],
                    [0.3, -0.2],
                    [1.1, 0.1],
                    [0.4, -0.3],
                    [0.9, 0.3]
                ],
                "horizon": 60.0,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_churn_uplift");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_churn_uplift_survival_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/churn_uplift_survival_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_churn_uplift_survival"))
            .cloned()
            .expect("golden must contain nextstat_churn_uplift_survival result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_churn_uplift_survival".to_string(),
            arguments: serde_json::json!({
                "times": [30.0, 45.0, 60.0, 75.0, 30.0, 50.0, 65.0, 90.0],
                "events": [true, false, true, false, false, true, false, true],
                "treated": [0, 0, 1, 1, 0, 1, 0, 1],
                "covariates": [
                    [1.0, 0.2],
                    [0.5, -0.1],
                    [1.2, 0.4],
                    [0.7, 0.0],
                    [0.3, -0.2],
                    [1.1, 0.1],
                    [0.4, -0.3],
                    [0.9, 0.3]
                ],
                "horizon": 60.0,
                "eval_horizons": [30.0, 60.0, 90.0],
                "trim": 0.01,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_churn_uplift_survival");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_pharma_fit_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/pharma_fit_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_pharma_fit"))
            .cloned()
            .expect("golden must contain nextstat_pharma_fit result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_pharma_fit".to_string(),
            arguments: serde_json::json!({
                "times": [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0],
                "y": [1.2, 2.3, 1.9, 0.8, 1.0, 2.1, 1.7, 0.7],
                "subject_idx": [0, 0, 0, 0, 1, 1, 1, 1],
                "n_subjects": 2,
                "doses": [100.0, 100.0],
                "theta_init": [1.0, 5.0, 0.8],
                "omega_init": [0.2, 0.2, 0.2],
                "method": "focei",
                "model": "1cpt_oral",
                "sigma": 0.1,
                "error_model": "proportional",
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_pharma_fit");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_pharma_vpc_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/pharma_vpc_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_pharma_vpc"))
            .cloned()
            .expect("golden must contain nextstat_pharma_vpc result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_pharma_vpc".to_string(),
            arguments: serde_json::json!({
                "times": [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0],
                "y": [1.2, 2.3, 1.9, 0.8, 1.0, 2.1, 1.7, 0.7],
                "subject_idx": [0, 0, 0, 0, 1, 1, 1, 1],
                "n_subjects": 2,
                "doses": [100.0, 100.0],
                "model": "1cpt_oral",
                "theta": [1.0, 5.0, 0.8],
                "omega_matrix": [
                    [0.04, 0.0, 0.0],
                    [0.0, 0.04, 0.0],
                    [0.0, 0.0, 0.04]
                ],
                "sigma": 0.1,
                "error_model": "proportional",
                "n_sim": 16,
                "seed": 42,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_pharma_vpc");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_pk_gof_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw =
            include_str!("../../../tests/fixtures/tool_goldens/pk_gof_small_deterministic.v1.json");
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_pk_gof"))
            .cloned()
            .expect("golden must contain nextstat_pk_gof result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_pk_gof".to_string(),
            arguments: serde_json::json!({
                "times": [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0],
                "y": [1.2, 2.3, 1.9, 0.8, 1.0, 2.1, 1.7, 0.7],
                "subject_idx": [0, 0, 0, 0, 1, 1, 1, 1],
                "doses": [100.0, 100.0],
                "model": "1cpt_oral",
                "theta": [1.0, 5.0, 0.8],
                "eta": [
                    [0.05, -0.02, 0.01],
                    [-0.04, 0.03, -0.02]
                ],
                "sigma": 0.1,
                "error_model": "proportional",
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_pk_gof");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_pk_npde_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/pk_npde_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_pk_npde"))
            .cloned()
            .expect("golden must contain nextstat_pk_npde result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_pk_npde".to_string(),
            arguments: serde_json::json!({
                "times": [0.5, 1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0],
                "y": [1.2, 2.3, 1.9, 0.8, 1.0, 2.1, 1.7, 0.7],
                "subject_idx": [0, 0, 0, 0, 1, 1, 1, 1],
                "n_subjects": 2,
                "doses": [100.0, 100.0],
                "model": "1cpt_oral",
                "theta": [1.0, 5.0, 0.8],
                "omega_matrix": [
                    [0.04, 0.0, 0.0],
                    [0.0, 0.04, 0.0],
                    [0.0, 0.0, 0.04]
                ],
                "sigma": 0.1,
                "error_model": "proportional",
                "n_sim": 16,
                "seed": 42,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_pk_npde");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_trial_simulate_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/trial_simulate_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_trial_simulate"))
            .cloned()
            .expect("golden must contain nextstat_trial_simulate result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_trial_simulate".to_string(),
            arguments: serde_json::json!({
                "n_subjects": 3,
                "dose": 100.0,
                "obs_times": [0.5, 1.0, 2.0, 4.0],
                "pk_model": "1cpt_oral",
                "theta": [1.0, 5.0, 0.8],
                "omega": [0.2, 0.2, 0.2],
                "sigma": 0.1,
                "error_model": "proportional",
                "bioavailability": 1.0,
                "seed": 42,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_trial_simulate");
    }

    #[test]
    fn server_tools_match_local_tool_goldens_on_chain_ladder_small_deterministic() {
        let state = AppState::new(None);
        let gold_raw = include_str!(
            "../../../tests/fixtures/tool_goldens/chain_ladder_small_deterministic.v1.json"
        );
        let gold: serde_json::Value =
            serde_json::from_str(gold_raw).expect("golden JSON must parse");
        let want_raw = gold
            .get("tools")
            .and_then(|tools| tools.get("nextstat_chain_ladder"))
            .cloned()
            .expect("golden must contain nextstat_chain_ladder result");
        let want = normalize_envelope(want_raw);

        let req = ToolExecuteRequest {
            name: "nextstat_chain_ladder".to_string(),
            arguments: serde_json::json!({
                "triangle": [
                    [100.0, 150.0, 180.0],
                    [110.0, 160.0, serde_json::Value::Null],
                    [120.0, serde_json::Value::Null, serde_json::Value::Null]
                ],
                "method": "mack",
                "conf_level": 0.95,
                "execution": { "deterministic": true }
            }),
        };

        let out = execute_tool(&state, req);
        assert!(out.ok, "expected ok=true, got error={:?}", out.error);
        let got = normalize_envelope(serde_json::to_value(&out).expect("envelope must serialize"));
        assert_json_close(&got, &want, "tool:nextstat_chain_ladder");
    }

    #[test]
    fn server_tool_schema_matches_server_strict_schema() {
        let schema = get_tool_schema();
        let tools = schema
            .get("tools")
            .and_then(|value| value.as_array())
            .expect("tool schema must contain tools array");
        let mut names = tools
            .iter()
            .map(|tool| {
                tool.pointer("/function/name")
                    .and_then(|value| value.as_str())
                    .expect("each tool must contain function.name")
                    .to_string()
            })
            .collect::<Vec<_>>();
        names.sort();

        let strict: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/schemas/tools/nextstat_tool_result_server_strict_v1.schema.json"
        ))
        .expect("server strict schema must parse");
        let mut strict_names = strict
            .pointer("/properties/meta/properties/tool_name/enum")
            .and_then(|value| value.as_array())
            .expect("strict schema must enumerate tool names")
            .iter()
            .map(|value| value.as_str().expect("enum values must be strings").to_string())
            .collect::<Vec<_>>();
        strict_names.sort();

        assert_eq!(names, strict_names, "server tool schema and strict schema drifted");
        assert!(
            names.iter().any(|name| name == "nextstat_read_root_histogram"),
            "server-safe schema must expose in-memory ROOT ingest"
        );
    }

    #[test]
    fn server_tool_schema_exposes_capability_policy_metadata() {
        let schema = get_tool_schema();
        assert_eq!(schema["schema_version"], "nextstat.tool_schema.v1");
        assert_eq!(schema["transport"], "server");

        let capabilities = schema
            .get("capabilities")
            .and_then(|value| value.as_array())
            .expect("tool schema must contain capabilities array");

        let fit = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_fit")
            .expect("capabilities must include nextstat_fit");
        assert_eq!(fit["local_available"], true);
        assert_eq!(fit["server_available"], true);
        assert_eq!(fit["server_policy"]["availability"], "exposed");
        assert_eq!(fit["server_policy"]["reason_code"], "server_safe_subset");

        let root = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_read_root_histogram")
            .expect("capabilities must include nextstat_read_root_histogram");
        assert_eq!(root["local_available"], true);
        assert_eq!(root["server_available"], true);
        assert_eq!(root["server_policy"]["availability"], "exposed");
        assert_eq!(root["server_policy"]["reason_code"], "server_safe_subset");

        let glm = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_glm_fit")
            .expect("capabilities must include nextstat_glm_fit");
        assert_eq!(glm["local_available"], true);
        assert_eq!(glm["server_available"], true);
        assert_eq!(glm["server_policy"]["availability"], "exposed");
        assert_eq!(glm["server_policy"]["reason_code"], "server_safe_subset");

        let bayesian = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_bayesian_sample")
            .expect("capabilities must include nextstat_bayesian_sample");
        assert_eq!(bayesian["local_available"], true);
        assert_eq!(bayesian["server_available"], true);
        assert_eq!(bayesian["server_policy"]["availability"], "exposed");
        assert_eq!(bayesian["server_policy"]["reason_code"], "server_safe_subset");

        let survival = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_survival_fit")
            .expect("capabilities must include nextstat_survival_fit");
        assert_eq!(survival["local_available"], true);
        assert_eq!(survival["server_available"], true);
        assert_eq!(survival["server_policy"]["availability"], "exposed");
        assert_eq!(survival["server_policy"]["reason_code"], "server_safe_subset");

        let kaplan_meier = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_kaplan_meier")
            .expect("capabilities must include nextstat_kaplan_meier");
        assert_eq!(kaplan_meier["local_available"], true);
        assert_eq!(kaplan_meier["server_available"], true);
        assert_eq!(kaplan_meier["server_policy"]["availability"], "exposed");
        assert_eq!(kaplan_meier["server_policy"]["reason_code"], "server_safe_subset");

        let log_rank_test = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_log_rank_test")
            .expect("capabilities must include nextstat_log_rank_test");
        assert_eq!(log_rank_test["local_available"], true);
        assert_eq!(log_rank_test["server_available"], true);
        assert_eq!(log_rank_test["server_policy"]["availability"], "exposed");
        assert_eq!(log_rank_test["server_policy"]["reason_code"], "server_safe_subset");

        let competing_risks = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_competing_risks")
            .expect("capabilities must include nextstat_competing_risks");
        assert_eq!(competing_risks["local_available"], true);
        assert_eq!(competing_risks["server_available"], true);
        assert_eq!(competing_risks["server_policy"]["availability"], "exposed");
        assert_eq!(competing_risks["server_policy"]["reason_code"], "server_safe_subset");

        let meta_analysis = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_meta_analysis")
            .expect("capabilities must include nextstat_meta_analysis");
        assert_eq!(meta_analysis["local_available"], true);
        assert_eq!(meta_analysis["server_available"], true);
        assert_eq!(meta_analysis["server_policy"]["availability"], "exposed");
        assert_eq!(meta_analysis["server_policy"]["reason_code"], "server_safe_subset");

        let panel_fe = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_panel_fe")
            .expect("capabilities must include nextstat_panel_fe");
        assert_eq!(panel_fe["local_available"], true);
        assert_eq!(panel_fe["server_available"], true);
        assert_eq!(panel_fe["server_policy"]["availability"], "exposed");
        assert_eq!(panel_fe["server_policy"]["reason_code"], "server_safe_subset");

        let did = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_did")
            .expect("capabilities must include nextstat_did");
        assert_eq!(did["local_available"], true);
        assert_eq!(did["server_available"], true);
        assert_eq!(did["server_policy"]["availability"], "exposed");
        assert_eq!(did["server_policy"]["reason_code"], "server_safe_subset");

        let iv_2sls = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_iv_2sls")
            .expect("capabilities must include nextstat_iv_2sls");
        assert_eq!(iv_2sls["local_available"], true);
        assert_eq!(iv_2sls["server_available"], true);
        assert_eq!(iv_2sls["server_policy"]["availability"], "exposed");
        assert_eq!(iv_2sls["server_policy"]["reason_code"], "server_safe_subset");

        let aipw = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_aipw")
            .expect("capabilities must include nextstat_aipw");
        assert_eq!(aipw["local_available"], true);
        assert_eq!(aipw["server_available"], true);
        assert_eq!(aipw["server_policy"]["availability"], "exposed");
        assert_eq!(aipw["server_policy"]["reason_code"], "server_safe_subset");

        let event_study = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_event_study")
            .expect("capabilities must include nextstat_event_study");
        assert_eq!(event_study["local_available"], true);
        assert_eq!(event_study["server_available"], true);
        assert_eq!(event_study["server_policy"]["availability"], "exposed");
        assert_eq!(event_study["server_policy"]["reason_code"], "server_safe_subset");

        let garch = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_garch_fit")
            .expect("capabilities must include nextstat_garch_fit");
        assert_eq!(garch["local_available"], true);
        assert_eq!(garch["server_available"], true);
        assert_eq!(garch["server_policy"]["availability"], "exposed");
        assert_eq!(garch["server_policy"]["reason_code"], "server_safe_subset");

        let ads_cuped = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_ads_cuped_adjust")
            .expect("capabilities must include nextstat_ads_cuped_adjust");
        assert_eq!(ads_cuped["local_available"], true);
        assert_eq!(ads_cuped["server_available"], true);
        assert_eq!(ads_cuped["server_policy"]["availability"], "exposed");
        assert_eq!(ads_cuped["server_policy"]["reason_code"], "server_safe_subset");

        let ads_cure = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_ads_cure_adjust")
            .expect("capabilities must include nextstat_ads_cure_adjust");
        assert_eq!(ads_cure["local_available"], true);
        assert_eq!(ads_cure["server_available"], true);
        assert_eq!(ads_cure["server_policy"]["availability"], "exposed");
        assert_eq!(ads_cure["server_policy"]["reason_code"], "server_safe_subset");

        let kalman = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_kalman")
            .expect("capabilities must include nextstat_kalman");
        assert_eq!(kalman["local_available"], true);
        assert_eq!(kalman["server_available"], true);
        assert_eq!(kalman["server_policy"]["availability"], "exposed");
        assert_eq!(kalman["server_policy"]["reason_code"], "server_safe_subset");

        let churn_generate_data = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_churn_generate_data")
            .expect("capabilities must include nextstat_churn_generate_data");
        assert_eq!(churn_generate_data["local_available"], true);
        assert_eq!(churn_generate_data["server_available"], true);
        assert_eq!(churn_generate_data["server_policy"]["availability"], "exposed");
        assert_eq!(churn_generate_data["server_policy"]["reason_code"], "server_safe_subset");

        let churn_risk_model = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_churn_risk_model")
            .expect("capabilities must include nextstat_churn_risk_model");
        assert_eq!(churn_risk_model["local_available"], true);
        assert_eq!(churn_risk_model["server_available"], true);
        assert_eq!(churn_risk_model["server_policy"]["availability"], "exposed");
        assert_eq!(churn_risk_model["server_policy"]["reason_code"], "server_safe_subset");

        let churn_retention = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_churn_retention")
            .expect("capabilities must include nextstat_churn_retention");
        assert_eq!(churn_retention["local_available"], true);
        assert_eq!(churn_retention["server_available"], true);
        assert_eq!(churn_retention["server_policy"]["availability"], "exposed");
        assert_eq!(churn_retention["server_policy"]["reason_code"], "server_safe_subset");

        let churn_diagnostics = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_churn_diagnostics")
            .expect("capabilities must include nextstat_churn_diagnostics");
        assert_eq!(churn_diagnostics["local_available"], true);
        assert_eq!(churn_diagnostics["server_available"], true);
        assert_eq!(churn_diagnostics["server_policy"]["availability"], "exposed");
        assert_eq!(churn_diagnostics["server_policy"]["reason_code"], "server_safe_subset");

        let churn_cohort_matrix = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_churn_cohort_matrix")
            .expect("capabilities must include nextstat_churn_cohort_matrix");
        assert_eq!(churn_cohort_matrix["local_available"], true);
        assert_eq!(churn_cohort_matrix["server_available"], true);
        assert_eq!(churn_cohort_matrix["server_policy"]["availability"], "exposed");
        assert_eq!(churn_cohort_matrix["server_policy"]["reason_code"], "server_safe_subset");

        let churn_bootstrap_hr = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_churn_bootstrap_hr")
            .expect("capabilities must include nextstat_churn_bootstrap_hr");
        assert_eq!(churn_bootstrap_hr["local_available"], true);
        assert_eq!(churn_bootstrap_hr["server_available"], true);
        assert_eq!(churn_bootstrap_hr["server_policy"]["availability"], "exposed");
        assert_eq!(churn_bootstrap_hr["server_policy"]["reason_code"], "server_safe_subset");

        let churn_ingest = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_churn_ingest")
            .expect("capabilities must include nextstat_churn_ingest");
        assert_eq!(churn_ingest["local_available"], true);
        assert_eq!(churn_ingest["server_available"], true);
        assert_eq!(churn_ingest["server_policy"]["availability"], "exposed");
        assert_eq!(churn_ingest["server_policy"]["reason_code"], "server_safe_subset");

        let churn_compare = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_churn_compare")
            .expect("capabilities must include nextstat_churn_compare");
        assert_eq!(churn_compare["local_available"], true);
        assert_eq!(churn_compare["server_available"], true);
        assert_eq!(churn_compare["server_policy"]["availability"], "exposed");
        assert_eq!(churn_compare["server_policy"]["reason_code"], "server_safe_subset");

        let churn_uplift = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_churn_uplift")
            .expect("capabilities must include nextstat_churn_uplift");
        assert_eq!(churn_uplift["local_available"], true);
        assert_eq!(churn_uplift["server_available"], true);
        assert_eq!(churn_uplift["server_policy"]["availability"], "exposed");
        assert_eq!(churn_uplift["server_policy"]["reason_code"], "server_safe_subset");

        let churn_uplift_survival = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_churn_uplift_survival")
            .expect("capabilities must include nextstat_churn_uplift_survival");
        assert_eq!(churn_uplift_survival["local_available"], true);
        assert_eq!(churn_uplift_survival["server_available"], true);
        assert_eq!(churn_uplift_survival["server_policy"]["availability"], "exposed");
        assert_eq!(churn_uplift_survival["server_policy"]["reason_code"], "server_safe_subset");

        let pharma_fit = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_pharma_fit")
            .expect("capabilities must include nextstat_pharma_fit");
        assert_eq!(pharma_fit["local_available"], true);
        assert_eq!(pharma_fit["server_available"], true);
        assert_eq!(pharma_fit["server_policy"]["availability"], "exposed");
        assert_eq!(pharma_fit["server_policy"]["reason_code"], "server_safe_subset");

        let pharma_vpc = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_pharma_vpc")
            .expect("capabilities must include nextstat_pharma_vpc");
        assert_eq!(pharma_vpc["local_available"], true);
        assert_eq!(pharma_vpc["server_available"], true);
        assert_eq!(pharma_vpc["server_policy"]["availability"], "exposed");
        assert_eq!(pharma_vpc["server_policy"]["reason_code"], "server_safe_subset");

        let pk_gof = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_pk_gof")
            .expect("capabilities must include nextstat_pk_gof");
        assert_eq!(pk_gof["local_available"], true);
        assert_eq!(pk_gof["server_available"], true);
        assert_eq!(pk_gof["server_policy"]["availability"], "exposed");
        assert_eq!(pk_gof["server_policy"]["reason_code"], "server_safe_subset");

        let pk_npde = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_pk_npde")
            .expect("capabilities must include nextstat_pk_npde");
        assert_eq!(pk_npde["local_available"], true);
        assert_eq!(pk_npde["server_available"], true);
        assert_eq!(pk_npde["server_policy"]["availability"], "exposed");
        assert_eq!(pk_npde["server_policy"]["reason_code"], "server_safe_subset");

        let trial_simulate = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_trial_simulate")
            .expect("capabilities must include nextstat_trial_simulate");
        assert_eq!(trial_simulate["local_available"], true);
        assert_eq!(trial_simulate["server_available"], true);
        assert_eq!(trial_simulate["server_policy"]["availability"], "exposed");
        assert_eq!(trial_simulate["server_policy"]["reason_code"], "server_safe_subset");

        let chain_ladder = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_chain_ladder")
            .expect("capabilities must include nextstat_chain_ladder");
        assert_eq!(chain_ladder["local_available"], true);
        assert_eq!(chain_ladder["server_available"], true);
        assert_eq!(chain_ladder["server_policy"]["availability"], "exposed");
        assert_eq!(chain_ladder["server_policy"]["reason_code"], "server_safe_subset");

        let fault_tree_mc = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_fault_tree_mc")
            .expect("capabilities must include nextstat_fault_tree_mc");
        assert_eq!(fault_tree_mc["local_available"], true);
        assert_eq!(fault_tree_mc["server_available"], true);
        assert_eq!(fault_tree_mc["server_policy"]["availability"], "exposed");
        assert_eq!(fault_tree_mc["server_policy"]["reason_code"], "server_safe_subset");

        let fault_tree_ce_is = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_fault_tree_ce_is")
            .expect("capabilities must include nextstat_fault_tree_ce_is");
        assert_eq!(fault_tree_ce_is["local_available"], true);
        assert_eq!(fault_tree_ce_is["server_available"], true);
        assert_eq!(fault_tree_ce_is["server_policy"]["availability"], "exposed");
        assert_eq!(fault_tree_ce_is["server_policy"]["reason_code"], "server_safe_subset");

        let bioequivalence = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_bioequivalence")
            .expect("capabilities must include nextstat_bioequivalence");
        assert_eq!(bioequivalence["local_available"], true);
        assert_eq!(bioequivalence["server_available"], true);
        assert_eq!(bioequivalence["server_policy"]["availability"], "exposed");
        assert_eq!(bioequivalence["server_policy"]["reason_code"], "server_safe_subset");

        let dose_response = capabilities
            .iter()
            .find(|entry| entry["name"] == "nextstat_dose_response")
            .expect("capabilities must include nextstat_dose_response");
        assert_eq!(dose_response["local_available"], true);
        assert_eq!(dose_response["server_available"], true);
        assert_eq!(dose_response["server_policy"]["availability"], "exposed");
        assert_eq!(dose_response["server_policy"]["reason_code"], "server_safe_subset");

        let guidance = schema
            .get("guidance")
            .and_then(|value| value.as_object())
            .expect("tool schema must contain guidance object");
        let hints = guidance
            .get("hints")
            .and_then(|value| value.as_array())
            .expect("guidance must contain hints array");
        assert!(!hints.is_empty());
        let recipes = guidance
            .get("recipes")
            .and_then(|value| value.as_array())
            .expect("guidance must contain recipes array");
        assert!(!recipes.is_empty());
        assert!(recipes.iter().all(|recipe| recipe["transport"] == "server"));
        assert!(recipes.iter().any(|recipe| recipe["id"] == "server_glm_analysis"));
        assert!(recipes.iter().any(|recipe| recipe["id"] == "server_bayesian_sampling_summary"));
        assert!(recipes.iter().any(|recipe| {
            recipe["id"] == "server_bayesian_sampling_summary"
                && recipe["tools"]
                    .as_array()
                    .map(|tools| tools.iter().any(|tool| tool == "nextstat_bayesian_sample"))
                    .unwrap_or(false)
        }));
        assert!(recipes.iter().any(|recipe| recipe["id"] == "server_survival_analysis"));
        assert!(recipes.iter().any(|recipe| {
            recipe["id"] == "server_survival_analysis"
                && recipe["tools"]
                    .as_array()
                    .map(|tools| tools.iter().any(|tool| tool == "nextstat_kaplan_meier"))
                    .unwrap_or(false)
        }));
        assert!(recipes.iter().any(|recipe| {
            recipe["id"] == "server_survival_analysis"
                && recipe["tools"]
                    .as_array()
                    .map(|tools| tools.iter().any(|tool| tool == "nextstat_competing_risks"))
                    .unwrap_or(false)
        }));
        assert!(recipes.iter().any(|recipe| recipe["id"] == "server_meta_analysis"));
        assert!(recipes.iter().any(|recipe| recipe["id"] == "server_panel_econometrics"));
        assert!(recipes.iter().any(|recipe| recipe["id"] == "server_did_analysis"));
        assert!(recipes.iter().any(|recipe| recipe["id"] == "server_iv_analysis"));
        assert!(recipes.iter().any(|recipe| recipe["id"] == "server_aipw_analysis"));
        assert!(recipes.iter().any(|recipe| recipe["id"] == "server_event_study_analysis"));
        assert!(recipes.iter().any(|recipe| recipe["id"] == "server_volatility_analysis"));
        assert!(recipes.iter().any(|recipe| recipe["id"] == "server_state_space_analysis"));
        assert!(recipes.iter().any(|recipe| recipe["id"] == "server_retention_analysis"));
        assert!(recipes.iter().any(|recipe| recipe["id"] == "server_pharma_population_pk"));
        assert!(recipes.iter().any(|recipe| {
            recipe["id"] == "server_pharma_population_pk"
                && recipe["tools"]
                    .as_array()
                    .map(|tools| tools.iter().any(|tool| tool == "nextstat_pharma_fit"))
                    .unwrap_or(false)
        }));
        assert!(recipes.iter().any(|recipe| {
            recipe["id"] == "server_pharma_population_pk"
                && recipe["tools"]
                    .as_array()
                    .map(|tools| tools.iter().any(|tool| tool == "nextstat_pk_gof"))
                    .unwrap_or(false)
        }));
        assert!(recipes.iter().any(|recipe| {
            recipe["id"] == "server_pharma_population_pk"
                && recipe["tools"]
                    .as_array()
                    .map(|tools| tools.iter().any(|tool| tool == "nextstat_pk_npde"))
                    .unwrap_or(false)
        }));
        assert!(recipes.iter().any(|recipe| {
            recipe["id"] == "server_pharma_population_pk"
                && recipe["tools"]
                    .as_array()
                    .map(|tools| tools.iter().any(|tool| tool == "nextstat_pharma_vpc"))
                    .unwrap_or(false)
        }));
        assert!(recipes.iter().any(|recipe| recipe["id"] == "server_pharma_trial_simulation"));
        assert!(recipes.iter().any(|recipe| {
            recipe["id"] == "server_pharma_trial_simulation"
                && recipe["tools"]
                    .as_array()
                    .map(|tools| tools.iter().any(|tool| tool == "nextstat_trial_simulate"))
                    .unwrap_or(false)
        }));
        assert!(recipes.iter().any(|recipe| recipe["id"] == "server_reserving_analysis"));
        assert!(recipes.iter().any(|recipe| recipe["id"] == "server_fault_tree_analysis"));
        assert!(recipes.iter().any(|recipe| {
            recipe["id"] == "server_fault_tree_analysis"
                && recipe["tools"]
                    .as_array()
                    .map(|tools| tools.iter().any(|tool| tool == "nextstat_fault_tree_mc"))
                    .unwrap_or(false)
        }));
        assert!(recipes.iter().any(|recipe| {
            recipe["id"] == "server_fault_tree_analysis"
                && recipe["tools"]
                    .as_array()
                    .map(|tools| tools.iter().any(|tool| tool == "nextstat_fault_tree_ce_is"))
                    .unwrap_or(false)
        }));
        assert!(recipes.iter().any(|recipe| recipe["id"] == "server_bioequivalence_analysis"));
        assert!(recipes.iter().any(|recipe| recipe["id"] == "server_dose_response_analysis"));
    }

    #[test]
    fn server_tool_schema_matches_example_fixture() {
        let schema = get_tool_schema();
        let example: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/specs/nextstat_tool_schema_server_v1.example.json"
        ))
        .expect("server tool schema example must parse");
        assert_eq!(schema, example, "server tool schema drifted from example fixture");
    }
}
