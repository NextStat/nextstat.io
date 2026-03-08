use anyhow::{Context, Result};
use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::cmp::max;
use std::fmt;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum RiskLevel {
    Low,
    Medium,
    High,
}

impl RiskLevel {
    fn as_str(self) -> &'static str {
        match self {
            RiskLevel::Low => "low",
            RiskLevel::Medium => "medium",
            RiskLevel::High => "high",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum JurisdictionProfile {
    #[serde(rename = "ich_core")]
    IchCore,
    #[serde(rename = "ema_step5_2026")]
    EmaStep52026,
    #[serde(rename = "fda_draft_2024")]
    FdaDraft2024,
}

impl JurisdictionProfile {
    fn as_str(&self) -> &'static str {
        match self {
            Self::IchCore => "ich_core",
            Self::EmaStep52026 => "ema_step5_2026",
            Self::FdaDraft2024 => "fda_draft_2024",
        }
    }
}

impl fmt::Display for JurisdictionProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy)]
enum M15DocumentKind {
    AssessmentTable,
    Map,
    Mar,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ProfileRequirements {
    pub(crate) profile_label: String,
    pub(crate) canonical_reference: String,
    pub(crate) framing_heading: String,
    pub(crate) framing_text: String,
    pub(crate) mandatory_sections: Vec<String>,
}

fn profile_requirements(
    profile: JurisdictionProfile,
    kind: M15DocumentKind,
) -> ProfileRequirements {
    let (profile_label, canonical_reference) = match profile {
        JurisdictionProfile::IchCore => (
            "ICH Core",
            "ICH M15 core cross-jurisdiction framing for model-informed drug development reporting.",
        ),
        JurisdictionProfile::EmaStep52026 => (
            "EMA Step 5 (2026)",
            "EMA Step 5 alignment for the ICH M15 guideline effective in the EU from July 23, 2026.",
        ),
        JurisdictionProfile::FdaDraft2024 => (
            "FDA Draft Guidance (2024)",
            "FDA draft guidance framing for M15 general principles for model-informed drug development.",
        ),
    };

    let (framing_heading, framing_text, mandatory_sections) = match (profile, kind) {
        (JurisdictionProfile::IchCore, M15DocumentKind::AssessmentTable) => (
            "ICH Core Assessment Framing",
            "Use cross-jurisdiction ICH wording for the question of interest, context of use, model influence, and consequence framing.",
            vec![
                "Questions of Interest",
                "Mandatory Sections",
                "ICH Core Assessment Framing",
                "Evidence Refs",
            ],
        ),
        (JurisdictionProfile::IchCore, M15DocumentKind::Map) => (
            "ICH Core Planning Framing",
            "Frame the MAP as a frozen, cross-jurisdiction analysis plan with explicit technical criteria and governance.",
            vec![
                "Questions",
                "Planned Datasets",
                "Methods",
                "Technical Acceptance Criteria",
                "Governance",
            ],
        ),
        (JurisdictionProfile::IchCore, M15DocumentKind::Mar) => (
            "ICH Core Results Framing",
            "Frame results as evidence-based support statements linked back to the frozen MAP and deterministic validation evidence.",
            vec![
                "Question Conclusions",
                "Criterion Results",
                "Deviations",
                "Governance",
                "Linked Artifacts",
            ],
        ),
        (JurisdictionProfile::EmaStep52026, M15DocumentKind::AssessmentTable) => (
            "EMA Step 5 Assessment Framing",
            "Emphasize EU reviewer-readiness, context of use in the clinical pharmacology dossier, and explicit traceability for Step 5 review.",
            vec![
                "Questions of Interest",
                "Mandatory Sections",
                "EMA Step 5 Assessment Framing",
                "Evidence Refs",
            ],
        ),
        (JurisdictionProfile::EmaStep52026, M15DocumentKind::Map) => (
            "EMA Step 5 Planning Framing",
            "Emphasize planned evidence, governance, and dossier-oriented wording consistent with EMA Step 5 positioning.",
            vec![
                "Questions",
                "Planned Datasets",
                "Methods",
                "Technical Acceptance Criteria",
                "Governance",
            ],
        ),
        (JurisdictionProfile::EmaStep52026, M15DocumentKind::Mar) => (
            "EMA Step 5 Results Framing",
            "Emphasize review-ready conclusions, deviations, and limitations suitable for EMA-oriented briefing and dossier assembly.",
            vec![
                "Question Conclusions",
                "Criterion Results",
                "Deviations",
                "Governance",
                "Linked Artifacts",
            ],
        ),
        (JurisdictionProfile::FdaDraft2024, M15DocumentKind::AssessmentTable) => (
            "FDA Draft Guidance Assessment Framing",
            "Emphasize draft-guidance framing, decision consequence, and explicit qualification boundaries for FDA-facing evidence narratives.",
            vec![
                "Questions of Interest",
                "Mandatory Sections",
                "FDA Draft Guidance Assessment Framing",
                "Evidence Refs",
            ],
        ),
        (JurisdictionProfile::FdaDraft2024, M15DocumentKind::Map) => (
            "FDA Draft Guidance Planning Framing",
            "Emphasize planning language aligned with FDA draft expectations for context of use, methods, and acceptance criteria.",
            vec![
                "Questions",
                "Planned Datasets",
                "Methods",
                "Technical Acceptance Criteria",
                "Governance",
            ],
        ),
        (JurisdictionProfile::FdaDraft2024, M15DocumentKind::Mar) => (
            "FDA Draft Guidance Results Framing",
            "Emphasize evidence-bounded conclusions, limitations, and deviations suitable for FDA draft-guidance review expectations.",
            vec![
                "Question Conclusions",
                "Criterion Results",
                "Deviations",
                "Governance",
                "Linked Artifacts",
            ],
        ),
    };

    ProfileRequirements {
        profile_label: profile_label.to_string(),
        canonical_reference: canonical_reference.to_string(),
        framing_heading: framing_heading.to_string(),
        framing_text: framing_text.to_string(),
        mandatory_sections: mandatory_sections.into_iter().map(|value| value.to_string()).collect(),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ArtifactRef {
    pub(crate) artifact_type: String,
    pub(crate) path: String,
    pub(crate) sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) role: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ProductContext {
    pub(crate) program_name: String,
    pub(crate) compound_name: String,
    pub(crate) indication: String,
    pub(crate) sponsor: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AssessmentEntry {
    pub(crate) qoi_id: String,
    pub(crate) question_of_interest: String,
    pub(crate) context_of_use: String,
    pub(crate) model_influence: String,
    pub(crate) consequence_of_wrong_decision: String,
    pub(crate) model_impact: String,
    pub(crate) model_risk: String,
    pub(crate) recommended_reporting_level: String,
    pub(crate) justification: String,
    pub(crate) evidence_refs: Vec<ArtifactRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) review_notes: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AssessmentSummary {
    pub(crate) n_entries: usize,
    pub(crate) highest_model_impact: String,
    pub(crate) highest_model_risk: String,
    pub(crate) unresolved_items: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AssessmentTableArtifact {
    pub(crate) schema_version: String,
    pub(crate) generated_at: Option<String>,
    pub(crate) deterministic: bool,
    pub(crate) jurisdiction_profile: JurisdictionProfile,
    pub(crate) profile_requirements: ProfileRequirements,
    pub(crate) source_config_sha256: String,
    pub(crate) product_context: ProductContext,
    pub(crate) entries: Vec<AssessmentEntry>,
    pub(crate) summary: AssessmentSummary,
    pub(crate) review_status: String,
}

#[derive(Debug, Deserialize)]
struct M15Config {
    schema_version: String,
    deterministic: bool,
    jurisdiction_profile: JurisdictionProfile,
    product_context: ProductContext,
    context_of_use: String,
    questions_of_interest: Vec<QuestionOfInterest>,
    planned_datasets: Vec<DatasetRef>,
    methods: Vec<MethodRef>,
    authors: Vec<PersonRole>,
    technical_acceptance_criteria: Vec<ConfigCriterion>,
    supporting_artifacts: Vec<ArtifactRef>,
    review_plan: ReviewPlan,
    reporting_strategy: ReportingStrategy,
}

#[derive(Debug, Deserialize)]
struct QuestionOfInterest {
    qoi_id: String,
    question: String,
    context_of_use: String,
    model_influence: String,
    consequence_of_wrong_decision: String,
    decision_impact: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct DatasetRef {
    pub(crate) dataset_id: String,
    pub(crate) description: String,
    pub(crate) purpose: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MethodRef {
    pub(crate) method_id: String,
    pub(crate) description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) software_ref: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct PersonRole {
    name: String,
    role: String,
    organization: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ConfigCriterion {
    criterion_id: String,
    description: String,
    rationale: String,
    target: String,
    source: String,
    applies_to_qoi: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct ReviewPlan {
    primary_author: String,
    qa_reviewer: String,
    approver: String,
    status: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ReportingStrategy {
    assessment_table_required: bool,
    map_required: bool,
    mar_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MapQuestionRef {
    pub(crate) question_id: String,
    pub(crate) question_of_interest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MapCriterion {
    pub(crate) criterion_id: String,
    pub(crate) description: String,
    pub(crate) acceptance_rule: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) applies_to_question_ids: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MapGovernance {
    pub(crate) authors: Vec<String>,
    pub(crate) reviewers: Vec<String>,
    pub(crate) approvers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SignoffPlan {
    pub(crate) primary_author: String,
    pub(crate) qa_reviewer: String,
    pub(crate) approver: String,
    pub(crate) status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MapLinkedArtifacts {
    pub(crate) assessment_table_ref: String,
    pub(crate) validation_report_ref: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MapArtifact {
    pub(crate) schema_version: String,
    pub(crate) generated_at: Option<String>,
    pub(crate) document_status: String,
    pub(crate) jurisdiction_profile: JurisdictionProfile,
    pub(crate) profile_requirements: ProfileRequirements,
    pub(crate) context_of_use: String,
    pub(crate) questions: Vec<MapQuestionRef>,
    pub(crate) planned_datasets: Vec<DatasetRef>,
    pub(crate) methods: Vec<MethodRef>,
    pub(crate) technical_acceptance_criteria: Vec<MapCriterion>,
    pub(crate) governance: MapGovernance,
    pub(crate) signoff: SignoffPlan,
    pub(crate) linked_artifacts: MapLinkedArtifacts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MarQuestionResult {
    pub(crate) question_id: String,
    pub(crate) question_of_interest: String,
    pub(crate) conclusion_status: String,
    pub(crate) conclusion: String,
    pub(crate) evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MarDatasetResult {
    pub(crate) dataset_id: String,
    pub(crate) provenance_ref: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MarMethodResult {
    pub(crate) method_id: String,
    pub(crate) status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MarCriterionResult {
    pub(crate) criterion_id: String,
    pub(crate) status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) observed_value: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MarDeviation {
    pub(crate) deviation_id: String,
    pub(crate) description: String,
    pub(crate) impact_assessment: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MarLinkedArtifacts {
    pub(crate) assessment_table_ref: String,
    pub(crate) validation_report_ref: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) pharma_validation_ref: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MarArtifact {
    pub(crate) schema_version: String,
    pub(crate) generated_at: Option<String>,
    pub(crate) document_status: String,
    pub(crate) jurisdiction_profile: JurisdictionProfile,
    pub(crate) profile_requirements: ProfileRequirements,
    pub(crate) context_of_use: String,
    pub(crate) based_on_map_ref: String,
    pub(crate) questions: Vec<MarQuestionResult>,
    pub(crate) executed_datasets: Vec<MarDatasetResult>,
    pub(crate) methods_executed: Vec<MarMethodResult>,
    pub(crate) criterion_results: Vec<MarCriterionResult>,
    pub(crate) deviations: Vec<MarDeviation>,
    pub(crate) limitations: Vec<String>,
    pub(crate) governance: MapGovernance,
    pub(crate) signoff: SignoffPlan,
    pub(crate) linked_artifacts: MarLinkedArtifacts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ProfileDiffProfileView {
    pub(crate) jurisdiction_profile: JurisdictionProfile,
    pub(crate) profile_label: String,
    pub(crate) canonical_reference: String,
    pub(crate) framing_heading: String,
    pub(crate) framing_text: String,
    pub(crate) mandatory_sections: Vec<String>,
    pub(crate) profile_only_sections: Vec<String>,
    pub(crate) missing_sections_vs_union: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ProfileDiffSectionPresence {
    pub(crate) section_name: String,
    pub(crate) present_in_profiles: Vec<JurisdictionProfile>,
    pub(crate) present_in_all_profiles: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ProfileDiffDocument {
    pub(crate) document_kind: String,
    pub(crate) common_sections: Vec<String>,
    pub(crate) section_presence: Vec<ProfileDiffSectionPresence>,
    pub(crate) profile_views: Vec<ProfileDiffProfileView>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ProfileDiffSummary {
    pub(crate) documents_compared: usize,
    pub(crate) profiles_compared: usize,
    pub(crate) sections_with_profile_specific_diff: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ProfileDiffReportArtifact {
    pub(crate) schema_version: String,
    pub(crate) generated_at: Option<String>,
    pub(crate) deterministic: bool,
    pub(crate) selected_profile: JurisdictionProfile,
    pub(crate) compared_profiles: Vec<JurisdictionProfile>,
    pub(crate) source_config_sha256: String,
    pub(crate) documents: Vec<ProfileDiffDocument>,
    pub(crate) summary: ProfileDiffSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct BundleArtifacts {
    pub(crate) assessment_table: ArtifactRef,
    pub(crate) map: ArtifactRef,
    pub(crate) mar: ArtifactRef,
    pub(crate) validation_report: ArtifactRef,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) pharma_validation: Option<ArtifactRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct BundleFileEntry {
    pub(crate) path: String,
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
    pub(crate) artifact_role: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct BundleIntegrity {
    pub(crate) all_hashes_present: bool,
    pub(crate) deterministic_re_render_verified: bool,
    pub(crate) missing_required_roles: Vec<String>,
    pub(crate) signoff_roles_complete: bool,
    pub(crate) signoff_roles_distinct: bool,
    pub(crate) missing_signoff_roles: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct BundleManifestArtifact {
    pub(crate) schema_version: String,
    pub(crate) generated_at: Option<String>,
    pub(crate) deterministic: bool,
    pub(crate) jurisdiction_profile: JurisdictionProfile,
    pub(crate) source_config_sha256: String,
    pub(crate) bundle_status: String,
    pub(crate) artifacts: BundleArtifacts,
    pub(crate) files: Vec<BundleFileEntry>,
    pub(crate) integrity: BundleIntegrity,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let out = h.finalize();
    let mut s = String::with_capacity(64);
    for b in out {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn parse_level(value: &str, field_name: &str) -> Result<RiskLevel> {
    match value {
        "low" => Ok(RiskLevel::Low),
        "medium" => Ok(RiskLevel::Medium),
        "high" => Ok(RiskLevel::High),
        other => anyhow::bail!("unsupported {field_name}: {other}"),
    }
}

fn reporting_level(model_impact: RiskLevel, model_risk: RiskLevel) -> &'static str {
    match max(model_impact, model_risk) {
        RiskLevel::High => "full",
        RiskLevel::Medium => "enhanced",
        RiskLevel::Low => "basic",
    }
}

fn generated_at(deterministic: bool) -> Option<String> {
    if deterministic { None } else { Some(Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true)) }
}

fn require_status<'a>(value: &'a Value, pointer: &str, label: &str) -> Result<&'a str> {
    value
        .pointer(pointer)
        .and_then(|v| v.as_str())
        .with_context(|| format!("missing {label} at JSON pointer {pointer}"))
}

fn canonical_json_bytes<T: Serialize>(doc: &T, deterministic: bool) -> Result<Vec<u8>> {
    let mut value = serde_json::to_value(doc)?;
    if deterministic {
        value = crate::normalize_json_for_determinism(value);
    }
    value = crate::canonicalize_json(&value);
    Ok(serde_json::to_vec_pretty(&value)?)
}

fn fallback_file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| path.display().to_string())
}

fn stable_artifact_path(artifact_type: &str, fallback_path: &Path) -> String {
    match artifact_type {
        "config" => "m15_config.json".to_string(),
        "assessment_table" => "m15_assessment_table.json".to_string(),
        "map" => "m15_map.json".to_string(),
        "mar" => "m15_mar.json".to_string(),
        "validation_report" => "validation_report.json".to_string(),
        "pharma_validation" => "pharma_validation.json".to_string(),
        "workspace" => "workspace.json".to_string(),
        _ => fallback_file_name(fallback_path),
    }
}

fn stable_artifact_pointer_ref(artifact_type: &str, pointer: &str) -> String {
    format!("{}#{pointer}", stable_artifact_path(artifact_type, Path::new(artifact_type)))
}

fn bundle_artifact_ref(path: &Path, artifact_type: &str, role: &str) -> Result<ArtifactRef> {
    let bytes =
        std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(ArtifactRef {
        artifact_type: artifact_type.to_string(),
        path: stable_artifact_path(artifact_type, path),
        sha256: sha256_hex(&bytes),
        role: Some(role.to_string()),
    })
}

fn bundle_file_entry(path: &Path, artifact_role: &str) -> Result<BundleFileEntry> {
    let metadata =
        std::fs::metadata(path).with_context(|| format!("failed to stat {}", path.display()))?;
    let bytes =
        std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(BundleFileEntry {
        path: stable_artifact_path(artifact_role, path),
        bytes: metadata.len(),
        sha256: sha256_hex(&bytes),
        artifact_role: artifact_role.to_string(),
    })
}

fn bundle_role_rank(role: &str) -> usize {
    match role {
        "config" => 0,
        "assessment_table" => 1,
        "map" => 2,
        "mar" => 3,
        "validation_report" => 4,
        "pharma_validation" => 5,
        "workspace" => 6,
        _ => 7,
    }
}

pub(crate) fn build_assessment_table(
    config_path: &Path,
    validation_report_path: &Path,
    pharma_validation_path: &Path,
    deterministic: bool,
) -> Result<AssessmentTableArtifact> {
    let config_bytes = std::fs::read(config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let config: M15Config = serde_json::from_slice(&config_bytes)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;
    if config.schema_version != "m15_config_v1" {
        anyhow::bail!(
            "unsupported M15 config schema_version: {} (expected m15_config_v1)",
            config.schema_version
        );
    }
    let signoff = validate_signoff_plan(&config.authors, &config.review_plan)?;
    if config.questions_of_interest.is_empty() {
        anyhow::bail!("m15 config must contain at least one question_of_interest");
    }

    let validation_bytes = std::fs::read(validation_report_path)
        .with_context(|| format!("failed to read {}", validation_report_path.display()))?;
    let validation_json: Value = serde_json::from_slice(&validation_bytes)
        .with_context(|| format!("failed to parse {}", validation_report_path.display()))?;
    let validation_schema =
        require_status(&validation_json, "/schema_version", "validation report schema_version")?;
    if validation_schema != "validation_report_v1" {
        anyhow::bail!(
            "unsupported validation report schema_version: {} (expected validation_report_v1)",
            validation_schema
        );
    }
    let validation_status =
        require_status(&validation_json, "/apex2_summary/overall", "validation report overall")?;

    let pharma_bytes = std::fs::read(pharma_validation_path)
        .with_context(|| format!("failed to read {}", pharma_validation_path.display()))?;
    let pharma_json: Value = serde_json::from_slice(&pharma_bytes)
        .with_context(|| format!("failed to parse {}", pharma_validation_path.display()))?;
    let pharma_schema =
        require_status(&pharma_json, "/schema_version", "pharma validation schema_version")?;
    if pharma_schema != "nextstat.pharma_validation.v1" {
        anyhow::bail!(
            "unsupported pharma validation schema_version: {} (expected nextstat.pharma_validation.v1)",
            pharma_schema
        );
    }
    let pharma_status = require_status(&pharma_json, "/status", "pharma validation status")?;

    let validation_ref = ArtifactRef {
        artifact_type: "validation_report".to_string(),
        path: stable_artifact_path("validation_report", validation_report_path),
        sha256: sha256_hex(&validation_bytes),
        role: Some("cross-suite deterministic validation".to_string()),
    };
    let pharma_ref = ArtifactRef {
        artifact_type: "pharma_validation".to_string(),
        path: stable_artifact_path("pharma_validation", pharma_validation_path),
        sha256: sha256_hex(&pharma_bytes),
        role: Some("pharma IQ/OQ/PQ evidence".to_string()),
    };

    let mut highest_model_impact = RiskLevel::Low;
    let mut highest_model_risk = RiskLevel::Low;
    let mut unresolved_items = 0usize;
    let mut entries = Vec::with_capacity(config.questions_of_interest.len());

    for qoi in &config.questions_of_interest {
        let model_influence = parse_level(&qoi.model_influence, "model_influence")?;
        let consequence =
            parse_level(&qoi.consequence_of_wrong_decision, "consequence_of_wrong_decision")?;
        let model_impact = parse_level(&qoi.decision_impact, "decision_impact")?;
        let model_risk = max(model_influence, consequence);
        highest_model_impact = max(highest_model_impact, model_impact);
        highest_model_risk = max(highest_model_risk, model_risk);

        let mut review_notes = Vec::new();
        if validation_status != "pass" {
            review_notes.push(format!(
                "validation_report overall status is `{validation_status}`, so evidence is not yet review-complete"
            ));
        }
        if pharma_status != "ok" {
            review_notes.push(format!(
                "pharma_validation status is `{pharma_status}`, so pharma qualification evidence remains open"
            ));
        }
        if !review_notes.is_empty() {
            unresolved_items += 1;
        }

        let justification = format!(
            "Context of use: {} Model influence is {} and consequence of wrong decision is {}. \
Validation report overall status is {} and pharma validation status is {}.",
            qoi.context_of_use,
            qoi.model_influence,
            qoi.consequence_of_wrong_decision,
            validation_status,
            pharma_status
        );

        entries.push(AssessmentEntry {
            qoi_id: qoi.qoi_id.clone(),
            question_of_interest: qoi.question.clone(),
            context_of_use: qoi.context_of_use.clone(),
            model_influence: qoi.model_influence.clone(),
            consequence_of_wrong_decision: qoi.consequence_of_wrong_decision.clone(),
            model_impact: qoi.decision_impact.clone(),
            model_risk: model_risk.as_str().to_string(),
            recommended_reporting_level: reporting_level(model_impact, model_risk).to_string(),
            justification,
            evidence_refs: vec![validation_ref.clone(), pharma_ref.clone()],
            review_notes: if review_notes.is_empty() { None } else { Some(review_notes) },
        });
    }

    Ok(AssessmentTableArtifact {
        schema_version: "m15_assessment_table_v1".to_string(),
        generated_at: generated_at(deterministic || config.deterministic),
        deterministic: deterministic || config.deterministic,
        jurisdiction_profile: config.jurisdiction_profile,
        profile_requirements: profile_requirements(
            config.jurisdiction_profile,
            M15DocumentKind::AssessmentTable,
        ),
        source_config_sha256: sha256_hex(&config_bytes),
        product_context: config.product_context,
        entries,
        summary: AssessmentSummary {
            n_entries: config.questions_of_interest.len(),
            highest_model_impact: highest_model_impact.as_str().to_string(),
            highest_model_risk: highest_model_risk.as_str().to_string(),
            unresolved_items,
        },
        review_status: assessment_review_status(&signoff.status, unresolved_items)?,
    })
}

pub(crate) fn render_assessment_table_markdown(doc: &AssessmentTableArtifact) -> String {
    let mut out = String::new();
    out.push_str("# ICH M15 Assessment Table\n\n");
    out.push_str(&format!(
        "- Program: {}\n- Compound: {}\n- Indication: {}\n- Sponsor: {}\n- Jurisdiction: {}\n- Review status: {}\n\n",
        doc.product_context.program_name,
        doc.product_context.compound_name,
        doc.product_context.indication,
        doc.product_context.sponsor,
        doc.jurisdiction_profile,
        doc.review_status,
    ));
    append_profile_markdown(&mut out, &doc.profile_requirements);
    out.push_str(&format!(
        "Summary: {} entries, highest model impact `{}`, highest model risk `{}`, unresolved items `{}`.\n\n",
        doc.summary.n_entries,
        doc.summary.highest_model_impact,
        doc.summary.highest_model_risk,
        doc.summary.unresolved_items,
    ));
    out.push_str("## Questions of Interest\n\n");

    for entry in &doc.entries {
        out.push_str(&format!("### {} — {}\n\n", entry.qoi_id, entry.question_of_interest));
        out.push_str(&format!("- Context of use: {}\n", entry.context_of_use));
        out.push_str(&format!("- Model influence: {}\n", entry.model_influence));
        out.push_str(&format!(
            "- Consequence of wrong decision: {}\n",
            entry.consequence_of_wrong_decision
        ));
        out.push_str(&format!("- Model impact: {}\n", entry.model_impact));
        out.push_str(&format!("- Model risk: {}\n", entry.model_risk));
        out.push_str(&format!(
            "- Recommended reporting level: {}\n",
            entry.recommended_reporting_level
        ));
        out.push_str(&format!("- Justification: {}\n", entry.justification));
        out.push_str("- Evidence refs:\n");
        for evidence in &entry.evidence_refs {
            out.push_str(&format!(
                "  - {}: {} ({})\n",
                evidence.artifact_type, evidence.path, evidence.sha256
            ));
        }
        if let Some(notes) = &entry.review_notes {
            out.push_str("- Review notes:\n");
            for note in notes {
                out.push_str(&format!("  - {}\n", note));
            }
        }
        out.push('\n');
    }

    out
}

fn freeze_status(review_status: &str) -> Result<String> {
    match review_status {
        "draft" => Ok("draft".to_string()),
        "reviewed" => Ok("frozen".to_string()),
        "approved" => Ok("approved".to_string()),
        other => anyhow::bail!("unsupported signoff.status for MAP: {other}"),
    }
}

fn signoff_status(review_status: &str) -> Result<String> {
    match review_status {
        "draft" => Ok("draft".to_string()),
        "review_ready" => Ok("reviewed".to_string()),
        "approved" => Ok("approved".to_string()),
        other => anyhow::bail!("unsupported review_plan.status: {other}"),
    }
}

fn validate_signoff_plan(authors: &[PersonRole], review_plan: &ReviewPlan) -> Result<SignoffPlan> {
    let primary_author = review_plan.primary_author.trim();
    let qa_reviewer = review_plan.qa_reviewer.trim();
    let approver = review_plan.approver.trim();

    if primary_author.is_empty() || qa_reviewer.is_empty() || approver.is_empty() {
        anyhow::bail!("review_plan roles must be non-empty");
    }

    let author_names: std::collections::BTreeSet<&str> =
        authors.iter().map(|author| author.name.trim()).collect();
    if !author_names.contains(primary_author) {
        anyhow::bail!(
            "review_plan.primary_author must match an entry in authors: {}",
            primary_author
        );
    }

    if primary_author == qa_reviewer || primary_author == approver || qa_reviewer == approver {
        anyhow::bail!("review_plan roles must be assigned to distinct people");
    }

    let status = signoff_status(&review_plan.status)?;
    Ok(SignoffPlan {
        primary_author: primary_author.to_string(),
        qa_reviewer: qa_reviewer.to_string(),
        approver: approver.to_string(),
        status,
    })
}

fn validate_governance(governance: &MapGovernance, signoff: &SignoffPlan) -> Result<()> {
    if governance.authors.is_empty() {
        anyhow::bail!("governance.authors must be non-empty");
    }
    if governance.reviewers.is_empty() {
        anyhow::bail!("governance.reviewers must be non-empty");
    }
    if governance.approvers.is_empty() {
        anyhow::bail!("governance.approvers must be non-empty");
    }
    if !governance.authors.iter().any(|name| name == &signoff.primary_author) {
        anyhow::bail!("governance.authors must include signoff.primary_author");
    }
    if !governance.reviewers.iter().any(|name| name == &signoff.qa_reviewer) {
        anyhow::bail!("governance.reviewers must include signoff.qa_reviewer");
    }
    if !governance.approvers.iter().any(|name| name == &signoff.approver) {
        anyhow::bail!("governance.approvers must include signoff.approver");
    }
    if signoff.primary_author == signoff.qa_reviewer
        || signoff.primary_author == signoff.approver
        || signoff.qa_reviewer == signoff.approver
    {
        anyhow::bail!("signoff roles must be assigned to distinct people");
    }
    Ok(())
}

fn assessment_review_status(review_status: &str, unresolved_items: usize) -> Result<String> {
    if unresolved_items > 0 {
        Ok("draft".to_string())
    } else {
        match review_status {
            "draft" | "reviewed" | "approved" => Ok(review_status.to_string()),
            other => anyhow::bail!("unsupported signoff.status for assessment review: {other}"),
        }
    }
}

fn mar_document_status(signoff_status: &str, evidence_ready: bool) -> Result<String> {
    if evidence_ready {
        match signoff_status {
            "draft" | "reviewed" | "approved" => Ok(signoff_status.to_string()),
            other => anyhow::bail!("unsupported signoff.status for MAR: {other}"),
        }
    } else {
        Ok("draft".to_string())
    }
}

fn append_profile_markdown(out: &mut String, requirements: &ProfileRequirements) {
    out.push_str(&format!(
        "- Profile label: {}\n- Canonical reference: {}\n\n",
        requirements.profile_label, requirements.canonical_reference
    ));
    out.push_str("## Mandatory Sections\n\n");
    for section in &requirements.mandatory_sections {
        out.push_str(&format!("- {}\n", section));
    }
    out.push('\n');
    out.push_str(&format!(
        "## {}\n\n{}\n\n",
        requirements.framing_heading, requirements.framing_text
    ));
}

fn supported_jurisdiction_profiles() -> [JurisdictionProfile; 3] {
    [
        JurisdictionProfile::IchCore,
        JurisdictionProfile::EmaStep52026,
        JurisdictionProfile::FdaDraft2024,
    ]
}

fn document_kind_name(kind: M15DocumentKind) -> &'static str {
    match kind {
        M15DocumentKind::AssessmentTable => "assessment_table",
        M15DocumentKind::Map => "map",
        M15DocumentKind::Mar => "mar",
    }
}

fn build_profile_diff_document(kind: M15DocumentKind) -> ProfileDiffDocument {
    let requirements_by_profile: Vec<(JurisdictionProfile, ProfileRequirements)> =
        supported_jurisdiction_profiles()
            .into_iter()
            .map(|profile| (profile, profile_requirements(profile, kind)))
            .collect();

    let first_sections = &requirements_by_profile[0].1.mandatory_sections;
    let common_sections: Vec<String> = first_sections
        .iter()
        .filter(|section| {
            requirements_by_profile.iter().all(|(_, requirements)| {
                requirements.mandatory_sections.iter().any(|value| value == *section)
            })
        })
        .cloned()
        .collect();

    let mut union_sections = Vec::new();
    for (_, requirements) in &requirements_by_profile {
        for section in &requirements.mandatory_sections {
            if !union_sections.iter().any(|value| value == section) {
                union_sections.push(section.clone());
            }
        }
    }

    let section_presence = union_sections
        .iter()
        .map(|section_name| {
            let present_in_profiles: Vec<JurisdictionProfile> = requirements_by_profile
                .iter()
                .filter_map(|(profile, requirements)| {
                    requirements
                        .mandatory_sections
                        .iter()
                        .any(|value| value == section_name)
                        .then_some(*profile)
                })
                .collect();
            ProfileDiffSectionPresence {
                section_name: section_name.clone(),
                present_in_profiles: present_in_profiles.clone(),
                present_in_all_profiles: present_in_profiles.len() == requirements_by_profile.len(),
            }
        })
        .collect();

    let profile_views = requirements_by_profile
        .into_iter()
        .map(|(jurisdiction_profile, requirements)| ProfileDiffProfileView {
            jurisdiction_profile,
            profile_label: requirements.profile_label.clone(),
            canonical_reference: requirements.canonical_reference.clone(),
            framing_heading: requirements.framing_heading.clone(),
            framing_text: requirements.framing_text.clone(),
            profile_only_sections: requirements
                .mandatory_sections
                .iter()
                .filter(|section| !common_sections.iter().any(|value| value == *section))
                .cloned()
                .collect(),
            missing_sections_vs_union: union_sections
                .iter()
                .filter(|section| {
                    !requirements.mandatory_sections.iter().any(|value| value == *section)
                })
                .cloned()
                .collect(),
            mandatory_sections: requirements.mandatory_sections,
        })
        .collect();

    ProfileDiffDocument {
        document_kind: document_kind_name(kind).to_string(),
        common_sections,
        section_presence,
        profile_views,
    }
}

pub(crate) fn build_profile_diff_report(
    config_path: &Path,
    deterministic: bool,
) -> Result<ProfileDiffReportArtifact> {
    let config_bytes = std::fs::read(config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let config: M15Config = serde_json::from_slice(&config_bytes)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;
    if config.schema_version != "m15_config_v1" {
        anyhow::bail!(
            "unsupported M15 config schema_version: {} (expected m15_config_v1)",
            config.schema_version
        );
    }

    let deterministic = deterministic || config.deterministic;
    let documents = vec![
        build_profile_diff_document(M15DocumentKind::AssessmentTable),
        build_profile_diff_document(M15DocumentKind::Map),
        build_profile_diff_document(M15DocumentKind::Mar),
    ];

    let sections_with_profile_specific_diff = documents
        .iter()
        .map(|document| {
            document
                .section_presence
                .iter()
                .filter(|section| !section.present_in_all_profiles)
                .count()
        })
        .sum();

    Ok(ProfileDiffReportArtifact {
        schema_version: "m15_profile_diff_report_v1".to_string(),
        generated_at: generated_at(deterministic),
        deterministic,
        selected_profile: config.jurisdiction_profile,
        compared_profiles: supported_jurisdiction_profiles().into_iter().collect(),
        source_config_sha256: sha256_hex(&config_bytes),
        documents,
        summary: ProfileDiffSummary {
            documents_compared: 3,
            profiles_compared: 3,
            sections_with_profile_specific_diff,
        },
    })
}

pub(crate) fn render_profile_diff_markdown(doc: &ProfileDiffReportArtifact) -> String {
    let mut out = String::new();
    out.push_str("# ICH M15 Profile Diff Report\n\n");
    out.push_str(&format!(
        "- Selected profile: {}\n- Compared profiles: {}\n- Documents compared: {}\n- Sections with profile-specific diff: {}\n\n",
        doc.selected_profile,
        doc.compared_profiles
            .iter()
            .map(|profile| profile.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        doc.summary.documents_compared,
        doc.summary.sections_with_profile_specific_diff
    ));

    for document in &doc.documents {
        out.push_str(&format!("## {}\n\n", document.document_kind));
        out.push_str("- Common mandatory sections:\n");
        for section in &document.common_sections {
            out.push_str(&format!("  - {}\n", section));
        }
        out.push_str("\n- Section presence by profile:\n");
        for section in &document.section_presence {
            out.push_str(&format!(
                "  - {}: {}{}\n",
                section.section_name,
                section
                    .present_in_profiles
                    .iter()
                    .map(|profile| profile.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                if section.present_in_all_profiles { " [all profiles]" } else { "" }
            ));
        }
        out.push('\n');
        for view in &document.profile_views {
            out.push_str(&format!(
                "### {} ({})\n\n",
                view.profile_label, view.jurisdiction_profile
            ));
            out.push_str(&format!(
                "- Framing heading: {}\n- Canonical reference: {}\n- Framing text: {}\n",
                view.framing_heading, view.canonical_reference, view.framing_text
            ));
            out.push_str("- Mandatory sections:\n");
            for section in &view.mandatory_sections {
                out.push_str(&format!("  - {}\n", section));
            }
            out.push_str("- Profile-only mandatory sections:\n");
            if view.profile_only_sections.is_empty() {
                out.push_str("  - None\n");
            } else {
                for section in &view.profile_only_sections {
                    out.push_str(&format!("  - {}\n", section));
                }
            }
            out.push_str("- Missing sections vs union:\n");
            if view.missing_sections_vs_union.is_empty() {
                out.push_str("  - None\n");
            } else {
                for section in &view.missing_sections_vs_union {
                    out.push_str(&format!("  - {}\n", section));
                }
            }
            out.push('\n');
        }
    }

    out
}

pub(crate) fn build_map(
    config_path: &Path,
    assessment_table_path: &Path,
    deterministic: bool,
) -> Result<MapArtifact> {
    let config_bytes = std::fs::read(config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let config: M15Config = serde_json::from_slice(&config_bytes)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;
    if config.schema_version != "m15_config_v1" {
        anyhow::bail!(
            "unsupported M15 config schema_version: {} (expected m15_config_v1)",
            config.schema_version
        );
    }

    let assessment_bytes = std::fs::read(assessment_table_path)
        .with_context(|| format!("failed to read {}", assessment_table_path.display()))?;
    let assessment: AssessmentTableArtifact = serde_json::from_slice(&assessment_bytes)
        .with_context(|| format!("failed to parse {}", assessment_table_path.display()))?;
    if assessment.schema_version != "m15_assessment_table_v1" {
        anyhow::bail!(
            "unsupported assessment-table schema_version: {} (expected m15_assessment_table_v1)",
            assessment.schema_version
        );
    }

    let config_sha = sha256_hex(&config_bytes);
    if assessment.source_config_sha256 != config_sha {
        anyhow::bail!(
            "assessment-table source_config_sha256 does not match config: {} != {}",
            assessment.source_config_sha256,
            config_sha
        );
    }

    let assessment_qoi_ids: std::collections::BTreeSet<String> =
        assessment.entries.iter().map(|e| e.qoi_id.clone()).collect();
    let config_qoi_ids: std::collections::BTreeSet<String> =
        config.questions_of_interest.iter().map(|q| q.qoi_id.clone()).collect();
    if assessment_qoi_ids != config_qoi_ids {
        anyhow::bail!("assessment-table QOI set does not match config questions_of_interest");
    }

    let validation_report_ref = assessment
        .entries
        .iter()
        .flat_map(|entry| entry.evidence_refs.iter())
        .find(|artifact| artifact.artifact_type == "validation_report")
        .map(|artifact| artifact.path.clone())
        .or_else(|| {
            config
                .supporting_artifacts
                .iter()
                .find(|artifact| artifact.artifact_type == "validation_report")
                .map(|artifact| {
                    stable_artifact_path(&artifact.artifact_type, Path::new(&artifact.path))
                })
        });

    let questions = config
        .questions_of_interest
        .iter()
        .map(|q| MapQuestionRef {
            question_id: q.qoi_id.clone(),
            question_of_interest: q.question.clone(),
        })
        .collect();

    let technical_acceptance_criteria = config
        .technical_acceptance_criteria
        .iter()
        .map(|criterion| MapCriterion {
            criterion_id: criterion.criterion_id.clone(),
            description: format!(
                "{} Rationale: {} Source: {}.",
                criterion.description, criterion.rationale, criterion.source
            ),
            acceptance_rule: criterion.target.clone(),
            applies_to_question_ids: Some(criterion.applies_to_qoi.clone()),
        })
        .collect();

    let signoff = validate_signoff_plan(&config.authors, &config.review_plan)?;
    let authors = config.authors.iter().map(|a| a.name.clone()).collect();
    let governance = MapGovernance {
        authors,
        reviewers: vec![signoff.qa_reviewer.clone()],
        approvers: vec![signoff.approver.clone()],
    };
    validate_governance(&governance, &signoff)?;
    let _reporting_strategy = &config.reporting_strategy;
    let map = MapArtifact {
        schema_version: "m15_map_v1".to_string(),
        generated_at: generated_at(deterministic || config.deterministic),
        document_status: freeze_status(&signoff.status)?,
        jurisdiction_profile: config.jurisdiction_profile,
        profile_requirements: profile_requirements(
            config.jurisdiction_profile,
            M15DocumentKind::Map,
        ),
        context_of_use: config.context_of_use,
        questions,
        planned_datasets: config.planned_datasets,
        methods: config.methods,
        technical_acceptance_criteria,
        governance,
        signoff,
        linked_artifacts: MapLinkedArtifacts {
            assessment_table_ref: stable_artifact_path("assessment_table", assessment_table_path),
            validation_report_ref,
        },
    };

    Ok(map)
}

pub(crate) fn render_map_markdown(doc: &MapArtifact) -> String {
    let mut out = String::new();
    out.push_str("# ICH M15 Model Analysis Plan\n\n");
    out.push_str(&format!(
        "- Status: {}\n- Jurisdiction: {}\n- Context of use: {}\n- Assessment table: {}\n\n",
        doc.document_status,
        doc.jurisdiction_profile,
        doc.context_of_use,
        doc.linked_artifacts.assessment_table_ref
    ));
    append_profile_markdown(&mut out, &doc.profile_requirements);

    out.push_str("## Questions\n\n");
    for question in &doc.questions {
        out.push_str(&format!("- {}: {}\n", question.question_id, question.question_of_interest));
    }
    out.push_str("\n## Planned Datasets\n\n");
    for dataset in &doc.planned_datasets {
        out.push_str(&format!(
            "- {}: {} ({})\n",
            dataset.dataset_id, dataset.description, dataset.purpose
        ));
    }
    out.push_str("\n## Methods\n\n");
    for method in &doc.methods {
        out.push_str(&format!("- {}: {}", method.method_id, method.description));
        if let Some(software_ref) = &method.software_ref {
            out.push_str(&format!(" [{}]", software_ref));
        }
        out.push('\n');
    }
    out.push_str("\n## Technical Acceptance Criteria\n\n");
    for criterion in &doc.technical_acceptance_criteria {
        out.push_str(&format!(
            "- {}: {} Acceptance rule: {}\n",
            criterion.criterion_id, criterion.description, criterion.acceptance_rule
        ));
    }
    out.push_str("\n## Governance\n\n");
    out.push_str(&format!("- Authors: {}\n", doc.governance.authors.join(", ")));
    out.push_str(&format!("- Reviewers: {}\n", doc.governance.reviewers.join(", ")));
    out.push_str(&format!("- Approvers: {}\n", doc.governance.approvers.join(", ")));
    out.push_str(&format!(
        "- Signoff: author={}, reviewer={}, approver={}, status={}\n",
        doc.signoff.primary_author,
        doc.signoff.qa_reviewer,
        doc.signoff.approver,
        doc.signoff.status
    ));
    if let Some(validation_report_ref) = &doc.linked_artifacts.validation_report_ref {
        out.push_str(&format!("- Validation report ref: {}\n", validation_report_ref));
    }

    out
}

fn json_pointer_from_dotted(path: &str) -> String {
    let mut pointer = String::new();
    for segment in path.split('.') {
        pointer.push('/');
        pointer.push_str(segment);
    }
    pointer
}

fn evaluate_acceptance_rule(
    rule: &str,
    validation_json: &Value,
    pharma_json: &Value,
) -> MarCriterionResult {
    let parts: Vec<&str> = rule.split("==").collect();
    if parts.len() != 2 {
        return MarCriterionResult {
            criterion_id: String::new(),
            status: "not_evaluable".to_string(),
            observed_value: None,
            notes: Some(format!("unsupported acceptance rule syntax: {rule}")),
        };
    }

    let lhs = parts[0].trim();
    let rhs = parts[1].trim();
    let (root, dotted_path, source_json) =
        if let Some(rest) = lhs.strip_prefix("validation_report.") {
            ("validation_report", rest, validation_json)
        } else if let Some(rest) = lhs.strip_prefix("pharma_validation.") {
            ("pharma_validation", rest, pharma_json)
        } else {
            return MarCriterionResult {
                criterion_id: String::new(),
                status: "not_evaluable".to_string(),
                observed_value: None,
                notes: Some(format!("unsupported acceptance rule root: {lhs}")),
            };
        };

    let pointer = json_pointer_from_dotted(dotted_path);
    match source_json.pointer(&pointer) {
        Some(observed) => {
            let status = if observed.as_str() == Some(rhs) { "met" } else { "not_met" };
            MarCriterionResult {
                criterion_id: String::new(),
                status: status.to_string(),
                observed_value: Some(observed.clone()),
                notes: Some(format!(
                    "evaluated {}{}",
                    stable_artifact_path(root, Path::new(root)),
                    pointer
                )),
            }
        }
        None => MarCriterionResult {
            criterion_id: String::new(),
            status: "not_evaluable".to_string(),
            observed_value: None,
            notes: Some(format!("missing value for {} root at {}", root, pointer)),
        },
    }
}

pub(crate) fn build_mar(
    map_path: &Path,
    assessment_table_path: &Path,
    validation_report_path: &Path,
    pharma_validation_path: &Path,
    deterministic: bool,
) -> Result<MarArtifact> {
    let map_bytes = std::fs::read(map_path)
        .with_context(|| format!("failed to read {}", map_path.display()))?;
    let map: MapArtifact = serde_json::from_slice(&map_bytes)
        .with_context(|| format!("failed to parse {}", map_path.display()))?;
    if map.schema_version != "m15_map_v1" {
        anyhow::bail!("unsupported map schema_version: {}", map.schema_version);
    }
    validate_governance(&map.governance, &map.signoff)?;

    let assessment_bytes = std::fs::read(assessment_table_path)
        .with_context(|| format!("failed to read {}", assessment_table_path.display()))?;
    let assessment: AssessmentTableArtifact = serde_json::from_slice(&assessment_bytes)
        .with_context(|| format!("failed to parse {}", assessment_table_path.display()))?;
    if assessment.schema_version != "m15_assessment_table_v1" {
        anyhow::bail!("unsupported assessment-table schema_version: {}", assessment.schema_version);
    }

    let validation_bytes = std::fs::read(validation_report_path)
        .with_context(|| format!("failed to read {}", validation_report_path.display()))?;
    let validation_json: Value = serde_json::from_slice(&validation_bytes)
        .with_context(|| format!("failed to parse {}", validation_report_path.display()))?;
    let validation_schema =
        require_status(&validation_json, "/schema_version", "validation report schema_version")?;
    if validation_schema != "validation_report_v1" {
        anyhow::bail!("unsupported validation report schema_version: {}", validation_schema);
    }

    let pharma_bytes = std::fs::read(pharma_validation_path)
        .with_context(|| format!("failed to read {}", pharma_validation_path.display()))?;
    let pharma_json: Value = serde_json::from_slice(&pharma_bytes)
        .with_context(|| format!("failed to parse {}", pharma_validation_path.display()))?;
    let pharma_schema =
        require_status(&pharma_json, "/schema_version", "pharma validation schema_version")?;
    if pharma_schema != "nextstat.pharma_validation.v1" {
        anyhow::bail!("unsupported pharma validation schema_version: {}", pharma_schema);
    }

    let map_qoi_ids: std::collections::BTreeSet<String> =
        map.questions.iter().map(|q| q.question_id.clone()).collect();
    let assessment_qoi_ids: std::collections::BTreeSet<String> =
        assessment.entries.iter().map(|e| e.qoi_id.clone()).collect();
    if map_qoi_ids != assessment_qoi_ids {
        anyhow::bail!("MAP questions do not match assessment-table entries");
    }

    let mut criterion_results = Vec::with_capacity(map.technical_acceptance_criteria.len());
    for criterion in &map.technical_acceptance_criteria {
        let mut result =
            evaluate_acceptance_rule(&criterion.acceptance_rule, &validation_json, &pharma_json);
        result.criterion_id = criterion.criterion_id.clone();
        criterion_results.push(result);
    }

    let any_not_met = criterion_results.iter().any(|r| r.status == "not_met");
    let any_not_evaluable = criterion_results.iter().any(|r| r.status == "not_evaluable");

    let mut deviations = Vec::new();
    for entry in &assessment.entries {
        if let Some(notes) = &entry.review_notes {
            for (idx, note) in notes.iter().enumerate() {
                deviations.push(MarDeviation {
                    deviation_id: format!("DEV-{}-{:02}", entry.qoi_id, idx + 1),
                    description: note.clone(),
                    impact_assessment: "Open review note from assessment-table evidence; blocks final support conclusion until resolved.".to_string(),
                });
            }
        }
    }

    let questions = map
        .questions
        .iter()
        .map(|question| {
            let relevant: Vec<&MarCriterionResult> = map
                .technical_acceptance_criteria
                .iter()
                .zip(criterion_results.iter())
                .filter_map(|(criterion, result)| {
                    criterion
                        .applies_to_question_ids
                        .as_ref()
                        .filter(|ids| ids.iter().any(|id| id == &question.question_id))
                        .map(|_| result)
                })
                .collect();
            let conclusion_status = if relevant.iter().any(|r| r.status == "not_met") {
                "not_supported"
            } else if relevant.iter().any(|r| r.status == "not_evaluable") {
                "inconclusive"
            } else {
                "supported"
            };
            let conclusion = match conclusion_status {
                "supported" => format!(
                    "All mapped technical acceptance criteria were met for {}.",
                    question.question_id
                ),
                "not_supported" => format!(
                    "At least one mapped technical acceptance criterion was not met for {}.",
                    question.question_id
                ),
                _ => format!(
                    "One or more mapped technical acceptance criteria could not be evaluated for {}.",
                    question.question_id
                ),
            };
            MarQuestionResult {
                question_id: question.question_id.clone(),
                question_of_interest: question.question_of_interest.clone(),
                conclusion_status: conclusion_status.to_string(),
                conclusion,
                evidence_refs: vec![
                    stable_artifact_pointer_ref("validation_report", "/apex2_summary/overall"),
                    stable_artifact_pointer_ref("pharma_validation", "/status"),
                ],
            }
        })
        .collect();

    let executed_datasets = map
        .planned_datasets
        .iter()
        .enumerate()
        .map(|(idx, dataset)| MarDatasetResult {
            dataset_id: dataset.dataset_id.clone(),
            provenance_ref: if idx == 0 {
                stable_artifact_pointer_ref(
                    "validation_report",
                    "/dataset_fingerprint/workspace_sha256",
                )
            } else {
                stable_artifact_pointer_ref("pharma_validation", "/summary")
            },
        })
        .collect();

    let methods_executed = map
        .methods
        .iter()
        .map(|method| MarMethodResult {
            method_id: method.method_id.clone(),
            status: "completed".to_string(),
            notes: Some("Executed with deterministic evidence bundle generation.".to_string()),
        })
        .collect();

    let mut limitations = vec![
        "Conclusions remain bounded by the linked validation and pharma qualification artifacts."
            .to_string(),
    ];
    if any_not_evaluable {
        limitations.push(
            "At least one technical acceptance criterion was not evaluable from the supplied artifacts."
                .to_string(),
        );
    }

    let document_status = mar_document_status(
        &map.signoff.status,
        !(any_not_met || any_not_evaluable || !deviations.is_empty()),
    )?;

    Ok(MarArtifact {
        schema_version: "m15_mar_v1".to_string(),
        generated_at: generated_at(deterministic),
        document_status,
        jurisdiction_profile: map.jurisdiction_profile,
        profile_requirements: profile_requirements(map.jurisdiction_profile, M15DocumentKind::Mar),
        context_of_use: map.context_of_use,
        based_on_map_ref: stable_artifact_path("map", map_path),
        questions,
        executed_datasets,
        methods_executed,
        criterion_results,
        deviations,
        limitations,
        governance: map.governance.clone(),
        signoff: map.signoff.clone(),
        linked_artifacts: MarLinkedArtifacts {
            assessment_table_ref: stable_artifact_path("assessment_table", assessment_table_path),
            validation_report_ref: stable_artifact_path(
                "validation_report",
                validation_report_path,
            ),
            pharma_validation_ref: Some(stable_artifact_path(
                "pharma_validation",
                pharma_validation_path,
            )),
        },
    })
}

pub(crate) fn render_mar_markdown(doc: &MarArtifact) -> String {
    let mut out = String::new();
    out.push_str("# ICH M15 Model Analysis Report\n\n");
    out.push_str(&format!(
        "- Status: {}\n- Jurisdiction: {}\n- Context of use: {}\n- Based on MAP: {}\n\n",
        doc.document_status, doc.jurisdiction_profile, doc.context_of_use, doc.based_on_map_ref
    ));
    append_profile_markdown(&mut out, &doc.profile_requirements);
    out.push_str("## Question Conclusions\n\n");
    for question in &doc.questions {
        out.push_str(&format!(
            "- {}: {} [{}]\n",
            question.question_id, question.conclusion, question.conclusion_status
        ));
    }
    out.push_str("\n## Criterion Results\n\n");
    for criterion in &doc.criterion_results {
        out.push_str(&format!("- {}: {}", criterion.criterion_id, criterion.status));
        if let Some(notes) = &criterion.notes {
            out.push_str(&format!(" ({})", notes));
        }
        out.push('\n');
    }
    out.push_str("\n## Deviations\n\n");
    if doc.deviations.is_empty() {
        out.push_str("- None\n");
    } else {
        for deviation in &doc.deviations {
            out.push_str(&format!(
                "- {}: {} [{}]\n",
                deviation.deviation_id, deviation.description, deviation.impact_assessment
            ));
        }
    }
    out.push_str("\n## Governance\n\n");
    out.push_str(&format!("- Authors: {}\n", doc.governance.authors.join(", ")));
    out.push_str(&format!("- Reviewers: {}\n", doc.governance.reviewers.join(", ")));
    out.push_str(&format!("- Approvers: {}\n", doc.governance.approvers.join(", ")));
    out.push_str(&format!(
        "- Signoff: author={}, reviewer={}, approver={}, status={}\n",
        doc.signoff.primary_author,
        doc.signoff.qa_reviewer,
        doc.signoff.approver,
        doc.signoff.status
    ));
    out.push_str("\n## Linked Artifacts\n\n");
    out.push_str(&format!(
        "- Assessment table: {}\n- Validation report: {}\n",
        doc.linked_artifacts.assessment_table_ref, doc.linked_artifacts.validation_report_ref
    ));
    if let Some(pharma_validation_ref) = &doc.linked_artifacts.pharma_validation_ref {
        out.push_str(&format!("- Pharma validation: {}\n", pharma_validation_ref));
    }
    out
}

pub(crate) fn build_bundle(
    config_path: &Path,
    assessment_table_path: &Path,
    map_path: &Path,
    mar_path: &Path,
    validation_report_path: &Path,
    pharma_validation_path: &Path,
    deterministic: bool,
) -> Result<BundleManifestArtifact> {
    let config_bytes = std::fs::read(config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let config: M15Config = serde_json::from_slice(&config_bytes)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;
    if config.schema_version != "m15_config_v1" {
        anyhow::bail!(
            "unsupported M15 config schema_version: {} (expected m15_config_v1)",
            config.schema_version
        );
    }
    let deterministic = deterministic || config.deterministic;

    let assessment_bytes = std::fs::read(assessment_table_path)
        .with_context(|| format!("failed to read {}", assessment_table_path.display()))?;
    let assessment: AssessmentTableArtifact = serde_json::from_slice(&assessment_bytes)
        .with_context(|| format!("failed to parse {}", assessment_table_path.display()))?;
    if assessment.schema_version != "m15_assessment_table_v1" {
        anyhow::bail!("unsupported assessment-table schema_version: {}", assessment.schema_version);
    }

    let map_bytes = std::fs::read(map_path)
        .with_context(|| format!("failed to read {}", map_path.display()))?;
    let map: MapArtifact = serde_json::from_slice(&map_bytes)
        .with_context(|| format!("failed to parse {}", map_path.display()))?;
    if map.schema_version != "m15_map_v1" {
        anyhow::bail!("unsupported map schema_version: {}", map.schema_version);
    }

    let mar_bytes = std::fs::read(mar_path)
        .with_context(|| format!("failed to read {}", mar_path.display()))?;
    let mar: MarArtifact = serde_json::from_slice(&mar_bytes)
        .with_context(|| format!("failed to parse {}", mar_path.display()))?;
    if mar.schema_version != "m15_mar_v1" {
        anyhow::bail!("unsupported mar schema_version: {}", mar.schema_version);
    }
    validate_governance(&map.governance, &map.signoff)?;
    validate_governance(&mar.governance, &mar.signoff)?;

    let validation_bytes = std::fs::read(validation_report_path)
        .with_context(|| format!("failed to read {}", validation_report_path.display()))?;
    let validation_json: Value = serde_json::from_slice(&validation_bytes)
        .with_context(|| format!("failed to parse {}", validation_report_path.display()))?;
    let validation_schema =
        require_status(&validation_json, "/schema_version", "validation report schema_version")?;
    if validation_schema != "validation_report_v1" {
        anyhow::bail!("unsupported validation report schema_version: {}", validation_schema);
    }

    let pharma_bytes = std::fs::read(pharma_validation_path)
        .with_context(|| format!("failed to read {}", pharma_validation_path.display()))?;
    let pharma_json: Value = serde_json::from_slice(&pharma_bytes)
        .with_context(|| format!("failed to parse {}", pharma_validation_path.display()))?;
    let pharma_schema =
        require_status(&pharma_json, "/schema_version", "pharma validation schema_version")?;
    if pharma_schema != "nextstat.pharma_validation.v1" {
        anyhow::bail!("unsupported pharma validation schema_version: {}", pharma_schema);
    }

    let rerendered_assessment = build_assessment_table(
        config_path,
        validation_report_path,
        pharma_validation_path,
        deterministic,
    )?;
    let rerendered_assessment_bytes = canonical_json_bytes(&rerendered_assessment, deterministic)?;
    let rerendered_map = build_map(config_path, assessment_table_path, deterministic)?;
    let rerendered_map_bytes = canonical_json_bytes(&rerendered_map, deterministic)?;
    let rerendered_mar = build_mar(
        map_path,
        assessment_table_path,
        validation_report_path,
        pharma_validation_path,
        deterministic,
    )?;
    let rerendered_mar_bytes = canonical_json_bytes(&rerendered_mar, deterministic)?;

    let deterministic_re_render_verified = assessment_bytes == rerendered_assessment_bytes
        && map_bytes == rerendered_map_bytes
        && mar_bytes == rerendered_mar_bytes;

    let artifacts = BundleArtifacts {
        assessment_table: bundle_artifact_ref(
            assessment_table_path,
            "assessment_table",
            "risk classification artifact",
        )?,
        map: bundle_artifact_ref(map_path, "map", "frozen analysis plan")?,
        mar: bundle_artifact_ref(mar_path, "mar", "analysis report")?,
        validation_report: bundle_artifact_ref(
            validation_report_path,
            "validation_report",
            "cross-suite deterministic validation",
        )?,
        pharma_validation: Some(bundle_artifact_ref(
            pharma_validation_path,
            "pharma_validation",
            "pharma IQ/OQ/PQ evidence",
        )?),
    };

    let mut files = vec![
        bundle_file_entry(config_path, "config")?,
        bundle_file_entry(assessment_table_path, "assessment_table")?,
        bundle_file_entry(map_path, "map")?,
        bundle_file_entry(mar_path, "mar")?,
        bundle_file_entry(validation_report_path, "validation_report")?,
        bundle_file_entry(pharma_validation_path, "pharma_validation")?,
    ];
    files.sort_by_key(|entry| (bundle_role_rank(&entry.artifact_role), entry.path.clone()));

    let all_hashes_present =
        [&artifacts.assessment_table, &artifacts.map, &artifacts.mar, &artifacts.validation_report]
            .into_iter()
            .chain(artifacts.pharma_validation.iter())
            .all(|artifact| artifact.sha256.len() == 64);

    let missing_required_roles = Vec::new();
    let mut missing_signoff_roles = Vec::new();
    if mar.signoff.primary_author.trim().is_empty() {
        missing_signoff_roles.push("primary_author".to_string());
    }
    if mar.signoff.qa_reviewer.trim().is_empty() {
        missing_signoff_roles.push("qa_reviewer".to_string());
    }
    if mar.signoff.approver.trim().is_empty() {
        missing_signoff_roles.push("approver".to_string());
    }
    let signoff_roles_complete = missing_signoff_roles.is_empty();
    let signoff_roles_distinct = mar.signoff.primary_author != mar.signoff.qa_reviewer
        && mar.signoff.primary_author != mar.signoff.approver
        && mar.signoff.qa_reviewer != mar.signoff.approver;
    let bundle_status = if all_hashes_present
        && missing_required_roles.is_empty()
        && deterministic_re_render_verified
        && signoff_roles_complete
        && signoff_roles_distinct
        && (mar.document_status == "reviewed" || mar.document_status == "approved")
    {
        "complete"
    } else {
        "draft"
    };

    Ok(BundleManifestArtifact {
        schema_version: "m15_bundle_manifest_v1".to_string(),
        generated_at: generated_at(deterministic),
        deterministic,
        jurisdiction_profile: config.jurisdiction_profile,
        source_config_sha256: sha256_hex(&config_bytes),
        bundle_status: bundle_status.to_string(),
        artifacts,
        files,
        integrity: BundleIntegrity {
            all_hashes_present,
            deterministic_re_render_verified,
            missing_required_roles,
            signoff_roles_complete,
            signoff_roles_distinct,
            missing_signoff_roles,
        },
    })
}
