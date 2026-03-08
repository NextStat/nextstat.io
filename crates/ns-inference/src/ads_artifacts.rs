//! Shared statistical artifact contract for ads inference surfaces.
//!
//! This module defines a reusable artifact envelope that can be consumed by
//! cloud transport, calculator/reporting surfaces, and MCP/agent integrations
//! without re-inventing ads-specific claimability or systematics semantics in
//! each layer.

use serde::{Deserialize, Serialize};
use serde_json::json;
use thiserror::Error;

use crate::ads_systematics::{
    AdsCombinationPolicy, AdsSystematic, AdsSystematicKind, AdsSystematicSourceClass,
    AdsUncertaintyLayer, default_systematic_by_code, default_systematic_by_id,
    default_systematics_slice,
};

pub const ADS_STATISTICAL_ARTIFACT_TYPE: &str = "ads_statistical_artifact";
pub const ADS_STATISTICAL_ARTIFACT_VERSION: &str = "1.0.0";
pub const ADS_SYSTEMATICS_REGISTRY_VERSION: &str = "v1";
pub const ADS_STATISTICAL_ARTIFACT_SCHEMA_ID: &str =
    "https://nextstat.io/schemas/ads/ads_statistical_artifact_v1.schema.json";

#[derive(Debug, Clone, PartialEq, Error)]
pub enum AdsArtifactValidationError {
    #[error("ads statistical artifact type mismatch: expected `{expected}`, got `{actual}`")]
    ArtifactTypeMismatch { expected: &'static str, actual: String },
    #[error("ads statistical artifact version mismatch: expected `{expected}`, got `{actual}`")]
    ArtifactVersionMismatch { expected: &'static str, actual: String },
    #[error(
        "ads statistical artifact systematics registry mismatch: expected `{expected}`, got `{actual}`"
    )]
    RegistryVersionMismatch { expected: &'static str, actual: String },
    #[error("ads statistical artifact `{field}` interval is invalid: lo {lo} > hi {hi}")]
    InvalidInterval { field: &'static str, lo: f64, hi: f64 },
    #[error("ads statistical artifact total uncertainty status `{status}` requires an interval")]
    MissingTotalInterval { status: &'static str },
    #[error(
        "ads statistical artifact total uncertainty status `{status}` must not include an interval"
    )]
    UnexpectedTotalInterval { status: &'static str },
    #[error("ads statistical artifact guardrail `{field}` must be non-negative, got {value}")]
    NegativeGuardrail { field: &'static str, value: f64 },
    #[error(
        "ads statistical artifact sequential looks are inconsistent: used {used} exceeds planned {planned}"
    )]
    InvalidSequentialLooks { used: i32, planned: i32 },
    #[error(
        "ads statistical artifact systematics entry count mismatch: expected {expected}, got {actual}"
    )]
    SystematicsEntryCountMismatch { expected: usize, actual: usize },
    #[error(
        "ads statistical artifact systematics entry #{index} expected code `{expected}`, got `{actual}`"
    )]
    SystematicCodeOrderMismatch { index: usize, expected: &'static str, actual: String },
    #[error(
        "ads statistical artifact systematics entry `{code}` has unknown systematic id `{systematic_id}`"
    )]
    UnknownSystematicCode { code: String, systematic_id: String },
    #[error(
        "ads statistical artifact systematics entry `{code}` must use canonical systematic id `{expected}`, got `{actual}`"
    )]
    SystematicIdMismatch { code: String, expected: &'static str, actual: String },
    #[error(
        "ads statistical artifact systematics entry `{code}` must preserve `{field}` from the shared registry"
    )]
    SystematicSemanticMismatch { code: String, field: &'static str },
    #[error("ads statistical artifact JSON failed to deserialize: {0}")]
    JsonDeserialization(String),
    #[error("ads statistical artifact JSON shape is non-canonical after validation")]
    NonCanonicalJsonShape,
    #[error("ads statistical artifact JSON failed to serialize: {0}")]
    JsonSerialization(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdsArtifactDecision {
    Continue,
    StopWinner,
    StopNoDifference,
    StopGuardrail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdsClaimStatus {
    ClaimableWin,
    PositiveSignalButNotClaimable,
    ClaimableNoDifference,
    NoDifferenceButNotClaimable,
    GuardrailBlocked,
    Inconclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdsPracticalThresholdStatus {
    ThresholdNotDeclared,
    ExceedsWinThreshold,
    ExceedsLossThreshold,
    InsideEquivalenceBand,
    InconclusiveRelativeToThreshold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdsArtifactSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdsCausalUncertaintyStatus {
    Separate,
    Combined,
    NotAvailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdsTotalUncertaintyStatus {
    Combined,
    NotCombinable,
    NotAvailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdsFitQuality {
    Good,
    Warning,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdsUncertaintyInterval {
    pub lo: f64,
    pub hi: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdsCausalUncertainty {
    pub status: AdsCausalUncertaintyStatus,
    #[serde(default)]
    pub diagnostics: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdsTotalUncertainty {
    pub status: AdsTotalUncertaintyStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interval: Option<AdsUncertaintyInterval>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdsClsGuardrail {
    pub enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_sensitivity_sigma: Option<f64>,
    pub claim_status: AdsClaimStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdsDecisionLossView {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hold: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revert: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdsPracticalSignificance {
    pub threshold_status: AdsPracticalThresholdStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta_win: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta_loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta_equiv: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tost_status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decision_loss: Option<AdsDecisionLossView>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdsArtifactResult {
    pub effect_estimate: f64,
    pub sampling_uncertainty: AdsUncertaintyInterval,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub measurement_uncertainty: Option<AdsUncertaintyInterval>,
    pub causal_uncertainty: AdsCausalUncertainty,
    pub total_uncertainty: AdsTotalUncertainty,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cls: Option<AdsClsGuardrail>,
    pub claim_status: AdsClaimStatus,
    pub practical_significance: AdsPracticalSignificance,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdsArtifactDiagnostics {
    pub fit_quality: AdsFitQuality,
    pub convergence: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gof_pvalue: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dominant_systematic: Option<String>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdsEventCounts {
    pub control: i64,
    pub variant: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdsArtifactGuardrails {
    pub minimum_events: i64,
    pub actual_events: AdsEventCounts,
    pub power_at_mde: f64,
    pub sequential_looks_used: i32,
    pub sequential_looks_planned: i32,
    pub alpha_spent: f64,
    pub alpha_remaining: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdsArtifactAssumption {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdsArtifactContractReference {
    #[serde(rename = "type")]
    pub artifact_type: String,
    pub version: String,
    pub systematics_registry_version: String,
}

impl AdsArtifactContractReference {
    pub fn shared() -> Self {
        Self {
            artifact_type: ADS_STATISTICAL_ARTIFACT_TYPE.to_string(),
            version: ADS_STATISTICAL_ARTIFACT_VERSION.to_string(),
            systematics_registry_version: ADS_SYSTEMATICS_REGISTRY_VERSION.to_string(),
        }
    }
}

pub fn ads_artifact_contract_reference_json_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "required": ["type", "version", "systematics_registry_version"],
        "properties": {
            "type": { "type": "string", "const": ADS_STATISTICAL_ARTIFACT_TYPE },
            "version": { "type": "string", "const": ADS_STATISTICAL_ARTIFACT_VERSION },
            "systematics_registry_version": {
                "type": "string",
                "const": ADS_SYSTEMATICS_REGISTRY_VERSION
            }
        },
        "additionalProperties": false
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdsArtifactSemanticContext {
    pub contract: AdsArtifactContractReference,
    pub systematics_profile: AdsSystematicsProfile,
    #[serde(default)]
    pub assumptions: Vec<AdsArtifactAssumption>,
}

impl AdsArtifactSemanticContext {
    pub fn new(
        systematics_profile: AdsSystematicsProfile,
        assumptions: Vec<AdsArtifactAssumption>,
    ) -> Self {
        Self { contract: AdsArtifactContractReference::shared(), systematics_profile, assumptions }
    }
}

pub fn ads_artifact_semantic_context_json_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "required": ["contract", "systematics_profile", "assumptions"],
        "properties": {
            "contract": ads_artifact_contract_reference_json_schema(),
            "systematics_profile": {
                "type": "object",
                "required": ["registry_version", "entries"],
                "properties": {
                    "registry_version": {
                        "type": "string",
                        "const": ADS_SYSTEMATICS_REGISTRY_VERSION
                    },
                    "entries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "code",
                                "systematic_id",
                                "enabled",
                                "uncertainty_layer",
                                "kind",
                                "source_class",
                                "combination_policy",
                                "nominal",
                                "sigma",
                                "constraint"
                            ],
                            "properties": {
                                "code": { "type": "string" },
                                "systematic_id": { "type": "string" },
                                "enabled": { "type": "boolean" },
                                "uncertainty_layer": {
                                    "type": "string",
                                    "enum": ["measurement", "causal"]
                                },
                                "kind": {
                                    "type": "string",
                                    "enum": ["norm_sys", "histo_sys"]
                                },
                                "source_class": {
                                    "type": "string",
                                    "enum": [
                                        "data_estimated",
                                        "hybrid_api_plus_benchmark",
                                        "hybrid_api_plus_prior",
                                        "policy_default_or_vendor_calibrated",
                                        "policy_default_or_incrementality_calibrated",
                                        "internal_nextstat_diagnostic"
                                    ]
                                },
                                "combination_policy": {
                                    "type": "string",
                                    "enum": [
                                        "profile_likelihood",
                                        "scenario_only",
                                        "sensitivity_first_then_profile",
                                        "sensitivity_only"
                                    ]
                                },
                                "nominal": { "type": "string" },
                                "sigma": { "type": "string" },
                                "constraint": { "type": "string" }
                            },
                            "additionalProperties": false
                        }
                    }
                },
                "additionalProperties": false
            },
            "assumptions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["key", "value"],
                    "properties": {
                        "key": { "type": "string" },
                        "value": { "type": "string" }
                    },
                    "additionalProperties": false
                }
            }
        },
        "additionalProperties": false
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdsArtifactMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_hash: Option<String>,
    pub engine_version: String,
    pub model_version: String,
    pub created_at: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdsClaimViolation {
    pub rule: String,
    pub severity: AdsArtifactSeverity,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdsLeakageIndicator {
    pub name: String,
    pub detected: bool,
    pub severity: AdsArtifactSeverity,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdsArtifactRigor {
    pub passed: bool,
    pub artifacts_complete: bool,
    pub claim_valid: bool,
    pub leakage_clean: bool,
    #[serde(default)]
    pub claim_violations: Vec<AdsClaimViolation>,
    #[serde(default)]
    pub leakage_indicators: Vec<AdsLeakageIndicator>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdsSystematicsProfileEntry {
    pub code: String,
    pub systematic_id: String,
    pub enabled: bool,
    pub uncertainty_layer: AdsUncertaintyLayer,
    pub kind: AdsSystematicKind,
    pub source_class: AdsSystematicSourceClass,
    pub combination_policy: AdsCombinationPolicy,
    pub nominal: String,
    pub sigma: String,
    pub constraint: String,
}

impl From<AdsSystematic> for AdsSystematicsProfileEntry {
    fn from(systematic: AdsSystematic) -> Self {
        Self {
            code: systematic.code.to_string(),
            systematic_id: systematic.systematic_id.to_string(),
            enabled: systematic.default_enabled,
            uncertainty_layer: systematic.uncertainty_layer,
            kind: systematic.kind,
            source_class: systematic.source_class,
            combination_policy: systematic.combination_policy,
            nominal: systematic.default_nominal.to_string(),
            sigma: systematic.default_sigma.to_string(),
            constraint: systematic.constraint.to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdsSystematicsProfile {
    pub registry_version: String,
    pub entries: Vec<AdsSystematicsProfileEntry>,
}

impl AdsSystematicsProfile {
    pub fn from_defaults() -> Self {
        Self {
            registry_version: ADS_SYSTEMATICS_REGISTRY_VERSION.to_string(),
            entries: default_systematics_slice()
                .iter()
                .copied()
                .map(AdsSystematicsProfileEntry::from)
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdsArtifactReferenceSummary {
    pub contract: AdsArtifactContractReference,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decision: Option<AdsArtifactDecision>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub claim_status: Option<AdsClaimStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
}

impl From<&AdsStatisticalArtifact> for AdsArtifactReferenceSummary {
    fn from(artifact: &AdsStatisticalArtifact) -> Self {
        Self {
            contract: AdsArtifactContractReference::shared(),
            decision: Some(artifact.decision),
            claim_status: Some(artifact.result.claim_status),
            created_at: Some(artifact.metadata.created_at.clone()),
        }
    }
}

pub fn ads_artifact_reference_summary_json_schema() -> serde_json::Value {
    json!({
        "type": "object",
        "required": ["contract"],
        "properties": {
            "contract": ads_artifact_contract_reference_json_schema(),
            "decision": {
                "type": ["string", "null"],
                "enum": [
                    "continue",
                    "stop_winner",
                    "stop_no_difference",
                    "stop_guardrail",
                    null
                ]
            },
            "claim_status": {
                "type": ["string", "null"],
                "enum": [
                    "claimable_win",
                    "positive_signal_but_not_claimable",
                    "claimable_no_difference",
                    "no_difference_but_not_claimable",
                    "guardrail_blocked",
                    "inconclusive",
                    null
                ]
            },
            "created_at": { "type": ["string", "null"] }
        },
        "additionalProperties": false
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdsStatisticalArtifact {
    #[serde(rename = "type")]
    pub artifact_type: String,
    pub version: String,
    pub decision: AdsArtifactDecision,
    pub result: AdsArtifactResult,
    pub diagnostics: AdsArtifactDiagnostics,
    pub guardrails: AdsArtifactGuardrails,
    #[serde(default)]
    pub assumptions: Vec<AdsArtifactAssumption>,
    pub systematics_profile: AdsSystematicsProfile,
    pub rigor: AdsArtifactRigor,
    pub metadata: AdsArtifactMetadata,
}

impl AdsStatisticalArtifact {
    pub fn new(
        decision: AdsArtifactDecision,
        result: AdsArtifactResult,
        diagnostics: AdsArtifactDiagnostics,
        guardrails: AdsArtifactGuardrails,
        metadata: AdsArtifactMetadata,
    ) -> Self {
        Self {
            artifact_type: ADS_STATISTICAL_ARTIFACT_TYPE.to_string(),
            version: ADS_STATISTICAL_ARTIFACT_VERSION.to_string(),
            decision,
            result,
            diagnostics,
            guardrails,
            assumptions: Vec::new(),
            systematics_profile: AdsSystematicsProfile::from_defaults(),
            rigor: AdsArtifactRigor {
                passed: true,
                artifacts_complete: true,
                claim_valid: true,
                leakage_clean: true,
                claim_violations: Vec::new(),
                leakage_indicators: Vec::new(),
            },
            metadata,
        }
    }
}

pub fn validate_ads_statistical_artifact(
    artifact: &AdsStatisticalArtifact,
) -> Result<(), AdsArtifactValidationError> {
    if artifact.artifact_type != ADS_STATISTICAL_ARTIFACT_TYPE {
        return Err(AdsArtifactValidationError::ArtifactTypeMismatch {
            expected: ADS_STATISTICAL_ARTIFACT_TYPE,
            actual: artifact.artifact_type.clone(),
        });
    }
    if artifact.version != ADS_STATISTICAL_ARTIFACT_VERSION {
        return Err(AdsArtifactValidationError::ArtifactVersionMismatch {
            expected: ADS_STATISTICAL_ARTIFACT_VERSION,
            actual: artifact.version.clone(),
        });
    }
    if artifact.systematics_profile.registry_version != ADS_SYSTEMATICS_REGISTRY_VERSION {
        return Err(AdsArtifactValidationError::RegistryVersionMismatch {
            expected: ADS_SYSTEMATICS_REGISTRY_VERSION,
            actual: artifact.systematics_profile.registry_version.clone(),
        });
    }

    validate_interval("sampling_uncertainty", &artifact.result.sampling_uncertainty)?;
    if let Some(interval) = artifact.result.measurement_uncertainty.as_ref() {
        validate_interval("measurement_uncertainty", interval)?;
    }
    match (
        artifact.result.total_uncertainty.status,
        artifact.result.total_uncertainty.interval.as_ref(),
    ) {
        (AdsTotalUncertaintyStatus::Combined, Some(interval)) => {
            validate_interval("total_uncertainty", interval)?;
        }
        (AdsTotalUncertaintyStatus::Combined, None) => {
            return Err(AdsArtifactValidationError::MissingTotalInterval { status: "combined" });
        }
        (AdsTotalUncertaintyStatus::NotCombinable, Some(_)) => {
            return Err(AdsArtifactValidationError::UnexpectedTotalInterval {
                status: "not_combinable",
            });
        }
        (AdsTotalUncertaintyStatus::NotAvailable, Some(_)) => {
            return Err(AdsArtifactValidationError::UnexpectedTotalInterval {
                status: "not_available",
            });
        }
        (_, None) => {}
    }

    validate_non_negative_guardrail("minimum_events", artifact.guardrails.minimum_events as f64)?;
    validate_non_negative_guardrail(
        "actual_events.control",
        artifact.guardrails.actual_events.control as f64,
    )?;
    validate_non_negative_guardrail(
        "actual_events.variant",
        artifact.guardrails.actual_events.variant as f64,
    )?;
    validate_non_negative_guardrail("power_at_mde", artifact.guardrails.power_at_mde)?;
    validate_non_negative_guardrail(
        "sequential_looks_used",
        artifact.guardrails.sequential_looks_used as f64,
    )?;
    validate_non_negative_guardrail(
        "sequential_looks_planned",
        artifact.guardrails.sequential_looks_planned as f64,
    )?;
    validate_non_negative_guardrail("alpha_spent", artifact.guardrails.alpha_spent)?;
    validate_non_negative_guardrail("alpha_remaining", artifact.guardrails.alpha_remaining)?;
    if artifact.guardrails.sequential_looks_used > artifact.guardrails.sequential_looks_planned {
        return Err(AdsArtifactValidationError::InvalidSequentialLooks {
            used: artifact.guardrails.sequential_looks_used,
            planned: artifact.guardrails.sequential_looks_planned,
        });
    }

    let defaults = default_systematics_slice();
    if artifact.systematics_profile.entries.len() != defaults.len() {
        return Err(AdsArtifactValidationError::SystematicsEntryCountMismatch {
            expected: defaults.len(),
            actual: artifact.systematics_profile.entries.len(),
        });
    }
    for (index, (entry, expected)) in
        artifact.systematics_profile.entries.iter().zip(defaults.iter()).enumerate()
    {
        if entry.code != expected.code {
            return Err(AdsArtifactValidationError::SystematicCodeOrderMismatch {
                index,
                expected: expected.code,
                actual: entry.code.clone(),
            });
        }
        let Some(default_by_code) = default_systematic_by_code(&entry.code) else {
            return Err(AdsArtifactValidationError::UnknownSystematicCode {
                code: entry.code.clone(),
                systematic_id: entry.systematic_id.clone(),
            });
        };
        if entry.systematic_id != expected.systematic_id {
            return Err(AdsArtifactValidationError::SystematicIdMismatch {
                code: entry.code.clone(),
                expected: expected.systematic_id,
                actual: entry.systematic_id.clone(),
            });
        }
        let Some(default_by_id) = default_systematic_by_id(&entry.systematic_id) else {
            return Err(AdsArtifactValidationError::UnknownSystematicCode {
                code: entry.code.clone(),
                systematic_id: entry.systematic_id.clone(),
            });
        };
        if default_by_code.systematic_id != default_by_id.systematic_id {
            return Err(AdsArtifactValidationError::SystematicIdMismatch {
                code: entry.code.clone(),
                expected: default_by_code.systematic_id,
                actual: entry.systematic_id.clone(),
            });
        }
        if entry.uncertainty_layer != expected.uncertainty_layer {
            return Err(AdsArtifactValidationError::SystematicSemanticMismatch {
                code: entry.code.clone(),
                field: "uncertainty_layer",
            });
        }
        if entry.kind != expected.kind {
            return Err(AdsArtifactValidationError::SystematicSemanticMismatch {
                code: entry.code.clone(),
                field: "kind",
            });
        }
        if entry.source_class != expected.source_class {
            return Err(AdsArtifactValidationError::SystematicSemanticMismatch {
                code: entry.code.clone(),
                field: "source_class",
            });
        }
        if entry.combination_policy != expected.combination_policy {
            return Err(AdsArtifactValidationError::SystematicSemanticMismatch {
                code: entry.code.clone(),
                field: "combination_policy",
            });
        }
    }

    Ok(())
}

pub fn serialize_validated_ads_statistical_artifact(
    artifact: &AdsStatisticalArtifact,
) -> Result<serde_json::Value, AdsArtifactValidationError> {
    validate_ads_statistical_artifact(artifact)?;
    serde_json::to_value(artifact)
        .map_err(|error| AdsArtifactValidationError::JsonSerialization(error.to_string()))
}

pub fn validate_ads_statistical_artifact_value(
    value: &serde_json::Value,
) -> Result<AdsStatisticalArtifact, AdsArtifactValidationError> {
    let artifact = serde_json::from_value::<AdsStatisticalArtifact>(value.clone())
        .map_err(|error| AdsArtifactValidationError::JsonDeserialization(error.to_string()))?;
    validate_ads_statistical_artifact(&artifact)?;
    let canonical = serialize_validated_ads_statistical_artifact(&artifact)?;
    if canonical != *value {
        return Err(AdsArtifactValidationError::NonCanonicalJsonShape);
    }
    Ok(artifact)
}

fn validate_interval(
    field: &'static str,
    interval: &AdsUncertaintyInterval,
) -> Result<(), AdsArtifactValidationError> {
    if interval.lo > interval.hi {
        return Err(AdsArtifactValidationError::InvalidInterval {
            field,
            lo: interval.lo,
            hi: interval.hi,
        });
    }
    Ok(())
}

fn validate_non_negative_guardrail(
    field: &'static str,
    value: f64,
) -> Result<(), AdsArtifactValidationError> {
    if value < 0.0 {
        return Err(AdsArtifactValidationError::NegativeGuardrail { field, value });
    }
    Ok(())
}

pub fn ads_statistical_artifact_json_schema() -> serde_json::Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": ADS_STATISTICAL_ARTIFACT_SCHEMA_ID,
        "title": "NextStat Ads Statistical Artifact",
        "type": "object",
        "required": [
            "type",
            "version",
            "decision",
            "result",
            "diagnostics",
            "guardrails",
            "systematics_profile",
            "rigor",
            "metadata"
        ],
        "properties": {
            "type": { "type": "string", "const": ADS_STATISTICAL_ARTIFACT_TYPE },
            "version": { "type": "string", "const": ADS_STATISTICAL_ARTIFACT_VERSION },
            "decision": {
                "type": "string",
                "enum": ["continue", "stop_winner", "stop_no_difference", "stop_guardrail"]
            },
            "result": {
                "type": "object",
                "required": [
                    "effect_estimate",
                    "sampling_uncertainty",
                    "causal_uncertainty",
                    "total_uncertainty",
                    "claim_status",
                    "practical_significance"
                ],
                "properties": {
                    "effect_estimate": { "type": "number" },
                    "sampling_uncertainty": {
                        "type": "object",
                        "required": ["lo", "hi"],
                        "properties": {
                            "lo": { "type": "number" },
                            "hi": { "type": "number" }
                        },
                        "additionalProperties": false
                    },
                    "measurement_uncertainty": {
                        "type": ["object", "null"],
                        "required": ["lo", "hi"],
                        "properties": {
                            "lo": { "type": "number" },
                            "hi": { "type": "number" }
                        },
                        "additionalProperties": false
                    },
                    "causal_uncertainty": {
                        "type": "object",
                        "required": ["status", "diagnostics"],
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["separate", "combined", "not_available"]
                            },
                            "diagnostics": {
                                "type": "array",
                                "items": { "type": "string" }
                            }
                        },
                        "additionalProperties": false
                    },
                    "total_uncertainty": {
                        "type": "object",
                        "required": ["status"],
                        "properties": {
                            "status": {
                                "type": "string",
                                "enum": ["combined", "not_combinable", "not_available"]
                            },
                            "interval": {
                                "type": ["object", "null"],
                                "required": ["lo", "hi"],
                                "properties": {
                                    "lo": { "type": "number" },
                                    "hi": { "type": "number" }
                                },
                                "additionalProperties": false
                            }
                        },
                        "additionalProperties": false
                    },
                    "cls": {
                        "type": ["object", "null"],
                        "required": ["enabled", "claim_status"],
                        "properties": {
                            "enabled": { "type": "boolean" },
                            "value": { "type": ["number", "null"] },
                            "expected_sensitivity_sigma": { "type": ["number", "null"] },
                            "claim_status": {
                                "type": "string",
                                "enum": [
                                    "claimable_win",
                                    "positive_signal_but_not_claimable",
                                    "claimable_no_difference",
                                    "no_difference_but_not_claimable",
                                    "guardrail_blocked",
                                    "inconclusive"
                                ]
                            }
                        },
                        "additionalProperties": false
                    },
                    "claim_status": {
                        "type": "string",
                        "enum": [
                            "claimable_win",
                            "positive_signal_but_not_claimable",
                            "claimable_no_difference",
                            "no_difference_but_not_claimable",
                            "guardrail_blocked",
                            "inconclusive"
                        ]
                    },
                    "practical_significance": {
                        "type": "object",
                        "required": ["threshold_status"],
                        "properties": {
                            "threshold_status": {
                                "type": "string",
                                "enum": [
                                    "threshold_not_declared",
                                    "exceeds_win_threshold",
                                    "exceeds_loss_threshold",
                                    "inside_equivalence_band",
                                    "inconclusive_relative_to_threshold"
                                ]
                            },
                            "delta_win": { "type": ["number", "null"] },
                            "delta_loss": { "type": ["number", "null"] },
                            "delta_equiv": { "type": ["number", "null"] },
                            "tost_status": { "type": ["string", "null"] },
                            "decision_loss": {
                                "type": ["object", "null"],
                                "properties": {
                                    "scale": { "type": ["number", "null"] },
                                    "hold": { "type": ["number", "null"] },
                                    "revert": { "type": ["number", "null"] }
                                },
                                "additionalProperties": false
                            }
                        },
                        "additionalProperties": false
                    }
                },
                "additionalProperties": false
            },
            "diagnostics": {
                "type": "object",
                "required": ["fit_quality", "convergence", "warnings"],
                "properties": {
                    "fit_quality": { "type": "string", "enum": ["good", "warning", "failed"] },
                    "convergence": { "type": "boolean" },
                    "gof_pvalue": { "type": ["number", "null"] },
                    "dominant_systematic": { "type": ["string", "null"] },
                    "warnings": {
                        "type": "array",
                        "items": { "type": "string" }
                    }
                },
                "additionalProperties": false
            },
            "guardrails": {
                "type": "object",
                "required": [
                    "minimum_events",
                    "actual_events",
                    "power_at_mde",
                    "sequential_looks_used",
                    "sequential_looks_planned",
                    "alpha_spent",
                    "alpha_remaining"
                ],
                "properties": {
                    "minimum_events": { "type": "integer" },
                    "actual_events": {
                        "type": "object",
                        "required": ["control", "variant"],
                        "properties": {
                            "control": { "type": "integer" },
                            "variant": { "type": "integer" }
                        },
                        "additionalProperties": false
                    },
                    "power_at_mde": { "type": "number" },
                    "sequential_looks_used": { "type": "integer" },
                    "sequential_looks_planned": { "type": "integer" },
                    "alpha_spent": { "type": "number" },
                    "alpha_remaining": { "type": "number" }
                },
                "additionalProperties": false
            },
            "assumptions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["key", "value"],
                    "properties": {
                        "key": { "type": "string" },
                        "value": { "type": "string" }
                    },
                    "additionalProperties": false
                }
            },
            "systematics_profile": {
                "type": "object",
                "required": ["registry_version", "entries"],
                "properties": {
                    "registry_version": {
                        "type": "string",
                        "const": ADS_SYSTEMATICS_REGISTRY_VERSION
                    },
                    "entries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "code",
                                "systematic_id",
                                "enabled",
                                "uncertainty_layer",
                                "kind",
                                "source_class",
                                "combination_policy",
                                "nominal",
                                "sigma",
                                "constraint"
                            ],
                            "properties": {
                                "code": { "type": "string" },
                                "systematic_id": { "type": "string" },
                                "enabled": { "type": "boolean" },
                                "uncertainty_layer": {
                                    "type": "string",
                                    "enum": ["measurement", "causal"]
                                },
                                "kind": {
                                    "type": "string",
                                    "enum": ["norm_sys", "histo_sys"]
                                },
                                "source_class": {
                                    "type": "string",
                                    "enum": [
                                        "data_estimated",
                                        "hybrid_api_plus_benchmark",
                                        "hybrid_api_plus_prior",
                                        "policy_default_or_vendor_calibrated",
                                        "policy_default_or_incrementality_calibrated",
                                        "internal_nextstat_diagnostic"
                                    ]
                                },
                                "combination_policy": {
                                    "type": "string",
                                    "enum": [
                                        "profile_likelihood",
                                        "scenario_only",
                                        "sensitivity_first_then_profile",
                                        "sensitivity_only"
                                    ]
                                },
                                "nominal": { "type": "string" },
                                "sigma": { "type": "string" },
                                "constraint": { "type": "string" }
                            },
                            "additionalProperties": false
                        }
                    }
                },
                "additionalProperties": false
            },
            "rigor": {
                "type": "object",
                "required": [
                    "passed",
                    "artifacts_complete",
                    "claim_valid",
                    "leakage_clean",
                    "claim_violations",
                    "leakage_indicators"
                ],
                "properties": {
                    "passed": { "type": "boolean" },
                    "artifacts_complete": { "type": "boolean" },
                    "claim_valid": { "type": "boolean" },
                    "leakage_clean": { "type": "boolean" },
                    "claim_violations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["rule", "severity", "message"],
                            "properties": {
                                "rule": { "type": "string" },
                                "severity": {
                                    "type": "string",
                                    "enum": ["info", "warning", "error"]
                                },
                                "message": { "type": "string" }
                            },
                            "additionalProperties": false
                        }
                    },
                    "leakage_indicators": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["name", "detected", "severity", "message"],
                            "properties": {
                                "name": { "type": "string" },
                                "detected": { "type": "boolean" },
                                "severity": {
                                    "type": "string",
                                    "enum": ["info", "warning", "error"]
                                },
                                "message": { "type": "string" }
                            },
                            "additionalProperties": false
                        }
                    }
                },
                "additionalProperties": false
            },
            "metadata": {
                "type": "object",
                "required": ["engine_version", "model_version", "created_at"],
                "properties": {
                    "data_hash": { "type": ["string", "null"] },
                    "engine_version": { "type": "string" },
                    "model_version": { "type": "string" },
                    "created_at": { "type": "string" }
                },
                "additionalProperties": false
            }
        },
        "additionalProperties": false
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_artifact() -> AdsStatisticalArtifact {
        let result = AdsArtifactResult {
            effect_estimate: 0.12,
            sampling_uncertainty: AdsUncertaintyInterval { lo: -0.04, hi: 0.04 },
            measurement_uncertainty: Some(AdsUncertaintyInterval { lo: -0.08, hi: 0.08 }),
            causal_uncertainty: AdsCausalUncertainty {
                status: AdsCausalUncertaintyStatus::Separate,
                diagnostics: vec![
                    "no_pretrend_violation".into(),
                    "no_major_overlap_warning".into(),
                ],
            },
            total_uncertainty: AdsTotalUncertainty {
                status: AdsTotalUncertaintyStatus::NotCombinable,
                interval: None,
            },
            cls: Some(AdsClsGuardrail {
                enabled: true,
                value: Some(0.023),
                expected_sensitivity_sigma: Some(3.2),
                claim_status: AdsClaimStatus::ClaimableWin,
            }),
            claim_status: AdsClaimStatus::ClaimableWin,
            practical_significance: AdsPracticalSignificance {
                threshold_status: AdsPracticalThresholdStatus::ExceedsWinThreshold,
                delta_win: Some(0.05),
                delta_loss: None,
                delta_equiv: Some(0.02),
                tost_status: None,
                decision_loss: Some(AdsDecisionLossView {
                    scale: Some(1200.0),
                    hold: Some(4200.0),
                    revert: Some(8500.0),
                }),
            },
        };

        let diagnostics = AdsArtifactDiagnostics {
            fit_quality: AdsFitQuality::Good,
            convergence: true,
            gof_pvalue: Some(0.28),
            dominant_systematic: Some("conversion_lag_curve".into()),
            warnings: vec![],
        };

        let guardrails = AdsArtifactGuardrails {
            minimum_events: 100,
            actual_events: AdsEventCounts { control: 134, variant: 157 },
            power_at_mde: 0.87,
            sequential_looks_used: 2,
            sequential_looks_planned: 5,
            alpha_spent: 0.012,
            alpha_remaining: 0.038,
        };

        let metadata = AdsArtifactMetadata {
            data_hash: Some("sha256:abc123".into()),
            engine_version: "ns-inference-0.9.9".into(),
            model_version: "ads-histfactory-v1".into(),
            created_at: "2026-03-09T11:00:00Z".into(),
        };

        let mut artifact = AdsStatisticalArtifact::new(
            AdsArtifactDecision::StopWinner,
            result,
            diagnostics,
            guardrails,
            metadata,
        );
        artifact.assumptions = vec![
            AdsArtifactAssumption { key: "attribution_window".into(), value: "7d_click".into() },
            AdsArtifactAssumption {
                key: "fraud_policy".into(),
                value: "iab_conservative_10pct".into(),
            },
        ];
        artifact
    }

    #[test]
    fn systematics_profile_defaults_track_registry_order() {
        let profile = AdsSystematicsProfile::from_defaults();
        assert_eq!(profile.registry_version, ADS_SYSTEMATICS_REGISTRY_VERSION);
        assert_eq!(profile.entries.len(), 6);
        assert_eq!(profile.entries[0].code, "S1");
        assert_eq!(profile.entries[0].systematic_id, "conversion_lag_curve");
        assert_eq!(profile.entries[5].code, "S6");
        assert_eq!(profile.entries[5].systematic_id, "experiment_interference");
    }

    #[test]
    fn statistical_artifact_roundtrip_preserves_claimability_and_metadata() {
        let artifact = sample_artifact();
        let json = serde_json::to_string_pretty(&artifact).expect("serialize");
        let roundtrip: AdsStatisticalArtifact = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(roundtrip.artifact_type, ADS_STATISTICAL_ARTIFACT_TYPE);
        assert_eq!(roundtrip.version, ADS_STATISTICAL_ARTIFACT_VERSION);
        assert_eq!(roundtrip.result.claim_status, AdsClaimStatus::ClaimableWin);
        assert_eq!(
            roundtrip.result.practical_significance.threshold_status,
            AdsPracticalThresholdStatus::ExceedsWinThreshold
        );
        assert_eq!(roundtrip.metadata.data_hash.as_deref(), Some("sha256:abc123"));
        assert_eq!(roundtrip.systematics_profile.entries.len(), 6);
        assert_eq!(roundtrip.assumptions.len(), 2);
    }

    #[test]
    fn systematics_profile_entry_carries_registry_semantics() {
        let profile = AdsSystematicsProfile::from_defaults();
        let fraud = profile
            .entries
            .iter()
            .find(|entry| entry.systematic_id == "residual_fraud_rate")
            .expect("fraud entry must exist");

        assert_eq!(fraud.source_class, AdsSystematicSourceClass::PolicyDefaultOrVendorCalibrated);
        assert_eq!(fraud.combination_policy, AdsCombinationPolicy::SensitivityFirstThenProfile);
        assert_eq!(fraud.nominal, "0.10");
        assert_eq!(fraud.sigma, "0.05");
    }

    #[test]
    fn artifact_contract_reference_matches_shared_constants() {
        let contract = AdsArtifactContractReference::shared();
        assert_eq!(contract.artifact_type, ADS_STATISTICAL_ARTIFACT_TYPE);
        assert_eq!(contract.version, ADS_STATISTICAL_ARTIFACT_VERSION);
        assert_eq!(contract.systematics_registry_version, ADS_SYSTEMATICS_REGISTRY_VERSION);
    }

    #[test]
    fn artifact_reference_summary_can_be_built_from_statistical_artifact() {
        let artifact = sample_artifact();
        let summary = AdsArtifactReferenceSummary::from(&artifact);

        assert_eq!(summary.contract.artifact_type, ADS_STATISTICAL_ARTIFACT_TYPE);
        assert_eq!(summary.contract.version, ADS_STATISTICAL_ARTIFACT_VERSION);
        assert_eq!(summary.claim_status, Some(AdsClaimStatus::ClaimableWin));
        assert_eq!(summary.decision, Some(AdsArtifactDecision::StopWinner));
        assert_eq!(summary.created_at.as_deref(), Some("2026-03-09T11:00:00Z"));
    }

    #[test]
    fn validate_artifact_value_rejects_non_canonical_extra_fields() {
        let mut artifact_value = serde_json::to_value(sample_artifact()).expect("serialize");
        artifact_value["unexpected"] = json!(true);

        let error = validate_ads_statistical_artifact_value(&artifact_value).expect_err("reject");
        assert_eq!(error, AdsArtifactValidationError::NonCanonicalJsonShape);
    }

    #[test]
    fn validate_artifact_rejects_total_interval_without_combined_status() {
        let mut artifact = sample_artifact();
        artifact.result.total_uncertainty.status = AdsTotalUncertaintyStatus::NotCombinable;
        artifact.result.total_uncertainty.interval =
            Some(AdsUncertaintyInterval { lo: -0.1, hi: 0.2 });

        let error = validate_ads_statistical_artifact(&artifact).expect_err("reject");
        assert_eq!(
            error,
            AdsArtifactValidationError::UnexpectedTotalInterval { status: "not_combinable" }
        );
    }

    #[test]
    fn validate_artifact_rejects_registry_semantic_drift() {
        let mut artifact = sample_artifact();
        artifact.systematics_profile.entries[0].systematic_id = "wrong_id".to_string();

        let error = validate_ads_statistical_artifact(&artifact).expect_err("reject");
        assert_eq!(
            error,
            AdsArtifactValidationError::SystematicIdMismatch {
                code: "S1".to_string(),
                expected: "conversion_lag_curve",
                actual: "wrong_id".to_string(),
            }
        );
    }

    #[test]
    fn json_schema_contract_tracks_shared_constants() {
        let schema = ads_statistical_artifact_json_schema();
        assert_eq!(schema["$id"], ADS_STATISTICAL_ARTIFACT_SCHEMA_ID);
        assert_eq!(schema["properties"]["type"]["const"], ADS_STATISTICAL_ARTIFACT_TYPE);
        assert_eq!(schema["properties"]["version"]["const"], ADS_STATISTICAL_ARTIFACT_VERSION);
        assert_eq!(
            schema["properties"]["systematics_profile"]["properties"]["registry_version"]["const"],
            ADS_SYSTEMATICS_REGISTRY_VERSION
        );
    }
}
