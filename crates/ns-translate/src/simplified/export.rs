use super::factorize::factorize_covariance_matrix;
use super::schema::{
    SIMPLIFIED_LIKELIHOOD_BASIS_METHOD_EIGEN,
    SIMPLIFIED_LIKELIHOOD_CONSTRAINT_COVARIANCE_SOURCE_ALIGNED_FIT_COVARIANCE,
    SIMPLIFIED_LIKELIHOOD_CONSTRAINT_COVARIANCE_SOURCE_SOURCE_MODEL_CONSTRAINTS,
    SIMPLIFIED_LIKELIHOOD_JACOBIAN_METHOD_FINITE_DIFFERENCE, SIMPLIFIED_LIKELIHOOD_SCHEMA_V0,
    SIMPLIFIED_LIKELIHOOD_SOURCE_FORMAT_DERIVED_FROM_WORKSPACE,
    SIMPLIFIED_LIKELIHOOD_SOURCE_WORKSPACE_FORMAT_HS3,
    SIMPLIFIED_LIKELIHOOD_SOURCE_WORKSPACE_FORMAT_PYHF, SimplifiedDiagnostics,
    SimplifiedLikelihoodBin, SimplifiedLikelihoodDerivation, SimplifiedLikelihoodMetadata,
    SimplifiedLikelihoodPoi, SimplifiedLikelihoodWorkspace, SimplifiedUncertaintyModel,
};
use nalgebra::DMatrix;
use ns_core::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

pub const SIMPLIFIED_LIKELIHOOD_DERIVE_SCHEMA_V0: &str = "nextstat_simplified_likelihood_derive_v0";
pub const SIMPLIFIED_LIKELIHOOD_EXPORT_REPORT_SCHEMA_V0: &str =
    "nextstat_simplified_likelihood_export_report_v0";
pub const SIMPLIFIED_LIKELIHOOD_EXPORT_REPORT_SUPPORT_CLASS_RESEARCH_GRADE: &str = "research-grade";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedLikelihoodDeriveConfig {
    pub schema_version: String,
    pub source_workspace: SimplifiedLikelihoodDeriveSourceWorkspace,
    pub fit_result: SimplifiedLikelihoodDeriveFitResult,
    pub selection: SimplifiedLikelihoodDeriveSelection,
    pub reduction: SimplifiedLikelihoodDeriveReduction,
    pub jacobian: SimplifiedLikelihoodDeriveJacobian,
    pub fidelity_smoke: SimplifiedLikelihoodDeriveFidelitySmoke,
    pub output_contract: SimplifiedLikelihoodDeriveOutputContract,
    #[serde(default)]
    pub unsupported_semantics: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedLikelihoodDeriveSourceWorkspace {
    pub format: String,
    #[serde(default)]
    pub schema_version: Option<String>,
    pub poi_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedLikelihoodDeriveFitResult {
    pub schema_version: String,
    pub background_state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedLikelihoodDeriveSelection {
    pub channels: Vec<String>,
    #[serde(default)]
    pub bins: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedLikelihoodDeriveReduction {
    pub output_uncertainty_model: String,
    pub basis_method: String,
    pub explained_variance_target: f64,
    pub constraint_covariance_source: String,
    #[serde(default)]
    pub max_components: Option<usize>,
    #[serde(default)]
    pub split_stat_covariance: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedLikelihoodDeriveJacobian {
    pub method: String,
    pub relative_step: f64,
    pub absolute_step_floor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedLikelihoodDeriveFidelitySmoke {
    pub random_draws: usize,
    pub qmu_test_mu: f64,
    pub upper_limit_cl: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedLikelihoodDeriveOutputContract {
    pub schema_version: String,
    pub require_factorization_diagnostics: bool,
    pub require_fidelity_diagnostics: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedLikelihoodExportReport {
    pub schema_version: String,
    pub status: String,
    pub support_class: String,
    pub source: SimplifiedLikelihoodExportReportSource,
    pub metadata: SimplifiedLikelihoodMetadata,
    pub selection: SimplifiedLikelihoodDeriveSelection,
    pub reduction: SimplifiedLikelihoodDeriveReduction,
    pub jacobian: SimplifiedLikelihoodDeriveJacobian,
    pub fidelity_smoke: SimplifiedLikelihoodDeriveFidelitySmoke,
    pub output: SimplifiedLikelihoodExportReportOutput,
    pub diagnostics: SimplifiedDiagnostics,
    #[serde(default)]
    pub unsupported_semantics: Vec<String>,
    pub explicit_boundaries: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedLikelihoodExportReportSource {
    pub workspace_format: String,
    #[serde(default)]
    pub workspace_schema_version: Option<String>,
    pub fit_result_schema_version: String,
    pub poi_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedLikelihoodExportReportOutput {
    pub schema_version: String,
    pub uncertainty_model_kind: String,
    pub bins_count: usize,
    pub full_nuisance_count: usize,
    pub reduced_nuisance_count: usize,
    pub reduction_ratio: f64,
    pub json_bytes: usize,
    pub json_sha256: String,
}

#[derive(Debug, Clone)]
pub struct SimplifiedLikelihoodDeriveMetadata {
    pub experiment: String,
    pub analysis_id: String,
    pub reference: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SimplifiedLikelihoodAlignedFitResult {
    pub schema_version: Option<String>,
    pub parameters: Vec<f64>,
    pub covariance: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SimplifiedLikelihoodDerivedCore {
    pub workspace: SimplifiedLikelihoodWorkspace,
    pub full_nuisance_count: usize,
    pub total_background_covariance: Vec<Vec<f64>>,
    pub retained_background_covariance: Vec<Vec<f64>>,
}

pub fn build_simplified_likelihood_export_report(
    config: &SimplifiedLikelihoodDeriveConfig,
    metadata: &SimplifiedLikelihoodDeriveMetadata,
    derived: &SimplifiedLikelihoodDerivedCore,
) -> Result<SimplifiedLikelihoodExportReport> {
    let diagnostics = derived.workspace.diagnostics.clone().ok_or_else(|| {
        Error::Validation(
            "derived simplified-likelihood workspace must include diagnostics before export report generation"
                .to_string(),
        )
    })?;
    if diagnostics.factorization.is_none() {
        return Err(Error::Validation(
            "derived simplified-likelihood workspace must include factorization diagnostics before export report generation"
                .to_string(),
        ));
    }
    if diagnostics.fidelity.is_none() {
        return Err(Error::Validation(
            "derived simplified-likelihood workspace must include fidelity diagnostics before export report generation"
                .to_string(),
        ));
    }

    let derivation = derived.workspace.derivation.as_ref().ok_or_else(|| {
        Error::Validation(
            "derived simplified-likelihood workspace must include derivation provenance before export report generation"
                .to_string(),
        )
    })?;
    let canonical_workspace_value =
        canonicalize_json_value(&serde_json::to_value(&derived.workspace).map_err(|err| {
            Error::Validation(format!("failed to convert derived workspace into JSON value: {err}"))
        })?);
    let serialized_workspace =
        serde_json::to_vec_pretty(&canonical_workspace_value).map_err(|err| {
            Error::Validation(format!("failed to serialize derived workspace: {err}"))
        })?;
    let reduced_nuisance_count = match &derived.workspace.uncertainty_model {
        SimplifiedUncertaintyModel::Basis { components } => components.len(),
        SimplifiedUncertaintyModel::Covariance { total_covariance, .. } => total_covariance.len(),
    };
    let uncertainty_model_kind = match &derived.workspace.uncertainty_model {
        SimplifiedUncertaintyModel::Basis { .. } => "basis",
        SimplifiedUncertaintyModel::Covariance { .. } => "covariance",
    };
    let reduction_ratio = if derived.full_nuisance_count == 0 {
        0.0
    } else {
        reduced_nuisance_count as f64 / derived.full_nuisance_count as f64
    };

    let mut explicit_boundaries = vec![
        "source_workspace.format=pyhf only".to_string(),
        "partial per-channel bin selection unsupported".to_string(),
        "derived_from_workspace preserves reduced nuisance coordinates, not source-level nuisance identities"
            .to_string(),
    ];
    if config.reduction.constraint_covariance_source
        == SIMPLIFIED_LIKELIHOOD_CONSTRAINT_COVARIANCE_SOURCE_SOURCE_MODEL_CONSTRAINTS
    {
        explicit_boundaries.push(
            "constraint_covariance_source=source_model_constraints currently supports Gaussian-constrained nuisances only"
                .to_string(),
        );
    }

    Ok(SimplifiedLikelihoodExportReport {
        schema_version: SIMPLIFIED_LIKELIHOOD_EXPORT_REPORT_SCHEMA_V0.to_string(),
        status: "ok".to_string(),
        support_class: SIMPLIFIED_LIKELIHOOD_EXPORT_REPORT_SUPPORT_CLASS_RESEARCH_GRADE.to_string(),
        source: SimplifiedLikelihoodExportReportSource {
            workspace_format: config.source_workspace.format.clone(),
            workspace_schema_version: derivation.source_workspace_schema_version.clone(),
            fit_result_schema_version: derivation
                .fit_result_schema_version
                .clone()
                .unwrap_or_else(|| config.fit_result.schema_version.clone()),
            poi_name: config.source_workspace.poi_name.clone(),
        },
        metadata: SimplifiedLikelihoodMetadata {
            experiment: metadata.experiment.clone(),
            analysis_id: metadata.analysis_id.clone(),
            source_format: derived.workspace.metadata.source_format.clone(),
            reference: metadata.reference.clone(),
            description: metadata.description.clone(),
        },
        selection: config.selection.clone(),
        reduction: config.reduction.clone(),
        jacobian: config.jacobian.clone(),
        fidelity_smoke: config.fidelity_smoke.clone(),
        output: SimplifiedLikelihoodExportReportOutput {
            schema_version: derived.workspace.schema_version.clone(),
            uncertainty_model_kind: uncertainty_model_kind.to_string(),
            bins_count: derived.workspace.bins.len(),
            full_nuisance_count: derived.full_nuisance_count,
            reduced_nuisance_count,
            reduction_ratio,
            json_bytes: serialized_workspace.len(),
            json_sha256: format!("{:x}", Sha256::digest(&serialized_workspace)),
        },
        diagnostics,
        unsupported_semantics: config.unsupported_semantics.clone(),
        explicit_boundaries,
    })
}

fn canonicalize_json_value(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<String> = map.keys().cloned().collect();
            keys.sort();
            let mut out = serde_json::Map::new();
            for key in keys {
                if let Some(entry) = map.get(&key) {
                    out.insert(key, canonicalize_json_value(entry));
                }
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(entries) => {
            serde_json::Value::Array(entries.iter().map(canonicalize_json_value).collect())
        }
        _ => value.clone(),
    }
}

pub fn validate_simplified_likelihood_derive_config(
    config: &SimplifiedLikelihoodDeriveConfig,
) -> Result<()> {
    if config.schema_version != SIMPLIFIED_LIKELIHOOD_DERIVE_SCHEMA_V0 {
        return Err(Error::Validation(format!(
            "expected schema_version '{}' but got '{}'",
            SIMPLIFIED_LIKELIHOOD_DERIVE_SCHEMA_V0, config.schema_version
        )));
    }

    match config.source_workspace.format.as_str() {
        SIMPLIFIED_LIKELIHOOD_SOURCE_WORKSPACE_FORMAT_PYHF
        | SIMPLIFIED_LIKELIHOOD_SOURCE_WORKSPACE_FORMAT_HS3 => {}
        other => {
            return Err(Error::Validation(format!(
                "source_workspace.format must be 'pyhf' or 'hs3', got '{other}'"
            )));
        }
    }
    validate_nonempty("source_workspace.poi_name", &config.source_workspace.poi_name)?;
    validate_optional_nonempty(
        "source_workspace.schema_version",
        config.source_workspace.schema_version.as_deref(),
    )?;

    validate_nonempty("fit_result.schema_version", &config.fit_result.schema_version)?;
    if config.fit_result.background_state != "postfit_background" {
        return Err(Error::Validation(format!(
            "fit_result.background_state must be 'postfit_background', got '{}'",
            config.fit_result.background_state
        )));
    }

    validate_nonempty_vec("selection.channels", &config.selection.channels)?;
    validate_optional_nonempty_vec("selection.bins", config.selection.bins.as_deref())?;

    if config.reduction.output_uncertainty_model != "basis" {
        return Err(Error::Validation(format!(
            "reduction.output_uncertainty_model must be 'basis', got '{}'",
            config.reduction.output_uncertainty_model
        )));
    }
    if config.reduction.basis_method != SIMPLIFIED_LIKELIHOOD_BASIS_METHOD_EIGEN {
        return Err(Error::Validation(format!(
            "reduction.basis_method must be '{}', got '{}'",
            SIMPLIFIED_LIKELIHOOD_BASIS_METHOD_EIGEN, config.reduction.basis_method
        )));
    }
    if !config.reduction.explained_variance_target.is_finite()
        || config.reduction.explained_variance_target <= 0.0
        || config.reduction.explained_variance_target > 1.0
    {
        return Err(Error::Validation(format!(
            "reduction.explained_variance_target must be in (0, 1], got {}",
            config.reduction.explained_variance_target
        )));
    }
    if matches!(config.reduction.max_components, Some(0)) {
        return Err(Error::Validation(
            "reduction.max_components must be >= 1 when provided".to_string(),
        ));
    }
    match config.reduction.constraint_covariance_source.as_str() {
        SIMPLIFIED_LIKELIHOOD_CONSTRAINT_COVARIANCE_SOURCE_SOURCE_MODEL_CONSTRAINTS
        | SIMPLIFIED_LIKELIHOOD_CONSTRAINT_COVARIANCE_SOURCE_ALIGNED_FIT_COVARIANCE => {}
        other => {
            return Err(Error::Validation(format!(
                "reduction.constraint_covariance_source must be '{}' or '{}', got '{other}'",
                SIMPLIFIED_LIKELIHOOD_CONSTRAINT_COVARIANCE_SOURCE_SOURCE_MODEL_CONSTRAINTS,
                SIMPLIFIED_LIKELIHOOD_CONSTRAINT_COVARIANCE_SOURCE_ALIGNED_FIT_COVARIANCE,
            )));
        }
    }

    if config.jacobian.method != SIMPLIFIED_LIKELIHOOD_JACOBIAN_METHOD_FINITE_DIFFERENCE {
        return Err(Error::Validation(format!(
            "jacobian.method must be '{}', got '{}'",
            SIMPLIFIED_LIKELIHOOD_JACOBIAN_METHOD_FINITE_DIFFERENCE, config.jacobian.method
        )));
    }
    if !config.jacobian.relative_step.is_finite() || config.jacobian.relative_step <= 0.0 {
        return Err(Error::Validation(format!(
            "jacobian.relative_step must be finite and > 0, got {}",
            config.jacobian.relative_step
        )));
    }
    if !config.jacobian.absolute_step_floor.is_finite()
        || config.jacobian.absolute_step_floor <= 0.0
    {
        return Err(Error::Validation(format!(
            "jacobian.absolute_step_floor must be finite and > 0, got {}",
            config.jacobian.absolute_step_floor
        )));
    }

    if config.fidelity_smoke.random_draws == 0 {
        return Err(Error::Validation("fidelity_smoke.random_draws must be >= 1".to_string()));
    }
    if !config.fidelity_smoke.qmu_test_mu.is_finite() {
        return Err(Error::Validation("fidelity_smoke.qmu_test_mu must be finite".to_string()));
    }
    if !config.fidelity_smoke.upper_limit_cl.is_finite()
        || config.fidelity_smoke.upper_limit_cl <= 0.0
        || config.fidelity_smoke.upper_limit_cl >= 1.0
    {
        return Err(Error::Validation(format!(
            "fidelity_smoke.upper_limit_cl must be in (0, 1), got {}",
            config.fidelity_smoke.upper_limit_cl
        )));
    }

    if config.output_contract.schema_version != SIMPLIFIED_LIKELIHOOD_SCHEMA_V0 {
        return Err(Error::Validation(format!(
            "output_contract.schema_version must be '{}', got '{}'",
            SIMPLIFIED_LIKELIHOOD_SCHEMA_V0, config.output_contract.schema_version
        )));
    }
    if !config.output_contract.require_factorization_diagnostics {
        return Err(Error::Validation(
            "output_contract.require_factorization_diagnostics must be true".to_string(),
        ));
    }
    if !config.output_contract.require_fidelity_diagnostics {
        return Err(Error::Validation(
            "output_contract.require_fidelity_diagnostics must be true".to_string(),
        ));
    }

    for (idx, entry) in config.unsupported_semantics.iter().enumerate() {
        validate_nonempty(&format!("unsupported_semantics[{idx}]"), entry)?;
    }

    Ok(())
}

pub fn derive_simplified_likelihood_core(
    workspace: &crate::pyhf::Workspace,
    model: &crate::pyhf::HistFactoryModel,
    fit_result: &SimplifiedLikelihoodAlignedFitResult,
    config: &SimplifiedLikelihoodDeriveConfig,
    metadata: &SimplifiedLikelihoodDeriveMetadata,
) -> Result<SimplifiedLikelihoodDerivedCore> {
    validate_simplified_likelihood_derive_config(config)?;

    let poi_idx = model
        .poi_index()
        .ok_or_else(|| Error::Validation("source workspace must define a POI".to_string()))?;
    let poi = model.parameters().get(poi_idx).ok_or_else(|| {
        Error::Validation(format!("POI index {poi_idx} is out of range for model parameters"))
    })?;
    if poi.name != config.source_workspace.poi_name {
        return Err(Error::Validation(format!(
            "source_workspace.poi_name mismatch: config='{}' model='{}'",
            config.source_workspace.poi_name, poi.name
        )));
    }

    if let Some(schema_version) = fit_result.schema_version.as_deref()
        && schema_version != config.fit_result.schema_version
    {
        return Err(Error::Validation(format!(
            "fit result schema mismatch: config='{}' fit='{}'",
            config.fit_result.schema_version, schema_version
        )));
    }

    let n_params = model.parameters().len();
    if fit_result.parameters.len() != n_params {
        return Err(Error::Validation(format!(
            "aligned fit parameter length mismatch: got {} expected {}",
            fit_result.parameters.len(),
            n_params
        )));
    }
    if fit_result.covariance.len() != n_params * n_params {
        return Err(Error::Validation(format!(
            "aligned fit covariance length mismatch: got {} expected {}",
            fit_result.covariance.len(),
            n_params * n_params
        )));
    }

    let selection =
        resolve_selection(model, &config.selection.channels, config.selection.bins.as_deref())?;

    let mut background_params = fit_result.parameters.clone();
    background_params[poi_idx] = 0.0;
    let background_model = model.with_fixed_param(poi_idx, 0.0);
    let background_all = background_model.expected_data(&background_params)?;
    let background_nominal = select_flat_entries(&background_all, &selection.flat_indices)?;

    let observed_all = flatten_observed_main(model);
    let observed = select_flat_entries(&observed_all, &selection.flat_indices)?;

    let mut signal_params = background_params.clone();
    signal_params[poi_idx] = 1.0;
    let signal_all = model.expected_data(&signal_params)?;
    let signal_total = select_flat_entries(&signal_all, &selection.flat_indices)?;
    let signal_nominal = signal_total
        .iter()
        .zip(background_nominal.iter())
        .map(|(signal_plus_background, background)| (signal_plus_background - background).max(0.0))
        .collect::<Vec<_>>();

    let nuisance_indices: Vec<usize> = (0..n_params).filter(|idx| *idx != poi_idx).collect();
    let nuisance_covariance = match config.reduction.constraint_covariance_source.as_str() {
        SIMPLIFIED_LIKELIHOOD_CONSTRAINT_COVARIANCE_SOURCE_SOURCE_MODEL_CONSTRAINTS => {
            source_constraint_covariance(model, &nuisance_indices)?
        }
        SIMPLIFIED_LIKELIHOOD_CONSTRAINT_COVARIANCE_SOURCE_ALIGNED_FIT_COVARIANCE => {
            select_covariance_submatrix(&fit_result.covariance, n_params, &nuisance_indices)?
        }
        other => {
            return Err(Error::Validation(format!(
                "unsupported reduction.constraint_covariance_source '{other}'"
            )));
        }
    };
    let jacobian = finite_difference_jacobian(
        &background_model,
        &background_params,
        &nuisance_indices,
        &selection.flat_indices,
        config.jacobian.relative_step,
        config.jacobian.absolute_step_floor,
    )?;
    let total_background_covariance =
        project_background_covariance(&jacobian, &nuisance_covariance)?;

    let stat_background_covariance = if config.reduction.split_stat_covariance {
        let stat_parameter_positions = stat_nuisance_positions(model, &nuisance_indices);
        if stat_parameter_positions.is_empty() {
            None
        } else {
            let stat_jacobian = select_matrix_columns(&jacobian, &stat_parameter_positions);
            let stat_covariance =
                select_nested_submatrix(&nuisance_covariance, &stat_parameter_positions)?;
            Some(project_background_covariance(&stat_jacobian, &stat_covariance)?)
        }
    } else {
        None
    };

    let factorized = factorize_covariance_matrix(
        &background_nominal,
        &total_background_covariance,
        stat_background_covariance.as_deref(),
        config.reduction.explained_variance_target,
        config.reduction.max_components,
        "sl_np_",
    )?;

    let source_workspace_schema_version =
        config.source_workspace.schema_version.clone().or_else(|| {
            workspace
                .version
                .as_ref()
                .map(|version| format!("pyhf_workspace_v{}", normalize_pyhf_version(version)))
        });

    let workspace = SimplifiedLikelihoodWorkspace {
        schema_version: SIMPLIFIED_LIKELIHOOD_SCHEMA_V0.to_string(),
        metadata: SimplifiedLikelihoodMetadata {
            experiment: metadata.experiment.clone(),
            analysis_id: metadata.analysis_id.clone(),
            source_format: SIMPLIFIED_LIKELIHOOD_SOURCE_FORMAT_DERIVED_FROM_WORKSPACE.to_string(),
            reference: metadata.reference.clone(),
            description: metadata.description.clone(),
        },
        poi: SimplifiedLikelihoodPoi {
            name: poi.name.clone(),
            init: 1.0,
            bounds: [poi.bounds.0, poi.bounds.1],
        },
        bins: selection.bins,
        observed,
        background_nominal,
        signal_nominal: Some(signal_nominal),
        uncertainty_model: SimplifiedUncertaintyModel::Basis { components: factorized.components },
        derivation: Some(SimplifiedLikelihoodDerivation {
            source_workspace_format: config.source_workspace.format.clone(),
            source_workspace_schema_version,
            fit_result_schema_version: Some(
                fit_result
                    .schema_version
                    .clone()
                    .unwrap_or_else(|| config.fit_result.schema_version.clone()),
            ),
            selected_channels: config.selection.channels.clone(),
            selected_bins: config.selection.bins.clone(),
            basis_method: config.reduction.basis_method.clone(),
            explained_variance_target: config.reduction.explained_variance_target,
            constraint_covariance_source: config.reduction.constraint_covariance_source.clone(),
            jacobian_method: config.jacobian.method.clone(),
            split_stat_covariance: config.reduction.split_stat_covariance,
        }),
        diagnostics: Some(SimplifiedDiagnostics {
            factorization: Some(factorized.diagnostics),
            fidelity: None,
        }),
    };

    Ok(SimplifiedLikelihoodDerivedCore {
        workspace,
        full_nuisance_count: nuisance_indices.len(),
        total_background_covariance,
        retained_background_covariance: factorized.retained_covariance,
    })
}

fn validate_nonempty(label: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(Error::Validation(format!("{label} must not be empty")));
    }
    Ok(())
}

fn validate_optional_nonempty(label: &str, value: Option<&str>) -> Result<()> {
    if let Some(value) = value {
        validate_nonempty(label, value)?;
    }
    Ok(())
}

fn validate_nonempty_vec(label: &str, values: &[String]) -> Result<()> {
    if values.is_empty() {
        return Err(Error::Validation(format!("{label} must contain at least one entry")));
    }
    for (idx, value) in values.iter().enumerate() {
        validate_nonempty(&format!("{label}[{idx}]"), value)?;
    }
    Ok(())
}

fn validate_optional_nonempty_vec(label: &str, values: Option<&[String]>) -> Result<()> {
    if let Some(values) = values {
        validate_nonempty_vec(label, values)?;
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct ResolvedSelection {
    bins: Vec<SimplifiedLikelihoodBin>,
    flat_indices: Vec<usize>,
}

fn resolve_selection(
    model: &crate::pyhf::HistFactoryModel,
    channels: &[String],
    bins: Option<&[String]>,
) -> Result<ResolvedSelection> {
    let mut channel_seen = HashSet::new();
    let mut requested_bin_map: HashMap<String, Vec<usize>> = HashMap::new();
    let mut requested_bin_seen = HashSet::new();

    if let Some(bins) = bins {
        for entry in bins {
            let (channel_name, bin_idx) = parse_bin_selector(entry)?;
            if !requested_bin_seen.insert((channel_name.clone(), bin_idx)) {
                return Err(Error::Validation(format!(
                    "selection.bins contains duplicate entry '{entry}'"
                )));
            }
            requested_bin_map.entry(channel_name).or_default().push(bin_idx);
        }
    }

    let mut selected_bins = Vec::new();
    let mut flat_indices = Vec::new();

    for channel_name in channels {
        if !channel_seen.insert(channel_name.as_str()) {
            return Err(Error::Validation(format!(
                "selection.channels contains duplicate channel '{channel_name}'"
            )));
        }
        let channel_idx = model.channel_index(channel_name).ok_or_else(|| {
            Error::Validation(format!(
                "selection.channels references unknown channel '{channel_name}'"
            ))
        })?;
        let n_bins = model.channel_bin_count(channel_idx)?;
        let channel_offset = model.channel_bin_offset(channel_idx)?;

        let requested_indices = requested_bin_map.remove(channel_name);
        let mut kept_indices = if let Some(mut indices) = requested_indices {
            indices.sort_unstable();
            indices.dedup();
            for &bin_idx in &indices {
                if bin_idx >= n_bins {
                    return Err(Error::Validation(format!(
                        "selection.bins references out-of-range bin '{channel_name}/bin{bin_idx}'"
                    )));
                }
            }
            if indices.len() != n_bins {
                return Err(Error::Validation(format!(
                    "partial-bin selection is not supported yet for channel '{channel_name}': got {} of {} bins",
                    indices.len(),
                    n_bins
                )));
            }
            indices
        } else {
            (0..n_bins).collect::<Vec<_>>()
        };

        kept_indices.sort_unstable();
        for bin_idx in kept_indices {
            selected_bins.push(SimplifiedLikelihoodBin {
                channel: channel_name.clone(),
                name: format!("bin{bin_idx}"),
            });
            flat_indices.push(channel_offset + bin_idx);
        }
    }

    if !requested_bin_map.is_empty() {
        let mut unknown = requested_bin_map.keys().cloned().collect::<Vec<_>>();
        unknown.sort();
        unknown.dedup();
        return Err(Error::Validation(format!(
            "selection.bins references channel(s) outside selection.channels: {}",
            unknown.join(", ")
        )));
    }

    Ok(ResolvedSelection { bins: selected_bins, flat_indices })
}

fn parse_bin_selector(entry: &str) -> Result<(String, usize)> {
    let Some((channel_name, bin_name)) = entry.split_once('/') else {
        return Err(Error::Validation(format!(
            "selection.bins entry '{entry}' must be '<channel>/bin<index>'"
        )));
    };
    if channel_name.trim().is_empty() || !bin_name.starts_with("bin") {
        return Err(Error::Validation(format!(
            "selection.bins entry '{entry}' must be '<channel>/bin<index>'"
        )));
    }
    let bin_idx = bin_name["bin".len()..].parse::<usize>().map_err(|_| {
        Error::Validation(format!(
            "selection.bins entry '{entry}' must use numeric bin suffixes like 'channel/bin0'"
        ))
    })?;
    Ok((channel_name.to_string(), bin_idx))
}

fn flatten_observed_main(model: &crate::pyhf::HistFactoryModel) -> Vec<f64> {
    let mut observed = Vec::new();
    for channel in model.observed_main_by_channel() {
        observed.extend(channel.y);
    }
    observed
}

fn select_flat_entries(values: &[f64], indices: &[usize]) -> Result<Vec<f64>> {
    let mut selected = Vec::with_capacity(indices.len());
    for &idx in indices {
        let value = values.get(idx).copied().ok_or_else(|| {
            Error::Validation(format!("selection resolved flat index {idx} outside vector bounds"))
        })?;
        selected.push(value);
    }
    Ok(selected)
}

fn select_covariance_submatrix(
    covariance: &[f64],
    dimension: usize,
    keep_indices: &[usize],
) -> Result<Vec<Vec<f64>>> {
    if covariance.len() != dimension * dimension {
        return Err(Error::Validation(format!(
            "covariance length mismatch: got {} expected {}",
            covariance.len(),
            dimension * dimension
        )));
    }

    let mut rows = Vec::with_capacity(keep_indices.len());
    for &row_idx in keep_indices {
        let mut row = Vec::with_capacity(keep_indices.len());
        for &col_idx in keep_indices {
            row.push(covariance[row_idx * dimension + col_idx]);
        }
        rows.push(row);
    }
    Ok(rows)
}

fn source_constraint_covariance(
    model: &crate::pyhf::HistFactoryModel,
    nuisance_indices: &[usize],
) -> Result<Vec<Vec<f64>>> {
    let mut rows = vec![vec![0.0; nuisance_indices.len()]; nuisance_indices.len()];
    for (position, &param_idx) in nuisance_indices.iter().enumerate() {
        let parameter = model.parameters().get(param_idx).ok_or_else(|| {
            Error::Validation(format!("nuisance parameter index {param_idx} is out of range"))
        })?;
        let variance = match &parameter.constraint_term {
            None => {
                if !parameter.constrained {
                    return Err(Error::Validation(format!(
                        "reduction.constraint_covariance_source='{}' requires Gaussian-constrained nuisance parameters; '{}' is unconstrained",
                        SIMPLIFIED_LIKELIHOOD_CONSTRAINT_COVARIANCE_SOURCE_SOURCE_MODEL_CONSTRAINTS,
                        parameter.name,
                    )));
                }
                let width = parameter.constraint_width.ok_or_else(|| {
                    Error::Validation(format!(
                        "reduction.constraint_covariance_source='{}' requires finite Gaussian widths; '{}' is missing constraint_width",
                        SIMPLIFIED_LIKELIHOOD_CONSTRAINT_COVARIANCE_SOURCE_SOURCE_MODEL_CONSTRAINTS,
                        parameter.name,
                    ))
                })?;
                if !width.is_finite() || width <= 0.0 {
                    return Err(Error::Validation(format!(
                        "Gaussian constraint width for '{}' must be finite and > 0, got {width}",
                        parameter.name,
                    )));
                }
                width * width
            }
            Some(crate::pyhf::ConstraintTerm::Uniform)
            | Some(crate::pyhf::ConstraintTerm::NoConstraint)
            | Some(crate::pyhf::ConstraintTerm::LogNormal { .. })
            | Some(crate::pyhf::ConstraintTerm::Gamma { .. }) => {
                return Err(Error::Validation(format!(
                    "reduction.constraint_covariance_source='{}' currently supports only Gaussian-constrained nuisances; '{}' uses a non-Gaussian or unsupported source constraint",
                    SIMPLIFIED_LIKELIHOOD_CONSTRAINT_COVARIANCE_SOURCE_SOURCE_MODEL_CONSTRAINTS,
                    parameter.name,
                )));
            }
        };
        rows[position][position] = variance;
    }
    Ok(rows)
}

fn finite_difference_jacobian(
    model: &crate::pyhf::HistFactoryModel,
    background_params: &[f64],
    nuisance_indices: &[usize],
    selected_flat_indices: &[usize],
    relative_step: f64,
    absolute_step_floor: f64,
) -> Result<Vec<Vec<f64>>> {
    let n_bins = selected_flat_indices.len();
    let mut jacobian = vec![vec![0.0; nuisance_indices.len()]; n_bins];

    for (col_idx, &param_idx) in nuisance_indices.iter().enumerate() {
        let parameter = model.parameters().get(param_idx).ok_or_else(|| {
            Error::Validation(format!("nuisance parameter index {param_idx} is out of range"))
        })?;
        let center = background_params[param_idx];
        let nominal_step = (center.abs() * relative_step).max(absolute_step_floor);
        let up_room = parameter.bounds.1 - center;
        let down_room = center - parameter.bounds.0;
        let up_step = nominal_step.min(up_room.max(0.0));
        let down_step = nominal_step.min(down_room.max(0.0));

        let column = if up_step > 0.0 && down_step > 0.0 {
            let up = evaluate_selected_with_param_shift(
                model,
                background_params,
                param_idx,
                up_step,
                selected_flat_indices,
            )?;
            let down = evaluate_selected_with_param_shift(
                model,
                background_params,
                param_idx,
                -down_step,
                selected_flat_indices,
            )?;
            up.iter()
                .zip(down.iter())
                .map(|(up_value, down_value)| (up_value - down_value) / (up_step + down_step))
                .collect::<Vec<_>>()
        } else if up_step > 0.0 {
            let base = select_flat_entries(
                &model.expected_data(background_params)?,
                selected_flat_indices,
            )?;
            let up = evaluate_selected_with_param_shift(
                model,
                background_params,
                param_idx,
                up_step,
                selected_flat_indices,
            )?;
            up.iter()
                .zip(base.iter())
                .map(|(up_value, base_value)| (up_value - base_value) / up_step)
                .collect::<Vec<_>>()
        } else if down_step > 0.0 {
            let base = select_flat_entries(
                &model.expected_data(background_params)?,
                selected_flat_indices,
            )?;
            let down = evaluate_selected_with_param_shift(
                model,
                background_params,
                param_idx,
                -down_step,
                selected_flat_indices,
            )?;
            base.iter()
                .zip(down.iter())
                .map(|(base_value, down_value)| (base_value - down_value) / down_step)
                .collect::<Vec<_>>()
        } else {
            vec![0.0; n_bins]
        };

        for (row_idx, value) in column.into_iter().enumerate() {
            jacobian[row_idx][col_idx] = value;
        }
    }

    Ok(jacobian)
}

fn evaluate_selected_with_param_shift(
    model: &crate::pyhf::HistFactoryModel,
    background_params: &[f64],
    param_idx: usize,
    delta: f64,
    selected_flat_indices: &[usize],
) -> Result<Vec<f64>> {
    let mut shifted = background_params.to_vec();
    shifted[param_idx] += delta;
    let expected = model.expected_data(&shifted)?;
    select_flat_entries(&expected, selected_flat_indices)
}

fn project_background_covariance(
    jacobian: &[Vec<f64>],
    nuisance_covariance: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>> {
    if jacobian.is_empty() {
        return Ok(Vec::new());
    }
    let n_bins = jacobian.len();
    let n_nuisances = jacobian[0].len();
    if nuisance_covariance.len() != n_nuisances
        || nuisance_covariance.iter().any(|row| row.len() != n_nuisances)
    {
        return Err(Error::Validation(format!(
            "nuisance covariance dimension mismatch: expected {n_nuisances}x{n_nuisances}"
        )));
    }

    let jacobian_matrix = DMatrix::from_fn(n_bins, n_nuisances, |row, col| jacobian[row][col]);
    let nuisance_matrix =
        DMatrix::from_fn(n_nuisances, n_nuisances, |row, col| nuisance_covariance[row][col]);
    let projected = &jacobian_matrix * nuisance_matrix * jacobian_matrix.transpose();
    Ok((0..projected.nrows())
        .map(|row| (0..projected.ncols()).map(|col| projected[(row, col)]).collect::<Vec<_>>())
        .collect())
}

fn stat_nuisance_positions(
    model: &crate::pyhf::HistFactoryModel,
    nuisance_indices: &[usize],
) -> Vec<usize> {
    let mut stat_like_indices = HashSet::new();
    for channel_idx in 0..model.n_channels() {
        let sample_names = model.sample_names(channel_idx);
        for sample_idx in 0..sample_names.len() {
            for modifier in model.sample_modifiers(channel_idx, sample_idx) {
                match modifier {
                    crate::pyhf::ModelModifier::ShapeSys { param_indices, .. }
                    | crate::pyhf::ModelModifier::StatError { param_indices, .. } => {
                        for &param_idx in param_indices {
                            stat_like_indices.insert(param_idx);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    nuisance_indices
        .iter()
        .enumerate()
        .filter_map(|(position, param_idx)| {
            stat_like_indices.contains(param_idx).then_some(position)
        })
        .collect()
}

fn select_matrix_columns(matrix: &[Vec<f64>], positions: &[usize]) -> Vec<Vec<f64>> {
    matrix
        .iter()
        .map(|row| positions.iter().map(|&position| row[position]).collect::<Vec<_>>())
        .collect()
}

fn select_nested_submatrix(matrix: &[Vec<f64>], positions: &[usize]) -> Result<Vec<Vec<f64>>> {
    let dimension = matrix.len();
    if matrix.iter().any(|row| row.len() != dimension) {
        return Err(Error::Validation("matrix rows must have equal length".to_string()));
    }
    Ok(positions
        .iter()
        .map(|&row| positions.iter().map(|&col| matrix[row][col]).collect::<Vec<_>>())
        .collect())
}

fn normalize_pyhf_version(version: &str) -> String {
    if version.starts_with('v') { version.to_string() } else { version.replace('.', "_") }
}
