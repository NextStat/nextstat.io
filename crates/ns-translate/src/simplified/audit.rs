use super::factorize::factorize_covariance_workspace;
use super::schema::{
    SimplifiedDiagnostics, SimplifiedLikelihoodWorkspace, SimplifiedUncertaintyModel,
};
use super::validate::validate_simplified_likelihood;
use serde::Serialize;
use std::collections::HashSet;

use ns_core::Result;

pub const SIMPLIFIED_LIKELIHOOD_AUDIT_SCHEMA_V0: &str = "nextstat_simplified_likelihood_audit_v0";

#[derive(Debug, Clone, Serialize)]
pub struct SimplifiedLikelihoodAudit {
    pub schema_version: String,
    pub input_schema_version: String,
    pub experiment: String,
    pub analysis_id: String,
    pub source_format: String,
    pub poi_name: String,
    pub channel_names: Vec<String>,
    pub channel_count: usize,
    pub total_bins: usize,
    pub has_signal: bool,
    pub uncertainty_model_kind: String,
    pub reduced_nuisance_count: usize,
    pub parameter_count_estimate: usize,
    pub total_observed_yield: f64,
    pub total_background_yield: f64,
    pub total_signal_yield: Option<f64>,
    pub input_has_factorization_diagnostics: bool,
    pub input_has_fidelity_diagnostics: bool,
    pub diagnostics: SimplifiedDiagnostics,
}

pub fn audit_simplified_likelihood(
    spec: &SimplifiedLikelihoodWorkspace,
) -> Result<SimplifiedLikelihoodAudit> {
    validate_simplified_likelihood(spec)?;

    let channel_names = ordered_channel_names(spec);
    let input_has_factorization_diagnostics =
        spec.diagnostics.as_ref().and_then(|diag| diag.factorization.as_ref()).is_some();
    let input_has_fidelity_diagnostics =
        spec.diagnostics.as_ref().and_then(|diag| diag.fidelity.as_ref()).is_some();

    let (uncertainty_model_kind, reduced_nuisance_count, factorization) = match &spec
        .uncertainty_model
    {
        SimplifiedUncertaintyModel::Basis { components } => (
            "basis".to_string(),
            components.len(),
            spec.diagnostics.as_ref().and_then(|diag| diag.factorization.clone()),
        ),
        SimplifiedUncertaintyModel::Covariance { .. } => {
            let factorized = factorize_covariance_workspace(spec)?;
            ("covariance".to_string(), factorized.components.len(), Some(factorized.diagnostics))
        }
    };

    let diagnostics = SimplifiedDiagnostics {
        factorization,
        fidelity: spec.diagnostics.as_ref().and_then(|diag| diag.fidelity.clone()),
    };

    Ok(SimplifiedLikelihoodAudit {
        schema_version: SIMPLIFIED_LIKELIHOOD_AUDIT_SCHEMA_V0.to_string(),
        input_schema_version: spec.schema_version.clone(),
        experiment: spec.metadata.experiment.clone(),
        analysis_id: spec.metadata.analysis_id.clone(),
        source_format: spec.metadata.source_format.clone(),
        poi_name: spec.poi.name.clone(),
        channel_names: channel_names.clone(),
        channel_count: channel_names.len(),
        total_bins: spec.bins.len(),
        has_signal: spec.signal_nominal.is_some(),
        uncertainty_model_kind,
        reduced_nuisance_count,
        parameter_count_estimate: reduced_nuisance_count + 1,
        total_observed_yield: spec.observed.iter().sum(),
        total_background_yield: spec.background_nominal.iter().sum(),
        total_signal_yield: spec.signal_nominal.as_ref().map(|values| values.iter().sum()),
        input_has_factorization_diagnostics,
        input_has_fidelity_diagnostics,
        diagnostics,
    })
}

fn ordered_channel_names(spec: &SimplifiedLikelihoodWorkspace) -> Vec<String> {
    let mut seen = HashSet::<String>::new();
    let mut names = Vec::new();
    for bin in &spec.bins {
        if seen.insert(bin.channel.clone()) {
            names.push(bin.channel.clone());
        }
    }
    names
}
