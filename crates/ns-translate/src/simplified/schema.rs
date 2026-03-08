use serde::{Deserialize, Serialize};

pub const SIMPLIFIED_LIKELIHOOD_SCHEMA_V0: &str = "nextstat_simplified_likelihood_v0";
pub const SIMPLIFIED_LIKELIHOOD_SOURCE_FORMAT_BASIS: &str = "basis";
pub const SIMPLIFIED_LIKELIHOOD_SOURCE_FORMAT_COVARIANCE: &str = "covariance";
pub const SIMPLIFIED_LIKELIHOOD_SOURCE_FORMAT_DERIVED_FROM_WORKSPACE: &str =
    "derived_from_workspace";
pub const SIMPLIFIED_LIKELIHOOD_SOURCE_WORKSPACE_FORMAT_PYHF: &str = "pyhf";
pub const SIMPLIFIED_LIKELIHOOD_SOURCE_WORKSPACE_FORMAT_HS3: &str = "hs3";
pub const SIMPLIFIED_LIKELIHOOD_BASIS_METHOD_EIGEN: &str = "eigen";
pub const SIMPLIFIED_LIKELIHOOD_JACOBIAN_METHOD_FINITE_DIFFERENCE: &str = "finite_difference";
pub const SIMPLIFIED_LIKELIHOOD_CONSTRAINT_COVARIANCE_SOURCE_SOURCE_MODEL_CONSTRAINTS: &str =
    "source_model_constraints";
pub const SIMPLIFIED_LIKELIHOOD_CONSTRAINT_COVARIANCE_SOURCE_ALIGNED_FIT_COVARIANCE: &str =
    "aligned_fit_covariance";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedLikelihoodWorkspace {
    pub schema_version: String,
    pub metadata: SimplifiedLikelihoodMetadata,
    pub poi: SimplifiedLikelihoodPoi,
    pub bins: Vec<SimplifiedLikelihoodBin>,
    pub observed: Vec<f64>,
    pub background_nominal: Vec<f64>,
    #[serde(default)]
    pub signal_nominal: Option<Vec<f64>>,
    pub uncertainty_model: SimplifiedUncertaintyModel,
    #[serde(default)]
    pub derivation: Option<SimplifiedLikelihoodDerivation>,
    #[serde(default)]
    pub diagnostics: Option<SimplifiedDiagnostics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedLikelihoodMetadata {
    pub experiment: String,
    pub analysis_id: String,
    pub source_format: String,
    pub reference: String,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedLikelihoodPoi {
    pub name: String,
    pub init: f64,
    pub bounds: [f64; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedLikelihoodBin {
    pub channel: String,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SimplifiedUncertaintyModel {
    Basis {
        #[serde(default)]
        components: Vec<SimplifiedBasisComponent>,
    },
    Covariance {
        total_covariance: Vec<Vec<f64>>,
        #[serde(default)]
        stat_covariance: Option<Vec<Vec<f64>>>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedBasisComponent {
    pub name: String,
    pub hi: Vec<f64>,
    pub lo: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedLikelihoodDerivation {
    pub source_workspace_format: String,
    #[serde(default)]
    pub source_workspace_schema_version: Option<String>,
    #[serde(default)]
    pub fit_result_schema_version: Option<String>,
    pub selected_channels: Vec<String>,
    #[serde(default)]
    pub selected_bins: Option<Vec<String>>,
    pub basis_method: String,
    pub explained_variance_target: f64,
    pub constraint_covariance_source: String,
    pub jacobian_method: String,
    #[serde(default)]
    pub split_stat_covariance: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimplifiedDiagnostics {
    #[serde(default)]
    pub factorization: Option<SimplifiedFactorizationDiagnostics>,
    #[serde(default)]
    pub fidelity: Option<SimplifiedFidelityDiagnostics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplifiedFactorizationDiagnostics {
    pub method: String,
    pub original_rank: usize,
    pub retained_rank: usize,
    pub explained_variance_fraction: f64,
    pub frobenius_residual: f64,
    #[serde(default)]
    pub clipped_negative_eigenvalues: usize,
    #[serde(default)]
    pub max_clipped_negative_eigenvalue_magnitude: f64,
    #[serde(default)]
    pub input_trace: f64,
    #[serde(default)]
    pub retained_trace: f64,
    #[serde(default)]
    pub stat_covariance_trace: Option<f64>,
    #[serde(default)]
    pub shared_systematic_trace: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimplifiedFidelityDiagnostics {
    #[serde(default)]
    pub nuisance_count_full: Option<usize>,
    #[serde(default)]
    pub nuisance_count_reduced: Option<usize>,
    #[serde(default)]
    pub bins_count: Option<usize>,
    #[serde(default)]
    pub relative_background_cov_residual: Option<f64>,
    #[serde(default)]
    pub max_abs_expected_delta_at_nominal: Option<f64>,
    #[serde(default)]
    pub max_abs_expected_delta_random_draws: Option<f64>,
    #[serde(default)]
    pub qmu_delta_smoke: Option<f64>,
    #[serde(default)]
    pub upper_limit_ratio_smoke: Option<f64>,
    #[serde(default)]
    pub max_abs_yield_delta: Option<f64>,
    #[serde(default)]
    pub max_rel_yield_delta: Option<f64>,
}
