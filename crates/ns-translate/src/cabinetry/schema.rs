//! Cabinetry YAML config schema types.

use serde::Deserialize;

/// Top-level cabinetry configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct CabinetryConfig {
    pub general: GeneralBlock,
    pub regions: Vec<RegionBlock>,
    pub samples: Vec<SampleBlock>,
    pub norm_factors: Vec<NormFactorBlock>,
    #[serde(default)]
    pub systematics: Vec<SystematicBlock>,
}

/// General measurement settings.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct GeneralBlock {
    pub measurement: String,
    #[serde(default, alias = "POI")]
    pub poi: String,
    pub input_path: String,
    pub histogram_folder: String,
    #[serde(default)]
    pub variation_path: String,
    #[serde(default)]
    pub fixed: Vec<FixedParam>,
}

/// A fixed parameter.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct FixedParam {
    pub name: String,
    pub value: f64,
}

/// A phase-space region (channel).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct RegionBlock {
    pub name: String,
    #[serde(default)]
    pub variable: Option<String>,
    #[serde(default)]
    pub binning: Option<Vec<f64>>,
    #[serde(default)]
    pub filter: Option<String>,
    #[serde(default)]
    pub region_path: Option<String>,
}

/// A data or MC sample.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct SampleBlock {
    pub name: String,
    #[serde(default)]
    pub tree: Option<String>,
    #[serde(default)]
    pub sample_path: Option<StringOrArray>,
    #[serde(default)]
    pub weight: Option<String>,
    #[serde(default)]
    pub filter: Option<String>,
    #[serde(default)]
    pub data: bool,
    #[serde(default)]
    pub disable_staterror: bool,
    #[serde(default)]
    pub regions: Option<StringOrArray>,
}

/// A normalization factor (free parameter).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct NormFactorBlock {
    pub name: String,
    #[serde(default)]
    pub samples: Option<StringOrArray>,
    #[serde(default)]
    pub regions: Option<StringOrArray>,
    #[serde(default)]
    pub nominal: Option<f64>,
    #[serde(default)]
    pub bounds: Option<[f64; 2]>,
}

/// A systematic uncertainty.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct SystematicBlock {
    pub name: String,
    #[serde(rename = "Type")]
    pub syst_type: SystematicType,
    pub up: VariationTemplate,
    pub down: VariationTemplate,
    #[serde(default)]
    pub samples: Option<StringOrArray>,
    #[serde(default)]
    pub regions: Option<StringOrArray>,
    #[serde(default)]
    pub modifier_name: Option<String>,
    #[serde(default)]
    pub smoothing: Option<SmoothingConfig>,
}

/// Systematic type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum SystematicType {
    Normalization,
    NormPlusShape,
}

/// Variation template for up/down systematics.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct VariationTemplate {
    #[serde(default)]
    pub normalization: Option<f64>,
    #[serde(default)]
    pub symmetrize: bool,
    #[serde(default)]
    pub tree: Option<String>,
    #[serde(default)]
    pub weight: Option<String>,
    #[serde(default)]
    pub variable: Option<String>,
    #[serde(default)]
    pub filter: Option<String>,
    #[serde(default)]
    pub sample_path: Option<StringOrArray>,
    #[serde(default)]
    pub region_path: Option<String>,
    #[serde(default)]
    pub variation_path: Option<String>,
}

/// Smoothing configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct SmoothingConfig {
    pub algorithm: String,
    #[serde(default)]
    pub regions: Option<StringOrArray>,
    #[serde(default)]
    pub samples: Option<StringOrArray>,
}

/// A field that can be a single string or an array of strings.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StringOrArray {
    Single(String),
    Array(Vec<String>),
}

impl StringOrArray {
    /// Convert to a list of strings.
    pub fn to_vec(&self) -> Vec<String> {
        match self {
            StringOrArray::Single(s) => vec![s.clone()],
            StringOrArray::Array(v) => v.clone(),
        }
    }

    /// Check if a name matches this filter.
    pub fn contains(&self, name: &str) -> bool {
        match self {
            StringOrArray::Single(s) => s == name,
            StringOrArray::Array(v) => v.iter().any(|s| s == name),
        }
    }
}
