//! pyhf JSON schema types

use serde::{Deserialize, Serialize};

/// pyhf workspace representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workspace {
    /// Channels
    pub channels: Vec<Channel>,
    /// Observations
    pub observations: Vec<Observation>,
    /// Measurements
    pub measurements: Vec<Measurement>,
    /// Schema version
    #[serde(default)]
    pub version: Option<String>,
}

/// Channel (region)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Channel {
    /// Channel name
    pub name: String,
    /// Samples in this channel
    pub samples: Vec<Sample>,
}

/// Sample (process)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    /// Sample name
    pub name: String,
    /// Expected event counts per bin
    pub data: Vec<f64>,
    /// Modifiers (systematics)
    pub modifiers: Vec<Modifier>,
}

/// Modifier (systematic uncertainty)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Modifier {
    /// normfactor: free-floating normalization (e.g., POI)
    #[serde(rename = "normfactor")]
    NormFactor {
        /// Modifier name.
        name: String,
        /// Optional modifier payload (currently unused).
        #[serde(default)]
        data: Option<serde_json::Value>,
    },

    /// normsys: log-normal normalization uncertainty
    #[serde(rename = "normsys")]
    NormSys {
        /// Modifier name.
        name: String,
        /// Log-normal parameters.
        data: NormSysData,
    },

    /// histosys: histogram-based shape uncertainty
    #[serde(rename = "histosys")]
    HistoSys {
        /// Modifier name.
        name: String,
        /// Up/down templates.
        data: HistoSysData,
    },

    /// shapesys: Gaussian constraint on bin-by-bin shape
    #[serde(rename = "shapesys")]
    ShapeSys {
        /// Modifier name.
        name: String,
        /// Per-bin uncertainties (σ) used for Barlow–Beeston auxiliary constraints.
        data: Vec<f64>,
    },

    /// shapefactor: unconstrained shape variation
    #[serde(rename = "shapefactor")]
    ShapeFactor {
        /// Modifier name.
        name: String,
        /// Optional modifier payload (currently unused).
        #[serde(default)]
        data: Option<serde_json::Value>,
    },

    /// staterror: Poisson statistical error
    #[serde(rename = "staterror")]
    StatError {
        /// Modifier name.
        name: String,
        /// Per-bin uncertainties (σ).
        data: Vec<f64>,
    },

    /// lumi: luminosity uncertainty
    #[serde(rename = "lumi")]
    Lumi {
        /// Modifier name.
        name: String,
        /// Optional modifier payload (currently unused).
        #[serde(default)]
        data: Option<serde_json::Value>,
    },
}

/// normsys data (hi/lo factors)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormSysData {
    /// High (up) multiplicative factor.
    pub hi: f64,
    /// Low (down) multiplicative factor.
    pub lo: f64,
}

/// histosys data (up/down histograms)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoSysData {
    /// High (up) template values.
    pub hi_data: Vec<f64>,
    /// Low (down) template values.
    pub lo_data: Vec<f64>,
}

/// Observation (data)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Channel name this observation belongs to
    pub name: String,
    /// Observed event counts per bin
    pub data: Vec<f64>,
}

/// Measurement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurement {
    /// Measurement name
    pub name: String,
    /// Configuration
    pub config: MeasurementConfig,
}

/// Measurement config
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementConfig {
    /// Parameter of interest
    pub poi: String,
    /// Parameter configurations
    #[serde(default)]
    pub parameters: Vec<ParameterConfig>,
}

/// Parameter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterConfig {
    /// Parameter name
    pub name: String,
    /// Initial values
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inits: Vec<f64>,
    /// Bounds [[min, max]]
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bounds: Vec<[f64; 2]>,
    /// Whether this parameter is fixed (frozen) in fits.
    ///
    /// Mirrors pyhf: `measurements[].config.parameters[].fixed`.
    #[serde(default)]
    pub fixed: bool,
    /// Auxiliary data (constraint centers)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub auxdata: Vec<f64>,
    /// Constraint widths
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sigmas: Vec<f64>,
    /// Non-standard extension: HistFactory `<ConstraintTerm>` metadata.
    ///
    /// This is ignored by pyhf, but preserved by NextStat to support ROOT/HistFactory/TREx
    /// constraint-term semantics when ingesting from `combination.xml`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraint: Option<ConstraintSpec>,
}

/// Non-standard extension: constraint-term specification (from HistFactory `<ConstraintTerm>`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintSpec {
    /// Constraint distribution type (case-insensitive).
    ///
    /// Common values: `Gamma`, `LogNormal`, `Uniform`, `NoConstraint`/`NoSyst`.
    #[serde(rename = "type")]
    pub constraint_type: String,
    /// Relative uncertainty parameter (HistFactory attribute `RelativeUncertainty`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rel_uncertainty: Option<f64>,
}
