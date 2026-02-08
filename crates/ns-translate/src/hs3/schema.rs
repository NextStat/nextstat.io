//! HS3 (HEP Statistics Serialization Standard) JSON schema types.
//!
//! These types model the HS3 v0.2 JSON structure as produced by ROOT 6.37+.
//! The top-level type is [`Hs3Workspace`].
//!
//! Distribution and modifier enums use custom `Deserialize` implementations
//! to gracefully handle unknown `type` tags (forward-compatibility with
//! future HS3 spec versions).

use serde::{Deserialize, Deserializer, Serialize, Serializer};

// ---------------------------------------------------------------------------
// Top-level workspace
// ---------------------------------------------------------------------------

/// Complete HS3 workspace — the top-level JSON object.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3Workspace {
    pub distributions: Vec<Hs3Distribution>,
    pub data: Vec<Hs3Data>,
    pub domains: Vec<Hs3Domain>,
    pub parameter_points: Vec<Hs3ParameterPointSet>,
    pub analyses: Vec<Hs3Analysis>,
    pub likelihoods: Vec<Hs3Likelihood>,
    pub metadata: Hs3Metadata,
    #[serde(default)]
    pub misc: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Distributions (tagged union on "type")
// ---------------------------------------------------------------------------

/// Distribution entry. Known types are deserialized into their respective
/// structs; unknown types are preserved as opaque JSON for forward-compat.
#[derive(Debug, Clone)]
pub enum Hs3Distribution {
    HistFactory(Hs3HistFactoryDist),
    Gaussian(Hs3GaussianDist),
    Poisson(Hs3PoissonDist),
    LogNormal(Hs3LogNormalDist),
    /// Unknown distribution type — preserved as raw JSON for roundtrip.
    Unknown(serde_json::Value),
}

impl<'de> Deserialize<'de> for Hs3Distribution {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let type_str = value
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| serde::de::Error::missing_field("type"))?;

        match type_str {
            "histfactory_dist" => serde_json::from_value(value.clone())
                .map(Hs3Distribution::HistFactory)
                .map_err(serde::de::Error::custom),
            "gaussian_dist" => serde_json::from_value(value.clone())
                .map(Hs3Distribution::Gaussian)
                .map_err(serde::de::Error::custom),
            "poisson_dist" => serde_json::from_value(value.clone())
                .map(Hs3Distribution::Poisson)
                .map_err(serde::de::Error::custom),
            "lognormal_dist" => serde_json::from_value(value.clone())
                .map(Hs3Distribution::LogNormal)
                .map_err(serde::de::Error::custom),
            _ => Ok(Hs3Distribution::Unknown(value)),
        }
    }
}

impl Serialize for Hs3Distribution {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Hs3Distribution::HistFactory(d) => d.serialize(serializer),
            Hs3Distribution::Gaussian(d) => d.serialize(serializer),
            Hs3Distribution::Poisson(d) => d.serialize(serializer),
            Hs3Distribution::LogNormal(d) => d.serialize(serializer),
            Hs3Distribution::Unknown(v) => v.serialize(serializer),
        }
    }
}

impl Hs3Distribution {
    /// Returns the distribution name, if available.
    pub fn name(&self) -> Option<&str> {
        match self {
            Hs3Distribution::HistFactory(d) => Some(&d.name),
            Hs3Distribution::Gaussian(d) => Some(&d.name),
            Hs3Distribution::Poisson(d) => Some(&d.name),
            Hs3Distribution::LogNormal(d) => Some(&d.name),
            Hs3Distribution::Unknown(v) => v.get("name").and_then(|n| n.as_str()),
        }
    }

    /// Returns the `"type"` tag string.
    pub fn type_tag(&self) -> &str {
        match self {
            Hs3Distribution::HistFactory(_) => "histfactory_dist",
            Hs3Distribution::Gaussian(_) => "gaussian_dist",
            Hs3Distribution::Poisson(_) => "poisson_dist",
            Hs3Distribution::LogNormal(_) => "lognormal_dist",
            Hs3Distribution::Unknown(v) => {
                v.get("type").and_then(|t| t.as_str()).unwrap_or("unknown")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HistFactory distribution (= one channel)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3HistFactoryDist {
    pub name: String,
    #[serde(rename = "type")]
    pub dist_type: String,
    pub axes: Vec<Hs3Axis>,
    pub samples: Vec<Hs3Sample>,
}

/// Sample within a HistFactory distribution.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3Sample {
    pub name: String,
    pub data: Hs3SampleData,
    pub modifiers: Vec<Hs3Modifier>,
}

/// Sample data block (nominal yields + optional per-bin errors for staterror).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3SampleData {
    pub contents: Vec<f64>,
    #[serde(default)]
    pub errors: Option<Vec<f64>>,
}

// ---------------------------------------------------------------------------
// Modifiers (tagged union on "type")
// ---------------------------------------------------------------------------

/// Modifier on a sample. Custom Deserialize handles unknown modifier types.
#[derive(Debug, Clone)]
pub enum Hs3Modifier {
    NormFactor {
        name: String,
        parameter: String,
    },
    NormSys {
        name: String,
        parameter: String,
        constraint_name: Option<String>,
        data: Hs3NormSysData,
    },
    HistoSys {
        name: String,
        parameter: String,
        constraint_name: String,
        data: Hs3HistoSysData,
    },
    StatError {
        name: String,
        parameters: Vec<String>,
        constraint_type: String,
    },
    ShapeSys {
        name: String,
        parameters: Vec<String>,
        constraint_type: Option<String>,
        data: Option<Hs3ShapeSysData>,
    },
    ShapeFactor {
        name: String,
        parameters: Option<Vec<String>>,
        parameter: Option<String>,
    },
    Lumi {
        name: String,
        parameter: String,
        constraint_name: Option<String>,
    },
    /// Unknown modifier type — preserved as raw JSON.
    Unknown(serde_json::Value),
}

/// Helper structs for serde deserialization of individual modifier variants.
mod modifier_de {
    use serde::Deserialize;

    #[derive(Deserialize)]
    pub struct NormFactor {
        pub name: String,
        pub parameter: String,
    }
    #[derive(Deserialize)]
    pub struct NormSys {
        pub name: String,
        pub parameter: String,
        #[serde(default)]
        pub constraint_name: Option<String>,
        pub data: super::Hs3NormSysData,
    }
    #[derive(Deserialize)]
    pub struct HistoSys {
        pub name: String,
        pub parameter: String,
        pub constraint_name: String,
        pub data: super::Hs3HistoSysData,
    }
    #[derive(Deserialize)]
    pub struct StatError {
        pub name: String,
        pub parameters: Vec<String>,
        pub constraint_type: String,
    }
    #[derive(Deserialize)]
    pub struct ShapeSys {
        pub name: String,
        pub parameters: Vec<String>,
        #[serde(default)]
        pub constraint_type: Option<String>,
        #[serde(default)]
        pub data: Option<super::Hs3ShapeSysData>,
    }
    #[derive(Deserialize)]
    pub struct ShapeFactor {
        pub name: String,
        #[serde(default)]
        pub parameters: Option<Vec<String>>,
        #[serde(default)]
        pub parameter: Option<String>,
    }
    #[derive(Deserialize)]
    pub struct Lumi {
        pub name: String,
        pub parameter: String,
        #[serde(default)]
        pub constraint_name: Option<String>,
    }
}

impl<'de> Deserialize<'de> for Hs3Modifier {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let type_str = value
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| serde::de::Error::missing_field("type"))?;

        match type_str {
            "normfactor" => {
                let m: modifier_de::NormFactor =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(Hs3Modifier::NormFactor {
                    name: m.name,
                    parameter: m.parameter,
                })
            }
            "normsys" => {
                let m: modifier_de::NormSys =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(Hs3Modifier::NormSys {
                    name: m.name,
                    parameter: m.parameter,
                    constraint_name: m.constraint_name,
                    data: m.data,
                })
            }
            "histosys" => {
                let m: modifier_de::HistoSys =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(Hs3Modifier::HistoSys {
                    name: m.name,
                    parameter: m.parameter,
                    constraint_name: m.constraint_name,
                    data: m.data,
                })
            }
            "staterror" => {
                let m: modifier_de::StatError =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(Hs3Modifier::StatError {
                    name: m.name,
                    parameters: m.parameters,
                    constraint_type: m.constraint_type,
                })
            }
            "shapesys" => {
                let m: modifier_de::ShapeSys =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(Hs3Modifier::ShapeSys {
                    name: m.name,
                    parameters: m.parameters,
                    constraint_type: m.constraint_type,
                    data: m.data,
                })
            }
            "shapefactor" => {
                let m: modifier_de::ShapeFactor =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(Hs3Modifier::ShapeFactor {
                    name: m.name,
                    parameters: m.parameters,
                    parameter: m.parameter,
                })
            }
            "lumi" => {
                let m: modifier_de::Lumi =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(Hs3Modifier::Lumi {
                    name: m.name,
                    parameter: m.parameter,
                    constraint_name: m.constraint_name,
                })
            }
            _ => Ok(Hs3Modifier::Unknown(value)),
        }
    }
}

impl Serialize for Hs3Modifier {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeMap;
        match self {
            Hs3Modifier::NormFactor { name, parameter } => {
                let mut map = serializer.serialize_map(Some(3))?;
                map.serialize_entry("type", "normfactor")?;
                map.serialize_entry("name", name)?;
                map.serialize_entry("parameter", parameter)?;
                map.end()
            }
            Hs3Modifier::NormSys {
                name,
                parameter,
                constraint_name,
                data,
            } => {
                let mut map = serializer.serialize_map(Some(5))?;
                map.serialize_entry("type", "normsys")?;
                map.serialize_entry("name", name)?;
                map.serialize_entry("parameter", parameter)?;
                map.serialize_entry("constraint_name", constraint_name)?;
                map.serialize_entry("data", data)?;
                map.end()
            }
            Hs3Modifier::HistoSys {
                name,
                parameter,
                constraint_name,
                data,
            } => {
                let mut map = serializer.serialize_map(Some(5))?;
                map.serialize_entry("type", "histosys")?;
                map.serialize_entry("name", name)?;
                map.serialize_entry("parameter", parameter)?;
                map.serialize_entry("constraint_name", constraint_name)?;
                map.serialize_entry("data", data)?;
                map.end()
            }
            Hs3Modifier::StatError {
                name,
                parameters,
                constraint_type,
            } => {
                let mut map = serializer.serialize_map(Some(4))?;
                map.serialize_entry("type", "staterror")?;
                map.serialize_entry("name", name)?;
                map.serialize_entry("parameters", parameters)?;
                map.serialize_entry("constraint_type", constraint_type)?;
                map.end()
            }
            Hs3Modifier::ShapeSys {
                name,
                parameters,
                constraint_type,
                data,
            } => {
                let mut map = serializer.serialize_map(Some(5))?;
                map.serialize_entry("type", "shapesys")?;
                map.serialize_entry("name", name)?;
                map.serialize_entry("parameters", parameters)?;
                if let Some(ct) = constraint_type {
                    map.serialize_entry("constraint_type", ct)?;
                }
                if let Some(d) = data {
                    map.serialize_entry("data", d)?;
                }
                map.end()
            }
            Hs3Modifier::ShapeFactor {
                name,
                parameters,
                parameter,
            } => {
                let mut map = serializer.serialize_map(Some(4))?;
                map.serialize_entry("type", "shapefactor")?;
                map.serialize_entry("name", name)?;
                if let Some(ps) = parameters {
                    map.serialize_entry("parameters", ps)?;
                }
                if let Some(p) = parameter {
                    map.serialize_entry("parameter", p)?;
                }
                map.end()
            }
            Hs3Modifier::Lumi {
                name,
                parameter,
                constraint_name,
            } => {
                let mut map = serializer.serialize_map(Some(4))?;
                map.serialize_entry("type", "lumi")?;
                map.serialize_entry("name", name)?;
                map.serialize_entry("parameter", parameter)?;
                if let Some(cn) = constraint_name {
                    map.serialize_entry("constraint_name", cn)?;
                }
                map.end()
            }
            Hs3Modifier::Unknown(v) => v.serialize(serializer),
        }
    }
}

impl Hs3Modifier {
    /// Returns the modifier name, if available.
    pub fn name(&self) -> Option<&str> {
        match self {
            Hs3Modifier::NormFactor { name, .. }
            | Hs3Modifier::NormSys { name, .. }
            | Hs3Modifier::HistoSys { name, .. }
            | Hs3Modifier::StatError { name, .. }
            | Hs3Modifier::ShapeSys { name, .. }
            | Hs3Modifier::ShapeFactor { name, .. }
            | Hs3Modifier::Lumi { name, .. } => Some(name),
            Hs3Modifier::Unknown(v) => v.get("name").and_then(|n| n.as_str()),
        }
    }

    /// Returns the `"type"` tag string.
    pub fn type_tag(&self) -> &str {
        match self {
            Hs3Modifier::NormFactor { .. } => "normfactor",
            Hs3Modifier::NormSys { .. } => "normsys",
            Hs3Modifier::HistoSys { .. } => "histosys",
            Hs3Modifier::StatError { .. } => "staterror",
            Hs3Modifier::ShapeSys { .. } => "shapesys",
            Hs3Modifier::ShapeFactor { .. } => "shapefactor",
            Hs3Modifier::Lumi { .. } => "lumi",
            Hs3Modifier::Unknown(v) => {
                v.get("type").and_then(|t| t.as_str()).unwrap_or("unknown")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Modifier data types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3NormSysData {
    pub hi: f64,
    pub lo: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3HistoSysData {
    pub hi: Hs3HistoTemplate,
    pub lo: Hs3HistoTemplate,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3HistoTemplate {
    pub contents: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3ShapeSysData {
    pub vals: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Constraint distributions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3GaussianDist {
    pub name: String,
    #[serde(rename = "type")]
    pub dist_type: String,
    /// Constrained parameter name.
    pub x: String,
    /// Global observable name (resolved to a value from parameter_points).
    pub mean: String,
    pub sigma: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3PoissonDist {
    pub name: String,
    #[serde(rename = "type")]
    pub dist_type: String,
    pub x: String,
    pub mean: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3LogNormalDist {
    pub name: String,
    #[serde(rename = "type")]
    pub dist_type: String,
    pub x: String,
    pub mean: String,
    pub sigma: f64,
}

// ---------------------------------------------------------------------------
// Axes
// ---------------------------------------------------------------------------

/// Axis definition (used in distributions and data).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3Axis {
    pub name: String,
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
    #[serde(default)]
    pub nbins: Option<usize>,
    #[serde(default)]
    pub edges: Option<Vec<f64>>,
}

// ---------------------------------------------------------------------------
// Binned data
// ---------------------------------------------------------------------------

/// Observed data entry (binned histogram).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3Data {
    pub name: String,
    #[serde(rename = "type")]
    pub data_type: String,
    #[serde(default)]
    pub axes: Vec<Hs3Axis>,
    pub contents: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Domains (parameter bounds)
// ---------------------------------------------------------------------------

/// Parameter domain (product of 1D intervals).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3Domain {
    pub name: String,
    #[serde(rename = "type")]
    pub domain_type: String,
    pub axes: Vec<Hs3DomainAxis>,
}

/// Single axis of a parameter domain.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3DomainAxis {
    pub name: String,
    pub min: f64,
    pub max: f64,
}

// ---------------------------------------------------------------------------
// Parameter points (init values)
// ---------------------------------------------------------------------------

/// Named set of parameter values (e.g., "default_values", "bestfit").
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3ParameterPointSet {
    pub name: String,
    pub parameters: Vec<Hs3ParameterValue>,
}

/// Single parameter value within a parameter point set.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3ParameterValue {
    pub name: String,
    pub value: f64,
}

// ---------------------------------------------------------------------------
// Analyses and likelihoods
// ---------------------------------------------------------------------------

/// Analysis definition — selects a likelihood and POIs.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3Analysis {
    pub name: String,
    pub likelihood: String,
    pub parameters_of_interest: Vec<String>,
    pub domains: Vec<String>,
}

/// Likelihood — maps distribution names to data names.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3Likelihood {
    pub name: String,
    pub distributions: Vec<String>,
    pub data: Vec<String>,
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3Metadata {
    pub hs3_version: String,
    #[serde(default)]
    pub packages: Option<Vec<Hs3Package>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Hs3Package {
    pub name: String,
    pub version: String,
}
