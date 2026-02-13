//! pyhf JSON schema types

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

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

// ---------------------------------------------------------------------------
// Modifier helpers
// ---------------------------------------------------------------------------

impl Modifier {
    /// The name of this modifier (shared across all variants).
    pub fn name(&self) -> &str {
        match self {
            Modifier::NormFactor { name, .. }
            | Modifier::NormSys { name, .. }
            | Modifier::HistoSys { name, .. }
            | Modifier::ShapeSys { name, .. }
            | Modifier::ShapeFactor { name, .. }
            | Modifier::StatError { name, .. }
            | Modifier::Lumi { name, .. } => name,
        }
    }

    /// The pyhf modifier type tag (e.g. `"normfactor"`, `"normsys"`).
    pub fn modifier_type(&self) -> &'static str {
        match self {
            Modifier::NormFactor { .. } => "normfactor",
            Modifier::NormSys { .. } => "normsys",
            Modifier::HistoSys { .. } => "histosys",
            Modifier::ShapeSys { .. } => "shapesys",
            Modifier::ShapeFactor { .. } => "shapefactor",
            Modifier::StatError { .. } => "staterror",
            Modifier::Lumi { .. } => "lumi",
        }
    }
}

// ---------------------------------------------------------------------------
// Workspace operations (pyhf parity: prune, sorted, digest, rename, combine)
// ---------------------------------------------------------------------------

/// Join strategy for [`Workspace::combine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombineJoin {
    /// Error if any channel names overlap.
    None,
    /// Union: channels from both workspaces. Overlapping channels must be identical.
    Outer,
    /// Keep left workspace's version of overlapping channels.
    LeftOuter,
    /// Keep right workspace's version of overlapping channels.
    RightOuter,
}

impl Workspace {
    // -- G2: prune --------------------------------------------------------

    /// Remove specified channels, samples, modifiers, and/or measurements.
    ///
    /// Returns a new workspace with the matching components removed.
    /// Mirrors `pyhf.workspace.Workspace.prune()`.
    pub fn prune(
        &self,
        channels: &[&str],
        samples: &[&str],
        modifiers: &[&str],
        measurements: &[&str],
    ) -> Self {
        let ch_set: HashSet<&str> = channels.iter().copied().collect();
        let samp_set: HashSet<&str> = samples.iter().copied().collect();
        let mod_set: HashSet<&str> = modifiers.iter().copied().collect();
        let meas_set: HashSet<&str> = measurements.iter().copied().collect();

        let new_channels: Vec<Channel> = self
            .channels
            .iter()
            .filter(|c| !ch_set.contains(c.name.as_str()))
            .map(|c| {
                let new_samples: Vec<Sample> = c
                    .samples
                    .iter()
                    .filter(|s| !samp_set.contains(s.name.as_str()))
                    .map(|s| {
                        let new_modifiers: Vec<Modifier> = s
                            .modifiers
                            .iter()
                            .filter(|m| !mod_set.contains(m.name()))
                            .cloned()
                            .collect();
                        Sample {
                            name: s.name.clone(),
                            data: s.data.clone(),
                            modifiers: new_modifiers,
                        }
                    })
                    .collect();
                Channel { name: c.name.clone(), samples: new_samples }
            })
            .collect();

        let remaining_ch_names: HashSet<&str> =
            new_channels.iter().map(|c| c.name.as_str()).collect();

        let new_observations: Vec<Observation> = self
            .observations
            .iter()
            .filter(|o| remaining_ch_names.contains(o.name.as_str()))
            .cloned()
            .collect();

        let new_measurements: Vec<Measurement> = self
            .measurements
            .iter()
            .filter(|m| !meas_set.contains(m.name.as_str()))
            .cloned()
            .collect();

        Workspace {
            channels: new_channels,
            observations: new_observations,
            measurements: new_measurements,
            version: self.version.clone(),
        }
    }

    // -- G4: sorted -------------------------------------------------------

    /// Return a new workspace with all components in canonical (sorted) order.
    ///
    /// - Channels sorted by name.
    /// - Samples sorted by name within each channel.
    /// - Modifiers sorted by (name, type) within each sample.
    /// - Observations sorted by channel name.
    /// - Measurements sorted by name.
    ///
    /// Mirrors `pyhf.workspace.Workspace.sorted()`.
    pub fn sorted(&self) -> Self {
        let mut channels = self.channels.clone();
        for ch in &mut channels {
            for samp in &mut ch.samples {
                samp.modifiers.sort_by(|a, b| {
                    a.name().cmp(b.name()).then_with(|| a.modifier_type().cmp(b.modifier_type()))
                });
            }
            ch.samples.sort_by(|a, b| a.name.cmp(&b.name));
        }
        channels.sort_by(|a, b| a.name.cmp(&b.name));

        let mut observations = self.observations.clone();
        observations.sort_by(|a, b| a.name.cmp(&b.name));

        let mut measurements = self.measurements.clone();
        measurements.sort_by(|a, b| a.name.cmp(&b.name));

        Workspace { channels, observations, measurements, version: self.version.clone() }
    }

    // -- G5: digest -------------------------------------------------------

    /// Compute a SHA-256 digest of the canonical (sorted) workspace JSON.
    ///
    /// Mirrors `pyhf.utils.digest()`. The output is a lowercase hex string.
    pub fn digest(&self) -> String {
        use sha2::{Digest, Sha256};
        let canonical = self.sorted();
        let json = serde_json::to_string(&canonical).unwrap_or_default();
        let hash = Sha256::digest(json.as_bytes());
        format!("{:x}", hash)
    }

    // -- G3: rename -------------------------------------------------------

    /// Rename channels, samples, modifiers, and/or measurements by name mapping.
    ///
    /// Returns a new workspace with all references updated consistently.
    /// Mirrors `pyhf.workspace.Workspace.rename()`.
    pub fn rename(
        &self,
        channels: &HashMap<String, String>,
        samples: &HashMap<String, String>,
        modifiers: &HashMap<String, String>,
        measurements: &HashMap<String, String>,
    ) -> Self {
        let rename_str = |s: &str, map: &HashMap<String, String>| -> String {
            map.get(s).cloned().unwrap_or_else(|| s.to_string())
        };

        let rename_modifier = |m: &Modifier| -> Modifier {
            match m {
                Modifier::NormFactor { name, data } => {
                    Modifier::NormFactor { name: rename_str(name, modifiers), data: data.clone() }
                }
                Modifier::NormSys { name, data } => {
                    Modifier::NormSys { name: rename_str(name, modifiers), data: data.clone() }
                }
                Modifier::HistoSys { name, data } => {
                    Modifier::HistoSys { name: rename_str(name, modifiers), data: data.clone() }
                }
                Modifier::ShapeSys { name, data } => {
                    Modifier::ShapeSys { name: rename_str(name, modifiers), data: data.clone() }
                }
                Modifier::ShapeFactor { name, data } => {
                    Modifier::ShapeFactor { name: rename_str(name, modifiers), data: data.clone() }
                }
                Modifier::StatError { name, data } => {
                    Modifier::StatError { name: rename_str(name, modifiers), data: data.clone() }
                }
                Modifier::Lumi { name, data } => {
                    Modifier::Lumi { name: rename_str(name, modifiers), data: data.clone() }
                }
            }
        };

        let new_channels: Vec<Channel> = self
            .channels
            .iter()
            .map(|c| Channel {
                name: rename_str(&c.name, channels),
                samples: c
                    .samples
                    .iter()
                    .map(|s| Sample {
                        name: rename_str(&s.name, samples),
                        data: s.data.clone(),
                        modifiers: s.modifiers.iter().map(&rename_modifier).collect(),
                    })
                    .collect(),
            })
            .collect();

        let new_observations: Vec<Observation> = self
            .observations
            .iter()
            .map(|o| Observation { name: rename_str(&o.name, channels), data: o.data.clone() })
            .collect();

        let new_measurements: Vec<Measurement> = self
            .measurements
            .iter()
            .map(|m| Measurement {
                name: rename_str(&m.name, measurements),
                config: MeasurementConfig {
                    poi: m.config.poi.clone(),
                    parameters: m
                        .config
                        .parameters
                        .iter()
                        .map(|p| ParameterConfig {
                            name: rename_str(&p.name, modifiers),
                            ..p.clone()
                        })
                        .collect(),
                },
            })
            .collect();

        Workspace {
            channels: new_channels,
            observations: new_observations,
            measurements: new_measurements,
            version: self.version.clone(),
        }
    }

    // -- G1: combine ------------------------------------------------------

    /// Combine two workspaces into one.
    ///
    /// Mirrors `pyhf.workspace.Workspace.combine()`.
    ///
    /// Join modes:
    /// - `None`: error if any channel names overlap.
    /// - `Outer`: overlapping channels must be identical (deep equality); kept once.
    /// - `LeftOuter`: overlapping channels use `self`'s version.
    /// - `RightOuter`: overlapping channels use `other`'s version.
    pub fn combine(&self, other: &Workspace, join: CombineJoin) -> Result<Self, String> {
        let self_ch_names: HashSet<&str> = self.channels.iter().map(|c| c.name.as_str()).collect();
        let other_ch_names: HashSet<&str> =
            other.channels.iter().map(|c| c.name.as_str()).collect();
        let overlap: HashSet<&str> = self_ch_names.intersection(&other_ch_names).copied().collect();

        if !overlap.is_empty() && join == CombineJoin::None {
            return Err(format!(
                "Cannot combine: overlapping channels {:?}. Use a join mode to resolve.",
                overlap
            ));
        }

        if join == CombineJoin::Outer {
            for ch_name in &overlap {
                let self_ch = self.channels.iter().find(|c| c.name == *ch_name).unwrap();
                let other_ch = other.channels.iter().find(|c| c.name == *ch_name).unwrap();
                let self_json = serde_json::to_string(self_ch).unwrap_or_default();
                let other_json = serde_json::to_string(other_ch).unwrap_or_default();
                if self_json != other_json {
                    return Err(format!(
                        "Cannot combine with join=Outer: channel '{}' differs between workspaces",
                        ch_name
                    ));
                }
            }
        }

        let mut channels = self.channels.clone();
        for ch in &other.channels {
            if overlap.contains(ch.name.as_str()) {
                match join {
                    CombineJoin::LeftOuter | CombineJoin::Outer => {
                        // Keep self's version (already in channels).
                    }
                    CombineJoin::RightOuter => {
                        if let Some(pos) = channels.iter().position(|c| c.name == ch.name) {
                            channels[pos] = ch.clone();
                        }
                    }
                    CombineJoin::None => unreachable!(),
                }
            } else {
                channels.push(ch.clone());
            }
        }

        let mut observations = self.observations.clone();
        for obs in &other.observations {
            if overlap.contains(obs.name.as_str()) {
                match join {
                    CombineJoin::LeftOuter | CombineJoin::Outer => {}
                    CombineJoin::RightOuter => {
                        if let Some(pos) = observations.iter().position(|o| o.name == obs.name) {
                            observations[pos] = obs.clone();
                        }
                    }
                    CombineJoin::None => unreachable!(),
                }
            } else {
                observations.push(obs.clone());
            }
        }

        let mut measurement_names: HashSet<String> =
            self.measurements.iter().map(|m| m.name.clone()).collect();
        let mut measurements_out = self.measurements.clone();
        for m in &other.measurements {
            if !measurement_names.contains(&m.name) {
                measurements_out.push(m.clone());
                measurement_names.insert(m.name.clone());
            }
        }

        Ok(Workspace {
            channels,
            observations,
            measurements: measurements_out,
            version: self.version.clone().or_else(|| other.version.clone()),
        })
    }
}
