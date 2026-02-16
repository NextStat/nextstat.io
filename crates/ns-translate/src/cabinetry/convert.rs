//! Cabinetry config parsing and conversion to pyhf `Workspace`.

use super::schema::*;
use crate::pyhf::schema::{
    Channel, Measurement, MeasurementConfig, Modifier, NormSysData, Observation, ParameterConfig,
    Sample, Workspace,
};

/// Parse a cabinetry YAML config string.
pub fn parse_cabinetry_config(yaml: &str) -> Result<CabinetryConfig, CabinetryError> {
    serde_yaml_ng::from_str(yaml).map_err(CabinetryError::Yaml)
}

/// Histograms for a (region, sample) pair, needed to build a full Workspace.
pub struct HistogramData {
    /// Region (channel) name.
    pub region: String,
    /// Sample name.
    pub sample: String,
    /// Bin contents (expected event counts per bin).
    pub nominal: Vec<f64>,
    /// Per-bin statistical uncertainties (for staterror modifier). Empty = auto from sqrt(nominal).
    pub stat_uncertainties: Vec<f64>,
    /// Systematic histograms: (syst_name, up_data, down_data).
    pub systematics: Vec<(String, Vec<f64>, Vec<f64>)>,
}

/// Convert a cabinetry config + pre-built histograms into a pyhf `Workspace`.
///
/// The caller is responsible for building histograms (cabinetry normally does this via
/// uproot). This function only performs structure mapping.
pub fn to_workspace(
    config: &CabinetryConfig,
    histograms: &[HistogramData],
    observations: &[(String, Vec<f64>)],
) -> Result<Workspace, CabinetryError> {
    let mut channels = Vec::new();

    for region in &config.regions {
        let mut samples = Vec::new();

        for sample_cfg in &config.samples {
            // Skip data samples (they go into observations).
            if sample_cfg.data {
                continue;
            }

            // Check region filter.
            if let Some(ref region_filter) = sample_cfg.regions
                && !region_filter.contains(&region.name)
            {
                continue;
            }

            // Find histogram data for this (region, sample) pair.
            let histo =
                histograms.iter().find(|h| h.region == region.name && h.sample == sample_cfg.name);

            let nominal = match histo {
                Some(h) => h.nominal.clone(),
                None => {
                    return Err(CabinetryError::MissingHistogram(
                        region.name.clone(),
                        sample_cfg.name.clone(),
                    ));
                }
            };

            let mut modifiers = Vec::new();

            // NormFactor modifiers.
            for nf in &config.norm_factors {
                if let Some(ref sf) = nf.samples
                    && !sf.contains(&sample_cfg.name)
                {
                    continue;
                }
                if let Some(ref rf) = nf.regions
                    && !rf.contains(&region.name)
                {
                    continue;
                }
                modifiers.push(Modifier::NormFactor { name: nf.name.clone(), data: None });
            }

            // Systematic modifiers.
            for syst in &config.systematics {
                if let Some(ref sf) = syst.samples
                    && !sf.contains(&sample_cfg.name)
                {
                    continue;
                }
                if let Some(ref rf) = syst.regions
                    && !rf.contains(&region.name)
                {
                    continue;
                }

                let mod_name = syst.modifier_name.as_deref().unwrap_or(&syst.name);

                match syst.syst_type {
                    SystematicType::Normalization => {
                        let hi = 1.0 + syst.up.normalization.unwrap_or(0.0);
                        let lo = 1.0 + syst.down.normalization.unwrap_or(0.0);
                        modifiers.push(Modifier::NormSys {
                            name: mod_name.to_string(),
                            data: NormSysData { hi, lo },
                        });
                    }
                    SystematicType::NormPlusShape => {
                        // If histograms provide shape systematics, use them.
                        if let Some(h) = histo {
                            if let Some((_, up_data, down_data)) =
                                h.systematics.iter().find(|(n, _, _)| n == mod_name)
                            {
                                modifiers.push(Modifier::HistoSys {
                                    name: mod_name.to_string(),
                                    data: crate::pyhf::schema::HistoSysData {
                                        hi_data: up_data.clone(),
                                        lo_data: down_data.clone(),
                                    },
                                });
                            } else if let (Some(up_norm), Some(down_norm)) =
                                (syst.up.normalization, syst.down.normalization)
                            {
                                // Fall back to normsys if only normalization is specified.
                                modifiers.push(Modifier::NormSys {
                                    name: mod_name.to_string(),
                                    data: NormSysData { hi: 1.0 + up_norm, lo: 1.0 + down_norm },
                                });
                            }
                        }
                    }
                }
            }

            // StatError modifier (if not disabled).
            if !sample_cfg.disable_staterror
                && let Some(h) = histo
            {
                let stat_unc = if h.stat_uncertainties.is_empty() {
                    nominal.iter().map(|v| v.sqrt()).collect()
                } else {
                    h.stat_uncertainties.clone()
                };
                modifiers.push(Modifier::StatError {
                    name: format!("staterror_{}", region.name),
                    data: stat_unc,
                });
            }

            samples.push(Sample { name: sample_cfg.name.clone(), data: nominal, modifiers });
        }

        channels.push(Channel { name: region.name.clone(), samples });
    }

    // Build observations.
    let obs: Vec<Observation> = observations
        .iter()
        .map(|(name, data)| Observation { name: name.clone(), data: data.clone() })
        .collect();

    // Build measurement.
    let poi_name = if config.general.poi.is_empty() {
        config.norm_factors.first().map(|nf| nf.name.clone()).unwrap_or_default()
    } else {
        config.general.poi.clone()
    };

    let mut parameters = Vec::new();
    for nf in &config.norm_factors {
        let mut pc = ParameterConfig {
            name: nf.name.clone(),
            inits: Vec::new(),
            bounds: Vec::new(),
            fixed: false,
            auxdata: Vec::new(),
            sigmas: Vec::new(),
            constraint: None,
        };
        if let Some(nom) = nf.nominal {
            pc.inits = vec![nom];
        }
        if let Some(b) = nf.bounds {
            pc.bounds = vec![b];
        }
        parameters.push(pc);
    }
    for fp in &config.general.fixed {
        if let Some(pc) = parameters.iter_mut().find(|p| p.name == fp.name) {
            pc.fixed = true;
            pc.inits = vec![fp.value];
        } else {
            parameters.push(ParameterConfig {
                name: fp.name.clone(),
                inits: vec![fp.value],
                bounds: Vec::new(),
                fixed: true,
                auxdata: Vec::new(),
                sigmas: Vec::new(),
                constraint: None,
            });
        }
    }

    let measurement = Measurement {
        name: config.general.measurement.clone(),
        config: MeasurementConfig { poi: poi_name, parameters },
    };

    Ok(Workspace {
        channels,
        observations: obs,
        measurements: vec![measurement],
        version: Some("1.0.0".into()),
    })
}

/// Errors from cabinetry config processing.
#[derive(Debug, thiserror::Error)]
pub enum CabinetryError {
    #[error("YAML parse error: {0}")]
    Yaml(serde_yaml_ng::Error),
    #[error("missing histogram for region '{0}', sample '{1}'")]
    MissingHistogram(String, String),
    #[error("conversion error: {0}")]
    Conversion(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
