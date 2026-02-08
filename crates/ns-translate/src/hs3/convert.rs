//! HS3 → HistFactoryModel conversion.
//!
//! Converts a [`ResolvedWorkspace`] (from [`super::resolve`]) into the
//! canonical [`HistFactoryModel`] used by all fitting code.

use super::resolve::*;
use super::schema::*;
use crate::pyhf::model::{
    AuxiliaryPoissonConstraint, HistFactoryModel, HistoSysInterpCode, ModelChannel, ModelModifier,
    ModelSample, NormSysInterpCode, Parameter,
};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Convert an HS3 JSON string to a `HistFactoryModel`.
///
/// - `analysis_name`: which analysis to build (`None` = first).
/// - `param_point_set`: which parameter_points set for init values (`None` = `"default_values"`).
/// - `normsys_interp`: interpolation code for NormSys (ROOT default: Code1).
/// - `histosys_interp`: interpolation code for HistoSys (ROOT default: Code0).
pub fn from_hs3(
    json: &str,
    analysis_name: Option<&str>,
    param_point_set: Option<&str>,
    normsys_interp: NormSysInterpCode,
    histosys_interp: HistoSysInterpCode,
) -> Result<HistFactoryModel, Hs3ConvertError> {
    let ws: Hs3Workspace = serde_json::from_str(json).map_err(Hs3ConvertError::Json)?;
    let resolved = resolve(&ws, analysis_name, param_point_set)
        .map_err(Hs3ConvertError::Resolve)?;
    convert(resolved, normsys_interp, histosys_interp)
}

/// Convenience: parse HS3 with ROOT HistFactory default interpolation codes
/// (Code1 for NormSys, Code0 for HistoSys).
pub fn from_hs3_default(json: &str) -> Result<HistFactoryModel, Hs3ConvertError> {
    from_hs3(
        json,
        None,
        None,
        NormSysInterpCode::Code1,
        HistoSysInterpCode::Code0,
    )
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum Hs3ConvertError {
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("HS3 resolve error: {0}")]
    Resolve(#[from] Hs3ResolveError),
    #[error("conversion error: {0}")]
    Conversion(String),
}

// ---------------------------------------------------------------------------
// Core conversion
// ---------------------------------------------------------------------------

/// Convert a `ResolvedWorkspace` into a `HistFactoryModel`.
pub fn convert(
    resolved: ResolvedWorkspace,
    normsys_interp: NormSysInterpCode,
    histosys_interp: HistoSysInterpCode,
) -> Result<HistFactoryModel, Hs3ConvertError> {
    let mut parameters: Vec<Parameter> = Vec::new();
    let mut param_map: HashMap<String, usize> = HashMap::new();

    // Default bounds
    const NP_LO: f64 = -5.0;
    const NP_HI: f64 = 5.0;
    const NORM_LO: f64 = 0.0;
    const NORM_HI: f64 = 10.0;
    const GAMMA_LO: f64 = 0.0;
    const GAMMA_HI: f64 = 10.0;

    // -----------------------------------------------------------------------
    // Step 1: Register POIs first
    // -----------------------------------------------------------------------
    for poi_name in &resolved.pois {
        let bounds = resolved
            .bounds
            .get(poi_name)
            .copied()
            .unwrap_or((NORM_LO, NORM_HI));
        let init = resolved.inits.get(poi_name).copied().unwrap_or(1.0);

        let idx = parameters.len();
        parameters.push(Parameter {
            name: poi_name.clone(),
            init,
            bounds,
            constrained: false,
            constraint_center: None,
            constraint_width: None,
        });
        param_map.insert(poi_name.clone(), idx);
    }

    let poi_index = if resolved.pois.is_empty() {
        None
    } else {
        Some(0) // first POI is primary
    };

    // -----------------------------------------------------------------------
    // Helper: get-or-create parameter index
    // -----------------------------------------------------------------------
    let get_or_create_param = |name: &str,
                                   params: &mut Vec<Parameter>,
                                   map: &mut HashMap<String, usize>,
                                   resolved: &ResolvedWorkspace,
                                   is_gamma: bool|
     -> usize {
        if let Some(&idx) = map.get(name) {
            return idx;
        }

        let constraint = resolved.constraints.get(name);
        let bounds = resolved.bounds.get(name).copied().unwrap_or_else(|| {
            if is_gamma {
                (GAMMA_LO, GAMMA_HI)
            } else if constraint.is_some() {
                (NP_LO, NP_HI)
            } else {
                (NORM_LO, NORM_HI)
            }
        });

        let init = resolved.inits.get(name).copied().unwrap_or_else(|| {
            if is_gamma {
                1.0
            } else if constraint.is_some() {
                0.0
            } else {
                1.0
            }
        });

        let (constrained, center, width) = match constraint {
            Some(ci) => (true, Some(ci.center), Some(ci.width)),
            None => (false, None, None),
        };

        let idx = params.len();
        params.push(Parameter {
            name: name.to_string(),
            init,
            bounds,
            constrained,
            constraint_center: center,
            constraint_width: width,
        });
        map.insert(name.to_string(), idx);
        idx
    };

    // -----------------------------------------------------------------------
    // Step 2: Pre-scan all channels to discover parameters and accumulate
    //         staterror uncertainties
    // -----------------------------------------------------------------------

    #[derive(Debug, Clone)]
    struct StatErrorAccum {
        sum_nominal: Vec<f64>,
        sum_uncert_sq: Vec<f64>,
    }

    // Channel-level staterror accumulation: (channel_idx, staterror_name) → accum
    let mut staterror_accum: HashMap<(usize, String), StatErrorAccum> = HashMap::new();

    for (ch_idx, ch) in resolved.channels.iter().enumerate() {
        for sample in &ch.samples {
            for m in &sample.modifiers {
                match m {
                    ResolvedModifier::NormFactor { param_name } => {
                        get_or_create_param(
                            param_name,
                            &mut parameters,
                            &mut param_map,
                            &resolved,
                            false,
                        );
                    }
                    ResolvedModifier::NormSys { param_name, .. } => {
                        get_or_create_param(
                            param_name,
                            &mut parameters,
                            &mut param_map,
                            &resolved,
                            false,
                        );
                    }
                    ResolvedModifier::HistoSys { param_name, .. } => {
                        get_or_create_param(
                            param_name,
                            &mut parameters,
                            &mut param_map,
                            &resolved,
                            false,
                        );
                    }
                    ResolvedModifier::StatError { param_names, .. } => {
                        for pn in param_names {
                            get_or_create_param(
                                pn,
                                &mut parameters,
                                &mut param_map,
                                &resolved,
                                true,
                            );
                        }
                        // Accumulate errors for sigma_rel computation
                        let staterror_key = (
                            ch_idx,
                            param_names
                                .first()
                                .map(|s| s.as_str())
                                .unwrap_or("")
                                .to_string(),
                        );
                        let errors = sample.errors.as_deref().unwrap_or(&[]);
                        let accum = staterror_accum
                            .entry(staterror_key)
                            .or_insert_with(|| StatErrorAccum {
                                sum_nominal: vec![0.0; ch.n_bins],
                                sum_uncert_sq: vec![0.0; ch.n_bins],
                            });
                        for bin in 0..ch.n_bins {
                            accum.sum_nominal[bin] += sample.nominal.get(bin).copied().unwrap_or(0.0);
                            let err = errors.get(bin).copied().unwrap_or(0.0);
                            accum.sum_uncert_sq[bin] += err * err;
                        }
                    }
                    ResolvedModifier::ShapeSys { param_names, .. } => {
                        for pn in param_names {
                            get_or_create_param(
                                pn,
                                &mut parameters,
                                &mut param_map,
                                &resolved,
                                true,
                            );
                        }
                    }
                    ResolvedModifier::ShapeFactor { param_names } => {
                        for pn in param_names {
                            get_or_create_param(
                                pn,
                                &mut parameters,
                                &mut param_map,
                                &resolved,
                                true,
                            );
                        }
                    }
                    ResolvedModifier::Lumi { param_name } => {
                        get_or_create_param(
                            param_name,
                            &mut parameters,
                            &mut param_map,
                            &resolved,
                            false,
                        );
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 3: Compute staterror sigma_rel and update parameter constraints
    // -----------------------------------------------------------------------
    for ((ch_idx, _), accum) in &staterror_accum {
        let ch = &resolved.channels[*ch_idx];
        for sample in &ch.samples {
            for m in &sample.modifiers {
                if let ResolvedModifier::StatError {
                    param_names,
                    constraint_type: _,
                } = m
                {
                    for (bin, pn) in param_names.iter().enumerate() {
                        if let Some(&pidx) = param_map.get(pn) {
                            let sum_nom = accum.sum_nominal.get(bin).copied().unwrap_or(0.0);
                            let sum_sq = accum.sum_uncert_sq.get(bin).copied().unwrap_or(0.0);
                            if sum_nom > 0.0 && sum_sq > 0.0 {
                                let sigma_rel = sum_sq.sqrt() / sum_nom;
                                parameters[pidx].constrained = true;
                                parameters[pidx].constraint_center = Some(1.0);
                                parameters[pidx].constraint_width = Some(sigma_rel);
                                // Set init to 1.0 for gamma parameters
                                if parameters[pidx].init == 0.0 {
                                    parameters[pidx].init = 1.0;
                                }
                            }
                        }
                    }
                    // Only process once per staterror group per channel
                    break;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 4: Build channels
    // -----------------------------------------------------------------------
    let mut channels: Vec<ModelChannel> = Vec::with_capacity(resolved.channels.len());

    for (_ch_idx, ch) in resolved.channels.iter().enumerate() {
        let mut model_samples = Vec::with_capacity(ch.samples.len());
        let mut auxiliary_data: Vec<AuxiliaryPoissonConstraint> = Vec::new();

        for (sample_idx, sample) in ch.samples.iter().enumerate() {
            let mut model_modifiers = Vec::with_capacity(sample.modifiers.len());

            for (mod_idx, m) in sample.modifiers.iter().enumerate() {
                match m {
                    ResolvedModifier::NormFactor { param_name } => {
                        let param_idx = param_map[param_name];
                        model_modifiers.push(ModelModifier::NormFactor { param_idx });
                    }
                    ResolvedModifier::NormSys {
                        param_name,
                        hi,
                        lo,
                    } => {
                        let param_idx = param_map[param_name];
                        model_modifiers.push(ModelModifier::NormSys {
                            param_idx,
                            hi_factor: *hi,
                            lo_factor: *lo,
                            interp_code: normsys_interp,
                        });
                    }
                    ResolvedModifier::HistoSys {
                        param_name,
                        hi_template,
                        lo_template,
                    } => {
                        let param_idx = param_map[param_name];
                        model_modifiers.push(ModelModifier::HistoSys {
                            param_idx,
                            hi_template: hi_template.clone(),
                            lo_template: lo_template.clone(),
                            interp_code: histosys_interp,
                        });
                    }
                    ResolvedModifier::StatError { param_names, .. } => {
                        let param_indices: Vec<usize> = param_names
                            .iter()
                            .map(|pn| param_map[pn])
                            .collect();
                        let uncertainties = sample
                            .errors
                            .as_deref()
                            .unwrap_or(&[])
                            .to_vec();
                        model_modifiers.push(ModelModifier::StatError {
                            param_indices,
                            uncertainties,
                        });
                    }
                    ResolvedModifier::ShapeSys {
                        param_names,
                        uncertainties,
                    } => {
                        let param_indices: Vec<usize> = param_names
                            .iter()
                            .map(|pn| param_map[pn])
                            .collect();

                        // Build Barlow-Beeston tau values and auxiliary data
                        let nominal = &sample.nominal;
                        let mut tau = vec![0.0; ch.n_bins];
                        for bin in 0..ch.n_bins {
                            let sigma = uncertainties.get(bin).copied().unwrap_or(0.0);
                            let nom = nominal.get(bin).copied().unwrap_or(0.0);
                            if sigma > 0.0 && nom > 0.0 {
                                tau[bin] = (nom / sigma) * (nom / sigma);
                            }
                        }

                        auxiliary_data.push(AuxiliaryPoissonConstraint {
                            sample_idx,
                            modifier_idx: mod_idx,
                            observed: tau.clone(), // for observed dataset, aux_obs = tau
                            tau,
                        });

                        model_modifiers.push(ModelModifier::ShapeSys {
                            param_indices,
                            uncertainties: uncertainties.clone(),
                            nominal_values: nominal.clone(),
                        });
                    }
                    ResolvedModifier::ShapeFactor { param_names } => {
                        let param_indices: Vec<usize> = param_names
                            .iter()
                            .map(|pn| param_map[pn])
                            .collect();
                        model_modifiers.push(ModelModifier::ShapeFactor { param_indices });
                    }
                    ResolvedModifier::Lumi { param_name } => {
                        let param_idx = param_map[param_name];
                        model_modifiers.push(ModelModifier::Lumi { param_idx });
                    }
                }
            }

            model_samples.push(ModelSample {
                name: sample.name.clone(),
                nominal: sample.nominal.clone(),
                modifiers: model_modifiers,
            });
        }

        channels.push(ModelChannel {
            name: ch.name.clone(),
            include_in_fit: true,
            samples: model_samples,
            observed: ch.observed.clone(),
            auxiliary_data,
        });
    }

    Ok(HistFactoryModel::from_parts(parameters, poi_index, channels))
}
