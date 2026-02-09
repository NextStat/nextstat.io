//! HS3 Export — serialize a `HistFactoryModel` back to HS3 JSON.
//!
//! Enables roundtrip workflows: load HS3 → fit → export updated HS3 with
//! bestfit parameter points.

use super::schema::*;
use crate::pyhf::model::{HistFactoryModel, ModelModifier};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Export a `HistFactoryModel` to an HS3 workspace.
///
/// - `analysis_name`: name for the exported analysis object.
/// - `bestfit_params`: if provided, included as an additional parameter_points set
///   with the given name and values.
/// - `original_hs3`: if provided, preserve metadata and misc from the original.
pub fn export_hs3(
    model: &HistFactoryModel,
    analysis_name: &str,
    bestfit_params: Option<(&str, &[f64])>,
    original_hs3: Option<&Hs3Workspace>,
) -> Hs3Workspace {
    let params = model.parameters();

    // -----------------------------------------------------------------------
    // Build distributions
    // -----------------------------------------------------------------------
    let mut distributions: Vec<Hs3Distribution> = Vec::new();
    let mut constraint_params_seen: HashSet<usize> = HashSet::new();

    // One histfactory_dist per channel
    for ch_idx in 0..model.n_channels() {
        let ch_name = model.channel_names()[ch_idx].clone();
        let n_bins = model.channel_bin_count(ch_idx).unwrap_or(0);

        let mut hs3_samples: Vec<Hs3Sample> = Vec::new();

        for (sample_idx, sample_name) in model.sample_names(ch_idx).iter().enumerate() {
            let nominal = model.sample_nominal(ch_idx, sample_idx).unwrap_or(&[]).to_vec();

            let mut modifiers: Vec<Hs3Modifier> = Vec::new();
            let mut sample_errors: Option<Vec<f64>> = None;

            // Walk internal modifiers for this sample
            for m in model.sample_modifiers(ch_idx, sample_idx) {
                match m {
                    ModelModifier::NormFactor { param_idx } => {
                        modifiers.push(Hs3Modifier::NormFactor {
                            name: params[*param_idx].name.clone(),
                            parameter: params[*param_idx].name.clone(),
                        });
                    }
                    ModelModifier::NormSys { param_idx, hi_factor, lo_factor, .. } => {
                        let pname = &params[*param_idx].name;
                        constraint_params_seen.insert(*param_idx);
                        modifiers.push(Hs3Modifier::NormSys {
                            name: pname.clone(),
                            parameter: pname.clone(),
                            constraint_name: Some(format!("{pname}Constraint")),
                            data: Hs3NormSysData { hi: *hi_factor, lo: *lo_factor },
                        });
                    }
                    ModelModifier::HistoSys { param_idx, hi_template, lo_template, .. } => {
                        let pname = &params[*param_idx].name;
                        constraint_params_seen.insert(*param_idx);
                        modifiers.push(Hs3Modifier::HistoSys {
                            name: pname.clone(),
                            parameter: pname.clone(),
                            constraint_name: format!("{pname}Constraint"),
                            data: Hs3HistoSysData {
                                hi: Hs3HistoTemplate { contents: hi_template.clone() },
                                lo: Hs3HistoTemplate { contents: lo_template.clone() },
                            },
                        });
                    }
                    ModelModifier::StatError { param_indices, uncertainties } => {
                        let param_names: Vec<String> =
                            param_indices.iter().map(|&idx| params[idx].name.clone()).collect();
                        for &idx in param_indices {
                            constraint_params_seen.insert(idx);
                        }
                        modifiers.push(Hs3Modifier::StatError {
                            name: format!("staterror_{ch_name}"),
                            parameters: param_names,
                            constraint_type: "Poisson".to_string(),
                        });
                        if !uncertainties.is_empty() {
                            sample_errors = Some(uncertainties.clone());
                        }
                    }
                    ModelModifier::Lumi { param_idx } => {
                        let pname = &params[*param_idx].name;
                        constraint_params_seen.insert(*param_idx);
                        modifiers.push(Hs3Modifier::Lumi {
                            name: pname.clone(),
                            parameter: pname.clone(),
                            constraint_name: Some(format!("{pname}Constraint")),
                        });
                    }
                    ModelModifier::ShapeSys { param_indices, uncertainties, .. } => {
                        let param_names: Vec<String> =
                            param_indices.iter().map(|&idx| params[idx].name.clone()).collect();
                        for &idx in param_indices {
                            constraint_params_seen.insert(idx);
                        }
                        modifiers.push(Hs3Modifier::ShapeSys {
                            name: format!("shapesys_{ch_name}_{sample_name}"),
                            parameters: param_names,
                            constraint_type: Some("Poisson".to_string()),
                            data: Some(Hs3ShapeSysData { vals: uncertainties.clone() }),
                        });
                    }
                    ModelModifier::ShapeFactor { param_indices } => {
                        let param_names: Vec<String> =
                            param_indices.iter().map(|&idx| params[idx].name.clone()).collect();
                        modifiers.push(Hs3Modifier::ShapeFactor {
                            name: format!("shapefactor_{ch_name}_{sample_name}"),
                            parameters: Some(param_names),
                            parameter: None,
                        });
                    }
                }
            }

            hs3_samples.push(Hs3Sample {
                name: sample_name.clone(),
                data: Hs3SampleData { contents: nominal, errors: sample_errors },
                modifiers,
            });
        }

        distributions.push(Hs3Distribution::HistFactory(Hs3HistFactoryDist {
            name: ch_name.clone(),
            dist_type: "histfactory_dist".to_string(),
            axes: vec![Hs3Axis {
                name: format!("obs_x_{ch_name}"),
                min: Some(0.0),
                max: Some(n_bins as f64),
                nbins: Some(n_bins),
                edges: None,
            }],
            samples: hs3_samples,
        }));
    }

    // Gaussian constraint distributions for constrained NPs
    for (pidx, param) in params.iter().enumerate() {
        if param.constrained && constraint_params_seen.contains(&pidx) {
            let center = param.constraint_center.unwrap_or(0.0);
            let width = param.constraint_width.unwrap_or(1.0);

            // Skip gamma (staterror) parameters — they use Poisson, not Gaussian
            if center == 1.0 && param.name.starts_with("gamma_") {
                continue;
            }

            let glob_obs_name = format!("nom_{}", param.name);
            distributions.push(Hs3Distribution::Gaussian(Hs3GaussianDist {
                name: format!("{}Constraint", param.name),
                dist_type: "gaussian_dist".to_string(),
                x: param.name.clone(),
                mean: glob_obs_name,
                sigma: width,
            }));
        }
    }

    // -----------------------------------------------------------------------
    // Build data
    // -----------------------------------------------------------------------
    let observed_data = model.observed_main_by_channel();
    let data: Vec<Hs3Data> = observed_data
        .iter()
        .map(|ch| Hs3Data {
            name: format!("obsData_{}", ch.channel_name),
            data_type: "binned".to_string(),
            axes: vec![],
            contents: ch.y.clone(),
        })
        .collect();

    // -----------------------------------------------------------------------
    // Build domains
    // -----------------------------------------------------------------------
    let mut np_axes: Vec<Hs3DomainAxis> = Vec::new();
    let mut poi_axes: Vec<Hs3DomainAxis> = Vec::new();
    let mut glob_obs_axes: Vec<Hs3DomainAxis> = Vec::new();

    let poi_index = model.poi_index();
    for (idx, param) in params.iter().enumerate() {
        let axis =
            Hs3DomainAxis { name: param.name.clone(), min: param.bounds.0, max: param.bounds.1 };

        if Some(idx) == poi_index {
            poi_axes.push(axis);
        } else if param.constrained {
            np_axes.push(axis);
            // Global observable for this NP
            let center = param.constraint_center.unwrap_or(0.0);
            glob_obs_axes.push(Hs3DomainAxis {
                name: format!("nom_{}", param.name),
                min: center - 10.0,
                max: center + 10.0,
            });
        } else {
            np_axes.push(axis);
        }
    }

    let domains = vec![
        Hs3Domain {
            name: format!("{analysis_name}_nuisance_parameters"),
            domain_type: "product_domain".to_string(),
            axes: np_axes,
        },
        Hs3Domain {
            name: format!("{analysis_name}_global_observables"),
            domain_type: "product_domain".to_string(),
            axes: glob_obs_axes,
        },
        Hs3Domain {
            name: format!("{analysis_name}_parameters_of_interest"),
            domain_type: "product_domain".to_string(),
            axes: poi_axes,
        },
    ];

    // -----------------------------------------------------------------------
    // Build parameter_points
    // -----------------------------------------------------------------------
    let mut parameter_points = vec![Hs3ParameterPointSet {
        name: "default_values".to_string(),
        parameters: params
            .iter()
            .map(|p| Hs3ParameterValue { name: p.name.clone(), value: p.init })
            .chain(
                // Add global observables
                params.iter().filter(|p| p.constrained).map(|p| Hs3ParameterValue {
                    name: format!("nom_{}", p.name),
                    value: p.constraint_center.unwrap_or(0.0),
                }),
            )
            .collect(),
    }];

    if let Some((name, values)) = bestfit_params {
        let pp: Vec<Hs3ParameterValue> = params
            .iter()
            .zip(values.iter())
            .map(|(p, &v)| Hs3ParameterValue { name: p.name.clone(), value: v })
            .collect();
        parameter_points.push(Hs3ParameterPointSet { name: name.to_string(), parameters: pp });
    }

    // -----------------------------------------------------------------------
    // Build analyses and likelihoods
    // -----------------------------------------------------------------------
    let dist_names: Vec<String> = model.channel_names().to_vec();

    let data_names: Vec<String> = dist_names.iter().map(|n| format!("obsData_{n}")).collect();

    let likelihood_name = analysis_name.to_string();

    let likelihoods = vec![Hs3Likelihood {
        name: likelihood_name.clone(),
        distributions: dist_names,
        data: data_names,
    }];

    let pois: Vec<String> =
        if let Some(poi_idx) = poi_index { vec![params[poi_idx].name.clone()] } else { vec![] };

    let analyses = vec![Hs3Analysis {
        name: analysis_name.to_string(),
        likelihood: likelihood_name,
        parameters_of_interest: pois,
        domains: domains.iter().map(|d| d.name.clone()).collect(),
    }];

    // -----------------------------------------------------------------------
    // Metadata
    // -----------------------------------------------------------------------
    let metadata = original_hs3.map(|ws| ws.metadata.clone()).unwrap_or(Hs3Metadata {
        hs3_version: "0.2".to_string(),
        packages: Some(vec![Hs3Package {
            name: "NextStat".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }]),
    });

    let misc = original_hs3.and_then(|ws| ws.misc.clone());

    Hs3Workspace {
        distributions,
        data,
        domains,
        parameter_points,
        analyses,
        likelihoods,
        metadata,
        misc,
    }
}

/// Convenience: export to JSON string.
pub fn export_hs3_json(
    model: &HistFactoryModel,
    analysis_name: &str,
    bestfit_params: Option<(&str, &[f64])>,
    original_hs3: Option<&Hs3Workspace>,
) -> Result<String, serde_json::Error> {
    let ws = export_hs3(model, analysis_name, bestfit_params, original_hs3);
    serde_json::to_string_pretty(&ws)
}
