//! HS3 Export — serialize a `HistFactoryModel` back to HS3 JSON.
//!
//! Enables roundtrip workflows: load HS3 → fit → export updated HS3 with
//! bestfit parameter points.

use super::schema::*;
use crate::pyhf::model::{HistFactoryModel, ModelModifier};
#[cfg(feature = "unbinned")]
use ns_core::traits::PoiModel;
#[cfg(feature = "unbinned")]
use std::collections::HashMap;
use std::collections::HashSet;
#[cfg(feature = "unbinned")]
use std::path::PathBuf;

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

// ---------------------------------------------------------------------------
// Unbinned (event-level) HS3 import/export — NextStat extension
// ---------------------------------------------------------------------------

#[cfg(feature = "unbinned")]
use ns_unbinned::UnbinnedModel as RustUnbinnedModel;
#[cfg(feature = "unbinned")]
use ns_unbinned::spec::{
    ChannelSpec, ConstraintSpec, DataFormat, DataSpec, ModelSpec, ObservableSpecV0, ParameterSpec,
    PdfSpec, ProcessSpec, RateModifierSpec, UNBINNED_SPEC_V0, UnbinnedSpecV0, YieldSpec,
};

#[cfg(feature = "unbinned")]
#[derive(Debug, thiserror::Error)]
pub enum Hs3UnbinnedImportError {
    #[error("analysis not found: {0}")]
    AnalysisNotFound(String),
    #[error("likelihood not found: {0}")]
    LikelihoodNotFound(String),
    #[error("parameter point set not found: {0}")]
    ParamPointSetNotFound(String),
    #[error("no unbinned distributions found in selected analysis")]
    NoUnbinnedDistributions,
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(feature = "unbinned")]
#[derive(Debug, thiserror::Error)]
pub enum Hs3HybridImportError {
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("binned HS3 import failed: {0}")]
    Binned(#[from] super::convert::Hs3ConvertError),
    #[error("unbinned HS3 import failed: {0}")]
    Unbinned(#[from] Hs3UnbinnedImportError),
}

#[cfg(feature = "unbinned")]
#[derive(Debug)]
pub struct ImportedHybridHs3 {
    pub binned: HistFactoryModel,
    pub unbinned_spec: UnbinnedSpecV0,
    pub shared_param_names: Vec<String>,
}

#[cfg(feature = "unbinned")]
#[derive(Debug, Clone)]
struct GaussianConstraintSpec {
    mean_param: String,
    sigma: f64,
}

#[cfg(feature = "unbinned")]
/// Import an unbinned spec from an HS3 workspace containing
/// `nextstat_unbinned_dist` distributions.
pub fn import_unbinned_hs3(
    ws: &Hs3Workspace,
    analysis_name: Option<&str>,
    param_point_set: Option<&str>,
) -> Result<UnbinnedSpecV0, Hs3UnbinnedImportError> {
    let analysis = select_analysis(ws, analysis_name)?;
    let likelihood =
        ws.likelihoods.iter().find(|l| l.name == analysis.likelihood).ok_or_else(|| {
            Hs3UnbinnedImportError::LikelihoodNotFound(analysis.likelihood.clone())
        })?;

    let mut dist_map: HashMap<&str, &Hs3Distribution> = HashMap::new();
    for dist in &ws.distributions {
        if let Some(name) = dist.name() {
            dist_map.insert(name, dist);
        }
    }

    let mut selected_unbinned: Vec<&Hs3UnbinnedDist> = Vec::new();
    for dist_name in &likelihood.distributions {
        if let Some(dist) = dist_map.get(dist_name.as_str())
            && let Hs3Distribution::Unbinned(d) = *dist
        {
            selected_unbinned.push(d);
        }
    }

    if selected_unbinned.is_empty() {
        selected_unbinned = ws
            .distributions
            .iter()
            .filter_map(|d| match d {
                Hs3Distribution::Unbinned(u) => Some(u),
                _ => None,
            })
            .collect();
    }

    if selected_unbinned.is_empty() {
        return Err(Hs3UnbinnedImportError::NoUnbinnedDistributions);
    }

    let param_points = select_param_points(ws, param_point_set)?;
    let mut init_map: HashMap<&str, f64> = HashMap::new();
    for pv in &param_points.parameters {
        init_map.insert(pv.name.as_str(), pv.value);
    }

    let mut bounds_map: HashMap<&str, (f64, f64)> = HashMap::new();
    for domain_name in &analysis.domains {
        if let Some(domain) = ws.domains.iter().find(|d| d.name == *domain_name) {
            for axis in &domain.axes {
                bounds_map.insert(axis.name.as_str(), (axis.min, axis.max));
            }
        }
    }

    let mut constraints: HashMap<String, GaussianConstraintSpec> = HashMap::new();
    for dist in &ws.distributions {
        if let Hs3Distribution::Gaussian(g) = dist {
            constraints.insert(
                g.x.clone(),
                GaussianConstraintSpec { mean_param: g.mean.clone(), sigma: g.sigma },
            );
        }
    }

    let mut channels: Vec<ChannelSpec> = Vec::new();
    let mut ordered_params: Vec<String> = Vec::new();
    let mut seen_params: HashSet<String> = HashSet::new();
    let mut yield_like_params: HashSet<String> = HashSet::new();

    for dist in selected_unbinned {
        let channel = hs3_unbinned_dist_to_spec(
            dist,
            &mut ordered_params,
            &mut seen_params,
            &mut yield_like_params,
        );
        channels.push(channel);
    }

    if ordered_params.is_empty() {
        ordered_params.push("mu".to_string());
        seen_params.insert("mu".to_string());
    }

    let mut parameters: Vec<ParameterSpec> = Vec::with_capacity(ordered_params.len());
    for name in ordered_params {
        let constraint = constraints.get(&name);
        let mean =
            constraint.and_then(|gc| init_map.get(gc.mean_param.as_str()).copied()).unwrap_or(0.0);
        let bounds = bounds_map.get(name.as_str()).copied().unwrap_or_else(|| {
            if yield_like_params.contains(&name) {
                (0.0, 10.0)
            } else if let Some(gc) = constraint {
                (mean - 5.0 * gc.sigma, mean + 5.0 * gc.sigma)
            } else {
                (-5.0, 5.0)
            }
        });
        let init = init_map
            .get(name.as_str())
            .copied()
            .unwrap_or_else(|| if yield_like_params.contains(&name) { 1.0 } else { mean });
        let constraint = constraint.map(|gc| ConstraintSpec::Gaussian { mean, sigma: gc.sigma });

        parameters.push(ParameterSpec { name, init, bounds: [bounds.0, bounds.1], constraint });
    }

    let poi = analysis
        .parameters_of_interest
        .iter()
        .find(|p| parameters.iter().any(|q| q.name == **p))
        .cloned()
        .or_else(|| parameters.first().map(|p| p.name.clone()));

    Ok(UnbinnedSpecV0 {
        schema_uri: None,
        schema_version: UNBINNED_SPEC_V0.to_string(),
        model: ModelSpec { poi, parameters },
        channels,
    })
}

#[cfg(feature = "unbinned")]
/// Import an unbinned spec from HS3 JSON.
pub fn import_unbinned_hs3_json(
    json: &str,
    analysis_name: Option<&str>,
    param_point_set: Option<&str>,
) -> Result<UnbinnedSpecV0, Hs3UnbinnedImportError> {
    let ws: Hs3Workspace = serde_json::from_str(json)?;
    import_unbinned_hs3(&ws, analysis_name, param_point_set)
}

#[cfg(feature = "unbinned")]
/// Import a hybrid HS3 workspace into binned HistFactory + unbinned spec pieces.
pub fn import_hybrid_hs3(
    ws: &Hs3Workspace,
    analysis_name: Option<&str>,
    param_point_set: Option<&str>,
) -> Result<ImportedHybridHs3, Hs3HybridImportError> {
    let json = serde_json::to_string(ws)?;
    let binned = super::convert::from_hs3(
        &json,
        analysis_name,
        param_point_set,
        crate::pyhf::model::NormSysInterpCode::Code1,
        crate::pyhf::model::HistoSysInterpCode::Code0,
    )?;
    let unbinned_spec = import_unbinned_hs3(ws, analysis_name, param_point_set)?;

    let binned_names: HashSet<&str> = binned.parameters().iter().map(|p| p.name.as_str()).collect();
    let shared_param_names: Vec<String> = unbinned_spec
        .model
        .parameters
        .iter()
        .filter(|p| binned_names.contains(p.name.as_str()))
        .map(|p| p.name.clone())
        .collect();

    Ok(ImportedHybridHs3 { binned, unbinned_spec, shared_param_names })
}

#[cfg(feature = "unbinned")]
/// Import a hybrid HS3 workspace from JSON.
pub fn import_hybrid_hs3_json(
    json: &str,
    analysis_name: Option<&str>,
    param_point_set: Option<&str>,
) -> Result<ImportedHybridHs3, Hs3HybridImportError> {
    let ws: Hs3Workspace = serde_json::from_str(json)?;
    import_hybrid_hs3(&ws, analysis_name, param_point_set)
}

#[cfg(feature = "unbinned")]
fn select_analysis<'a>(
    ws: &'a Hs3Workspace,
    analysis_name: Option<&str>,
) -> Result<&'a Hs3Analysis, Hs3UnbinnedImportError> {
    match analysis_name {
        Some(name) => ws
            .analyses
            .iter()
            .find(|a| a.name == name)
            .ok_or_else(|| Hs3UnbinnedImportError::AnalysisNotFound(name.to_string())),
        None => ws
            .analyses
            .first()
            .ok_or_else(|| Hs3UnbinnedImportError::AnalysisNotFound("(empty)".to_string())),
    }
}

#[cfg(feature = "unbinned")]
fn select_param_points<'a>(
    ws: &'a Hs3Workspace,
    set_name: Option<&str>,
) -> Result<&'a Hs3ParameterPointSet, Hs3UnbinnedImportError> {
    if let Some(name) = set_name {
        return ws
            .parameter_points
            .iter()
            .find(|pp| pp.name == name)
            .ok_or_else(|| Hs3UnbinnedImportError::ParamPointSetNotFound(name.to_string()));
    }
    ws.parameter_points
        .iter()
        .find(|pp| pp.name == "default_values")
        .or_else(|| ws.parameter_points.first())
        .ok_or_else(|| Hs3UnbinnedImportError::ParamPointSetNotFound("default_values".to_string()))
}

#[cfg(feature = "unbinned")]
fn remember_param(name: &str, ordered: &mut Vec<String>, seen: &mut HashSet<String>) {
    if seen.insert(name.to_string()) {
        ordered.push(name.to_string());
    }
}

#[cfg(feature = "unbinned")]
fn hs3_unbinned_dist_to_spec(
    dist: &Hs3UnbinnedDist,
    ordered_params: &mut Vec<String>,
    seen_params: &mut HashSet<String>,
    yield_like_params: &mut HashSet<String>,
) -> ChannelSpec {
    let observables: Vec<ObservableSpecV0> = dist
        .observables
        .iter()
        .map(|o| ObservableSpecV0 {
            name: o.name.clone(),
            expr: o.expr.clone(),
            bounds: [o.min, o.max],
        })
        .collect();

    let mut observable_bounds: HashMap<String, (f64, f64)> = HashMap::new();
    for o in &dist.observables {
        observable_bounds.insert(o.name.clone(), (o.min, o.max));
    }

    let data = if let Some(src) = &dist.data_source {
        let format = src.format.as_deref().map(str::to_ascii_lowercase);
        match format.as_deref() {
            Some("parquet") => DataSpec {
                file: PathBuf::from(&src.file),
                tree: None,
                channel: None,
                selection: src.selection.clone(),
                weight: src.weight.clone(),
                format: Some(DataFormat::Parquet),
            },
            _ => DataSpec {
                file: PathBuf::from(&src.file),
                tree: Some(src.tree.clone().unwrap_or_else(|| "Events".to_string())),
                channel: None,
                selection: src.selection.clone(),
                weight: src.weight.clone(),
                format: Some(DataFormat::Root),
            },
        }
    } else {
        DataSpec {
            file: PathBuf::from("events.root"),
            tree: Some("Events".to_string()),
            channel: None,
            selection: None,
            weight: None,
            format: Some(DataFormat::Root),
        }
    };

    let processes: Vec<ProcessSpec> = dist
        .processes
        .iter()
        .map(|proc| {
            remember_pdf_params(&proc.pdf, ordered_params, seen_params);
            let yield_spec = hs3_yield_to_spec(
                &proc.yield_spec,
                ordered_params,
                seen_params,
                yield_like_params,
                &dist.name,
                &proc.name,
            );
            ProcessSpec {
                name: proc.name.clone(),
                pdf: hs3_pdf_to_spec(&proc.pdf, &observable_bounds, &dist.name, &proc.name),
                yield_spec,
            }
        })
        .collect();

    ChannelSpec { name: dist.name.clone(), include_in_fit: true, data, observables, processes }
}

#[cfg(feature = "unbinned")]
fn remember_pdf_params(
    pdf: &Hs3UnbinnedPdf,
    ordered: &mut Vec<String>,
    seen: &mut HashSet<String>,
) {
    for p in &pdf.params {
        remember_param(p, ordered, seen);
    }
    if let Some(params) = &pdf.context_params {
        for p in params {
            remember_param(p, ordered, seen);
        }
    }
    if let Some(systematics) = &pdf.systematics {
        for p in systematics {
            remember_param(p, ordered, seen);
        }
    }
    if let Some(components) = &pdf.components {
        for c in components {
            remember_pdf_params(c, ordered, seen);
        }
    }
}

#[cfg(feature = "unbinned")]
fn hs3_yield_to_spec(
    y: &Hs3UnbinnedYield,
    ordered_params: &mut Vec<String>,
    seen_params: &mut HashSet<String>,
    yield_like_params: &mut HashSet<String>,
    channel_name: &str,
    process_name: &str,
) -> YieldSpec {
    let modifiers: Vec<RateModifierSpec> = y
        .modifiers
        .iter()
        .filter_map(|m| {
            remember_param(&m.param, ordered_params, seen_params);
            let mod_type = m.mod_type.to_ascii_lowercase();
            if mod_type == "normsys" {
                Some(RateModifierSpec::NormSys { param: m.param.clone(), lo: m.lo, hi: m.hi })
            } else if mod_type == "weightsys" {
                Some(RateModifierSpec::WeightSys {
                    param: m.param.clone(),
                    lo: m.lo,
                    hi: m.hi,
                    interp_code: None,
                })
            } else {
                log::warn!(
                    "HS3 unbinned modifier '{}' in channel '{}' process '{}' is unsupported; skipping",
                    m.mod_type,
                    channel_name,
                    process_name
                );
                None
            }
        })
        .collect();

    match y.yield_type.to_ascii_lowercase().as_str() {
        "fixed" => YieldSpec::Fixed { value: y.value.unwrap_or(0.0), modifiers },
        "parameter" => {
            let name = y.parameter.clone().unwrap_or_else(|| "yield".to_string());
            remember_param(&name, ordered_params, seen_params);
            yield_like_params.insert(name.clone());
            YieldSpec::Parameter { name, modifiers }
        }
        "scaled" => {
            let scale = y.scale.clone().unwrap_or_else(|| "mu".to_string());
            remember_param(&scale, ordered_params, seen_params);
            yield_like_params.insert(scale.clone());
            YieldSpec::Scaled { base_yield: y.base_yield.unwrap_or(1.0), scale, modifiers }
        }
        other => {
            log::warn!(
                "HS3 unbinned yield type '{}' in channel '{}' process '{}' is unsupported; using fixed=1",
                other,
                channel_name,
                process_name
            );
            YieldSpec::Fixed { value: 1.0, modifiers }
        }
    }
}

#[cfg(feature = "unbinned")]
fn hs3_pdf_to_spec(
    pdf: &Hs3UnbinnedPdf,
    observable_bounds: &HashMap<String, (f64, f64)>,
    channel_name: &str,
    process_name: &str,
) -> PdfSpec {
    let observable = pick_observable(pdf, observable_bounds);
    let make_fallback = || fallback_histogram(pdf, observable_bounds);

    match pdf.pdf_type.to_ascii_lowercase().as_str() {
        "gaussian" => PdfSpec::Gaussian { observable, params: pdf.params.clone() },
        "crystal_ball" => PdfSpec::CrystalBall { observable, params: pdf.params.clone() },
        "double_crystal_ball" => {
            PdfSpec::DoubleCrystalBall { observable, params: pdf.params.clone() }
        }
        "exponential" => PdfSpec::Exponential { observable, params: pdf.params.clone() },
        "argus" => PdfSpec::Argus { observable, params: pdf.params.clone() },
        "voigtian" => PdfSpec::Voigtian { observable, params: pdf.params.clone() },
        "chebyshev" => PdfSpec::Chebyshev { observable, params: pdf.params.clone() },
        "histogram" => {
            let edges = pdf.bin_edges.clone();
            let content = pdf.bin_content.clone();
            match (edges, content) {
                (Some(bin_edges), Some(bin_content))
                    if bin_edges.len() == bin_content.len().saturating_add(1)
                        && !bin_content.is_empty() =>
                {
                    PdfSpec::Histogram { observable, bin_edges, bin_content, pseudo_count: None }
                }
                _ => {
                    log::warn!(
                        "HS3 histogram PDF in channel '{}' process '{}' has invalid bins; using fallback",
                        channel_name,
                        process_name
                    );
                    make_fallback()
                }
            }
        }
        "kde" => {
            let (lo, hi) = observable_bounds.get(&observable).copied().unwrap_or((0.0, 1.0));
            let centers = pdf.centers.clone().unwrap_or_else(|| vec![(lo + hi) * 0.5]);
            PdfSpec::Kde {
                observable,
                bandwidth: pdf.bandwidth.unwrap_or(1.0),
                centers,
                weights: pdf.weights.clone(),
            }
        }
        "spline" => {
            if let (Some(knots_x), Some(knots_y)) = (pdf.knots_x.clone(), pdf.knots_y.clone())
                && knots_x.len() == knots_y.len()
                && knots_x.len() >= 2
            {
                PdfSpec::Spline { observable, knots_x, knots_y }
            } else {
                log::warn!(
                    "HS3 spline PDF in channel '{}' process '{}' has invalid knots; using fallback",
                    channel_name,
                    process_name
                );
                make_fallback()
            }
        }
        "product" => {
            let components = pdf
                .components
                .clone()
                .unwrap_or_default()
                .iter()
                .map(|c| hs3_pdf_to_spec(c, observable_bounds, channel_name, process_name))
                .collect::<Vec<_>>();
            if components.is_empty() {
                log::warn!(
                    "HS3 product PDF in channel '{}' process '{}' has no components; using fallback",
                    channel_name,
                    process_name
                );
                make_fallback()
            } else {
                PdfSpec::Product { components }
            }
        }
        "flow" => {
            if let Some(manifest) = &pdf.manifest {
                PdfSpec::Flow { manifest: PathBuf::from(manifest) }
            } else {
                log::warn!(
                    "HS3 flow PDF in channel '{}' process '{}' has no manifest; using fallback",
                    channel_name,
                    process_name
                );
                make_fallback()
            }
        }
        "conditional_flow" => {
            if let Some(manifest) = &pdf.manifest {
                PdfSpec::ConditionalFlow {
                    manifest: PathBuf::from(manifest),
                    context_params: pdf.context_params.clone().unwrap_or_default(),
                }
            } else {
                log::warn!(
                    "HS3 conditional_flow PDF in channel '{}' process '{}' has no manifest; using fallback",
                    channel_name,
                    process_name
                );
                make_fallback()
            }
        }
        "dcr_surrogate" => {
            if let Some(manifest) = &pdf.manifest {
                PdfSpec::DcrSurrogate {
                    manifest: PathBuf::from(manifest),
                    systematics: pdf.systematics.clone().unwrap_or_default(),
                }
            } else {
                log::warn!(
                    "HS3 dcr_surrogate PDF in channel '{}' process '{}' has no manifest; using fallback",
                    channel_name,
                    process_name
                );
                make_fallback()
            }
        }
        _ => {
            log::warn!(
                "HS3 PDF type '{}' in channel '{}' process '{}' is not directly representable in spec; using fallback",
                pdf.pdf_type,
                channel_name,
                process_name
            );
            make_fallback()
        }
    }
}

#[cfg(feature = "unbinned")]
fn pick_observable(
    pdf: &Hs3UnbinnedPdf,
    observable_bounds: &HashMap<String, (f64, f64)>,
) -> String {
    if let Some(name) = &pdf.observable {
        return name.clone();
    }
    observable_bounds.keys().next().cloned().unwrap_or_else(|| "x".to_string())
}

#[cfg(feature = "unbinned")]
fn fallback_histogram(
    pdf: &Hs3UnbinnedPdf,
    observable_bounds: &HashMap<String, (f64, f64)>,
) -> PdfSpec {
    let observable = pick_observable(pdf, observable_bounds);
    let (mut lo, mut hi) = observable_bounds.get(&observable).copied().unwrap_or((0.0, 1.0));
    if !lo.is_finite() || !hi.is_finite() || hi <= lo {
        lo = 0.0;
        hi = 1.0;
    }
    PdfSpec::Histogram {
        observable,
        bin_edges: vec![lo, hi],
        bin_content: vec![1.0],
        pseudo_count: None,
    }
}

#[cfg(feature = "unbinned")]
fn spec_pdf_to_hs3(pdf: &PdfSpec) -> Hs3UnbinnedPdf {
    match pdf {
        PdfSpec::Gaussian { observable, params } => Hs3UnbinnedPdf {
            pdf_type: "gaussian".into(),
            observable: Some(observable.clone()),
            params: params.clone(),
            ..Default::default()
        },
        PdfSpec::CrystalBall { observable, params } => Hs3UnbinnedPdf {
            pdf_type: "crystal_ball".into(),
            observable: Some(observable.clone()),
            params: params.clone(),
            ..Default::default()
        },
        PdfSpec::DoubleCrystalBall { observable, params } => Hs3UnbinnedPdf {
            pdf_type: "double_crystal_ball".into(),
            observable: Some(observable.clone()),
            params: params.clone(),
            ..Default::default()
        },
        PdfSpec::Exponential { observable, params } => Hs3UnbinnedPdf {
            pdf_type: "exponential".into(),
            observable: Some(observable.clone()),
            params: params.clone(),
            ..Default::default()
        },
        PdfSpec::Argus { observable, params } => Hs3UnbinnedPdf {
            pdf_type: "argus".into(),
            observable: Some(observable.clone()),
            params: params.clone(),
            ..Default::default()
        },
        PdfSpec::Voigtian { observable, params } => Hs3UnbinnedPdf {
            pdf_type: "voigtian".into(),
            observable: Some(observable.clone()),
            params: params.clone(),
            ..Default::default()
        },
        PdfSpec::Chebyshev { observable, params } => Hs3UnbinnedPdf {
            pdf_type: "chebyshev".into(),
            observable: Some(observable.clone()),
            params: params.clone(),
            ..Default::default()
        },
        PdfSpec::Histogram { observable, bin_edges, bin_content, .. } => Hs3UnbinnedPdf {
            pdf_type: "histogram".into(),
            observable: Some(observable.clone()),
            bin_edges: Some(bin_edges.clone()),
            bin_content: Some(bin_content.clone()),
            ..Default::default()
        },
        PdfSpec::Kde { observable, bandwidth, centers, weights } => Hs3UnbinnedPdf {
            pdf_type: "kde".into(),
            observable: Some(observable.clone()),
            bandwidth: Some(*bandwidth),
            centers: Some(centers.clone()),
            weights: weights.clone(),
            ..Default::default()
        },
        PdfSpec::Spline { observable, knots_x, knots_y } => Hs3UnbinnedPdf {
            pdf_type: "spline".into(),
            observable: Some(observable.clone()),
            knots_x: Some(knots_x.clone()),
            knots_y: Some(knots_y.clone()),
            ..Default::default()
        },
        PdfSpec::Product { components } => Hs3UnbinnedPdf {
            pdf_type: "product".into(),
            components: Some(components.iter().map(spec_pdf_to_hs3).collect()),
            ..Default::default()
        },
        PdfSpec::Flow { manifest } => Hs3UnbinnedPdf {
            pdf_type: "flow".into(),
            manifest: Some(manifest.display().to_string()),
            ..Default::default()
        },
        PdfSpec::ConditionalFlow { manifest, context_params } => Hs3UnbinnedPdf {
            pdf_type: "conditional_flow".into(),
            manifest: Some(manifest.display().to_string()),
            context_params: Some(context_params.clone()),
            ..Default::default()
        },
        PdfSpec::DcrSurrogate { manifest, systematics } => Hs3UnbinnedPdf {
            pdf_type: "dcr_surrogate".into(),
            manifest: Some(manifest.display().to_string()),
            systematics: Some(systematics.clone()),
            ..Default::default()
        },
        PdfSpec::HistogramFromTree { observable, bin_edges, .. } => Hs3UnbinnedPdf {
            pdf_type: "histogram_from_tree".into(),
            observable: Some(observable.clone()),
            bin_edges: Some(bin_edges.clone()),
            ..Default::default()
        },
        PdfSpec::KdeFromTree { observable, bandwidth, .. } => Hs3UnbinnedPdf {
            pdf_type: "kde_from_tree".into(),
            observable: Some(observable.clone()),
            bandwidth: Some(*bandwidth),
            ..Default::default()
        },
    }
}

#[cfg(feature = "unbinned")]
fn spec_yield_to_hs3(y: &YieldSpec) -> Hs3UnbinnedYield {
    match y {
        YieldSpec::Fixed { value, modifiers } => Hs3UnbinnedYield {
            yield_type: "fixed".into(),
            value: Some(*value),
            parameter: None,
            base_yield: None,
            scale: None,
            modifiers: modifiers.iter().map(spec_rate_mod_to_hs3).collect(),
        },
        YieldSpec::Parameter { name, modifiers } => Hs3UnbinnedYield {
            yield_type: "parameter".into(),
            value: None,
            parameter: Some(name.clone()),
            base_yield: None,
            scale: None,
            modifiers: modifiers.iter().map(spec_rate_mod_to_hs3).collect(),
        },
        YieldSpec::Scaled { base_yield, scale, modifiers } => Hs3UnbinnedYield {
            yield_type: "scaled".into(),
            value: None,
            parameter: None,
            base_yield: Some(*base_yield),
            scale: Some(scale.clone()),
            modifiers: modifiers.iter().map(spec_rate_mod_to_hs3).collect(),
        },
    }
}

#[cfg(feature = "unbinned")]
fn spec_rate_mod_to_hs3(m: &RateModifierSpec) -> Hs3UnbinnedRateModifier {
    match m {
        RateModifierSpec::NormSys { param, lo, hi } => Hs3UnbinnedRateModifier {
            mod_type: "normsys".into(),
            param: param.clone(),
            lo: *lo,
            hi: *hi,
        },
        RateModifierSpec::WeightSys { param, lo, hi, .. } => Hs3UnbinnedRateModifier {
            mod_type: "weightsys".into(),
            param: param.clone(),
            lo: *lo,
            hi: *hi,
        },
    }
}

#[cfg(feature = "unbinned")]
fn spec_channel_to_hs3_dist(ch: &ChannelSpec) -> Hs3UnbinnedDist {
    let observables: Vec<Hs3UnbinnedObservable> = ch
        .observables
        .iter()
        .map(|o| Hs3UnbinnedObservable {
            name: o.name.clone(),
            min: o.bounds[0],
            max: o.bounds[1],
            expr: o.expr.clone(),
        })
        .collect();

    let processes: Vec<Hs3UnbinnedProcess> = ch
        .processes
        .iter()
        .map(|p| Hs3UnbinnedProcess {
            name: p.name.clone(),
            pdf: spec_pdf_to_hs3(&p.pdf),
            yield_spec: spec_yield_to_hs3(&p.yield_spec),
        })
        .collect();

    let data_source = Some(Hs3UnbinnedDataSource {
        file: ch.data.file.display().to_string(),
        format: ch.data.format.map(|f| match f {
            ns_unbinned::spec::DataFormat::Root => "root".into(),
            ns_unbinned::spec::DataFormat::Parquet => "parquet".into(),
        }),
        tree: ch.data.tree.clone(),
        selection: ch.data.selection.clone(),
        weight: ch.data.weight.clone(),
    });

    Hs3UnbinnedDist {
        name: ch.name.clone(),
        dist_type: "nextstat_unbinned_dist".into(),
        observables,
        processes,
        data_source,
    }
}

#[cfg(feature = "unbinned")]
/// Export an `UnbinnedModel` (from spec) to an HS3 workspace.
///
/// Uses the `UnbinnedSpecV0` for PDF/channel structure and the compiled
/// `UnbinnedModel` for parameter metadata (bounds, inits, constraints).
pub fn export_unbinned_hs3(
    spec: &UnbinnedSpecV0,
    model: &RustUnbinnedModel,
    analysis_name: &str,
    bestfit_params: Option<(&str, &[f64])>,
) -> Hs3Workspace {
    let params = model.parameters();

    let mut distributions: Vec<Hs3Distribution> = Vec::new();

    for ch in &spec.channels {
        distributions.push(Hs3Distribution::Unbinned(spec_channel_to_hs3_dist(ch)));
    }

    // Gaussian constraints
    for p in params {
        if let Some(ns_unbinned::Constraint::Gaussian { mean: _, sigma }) = &p.constraint {
            let glob_obs_name = format!("nom_{}", p.name);
            distributions.push(Hs3Distribution::Gaussian(Hs3GaussianDist {
                name: format!("{}Constraint", p.name),
                dist_type: "gaussian_dist".into(),
                x: p.name.clone(),
                mean: glob_obs_name,
                sigma: *sigma,
            }));
        }
    }

    // Data entries (by design, unbinned data lives out-of-band and is referenced by `data_source`).
    let data: Vec<Hs3Data> = spec
        .channels
        .iter()
        .map(|ch| Hs3Data {
            name: format!("obsData_{}", ch.name),
            data_type: "unbinned_ref".into(),
            axes: vec![],
            contents: vec![],
        })
        .collect();

    // Domains
    let poi_index = model.poi_index();
    let mut np_axes = Vec::new();
    let mut poi_axes = Vec::new();
    let mut glob_obs_axes = Vec::new();

    for (idx, p) in params.iter().enumerate() {
        let axis = Hs3DomainAxis { name: p.name.clone(), min: p.bounds.0, max: p.bounds.1 };
        if Some(idx) == poi_index {
            poi_axes.push(axis);
        } else {
            np_axes.push(axis);
        }
        if let Some(ns_unbinned::Constraint::Gaussian { mean, .. }) = &p.constraint {
            glob_obs_axes.push(Hs3DomainAxis {
                name: format!("nom_{}", p.name),
                min: mean - 10.0,
                max: mean + 10.0,
            });
        }
    }

    let domains = vec![
        Hs3Domain {
            name: format!("{analysis_name}_nuisance_parameters"),
            domain_type: "product_domain".into(),
            axes: np_axes,
        },
        Hs3Domain {
            name: format!("{analysis_name}_global_observables"),
            domain_type: "product_domain".into(),
            axes: glob_obs_axes,
        },
        Hs3Domain {
            name: format!("{analysis_name}_parameters_of_interest"),
            domain_type: "product_domain".into(),
            axes: poi_axes,
        },
    ];

    // Parameter points
    let mut parameter_points = vec![Hs3ParameterPointSet {
        name: "default_values".into(),
        parameters: params
            .iter()
            .map(|p| Hs3ParameterValue { name: p.name.clone(), value: p.init })
            .chain(params.iter().filter_map(|p| {
                if let Some(ns_unbinned::Constraint::Gaussian { mean, .. }) = &p.constraint {
                    Some(Hs3ParameterValue { name: format!("nom_{}", p.name), value: *mean })
                } else {
                    None
                }
            }))
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

    // Analyses + likelihoods
    let dist_names: Vec<String> = spec.channels.iter().map(|ch| ch.name.clone()).collect();
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

    Hs3Workspace {
        distributions,
        data,
        domains,
        parameter_points,
        analyses,
        likelihoods,
        metadata: Hs3Metadata {
            hs3_version: "0.2".into(),
            packages: Some(vec![Hs3Package {
                name: "NextStat".into(),
                version: env!("CARGO_PKG_VERSION").into(),
            }]),
        },
        misc: Some(serde_json::json!({
            "nextstat_extensions": ["unbinned_dist"],
            "unbinned_spec_version": spec.schema_version,
        })),
    }
}

#[cfg(feature = "unbinned")]
/// Export a hybrid (binned + unbinned) model to an HS3 workspace.
///
/// Combines a binned `HistFactoryModel` and an unbinned spec/model into a
/// single HS3 workspace with both `histfactory_dist` and `nextstat_unbinned_dist`
/// distributions sharing the same parameter space.
pub fn export_hybrid_hs3(
    binned: &HistFactoryModel,
    unbinned_spec: &UnbinnedSpecV0,
    unbinned_model: &RustUnbinnedModel,
    analysis_name: &str,
    bestfit_params: Option<(&str, &[f64])>,
    shared_param_names: &[String],
) -> Hs3Workspace {
    let mut binned_ws = export_hs3(binned, analysis_name, None, None);
    let unbinned_ws = export_unbinned_hs3(unbinned_spec, unbinned_model, analysis_name, None);

    // Merge unbinned distributions
    for d in &unbinned_ws.distributions {
        binned_ws.distributions.push(d.clone());
    }

    // Merge unbinned data entries
    for d in &unbinned_ws.data {
        binned_ws.data.push(d.clone());
    }

    // Merge unbinned-only parameters into domains and parameter_points
    let binned_param_names: HashSet<String> =
        binned.parameters().iter().map(|p| p.name.clone()).collect();

    for p in unbinned_model.parameters() {
        if !binned_param_names.contains(&p.name) {
            // Add to NP domain
            if let Some(domain) = binned_ws.domains.first_mut() {
                domain.axes.push(Hs3DomainAxis {
                    name: p.name.clone(),
                    min: p.bounds.0,
                    max: p.bounds.1,
                });
            }
            // Add to default_values
            if let Some(pp) = binned_ws.parameter_points.first_mut() {
                pp.parameters.push(Hs3ParameterValue { name: p.name.clone(), value: p.init });
            }
            // Add global observable if constrained
            if let Some(ns_unbinned::Constraint::Gaussian { mean, .. }) = &p.constraint {
                if let Some(domain) = binned_ws.domains.get_mut(1) {
                    domain.axes.push(Hs3DomainAxis {
                        name: format!("nom_{}", p.name),
                        min: mean - 10.0,
                        max: mean + 10.0,
                    });
                }
                if let Some(pp) = binned_ws.parameter_points.first_mut() {
                    pp.parameters
                        .push(Hs3ParameterValue { name: format!("nom_{}", p.name), value: *mean });
                }
            }
        }
    }

    // Update likelihood to include unbinned distributions
    if let Some(lh) = binned_ws.likelihoods.first_mut() {
        for ch in &unbinned_spec.channels {
            lh.distributions.push(ch.name.clone());
            lh.data.push(format!("obsData_{}", ch.name));
        }
    }

    // Add bestfit parameter points (hybrid global vector)
    if let Some((name, values)) = bestfit_params {
        // Collect all parameter names in hybrid global order
        let all_names: Vec<String> = {
            let mut names: Vec<String> =
                binned.parameters().iter().map(|p| p.name.clone()).collect();
            for p in unbinned_model.parameters() {
                if !binned_param_names.contains(&p.name) {
                    names.push(p.name.clone());
                }
            }
            names
        };
        let pp: Vec<Hs3ParameterValue> = all_names
            .iter()
            .zip(values.iter())
            .map(|(n, &v)| Hs3ParameterValue { name: n.clone(), value: v })
            .collect();
        binned_ws
            .parameter_points
            .push(Hs3ParameterPointSet { name: name.to_string(), parameters: pp });
    }

    // Update metadata
    binned_ws.misc = Some(serde_json::json!({
        "nextstat_extensions": ["unbinned_dist", "hybrid_likelihood"],
        "unbinned_spec_version": unbinned_spec.schema_version,
        "n_shared_parameters": shared_param_names.len(),
        "shared_parameters": shared_param_names,
    }));

    binned_ws
}

/// Convenience: export unbinned to JSON string.
#[cfg(feature = "unbinned")]
pub fn export_unbinned_hs3_json(
    spec: &UnbinnedSpecV0,
    model: &RustUnbinnedModel,
    analysis_name: &str,
    bestfit_params: Option<(&str, &[f64])>,
) -> Result<String, serde_json::Error> {
    let ws = export_unbinned_hs3(spec, model, analysis_name, bestfit_params);
    serde_json::to_string_pretty(&ws)
}

/// Convenience: export hybrid to JSON string.
#[cfg(feature = "unbinned")]
pub fn export_hybrid_hs3_json(
    binned: &HistFactoryModel,
    unbinned_spec: &UnbinnedSpecV0,
    unbinned_model: &RustUnbinnedModel,
    analysis_name: &str,
    bestfit_params: Option<(&str, &[f64])>,
    shared_param_names: &[String],
) -> Result<String, serde_json::Error> {
    let ws = export_hybrid_hs3(
        binned,
        unbinned_spec,
        unbinned_model,
        analysis_name,
        bestfit_params,
        shared_param_names,
    );
    serde_json::to_string_pretty(&ws)
}
