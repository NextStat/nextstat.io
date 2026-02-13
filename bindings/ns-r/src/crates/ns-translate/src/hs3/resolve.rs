//! Two-pass reference resolver for HS3 workspaces.
//!
//! HS3 JSON uses named cross-references (e.g., a modifier's `parameter` field
//! references a parameter defined in `domains[]` and `parameter_points[]`).
//! This module resolves all references into a flat [`ResolvedWorkspace`] that
//! can be directly converted to `HistFactoryModel` without further name lookups.
//!
//! # Two-Pass Architecture
//!
//! **Pass 1 — Index:** Build `HashMap<String, &T>` for every named object
//! (distributions, data, domains, constraints, parameter points).
//!
//! **Pass 2 — Resolve:** For a selected analysis, walk each channel's samples
//! and modifiers, resolving every name reference against the indices built in
//! Pass 1.

use super::schema::*;
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Public result types
// ---------------------------------------------------------------------------

/// Fully resolved workspace ready for conversion to `HistFactoryModel`.
#[derive(Debug, Clone)]
pub struct ResolvedWorkspace {
    /// Selected analysis name.
    pub analysis_name: String,
    /// Parameters of interest (names).
    pub pois: Vec<String>,
    /// Resolved channels: each is a (distribution, observed_data) pair.
    pub channels: Vec<ResolvedChannel>,
    /// All parameter constraints (param_name → ConstraintInfo).
    pub constraints: HashMap<String, ConstraintInfo>,
    /// All parameter bounds (param_name → (min, max)).
    pub bounds: HashMap<String, (f64, f64)>,
    /// Parameter init values (param_name → init).
    pub inits: HashMap<String, f64>,
    /// Global observable values (glob_obs_name → value).
    pub global_observables: HashMap<String, f64>,
    /// HS3 metadata (preserved for roundtrip).
    pub metadata: Hs3Metadata,
    /// Opaque misc section (preserved for roundtrip).
    pub misc: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct ResolvedChannel {
    pub name: String,
    pub n_bins: usize,
    pub observed: Vec<f64>,
    pub samples: Vec<ResolvedSample>,
}

#[derive(Debug, Clone)]
pub struct ResolvedSample {
    pub name: String,
    pub nominal: Vec<f64>,
    pub errors: Option<Vec<f64>>,
    pub modifiers: Vec<ResolvedModifier>,
}

#[derive(Debug, Clone)]
pub enum ResolvedModifier {
    NormFactor { param_name: String },
    NormSys { param_name: String, hi: f64, lo: f64 },
    HistoSys { param_name: String, hi_template: Vec<f64>, lo_template: Vec<f64> },
    StatError { param_names: Vec<String>, constraint_type: StatConstraintType },
    ShapeSys { param_names: Vec<String>, uncertainties: Vec<f64> },
    ShapeFactor { param_names: Vec<String> },
    Lumi { param_name: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatConstraintType {
    Poisson,
    Gaussian,
}

#[derive(Debug, Clone)]
pub struct ConstraintInfo {
    pub kind: ConstraintKind,
    /// Constraint center value (e.g. 0.0 for standard alpha NPs).
    pub center: f64,
    /// Constraint width (sigma for Gaussian; unused for Poisson).
    pub width: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    Gaussian,
    Poisson,
    LogNormal,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum Hs3ResolveError {
    #[error("analysis not found: {0}")]
    AnalysisNotFound(String),
    #[error("likelihood not found: {0}")]
    LikelihoodNotFound(String),
    #[error("distribution not found: {0}")]
    DistributionNotFound(String),
    #[error("data not found: {0}")]
    DataNotFound(String),
    #[error("domain not found: {0}")]
    DomainNotFound(String),
    #[error("parameter point set not found: {0}")]
    ParamPointSetNotFound(String),
    #[error("distribution {0} is not a histfactory_dist")]
    NotHistFactoryDist(String),
    #[error(
        "bin count mismatch in channel {channel}: observed has {obs_bins} bins but sample {sample} has {sample_bins}"
    )]
    BinCountMismatch { channel: String, obs_bins: usize, sample: String, sample_bins: usize },
    #[error("duplicate name in {section}: {name}")]
    DuplicateName { section: String, name: String },
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Resolve an HS3 workspace for a specific analysis.
///
/// - `analysis_name`: which analysis to build (`None` = first).
/// - `param_point_set`: which parameter_points set to use for init values
///   (`None` = `"default_values"` or the first available set).
pub fn resolve(
    ws: &Hs3Workspace,
    analysis_name: Option<&str>,
    param_point_set: Option<&str>,
) -> Result<ResolvedWorkspace, Hs3ResolveError> {
    // -----------------------------------------------------------------------
    // Pass 1: Index all named objects
    // -----------------------------------------------------------------------

    // -- Distributions by name --
    let mut dist_map: HashMap<&str, &Hs3Distribution> = HashMap::new();
    for d in &ws.distributions {
        if let Some(name) = d.name() {
            dist_map.insert(name, d);
        }
    }

    // -- Gaussian constraints: parameter_name → ConstraintInfo --
    let mut constraint_by_param: HashMap<String, ConstraintInfo> = HashMap::new();
    for d in &ws.distributions {
        match d {
            Hs3Distribution::Gaussian(g) => {
                constraint_by_param.insert(
                    g.x.clone(),
                    ConstraintInfo {
                        kind: ConstraintKind::Gaussian,
                        center: 0.0, // will be overwritten from global observables
                        width: g.sigma,
                    },
                );
            }
            Hs3Distribution::Poisson(p) => {
                constraint_by_param.insert(
                    p.x.clone(),
                    ConstraintInfo { kind: ConstraintKind::Poisson, center: 1.0, width: 0.0 },
                );
            }
            Hs3Distribution::LogNormal(l) => {
                constraint_by_param.insert(
                    l.x.clone(),
                    ConstraintInfo { kind: ConstraintKind::LogNormal, center: 0.0, width: l.sigma },
                );
            }
            _ => {}
        }
    }

    // -- Data by name --
    let mut data_map: HashMap<&str, &Hs3Data> = HashMap::new();
    for d in &ws.data {
        data_map.insert(&d.name, d);
    }

    // -- Domains: merge all axes into param_bounds --
    let mut domain_map: HashMap<&str, &Hs3Domain> = HashMap::new();
    for d in &ws.domains {
        domain_map.insert(&d.name, d);
    }

    // -- Parameter points: index by set name --
    let mut pp_map: HashMap<&str, &Hs3ParameterPointSet> = HashMap::new();
    for pp in &ws.parameter_points {
        pp_map.insert(&pp.name, pp);
    }

    // -- Analyses and likelihoods by name --
    let mut analysis_map: HashMap<&str, &Hs3Analysis> = HashMap::new();
    for a in &ws.analyses {
        analysis_map.insert(&a.name, a);
    }
    let mut likelihood_map: HashMap<&str, &Hs3Likelihood> = HashMap::new();
    for l in &ws.likelihoods {
        likelihood_map.insert(&l.name, l);
    }

    // -----------------------------------------------------------------------
    // Select analysis
    // -----------------------------------------------------------------------
    let analysis = match analysis_name {
        Some(name) => analysis_map
            .get(name)
            .ok_or_else(|| Hs3ResolveError::AnalysisNotFound(name.to_string()))?,
        None => ws
            .analyses
            .first()
            .ok_or_else(|| Hs3ResolveError::AnalysisNotFound("(empty)".to_string()))?,
    };

    let likelihood = likelihood_map
        .get(analysis.likelihood.as_str())
        .ok_or_else(|| Hs3ResolveError::LikelihoodNotFound(analysis.likelihood.clone()))?;

    // -----------------------------------------------------------------------
    // Merge bounds from all referenced domains
    // -----------------------------------------------------------------------
    let mut param_bounds: HashMap<String, (f64, f64)> = HashMap::new();
    let mut global_obs_names: HashSet<String> = HashSet::new();

    for domain_name in &analysis.domains {
        if let Some(domain) = domain_map.get(domain_name.as_str()) {
            let is_global_obs = domain_name.contains("global_observables");
            for axis in &domain.axes {
                param_bounds.insert(axis.name.clone(), (axis.min, axis.max));
                if is_global_obs {
                    global_obs_names.insert(axis.name.clone());
                }
            }
        }
        // Silently skip missing domains (some may be informational)
    }

    // -----------------------------------------------------------------------
    // Load parameter init values from selected param_point_set
    // -----------------------------------------------------------------------
    let pp_set_name = param_point_set.unwrap_or("default_values");
    let pp_set = match pp_map.get(pp_set_name) {
        Some(pp) => *pp,
        None => ws
            .parameter_points
            .first()
            .ok_or_else(|| Hs3ResolveError::ParamPointSetNotFound(pp_set_name.to_string()))?,
    };

    let mut param_inits: HashMap<String, f64> = HashMap::new();
    let mut global_obs_values: HashMap<String, f64> = HashMap::new();

    for pv in &pp_set.parameters {
        // Skip binWidth_* parameters (ROOT internal bookkeeping)
        if pv.name.starts_with("binWidth_") {
            continue;
        }
        if global_obs_names.contains(&pv.name) {
            global_obs_values.insert(pv.name.clone(), pv.value);
        } else {
            param_inits.insert(pv.name.clone(), pv.value);
        }
    }

    // -----------------------------------------------------------------------
    // Update constraint centers from global observable values
    // -----------------------------------------------------------------------
    // Gaussian constraint `mean` field references a global observable by name.
    // Look up the global observable value and use it as the constraint center.
    for d in &ws.distributions {
        if let Hs3Distribution::Gaussian(g) = d
            && let Some(center_val) = global_obs_values.get(&g.mean)
            && let Some(ci) = constraint_by_param.get_mut(&g.x)
        {
            ci.center = *center_val;
        }
    }

    // -----------------------------------------------------------------------
    // Pass 2: Resolve channels
    // -----------------------------------------------------------------------
    let mut channels = Vec::with_capacity(likelihood.distributions.len());

    for (dist_name, data_name) in likelihood.distributions.iter().zip(likelihood.data.iter()) {
        // Look up distribution by name.
        let dist = dist_map
            .get(dist_name.as_str())
            .ok_or_else(|| Hs3ResolveError::DistributionNotFound(dist_name.clone()))?;

        // HS3 likelihoods may include NextStat extension distributions (e.g. unbinned)
        // and future non-HistFactory objects. Those are ignored by the HistFactory
        // resolver instead of hard-failing, so hybrid workspaces remain ingestible.
        let hf = match dist {
            Hs3Distribution::HistFactory(hf) => hf,
            _ => continue,
        };

        // Look up observed data
        let obs_data = data_map
            .get(data_name.as_str())
            .ok_or_else(|| Hs3ResolveError::DataNotFound(data_name.clone()))?;

        let n_bins = obs_data.contents.len();

        // Resolve samples
        let mut resolved_samples = Vec::with_capacity(hf.samples.len());
        for sample in &hf.samples {
            // Validate bin count
            if sample.data.contents.len() != n_bins {
                return Err(Hs3ResolveError::BinCountMismatch {
                    channel: hf.name.clone(),
                    obs_bins: n_bins,
                    sample: sample.name.clone(),
                    sample_bins: sample.data.contents.len(),
                });
            }

            // Resolve modifiers
            let mut resolved_mods = Vec::with_capacity(sample.modifiers.len());
            for m in &sample.modifiers {
                if let Some(rm) = resolve_modifier(m)? {
                    resolved_mods.push(rm);
                }
                // Unknown modifiers are silently skipped (forward-compat)
            }

            resolved_samples.push(ResolvedSample {
                name: sample.name.clone(),
                nominal: sample.data.contents.clone(),
                errors: sample.data.errors.clone(),
                modifiers: resolved_mods,
            });
        }

        channels.push(ResolvedChannel {
            name: hf.name.clone(),
            n_bins,
            observed: obs_data.contents.clone(),
            samples: resolved_samples,
        });
    }

    // -----------------------------------------------------------------------
    // Build result
    // -----------------------------------------------------------------------
    Ok(ResolvedWorkspace {
        analysis_name: analysis.name.clone(),
        pois: analysis.parameters_of_interest.clone(),
        channels,
        constraints: constraint_by_param,
        bounds: param_bounds,
        inits: param_inits,
        global_observables: global_obs_values,
        metadata: ws.metadata.clone(),
        misc: ws.misc.clone(),
    })
}

// ---------------------------------------------------------------------------
// Helper: resolve a single modifier
// ---------------------------------------------------------------------------

fn resolve_modifier(m: &Hs3Modifier) -> Result<Option<ResolvedModifier>, Hs3ResolveError> {
    match m {
        Hs3Modifier::NormFactor { parameter, .. } => {
            Ok(Some(ResolvedModifier::NormFactor { param_name: parameter.clone() }))
        }
        Hs3Modifier::NormSys { parameter, data, .. } => Ok(Some(ResolvedModifier::NormSys {
            param_name: parameter.clone(),
            hi: data.hi,
            lo: data.lo,
        })),
        Hs3Modifier::HistoSys { parameter, data, .. } => Ok(Some(ResolvedModifier::HistoSys {
            param_name: parameter.clone(),
            hi_template: data.hi.contents.clone(),
            lo_template: data.lo.contents.clone(),
        })),
        Hs3Modifier::StatError { parameters, constraint_type, .. } => {
            let ct = match constraint_type.as_str() {
                "Poisson" => StatConstraintType::Poisson,
                "Gaussian" => StatConstraintType::Gaussian,
                _ => StatConstraintType::Poisson, // default to Poisson
            };
            Ok(Some(ResolvedModifier::StatError {
                param_names: parameters.clone(),
                constraint_type: ct,
            }))
        }
        Hs3Modifier::ShapeSys { parameters, data, .. } => {
            let uncertainties = data.as_ref().map(|d| d.vals.clone()).unwrap_or_default();
            Ok(Some(ResolvedModifier::ShapeSys { param_names: parameters.clone(), uncertainties }))
        }
        Hs3Modifier::ShapeFactor { parameters, parameter, .. } => {
            // ShapeFactor can have either `parameters: [...]` or `parameter: "..."`.
            let param_names = if let Some(ps) = parameters {
                ps.clone()
            } else if let Some(p) = parameter {
                vec![p.clone()]
            } else {
                vec![]
            };
            Ok(Some(ResolvedModifier::ShapeFactor { param_names }))
        }
        Hs3Modifier::Lumi { parameter, .. } => {
            Ok(Some(ResolvedModifier::Lumi { param_name: parameter.clone() }))
        }
        Hs3Modifier::Unknown(_) => Ok(None), // skip unknown modifiers
    }
}
