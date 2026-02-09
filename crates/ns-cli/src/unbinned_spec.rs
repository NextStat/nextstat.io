//! Unbinned (event-level) spec v0 parsing + compilation into `ns-unbinned` models.
//!
//! This is intentionally minimal for Phase 1:
//! - data: ROOT TTree ingestion via `ns-root`
//! - PDFs: Gaussian, Exponential, CrystalBall, DoubleCrystalBall, Chebyshev, Histogram, KDE
//! - yields: fixed / free parameter / scaled (signal-strength style)
//! - constraints: Gaussian (nuisance priors)
//! - rate systematics (Phase 2): NormSys (HistFactory-style)

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ns_root::RootFile;
use ns_unbinned::{
    ChebyshevPdf, Constraint, CrystalBallPdf, DoubleCrystalBallPdf, EventStore, ExponentialPdf,
    GaussianPdf, HistogramPdf, KdePdf, ObservableSpec, Parameter, Process, RateModifier,
    UnbinnedChannel, UnbinnedModel, UnbinnedPdf, YieldExpr,
};

pub const UNBINNED_SPEC_V0: &str = "nextstat_unbinned_spec_v0";

#[derive(Debug, Clone, Deserialize)]
pub struct UnbinnedSpecV0 {
    #[serde(rename = "$schema")]
    #[allow(dead_code)]
    pub schema_uri: Option<String>,
    pub schema_version: String,
    pub model: ModelSpec,
    pub channels: Vec<ChannelSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelSpec {
    /// Optional parameter-of-interest name.
    #[serde(default)]
    pub poi: Option<String>,
    pub parameters: Vec<ParameterSpec>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ParameterSpec {
    pub name: String,
    pub init: f64,
    pub bounds: [f64; 2],
    #[serde(default)]
    pub constraint: Option<ConstraintSpec>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConstraintSpec {
    Gaussian { mean: f64, sigma: f64 },
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChannelSpec {
    pub name: String,
    #[serde(default = "default_true")]
    pub include_in_fit: bool,
    pub data: DataSpec,
    pub observables: Vec<ObservableSpecV0>,
    pub processes: Vec<ProcessSpec>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct DataSpec {
    pub file: PathBuf,
    pub tree: String,
    #[serde(default)]
    pub selection: Option<String>,
    /// Optional per-event weight expression.
    ///
    /// Phase 1: observed data weights are not supported by `ns-unbinned::UnbinnedModel`,
    /// so this must be omitted.
    #[serde(default)]
    pub weight: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ObservableSpecV0 {
    pub name: String,
    #[serde(default)]
    pub expr: Option<String>,
    pub bounds: [f64; 2],
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProcessSpec {
    pub name: String,
    pub pdf: PdfSpec,
    #[serde(rename = "yield")]
    pub yield_spec: YieldSpec,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PdfSpec {
    Gaussian {
        observable: String,
        params: Vec<String>,
    },
    CrystalBall {
        observable: String,
        params: Vec<String>,
    },
    DoubleCrystalBall {
        observable: String,
        params: Vec<String>,
    },
    Exponential {
        observable: String,
        params: Vec<String>,
    },
    Chebyshev {
        observable: String,
        params: Vec<String>,
    },
    Histogram {
        observable: String,
        bin_edges: Vec<f64>,
        bin_content: Vec<f64>,
        #[serde(default)]
        pseudo_count: Option<f64>,
    },
    HistogramFromTree {
        observable: String,
        bin_edges: Vec<f64>,
        #[serde(default)]
        pseudo_count: Option<f64>,
        source: DataSpec,
        #[serde(default)]
        max_events: Option<usize>,
    },
    Kde {
        observable: String,
        bandwidth: f64,
        centers: Vec<f64>,
        #[serde(default)]
        weights: Option<Vec<f64>>,
    },
    KdeFromTree {
        observable: String,
        bandwidth: f64,
        source: DataSpec,
        #[serde(default)]
        max_events: Option<usize>,
    },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum YieldSpec {
    Fixed {
        value: f64,
        #[serde(default)]
        modifiers: Vec<RateModifierSpec>,
    },
    Parameter {
        name: String,
        #[serde(default)]
        modifiers: Vec<RateModifierSpec>,
    },
    Scaled {
        base_yield: f64,
        scale: String,
        #[serde(default)]
        modifiers: Vec<RateModifierSpec>,
    },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RateModifierSpec {
    /// HistFactory-like NormSys modifier (piecewise exponential interpolation).
    #[serde(rename = "normsys", alias = "norm_sys")]
    NormSys { param: String, lo: f64, hi: f64 },
}

pub fn read_unbinned_spec(path: &Path) -> Result<UnbinnedSpecV0> {
    let bytes =
        std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let spec: UnbinnedSpecV0 = serde_yaml_ng::from_slice(&bytes).with_context(|| {
        format!("failed to parse unbinned spec (YAML/JSON) from {}", path.display())
    })?;

    if spec.schema_version != UNBINNED_SPEC_V0 {
        anyhow::bail!(
            "unsupported schema_version: {} (expected {})",
            spec.schema_version,
            UNBINNED_SPEC_V0
        );
    }
    Ok(spec)
}

pub fn compile_unbinned_model(spec: &UnbinnedSpecV0, spec_path: &Path) -> Result<UnbinnedModel> {
    if spec.model.parameters.is_empty() {
        anyhow::bail!("unbinned spec: model.parameters must be non-empty");
    }
    if spec.channels.is_empty() {
        anyhow::bail!("unbinned spec: channels must be non-empty");
    }

    let mut parameters = Vec::with_capacity(spec.model.parameters.len());
    let mut index_by_name = HashMap::<String, usize>::new();

    for (i, p) in spec.model.parameters.iter().enumerate() {
        if index_by_name.insert(p.name.clone(), i).is_some() {
            anyhow::bail!("duplicate parameter name: '{}'", p.name);
        }
        let constraint = p.constraint.as_ref().map(|ConstraintSpec::Gaussian { mean, sigma }| {
            Constraint::Gaussian { mean: *mean, sigma: *sigma }
        });
        parameters.push(Parameter {
            name: p.name.clone(),
            init: p.init,
            bounds: (p.bounds[0], p.bounds[1]),
            constraint,
        });
    }

    let poi_index = match &spec.model.poi {
        None => None,
        Some(name) => Some(
            *index_by_name
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("unknown POI parameter name: '{name}'"))?,
        ),
    };

    let base_dir = spec_path.parent().unwrap_or_else(|| Path::new("."));

    let mut channels = Vec::with_capacity(spec.channels.len());
    for ch in &spec.channels {
        if ch.processes.is_empty() {
            anyhow::bail!("channel '{}' has no processes", ch.name);
        }
        if ch.observables.is_empty() {
            anyhow::bail!("channel '{}' has no observables", ch.name);
        }
        if ch.data.weight.is_some() {
            anyhow::bail!(
                "channel '{}' specifies data.weight, but weighted observed data is not supported in Phase 1",
                ch.name
            );
        }

        // Resolve data file relative to the spec location.
        let root_path = if ch.data.file.is_absolute() {
            ch.data.file.clone()
        } else {
            base_dir.join(&ch.data.file)
        };

        let root = RootFile::open(&root_path)
            .with_context(|| format!("failed to open ROOT file {}", root_path.display()))?;

        let observables: Vec<ObservableSpec> = ch
            .observables
            .iter()
            .map(|o| {
                let expr = o.expr.clone().unwrap_or_else(|| o.name.clone());
                ObservableSpec { name: o.name.clone(), expr, bounds: (o.bounds[0], o.bounds[1]) }
            })
            .collect();

        let store = EventStore::from_tree(
            &root,
            &ch.data.tree,
            &observables,
            ch.data.selection.as_deref(),
            None,
        )
        .with_context(|| {
            format!("failed to load channel '{}' data from {}", ch.name, root_path.display())
        })?;

        let store = Arc::new(store);

        let observable_names: HashMap<&str, ()> =
            observables.iter().map(|o| (o.name.as_str(), ())).collect();
        let observable_specs: HashMap<&str, &ObservableSpec> =
            observables.iter().map(|o| (o.name.as_str(), o)).collect();

        let mut processes = Vec::with_capacity(ch.processes.len());
        for proc in &ch.processes {
            let (pdf, shape_param_names, observable_name): (
                Arc<dyn UnbinnedPdf>,
                Vec<String>,
                String,
            ) = match &proc.pdf {
                PdfSpec::Gaussian { observable, params } => {
                    if params.len() != 2 {
                        anyhow::bail!(
                            "process '{}' gaussian params must have length 2 (mu,sigma), got {}",
                            proc.name,
                            params.len()
                        );
                    }
                    (
                        Arc::new(GaussianPdf::new(observable.clone())),
                        params.clone(),
                        observable.clone(),
                    )
                }
                PdfSpec::CrystalBall { observable, params } => {
                    if params.len() != 4 {
                        anyhow::bail!(
                            "process '{}' crystal_ball params must have length 4 (mu,sigma,alpha,n), got {}",
                            proc.name,
                            params.len()
                        );
                    }
                    (
                        Arc::new(CrystalBallPdf::new(observable.clone())),
                        params.clone(),
                        observable.clone(),
                    )
                }
                PdfSpec::DoubleCrystalBall { observable, params } => {
                    if params.len() != 6 {
                        anyhow::bail!(
                            "process '{}' double_crystal_ball params must have length 6 (mu,sigma,alpha_l,n_l,alpha_r,n_r), got {}",
                            proc.name,
                            params.len()
                        );
                    }
                    (
                        Arc::new(DoubleCrystalBallPdf::new(observable.clone())),
                        params.clone(),
                        observable.clone(),
                    )
                }
                PdfSpec::Exponential { observable, params } => {
                    if params.len() != 1 {
                        anyhow::bail!(
                            "process '{}' exponential params must have length 1 (lambda), got {}",
                            proc.name,
                            params.len()
                        );
                    }
                    (
                        Arc::new(ExponentialPdf::new(observable.clone())),
                        params.clone(),
                        observable.clone(),
                    )
                }
                PdfSpec::Chebyshev { observable, params } => {
                    if params.is_empty() {
                        anyhow::bail!(
                            "process '{}' chebyshev params must be non-empty (c_1..c_order), got 0",
                            proc.name
                        );
                    }
                    (
                        Arc::new(ChebyshevPdf::new(observable.clone(), params.len())?),
                        params.clone(),
                        observable.clone(),
                    )
                }
                PdfSpec::Histogram { observable, bin_edges, bin_content, pseudo_count } => {
                    let pc = pseudo_count.unwrap_or(0.0);
                    (
                        Arc::new(HistogramPdf::from_edges_and_contents(
                            observable.clone(),
                            bin_edges.clone(),
                            bin_content.clone(),
                            pc,
                        )?),
                        Vec::new(),
                        observable.clone(),
                    )
                }
                PdfSpec::HistogramFromTree {
                    observable,
                    bin_edges,
                    pseudo_count,
                    source,
                    max_events,
                } => {
                    let Some(obs) = observable_specs.get(observable.as_str()) else {
                        anyhow::bail!(
                            "process '{}' references unknown observable '{}' (channel '{}')",
                            proc.name,
                            observable,
                            ch.name
                        );
                    };
                    // Ensure support matches exactly for correct normalization.
                    let eps = 1e-12;
                    let edge_min = *bin_edges.first().unwrap_or(&f64::NAN);
                    let edge_max = *bin_edges.last().unwrap_or(&f64::NAN);
                    if (edge_min - obs.bounds.0).abs() > eps
                        || (edge_max - obs.bounds.1).abs() > eps
                    {
                        anyhow::bail!(
                            "process '{}' histogram_from_tree bin_edges support [{edge_min}, {edge_max}] must match observable '{}' bounds {:?} (channel '{}')",
                            proc.name,
                            observable,
                            obs.bounds,
                            ch.name
                        );
                    }

                    let src_path = if source.file.is_absolute() {
                        source.file.clone()
                    } else {
                        base_dir.join(&source.file)
                    };
                    let src_root = RootFile::open(&src_path).with_context(|| {
                        format!("failed to open ROOT file {}", src_path.display())
                    })?;

                    let train_obs = vec![(*obs).clone()];
                    let train = EventStore::from_tree(
                            &src_root,
                            &source.tree,
                            &train_obs,
                            source.selection.as_deref(),
                            source.weight.as_deref(),
                        )
                        .with_context(|| {
                            format!(
                                "failed to load histogram training data for process '{}' (channel '{}') from {}",
                                proc.name,
                                ch.name,
                                src_path.display()
                            )
                        })?;

                    let mut xs = train
                        .column(observable.as_str())
                        .ok_or_else(|| {
                            anyhow::anyhow!("histogram training data missing column '{observable}'")
                        })?
                        .to_vec();
                    let mut ws = train.weights().map(|w| w.to_vec());

                    if let Some(max) = *max_events {
                        if max == 0 {
                            anyhow::bail!(
                                "process '{}' histogram_from_tree max_events must be > 0",
                                proc.name
                            );
                        }
                        let keep = max.min(xs.len());
                        xs.truncate(keep);
                        if let Some(w) = ws.as_mut() {
                            w.truncate(keep);
                        }
                    }

                    let n_bins = bin_edges.len().saturating_sub(1);
                    if n_bins == 0 {
                        anyhow::bail!(
                            "process '{}' histogram_from_tree requires at least 2 bin_edges",
                            proc.name
                        );
                    }

                    let mut bin_content = vec![0.0f64; n_bins];
                    for (i, &x) in xs.iter().enumerate() {
                        if x >= edge_max {
                            bin_content[n_bins - 1] += ws.as_ref().map(|w| w[i]).unwrap_or(1.0);
                            continue;
                        }
                        // `k` is the number of edges <= x, so bin index is k-1.
                        let k = bin_edges.partition_point(|e| *e <= x);
                        if k == 0 || k > n_bins {
                            anyhow::bail!(
                                "process '{}' histogram_from_tree: x out of range: x={x} not in [{edge_min}, {edge_max}]",
                                proc.name
                            );
                        }
                        let idx = k - 1;
                        bin_content[idx] += ws.as_ref().map(|w| w[i]).unwrap_or(1.0);
                    }

                    let pc = pseudo_count.unwrap_or(0.0);
                    (
                        Arc::new(HistogramPdf::from_edges_and_contents(
                            observable.clone(),
                            bin_edges.clone(),
                            bin_content,
                            pc,
                        )?),
                        Vec::new(),
                        observable.clone(),
                    )
                }
                PdfSpec::Kde { observable, bandwidth, centers, weights } => {
                    let Some(obs) = observable_specs.get(observable.as_str()) else {
                        anyhow::bail!(
                            "process '{}' references unknown observable '{}' (channel '{}')",
                            proc.name,
                            observable,
                            ch.name
                        );
                    };
                    (
                        Arc::new(KdePdf::from_samples(
                            observable.clone(),
                            obs.bounds,
                            centers.clone(),
                            weights.clone(),
                            *bandwidth,
                        )?),
                        Vec::new(),
                        observable.clone(),
                    )
                }
                PdfSpec::KdeFromTree { observable, bandwidth, source, max_events } => {
                    let Some(obs) = observable_specs.get(observable.as_str()) else {
                        anyhow::bail!(
                            "process '{}' references unknown observable '{}' (channel '{}')",
                            proc.name,
                            observable,
                            ch.name
                        );
                    };

                    let src_path = if source.file.is_absolute() {
                        source.file.clone()
                    } else {
                        base_dir.join(&source.file)
                    };
                    let src_root = RootFile::open(&src_path).with_context(|| {
                        format!("failed to open ROOT file {}", src_path.display())
                    })?;

                    let train_obs = vec![(*obs).clone()];
                    let train = EventStore::from_tree(
                            &src_root,
                            &source.tree,
                            &train_obs,
                            source.selection.as_deref(),
                            source.weight.as_deref(),
                        )
                        .with_context(|| {
                            format!(
                                "failed to load KDE training data for process '{}' (channel '{}') from {}",
                                proc.name,
                                ch.name,
                                src_path.display()
                            )
                        })?;

                    let mut centers = train
                        .column(observable.as_str())
                        .ok_or_else(|| {
                            anyhow::anyhow!("KDE training data missing column '{observable}'")
                        })?
                        .to_vec();
                    let mut weights = train.weights().map(|w| w.to_vec());

                    if let Some(max) = *max_events {
                        if max == 0 {
                            anyhow::bail!(
                                "process '{}' kde_from_tree max_events must be > 0",
                                proc.name
                            );
                        }
                        let keep = max.min(centers.len());
                        centers.truncate(keep);
                        if let Some(w) = weights.as_mut() {
                            w.truncate(keep);
                        }
                    }

                    (
                        Arc::new(KdePdf::from_samples(
                            observable.clone(),
                            obs.bounds,
                            centers,
                            weights,
                            *bandwidth,
                        )?),
                        Vec::new(),
                        observable.clone(),
                    )
                }
            };

            if !observable_names.contains_key(observable_name.as_str()) {
                anyhow::bail!(
                    "process '{}' references unknown observable '{}' (channel '{}')",
                    proc.name,
                    observable_name,
                    ch.name
                );
            }

            let mut shape_param_indices = Vec::with_capacity(shape_param_names.len());
            for p in shape_param_names {
                let idx = *index_by_name.get(&p).ok_or_else(|| {
                    anyhow::anyhow!(
                        "process '{}' references unknown shape parameter '{}'",
                        proc.name,
                        p
                    )
                })?;
                shape_param_indices.push(idx);
            }

            let yield_expr = match &proc.yield_spec {
                YieldSpec::Fixed { value, modifiers } => wrap_yield_modifiers(
                    YieldExpr::Fixed(*value),
                    modifiers,
                    &index_by_name,
                    &proc.name,
                )?,
                YieldSpec::Parameter { name, modifiers } => {
                    let idx = *index_by_name.get(name).ok_or_else(|| {
                        anyhow::anyhow!(
                            "process '{}' references unknown yield parameter '{}'",
                            proc.name,
                            name
                        )
                    })?;
                    wrap_yield_modifiers(
                        YieldExpr::Parameter { index: idx },
                        modifiers,
                        &index_by_name,
                        &proc.name,
                    )?
                }
                YieldSpec::Scaled { base_yield, scale, modifiers } => {
                    let idx = *index_by_name.get(scale).ok_or_else(|| {
                        anyhow::anyhow!(
                            "process '{}' references unknown scale parameter '{}'",
                            proc.name,
                            scale
                        )
                    })?;
                    wrap_yield_modifiers(
                        YieldExpr::Scaled { base_yield: *base_yield, scale_index: idx },
                        modifiers,
                        &index_by_name,
                        &proc.name,
                    )?
                }
            };

            processes.push(Process {
                name: proc.name.clone(),
                pdf,
                shape_param_indices,
                yield_expr,
            });
        }

        channels.push(UnbinnedChannel {
            name: ch.name.clone(),
            include_in_fit: ch.include_in_fit,
            data: store,
            processes,
        });
    }

    UnbinnedModel::new(parameters, channels, poi_index).context("failed to construct UnbinnedModel")
}

fn wrap_yield_modifiers(
    base: YieldExpr,
    modifiers: &[RateModifierSpec],
    index_by_name: &HashMap<String, usize>,
    process_name: &str,
) -> Result<YieldExpr> {
    if modifiers.is_empty() {
        return Ok(base);
    }
    let mut compiled = Vec::with_capacity(modifiers.len());
    for m in modifiers {
        match m {
            RateModifierSpec::NormSys { param, lo, hi } => {
                let idx = *index_by_name.get(param).ok_or_else(|| {
                    anyhow::anyhow!(
                        "process '{}' references unknown NormSys nuisance parameter '{}'",
                        process_name,
                        param
                    )
                })?;
                compiled.push(RateModifier::NormSys { alpha_index: idx, lo: *lo, hi: *hi });
            }
        }
    }
    Ok(YieldExpr::Modified { base: Box::new(base), modifiers: compiled })
}
