//! Unbinned (event-level) spec v0 parsing + compilation into `ns-unbinned` models.
//!
//! This is intentionally minimal for Phase 1:
//! - data: ROOT TTree ingestion via `ns-root`
//! - PDFs: Gaussian, Exponential, CrystalBall, DoubleCrystalBall, Chebyshev, Histogram, KDE,
//!   and a few common 1D shapes (Argus, Voigtian, Spline).
//! - yields: fixed / free parameter / scaled (signal-strength style)
//! - constraints: Gaussian (nuisance priors)
//! - rate systematics (Phase 2): NormSys + WeightSys (HistFactory-style)

#![allow(missing_docs)]

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ns_root::RootFile;

use crate::{
    ArgusPdf, ChebyshevPdf, Constraint, CrystalBallPdf, DoubleCrystalBallPdf, EventStore,
    ExponentialPdf, GaussianPdf, HistoSysInterpCode, HistogramPdf, HistogramSystematic,
    HorizontalMorphingKdePdf, KdeHorizontalSystematic, KdePdf, KdeWeightSystematic,
    MorphingHistogramPdf, MorphingKdePdf, ObservableSpec, Parameter, Process, RateModifier,
    SplinePdf, UnbinnedChannel, UnbinnedModel, UnbinnedPdf, VoigtianPdf, YieldExpr,
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
    /// TTree name inside a ROOT file.  Required when `format` is `root` (or inferred as ROOT).
    #[serde(default)]
    pub tree: Option<String>,
    /// Optional channel name for multi-channel Parquet files (uses the `_channel` column).
    ///
    /// Parquet only. When set, the reader filters the Parquet file to rows with
    /// `_channel == channel`.
    #[serde(default)]
    pub channel: Option<String>,
    #[serde(default)]
    pub selection: Option<String>,
    /// Optional per-event weight expression.
    ///
    /// Weight expressions use the `ns-root` expression language (ROOT sources only).
    #[serde(default)]
    pub weight: Option<String>,
    /// Explicit data format override.  If omitted, inferred from file extension:
    /// `.root` → ROOT TTree, `.parquet` / `.pq` → Parquet.
    #[serde(default)]
    pub format: Option<DataFormat>,
}

/// Data file format for unbinned event ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataFormat {
    /// ROOT TTree (requires `tree` field).
    Root,
    /// Apache Parquet (observable bounds read from file metadata).
    Parquet,
}

impl DataSpec {
    /// Resolve the effective data format, falling back to file-extension heuristic.
    pub fn effective_format(&self) -> DataFormat {
        if let Some(f) = self.format {
            return f;
        }
        match self.file.extension().and_then(|e| e.to_str()) {
            Some("parquet" | "pq") => DataFormat::Parquet,
            _ => DataFormat::Root,
        }
    }

    /// Return the tree name, or error if required but missing.
    pub fn tree_name(&self) -> anyhow::Result<&str> {
        self.tree.as_deref().ok_or_else(|| {
            anyhow::anyhow!("data.tree is required for ROOT files (file: {})", self.file.display())
        })
    }
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
    /// ARGUS background shape (normalized on the EventStore bounds).
    ///
    /// Shape parameters: `[c, p]` (cutoff is taken from the observable upper bound).
    Argus {
        observable: String,
        params: Vec<String>,
    },
    /// Voigtian line-shape (Gaussian ⊗ Breit-Wigner), normalized on the EventStore bounds.
    ///
    /// Shape parameters: `[mu, sigma, gamma]`.
    Voigtian {
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
        #[serde(default)]
        weight_systematics: Vec<WeightSystematicSpec>,
        #[serde(default)]
        horizontal_systematics: Vec<HorizontalSystematicSpec>,
    },
    Kde {
        observable: String,
        bandwidth: f64,
        centers: Vec<f64>,
        #[serde(default)]
        weights: Option<Vec<f64>>,
    },
    /// 1D monotonic cubic spline template PDF defined by density values at knots.
    ///
    /// This PDF has no shape parameters; it is a fixed template. Knot support must
    /// match the observable bounds exactly (within epsilon) for correct normalization.
    Spline {
        observable: String,
        knots_x: Vec<f64>,
        knots_y: Vec<f64>,
    },
    /// Product of independent component PDFs: `p(x,y,...) = p1(x) * p2(y) * ...`.
    ///
    /// This is a CPU-only composition helper intended for simple multi-observable models
    /// where the independence assumption is acceptable.
    ///
    /// Current limitation: `components` must be a non-empty list of *inline* PDFs
    /// (no `*_from_tree` and no neural PDFs). Nested `product` is not supported.
    Product {
        components: Vec<PdfSpec>,
    },
    KdeFromTree {
        observable: String,
        bandwidth: f64,
        source: DataSpec,
        #[serde(default)]
        max_events: Option<usize>,
        #[serde(default)]
        weight_systematics: Vec<WeightSystematicSpec>,
        #[serde(default)]
        horizontal_systematics: Vec<HorizontalSystematicSpec>,
    },
    /// ONNX normalizing flow (unconditional). Requires feature `neural`.
    ///
    /// ```yaml
    /// pdf:
    ///   type: flow
    ///   manifest: "models/signal_flow/flow_manifest.json"
    /// ```
    Flow {
        /// Path to `flow_manifest.json` (relative to the spec file).
        manifest: PathBuf,
    },
    /// ONNX normalizing flow with context parameters (conditional). Requires feature `neural`.
    ///
    /// ```yaml
    /// pdf:
    ///   type: conditional_flow
    ///   manifest: "models/signal_flow/flow_manifest.json"
    ///   context_params: ["alpha_syst1", "alpha_syst2"]
    /// ```
    ConditionalFlow {
        /// Path to `flow_manifest.json` (relative to the spec file).
        manifest: PathBuf,
        /// Names of model parameters fed as context to the flow.
        context_params: Vec<String>,
    },
    /// Neural DCR surrogate replacing binned template morphing. Requires feature `neural`.
    ///
    /// ```yaml
    /// pdf:
    ///   type: dcr_surrogate
    ///   manifest: "models/bkg_dcr/flow_manifest.json"
    ///   systematics: ["jes_alpha", "jer_alpha"]
    /// ```
    DcrSurrogate {
        /// Path to `flow_manifest.json` (relative to the spec file).
        manifest: PathBuf,
        /// Names of systematic nuisance parameters fed as context to the flow.
        systematics: Vec<String>,
    },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoSysInterpCodeSpec {
    Code0,
    Code4p,
}

fn default_histosys_interp_code() -> HistoSysInterpCodeSpec {
    HistoSysInterpCodeSpec::Code0
}

impl HistoSysInterpCodeSpec {
    fn as_unbinned(&self) -> HistoSysInterpCode {
        match self {
            HistoSysInterpCodeSpec::Code0 => HistoSysInterpCode::Code0,
            HistoSysInterpCodeSpec::Code4p => HistoSysInterpCode::Code4p,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct WeightSystematicSpec {
    /// Name of the nuisance parameter `α` controlling the variation.
    pub param: String,
    /// Expression evaluating to a non-negative *ratio* `w_up / w_nom` per event.
    pub up: String,
    /// Expression evaluating to a non-negative *ratio* `w_down / w_nom` per event.
    pub down: String,
    /// Interpolation code (default: code0 / piecewise linear).
    #[serde(default = "default_histosys_interp_code")]
    pub interp: HistoSysInterpCodeSpec,
    /// Apply this systematic to the PDF shape (template morphing).
    #[serde(default = "default_true")]
    pub apply_to_shape: bool,
    /// Apply this systematic to the process yield (rate modifier from total up/down weights).
    #[serde(default = "default_true")]
    pub apply_to_yield: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HorizontalSystematicSpec {
    /// Name of the nuisance parameter `α` controlling the variation.
    pub param: String,
    /// Expression evaluating to the observable value at `α = +1` (e.g. shifted branch or formula).
    pub up: String,
    /// Expression evaluating to the observable value at `α = -1`.
    pub down: String,
    /// Interpolation code (default: code0 / piecewise linear).
    #[serde(default = "default_histosys_interp_code")]
    pub interp: HistoSysInterpCodeSpec,
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
    /// HistFactory-like WeightSys modifier (template interpolation on a scalar yield factor).
    #[serde(rename = "weightsys", alias = "weight_sys")]
    WeightSys {
        param: String,
        lo: f64,
        hi: f64,
        #[serde(default)]
        interp_code: Option<String>,
    },
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

        // Resolve data file relative to the spec location.
        let data_path = if ch.data.file.is_absolute() {
            ch.data.file.clone()
        } else {
            base_dir.join(&ch.data.file)
        };

        let observables: Vec<ObservableSpec> = ch
            .observables
            .iter()
            .map(|o| {
                let expr = o.expr.clone().unwrap_or_else(|| o.name.clone());
                ObservableSpec { name: o.name.clone(), expr, bounds: (o.bounds[0], o.bounds[1]) }
            })
            .collect();

        let store = match ch.data.effective_format() {
            DataFormat::Root => {
                if ch.data.channel.is_some() {
                    anyhow::bail!(
                        "channel '{}': data.channel is only supported for Parquet data sources",
                        ch.name
                    );
                }
                let tree_name = ch.data.tree_name().with_context(|| {
                    format!("channel '{}' uses ROOT format but has no tree name", ch.name)
                })?;
                let root = RootFile::open(&data_path)
                    .with_context(|| format!("failed to open ROOT file {}", data_path.display()))?;
                EventStore::from_tree(
                    &root,
                    tree_name,
                    &observables,
                    ch.data.selection.as_deref(),
                    ch.data.weight.as_deref(),
                )
                .with_context(|| {
                    format!(
                        "failed to load channel '{}' data from {}",
                        ch.name,
                        data_path.display()
                    )
                })?
            }
            #[cfg(feature = "arrow-io")]
            DataFormat::Parquet => {
                if ch.data.selection.is_some() {
                    anyhow::bail!(
                        "channel '{}': selection expressions are not supported for Parquet data sources (apply selections before writing Parquet)",
                        ch.name
                    );
                }
                if ch.data.weight.is_some() {
                    anyhow::bail!(
                        "channel '{}': data.weight is not supported for Parquet data sources (write weights into the Parquet file instead)",
                        ch.name
                    );
                }
                if let Some(ch_name) = ch.data.channel.as_deref() {
                    EventStore::from_parquet_channel(&data_path, Some(&observables), ch_name)
                        .with_context(|| {
                            format!(
                                "failed to load channel '{}' data from Parquet {} (channel='{}')",
                                ch.name,
                                data_path.display(),
                                ch_name
                            )
                        })?
                } else {
                    EventStore::from_parquet(&data_path, Some(&observables)).with_context(|| {
                        format!(
                            "failed to load channel '{}' data from Parquet {}",
                            ch.name,
                            data_path.display()
                        )
                    })?
                }
            }
            #[cfg(not(feature = "arrow-io"))]
            DataFormat::Parquet => {
                anyhow::bail!(
                    "channel '{}' references a Parquet file but the 'arrow-io' feature is not enabled",
                    ch.name
                );
            }
        };

        let store = Arc::new(store);

        let observable_names: HashMap<&str, ()> =
            observables.iter().map(|o| (o.name.as_str(), ())).collect();
        let observable_specs: HashMap<&str, &ObservableSpec> =
            observables.iter().map(|o| (o.name.as_str(), o)).collect();

        let mut processes = Vec::with_capacity(ch.processes.len());
        for proc in &ch.processes {
            let validate_param_min_strict = |param_name: &str, min_exclusive: f64, rule: &str| {
                let pidx = *index_by_name.get(param_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "process '{}' references unknown parameter '{}' (channel '{}')",
                        proc.name,
                        param_name,
                        ch.name
                    )
                })?;
                let p = &parameters[pidx];
                if !p.init.is_finite() || p.init <= min_exclusive {
                    anyhow::bail!(
                        "process '{}' parameter '{}' init={} violates {} (channel '{}')",
                        proc.name,
                        param_name,
                        p.init,
                        rule,
                        ch.name
                    );
                }
                if !p.bounds.0.is_finite() || !p.bounds.1.is_finite() || p.bounds.0 <= min_exclusive
                {
                    anyhow::bail!(
                        "process '{}' parameter '{}' bounds {:?} violate {} (channel '{}')",
                        proc.name,
                        param_name,
                        p.bounds,
                        rule,
                        ch.name
                    );
                }
                Ok::<(), anyhow::Error>(())
            };

            let (pdf, shape_param_names, extra_yield_modifiers, observable_name): (
                Arc<dyn UnbinnedPdf>,
                Vec<String>,
                Vec<RateModifier>,
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
                        Vec::new(),
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
                    validate_param_min_strict(
                        &params[2],
                        0.0,
                        "CrystalBall requires alpha > 0 (strictly)",
                    )?;
                    validate_param_min_strict(
                        &params[3],
                        1.0,
                        "CrystalBall requires n > 1 (strictly, e.g. lower bound 1.01)",
                    )?;
                    (
                        Arc::new(CrystalBallPdf::new(observable.clone())),
                        params.clone(),
                        Vec::new(),
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
                    validate_param_min_strict(
                        &params[2],
                        0.0,
                        "DoubleCrystalBall requires alpha_l > 0 (strictly)",
                    )?;
                    validate_param_min_strict(
                        &params[3],
                        1.0,
                        "DoubleCrystalBall requires n_l > 1 (strictly, e.g. lower bound 1.01)",
                    )?;
                    validate_param_min_strict(
                        &params[4],
                        0.0,
                        "DoubleCrystalBall requires alpha_r > 0 (strictly)",
                    )?;
                    validate_param_min_strict(
                        &params[5],
                        1.0,
                        "DoubleCrystalBall requires n_r > 1 (strictly, e.g. lower bound 1.01)",
                    )?;
                    (
                        Arc::new(DoubleCrystalBallPdf::new(observable.clone())),
                        params.clone(),
                        Vec::new(),
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
                        Vec::new(),
                        observable.clone(),
                    )
                }
                PdfSpec::Argus { observable, params } => {
                    if params.len() != 2 {
                        anyhow::bail!(
                            "process '{}' argus params must have length 2 (c,p), got {}",
                            proc.name,
                            params.len()
                        );
                    }
                    (
                        Arc::new(ArgusPdf::new(observable.clone())),
                        params.clone(),
                        Vec::new(),
                        observable.clone(),
                    )
                }
                PdfSpec::Voigtian { observable, params } => {
                    if params.len() != 3 {
                        anyhow::bail!(
                            "process '{}' voigtian params must have length 3 (mu,sigma,gamma), got {}",
                            proc.name,
                            params.len()
                        );
                    }
                    (
                        Arc::new(VoigtianPdf::new(observable.clone())),
                        params.clone(),
                        Vec::new(),
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
                        Vec::new(),
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
                        Vec::new(),
                        observable.clone(),
                    )
                }
                PdfSpec::Spline { observable, knots_x, knots_y } => {
                    let Some(obs) = observable_specs.get(observable.as_str()) else {
                        anyhow::bail!(
                            "process '{}' references unknown observable '{}' (channel '{}')",
                            proc.name,
                            observable,
                            ch.name
                        );
                    };

                    // Require that spline support matches event support for correct normalization.
                    let eps = 1e-12;
                    let x_min = *knots_x.first().unwrap_or(&f64::NAN);
                    let x_max = *knots_x.last().unwrap_or(&f64::NAN);
                    if (x_min - obs.bounds.0).abs() > eps || (x_max - obs.bounds.1).abs() > eps {
                        anyhow::bail!(
                            "process '{}' spline support [{x_min}, {x_max}] must match observable '{}' bounds {:?} (channel '{}')",
                            proc.name,
                            observable,
                            obs.bounds,
                            ch.name
                        );
                    }

                    (
                        Arc::new(SplinePdf::from_knots(
                            observable.clone(),
                            knots_x.clone(),
                            knots_y.clone(),
                        )?),
                        Vec::new(),
                        Vec::new(),
                        observable.clone(),
                    )
                }
                PdfSpec::Product { components } => {
                    if components.is_empty() {
                        anyhow::bail!(
                            "process '{}' product requires a non-empty components list (channel '{}')",
                            proc.name,
                            ch.name
                        );
                    }

                    let mut compiled = Vec::<Arc<dyn UnbinnedPdf>>::with_capacity(components.len());
                    let mut shape_param_names = Vec::<String>::new();
                    let mut extra_yield_modifiers = Vec::<RateModifier>::new();
                    let mut claimed = HashSet::<String>::new();
                    let mut primary_obs: Option<String> = None;

                    for (i, c) in components.iter().enumerate() {
                        let (pdf_i, shape_i, extra_i, obs_i): (
                            Arc<dyn UnbinnedPdf>,
                            Vec<String>,
                            Vec<RateModifier>,
                            String,
                        ) = match c {
                            PdfSpec::Gaussian { observable, params } => {
                                if params.len() != 2 {
                                    anyhow::bail!(
                                        "process '{}' product component {} gaussian params must have length 2 (mu,sigma), got {} (channel '{}')",
                                        proc.name,
                                        i,
                                        params.len(),
                                        ch.name
                                    );
                                }
                                (
                                    Arc::new(GaussianPdf::new(observable.clone())),
                                    params.clone(),
                                    Vec::new(),
                                    observable.clone(),
                                )
                            }
                            PdfSpec::CrystalBall { observable, params } => {
                                if params.len() != 4 {
                                    anyhow::bail!(
                                        "process '{}' product component {} crystal_ball params must have length 4 (mu,sigma,alpha,n), got {} (channel '{}')",
                                        proc.name,
                                        i,
                                        params.len(),
                                        ch.name
                                    );
                                }
                                (
                                    Arc::new(CrystalBallPdf::new(observable.clone())),
                                    params.clone(),
                                    Vec::new(),
                                    observable.clone(),
                                )
                            }
                            PdfSpec::DoubleCrystalBall { observable, params } => {
                                if params.len() != 6 {
                                    anyhow::bail!(
                                        "process '{}' product component {} double_crystal_ball params must have length 6 (mu,sigma,alpha_l,n_l,alpha_r,n_r), got {} (channel '{}')",
                                        proc.name,
                                        i,
                                        params.len(),
                                        ch.name
                                    );
                                }
                                (
                                    Arc::new(DoubleCrystalBallPdf::new(observable.clone())),
                                    params.clone(),
                                    Vec::new(),
                                    observable.clone(),
                                )
                            }
                            PdfSpec::Exponential { observable, params } => {
                                if params.len() != 1 {
                                    anyhow::bail!(
                                        "process '{}' product component {} exponential params must have length 1 (lambda), got {} (channel '{}')",
                                        proc.name,
                                        i,
                                        params.len(),
                                        ch.name
                                    );
                                }
                                (
                                    Arc::new(ExponentialPdf::new(observable.clone())),
                                    params.clone(),
                                    Vec::new(),
                                    observable.clone(),
                                )
                            }
                            PdfSpec::Argus { observable, params } => {
                                if params.len() != 2 {
                                    anyhow::bail!(
                                        "process '{}' product component {} argus params must have length 2 (c,p), got {} (channel '{}')",
                                        proc.name,
                                        i,
                                        params.len(),
                                        ch.name
                                    );
                                }
                                (
                                    Arc::new(ArgusPdf::new(observable.clone())),
                                    params.clone(),
                                    Vec::new(),
                                    observable.clone(),
                                )
                            }
                            PdfSpec::Voigtian { observable, params } => {
                                if params.len() != 3 {
                                    anyhow::bail!(
                                        "process '{}' product component {} voigtian params must have length 3 (mu,sigma,gamma), got {} (channel '{}')",
                                        proc.name,
                                        i,
                                        params.len(),
                                        ch.name
                                    );
                                }
                                (
                                    Arc::new(VoigtianPdf::new(observable.clone())),
                                    params.clone(),
                                    Vec::new(),
                                    observable.clone(),
                                )
                            }
                            PdfSpec::Chebyshev { observable, params } => {
                                if params.is_empty() {
                                    anyhow::bail!(
                                        "process '{}' product component {} chebyshev params must be non-empty (c_1..c_order), got 0 (channel '{}')",
                                        proc.name,
                                        i,
                                        ch.name
                                    );
                                }
                                (
                                    Arc::new(ChebyshevPdf::new(observable.clone(), params.len())?),
                                    params.clone(),
                                    Vec::new(),
                                    observable.clone(),
                                )
                            }
                            PdfSpec::Histogram {
                                observable,
                                bin_edges,
                                bin_content,
                                pseudo_count,
                            } => {
                                let pc = pseudo_count.unwrap_or(0.0);
                                (
                                    Arc::new(HistogramPdf::from_edges_and_contents(
                                        observable.clone(),
                                        bin_edges.clone(),
                                        bin_content.clone(),
                                        pc,
                                    )?),
                                    Vec::new(),
                                    Vec::new(),
                                    observable.clone(),
                                )
                            }
                            PdfSpec::Kde { observable, bandwidth, centers, weights } => {
                                let Some(obs) = observable_specs.get(observable.as_str()) else {
                                    anyhow::bail!(
                                        "process '{}' product component {} references unknown observable '{}' (channel '{}')",
                                        proc.name,
                                        i,
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
                                    Vec::new(),
                                    observable.clone(),
                                )
                            }
                            PdfSpec::Spline { observable, knots_x, knots_y } => {
                                let Some(obs) = observable_specs.get(observable.as_str()) else {
                                    anyhow::bail!(
                                        "process '{}' product component {} references unknown observable '{}' (channel '{}')",
                                        proc.name,
                                        i,
                                        observable,
                                        ch.name
                                    );
                                };
                                let eps = 1e-12;
                                let x_min = *knots_x.first().unwrap_or(&f64::NAN);
                                let x_max = *knots_x.last().unwrap_or(&f64::NAN);
                                if (x_min - obs.bounds.0).abs() > eps
                                    || (x_max - obs.bounds.1).abs() > eps
                                {
                                    anyhow::bail!(
                                        "process '{}' product component {} spline support [{x_min}, {x_max}] must match observable '{}' bounds {:?} (channel '{}')",
                                        proc.name,
                                        i,
                                        observable,
                                        obs.bounds,
                                        ch.name
                                    );
                                }
                                (
                                    Arc::new(SplinePdf::from_knots(
                                        observable.clone(),
                                        knots_x.clone(),
                                        knots_y.clone(),
                                    )?),
                                    Vec::new(),
                                    Vec::new(),
                                    observable.clone(),
                                )
                            }
                            PdfSpec::Product { .. } => {
                                anyhow::bail!(
                                    "process '{}' product component {}: nested product is not supported (channel '{}')",
                                    proc.name,
                                    i,
                                    ch.name
                                );
                            }
                            other => {
                                anyhow::bail!(
                                    "process '{}' product component {}: unsupported pdf {:?}. \
                                     Product components must be inline PDFs (no *_from_tree and no neural PDFs) (channel '{}')",
                                    proc.name,
                                    i,
                                    other,
                                    ch.name
                                );
                            }
                        };

                        if !observable_names.contains_key(obs_i.as_str()) {
                            anyhow::bail!(
                                "process '{}' product component {} references unknown observable '{}' (channel '{}')",
                                proc.name,
                                i,
                                obs_i,
                                ch.name
                            );
                        }
                        if !claimed.insert(obs_i.clone()) {
                            anyhow::bail!(
                                "process '{}' product requires disjoint observables, but '{}' is used more than once (channel '{}')",
                                proc.name,
                                obs_i,
                                ch.name
                            );
                        }
                        if primary_obs.is_none() {
                            primary_obs = Some(obs_i.clone());
                        }

                        compiled.push(pdf_i);
                        shape_param_names.extend(shape_i);
                        extra_yield_modifiers.extend(extra_i);
                    }

                    let product = crate::pdf::ProductPdf::new(compiled)?;
                    (
                        Arc::new(product) as Arc<dyn UnbinnedPdf>,
                        shape_param_names,
                        extra_yield_modifiers,
                        primary_obs.unwrap_or_else(|| "<product>".to_string()),
                    )
                }
                PdfSpec::HistogramFromTree {
                    observable,
                    bin_edges,
                    pseudo_count,
                    source,
                    max_events,
                    weight_systematics,
                    horizontal_systematics,
                } => {
                    if source.effective_format() != DataFormat::Root {
                        anyhow::bail!(
                            "process '{}' histogram_from_tree requires a ROOT source (got Parquet): source.file={}",
                            proc.name,
                            source.file.display()
                        );
                    }
                    if source.channel.is_some() {
                        anyhow::bail!(
                            "process '{}' histogram_from_tree: source.channel is not supported (ROOT sources do not have channels)",
                            proc.name
                        );
                    }
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

                    let mut train_obs = Vec::<ObservableSpec>::with_capacity(
                        1 + horizontal_systematics.len().saturating_mul(2),
                    );
                    train_obs.push((*obs).clone());

                    let mut extra_weight_exprs =
                        Vec::<&str>::with_capacity(weight_systematics.len() * 2);
                    let mut seen_params = HashSet::<&str>::with_capacity(weight_systematics.len());
                    let mut weight_shape_params =
                        HashSet::<&str>::with_capacity(weight_systematics.len());
                    for s in weight_systematics {
                        if s.param.trim().is_empty() {
                            anyhow::bail!(
                                "process '{}' histogram_from_tree: weight systematic param name must be non-empty",
                                proc.name
                            );
                        }
                        if !seen_params.insert(s.param.as_str()) {
                            anyhow::bail!(
                                "process '{}' histogram_from_tree: duplicate weight systematic param '{}'",
                                proc.name,
                                s.param
                            );
                        }
                        if s.up.trim().is_empty() || s.down.trim().is_empty() {
                            anyhow::bail!(
                                "process '{}' histogram_from_tree: weight systematic '{}' requires non-empty up/down expressions",
                                proc.name,
                                s.param
                            );
                        }
                        if !s.apply_to_shape && !s.apply_to_yield {
                            anyhow::bail!(
                                "process '{}' histogram_from_tree: weight systematic '{}' is a no-op (apply_to_shape=false and apply_to_yield=false)",
                                proc.name,
                                s.param
                            );
                        }
                        if s.apply_to_shape {
                            weight_shape_params.insert(s.param.as_str());
                        }
                        extra_weight_exprs.push(s.up.as_str());
                        extra_weight_exprs.push(s.down.as_str());
                    }

                    let mut horizontal_cols =
                        Vec::<(String, String, String, HistoSysInterpCode)>::new();
                    if !horizontal_systematics.is_empty() {
                        horizontal_cols.reserve(horizontal_systematics.len());

                        let mut seen_horizontal_params =
                            HashSet::<&str>::with_capacity(horizontal_systematics.len());
                        for s in horizontal_systematics {
                            if s.param.trim().is_empty() {
                                anyhow::bail!(
                                    "process '{}' histogram_from_tree: horizontal systematic param name must be non-empty",
                                    proc.name
                                );
                            }
                            if !seen_horizontal_params.insert(s.param.as_str()) {
                                anyhow::bail!(
                                    "process '{}' histogram_from_tree: duplicate horizontal systematic param '{}'",
                                    proc.name,
                                    s.param
                                );
                            }
                            if weight_shape_params.contains(s.param.as_str()) {
                                anyhow::bail!(
                                    "process '{}' histogram_from_tree: horizontal systematic '{}' conflicts with weight_systematics shape modifier of the same param (choose one shape source for this nuisance)",
                                    proc.name,
                                    s.param
                                );
                            }
                            let _ = *index_by_name.get(s.param.as_str()).ok_or_else(|| {
                                anyhow::anyhow!(
                                    "process '{}' histogram_from_tree references unknown horizontal systematic nuisance parameter '{}'",
                                    proc.name,
                                    s.param
                                )
                            })?;

                            if s.up.trim().is_empty() || s.down.trim().is_empty() {
                                anyhow::bail!(
                                    "process '{}' histogram_from_tree: horizontal systematic '{}' requires non-empty up/down expressions",
                                    proc.name,
                                    s.param
                                );
                            }

                            let up_name = format!("__{observable}__{}__up", s.param);
                            let down_name = format!("__{observable}__{}__down", s.param);
                            if observable_specs.contains_key(up_name.as_str())
                                || observable_specs.contains_key(down_name.as_str())
                            {
                                anyhow::bail!(
                                    "process '{}' histogram_from_tree: internal horizontal systematic column name collision for '{}'",
                                    proc.name,
                                    s.param
                                );
                            }

                            train_obs.push(ObservableSpec::expression(
                                up_name.clone(),
                                s.up.clone(),
                                obs.bounds,
                            ));
                            train_obs.push(ObservableSpec::expression(
                                down_name.clone(),
                                s.down.clone(),
                                obs.bounds,
                            ));

                            horizontal_cols.push((
                                s.param.clone(),
                                up_name,
                                down_name,
                                s.interp.as_unbinned(),
                            ));
                        }
                    }

                    let src_tree_name = source.tree_name().with_context(|| {
                        format!(
                            "process '{}' histogram_from_tree source requires tree name",
                            proc.name
                        )
                    })?;
                    let (train, mut extra_weights) = EventStore::from_tree_with_extra_weights(
                        &src_root,
                        src_tree_name,
                        &train_obs,
                        source.selection.as_deref(),
                        source.weight.as_deref(),
                        &extra_weight_exprs,
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

                    let mut horizontal_values =
                        Vec::<(String, Vec<f64>, Vec<f64>, HistoSysInterpCode)>::new();
                    if !horizontal_cols.is_empty() {
                        horizontal_values.reserve(horizontal_cols.len());
                        for (param, up_col, down_col, interp_code) in &horizontal_cols {
                            let x_up = train
                                .column(up_col.as_str())
                                .ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "histogram training data missing column '{}'",
                                        up_col
                                    )
                                })?
                                .to_vec();
                            let x_down = train
                                .column(down_col.as_str())
                                .ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "histogram training data missing column '{}'",
                                        down_col
                                    )
                                })?
                                .to_vec();
                            horizontal_values.push((param.clone(), x_up, x_down, *interp_code));
                        }
                    }

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
                        for w in &mut extra_weights {
                            w.truncate(keep);
                        }
                        for (_param, x_up, x_down, _interp_code) in &mut horizontal_values {
                            x_up.truncate(keep);
                            x_down.truncate(keep);
                        }
                    }

                    let n_bins = bin_edges.len().saturating_sub(1);
                    if n_bins == 0 {
                        anyhow::bail!(
                            "process '{}' histogram_from_tree requires at least 2 bin_edges",
                            proc.name
                        );
                    }

                    let bin_index = |x: f64, label: &str| -> Result<usize> {
                        if x >= edge_max {
                            return Ok(n_bins - 1);
                        }
                        // `k` is the number of edges <= x, so bin index is k-1.
                        let k = bin_edges.partition_point(|e| *e <= x);
                        if k == 0 || k > n_bins {
                            anyhow::bail!(
                                "process '{}' histogram_from_tree: {label} x out of range: x={x} not in [{edge_min}, {edge_max}]",
                                proc.name
                            );
                        }
                        Ok(k - 1)
                    };

                    // Precompute bin indices once.
                    let mut bin_idx = Vec::with_capacity(xs.len());
                    for &x in &xs {
                        bin_idx.push(bin_index(x, "nominal")?);
                    }

                    let mut nominal_bin_content = vec![0.0f64; n_bins];
                    let mut total_nom = 0.0f64;
                    for (i, &idx) in bin_idx.iter().enumerate() {
                        let w0 = ws.as_ref().map(|w| w[i]).unwrap_or(1.0);
                        total_nom += w0;
                        nominal_bin_content[idx] += w0;
                    }

                    if !(total_nom.is_finite() && total_nom > 0.0) {
                        anyhow::bail!(
                            "process '{}' histogram_from_tree: nominal sum of weights must be finite and > 0, got {total_nom}",
                            proc.name
                        );
                    }

                    let pc = pseudo_count.unwrap_or(0.0);

                    let mut shape_param_names = Vec::<String>::new();
                    let mut systematics = Vec::<HistogramSystematic>::new();
                    let mut yield_modifiers = Vec::<RateModifier>::new();

                    if !weight_systematics.is_empty() {
                        if extra_weights.len() != weight_systematics.len() * 2 {
                            anyhow::bail!(
                                "process '{}' histogram_from_tree: internal error: expected {} extra weight columns, got {}",
                                proc.name,
                                weight_systematics.len() * 2,
                                extra_weights.len()
                            );
                        }

                        for (sidx, s) in weight_systematics.iter().enumerate() {
                            let interp_code = s.interp.as_unbinned();

                            let up_ratio = &extra_weights[2 * sidx];
                            let down_ratio = &extra_weights[2 * sidx + 1];

                            if up_ratio.len() != xs.len() || down_ratio.len() != xs.len() {
                                anyhow::bail!(
                                    "process '{}' histogram_from_tree: weight systematic '{}' length mismatch: expected {} events, got up={} down={}",
                                    proc.name,
                                    s.param,
                                    xs.len(),
                                    up_ratio.len(),
                                    down_ratio.len()
                                );
                            }

                            let needs_shape = s.apply_to_shape;
                            let needs_yield = s.apply_to_yield;

                            let mut up_bins =
                                if needs_shape { vec![0.0f64; n_bins] } else { Vec::new() };
                            let mut down_bins =
                                if needs_shape { vec![0.0f64; n_bins] } else { Vec::new() };
                            let mut total_up = 0.0f64;
                            let mut total_down = 0.0f64;

                            for (i, &idx) in bin_idx.iter().enumerate() {
                                let w0 = ws.as_ref().map(|w| w[i]).unwrap_or(1.0);
                                let wu = w0 * up_ratio[i];
                                let wd = w0 * down_ratio[i];

                                if needs_shape {
                                    up_bins[idx] += wu;
                                    down_bins[idx] += wd;
                                }
                                if needs_yield {
                                    total_up += wu;
                                    total_down += wd;
                                }
                            }

                            if needs_shape {
                                shape_param_names.push(s.param.clone());
                                systematics.push(HistogramSystematic {
                                    down: down_bins,
                                    up: up_bins,
                                    interp_code,
                                });
                            }

                            if needs_yield {
                                if !(total_up.is_finite()
                                    && total_up > 0.0
                                    && total_down.is_finite()
                                    && total_down > 0.0)
                                {
                                    anyhow::bail!(
                                        "process '{}' histogram_from_tree: weight systematic '{}' totals must be finite and > 0, got up={total_up} down={total_down}",
                                        proc.name,
                                        s.param
                                    );
                                }
                                let hi = total_up / total_nom;
                                let lo = total_down / total_nom;
                                if !(hi.is_finite() && hi > 0.0 && lo.is_finite() && lo > 0.0) {
                                    anyhow::bail!(
                                        "process '{}' histogram_from_tree: weight systematic '{}' yield factors must be finite and > 0, got lo={lo}, hi={hi}",
                                        proc.name,
                                        s.param
                                    );
                                }
                                let alpha_index = *index_by_name.get(&s.param).ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "process '{}' references unknown weight systematic nuisance parameter '{}'",
                                        proc.name,
                                        s.param
                                    )
                                })?;
                                yield_modifiers.push(RateModifier::WeightSys {
                                    alpha_index,
                                    lo,
                                    hi,
                                    interp_code,
                                });
                            }
                        }
                    }

                    if !horizontal_values.is_empty() {
                        for (param, x_up, x_down, interp_code) in horizontal_values {
                            if shape_param_names.iter().any(|p| p == &param) {
                                anyhow::bail!(
                                    "process '{}' histogram_from_tree: duplicate shape systematic param '{}'",
                                    proc.name,
                                    param
                                );
                            }

                            if x_up.len() != xs.len() || x_down.len() != xs.len() {
                                anyhow::bail!(
                                    "process '{}' histogram_from_tree: horizontal systematic '{}' length mismatch: expected {} events, got up={} down={}",
                                    proc.name,
                                    param,
                                    xs.len(),
                                    x_up.len(),
                                    x_down.len()
                                );
                            }

                            let label_up = format!("horizontal systematic '{param}'/up");
                            let label_down = format!("horizontal systematic '{param}'/down");

                            let mut up_bins = vec![0.0f64; n_bins];
                            let mut down_bins = vec![0.0f64; n_bins];
                            for i in 0..xs.len() {
                                let w0 = ws.as_ref().map(|w| w[i]).unwrap_or(1.0);
                                let idx_up = bin_index(x_up[i], &label_up)?;
                                let idx_down = bin_index(x_down[i], &label_down)?;
                                up_bins[idx_up] += w0;
                                down_bins[idx_down] += w0;
                            }

                            shape_param_names.push(param);
                            systematics.push(HistogramSystematic {
                                down: down_bins,
                                up: up_bins,
                                interp_code,
                            });
                        }
                    }

                    let pdf: Arc<dyn UnbinnedPdf> = if systematics.is_empty() {
                        Arc::new(HistogramPdf::from_edges_and_contents(
                            observable.clone(),
                            bin_edges.clone(),
                            nominal_bin_content,
                            pc,
                        )?)
                    } else {
                        Arc::new(MorphingHistogramPdf::new(
                            observable.clone(),
                            bin_edges.clone(),
                            nominal_bin_content,
                            systematics,
                            pc,
                        )?)
                    };

                    (pdf, shape_param_names, yield_modifiers, observable.clone())
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
                        Vec::new(),
                        observable.clone(),
                    )
                }
                PdfSpec::KdeFromTree {
                    observable,
                    bandwidth,
                    source,
                    max_events,
                    weight_systematics,
                    horizontal_systematics,
                } => {
                    if source.effective_format() != DataFormat::Root {
                        anyhow::bail!(
                            "process '{}' kde_from_tree requires a ROOT source (got Parquet): source.file={}",
                            proc.name,
                            source.file.display()
                        );
                    }
                    if source.channel.is_some() {
                        anyhow::bail!(
                            "process '{}' kde_from_tree: source.channel is not supported (ROOT sources do not have channels)",
                            proc.name
                        );
                    }
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

                    let mut train_obs = Vec::<ObservableSpec>::with_capacity(
                        1 + horizontal_systematics.len().saturating_mul(2),
                    );
                    train_obs.push((*obs).clone());

                    let mut extra_weight_exprs =
                        Vec::<&str>::with_capacity(weight_systematics.len() * 2);
                    let mut seen_params = HashSet::<&str>::with_capacity(weight_systematics.len());
                    let mut weight_shape_params =
                        HashSet::<&str>::with_capacity(weight_systematics.len());
                    for s in weight_systematics {
                        if s.param.trim().is_empty() {
                            anyhow::bail!(
                                "process '{}' kde_from_tree: weight systematic param name must be non-empty",
                                proc.name
                            );
                        }
                        if !seen_params.insert(s.param.as_str()) {
                            anyhow::bail!(
                                "process '{}' kde_from_tree: duplicate weight systematic param '{}'",
                                proc.name,
                                s.param
                            );
                        }
                        if s.up.trim().is_empty() || s.down.trim().is_empty() {
                            anyhow::bail!(
                                "process '{}' kde_from_tree: weight systematic '{}' requires non-empty up/down expressions",
                                proc.name,
                                s.param
                            );
                        }
                        if !s.apply_to_shape && !s.apply_to_yield {
                            anyhow::bail!(
                                "process '{}' kde_from_tree: weight systematic '{}' is a no-op (apply_to_shape=false and apply_to_yield=false)",
                                proc.name,
                                s.param
                            );
                        }
                        if s.apply_to_shape {
                            weight_shape_params.insert(s.param.as_str());
                        }
                        extra_weight_exprs.push(s.up.as_str());
                        extra_weight_exprs.push(s.down.as_str());
                    }

                    let mut horizontal_cols =
                        Vec::<(String, String, String, HistoSysInterpCode)>::new();
                    if !horizontal_systematics.is_empty() {
                        horizontal_cols.reserve(horizontal_systematics.len());

                        let mut seen_horizontal_params =
                            HashSet::<&str>::with_capacity(horizontal_systematics.len());
                        for s in horizontal_systematics {
                            if s.param.trim().is_empty() {
                                anyhow::bail!(
                                    "process '{}' kde_from_tree: horizontal systematic param name must be non-empty",
                                    proc.name
                                );
                            }
                            if !seen_horizontal_params.insert(s.param.as_str()) {
                                anyhow::bail!(
                                    "process '{}' kde_from_tree: duplicate horizontal systematic param '{}'",
                                    proc.name,
                                    s.param
                                );
                            }
                            if weight_shape_params.contains(s.param.as_str()) {
                                anyhow::bail!(
                                    "process '{}' kde_from_tree: horizontal systematic '{}' conflicts with weight_systematics shape modifier of the same param (choose one shape source for this nuisance)",
                                    proc.name,
                                    s.param
                                );
                            }
                            let _ = *index_by_name.get(s.param.as_str()).ok_or_else(|| {
                                anyhow::anyhow!(
                                    "process '{}' kde_from_tree references unknown horizontal systematic nuisance parameter '{}'",
                                    proc.name,
                                    s.param
                                )
                            })?;

                            if s.up.trim().is_empty() || s.down.trim().is_empty() {
                                anyhow::bail!(
                                    "process '{}' kde_from_tree: horizontal systematic '{}' requires non-empty up/down expressions",
                                    proc.name,
                                    s.param
                                );
                            }

                            let up_name = format!("__{observable}__{}__up", s.param);
                            let down_name = format!("__{observable}__{}__down", s.param);
                            if observable_specs.contains_key(up_name.as_str())
                                || observable_specs.contains_key(down_name.as_str())
                            {
                                anyhow::bail!(
                                    "process '{}' kde_from_tree: internal horizontal systematic column name collision for '{}'",
                                    proc.name,
                                    s.param
                                );
                            }

                            train_obs.push(ObservableSpec::expression(
                                up_name.clone(),
                                s.up.clone(),
                                obs.bounds,
                            ));
                            train_obs.push(ObservableSpec::expression(
                                down_name.clone(),
                                s.down.clone(),
                                obs.bounds,
                            ));

                            horizontal_cols.push((
                                s.param.clone(),
                                up_name,
                                down_name,
                                s.interp.as_unbinned(),
                            ));
                        }
                    }

                    let src_tree_name = source.tree_name().with_context(|| {
                        format!("process '{}' kde_from_tree source requires tree name", proc.name)
                    })?;
                    let (train, mut extra_weights) = EventStore::from_tree_with_extra_weights(
                        &src_root,
                        src_tree_name,
                        &train_obs,
                        source.selection.as_deref(),
                        source.weight.as_deref(),
                        &extra_weight_exprs,
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

                    let mut horizontal_values =
                        Vec::<(String, Vec<f64>, Vec<f64>, HistoSysInterpCode)>::new();
                    if !horizontal_cols.is_empty() {
                        horizontal_values.reserve(horizontal_cols.len());
                        for (param, up_col, down_col, interp_code) in &horizontal_cols {
                            let x_up = train
                                .column(up_col.as_str())
                                .ok_or_else(|| {
                                    anyhow::anyhow!("KDE training data missing column '{}'", up_col)
                                })?
                                .to_vec();
                            let x_down = train
                                .column(down_col.as_str())
                                .ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "KDE training data missing column '{}'",
                                        down_col
                                    )
                                })?
                                .to_vec();
                            horizontal_values.push((param.clone(), x_up, x_down, *interp_code));
                        }
                    }

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
                        for w in &mut extra_weights {
                            w.truncate(keep);
                        }
                        for (_param, x_up, x_down, _interp_code) in &mut horizontal_values {
                            x_up.truncate(keep);
                            x_down.truncate(keep);
                        }
                    }

                    let weights_nom = match weights {
                        Some(w) => w,
                        None => vec![1.0f64; centers.len()],
                    };
                    let total_nom: f64 = weights_nom.iter().sum();
                    if !(total_nom.is_finite() && total_nom > 0.0) {
                        anyhow::bail!(
                            "process '{}' kde_from_tree: nominal sum of weights must be finite and > 0, got {total_nom}",
                            proc.name
                        );
                    }

                    let mut shape_param_names = Vec::<String>::new();
                    let mut weight_shape_systematics = Vec::<KdeWeightSystematic>::new();
                    let mut center_systematics = Vec::<KdeHorizontalSystematic>::new();
                    let mut yield_modifiers = Vec::<RateModifier>::new();

                    if !weight_systematics.is_empty() {
                        if extra_weights.len() != weight_systematics.len() * 2 {
                            anyhow::bail!(
                                "process '{}' kde_from_tree: internal error: expected {} extra weight columns, got {}",
                                proc.name,
                                weight_systematics.len() * 2,
                                extra_weights.len()
                            );
                        }

                        for (sidx, s) in weight_systematics.iter().enumerate() {
                            let interp_code = s.interp.as_unbinned();
                            let up_ratio = &extra_weights[2 * sidx];
                            let down_ratio = &extra_weights[2 * sidx + 1];
                            if up_ratio.len() != centers.len() || down_ratio.len() != centers.len()
                            {
                                anyhow::bail!(
                                    "process '{}' kde_from_tree: weight systematic '{}' length mismatch: expected {} events, got up={} down={}",
                                    proc.name,
                                    s.param,
                                    centers.len(),
                                    up_ratio.len(),
                                    down_ratio.len()
                                );
                            }

                            if s.apply_to_shape {
                                shape_param_names.push(s.param.clone());
                                let mut up_w = Vec::with_capacity(centers.len());
                                let mut down_w = Vec::with_capacity(centers.len());
                                for i in 0..centers.len() {
                                    up_w.push(weights_nom[i] * up_ratio[i]);
                                    down_w.push(weights_nom[i] * down_ratio[i]);
                                }
                                weight_shape_systematics.push(KdeWeightSystematic {
                                    down: down_w,
                                    up: up_w,
                                    interp_code,
                                });
                            }

                            if s.apply_to_yield {
                                let mut total_up = 0.0f64;
                                let mut total_down = 0.0f64;
                                for i in 0..centers.len() {
                                    total_up += weights_nom[i] * up_ratio[i];
                                    total_down += weights_nom[i] * down_ratio[i];
                                }
                                if !(total_up.is_finite()
                                    && total_up > 0.0
                                    && total_down.is_finite()
                                    && total_down > 0.0)
                                {
                                    anyhow::bail!(
                                        "process '{}' kde_from_tree: weight systematic '{}' totals must be finite and > 0, got up={total_up} down={total_down}",
                                        proc.name,
                                        s.param
                                    );
                                }
                                let hi = total_up / total_nom;
                                let lo = total_down / total_nom;
                                if !(hi.is_finite() && hi > 0.0 && lo.is_finite() && lo > 0.0) {
                                    anyhow::bail!(
                                        "process '{}' kde_from_tree: weight systematic '{}' yield factors must be finite and > 0, got lo={lo}, hi={hi}",
                                        proc.name,
                                        s.param
                                    );
                                }
                                let alpha_index = *index_by_name.get(&s.param).ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "process '{}' references unknown weight systematic nuisance parameter '{}'",
                                        proc.name,
                                        s.param
                                    )
                                })?;
                                yield_modifiers.push(RateModifier::WeightSys {
                                    alpha_index,
                                    lo,
                                    hi,
                                    interp_code,
                                });
                            }
                        }
                    }

                    if !horizontal_values.is_empty() {
                        for (param, x_up, x_down, interp_code) in horizontal_values {
                            if shape_param_names.iter().any(|p| p == &param) {
                                anyhow::bail!(
                                    "process '{}' kde_from_tree: duplicate shape systematic param '{}'",
                                    proc.name,
                                    param
                                );
                            }
                            if x_up.len() != centers.len() || x_down.len() != centers.len() {
                                anyhow::bail!(
                                    "process '{}' kde_from_tree: horizontal systematic '{}' length mismatch: expected {} events, got up={} down={}",
                                    proc.name,
                                    param,
                                    centers.len(),
                                    x_up.len(),
                                    x_down.len()
                                );
                            }

                            shape_param_names.push(param);
                            center_systematics.push(KdeHorizontalSystematic {
                                down: x_down,
                                up: x_up,
                                interp_code,
                            });
                        }
                    }

                    let pdf: Arc<dyn UnbinnedPdf> = if center_systematics.is_empty() {
                        if weight_shape_systematics.is_empty() {
                            Arc::new(KdePdf::from_samples(
                                observable.clone(),
                                obs.bounds,
                                centers,
                                Some(weights_nom),
                                *bandwidth,
                            )?)
                        } else {
                            Arc::new(MorphingKdePdf::new(
                                observable.clone(),
                                obs.bounds,
                                centers,
                                weights_nom,
                                weight_shape_systematics,
                                *bandwidth,
                            )?)
                        }
                    } else {
                        Arc::new(HorizontalMorphingKdePdf::new(
                            observable.clone(),
                            obs.bounds,
                            centers,
                            weights_nom,
                            weight_shape_systematics,
                            center_systematics,
                            *bandwidth,
                        )?)
                    };

                    (pdf, shape_param_names, yield_modifiers, observable.clone())
                }
                #[cfg(feature = "neural")]
                PdfSpec::Flow { manifest } => {
                    let manifest_path = if manifest.is_absolute() {
                        manifest.clone()
                    } else {
                        base_dir.join(manifest)
                    };
                    let flow = crate::pdf::FlowPdf::from_manifest(&manifest_path, &[])
                        .with_context(|| {
                            format!(
                                "process '{}' flow: failed to load from {}",
                                proc.name,
                                manifest_path.display()
                            )
                        })?;
                    let obs_names = flow.observables().to_vec();
                    if obs_names.is_empty() {
                        anyhow::bail!(
                            "process '{}' flow: manifest declares 0 observables",
                            proc.name
                        );
                    }
                    for obs_name in &obs_names {
                        if !observable_names.contains_key(obs_name.as_str()) {
                            anyhow::bail!(
                                "process '{}' flow: manifest observable '{}' not declared in channel '{}' observables",
                                proc.name,
                                obs_name,
                                ch.name
                            );
                        }
                    }
                    let primary_obs = obs_names[0].clone();
                    (Arc::new(flow) as Arc<dyn UnbinnedPdf>, Vec::new(), Vec::new(), primary_obs)
                }
                #[cfg(feature = "neural")]
                PdfSpec::ConditionalFlow { manifest, context_params } => {
                    let manifest_path = if manifest.is_absolute() {
                        manifest.clone()
                    } else {
                        base_dir.join(manifest)
                    };
                    let mut ctx_indices = Vec::with_capacity(context_params.len());
                    for cp in context_params {
                        let idx = *index_by_name.get(cp).ok_or_else(|| {
                            anyhow::anyhow!(
                                "process '{}' conditional_flow: unknown context parameter '{}'",
                                proc.name,
                                cp
                            )
                        })?;
                        ctx_indices.push(idx);
                    }
                    let flow = crate::pdf::FlowPdf::from_manifest(&manifest_path, &ctx_indices)
                        .with_context(|| {
                            format!(
                                "process '{}' conditional_flow: failed to load from {}",
                                proc.name,
                                manifest_path.display()
                            )
                        })?;
                    let obs_names = flow.observables().to_vec();
                    if obs_names.is_empty() {
                        anyhow::bail!(
                            "process '{}' conditional_flow: manifest declares 0 observables",
                            proc.name
                        );
                    }
                    for obs_name in &obs_names {
                        if !observable_names.contains_key(obs_name.as_str()) {
                            anyhow::bail!(
                                "process '{}' conditional_flow: manifest observable '{}' not declared in channel '{}' observables",
                                proc.name,
                                obs_name,
                                ch.name
                            );
                        }
                    }
                    let primary_obs = obs_names[0].clone();
                    (
                        Arc::new(flow) as Arc<dyn UnbinnedPdf>,
                        context_params.clone(),
                        Vec::new(),
                        primary_obs,
                    )
                }
                #[cfg(feature = "neural")]
                PdfSpec::DcrSurrogate { manifest, systematics } => {
                    let manifest_path = if manifest.is_absolute() {
                        manifest.clone()
                    } else {
                        base_dir.join(manifest)
                    };
                    let mut syst_indices = Vec::with_capacity(systematics.len());
                    for s in systematics {
                        let idx = *index_by_name.get(s).ok_or_else(|| {
                            anyhow::anyhow!(
                                "process '{}' dcr_surrogate: unknown systematic parameter '{}'",
                                proc.name,
                                s
                            )
                        })?;
                        syst_indices.push(idx);
                    }
                    let dcr = crate::pdf::DcrSurrogate::from_manifest(
                        &manifest_path,
                        &syst_indices,
                        systematics.clone(),
                        proc.name.clone(),
                    )
                    .with_context(|| {
                        format!(
                            "process '{}' dcr_surrogate: failed to load from {}",
                            proc.name,
                            manifest_path.display()
                        )
                    })?;
                    let obs_names = dcr.observables().to_vec();
                    if obs_names.is_empty() {
                        anyhow::bail!(
                            "process '{}' dcr_surrogate: manifest declares 0 observables",
                            proc.name
                        );
                    }
                    for obs_name in &obs_names {
                        if !observable_names.contains_key(obs_name.as_str()) {
                            anyhow::bail!(
                                "process '{}' dcr_surrogate: manifest observable '{}' not declared in channel '{}' observables",
                                proc.name,
                                obs_name,
                                ch.name
                            );
                        }
                    }
                    let primary_obs = obs_names[0].clone();
                    (
                        Arc::new(dcr) as Arc<dyn UnbinnedPdf>,
                        systematics.clone(),
                        Vec::new(),
                        primary_obs,
                    )
                }
                #[cfg(not(feature = "neural"))]
                PdfSpec::Flow { .. }
                | PdfSpec::ConditionalFlow { .. }
                | PdfSpec::DcrSurrogate { .. } => {
                    anyhow::bail!(
                        "process '{}': flow/conditional_flow/dcr_surrogate PDFs require the 'neural' feature (cargo build --features neural)",
                        proc.name
                    );
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

            let yield_expr = apply_extra_yield_modifiers(yield_expr, extra_yield_modifiers);

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

fn apply_extra_yield_modifiers(base: YieldExpr, extra: Vec<RateModifier>) -> YieldExpr {
    if extra.is_empty() {
        return base;
    }

    match base {
        YieldExpr::Modified { base, mut modifiers } => {
            modifiers.extend(extra);
            YieldExpr::Modified { base, modifiers }
        }
        other => YieldExpr::Modified { base: Box::new(other), modifiers: extra },
    }
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
            RateModifierSpec::WeightSys { param, lo, hi, interp_code } => {
                let idx = *index_by_name.get(param).ok_or_else(|| {
                    anyhow::anyhow!(
                        "process '{}' references unknown WeightSys nuisance parameter '{}'",
                        process_name,
                        param
                    )
                })?;
                let code = match interp_code.as_deref() {
                    Some("code4p") | Some("Code4p") => HistoSysInterpCode::Code4p,
                    _ => HistoSysInterpCode::Code0,
                };
                compiled.push(RateModifier::WeightSys {
                    alpha_index: idx,
                    lo: *lo,
                    hi: *hi,
                    interp_code: code,
                });
            }
        }
    }
    Ok(YieldExpr::Modified { base: Box::new(base), modifiers: compiled })
}
