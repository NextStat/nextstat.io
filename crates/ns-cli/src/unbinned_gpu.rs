//! Unbinned GPU compilation + model wrapper for GPU-accelerated inference.
//!
//! This module is intentionally conservative: Phase 1 GPU supports a closed set of PDFs
//! (Gaussian/Exponential/CrystalBall/DoubleCrystalBall/Chebyshev) and yield expressions
//! (fixed/parameter/scaled, with NormSys/WeightSys rate modifiers). Everything else should
//! fall back to the CPU path.

use anyhow::{Context, Result};
use ns_compute::unbinned_types::{self, UnbinnedGpuModelData};
use ns_core::traits::{FixedParamModel, LogDensityModel, PoiModel, PreparedModelRef};
use ns_root::RootFile;
use ns_unbinned::{EventStore, ObservableSpec};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::unbinned_spec::{
    ConstraintSpec, DataFormat, DataSpec, HistoSysInterpCodeSpec, HorizontalSystematicSpec,
    PdfSpec, RateModifierSpec, UnbinnedSpecV0, WeightSystematicSpec, YieldSpec,
};

const GPU_MAX_CHEBYSHEV_ORDER: usize = 16;
const HISTOGRAM_SUPPORT_EPS: f64 = 1e-12;

fn push_histogram_aux(
    pdf_aux_f64: &mut Vec<f64>,
    proc_name: &str,
    ch_name: &str,
    observable: &str,
    expected_observable: &str,
    obs_bounds: (f64, f64),
    bin_edges: &[f64],
    bin_content: &[f64],
    pseudo_count: Option<f64>,
) -> Result<(u32, u32)> {
    if observable != expected_observable {
        anyhow::bail!(
            "process '{}' pdf.observable '{}' must match channel '{}' observable '{}'",
            proc_name,
            observable,
            ch_name,
            expected_observable
        );
    }
    if bin_edges.len() < 2 {
        anyhow::bail!(
            "process '{}' histogram requires at least 2 bin edges, got {}",
            proc_name,
            bin_edges.len()
        );
    }
    if bin_content.len() + 1 != bin_edges.len() {
        anyhow::bail!(
            "process '{}' histogram bin_content length mismatch: expected {}, got {}",
            proc_name,
            bin_edges.len() - 1,
            bin_content.len()
        );
    }
    let pseudo_count = pseudo_count.unwrap_or(0.0);
    if !pseudo_count.is_finite() || pseudo_count < 0.0 {
        anyhow::bail!(
            "process '{}' histogram pseudo_count must be finite and >= 0, got {}",
            proc_name,
            pseudo_count
        );
    }
    for (i, &e) in bin_edges.iter().enumerate() {
        if !e.is_finite() {
            anyhow::bail!(
                "process '{}' histogram bin_edges[{}] must be finite, got {}",
                proc_name,
                i,
                e
            );
        }
        if i > 0 && bin_edges[i - 1] >= e {
            anyhow::bail!(
                "process '{}' histogram bin edges must be strictly increasing (edges[{}]={} >= edges[{}]={})",
                proc_name,
                i - 1,
                bin_edges[i - 1],
                i,
                e
            );
        }
    }
    for (i, &w) in bin_content.iter().enumerate() {
        if !w.is_finite() || w < 0.0 {
            anyhow::bail!(
                "process '{}' histogram bin_content[{}] must be finite and >= 0, got {}",
                proc_name,
                i,
                w
            );
        }
    }

    // Require that histogram support matches event support, mirroring HistogramPdf.
    let x_min = bin_edges[0];
    let x_max = *bin_edges.last().unwrap_or(&x_min);
    let (a, b) = obs_bounds;
    if (a - x_min).abs() > HISTOGRAM_SUPPORT_EPS || (b - x_max).abs() > HISTOGRAM_SUPPORT_EPS {
        anyhow::bail!(
            "process '{}' histogram support mismatch: pdf=[{}, {}], events=({}, {}) (channel '{}')",
            proc_name,
            x_min,
            x_max,
            a,
            b,
            ch_name
        );
    }

    let mut total = 0.0f64;
    for &w in bin_content {
        total += w + pseudo_count;
    }
    if !(total.is_finite() && total > 0.0) {
        anyhow::bail!(
            "process '{}' histogram total content must be finite and > 0 after pseudo_count, got {}",
            proc_name,
            total
        );
    }
    let log_total = total.ln();

    let n_bins = bin_content.len();
    let mut log_density = Vec::with_capacity(n_bins);
    for i in 0..n_bins {
        let w = bin_content[i] + pseudo_count;
        let width = bin_edges[i + 1] - bin_edges[i];
        if !(width.is_finite() && width > 0.0) {
            anyhow::bail!(
                "process '{}' histogram bin width must be finite and > 0, got width={} for bin {}",
                proc_name,
                width,
                i
            );
        }
        if w > 0.0 {
            log_density.push(w.ln() - log_total - width.ln());
        } else {
            log_density.push(f64::NEG_INFINITY);
        }
    }

    let aux_offset = pdf_aux_f64.len() as u32;
    pdf_aux_f64.extend_from_slice(bin_edges);
    pdf_aux_f64.extend_from_slice(&log_density);
    let aux_len = (bin_edges.len() + log_density.len()) as u32;

    Ok((aux_offset, aux_len))
}

fn lower_histogram_from_tree_to_gpu_histogram(
    pdf_aux_f64: &mut Vec<f64>,
    base_dir: &Path,
    index_by_name: &HashMap<String, usize>,
    proc_name: &str,
    ch_name: &str,
    expected_observable: &str,
    expected_observable_expr: &str,
    obs_bounds: (f64, f64),
    observable: &str,
    bin_edges: &[f64],
    pseudo_count: Option<f64>,
    source: &DataSpec,
    max_events: Option<usize>,
    weight_systematics: &[WeightSystematicSpec],
    horizontal_systematics: &[HorizontalSystematicSpec],
) -> Result<(u32, u32, Vec<unbinned_types::GpuUnbinnedRateModifierDesc>)> {
    if !horizontal_systematics.is_empty() {
        anyhow::bail!(
            "process '{}' histogram_from_tree with horizontal_systematics is not supported by unbinned --gpu (shape morphing is CPU-only for now)",
            proc_name
        );
    }
    if source.effective_format() != DataFormat::Root {
        anyhow::bail!(
            "process '{}' histogram_from_tree requires a ROOT source for unbinned --gpu (got Parquet): source.file={}",
            proc_name,
            source.file.display()
        );
    }
    if source.channel.is_some() {
        anyhow::bail!(
            "process '{}' histogram_from_tree: source.channel is not supported (ROOT sources do not have channels)",
            proc_name
        );
    }

    let mut extra_weight_exprs = Vec::<&str>::new();
    let mut yield_weight_systematics = Vec::<&WeightSystematicSpec>::new();
    let mut seen_weight_params = HashSet::<&str>::new();
    for s in weight_systematics {
        if s.param.trim().is_empty() {
            anyhow::bail!(
                "process '{}' histogram_from_tree: weight systematic param name must be non-empty",
                proc_name
            );
        }
        if !seen_weight_params.insert(s.param.as_str()) {
            anyhow::bail!(
                "process '{}' histogram_from_tree: duplicate weight systematic param '{}'",
                proc_name,
                s.param
            );
        }
        if s.up.trim().is_empty() || s.down.trim().is_empty() {
            anyhow::bail!(
                "process '{}' histogram_from_tree: weight systematic '{}' requires non-empty up/down expressions",
                proc_name,
                s.param
            );
        }
        if !s.apply_to_shape && !s.apply_to_yield {
            anyhow::bail!(
                "process '{}' histogram_from_tree: weight systematic '{}' is a no-op (apply_to_shape=false and apply_to_yield=false)",
                proc_name,
                s.param
            );
        }
        if s.apply_to_shape {
            anyhow::bail!(
                "process '{}' histogram_from_tree: weight systematic '{}' with apply_to_shape=true is not supported by unbinned --gpu (shape morphing is CPU-only for now)",
                proc_name,
                s.param
            );
        }
        if s.apply_to_yield {
            extra_weight_exprs.push(s.up.as_str());
            extra_weight_exprs.push(s.down.as_str());
            yield_weight_systematics.push(s);
        }
    }

    let src_path =
        if source.file.is_absolute() { source.file.clone() } else { base_dir.join(&source.file) };
    let src_root = RootFile::open(&src_path)
        .with_context(|| format!("failed to open ROOT file {}", src_path.display()))?;
    let src_tree_name = source.tree_name().with_context(|| {
        format!("process '{}' histogram_from_tree source requires tree name", proc_name)
    })?;

    let train_observables = vec![ObservableSpec::expression(
        expected_observable.to_string(),
        expected_observable_expr.to_string(),
        obs_bounds,
    )];
    let (train, mut extra_weights) = EventStore::from_tree_with_extra_weights(
        &src_root,
        src_tree_name,
        &train_observables,
        source.selection.as_deref(),
        source.weight.as_deref(),
        &extra_weight_exprs,
    )
    .with_context(|| {
        format!(
            "failed to load histogram_from_tree source data for process '{}' (channel '{}') from {}",
            proc_name,
            ch_name,
            src_path.display()
        )
    })?;

    let mut xs = train
        .column(expected_observable)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "histogram_from_tree source data missing column '{expected_observable}'"
            )
        })?
        .to_vec();
    let mut ws = train.weights().map(|w| w.to_vec());

    if let Some(max) = max_events {
        if max == 0 {
            anyhow::bail!("process '{}' histogram_from_tree max_events must be > 0", proc_name);
        }
        let keep = max.min(xs.len());
        xs.truncate(keep);
        if let Some(w) = ws.as_mut() {
            w.truncate(keep);
        }
        for w in &mut extra_weights {
            w.truncate(keep);
        }
    }

    let n_bins = bin_edges.len().saturating_sub(1);
    if n_bins == 0 {
        anyhow::bail!("process '{}' histogram_from_tree requires at least 2 bin_edges", proc_name);
    }
    let edge_min = bin_edges[0];
    let edge_max = *bin_edges.last().unwrap_or(&edge_min);
    let bin_index = |x: f64| -> Result<usize> {
        if !x.is_finite() {
            anyhow::bail!(
                "process '{}' histogram_from_tree: non-finite event value {}",
                proc_name,
                x
            );
        }
        if x < edge_min || x > edge_max {
            anyhow::bail!(
                "process '{}' histogram_from_tree: x out of range: x={} not in [{}, {}]",
                proc_name,
                x,
                edge_min,
                edge_max
            );
        }
        if x >= edge_max {
            return Ok(n_bins - 1);
        }
        let k = bin_edges.partition_point(|e| *e <= x);
        if k == 0 || k > n_bins {
            anyhow::bail!(
                "process '{}' histogram_from_tree: failed to locate bin for x={} in [{}, {}]",
                proc_name,
                x,
                edge_min,
                edge_max
            );
        }
        Ok(k - 1)
    };

    let mut bin_idx = Vec::with_capacity(xs.len());
    for &x in &xs {
        bin_idx.push(bin_index(x)?);
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
            "process '{}' histogram_from_tree: nominal sum of weights must be finite and > 0, got {}",
            proc_name,
            total_nom
        );
    }

    if extra_weights.len() != yield_weight_systematics.len().saturating_mul(2) {
        anyhow::bail!(
            "process '{}' histogram_from_tree: internal error: expected {} extra weight columns, got {}",
            proc_name,
            yield_weight_systematics.len() * 2,
            extra_weights.len()
        );
    }

    let mut derived_rate_modifiers =
        Vec::<unbinned_types::GpuUnbinnedRateModifierDesc>::with_capacity(
            yield_weight_systematics.len(),
        );
    for (sidx, s) in yield_weight_systematics.iter().enumerate() {
        let up_ratio = &extra_weights[2 * sidx];
        let down_ratio = &extra_weights[2 * sidx + 1];
        if up_ratio.len() != xs.len() || down_ratio.len() != xs.len() {
            anyhow::bail!(
                "process '{}' histogram_from_tree: weight systematic '{}' length mismatch: expected {} events, got up={} down={}",
                proc_name,
                s.param,
                xs.len(),
                up_ratio.len(),
                down_ratio.len()
            );
        }

        let mut total_up = 0.0f64;
        let mut total_down = 0.0f64;
        for i in 0..xs.len() {
            let w0 = ws.as_ref().map(|w| w[i]).unwrap_or(1.0);
            total_up += w0 * up_ratio[i];
            total_down += w0 * down_ratio[i];
        }
        if !(total_up.is_finite() && total_up > 0.0 && total_down.is_finite() && total_down > 0.0) {
            anyhow::bail!(
                "process '{}' histogram_from_tree: weight systematic '{}' totals must be finite and > 0, got up={} down={}",
                proc_name,
                s.param,
                total_up,
                total_down
            );
        }
        let hi = total_up / total_nom;
        let lo = total_down / total_nom;
        if !(hi.is_finite() && hi > 0.0 && lo.is_finite() && lo > 0.0) {
            anyhow::bail!(
                "process '{}' histogram_from_tree: weight systematic '{}' yield factors must be finite and > 0, got lo={}, hi={}",
                proc_name,
                s.param,
                lo,
                hi
            );
        }

        let alpha_idx = *index_by_name.get(s.param.as_str()).ok_or_else(|| {
            anyhow::anyhow!(
                "process '{}' histogram_from_tree references unknown weight systematic nuisance parameter '{}'",
                proc_name,
                s.param
            )
        })?;
        let interp_code = match &s.interp {
            HistoSysInterpCodeSpec::Code0 => 0u32,
            HistoSysInterpCodeSpec::Code4p => 1u32,
        };
        derived_rate_modifiers.push(unbinned_types::GpuUnbinnedRateModifierDesc {
            kind: unbinned_types::rate_modifier_kind::WEIGHT_SYS,
            alpha_param_idx: alpha_idx as u32,
            interp_code,
            _pad: 0,
            lo,
            hi,
        });
    }

    let (aux_offset, aux_len) = push_histogram_aux(
        pdf_aux_f64,
        proc_name,
        ch_name,
        observable,
        expected_observable,
        obs_bounds,
        bin_edges,
        &nominal_bin_content,
        pseudo_count,
    )?;

    Ok((aux_offset, aux_len, derived_rate_modifiers))
}

/// Minimal parameter metadata for unbinned GPU models.
#[derive(Debug, Clone)]
pub(crate) struct UnbinnedGpuMeta {
    pub parameter_names: Vec<String>,
    pub bounds: Vec<(f64, f64)>,
    pub init: Vec<f64>,
    pub poi_index: Option<usize>,
}

pub(crate) trait UnbinnedAccel: Send {
    fn n_params(&self) -> usize;
    fn single_nll(&mut self, params: &[f64]) -> ns_core::Result<f64>;
    fn single_nll_grad(&mut self, params: &[f64]) -> ns_core::Result<(f64, Vec<f64>)>;
}

/// Multi-channel wrapper: sums NLL/grad across multiple per-channel accelerators.
///
/// This keeps kernels simple (1 dataset per dispatch) while allowing multi-channel
/// unbinned fits on GPU by evaluating each included channel independently.
pub(crate) struct MultiUnbinnedAccel<A: UnbinnedAccel> {
    accels: Vec<A>,
    n_params: usize,
}

impl<A: UnbinnedAccel> MultiUnbinnedAccel<A> {
    fn new(accels: Vec<A>) -> ns_core::Result<Self> {
        if accels.is_empty() {
            return Err(ns_core::Error::Validation(
                "MultiUnbinnedAccel requires at least 1 channel accelerator".into(),
            ));
        }
        let n_params = accels[0].n_params();
        for (i, a) in accels.iter().enumerate() {
            if a.n_params() != n_params {
                return Err(ns_core::Error::Validation(format!(
                    "MultiUnbinnedAccel channel {i} has n_params={}, expected {n_params}",
                    a.n_params()
                )));
            }
        }
        Ok(Self { accels, n_params })
    }
}

impl<A: UnbinnedAccel> UnbinnedAccel for MultiUnbinnedAccel<A> {
    fn n_params(&self) -> usize {
        self.n_params
    }

    fn single_nll(&mut self, params: &[f64]) -> ns_core::Result<f64> {
        let mut sum = 0.0f64;
        for a in &mut self.accels {
            sum += a.single_nll(params)?;
        }
        Ok(sum)
    }

    fn single_nll_grad(&mut self, params: &[f64]) -> ns_core::Result<(f64, Vec<f64>)> {
        let mut sum_nll = 0.0f64;
        let mut sum_grad = vec![0.0f64; self.n_params];
        for a in &mut self.accels {
            let (nll, g) = a.single_nll_grad(params)?;
            sum_nll += nll;
            if g.len() != sum_grad.len() {
                return Err(ns_core::Error::Validation(format!(
                    "MultiUnbinnedAccel grad length mismatch: expected {}, got {}",
                    sum_grad.len(),
                    g.len()
                )));
            }
            for i in 0..sum_grad.len() {
                sum_grad[i] += g[i];
            }
        }
        Ok((sum_nll, sum_grad))
    }
}

#[cfg(feature = "cuda")]
impl UnbinnedAccel for ns_compute::cuda_unbinned::CudaUnbinnedAccelerator {
    fn n_params(&self) -> usize {
        ns_compute::cuda_unbinned::CudaUnbinnedAccelerator::n_params(self)
    }
    fn single_nll(&mut self, params: &[f64]) -> ns_core::Result<f64> {
        ns_compute::cuda_unbinned::CudaUnbinnedAccelerator::single_nll(self, params)
    }
    fn single_nll_grad(&mut self, params: &[f64]) -> ns_core::Result<(f64, Vec<f64>)> {
        ns_compute::cuda_unbinned::CudaUnbinnedAccelerator::single_nll_grad(self, params)
    }
}

#[cfg(feature = "metal")]
impl UnbinnedAccel for ns_compute::metal_unbinned::MetalUnbinnedAccelerator {
    fn n_params(&self) -> usize {
        ns_compute::metal_unbinned::MetalUnbinnedAccelerator::n_params(self)
    }
    fn single_nll(&mut self, params: &[f64]) -> ns_core::Result<f64> {
        ns_compute::metal_unbinned::MetalUnbinnedAccelerator::single_nll(self, params)
    }
    fn single_nll_grad(&mut self, params: &[f64]) -> ns_core::Result<(f64, Vec<f64>)> {
        ns_compute::metal_unbinned::MetalUnbinnedAccelerator::single_nll_grad(self, params)
    }
}

/// Unbinned model wrapper that routes NLL/gradient evaluations to a GPU accelerator.
pub(crate) struct UnbinnedGpuModel<A: UnbinnedAccel> {
    meta: UnbinnedGpuMeta,
    accel: Arc<Mutex<A>>,
}

impl<A: UnbinnedAccel> UnbinnedGpuModel<A> {
    pub fn meta(&self) -> &UnbinnedGpuMeta {
        &self.meta
    }
}

impl<A: UnbinnedAccel> LogDensityModel for UnbinnedGpuModel<A> {
    type Prepared<'a>
        = PreparedModelRef<'a, Self>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        self.meta.init.len()
    }

    fn parameter_names(&self) -> Vec<String> {
        self.meta.parameter_names.clone()
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        self.meta.bounds.clone()
    }

    fn parameter_init(&self) -> Vec<f64> {
        self.meta.init.clone()
    }

    fn nll(&self, params: &[f64]) -> ns_core::Result<f64> {
        let mut a = self.accel.lock().map_err(|_| {
            ns_core::Error::Computation("mutex poisoned (unbinned gpu accelerator)".into())
        })?;
        a.single_nll(params)
    }

    fn grad_nll(&self, params: &[f64]) -> ns_core::Result<Vec<f64>> {
        let mut a = self.accel.lock().map_err(|_| {
            ns_core::Error::Computation("mutex poisoned (unbinned gpu accelerator)".into())
        })?;
        let (_nll, g) = a.single_nll_grad(params)?;
        Ok(g)
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }

    fn prefer_fused_eval_grad(&self) -> bool {
        true
    }

    fn nll_grad_prepared(
        &self,
        _prepared: &Self::Prepared<'_>,
        params: &[f64],
    ) -> ns_core::Result<(f64, Vec<f64>)> {
        let mut a = self.accel.lock().map_err(|_| {
            ns_core::Error::Computation("mutex poisoned (unbinned gpu accelerator)".into())
        })?;
        a.single_nll_grad(params)
    }
}

impl<A: UnbinnedAccel> PoiModel for UnbinnedGpuModel<A> {
    fn poi_index(&self) -> Option<usize> {
        self.meta.poi_index
    }
}

impl<A: UnbinnedAccel> FixedParamModel for UnbinnedGpuModel<A> {
    fn with_fixed_param(&self, param_idx: usize, value: f64) -> Self {
        let mut meta = self.meta.clone();
        if let Some(b) = meta.bounds.get_mut(param_idx) {
            *b = (value, value);
        }
        if let Some(v) = meta.init.get_mut(param_idx) {
            *v = value;
        }
        Self { meta, accel: self.accel.clone() }
    }
}

fn build_meta(spec: &UnbinnedSpecV0) -> Result<(UnbinnedGpuMeta, HashMap<String, usize>)> {
    let mut index_by_name = HashMap::<String, usize>::new();
    let mut parameter_names = Vec::with_capacity(spec.model.parameters.len());
    let mut bounds = Vec::with_capacity(spec.model.parameters.len());
    let mut init = Vec::with_capacity(spec.model.parameters.len());

    for (i, p) in spec.model.parameters.iter().enumerate() {
        if index_by_name.insert(p.name.clone(), i).is_some() {
            anyhow::bail!("duplicate parameter name: '{}'", p.name);
        }
        parameter_names.push(p.name.clone());
        bounds.push((p.bounds[0], p.bounds[1]));
        init.push(p.init);
    }

    let poi_index = match &spec.model.poi {
        None => None,
        Some(name) => Some(
            *index_by_name
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("unknown POI parameter name: '{name}'"))?,
        ),
    };

    Ok((UnbinnedGpuMeta { parameter_names, bounds, init, poi_index }, index_by_name))
}

fn build_gpu_data(
    spec: &UnbinnedSpecV0,
    spec_path: &Path,
) -> Result<(UnbinnedGpuMeta, Vec<UnbinnedGpuModelData>)> {
    let (meta, index_by_name) = build_meta(spec)?;

    // Gaussian constraints (global; applied once).
    let mut gauss_constraints = Vec::<unbinned_types::GpuUnbinnedGaussConstraintEntry>::new();
    let mut constraint_const = 0.0f64;
    for (idx, p) in spec.model.parameters.iter().enumerate() {
        let Some(c) = &p.constraint else { continue };
        match c {
            ConstraintSpec::Gaussian { mean, sigma } => {
                if !(sigma.is_finite() && *sigma > 0.0) {
                    anyhow::bail!(
                        "Gaussian constraint sigma must be finite and > 0 for '{}', got {}",
                        p.name,
                        sigma
                    );
                }
                gauss_constraints.push(unbinned_types::GpuUnbinnedGaussConstraintEntry {
                    center: *mean,
                    inv_width: 1.0 / *sigma,
                    param_idx: idx as u32,
                    _pad: 0,
                });
                constraint_const += sigma.ln() + 0.5 * (2.0 * std::f64::consts::PI).ln();
            }
        }
    }

    // Collect included channels. We keep kernels 1-channel and sum across channels in Rust.
    // Resolve data file relative to the spec location.
    let base_dir = spec_path.parent().unwrap_or_else(|| Path::new("."));
    let mut datas = Vec::<UnbinnedGpuModelData>::new();

    for ch in spec.channels.iter().filter(|c| c.include_in_fit) {
        if ch.observables.len() != 1 {
            anyhow::bail!(
                "unbinned --gpu currently supports exactly 1 observable per channel; channel '{}' has {}",
                ch.name,
                ch.observables.len()
            );
        }
        if ch.processes.is_empty() {
            anyhow::bail!("channel '{}' has no processes", ch.name);
        }

        // Resolve data file relative to the spec location.
        let data_path = if ch.data.file.is_absolute() {
            ch.data.file.clone()
        } else {
            base_dir.join(&ch.data.file)
        };

        let obs0 = &ch.observables[0];
        let obs_expr = obs0.expr.clone().unwrap_or_else(|| obs0.name.clone());
        let bounds0 = (obs0.bounds[0], obs0.bounds[1]);
        let observables = vec![ObservableSpec {
            name: obs0.name.clone(),
            expr: obs_expr.clone(),
            bounds: bounds0,
        }];

        let store = match ch.data.effective_format() {
            crate::unbinned_spec::DataFormat::Root => {
                if ch.data.channel.is_some() {
                    anyhow::bail!(
                        "channel '{}': data.channel is only supported for Parquet data sources",
                        ch.name
                    );
                }
                let root = RootFile::open(&data_path)
                    .with_context(|| format!("failed to open ROOT file {}", data_path.display()))?;
                let tree_name = ch.data.tree_name().with_context(|| {
                    format!("channel '{}' uses ROOT format but has no tree name", ch.name)
                })?;
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
            crate::unbinned_spec::DataFormat::Parquet => {
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
        };

        let xs = store.column(obs0.name.as_str()).ok_or_else(|| {
            anyhow::anyhow!(
                "missing column '{}' in observed data (channel '{}')",
                obs0.name,
                ch.name
            )
        })?;

        let (a, b) = store.bounds(obs0.name.as_str()).ok_or_else(|| {
            anyhow::anyhow!("missing bounds for observable '{}' (channel '{}')", obs0.name, ch.name)
        })?;

        let n_events = store.n_events();
        let mut obs_soa = Vec::with_capacity(n_events);
        obs_soa.extend_from_slice(xs);
        let event_weights = store.weights().map(|w| w.to_vec());

        if let Some(ref w) = event_weights {
            let sum_w: f64 = w.iter().sum();
            let sum_w2: f64 = w.iter().map(|wi| wi * wi).sum();
            let n_eff = if sum_w2 > 0.0 { (sum_w * sum_w) / sum_w2 } else { 0.0 };
            let w_min_pos = w.iter().copied().filter(|&wi| wi > 0.0).fold(f64::INFINITY, f64::min);
            let w_max = w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let ratio =
                if w_min_pos > 0.0 && w_min_pos.is_finite() { w_max / w_min_pos } else { f64::NAN };
            eprintln!(
                "  channel '{}': N={}, sum(w)={:.2}, N_eff={:.1}, w_range=[{:.4}, {:.4}], max/min={:.1}",
                ch.name, n_events, sum_w, n_eff, w_min_pos, w_max, ratio
            );
            if ratio > 100.0 {
                eprintln!(
                    "  WARNING: channel '{}' has extreme weight ratio {:.1} (max/min > 100). \
                     Effective sample size is {:.0} ({:.1}% of N={}). \
                     Consider rebinning or capping weights.",
                    ch.name,
                    ratio,
                    n_eff,
                    100.0 * n_eff / n_events as f64,
                    n_events
                );
            }
        }

        let mut processes =
            Vec::<unbinned_types::GpuUnbinnedProcessDesc>::with_capacity(ch.processes.len());
        let mut shape_param_indices = Vec::<u32>::new();
        let mut rate_modifiers = Vec::<unbinned_types::GpuUnbinnedRateModifierDesc>::new();
        let mut pdf_aux_f64 = Vec::<f64>::new();

        for proc in &ch.processes {
            let mut derived_rate_modifiers =
                Vec::<unbinned_types::GpuUnbinnedRateModifierDesc>::new();
            let (pdf_kind, shape_names, pdf_aux_offset, pdf_aux_len): (u32, Vec<String>, u32, u32) =
                match &proc.pdf {
                    PdfSpec::Gaussian { observable, params } => {
                        if observable != &obs0.name {
                            anyhow::bail!(
                                "process '{}' pdf.observable '{}' must match channel '{}' observable '{}'",
                                proc.name,
                                observable,
                                ch.name,
                                obs0.name
                            );
                        }
                        if params.len() != 2 {
                            anyhow::bail!(
                                "process '{}' gaussian params must have length 2 (mu,sigma), got {}",
                                proc.name,
                                params.len()
                            );
                        }
                        (unbinned_types::pdf_kind::GAUSSIAN, params.clone(), 0, 0)
                    }
                    PdfSpec::Exponential { observable, params } => {
                        if observable != &obs0.name {
                            anyhow::bail!(
                                "process '{}' pdf.observable '{}' must match channel '{}' observable '{}'",
                                proc.name,
                                observable,
                                ch.name,
                                obs0.name
                            );
                        }
                        if params.len() != 1 {
                            anyhow::bail!(
                                "process '{}' exponential params must have length 1 (lambda), got {}",
                                proc.name,
                                params.len()
                            );
                        }
                        (unbinned_types::pdf_kind::EXPONENTIAL, params.clone(), 0, 0)
                    }
                    PdfSpec::CrystalBall { observable, params } => {
                        if observable != &obs0.name {
                            anyhow::bail!(
                                "process '{}' pdf.observable '{}' must match channel '{}' observable '{}'",
                                proc.name,
                                observable,
                                ch.name,
                                obs0.name
                            );
                        }
                        if params.len() != 4 {
                            anyhow::bail!(
                                "process '{}' crystal_ball params must have length 4 (mu,sigma,alpha,n), got {}",
                                proc.name,
                                params.len()
                            );
                        }
                        (unbinned_types::pdf_kind::CRYSTAL_BALL, params.clone(), 0, 0)
                    }
                    PdfSpec::DoubleCrystalBall { observable, params } => {
                        if observable != &obs0.name {
                            anyhow::bail!(
                                "process '{}' pdf.observable '{}' must match channel '{}' observable '{}'",
                                proc.name,
                                observable,
                                ch.name,
                                obs0.name
                            );
                        }
                        if params.len() != 6 {
                            anyhow::bail!(
                                "process '{}' double_crystal_ball params must have length 6 (mu,sigma,alpha_l,n_l,alpha_r,n_r), got {}",
                                proc.name,
                                params.len()
                            );
                        }
                        (unbinned_types::pdf_kind::DOUBLE_CRYSTAL_BALL, params.clone(), 0, 0)
                    }
                    PdfSpec::Chebyshev { observable, params } => {
                        if observable != &obs0.name {
                            anyhow::bail!(
                                "process '{}' pdf.observable '{}' must match channel '{}' observable '{}'",
                                proc.name,
                                observable,
                                ch.name,
                                obs0.name
                            );
                        }
                        if params.is_empty() {
                            anyhow::bail!(
                                "process '{}' chebyshev params must have length >= 1 (c_1..c_m), got 0",
                                proc.name
                            );
                        }
                        if params.len() > GPU_MAX_CHEBYSHEV_ORDER {
                            anyhow::bail!(
                                "process '{}' chebyshev order too large for --gpu: got {}, max {}",
                                proc.name,
                                params.len(),
                                GPU_MAX_CHEBYSHEV_ORDER
                            );
                        }
                        (unbinned_types::pdf_kind::CHEBYSHEV, params.clone(), 0, 0)
                    }
                    PdfSpec::Histogram { observable, bin_edges, bin_content, pseudo_count } => {
                        let (aux_offset, aux_len) = push_histogram_aux(
                            &mut pdf_aux_f64,
                            &proc.name,
                            &ch.name,
                            observable,
                            &obs0.name,
                            (a, b),
                            bin_edges,
                            bin_content,
                            *pseudo_count,
                        )?;
                        (unbinned_types::pdf_kind::HISTOGRAM, Vec::new(), aux_offset, aux_len)
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
                        let (aux_offset, aux_len, derived) =
                            lower_histogram_from_tree_to_gpu_histogram(
                                &mut pdf_aux_f64,
                                base_dir,
                                &index_by_name,
                                &proc.name,
                                &ch.name,
                                &obs0.name,
                                &obs_expr,
                                (a, b),
                                observable,
                                bin_edges,
                                *pseudo_count,
                                source,
                                *max_events,
                                weight_systematics,
                                horizontal_systematics,
                            )?;
                        derived_rate_modifiers = derived;
                        (unbinned_types::pdf_kind::HISTOGRAM, Vec::new(), aux_offset, aux_len)
                    }
                    other => {
                        anyhow::bail!(
                            "unbinned --gpu supports only gaussian/exponential/crystal_ball/double_crystal_ball/chebyshev/histogram/histogram_from_tree PDFs for now (CPU-only: argus/voigtian/spline/kde/kde_from_tree/flows); process '{}' (channel '{}') has unsupported pdf {:?}",
                            proc.name,
                            ch.name,
                            other
                        );
                    }
                };

            let shape_param_offset = shape_param_indices.len() as u32;
            for name in &shape_names {
                let idx = *index_by_name.get(name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "process '{}' references unknown shape parameter '{}' (channel '{}')",
                        proc.name,
                        name,
                        ch.name
                    )
                })?;
                shape_param_indices.push(idx as u32);
            }

            let (yield_kind, base_yield, yield_param_idx, modifiers): (
                u32,
                f64,
                u32,
                &[RateModifierSpec],
            ) = match &proc.yield_spec {
                YieldSpec::Fixed { value, modifiers } => {
                    (unbinned_types::yield_kind::FIXED, *value, 0u32, modifiers.as_slice())
                }
                YieldSpec::Parameter { name, modifiers } => {
                    let idx = *index_by_name.get(name).ok_or_else(|| {
                        anyhow::anyhow!(
                            "process '{}' references unknown yield parameter '{}' (channel '{}')",
                            proc.name,
                            name,
                            ch.name
                        )
                    })?;
                    (unbinned_types::yield_kind::PARAMETER, 0.0, idx as u32, modifiers.as_slice())
                }
                YieldSpec::Scaled { base_yield, scale, modifiers } => {
                    let idx = *index_by_name.get(scale).ok_or_else(|| {
                        anyhow::anyhow!(
                            "process '{}' references unknown scale parameter '{}' (channel '{}')",
                            proc.name,
                            scale,
                            ch.name
                        )
                    })?;
                    (
                        unbinned_types::yield_kind::SCALED,
                        *base_yield,
                        idx as u32,
                        modifiers.as_slice(),
                    )
                }
            };

            let rate_mod_offset = rate_modifiers.len() as u32;
            for m in modifiers {
                match m {
                    RateModifierSpec::NormSys { param, lo, hi } => {
                        if !(*lo).is_finite() || *lo <= 0.0 || !(*hi).is_finite() || *hi <= 0.0 {
                            anyhow::bail!(
                                "process '{}' has invalid NormSys lo/hi (must be finite and > 0): lo={}, hi={}",
                                proc.name,
                                lo,
                                hi
                            );
                        }
                        let alpha_idx = *index_by_name.get(param).ok_or_else(|| {
                            anyhow::anyhow!(
                                "process '{}' references unknown NormSys parameter '{}' (channel '{}')",
                                proc.name,
                                param,
                                ch.name
                            )
                        })?;
                        rate_modifiers.push(unbinned_types::GpuUnbinnedRateModifierDesc {
                            kind: unbinned_types::rate_modifier_kind::NORM_SYS,
                            alpha_param_idx: alpha_idx as u32,
                            interp_code: 0,
                            _pad: 0,
                            lo: *lo,
                            hi: *hi,
                        });
                    }
                    RateModifierSpec::WeightSys { param, lo, hi, interp_code } => {
                        if !(*lo).is_finite() || *lo <= 0.0 || !(*hi).is_finite() || *hi <= 0.0 {
                            anyhow::bail!(
                                "process '{}' has invalid WeightSys lo/hi (must be finite and > 0): lo={}, hi={}",
                                proc.name,
                                lo,
                                hi
                            );
                        }
                        let alpha_idx = *index_by_name.get(param).ok_or_else(|| {
                            anyhow::anyhow!(
                                "process '{}' references unknown WeightSys parameter '{}' (channel '{}')",
                                proc.name,
                                param,
                                ch.name
                            )
                        })?;
                        let gpu_interp = match interp_code.as_deref() {
                            Some("code4p") | Some("Code4p") => 1u32,
                            _ => 0u32,
                        };
                        rate_modifiers.push(unbinned_types::GpuUnbinnedRateModifierDesc {
                            kind: unbinned_types::rate_modifier_kind::WEIGHT_SYS,
                            alpha_param_idx: alpha_idx as u32,
                            interp_code: gpu_interp,
                            _pad: 0,
                            lo: *lo,
                            hi: *hi,
                        });
                    }
                }
            }
            rate_modifiers.extend(derived_rate_modifiers);
            let n_rate_mods = (rate_modifiers.len() as u32) - rate_mod_offset;

            processes.push(unbinned_types::GpuUnbinnedProcessDesc {
                base_yield,
                pdf_kind,
                yield_kind,
                obs_index: 0,
                shape_param_offset,
                n_shape_params: shape_names.len() as u32,
                yield_param_idx,
                rate_mod_offset,
                n_rate_mods,
                pdf_aux_offset,
                pdf_aux_len,
            });
        }

        datas.push(UnbinnedGpuModelData {
            n_params: meta.parameter_names.len(),
            n_obs: 1,
            n_events,
            obs_bounds: vec![(a, b)],
            obs_soa,
            event_weights,
            processes,
            rate_modifiers,
            shape_param_indices,
            pdf_aux_f64,
            gauss_constraints: Vec::new(),
            constraint_const: 0.0,
        });
    }

    if datas.is_empty() {
        anyhow::bail!("unbinned --gpu requires at least one channel with include_in_fit=true");
    }

    // Attach constraints to the first included channel only.
    datas[0].gauss_constraints = gauss_constraints;
    datas[0].constraint_const = constraint_const;

    Ok((meta, datas))
}

/// Build a GPU-lowered *static* unbinned model representation from the spec.
///
/// Unlike `build_gpu_data`, this does not load observed data from ROOT. It is intended for
/// GPU batch toy fitting, where toy datasets are provided separately.
pub(crate) fn build_gpu_static_data(
    spec: &UnbinnedSpecV0,
) -> Result<(UnbinnedGpuMeta, UnbinnedGpuModelData)> {
    if spec.channels.len() != 1 {
        anyhow::bail!(
            "unbinned --gpu currently supports exactly 1 channel, got {}",
            spec.channels.len()
        );
    }
    let ch = &spec.channels[0];
    if !ch.include_in_fit {
        anyhow::bail!("unbinned --gpu channel must have include_in_fit=true");
    }
    if ch.observables.len() != 1 {
        anyhow::bail!(
            "unbinned --gpu currently supports exactly 1 observable, got {}",
            ch.observables.len()
        );
    }
    if ch.processes.is_empty() {
        anyhow::bail!("channel '{}' has no processes", ch.name);
    }

    let (meta, index_by_name) = build_meta(spec)?;

    let obs0 = &ch.observables[0];
    let (a, b) = (obs0.bounds[0], obs0.bounds[1]);
    if !(a.is_finite() && b.is_finite() && a < b) {
        anyhow::bail!("invalid observable bounds for '{}': [{a}, {b}]", obs0.name);
    }

    // Processes + shape param indices (flat).
    let mut processes =
        Vec::<unbinned_types::GpuUnbinnedProcessDesc>::with_capacity(ch.processes.len());
    let mut shape_param_indices = Vec::<u32>::new();
    let mut rate_modifiers = Vec::<unbinned_types::GpuUnbinnedRateModifierDesc>::new();
    let mut pdf_aux_f64 = Vec::<f64>::new();

    for proc in &ch.processes {
        let (pdf_kind, shape_names, pdf_aux_offset, pdf_aux_len): (u32, Vec<String>, u32, u32) =
            match &proc.pdf {
                PdfSpec::Gaussian { observable, params } => {
                    if observable != &obs0.name {
                        anyhow::bail!(
                            "process '{}' pdf.observable '{}' must match channel observable '{}'",
                            proc.name,
                            observable,
                            obs0.name
                        );
                    }
                    if params.len() != 2 {
                        anyhow::bail!(
                            "process '{}' gaussian params must have length 2 (mu,sigma), got {}",
                            proc.name,
                            params.len()
                        );
                    }
                    (unbinned_types::pdf_kind::GAUSSIAN, params.clone(), 0, 0)
                }
                PdfSpec::Exponential { observable, params } => {
                    if observable != &obs0.name {
                        anyhow::bail!(
                            "process '{}' pdf.observable '{}' must match channel observable '{}'",
                            proc.name,
                            observable,
                            obs0.name
                        );
                    }
                    if params.len() != 1 {
                        anyhow::bail!(
                            "process '{}' exponential params must have length 1 (lambda), got {}",
                            proc.name,
                            params.len()
                        );
                    }
                    (unbinned_types::pdf_kind::EXPONENTIAL, params.clone(), 0, 0)
                }
                PdfSpec::CrystalBall { observable, params } => {
                    if observable != &obs0.name {
                        anyhow::bail!(
                            "process '{}' pdf.observable '{}' must match channel observable '{}'",
                            proc.name,
                            observable,
                            obs0.name
                        );
                    }
                    if params.len() != 4 {
                        anyhow::bail!(
                            "process '{}' crystal_ball params must have length 4 (mu,sigma,alpha,n), got {}",
                            proc.name,
                            params.len()
                        );
                    }
                    (unbinned_types::pdf_kind::CRYSTAL_BALL, params.clone(), 0, 0)
                }
                PdfSpec::DoubleCrystalBall { observable, params } => {
                    if observable != &obs0.name {
                        anyhow::bail!(
                            "process '{}' pdf.observable '{}' must match channel observable '{}'",
                            proc.name,
                            observable,
                            obs0.name
                        );
                    }
                    if params.len() != 6 {
                        anyhow::bail!(
                            "process '{}' double_crystal_ball params must have length 6 (mu,sigma,alpha_l,n_l,alpha_r,n_r), got {}",
                            proc.name,
                            params.len()
                        );
                    }
                    (unbinned_types::pdf_kind::DOUBLE_CRYSTAL_BALL, params.clone(), 0, 0)
                }
                PdfSpec::Chebyshev { observable, params } => {
                    if observable != &obs0.name {
                        anyhow::bail!(
                            "process '{}' pdf.observable '{}' must match channel observable '{}'",
                            proc.name,
                            observable,
                            obs0.name
                        );
                    }
                    if params.is_empty() {
                        anyhow::bail!(
                            "process '{}' chebyshev params must have length >= 1 (c_1..c_m), got 0",
                            proc.name
                        );
                    }
                    if params.len() > GPU_MAX_CHEBYSHEV_ORDER {
                        anyhow::bail!(
                            "process '{}' chebyshev order too large for --gpu: got {}, max {}",
                            proc.name,
                            params.len(),
                            GPU_MAX_CHEBYSHEV_ORDER
                        );
                    }
                    (unbinned_types::pdf_kind::CHEBYSHEV, params.clone(), 0, 0)
                }
                PdfSpec::Histogram { observable, bin_edges, bin_content, pseudo_count } => {
                    let (aux_offset, aux_len) = push_histogram_aux(
                        &mut pdf_aux_f64,
                        &proc.name,
                        &ch.name,
                        observable,
                        &obs0.name,
                        (a, b),
                        bin_edges,
                        bin_content,
                        *pseudo_count,
                    )?;
                    (unbinned_types::pdf_kind::HISTOGRAM, Vec::new(), aux_offset, aux_len)
                }
                other => {
                    anyhow::bail!(
                        "unbinned --gpu supports only gaussian/exponential/crystal_ball/double_crystal_ball/chebyshev/histogram PDFs for now (CPU-only: argus/voigtian/spline/kde/*_from_tree/flows); process '{}' has unsupported pdf {:?}",
                        proc.name,
                        other
                    );
                }
            };

        let shape_param_offset = shape_param_indices.len() as u32;
        for name in &shape_names {
            let idx = *index_by_name.get(name).ok_or_else(|| {
                anyhow::anyhow!(
                    "process '{}' references unknown shape parameter '{}'",
                    proc.name,
                    name
                )
            })?;
            shape_param_indices.push(idx as u32);
        }

        let (yield_kind, base_yield, yield_param_idx, modifiers): (
            u32,
            f64,
            u32,
            &[RateModifierSpec],
        ) = match &proc.yield_spec {
            YieldSpec::Fixed { value, modifiers } => {
                (unbinned_types::yield_kind::FIXED, *value, 0u32, modifiers.as_slice())
            }
            YieldSpec::Parameter { name, modifiers } => {
                let idx = *index_by_name.get(name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "process '{}' references unknown yield parameter '{}'",
                        proc.name,
                        name
                    )
                })?;
                (unbinned_types::yield_kind::PARAMETER, 0.0, idx as u32, modifiers.as_slice())
            }
            YieldSpec::Scaled { base_yield, scale, modifiers } => {
                let idx = *index_by_name.get(scale).ok_or_else(|| {
                    anyhow::anyhow!(
                        "process '{}' references unknown scale parameter '{}'",
                        proc.name,
                        scale
                    )
                })?;
                (unbinned_types::yield_kind::SCALED, *base_yield, idx as u32, modifiers.as_slice())
            }
        };

        let rate_mod_offset = rate_modifiers.len() as u32;
        for m in modifiers {
            match m {
                RateModifierSpec::NormSys { param, lo, hi } => {
                    if !(*lo).is_finite() || *lo <= 0.0 || !(*hi).is_finite() || *hi <= 0.0 {
                        anyhow::bail!(
                            "process '{}' has invalid NormSys lo/hi (must be finite and > 0): lo={}, hi={}",
                            proc.name,
                            lo,
                            hi
                        );
                    }
                    let alpha_idx = *index_by_name.get(param).ok_or_else(|| {
                        anyhow::anyhow!(
                            "process '{}' references unknown NormSys parameter '{}'",
                            proc.name,
                            param
                        )
                    })?;
                    rate_modifiers.push(unbinned_types::GpuUnbinnedRateModifierDesc {
                        kind: unbinned_types::rate_modifier_kind::NORM_SYS,
                        alpha_param_idx: alpha_idx as u32,
                        interp_code: 0,
                        _pad: 0,
                        lo: *lo,
                        hi: *hi,
                    });
                }
                RateModifierSpec::WeightSys { param, lo, hi, interp_code } => {
                    if !(*lo).is_finite() || *lo <= 0.0 || !(*hi).is_finite() || *hi <= 0.0 {
                        anyhow::bail!(
                            "process '{}' has invalid WeightSys lo/hi (must be finite and > 0): lo={}, hi={}",
                            proc.name,
                            lo,
                            hi
                        );
                    }
                    let alpha_idx = *index_by_name.get(param).ok_or_else(|| {
                        anyhow::anyhow!(
                            "process '{}' references unknown WeightSys parameter '{}'",
                            proc.name,
                            param
                        )
                    })?;
                    let gpu_interp = match interp_code.as_deref() {
                        Some("code4p") | Some("Code4p") => 1u32,
                        _ => 0u32,
                    };
                    rate_modifiers.push(unbinned_types::GpuUnbinnedRateModifierDesc {
                        kind: unbinned_types::rate_modifier_kind::WEIGHT_SYS,
                        alpha_param_idx: alpha_idx as u32,
                        interp_code: gpu_interp,
                        _pad: 0,
                        lo: *lo,
                        hi: *hi,
                    });
                }
            }
        }
        let n_rate_mods = (rate_modifiers.len() as u32) - rate_mod_offset;

        processes.push(unbinned_types::GpuUnbinnedProcessDesc {
            base_yield,
            pdf_kind,
            yield_kind,
            obs_index: 0,
            shape_param_offset,
            n_shape_params: shape_names.len() as u32,
            yield_param_idx,
            rate_mod_offset,
            n_rate_mods,
            pdf_aux_offset,
            pdf_aux_len,
        });
    }

    // Gaussian constraints.
    let mut gauss_constraints = Vec::<unbinned_types::GpuUnbinnedGaussConstraintEntry>::new();
    let mut constraint_const = 0.0f64;
    for (idx, p) in spec.model.parameters.iter().enumerate() {
        let Some(c) = &p.constraint else { continue };
        match c {
            ConstraintSpec::Gaussian { mean, sigma } => {
                if !(sigma.is_finite() && *sigma > 0.0) {
                    anyhow::bail!(
                        "Gaussian constraint sigma must be finite and > 0 for '{}', got {}",
                        p.name,
                        sigma
                    );
                }
                gauss_constraints.push(unbinned_types::GpuUnbinnedGaussConstraintEntry {
                    center: *mean,
                    inv_width: 1.0 / *sigma,
                    param_idx: idx as u32,
                    _pad: 0,
                });
                constraint_const += sigma.ln() + 0.5 * (2.0 * std::f64::consts::PI).ln();
            }
        }
    }

    let gpu = UnbinnedGpuModelData {
        n_params: meta.parameter_names.len(),
        n_obs: 1,
        n_events: 0,
        obs_bounds: vec![(a, b)],
        obs_soa: Vec::new(),
        event_weights: None,
        processes,
        rate_modifiers,
        shape_param_indices,
        pdf_aux_f64,
        gauss_constraints,
        constraint_const,
    };

    Ok((meta, gpu))
}

/// Build GPU-lowered *static* unbinned model representations from the spec (one per channel).
///
/// Unlike `build_gpu_data`, this does not load observed data from ROOT. It is intended for
/// GPU batch toy fitting, where toy datasets are provided separately.
///
/// Notes:
/// - Only channels with `include_in_fit=true` are lowered.
/// - Constraints are attached to the first included channel only (to avoid double counting).
pub(crate) fn build_gpu_static_datas(
    spec: &UnbinnedSpecV0,
    spec_path: &Path,
) -> Result<(UnbinnedGpuMeta, Vec<UnbinnedGpuModelData>)> {
    let (meta, index_by_name) = build_meta(spec)?;
    let base_dir = spec_path.parent().unwrap_or_else(|| Path::new("."));

    let mut datas = Vec::<UnbinnedGpuModelData>::new();

    // Gaussian constraints (attached once).
    let mut gauss_constraints = Vec::<unbinned_types::GpuUnbinnedGaussConstraintEntry>::new();
    let mut constraint_const = 0.0f64;
    for (idx, p) in spec.model.parameters.iter().enumerate() {
        let Some(c) = &p.constraint else { continue };
        match c {
            ConstraintSpec::Gaussian { mean, sigma } => {
                if !(sigma.is_finite() && *sigma > 0.0) {
                    anyhow::bail!(
                        "Gaussian constraint sigma must be finite and > 0 for '{}', got {}",
                        p.name,
                        sigma
                    );
                }
                gauss_constraints.push(unbinned_types::GpuUnbinnedGaussConstraintEntry {
                    center: *mean,
                    inv_width: 1.0 / *sigma,
                    param_idx: idx as u32,
                    _pad: 0,
                });
                constraint_const += sigma.ln() + 0.5 * (2.0 * std::f64::consts::PI).ln();
            }
        }
    }

    for ch in &spec.channels {
        if !ch.include_in_fit {
            continue;
        }
        if ch.observables.len() != 1 {
            anyhow::bail!(
                "unbinned --gpu currently supports exactly 1 observable per channel, got {} in channel '{}'",
                ch.observables.len(),
                ch.name
            );
        }
        if ch.processes.is_empty() {
            anyhow::bail!("channel '{}' has no processes", ch.name);
        }

        let obs0 = &ch.observables[0];
        let obs_expr = obs0.expr.clone().unwrap_or_else(|| obs0.name.clone());
        let (a, b) = (obs0.bounds[0], obs0.bounds[1]);
        if !(a.is_finite() && b.is_finite() && a < b) {
            anyhow::bail!("invalid observable bounds for '{}': [{a}, {b}]", obs0.name);
        }

        let mut processes =
            Vec::<unbinned_types::GpuUnbinnedProcessDesc>::with_capacity(ch.processes.len());
        let mut shape_param_indices = Vec::<u32>::new();
        let mut rate_modifiers = Vec::<unbinned_types::GpuUnbinnedRateModifierDesc>::new();
        let mut pdf_aux_f64 = Vec::<f64>::new();

        for proc in &ch.processes {
            let mut derived_rate_modifiers =
                Vec::<unbinned_types::GpuUnbinnedRateModifierDesc>::new();
            let (pdf_kind, shape_names, pdf_aux_offset, pdf_aux_len): (u32, Vec<String>, u32, u32) =
                match &proc.pdf {
                    PdfSpec::Gaussian { observable, params } => {
                        if observable != &obs0.name {
                            anyhow::bail!(
                                "process '{}' pdf.observable '{}' must match channel observable '{}' (channel '{}')",
                                proc.name,
                                observable,
                                obs0.name,
                                ch.name
                            );
                        }
                        if params.len() != 2 {
                            anyhow::bail!(
                                "process '{}' gaussian params must have length 2 (mu,sigma), got {}",
                                proc.name,
                                params.len()
                            );
                        }
                        (unbinned_types::pdf_kind::GAUSSIAN, params.clone(), 0, 0)
                    }
                    PdfSpec::Exponential { observable, params } => {
                        if observable != &obs0.name {
                            anyhow::bail!(
                                "process '{}' pdf.observable '{}' must match channel observable '{}' (channel '{}')",
                                proc.name,
                                observable,
                                obs0.name,
                                ch.name
                            );
                        }
                        if params.len() != 1 {
                            anyhow::bail!(
                                "process '{}' exponential params must have length 1 (lambda), got {}",
                                proc.name,
                                params.len()
                            );
                        }
                        (unbinned_types::pdf_kind::EXPONENTIAL, params.clone(), 0, 0)
                    }
                    PdfSpec::CrystalBall { observable, params } => {
                        if observable != &obs0.name {
                            anyhow::bail!(
                                "process '{}' pdf.observable '{}' must match channel observable '{}' (channel '{}')",
                                proc.name,
                                observable,
                                obs0.name,
                                ch.name
                            );
                        }
                        if params.len() != 4 {
                            anyhow::bail!(
                                "process '{}' crystal_ball params must have length 4 (mu,sigma,alpha,n), got {}",
                                proc.name,
                                params.len()
                            );
                        }
                        (unbinned_types::pdf_kind::CRYSTAL_BALL, params.clone(), 0, 0)
                    }
                    PdfSpec::DoubleCrystalBall { observable, params } => {
                        if observable != &obs0.name {
                            anyhow::bail!(
                                "process '{}' pdf.observable '{}' must match channel observable '{}' (channel '{}')",
                                proc.name,
                                observable,
                                obs0.name,
                                ch.name
                            );
                        }
                        if params.len() != 6 {
                            anyhow::bail!(
                                "process '{}' double_crystal_ball params must have length 6 (mu,sigma,alpha_l,n_l,alpha_r,n_r), got {}",
                                proc.name,
                                params.len()
                            );
                        }
                        (unbinned_types::pdf_kind::DOUBLE_CRYSTAL_BALL, params.clone(), 0, 0)
                    }
                    PdfSpec::Chebyshev { observable, params } => {
                        if observable != &obs0.name {
                            anyhow::bail!(
                                "process '{}' pdf.observable '{}' must match channel observable '{}' (channel '{}')",
                                proc.name,
                                observable,
                                obs0.name,
                                ch.name
                            );
                        }
                        if params.is_empty() {
                            anyhow::bail!(
                                "process '{}' chebyshev params must have length >= 1 (c_1..c_m), got 0",
                                proc.name
                            );
                        }
                        if params.len() > GPU_MAX_CHEBYSHEV_ORDER {
                            anyhow::bail!(
                                "process '{}' chebyshev order too large for --gpu: got {}, max {}",
                                proc.name,
                                params.len(),
                                GPU_MAX_CHEBYSHEV_ORDER
                            );
                        }
                        (unbinned_types::pdf_kind::CHEBYSHEV, params.clone(), 0, 0)
                    }
                    PdfSpec::Histogram { observable, bin_edges, bin_content, pseudo_count } => {
                        let (aux_offset, aux_len) = push_histogram_aux(
                            &mut pdf_aux_f64,
                            &proc.name,
                            &ch.name,
                            observable,
                            &obs0.name,
                            (a, b),
                            bin_edges,
                            bin_content,
                            *pseudo_count,
                        )?;
                        (unbinned_types::pdf_kind::HISTOGRAM, Vec::new(), aux_offset, aux_len)
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
                        let (aux_offset, aux_len, derived) =
                            lower_histogram_from_tree_to_gpu_histogram(
                                &mut pdf_aux_f64,
                                base_dir,
                                &index_by_name,
                                &proc.name,
                                &ch.name,
                                &obs0.name,
                                &obs_expr,
                                (a, b),
                                observable,
                                bin_edges,
                                *pseudo_count,
                                source,
                                *max_events,
                                weight_systematics,
                                horizontal_systematics,
                            )?;
                        derived_rate_modifiers = derived;
                        (unbinned_types::pdf_kind::HISTOGRAM, Vec::new(), aux_offset, aux_len)
                    }
                    other => {
                        anyhow::bail!(
                            "unbinned --gpu supports only gaussian/exponential/crystal_ball/double_crystal_ball/chebyshev/histogram/histogram_from_tree PDFs for now (CPU-only: argus/voigtian/spline/kde/kde_from_tree/flows); process '{}' (channel '{}') has unsupported pdf {:?}",
                            proc.name,
                            ch.name,
                            other
                        );
                    }
                };

            let shape_param_offset = shape_param_indices.len() as u32;
            for name in &shape_names {
                let idx = *index_by_name.get(name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "process '{}' references unknown shape parameter '{}' (channel '{}')",
                        proc.name,
                        name,
                        ch.name
                    )
                })?;
                shape_param_indices.push(idx as u32);
            }

            let (yield_kind, base_yield, yield_param_idx, modifiers): (
                u32,
                f64,
                u32,
                &[RateModifierSpec],
            ) = match &proc.yield_spec {
                YieldSpec::Fixed { value, modifiers } => {
                    (unbinned_types::yield_kind::FIXED, *value, 0u32, modifiers.as_slice())
                }
                YieldSpec::Parameter { name, modifiers } => {
                    let idx = *index_by_name.get(name).ok_or_else(|| {
                        anyhow::anyhow!(
                            "process '{}' references unknown yield parameter '{}' (channel '{}')",
                            proc.name,
                            name,
                            ch.name
                        )
                    })?;
                    (unbinned_types::yield_kind::PARAMETER, 0.0, idx as u32, modifiers.as_slice())
                }
                YieldSpec::Scaled { base_yield, scale, modifiers } => {
                    let idx = *index_by_name.get(scale).ok_or_else(|| {
                        anyhow::anyhow!(
                            "process '{}' references unknown scale parameter '{}' (channel '{}')",
                            proc.name,
                            scale,
                            ch.name
                        )
                    })?;
                    (
                        unbinned_types::yield_kind::SCALED,
                        *base_yield,
                        idx as u32,
                        modifiers.as_slice(),
                    )
                }
            };

            let rate_mod_offset = rate_modifiers.len() as u32;
            for m in modifiers {
                match m {
                    RateModifierSpec::NormSys { param, lo, hi } => {
                        if !(*lo).is_finite() || *lo <= 0.0 || !(*hi).is_finite() || *hi <= 0.0 {
                            anyhow::bail!(
                                "process '{}' has invalid NormSys lo/hi (must be finite and > 0): lo={}, hi={} (channel '{}')",
                                proc.name,
                                lo,
                                hi,
                                ch.name
                            );
                        }
                        let alpha_idx = *index_by_name.get(param).ok_or_else(|| {
                            anyhow::anyhow!(
                                "process '{}' references unknown NormSys parameter '{}' (channel '{}')",
                                proc.name,
                                param,
                                ch.name
                            )
                        })?;
                        rate_modifiers.push(unbinned_types::GpuUnbinnedRateModifierDesc {
                            kind: unbinned_types::rate_modifier_kind::NORM_SYS,
                            alpha_param_idx: alpha_idx as u32,
                            interp_code: 0,
                            _pad: 0,
                            lo: *lo,
                            hi: *hi,
                        });
                    }
                    RateModifierSpec::WeightSys { param, lo, hi, interp_code } => {
                        if !(*lo).is_finite() || *lo <= 0.0 || !(*hi).is_finite() || *hi <= 0.0 {
                            anyhow::bail!(
                                "process '{}' has invalid WeightSys lo/hi (must be finite and > 0): lo={}, hi={} (channel '{}')",
                                proc.name,
                                lo,
                                hi,
                                ch.name
                            );
                        }
                        let alpha_idx = *index_by_name.get(param).ok_or_else(|| {
                            anyhow::anyhow!(
                                "process '{}' references unknown WeightSys parameter '{}' (channel '{}')",
                                proc.name,
                                param,
                                ch.name
                            )
                        })?;
                        let gpu_interp = match interp_code.as_deref() {
                            Some("code4p") | Some("Code4p") => 1u32,
                            _ => 0u32,
                        };
                        rate_modifiers.push(unbinned_types::GpuUnbinnedRateModifierDesc {
                            kind: unbinned_types::rate_modifier_kind::WEIGHT_SYS,
                            alpha_param_idx: alpha_idx as u32,
                            interp_code: gpu_interp,
                            _pad: 0,
                            lo: *lo,
                            hi: *hi,
                        });
                    }
                }
            }
            rate_modifiers.extend(derived_rate_modifiers);
            let n_rate_mods = (rate_modifiers.len() as u32) - rate_mod_offset;

            processes.push(unbinned_types::GpuUnbinnedProcessDesc {
                base_yield,
                pdf_kind,
                yield_kind,
                obs_index: 0,
                shape_param_offset,
                n_shape_params: shape_names.len() as u32,
                yield_param_idx,
                rate_mod_offset,
                n_rate_mods,
                pdf_aux_offset,
                pdf_aux_len,
            });
        }

        datas.push(UnbinnedGpuModelData {
            n_params: meta.parameter_names.len(),
            n_obs: 1,
            n_events: 0,
            obs_bounds: vec![(a, b)],
            obs_soa: Vec::new(),
            event_weights: None,
            processes,
            rate_modifiers,
            shape_param_indices,
            pdf_aux_f64,
            gauss_constraints: Vec::new(),
            constraint_const: 0.0,
        });
    }

    if datas.is_empty() {
        anyhow::bail!("unbinned --gpu requires at least one channel with include_in_fit=true");
    }

    // Attach constraints to the first included channel only.
    datas[0].gauss_constraints = gauss_constraints;
    datas[0].constraint_const = constraint_const;

    Ok((meta, datas))
}

#[cfg(feature = "cuda")]
pub fn compile_cuda(
    spec: &UnbinnedSpecV0,
    spec_path: &Path,
) -> Result<UnbinnedGpuModel<MultiUnbinnedAccel<ns_compute::cuda_unbinned::CudaUnbinnedAccelerator>>>
{
    let (meta, gpu_datas) = build_gpu_data(spec, spec_path)?;
    let mut accels = Vec::with_capacity(gpu_datas.len());
    for (ch_idx, gpu_data) in gpu_datas.iter().enumerate() {
        let accel =
            ns_compute::cuda_unbinned::CudaUnbinnedAccelerator::from_unbinned_data(gpu_data)
                .with_context(|| {
                    format!("failed to create CUDA unbinned accelerator (channel index {ch_idx})")
                })?;
        accels.push(accel);
    }
    let multi = MultiUnbinnedAccel::new(accels).map_err(|e| anyhow::anyhow!(e))?;
    Ok(UnbinnedGpuModel { meta, accel: Arc::new(Mutex::new(multi)) })
}

#[cfg(feature = "metal")]
pub fn compile_metal(
    spec: &UnbinnedSpecV0,
    spec_path: &Path,
) -> Result<
    UnbinnedGpuModel<MultiUnbinnedAccel<ns_compute::metal_unbinned::MetalUnbinnedAccelerator>>,
> {
    let (meta, gpu_datas) = build_gpu_data(spec, spec_path)?;
    let mut accels = Vec::with_capacity(gpu_datas.len());
    for (ch_idx, gpu_data) in gpu_datas.iter().enumerate() {
        let accel =
            ns_compute::metal_unbinned::MetalUnbinnedAccelerator::from_unbinned_data(gpu_data)
                .with_context(|| {
                    format!("failed to create Metal unbinned accelerator (channel index {ch_idx})")
                })?;
        accels.push(accel);
    }
    let multi = MultiUnbinnedAccel::new(accels).map_err(|e| anyhow::anyhow!(e))?;
    Ok(UnbinnedGpuModel { meta, accel: Arc::new(Mutex::new(multi)) })
}

// ── G2-R1: Flow PDF detection + CUDA batch config builder ─────────────────

/// Returns `true` if any included channel has a Flow, ConditionalFlow, or DcrSurrogate PDF.
pub(crate) fn spec_has_flow_pdfs(spec: &UnbinnedSpecV0) -> bool {
    spec.channels.iter().filter(|ch| ch.include_in_fit).any(|ch| {
        ch.processes.iter().any(|p| {
            matches!(
                &p.pdf,
                PdfSpec::Flow { .. }
                    | PdfSpec::ConditionalFlow { .. }
                    | PdfSpec::DcrSurrogate { .. }
            )
        })
    })
}

/// Build Gaussian constraint entries from the spec (reusable across analytical and flow paths).
pub(crate) fn build_gauss_constraints(
    spec: &UnbinnedSpecV0,
) -> Result<(Vec<unbinned_types::GpuUnbinnedGaussConstraintEntry>, f64)> {
    let mut gauss_constraints = Vec::new();
    let mut constraint_const = 0.0f64;
    for (idx, p) in spec.model.parameters.iter().enumerate() {
        let Some(c) = &p.constraint else { continue };
        match c {
            ConstraintSpec::Gaussian { mean, sigma } => {
                if !(sigma.is_finite() && *sigma > 0.0) {
                    anyhow::bail!(
                        "Gaussian constraint sigma must be finite and > 0 for '{}', got {}",
                        p.name,
                        sigma
                    );
                }
                gauss_constraints.push(unbinned_types::GpuUnbinnedGaussConstraintEntry {
                    center: *mean,
                    inv_width: 1.0 / *sigma,
                    param_idx: idx as u32,
                    _pad: 0,
                });
                constraint_const += sigma.ln() + 0.5 * (2.0 * std::f64::consts::PI).ln();
            }
        }
    }
    Ok((gauss_constraints, constraint_const))
}

/// Build a `FlowBatchNllConfig` from the spec for a single included channel.
///
/// Returns `(config, process_pdf_indices)` where `process_pdf_indices[p]` is the index
/// into `model.channels()[ch_idx].processes` for the p-th process.
#[cfg(feature = "cuda")]
pub(crate) fn build_flow_batch_config(
    spec: &UnbinnedSpecV0,
    n_toys: usize,
    toy_offsets: Vec<u32>,
    total_events: usize,
) -> Result<ns_compute::cuda_flow_batch::FlowBatchNllConfig> {
    let (gauss_constraints, constraint_const) = build_gauss_constraints(spec)?;

    let (meta, index_by_name) = build_meta(spec)?;
    let n_params = meta.init.len();

    let included_channels: Vec<&crate::unbinned_spec::ChannelSpec> =
        spec.channels.iter().filter(|ch| ch.include_in_fit).collect();
    if included_channels.len() != 1 {
        anyhow::bail!(
            "--gpu cuda with flow PDFs currently supports exactly 1 included channel, got {}",
            included_channels.len()
        );
    }
    let ch = included_channels[0];

    let mut processes = Vec::with_capacity(ch.processes.len());
    for proc in &ch.processes {
        // Validate: no rate modifiers on flow processes (not supported by flow batch kernel).
        let has_modifiers = match &proc.yield_spec {
            YieldSpec::Fixed { modifiers, .. } => !modifiers.is_empty(),
            YieldSpec::Parameter { modifiers, .. } => !modifiers.is_empty(),
            YieldSpec::Scaled { modifiers, .. } => !modifiers.is_empty(),
        };
        if has_modifiers {
            anyhow::bail!(
                "process '{}': --gpu cuda with flow PDFs does not support rate modifiers (NormSys/WeightSys). Use CPU path instead.",
                proc.name
            );
        }

        let (base_yield, yield_param_idx, yield_is_scaled) = match &proc.yield_spec {
            YieldSpec::Fixed { value, .. } => (*value, None, false),
            YieldSpec::Parameter { name, .. } => {
                let idx = *index_by_name.get(name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "process '{}' references unknown yield parameter '{}'",
                        proc.name,
                        name
                    )
                })?;
                (0.0, Some(idx), false)
            }
            YieldSpec::Scaled { base_yield, scale, .. } => {
                let idx = *index_by_name.get(scale).ok_or_else(|| {
                    anyhow::anyhow!(
                        "process '{}' references unknown scale parameter '{}'",
                        proc.name,
                        scale
                    )
                })?;
                (*base_yield, Some(idx), true)
            }
        };

        processes.push(ns_compute::cuda_flow_batch::FlowBatchProcessDesc {
            base_yield,
            yield_param_idx,
            yield_is_scaled,
        });
    }

    Ok(ns_compute::cuda_flow_batch::FlowBatchNllConfig {
        total_events,
        n_toys,
        toy_offsets,
        processes,
        n_params,
        gauss_constraints,
        constraint_const,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::{Path, PathBuf};

    fn fixture_root_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("tests/fixtures/simple_tree.root")
            .canonicalize()
            .unwrap_or_else(|_| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../..")
                    .join("tests/fixtures/simple_tree.root")
            })
    }

    fn make_histogram_from_tree_spec(
        root: &Path,
        apply_to_shape: bool,
        with_horizontal: bool,
    ) -> UnbinnedSpecV0 {
        let mut pdf = json!({
            "type": "histogram_from_tree",
            "observable": "mbb",
            "bin_edges": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
            "pseudo_count": 0.5,
            "max_events": 500,
            "source": { "file": root, "tree": "events", "weight": "weight_mc" },
            "weight_systematics": [
                {
                    "param": "alpha_jes",
                    "up": "weight_jes_up/weight_mc",
                    "down": "weight_jes_down/weight_mc",
                    "interp": "code4p",
                    "apply_to_shape": apply_to_shape,
                    "apply_to_yield": true
                }
            ]
        });
        if with_horizontal {
            pdf["horizontal_systematics"] = json!([
                { "param": "alpha_jes_h", "up": "mbb*1.02", "down": "mbb*0.98", "interp": "code4p" }
            ]);
        }

        serde_json::from_value(json!({
            "schema_version": "nextstat_unbinned_spec_v0",
            "model": {
                "parameters": [
                    { "name": "nu", "init": 900.0, "bounds": [0.0, 5000.0] },
                    { "name": "alpha_jes", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } },
                    { "name": "alpha_jes_h", "init": 0.0, "bounds": [-5.0, 5.0], "constraint": { "type": "gaussian", "mean": 0.0, "sigma": 1.0 } }
                ]
            },
            "channels": [
                {
                    "name": "SR",
                    "include_in_fit": true,
                    "data": { "file": root, "tree": "events" },
                    "observables": [ { "name": "mbb", "bounds": [0.0, 500.0] } ],
                    "processes": [
                        {
                            "name": "p",
                            "pdf": pdf,
                            "yield": { "type": "parameter", "name": "nu" }
                        }
                    ]
                }
            ]
        }))
        .expect("valid unbinned spec")
    }

    #[test]
    fn gpu_lowering_supports_histogram_from_tree_yield_only_systematics() {
        let root = fixture_root_path();
        if !root.exists() {
            return;
        }
        let spec = make_histogram_from_tree_spec(&root, false, false);
        let cfg_path = Path::new("/tmp/unbinned_spec_gpu_hist_from_tree.json");

        let (_meta, datas) =
            build_gpu_data(&spec, cfg_path).expect("GPU lowering should support yield-only mode");
        assert_eq!(datas.len(), 1);
        assert_eq!(datas[0].processes.len(), 1);
        assert_eq!(datas[0].processes[0].pdf_kind, unbinned_types::pdf_kind::HISTOGRAM);
        assert_eq!(datas[0].rate_modifiers.len(), 1);
        assert_eq!(datas[0].rate_modifiers[0].kind, unbinned_types::rate_modifier_kind::WEIGHT_SYS);

        let (_meta_static, static_datas) = build_gpu_static_datas(&spec, cfg_path)
            .expect("static GPU lowering should support yield-only mode");
        assert_eq!(static_datas.len(), 1);
        assert_eq!(static_datas[0].processes[0].pdf_kind, unbinned_types::pdf_kind::HISTOGRAM);
        assert_eq!(static_datas[0].rate_modifiers.len(), 1);
    }

    #[test]
    fn gpu_lowering_rejects_histogram_from_tree_shape_morphing() {
        let root = fixture_root_path();
        if !root.exists() {
            return;
        }
        let spec = make_histogram_from_tree_spec(&root, true, false);
        let cfg_path = Path::new("/tmp/unbinned_spec_gpu_hist_from_tree_shape.json");

        let err = build_gpu_data(&spec, cfg_path).expect_err("shape morphing should be rejected");
        let msg = format!("{err:#}");
        assert!(msg.contains("apply_to_shape=true"), "unexpected error message: {msg}");
    }

    #[test]
    fn gpu_lowering_rejects_histogram_from_tree_horizontal_systematics() {
        let root = fixture_root_path();
        if !root.exists() {
            return;
        }
        let spec = make_histogram_from_tree_spec(&root, false, true);
        let cfg_path = Path::new("/tmp/unbinned_spec_gpu_hist_from_tree_horizontal.json");

        let err = build_gpu_data(&spec, cfg_path)
            .expect_err("horizontal systematics should be rejected on GPU");
        let msg = format!("{err:#}");
        assert!(msg.contains("horizontal_systematics"), "unexpected error message: {msg}");
    }
}
