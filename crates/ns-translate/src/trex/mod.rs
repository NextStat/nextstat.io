//! TRExFitter config importer (subset) for TREx replacement workflows.
//!
//! This module parses a common, line-based TRExFitter-style config format and
//! converts it into a NextStat-compatible pyhf JSON `Workspace` via the existing
//! ntuple → workspace pipeline (`NtupleWorkspaceBuilder`).
//!
//! Current scope:
//! - Supports `ReadFrom: NTUP` (or omitted; defaults to NTUP).
//! - Supports `Region:` blocks (channel variable/binning/selection).
//! - Supports `Sample:` blocks (file/weight/type/regions + simple modifiers).
//! - Supports `Systematic:` blocks (norm/weight/tree) applied by sample/region.
//!
//! Not implemented (yet):
//! - `ReadFrom: HIST` and more advanced TRExFitter features (smoothing, pruning,
//!   symmetrisation, multi-POI, etc.).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use ns_core::{Error, Result};

use crate::ntuple::{ChannelConfig, NtupleModifier, NtupleWorkspaceBuilder, SampleConfig};
use crate::pyhf::Workspace;

#[derive(Debug, Clone)]
struct Attr {
    key: String,
    value: String,
    line: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Job,
    Region,
    Sample,
    Systematic,
}

#[derive(Debug, Clone)]
struct RawBlock {
    kind: BlockKind,
    name: String,
    attrs: Vec<Attr>,
}

fn strip_comment(line: &str) -> &str {
    // TRExFitter configs commonly use '#' for comments. Keep it conservative:
    // do not attempt quote-aware parsing.
    match line.find('#') {
        Some(i) => &line[..i],
        None => line,
    }
}

fn unquote(s: &str) -> &str {
    let s = s.trim();
    if s.len() >= 2 {
        let bytes = s.as_bytes();
        let first = bytes[0] as char;
        let last = bytes[bytes.len() - 1] as char;
        if (first == '"' && last == '"') || (first == '\'' && last == '\'') {
            return &s[1..s.len() - 1];
        }
    }
    s
}

fn split_kv(line: &str) -> Option<(String, String)> {
    let idx = line.find(':')?;
    let key = line[..idx].trim();
    if key.is_empty() {
        return None;
    }
    let value = unquote(line[idx + 1..].trim()).to_string();
    Some((key.to_string(), value))
}

fn key_eq(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

fn parse_list(value: &str) -> Vec<String> {
    value
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().to_string())
        .collect()
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" => Some(true),
        "1" | "true" | "yes" | "y" | "on" => Some(true),
        "0" | "false" | "no" | "n" | "off" => Some(false),
        _ => None,
    }
}

fn parse_f64(s: &str) -> Result<f64> {
    s.trim().parse::<f64>().map_err(|_| Error::Validation(format!("invalid float: {s:?}")))
}

fn parse_binning(value: &str) -> Result<Vec<f64>> {
    // Accept "0, 50, 100, ..." and "[0,50,100]".
    let v = value.trim().trim_start_matches('[').trim_end_matches(']');
    let parts: Vec<&str> = v.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    if parts.len() >= 2 {
        let mut edges = Vec::with_capacity(parts.len());
        for p in parts {
            edges.push(parse_f64(p)?);
        }
        return Ok(edges);
    }

    // Fallback: whitespace-separated.
    let parts_ws: Vec<&str> = v.split_whitespace().filter(|s| !s.is_empty()).collect();
    if parts_ws.len() >= 2 {
        let mut edges = Vec::with_capacity(parts_ws.len());
        for p in parts_ws {
            edges.push(parse_f64(p)?);
        }
        return Ok(edges);
    }

    Err(Error::Validation(format!("binning must have >= 2 edges, got: {value:?}")))
}

fn parse_raw(text: &str) -> Result<(Vec<Attr>, Vec<RawBlock>)> {
    let mut globals: Vec<Attr> = Vec::new();
    let mut blocks: Vec<RawBlock> = Vec::new();
    let mut current: Option<RawBlock> = None;

    for (i, raw_line) in text.lines().enumerate() {
        let line_no = i + 1;
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }
        let Some((key, value)) = split_kv(line) else {
            continue;
        };

        let kind = if key_eq(&key, "Job") {
            Some(BlockKind::Job)
        } else if key_eq(&key, "Region") || key_eq(&key, "Channel") {
            Some(BlockKind::Region)
        } else if key_eq(&key, "Sample") {
            Some(BlockKind::Sample)
        } else if key_eq(&key, "Systematic") {
            Some(BlockKind::Systematic)
        } else {
            None
        };

        if let Some(k) = kind {
            if let Some(b) = current.take() {
                blocks.push(b);
            }
            let name = value.trim().to_string();
            if name.is_empty() {
                return Err(Error::Validation(format!(
                    "{key} block missing name (line {line_no})"
                )));
            }
            current = Some(RawBlock { kind: k, name, attrs: Vec::new() });
            continue;
        }

        let attr = Attr { key, value, line: line_no };
        if let Some(ref mut b) = current {
            b.attrs.push(attr);
        } else {
            globals.push(attr);
        }
    }

    if let Some(b) = current.take() {
        blocks.push(b);
    }

    Ok((globals, blocks))
}

fn last_attr_value(attrs: &[Attr], key: &str) -> Option<String> {
    attrs.iter().rev().find(|a| key_eq(&a.key, key)).map(|a| a.value.clone())
}

fn all_attr_values(attrs: &[Attr], key: &str) -> Vec<String> {
    attrs.iter().filter(|a| key_eq(&a.key, key)).map(|a| a.value.clone()).collect()
}

/// Parsed TRExFitter-style configuration (subset).
#[derive(Debug, Clone)]
pub struct TrexConfig {
    /// `ReadFrom` mode, e.g. `NTUP` or `HIST`.
    pub read_from: Option<String>,
    /// Default TTree name for ntuple mode.
    pub tree_name: Option<String>,
    /// Measurement name (HistFactory/pyhf concept).
    pub measurement: Option<String>,
    /// Parameter of interest name.
    pub poi: Option<String>,
    /// Regions/channels in input order.
    pub regions: Vec<TrexRegion>,
    /// Samples in input order.
    pub samples: Vec<TrexSample>,
    /// Systematics in input order.
    pub systematics: Vec<TrexSystematic>,
}

/// One TREx region/channel block.
#[derive(Debug, Clone)]
pub struct TrexRegion {
    /// Region name.
    pub name: String,
    /// Variable expression to histogram.
    pub variable: String,
    /// Bin edges.
    pub binning: Vec<f64>,
    /// Optional selection/cut expression.
    pub selection: Option<String>,
    /// Optional explicit data file (ntuple).
    pub data_file: Option<PathBuf>,
    /// Optional override TTree name for data.
    pub data_tree_name: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SampleKind {
    Data,
    Mc,
}

/// One TREx sample block.
#[derive(Debug, Clone)]
pub struct TrexSample {
    /// Sample name.
    pub name: String,
    kind: SampleKind,
    /// ROOT file path.
    pub file: PathBuf,
    /// Optional tree name override for this sample.
    pub tree_name: Option<String>,
    /// Optional weight expression.
    pub weight: Option<String>,
    /// Optional region filter.
    pub regions: Option<Vec<String>>,
    /// Modifiers applied to this sample.
    pub modifiers: Vec<NtupleModifier>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SystKind {
    Norm,
    Weight,
    Tree,
}

/// One TREx systematic block.
#[derive(Debug, Clone)]
pub struct TrexSystematic {
    /// Systematic name (parameter name).
    pub name: String,
    kind: SystKind,
    /// Sample filter (names).
    pub samples: Vec<String>,
    /// Optional region filter (names).
    pub regions: Option<Vec<String>>,
    // Payload:
    lo: Option<f64>,
    hi: Option<f64>,
    weight_up: Option<String>,
    weight_down: Option<String>,
    file_up: Option<PathBuf>,
    file_down: Option<PathBuf>,
    tree_name: Option<String>,
}

impl TrexConfig {
    /// Parse a TRExFitter-style config file from a string.
    pub fn parse_str(text: &str) -> Result<Self> {
        let (globals, blocks) = parse_raw(text)?;

        // Globals (can also be present inside a Job block).
        let mut read_from = last_attr_value(&globals, "ReadFrom");
        let mut tree_name =
            last_attr_value(&globals, "TreeName").or_else(|| last_attr_value(&globals, "Tree"));
        let mut measurement = last_attr_value(&globals, "Measurement");
        let mut poi = last_attr_value(&globals, "POI").or_else(|| last_attr_value(&globals, "Poi"));

        let mut regions = Vec::new();
        let mut samples = Vec::new();
        let mut systematics = Vec::new();

        for b in blocks {
            match b.kind {
                BlockKind::Job => {
                    // Treat Job attrs as global overrides.
                    if let Some(v) = last_attr_value(&b.attrs, "ReadFrom") {
                        read_from = Some(v);
                    }
                    if let Some(v) = last_attr_value(&b.attrs, "TreeName")
                        .or_else(|| last_attr_value(&b.attrs, "Tree"))
                    {
                        tree_name = Some(v);
                    }
                    if let Some(v) = last_attr_value(&b.attrs, "Measurement") {
                        measurement = Some(v);
                    }
                    if let Some(v) = last_attr_value(&b.attrs, "POI")
                        .or_else(|| last_attr_value(&b.attrs, "Poi"))
                    {
                        poi = Some(v);
                    }
                }
                BlockKind::Region => {
                    regions.push(parse_region_block(&b)?);
                }
                BlockKind::Sample => {
                    samples.push(parse_sample_block(&b)?);
                }
                BlockKind::Systematic => {
                    systematics.push(parse_systematic_block(&b)?);
                }
            }
        }

        Ok(Self { read_from, tree_name, measurement, poi, regions, samples, systematics })
    }
}

fn parse_region_block(b: &RawBlock) -> Result<TrexRegion> {
    let name = b.name.clone();
    let variable = last_attr_value(&b.attrs, "Variable")
        .or_else(|| last_attr_value(&b.attrs, "Var"))
        .ok_or_else(|| {
            Error::Validation(format!(
                "Region '{name}' missing Variable (line {})",
                b.attrs.first().map(|a| a.line).unwrap_or(0)
            ))
        })?;
    let binning_str = last_attr_value(&b.attrs, "Binning")
        .or_else(|| last_attr_value(&b.attrs, "BinEdges"))
        .ok_or_else(|| Error::Validation(format!("Region '{name}' missing Binning")))?;
    let binning = parse_binning(&binning_str)?;
    if binning.len() < 2 {
        return Err(Error::Validation(format!("Region '{name}' Binning must have >=2 edges")));
    }

    let selection =
        last_attr_value(&b.attrs, "Selection").or_else(|| last_attr_value(&b.attrs, "Cut"));
    let data_file = last_attr_value(&b.attrs, "DataFile").map(PathBuf::from);
    let data_tree_name =
        last_attr_value(&b.attrs, "DataTreeName").or_else(|| last_attr_value(&b.attrs, "DataTree"));

    Ok(TrexRegion { name, variable, binning, selection, data_file, data_tree_name })
}

fn parse_sample_kind(value: Option<String>) -> SampleKind {
    let Some(v) = value else { return SampleKind::Mc };
    let v = v.trim().to_ascii_lowercase();
    if v == "data" { SampleKind::Data } else { SampleKind::Mc }
}

fn parse_sample_block(b: &RawBlock) -> Result<TrexSample> {
    let name = b.name.clone();

    let kind = parse_sample_kind(last_attr_value(&b.attrs, "Type"));

    let file = last_attr_value(&b.attrs, "File")
        .or_else(|| last_attr_value(&b.attrs, "Path"))
        .ok_or_else(|| Error::Validation(format!("Sample '{name}' missing File")))?;
    let file = PathBuf::from(file);

    let tree_name =
        last_attr_value(&b.attrs, "TreeName").or_else(|| last_attr_value(&b.attrs, "Tree"));
    let weight = last_attr_value(&b.attrs, "Weight");

    let regions = last_attr_value(&b.attrs, "Regions")
        .map(|v| parse_list(&v))
        .map(|xs| xs.into_iter().collect::<Vec<_>>())
        .filter(|xs| !xs.is_empty());

    let mut modifiers: Vec<NtupleModifier> = Vec::new();

    for v in all_attr_values(&b.attrs, "NormFactor") {
        for nf in parse_list(&v) {
            modifiers.push(NtupleModifier::NormFactor { name: nf });
        }
    }

    for v in all_attr_values(&b.attrs, "NormSys") {
        // Accept: "name lo hi" or "name lo,hi" or "name,lo,hi"
        let vv = v.replace(',', " ");
        let toks: Vec<&str> = vv.split_whitespace().collect();
        if toks.len() < 3 {
            return Err(Error::Validation(format!("Sample '{name}' invalid NormSys: {v:?}")));
        }
        let sys_name = toks[0].to_string();
        let lo = parse_f64(toks[1])?;
        let hi = parse_f64(toks[2])?;
        modifiers.push(NtupleModifier::NormSys { name: sys_name, lo, hi });
    }

    for v in all_attr_values(&b.attrs, "StatError") {
        if parse_bool(&v).unwrap_or(false) {
            modifiers.push(NtupleModifier::StatError);
        }
    }
    // Support a bare "StatError:" with empty value.
    if b.attrs.iter().any(|a| key_eq(&a.key, "StatError") && a.value.trim().is_empty()) {
        modifiers.push(NtupleModifier::StatError);
    }

    Ok(TrexSample { name, kind, file, tree_name, weight, regions, modifiers })
}

fn parse_syst_kind(value: &str) -> Option<SystKind> {
    let v = value.trim().to_ascii_lowercase();
    match v.as_str() {
        "norm" | "normsys" => Some(SystKind::Norm),
        "weight" | "weightsys" => Some(SystKind::Weight),
        "tree" | "treesys" => Some(SystKind::Tree),
        _ => None,
    }
}

fn parse_systematic_block(b: &RawBlock) -> Result<TrexSystematic> {
    let name = b.name.clone();

    let type_str = last_attr_value(&b.attrs, "Type")
        .ok_or_else(|| Error::Validation(format!("Systematic '{name}' missing Type")))?;
    let kind = parse_syst_kind(&type_str).ok_or_else(|| {
        Error::Validation(format!("Systematic '{name}' unknown Type: {type_str:?}"))
    })?;

    let samples_val = last_attr_value(&b.attrs, "Samples")
        .ok_or_else(|| Error::Validation(format!("Systematic '{name}' missing Samples")))?;
    let samples = parse_list(&samples_val);
    if samples.is_empty() {
        return Err(Error::Validation(format!("Systematic '{name}' Samples list is empty")));
    }

    let regions =
        last_attr_value(&b.attrs, "Regions").map(|v| parse_list(&v)).filter(|xs| !xs.is_empty());

    let mut out = TrexSystematic {
        name,
        kind,
        samples,
        regions,
        lo: None,
        hi: None,
        weight_up: None,
        weight_down: None,
        file_up: None,
        file_down: None,
        tree_name: None,
    };

    match kind {
        SystKind::Norm => {
            let lo = last_attr_value(&b.attrs, "Lo").or_else(|| last_attr_value(&b.attrs, "Down"));
            let hi = last_attr_value(&b.attrs, "Hi").or_else(|| last_attr_value(&b.attrs, "Up"));
            let (Some(lo), Some(hi)) = (lo, hi) else {
                return Err(Error::Validation(format!(
                    "Systematic '{}' (norm) requires Lo/Hi",
                    out.name
                )));
            };
            out.lo = Some(parse_f64(&lo)?);
            out.hi = Some(parse_f64(&hi)?);
        }
        SystKind::Weight => {
            let up =
                last_attr_value(&b.attrs, "WeightUp").or_else(|| last_attr_value(&b.attrs, "Up"));
            let down = last_attr_value(&b.attrs, "WeightDown")
                .or_else(|| last_attr_value(&b.attrs, "Down"));
            let (Some(up), Some(down)) = (up, down) else {
                return Err(Error::Validation(format!(
                    "Systematic '{}' (weight) requires WeightUp/WeightDown",
                    out.name
                )));
            };
            out.weight_up = Some(up);
            out.weight_down = Some(down);
        }
        SystKind::Tree => {
            let up = last_attr_value(&b.attrs, "FileUp")
                .or_else(|| last_attr_value(&b.attrs, "UpFile"))
                .or_else(|| last_attr_value(&b.attrs, "Up"));
            let down = last_attr_value(&b.attrs, "FileDown")
                .or_else(|| last_attr_value(&b.attrs, "DownFile"))
                .or_else(|| last_attr_value(&b.attrs, "Down"));
            let (Some(up), Some(down)) = (up, down) else {
                return Err(Error::Validation(format!(
                    "Systematic '{}' (tree) requires FileUp/FileDown",
                    out.name
                )));
            };
            out.file_up = Some(PathBuf::from(up));
            out.file_down = Some(PathBuf::from(down));
            out.tree_name =
                last_attr_value(&b.attrs, "TreeName").or_else(|| last_attr_value(&b.attrs, "Tree"));
        }
    }

    Ok(out)
}

fn sys_applies(sys: &TrexSystematic, region_name: &str, sample_name: &str) -> bool {
    if !sys.samples.iter().any(|s| s == sample_name) {
        return false;
    }
    if let Some(ref regions) = sys.regions {
        regions.iter().any(|r| r == region_name)
    } else {
        true
    }
}

fn sample_applies(sample: &TrexSample, region_name: &str) -> bool {
    if let Some(ref regions) = sample.regions {
        regions.iter().any(|r| r == region_name)
    } else {
        true
    }
}

fn sys_to_modifier(sys: &TrexSystematic) -> Result<NtupleModifier> {
    match sys.kind {
        SystKind::Norm => Ok(NtupleModifier::NormSys {
            name: sys.name.clone(),
            lo: sys.lo.ok_or_else(|| {
                Error::Validation(format!("Systematic '{}' missing lo", sys.name))
            })?,
            hi: sys.hi.ok_or_else(|| {
                Error::Validation(format!("Systematic '{}' missing hi", sys.name))
            })?,
        }),
        SystKind::Weight => Ok(NtupleModifier::WeightSys {
            name: sys.name.clone(),
            weight_up: sys.weight_up.clone().ok_or_else(|| {
                Error::Validation(format!("Systematic '{}' missing weight_up", sys.name))
            })?,
            weight_down: sys.weight_down.clone().ok_or_else(|| {
                Error::Validation(format!("Systematic '{}' missing weight_down", sys.name))
            })?,
        }),
        SystKind::Tree => Ok(NtupleModifier::TreeSys {
            name: sys.name.clone(),
            file_up: sys.file_up.clone().ok_or_else(|| {
                Error::Validation(format!("Systematic '{}' missing file_up", sys.name))
            })?,
            file_down: sys.file_down.clone().ok_or_else(|| {
                Error::Validation(format!("Systematic '{}' missing file_down", sys.name))
            })?,
            tree_name: sys.tree_name.clone(),
        }),
    }
}

/// Build a pyhf JSON `Workspace` from a parsed config (ntuple mode).
///
/// `base_dir` is used to resolve relative ROOT file paths (as in
/// `NtupleWorkspaceBuilder::ntuple_path`).
pub fn workspace_from_config(cfg: &TrexConfig, base_dir: &Path) -> Result<Workspace> {
    let read_from = cfg.read_from.as_deref().unwrap_or("NTUP").trim().to_ascii_uppercase();
    if read_from != "NTUP" {
        return Err(Error::NotImplemented(format!(
            "TREx config ReadFrom={read_from} is not supported yet (only NTUP)"
        )));
    }

    let tree_name = cfg.tree_name.clone().unwrap_or_else(|| "events".to_string());
    let measurement = cfg.measurement.clone().unwrap_or_else(|| "meas".to_string());
    let poi = cfg.poi.clone().unwrap_or_else(|| "mu".to_string());

    if cfg.regions.is_empty() {
        return Err(Error::Validation("TREx config has no Region blocks".into()));
    }
    if cfg.samples.is_empty() {
        return Err(Error::Validation("TREx config has no Sample blocks".into()));
    }

    // Basic validation: sample names are unique.
    let mut seen: HashSet<&str> = HashSet::new();
    for s in &cfg.samples {
        if !seen.insert(s.name.as_str()) {
            return Err(Error::Validation(format!("duplicate Sample name: {}", s.name)));
        }
    }

    let mut builder = NtupleWorkspaceBuilder::new()
        .ntuple_path(base_dir.to_path_buf())
        .tree_name(tree_name)
        .measurement(measurement, poi);

    // Quick index for data sample discovery.
    let data_samples: Vec<&TrexSample> =
        cfg.samples.iter().filter(|s| s.kind == SampleKind::Data).collect();

    for region in &cfg.regions {
        let mut ch =
            ChannelConfig::new(&region.name).variable(&region.variable).binning(&region.binning);

        if let Some(ref sel) = region.selection {
            ch = ch.selection(sel);
        }

        if let Some(ref data_file) = region.data_file {
            ch.data_file = Some(data_file.clone());
            ch.data_tree_name = region.data_tree_name.clone();
        } else {
            // Optional: infer data file from a DATA sample that applies to this region.
            let mut data_hits: Vec<&TrexSample> =
                data_samples.iter().copied().filter(|s| sample_applies(s, &region.name)).collect();
            data_hits.sort_by(|a, b| a.name.cmp(&b.name));
            if data_hits.len() > 1 {
                return Err(Error::Validation(format!(
                    "Region '{}' has multiple DATA samples (specify DataFile in Region): {}",
                    region.name,
                    data_hits.iter().map(|s| s.name.as_str()).collect::<Vec<_>>().join(", ")
                )));
            }
            if let Some(ds) = data_hits.pop() {
                ch.data_file = Some(ds.file.clone());
                ch.data_tree_name = ds.tree_name.clone();
            }
        }

        // Add samples in the input order.
        for s in &cfg.samples {
            if s.kind == SampleKind::Data {
                continue;
            }
            if !sample_applies(s, &region.name) {
                continue;
            }

            let mut sc = SampleConfig::new(&s.name, s.file.clone());
            sc.tree_name = s.tree_name.clone();
            sc.weight = s.weight.clone();
            sc.modifiers = s.modifiers.clone();

            for sys in &cfg.systematics {
                if sys_applies(sys, &region.name, &s.name) {
                    sc.modifiers.push(sys_to_modifier(sys)?);
                }
            }

            ch.samples.push(sc);
        }

        builder = builder.add_channel(ch);
    }

    builder.build()
}

/// Convenience: parse + build workspace in one step.
pub fn workspace_from_str(text: &str, base_dir: &Path) -> Result<Workspace> {
    let cfg = TrexConfig::parse_str(text)?;
    workspace_from_config(&cfg, base_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    #[test]
    fn trex_import_smoke_ntup_to_workspace() {
        let cfg = r#"
ReadFrom: NTUP
TreeName: events
Measurement: meas
POI: mu

Region: SR
Variable: mbb
Binning: 0, 50, 100, 150, 200, 300
Selection: njet >= 4

Sample: signal
Type: SIGNAL
File: tests/fixtures/simple_tree.root
Weight: weight_mc
Regions: SR
NormFactor: mu
StatError: true

Systematic: jes
Type: weight
Samples: signal
Regions: SR
WeightUp: weight_jes_up
WeightDown: weight_jes_down
"#;

        let ws = workspace_from_str(cfg, &repo_root()).expect("workspace build");
        assert_eq!(ws.channels.len(), 1);
        assert_eq!(ws.observations.len(), 1);
        assert_eq!(ws.channels[0].name, "SR");
        assert_eq!(ws.observations[0].name, "SR");

        let ch = &ws.channels[0];
        assert_eq!(ch.samples.len(), 1);
        assert_eq!(ch.samples[0].name, "signal");
        assert_eq!(ch.samples[0].data.len(), 5);

        // Asimov by default: observation equals sum of nominal samples.
        assert_eq!(ws.observations[0].data.len(), 5);
        assert_eq!(ws.observations[0].data, ch.samples[0].data);

        // Modifiers: normfactor + staterror + histosys (weightsys).
        let mut kinds: Vec<String> = ch.samples[0]
            .modifiers
            .iter()
            .map(|m| match m {
                crate::pyhf::Modifier::NormFactor { .. } => "normfactor",
                crate::pyhf::Modifier::StatError { .. } => "staterror",
                crate::pyhf::Modifier::HistoSys { .. } => "histosys",
                crate::pyhf::Modifier::NormSys { .. } => "normsys",
                crate::pyhf::Modifier::ShapeSys { .. } => "shapesys",
                crate::pyhf::Modifier::ShapeFactor { .. } => "shapefactor",
                crate::pyhf::Modifier::Lumi { .. } => "lumi",
            })
            .map(|s| s.to_string())
            .collect();
        kinds.sort();
        assert_eq!(kinds, vec!["histosys", "normfactor", "staterror"]);
    }

    #[test]
    fn trex_import_rejects_hist_mode() {
        let cfg = r#"
ReadFrom: HIST
Region: SR
Variable: mbb
Binning: 0, 1
Sample: x
File: tests/fixtures/simple_tree.root
"#;
        let err = workspace_from_str(cfg, &repo_root()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Not implemented"));
        assert!(msg.contains("HIST"));
    }
}
