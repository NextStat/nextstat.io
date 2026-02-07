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
//! - Full TRExFitter config surface (many optional blocks/knobs are ignored).
//! - `ReadFrom: HIST` supports importing a HistFactory export (`combination.xml`) plus TREx-like
//!   region/sample masking; it does **not** re-derive histograms/variations from NTUP inputs.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use ns_core::{Error, Result};
use ns_root::CompiledExpr;
use serde::Serialize;

use crate::ntuple::{ChannelConfig, NtupleModifier, NtupleWorkspaceBuilder, SampleConfig};
use crate::pyhf::{Channel, Observation, Sample, Workspace};

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
    NormFactor,
}

#[derive(Debug, Clone)]
struct RawBlock {
    kind: BlockKind,
    name: String,
    ctx_region: Option<String>,
    ctx_sample: Option<String>,
    closed_explicitly: bool,
    attrs: Vec<Attr>,
}

fn strip_comment(line: &str) -> &str {
    // TRExFitter configs commonly use '#' for comments; sometimes also '//' or '%' is used.
    // Be quote-aware so values like "w#1" keep the '#'.
    let mut in_single = false;
    let mut in_double = false;
    let mut escape = false;
    let bytes = line.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if escape {
            escape = false;
            i += 1;
            continue;
        }
        if in_single || in_double {
            if c == '\\' {
                escape = true;
                i += 1;
                continue;
            }
            if in_single && c == '\'' {
                in_single = false;
            } else if in_double && c == '"' {
                in_double = false;
            }
            i += 1;
            continue;
        }

        if c == '\'' {
            in_single = true;
            i += 1;
            continue;
        }
        if c == '"' {
            in_double = true;
            i += 1;
            continue;
        }

        if c == '#' {
            return &line[..i];
        }
        if c == '/' && i + 1 < bytes.len() && (bytes[i + 1] as char) == '/' {
            return &line[..i];
        }
        if c == '%' && (i == 0 || (bytes[i - 1] as char).is_whitespace()) {
            return &line[..i];
        }

        i += 1;
    }
    line
}

fn unquote(s: &str) -> String {
    let s = s.trim();
    if s.len() < 2 {
        return s.to_string();
    }
    let bytes = s.as_bytes();
    let first = bytes[0] as char;
    let last = bytes[bytes.len() - 1] as char;
    let is_quoted = (first == '"' && last == '"') || (first == '\'' && last == '\'');
    if !is_quoted {
        return s.to_string();
    }
    let inner = &s[1..s.len() - 1];
    // Minimal unescape: keep it conservative (TREx configs often use simple escaping).
    let mut out = String::with_capacity(inner.len());
    let mut escape = false;
    for ch in inner.chars() {
        if escape {
            let mapped = match ch {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                '\\' => '\\',
                '"' => '"',
                '\'' => '\'',
                other => other,
            };
            out.push(mapped);
            escape = false;
            continue;
        }
        if ch == '\\' {
            escape = true;
            continue;
        }
        out.push(ch);
    }
    out
}

fn split_kv(line: &str) -> Option<(String, String)> {
    // Split on ':' or '=' outside quotes. Also accept bare keys as flags.
    let mut in_single = false;
    let mut in_double = false;
    let mut escape = false;
    let bytes = line.as_bytes();
    let mut sep_idx: Option<usize> = None;
    for i in 0..bytes.len() {
        let c = bytes[i] as char;
        if escape {
            escape = false;
            continue;
        }
        if in_single || in_double {
            if c == '\\' {
                escape = true;
                continue;
            }
            if in_single && c == '\'' {
                in_single = false;
            } else if in_double && c == '"' {
                in_double = false;
            }
            continue;
        }

        if c == '\'' {
            in_single = true;
            continue;
        }
        if c == '"' {
            in_double = true;
            continue;
        }
        if c == ':' || c == '=' {
            sep_idx = Some(i);
            break;
        }
    }

    let (key, value) = match sep_idx {
        Some(idx) => (line[..idx].trim(), line[idx + 1..].trim()),
        None => (line.trim(), ""),
    };
    if key.is_empty() {
        return None;
    }

    let value = value.trim();
    // Only unquote a pure quoted scalar; leave lists like ["a", "b"] intact.
    let value = if (value.starts_with('"') && value.ends_with('"'))
        || (value.starts_with('\'') && value.ends_with('\''))
    {
        unquote(value)
    } else {
        value.to_string()
    };
    Some((key.to_string(), value))
}

fn key_eq(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

fn parse_list(value: &str) -> Vec<String> {
    // Accept:
    // - "a,b,c"
    // - "a b c"
    // - "[a, b, \"c d\"]"
    // - "\"a b\" c"
    let v = value.trim();
    let inner = v.strip_prefix('[').and_then(|s| s.strip_suffix(']')).unwrap_or(v);

    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut escape = false;

    let push_cur = |out: &mut Vec<String>, cur: &mut String| {
        let s = cur.trim();
        if !s.is_empty() {
            let s = if (s.starts_with('"') && s.ends_with('"'))
                || (s.starts_with('\'') && s.ends_with('\''))
            {
                unquote(s)
            } else {
                s.to_string()
            };
            if !s.trim().is_empty() {
                out.push(s);
            }
        }
        cur.clear();
    };

    for ch in inner.chars() {
        if escape {
            cur.push(ch);
            escape = false;
            continue;
        }
        if in_single || in_double {
            if ch == '\\' {
                cur.push(ch);
                escape = true;
                continue;
            }
            if in_single && ch == '\'' {
                in_single = false;
            } else if in_double && ch == '"' {
                in_double = false;
            }
            cur.push(ch);
            continue;
        }

        match ch {
            '\'' => {
                in_single = true;
                cur.push(ch);
            }
            '"' => {
                in_double = true;
                cur.push(ch);
            }
            ',' => push_cur(&mut out, &mut cur),
            c if c.is_whitespace() => push_cur(&mut out, &mut cur),
            _ => cur.push(ch),
        }
    }
    push_cur(&mut out, &mut cur);

    out
}

fn parse_semicolon_list(value: &str) -> Vec<String> {
    // TREx configs sometimes use semicolon-separated lists, commonly for Region names:
    //   Region: "a";"b"
    let v = value.trim();
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut escape = false;

    let push_cur = |out: &mut Vec<String>, cur: &mut String| {
        let s = cur.trim();
        if !s.is_empty() {
            let s = if (s.starts_with('"') && s.ends_with('"'))
                || (s.starts_with('\'') && s.ends_with('\''))
            {
                unquote(s)
            } else {
                s.to_string()
            };
            if !s.trim().is_empty() {
                out.push(s);
            }
        }
        cur.clear();
    };

    for ch in v.chars() {
        if escape {
            cur.push(ch);
            escape = false;
            continue;
        }
        if in_single || in_double {
            if ch == '\\' {
                cur.push(ch);
                escape = true;
                continue;
            }
            if in_single && ch == '\'' {
                in_single = false;
            } else if in_double && ch == '"' {
                in_double = false;
            }
            cur.push(ch);
            continue;
        }

        match ch {
            '\'' => {
                in_single = true;
                cur.push(ch);
            }
            '"' => {
                in_double = true;
                cur.push(ch);
            }
            ';' => push_cur(&mut out, &mut cur),
            _ => cur.push(ch),
        }
    }
    push_cur(&mut out, &mut cur);
    out
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
    // Accept "0, 50, 100, ...", "0; 50; 100; ...", and "[0,50,100]".
    let v = value.trim().trim_start_matches('[').trim_end_matches(']');
    let parts: Vec<&str> =
        v.split([',', ';']).map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
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
    let mut stack: Vec<RawBlock> = Vec::new();

    fn can_nest(parent: BlockKind, child: BlockKind) -> bool {
        match parent {
            BlockKind::Job => true,
            BlockKind::Region => matches!(child, BlockKind::Sample | BlockKind::Systematic),
            BlockKind::Sample => matches!(child, BlockKind::Systematic),
            BlockKind::Systematic => false,
            BlockKind::NormFactor => false,
        }
    }

    fn close_top(stack: &mut Vec<RawBlock>, blocks: &mut Vec<RawBlock>, explicit: bool) {
        if let Some(mut b) = stack.pop() {
            b.closed_explicitly = explicit;
            blocks.push(b);
        }
    }

    for (i, raw_line) in text.lines().enumerate() {
        let line_no = i + 1;
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }
        let Some((key, value)) = split_kv(line) else {
            continue;
        };

        // Ambiguity: some NextStat-compatible minimal configs use `NormFactor:` as a Sample attribute,
        // while full TREx configs use `NormFactor:` as a dedicated block. To preserve back-compat,
        // treat `NormFactor:` as a block only when not currently inside Region/Sample/Systematic.
        let can_start_normfactor_block = match stack.last() {
            Some(top) => !matches!(top.kind, BlockKind::Region | BlockKind::Sample | BlockKind::Systematic),
            None => true,
        };

        let kind = if key_eq(&key, "Job") {
            Some(BlockKind::Job)
        } else if key_eq(&key, "Region") || key_eq(&key, "Channel") {
            Some(BlockKind::Region)
        } else if key_eq(&key, "Sample") {
            Some(BlockKind::Sample)
        } else if key_eq(&key, "Systematic") {
            Some(BlockKind::Systematic)
        } else if key_eq(&key, "NormFactor") && can_start_normfactor_block {
            Some(BlockKind::NormFactor)
        } else if key_eq(&key, "EndJob") {
            Some(BlockKind::Job)
        } else if key_eq(&key, "EndRegion") || key_eq(&key, "EndChannel") {
            Some(BlockKind::Region)
        } else if key_eq(&key, "EndSample") {
            Some(BlockKind::Sample)
        } else if key_eq(&key, "EndSystematic") {
            Some(BlockKind::Systematic)
        } else if key_eq(&key, "EndNormFactor") {
            Some(BlockKind::NormFactor)
        } else {
            None
        };

        if let Some(k) = kind {
            // End markers: pop matching kind.
            if key.to_ascii_lowercase().starts_with("end") {
                // Close inner blocks implicitly until we find the matching one.
                let mut found = None;
                while let Some(top) = stack.pop() {
                    if top.kind == k {
                        found = Some(top);
                        break;
                    }
                    blocks.push(RawBlock { closed_explicitly: false, ..top });
                }
                let Some(top) = found else {
                    return Err(Error::Validation(format!(
                        "{key} without an open block (line {line_no})"
                    )));
                };
                blocks.push(RawBlock { closed_explicitly: true, ..top });
                continue;
            }

            // Start marker: push a new block, preserving context.
            let name = value.trim().to_string();
            if name.is_empty() {
                return Err(Error::Validation(format!(
                    "{key} block missing name (line {line_no})"
                )));
            }

            // Implicitly close blocks that cannot contain this new kind (legacy flat configs).
            while let Some(top) = stack.last() {
                if can_nest(top.kind, k) {
                    break;
                }
                close_top(&mut stack, &mut blocks, false);
            }

            let ctx_region =
                stack.iter().rev().find(|b| b.kind == BlockKind::Region).map(|b| b.name.clone());
            let ctx_sample =
                stack.iter().rev().find(|b| b.kind == BlockKind::Sample).map(|b| b.name.clone());
            stack.push(RawBlock {
                kind: k,
                name,
                ctx_region,
                ctx_sample,
                closed_explicitly: false,
                attrs: Vec::new(),
            });
            continue;
        }

        let attr = Attr { key, value, line: line_no };
        if let Some(top) = stack.last_mut() {
            top.attrs.push(attr);
        } else {
            globals.push(attr);
        }
    }

    // Flush any still-open blocks (common in the minimal subset format).
    while !stack.is_empty() {
        close_top(&mut stack, &mut blocks, false);
    }

    Ok((globals, blocks))
}

fn last_attr_value(attrs: &[Attr], key: &str) -> Option<String> {
    attrs.iter().rev().find(|a| key_eq(&a.key, key)).map(|a| a.value.clone())
}

fn all_attr_values(attrs: &[Attr], key: &str) -> Vec<String> {
    attrs.iter().filter(|a| key_eq(&a.key, key)).map(|a| a.value.clone()).collect()
}

fn has_attr(attrs: &[Attr], key: &str) -> bool {
    attrs.iter().any(|a| key_eq(&a.key, key))
}

/// Parsed TRExFitter-style configuration (subset).
#[derive(Debug, Clone)]
pub struct TrexConfig {
    /// `ReadFrom` mode, e.g. `NTUP` or `HIST`.
    pub read_from: Option<String>,
    /// HistFactory export directory (HIST mode) for resolving `combination.xml`.
    pub histo_path: Option<PathBuf>,
    /// Optional explicit HistFactory `combination.xml` path (HIST mode).
    pub combination_xml: Option<PathBuf>,
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
    Histo,
    Shape,
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
    weight_base: Option<String>,
    weight_up_suffix: Option<String>,
    weight_down_suffix: Option<String>,
    weight_suf_up: Option<String>,
    weight_suf_down: Option<String>,
    weight_up: Option<String>,
    weight_down: Option<String>,
    file_up: Option<PathBuf>,
    file_down: Option<PathBuf>,
    tree_name: Option<String>,
    // HISTO fields
    histo_name_up: Option<String>,
    histo_name_down: Option<String>,
    histo_file_up: Option<PathBuf>,
    histo_file_down: Option<PathBuf>,
    // Optional TREx knobs (currently parsed for coverage/parity work).
    smoothing: Option<String>,
    drop_norm: Option<String>,
    pruning: Option<String>,
}

/// A single unrecognized `key: value` attribute encountered while parsing a TREx config.
#[derive(Debug, Clone, Serialize)]
pub struct TrexUnknownAttr {
    /// Where it was found: `"global"` or `"block"`.
    pub scope: String,
    /// Block kind when `scope="block"` (e.g. `Region`, `Sample`, `Systematic`).
    pub block_kind: Option<String>,
    /// Block name when `scope="block"` (e.g. region/sample/systematic name).
    pub block_name: Option<String>,
    /// 1-based line number in the original config text.
    pub line: usize,
    /// Attribute key as it appeared in the config.
    pub key: String,
    /// Raw attribute value (unquoted/unescaped).
    pub value: String,
}

/// A best-effort "what did we understand?" report for TREx config parsing.
///
/// This is intended to help parity work against legacy configs: it flags keys/attrs that
/// NextStat currently ignores so we can iterate coverage intentionally.
#[derive(Debug, Clone, Serialize)]
pub struct TrexCoverageReport {
    /// Schema identifier for this JSON report.
    pub schema_version: String,
    /// Count of parsed global attributes.
    pub n_globals: usize,
    /// Count of parsed blocks (Region/Sample/Systematic/Job/...).
    pub n_blocks: usize,
    /// Count of parsed attributes inside blocks.
    pub n_block_attrs: usize,
    /// List of unknown attributes (not currently recognized by the importer).
    pub unknown: Vec<TrexUnknownAttr>,
}

/// A single expression-bearing `key: value` attribute encountered while parsing a TREx config.
///
/// Used for NTUP-mode parity work: surfaces unsupported constructs and required branch names early.
#[derive(Debug, Clone, Serialize)]
pub struct TrexExprCoverageItem {
    /// Where it was found: `"global"` or `"block"`.
    pub scope: String,
    /// Block kind when `scope="block"` (e.g. `Region`, `Sample`, `Systematic`).
    pub block_kind: Option<String>,
    /// Block name when `scope="block"` (e.g. region/sample/systematic name).
    pub block_name: Option<String>,
    /// Nested block context: parent Region name, when present.
    pub ctx_region: Option<String>,
    /// Nested block context: parent Sample name, when present.
    pub ctx_sample: Option<String>,
    /// 1-based line number in the original config text (best effort).
    pub line: usize,
    /// Attribute key as it appeared in the config (e.g. `Selection`, `WeightUp`).
    pub key: String,
    /// Semantic role of the expression (`selection`, `weight`, `variable`, `weight_up`, ...).
    pub role: String,
    /// Expression text (unquoted/unescaped). For derived expressions, this is the derived value.
    pub expr: String,
    /// Whether this expression was derived (e.g. WeightBase + suffix expansion).
    pub derived: bool,
    /// Whether compilation succeeded.
    pub ok: bool,
    /// Compilation error message when `ok=false`.
    pub error: Option<String>,
    /// Branch names referenced by this expression (ordered by first occurrence).
    pub required_branches: Vec<String>,
}

/// Best-effort report of which TREx config expressions compile under NextStat's ROOT-dialect engine.
#[derive(Debug, Clone, Serialize)]
pub struct TrexExprCoverageReport {
    /// Schema identifier for this JSON report.
    pub schema_version: String,
    /// Count of expressions discovered.
    pub n_exprs: usize,
    /// Count of expressions that compiled successfully.
    pub n_ok: usize,
    /// Count of expressions that failed compilation.
    pub n_err: usize,
    /// Per-expression records (includes successes + failures).
    pub items: Vec<TrexExprCoverageItem>,
}

impl TrexConfig {
    /// Parse a TRExFitter-style config file from a string.
    pub fn parse_str(text: &str) -> Result<Self> {
        let (globals, blocks) = parse_raw(text)?;
        Self::parse_from_raw(globals, blocks)
    }

    /// Parse a TRExFitter-style config from text and also return a coverage report.
    pub fn parse_str_with_coverage(text: &str) -> Result<(Self, TrexCoverageReport)> {
        let (globals, blocks) = parse_raw(text)?;
        let cfg = Self::parse_from_raw(globals.clone(), blocks.clone())?;
        let report = trex_coverage_from_raw(&globals, &blocks);
        Ok((cfg, report))
    }

    fn parse_from_raw(globals: Vec<Attr>, blocks: Vec<RawBlock>) -> Result<Self> {
        // Globals (can also be present inside a Job block).
        let mut read_from = last_attr_value(&globals, "ReadFrom");
        let mut histo_path = last_attr_value(&globals, "HistoPath")
            .or_else(|| last_attr_value(&globals, "HistPath"))
            .or_else(|| last_attr_value(&globals, "ExportDir"));
        let mut combination_xml = last_attr_value(&globals, "CombinationXml")
            .or_else(|| last_attr_value(&globals, "HistFactoryXml"))
            .or_else(|| last_attr_value(&globals, "CombinationXML"));
        let mut tree_name =
            last_attr_value(&globals, "TreeName").or_else(|| last_attr_value(&globals, "Tree"));
        tree_name = tree_name.or_else(|| last_attr_value(&globals, "NtupleName"));
        let mut measurement = last_attr_value(&globals, "Measurement");
        let mut poi = last_attr_value(&globals, "POI").or_else(|| last_attr_value(&globals, "Poi"));

        let mut regions = Vec::new();
        let mut samples = Vec::new();
        let mut systematics = Vec::new();
        let mut norm_factors: Vec<(String, Vec<String>)> = Vec::new();

        for b in blocks {
            match b.kind {
                BlockKind::Job => {
                    // Treat Job attrs as global overrides.
                    if let Some(v) = last_attr_value(&b.attrs, "ReadFrom") {
                        read_from = Some(v);
                    }
                    if let Some(v) = last_attr_value(&b.attrs, "HistoPath")
                        .or_else(|| last_attr_value(&b.attrs, "HistPath"))
                        .or_else(|| last_attr_value(&b.attrs, "ExportDir"))
                    {
                        histo_path = Some(v);
                    }
                    if let Some(v) = last_attr_value(&b.attrs, "CombinationXml")
                        .or_else(|| last_attr_value(&b.attrs, "HistFactoryXml"))
                        .or_else(|| last_attr_value(&b.attrs, "CombinationXML"))
                    {
                        combination_xml = Some(v);
                    }
                    if let Some(v) = last_attr_value(&b.attrs, "TreeName")
                        .or_else(|| last_attr_value(&b.attrs, "Tree"))
                    {
                        tree_name = Some(v);
                    }
                    if let Some(v) = last_attr_value(&b.attrs, "NtupleName") {
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
                    let base_region = parse_region_block(&b)?;
                    let names = if base_region.name.contains(';') {
                        parse_semicolon_list(&base_region.name)
                    } else {
                        vec![base_region.name.clone()]
                    };
                    if names.len() == 1 {
                        regions.push(base_region);
                    } else {
                        for n in names {
                            let mut r = base_region.clone();
                            r.name = n;
                            regions.push(r);
                        }
                    }
                }
                BlockKind::Sample => {
                    // Skip override-only nested samples (Region → Sample without File/Path).
                    // These are handled by `workspace_from_str` composition rules.
                    if b.ctx_region.is_some()
                        && !has_attr(&b.attrs, "File")
                        && !has_attr(&b.attrs, "Path")
                        && !has_attr(&b.attrs, "NtupleFile")
                        && !has_attr(&b.attrs, "NtupleFiles")
                    {
                        continue;
                    }
                    samples.push(parse_sample_block(&b)?);
                }
                BlockKind::Systematic => {
                    match parse_systematic_block(&b) {
                        Ok(s) => systematics.push(s),
                        Err(Error::NotImplemented(_) | Error::Validation(_)) => {
                            // Best-effort: skip systematics with unsupported types or
                            // missing required fields (e.g. HISTO without HistoNameUp).
                            continue;
                        }
                        Err(e) => return Err(e),
                    }
                }
                BlockKind::NormFactor => {
                    let nf_name = b.name.clone();
                    let samples_val =
                        last_attr_value(&b.attrs, "Samples").unwrap_or_else(|| "all".to_string());
                    let targets = parse_list(&samples_val);
                    norm_factors.push((nf_name, targets));
                }
            }
        }

        if !norm_factors.is_empty() {
            for (nf, targets) in norm_factors {
                for s in &mut samples {
                    if s.kind == SampleKind::Data {
                        continue;
                    }
                    if targets.iter().any(|t| t == &s.name || t.eq_ignore_ascii_case("all")) {
                        s.modifiers.push(NtupleModifier::NormFactor { name: nf.clone() });
                    }
                }
            }
        }

        Ok(Self {
            read_from,
            histo_path: histo_path.map(PathBuf::from),
            combination_xml: combination_xml.map(PathBuf::from),
            tree_name,
            measurement,
            poi,
            regions,
            samples,
            systematics,
        })
    }
}

/// Parse a TRExFitter-style config from text and produce an expression-coverage report.
///
/// This is intended to help parity work against legacy configs: it reports expression-bearing
/// attributes (selection/weight/variable + weight systematics), their compilation status, and
/// required branch names.
pub fn expr_coverage_from_str(text: &str) -> Result<TrexExprCoverageReport> {
    let (globals, blocks) = parse_raw(text)?;
    Ok(trex_expr_coverage_from_raw(&globals, &blocks))
}

fn parse_region_block(b: &RawBlock) -> Result<TrexRegion> {
    let name = b.name.clone();

    fn parse_variable_spec(value: &str) -> Option<(String, Vec<f64>)> {
        // TREx configs often encode variable + binning on a single line:
        //   Variable: "jet_pt",6,200,800
        //   Variable: "x",0,1,2,3  (edges)
        let toks = parse_list(value);
        if toks.len() < 2 {
            return None;
        }
        let expr = toks[0].trim().to_string();
        let nums: Vec<f64> = toks[1..].iter().filter_map(|s| s.trim().parse::<f64>().ok()).collect();
        if nums.len() != toks.len() - 1 {
            return None;
        }

        // Heuristic:
        // - 3 numeric fields: treat as (nbins, low, high) if nbins is an integer.
        // - >=2 numeric fields: treat as explicit edges.
        if nums.len() == 3 {
            let nbins_f = nums[0];
            let nbins = nbins_f.round();
            if (nbins - nbins_f).abs() < 1e-12 && nbins >= 1.0 {
                let n = nbins as usize;
                let lo = nums[1];
                let hi = nums[2];
                if hi > lo {
                    let step = (hi - lo) / (n as f64);
                    let mut edges = Vec::with_capacity(n + 1);
                    for i in 0..=n {
                        edges.push(lo + step * (i as f64));
                    }
                    return Some((expr, edges));
                }
            }
        }
        if nums.len() >= 2 {
            return Some((expr, nums));
        }
        None
    }

    let variable_raw = last_attr_value(&b.attrs, "Variable")
        .or_else(|| last_attr_value(&b.attrs, "Var"))
        .ok_or_else(|| {
            Error::Validation(format!(
                "Region '{name}' missing Variable (line {})",
                b.attrs.first().map(|a| a.line).unwrap_or(0)
            ))
        })?;

    let (variable, binning) = if let Some(binning_str) = last_attr_value(&b.attrs, "Binning")
        .or_else(|| last_attr_value(&b.attrs, "BinEdges"))
    {
        (variable_raw, parse_binning(&binning_str)?)
    } else if let Some((expr, edges)) = parse_variable_spec(&variable_raw) {
        (expr, edges)
    } else {
        return Err(Error::Validation(format!(
            "Region '{name}' missing Binning; expected either `Binning:`/`BinEdges:` or a TREx-style `Variable: expr,nbins,lo,hi`"
        )));
    };

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

fn trex_coverage_from_raw(globals: &[Attr], blocks: &[RawBlock]) -> TrexCoverageReport {
    fn known_global(key: &str) -> bool {
        [
            "ReadFrom",
            "HistoPath",
            "HistPath",
            "ExportDir",
            "CombinationXml",
            "CombinationXML",
            "HistFactoryXml",
            "TreeName",
            "Tree",
            "Measurement",
            "POI",
            "Poi",
        ]
        .iter()
        .any(|k| key_eq(key, k))
    }

    fn known_in_block(kind: BlockKind, key: &str) -> bool {
        match kind {
            BlockKind::Job => known_global(key),
            BlockKind::Region => [
                "Variable",
                "Var",
                "Binning",
                "BinEdges",
                "Selection",
                "Cut",
                "Weight",
                "DataFile",
                "DataTreeName",
                "DataTree",
            ]
            .iter()
            .any(|k| key_eq(key, k)),
            BlockKind::Sample => [
                "Type",
                "File",
                "Path",
                "NtupleFile",
                "NtupleFiles",
                "NtupleName",
                "TreeName",
                "Tree",
                "Weight",
                "MCweight",
                "Regions",
                "Selection",
                "Cut",
                "Variable",
                "Var",
                "NormFactor",
                "NormSys",
                "StatError",
            ]
            .iter()
            .any(|k| key_eq(key, k)),
            BlockKind::Systematic => [
                "Type",
                "Samples",
                "Regions",
                "Lo",
                "Down",
                "Hi",
                "Up",
                "OverallUp",
                "OverallDown",
                "WeightUp",
                "WeightDown",
                "WeightSufUp",
                "WeightSufDown",
                "WeightBase",
                "WeightUpSuffix",
                "WeightDownSuffix",
                "UpSuffix",
                "DownSuffix",
                "SuffixUp",
                "SuffixDown",
                "FileUp",
                "UpFile",
                "FileDown",
                "DownFile",
                "TreeName",
                "Tree",
                // HISTO/SHAPE systematics.
                "HistoNameUp",
                "HistoNameDown",
                "HistoUp",
                "HistoDown",
                "NameUp",
                "NameDown",
                "HistoFileUp",
                "HistoFileDown",
                "HistoPathUp",
                "HistoPathDown",
                // Optional TREx knobs.
                "Smoothing",
                "DropNorm",
                "Pruning",
            ]
            .iter()
            .any(|k| key_eq(key, k)),
            BlockKind::NormFactor => [
                "Title",
                "Nominal",
                "Min",
                "Max",
                "Samples",
                "Constant",
            ]
            .iter()
            .any(|k| key_eq(key, k)),
        }
    }

    let mut unknown: Vec<TrexUnknownAttr> = Vec::new();
    for a in globals {
        if !known_global(&a.key) {
            unknown.push(TrexUnknownAttr {
                scope: "global".to_string(),
                block_kind: None,
                block_name: None,
                line: a.line,
                key: a.key.clone(),
                value: a.value.clone(),
            });
        }
    }

    let mut n_block_attrs = 0usize;
    for b in blocks {
        for a in &b.attrs {
            n_block_attrs += 1;
            if !known_in_block(b.kind, &a.key) {
                unknown.push(TrexUnknownAttr {
                    scope: "block".to_string(),
                    block_kind: Some(format!("{:?}", b.kind)),
                    block_name: Some(b.name.clone()),
                    line: a.line,
                    key: a.key.clone(),
                    value: a.value.clone(),
                });
            }
        }
    }

    TrexCoverageReport {
        schema_version: "trex_config_coverage_v0".to_string(),
        n_globals: globals.len(),
        n_blocks: blocks.len(),
        n_block_attrs,
        unknown,
    }
}

fn last_attr<'a>(attrs: &'a [Attr], key: &str) -> Option<&'a Attr> {
    attrs.iter().rev().find(|a| key_eq(&a.key, key))
}

fn trex_expr_coverage_from_raw(globals: &[Attr], blocks: &[RawBlock]) -> TrexExprCoverageReport {
    fn compile(expr: &str) -> (bool, Option<String>, Vec<String>) {
        match CompiledExpr::compile(expr) {
            Ok(c) => (true, None, c.required_branches),
            Err(e) => (false, Some(e.to_string()), Vec::new()),
        }
    }

    fn normalize_variable_expr(raw: &str) -> String {
        // TREx configs often encode variable + binning on a single line:
        //   Variable: "jet_pt",6,200,800
        // Keep only the first field (variable expression).
        let s = raw.trim();
        let first = s.split(',').next().unwrap_or(s).trim();
        if (first.starts_with('"') && first.ends_with('"')) || (first.starts_with('\'') && first.ends_with('\'')) {
            unquote(first)
        } else {
            first.to_string()
        }
    }

    fn push_item(
        out: &mut Vec<TrexExprCoverageItem>,
        scope: &str,
        block_kind: Option<String>,
        block_name: Option<String>,
        ctx_region: Option<String>,
        ctx_sample: Option<String>,
        line: usize,
        key: &str,
        role: &str,
        expr: String,
        derived: bool,
    ) {
        let expr_norm = if role == "variable" { normalize_variable_expr(&expr) } else { expr };
        let expr_trim = expr_norm.trim();
        if expr_trim.is_empty() {
            return;
        }
        let (ok, error, required_branches) = compile(expr_trim);
        out.push(TrexExprCoverageItem {
            scope: scope.to_string(),
            block_kind,
            block_name,
            ctx_region,
            ctx_sample,
            line,
            key: key.to_string(),
            role: role.to_string(),
            expr: expr_trim.to_string(),
            derived,
            ok,
            error,
            required_branches,
        });
    }

    fn role_for_global_key(key: &str) -> Option<&'static str> {
        if key_eq(key, "Selection") || key_eq(key, "Cut") {
            Some("selection")
        } else if key_eq(key, "Weight") || key_eq(key, "MCweight") {
            Some("weight")
        } else if key_eq(key, "Variable") || key_eq(key, "Var") {
            Some("variable")
        } else {
            None
        }
    }

    fn role_for_block_key(
        kind: BlockKind,
        syst_kind: Option<SystKind>,
        key: &str,
    ) -> Option<&'static str> {
        match kind {
            BlockKind::Region | BlockKind::Sample => {
                if key_eq(key, "Selection") || key_eq(key, "Cut") {
                    Some("selection")
                } else if key_eq(key, "Weight") || key_eq(key, "MCweight") {
                    Some("weight")
                } else if key_eq(key, "Variable") || key_eq(key, "Var") {
                    Some("variable")
                } else {
                    None
                }
            }
            BlockKind::Systematic => {
                if syst_kind != Some(SystKind::Weight) {
                    return None;
                }
                if key_eq(key, "WeightUp") || key_eq(key, "Up") {
                    Some("weight_up")
                } else if key_eq(key, "WeightDown") || key_eq(key, "Down") {
                    Some("weight_down")
                } else if key_eq(key, "WeightSufUp") {
                    Some("weight_suf_up")
                } else if key_eq(key, "WeightSufDown") {
                    Some("weight_suf_down")
                } else if key_eq(key, "WeightBase") || key_eq(key, "Weight") {
                    Some("weight_base")
                } else {
                    None
                }
            }
            BlockKind::Job => None,
            BlockKind::NormFactor => None,
        }
    }

    let mut items: Vec<TrexExprCoverageItem> = Vec::new();

    for a in globals {
        if let Some(role) = role_for_global_key(&a.key) {
            push_item(
                &mut items,
                "global",
                None,
                None,
                None,
                None,
                a.line,
                &a.key,
                role,
                a.value.clone(),
                false,
            );
        }
    }

    for b in blocks {
        let syst_kind = if b.kind == BlockKind::Systematic {
            last_attr_value(&b.attrs, "Type").and_then(|t| parse_syst_kind(&t))
        } else {
            None
        };

        for a in &b.attrs {
            if let Some(role) = role_for_block_key(b.kind, syst_kind, &a.key) {
                push_item(
                    &mut items,
                    "block",
                    Some(format!("{:?}", b.kind)),
                    Some(b.name.clone()),
                    b.ctx_region.clone(),
                    b.ctx_sample.clone(),
                    a.line,
                    &a.key,
                    role,
                    a.value.clone(),
                    false,
                );
            }
        }

        // For weight systematics, also validate derived WeightUp/WeightDown from suffix expansion.
        if b.kind == BlockKind::Systematic && syst_kind == Some(SystKind::Weight) {
            let has_explicit_up =
                last_attr(&b.attrs, "WeightUp").is_some() || last_attr(&b.attrs, "Up").is_some();
            let has_explicit_down = last_attr(&b.attrs, "WeightDown").is_some()
                || last_attr(&b.attrs, "Down").is_some();
            if !(has_explicit_up || has_explicit_down) {
                let base = last_attr(&b.attrs, "WeightBase").or_else(|| last_attr(&b.attrs, "Weight"));
                let up_suf = last_attr(&b.attrs, "WeightUpSuffix")
                    .or_else(|| last_attr(&b.attrs, "UpSuffix"))
                    .or_else(|| last_attr(&b.attrs, "SuffixUp"));
                let down_suf = last_attr(&b.attrs, "WeightDownSuffix")
                    .or_else(|| last_attr(&b.attrs, "DownSuffix"))
                    .or_else(|| last_attr(&b.attrs, "SuffixDown"));
                if let (Some(base), Some(up_suf), Some(down_suf)) = (base, up_suf, down_suf) {
                    let line = base.line;
                    let base_trim = base.value.trim();
                    let up_trim = up_suf.value.trim();
                    let down_trim = down_suf.value.trim();
                    if !(base_trim.is_empty() || up_trim.is_empty() || down_trim.is_empty()) {
                        push_item(
                            &mut items,
                            "block",
                            Some(format!("{:?}", b.kind)),
                            Some(b.name.clone()),
                            b.ctx_region.clone(),
                            b.ctx_sample.clone(),
                            line,
                            "WeightUp",
                            "weight_up",
                            format!("{base_trim}{up_trim}"),
                            true,
                        );
                        push_item(
                            &mut items,
                            "block",
                            Some(format!("{:?}", b.kind)),
                            Some(b.name.clone()),
                            b.ctx_region.clone(),
                            b.ctx_sample.clone(),
                            line,
                            "WeightDown",
                            "weight_down",
                            format!("{base_trim}{down_trim}"),
                            true,
                        );
                    }
                }
            }
        }
    }

    let n_exprs = items.len();
    let n_ok = items.iter().filter(|x| x.ok).count();
    let n_err = n_exprs - n_ok;
    TrexExprCoverageReport {
        schema_version: "trex_expr_coverage_v0".to_string(),
        n_exprs,
        n_ok,
        n_err,
        items,
    }
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
        .or_else(|| last_attr_value(&b.attrs, "NtupleFile"))
        .or_else(|| {
            last_attr_value(&b.attrs, "NtupleFiles").and_then(|v| parse_list(&v).into_iter().next())
        })
        .ok_or_else(|| {
            Error::Validation(format!(
                "Sample '{name}' missing File (expected File/Path or NtupleFile/NtupleFiles)"
            ))
        })?;
    let file = PathBuf::from(file);

    let tree_name = last_attr_value(&b.attrs, "TreeName")
        .or_else(|| last_attr_value(&b.attrs, "Tree"))
        .or_else(|| last_attr_value(&b.attrs, "NtupleName"));
    let weight = last_attr_value(&b.attrs, "Weight").or_else(|| last_attr_value(&b.attrs, "MCweight"));

    let mut regions = last_attr_value(&b.attrs, "Regions")
        .map(|v| parse_list(&v))
        .map(|xs| xs.into_iter().collect::<Vec<_>>())
        .filter(|xs| !xs.is_empty());
    // Legacy nested configs often scope samples inside a Region block instead of using `Regions:`.
    if regions.is_none()
        && let Some(ref r) = b.ctx_region
    {
        regions = Some(vec![r.clone()]);
    }

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
        "norm" | "normsys" | "overall" | "overallsys" => Some(SystKind::Norm),
        "weight" | "weightsys" => Some(SystKind::Weight),
        "tree" | "treesys" => Some(SystKind::Tree),
        "histo" | "histosys" => Some(SystKind::Histo),
        "shape" | "shapesys" => Some(SystKind::Shape),
        _ => None,
        }
    }

fn parse_systematic_block(b: &RawBlock) -> Result<TrexSystematic> {
    let name = b.name.clone();

    let kind = if let Some(type_str) = last_attr_value(&b.attrs, "Type") {
        parse_syst_kind(&type_str).ok_or_else(|| {
            Error::NotImplemented(format!("Systematic '{name}' unsupported Type: {type_str:?}"))
        })?
    } else {
        // Best-effort inference for legacy configs that omit `Type:`.
        let has_weight = [
            "WeightUp",
            "WeightDown",
            "WeightBase",
            "WeightSufUp",
            "WeightSufDown",
            "WeightUpSuffix",
            "WeightDownSuffix",
            "UpSuffix",
            "DownSuffix",
            "SuffixUp",
            "SuffixDown",
        ]
        .iter()
        .any(|k| has_attr(&b.attrs, k));
        let has_norm = ["OverallUp", "OverallDown", "Lo", "Hi", "Up", "Down"]
            .iter()
            .any(|k| has_attr(&b.attrs, k));
        let has_tree = ["FileUp", "UpFile", "FileDown", "DownFile"]
            .iter()
            .any(|k| has_attr(&b.attrs, k));
        let has_histo = ["HistoNameUp", "HistoNameDown", "HistoUp", "HistoDown"]
            .iter()
            .any(|k| has_attr(&b.attrs, k));

        if has_weight {
            SystKind::Weight
        } else if has_norm {
            SystKind::Norm
        } else if has_histo {
            SystKind::Histo
        } else if has_tree {
            SystKind::Tree
        } else {
            return Err(Error::NotImplemented(format!(
                "Systematic '{name}' missing Type and cannot be inferred"
            )));
        }
    };

    let samples = if let Some(samples_val) = last_attr_value(&b.attrs, "Samples") {
        parse_list(&samples_val)
    } else {
        // Nested systematic under a Sample block: infer the sample scope.
        b.ctx_sample.clone().map(|s| vec![s]).unwrap_or_default()
    };
    if samples.is_empty() {
        return Err(Error::Validation(format!("Systematic '{name}' Samples list is empty")));
    }

    let mut regions =
        last_attr_value(&b.attrs, "Regions").map(|v| parse_list(&v)).filter(|xs| !xs.is_empty());
    if regions.is_none()
        && let Some(ref r) = b.ctx_region
    {
        regions = Some(vec![r.clone()]);
    }

    let mut out = TrexSystematic {
        name,
        kind,
        samples,
        regions,
        lo: None,
        hi: None,
        weight_base: None,
        weight_up_suffix: None,
        weight_down_suffix: None,
        weight_suf_up: None,
        weight_suf_down: None,
        weight_up: None,
        weight_down: None,
        file_up: None,
        file_down: None,
        tree_name: None,
        histo_name_up: None,
        histo_name_down: None,
        histo_file_up: None,
        histo_file_down: None,
        smoothing: last_attr_value(&b.attrs, "Smoothing"),
        drop_norm: last_attr_value(&b.attrs, "DropNorm"),
        pruning: last_attr_value(&b.attrs, "Pruning"),
    };

    match kind {
        SystKind::Norm => {
            let lo = last_attr_value(&b.attrs, "Lo").or_else(|| last_attr_value(&b.attrs, "Down"));
            let hi = last_attr_value(&b.attrs, "Hi").or_else(|| last_attr_value(&b.attrs, "Up"));
            if let (Some(lo), Some(hi)) = (lo, hi) {
                out.lo = Some(parse_f64(&lo)?);
                out.hi = Some(parse_f64(&hi)?);
            } else {
                // TREx "OVERALL" systematics use OverallUp/OverallDown.
                let up = last_attr_value(&b.attrs, "OverallUp");
                let down = last_attr_value(&b.attrs, "OverallDown");
                let (Some(up), Some(down)) = (up, down) else {
                    return Err(Error::Validation(format!(
                        "Systematic '{}' (norm/overall) requires Lo/Hi (or OverallUp/OverallDown)",
                        out.name
                    )));
                };
                let up_v = parse_f64(&up)?;
                let down_v = parse_f64(&down)?;

                // Heuristic: treat small values as fractional shifts (±0.02), else as absolute factors (1.02).
                let frac = up_v.abs() < 0.5 && down_v.abs() < 0.5;
                let (hi_v, lo_v) = if frac { (1.0 + up_v, 1.0 + down_v) } else { (up_v, down_v) };
                out.lo = Some(lo_v);
                out.hi = Some(hi_v);
            }
        }
        SystKind::Weight => {
            let up =
                last_attr_value(&b.attrs, "WeightUp").or_else(|| last_attr_value(&b.attrs, "Up"));
            let down = last_attr_value(&b.attrs, "WeightDown")
                .or_else(|| last_attr_value(&b.attrs, "Down"));

            let suf_up = last_attr_value(&b.attrs, "WeightSufUp");
            let suf_down = last_attr_value(&b.attrs, "WeightSufDown");

            // Weight-suffix expansion (TREx-like convenience):
            // allow specifying a base + up/down suffixes instead of full expressions.
            // Example:
            //   WeightBase: weight_jes
            //   WeightUpSuffix: _up
            //   WeightDownSuffix: _down
            let base = last_attr_value(&b.attrs, "WeightBase")
                .or_else(|| last_attr_value(&b.attrs, "Weight"));
            let up_suf = last_attr_value(&b.attrs, "WeightUpSuffix")
                .or_else(|| last_attr_value(&b.attrs, "UpSuffix"))
                .or_else(|| last_attr_value(&b.attrs, "SuffixUp"));
            let down_suf = last_attr_value(&b.attrs, "WeightDownSuffix")
                .or_else(|| last_attr_value(&b.attrs, "DownSuffix"))
                .or_else(|| last_attr_value(&b.attrs, "SuffixDown"));

            if up.is_some() || down.is_some() {
                let (Some(up), Some(down)) = (up, down) else {
                    return Err(Error::Validation(format!(
                        "Systematic '{}' (weight) requires both WeightUp and WeightDown",
                        out.name
                    )));
                };
                out.weight_up = Some(up);
                out.weight_down = Some(down);
            } else if suf_up.is_some() || suf_down.is_some() {
                let (Some(su), Some(sd)) = (suf_up, suf_down) else {
                    return Err(Error::Validation(format!(
                        "Systematic '{}' (weight) requires both WeightSufUp and WeightSufDown",
                        out.name
                    )));
                };
                out.weight_suf_up = Some(su);
                out.weight_suf_down = Some(sd);
            } else if base.is_some() || up_suf.is_some() || down_suf.is_some() {
                let (Some(base), Some(up_suf), Some(down_suf)) = (base, up_suf, down_suf) else {
                    return Err(Error::Validation(format!(
                        "Systematic '{}' (weight) requires WeightBase/WeightUpSuffix/WeightDownSuffix (or WeightUp/WeightDown)",
                        out.name
                    )));
                };
                out.weight_base = Some(base.clone());
                out.weight_up_suffix = Some(up_suf.clone());
                out.weight_down_suffix = Some(down_suf.clone());
                out.weight_up = Some(format!("{base}{up_suf}"));
                out.weight_down = Some(format!("{base}{down_suf}"));
            } else {
                return Err(Error::Validation(format!(
                    "Systematic '{}' (weight) requires WeightUp/WeightDown, WeightSufUp/WeightSufDown, or suffix expansion fields",
                    out.name
                )));
            }
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
        SystKind::Histo | SystKind::Shape => {
            // TREx HISTO: read up/down variation histograms (TH1) from ROOT files.
            // Keys: HistoNameUp/HistoNameDown (histogram object names),
            //       HistoFileUp/HistoFileDown or HistoPathUp/HistoPathDown (ROOT files).
            let name_up = last_attr_value(&b.attrs, "HistoNameUp")
                .or_else(|| last_attr_value(&b.attrs, "HistoUp"))
                .or_else(|| last_attr_value(&b.attrs, "NameUp"));
            let name_down = last_attr_value(&b.attrs, "HistoNameDown")
                .or_else(|| last_attr_value(&b.attrs, "HistoDown"))
                .or_else(|| last_attr_value(&b.attrs, "NameDown"));
            let (Some(name_up), Some(name_down)) = (name_up, name_down) else {
                return Err(Error::Validation(format!(
                    "Systematic '{}' (histo) requires HistoNameUp/HistoNameDown",
                    out.name
                )));
            };
            out.histo_name_up = Some(name_up);
            out.histo_name_down = Some(name_down);
            // Optional: separate files for up/down (defaults to sample's nominal file).
            out.histo_file_up = last_attr_value(&b.attrs, "HistoFileUp")
                .or_else(|| last_attr_value(&b.attrs, "HistoPathUp"))
                .or_else(|| last_attr_value(&b.attrs, "FileUp"))
                .map(PathBuf::from);
            out.histo_file_down = last_attr_value(&b.attrs, "HistoFileDown")
                .or_else(|| last_attr_value(&b.attrs, "HistoPathDown"))
                .or_else(|| last_attr_value(&b.attrs, "FileDown"))
                .map(PathBuf::from);
        }
    }

    Ok(out)
}

fn sys_applies(sys: &TrexSystematic, region_name: &str, sample_name: &str) -> bool {
    if !sys.samples.iter().any(|s| s == sample_name || s.eq_ignore_ascii_case("all")) {
        return false;
    }
    if let Some(ref regions) = sys.regions {
        regions.iter().any(|r| r == region_name || r.eq_ignore_ascii_case("all"))
    } else {
        true
    }
}

fn sample_applies(sample: &TrexSample, region_name: &str) -> bool {
    if let Some(ref regions) = sample.regions {
        regions.iter().any(|r| r == region_name || r.eq_ignore_ascii_case("all"))
    } else {
        true
    }
}

#[derive(Debug, Clone, Default)]
struct RegionSampleOverrides {
    selection: Option<String>,
    weight: Option<String>,
    variable: Option<String>,
}

fn expr_mul(parts: impl IntoIterator<Item = String>) -> Option<String> {
    let mut out: Option<String> = None;
    for raw in parts {
        let p = raw.trim();
        if p.is_empty() {
            continue;
        }
        out = Some(match out {
            None => format!("({p})"),
            Some(prev) => format!("{prev} * ({p})"),
        });
    }
    out
}

fn expr_and(parts: impl IntoIterator<Item = String>) -> Option<String> {
    let mut out: Option<String> = None;
    for raw in parts {
        let p = raw.trim();
        if p.is_empty() {
            continue;
        }
        out = Some(match out {
            None => format!("({p})"),
            Some(prev) => format!("{prev} && ({p})"),
        });
    }
    out
}

fn collect_region_weights_and_sample_overrides(
    blocks: &[RawBlock],
) -> (
    HashMap<String, String>,
    HashMap<(String, String), RegionSampleOverrides>,
    HashMap<String, String>,
    HashMap<String, String>,
) {
    let mut region_weight: HashMap<String, String> = HashMap::new();
    let mut overrides: HashMap<(String, String), RegionSampleOverrides> = HashMap::new();
    let mut sample_selection: HashMap<String, String> = HashMap::new();
    let mut sample_variable: HashMap<String, String> = HashMap::new();

    for b in blocks {
        match b.kind {
            BlockKind::Region => {
                // Top-level region: capture region-wide Weight (applies to all samples in the region).
                if b.ctx_region.is_none()
                    && b.ctx_sample.is_none()
                    && let Some(w) = last_attr_value(&b.attrs, "Weight")
                    && !w.trim().is_empty()
                {
                    region_weight.insert(b.name.clone(), w);
                }
            }
            BlockKind::Sample => {
                // Override-only sample nested under a Region (no File/Path): per-(region,sample) overrides.
                if b.ctx_region.is_some()
                    && !has_attr(&b.attrs, "File")
                    && !has_attr(&b.attrs, "Path")
                {
                    let region_name = b.ctx_region.clone().unwrap();
                    let key = (region_name, b.name.clone());
                    let entry = overrides.entry(key).or_default();

                    if let Some(sel) = last_attr_value(&b.attrs, "Selection")
                        .or_else(|| last_attr_value(&b.attrs, "Cut"))
                        && !sel.trim().is_empty()
                    {
                        entry.selection = Some(sel);
                    }
                    if let Some(w) = last_attr_value(&b.attrs, "Weight")
                        && !w.trim().is_empty()
                    {
                        entry.weight = Some(w);
                    }
                    if let Some(v) = last_attr_value(&b.attrs, "Variable")
                        .or_else(|| last_attr_value(&b.attrs, "Var"))
                        && !v.trim().is_empty()
                    {
                        entry.variable = Some(v);
                    }
                    continue;
                }

                // Full sample blocks (with File/Path): capture optional global Selection/Cut and Variable/Var.
                if has_attr(&b.attrs, "File") || has_attr(&b.attrs, "Path") {
                    if let Some(sel) = last_attr_value(&b.attrs, "Selection")
                        .or_else(|| last_attr_value(&b.attrs, "Cut"))
                        && !sel.trim().is_empty()
                    {
                        sample_selection.insert(b.name.clone(), sel);
                    }
                    if let Some(v) = last_attr_value(&b.attrs, "Variable")
                        .or_else(|| last_attr_value(&b.attrs, "Var"))
                        && !v.trim().is_empty()
                    {
                        sample_variable.insert(b.name.clone(), v);
                    }
                }
            }
            _ => {}
        }
    }

    (region_weight, overrides, sample_selection, sample_variable)
}

fn enforce_variable_rules(
    region_name: &str,
    region_variable: &str,
    sample_name: &str,
    sample_var: Option<&String>,
    override_var: Option<&String>,
) -> Result<()> {
    if let Some(v) = sample_var
        && !v.trim().is_empty()
        && v != region_variable
    {
        return Err(Error::Validation(format!(
            "per-sample variable override is not supported (variable is channel-scoped): region='{region_name}' sample='{sample_name}' Sample.Variable='{v}' != Region.Variable='{region_variable}'. Split into separate Regions if you need different variables."
        )));
    }
    if let Some(v) = override_var
        && !v.trim().is_empty()
        && v != region_variable
    {
        return Err(Error::Validation(format!(
            "per-sample variable override is not supported (variable is channel-scoped): region='{region_name}' sample='{sample_name}' Override.Variable='{v}' != Region.Variable='{region_variable}'. Split into separate Regions if you need different variables."
        )));
    }
    Ok(())
}

fn sys_to_modifier(sys: &TrexSystematic, base_weight: Option<&str>) -> Result<NtupleModifier> {
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
        SystKind::Weight => {
            if let (Some(su), Some(sd)) =
                (sys.weight_suf_up.as_deref(), sys.weight_suf_down.as_deref())
            {
                let up = expr_mul(
                    [base_weight.map(|s| s.to_string()), Some(su.to_string())]
                        .into_iter()
                        .flatten(),
                )
                .ok_or_else(|| {
                    Error::Validation(format!("Systematic '{}' empty WeightSufUp", sys.name))
                })?;
                let down = expr_mul(
                    [base_weight.map(|s| s.to_string()), Some(sd.to_string())]
                        .into_iter()
                        .flatten(),
                )
                .ok_or_else(|| {
                    Error::Validation(format!("Systematic '{}' empty WeightSufDown", sys.name))
                })?;
                Ok(NtupleModifier::WeightSys {
                    name: sys.name.clone(),
                    weight_up: up,
                    weight_down: down,
                })
            } else {
                Ok(NtupleModifier::WeightSys {
                    name: sys.name.clone(),
                    weight_up: sys.weight_up.clone().ok_or_else(|| {
                        Error::Validation(format!("Systematic '{}' missing weight_up", sys.name))
                    })?,
                    weight_down: sys.weight_down.clone().ok_or_else(|| {
                        Error::Validation(format!("Systematic '{}' missing weight_down", sys.name))
                    })?,
                })
            }
        }
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
        SystKind::Histo | SystKind::Shape => Ok(NtupleModifier::HistoSys {
            name: sys.name.clone(),
            histo_name_up: sys.histo_name_up.clone().ok_or_else(|| {
                Error::Validation(format!("Systematic '{}' missing histo_name_up", sys.name))
            })?,
            histo_name_down: sys.histo_name_down.clone().ok_or_else(|| {
                Error::Validation(format!("Systematic '{}' missing histo_name_down", sys.name))
            })?,
            file_up: sys.histo_file_up.clone(),
            file_down: sys.histo_file_down.clone(),
        }),
    }
}

/// Build a pyhf JSON `Workspace` from a parsed config (ntuple mode).
///
/// `base_dir` is used to resolve relative ROOT file paths (as in
/// `NtupleWorkspaceBuilder::ntuple_path`).
pub fn workspace_from_config(cfg: &TrexConfig, base_dir: &Path) -> Result<Workspace> {
    let read_from = cfg.read_from.as_deref().unwrap_or("NTUP").trim().to_ascii_uppercase();
    if read_from == "HIST" {
        return workspace_from_hist_mode(cfg, base_dir);
    }
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
                    sc.modifiers.push(sys_to_modifier(sys, sc.weight.as_deref())?);
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
    let (globals, blocks) = parse_raw(text)?;
    let cfg = TrexConfig::parse_from_raw(globals, blocks.clone())?;

    let (region_weight, overrides, sample_selection, sample_variable) =
        collect_region_weights_and_sample_overrides(&blocks);

    // Re-implement `workspace_from_config` with override-aware composition rules.
    let read_from = cfg.read_from.as_deref().unwrap_or("NTUP").trim().to_ascii_uppercase();
    if read_from == "HIST" {
        return workspace_from_hist_mode(&cfg, base_dir);
    }
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

            let ov = overrides.get(&(region.name.clone(), s.name.clone()));
            enforce_variable_rules(
                &region.name,
                &region.variable,
                &s.name,
                sample_variable.get(&s.name),
                ov.and_then(|x| x.variable.as_ref()),
            )?;

            // Composition rules:
            // - channel selection = region.Selection (shared)
            // - per-sample Selection/Cut is applied multiplicatively via the sample weight (selection evaluates to 0/1)
            // - Weight = region.Weight * sample.Weight * override.Weight
            let sel = expr_and(
                [sample_selection.get(&s.name).cloned(), ov.and_then(|x| x.selection.clone())]
                    .into_iter()
                    .flatten(),
            );

            let w = expr_mul(
                [
                    region_weight.get(&region.name).cloned(),
                    s.weight.clone(),
                    ov.and_then(|x| x.weight.clone()),
                    sel,
                ]
                .into_iter()
                .flatten(),
            );

            let mut sc = SampleConfig::new(&s.name, s.file.clone());
            sc.tree_name = s.tree_name.clone();
            sc.weight = w;
            sc.modifiers = s.modifiers.clone();

            for sys in &cfg.systematics {
                if sys_applies(sys, &region.name, &s.name) {
                    sc.modifiers.push(sys_to_modifier(sys, sc.weight.as_deref())?);
                }
            }

            ch.samples.push(sc);
        }

        builder = builder.add_channel(ch);
    }

    builder.build()
}

fn resolve_rel(base_dir: &Path, p: &Path) -> PathBuf {
    if p.is_absolute() { p.to_path_buf() } else { base_dir.join(p) }
}

fn discover_single_combination_xml(dir: &Path) -> Result<PathBuf> {
    let direct = dir.join("combination.xml");
    if direct.is_file() {
        return Ok(direct);
    }

    let mut hits: Vec<PathBuf> = Vec::new();
    let mut stack: Vec<PathBuf> = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for ent in std::fs::read_dir(&d)? {
            let ent = ent?;
            let p = ent.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if p.file_name().and_then(|s| s.to_str()) == Some("combination.xml") {
                hits.push(p);
                if hits.len() > 1 {
                    break;
                }
            }
        }
        if hits.len() > 1 {
            break;
        }
    }

    if hits.is_empty() {
        return Err(Error::Validation(format!(
            "HIST mode requires a HistFactory export directory containing combination.xml (none found under {})",
            dir.display()
        )));
    }
    if hits.len() > 1 {
        return Err(Error::Validation(format!(
            "multiple combination.xml found under {} (expected exactly 1)",
            dir.display()
        )));
    }
    Ok(hits.remove(0))
}

fn workspace_from_hist_mode(cfg: &TrexConfig, base_dir: &Path) -> Result<Workspace> {
    // Resolve `combination.xml` path.
    //
    // Important: TREx-style HIST workflows typically treat `HistoPath` as the export root,
    // even when `combination.xml` lives under `config/`. HistFactory XML then references
    // ROOT files and channel XMLs via paths relative to that export root.
    //
    // Therefore, when `HistoPath` is provided, we use it as the base directory for
    // resolving relative paths during HistFactory import.
    let mut basedir: Option<PathBuf> = None;

    let combo = if let Some(ref p) = cfg.combination_xml {
        let combo = resolve_rel(base_dir, p);
        // If the caller also provided `HistoPath`, prefer it as the base directory (TREx semantics).
        if basedir.is_none() {
            basedir = cfg.histo_path.as_ref().map(|hp| resolve_rel(base_dir, hp));
        }
        combo
    } else if let Some(ref p) = cfg.histo_path {
        let resolved = resolve_rel(base_dir, p);
        basedir = Some(resolved.clone());
        if resolved.is_file() { resolved } else { discover_single_combination_xml(&resolved)? }
    } else {
        return Err(Error::Validation(
            "ReadFrom=HIST requires HistoPath (export dir) or CombinationXml (path to combination.xml)".to_string(),
        ));
    };

    let basedir = basedir.as_ref().map(|p| {
        if p.is_file() {
            p.parent().unwrap_or_else(|| Path::new(".")).to_path_buf()
        } else {
            p.clone()
        }
    });

    let mut ws = crate::histfactory::from_xml_with_basedir(&combo, basedir.as_deref())?;

    // Partial TREx semantics: region/sample filtering in HIST mode.
    //
    // In full TRExFitter, the `.config` can be used to select a subset of regions (channels)
    // and to mask samples per-region via `Sample: ... Regions: ...`. When importing from an
    // existing HistFactory export, we treat these as *filters* over the imported workspace.
    ws = apply_hist_mode_filters(ws, cfg)?;

    Ok(ws)
}

fn apply_hist_mode_filters(mut ws: Workspace, cfg: &TrexConfig) -> Result<Workspace> {
    let explicit_channel_selection = !cfg.regions.is_empty();

    // Region/channel selection: if the config defines Region blocks, we treat them as the
    // include-list for channels, in the config order.
    if explicit_channel_selection {
        let mut want: Vec<String> = Vec::new();
        let mut seen: HashSet<&str> = HashSet::new();
        for r in &cfg.regions {
            if !seen.insert(r.name.as_str()) {
                return Err(Error::Validation(format!("duplicate Region name: {}", r.name)));
            }
            want.push(r.name.clone());
        }

        let mut by_name: HashMap<String, Channel> =
            ws.channels.drain(..).map(|c| (c.name.clone(), c)).collect();
        let mut obs_by_name: HashMap<String, Observation> =
            ws.observations.drain(..).map(|o| (o.name.clone(), o)).collect();

        let mut channels: Vec<Channel> = Vec::with_capacity(want.len());
        let mut observations: Vec<Observation> = Vec::with_capacity(want.len());
        let mut missing: Vec<String> = Vec::new();

        for name in want {
            let Some(ch) = by_name.remove(&name) else {
                missing.push(name);
                continue;
            };
            let Some(obs) = obs_by_name.remove(&ch.name) else {
                return Err(Error::Validation(format!(
                    "missing Observation for channel '{}' in imported workspace",
                    ch.name
                )));
            };
            channels.push(ch);
            observations.push(obs);
        }

        if !missing.is_empty() {
            missing.sort();
            return Err(Error::Validation(format!(
                "HIST mode region selection requested missing channel(s): {}",
                missing.join(", ")
            )));
        }

        ws.channels = channels;
        ws.observations = observations;
    }

    // Sample selection: if the config defines Sample blocks, treat them as an include-list
    // per channel, respecting per-sample `Regions:` filters.
    if !cfg.samples.is_empty() {
        // Preserve config sample order.
        let mut want: Vec<&TrexSample> = Vec::new();
        let mut seen: HashSet<&str> = HashSet::new();
        for s in &cfg.samples {
            if s.kind == SampleKind::Data {
                continue;
            }
            if !seen.insert(s.name.as_str()) {
                return Err(Error::Validation(format!("duplicate Sample name: {}", s.name)));
            }
            want.push(s);
        }

        let mut kept_channels: Vec<Channel> = Vec::with_capacity(ws.channels.len());
        let mut kept_observations: Vec<Observation> = Vec::with_capacity(ws.observations.len());
        let mut obs_by_name: HashMap<String, Observation> =
            ws.observations.drain(..).map(|o| (o.name.clone(), o)).collect();

        for mut ch in ws.channels.drain(..) {
            let mut by_name: HashMap<String, Sample> =
                ch.samples.drain(..).map(|s| (s.name.clone(), s)).collect();

            let mut new_samples: Vec<Sample> = Vec::new();
            for s in &want {
                // Respect per-sample region filter: if the sample doesn't apply to this region,
                // skip it even if it's in the global include-list.
                if !sample_applies(s, &ch.name) {
                    continue;
                }

                match by_name.remove(&s.name) {
                    Some(sample) => new_samples.push(sample),
                    None => {
                        return Err(Error::Validation(format!(
                            "HIST mode sample selection requested missing sample '{}' in channel '{}'",
                            s.name, ch.name
                        )));
                    }
                }
            }

            if new_samples.is_empty() {
                if explicit_channel_selection {
                    return Err(Error::Validation(format!(
                        "HIST mode filters removed all samples from explicitly selected channel '{}'",
                        ch.name
                    )));
                }
                // TREx-like behavior: if channels were not explicitly selected, drop empty channels.
                continue;
            }

            ch.samples = new_samples;

            let obs = obs_by_name.remove(&ch.name).ok_or_else(|| {
                Error::Validation(format!(
                    "missing Observation for channel '{}' in imported workspace",
                    ch.name
                ))
            })?;

            kept_channels.push(ch);
            kept_observations.push(obs);
        }

        if kept_channels.is_empty() {
            return Err(Error::Validation(
                "HIST mode sample filters removed all channels (no non-empty channels remain)"
                    .to_string(),
            ));
        }

        ws.channels = kept_channels;
        ws.observations = kept_observations;
    }

    Ok(ws)
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
    fn trex_parse_systematic_type_shape() {
        let cfg = r#"
ReadFrom: NTUP
TreeName: events
Measurement: meas
POI: mu

Region: SR
Variable: mbb
Binning: 0, 50, 100

Sample: signal
Type: SIGNAL
File: tests/fixtures/simple_tree.root
Weight: 1
Regions: SR

Systematic: jes
Type: SHAPE
Samples: signal
Regions: SR
HistoNameUp: jes_up
HistoNameDown: jes_down
Smoothing: 1
DropNorm: false
Pruning: 0.02
"#;

        let cfg = TrexConfig::parse_str(cfg).expect("parse");
        assert_eq!(cfg.systematics.len(), 1);
        let sys = &cfg.systematics[0];
        assert_eq!(sys.kind, SystKind::Shape);
        assert_eq!(sys.histo_name_up.as_deref(), Some("jes_up"));
        assert_eq!(sys.histo_name_down.as_deref(), Some("jes_down"));
        assert_eq!(sys.smoothing.as_deref(), Some("1"));
        assert_eq!(sys.drop_norm.as_deref(), Some("false"));
        assert_eq!(sys.pruning.as_deref(), Some("0.02"));
    }

    #[test]
    fn trex_coverage_recognizes_smoothing_and_pruning_attrs() {
        let cfg = r#"
ReadFrom: NTUP
TreeName: events
Measurement: meas
POI: mu

Region: SR
Variable: mbb
Binning: 0, 50, 100

Sample: signal
Type: SIGNAL
File: tests/fixtures/simple_tree.root
Weight: 1
Regions: SR

Systematic: jes
Type: SHAPE
Samples: signal
Regions: SR
HistoNameUp: jes_up
HistoNameDown: jes_down
Smoothing: 1
DropNorm: false
Pruning: 0.02
"#;

        let (_cfg, report) = TrexConfig::parse_str_with_coverage(cfg).expect("parse");
        assert!(
            report.unknown.is_empty(),
            "unexpected unknown attrs: {:?}",
            report.unknown
        );
    }

    #[test]
    fn trex_import_applies_sample_specific_selection_via_weight() {
        let cfg = r#"
ReadFrom: NTUP
TreeName: events
Measurement: meas
POI: mu

Region: SR
Variable: mbb
Binning: 0, 50, 100, 150, 200, 300

Sample: a
Type: SIGNAL
File: tests/fixtures/simple_tree.root
Weight: 1
Selection: njet >= 4
Regions: SR

Sample: b
Type: BACKGROUND
File: tests/fixtures/simple_tree.root
Weight: 1
Regions: SR
"#;

        let ws = workspace_from_str(cfg, &repo_root()).expect("workspace build");
        let ch = &ws.channels[0];
        assert_eq!(ch.samples.len(), 2);

        let sum_a: f64 = ch.samples.iter().find(|s| s.name == "a").unwrap().data.iter().sum();
        let sum_b: f64 = ch.samples.iter().find(|s| s.name == "b").unwrap().data.iter().sum();
        assert!(sum_a < sum_b, "sample-specific selection should reduce only sample a");
    }

    #[test]
    fn trex_import_applies_region_weight_to_all_samples() {
        let cfg = r#"
ReadFrom: NTUP
TreeName: events
Measurement: meas
POI: mu

Region: SR
Variable: mbb
Binning: 0, 50, 100, 150, 200, 300
Weight: 2

Sample: a
Type: SIGNAL
File: tests/fixtures/simple_tree.root
Weight: 1
Regions: SR

Sample: b
Type: BACKGROUND
File: tests/fixtures/simple_tree.root
Weight: 1
Regions: SR
"#;

        let ws = workspace_from_str(cfg, &repo_root()).expect("workspace build");
        let ch = &ws.channels[0];
        let sum_a: f64 = ch.samples.iter().find(|s| s.name == "a").unwrap().data.iter().sum();
        let sum_b: f64 = ch.samples.iter().find(|s| s.name == "b").unwrap().data.iter().sum();
        assert!((sum_a - sum_b).abs() < 1e-9, "region Weight should scale both samples equally");
    }

    #[test]
    fn trex_import_applies_nested_sample_override_weight() {
        let cfg = r#"
ReadFrom: NTUP
TreeName: events
Measurement: meas
POI: mu

Region: SR
Variable: mbb
Binning: 0, 50, 100, 150, 200, 300

Sample: a
Type: SIGNAL
File: tests/fixtures/simple_tree.root
Weight: 1
EndSample: a

Sample: b
Type: BACKGROUND
File: tests/fixtures/simple_tree.root
Weight: 1
EndSample: b

# Override-only sample scoped to SR (no File): multiplies sample a weights by 3.
Sample: a
Weight: 3
EndSample: a
EndRegion: SR
"#;

        let ws = workspace_from_str(cfg, &repo_root()).expect("workspace build");
        let ch = &ws.channels[0];
        let sum_a: f64 = ch.samples.iter().find(|s| s.name == "a").unwrap().data.iter().sum();
        let sum_b: f64 = ch.samples.iter().find(|s| s.name == "b").unwrap().data.iter().sum();
        assert!(
            (sum_a - 3.0 * sum_b).abs() < 1e-8,
            "nested override Weight should multiply only sample a"
        );
    }

    #[test]
    fn trex_import_rejects_variable_override_mismatch() {
        let cfg = r#"
ReadFrom: NTUP
TreeName: events
Measurement: meas
POI: mu

Region: SR
Variable: mbb
Binning: 0, 50, 100, 150, 200, 300

Sample: a
Type: SIGNAL
File: tests/fixtures/simple_tree.root
Weight: 1
EndSample: a

Sample: a
Variable: other_var
EndSample: a
EndRegion: SR
"#;

        let err = workspace_from_str(cfg, &repo_root()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("per-sample variable override is not supported"), "msg={msg}");
    }

    #[test]
    fn trex_import_weight_suffix_expansion_matches_explicit_weights() {
        let cfg_explicit = r#"
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

        let cfg_suffix = r#"
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
WeightBase: weight_jes
WeightUpSuffix: _up
WeightDownSuffix: _down
"#;

        let ws_a = workspace_from_str(cfg_explicit, &repo_root()).expect("explicit ws build");
        let ws_b = workspace_from_str(cfg_suffix, &repo_root()).expect("suffix ws build");

        let a = &ws_a.channels[0].samples[0];
        let b = &ws_b.channels[0].samples[0];
        assert_eq!(a.data, b.data, "nominal histogram must match");

        fn find_histosys(s: &crate::pyhf::schema::Sample) -> &crate::pyhf::schema::Modifier {
            s.modifiers
                .iter()
                .find(|m| matches!(m, crate::pyhf::schema::Modifier::HistoSys { name, .. } if name == "jes"))
                .expect("missing histosys jes")
        }

        match (find_histosys(a), find_histosys(b)) {
            (
                crate::pyhf::schema::Modifier::HistoSys { data: da, .. },
                crate::pyhf::schema::Modifier::HistoSys { data: db, .. },
            ) => {
                assert_eq!(da.hi_data, db.hi_data);
                assert_eq!(da.lo_data, db.lo_data);
            }
            _ => unreachable!("histosys matcher"),
        }
    }

    #[test]
    fn trex_import_hist_mode_from_histfactory_export_dir() {
        let cfg = r#"
ReadFrom: HIST
HistoPath: tests/fixtures/histfactory
"#;
        let ws = workspace_from_str(cfg, &repo_root()).expect("HIST mode workspace");
        let got: serde_json::Value = serde_json::to_value(&ws).expect("to_value");
        let want: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(repo_root().join("tests/fixtures/histfactory/workspace.json"))
                .unwrap(),
        )
        .expect("fixture JSON");
        assert_eq!(got, want, "workspace mismatch for HIST-mode import");
    }

    #[test]
    fn trex_import_hist_mode_uses_histopath_as_basedir_for_pyhf_fixtures() {
        // In the pyhf validation fixtures, combination.xml lives under `config/` but ROOT inputs
        // are referenced relative to the export root. HIST mode should treat HistoPath as basedir.
        let cfg = r#"
ReadFrom: HIST
HistoPath: tests/fixtures/pyhf_multichannel
CombinationXml: tests/fixtures/pyhf_multichannel/config/example.xml
"#;
        let ws = workspace_from_str(cfg, &repo_root()).expect("HIST mode workspace (pyhf fixture)");
        let names: Vec<&str> = ws.channels.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["channel1", "channel2"]);
    }

    #[test]
    fn trex_import_hist_mode_filters_channels_by_region_blocks() {
        let cfg = r#"
ReadFrom: HIST
HistoPath: tests/fixtures/pyhf_multichannel
CombinationXml: tests/fixtures/pyhf_multichannel/config/example.xml

Region: channel2
Variable: x
Binning: 0, 1
"#;
        let ws = workspace_from_str(cfg, &repo_root()).expect("HIST mode workspace (filtered)");
        assert_eq!(ws.channels.len(), 1);
        assert_eq!(ws.channels[0].name, "channel2");
        assert_eq!(ws.observations.len(), 1);
        assert_eq!(ws.observations[0].name, "channel2");
    }

    #[test]
    fn trex_import_hist_mode_filters_samples_by_sample_blocks_and_regions_list() {
        let cfg = r#"
ReadFrom: HIST
HistoPath: tests/fixtures/pyhf_multichannel
CombinationXml: tests/fixtures/pyhf_multichannel/config/example.xml

Sample: bkg
Type: BACKGROUND
File: ignored.root
Regions: channel1, channel2
"#;
        let ws =
            workspace_from_str(cfg, &repo_root()).expect("HIST mode workspace (sample filtered)");
        assert_eq!(ws.channels.len(), 2);
        assert_eq!(ws.channels[0].name, "channel1");
        assert_eq!(ws.channels[0].samples.len(), 1);
        assert_eq!(ws.channels[0].samples[0].name, "bkg");
        assert_eq!(ws.channels[1].name, "channel2");
        assert_eq!(ws.channels[1].samples.len(), 1);
        assert_eq!(ws.channels[1].samples[0].name, "bkg");
    }

    #[test]
    fn trex_import_hist_mode_errors_when_requested_sample_missing_in_channel() {
        // `signal` does not exist in channel2 in the imported workspace, so if the config
        // requests it in channel2 we should fail loudly.
        let cfg = r#"
ReadFrom: HIST
HistoPath: tests/fixtures/pyhf_multichannel
CombinationXml: tests/fixtures/pyhf_multichannel/config/example.xml

Sample: signal
Type: SIGNAL
File: ignored.root
Regions: channel2
"#;
        let err = workspace_from_str(cfg, &repo_root()).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("missing sample 'signal'") && msg.contains("channel 'channel2'"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn trex_config_parses_quoted_values_and_lists() {
        let cfg = r#"
ReadFrom: NTUP
TreeName: events

Region: SR
Variable: mbb
Binning: [0, 1]

Sample: signal
File: "tests/fixtures/simple_tree.root"
Weight: "w#mc"  # comment after quoted value
Regions: ["SR", "CR 1"]

Systematic: jes
Type: weight
Samples: ["signal"]
Regions: ["SR"]
WeightUp: "w_jes_up"
WeightDown: "w_jes_down"
"#;

        let parsed = TrexConfig::parse_str(cfg).expect("parse_str");
        assert_eq!(parsed.regions.len(), 1);
        assert_eq!(parsed.samples.len(), 1);
        assert_eq!(parsed.systematics.len(), 1);
        assert_eq!(parsed.samples[0].weight.as_deref(), Some("w#mc"));
        assert_eq!(
            parsed.samples[0].regions.as_ref().unwrap(),
            &vec!["SR".to_string(), "CR 1".to_string()]
        );
        assert_eq!(parsed.systematics[0].samples, vec!["signal".to_string()]);
        assert_eq!(parsed.systematics[0].regions.as_ref().unwrap(), &vec!["SR".to_string()]);
    }

    #[test]
    fn trex_config_supports_explicit_end_markers_and_nested_blocks() {
        let cfg = r#"
ReadFrom: NTUP
TreeName: events

Region: SR
Variable: mbb
Binning: 0, 1

Sample: signal
File: tests/fixtures/simple_tree.root

Systematic: jes
Type: weight
WeightUp: w_jes_up
WeightDown: w_jes_down
EndSystematic: jes

EndSample: signal

Selection: njet >= 4
EndRegion: SR
"#;

        let parsed = TrexConfig::parse_str(cfg).expect("parse_str");
        assert_eq!(parsed.regions.len(), 1);
        assert_eq!(parsed.regions[0].selection.as_deref(), Some("njet >= 4"));

        // Sample is nested under Region and has no explicit `Regions:`; we infer it when EndSample is present.
        assert_eq!(parsed.samples.len(), 1);
        assert_eq!(parsed.samples[0].regions.as_ref().unwrap(), &vec!["SR".to_string()]);

        // Systematic is nested under Sample and has no explicit `Samples:`/`Regions:`; infer scopes.
        assert_eq!(parsed.systematics.len(), 1);
        assert_eq!(parsed.systematics[0].samples, vec!["signal".to_string()]);
        assert_eq!(parsed.systematics[0].regions.as_ref().unwrap(), &vec!["SR".to_string()]);
    }

    #[test]
    fn trex_expr_coverage_reports_ok_and_err() {
        let cfg = r#"
ReadFrom: NTUP
TreeName: events

Region: SR
Variable: jet_pt[0]
Selection: njet >= 4 && (pt_lead > 25.0)

Sample: signal
File: tests/fixtures/simple_tree.root
Weight: weight_mc

Systematic: jes
Type: weight
WeightBase: weight_jes
WeightUpSuffix: _up
WeightDownSuffix: _down

Sample: bad
File: tests/fixtures/simple_tree.root
Selection: x +
"#;

        let rep = expr_coverage_from_str(cfg).expect("expr coverage");
        assert_eq!(rep.schema_version, "trex_expr_coverage_v0");
        assert!(rep.n_exprs >= 4, "expected at least 4 expressions, got={}", rep.n_exprs);
        assert!(rep.n_err >= 1, "expected at least one compile error");

        let bad = rep
            .items
            .iter()
            .find(|x| x.block_kind.as_deref() == Some("Sample") && x.block_name.as_deref() == Some("bad"))
            .expect("bad sample item");
        assert!(!bad.ok, "expected bad expression to fail");

        let derived_up = rep.items.iter().find(|x| x.derived && x.key == "WeightUp").expect("derived WeightUp");
        assert!(derived_up.ok, "derived WeightUp should compile, err={:?}", derived_up.error);
        assert!(derived_up.required_branches.iter().any(|b| b == "weight_jes_up"));
    }

    #[test]
    fn trex_parser_strips_percent_comments() {
        let cfg = r#"
Job: foo
  MCweight: w1 * w2 % comment
"#;
        let (_globals, blocks) = parse_raw(cfg).expect("parse_raw");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].kind, BlockKind::Job);
        let mcw = last_attr_value(&blocks[0].attrs, "MCweight").unwrap_or_default();
        assert_eq!(mcw.trim(), "w1 * w2");
    }

    #[test]
    fn trex_expr_coverage_normalizes_variable_specs_with_commas() {
        let cfg = r#"
ReadFrom: NTUP
TreeName: events

Region: SR
Variable: "lep_Pt_0/1e3",100,0,1000 % nbins,low,high
Binning: 0, 1
Selection: lep_Pt_0 > 0
"#;
        let rep = expr_coverage_from_str(cfg).expect("expr coverage");
        let var = rep
            .items
            .iter()
            .find(|x| x.block_kind.as_deref() == Some("Region") && x.role == "variable")
            .expect("variable item");
        assert_eq!(var.expr, "lep_Pt_0/1e3");
        assert!(var.ok, "normalized variable should compile, err={:?}", var.error);
    }

    #[test]
    fn trex_region_parses_uniform_binning_from_variable_spec() {
        let cfg = r#"
ReadFrom: NTUP
TreeName: events

Region: SR
Variable: "x",4,0,2
Selection: x > 0

Sample: s
File: tests/fixtures/simple_tree.root
"#;

        let parsed = TrexConfig::parse_str(cfg).expect("parse_str");
        assert_eq!(parsed.regions.len(), 1);
        assert_eq!(parsed.regions[0].name, "SR");
        assert_eq!(parsed.regions[0].variable, "x");
        assert_eq!(parsed.regions[0].binning.len(), 5);
        assert!((parsed.regions[0].binning[0] - 0.0).abs() < 1e-12);
        assert!((parsed.regions[0].binning[4] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn trex_sample_accepts_ntuplefile_and_mcweight_aliases() {
        let cfg = r#"
ReadFrom: NTUP
NtupleName: reco

Region: SR
Variable: x,2,0,2
Selection: x > 0

Sample: sig
Type: SIGNAL
NtupleFile: prediction.root
MCweight: w_mc
"#;

        let parsed = TrexConfig::parse_str(cfg).expect("parse_str");
        assert_eq!(parsed.tree_name.as_deref(), Some("reco"));
        assert_eq!(parsed.samples.len(), 1);
        assert_eq!(parsed.samples[0].name, "sig");
        assert_eq!(parsed.samples[0].file, std::path::PathBuf::from("prediction.root"));
        assert_eq!(parsed.samples[0].weight.as_deref(), Some("w_mc"));
    }

    #[test]
    fn trex_parses_histo_systematic_type() {
        let cfg = r#"
ReadFrom: NTUP
TreeName: events

Region: SR
Variable: x,2,0,2
Selection: x > 0

Sample: sig
File: tests/fixtures/simple_tree.root

Systematic: lumi
Type: OVERALL
OverallUp: 0.02
OverallDown: -0.02
Samples: all

Systematic: jes
Type: HISTO
HistoNameUp: jes_up
HistoNameDown: jes_down
HistoFileUp: syst_up.root
HistoFileDown: syst_down.root
Samples: sig
"#;

        let parsed = TrexConfig::parse_str(cfg).expect("parse_str");
        assert_eq!(parsed.systematics.len(), 2, "both OVERALL and HISTO should be parsed");

        let lumi = &parsed.systematics[0];
        assert_eq!(lumi.name, "lumi");
        assert_eq!(lumi.kind, SystKind::Norm);
        assert!((lumi.hi.unwrap_or(0.0) - 1.02).abs() < 1e-12);
        assert!((lumi.lo.unwrap_or(0.0) - 0.98).abs() < 1e-12);

        let jes = &parsed.systematics[1];
        assert_eq!(jes.name, "jes");
        assert_eq!(jes.kind, SystKind::Histo);
        assert_eq!(jes.histo_name_up.as_deref(), Some("jes_up"));
        assert_eq!(jes.histo_name_down.as_deref(), Some("jes_down"));
        assert_eq!(jes.histo_file_up.as_ref().unwrap().to_str().unwrap(), "syst_up.root");
        assert_eq!(jes.histo_file_down.as_ref().unwrap().to_str().unwrap(), "syst_down.root");
        assert_eq!(jes.samples, vec!["sig".to_string()]);
    }

    #[test]
    fn trex_skips_histo_systematic_without_required_fields() {
        // HISTO without HistoNameUp/HistoNameDown → Validation error → skipped in best-effort
        let cfg = r#"
ReadFrom: NTUP
TreeName: events

Region: SR
Variable: x,2,0,2
Selection: x > 0

Sample: sig
File: tests/fixtures/simple_tree.root

Systematic: lumi
Type: OVERALL
OverallUp: 0.02
OverallDown: -0.02
Samples: all

Systematic: jes_bad
Type: HISTO
Samples: sig
"#;

        // This should NOT error out — the bad HISTO should be skipped like any invalid systematic.
        let parsed = TrexConfig::parse_str(cfg).expect("parse_str");
        assert_eq!(parsed.systematics.len(), 1, "only lumi should be parsed; jes_bad skipped");
        assert_eq!(parsed.systematics[0].name, "lumi");
    }

    #[test]
    fn trex_systematic_infers_weight_type_from_weightsuf_fields() {
        let cfg = r#"
ReadFrom: NTUP
TreeName: events

Region: SR
Variable: x,2,0,2
Selection: x > 0

Sample: sig
File: tests/fixtures/simple_tree.root
Weight: w0

Systematic: varR
Samples: sig
WeightSufUp: w_up
WeightSufDown: w_down
"#;

        let parsed = TrexConfig::parse_str(cfg).expect("parse_str");
        assert_eq!(parsed.systematics.len(), 1);
        let sys = &parsed.systematics[0];
        assert_eq!(sys.name, "varR");
        assert_eq!(sys.kind, SystKind::Weight);
        assert_eq!(sys.weight_suf_up.as_deref(), Some("w_up"));
        assert_eq!(sys.weight_suf_down.as_deref(), Some("w_down"));
    }

    #[test]
    fn trex_normfactor_sample_attr_backcompat_is_supported() {
        let cfg = r#"
ReadFrom: NTUP
TreeName: events

Region: SR
Variable: x,2,0,2
Selection: x > 0

Sample: sig
File: tests/fixtures/simple_tree.root
NormFactor: mu
"#;

        let parsed = TrexConfig::parse_str(cfg).expect("parse_str");
        assert_eq!(parsed.samples.len(), 1);
        assert!(
            parsed.samples[0]
                .modifiers
                .iter()
                .any(|m| matches!(m, NtupleModifier::NormFactor { name } if name == "mu")),
            "expected sample-level NormFactor modifier"
        );
    }

    #[test]
    fn trex_normfactor_block_applies_modifier_to_target_samples() {
        let cfg = r#"
ReadFrom: NTUP
TreeName: events

Region: SR
Variable: x,2,0,2
Selection: x > 0

Sample: sig
File: tests/fixtures/simple_tree.root

NormFactor: SigXsecOverSM
Samples: sig
"#;

        let parsed = TrexConfig::parse_str(cfg).expect("parse_str");
        assert_eq!(parsed.samples.len(), 1);
        assert!(
            parsed.samples[0]
                .modifiers
                .iter()
                .any(|m| matches!(m, NtupleModifier::NormFactor { name } if name == "SigXsecOverSM")),
            "expected NormFactor block to attach modifier"
        );
    }
}
