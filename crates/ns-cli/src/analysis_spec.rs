//! Analysis spec v0 (YAML) parsing + semantic validation.
//!
//! This is the NextStat-native replacement surface for TRExFitter configs:
//! a single YAML file drives import → fit → scan → report.

use anyhow::Result;
use serde::Deserialize;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use crate::run::RunConfig;

const SPEC_V0: &str = "trex_analysis_spec_v0";

#[derive(Debug, Clone)]
pub enum AnyRunConfig {
    Legacy(RunConfig),
    SpecV0(AnalysisSpecV0),
}

pub fn read_any_run_config(path: &Path) -> Result<AnyRunConfig> {
    let bytes = std::fs::read(path)?;

    // YAML parser can also read JSON (YAML is a superset), so we can use it for detection.
    let probe: serde_yaml_ng::Value = serde_yaml_ng::from_slice(&bytes)?;
    let schema_version =
        probe.get("schema_version").and_then(|v| v.as_str()).map(|s| s.to_string());

    if schema_version.as_deref() == Some(SPEC_V0) {
        let spec: AnalysisSpecV0 = serde_yaml_ng::from_slice(&bytes)?;
        return Ok(AnyRunConfig::SpecV0(spec));
    }

    Ok(AnyRunConfig::Legacy(crate::run::read_run_config(path)?))
}

#[derive(Debug, Clone)]
pub struct RunPlan {
    pub threads: usize,
    pub workspace_json: PathBuf,
    pub import: Option<ImportPlan>,
    pub preprocess: Option<PreprocessPlan>,
    pub fit: Option<PathBuf>,
    pub profile_scan: Option<ProfileScanPlan>,
    pub report: Option<ReportPlan>,
}

#[derive(Debug, Clone)]
pub struct PreprocessPlan {
    pub config_json: PathBuf,
    pub provenance_json: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub enum ImportPlan {
    HistfactoryXml { histfactory_xml: PathBuf },
    TrexConfigTxt { config_path: PathBuf, base_dir: PathBuf },
    TrexConfigYaml { config_text: String, base_dir: PathBuf },
}

#[derive(Debug, Clone)]
pub struct ProfileScanPlan {
    pub start: f64,
    pub stop: f64,
    pub points: usize,
    pub output_json: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ReportPlan {
    pub histfactory_xml: PathBuf,
    pub out_dir: PathBuf,
    pub overwrite: bool,
    pub include_covariance: bool,
    pub render: bool,
    pub pdf: Option<PathBuf>,
    pub svg_dir: Option<PathBuf>,
    pub python: Option<PathBuf>,
    pub skip_uncertainty: bool,
    pub uncertainty_grouping: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnalysisSpecV0 {
    #[serde(rename = "$schema")]
    #[allow(dead_code)]
    pub schema_uri: Option<String>,
    pub schema_version: String,
    #[allow(dead_code)]
    pub analysis: AnalysisMeta,
    pub inputs: Inputs,
    pub execution: Execution,
    #[allow(dead_code)]
    pub gates: Gates,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnalysisMeta {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Inputs {
    pub mode: String,
    #[serde(default)]
    pub histfactory: Option<HistfactoryInputs>,
    #[serde(default)]
    pub trex_config_txt: Option<TrexConfigTxtInputs>,
    #[serde(default)]
    pub trex_config_yaml: Option<TrexConfigYamlInputs>,
    #[serde(default)]
    pub workspace_json: Option<WorkspaceJsonInputs>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HistfactoryInputs {
    pub export_dir: PathBuf,
    pub combination_xml: Option<PathBuf>,
    #[allow(dead_code)]
    pub measurement: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrexConfigTxtInputs {
    pub config_path: PathBuf,
    pub base_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrexConfigYamlInputs {
    pub read_from: String,
    #[serde(default)]
    pub base_dir: Option<PathBuf>,
    #[serde(default)]
    pub histo_path: Option<PathBuf>,
    #[serde(default)]
    pub combination_xml: Option<PathBuf>,
    #[serde(default)]
    pub tree_name: Option<String>,
    #[serde(default)]
    pub measurement: Option<String>,
    #[serde(default)]
    pub poi: Option<String>,
    #[serde(default)]
    pub regions: Vec<TrexYamlRegion>,
    #[serde(default)]
    pub samples: Vec<TrexYamlSample>,
    #[serde(default)]
    pub systematics: Vec<TrexYamlSystematic>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrexYamlRegion {
    pub name: String,
    pub variable: String,
    pub binning_edges: Vec<f64>,
    #[serde(default)]
    pub selection: Option<String>,
    #[serde(default)]
    pub data_file: Option<PathBuf>,
    #[serde(default)]
    pub data_tree_name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrexYamlSampleKind {
    Data,
    Mc,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrexYamlNormSys {
    pub name: String,
    pub lo: f64,
    pub hi: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrexYamlSample {
    pub name: String,
    pub kind: TrexYamlSampleKind,
    pub file: PathBuf,
    #[serde(default)]
    pub tree_name: Option<String>,
    #[serde(default)]
    pub weight: Option<String>,
    #[serde(default)]
    pub regions: Option<Vec<String>>,
    #[serde(default)]
    pub norm_factors: Vec<String>,
    #[serde(default)]
    pub norm_sys: Vec<TrexYamlNormSys>,
    #[serde(default)]
    pub stat_error: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrexYamlSystematicType {
    Norm,
    Weight,
    Tree,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrexYamlSystematic {
    pub name: String,
    #[serde(rename = "type")]
    pub kind: TrexYamlSystematicType,
    pub samples: Vec<String>,
    #[serde(default)]
    pub regions: Option<Vec<String>>,
    #[serde(default)]
    pub lo: Option<f64>,
    #[serde(default)]
    pub hi: Option<f64>,
    #[serde(default)]
    pub weight_up: Option<String>,
    #[serde(default)]
    pub weight_down: Option<String>,
    #[serde(default)]
    pub file_up: Option<PathBuf>,
    #[serde(default)]
    pub file_down: Option<PathBuf>,
    #[serde(default)]
    pub tree_name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorkspaceJsonInputs {
    pub path: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Execution {
    pub determinism: Determinism,
    pub import: ImportStep,
    #[serde(default)]
    pub preprocessing: Option<PreprocessingStep>,
    pub fit: FitStep,
    pub profile_scan: ProfileScanStep,
    pub report: ReportStep,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PreprocessingStep {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub steps: Vec<serde_yaml_ng::Value>,
    #[serde(default)]
    pub provenance_json: Option<PathBuf>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct Determinism {
    pub threads: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ImportStep {
    pub enabled: bool,
    pub output_json: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FitStep {
    pub enabled: bool,
    pub output_json: PathBuf,
    #[serde(default)]
    pub fit_regions: Vec<String>,
    #[serde(default)]
    pub validation_regions: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProfileScanStep {
    pub enabled: bool,
    pub start: f64,
    pub stop: f64,
    pub points: usize,
    pub output_json: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReportStep {
    pub enabled: bool,
    pub out_dir: PathBuf,
    pub overwrite: bool,
    pub include_covariance: bool,
    pub histfactory_xml: Option<PathBuf>,
    pub render: RenderStep,
    pub skip_uncertainty: bool,
    pub uncertainty_grouping: String,
    #[serde(default)]
    pub blind_regions: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RenderStep {
    pub enabled: bool,
    pub pdf: Option<PathBuf>,
    pub svg_dir: Option<PathBuf>,
    pub python: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gates {
    pub baseline_compare: BaselineCompareGate,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BaselineCompareGate {
    pub enabled: bool,
    #[allow(dead_code)]
    pub baseline_dir: PathBuf,
    #[allow(dead_code)]
    pub require_same_host: bool,
    #[allow(dead_code)]
    pub max_slowdown: f64,
}

fn resolve_path(base_dir: &Path, p: &Path) -> PathBuf {
    if p.is_absolute() { p.to_path_buf() } else { base_dir.join(p) }
}

fn contains_duplicates(values: &[String]) -> Option<String> {
    let mut xs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
    xs.sort();
    for w in xs.windows(2) {
        if w[0] == w[1] {
            return Some(w[0].to_string());
        }
    }
    None
}

fn quote_if_needed(raw: &str) -> Result<String> {
    let s = raw.trim();
    if s.is_empty() {
        anyhow::bail!("value cannot be empty");
    }
    let needs_quote = s.chars().any(|c| c.is_whitespace()) || s.contains('#');
    if !needs_quote {
        return Ok(s.to_string());
    }
    if !s.contains('"') {
        return Ok(format!("\"{s}\""));
    }
    if !s.contains('\'') {
        return Ok(format!("'{s}'"));
    }
    anyhow::bail!("cannot safely quote value containing both single and double quotes: {s:?}");
}

fn fmt_edges(edges: &[f64]) -> String {
    edges.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")
}

fn fmt_list(values: &[String]) -> String {
    values.join(", ")
}

fn render_trex_config_yaml_to_txt(cfg: &TrexConfigYamlInputs) -> Result<String> {
    let read_from = cfg.read_from.trim().to_ascii_uppercase();
    match read_from.as_str() {
        "NTUP" => {
            if cfg.regions.is_empty() {
                anyhow::bail!("inputs.trex_config_yaml.regions must be non-empty");
            }
            if cfg.samples.is_empty() {
                anyhow::bail!("inputs.trex_config_yaml.samples must be non-empty");
            }

            let region_names: Vec<String> = cfg.regions.iter().map(|r| r.name.clone()).collect();
            if let Some(dup) = contains_duplicates(&region_names) {
                anyhow::bail!("duplicate region name: {dup}");
            }
            let sample_names: Vec<String> = cfg.samples.iter().map(|s| s.name.clone()).collect();
            if let Some(dup) = contains_duplicates(&sample_names) {
                anyhow::bail!("duplicate sample name: {dup}");
            }

            for r in &cfg.regions {
                if r.binning_edges.len() < 2 {
                    anyhow::bail!("region '{}' binning_edges must have >= 2 edges", r.name);
                }
            }

            let tree_name = cfg.tree_name.as_deref().unwrap_or("events");
            let measurement = cfg.measurement.as_deref().unwrap_or("meas");
            let poi = cfg.poi.as_deref().unwrap_or("mu");

            let mut out = String::new();
            out.push_str(&format!("ReadFrom: {read_from}\n"));
            out.push_str(&format!("TreeName: {}\n", quote_if_needed(tree_name)?));
            out.push_str(&format!("Measurement: {}\n", quote_if_needed(measurement)?));
            out.push_str(&format!("POI: {}\n\n", quote_if_needed(poi)?));

            for r in &cfg.regions {
                out.push_str(&format!("Region: {}\n", quote_if_needed(&r.name)?));
                out.push_str(&format!("Variable: {}\n", r.variable.trim()));
                out.push_str(&format!("Binning: {}\n", fmt_edges(&r.binning_edges)));
                if let Some(ref sel) = r.selection {
                    let sel = sel.trim();
                    if !sel.is_empty() {
                        out.push_str(&format!("Selection: {sel}\n"));
                    }
                }
                if let Some(ref df) = r.data_file {
                    out.push_str(&format!(
                        "DataFile: {}\n",
                        quote_if_needed(&df.display().to_string())?
                    ));
                    if let Some(ref tn) = r.data_tree_name {
                        out.push_str(&format!("DataTreeName: {}\n", quote_if_needed(tn)?));
                    }
                }
                out.push('\n');
            }

            for s in &cfg.samples {
                out.push_str(&format!("Sample: {}\n", quote_if_needed(&s.name)?));
                let kind = match s.kind {
                    TrexYamlSampleKind::Data => "data",
                    TrexYamlSampleKind::Mc => "mc",
                };
                out.push_str(&format!("Type: {kind}\n"));
                out.push_str(&format!(
                    "File: {}\n",
                    quote_if_needed(&s.file.display().to_string())?
                ));
                if let Some(ref tn) = s.tree_name {
                    out.push_str(&format!("TreeName: {}\n", quote_if_needed(tn)?));
                }
                if let Some(ref w) = s.weight {
                    let w = w.trim();
                    if !w.is_empty() {
                        out.push_str(&format!("Weight: {w}\n"));
                    }
                }
                if let Some(ref rs) = s.regions {
                    if !rs.is_empty() {
                        out.push_str(&format!("Regions: {}\n", fmt_list(rs)));
                    }
                }
                for nf in &s.norm_factors {
                    let nf = nf.trim();
                    if !nf.is_empty() {
                        out.push_str(&format!("NormFactor: {nf}\n"));
                    }
                }
                for ns in &s.norm_sys {
                    out.push_str(&format!(
                        "NormSys: {} {} {}\n",
                        ns.name.trim(),
                        ns.lo,
                        ns.hi
                    ));
                }
                if s.stat_error {
                    out.push_str("StatError: true\n");
                }
                out.push('\n');
            }

            for sys in &cfg.systematics {
                if sys.samples.is_empty() {
                    anyhow::bail!("systematic '{}' samples must be non-empty", sys.name);
                }
                out.push_str(&format!("Systematic: {}\n", quote_if_needed(&sys.name)?));
                let t = match sys.kind {
                    TrexYamlSystematicType::Norm => "norm",
                    TrexYamlSystematicType::Weight => "weight",
                    TrexYamlSystematicType::Tree => "tree",
                };
                out.push_str(&format!("Type: {t}\n"));
                out.push_str(&format!("Samples: {}\n", fmt_list(&sys.samples)));
                if let Some(ref rs) = sys.regions {
                    if !rs.is_empty() {
                        out.push_str(&format!("Regions: {}\n", fmt_list(rs)));
                    }
                }

                match sys.kind {
                    TrexYamlSystematicType::Norm => {
                        let (Some(lo), Some(hi)) = (sys.lo, sys.hi) else {
                            anyhow::bail!("systematic '{}' type=norm requires lo and hi", sys.name);
                        };
                        out.push_str(&format!("Lo: {lo}\n"));
                        out.push_str(&format!("Hi: {hi}\n"));
                    }
                    TrexYamlSystematicType::Weight => {
                        let (Some(up), Some(down)) = (&sys.weight_up, &sys.weight_down) else {
                            anyhow::bail!(
                                "systematic '{}' type=weight requires weight_up and weight_down",
                                sys.name
                            );
                        };
                        out.push_str(&format!("WeightUp: {}\n", up.trim()));
                        out.push_str(&format!("WeightDown: {}\n", down.trim()));
                    }
                    TrexYamlSystematicType::Tree => {
                        let (Some(up), Some(down)) = (&sys.file_up, &sys.file_down) else {
                            anyhow::bail!(
                                "systematic '{}' type=tree requires file_up and file_down",
                                sys.name
                            );
                        };
                        out.push_str(&format!(
                            "FileUp: {}\n",
                            quote_if_needed(&up.display().to_string())?
                        ));
                        out.push_str(&format!(
                            "FileDown: {}\n",
                            quote_if_needed(&down.display().to_string())?
                        ));
                        if let Some(ref tn) = sys.tree_name {
                            out.push_str(&format!("TreeName: {}\n", quote_if_needed(tn)?));
                        }
                    }
                }

                out.push('\n');
            }

            Ok(out)
        }
        "HIST" => {
            if cfg.histo_path.is_none() && cfg.combination_xml.is_none() {
                anyhow::bail!(
                    "inputs.trex_config_yaml.read_from=HIST requires histo_path or combination_xml"
                );
            }
            let mut out = String::new();
            out.push_str("ReadFrom: HIST\n");
            if let Some(ref p) = cfg.histo_path {
                out.push_str(&format!(
                    "HistoPath: {}\n",
                    quote_if_needed(&p.display().to_string())?
                ));
            }
            if let Some(ref p) = cfg.combination_xml {
                out.push_str(&format!(
                    "CombinationXml: {}\n",
                    quote_if_needed(&p.display().to_string())?
                ));
            }
            if let Some(ref m) = cfg.measurement {
                let m = m.trim();
                if !m.is_empty() {
                    out.push_str(&format!("Measurement: {}\n", quote_if_needed(m)?));
                }
            }
            if let Some(ref poi) = cfg.poi {
                let poi = poi.trim();
                if !poi.is_empty() {
                    out.push_str(&format!("POI: {}\n", quote_if_needed(poi)?));
                }
            }
            out.push('\n');
            Ok(out)
        }
        other => anyhow::bail!(
            "inputs.trex_config_yaml.read_from must be NTUP or HIST in v0, got={}",
            other
        ),
    }
}

fn find_combination_xml(export_dir: &Path) -> Result<PathBuf> {
    // First: common TREx/HistFactory convention.
    let direct = export_dir.join("combination.xml");
    if direct.is_file() {
        return Ok(direct);
    }

    // Recursive search with deterministic ordering. Keep it bounded.
    let mut queue: VecDeque<(PathBuf, usize)> = VecDeque::new();
    queue.push_back((export_dir.to_path_buf(), 0));
    let mut hits: Vec<PathBuf> = Vec::new();

    while let Some((dir, depth)) = queue.pop_front() {
        if depth > 6 {
            continue;
        }
        let mut entries: Vec<PathBuf> = match std::fs::read_dir(&dir) {
            Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).collect(),
            Err(_) => continue,
        };
        entries.sort();

        for p in entries {
            if p.is_dir() {
                queue.push_back((p, depth + 1));
                continue;
            }
            if p.file_name().and_then(|s| s.to_str()) == Some("combination.xml") {
                hits.push(p);
            }
        }
    }

    hits.sort();
    hits.dedup();

    match hits.len() {
        0 => anyhow::bail!(
            "failed to auto-discover combination.xml under export_dir={}",
            export_dir.display()
        ),
        1 => Ok(hits.remove(0)),
        _ => anyhow::bail!(
            "multiple combination.xml files found under export_dir={}; set inputs.histfactory.combination_xml explicitly (first few: {})",
            export_dir.display(),
            hits.iter().take(5).map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
        ),
    }
}

impl AnalysisSpecV0 {
    pub fn to_run_plan(&self, config_path: &Path) -> Result<RunPlan> {
        if self.schema_version != SPEC_V0 {
            anyhow::bail!(
                "unsupported schema_version for analysis spec: got={} expected={}",
                self.schema_version,
                SPEC_V0
            );
        }

        let cfg_dir = config_path.parent().unwrap_or_else(|| Path::new("."));
        let threads = self.execution.determinism.threads.max(1);

        let mode = self.inputs.mode.as_str();

        // Workspace selection/build
        let (workspace_json, import_plan, histfactory_xml_hint): (
            PathBuf,
            Option<ImportPlan>,
            Option<PathBuf>,
        ) = match mode {
            "histfactory_xml" => {
                let hf = self.inputs.histfactory.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("inputs.histfactory is required for mode=histfactory_xml")
                })?;

                if !self.execution.import.enabled {
                    anyhow::bail!(
                        "execution.import.enabled must be true for mode=histfactory_xml (needs a workspace.json path)"
                    );
                }

                let export_dir = resolve_path(cfg_dir, &hf.export_dir);
                let xml = if let Some(ref p) = hf.combination_xml {
                    resolve_path(cfg_dir, p)
                } else {
                    find_combination_xml(&export_dir)?
                };
                if !xml.is_file() {
                    anyhow::bail!("histfactory combination.xml not found: {}", xml.display());
                }

                let ws_out = resolve_path(cfg_dir, &self.execution.import.output_json);
                (
                    ws_out,
                    Some(ImportPlan::HistfactoryXml { histfactory_xml: xml.clone() }),
                    Some(xml),
                )
            }
            "trex_config_txt" => {
                let tc = self.inputs.trex_config_txt.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("inputs.trex_config_txt is required for mode=trex_config_txt")
                })?;
                if !self.execution.import.enabled {
                    anyhow::bail!(
                        "execution.import.enabled must be true for mode=trex_config_txt (needs a workspace.json path)"
                    );
                }

                let config_path = resolve_path(cfg_dir, &tc.config_path);
                let base_dir =
                    tc.base_dir.as_ref().map(|p| resolve_path(cfg_dir, p)).unwrap_or_else(|| {
                        config_path.parent().unwrap_or_else(|| Path::new(".")).to_path_buf()
                    });

                let ws_out = resolve_path(cfg_dir, &self.execution.import.output_json);
                (ws_out, Some(ImportPlan::TrexConfigTxt { config_path, base_dir }), None)
            }
            "workspace_json" => {
                let ws = self.inputs.workspace_json.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("inputs.workspace_json is required for mode=workspace_json")
                })?;
                let p = resolve_path(cfg_dir, &ws.path);
                if !p.is_file() {
                    anyhow::bail!("workspace JSON not found: {}", p.display());
                }
                (p, None, None)
            }
            "trex_config_yaml" => {
                let ty = self.inputs.trex_config_yaml.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("inputs.trex_config_yaml is required for mode=trex_config_yaml")
                })?;
                if !self.execution.import.enabled {
                    anyhow::bail!(
                        "execution.import.enabled must be true for mode=trex_config_yaml (needs a workspace.json path)"
                    );
                }

                let base_dir = if let Some(ref p) = ty.base_dir {
                    resolve_path(cfg_dir, p)
                } else {
                    cfg_dir.to_path_buf()
                };

                let config_text = render_trex_config_yaml_to_txt(ty)?;
                let ws_out = resolve_path(cfg_dir, &self.execution.import.output_json);
                (
                    ws_out,
                    Some(ImportPlan::TrexConfigYaml { config_text, base_dir }),
                    None,
                )
            }
            other => anyhow::bail!("unknown inputs.mode: {other}"),
        };

        // Fit (optional)
        let fit = if self.execution.fit.enabled {
            Some(resolve_path(cfg_dir, &self.execution.fit.output_json))
        } else {
            None
        };

        // Scan (optional)
        let profile_scan = if self.execution.profile_scan.enabled {
            if self.execution.profile_scan.points < 2 {
                anyhow::bail!("execution.profile_scan.points must be >= 2");
            }
            Some(ProfileScanPlan {
                start: self.execution.profile_scan.start,
                stop: self.execution.profile_scan.stop,
                points: self.execution.profile_scan.points,
                output_json: resolve_path(cfg_dir, &self.execution.profile_scan.output_json),
            })
        } else {
            None
        };

        // Report (optional)
        let report = if self.execution.report.enabled {
            let out_dir = resolve_path(cfg_dir, &self.execution.report.out_dir);

            let report_xml = if let Some(ref p) = self.execution.report.histfactory_xml {
                resolve_path(cfg_dir, p)
            } else if let Some(p) = histfactory_xml_hint.as_ref() {
                p.clone()
            } else {
                anyhow::bail!(
                    "execution.report.histfactory_xml is required when inputs.mode={} (bin edges extraction needs combination.xml)",
                    mode
                );
            };
            if !report_xml.is_file() {
                anyhow::bail!("report histfactory_xml not found: {}", report_xml.display());
            }

            Some(ReportPlan {
                histfactory_xml: report_xml,
                out_dir,
                overwrite: self.execution.report.overwrite,
                include_covariance: self.execution.report.include_covariance,
                render: self.execution.report.render.enabled,
                pdf: self.execution.report.render.pdf.as_ref().map(|p| resolve_path(cfg_dir, p)),
                svg_dir: self
                    .execution
                    .report
                    .render
                    .svg_dir
                    .as_ref()
                    .map(|p| resolve_path(cfg_dir, p)),
                python: self
                    .execution
                    .report
                    .render
                    .python
                    .as_ref()
                    .map(|p| resolve_path(cfg_dir, p)),
                skip_uncertainty: self.execution.report.skip_uncertainty,
                uncertainty_grouping: self.execution.report.uncertainty_grouping.clone(),
            })
        } else {
            None
        };

        // Preprocessing (optional)
        let preprocess = if let Some(ref pp) = self.execution.preprocessing {
            if pp.enabled {
                // Write the preprocessing config (steps array) to a temp JSON next to workspace.
                let config_json = workspace_json
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .join("_preprocess_config.json");
                let provenance_json = pp
                    .provenance_json
                    .as_ref()
                    .map(|p| resolve_path(cfg_dir, p));
                Some(PreprocessPlan { config_json, provenance_json })
            } else {
                None
            }
        } else {
            None
        };

        Ok(RunPlan { threads, workspace_json, import: import_plan, preprocess, fit, profile_scan, report })
    }
}
