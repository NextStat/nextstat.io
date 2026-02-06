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
    let schema_version = probe
        .get("schema_version")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

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
    pub fit: Option<PathBuf>,
    pub profile_scan: Option<ProfileScanPlan>,
    pub report: Option<ReportPlan>,
}

#[derive(Debug, Clone)]
pub enum ImportPlan {
    HistfactoryXml { histfactory_xml: PathBuf },
    TrexConfigTxt { config_path: PathBuf, base_dir: PathBuf },
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
    // Parsed, but not executed by `nextstat run` yet.
    #[allow(dead_code)]
    pub read_from: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorkspaceJsonInputs {
    pub path: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Execution {
    pub determinism: Determinism,
    pub import: ImportStep,
    pub fit: FitStep,
    pub profile_scan: ProfileScanStep,
    pub report: ReportStep,
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
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        base_dir.join(p)
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
            hits.iter()
                .take(5)
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
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
        let (workspace_json, import_plan, histfactory_xml_hint): (PathBuf, Option<ImportPlan>, Option<PathBuf>) =
            match mode {
                "histfactory_xml" => {
                    let hf = self
                        .inputs
                        .histfactory
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("inputs.histfactory is required for mode=histfactory_xml"))?;

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
                    let base_dir = tc
                        .base_dir
                        .as_ref()
                        .map(|p| resolve_path(cfg_dir, p))
                        .unwrap_or_else(|| {
                            config_path.parent().unwrap_or_else(|| Path::new(".")).to_path_buf()
                        });

                    let ws_out = resolve_path(cfg_dir, &self.execution.import.output_json);
                    (
                        ws_out,
                        Some(ImportPlan::TrexConfigTxt { config_path, base_dir }),
                        None,
                    )
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
                    anyhow::bail!("mode=trex_config_yaml is not supported by `nextstat run` yet")
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
                anyhow::bail!(
                    "report histfactory_xml not found: {}",
                    report_xml.display()
                );
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

        Ok(RunPlan {
            threads,
            workspace_json,
            import: import_plan,
            fit,
            profile_scan,
            report,
        })
    }
}

