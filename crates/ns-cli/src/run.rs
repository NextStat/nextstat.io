//! `nextstat run` orchestration (Phase 0 foundation).

use anyhow::Result;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Deserialize)]
pub struct RunConfig {
    /// Path to HistFactory `combination.xml` (and referenced files).
    pub histfactory_xml: PathBuf,
    /// Output directory for this run (inputs + artifacts).
    pub out_dir: PathBuf,

    /// Allow writing into a non-empty `out_dir` (overwrites known filenames).
    #[serde(default)]
    pub overwrite: bool,

    /// Threads (0 = auto). Use 1 for deterministic parity.
    #[serde(default = "default_threads")]
    pub threads: usize,

    /// Make JSON output deterministic (stable ordering; normalize timestamps/timings).
    #[serde(default)]
    pub deterministic: bool,

    /// Also include the raw covariance matrix in correlation artifacts (if available).
    #[serde(default)]
    pub include_covariance: bool,

    /// Skip the uncertainty breakdown artifact (ranking-based).
    #[serde(default)]
    pub skip_uncertainty: bool,

    /// Uncertainty grouping policy (currently: `prefix_1`).
    #[serde(default = "default_uncertainty_grouping")]
    pub uncertainty_grouping: String,

    /// Optional report rendering via Python.
    #[serde(default)]
    pub render: bool,
    #[serde(default)]
    pub pdf: Option<PathBuf>,
    #[serde(default)]
    pub svg_dir: Option<PathBuf>,
    #[serde(default)]
    pub python: Option<PathBuf>,
}

fn default_threads() -> usize {
    1
}

fn default_uncertainty_grouping() -> String {
    "prefix_1".to_string()
}

pub fn read_run_config(path: &Path) -> Result<RunConfig> {
    let bytes = std::fs::read(path)?;
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
    let cfg: RunConfig = if ext == "json" {
        serde_json::from_slice(&bytes)?
    } else {
        // Default: YAML (serde_yaml_ng).
        serde_yaml_ng::from_slice(&bytes)?
    };
    Ok(cfg)
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RunPaths {
    pub out_dir: PathBuf,
    pub inputs_dir: PathBuf,
    pub artifacts_dir: PathBuf,
    pub workspace_json: PathBuf,
}

pub fn derive_paths(out_dir: &Path) -> RunPaths {
    let inputs_dir = out_dir.join("inputs");
    let artifacts_dir = out_dir.join("artifacts");
    let workspace_json = inputs_dir.join("workspace.json");
    RunPaths { out_dir: out_dir.to_path_buf(), inputs_dir, artifacts_dir, workspace_json }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let out = h.finalize();
    let mut s = String::with_capacity(64);
    for b in out {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn sha256_file(path: &Path) -> Result<String> {
    Ok(sha256_hex(&std::fs::read(path)?))
}

fn file_size(path: &Path) -> Result<u64> {
    Ok(std::fs::metadata(path)?.len())
}

fn ensure_empty_dir(dir: &Path) -> Result<()> {
    if dir.exists() {
        if !dir.is_dir() {
            anyhow::bail!("bundle path exists but is not a directory: {}", dir.display());
        }
        if dir.read_dir()?.next().is_some() {
            anyhow::bail!("bundle directory must be empty: {}", dir.display());
        }
    } else {
        std::fs::create_dir_all(dir)?;
    }
    Ok(())
}

fn copy_file_into(src: &Path, dst: &Path) -> Result<()> {
    if let Some(parent) = dst.parent()
        && !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    std::fs::copy(src, dst)?;
    Ok(())
}

fn copy_tree_filtered(src_dir: &Path, dst_dir: &Path, exts: &[&str]) -> Result<Vec<PathBuf>> {
    let mut copied = Vec::new();
    if !src_dir.exists() {
        return Ok(copied);
    }
    let mut entries: Vec<PathBuf> =
        std::fs::read_dir(src_dir)?.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        let rel = p.strip_prefix(src_dir).unwrap_or(&p);
        if p.is_dir() {
            copied.extend(copy_tree_filtered(&p, &dst_dir.join(rel), exts)?);
            continue;
        }
        let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
        if exts.iter().any(|e| *e == ext) {
            let dst = dst_dir.join(rel);
            copy_file_into(&p, &dst)?;
            copied.push(dst);
        }
    }
    Ok(copied)
}

#[derive(Debug, Clone, serde::Serialize)]
struct ProvenanceFile {
    role: String,
    original_path: String,
    bundle_path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Clone, serde::Serialize)]
struct Provenance {
    created_unix_ms: u128,
    config_type: String,
    config_path: String,
    config_sha256: String,
    inputs: Vec<ProvenanceFile>,
    outputs: Vec<ProvenanceFile>,
    notes: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct Manifest {
    bundle_version: u32,
    files: Vec<ManifestFile>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ManifestFile {
    path: String,
    bytes: u64,
    sha256: String,
}

fn rel_display(root: &Path, p: &Path) -> String {
    p.strip_prefix(root).unwrap_or(p).display().to_string()
}

fn record_prov_file(
    list: &mut Vec<ProvenanceFile>,
    role: &str,
    original: &Path,
    bundle_root: &Path,
    bundle_path: &Path,
) -> Result<()> {
    if !bundle_path.is_file() {
        anyhow::bail!(
            "expected bundle file for role={role} but it does not exist: {}",
            bundle_path.display()
        );
    }
    list.push(ProvenanceFile {
        role: role.to_string(),
        original_path: original.display().to_string(),
        bundle_path: rel_display(bundle_root, bundle_path),
        bytes: file_size(bundle_path)?,
        sha256: sha256_file(bundle_path)?,
    });
    Ok(())
}

fn write_manifest(bundle_dir: &Path) -> Result<()> {
    let mut files = Vec::new();
    for entry in walk_files(bundle_dir)? {
        if entry.file_name().and_then(|s| s.to_str()) == Some("manifest.json") {
            continue;
        }
        files.push(ManifestFile {
            path: rel_display(bundle_dir, &entry),
            bytes: file_size(&entry)?,
            sha256: sha256_file(&entry)?,
        });
    }

    files.sort_by(|a, b| a.path.cmp(&b.path));
    let manifest = Manifest { bundle_version: 1, files };
    std::fs::write(bundle_dir.join("manifest.json"), serde_json::to_string_pretty(&manifest)?)?;
    Ok(())
}

fn write_meta(bundle_dir: &Path, meta: &serde_json::Value) -> Result<()> {
    std::fs::write(bundle_dir.join("meta.json"), serde_json::to_string_pretty(meta)?)?;
    Ok(())
}

/// Best-effort “run bundle” writer for `nextstat run`.
///
/// This is intentionally small and will evolve as the Phase 0 bundle contract
/// is finalized (see TREx replacement parity docs).
pub fn write_run_bundle(bundle_dir: &Path, config_path: &Path, cfg: &RunConfig) -> Result<()> {
    ensure_empty_dir(bundle_dir)?;

    let inputs_dir = bundle_dir.join("inputs");
    let outputs_dir = bundle_dir.join("outputs");
    std::fs::create_dir_all(&inputs_dir)?;
    std::fs::create_dir_all(&outputs_dir)?;

    let created_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let config_sha256 = sha256_file(config_path)?;

    // Copy config.
    let config_copy = inputs_dir.join("run_config.yaml");
    copy_file_into(config_path, &config_copy)?;

    // Copy the HistFactory export directory (xml/root) to preserve referenced files.
    let hf_dir = cfg.histfactory_xml.parent().unwrap_or_else(|| Path::new("."));
    copy_tree_filtered(hf_dir, &inputs_dir.join("histfactory"), &["xml", "root"])?;

    // Copy run outputs (inputs+artifacts) verbatim.
    let run_outputs = outputs_dir.join("run");
    copy_tree_filtered(&cfg.out_dir, &run_outputs, &["json", "csv", "tex", "pdf", "svg"])?;

    // Provenance (intermediate hashes).
    let mut prov_inputs = Vec::new();
    let mut prov_outputs = Vec::new();
    let mut notes = Vec::new();

    record_prov_file(&mut prov_inputs, "run_config", config_path, bundle_dir, &config_copy)?;

    let hf_rel = cfg.histfactory_xml.strip_prefix(hf_dir).unwrap_or(cfg.histfactory_xml.as_path());
    let hf_copy = inputs_dir.join("histfactory").join(hf_rel);
    if hf_copy.is_file() {
        record_prov_file(
            &mut prov_inputs,
            "histfactory_xml",
            &cfg.histfactory_xml,
            bundle_dir,
            &hf_copy,
        )?;
    } else {
        notes.push(format!(
            "histfactory_xml copy not found at expected path (original={}, expected_copy={})",
            cfg.histfactory_xml.display(),
            hf_copy.display()
        ));
    }

    let key_outputs: &[(&str, &str)] = &[
        ("workspace_json", "inputs/workspace.json"),
        ("fit_json", "artifacts/fit.json"),
        ("distributions_json", "artifacts/distributions.json"),
        ("yields_json", "artifacts/yields.json"),
        ("pulls_json", "artifacts/pulls.json"),
        ("corr_json", "artifacts/corr.json"),
        ("uncertainty_json", "artifacts/uncertainty.json"),
        ("yields_csv", "artifacts/yields.csv"),
        ("yields_tex", "artifacts/yields.tex"),
        ("report_pdf", "artifacts/report.pdf"),
    ];
    for (role, rel) in key_outputs {
        let original = cfg.out_dir.join(rel);
        if !original.exists() {
            continue;
        }
        let copy = run_outputs.join(rel);
        if copy.exists() {
            record_prov_file(&mut prov_outputs, role, &original, bundle_dir, &copy)?;
        }
    }

    let prov = Provenance {
        created_unix_ms,
        config_type: "run_config_legacy".to_string(),
        config_path: config_path.display().to_string(),
        config_sha256,
        inputs: prov_inputs,
        outputs: prov_outputs,
        notes,
    };
    std::fs::write(bundle_dir.join("provenance.json"), serde_json::to_string_pretty(&prov)?)?;

    // Meta (small, stable-ish).
    let meta = serde_json::json!({
        "tool": "nextstat",
        "tool_version": ns_core::VERSION,
        "command": "run",
        "created_unix_ms": created_unix_ms,
        "config_type": "run_config_legacy",
        "args": {
            "config_sha256": sha256_file(config_path)?,
            "histfactory_xml_sha256": sha256_file(&cfg.histfactory_xml)?,
            "out_dir": cfg.out_dir,
            "threads": cfg.threads,
        },
        "provenance": "provenance.json"
    });
    write_meta(bundle_dir, &meta)?;
    write_manifest(bundle_dir)?;

    Ok(())
}

pub fn write_run_bundle_spec_v0(
    bundle_dir: &Path,
    config_path: &Path,
    spec: &crate::analysis_spec::AnalysisSpecV0,
    plan: &crate::analysis_spec::RunPlan,
) -> Result<()> {
    ensure_empty_dir(bundle_dir)?;

    let created_unix_ms = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let config_sha256 = sha256_file(config_path)?;

    let inputs_dir = bundle_dir.join("inputs");
    let outputs_dir = bundle_dir.join("outputs");
    std::fs::create_dir_all(&inputs_dir)?;
    std::fs::create_dir_all(&outputs_dir)?;

    // Copy config.
    let config_copy = inputs_dir.join("analysis.yaml");
    copy_file_into(config_path, &config_copy)?;

    let mut prov_inputs = Vec::new();
    let mut prov_outputs = Vec::new();
    let mut notes = Vec::new();
    record_prov_file(&mut prov_inputs, "analysis_spec", config_path, bundle_dir, &config_copy)?;

    // Copy HistFactory inputs when available (best-effort).
    let mut histfactory_xmls: Vec<PathBuf> = Vec::new();
    if let Some(import) = plan.import.as_ref()
        && let crate::analysis_spec::ImportPlan::HistfactoryXml { histfactory_xml } = import {
            histfactory_xmls.push(histfactory_xml.clone());
        }
    if let Some(report) = plan.report.as_ref() {
        histfactory_xmls.push(report.histfactory_xml.clone());
    }
    histfactory_xmls.sort();
    histfactory_xmls.dedup();

    for (i, xml) in histfactory_xmls.iter().enumerate() {
        let hf_dir = xml.parent().unwrap_or_else(|| Path::new("."));
        copy_tree_filtered(hf_dir, &inputs_dir.join("histfactory"), &["xml", "root"])?;
        let rel = xml.strip_prefix(hf_dir).unwrap_or(xml.as_path());
        let copy = inputs_dir.join("histfactory").join(rel);
        if copy.is_file() {
            record_prov_file(
                &mut prov_inputs,
                &format!("histfactory_xml[{i}]"),
                xml,
                bundle_dir,
                &copy,
            )?;
        }
    }

    // Copy trex config text if used.
    if let Some(import) = plan.import.as_ref()
        && let crate::analysis_spec::ImportPlan::TrexConfigTxt { config_path, .. } = import {
            let dst = inputs_dir.join("trex").join("config.txt");
            copy_file_into(config_path, &dst)?;
            record_prov_file(&mut prov_inputs, "trex_config_txt", config_path, bundle_dir, &dst)?;
            notes.push(
                "trex_config_txt mode: run bundle v1 currently records only the config text (not referenced ROOT files)"
                    .to_string(),
            );
        }

    // Workspace input (workspace_json mode): copy the input workspace.
    if spec.inputs.mode == "workspace_json" {
        let dst = inputs_dir.join("workspace.json");
        copy_file_into(&plan.workspace_json, &dst)?;
        record_prov_file(
            &mut prov_inputs,
            "workspace_json_input",
            &plan.workspace_json,
            bundle_dir,
            &dst,
        )?;
    }

    // Step outputs with stable bundle paths.
    if plan.import.is_some() && plan.workspace_json.exists() {
        let dst = outputs_dir.join("workspace.json");
        copy_file_into(&plan.workspace_json, &dst)?;
        record_prov_file(
            &mut prov_outputs,
            "workspace_json",
            &plan.workspace_json,
            bundle_dir,
            &dst,
        )?;
    }
    if let Some(fit) = plan.fit.as_ref()
        && fit.exists() {
            let dst = outputs_dir.join("fit.json");
            copy_file_into(fit, &dst)?;
            record_prov_file(&mut prov_outputs, "fit_json", fit, bundle_dir, &dst)?;
        }
    if let Some(scan) = plan.profile_scan.as_ref()
        && scan.output_json.exists() {
            let dst = outputs_dir.join("scan.json");
            copy_file_into(&scan.output_json, &dst)?;
            record_prov_file(&mut prov_outputs, "scan_json", &scan.output_json, bundle_dir, &dst)?;
        }
    if let Some(report) = plan.report.as_ref() {
        let dst_dir = outputs_dir.join("report");
        copy_tree_filtered(&report.out_dir, &dst_dir, &["json", "csv", "tex", "pdf", "svg"])?;

        let key_reports: &[(&str, &str)] = &[
            ("report_fit_json", "fit.json"),
            ("report_distributions_json", "distributions.json"),
            ("report_yields_json", "yields.json"),
            ("report_pulls_json", "pulls.json"),
            ("report_corr_json", "corr.json"),
            ("report_uncertainty_json", "uncertainty.json"),
            ("report_pdf", "report.pdf"),
        ];
        for (role, name) in key_reports {
            let original = report.out_dir.join(name);
            let copy = dst_dir.join(name);
            if original.exists() && copy.exists() {
                record_prov_file(&mut prov_outputs, role, &original, bundle_dir, &copy)?;
            }
        }
    }

    let prov = Provenance {
        created_unix_ms,
        config_type: "analysis_spec_v0".to_string(),
        config_path: config_path.display().to_string(),
        config_sha256,
        inputs: prov_inputs,
        outputs: prov_outputs,
        notes,
    };
    std::fs::write(bundle_dir.join("provenance.json"), serde_json::to_string_pretty(&prov)?)?;

    let meta = serde_json::json!({
        "tool": "nextstat",
        "tool_version": ns_core::VERSION,
        "command": "run",
        "created_unix_ms": created_unix_ms,
        "config_type": "analysis_spec_v0",
        "schema_version": spec.schema_version,
        "inputs": { "mode": spec.inputs.mode },
        "plan": {
            "threads": plan.threads,
            "workspace_json": plan.workspace_json,
            "import": plan.import.as_ref().map(|_| true).unwrap_or(false),
            "fit": plan.fit.is_some(),
            "profile_scan": plan.profile_scan.is_some(),
            "report": plan.report.is_some(),
        },
        "args": {
            "config_sha256": sha256_file(config_path)?,
        },
        "provenance": "provenance.json"
    });
    write_meta(bundle_dir, &meta)?;
    write_manifest(bundle_dir)?;

    Ok(())
}

fn walk_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    if !dir.exists() {
        return Ok(out);
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let p = entry.path();
        if p.is_dir() {
            out.extend(walk_files(&p)?);
        } else {
            out.push(p);
        }
    }
    Ok(out)
}
