//! `nextstat run` orchestration (Phase 0 foundation).

use anyhow::Result;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

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
    if let Some(parent) = dst.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::copy(src, dst)?;
    Ok(())
}

fn copy_tree_filtered(src_dir: &Path, dst_dir: &Path, exts: &[&str]) -> Result<Vec<PathBuf>> {
    let mut copied = Vec::new();
    if !src_dir.exists() {
        return Ok(copied);
    }
    for entry in std::fs::read_dir(src_dir)? {
        let entry = entry?;
        let p = entry.path();
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

    // Copy config.
    copy_file_into(config_path, &inputs_dir.join("run_config.yaml"))?;

    // Copy the HistFactory export directory (xml/root) to preserve referenced files.
    let hf_dir = cfg.histfactory_xml.parent().unwrap_or_else(|| Path::new("."));
    copy_tree_filtered(hf_dir, &inputs_dir.join("histfactory"), &["xml", "root"])?;

    // Copy run outputs (inputs+artifacts) verbatim.
    copy_tree_filtered(
        &cfg.out_dir,
        &outputs_dir.join("run"),
        &["json", "csv", "tex", "pdf", "svg"],
    )?;

    // Meta (small, stable).
    let meta = serde_json::json!({
        "tool": "nextstat",
        "tool_version": ns_core::VERSION,
        "command": "run",
        "args": {
            "histfactory_xml_sha256": sha256_file(&cfg.histfactory_xml)?,
            "out_dir": cfg.out_dir,
            "threads": cfg.threads,
        },
    });
    std::fs::write(bundle_dir.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;

    // Manifest (sha256 + size).
    let mut files = Vec::new();
    for rel in ["meta.json", "inputs/run_config.yaml"] {
        let p = bundle_dir.join(rel);
        files.push(ManifestFile {
            path: rel.to_string(),
            bytes: file_size(&p)?,
            sha256: sha256_file(&p)?,
        });
    }

    for root in [inputs_dir.as_path(), outputs_dir.as_path()] {
        for entry in walk_files(root)? {
            let rel = entry.strip_prefix(bundle_dir).unwrap_or(&entry);
            files.push(ManifestFile {
                path: rel.display().to_string(),
                bytes: file_size(&entry)?,
                sha256: sha256_file(&entry)?,
            });
        }
    }

    // Deterministic ordering.
    files.sort_by(|a, b| a.path.cmp(&b.path));

    let manifest = Manifest { bundle_version: 1, files };
    std::fs::write(bundle_dir.join("manifest.json"), serde_json::to_string_pretty(&manifest)?)?;

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
