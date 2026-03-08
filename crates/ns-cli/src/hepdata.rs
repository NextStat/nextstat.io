use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Instant;
use tar::Archive;
use zip::ZipArchive;

const EMBEDDED_MANIFEST_JSON: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/hepdata/manifest.json"));
const HEPDATA_IMPORT_SCHEMA_V1: &str = "nextstat.hepdata_import.v1";
const HEPDATA_LOCK_SCHEMA_V1: &str = "nextstat.hepdata_lock.v1";

#[derive(Debug, Deserialize, Serialize)]
struct ManifestFile {
    datasets: Vec<ManifestDataset>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ManifestDataset {
    id: String,
    name: Option<String>,
    doi: String,
    #[serde(default)]
    materialize: MaterializeSpec,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct MaterializeSpec {
    #[serde(default = "default_true")]
    bkgonly: bool,
    #[serde(default)]
    patches: Vec<PatchMaterialization>,
    bkgonly_filename: Option<String>,
    patchset_filename: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct PatchMaterialization {
    id: Option<String>,
    patch_name: Option<String>,
    bkgonly_filename: Option<String>,
    patchset_filename: Option<String>,
}

#[derive(Debug, Serialize)]
struct MaterializeSummary {
    schema_version: &'static str,
    mode: &'static str,
    source: &'static str,
    source_mode: &'static str,
    manifest: String,
    out_dir: String,
    cache_dir: String,
    lock: String,
    datasets: Vec<SummaryDataset>,
}

#[derive(Debug, Serialize)]
struct SummaryDataset {
    id: String,
    name: String,
    doi: String,
    download: LockDownload,
    inputs: DatasetInputs,
    #[serde(skip_serializing_if = "Option::is_none")]
    timings: Option<DatasetTimings>,
    materialized: Vec<SummaryMaterialized>,
}

#[derive(Debug, Serialize)]
struct SummaryMaterialized {
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    patch_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    patch_name: Option<String>,
    path: String,
}

#[derive(Debug, Serialize)]
struct CatalogSummary {
    schema_version: &'static str,
    mode: &'static str,
    source: &'static str,
    source_mode: &'static str,
    manifest: String,
    datasets: Vec<CatalogDataset>,
}

#[derive(Debug, Serialize)]
struct CatalogDataset {
    id: String,
    name: String,
    doi: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    download: Option<LockDownload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inputs: Option<DatasetInputs>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timings: Option<DatasetTimings>,
    materialize: CatalogMaterialize,
}

#[derive(Debug, Serialize)]
struct CatalogMaterialize {
    bkgonly: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    bkgonly_filename: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    patchset_filename: Option<String>,
    patches: Vec<CatalogPatch>,
}

#[derive(Debug, Serialize)]
struct CatalogPatch {
    id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    patch_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bkgonly_filename: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    patchset_filename: Option<String>,
}

#[derive(Debug, Serialize)]
struct Lockfile {
    schema_version: &'static str,
    source_mode: &'static str,
    generated_by: &'static str,
    datasets: Vec<LockDataset>,
}

#[derive(Debug, Serialize)]
struct LockDataset {
    id: String,
    name: String,
    doi: String,
    download: LockDownload,
    inputs: DatasetInputs,
    materialized: Vec<LockMaterialized>,
}

#[derive(Debug, Clone, Serialize)]
struct LockDownload {
    url: String,
    mode: String,
    cached: bool,
    path: String,
    sha256: String,
}

#[derive(Debug, Clone, Serialize)]
struct DatasetInputs {
    #[serde(skip_serializing_if = "Option::is_none")]
    bkgonly_filename: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    patchset_filename: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    available_patch_names: Vec<String>,
}

#[derive(Debug, Serialize)]
struct LockMaterialized {
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    patch_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    patch_name: Option<String>,
    path: String,
    sha256: String,
}

#[derive(Debug, Clone, Default, Serialize)]
struct DatasetTimings {
    total_s: f64,
    archive_prepare_s: f64,
    download_s: f64,
    extract_archive_s: f64,
    extract_nested_archives_s: f64,
    inspect_inputs_s: f64,
    materialize_bkgonly_s: f64,
    materialize_patches_s: f64,
    materialize_total_s: f64,
}

fn default_true() -> bool {
    true
}

pub fn cmd_import_hepdata(
    manifest_path: Option<&PathBuf>,
    list: bool,
    list_patches: bool,
    datasets: &[String],
    doi: Option<&str>,
    dataset_id: Option<&str>,
    display_name: Option<&str>,
    bkgonly_filename: Option<&str>,
    patchset_filename: Option<&str>,
    patches: &[String],
    out_dir: &PathBuf,
    cache_dir: Option<&PathBuf>,
    lock_path: Option<&PathBuf>,
    clean: bool,
    offline: bool,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    if list && list_patches {
        anyhow::bail!("--list and --list-patches cannot be combined");
    }
    let direct_mode = direct_mode_requested(
        doi,
        dataset_id,
        display_name,
        bkgonly_filename,
        patchset_filename,
        patches,
    );
    if list_patches {
        if !direct_mode {
            anyhow::bail!("--list-patches requires direct DOI import flags");
        }
        if manifest_path.is_some() {
            anyhow::bail!("--manifest cannot be combined with direct DOI import flags");
        }
        if !datasets.is_empty() {
            anyhow::bail!("--dataset cannot be combined with direct DOI import flags");
        }
        let dataset = build_direct_dataset(
            doi,
            dataset_id,
            display_name,
            bkgonly_filename,
            patchset_filename,
            patches,
        )?;
        let cache_dir = cache_dir.cloned().unwrap_or_else(|| out_dir.join("_cache"));
        return emit_direct_patch_catalog(&dataset, &cache_dir, offline, bundle);
    }
    let (manifest_source, selected) = if direct_mode {
        if list {
            anyhow::bail!("--list cannot be combined with direct DOI import flags");
        }
        if manifest_path.is_some() {
            anyhow::bail!("--manifest cannot be combined with direct DOI import flags");
        }
        if !datasets.is_empty() {
            anyhow::bail!("--dataset cannot be combined with direct DOI import flags");
        }
        (
            "direct".to_string(),
            vec![build_direct_dataset(
                doi,
                dataset_id,
                display_name,
                bkgonly_filename,
                patchset_filename,
                patches,
            )?],
        )
    } else {
        let manifest_source = manifest_path
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "embedded".to_string());
        let manifest = load_manifest(manifest_path)?;
        let selected = select_datasets(&manifest.datasets, datasets)?;
        if list {
            return emit_catalog(manifest_path, &manifest_source, &selected, bundle);
        }
        (manifest_source, selected)
    };

    let cache_dir = cache_dir.cloned().unwrap_or_else(|| out_dir.join("_cache"));
    let lock_path = lock_path.cloned().unwrap_or_else(|| out_dir.join("workspaces.lock.json"));

    if clean {
        let _ = std::fs::remove_dir_all(out_dir);
        if cache_dir != *out_dir {
            let _ = std::fs::remove_dir_all(&cache_dir);
        }
        let _ = std::fs::remove_file(&lock_path);
    }

    std::fs::create_dir_all(out_dir)?;
    std::fs::create_dir_all(&cache_dir)?;
    if let Some(parent) = lock_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }

    let mut lockfile = Lockfile {
        schema_version: HEPDATA_LOCK_SCHEMA_V1,
        source_mode: if direct_mode { "direct_doi" } else { "curated" },
        generated_by: "nextstat import hepdata",
        datasets: Vec::new(),
    };
    let mut summary = MaterializeSummary {
        schema_version: HEPDATA_IMPORT_SCHEMA_V1,
        mode: "materialize",
        source: "hepdata",
        source_mode: if direct_mode { "direct_doi" } else { "curated" },
        manifest: manifest_source.clone(),
        out_dir: out_dir.display().to_string(),
        cache_dir: cache_dir.display().to_string(),
        lock: lock_path.display().to_string(),
        datasets: Vec::new(),
    };

    for dataset in &selected {
        let imported = import_dataset(dataset, out_dir, &cache_dir, offline)?;
        lockfile.datasets.push(imported.lock_dataset);
        summary.datasets.push(imported.summary_dataset);
    }

    write_json_pretty(&lock_path, &lockfile)?;

    let summary_json = serde_json::to_value(&summary)?;
    if let Some(bundle_dir) = bundle {
        let bundle_input = stage_bundle_input(manifest_path, direct_mode, out_dir, &selected)?;
        crate::report::write_bundle(
            bundle_dir,
            "import_hepdata",
            materialize_bundle_request(
                &manifest_source,
                direct_mode,
                datasets,
                doi,
                dataset_id,
                display_name,
                bkgonly_filename,
                patchset_filename,
                patches,
                out_dir,
                &cache_dir,
                &lock_path,
                clean,
                offline,
                &selected,
            ),
            &bundle_input,
            &summary_json,
            false,
        )?;
    }

    println!("{}", serde_json::to_string_pretty(&summary_json)?);
    Ok(())
}

fn emit_catalog(
    manifest_path: Option<&PathBuf>,
    manifest_source: &str,
    datasets: &[ManifestDataset],
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let summary = CatalogSummary {
        schema_version: HEPDATA_IMPORT_SCHEMA_V1,
        mode: "catalog",
        source: "hepdata",
        source_mode: "curated",
        manifest: manifest_source.to_string(),
        datasets: datasets.iter().map(|dataset| catalog_dataset(dataset, None, None)).collect(),
    };

    let bundle_input = if let Some(path) = manifest_path {
        path.clone()
    } else {
        let embedded_path = std::env::temp_dir().join(format!(
            "nextstat_hepdata_manifest_{}_{}.json",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        if bundle.is_some() {
            std::fs::write(&embedded_path, EMBEDDED_MANIFEST_JSON)?;
        }
        embedded_path
    };

    let summary_json = serde_json::to_value(&summary)?;
    if let Some(bundle_dir) = bundle {
        crate::report::write_bundle(
            bundle_dir,
            "import_hepdata_catalog",
            serde_json::json!({
                "manifest": manifest_source,
                "dataset": datasets.iter().map(|dataset| dataset.id.clone()).collect::<Vec<_>>(),
                "list": true,
            }),
            if manifest_path.is_some() || bundle_input.exists() {
                &bundle_input
            } else {
                anyhow::bail!("embedded manifest staging file missing for hepdata catalog bundle");
            },
            &summary_json,
            false,
        )?;
    }

    println!("{}", serde_json::to_string_pretty(&summary_json)?);
    Ok(())
}

fn emit_direct_patch_catalog(
    dataset: &ManifestDataset,
    cache_dir: &Path,
    offline: bool,
    bundle: Option<&PathBuf>,
) -> Result<()> {
    let total_started_at = Instant::now();
    let prepared = prepare_dataset_archive(dataset, cache_dir, offline)?;
    let inspect_started_at = Instant::now();
    let inputs = inspect_dataset_inputs(&prepared.extracted_dir, &dataset.materialize, true, true)?;
    let mut timings = prepared.timings.clone();
    timings.inspect_inputs_s = elapsed_s(inspect_started_at);
    timings.total_s = elapsed_s(total_started_at);
    let discovered_patches = inputs
        .available_patch_names
        .iter()
        .map(|patch_name| CatalogPatch {
            id: patch_name.clone(),
            patch_name: Some(patch_name.clone()),
            bkgonly_filename: None,
            patchset_filename: None,
        })
        .collect();
    let materialize = CatalogMaterialize {
        bkgonly: dataset.materialize.bkgonly,
        bkgonly_filename: inputs.bkgonly_filename.clone(),
        patchset_filename: inputs.patchset_filename.clone(),
        patches: discovered_patches,
    };
    let summary = CatalogSummary {
        schema_version: HEPDATA_IMPORT_SCHEMA_V1,
        mode: "catalog",
        source: "hepdata",
        source_mode: "direct_doi",
        manifest: "direct".to_string(),
        datasets: vec![CatalogDataset {
            id: dataset.id.clone(),
            name: dataset.name.clone().unwrap_or_else(|| dataset.id.clone()),
            doi: dataset.doi.clone(),
            download: Some(prepared.download.clone()),
            inputs: Some(inputs),
            timings: Some(timings),
            materialize,
        }],
    };

    let summary_json = serde_json::to_value(&summary)?;
    if let Some(bundle_dir) = bundle {
        let bundle_input =
            stage_bundle_input(None, true, cache_dir, std::slice::from_ref(dataset))?;
        crate::report::write_bundle(
            bundle_dir,
            "import_hepdata_patch_catalog",
            serde_json::json!({
            "manifest": "direct",
            "source_mode": "direct_doi",
            "dataset": [dataset.id.clone()],
            "doi": dataset.doi.clone(),
            "list_patches": true,
            "cache_dir": cache_dir,
            "offline": offline,
            }),
            &bundle_input,
            &summary_json,
            false,
        )?;
    }

    println!("{}", serde_json::to_string_pretty(&summary_json)?);
    Ok(())
}

struct ImportedDataset {
    summary_dataset: SummaryDataset,
    lock_dataset: LockDataset,
}

struct PreparedDatasetArchive {
    extracted_dir: PathBuf,
    download: LockDownload,
    timings: DatasetTimings,
}

fn elapsed_s(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64()
}

fn load_manifest(path: Option<&PathBuf>) -> Result<ManifestFile> {
    let raw = if let Some(path) = path {
        std::fs::read_to_string(path)
            .with_context(|| format!("failed to read HEPData manifest: {}", path.display()))?
    } else {
        EMBEDDED_MANIFEST_JSON.to_string()
    };
    serde_json::from_str(&raw).context("failed to parse HEPData manifest JSON")
}

fn direct_mode_requested(
    doi: Option<&str>,
    dataset_id: Option<&str>,
    display_name: Option<&str>,
    bkgonly_filename: Option<&str>,
    patchset_filename: Option<&str>,
    patches: &[String],
) -> bool {
    doi.is_some()
        || dataset_id.is_some()
        || display_name.is_some()
        || bkgonly_filename.is_some()
        || patchset_filename.is_some()
        || !patches.is_empty()
}

fn build_direct_dataset(
    doi: Option<&str>,
    dataset_id: Option<&str>,
    display_name: Option<&str>,
    bkgonly_filename: Option<&str>,
    patchset_filename: Option<&str>,
    patches: &[String],
) -> Result<ManifestDataset> {
    let doi = required_direct_value("--doi", doi)?;
    let dataset_id = required_direct_value("--dataset-id", dataset_id)?;
    let display_name = optional_direct_value("--display-name", display_name)?;
    let bkgonly_filename = optional_direct_value("--bkgonly-filename", bkgonly_filename)?;
    let patchset_filename = optional_direct_value("--patchset-filename", patchset_filename)?;

    let mut seen_patch_names = BTreeSet::new();
    let mut materialized_patches = Vec::with_capacity(patches.len());
    for patch in patches {
        let (patch_id, patch_name) = parse_direct_patch_arg(patch)?;
        if !seen_patch_names.insert(patch_id.clone()) {
            anyhow::bail!("duplicate --patch id: {}", patch_id);
        }
        materialized_patches.push(PatchMaterialization {
            id: Some(patch_id),
            patch_name,
            bkgonly_filename: None,
            patchset_filename: None,
        });
    }

    Ok(ManifestDataset {
        id: dataset_id.clone(),
        name: Some(display_name.unwrap_or_else(|| dataset_id.clone())),
        doi,
        materialize: MaterializeSpec {
            bkgonly: true,
            patches: materialized_patches,
            bkgonly_filename,
            patchset_filename,
        },
    })
}

fn required_direct_value(flag: &str, value: Option<&str>) -> Result<String> {
    match value {
        Some(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                anyhow::bail!("{flag} must be non-empty");
            }
            Ok(trimmed.to_string())
        }
        None => anyhow::bail!("{flag} is required when using direct DOI import mode"),
    }
}

fn optional_direct_value(flag: &str, value: Option<&str>) -> Result<Option<String>> {
    match value {
        Some(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                anyhow::bail!("{flag} must be non-empty when provided");
            }
            Ok(Some(trimmed.to_string()))
        }
        None => Ok(None),
    }
}

fn is_safe_direct_patch_token(value: &str) -> bool {
    value.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

fn parse_direct_patch_arg(raw: &str) -> Result<(String, Option<String>)> {
    let raw = raw.trim();
    if raw.is_empty() {
        anyhow::bail!("--patch values must be non-empty");
    }

    let (patch_id, patch_name) = match raw.split_once('=') {
        Some((patch_id, patch_name)) => (patch_id.trim(), Some(patch_name.trim())),
        None => (raw, None),
    };
    if patch_id.is_empty() {
        anyhow::bail!("--patch id must be non-empty");
    }
    if !is_safe_direct_patch_token(patch_id) {
        anyhow::bail!(
            "--patch id '{}' contains unsupported characters; use only [A-Za-z0-9._-]",
            patch_id
        );
    }

    let patch_name = match patch_name {
        Some("") => {
            anyhow::bail!("--patch patch-name must be non-empty when using <id>=<patch_name>")
        }
        Some(patch_name) => Some(patch_name.to_string()),
        None => None,
    };
    Ok((patch_id.to_string(), patch_name))
}

fn stage_bundle_input(
    manifest_path: Option<&PathBuf>,
    direct_mode: bool,
    out_dir: &Path,
    selected: &[ManifestDataset],
) -> Result<PathBuf> {
    if let Some(path) = manifest_path {
        return Ok(path.clone());
    }

    let staged_path = if direct_mode {
        out_dir.join(".hepdata_manifest.direct.json")
    } else {
        out_dir.join(".hepdata_manifest.embedded.json")
    };
    if direct_mode {
        write_json_pretty(&staged_path, &ManifestFile { datasets: selected.to_vec() })?;
    } else {
        std::fs::write(&staged_path, EMBEDDED_MANIFEST_JSON)?;
    }
    Ok(staged_path)
}

fn materialize_bundle_request(
    manifest_source: &str,
    direct_mode: bool,
    requested_datasets: &[String],
    doi: Option<&str>,
    dataset_id: Option<&str>,
    display_name: Option<&str>,
    bkgonly_filename: Option<&str>,
    patchset_filename: Option<&str>,
    patches: &[String],
    out_dir: &Path,
    cache_dir: &Path,
    lock_path: &Path,
    clean: bool,
    offline: bool,
    selected: &[ManifestDataset],
) -> serde_json::Value {
    let mut request = serde_json::json!({
        "manifest": manifest_source,
        "dataset": requested_datasets,
        "out_dir": out_dir,
        "cache_dir": cache_dir,
        "lock": lock_path,
        "clean": clean,
        "offline": offline,
    });
    if direct_mode {
        request["source_mode"] = serde_json::json!("direct_doi");
        request["dataset"] = serde_json::json!(
            selected.iter().map(|dataset| dataset.id.clone()).collect::<Vec<_>>()
        );
        request["doi"] = serde_json::json!(doi);
        request["dataset_id"] = serde_json::json!(dataset_id);
        request["display_name"] =
            serde_json::json!(display_name.map(str::trim).filter(|value| !value.is_empty()));
        request["bkgonly_filename"] = serde_json::json!(bkgonly_filename);
        request["patchset_filename"] = serde_json::json!(patchset_filename);
        request["patch"] = serde_json::json!(patches);
    }
    request
}

fn catalog_dataset(
    dataset: &ManifestDataset,
    download: Option<LockDownload>,
    inputs: Option<DatasetInputs>,
) -> CatalogDataset {
    CatalogDataset {
        id: dataset.id.clone(),
        name: dataset.name.clone().unwrap_or_else(|| dataset.id.clone()),
        doi: dataset.doi.clone(),
        download,
        inputs,
        timings: None,
        materialize: CatalogMaterialize {
            bkgonly: dataset.materialize.bkgonly,
            bkgonly_filename: dataset.materialize.bkgonly_filename.clone(),
            patchset_filename: dataset.materialize.patchset_filename.clone(),
            patches: dataset
                .materialize
                .patches
                .iter()
                .map(|patch| CatalogPatch {
                    id: patch.id.clone().unwrap_or_else(|| "patch".to_string()),
                    patch_name: patch.patch_name.clone(),
                    bkgonly_filename: patch.bkgonly_filename.clone(),
                    patchset_filename: patch.patchset_filename.clone(),
                })
                .collect(),
        },
    }
}

fn select_datasets(all: &[ManifestDataset], requested: &[String]) -> Result<Vec<ManifestDataset>> {
    if requested.is_empty() {
        return Ok(all.to_vec());
    }
    let mut selected = Vec::new();
    for dataset in all {
        if requested.iter().any(|id| id == &dataset.id) {
            selected.push(dataset.clone());
        }
    }
    let missing: Vec<String> = requested
        .iter()
        .filter(|id| !selected.iter().any(|dataset| &dataset.id == *id))
        .cloned()
        .collect();
    if !missing.is_empty() {
        anyhow::bail!("unknown HEPData dataset id(s): {}", missing.join(", "));
    }
    Ok(selected)
}

fn prepare_dataset_archive(
    dataset: &ManifestDataset,
    cache_dir: &Path,
    offline: bool,
) -> Result<PreparedDatasetArchive> {
    let archive_started_at = Instant::now();
    let slug = dataset.id.replace('/', "_");
    let ds_cache = cache_dir.join(&slug);
    let archive_path = ds_cache.join("download");
    let extracted_dir = ds_cache.join("extracted");
    let mut timings = DatasetTimings::default();

    std::fs::create_dir_all(&ds_cache)?;

    let cached = archive_path.exists() && std::fs::metadata(&archive_path)?.len() > 0;
    if !cached {
        if offline {
            anyhow::bail!(
                "offline mode requested, but cached download is missing for {} at {}",
                dataset.id,
                archive_path.display()
            );
        }
        let download_started_at = Instant::now();
        download_archive(&dataset.doi, &archive_path)?;
        timings.download_s = elapsed_s(download_started_at);
    }

    if extracted_dir.exists() {
        std::fs::remove_dir_all(&extracted_dir)?;
    }
    let extract_started_at = Instant::now();
    extract_archive(&archive_path, &extracted_dir)?;
    timings.extract_archive_s = elapsed_s(extract_started_at);
    let nested_started_at = Instant::now();
    let _ = extract_nested_archives(&extracted_dir)?;
    timings.extract_nested_archives_s = elapsed_s(nested_started_at);
    timings.archive_prepare_s = elapsed_s(archive_started_at);

    Ok(PreparedDatasetArchive {
        extracted_dir,
        download: LockDownload {
            url: dataset.doi.clone(),
            mode: if cached { "cached".to_string() } else { "network".to_string() },
            cached,
            path: archive_path.display().to_string(),
            sha256: sha256_file(&archive_path)?,
        },
        timings,
    })
}

fn import_dataset(
    dataset: &ManifestDataset,
    out_dir: &Path,
    cache_dir: &Path,
    offline: bool,
) -> Result<ImportedDataset> {
    let total_started_at = Instant::now();
    let slug = dataset.id.replace('/', "_");
    let ds_out_dir = out_dir.join(&slug);

    std::fs::create_dir_all(&ds_out_dir)?;

    let prepared = prepare_dataset_archive(dataset, cache_dir, offline)?;
    let inspect_started_at = Instant::now();
    let inputs = inspect_dataset_inputs(
        &prepared.extracted_dir,
        &dataset.materialize,
        dataset.materialize.bkgonly || !dataset.materialize.patches.is_empty(),
        !dataset.materialize.patches.is_empty(),
    )?;
    let mut timings = prepared.timings.clone();
    timings.inspect_inputs_s = elapsed_s(inspect_started_at);

    let mut summary_materialized = Vec::new();
    let mut lock_materialized = Vec::new();

    if dataset.materialize.bkgonly {
        let bkgonly_started_at = Instant::now();
        let path = materialize_bkgonly(
            &prepared.extracted_dir,
            &ds_out_dir,
            dataset.materialize.bkgonly_filename.as_deref(),
        )?;
        timings.materialize_bkgonly_s = elapsed_s(bkgonly_started_at);
        summary_materialized.push(SummaryMaterialized {
            kind: "bkgonly".to_string(),
            patch_id: None,
            patch_name: None,
            path: path.display().to_string(),
        });
        lock_materialized.push(LockMaterialized {
            kind: "bkgonly".to_string(),
            patch_id: None,
            patch_name: None,
            path: path.display().to_string(),
            sha256: sha256_file(&path)?,
        });
    }

    let patches_started_at = Instant::now();
    for patch in &dataset.materialize.patches {
        let patch_id = patch.id.clone().unwrap_or_else(|| "patch".to_string());
        let path =
            materialize_patch(&prepared.extracted_dir, &ds_out_dir, patch, &dataset.materialize)?;
        summary_materialized.push(SummaryMaterialized {
            kind: "patched".to_string(),
            patch_id: Some(patch_id.clone()),
            patch_name: patch.patch_name.clone(),
            path: path.display().to_string(),
        });
        lock_materialized.push(LockMaterialized {
            kind: "patched".to_string(),
            patch_id: Some(patch_id),
            patch_name: patch.patch_name.clone(),
            path: path.display().to_string(),
            sha256: sha256_file(&path)?,
        });
    }
    if !dataset.materialize.patches.is_empty() {
        timings.materialize_patches_s = elapsed_s(patches_started_at);
    }
    timings.materialize_total_s = timings.materialize_bkgonly_s + timings.materialize_patches_s;
    timings.total_s = elapsed_s(total_started_at);

    Ok(ImportedDataset {
        summary_dataset: SummaryDataset {
            id: dataset.id.clone(),
            name: dataset.name.clone().unwrap_or_else(|| dataset.id.clone()),
            doi: dataset.doi.clone(),
            download: prepared.download.clone(),
            inputs: inputs.clone(),
            timings: Some(timings),
            materialized: summary_materialized,
        },
        lock_dataset: LockDataset {
            id: dataset.id.clone(),
            name: dataset.name.clone().unwrap_or_else(|| dataset.id.clone()),
            doi: dataset.doi.clone(),
            download: prepared.download,
            inputs,
            materialized: lock_materialized,
        },
    })
}

fn inspect_dataset_inputs(
    extracted_dir: &Path,
    defaults: &MaterializeSpec,
    resolve_bkgonly: bool,
    resolve_patchset: bool,
) -> Result<DatasetInputs> {
    let bkgonly_filename = if resolve_bkgonly {
        Some(path_filename_string(&find_optional(
            extracted_dir,
            defaults.bkgonly_filename.as_deref(),
            "BkgOnly.json",
        )?)?)
    } else {
        None
    };

    let (patchset_filename, available_patch_names) = if resolve_patchset {
        let path =
            find_optional(extracted_dir, defaults.patchset_filename.as_deref(), "patchset.json")?;
        let patchset = read_patchset(&path)?;
        let patch_names = patchset.patch_names().into_iter().map(ToString::to_string).collect();
        (Some(path_filename_string(&path)?), patch_names)
    } else {
        (None, Vec::new())
    };

    Ok(DatasetInputs { bkgonly_filename, patchset_filename, available_patch_names })
}

fn materialize_bkgonly(
    extracted_dir: &Path,
    out_dir: &Path,
    bkgonly_filename: Option<&str>,
) -> Result<PathBuf> {
    let src = find_optional(extracted_dir, bkgonly_filename, "BkgOnly.json")?;
    let dst = out_dir.join("BkgOnly.json");
    std::fs::copy(&src, &dst)
        .with_context(|| format!("failed to copy {} -> {}", src.display(), dst.display()))?;
    Ok(dst)
}

fn materialize_patch(
    extracted_dir: &Path,
    out_dir: &Path,
    patch: &PatchMaterialization,
    defaults: &MaterializeSpec,
) -> Result<PathBuf> {
    let bkgonly_filename =
        patch.bkgonly_filename.as_deref().or(defaults.bkgonly_filename.as_deref());
    let patchset_filename =
        patch.patchset_filename.as_deref().or(defaults.patchset_filename.as_deref());

    let bkgonly_path = find_optional(extracted_dir, bkgonly_filename, "BkgOnly.json")?;
    let patchset_path = find_optional(extracted_dir, patchset_filename, "patchset.json")?;

    let base_json: serde_json::Value = serde_json::from_slice(&std::fs::read(&bkgonly_path)?)
        .with_context(|| format!("failed to parse JSON workspace: {}", bkgonly_path.display()))?;
    let patchset = read_patchset(&patchset_path)?;
    let patched = patchset.apply_to_value(&base_json, patch.patch_name.as_deref())?;

    let patch_id = patch.id.as_deref().unwrap_or("patch");
    let dst = out_dir.join(format!("patched__{patch_id}.json"));
    write_json_pretty(&dst, &patched)?;
    Ok(dst)
}

fn download_archive(url: &str, dest: &Path) -> Result<()> {
    if let Some(parent) = dest.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let response = ureq::get(url)
        .set("User-Agent", "nextstat-hepdata-import/1.0")
        .set("Accept", "*/*")
        .call()
        .with_context(|| format!("failed to download HEPData bundle: {url}"))?;
    let mut reader = response.into_reader();
    let mut file = File::create(dest)
        .with_context(|| format!("failed to create archive file: {}", dest.display()))?;
    std::io::copy(&mut reader, &mut file)
        .with_context(|| format!("failed to write archive file: {}", dest.display()))?;
    file.flush()?;
    Ok(())
}

fn extract_archive(archive_path: &Path, dest: &Path) -> Result<()> {
    std::fs::create_dir_all(dest)?;
    if try_extract_zip(archive_path, dest)? {
        return Ok(());
    }
    if try_extract_tar_gz(archive_path, dest)? {
        return Ok(());
    }
    if try_extract_tar(archive_path, dest)? {
        return Ok(());
    }
    anyhow::bail!("unknown archive format: {}", archive_path.display())
}

fn try_extract_zip(archive_path: &Path, dest: &Path) -> Result<bool> {
    let file = File::open(archive_path)?;
    let mut archive = match ZipArchive::new(file) {
        Ok(archive) => archive,
        Err(_) => return Ok(false),
    };

    for i in 0..archive.len() {
        let mut member = archive.by_index(i)?;
        let rel = member.enclosed_name().ok_or_else(|| {
            anyhow::anyhow!("refusing to extract zip member outside destination: {}", member.name())
        })?;
        let out = dest.join(rel);
        if member.is_dir() {
            std::fs::create_dir_all(&out)?;
            continue;
        }
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut out_file = File::create(&out)?;
        std::io::copy(&mut member, &mut out_file)?;
    }
    Ok(true)
}

fn try_extract_tar_gz(archive_path: &Path, dest: &Path) -> Result<bool> {
    let file = File::open(archive_path)?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    match unpack_tar(&mut archive, dest) {
        Ok(()) => Ok(true),
        Err(_) => Ok(false),
    }
}

fn try_extract_tar(archive_path: &Path, dest: &Path) -> Result<bool> {
    let file = File::open(archive_path)?;
    let mut archive = Archive::new(file);
    match unpack_tar(&mut archive, dest) {
        Ok(()) => Ok(true),
        Err(_) => Ok(false),
    }
}

fn unpack_tar<R: Read>(archive: &mut Archive<R>, dest: &Path) -> Result<()> {
    for entry in archive.entries()? {
        let mut entry = entry?;
        let rel = sanitize_relative_path(&entry.path()?)?;
        let out = dest.join(rel);
        if entry.header().entry_type().is_dir() {
            std::fs::create_dir_all(&out)?;
            continue;
        }
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        entry.unpack(&out)?;
    }
    Ok(())
}

fn extract_nested_archives(root: &Path) -> Result<usize> {
    let mut files = Vec::new();
    collect_files(root, &mut files)?;
    let mut extracted = 0usize;
    for file in files {
        if !should_extract_nested(&file) {
            continue;
        }
        if sibling_already_exists(&file) {
            continue;
        }
        let parent = file.parent().unwrap_or(root);
        if extract_archive(&file, parent).is_ok() {
            extracted += 1;
        }
    }
    Ok(extracted)
}

fn collect_files(root: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, out)?;
        } else if path.is_file() {
            out.push(path);
        }
    }
    Ok(())
}

fn should_extract_nested(path: &Path) -> bool {
    let name = path.file_name().and_then(OsStr::to_str).unwrap_or_default();
    name.ends_with(".json.tgz")
        || name.ends_with(".json.tar.gz")
        || name.ends_with(".json.tar")
        || name.ends_with(".tgz")
}

fn sibling_already_exists(path: &Path) -> bool {
    let name = path.file_name().and_then(OsStr::to_str).unwrap_or_default();
    if name.ends_with(".json.tgz") {
        return path.with_extension("").exists();
    }
    if name.ends_with(".tgz") || name.ends_with(".tar") || name.ends_with(".gz") {
        return path.with_extension("").exists();
    }
    false
}

fn sanitize_relative_path(path: &Path) -> Result<PathBuf> {
    let mut clean = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => clean.push(part),
            Component::CurDir => {}
            Component::Prefix(_) | Component::RootDir | Component::ParentDir => {
                anyhow::bail!("refusing to extract path outside destination: {}", path.display())
            }
        }
    }
    Ok(clean)
}

fn find_optional(root: &Path, filename: Option<&str>, default: &str) -> Result<PathBuf> {
    if let Some(filename) = filename { find_one(root, filename) } else { find_one(root, default) }
}

fn find_one(root: &Path, filename: &str) -> Result<PathBuf> {
    let mut matches = Vec::new();
    find_named_file(root, filename, &mut matches)?;
    if matches.is_empty() {
        anyhow::bail!("did not find '{filename}' under {}", root.display());
    }
    matches.sort_by_key(|path| path.components().count());
    Ok(matches.remove(0))
}

fn find_named_file(root: &Path, filename: &str, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            find_named_file(&path, filename, out)?;
        } else if path.file_name().and_then(OsStr::to_str) == Some(filename) {
            out.push(path);
        }
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .with_context(|| format!("failed to open file for sha256: {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn read_patchset(path: &Path) -> Result<ns_translate::pyhf::PatchSet> {
    serde_json::from_slice(&std::fs::read(path)?)
        .with_context(|| format!("failed to parse patchset JSON: {}", path.display()))
}

fn path_filename_string(path: &Path) -> Result<String> {
    path.file_name()
        .and_then(OsStr::to_str)
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("path has no terminal filename: {}", path.display()))
}

fn write_json_pretty<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(value)?)
        .with_context(|| format!("failed to write JSON file: {}", path.display()))?;
    Ok(())
}
