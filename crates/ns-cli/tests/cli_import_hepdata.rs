use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

fn bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_nextstat"))
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn fixture_path(rel: &str) -> PathBuf {
    repo_root().join(rel)
}

fn tmp_path(name: &str) -> PathBuf {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!("nextstat_cli_{}_{}_{}", std::process::id(), nanos, name));
    p
}

fn run(args: &[&str]) -> Output {
    Command::new(bin_path())
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?} {:?}: {}", bin_path(), args, e))
}

fn seed_cached_download(cache_dir: &Path, dataset_id: &str, fixture_rel: &str) -> PathBuf {
    let slug = dataset_id.replace('/', "_");
    let ds_cache = cache_dir.join(slug);
    std::fs::create_dir_all(&ds_cache).unwrap();
    let archive_path = ds_cache.join("download");
    std::fs::copy(fixture_path(fixture_rel), &archive_path).unwrap();
    archive_path
}

fn start_static_http_server(bytes: Vec<u8>) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind local HTTP listener");
    let addr = listener.local_addr().expect("local addr");
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept HTTP request");
        let mut request = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            let n = stream.read(&mut buf).expect("read HTTP request");
            if n == 0 {
                break;
            }
            request.extend_from_slice(&buf[..n]);
            if request.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
        assert!(
            request.starts_with(b"GET /download HTTP/1.1")
                || request.starts_with(b"GET /download HTTP/1.0"),
            "unexpected HTTP request: {}",
            String::from_utf8_lossy(&request)
        );
        let headers = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/octet-stream\r\nConnection: close\r\n\r\n",
            bytes.len()
        );
        stream.write_all(headers.as_bytes()).expect("write HTTP response headers");
        stream.write_all(&bytes).expect("write HTTP response body");
        stream.flush().expect("flush HTTP response");
    });
    (format!("http://127.0.0.1:{}/download", addr.port()), handle)
}

#[test]
fn import_hepdata_offline_materializes_selected_dataset_from_cached_archive() {
    let dataset_id = "hepdata.90607.v3.r3";
    let cache_dir = tmp_path("hepdata_cache");
    let out_dir = tmp_path("hepdata_out");
    let lock_path = tmp_path("hepdata_lock.json");

    seed_cached_download(
        &cache_dir,
        dataset_id,
        "tests/hepdata/_cache/hepdata.90607.v3.r3/download",
    );

    let out = run(&[
        "import",
        "hepdata",
        "--dataset",
        dataset_id,
        "--cache-dir",
        cache_dir.to_string_lossy().as_ref(),
        "--out-dir",
        out_dir.to_string_lossy().as_ref(),
        "--lock",
        lock_path.to_string_lossy().as_ref(),
        "--offline",
    ]);
    assert!(
        out.status.success(),
        "import hepdata should succeed, stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let summary: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be JSON summary");
    assert_eq!(
        summary.get("schema_version").and_then(|v| v.as_str()),
        Some("nextstat.hepdata_import.v1")
    );
    assert_eq!(summary.get("mode").and_then(|v| v.as_str()), Some("materialize"));
    assert_eq!(summary.get("source").and_then(|v| v.as_str()), Some("hepdata"));
    assert_eq!(summary.get("source_mode").and_then(|v| v.as_str()), Some("curated"));
    assert_eq!(summary.get("manifest").and_then(|v| v.as_str()), Some("embedded"));

    let datasets = summary.get("datasets").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    assert_eq!(datasets.len(), 1, "expected one dataset in summary");
    assert_eq!(datasets[0].get("id").and_then(|v| v.as_str()), Some(dataset_id));
    assert_eq!(
        datasets[0].get("download").and_then(|v| v.get("mode")).and_then(|v| v.as_str()),
        Some("cached")
    );
    assert_eq!(
        datasets[0].get("inputs").and_then(|v| v.get("bkgonly_filename")).and_then(|v| v.as_str()),
        Some("BkgOnly.json")
    );
    assert_eq!(
        datasets[0].get("inputs").and_then(|v| v.get("patchset_filename")).and_then(|v| v.as_str()),
        Some("patchset.json")
    );
    let available_patch_names = datasets[0]
        .get("inputs")
        .and_then(|v| v.get("available_patch_names"))
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    assert!(
        available_patch_names.iter().any(|entry| entry.as_str() == Some("C1N2_Wh_hbb_1000_0")),
        "expected patch provenance in summary: {available_patch_names:?}"
    );
    let timings = datasets[0]
        .get("timings")
        .and_then(|v| v.as_object())
        .cloned()
        .expect("expected timings block in materialize summary");
    assert_eq!(
        timings.get("download_s").and_then(|v| v.as_f64()),
        Some(0.0),
        "offline cached import should not report network download time"
    );
    assert!(
        timings.get("materialize_total_s").and_then(|v| v.as_f64()).unwrap_or_default() > 0.0,
        "expected non-zero materialize timing in summary: {timings:?}"
    );

    let ds_out = out_dir.join(dataset_id);
    assert!(ds_out.join("BkgOnly.json").exists(), "missing BkgOnly.json");
    assert!(ds_out.join("patched__first_patch.json").exists(), "missing patched workspace");
    assert!(lock_path.exists(), "missing lockfile: {}", lock_path.display());

    let lock: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&lock_path).unwrap()).expect("lock should be JSON");
    assert_eq!(
        lock.get("schema_version").and_then(|v| v.as_str()),
        Some("nextstat.hepdata_lock.v1")
    );
    assert_eq!(lock.get("source_mode").and_then(|v| v.as_str()), Some("curated"));
    assert_eq!(lock.get("generated_by").and_then(|v| v.as_str()), Some("nextstat import hepdata"));
    let lock_datasets =
        lock.get("datasets").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    assert_eq!(lock_datasets.len(), 1, "expected one dataset in lockfile");
    assert_eq!(lock_datasets[0].get("id").and_then(|v| v.as_str()), Some(dataset_id));
    assert_eq!(
        lock_datasets[0].get("download").and_then(|v| v.get("mode")).and_then(|v| v.as_str()),
        Some("cached")
    );
    assert_eq!(
        lock_datasets[0]
            .get("inputs")
            .and_then(|v| v.get("patchset_filename"))
            .and_then(|v| v.as_str()),
        Some("patchset.json")
    );

    let _ = std::fs::remove_dir_all(&cache_dir);
    let _ = std::fs::remove_dir_all(&out_dir);
    let _ = std::fs::remove_file(&lock_path);
}

#[test]
fn import_hepdata_offline_errors_when_cached_download_is_missing() {
    let cache_dir = tmp_path("hepdata_cache_missing");
    let out_dir = tmp_path("hepdata_out_missing");
    let dataset_id = "hepdata.116034.v1.r34";

    let out = run(&[
        "import",
        "hepdata",
        "--dataset",
        dataset_id,
        "--cache-dir",
        cache_dir.to_string_lossy().as_ref(),
        "--out-dir",
        out_dir.to_string_lossy().as_ref(),
        "--offline",
    ]);
    assert!(!out.status.success(), "import hepdata should fail without cached download");

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("offline mode requested") && stderr.contains(dataset_id),
        "expected offline cache miss error, got stderr:\n{stderr}"
    );

    let _ = std::fs::remove_dir_all(&cache_dir);
    let _ = std::fs::remove_dir_all(&out_dir);
}

#[test]
fn import_hepdata_list_returns_curated_catalog_json() {
    let out = run(&["import", "hepdata", "--list"]);
    assert!(
        out.status.success(),
        "hepdata --list should succeed, stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let summary: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be JSON catalog");
    assert_eq!(
        summary.get("schema_version").and_then(|v| v.as_str()),
        Some("nextstat.hepdata_import.v1")
    );
    assert_eq!(summary.get("mode").and_then(|v| v.as_str()), Some("catalog"));
    assert_eq!(summary.get("source").and_then(|v| v.as_str()), Some("hepdata"));
    assert_eq!(summary.get("manifest").and_then(|v| v.as_str()), Some("embedded"));

    let datasets = summary.get("datasets").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    assert!(datasets.len() >= 2, "expected embedded catalog datasets, got: {datasets:?}");
    let first = &datasets[0];
    assert!(first.get("id").and_then(|v| v.as_str()).is_some());
    assert!(first.get("name").and_then(|v| v.as_str()).is_some());
    assert!(first.get("doi").and_then(|v| v.as_str()).is_some());
    assert!(
        first.get("materialize").and_then(|v| v.as_object()).is_some(),
        "expected materialize block in catalog dataset"
    );
}

#[test]
fn import_hepdata_list_patches_returns_direct_doi_catalog_json() {
    let dataset_id = "custom.hepdata.90607.v3.r3.catalog";
    let cache_dir = tmp_path("hepdata_direct_catalog_cache");

    seed_cached_download(
        &cache_dir,
        dataset_id,
        "tests/hepdata/_cache/hepdata.90607.v3.r3/download",
    );

    let out = run(&[
        "import",
        "hepdata",
        "--list-patches",
        "--doi",
        "https://doi.org/10.17182/hepdata.90607.v3/r3",
        "--dataset-id",
        dataset_id,
        "--bkgonly-filename",
        "BkgOnly.json",
        "--patchset-filename",
        "patchset.json",
        "--cache-dir",
        cache_dir.to_string_lossy().as_ref(),
        "--offline",
    ]);
    assert!(
        out.status.success(),
        "hepdata --list-patches should succeed, stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let summary: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be JSON patch catalog");
    assert_eq!(
        summary.get("schema_version").and_then(|v| v.as_str()),
        Some("nextstat.hepdata_import.v1")
    );
    assert_eq!(summary.get("mode").and_then(|v| v.as_str()), Some("catalog"));
    assert_eq!(summary.get("source_mode").and_then(|v| v.as_str()), Some("direct_doi"));
    assert_eq!(summary.get("manifest").and_then(|v| v.as_str()), Some("direct"));

    let datasets = summary.get("datasets").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    assert_eq!(datasets.len(), 1, "expected one direct DOI catalog dataset");
    assert_eq!(datasets[0].get("id").and_then(|v| v.as_str()), Some(dataset_id));
    assert_eq!(
        datasets[0].get("download").and_then(|v| v.get("mode")).and_then(|v| v.as_str()),
        Some("cached")
    );
    assert_eq!(
        datasets[0].get("inputs").and_then(|v| v.get("patchset_filename")).and_then(|v| v.as_str()),
        Some("patchset.json")
    );
    let timings = datasets[0]
        .get("timings")
        .and_then(|v| v.as_object())
        .cloned()
        .expect("expected timings block in direct DOI patch catalog");
    assert_eq!(
        timings.get("download_s").and_then(|v| v.as_f64()),
        Some(0.0),
        "cached patch catalog should not report network download time"
    );
    assert!(
        timings.get("inspect_inputs_s").and_then(|v| v.as_f64()).unwrap_or_default() > 0.0,
        "expected archive inspection timing in direct DOI patch catalog: {timings:?}"
    );

    let materialize = datasets[0]
        .get("materialize")
        .and_then(|v| v.as_object())
        .cloned()
        .expect("expected materialize block");
    let patches =
        materialize.get("patches").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    assert!(
        patches.iter().any(|entry| {
            entry.get("id").and_then(|v| v.as_str()) == Some("C1N2_Wh_hbb_1000_0")
                && entry.get("patch_name").and_then(|v| v.as_str()) == Some("C1N2_Wh_hbb_1000_0")
        }),
        "expected discovered patch names in direct DOI catalog: {patches:?}"
    );

    let _ = std::fs::remove_dir_all(&cache_dir);
}

#[test]
fn import_hepdata_direct_doi_requires_dataset_id() {
    let out = run(&["import", "hepdata", "--doi", "https://doi.org/10.17182/hepdata.90607.v3/r3"]);
    assert!(!out.status.success(), "direct DOI import without dataset id should fail");

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--dataset-id is required"),
        "expected explicit dataset-id validation, got stderr:\n{stderr}"
    );
}

#[test]
fn import_hepdata_offline_materializes_direct_doi_dataset_from_cached_archive() {
    let dataset_id = "custom.hepdata.90607.v3.r3";
    let cache_dir = tmp_path("hepdata_direct_cache");
    let out_dir = tmp_path("hepdata_direct_out");
    let lock_path = tmp_path("hepdata_direct_lock.json");

    seed_cached_download(
        &cache_dir,
        dataset_id,
        "tests/hepdata/_cache/hepdata.90607.v3.r3/download",
    );

    let out = run(&[
        "import",
        "hepdata",
        "--doi",
        "https://doi.org/10.17182/hepdata.90607.v3/r3",
        "--dataset-id",
        dataset_id,
        "--display-name",
        "Custom 90607",
        "--bkgonly-filename",
        "BkgOnly.json",
        "--patchset-filename",
        "patchset.json",
        "--patch",
        "first_patch",
        "--cache-dir",
        cache_dir.to_string_lossy().as_ref(),
        "--out-dir",
        out_dir.to_string_lossy().as_ref(),
        "--lock",
        lock_path.to_string_lossy().as_ref(),
        "--offline",
    ]);
    assert!(
        out.status.success(),
        "direct DOI import should succeed, stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let summary: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be JSON summary");
    assert_eq!(
        summary.get("schema_version").and_then(|v| v.as_str()),
        Some("nextstat.hepdata_import.v1")
    );
    assert_eq!(summary.get("mode").and_then(|v| v.as_str()), Some("materialize"));
    assert_eq!(summary.get("source_mode").and_then(|v| v.as_str()), Some("direct_doi"));

    let datasets = summary.get("datasets").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    assert_eq!(datasets.len(), 1, "expected one direct DOI dataset");
    assert_eq!(datasets[0].get("id").and_then(|v| v.as_str()), Some(dataset_id));
    assert_eq!(datasets[0].get("name").and_then(|v| v.as_str()), Some("Custom 90607"));
    assert_eq!(
        datasets[0].get("doi").and_then(|v| v.as_str()),
        Some("https://doi.org/10.17182/hepdata.90607.v3/r3")
    );
    assert_eq!(
        datasets[0].get("download").and_then(|v| v.get("mode")).and_then(|v| v.as_str()),
        Some("cached")
    );
    assert_eq!(
        datasets[0].get("inputs").and_then(|v| v.get("patchset_filename")).and_then(|v| v.as_str()),
        Some("patchset.json")
    );
    let timings = datasets[0]
        .get("timings")
        .and_then(|v| v.as_object())
        .cloned()
        .expect("expected timings block in direct DOI materialize summary");
    assert_eq!(
        timings.get("download_s").and_then(|v| v.as_f64()),
        Some(0.0),
        "cached direct DOI import should not report network download time"
    );
    assert!(
        timings.get("materialize_total_s").and_then(|v| v.as_f64()).unwrap_or_default() > 0.0,
        "expected non-zero materialize timing in direct DOI summary: {timings:?}"
    );

    let ds_out = out_dir.join(dataset_id);
    assert!(ds_out.join("BkgOnly.json").exists(), "missing direct DOI BkgOnly.json");
    assert!(
        ds_out.join("patched__first_patch.json").exists(),
        "missing direct DOI patched workspace"
    );
    assert!(lock_path.exists(), "missing direct DOI lockfile");

    let lock: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&lock_path).unwrap()).expect("lock should be JSON");
    assert_eq!(lock.get("source_mode").and_then(|v| v.as_str()), Some("direct_doi"));

    let _ = std::fs::remove_dir_all(&cache_dir);
    let _ = std::fs::remove_dir_all(&out_dir);
    let _ = std::fs::remove_file(&lock_path);
}

#[test]
fn import_hepdata_offline_materializes_direct_doi_named_patch_from_cached_archive() {
    let dataset_id = "custom.hepdata.90607.v3.r3.named";
    let cache_dir = tmp_path("hepdata_direct_named_cache");
    let out_dir = tmp_path("hepdata_direct_named_out");

    seed_cached_download(
        &cache_dir,
        dataset_id,
        "tests/hepdata/_cache/hepdata.90607.v3.r3/download",
    );

    let out = run(&[
        "import",
        "hepdata",
        "--doi",
        "https://doi.org/10.17182/hepdata.90607.v3/r3",
        "--dataset-id",
        dataset_id,
        "--bkgonly-filename",
        "BkgOnly.json",
        "--patchset-filename",
        "patchset.json",
        "--patch",
        "signal_point=C1N2_Wh_hbb_1000_0",
        "--cache-dir",
        cache_dir.to_string_lossy().as_ref(),
        "--out-dir",
        out_dir.to_string_lossy().as_ref(),
        "--offline",
    ]);
    assert!(
        out.status.success(),
        "direct DOI named patch import should succeed, stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let summary: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be JSON summary");
    let datasets = summary.get("datasets").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    assert_eq!(datasets.len(), 1, "expected one direct DOI dataset");

    let materialized =
        datasets[0].get("materialized").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    assert!(
        materialized.iter().any(|entry| {
            entry.get("patch_id").and_then(|v| v.as_str()) == Some("signal_point")
                && entry.get("patch_name").and_then(|v| v.as_str()) == Some("C1N2_Wh_hbb_1000_0")
        }),
        "expected summary to preserve direct DOI patch id/name mapping: {materialized:?}"
    );

    let ds_out = out_dir.join(dataset_id);
    assert!(
        ds_out.join("patched__signal_point.json").exists(),
        "missing direct DOI named patch workspace"
    );

    let _ = std::fs::remove_dir_all(&cache_dir);
    let _ = std::fs::remove_dir_all(&out_dir);
}

#[test]
fn import_hepdata_downloads_direct_doi_archive_from_local_http_server() {
    let dataset_id = "custom.hepdata.90607.v3.r3.network";
    let cache_dir = tmp_path("hepdata_direct_network_cache");
    let out_dir = tmp_path("hepdata_direct_network_out");
    let lock_path = tmp_path("hepdata_direct_network_lock.json");
    let fixture_bytes =
        std::fs::read(fixture_path("tests/hepdata/_cache/hepdata.90607.v3.r3/download"))
            .expect("read HEPData fixture archive");
    let (doi_url, server) = start_static_http_server(fixture_bytes);

    let out = run(&[
        "import",
        "hepdata",
        "--doi",
        doi_url.as_str(),
        "--dataset-id",
        dataset_id,
        "--display-name",
        "Network 90607",
        "--bkgonly-filename",
        "BkgOnly.json",
        "--patchset-filename",
        "patchset.json",
        "--patch",
        "first_patch",
        "--cache-dir",
        cache_dir.to_string_lossy().as_ref(),
        "--out-dir",
        out_dir.to_string_lossy().as_ref(),
        "--lock",
        lock_path.to_string_lossy().as_ref(),
    ]);
    server.join().expect("local HTTP server should finish cleanly");
    assert!(
        out.status.success(),
        "direct DOI network import should succeed, stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let summary: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be JSON summary");
    assert_eq!(summary.get("source_mode").and_then(|v| v.as_str()), Some("direct_doi"));
    let datasets = summary.get("datasets").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    assert_eq!(datasets.len(), 1, "expected one direct DOI network dataset");
    let timings = datasets[0]
        .get("timings")
        .and_then(|v| v.as_object())
        .cloned()
        .expect("expected timings block in network materialize summary");
    assert!(
        timings.get("download_s").and_then(|v| v.as_f64()).unwrap_or_default() > 0.0,
        "expected non-zero network download timing in summary: {timings:?}"
    );
    assert!(
        timings.get("materialize_total_s").and_then(|v| v.as_f64()).unwrap_or_default() > 0.0,
        "expected non-zero materialize timing in network summary: {timings:?}"
    );

    let datasets = summary.get("datasets").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    assert_eq!(datasets.len(), 1, "expected one direct DOI dataset");
    assert_eq!(
        datasets[0].get("download").and_then(|v| v.get("mode")).and_then(|v| v.as_str()),
        Some("network")
    );
    assert_eq!(
        datasets[0].get("download").and_then(|v| v.get("cached")).and_then(|v| v.as_bool()),
        Some(false)
    );

    let cache_archive = cache_dir.join(dataset_id).join("download");
    assert!(cache_archive.exists(), "expected downloaded archive in cache");
    let ds_out = out_dir.join(dataset_id);
    assert!(ds_out.join("BkgOnly.json").exists(), "missing network BkgOnly.json");
    assert!(ds_out.join("patched__first_patch.json").exists(), "missing network patched workspace");

    let lock: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&lock_path).unwrap()).expect("lock should be JSON");
    let lock_datasets =
        lock.get("datasets").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    assert_eq!(lock_datasets.len(), 1, "expected one dataset in lockfile");
    assert_eq!(
        lock_datasets[0].get("download").and_then(|v| v.get("mode")).and_then(|v| v.as_str()),
        Some("network")
    );

    let _ = std::fs::remove_dir_all(&cache_dir);
    let _ = std::fs::remove_dir_all(&out_dir);
    let _ = std::fs::remove_file(&lock_path);
}
