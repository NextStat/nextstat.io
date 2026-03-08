use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_nextstat"))
}

fn repo_root() -> PathBuf {
    // crates/ns-cli -> repo root
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn fixture_path(rel: &str) -> PathBuf {
    repo_root().join("tests/fixtures").join(rel)
}

fn tmp_path(filename: &str) -> PathBuf {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!("nextstat_cli_{}_{}_{}", std::process::id(), nanos, filename));
    p
}

fn run(args: &[&str]) -> Output {
    Command::new(bin_path())
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?} {:?}: {}", bin_path(), args, e))
}

fn ensure_clean_path(p: &PathBuf) {
    let _ = std::fs::remove_dir_all(p);
    let _ = std::fs::remove_file(p);
}

fn repo_path(rel: &str) -> PathBuf {
    repo_root().join(rel)
}

fn seed_cached_download(cache_dir: &Path, dataset_id: &str, fixture_rel: &str) {
    let slug = dataset_id.replace('/', "_");
    let ds_cache = cache_dir.join(slug);
    std::fs::create_dir_all(&ds_cache).unwrap();
    let archive_path = ds_cache.join("download");
    std::fs::copy(repo_path(fixture_rel), &archive_path).unwrap();
}

#[test]
fn bundle_import_histfactory() {
    let xml = fixture_path("histfactory/combination.xml");
    assert!(xml.exists(), "missing fixture: {}", xml.display());

    let bundle_dir = tmp_path("bundle_import_histfactory");
    ensure_clean_path(&bundle_dir);

    let out = run(&[
        "--bundle",
        bundle_dir.to_string_lossy().as_ref(),
        "import",
        "histfactory",
        "--xml",
        xml.to_string_lossy().as_ref(),
    ]);
    assert!(
        out.status.success(),
        "expected success, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(bundle_dir.join("meta.json").exists());
    assert!(bundle_dir.join("manifest.json").exists());
    assert!(bundle_dir.join("inputs/input.json").exists());
    assert!(bundle_dir.join("outputs/result.json").exists());

    let _ = std::fs::remove_dir_all(&bundle_dir);
}

#[test]
fn bundle_import_trex_config() {
    let root = repo_root();
    let config = root.join("docs/examples/trex_config_ntup_minimal.txt");
    assert!(config.exists(), "missing config: {}", config.display());

    let ws_path = tmp_path("bundle_import_trex_ws.json");
    let bundle_dir = tmp_path("bundle_import_trex_config");
    ensure_clean_path(&ws_path);
    ensure_clean_path(&bundle_dir);

    let out = run(&[
        "--bundle",
        bundle_dir.to_string_lossy().as_ref(),
        "import",
        "trex-config",
        "--config",
        config.to_string_lossy().as_ref(),
        "--base-dir",
        root.to_string_lossy().as_ref(),
        "--output",
        ws_path.to_string_lossy().as_ref(),
    ]);
    assert!(
        out.status.success(),
        "expected success, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(ws_path.exists(), "missing workspace output: {}", ws_path.display());
    assert!(bundle_dir.join("meta.json").exists());
    assert!(bundle_dir.join("manifest.json").exists());
    assert!(bundle_dir.join("inputs/input.json").exists());
    assert!(bundle_dir.join("outputs/result.json").exists());

    let _ = std::fs::remove_file(&ws_path);
    let _ = std::fs::remove_dir_all(&bundle_dir);
}

#[test]
fn bundle_import_hepdata() {
    let dataset_id = "hepdata.90607.v3.r3";
    let cache_dir = tmp_path("bundle_hepdata_cache");
    let out_dir = tmp_path("bundle_hepdata_out");
    let lock_path = tmp_path("bundle_hepdata_lock.json");
    let bundle_dir = tmp_path("bundle_import_hepdata");

    ensure_clean_path(&cache_dir);
    ensure_clean_path(&out_dir);
    ensure_clean_path(&lock_path);
    ensure_clean_path(&bundle_dir);
    seed_cached_download(
        &cache_dir,
        dataset_id,
        "tests/hepdata/_cache/hepdata.90607.v3.r3/download",
    );

    let out = run(&[
        "--bundle",
        bundle_dir.to_string_lossy().as_ref(),
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
        "expected success, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out_dir.join(dataset_id).join("BkgOnly.json").exists());
    assert!(lock_path.exists());
    assert!(bundle_dir.join("meta.json").exists());
    assert!(bundle_dir.join("manifest.json").exists());
    assert!(bundle_dir.join("inputs/input.json").exists());
    assert!(bundle_dir.join("outputs/result.json").exists());

    let _ = std::fs::remove_dir_all(&cache_dir);
    let _ = std::fs::remove_dir_all(&out_dir);
    let _ = std::fs::remove_file(&lock_path);
    let _ = std::fs::remove_dir_all(&bundle_dir);
}

#[test]
fn bundle_import_hepdata_catalog() {
    let bundle_dir = tmp_path("bundle_import_hepdata_catalog");
    ensure_clean_path(&bundle_dir);

    let out =
        run(&["--bundle", bundle_dir.to_string_lossy().as_ref(), "import", "hepdata", "--list"]);
    assert!(
        out.status.success(),
        "expected success, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(bundle_dir.join("meta.json").exists());
    assert!(bundle_dir.join("manifest.json").exists());
    assert!(bundle_dir.join("inputs/input.json").exists());
    assert!(bundle_dir.join("outputs/result.json").exists());

    let _ = std::fs::remove_dir_all(&bundle_dir);
}

#[test]
fn bundle_import_hepdata_direct_patch_catalog() {
    let dataset_id = "custom.hepdata.90607.v3.r3.catalog";
    let cache_dir = tmp_path("bundle_hepdata_direct_catalog_cache");
    let bundle_dir = tmp_path("bundle_import_hepdata_direct_patch_catalog");

    ensure_clean_path(&cache_dir);
    ensure_clean_path(&bundle_dir);
    seed_cached_download(
        &cache_dir,
        dataset_id,
        "tests/hepdata/_cache/hepdata.90607.v3.r3/download",
    );

    let out = run(&[
        "--bundle",
        bundle_dir.to_string_lossy().as_ref(),
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
        "expected success, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(bundle_dir.join("meta.json").exists());
    assert!(bundle_dir.join("manifest.json").exists());
    assert!(bundle_dir.join("inputs/input.json").exists());
    assert!(bundle_dir.join("outputs/result.json").exists());

    let _ = std::fs::remove_dir_all(&cache_dir);
    let _ = std::fs::remove_dir_all(&bundle_dir);
}

#[test]
fn bundle_viz_distributions() {
    let ws = fixture_path("histfactory/workspace.json");
    let combo = fixture_path("histfactory/combination.xml");
    assert!(ws.exists(), "missing fixture: {}", ws.display());
    assert!(combo.exists(), "missing fixture: {}", combo.display());

    let out_path = tmp_path("viz_distributions.json");
    let bundle_dir = tmp_path("bundle_viz_distributions");
    ensure_clean_path(&out_path);
    ensure_clean_path(&bundle_dir);

    let out = run(&[
        "--bundle",
        bundle_dir.to_string_lossy().as_ref(),
        "viz",
        "distributions",
        "--input",
        ws.to_string_lossy().as_ref(),
        "--histfactory-xml",
        combo.to_string_lossy().as_ref(),
        "--output",
        out_path.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(
        out.status.success(),
        "expected success, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out_path.exists());
    assert!(bundle_dir.join("meta.json").exists());
    assert!(bundle_dir.join("manifest.json").exists());
    assert!(bundle_dir.join("inputs/input.json").exists());
    assert!(bundle_dir.join("outputs/result.json").exists());

    let _ = std::fs::remove_file(&out_path);
    let _ = std::fs::remove_dir_all(&bundle_dir);
}

#[test]
fn bundle_viz_pulls_and_corr() {
    let ws = fixture_path("histfactory/workspace.json");
    assert!(ws.exists(), "missing fixture: {}", ws.display());

    let fit_path = tmp_path("fit.json");
    ensure_clean_path(&fit_path);
    let out_fit = run(&[
        "fit",
        "--input",
        ws.to_string_lossy().as_ref(),
        "--output",
        fit_path.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(
        out_fit.status.success(),
        "fit should succeed, stderr={}",
        String::from_utf8_lossy(&out_fit.stderr)
    );
    assert!(fit_path.exists());

    let pulls_out = tmp_path("pulls.json");
    let pulls_bundle = tmp_path("bundle_viz_pulls");
    ensure_clean_path(&pulls_out);
    ensure_clean_path(&pulls_bundle);
    let out = run(&[
        "--bundle",
        pulls_bundle.to_string_lossy().as_ref(),
        "viz",
        "pulls",
        "--input",
        ws.to_string_lossy().as_ref(),
        "--fit",
        fit_path.to_string_lossy().as_ref(),
        "--output",
        pulls_out.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(
        out.status.success(),
        "viz pulls should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(pulls_bundle.join("meta.json").exists());
    assert!(pulls_bundle.join("manifest.json").exists());

    let corr_out = tmp_path("corr.json");
    let corr_bundle = tmp_path("bundle_viz_corr");
    ensure_clean_path(&corr_out);
    ensure_clean_path(&corr_bundle);
    let out = run(&[
        "--bundle",
        corr_bundle.to_string_lossy().as_ref(),
        "viz",
        "corr",
        "--input",
        ws.to_string_lossy().as_ref(),
        "--fit",
        fit_path.to_string_lossy().as_ref(),
        "--output",
        corr_out.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(
        out.status.success(),
        "viz corr should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(corr_bundle.join("meta.json").exists());
    assert!(corr_bundle.join("manifest.json").exists());

    let _ = std::fs::remove_file(&fit_path);
    let _ = std::fs::remove_file(&pulls_out);
    let _ = std::fs::remove_file(&corr_out);
    let _ = std::fs::remove_dir_all(&pulls_bundle);
    let _ = std::fs::remove_dir_all(&corr_bundle);
}
