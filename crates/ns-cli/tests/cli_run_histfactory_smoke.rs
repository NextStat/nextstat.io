use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures").join(rel)
}

fn unique_tmp_dir(name: &str) -> PathBuf {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("nextstat_{name}_{pid}_{n}"))
}

fn write_run_config_yaml(path: &Path, histfactory_xml: &Path, out_dir: &Path) {
    let yaml = format!(
        "histfactory_xml: \"{}\"\nout_dir: \"{}\"\noverwrite: true\nthreads: 1\nrender: false\n",
        histfactory_xml.display(),
        out_dir.display()
    );
    std::fs::write(path, yaml).unwrap();
}

#[test]
fn cli_run_histfactory_smoke_writes_bundle_and_artifacts() {
    let bin = env!("CARGO_BIN_EXE_nextstat");

    let hf_xml = fixture_path("histfactory/combination.xml");
    assert!(hf_xml.exists(), "missing fixture: {}", hf_xml.display());

    let out_dir = unique_tmp_dir("run_out");
    let bundle_dir = unique_tmp_dir("run_bundle");
    let cfg_path = unique_tmp_dir("run_cfg").with_extension("yaml");

    // Ensure clean.
    let _ = std::fs::remove_dir_all(&out_dir);
    let _ = std::fs::remove_dir_all(&bundle_dir);
    let _ = std::fs::remove_file(&cfg_path);

    write_run_config_yaml(&cfg_path, &hf_xml, &out_dir);

    let status = Command::new(bin)
        .arg("--bundle")
        .arg(&bundle_dir)
        .arg("run")
        .arg("--config")
        .arg(&cfg_path)
        .status()
        .expect("failed to run nextstat");
    assert!(status.success(), "nextstat run returned non-zero");

    // Out dir structure.
    assert!(out_dir.join("inputs/workspace.json").exists());
    assert!(out_dir.join("artifacts/fit.json").exists());
    assert!(out_dir.join("artifacts/yields.json").exists());
    assert!(out_dir.join("artifacts/distributions.json").exists());

    // Bundle presence (best-effort contract).
    assert!(bundle_dir.join("meta.json").exists());
    assert!(bundle_dir.join("provenance.json").exists());
    assert!(bundle_dir.join("manifest.json").exists());
    assert!(bundle_dir.join("inputs/run_config.yaml").exists());
    assert!(
        bundle_dir.join("inputs/histfactory/combination.xml").exists(),
        "expected HistFactory inputs to be copied into bundle"
    );

    // Cleanup best-effort.
    let _ = std::fs::remove_dir_all(&out_dir);
    let _ = std::fs::remove_dir_all(&bundle_dir);
    let _ = std::fs::remove_file(&cfg_path);
}
