use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_nextstat"))
}

fn repo_root() -> PathBuf {
    // crates/ns-cli -> repo root
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn fixture_path(name: &str) -> PathBuf {
    repo_root().join("tests/fixtures").join(name)
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

#[test]
fn viz_corr_smoke_histfactory_fixture() {
    let ws = fixture_path("histfactory/workspace.json");
    assert!(ws.exists(), "missing fixture: {}", ws.display());

    // Fit -> fit.json
    let fit_out = tmp_path("fit_histfactory_for_corr.json");
    let out_fit = run(&[
        "fit",
        "--input",
        ws.to_string_lossy().as_ref(),
        "--output",
        fit_out.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(
        out_fit.status.success(),
        "fit should succeed, stderr={}",
        String::from_utf8_lossy(&out_fit.stderr)
    );

    // viz corr -> corr.json
    let corr_out = tmp_path("viz_corr.json");
    let out_viz = run(&[
        "viz",
        "corr",
        "--input",
        ws.to_string_lossy().as_ref(),
        "--fit",
        fit_out.to_string_lossy().as_ref(),
        "--output",
        corr_out.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(
        out_viz.status.success(),
        "viz corr should succeed, stderr={}",
        String::from_utf8_lossy(&out_viz.stderr)
    );

    let artifact: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&corr_out).unwrap()).unwrap();
    assert_eq!(
        artifact.get("schema_version").and_then(|v| v.as_str()),
        Some("trex_report_corr_v0")
    );

    let names = artifact
        .get("parameter_names")
        .and_then(|v| v.as_array())
        .expect("parameter_names should be an array");
    let corr = artifact.get("corr").and_then(|v| v.as_array()).expect("corr should be an array");
    assert!(!names.is_empty(), "expected non-empty parameter_names");
    assert_eq!(corr.len(), names.len(), "corr must be NxN");

    for row in corr {
        let row = row.as_array().expect("corr row must be an array");
        assert_eq!(row.len(), names.len(), "corr must be NxN");
    }

    let _ = std::fs::remove_file(&fit_out);
    let _ = std::fs::remove_file(&corr_out);
}
