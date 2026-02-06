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
fn viz_pulls_smoke_histfactory_fixture() {
    let ws = fixture_path("histfactory/workspace.json");
    assert!(ws.exists(), "missing fixture: {}", ws.display());

    // Fit -> fit.json
    let fit_out = tmp_path("fit_histfactory_for_pulls.json");
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

    // viz pulls -> pulls.json
    let pulls_out = tmp_path("viz_pulls.json");
    let out_viz = run(&[
        "viz",
        "pulls",
        "--input",
        ws.to_string_lossy().as_ref(),
        "--fit",
        fit_out.to_string_lossy().as_ref(),
        "--output",
        pulls_out.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(
        out_viz.status.success(),
        "viz pulls should succeed, stderr={}",
        String::from_utf8_lossy(&out_viz.stderr)
    );

    let artifact: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&pulls_out).unwrap()).unwrap();
    assert_eq!(
        artifact.get("schema_version").and_then(|v| v.as_str()),
        Some("trex_report_pulls_v0")
    );

    let entries = artifact
        .get("entries")
        .and_then(|v| v.as_array())
        .expect("entries should be an array");
    assert!(!entries.is_empty(), "expected non-empty entries");

    let mut names: Vec<String> = entries
        .iter()
        .map(|e| e.get("name").and_then(|v| v.as_str()).unwrap().to_string())
        .collect();
    names.sort();
    assert!(names.iter().any(|n| n == "bkg_norm"), "expected bkg_norm entry, got {:?}", names);
    assert!(
        names.iter().any(|n| n.starts_with("staterror_SR[")),
        "expected staterror_SR[...] entries, got {:?}",
        names
    );

    for e in entries {
        let pull = e.get("pull").and_then(|v| v.as_f64()).unwrap();
        let constraint = e.get("constraint").and_then(|v| v.as_f64()).unwrap();
        assert!(pull.is_finite(), "pull must be finite");
        assert!(constraint.is_finite(), "constraint must be finite");
    }

    let _ = std::fs::remove_file(&fit_out);
    let _ = std::fs::remove_file(&pulls_out);
}

