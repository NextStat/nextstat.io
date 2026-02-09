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

fn tmp_dir(name: &str) -> PathBuf {
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

fn read_json(path: &PathBuf) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap())
        .unwrap_or_else(|e| panic!("invalid JSON at {}: {}", path.display(), e))
}

#[test]
fn report_generates_artifacts_dir_without_matplotlib() {
    let ws = fixture_path("histfactory/workspace.json");
    let xml = fixture_path("histfactory/combination.xml");
    assert!(ws.exists(), "missing fixture: {}", ws.display());
    assert!(xml.exists(), "missing fixture: {}", xml.display());

    let out_dir = tmp_dir("report_artifacts");

    let out = run(&[
        "report",
        "--input",
        ws.to_string_lossy().as_ref(),
        "--histfactory-xml",
        xml.to_string_lossy().as_ref(),
        "--out-dir",
        out_dir.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(
        out.status.success(),
        "report should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let fit = out_dir.join("fit.json");
    let dist = out_dir.join("distributions.json");
    let pulls = out_dir.join("pulls.json");
    let corr = out_dir.join("corr.json");
    let yields = out_dir.join("yields.json");

    assert!(fit.exists(), "missing fit.json: {}", fit.display());
    assert!(dist.exists(), "missing distributions.json: {}", dist.display());
    assert!(pulls.exists(), "missing pulls.json: {}", pulls.display());
    assert!(corr.exists(), "missing corr.json: {}", corr.display());
    assert!(yields.exists(), "missing yields.json: {}", yields.display());

    assert_eq!(
        read_json(&dist).get("schema_version").and_then(|v| v.as_str()),
        Some("trex_report_distributions_v0")
    );
    assert_eq!(
        read_json(&pulls).get("schema_version").and_then(|v| v.as_str()),
        Some("trex_report_pulls_v0")
    );
    assert_eq!(
        read_json(&corr).get("schema_version").and_then(|v| v.as_str()),
        Some("trex_report_corr_v0")
    );
    assert_eq!(
        read_json(&yields).get("schema_version").and_then(|v| v.as_str()),
        Some("trex_report_yields_v0")
    );

    let _ = std::fs::remove_dir_all(&out_dir);
}
