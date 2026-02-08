use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_nextstat"))
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn fixture_path(name: &str) -> PathBuf {
    repo_root().join("tests/fixtures").join(name)
}

fn tmp_file_path(suffix: &str) -> PathBuf {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!(
        "nextstat_validation_report_{}_{}_{}.json",
        std::process::id(),
        nanos,
        suffix
    ));
    p
}

fn run(args: &[&str]) -> Output {
    Command::new(bin_path())
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?} {:?}: {}", bin_path(), args, e))
}

#[test]
fn validation_report_writes_json_artifact() {
    let apex2 = fixture_path("apex2_master_min.json");
    let workspace = fixture_path("simple_workspace.json");
    assert!(apex2.exists(), "missing fixture: {}", apex2.display());
    assert!(workspace.exists(), "missing fixture: {}", workspace.display());

    let out_path = tmp_file_path("out");
    let out = run(&[
        "validation-report",
        "--apex2",
        apex2.to_string_lossy().as_ref(),
        "--workspace",
        workspace.to_string_lossy().as_ref(),
        "--out",
        out_path.to_string_lossy().as_ref(),
        "--deterministic",
    ]);
    assert!(
        out.status.success(),
        "validation-report should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let bytes = std::fs::read(&out_path).expect("output JSON should exist");
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("output should be JSON");
    assert_eq!(v.get("schema_version").and_then(|x| x.as_str()), Some("validation_report_v1"));
    assert_eq!(v.get("deterministic").and_then(|x| x.as_bool()), Some(true));
    assert_eq!(
        v.pointer("/apex2_summary/overall").and_then(|x| x.as_str()),
        Some("pass")
    );

    let _ = std::fs::remove_file(&out_path);
}

