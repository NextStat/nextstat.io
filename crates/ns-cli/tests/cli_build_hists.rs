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

#[test]
fn build_hists_writes_workspace_json_deterministically() {
    let cfg = fixture_path("trex_config/minimal_tutorial.config");
    assert!(cfg.exists(), "missing fixture: {}", cfg.display());

    let base_dir = repo_root();

    let out_a = tmp_dir("build_hists_a");
    let out_b = tmp_dir("build_hists_b");

    let out = run(&[
        "build-hists",
        "--config",
        cfg.to_string_lossy().as_ref(),
        "--base-dir",
        base_dir.to_string_lossy().as_ref(),
        "--out-dir",
        out_a.to_string_lossy().as_ref(),
    ]);
    assert!(
        out.status.success(),
        "build-hists should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let ws_a = out_a.join("workspace.json");
    assert!(ws_a.exists(), "missing workspace.json: {}", ws_a.display());

    let out = run(&[
        "build-hists",
        "--config",
        cfg.to_string_lossy().as_ref(),
        "--base-dir",
        base_dir.to_string_lossy().as_ref(),
        "--out-dir",
        out_b.to_string_lossy().as_ref(),
    ]);
    assert!(
        out.status.success(),
        "build-hists should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let ws_b = out_b.join("workspace.json");
    assert!(ws_b.exists(), "missing workspace.json: {}", ws_b.display());

    let a = std::fs::read(&ws_a).unwrap();
    let b = std::fs::read(&ws_b).unwrap();
    assert_eq!(a, b, "workspace.json should be deterministic for the same inputs");

    let _ = std::fs::remove_dir_all(&out_a);
    let _ = std::fs::remove_dir_all(&out_b);
}
