use std::path::PathBuf;
use std::process::{Command, Output};

fn bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_nextstat"))
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn fixture_path(name: &str) -> PathBuf {
    repo_root().join("tests/fixtures").join(name)
}

fn run(args: &[&str]) -> Output {
    Command::new(bin_path())
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?} {:?}: {}", bin_path(), args, e))
}

#[test]
fn viz_profile_contract() {
    let input = fixture_path("simple_workspace.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "viz",
        "profile",
        "--input",
        input.to_string_lossy().as_ref(),
        "--start",
        "0.0",
        "--stop",
        "2.0",
        "--points",
        "11",
        "--threads",
        "1",
    ]);

    assert!(
        out.status.success(),
        "viz profile should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert!(v.get("mu_hat").and_then(|x| x.as_f64()).unwrap().is_finite());
    let points = v.get("points").and_then(|x| x.as_array()).expect("points should be array");
    assert_eq!(points.len(), 11);
}

#[test]
fn viz_cls_contract() {
    let input = fixture_path("simple_workspace.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "viz",
        "cls",
        "--input",
        input.to_string_lossy().as_ref(),
        "--alpha",
        "0.05",
        "--scan-start",
        "0.0",
        "--scan-stop",
        "5.0",
        "--scan-points",
        "21",
        "--threads",
        "1",
    ]);

    assert!(
        out.status.success(),
        "viz cls should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert!(v.get("obs_limit").and_then(|x| x.as_f64()).unwrap().is_finite());
    let exp = v.get("exp_limits").and_then(|x| x.as_array()).expect("exp_limits should be array");
    assert_eq!(exp.len(), 5);
    let points = v.get("points").and_then(|x| x.as_array()).expect("points should be array");
    assert_eq!(points.len(), 21);
    let first = points[0].as_object().expect("point should be object");
    assert!(first.get("cls").and_then(|x| x.as_f64()).unwrap().is_finite());
    let expected =
        first.get("expected").and_then(|x| x.as_array()).expect("point.expected should be array");
    assert_eq!(expected.len(), 5);
}
