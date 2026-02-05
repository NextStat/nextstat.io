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
fn upper_limit_scan_writes_expected_contract() {
    let input = fixture_path("simple_workspace.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "upper-limit",
        "--input",
        input.to_string_lossy().as_ref(),
        "--alpha",
        "0.05",
        "--scan-start",
        "0.0",
        "--scan-stop",
        "5.0",
        "--scan-points",
        "201",
        "--threads",
        "1",
    ]);

    assert!(
        out.status.success(),
        "upper-limit scan should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_eq!(v.get("mode").and_then(|x| x.as_str()), Some("scan"));
    assert!(v.get("obs_limit").and_then(|x| x.as_f64()).unwrap().is_finite());
    let exp = v.get("exp_limits").and_then(|x| x.as_array()).expect("exp_limits should be array");
    assert_eq!(exp.len(), 5);
}

#[test]
fn hypotest_expected_set_contract() {
    let input = fixture_path("simple_workspace.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "hypotest",
        "--input",
        input.to_string_lossy().as_ref(),
        "--mu",
        "1.0",
        "--expected-set",
        "--threads",
        "1",
    ]);

    assert!(
        out.status.success(),
        "hypotest should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    let expected = v
        .get("expected_set")
        .and_then(|x| x.as_object())
        .expect("expected_set should be an object when enabled");
    let cls = expected
        .get("cls")
        .and_then(|x| x.as_array())
        .expect("expected_set.cls should be an array");
    assert_eq!(cls.len(), 5);
}
