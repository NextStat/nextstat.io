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

fn assert_metrics_contract(v: &serde_json::Value, command: &str) {
    assert_eq!(
        v.get("schema_version").and_then(|x| x.as_str()),
        Some("nextstat_metrics_v0"),
        "schema_version mismatch: {v}"
    );
    assert_eq!(v.get("tool").and_then(|x| x.as_str()), Some("nextstat"));
    assert_eq!(v.get("command").and_then(|x| x.as_str()), Some(command));
    assert!(v.get("created_unix_ms").is_some(), "missing created_unix_ms");
    let timing = v.get("timing").and_then(|x| x.as_object()).expect("timing must be object");
    let t = timing
        .get("wall_time_s")
        .and_then(|x| x.as_f64())
        .expect("timing.wall_time_s must be number");
    assert!(t.is_finite() && t >= 0.0);
    assert!(v.get("metrics").and_then(|x| x.as_object()).is_some(), "metrics must be object");
}

fn tmp_path(filename: &str) -> PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!("nextstat_cli_{}_{}_{}", std::process::id(), nanos, filename));
    p
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
        "41",
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
fn upper_limit_writes_metrics_json_to_file() {
    let input = fixture_path("simple_workspace.json");
    let metrics = tmp_path("upper_limit_metrics.json");

    let out = run(&[
        "upper-limit",
        "--input",
        input.to_string_lossy().as_ref(),
        "--alpha",
        "0.05",
        "--scan-start",
        "0.0",
        "--scan-stop",
        "2.0",
        "--scan-points",
        "5",
        "--json-metrics",
        metrics.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(
        out.status.success(),
        "upper-limit should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(metrics.exists(), "expected metrics file to exist: {}", metrics.display());

    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&metrics).unwrap()).expect("metrics should be JSON");
    assert_metrics_contract(&v, "upper_limit");
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

#[test]
fn hypotest_writes_metrics_json_to_file() {
    let input = fixture_path("simple_workspace.json");
    let metrics = tmp_path("hypotest_metrics.json");

    let out = run(&[
        "hypotest",
        "--input",
        input.to_string_lossy().as_ref(),
        "--mu",
        "1.0",
        "--json-metrics",
        metrics.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);

    assert!(
        out.status.success(),
        "hypotest should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(metrics.exists(), "expected metrics file to exist: {}", metrics.display());

    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&metrics).unwrap()).expect("metrics should be JSON");
    assert_metrics_contract(&v, "hypotest");
}

#[test]
fn hypotest_toys_expected_set_contract() {
    let input = fixture_path("simple_workspace.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "hypotest-toys",
        "--input",
        input.to_string_lossy().as_ref(),
        "--mu",
        "1.0",
        "--n-toys",
        "3",
        "--seed",
        "123",
        "--expected-set",
        "--threads",
        "1",
    ]);

    assert!(
        out.status.success(),
        "hypotest-toys should succeed, stderr={}",
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
