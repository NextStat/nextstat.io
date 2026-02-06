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
fn timeseries_kalman_filter_contract() {
    let input = fixture_path("kalman_1d.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-filter",
        "--input",
        input.to_string_lossy().as_ref(),
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-filter should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert!(v.get("log_likelihood").and_then(|x| x.as_f64()).unwrap().is_finite());
    let fm = v.get("filtered_means").and_then(|x| x.as_array()).expect("filtered_means should be array");
    assert_eq!(fm.len(), 4);
}

#[test]
fn timeseries_kalman_filter_local_level_contract() {
    let input = fixture_path("kalman_local_level.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-filter",
        "--input",
        input.to_string_lossy().as_ref(),
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-filter (local_level) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert!(v.get("log_likelihood").and_then(|x| x.as_f64()).unwrap().is_finite());
}

#[test]
fn timeseries_kalman_filter_local_linear_trend_contract() {
    let input = fixture_path("kalman_local_linear_trend.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-filter",
        "--input",
        input.to_string_lossy().as_ref(),
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-filter (local_linear_trend) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert!(v.get("log_likelihood").and_then(|x| x.as_f64()).unwrap().is_finite());
}

#[test]
fn timeseries_kalman_filter_ar1_contract() {
    let input = fixture_path("kalman_ar1.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-filter",
        "--input",
        input.to_string_lossy().as_ref(),
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-filter (ar1) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert!(v.get("log_likelihood").and_then(|x| x.as_f64()).unwrap().is_finite());
}

#[test]
fn timeseries_kalman_filter_local_level_seasonal_contract() {
    let input = fixture_path("kalman_local_level_seasonal.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-filter",
        "--input",
        input.to_string_lossy().as_ref(),
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-filter (local_level_seasonal) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert!(v.get("log_likelihood").and_then(|x| x.as_f64()).unwrap().is_finite());
}

#[test]
fn timeseries_kalman_filter_local_linear_trend_seasonal_contract() {
    let input = fixture_path("kalman_local_linear_trend_seasonal.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-filter",
        "--input",
        input.to_string_lossy().as_ref(),
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-filter (local_linear_trend_seasonal) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert!(v.get("log_likelihood").and_then(|x| x.as_f64()).unwrap().is_finite());
}

#[test]
fn timeseries_kalman_filter_allows_missing_null() {
    let input = fixture_path("kalman_1d_missing.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-filter",
        "--input",
        input.to_string_lossy().as_ref(),
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-filter should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert!(v.get("log_likelihood").and_then(|x| x.as_f64()).unwrap().is_finite());
}

#[test]
fn timeseries_kalman_smooth_contract() {
    let input = fixture_path("kalman_1d.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-smooth",
        "--input",
        input.to_string_lossy().as_ref(),
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-smooth should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    let sm = v.get("smoothed_means").and_then(|x| x.as_array()).expect("smoothed_means should be array");
    assert_eq!(sm.len(), 4);
}

#[test]
fn timeseries_kalman_em_contract() {
    let input = fixture_path("kalman_1d.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-em",
        "--input",
        input.to_string_lossy().as_ref(),
        "--max-iter",
        "5",
        "--tol",
        "1e-12",
        "--min-diag",
        "1e-9",
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-em should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert!(v.get("loglik_trace").and_then(|x| x.as_array()).unwrap().len() >= 2);
    let q = v.get("q").and_then(|x| x.as_array()).expect("q should be array");
    assert_eq!(q.len(), 1);
}

#[test]
fn timeseries_kalman_forecast_contract() {
    let input = fixture_path("kalman_1d.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-forecast",
        "--input",
        input.to_string_lossy().as_ref(),
        "--steps",
        "3",
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-forecast should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_eq!(
        v.get("obs_means").and_then(|x| x.as_array()).unwrap().len(),
        3
    );
}

#[test]
fn timeseries_kalman_simulate_contract() {
    let input = fixture_path("kalman_1d.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&[
        "timeseries",
        "kalman-simulate",
        "--input",
        input.to_string_lossy().as_ref(),
        "--t-max",
        "5",
        "--seed",
        "123",
    ]);

    assert!(
        out.status.success(),
        "timeseries kalman-simulate should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_eq!(v.get("xs").and_then(|x| x.as_array()).unwrap().len(), 5);
    assert_eq!(v.get("ys").and_then(|x| x.as_array()).unwrap().len(), 5);
}
