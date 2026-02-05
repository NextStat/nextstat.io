use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ns-cli"))
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

fn assert_json_contract(v: &serde_json::Value) {
    let param_names = v
        .get("parameter_names")
        .and_then(|x| x.as_array())
        .expect("parameter_names should be an array");
    assert!(!param_names.is_empty(), "parameter_names should be non-empty");

    let bestfit = v.get("bestfit").and_then(|x| x.as_array()).expect("bestfit should be an array");
    let unc = v
        .get("uncertainties")
        .and_then(|x| x.as_array())
        .expect("uncertainties should be an array");
    assert_eq!(bestfit.len(), param_names.len(), "bestfit length must match parameter_names");
    assert_eq!(unc.len(), param_names.len(), "uncertainties length must match parameter_names");

    let nll = v.get("nll").and_then(|x| x.as_f64()).expect("nll should be a number");
    assert!(nll.is_finite(), "nll must be finite");

    let converged =
        v.get("converged").and_then(|x| x.as_bool()).expect("converged should be a boolean");
    // Don't assert it must be true in all environments, but it should exist.
    let _ = converged;

    let n_eval = v
        .get("n_evaluations")
        .and_then(|x| x.as_u64())
        .expect("n_evaluations should be an integer");
    assert!(n_eval > 0, "n_evaluations should be > 0");
}

#[test]
fn version_smoke() {
    let out = run(&["version"]);
    assert!(out.status.success(), "version should succeed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("nextstat "), "unexpected stdout: {}", stdout);
}

#[test]
fn fit_writes_valid_json_to_stdout() {
    let input = fixture_path("simple_workspace.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&["fit", "--input", input.to_string_lossy().as_ref(), "--threads", "1"]);
    assert!(
        out.status.success(),
        "fit should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    assert_json_contract(&v);
}

#[test]
fn fit_writes_valid_json_to_file() {
    let input = fixture_path("simple_workspace.json");
    let output = tmp_path("fit_out.json");

    let out = run(&[
        "fit",
        "--input",
        input.to_string_lossy().as_ref(),
        "--output",
        output.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(
        out.status.success(),
        "fit should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(output.exists(), "expected output file to exist: {}", output.display());

    let bytes = std::fs::read(&output).unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("output file should be JSON");
    assert_json_contract(&v);

    let _ = std::fs::remove_file(&output);
}

#[test]
fn fit_errors_on_missing_input() {
    let missing = tmp_path("does_not_exist.json");
    let out = run(&["fit", "--input", missing.to_string_lossy().as_ref(), "--threads", "1"]);
    assert!(!out.status.success(), "expected failure for missing input");
}

#[test]
fn fit_errors_on_invalid_json() {
    let bad = tmp_path("bad.json");
    std::fs::write(&bad, "{").unwrap();

    let out = run(&["fit", "--input", bad.to_string_lossy().as_ref(), "--threads", "1"]);
    assert!(!out.status.success(), "expected failure for invalid JSON");

    let _ = std::fs::remove_file(&bad);
}

#[test]
fn fit_errors_on_length_mismatch_fixture() {
    let input = fixture_path("bad_observations_length_mismatch.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out = run(&["fit", "--input", input.to_string_lossy().as_ref(), "--threads", "1"]);
    assert!(!out.status.success(), "expected failure for length mismatch");

    // Try to keep this robust: error message can evolve.
    let stderr = String::from_utf8_lossy(&out.stderr).to_lowercase();
    assert!(
        stderr.contains("length") || stderr.contains("mismatch") || stderr.contains("validation"),
        "unexpected stderr: {}",
        stderr
    );
}
