use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

/// Compare two markdown strings allowing tiny floating-point rounding differences.
/// Numbers are extracted from each string and compared with relative tolerance 1e-4.
fn assert_markdown_close(actual: &str, expected: &str) {
    let re = regex::Regex::new(r"[0-9]+\.[0-9]+").unwrap();
    let strip_numbers = |s: &str| re.replace_all(s, "###").to_string();
    assert_eq!(
        strip_numbers(actual),
        strip_numbers(expected),
        "markdown structure mismatch (non-numeric parts differ)"
    );
    let actual_nums: Vec<f64> =
        re.find_iter(actual).filter_map(|m| m.as_str().parse().ok()).collect();
    let expected_nums: Vec<f64> =
        re.find_iter(expected).filter_map(|m| m.as_str().parse().ok()).collect();
    assert_eq!(actual_nums.len(), expected_nums.len(), "different number count");
    for (i, (a, e)) in actual_nums.iter().zip(expected_nums.iter()).enumerate() {
        let tol = 1e-4_f64.max(e.abs() * 1e-4);
        assert!(
            (a - e).abs() <= tol,
            "numeric mismatch at position {i}: actual={a} expected={e} tol={tol}"
        );
    }
}

fn bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_nextstat"))
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

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn full_literature_fixture() -> PathBuf {
    repo_root().join("tests/fixtures/measurement_combine_gvm_topmass_full.json")
}

fn calibration_outlier_input_fixture() -> PathBuf {
    repo_root().join("tests/fixtures/measurement_combine_calibration_outlier_input.json")
}

fn calibration_outlier_report_fixture() -> PathBuf {
    repo_root().join("tests/fixtures/measurement_combine_calibration_outlier_report.json")
}

fn calibration_outlier_study_report_fixture() -> PathBuf {
    repo_root().join("tests/fixtures/measurement_combine_calibration_outlier_study_report.json")
}

fn scenario_outlier_input_fixture() -> PathBuf {
    repo_root().join("tests/fixtures/measurement_combine_outlier_scenarios.json")
}

fn scenario_outlier_report_fixture() -> PathBuf {
    repo_root().join("tests/fixtures/measurement_combine_outlier_scenario_study_report.json")
}

fn scenario_outlier_calibration_campaign_report_fixture() -> PathBuf {
    repo_root().join("tests/fixtures/measurement_combine_outlier_calibration_campaign_report.json")
}

fn scenario_outlier_calibration_campaign_summary_fixture() -> PathBuf {
    repo_root().join("tests/fixtures/measurement_combine_outlier_calibration_campaign_summary.json")
}

fn scenario_outlier_calibration_campaign_summary_markdown_fixture() -> PathBuf {
    repo_root().join("tests/fixtures/measurement_combine_outlier_calibration_campaign_summary.md")
}

fn stable_first_example_dir() -> PathBuf {
    repo_root().join("docs/examples/gvm-stable-first")
}

fn calibration_campaign_brief_fixture() -> PathBuf {
    repo_root().join("tests/fixtures/measurement_combine_calibration_campaign_brief.json")
}

fn calibration_campaign_brief_markdown_fixture() -> PathBuf {
    repo_root().join("tests/fixtures/measurement_combine_calibration_campaign_brief.md")
}

fn calibration_campaign_topmass_only_brief_fixture() -> PathBuf {
    repo_root().join("tests/fixtures/measurement_combine_calibration_topmass_only_brief.json")
}

fn calibration_campaign_family_report_fixture() -> PathBuf {
    repo_root().join("tests/fixtures/measurement_combine_calibration_campaign_family_report.json")
}

fn calibration_campaign_family_report_markdown_fixture() -> PathBuf {
    repo_root().join("tests/fixtures/measurement_combine_calibration_campaign_family_report.md")
}

fn calibration_campaign_family_matrix_fixture() -> PathBuf {
    repo_root().join("tests/fixtures/measurement_combine_calibration_campaign_family_matrix.json")
}

fn calibration_campaign_family_matrix_markdown_fixture() -> PathBuf {
    repo_root().join("tests/fixtures/measurement_combine_calibration_campaign_family_matrix.md")
}

fn calibration_campaign_topmass_only_family_report_fixture() -> PathBuf {
    repo_root()
        .join("tests/fixtures/measurement_combine_calibration_topmass_only_family_report.json")
}

fn calibration_campaign_topmass_only_family_matrix_fixture() -> PathBuf {
    repo_root()
        .join("tests/fixtures/measurement_combine_calibration_topmass_only_family_matrix.json")
}

fn calibration_campaign_portfolio_fixture() -> PathBuf {
    repo_root().join("tests/fixtures/measurement_combine_calibration_campaign_portfolio.json")
}

fn calibration_campaign_portfolio_markdown_fixture() -> PathBuf {
    repo_root().join("tests/fixtures/measurement_combine_calibration_campaign_portfolio.md")
}

fn calibration_campaign_portfolio_stability_fixture() -> PathBuf {
    repo_root()
        .join("tests/fixtures/measurement_combine_calibration_campaign_portfolio_stability.json")
}

fn calibration_campaign_portfolio_stability_markdown_fixture() -> PathBuf {
    repo_root()
        .join("tests/fixtures/measurement_combine_calibration_campaign_portfolio_stability.md")
}

fn full_literature_calibration_campaign_summary_fixture() -> PathBuf {
    repo_root()
        .join("tests/fixtures/measurement_combine_calibration_topmass_full_campaign_summary.json")
}

fn scenario_solver_parity_report_fixture() -> PathBuf {
    repo_root().join("tests/fixtures/measurement_combine_solver_parity_scenario_study_report.json")
}

fn scenario_solver_parity_markdown_fixture() -> PathBuf {
    repo_root().join("tests/fixtures/measurement_combine_solver_parity_scenario_study_report.md")
}

fn scenario_solver_parity_digest_fixture() -> PathBuf {
    repo_root().join("tests/fixtures/measurement_combine_solver_parity_scenario_study_digest.json")
}

fn scenario_solver_parity_digest_markdown_fixture() -> PathBuf {
    repo_root().join("tests/fixtures/measurement_combine_solver_parity_scenario_study_digest.md")
}

fn calibration_campaign_solver_parity_report_fixture() -> PathBuf {
    repo_root()
        .join("tests/fixtures/measurement_combine_solver_parity_calibration_campaign_report.json")
}

fn calibration_campaign_solver_parity_markdown_fixture() -> PathBuf {
    repo_root()
        .join("tests/fixtures/measurement_combine_solver_parity_calibration_campaign_report.md")
}

fn calibration_campaign_solver_parity_digest_fixture() -> PathBuf {
    repo_root()
        .join("tests/fixtures/measurement_combine_solver_parity_calibration_campaign_digest.json")
}

fn calibration_campaign_solver_parity_digest_markdown_fixture() -> PathBuf {
    repo_root()
        .join("tests/fixtures/measurement_combine_solver_parity_calibration_campaign_digest.md")
}

fn assert_json_close(actual: &serde_json::Value, expected: &serde_json::Value, path: &str) {
    match (actual, expected) {
        (serde_json::Value::Object(a), serde_json::Value::Object(e)) => {
            assert_eq!(a.len(), e.len(), "object key count mismatch at {path}");
            for (key, expected_value) in e {
                let child = format!("{path}.{key}");
                let actual_value = a.get(key).unwrap_or_else(|| panic!("missing key at {child}"));
                assert_json_close(actual_value, expected_value, &child);
            }
        }
        (serde_json::Value::Array(a), serde_json::Value::Array(e)) => {
            assert_eq!(a.len(), e.len(), "array length mismatch at {path}");
            for (idx, (actual_value, expected_value)) in a.iter().zip(e.iter()).enumerate() {
                let child = format!("{path}[{idx}]");
                assert_json_close(actual_value, expected_value, &child);
            }
        }
        (serde_json::Value::Number(a), serde_json::Value::Number(e)) => {
            let actual_value = a.as_f64().expect("actual number should convert to f64");
            let expected_value = e.as_f64().expect("expected number should convert to f64");
            let tol = 1e-4_f64.max(1e-4_f64 * actual_value.abs().max(expected_value.abs()));
            assert!(
                (actual_value - expected_value).abs() <= tol,
                "numeric mismatch at {path}: actual={actual_value} expected={expected_value} tol={tol}"
            );
        }
        _ => {
            assert_eq!(actual, expected, "value mismatch at {path}");
        }
    }
}

fn assert_json_matches_fixture(actual_json: &str, fixture: &Path) {
    let actual: serde_json::Value =
        serde_json::from_str(actual_json).expect("actual output should be valid JSON");
    let expected: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fixture).expect("fixture should exist"))
            .expect("fixture should be valid JSON");
    assert_json_close(&actual, &expected, "$");
}

#[test]
fn combine_measurements_writes_valid_json() {
    let input = tmp_path("measurement_combine.json");
    std::fs::write(
        &input,
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": "nextstat_measurement_combination_v0",
            "poi": "mu",
            "measurements": [
                {"name": "m1", "value": 1.0},
                {"name": "m2", "value": 3.0}
            ],
            "stat_covariance": [
                [1.0, 0.0],
                [0.0, 4.0]
            ],
            "systematics": []
        }))
        .unwrap(),
    )
    .unwrap();

    let out = run(&[
        "combine-measurements",
        "--input",
        input.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(
        out.status.success(),
        "combine-measurements should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("stdout should be JSON");
    assert_eq!(
        v.get("schema_version").and_then(|x| x.as_str()),
        Some("nextstat_measurement_combination_result_v0")
    );
    assert_eq!(v.get("stability").and_then(|x| x.as_str()), Some("stable"));
    let mu_hat = v.get("mu_hat").and_then(|x| x.as_f64()).expect("mu_hat should be a float");
    assert!((mu_hat - 1.4).abs() < 1e-12, "unexpected mu_hat: {mu_hat}");
    let bartlett = v
        .get("diagnostics")
        .and_then(|x| x.get("bartlett"))
        .and_then(|x| x.as_object())
        .expect("diagnostics.bartlett should be present");
    assert_eq!(bartlett.get("supported").and_then(|x| x.as_bool()), Some(false));
    let validity = v
        .get("diagnostics")
        .and_then(|x| x.get("perturbative_validity"))
        .and_then(|x| x.as_object())
        .expect("diagnostics.perturbative_validity should be present");
    assert_eq!(validity.get("threshold").and_then(|x| x.as_f64()), Some(1.0));
}

#[test]
fn combine_measurements_build_spec_accepts_tabular_bundle() {
    let measurements = tmp_path("measurement_combine_measurements.csv");
    let stat_covariance = tmp_path("measurement_combine_stat_covariance.csv");
    let systematics = tmp_path("measurement_combine_systematics.csv");
    let correlations = tmp_path("measurement_combine_correlations.csv");

    std::fs::write(&measurements, "name,value\nm1,1.0\nm2,3.0\n").unwrap();
    std::fs::write(&stat_covariance, "measurement,m2,m1\nm2,4.0,0.0\nm1,0.0,1.0\n").unwrap();
    std::fs::write(
        &systematics,
        "systematic,measurement,magnitude,error_on_error,aux_mean\ns1,m1,0.2,0.1,0.0\ns1,m2,0.3,0.1,0.0\n",
    )
    .unwrap();
    std::fs::write(
        &correlations,
        "systematic,row_measurement,col_measurement,corr\ns1,m1,m2,0.5\n",
    )
    .unwrap();

    let out = run(&[
        "combine-measurements-build-spec",
        "--poi",
        "mu",
        "--measurements",
        measurements.to_string_lossy().as_ref(),
        "--stat-covariance",
        stat_covariance.to_string_lossy().as_ref(),
        "--systematics",
        systematics.to_string_lossy().as_ref(),
        "--correlations",
        correlations.to_string_lossy().as_ref(),
    ]);
    assert!(
        out.status.success(),
        "combine-measurements-build-spec should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let spec: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be JSON");
    assert_eq!(
        spec.get("schema_version").and_then(|x| x.as_str()),
        Some("nextstat_measurement_combination_v0")
    );
    assert_eq!(spec.get("poi").and_then(|x| x.as_str()), Some("mu"));
    assert_eq!(
        spec.get("systematics")
            .and_then(|x| x.as_array())
            .and_then(|arr| arr.first())
            .and_then(|x| x.get("corr"))
            .cloned(),
        Some(serde_json::json!([[1.0, 0.5], [0.5, 1.0]]))
    );
}

#[test]
fn stable_first_example_bundle_roundtrips_through_cli() {
    let example_dir = stable_first_example_dir();
    let manifest = example_dir.join("manifest.yaml");
    let measurements = example_dir.join("measurements.csv");
    let stat_covariance = example_dir.join("stat_covariance.csv");
    let systematics = example_dir.join("systematics.csv");
    let correlations = example_dir.join("correlations.csv");
    let committed_spec: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(example_dir.join("spec.json")).expect("example spec should exist"),
    )
    .expect("example spec should be valid JSON");

    let built = run(&[
        "combine-measurements-build-spec",
        "--poi",
        "mu",
        "--measurements",
        measurements.to_string_lossy().as_ref(),
        "--stat-covariance",
        stat_covariance.to_string_lossy().as_ref(),
        "--systematics",
        systematics.to_string_lossy().as_ref(),
        "--correlations",
        correlations.to_string_lossy().as_ref(),
    ]);
    assert!(
        built.status.success(),
        "example build-spec should succeed, stderr={}",
        String::from_utf8_lossy(&built.stderr)
    );
    let built_spec: serde_json::Value =
        serde_json::from_slice(&built.stdout).expect("example build-spec output should be JSON");
    assert_eq!(built_spec, committed_spec, "example bundle should match committed spec");

    let built_from_manifest = run(&[
        "combine-measurements-build-spec",
        "--manifest",
        manifest.to_string_lossy().as_ref(),
    ]);
    assert!(
        built_from_manifest.status.success(),
        "example build-spec manifest path should succeed, stderr={}",
        String::from_utf8_lossy(&built_from_manifest.stderr)
    );
    let built_manifest_spec: serde_json::Value =
        serde_json::from_slice(&built_from_manifest.stdout)
            .expect("example build-spec manifest output should be JSON");
    assert_eq!(
        built_manifest_spec, committed_spec,
        "example manifest bundle should match committed spec"
    );

    let built_spec_path = tmp_path("measurement_combine_example_spec.json");
    std::fs::write(
        &built_spec_path,
        serde_json::to_string_pretty(&built_spec).expect("built spec should serialize"),
    )
    .unwrap();

    let fit = run(&[
        "combine-measurements",
        "--input",
        built_spec_path.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(
        fit.status.success(),
        "example combine-measurements should succeed, stderr={}",
        String::from_utf8_lossy(&fit.stderr)
    );
    let fit_json: serde_json::Value =
        serde_json::from_slice(&fit.stdout).expect("example fit output should be JSON");
    assert_eq!(fit_json.get("stability").and_then(|x| x.as_str()), Some("stable"));

    let calibrate = run(&[
        "combine-measurements-calibrate",
        "--input",
        built_spec_path.to_string_lossy().as_ref(),
        "--threads",
        "1",
        "--n-toys",
        "8",
        "--seed",
        "42",
    ]);
    assert!(
        calibrate.status.success(),
        "example calibrate should succeed, stderr={}",
        String::from_utf8_lossy(&calibrate.stderr)
    );
    let calibration_json: serde_json::Value = serde_json::from_slice(&calibrate.stdout)
        .expect("example calibration output should be JSON");
    assert_eq!(calibration_json.get("stability").and_then(|x| x.as_str()), Some("stable"));

    let study = run(&[
        "combine-measurements-calibrate-study",
        "--input",
        built_spec_path.to_string_lossy().as_ref(),
        "--threads",
        "1",
        "--n-toys",
        "8",
        "--seeds",
        "42,43",
    ]);
    assert!(
        study.status.success(),
        "example calibrate-study should succeed, stderr={}",
        String::from_utf8_lossy(&study.stderr)
    );
    let study_json: serde_json::Value =
        serde_json::from_slice(&study.stdout).expect("example study output should be JSON");
    assert_eq!(study_json.get("stability").and_then(|x| x.as_str()), Some("stable"));
}

#[test]
fn combine_measurements_rejects_negative_error_on_error() {
    let input = tmp_path("measurement_combine_invalid.json");
    std::fs::write(
        &input,
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": "nextstat_measurement_combination_v0",
            "poi": "mu",
            "measurements": [
                {"name": "m1", "value": 1.0}
            ],
            "stat_covariance": [
                [1.0]
            ],
            "systematics": [{
                "name": "s1",
                "magnitudes": [0.2],
                "corr": [[1.0]],
                "error_on_error": -0.1
            }]
        }))
        .unwrap(),
    )
    .unwrap();

    let out = run(&["combine-measurements", "--input", input.to_string_lossy().as_ref()]);
    assert!(!out.status.success(), "expected command failure");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("error_on_error"), "unexpected stderr: {stderr}");
}

#[test]
fn combine_measurements_is_deterministic_for_threads_one() {
    let fixture = full_literature_fixture();
    let first = run(&[
        "combine-measurements",
        "--input",
        fixture.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(first.status.success(), "first run failed: {}", String::from_utf8_lossy(&first.stderr));

    let second = run(&[
        "combine-measurements",
        "--input",
        fixture.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(
        second.status.success(),
        "second run failed: {}",
        String::from_utf8_lossy(&second.stderr)
    );

    assert_eq!(first.stdout, second.stdout, "expected byte-identical JSON output");
}

#[test]
fn combine_measurements_matches_full_literature_baseline() {
    let fixture = full_literature_fixture();
    let out = run(&[
        "combine-measurements",
        "--input",
        fixture.to_string_lossy().as_ref(),
        "--threads",
        "1",
        "--ci-level",
        "0.683",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("stdout should be JSON");
    let mu_hat = v.get("mu_hat").and_then(|x| x.as_f64()).expect("mu_hat should be a float");
    let ci = v
        .get("confidence_interval")
        .and_then(|x| x.as_object())
        .expect("confidence_interval should be present");
    let lower = ci.get("lower").and_then(|x| x.as_f64()).expect("lower should be a float");
    let upper = ci.get("upper").and_then(|x| x.as_f64()).expect("upper should be a float");
    let half_width = (upper - lower) / 2.0;
    assert!((mu_hat - 172.51).abs() <= 5e-3, "unexpected mu_hat: {mu_hat}");
    assert!((half_width - 0.33).abs() <= 1e-2, "unexpected half-width: {half_width}");
}

#[test]
fn combine_measurements_surfaces_bartlett_for_full_correlated_gvm_case() {
    let fixture = full_literature_fixture();
    let spec_text = std::fs::read_to_string(&fixture).expect("fixture should exist");
    let mut spec_json: serde_json::Value =
        serde_json::from_str(&spec_text).expect("fixture should be valid JSON");
    let systematics = spec_json
        .get_mut("systematics")
        .and_then(|x| x.as_array_mut())
        .expect("systematics should be an array");
    let bjes = systematics
        .iter_mut()
        .find(|s| s.get("name").and_then(|x| x.as_str()) == Some("b-JES"))
        .expect("b-JES systematic should be present");
    bjes["error_on_error"] = serde_json::json!(0.5);

    let input = tmp_path("measurement_combine_full_gvm.json");
    std::fs::write(&input, serde_json::to_string_pretty(&spec_json).unwrap()).unwrap();

    let out = run(&[
        "combine-measurements",
        "--input",
        input.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("stdout should be JSON");
    let bartlett = v
        .get("diagnostics")
        .and_then(|x| x.get("bartlett"))
        .and_then(|x| x.as_object())
        .expect("diagnostics.bartlett should be present");
    assert_eq!(bartlett.get("supported").and_then(|x| x.as_bool()), Some(true));
    assert_eq!(bartlett.get("method").and_then(|x| x.as_str()), Some("lawley_order_eps2_general"));
    assert!(bartlett.get("b_mu").and_then(|x| x.as_f64()).unwrap().is_finite());
    assert!(bartlett.get("q_star").and_then(|x| x.as_f64()).unwrap().is_finite());
}

#[test]
fn combine_measurements_accepts_solver_flag_for_paper_modes() {
    let fixture = full_literature_fixture();
    let spec_text = std::fs::read_to_string(&fixture).expect("fixture should exist");
    let mut spec_json: serde_json::Value =
        serde_json::from_str(&spec_text).expect("fixture should be valid JSON");
    let systematics = spec_json
        .get_mut("systematics")
        .and_then(|x| x.as_array_mut())
        .expect("systematics should be an array");
    let bjes = systematics
        .iter_mut()
        .find(|s| s.get("name").and_then(|x| x.as_str()) == Some("b-JES"))
        .expect("b-JES systematic should be present");
    bjes["error_on_error"] = serde_json::json!(0.05);

    let input = tmp_path("measurement_combine_solver_modes.json");
    std::fs::write(&input, serde_json::to_string_pretty(&spec_json).unwrap()).unwrap();

    let paper = run(&[
        "combine-measurements",
        "--input",
        input.to_string_lossy().as_ref(),
        "--solver",
        "numerical-paper",
        "--threads",
        "1",
    ]);
    assert!(
        paper.status.success(),
        "paper solver failed: {}",
        String::from_utf8_lossy(&paper.stderr)
    );
    let paper_json: serde_json::Value =
        serde_json::from_slice(&paper.stdout).expect("stdout should be JSON");
    assert_eq!(
        paper_json.get("optimizer").and_then(|x| x.get("method")).and_then(|x| x.as_str()),
        Some("numerical_profile_gvm_original_theta")
    );

    let analytic = run(&[
        "combine-measurements",
        "--input",
        input.to_string_lossy().as_ref(),
        "--solver",
        "analytic-perturbative",
        "--threads",
        "1",
    ]);
    assert!(
        analytic.status.success(),
        "analytic solver failed: {}",
        String::from_utf8_lossy(&analytic.stderr)
    );
    let analytic_json: serde_json::Value =
        serde_json::from_slice(&analytic.stdout).expect("stdout should be JSON");
    assert_eq!(
        analytic_json.get("optimizer").and_then(|x| x.get("method")).and_then(|x| x.as_str()),
        Some("analytic_perturbative_order_eps2")
    );
}

#[test]
fn combine_measurements_defaults_to_auto_solver_contract() {
    let fixture = full_literature_fixture();
    let spec_text = std::fs::read_to_string(&fixture).expect("fixture should exist");
    let mut spec_json: serde_json::Value =
        serde_json::from_str(&spec_text).expect("fixture should be valid JSON");
    let systematics = spec_json
        .get_mut("systematics")
        .and_then(|x| x.as_array_mut())
        .expect("systematics should be an array");
    let bjes = systematics
        .iter_mut()
        .find(|s| s.get("name").and_then(|x| x.as_str()) == Some("b-JES"))
        .expect("b-JES systematic should be present");
    bjes["error_on_error"] = serde_json::json!(0.05);

    let input = tmp_path("measurement_combine_auto_default.json");
    std::fs::write(&input, serde_json::to_string_pretty(&spec_json).unwrap()).unwrap();

    let out = run(&[
        "combine-measurements",
        "--input",
        input.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("stdout should be JSON");
    assert_eq!(
        v.get("optimizer").and_then(|x| x.get("method")).and_then(|x| x.as_str()),
        Some("analytic_perturbative_order_eps2")
    );
    assert_eq!(
        v.get("diagnostics").and_then(|x| x.get("requested_solver")).and_then(|x| x.as_str()),
        Some("auto")
    );
    assert_eq!(
        v.get("diagnostics").and_then(|x| x.get("effective_solver")).and_then(|x| x.as_str()),
        Some("analytic-perturbative")
    );
}

#[test]
fn combine_measurements_default_auto_falls_back_to_numerical_paper_when_needed() {
    let input = tmp_path("measurement_combine_auto_fallback.json");
    std::fs::write(
        &input,
        r#"{
  "schema_version": "nextstat_measurement_combination_v0",
  "poi": "mu",
  "measurements": [
    {"name": "m1", "value": 0.0},
    {"name": "m2", "value": 0.1},
    {"name": "m3", "value": 3.0}
  ],
  "stat_covariance": [
    [0.04, 0.0, 0.0],
    [0.0, 0.04, 0.0],
    [0.0, 0.0, 0.04]
  ],
  "systematics": [
    {
      "name": "new",
      "magnitudes": [0.0, 0.0, 0.2],
      "corr": [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
      ],
      "error_on_error": 1.5
    }
  ]
}"#,
    )
    .unwrap();

    let out = run(&[
        "combine-measurements",
        "--input",
        input.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("stdout should be JSON");
    assert_eq!(
        v.get("optimizer").and_then(|x| x.get("method")).and_then(|x| x.as_str()),
        Some("numerical_profile_gvm_original_theta")
    );
    assert_eq!(
        v.get("diagnostics").and_then(|x| x.get("requested_solver")).and_then(|x| x.as_str()),
        Some("auto")
    );
    assert_eq!(
        v.get("diagnostics").and_then(|x| x.get("effective_solver")).and_then(|x| x.as_str()),
        Some("numerical-paper")
    );
}

#[test]
fn combine_measurements_calibrate_writes_valid_json() {
    let input = calibration_outlier_input_fixture();

    let out = run(&[
        "combine-measurements-calibrate",
        "--input",
        input.to_string_lossy().as_ref(),
        "--threads",
        "1",
        "--n-toys",
        "16",
        "--seed",
        "123",
    ]);
    assert!(
        out.status.success(),
        "combine-measurements-calibrate should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("stdout should be JSON");
    assert_eq!(
        v.get("schema_version").and_then(|x| x.as_str()),
        Some("nextstat_measurement_combination_calibration_v0")
    );
    assert_eq!(v.get("stability").and_then(|x| x.as_str()), Some("stable"));
    assert_eq!(v.get("n_toys").and_then(|x| x.as_u64()), Some(16));
    let summary = v.get("summary").and_then(|x| x.as_object()).expect("summary should be present");
    assert!(summary.get("mean_q").and_then(|x| x.as_f64()).unwrap().is_finite());
    assert!(summary.get("mean_q_star").and_then(|x| x.as_f64()).unwrap().is_finite());
    assert!(summary.get("mean_sigma").and_then(|x| x.as_f64()).unwrap().is_finite());
    assert!(
        summary.get("mean_sigma_star_to_sigma_ratio").and_then(|x| x.as_f64()).unwrap().is_finite()
    );
}

#[test]
fn combine_measurements_calibrate_matches_committed_artifact() {
    let input = calibration_outlier_input_fixture();

    let out = run(&[
        "combine-measurements-calibrate",
        "--input",
        input.to_string_lossy().as_ref(),
        "--threads",
        "1",
        "--n-toys",
        "16",
        "--seed",
        "123",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    assert_json_matches_fixture(&actual, &calibration_outlier_report_fixture());
}

#[test]
fn combine_measurements_calibrate_study_writes_valid_json() {
    let input = calibration_outlier_input_fixture();

    let out = run(&[
        "combine-measurements-calibrate-study",
        "--input",
        input.to_string_lossy().as_ref(),
        "--threads",
        "1",
        "--n-toys",
        "16",
        "--seeds",
        "123,124,125",
    ]);
    assert!(
        out.status.success(),
        "combine-measurements-calibrate-study should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("stdout should be JSON");
    assert_eq!(
        v.get("schema_version").and_then(|x| x.as_str()),
        Some("nextstat_measurement_combination_calibration_study_v0")
    );
    assert_eq!(v.get("stability").and_then(|x| x.as_str()), Some("stable"));
    assert_eq!(v.get("seeds").and_then(|x| x.as_array()).map(|x| x.len()), Some(3));
    let aggregate =
        v.get("aggregate").and_then(|x| x.as_object()).expect("aggregate should be present");
    assert!(aggregate.get("mean_of_mean_q").and_then(|x| x.as_f64()).unwrap().is_finite());
    assert!(
        aggregate
            .get("max_abs_mean_sigma_star_to_sigma_ratio_delta_from_reference")
            .and_then(|x| x.as_f64())
            .unwrap()
            .is_finite()
    );
}

#[test]
fn combine_measurements_calibrate_study_matches_committed_artifact() {
    let input = calibration_outlier_input_fixture();

    let out = run(&[
        "combine-measurements-calibrate-study",
        "--input",
        input.to_string_lossy().as_ref(),
        "--threads",
        "1",
        "--n-toys",
        "16",
        "--seeds",
        "123,124,125",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    assert_json_matches_fixture(&actual, &calibration_outlier_study_report_fixture());
}

#[test]
fn combine_measurements_scenario_study_writes_valid_json() {
    let input = calibration_outlier_input_fixture();
    let scenarios = scenario_outlier_input_fixture();

    let out = run(&[
        "combine-measurements-scenario-study",
        "--input",
        input.to_string_lossy().as_ref(),
        "--scenarios",
        scenarios.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(
        out.status.success(),
        "combine-measurements-scenario-study should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("stdout should be JSON");
    assert_eq!(
        v.get("schema_version").and_then(|x| x.as_str()),
        Some("nextstat_measurement_combination_scenario_study_v0")
    );
    assert_eq!(v.get("stability").and_then(|x| x.as_str()), Some("research-grade"));
    let aggregate =
        v.get("aggregate").and_then(|x| x.as_object()).expect("aggregate should be present");
    assert_eq!(aggregate.get("n_scenarios").and_then(|x| x.as_u64()), Some(3));
    assert!(
        aggregate.get("max_sigma_ratio_to_baseline").and_then(|x| x.as_f64()).unwrap().is_finite()
    );
}

#[test]
fn combine_measurements_scenario_study_matches_committed_artifact() {
    let input = calibration_outlier_input_fixture();
    let scenarios = scenario_outlier_input_fixture();

    let out = run(&[
        "combine-measurements-scenario-study",
        "--input",
        input.to_string_lossy().as_ref(),
        "--scenarios",
        scenarios.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    assert_json_matches_fixture(&actual, &scenario_outlier_report_fixture());
}

#[test]
fn combine_measurements_scenario_study_accepts_solver_flag_for_paper_modes() {
    let scenarios_path = tmp_path("paper_scenarios.json");
    std::fs::write(
        &scenarios_path,
        r#"{
  "schema_version": "nextstat_measurement_combination_scenarios_v0",
  "scenarios": [
    {
      "name": "bjes_0p05",
      "error_on_error": [
        {
          "systematic": "b-JES",
          "value": 0.05
        }
      ]
    }
  ]
}"#,
    )
    .unwrap();

    let out = run(&[
        "combine-measurements-scenario-study",
        "--input",
        full_literature_fixture().to_string_lossy().as_ref(),
        "--scenarios",
        scenarios_path.to_string_lossy().as_ref(),
        "--solver",
        "analytic-perturbative",
        "--threads",
        "1",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("stdout should be JSON");
    assert_eq!(
        v["scenarios"][0]["result"]["optimizer"]["method"].as_str(),
        Some("analytic_perturbative_order_eps2")
    );
}

#[test]
fn combine_measurements_calibration_campaign_writes_valid_json() {
    let input = calibration_outlier_input_fixture();
    let scenarios = scenario_outlier_input_fixture();

    let out = run(&[
        "combine-measurements-calibration-campaign",
        "--input",
        input.to_string_lossy().as_ref(),
        "--scenarios",
        scenarios.to_string_lossy().as_ref(),
        "--threads",
        "1",
        "--n-toys",
        "16",
        "--seeds",
        "123,124,125",
    ]);
    assert!(
        out.status.success(),
        "combine-measurements-calibration-campaign should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("stdout should be JSON");
    assert_eq!(
        v.get("schema_version").and_then(|x| x.as_str()),
        Some("nextstat_measurement_combination_calibration_campaign_v0")
    );
    assert_eq!(v.get("stability").and_then(|x| x.as_str()), Some("research-grade"));
    let aggregate =
        v.get("aggregate").and_then(|x| x.as_object()).expect("aggregate should be present");
    assert_eq!(aggregate.get("n_scenarios").and_then(|x| x.as_u64()), Some(3));
    assert!(
        aggregate
            .get("max_calibration_mean_sigma_star_to_sigma_ratio")
            .and_then(|x| x.as_f64())
            .unwrap()
            .is_finite()
    );
}

#[test]
fn combine_measurements_calibration_campaign_matches_committed_artifact() {
    let input = calibration_outlier_input_fixture();
    let scenarios = scenario_outlier_input_fixture();

    let out = run(&[
        "combine-measurements-calibration-campaign",
        "--input",
        input.to_string_lossy().as_ref(),
        "--scenarios",
        scenarios.to_string_lossy().as_ref(),
        "--threads",
        "1",
        "--n-toys",
        "16",
        "--seeds",
        "123,124,125",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    assert_json_matches_fixture(&actual, &scenario_outlier_calibration_campaign_report_fixture());
}

#[test]
fn combine_measurements_calibration_campaign_accepts_solver_flag_for_paper_modes() {
    let scenarios_path = tmp_path("paper_campaign_scenarios.json");
    std::fs::write(
        &scenarios_path,
        r#"{
  "schema_version": "nextstat_measurement_combination_scenarios_v0",
  "scenarios": [
    {
      "name": "bjes_0p05",
      "error_on_error": [
        {
          "systematic": "b-JES",
          "value": 0.05
        }
      ]
    }
  ]
}"#,
    )
    .unwrap();

    let out = run(&[
        "combine-measurements-calibration-campaign",
        "--input",
        full_literature_fixture().to_string_lossy().as_ref(),
        "--scenarios",
        scenarios_path.to_string_lossy().as_ref(),
        "--solver",
        "analytic-perturbative",
        "--threads",
        "1",
        "--n-toys",
        "8",
        "--seeds",
        "2026,2027",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("stdout should be JSON");
    assert_eq!(
        v["scenarios"][0]["fit"]["optimizer"]["method"].as_str(),
        Some("analytic_perturbative_order_eps2")
    );
    assert_eq!(
        v["scenarios"][0]["calibration"]["toy_generation_method"].as_str(),
        Some("measurement_side_gvm_unbiased_sigma2_star_normalized_spec_original_theta")
    );
}

#[test]
fn combine_measurements_solver_parity_scenario_study_writes_valid_json() {
    let scenarios_path = tmp_path("paper_parity_scenarios.json");
    std::fs::write(
        &scenarios_path,
        r#"{
  "schema_version": "nextstat_measurement_combination_scenarios_v0",
  "scenarios": [
    {
      "name": "bjes_0p05",
      "error_on_error": [
        {
          "systematic": "b-JES",
          "value": 0.05
        }
      ]
    }
  ]
}"#,
    )
    .unwrap();

    let out = run(&[
        "combine-measurements-solver-parity-scenario-study",
        "--input",
        full_literature_fixture().to_string_lossy().as_ref(),
        "--scenarios",
        scenarios_path.to_string_lossy().as_ref(),
        "--lhs-solver",
        "numerical-paper",
        "--rhs-solver",
        "analytic-perturbative",
        "--threads",
        "1",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("stdout should be JSON");
    assert_eq!(
        v["schema_version"].as_str(),
        Some("nextstat_measurement_combination_scenario_study_solver_parity_v0")
    );
    assert_eq!(v["lhs_solver"].as_str(), Some("numerical-paper"));
    assert_eq!(v["rhs_solver"].as_str(), Some("analytic-perturbative"));
}

#[test]
fn combine_measurements_solver_parity_scenario_study_matches_committed_artifact() {
    let scenarios_path = tmp_path("paper_parity_scenarios_match.json");
    std::fs::write(
        &scenarios_path,
        r#"{
  "schema_version": "nextstat_measurement_combination_scenarios_v0",
  "scenarios": [
    {
      "name": "bjes_0p05",
      "error_on_error": [
        {
          "systematic": "b-JES",
          "value": 0.05
        }
      ]
    }
  ]
}"#,
    )
    .unwrap();

    let out = run(&[
        "combine-measurements-solver-parity-scenario-study",
        "--input",
        full_literature_fixture().to_string_lossy().as_ref(),
        "--scenarios",
        scenarios_path.to_string_lossy().as_ref(),
        "--lhs-solver",
        "numerical-paper",
        "--rhs-solver",
        "analytic-perturbative",
        "--threads",
        "1",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    assert_json_matches_fixture(&actual, &scenario_solver_parity_report_fixture());
}

#[test]
fn combine_measurements_solver_parity_scenario_study_from_reports_matches_committed_artifact() {
    let scenarios_path = tmp_path("paper_parity_scenarios_from_reports.json");
    std::fs::write(
        &scenarios_path,
        r#"{
  "schema_version": "nextstat_measurement_combination_scenarios_v0",
  "scenarios": [
    {
      "name": "bjes_0p05",
      "error_on_error": [
        {
          "systematic": "b-JES",
          "value": 0.05
        }
      ]
    }
  ]
}"#,
    )
    .unwrap();
    let lhs_report = tmp_path("paper_parity_scenario_lhs.json");
    let rhs_report = tmp_path("paper_parity_scenario_rhs.json");
    for (solver, path) in [("numerical-paper", &lhs_report), ("analytic-perturbative", &rhs_report)]
    {
        let out = run(&[
            "combine-measurements-scenario-study",
            "--input",
            full_literature_fixture().to_string_lossy().as_ref(),
            "--scenarios",
            scenarios_path.to_string_lossy().as_ref(),
            "--solver",
            solver,
            "--threads",
            "1",
            "--output",
            path.to_string_lossy().as_ref(),
        ]);
        assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    }

    let out = run(&[
        "combine-measurements-solver-parity-scenario-study-from-reports",
        "--lhs",
        lhs_report.to_string_lossy().as_ref(),
        "--rhs",
        rhs_report.to_string_lossy().as_ref(),
        "--lhs-solver",
        "numerical-paper",
        "--rhs-solver",
        "analytic-perturbative",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    assert_json_matches_fixture(&actual, &scenario_solver_parity_report_fixture());
}

#[test]
fn combine_measurements_solver_parity_scenario_study_markdown_matches_committed_artifact() {
    let scenarios_path = tmp_path("paper_parity_scenarios_markdown.json");
    std::fs::write(
        &scenarios_path,
        r#"{
  "schema_version": "nextstat_measurement_combination_scenarios_v0",
  "scenarios": [
    {
      "name": "bjes_0p05",
      "error_on_error": [
        {
          "systematic": "b-JES",
          "value": 0.05
        }
      ]
    }
  ]
}"#,
    )
    .unwrap();

    let out = run(&[
        "combine-measurements-solver-parity-scenario-study",
        "--input",
        full_literature_fixture().to_string_lossy().as_ref(),
        "--scenarios",
        scenarios_path.to_string_lossy().as_ref(),
        "--lhs-solver",
        "numerical-paper",
        "--rhs-solver",
        "analytic-perturbative",
        "--format",
        "markdown",
        "--threads",
        "1",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    let expected =
        std::fs::read_to_string(scenario_solver_parity_markdown_fixture()).expect("fixture");
    assert_markdown_close(&actual, &expected);
}

#[test]
fn combine_measurements_solver_parity_scenario_study_summarize_matches_fixture() {
    let scenarios_path = tmp_path("paper_parity_scenarios_summary.json");
    std::fs::write(
        &scenarios_path,
        r#"{
  "schema_version": "nextstat_measurement_combination_scenarios_v0",
  "scenarios": [
    {
      "name": "bjes_0p05",
      "error_on_error": [
        {
          "systematic": "b-JES",
          "value": 0.05
        }
      ]
    }
  ]
}"#,
    )
    .unwrap();
    let report_path = tmp_path("paper_parity_scenarios_report.json");
    let parity = run(&[
        "combine-measurements-solver-parity-scenario-study",
        "--input",
        full_literature_fixture().to_string_lossy().as_ref(),
        "--scenarios",
        scenarios_path.to_string_lossy().as_ref(),
        "--lhs-solver",
        "numerical-paper",
        "--rhs-solver",
        "analytic-perturbative",
        "--threads",
        "1",
        "--output",
        report_path.to_string_lossy().as_ref(),
    ]);
    assert!(parity.status.success(), "command failed: {}", String::from_utf8_lossy(&parity.stderr));

    let out = run(&[
        "combine-measurements-solver-parity-scenario-study-summarize",
        "--input",
        report_path.to_string_lossy().as_ref(),
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    assert_json_matches_fixture(&actual, &scenario_solver_parity_digest_fixture());
}

#[test]
fn combine_measurements_solver_parity_scenario_study_summarize_markdown_matches_fixture() {
    let scenarios_path = tmp_path("paper_parity_scenarios_summary_markdown.json");
    std::fs::write(
        &scenarios_path,
        r#"{
  "schema_version": "nextstat_measurement_combination_scenarios_v0",
  "scenarios": [
    {
      "name": "bjes_0p05",
      "error_on_error": [
        {
          "systematic": "b-JES",
          "value": 0.05
        }
      ]
    }
  ]
}"#,
    )
    .unwrap();
    let report_path = tmp_path("paper_parity_scenarios_report_markdown.json");
    let parity = run(&[
        "combine-measurements-solver-parity-scenario-study",
        "--input",
        full_literature_fixture().to_string_lossy().as_ref(),
        "--scenarios",
        scenarios_path.to_string_lossy().as_ref(),
        "--lhs-solver",
        "numerical-paper",
        "--rhs-solver",
        "analytic-perturbative",
        "--threads",
        "1",
        "--output",
        report_path.to_string_lossy().as_ref(),
    ]);
    assert!(parity.status.success(), "command failed: {}", String::from_utf8_lossy(&parity.stderr));

    let out = run(&[
        "combine-measurements-solver-parity-scenario-study-summarize",
        "--input",
        report_path.to_string_lossy().as_ref(),
        "--format",
        "markdown",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    let expected =
        std::fs::read_to_string(scenario_solver_parity_digest_markdown_fixture()).expect("fixture");
    assert_markdown_close(&actual, &expected);
}

#[test]
fn combine_measurements_solver_parity_calibration_campaign_writes_valid_json() {
    let scenarios_path = tmp_path("paper_parity_campaign_scenarios.json");
    std::fs::write(
        &scenarios_path,
        r#"{
  "schema_version": "nextstat_measurement_combination_scenarios_v0",
  "scenarios": [
    {
      "name": "bjes_0p05",
      "error_on_error": [
        {
          "systematic": "b-JES",
          "value": 0.05
        }
      ]
    }
  ]
}"#,
    )
    .unwrap();

    let out = run(&[
        "combine-measurements-solver-parity-calibration-campaign",
        "--input",
        full_literature_fixture().to_string_lossy().as_ref(),
        "--scenarios",
        scenarios_path.to_string_lossy().as_ref(),
        "--lhs-solver",
        "numerical-paper",
        "--rhs-solver",
        "analytic-perturbative",
        "--threads",
        "1",
        "--n-toys",
        "8",
        "--seeds",
        "2026,2027",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("stdout should be JSON");
    assert_eq!(
        v["schema_version"].as_str(),
        Some("nextstat_measurement_combination_calibration_campaign_solver_parity_v0")
    );
    assert_eq!(v["lhs_solver"].as_str(), Some("numerical-paper"));
    assert_eq!(v["rhs_solver"].as_str(), Some("analytic-perturbative"));
    assert!(v["aggregate"]["max_fit_sigma_rel_diff"].as_f64().unwrap().is_finite());
}

#[test]
fn combine_measurements_solver_parity_calibration_campaign_matches_committed_artifact() {
    let scenarios_path = tmp_path("paper_parity_campaign_scenarios_match.json");
    std::fs::write(
        &scenarios_path,
        r#"{
  "schema_version": "nextstat_measurement_combination_scenarios_v0",
  "scenarios": [
    {
      "name": "bjes_0p05",
      "error_on_error": [
        {
          "systematic": "b-JES",
          "value": 0.05
        }
      ]
    }
  ]
}"#,
    )
    .unwrap();

    let out = run(&[
        "combine-measurements-solver-parity-calibration-campaign",
        "--input",
        full_literature_fixture().to_string_lossy().as_ref(),
        "--scenarios",
        scenarios_path.to_string_lossy().as_ref(),
        "--lhs-solver",
        "numerical-paper",
        "--rhs-solver",
        "analytic-perturbative",
        "--threads",
        "1",
        "--n-toys",
        "8",
        "--seeds",
        "2026,2027",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    assert_json_matches_fixture(&actual, &calibration_campaign_solver_parity_report_fixture());
}

#[test]
fn combine_measurements_solver_parity_calibration_campaign_from_reports_matches_committed_artifact()
{
    let scenarios_path = tmp_path("paper_parity_campaign_from_reports.json");
    std::fs::write(
        &scenarios_path,
        r#"{
  "schema_version": "nextstat_measurement_combination_scenarios_v0",
  "scenarios": [
    {
      "name": "bjes_0p05",
      "error_on_error": [
        {
          "systematic": "b-JES",
          "value": 0.05
        }
      ]
    }
  ]
}"#,
    )
    .unwrap();
    let lhs_report = tmp_path("paper_parity_campaign_lhs.json");
    let rhs_report = tmp_path("paper_parity_campaign_rhs.json");
    for (solver, path) in [("numerical-paper", &lhs_report), ("analytic-perturbative", &rhs_report)]
    {
        let out = run(&[
            "combine-measurements-calibration-campaign",
            "--input",
            full_literature_fixture().to_string_lossy().as_ref(),
            "--scenarios",
            scenarios_path.to_string_lossy().as_ref(),
            "--solver",
            solver,
            "--threads",
            "1",
            "--n-toys",
            "8",
            "--seeds",
            "2026,2027",
            "--output",
            path.to_string_lossy().as_ref(),
        ]);
        assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    }

    let out = run(&[
        "combine-measurements-solver-parity-calibration-campaign-from-reports",
        "--lhs",
        lhs_report.to_string_lossy().as_ref(),
        "--rhs",
        rhs_report.to_string_lossy().as_ref(),
        "--lhs-solver",
        "numerical-paper",
        "--rhs-solver",
        "analytic-perturbative",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    assert_json_matches_fixture(&actual, &calibration_campaign_solver_parity_report_fixture());
}

#[test]
fn combine_measurements_solver_parity_calibration_campaign_markdown_matches_committed_artifact() {
    let scenarios_path = tmp_path("paper_parity_campaign_scenarios_markdown.json");
    std::fs::write(
        &scenarios_path,
        r#"{
  "schema_version": "nextstat_measurement_combination_scenarios_v0",
  "scenarios": [
    {
      "name": "bjes_0p05",
      "error_on_error": [
        {
          "systematic": "b-JES",
          "value": 0.05
        }
      ]
    }
  ]
}"#,
    )
    .unwrap();

    let out = run(&[
        "combine-measurements-solver-parity-calibration-campaign",
        "--input",
        full_literature_fixture().to_string_lossy().as_ref(),
        "--scenarios",
        scenarios_path.to_string_lossy().as_ref(),
        "--lhs-solver",
        "numerical-paper",
        "--rhs-solver",
        "analytic-perturbative",
        "--format",
        "markdown",
        "--threads",
        "1",
        "--n-toys",
        "8",
        "--seeds",
        "2026,2027",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    let expected = std::fs::read_to_string(calibration_campaign_solver_parity_markdown_fixture())
        .expect("fixture");
    assert_markdown_close(&actual, &expected);
}

#[test]
fn combine_measurements_solver_parity_calibration_campaign_summarize_matches_fixture() {
    let scenarios_path = tmp_path("paper_parity_campaign_summary.json");
    std::fs::write(
        &scenarios_path,
        r#"{
  "schema_version": "nextstat_measurement_combination_scenarios_v0",
  "scenarios": [
    {
      "name": "bjes_0p05",
      "error_on_error": [
        {
          "systematic": "b-JES",
          "value": 0.05
        }
      ]
    }
  ]
}"#,
    )
    .unwrap();
    let report_path = tmp_path("paper_parity_campaign_report_summary.json");
    let parity = run(&[
        "combine-measurements-solver-parity-calibration-campaign",
        "--input",
        full_literature_fixture().to_string_lossy().as_ref(),
        "--scenarios",
        scenarios_path.to_string_lossy().as_ref(),
        "--lhs-solver",
        "numerical-paper",
        "--rhs-solver",
        "analytic-perturbative",
        "--threads",
        "1",
        "--n-toys",
        "8",
        "--seeds",
        "2026,2027",
        "--output",
        report_path.to_string_lossy().as_ref(),
    ]);
    assert!(parity.status.success(), "command failed: {}", String::from_utf8_lossy(&parity.stderr));

    let out = run(&[
        "combine-measurements-solver-parity-calibration-campaign-summarize",
        "--input",
        report_path.to_string_lossy().as_ref(),
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    assert_json_matches_fixture(&actual, &calibration_campaign_solver_parity_digest_fixture());
}

#[test]
fn combine_measurements_solver_parity_calibration_campaign_summarize_markdown_matches_fixture() {
    let scenarios_path = tmp_path("paper_parity_campaign_summary_markdown.json");
    std::fs::write(
        &scenarios_path,
        r#"{
  "schema_version": "nextstat_measurement_combination_scenarios_v0",
  "scenarios": [
    {
      "name": "bjes_0p05",
      "error_on_error": [
        {
          "systematic": "b-JES",
          "value": 0.05
        }
      ]
    }
  ]
}"#,
    )
    .unwrap();
    let report_path = tmp_path("paper_parity_campaign_report_summary_markdown.json");
    let parity = run(&[
        "combine-measurements-solver-parity-calibration-campaign",
        "--input",
        full_literature_fixture().to_string_lossy().as_ref(),
        "--scenarios",
        scenarios_path.to_string_lossy().as_ref(),
        "--lhs-solver",
        "numerical-paper",
        "--rhs-solver",
        "analytic-perturbative",
        "--threads",
        "1",
        "--n-toys",
        "8",
        "--seeds",
        "2026,2027",
        "--output",
        report_path.to_string_lossy().as_ref(),
    ]);
    assert!(parity.status.success(), "command failed: {}", String::from_utf8_lossy(&parity.stderr));

    let out = run(&[
        "combine-measurements-solver-parity-calibration-campaign-summarize",
        "--input",
        report_path.to_string_lossy().as_ref(),
        "--format",
        "markdown",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    let expected =
        std::fs::read_to_string(calibration_campaign_solver_parity_digest_markdown_fixture())
            .expect("fixture");
    assert_markdown_close(&actual, &expected);
}

#[test]
fn combine_measurements_calibration_campaign_summarize_writes_valid_json() {
    let input = scenario_outlier_calibration_campaign_report_fixture();

    let out = run(&[
        "combine-measurements-calibration-campaign-summarize",
        "--input",
        input.to_string_lossy().as_ref(),
    ]);
    assert!(
        out.status.success(),
        "combine-measurements-calibration-campaign-summarize should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).expect("stdout should be JSON");
    assert_eq!(
        v.get("schema_version").and_then(|x| x.as_str()),
        Some("nextstat_measurement_combination_calibration_campaign_summary_v0")
    );
    assert_eq!(v.get("stability").and_then(|x| x.as_str()), Some("research-grade"));
    assert_eq!(v.get("dominant_calibration_scenario").and_then(|x| x.as_str()), Some("new_0p5"));
}

#[test]
fn combine_measurements_calibration_campaign_summarize_matches_committed_artifact() {
    let input = scenario_outlier_calibration_campaign_report_fixture();

    let out = run(&[
        "combine-measurements-calibration-campaign-summarize",
        "--input",
        input.to_string_lossy().as_ref(),
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    assert_json_matches_fixture(&actual, &scenario_outlier_calibration_campaign_summary_fixture());
}

#[test]
fn combine_measurements_calibration_campaign_summarize_markdown_matches_committed_artifact() {
    let input = scenario_outlier_calibration_campaign_report_fixture();

    let out = run(&[
        "combine-measurements-calibration-campaign-summarize",
        "--input",
        input.to_string_lossy().as_ref(),
        "--format",
        "markdown",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    let expected =
        std::fs::read_to_string(scenario_outlier_calibration_campaign_summary_markdown_fixture())
            .expect("markdown fixture should exist");
    assert_markdown_close(&actual, &expected);
}

#[test]
fn combine_measurements_calibration_campaign_brief_matches_committed_artifact() {
    let out = run(&[
        "combine-measurements-calibration-campaign-brief",
        "--input",
        scenario_outlier_calibration_campaign_summary_fixture().to_string_lossy().as_ref(),
        "--input",
        full_literature_calibration_campaign_summary_fixture().to_string_lossy().as_ref(),
        "--label",
        "outlier",
        "--label",
        "topmass_full",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    assert_json_matches_fixture(&actual, &calibration_campaign_brief_fixture());
}

#[test]
fn combine_measurements_calibration_campaign_brief_markdown_matches_committed_artifact() {
    let out = run(&[
        "combine-measurements-calibration-campaign-brief",
        "--input",
        scenario_outlier_calibration_campaign_summary_fixture().to_string_lossy().as_ref(),
        "--input",
        full_literature_calibration_campaign_summary_fixture().to_string_lossy().as_ref(),
        "--label",
        "outlier",
        "--label",
        "topmass_full",
        "--format",
        "markdown",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    let expected = std::fs::read_to_string(calibration_campaign_brief_markdown_fixture())
        .expect("markdown fixture should exist");
    assert_markdown_close(&actual, &expected);
}

#[test]
fn combine_measurements_calibration_campaign_brief_single_artifact_matches_committed_artifact() {
    let out = run(&[
        "combine-measurements-calibration-campaign-brief",
        "--input",
        full_literature_calibration_campaign_summary_fixture().to_string_lossy().as_ref(),
        "--label",
        "topmass_full",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    assert_json_matches_fixture(&actual, &calibration_campaign_topmass_only_brief_fixture());
}

#[test]
fn combine_measurements_calibration_campaign_family_report_matches_committed_artifact() {
    let out = run(&[
        "combine-measurements-calibration-campaign-family-report",
        "--input",
        calibration_campaign_brief_fixture().to_string_lossy().as_ref(),
        "--input",
        calibration_campaign_topmass_only_brief_fixture().to_string_lossy().as_ref(),
        "--label",
        "cross_fixture",
        "--label",
        "topmass_only",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    assert_json_matches_fixture(&actual, &calibration_campaign_family_report_fixture());
}

#[test]
fn combine_measurements_calibration_campaign_family_report_markdown_matches_committed_artifact() {
    let out = run(&[
        "combine-measurements-calibration-campaign-family-report",
        "--input",
        calibration_campaign_brief_fixture().to_string_lossy().as_ref(),
        "--input",
        calibration_campaign_topmass_only_brief_fixture().to_string_lossy().as_ref(),
        "--label",
        "cross_fixture",
        "--label",
        "topmass_only",
        "--format",
        "markdown",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    let expected = std::fs::read_to_string(calibration_campaign_family_report_markdown_fixture())
        .expect("markdown fixture should exist");
    assert_markdown_close(&actual, &expected);
}

#[test]
fn combine_measurements_calibration_campaign_family_matrix_matches_committed_artifact() {
    let out = run(&[
        "combine-measurements-calibration-campaign-family-matrix",
        "--input",
        calibration_campaign_family_report_fixture().to_string_lossy().as_ref(),
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    assert_json_matches_fixture(&actual, &calibration_campaign_family_matrix_fixture());
}

#[test]
fn combine_measurements_calibration_campaign_family_matrix_markdown_matches_committed_artifact() {
    let out = run(&[
        "combine-measurements-calibration-campaign-family-matrix",
        "--input",
        calibration_campaign_family_report_fixture().to_string_lossy().as_ref(),
        "--format",
        "markdown",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    let expected = std::fs::read_to_string(calibration_campaign_family_matrix_markdown_fixture())
        .expect("markdown fixture should exist");
    assert_markdown_close(&actual, &expected);
}

#[test]
fn combine_measurements_calibration_campaign_topmass_only_family_report_matches_committed_artifact()
{
    let out = run(&[
        "combine-measurements-calibration-campaign-family-report",
        "--input",
        calibration_campaign_topmass_only_brief_fixture().to_string_lossy().as_ref(),
        "--label",
        "topmass_only",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    assert_json_matches_fixture(
        &actual,
        &calibration_campaign_topmass_only_family_report_fixture(),
    );
}

#[test]
fn combine_measurements_calibration_campaign_topmass_only_family_matrix_matches_committed_artifact()
{
    let out = run(&[
        "combine-measurements-calibration-campaign-family-matrix",
        "--input",
        calibration_campaign_topmass_only_family_report_fixture().to_string_lossy().as_ref(),
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    assert_json_matches_fixture(
        &actual,
        &calibration_campaign_topmass_only_family_matrix_fixture(),
    );
}

#[test]
fn combine_measurements_calibration_campaign_portfolio_matches_committed_artifact() {
    let out = run(&[
        "combine-measurements-calibration-campaign-portfolio",
        "--input",
        calibration_campaign_family_matrix_fixture().to_string_lossy().as_ref(),
        "--input",
        calibration_campaign_topmass_only_family_matrix_fixture().to_string_lossy().as_ref(),
        "--label",
        "cross_portfolio",
        "--label",
        "topmass_only_portfolio",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    assert_json_matches_fixture(&actual, &calibration_campaign_portfolio_fixture());
}

#[test]
fn combine_measurements_calibration_campaign_portfolio_markdown_matches_committed_artifact() {
    let out = run(&[
        "combine-measurements-calibration-campaign-portfolio",
        "--input",
        calibration_campaign_family_matrix_fixture().to_string_lossy().as_ref(),
        "--input",
        calibration_campaign_topmass_only_family_matrix_fixture().to_string_lossy().as_ref(),
        "--label",
        "cross_portfolio",
        "--label",
        "topmass_only_portfolio",
        "--format",
        "markdown",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    let expected = std::fs::read_to_string(calibration_campaign_portfolio_markdown_fixture())
        .expect("markdown fixture should exist");
    assert_markdown_close(&actual, &expected);
}

#[test]
fn combine_measurements_calibration_campaign_portfolio_stability_matches_committed_artifact() {
    let out = run(&[
        "combine-measurements-calibration-campaign-portfolio-stability",
        "--input",
        calibration_campaign_portfolio_fixture().to_string_lossy().as_ref(),
        "--input",
        calibration_campaign_portfolio_fixture().to_string_lossy().as_ref(),
        "--label",
        "seedgrid_a",
        "--label",
        "seedgrid_b",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    assert_json_matches_fixture(&actual, &calibration_campaign_portfolio_stability_fixture());
}

#[test]
fn combine_measurements_calibration_campaign_portfolio_stability_markdown_matches_committed_artifact()
 {
    let out = run(&[
        "combine-measurements-calibration-campaign-portfolio-stability",
        "--input",
        calibration_campaign_portfolio_fixture().to_string_lossy().as_ref(),
        "--input",
        calibration_campaign_portfolio_fixture().to_string_lossy().as_ref(),
        "--label",
        "seedgrid_a",
        "--label",
        "seedgrid_b",
        "--format",
        "markdown",
    ]);
    assert!(out.status.success(), "command failed: {}", String::from_utf8_lossy(&out.stderr));
    let actual = String::from_utf8(out.stdout).expect("stdout should be UTF-8");
    let expected =
        std::fs::read_to_string(calibration_campaign_portfolio_stability_markdown_fixture())
            .expect("markdown fixture should exist");
    assert_markdown_close(&actual, &expected);
}
