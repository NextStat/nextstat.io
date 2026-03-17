use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("repo root should exist")
}

fn fixture_path(name: &str) -> PathBuf {
    repo_root().join("tests").join("fixtures").join(name)
}

fn doc_spec_path(name: &str) -> PathBuf {
    repo_root().join("docs").join("specs").join("hep").join(name)
}

fn tmp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!(
        "nextstat_cli_simplified_{}_{}_{}",
        std::process::id(),
        nanos,
        name
    ))
}

fn run_nextstat_json(args: &[&str]) -> serde_json::Value {
    let exe = env!("CARGO_BIN_EXE_nextstat");
    let out = Command::new(exe).args(args).output().expect("failed to run nextstat");

    if !out.status.success() {
        panic!(
            "nextstat failed: status={}\nstdout:\n{}\nstderr:\n{}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
    }

    serde_json::from_slice(&out.stdout).expect("nextstat should emit valid JSON")
}

fn run_nextstat_text(args: &[&str]) -> String {
    let exe = env!("CARGO_BIN_EXE_nextstat");
    let out = Command::new(exe).args(args).output().expect("failed to run nextstat");

    if !out.status.success() {
        panic!(
            "nextstat failed: status={}\nstdout:\n{}\nstderr:\n{}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
    }

    String::from_utf8(out.stdout).expect("nextstat should emit UTF-8 text")
}

fn derive_config_json_with_selection(
    max_components: Option<usize>,
    explained_variance_target: f64,
    random_draws: usize,
    channels: &[&str],
    bins: &[&str],
    constraint_covariance_source: &str,
) -> String {
    let mut reduction = serde_json::json!({
        "output_uncertainty_model": "basis",
        "basis_method": "eigen",
        "explained_variance_target": explained_variance_target,
        "constraint_covariance_source": constraint_covariance_source,
        "split_stat_covariance": true,
    });
    if let Some(value) = max_components {
        reduction["max_components"] = serde_json::json!(value);
    }
    serde_json::to_string_pretty(&serde_json::json!({
        "schema_version": "nextstat_simplified_likelihood_derive_v0",
        "source_workspace": {
            "format": "pyhf",
            "schema_version": "pyhf_workspace_v1",
            "poi_name": "mu"
        },
        "fit_result": {
            "schema_version": "nextstat_fit_result_v0",
            "background_state": "postfit_background"
        },
        "selection": {
            "channels": channels,
            "bins": bins,
        },
        "reduction": reduction,
        "jacobian": {
            "method": "finite_difference",
            "relative_step": 0.01,
            "absolute_step_floor": 0.000001
        },
        "fidelity_smoke": {
            "random_draws": random_draws,
            "qmu_test_mu": 1.0,
            "upper_limit_cl": 0.95
        },
        "output_contract": {
            "schema_version": "nextstat_simplified_likelihood_v0",
            "require_factorization_diagnostics": true,
            "require_fidelity_diagnostics": true
        }
    }))
    .expect("derive config should serialize")
}

fn derive_config_json(
    max_components: Option<usize>,
    explained_variance_target: f64,
    random_draws: usize,
) -> String {
    derive_config_json_with_selection(
        max_components,
        explained_variance_target,
        random_draws,
        &["singlechannel"],
        &["singlechannel/bin0", "singlechannel/bin1"],
        "aligned_fit_covariance",
    )
}

fn export_derived_simplified_workspace(
    name: &str,
    max_components: Option<usize>,
    explained_variance_target: f64,
    random_draws: usize,
) -> (PathBuf, PathBuf, serde_json::Value) {
    let (work_dir, output_path, _report_path, simplified, _report) =
        export_derived_simplified_workspace_with_optional_report(
            name,
            max_components,
            explained_variance_target,
            random_draws,
            false,
        );
    (work_dir, output_path, simplified)
}

fn export_derived_simplified_workspace_with_report(
    name: &str,
    max_components: Option<usize>,
    explained_variance_target: f64,
    random_draws: usize,
) -> (PathBuf, PathBuf, PathBuf, serde_json::Value, serde_json::Value) {
    let (work_dir, output_path, report_path, simplified, report) =
        export_derived_simplified_workspace_with_optional_report(
            name,
            max_components,
            explained_variance_target,
            random_draws,
            true,
        );
    (
        work_dir,
        output_path,
        report_path.expect("report path should be present"),
        simplified,
        report.expect("report JSON should be present"),
    )
}

fn export_derived_simplified_workspace_with_optional_report(
    name: &str,
    max_components: Option<usize>,
    explained_variance_target: f64,
    random_draws: usize,
    emit_report: bool,
) -> (PathBuf, PathBuf, Option<PathBuf>, serde_json::Value, Option<serde_json::Value>) {
    let input = fixture_path("simple_workspace.json");
    let work_dir = tmp_dir(name);
    let _ = std::fs::remove_dir_all(&work_dir);
    std::fs::create_dir_all(&work_dir).expect("temp dir should be creatable");

    let fit_path = work_dir.join("fit.json");
    let derive_config_path = work_dir.join("derive.json");
    let output_path = work_dir.join("simplified.json");
    let report_path = work_dir.join("simplified_export_report.json");

    let fit_status = Command::new(env!("CARGO_BIN_EXE_nextstat"))
        .args([
            "fit",
            "--input",
            input.to_string_lossy().as_ref(),
            "--output",
            fit_path.to_string_lossy().as_ref(),
            "--threads",
            "1",
        ])
        .output()
        .expect("fit command should run");
    assert!(
        fit_status.status.success(),
        "fit should succeed before simplify, stderr={}",
        String::from_utf8_lossy(&fit_status.stderr)
    );

    std::fs::write(
        &derive_config_path,
        derive_config_json(max_components, explained_variance_target, random_draws),
    )
    .expect("derive config should be writable");

    let mut args = vec![
        "simplify".to_string(),
        "workspace".to_string(),
        "--input".to_string(),
        input.to_string_lossy().into_owned(),
        "--fit".to_string(),
        fit_path.to_string_lossy().into_owned(),
        "--derive-config".to_string(),
        derive_config_path.to_string_lossy().into_owned(),
        "--experiment".to_string(),
        "ATLAS".to_string(),
        "--analysis-id".to_string(),
        name.to_string(),
        "--reference".to_string(),
        "internal-test".to_string(),
        "--output".to_string(),
        output_path.to_string_lossy().into_owned(),
    ];
    if emit_report {
        args.push("--report".to_string());
        args.push(report_path.to_string_lossy().into_owned());
    }
    args.push("--threads".to_string());
    args.push("1".to_string());

    let out = Command::new(env!("CARGO_BIN_EXE_nextstat"))
        .args(&args)
        .output()
        .expect("simplify command should run");
    assert!(
        out.status.success(),
        "simplify workspace should succeed, stdout=\n{}\nstderr=\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let simplified: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&output_path).expect("simplified output should be written"),
    )
    .expect("simplified output should be valid JSON");
    let report = if emit_report {
        Some(
            serde_json::from_slice(
                &std::fs::read(&report_path).expect("export report should be written"),
            )
            .expect("export report should be valid JSON"),
        )
    } else {
        None
    };
    (work_dir, output_path, if emit_report { Some(report_path) } else { None }, simplified, report)
}

fn export_public_style_simplified_workspace_with_report(
    name: &str,
    fixture_name: &str,
) -> (PathBuf, PathBuf, PathBuf, serde_json::Value, serde_json::Value) {
    let input = fixture_path(fixture_name);
    let work_dir = tmp_dir(name);
    let _ = std::fs::remove_dir_all(&work_dir);
    std::fs::create_dir_all(&work_dir).expect("temp dir should be creatable");

    let fit_path = work_dir.join("fit.json");
    let derive_config_path = work_dir.join("derive.json");
    let output_path = work_dir.join("simplified.json");
    let report_path = work_dir.join("simplified_export_report.json");

    let fit_status = Command::new(env!("CARGO_BIN_EXE_nextstat"))
        .args([
            "fit",
            "--input",
            input.to_string_lossy().as_ref(),
            "--output",
            fit_path.to_string_lossy().as_ref(),
            "--threads",
            "1",
        ])
        .output()
        .expect("fit command should run");
    assert!(
        fit_status.status.success(),
        "fit should succeed before simplify, stderr={}",
        String::from_utf8_lossy(&fit_status.stderr)
    );

    std::fs::write(
        &derive_config_path,
        derive_config_json_with_selection(
            Some(2),
            0.999,
            8,
            &["SR", "CR"],
            &["SR/bin0", "SR/bin1", "CR/bin0", "CR/bin1"],
            "source_model_constraints",
        ),
    )
    .expect("derive config should be writable");

    let out = Command::new(env!("CARGO_BIN_EXE_nextstat"))
        .args([
            "simplify",
            "workspace",
            "--input",
            input.to_string_lossy().as_ref(),
            "--fit",
            fit_path.to_string_lossy().as_ref(),
            "--derive-config",
            derive_config_path.to_string_lossy().as_ref(),
            "--experiment",
            "CMS",
            "--analysis-id",
            name,
            "--reference",
            "internal-test",
            "--output",
            output_path.to_string_lossy().as_ref(),
            "--report",
            report_path.to_string_lossy().as_ref(),
            "--threads",
            "1",
        ])
        .output()
        .expect("simplify command should run");
    assert!(
        out.status.success(),
        "simplify workspace should succeed, stdout=\n{}\nstderr=\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let simplified: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&output_path).expect("simplified output should be written"),
    )
    .expect("simplified output should be valid JSON");
    let report: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&report_path).expect("export report should be written"),
    )
    .expect("export report should be valid JSON");
    (work_dir, output_path, report_path, simplified, report)
}

#[test]
fn audit_accepts_basis_simplified_likelihood_input() {
    let input = fixture_path("sl_basis_two_bin.json");
    let input = input.to_string_lossy().to_string();

    let audit = run_nextstat_json(&["audit", "--input", &input, "--format", "json"]);

    assert_eq!(
        audit.get("schema_version").and_then(|v| v.as_str()),
        Some("nextstat_simplified_likelihood_audit_v0")
    );
    assert_eq!(
        audit.get("input_schema_version").and_then(|v| v.as_str()),
        Some("nextstat_simplified_likelihood_v0")
    );
    assert_eq!(audit.get("uncertainty_model_kind").and_then(|v| v.as_str()), Some("basis"));
    assert_eq!(audit.get("reduced_nuisance_count").and_then(|v| v.as_u64()), Some(1));
    assert_eq!(audit.get("parameter_count_estimate").and_then(|v| v.as_u64()), Some(2));
}

#[test]
fn audit_accepts_covariance_simplified_likelihood_input_and_reports_factorization() {
    let input = fixture_path("sl_covariance_three_bin.json");
    let input = input.to_string_lossy().to_string();

    let audit = run_nextstat_json(&["audit", "--input", &input, "--format", "json"]);

    assert_eq!(
        audit.get("schema_version").and_then(|v| v.as_str()),
        Some("nextstat_simplified_likelihood_audit_v0")
    );
    assert_eq!(
        audit.get("input_schema_version").and_then(|v| v.as_str()),
        Some("nextstat_simplified_likelihood_v0")
    );
    assert_eq!(audit.get("uncertainty_model_kind").and_then(|v| v.as_str()), Some("covariance"));
    assert_eq!(audit.get("reduced_nuisance_count").and_then(|v| v.as_u64()), Some(3));
    let factorization = audit
        .get("diagnostics")
        .and_then(|v| v.get("factorization"))
        .expect("diagnostics.factorization should be present");
    assert_eq!(factorization.get("retained_rank").and_then(|v| v.as_u64()), Some(3));
    assert_eq!(
        factorization.get("method").and_then(|v| v.as_str()),
        Some("symmetric_eigendecomposition")
    );
}

#[test]
fn audit_text_reports_simplified_factorization_summary() {
    let input = fixture_path("sl_covariance_three_bin.json");
    let input = input.to_string_lossy().to_string();

    let audit = run_nextstat_text(&["audit", "--input", &input]);

    assert!(audit.contains("Simplified likelihood:"));
    assert!(audit.contains("Uncertainty model: covariance"));
    assert!(audit.contains("Factorization:"));
    assert!(audit.contains("Method: symmetric_eigendecomposition"));
    assert!(audit.contains("Rank: 3 -> 3"));
}

#[test]
fn audit_accepts_derived_simplified_likelihood_example_and_reports_fidelity() {
    let input = doc_spec_path("simplified_likelihood_derived_from_workspace_v0.example.json");
    let input = input.to_string_lossy().to_string();

    let audit = run_nextstat_json(&["audit", "--input", &input, "--format", "json"]);

    assert_eq!(audit.get("source_format").and_then(|v| v.as_str()), Some("derived_from_workspace"));
    let fidelity = audit
        .get("diagnostics")
        .and_then(|v| v.get("fidelity"))
        .expect("diagnostics.fidelity should be present");
    assert_eq!(fidelity.get("nuisance_count_full").and_then(|v| v.as_u64()), Some(12));
    assert_eq!(fidelity.get("nuisance_count_reduced").and_then(|v| v.as_u64()), Some(2));
}

#[test]
fn fit_accepts_simplified_likelihood_input() {
    let input = fixture_path("sl_basis_two_bin.json");
    let input = input.to_string_lossy().to_string();

    let fit = run_nextstat_json(&["fit", "--input", &input, "--threads", "1"]);
    let names = fit
        .get("parameter_names")
        .and_then(|v| v.as_array())
        .expect("fit output should contain parameter_names");

    assert_eq!(names.len(), 2, "expected POI plus one reduced nuisance");
    assert_eq!(names[0].as_str(), Some("mu"));
    assert_eq!(names[1].as_str(), Some("sl_np_000"));
    assert!(fit.get("converged").and_then(|v| v.as_bool()).unwrap_or(false));
}

#[test]
fn fit_accepts_covariance_simplified_likelihood_input() {
    let input = fixture_path("sl_covariance_three_bin.json");
    let input = input.to_string_lossy().to_string();

    let fit = run_nextstat_json(&["fit", "--input", &input, "--threads", "1"]);
    let names = fit
        .get("parameter_names")
        .and_then(|v| v.as_array())
        .expect("fit output should contain parameter_names");

    assert_eq!(names[0].as_str(), Some("mu"));
    assert_eq!(names.len(), 4, "expected POI plus rank-3 covariance basis");
    assert!(fit.get("converged").and_then(|v| v.as_bool()).unwrap_or(false));
}

#[test]
fn upper_limit_accepts_simplified_likelihood_input() {
    let input = fixture_path("sl_basis_two_bin.json");
    let input = input.to_string_lossy().to_string();

    let out =
        run_nextstat_json(&["upper-limit", "--input", &input, "--expected", "--threads", "1"]);

    assert!(out.get("obs_limit").and_then(|v| v.as_f64()).unwrap().is_finite());
    let exp_limits = out
        .get("exp_limits")
        .and_then(|v| v.as_array())
        .expect("upper-limit output should contain exp_limits");
    assert_eq!(exp_limits.len(), 5);
}

#[test]
fn upper_limit_accepts_covariance_simplified_likelihood_input() {
    let input = fixture_path("sl_covariance_three_bin.json");
    let input = input.to_string_lossy().to_string();

    let out =
        run_nextstat_json(&["upper-limit", "--input", &input, "--expected", "--threads", "1"]);

    assert!(out.get("obs_limit").and_then(|v| v.as_f64()).unwrap().is_finite());
    let exp_limits = out
        .get("exp_limits")
        .and_then(|v| v.as_array())
        .expect("upper-limit output should contain exp_limits");
    assert_eq!(exp_limits.len(), 5);
}

#[test]
fn simplify_workspace_emits_derived_simplified_likelihood_artifact() {
    let (work_dir, _output_path, simplified) =
        export_derived_simplified_workspace("cli-simple-derived", Some(1), 0.9, 8);
    assert_eq!(
        simplified.get("schema_version").and_then(|v| v.as_str()),
        Some("nextstat_simplified_likelihood_v0")
    );
    assert_eq!(
        simplified.get("metadata").and_then(|v| v.get("source_format")).and_then(|v| v.as_str()),
        Some("derived_from_workspace")
    );
    assert_eq!(
        simplified
            .get("uncertainty_model")
            .and_then(|v| v.get("components"))
            .and_then(|v| v.as_array())
            .map(|v| v.len()),
        Some(1)
    );
    assert!(simplified.get("diagnostics").and_then(|v| v.get("factorization")).is_some());
    assert!(simplified.get("diagnostics").and_then(|v| v.get("fidelity")).is_some());

    let _ = std::fs::remove_dir_all(&work_dir);
}

#[test]
fn simplify_workspace_emits_export_report_artifact() {
    let (work_dir, output_path, _report_path, simplified, report) =
        export_derived_simplified_workspace_with_report(
            "cli-simple-derived-report",
            Some(1),
            0.9,
            8,
        );
    let simplified_bytes =
        std::fs::read(&output_path).expect("simplified artifact should have been written");
    let simplified_digest = format!("{:x}", Sha256::digest(&simplified_bytes));

    assert_eq!(
        report.get("schema_version").and_then(|v| v.as_str()),
        Some("nextstat_simplified_likelihood_export_report_v0")
    );
    assert_eq!(report.get("status").and_then(|v| v.as_str()), Some("ok"));
    assert_eq!(report.get("support_class").and_then(|v| v.as_str()), Some("research-grade"));
    assert_eq!(
        report.get("source").and_then(|v| v.get("workspace_format")).and_then(|v| v.as_str()),
        Some("pyhf")
    );
    assert_eq!(
        report.get("metadata").and_then(|v| v.get("analysis_id")).and_then(|v| v.as_str()),
        Some("cli-simple-derived-report")
    );
    assert_eq!(
        report.get("output").and_then(|v| v.get("schema_version")).and_then(|v| v.as_str()),
        Some("nextstat_simplified_likelihood_v0")
    );
    assert_eq!(
        report.get("output").and_then(|v| v.get("json_bytes")).and_then(|v| v.as_u64()),
        Some(simplified_bytes.len() as u64)
    );
    assert_eq!(
        report.get("output").and_then(|v| v.get("json_sha256")).and_then(|v| v.as_str()),
        Some(simplified_digest.as_str())
    );
    assert_eq!(
        report.get("output").and_then(|v| v.get("bins_count")).and_then(|v| v.as_u64()),
        Some(
            simplified
                .get("bins")
                .and_then(|v| v.as_array())
                .expect("simplified workspace should contain bins")
                .len() as u64
        )
    );
    assert_eq!(
        report.get("output").and_then(|v| v.get("reduced_nuisance_count")).and_then(|v| v.as_u64()),
        Some(
            simplified
                .get("uncertainty_model")
                .and_then(|v| v.get("components"))
                .and_then(|v| v.as_array())
                .expect("derived simplified workspace should expose basis components")
                .len() as u64
        )
    );
    assert!(report.get("diagnostics").and_then(|v| v.get("factorization")).is_some());
    assert!(report.get("diagnostics").and_then(|v| v.get("fidelity")).is_some());
    assert!(report.get("explicit_boundaries").and_then(|v| v.as_array()).is_some_and(|entries| {
        entries.iter().any(|entry| {
            entry.as_str().is_some_and(|value| value.contains("pyhf") || value.contains("partial"))
        })
    }));

    let _ = std::fs::remove_dir_all(&work_dir);
}

#[test]
fn simplify_workspace_round_trips_derived_artifact_through_runtime_surface() {
    let (work_dir, output_path, simplified) =
        export_derived_simplified_workspace("cli-simple-derived-roundtrip", Some(2), 0.999, 8);
    let input = output_path.to_string_lossy().to_string();

    let reduced_nuisance_count = simplified
        .get("diagnostics")
        .and_then(|v| v.get("fidelity"))
        .and_then(|v| v.get("nuisance_count_reduced"))
        .and_then(|v| v.as_u64())
        .expect("derived fidelity should expose nuisance_count_reduced");

    let audit = run_nextstat_json(&["audit", "--input", &input, "--format", "json"]);
    assert_eq!(audit.get("source_format").and_then(|v| v.as_str()), Some("derived_from_workspace"));
    assert_eq!(
        audit.get("reduced_nuisance_count").and_then(|v| v.as_u64()),
        Some(reduced_nuisance_count)
    );
    assert!(audit.get("diagnostics").and_then(|v| v.get("fidelity")).is_some());

    let fit = run_nextstat_json(&["fit", "--input", &input, "--threads", "1"]);
    let names = fit
        .get("parameter_names")
        .and_then(|v| v.as_array())
        .expect("fit output should contain parameter_names");
    assert_eq!(names[0].as_str(), Some("mu"));
    assert_eq!(names.len() as u64, reduced_nuisance_count + 1);
    assert!(fit.get("converged").and_then(|v| v.as_bool()).unwrap_or(false));

    let upper_limit =
        run_nextstat_json(&["upper-limit", "--input", &input, "--expected", "--threads", "1"]);
    assert!(upper_limit.get("obs_limit").and_then(|v| v.as_f64()).unwrap().is_finite());
    let exp_limits = upper_limit
        .get("exp_limits")
        .and_then(|v| v.as_array())
        .expect("upper-limit output should contain exp_limits");
    assert_eq!(exp_limits.len(), 5);

    let scan = run_nextstat_json(&[
        "scan",
        "--input",
        &input,
        "--start",
        "0.0",
        "--stop",
        "2.0",
        "--points",
        "5",
        "--threads",
        "1",
    ]);
    assert!(scan.get("mu_hat").and_then(|v| v.as_f64()).unwrap().is_finite());
    let points =
        scan.get("points").and_then(|v| v.as_array()).expect("scan output should contain points");
    assert_eq!(points.len(), 5);
    assert!(points.iter().all(|point| {
        point.get("mu").and_then(|v| v.as_f64()).unwrap().is_finite()
            && point.get("q_mu").and_then(|v| v.as_f64()).unwrap().is_finite()
    }));

    let _ = std::fs::remove_dir_all(&work_dir);
}

#[test]
fn simplify_workspace_emits_bounded_fidelity_diagnostics_for_high_fidelity_export() {
    let (work_dir, _output_path, simplified) =
        export_derived_simplified_workspace("cli-simple-derived-fidelity", Some(2), 0.999, 8);
    let fidelity = simplified
        .get("diagnostics")
        .and_then(|v| v.get("fidelity"))
        .expect("derived export should carry fidelity diagnostics");

    assert_eq!(fidelity.get("nuisance_count_full").and_then(|v| v.as_u64()), Some(2));
    assert_eq!(fidelity.get("nuisance_count_reduced").and_then(|v| v.as_u64()), Some(2));
    assert_eq!(fidelity.get("bins_count").and_then(|v| v.as_u64()), Some(2));

    let relative_background_cov_residual = fidelity
        .get("relative_background_cov_residual")
        .and_then(|v| v.as_f64())
        .expect("fidelity.relative_background_cov_residual should be f64");
    let max_abs_expected_delta_at_nominal = fidelity
        .get("max_abs_expected_delta_at_nominal")
        .and_then(|v| v.as_f64())
        .expect("fidelity.max_abs_expected_delta_at_nominal should be f64");
    let max_abs_expected_delta_random_draws = fidelity
        .get("max_abs_expected_delta_random_draws")
        .and_then(|v| v.as_f64())
        .expect("fidelity.max_abs_expected_delta_random_draws should be f64");
    let qmu_delta_smoke = fidelity
        .get("qmu_delta_smoke")
        .and_then(|v| v.as_f64())
        .expect("fidelity.qmu_delta_smoke should be f64");
    let upper_limit_ratio_smoke = fidelity
        .get("upper_limit_ratio_smoke")
        .and_then(|v| v.as_f64())
        .expect("fidelity.upper_limit_ratio_smoke should be f64");
    let max_abs_yield_delta = fidelity
        .get("max_abs_yield_delta")
        .and_then(|v| v.as_f64())
        .expect("fidelity.max_abs_yield_delta should be f64");
    let max_rel_yield_delta = fidelity
        .get("max_rel_yield_delta")
        .and_then(|v| v.as_f64())
        .expect("fidelity.max_rel_yield_delta should be f64");

    assert!(relative_background_cov_residual.is_finite());
    assert!(max_abs_expected_delta_at_nominal.is_finite());
    assert!(max_abs_expected_delta_random_draws.is_finite());
    assert!(qmu_delta_smoke.is_finite());
    assert!(upper_limit_ratio_smoke.is_finite());
    assert!(max_abs_yield_delta.is_finite());
    assert!(max_rel_yield_delta.is_finite());

    assert!(
        relative_background_cov_residual <= 1e-10,
        "expected full-rank export to preserve covariance, got residual={relative_background_cov_residual}"
    );
    assert!(
        qmu_delta_smoke <= 0.05,
        "expected q_mu fidelity smoke to stay within stable reinterpretation envelope, got {qmu_delta_smoke}"
    );
    assert!(
        (0.95..=1.05).contains(&upper_limit_ratio_smoke),
        "expected upper-limit fidelity ratio near unity, got {upper_limit_ratio_smoke}"
    );
    assert!(
        max_abs_expected_delta_at_nominal <= 1.0,
        "expected nominal yield drift to stay bounded, got {max_abs_expected_delta_at_nominal}"
    );
    assert!(
        max_abs_expected_delta_random_draws <= 1.0,
        "expected random-draw yield drift to stay bounded, got {max_abs_expected_delta_random_draws}"
    );
    assert!(
        max_abs_yield_delta <= 1.0,
        "expected overall absolute yield drift to stay bounded, got {max_abs_yield_delta}"
    );
    assert!(
        max_rel_yield_delta <= 0.02,
        "expected overall relative yield drift to stay bounded, got {max_rel_yield_delta}"
    );

    let _ = std::fs::remove_dir_all(&work_dir);
}

#[test]
fn simplify_workspace_keeps_public_export_smoke_fidelity_within_apex2_gate() {
    let (work_dir, _output_path, _report_path, simplified, report) =
        export_public_style_simplified_workspace_with_report(
            "cli-public-export-fidelity",
            "cms_public_sr_cr_gaussian_workspace.json",
        );

    let fidelity = simplified
        .get("diagnostics")
        .and_then(|v| v.get("fidelity"))
        .expect("derived export should carry fidelity diagnostics");
    let report_fidelity = report
        .get("diagnostics")
        .and_then(|v| v.get("fidelity"))
        .expect("export report should carry fidelity diagnostics");

    let qmu_delta_smoke = fidelity
        .get("qmu_delta_smoke")
        .and_then(|v| v.as_f64())
        .expect("fidelity.qmu_delta_smoke should be f64");
    let upper_limit_ratio_smoke = fidelity
        .get("upper_limit_ratio_smoke")
        .and_then(|v| v.as_f64())
        .expect("fidelity.upper_limit_ratio_smoke should be f64");

    assert!(
        qmu_delta_smoke <= 0.1,
        "expected public export q_mu smoke to stay inside Apex2 gate, got {qmu_delta_smoke}"
    );
    assert!(
        (0.95..=1.05).contains(&upper_limit_ratio_smoke),
        "expected public export upper-limit smoke ratio inside Apex2 gate, got {upper_limit_ratio_smoke}"
    );
    assert_eq!(
        report_fidelity.get("qmu_delta_smoke").and_then(|v| v.as_f64()),
        Some(qmu_delta_smoke)
    );
    assert_eq!(
        report_fidelity.get("upper_limit_ratio_smoke").and_then(|v| v.as_f64()),
        Some(upper_limit_ratio_smoke)
    );

    let _ = std::fs::remove_dir_all(&work_dir);
}

#[test]
fn ranking_on_covariance_simplified_likelihood_warns_about_reduced_semantics() {
    let input = fixture_path("sl_covariance_three_bin.json");
    let out = Command::new(env!("CARGO_BIN_EXE_nextstat"))
        .args(["viz", "ranking", "--input", input.to_string_lossy().as_ref(), "--threads", "1"])
        .output()
        .expect("viz ranking should run");
    assert!(
        out.status.success(),
        "viz ranking should succeed on covariance simplified-likelihood, stdout=\n{}\nstderr=\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let ranking: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("ranking output should be valid JSON");
    let names =
        ranking.get("names").and_then(|v| v.as_array()).expect("ranking.names should be present");
    assert!(!names.is_empty(), "ranking.names should not be empty");
    assert!(
        names
            .iter()
            .all(|name| { name.as_str().is_some_and(|value| value.starts_with("sl_cov_")) })
    );

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("reduced nuisance coordinates"));
    assert!(stderr.contains("covariance-form simplified-likelihood"));
}

#[test]
fn ranking_on_derived_simplified_likelihood_warns_about_source_semantics_boundary() {
    let (work_dir, output_path, _simplified) =
        export_derived_simplified_workspace("cli-simple-derived-ranking", Some(1), 0.9, 8);
    let out = Command::new(env!("CARGO_BIN_EXE_nextstat"))
        .args([
            "viz",
            "ranking",
            "--input",
            output_path.to_string_lossy().as_ref(),
            "--threads",
            "1",
        ])
        .output()
        .expect("viz ranking should run");
    assert!(
        out.status.success(),
        "viz ranking should succeed on derived simplified-likelihood, stdout=\n{}\nstderr=\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let ranking: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("ranking output should be valid JSON");
    let names =
        ranking.get("names").and_then(|v| v.as_array()).expect("ranking.names should be present");
    assert!(!names.is_empty(), "ranking.names should not be empty");
    assert!(
        names.iter().all(|name| { name.as_str().is_some_and(|value| value.starts_with("sl_np_")) })
    );

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("reduced nuisance coordinates"));
    assert!(stderr.contains("derived_from_workspace simplified-likelihood v0"));

    let _ = std::fs::remove_dir_all(&work_dir);
}

#[test]
fn simplify_workspace_rejects_fit_result_without_covariance() {
    let input = fixture_path("simple_workspace.json");
    let work_dir = tmp_dir("simplify_workspace_missing_covariance");
    let _ = std::fs::remove_dir_all(&work_dir);
    std::fs::create_dir_all(&work_dir).expect("temp dir should be creatable");

    let fit_path = work_dir.join("fit_missing_covariance.json");
    let derive_config_path = work_dir.join("derive.json");

    std::fs::write(
        &fit_path,
        r#"{
  "parameter_names": ["mu", "uncorr_bkguncrt[0]", "uncorr_bkguncrt[1]"],
  "bestfit": [0.0, 1.0, 1.0]
}"#,
    )
    .expect("fit fixture should be writable");
    std::fs::write(
        &derive_config_path,
        r#"{
  "schema_version": "nextstat_simplified_likelihood_derive_v0",
  "source_workspace": {
    "format": "pyhf",
    "schema_version": "pyhf_workspace_v1",
    "poi_name": "mu"
  },
  "fit_result": {
    "schema_version": "nextstat_fit_result_v0",
    "background_state": "postfit_background"
  },
  "selection": {
    "channels": ["singlechannel"],
    "bins": ["singlechannel/bin0", "singlechannel/bin1"]
  },
  "reduction": {
    "output_uncertainty_model": "basis",
    "basis_method": "eigen",
    "explained_variance_target": 0.95,
    "constraint_covariance_source": "aligned_fit_covariance",
    "split_stat_covariance": true
  },
  "jacobian": {
    "method": "finite_difference",
    "relative_step": 0.01,
    "absolute_step_floor": 0.000001
  },
  "fidelity_smoke": {
    "random_draws": 4,
    "qmu_test_mu": 1.0,
    "upper_limit_cl": 0.95
  },
  "output_contract": {
    "schema_version": "nextstat_simplified_likelihood_v0",
    "require_factorization_diagnostics": true,
    "require_fidelity_diagnostics": true
  }
}"#,
    )
    .expect("derive config should be writable");

    let out = Command::new(env!("CARGO_BIN_EXE_nextstat"))
        .args([
            "simplify",
            "workspace",
            "--input",
            input.to_string_lossy().as_ref(),
            "--fit",
            fit_path.to_string_lossy().as_ref(),
            "--derive-config",
            derive_config_path.to_string_lossy().as_ref(),
            "--experiment",
            "ATLAS",
            "--analysis-id",
            "cli-simple-derived-bad-fit",
            "--reference",
            "internal-test",
            "--threads",
            "1",
        ])
        .output()
        .expect("simplify command should run");
    assert!(
        !out.status.success(),
        "simplify workspace should reject fit result without covariance"
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("covariance"),
        "stderr should mention missing covariance, got:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let _ = std::fs::remove_dir_all(&work_dir);
}
