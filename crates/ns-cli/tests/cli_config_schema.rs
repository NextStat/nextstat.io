use std::path::PathBuf;
use std::process::{Command, Output};

fn bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_nextstat"))
}

fn run(args: &[&str]) -> Output {
    Command::new(bin_path())
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?} {:?}: {}", bin_path(), args, e))
}

#[test]
fn config_schema_default_is_analysis_spec_v0_json() {
    let out = run(&["config", "schema"]);
    assert!(
        out.status.success(),
        "config schema should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("schema output should be valid JSON");
    assert_eq!(v["$id"], "https://nextstat.io/schemas/trex/analysis_spec_v0.schema.json");
    assert_eq!(v["$schema"], "https://json-schema.org/draft/2020-12/schema");
}

#[test]
fn config_schema_can_emit_report_schema() {
    let out = run(&["config", "schema", "--name", "report_yields_v0"]);
    assert!(
        out.status.success(),
        "config schema --name report_yields_v0 should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("schema output should be valid JSON");
    assert_eq!(v["$schema"], "https://json-schema.org/draft/2020-12/schema");
}

#[test]
fn config_schema_can_emit_validation_report_schema() {
    let out = run(&["config", "schema", "--name", "validation_report_v1"]);
    assert!(
        out.status.success(),
        "config schema --name validation_report_v1 should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("schema output should be valid JSON");
    assert_eq!(v["$id"], "https://nextstat.io/schemas/validation/validation_report_v1.schema.json");
}

#[test]
fn config_schema_can_emit_m15_schemas() {
    let cases = [
        ("m15_config_v1", "https://nextstat.io/schemas/validation/m15_config_v1.schema.json"),
        (
            "m15_assessment_table_v1",
            "https://nextstat.io/schemas/validation/m15_assessment_table_v1.schema.json",
        ),
        ("m15_map_v1", "https://nextstat.io/schemas/validation/m15_map_v1.schema.json"),
        ("m15_mar_v1", "https://nextstat.io/schemas/validation/m15_mar_v1.schema.json"),
        (
            "m15_profile_diff_report_v1",
            "https://nextstat.io/schemas/validation/m15_profile_diff_report_v1.schema.json",
        ),
        (
            "m15_bundle_manifest_v1",
            "https://nextstat.io/schemas/validation/m15_bundle_manifest_v1.schema.json",
        ),
    ];

    for (name, expected_id) in cases {
        let out = run(&["config", "schema", "--name", name]);
        assert!(
            out.status.success(),
            "config schema --name {} should succeed, stderr={}",
            name,
            String::from_utf8_lossy(&out.stderr)
        );
        let v: serde_json::Value =
            serde_json::from_slice(&out.stdout).expect("schema output should be valid JSON");
        assert_eq!(v["$id"], expected_id);
        assert_eq!(v["$schema"], "https://json-schema.org/draft/2020-12/schema");
    }
}

#[test]
fn config_schema_can_emit_hepdata_schemas() {
    let cases = [
        ("hepdata_import_v1", "https://nextstat.io/schemas/io/hepdata_import_v1.schema.json"),
        ("hepdata_lock_v1", "https://nextstat.io/schemas/io/hepdata_lock_v1.schema.json"),
    ];

    for (name, expected_id) in cases {
        let out = run(&["config", "schema", "--name", name]);
        assert!(
            out.status.success(),
            "config schema --name {} should succeed, stderr={}",
            name,
            String::from_utf8_lossy(&out.stderr)
        );
        let v: serde_json::Value =
            serde_json::from_slice(&out.stdout).expect("schema output should be valid JSON");
        assert_eq!(v["$id"], expected_id);
        assert_eq!(v["$schema"], "https://json-schema.org/draft/2020-12/schema");
    }
}

#[test]
fn config_schema_can_emit_unbinned_spec_schema() {
    let out = run(&["config", "schema", "--name", "unbinned_spec_v0"]);
    assert!(
        out.status.success(),
        "config schema --name unbinned_spec_v0 should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("schema output should be valid JSON");
    assert_eq!(v["$id"], "https://nextstat.io/schemas/unbinned/unbinned_spec_v0.schema.json");
    assert_eq!(v["$schema"], "https://json-schema.org/draft/2020-12/schema");
}

#[test]
fn config_schema_can_emit_beta_binomial_design_schemas() {
    let cases = [
        (
            "beta_binomial_design_v0",
            "https://nextstat.io/schemas/pharma/beta_binomial_design_v0.schema.json",
        ),
        (
            "beta_binomial_design_analysis_v0",
            "https://nextstat.io/schemas/pharma/beta_binomial_design_analysis_v0.schema.json",
        ),
        (
            "beta_binomial_operating_characteristics_v0",
            "https://nextstat.io/schemas/pharma/beta_binomial_operating_characteristics_v0.schema.json",
        ),
        (
            "normal_normal_design_v0",
            "https://nextstat.io/schemas/pharma/normal_normal_design_v0.schema.json",
        ),
        (
            "normal_normal_design_analysis_v0",
            "https://nextstat.io/schemas/pharma/normal_normal_design_analysis_v0.schema.json",
        ),
        (
            "normal_normal_operating_characteristics_v0",
            "https://nextstat.io/schemas/pharma/normal_normal_operating_characteristics_v0.schema.json",
        ),
        (
            "beta_binomial_posterior_predictive_v0",
            "https://nextstat.io/schemas/pharma/beta_binomial_posterior_predictive_v0.schema.json",
        ),
        (
            "normal_normal_posterior_predictive_v0",
            "https://nextstat.io/schemas/pharma/normal_normal_posterior_predictive_v0.schema.json",
        ),
        (
            "beta_binomial_prior_sensitivity_campaign_v0",
            "https://nextstat.io/schemas/pharma/beta_binomial_prior_sensitivity_campaign_v0.schema.json",
        ),
        (
            "beta_binomial_prior_sensitivity_report_v0",
            "https://nextstat.io/schemas/pharma/beta_binomial_prior_sensitivity_report_v0.schema.json",
        ),
        (
            "beta_binomial_design_report_v0",
            "https://nextstat.io/schemas/pharma/beta_binomial_design_report_v0.schema.json",
        ),
        (
            "normal_normal_prior_sensitivity_campaign_v0",
            "https://nextstat.io/schemas/pharma/normal_normal_prior_sensitivity_campaign_v0.schema.json",
        ),
        (
            "normal_normal_prior_sensitivity_report_v0",
            "https://nextstat.io/schemas/pharma/normal_normal_prior_sensitivity_report_v0.schema.json",
        ),
        (
            "normal_normal_design_report_v0",
            "https://nextstat.io/schemas/pharma/normal_normal_design_report_v0.schema.json",
        ),
        (
            "bayesian_design_report_bundle_v0",
            "https://nextstat.io/schemas/pharma/bayesian_design_report_bundle_v0.schema.json",
        ),
        (
            "bayesian_design_regulatory_appendix_v0",
            "https://nextstat.io/schemas/pharma/bayesian_design_regulatory_appendix_v0.schema.json",
        ),
        (
            "bayesian_prior_conflict_diagnostic_v0",
            "https://nextstat.io/schemas/pharma/bayesian_prior_conflict_diagnostic_v0.schema.json",
        ),
        (
            "bayesian_historical_control_borrowing_policy_v0",
            "https://nextstat.io/schemas/pharma/bayesian_historical_control_borrowing_policy_v0.schema.json",
        ),
        (
            "bayesian_historical_control_borrowing_review_v0",
            "https://nextstat.io/schemas/pharma/bayesian_historical_control_borrowing_review_v0.schema.json",
        ),
        (
            "bayesian_historical_control_borrowing_operating_characteristics_v0",
            "https://nextstat.io/schemas/pharma/bayesian_historical_control_borrowing_operating_characteristics_v0.schema.json",
        ),
        (
            "bayesian_robust_mixture_prior_policy_v0",
            "https://nextstat.io/schemas/pharma/bayesian_robust_mixture_prior_policy_v0.schema.json",
        ),
        (
            "bayesian_robust_mixture_prior_review_v0",
            "https://nextstat.io/schemas/pharma/bayesian_robust_mixture_prior_review_v0.schema.json",
        ),
        (
            "bayesian_robust_mixture_prior_operating_characteristics_v0",
            "https://nextstat.io/schemas/pharma/bayesian_robust_mixture_prior_operating_characteristics_v0.schema.json",
        ),
        (
            "simplified_likelihood_v0",
            "https://nextstat.io/schemas/hep/simplified_likelihood_v0.schema.json",
        ),
        (
            "simplified_likelihood_audit_v0",
            "https://nextstat.io/schemas/hep/simplified_likelihood_audit_v0.schema.json",
        ),
        (
            "simplified_likelihood_derive_v0",
            "https://nextstat.io/schemas/hep/simplified_likelihood_derive_v0.schema.json",
        ),
        (
            "simplified_likelihood_export_report_v0",
            "https://nextstat.io/schemas/hep/simplified_likelihood_export_report_v0.schema.json",
        ),
        (
            "simplified_likelihood_promotion_evidence_bundle_v0",
            "https://nextstat.io/schemas/benchmarks/simplified_likelihood_promotion_evidence_bundle_v0.schema.json",
        ),
        (
            "simplified_likelihood_promotion_evidence_check_v0",
            "https://nextstat.io/schemas/benchmarks/simplified_likelihood_promotion_evidence_check_v0.schema.json",
        ),
        (
            "simplified_likelihood_promotion_bundle_promotion_report_v0",
            "https://nextstat.io/schemas/benchmarks/simplified_likelihood_promotion_bundle_promotion_report_v0.schema.json",
        ),
        (
            "simplified_likelihood_export_benchmark_snapshot_report_v0",
            "https://nextstat.io/schemas/benchmarks/simplified_likelihood_export_benchmark_snapshot_report_v0.schema.json",
        ),
        (
            "simplified_likelihood_export_public_validation_report_v0",
            "https://nextstat.io/schemas/benchmarks/simplified_likelihood_export_public_validation_report_v0.schema.json",
        ),
        (
            "simplified_likelihood_exporter_promotion_evidence_bundle_v0",
            "https://nextstat.io/schemas/benchmarks/simplified_likelihood_exporter_promotion_evidence_bundle_v0.schema.json",
        ),
        (
            "simplified_likelihood_exporter_promotion_evidence_check_v0",
            "https://nextstat.io/schemas/benchmarks/simplified_likelihood_exporter_promotion_evidence_check_v0.schema.json",
        ),
        (
            "simplified_likelihood_exporter_promotion_bundle_promotion_report_v0",
            "https://nextstat.io/schemas/benchmarks/simplified_likelihood_exporter_promotion_bundle_promotion_report_v0.schema.json",
        ),
        (
            "simplified_likelihood_exporter_stable_review_assessment_v0",
            "https://nextstat.io/schemas/benchmarks/simplified_likelihood_exporter_stable_review_assessment_v0.schema.json",
        ),
        (
            "simplified_likelihood_exporter_stable_source_semantics_boundary_v0",
            "https://nextstat.io/schemas/benchmarks/simplified_likelihood_exporter_stable_source_semantics_boundary_v0.schema.json",
        ),
        (
            "simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0",
            "https://nextstat.io/schemas/benchmarks/simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0.schema.json",
        ),
        (
            "simplified_likelihood_exporter_stable_candidate_review_packet_v0",
            "https://nextstat.io/schemas/benchmarks/simplified_likelihood_exporter_stable_candidate_review_packet_v0.schema.json",
        ),
        (
            "simplified_likelihood_exporter_stable_evidence_policy_v0",
            "https://nextstat.io/schemas/benchmarks/simplified_likelihood_exporter_stable_evidence_policy_v0.schema.json",
        ),
        (
            "simplified_likelihood_exporter_stable_evidence_freshness_report_v0",
            "https://nextstat.io/schemas/benchmarks/simplified_likelihood_exporter_stable_evidence_freshness_report_v0.schema.json",
        ),
        (
            "simplified_likelihood_exporter_stable_promotion_decision_v0",
            "https://nextstat.io/schemas/benchmarks/simplified_likelihood_exporter_stable_promotion_decision_v0.schema.json",
        ),
    ];

    for (name, expected_id) in cases {
        let out = run(&["config", "schema", "--name", name]);
        assert!(
            out.status.success(),
            "config schema --name {} should succeed, stderr={}",
            name,
            String::from_utf8_lossy(&out.stderr)
        );
        let v: serde_json::Value =
            serde_json::from_slice(&out.stdout).expect("schema output should be valid JSON");
        assert_eq!(v["$id"], expected_id);
        assert_eq!(v["$schema"], "https://json-schema.org/draft/2020-12/schema");
    }
}
