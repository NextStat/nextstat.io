use ns_inference::{
    ADS_STATISTICAL_ARTIFACT_SCHEMA_ID, ADS_STATISTICAL_ARTIFACT_TYPE,
    ADS_STATISTICAL_ARTIFACT_VERSION, ADS_SYSTEMATICS_REGISTRY_VERSION, AdsArtifactAssumption,
    AdsArtifactDecision, AdsArtifactDiagnostics, AdsArtifactGuardrails, AdsArtifactMetadata,
    AdsArtifactReferenceSummary, AdsArtifactResult, AdsArtifactRigor, AdsCausalUncertainty,
    AdsCausalUncertaintyStatus, AdsClaimStatus, AdsClsGuardrail, AdsDecisionLossView,
    AdsEventCounts, AdsFitQuality, AdsLeakageIndicator, AdsPracticalSignificance,
    AdsPracticalThresholdStatus, AdsStatisticalArtifact, AdsTotalUncertainty,
    AdsTotalUncertaintyStatus, AdsUncertaintyInterval, ads_statistical_artifact_json_schema,
};
use serde_json::Value;
use std::path::PathBuf;

fn docs_schema_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop();
    path.pop();
    path.push("docs/schemas/ads/ads_statistical_artifact_v1.schema.json");
    path
}

fn docs_example_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop();
    path.pop();
    path.push("docs/examples/ads/ads_statistical_artifact_v1.json");
    path
}

fn sample_artifact() -> AdsStatisticalArtifact {
    let result = AdsArtifactResult {
        effect_estimate: 0.12,
        sampling_uncertainty: AdsUncertaintyInterval { lo: -0.04, hi: 0.04 },
        measurement_uncertainty: Some(AdsUncertaintyInterval { lo: -0.08, hi: 0.08 }),
        causal_uncertainty: AdsCausalUncertainty {
            status: AdsCausalUncertaintyStatus::Separate,
            diagnostics: vec!["no_pretrend_violation".into(), "no_major_overlap_warning".into()],
        },
        total_uncertainty: AdsTotalUncertainty {
            status: AdsTotalUncertaintyStatus::NotCombinable,
            interval: None,
        },
        cls: Some(AdsClsGuardrail {
            enabled: true,
            value: Some(0.023),
            expected_sensitivity_sigma: Some(3.2),
            claim_status: AdsClaimStatus::ClaimableWin,
        }),
        claim_status: AdsClaimStatus::ClaimableWin,
        practical_significance: AdsPracticalSignificance {
            threshold_status: AdsPracticalThresholdStatus::ExceedsWinThreshold,
            delta_win: Some(0.05),
            delta_loss: None,
            delta_equiv: Some(0.02),
            tost_status: None,
            decision_loss: Some(AdsDecisionLossView {
                scale: Some(1200.0),
                hold: Some(4200.0),
                revert: Some(8500.0),
            }),
        },
    };

    let diagnostics = AdsArtifactDiagnostics {
        fit_quality: AdsFitQuality::Good,
        convergence: true,
        gof_pvalue: Some(0.28),
        dominant_systematic: Some("conversion_lag_curve".into()),
        warnings: vec![],
    };

    let guardrails = AdsArtifactGuardrails {
        minimum_events: 100,
        actual_events: AdsEventCounts { control: 134, variant: 157 },
        power_at_mde: 0.87,
        sequential_looks_used: 2,
        sequential_looks_planned: 5,
        alpha_spent: 0.012,
        alpha_remaining: 0.038,
    };

    let metadata = AdsArtifactMetadata {
        data_hash: Some("sha256:abc123".into()),
        engine_version: "ns-inference-0.9.9".into(),
        model_version: "ads-histfactory-v1".into(),
        created_at: "2026-03-09T11:00:00Z".into(),
    };

    let mut artifact = AdsStatisticalArtifact::new(
        AdsArtifactDecision::StopWinner,
        result,
        diagnostics,
        guardrails,
        metadata,
    );
    artifact.assumptions = vec![
        AdsArtifactAssumption { key: "attribution_window".into(), value: "7d_click".into() },
        AdsArtifactAssumption {
            key: "fraud_policy".into(),
            value: "iab_conservative_10pct".into(),
        },
    ];
    artifact.rigor = AdsArtifactRigor {
        passed: true,
        artifacts_complete: true,
        claim_valid: true,
        leakage_clean: true,
        claim_violations: Vec::new(),
        leakage_indicators: Vec::<AdsLeakageIndicator>::new(),
    };
    artifact
}

#[test]
fn docs_schema_matches_shared_runtime_schema() {
    let schema: Value = serde_json::from_str(
        &std::fs::read_to_string(docs_schema_path()).expect("schema fixture should exist"),
    )
    .expect("schema fixture should be valid JSON");

    assert_eq!(schema, ads_statistical_artifact_json_schema());
    assert_eq!(schema["$id"], ADS_STATISTICAL_ARTIFACT_SCHEMA_ID);
}

#[test]
fn docs_example_matches_committed_artifact_output() {
    let expected: Value = serde_json::from_str(
        &std::fs::read_to_string(docs_example_path()).expect("example fixture should exist"),
    )
    .expect("example fixture should be valid JSON");
    let actual = serde_json::to_value(sample_artifact()).expect("sample artifact should serialize");

    assert_eq!(actual, expected);
}

#[test]
fn docs_example_deserializes_as_shared_artifact_and_summary() {
    let example =
        std::fs::read_to_string(docs_example_path()).expect("example fixture should exist");
    let artifact: AdsStatisticalArtifact =
        serde_json::from_str(&example).expect("example must deserialize as AdsStatisticalArtifact");
    let summary = AdsArtifactReferenceSummary::from(&artifact);

    assert_eq!(artifact.artifact_type, ADS_STATISTICAL_ARTIFACT_TYPE);
    assert_eq!(artifact.version, ADS_STATISTICAL_ARTIFACT_VERSION);
    assert_eq!(summary.contract.artifact_type, ADS_STATISTICAL_ARTIFACT_TYPE);
    assert_eq!(summary.contract.version, ADS_STATISTICAL_ARTIFACT_VERSION);
    assert_eq!(summary.contract.systematics_registry_version, ADS_SYSTEMATICS_REGISTRY_VERSION);
    assert_eq!(summary.claim_status, Some(AdsClaimStatus::ClaimableWin));
}
