use super::audit::audit_simplified_likelihood;
use super::convert::{simplified_to_model, simplified_to_workspace};
use super::export::{
    SimplifiedLikelihoodAlignedFitResult, SimplifiedLikelihoodDeriveConfig,
    SimplifiedLikelihoodDeriveMetadata, build_simplified_likelihood_export_report,
    derive_simplified_likelihood_core, validate_simplified_likelihood_derive_config,
};
use super::factorize::factorize_covariance_workspace;
use super::schema::{
    SimplifiedBasisComponent, SimplifiedFidelityDiagnostics, SimplifiedLikelihoodWorkspace,
    SimplifiedUncertaintyModel,
};
use super::validate::validate_simplified_likelihood;

fn assert_vec_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "vector length mismatch");
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!((*a - *e).abs() <= 1e-12, "mismatch at index {idx}: actual={} expected={}", a, e);
    }
}

fn assert_matrix_close(actual: &[Vec<f64>], expected: &[Vec<f64>], tol: f64) {
    assert_eq!(actual.len(), expected.len(), "matrix row mismatch");
    for (row_idx, (actual_row, expected_row)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(actual_row.len(), expected_row.len(), "matrix column mismatch at row {row_idx}");
        for (col_idx, (a, e)) in actual_row.iter().zip(expected_row.iter()).enumerate() {
            assert!(
                (*a - *e).abs() <= tol,
                "matrix mismatch at ({row_idx}, {col_idx}): actual={} expected={} tol={tol}",
                a,
                e
            );
        }
    }
}

fn reconstruct_covariance(
    nominal: &[f64],
    components: &[SimplifiedBasisComponent],
) -> Vec<Vec<f64>> {
    let n = nominal.len();
    let mut covariance = vec![vec![0.0; n]; n];

    for component in components {
        let mut shift = vec![0.0; n];
        for idx in 0..n {
            let hi_shift = component.hi[idx] - nominal[idx];
            let lo_shift = nominal[idx] - component.lo[idx];
            assert!(
                (hi_shift - lo_shift).abs() <= 1e-10,
                "component {} is not symmetric around nominal at bin {}",
                component.name,
                idx
            );
            shift[idx] = hi_shift;
        }

        for i in 0..n {
            for j in 0..n {
                covariance[i][j] += shift[i] * shift[j];
            }
        }
    }

    covariance
}

#[test]
fn test_deserialize_simplified_likelihood_fixture() {
    let json = include_str!("../../../../tests/fixtures/sl_basis_two_bin.json");
    let spec: SimplifiedLikelihoodWorkspace =
        serde_json::from_str(json).expect("simplified likelihood fixture should deserialize");

    assert_eq!(spec.schema_version, "nextstat_simplified_likelihood_v0");
}

#[test]
fn test_validate_simplified_likelihood_fixture() {
    let json = include_str!("../../../../tests/fixtures/sl_basis_two_bin.json");
    let spec: SimplifiedLikelihoodWorkspace =
        serde_json::from_str(json).expect("simplified likelihood fixture should deserialize");

    validate_simplified_likelihood(&spec).expect("fixture should validate");
}

#[test]
fn test_validate_rejects_basis_component_length_mismatch() {
    let json = r#"{
      "schema_version": "nextstat_simplified_likelihood_v0",
      "metadata": {
        "experiment": "CMS",
        "analysis_id": "broken",
        "source_format": "basis",
        "reference": "internal-test"
      },
      "poi": {
        "name": "mu",
        "init": 1.0,
        "bounds": [0.0, 10.0]
      },
      "bins": [
        { "channel": "SR", "name": "bin0" },
        { "channel": "SR", "name": "bin1" }
      ],
      "observed": [4.0, 5.0],
      "background_nominal": [3.2, 4.8],
      "uncertainty_model": {
        "kind": "basis",
        "components": [
          {
            "name": "np0",
            "hi": [3.5],
            "lo": [2.9, 4.4]
          }
        ]
      }
    }"#;
    let spec: SimplifiedLikelihoodWorkspace =
        serde_json::from_str(json).expect("broken fixture should still deserialize");

    let err = validate_simplified_likelihood(&spec)
        .expect_err("validator should reject basis component length mismatch");
    assert!(
        err.to_string().contains("length"),
        "validator error should mention length mismatch, got: {err}"
    );
}

#[test]
fn test_validate_covariance_fixture() {
    let json = include_str!("../../../../tests/fixtures/sl_covariance_three_bin.json");
    let spec: SimplifiedLikelihoodWorkspace =
        serde_json::from_str(json).expect("covariance fixture should deserialize");

    validate_simplified_likelihood(&spec).expect("covariance fixture should validate");
}

#[test]
fn test_validate_rejects_materially_non_psd_covariance() {
    let json = r#"{
      "schema_version": "nextstat_simplified_likelihood_v0",
      "metadata": {
        "experiment": "CMS",
        "analysis_id": "broken-covariance",
        "source_format": "covariance",
        "reference": "internal-test"
      },
      "poi": {
        "name": "mu",
        "init": 1.0,
        "bounds": [0.0, 10.0]
      },
      "bins": [
        { "channel": "SR", "name": "bin0" },
        { "channel": "SR", "name": "bin1" }
      ],
      "observed": [5.0, 4.0],
      "background_nominal": [4.5, 3.9],
      "uncertainty_model": {
        "kind": "covariance",
        "total_covariance": [
          [1.0, 2.0],
          [2.0, 1.0]
        ]
      }
    }"#;
    let spec: SimplifiedLikelihoodWorkspace =
        serde_json::from_str(json).expect("broken covariance fixture should deserialize");

    let err = validate_simplified_likelihood(&spec)
        .expect_err("validator should reject materially non-PSD covariance");
    assert!(
        err.to_string().contains("positive semidefinite"),
        "validator error should mention PSD, got: {err}"
    );
}

#[test]
fn test_validate_rejects_derived_source_without_derivation_and_fidelity() {
    let json = r#"{
      "schema_version": "nextstat_simplified_likelihood_v0",
      "metadata": {
        "experiment": "ATLAS",
        "analysis_id": "derived-missing-provenance",
        "source_format": "derived_from_workspace",
        "reference": "internal-test"
      },
      "poi": {
        "name": "mu",
        "init": 1.0,
        "bounds": [0.0, 10.0]
      },
      "bins": [
        { "channel": "SR", "name": "bin0" }
      ],
      "observed": [5.0],
      "background_nominal": [4.0],
      "signal_nominal": [0.8],
      "uncertainty_model": {
        "kind": "basis",
        "components": []
      }
    }"#;
    let spec: SimplifiedLikelihoodWorkspace =
        serde_json::from_str(json).expect("derived fixture should deserialize");

    let err = validate_simplified_likelihood(&spec)
        .expect_err("derived artifacts must require derivation and fidelity diagnostics");
    assert!(
        err.to_string().contains("derivation"),
        "validator error should mention derivation/fidelity requirements, got: {err}"
    );
}

#[test]
fn test_validate_derived_simplified_likelihood_example() {
    let json = include_str!(
        "../../../../docs/specs/hep/simplified_likelihood_derived_from_workspace_v0.example.json"
    );
    let spec: SimplifiedLikelihoodWorkspace = serde_json::from_str(json)
        .expect("derived simplified likelihood example should deserialize");

    validate_simplified_likelihood(&spec)
        .expect("derived simplified likelihood example should validate");
}

#[test]
fn test_audit_basis_fixture_reports_public_contract_fields() {
    let json = include_str!("../../../../tests/fixtures/sl_basis_two_bin.json");
    let spec: SimplifiedLikelihoodWorkspace =
        serde_json::from_str(json).expect("basis fixture should deserialize");

    let audit = audit_simplified_likelihood(&spec).expect("basis fixture should audit");

    assert_eq!(audit.schema_version, "nextstat_simplified_likelihood_audit_v0");
    assert_eq!(audit.input_schema_version, "nextstat_simplified_likelihood_v0");
    assert_eq!(audit.uncertainty_model_kind, "basis");
    assert_eq!(audit.channel_names, vec!["SR".to_string()]);
    assert_eq!(audit.channel_count, 1);
    assert_eq!(audit.total_bins, 2);
    assert_eq!(audit.reduced_nuisance_count, 1);
    assert_eq!(audit.parameter_count_estimate, 2);
    assert!(audit.has_signal);
    assert!(!audit.input_has_factorization_diagnostics);
    assert!(audit.diagnostics.factorization.is_none());
}

#[test]
fn test_audit_covariance_fixture_reports_computed_factorization_diagnostics() {
    let json = include_str!("../../../../tests/fixtures/sl_covariance_three_bin.json");
    let spec: SimplifiedLikelihoodWorkspace =
        serde_json::from_str(json).expect("covariance fixture should deserialize");

    let audit = audit_simplified_likelihood(&spec).expect("covariance fixture should audit");

    assert_eq!(audit.schema_version, "nextstat_simplified_likelihood_audit_v0");
    assert_eq!(audit.input_schema_version, "nextstat_simplified_likelihood_v0");
    assert_eq!(audit.uncertainty_model_kind, "covariance");
    assert_eq!(audit.channel_names, vec!["SR_low".to_string(), "SR_high".to_string()]);
    assert_eq!(audit.channel_count, 2);
    assert_eq!(audit.total_bins, 3);
    assert_eq!(audit.reduced_nuisance_count, 3);
    assert_eq!(audit.parameter_count_estimate, 4);
    assert!(audit.has_signal);
    let factorization =
        audit.diagnostics.factorization.expect("covariance audit should expose factorization");
    assert_eq!(factorization.method, "symmetric_eigendecomposition");
    assert_eq!(factorization.retained_rank, 3);
    assert!(factorization.frobenius_residual <= 1e-10);
}

#[test]
fn test_deserialize_and_validate_simplified_likelihood_derive_config_example() {
    let json =
        include_str!("../../../../docs/specs/hep/simplified_likelihood_derive_v0.example.json");
    let config: SimplifiedLikelihoodDeriveConfig =
        serde_json::from_str(json).expect("derive config example should deserialize");

    validate_simplified_likelihood_derive_config(&config)
        .expect("derive config example should validate");
}

#[test]
fn test_validate_rejects_simplified_likelihood_derive_config_with_invalid_target() {
    let json = r#"{
      "schema_version": "nextstat_simplified_likelihood_derive_v0",
      "source_workspace": {
        "format": "pyhf",
        "poi_name": "mu"
      },
      "fit_result": {
        "schema_version": "nextstat_fit_result_v0",
        "background_state": "postfit_background"
      },
      "selection": {
        "channels": ["SR"]
      },
      "reduction": {
        "output_uncertainty_model": "basis",
        "basis_method": "eigen",
        "explained_variance_target": 0.0,
        "constraint_covariance_source": "source_model_constraints",
        "split_stat_covariance": true
      },
      "jacobian": {
        "method": "finite_difference",
        "relative_step": 0.01,
        "absolute_step_floor": 0.000001
      },
      "fidelity_smoke": {
        "random_draws": 64,
        "qmu_test_mu": 1.0,
        "upper_limit_cl": 0.95
      },
      "output_contract": {
        "schema_version": "nextstat_simplified_likelihood_v0",
        "require_factorization_diagnostics": true,
        "require_fidelity_diagnostics": true
      }
    }"#;
    let config: SimplifiedLikelihoodDeriveConfig =
        serde_json::from_str(json).expect("invalid derive config should deserialize");

    let err = validate_simplified_likelihood_derive_config(&config)
        .expect_err("validator should reject zero explained_variance_target");
    assert!(
        err.to_string().contains("explained_variance_target"),
        "validator error should mention explained_variance_target, got: {err}"
    );
}

#[test]
fn test_derive_simplified_likelihood_core_from_workspace_reduces_to_requested_basis_rank() {
    let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
    let workspace: crate::pyhf::Workspace =
        serde_json::from_str(json).expect("simple workspace fixture should deserialize");
    let model = crate::pyhf::HistFactoryModel::from_workspace_with_settings(
        &workspace,
        crate::pyhf::NormSysInterpCode::Code1,
        crate::pyhf::HistoSysInterpCode::Code0,
    )
    .expect("simple workspace should convert to model");

    let config_json = r#"{
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
        "explained_variance_target": 0.9,
        "constraint_covariance_source": "aligned_fit_covariance",
        "max_components": 1,
        "split_stat_covariance": true
      },
      "jacobian": {
        "method": "finite_difference",
        "relative_step": 0.01,
        "absolute_step_floor": 0.000001
      },
      "fidelity_smoke": {
        "random_draws": 8,
        "qmu_test_mu": 1.0,
        "upper_limit_cl": 0.95
      },
      "output_contract": {
        "schema_version": "nextstat_simplified_likelihood_v0",
        "require_factorization_diagnostics": true,
        "require_fidelity_diagnostics": true
      }
    }"#;
    let config: SimplifiedLikelihoodDeriveConfig =
        serde_json::from_str(config_json).expect("derive config should deserialize");
    validate_simplified_likelihood_derive_config(&config).expect("derive config should validate");

    let fit_result = SimplifiedLikelihoodAlignedFitResult {
        schema_version: Some("nextstat_fit_result_v0".to_string()),
        parameters: vec![0.0, 1.0, 1.0],
        covariance: vec![
            0.0, 0.0, 0.0, //
            0.0, 0.09, 0.12, //
            0.0, 0.12, 0.16,
        ],
    };
    let metadata = SimplifiedLikelihoodDeriveMetadata {
        experiment: "ATLAS".to_string(),
        analysis_id: "unit-simple-derived".to_string(),
        reference: "internal-test".to_string(),
        description: Some("unit test".to_string()),
    };

    let mut derived =
        derive_simplified_likelihood_core(&workspace, &model, &fit_result, &config, &metadata)
            .expect("workspace should derive to simplified likelihood core");

    assert_eq!(derived.full_nuisance_count, 2);
    assert_eq!(derived.workspace.metadata.source_format, "derived_from_workspace");
    assert_eq!(derived.workspace.background_nominal, vec![50.0, 60.0]);
    assert_eq!(
        derived.workspace.signal_nominal.as_ref().expect("signal nominal should be present"),
        &vec![5.0, 10.0]
    );
    assert_eq!(
        derived.workspace.derivation.as_ref().expect("derivation should be present").selected_bins,
        Some(vec!["singlechannel/bin0".to_string(), "singlechannel/bin1".to_string()])
    );
    assert_eq!(
        derived
            .workspace
            .derivation
            .as_ref()
            .expect("derivation should be present")
            .constraint_covariance_source,
        "aligned_fit_covariance"
    );
    match &derived.workspace.uncertainty_model {
        SimplifiedUncertaintyModel::Basis { components } => {
            assert_eq!(components.len(), 1, "max_components=1 should cap retained basis rank");
        }
        SimplifiedUncertaintyModel::Covariance { .. } => {
            panic!("derive runtime should emit basis uncertainty model");
        }
    }

    let factorization = derived
        .workspace
        .diagnostics
        .as_ref()
        .and_then(|diagnostics| diagnostics.factorization.as_ref())
        .expect("factorization diagnostics should be present");
    let retained_rank = factorization.retained_rank;
    let explained_variance_fraction = factorization.explained_variance_fraction;
    assert_eq!(retained_rank, 1);
    assert!(explained_variance_fraction >= 0.9);

    let diagnostics = derived.workspace.diagnostics.get_or_insert_with(Default::default);
    diagnostics.fidelity = Some(SimplifiedFidelityDiagnostics {
        nuisance_count_full: Some(derived.full_nuisance_count),
        nuisance_count_reduced: Some(retained_rank),
        bins_count: Some(derived.workspace.bins.len()),
        relative_background_cov_residual: Some(0.0),
        max_abs_expected_delta_at_nominal: Some(0.0),
        max_abs_expected_delta_random_draws: Some(0.0),
        qmu_delta_smoke: Some(0.0),
        upper_limit_ratio_smoke: Some(1.0),
        max_abs_yield_delta: Some(0.0),
        max_rel_yield_delta: Some(0.0),
    });
    validate_simplified_likelihood(&derived.workspace)
        .expect("augmented derived workspace should validate");

    let report = build_simplified_likelihood_export_report(&config, &metadata, &derived)
        .expect("derived workspace should emit export report");
    assert_eq!(report.schema_version, "nextstat_simplified_likelihood_export_report_v0");
    assert_eq!(report.status, "ok");
    assert_eq!(report.support_class, "research-grade");
    assert_eq!(report.source.workspace_format, "pyhf");
    assert_eq!(report.source.poi_name, "mu");
    assert_eq!(report.metadata.analysis_id, "unit-simple-derived");
    assert_eq!(report.output.schema_version, "nextstat_simplified_likelihood_v0");
    assert_eq!(report.output.uncertainty_model_kind, "basis");
    assert_eq!(report.output.bins_count, derived.workspace.bins.len());
    assert_eq!(report.output.full_nuisance_count, derived.full_nuisance_count);
    assert_eq!(report.output.reduced_nuisance_count, retained_rank);
    assert_eq!(report.output.reduction_ratio, 0.5);
    assert_eq!(report.reduction.constraint_covariance_source, "aligned_fit_covariance");
    assert_eq!(report.diagnostics.factorization.unwrap().retained_rank, retained_rank);
    assert_eq!(report.diagnostics.fidelity.unwrap().nuisance_count_reduced, Some(retained_rank));
    assert_eq!(
        report.explicit_boundaries,
        vec![
            "source_workspace.format=pyhf only".to_string(),
            "partial per-channel bin selection unsupported".to_string(),
            "derived_from_workspace preserves reduced nuisance coordinates, not source-level nuisance identities"
                .to_string(),
        ]
    );
}

#[test]
fn test_derive_simplified_likelihood_core_rejects_partial_channel_bin_selection() {
    let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
    let workspace: crate::pyhf::Workspace =
        serde_json::from_str(json).expect("simple workspace fixture should deserialize");
    let model = crate::pyhf::HistFactoryModel::from_workspace_with_settings(
        &workspace,
        crate::pyhf::NormSysInterpCode::Code1,
        crate::pyhf::HistoSysInterpCode::Code0,
    )
    .expect("simple workspace should convert to model");

    let config_json = r#"{
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
        "bins": ["singlechannel/bin0"]
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
        "random_draws": 8,
        "qmu_test_mu": 1.0,
        "upper_limit_cl": 0.95
      },
      "output_contract": {
        "schema_version": "nextstat_simplified_likelihood_v0",
        "require_factorization_diagnostics": true,
        "require_fidelity_diagnostics": true
      }
    }"#;
    let config: SimplifiedLikelihoodDeriveConfig =
        serde_json::from_str(config_json).expect("derive config should deserialize");
    let fit_result = SimplifiedLikelihoodAlignedFitResult {
        schema_version: Some("nextstat_fit_result_v0".to_string()),
        parameters: vec![0.0, 1.0, 1.0],
        covariance: vec![
            0.0, 0.0, 0.0, //
            0.0, 0.09, 0.0, //
            0.0, 0.0, 0.16,
        ],
    };
    let metadata = SimplifiedLikelihoodDeriveMetadata {
        experiment: "ATLAS".to_string(),
        analysis_id: "unit-simple-derived-partial".to_string(),
        reference: "internal-test".to_string(),
        description: None,
    };

    let err =
        derive_simplified_likelihood_core(&workspace, &model, &fit_result, &config, &metadata)
            .expect_err("partial bin selections should be rejected explicitly");
    assert!(
        err.to_string().contains("partial-bin"),
        "error should mention partial-bin selection boundary, got: {err}"
    );
}

#[test]
fn test_derive_simplified_likelihood_core_rejects_non_gaussian_source_constraints() {
    let json = r#"{
      "channels": [
        {
          "name": "SR",
          "samples": [
            {
              "name": "signal",
              "data": [1.0],
              "modifiers": [{"name": "mu", "type": "normfactor", "data": null}]
            },
            {
              "name": "background",
              "data": [10.0],
              "modifiers": [
                {
                  "name": "syst",
                  "type": "normsys",
                  "data": {"hi": 1.1, "lo": 0.9}
                }
              ]
            }
          ]
        }
      ],
      "observations": [{"name": "SR", "data": [11.0]}],
      "measurements": [
        {
          "name": "m",
          "config": {
            "poi": "mu",
            "parameters": [
              {
                "name": "syst",
                "constraint": {"type": "LogNormal", "rel_uncertainty": 0.1}
              }
            ]
          }
        }
      ],
      "version": "1.0.0"
    }"#;
    let workspace: crate::pyhf::Workspace =
        serde_json::from_str(json).expect("workspace with lognormal normsys should deserialize");
    let model = crate::pyhf::HistFactoryModel::from_workspace_with_settings(
        &workspace,
        crate::pyhf::NormSysInterpCode::Code1,
        crate::pyhf::HistoSysInterpCode::Code0,
    )
    .expect("workspace should convert to model");

    let config_json = r#"{
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
        "channels": ["SR"]
      },
      "reduction": {
        "output_uncertainty_model": "basis",
        "basis_method": "eigen",
        "explained_variance_target": 0.95,
        "constraint_covariance_source": "source_model_constraints",
        "split_stat_covariance": false
      },
      "jacobian": {
        "method": "finite_difference",
        "relative_step": 0.01,
        "absolute_step_floor": 0.000001
      },
      "fidelity_smoke": {
        "random_draws": 8,
        "qmu_test_mu": 1.0,
        "upper_limit_cl": 0.95
      },
      "output_contract": {
        "schema_version": "nextstat_simplified_likelihood_v0",
        "require_factorization_diagnostics": true,
        "require_fidelity_diagnostics": true
      }
    }"#;
    let config: SimplifiedLikelihoodDeriveConfig =
        serde_json::from_str(config_json).expect("derive config should deserialize");
    let fit_result = SimplifiedLikelihoodAlignedFitResult {
        schema_version: Some("nextstat_fit_result_v0".to_string()),
        parameters: vec![1.0, 0.0],
        covariance: vec![
            1.0, 0.0, //
            0.0, 1.0,
        ],
    };
    let metadata = SimplifiedLikelihoodDeriveMetadata {
        experiment: "ATLAS".to_string(),
        analysis_id: "unit-lognormal-reject".to_string(),
        reference: "internal-test".to_string(),
        description: None,
    };

    let err =
        derive_simplified_likelihood_core(&workspace, &model, &fit_result, &config, &metadata)
            .expect_err("non-Gaussian source constraints should be rejected");
    assert!(
        err.to_string().contains("Gaussian-constrained nuisances"),
        "error should mention Gaussian-constrained nuisances, got: {err}"
    );
}

#[test]
fn test_factorize_covariance_fixture_reconstructs_total_covariance() {
    let json = include_str!("../../../../tests/fixtures/sl_covariance_three_bin.json");
    let spec: SimplifiedLikelihoodWorkspace =
        serde_json::from_str(json).expect("covariance fixture should deserialize");

    let result =
        factorize_covariance_workspace(&spec).expect("covariance fixture should factorize");
    let rebuilt = reconstruct_covariance(&spec.background_nominal, &result.components);

    match &spec.uncertainty_model {
        super::schema::SimplifiedUncertaintyModel::Covariance { total_covariance, .. } => {
            assert_matrix_close(&rebuilt, total_covariance, 1e-10);
        }
        super::schema::SimplifiedUncertaintyModel::Basis { .. } => {
            panic!("expected covariance uncertainty model");
        }
    }

    assert_eq!(result.diagnostics.method, "symmetric_eigendecomposition");
    assert_eq!(result.diagnostics.original_rank, 3);
    assert_eq!(result.diagnostics.retained_rank, 3);
    assert!((result.diagnostics.explained_variance_fraction - 1.0).abs() <= 1e-12);
    assert!(result.diagnostics.frobenius_residual <= 1e-10);
}

#[test]
fn test_factorize_covariance_clips_tiny_negative_eigenvalue_and_logs_it() {
    let json = r#"{
      "schema_version": "nextstat_simplified_likelihood_v0",
      "metadata": {
        "experiment": "CMS",
        "analysis_id": "tiny-negative-eigen",
        "source_format": "covariance",
        "reference": "internal-test"
      },
      "poi": {
        "name": "mu",
        "init": 1.0,
        "bounds": [0.0, 10.0]
      },
      "bins": [
        { "channel": "SR", "name": "bin0" },
        { "channel": "SR", "name": "bin1" }
      ],
      "observed": [10.0, 9.0],
      "background_nominal": [8.0, 7.0],
      "uncertainty_model": {
        "kind": "covariance",
        "total_covariance": [
          [0.5, 0.50000000005],
          [0.50000000005, 0.5]
        ]
      }
    }"#;
    let spec: SimplifiedLikelihoodWorkspace =
        serde_json::from_str(json).expect("tiny-negative covariance should deserialize");

    validate_simplified_likelihood(&spec).expect("tiny-negative eigenvalue should pass validation");
    let result =
        factorize_covariance_workspace(&spec).expect("tiny-negative eigenvalue should clip");

    assert_eq!(result.diagnostics.clipped_negative_eigenvalues, 1);
    assert!(result.diagnostics.max_clipped_negative_eigenvalue_magnitude > 0.0);
    assert_eq!(result.diagnostics.retained_rank, 1);
    assert!((result.diagnostics.explained_variance_fraction - 1.0).abs() <= 1e-10);
}

#[test]
fn test_convert_basis_fixture_to_workspace_preserves_structure() {
    let json = include_str!("../../../../tests/fixtures/sl_basis_two_bin.json");
    let spec: SimplifiedLikelihoodWorkspace =
        serde_json::from_str(json).expect("simplified likelihood fixture should deserialize");

    let ws = simplified_to_workspace(&spec).expect("basis fixture should convert to workspace");

    assert_eq!(ws.channels.len(), 1);
    assert_eq!(ws.channels[0].name, "SR");
    assert_eq!(ws.channels[0].samples.len(), 2);
    assert_eq!(ws.channels[0].samples[0].name, "signal");
    assert_eq!(ws.channels[0].samples[1].name, "total_background");
    assert_eq!(ws.channels[0].samples[1].modifiers.len(), 1);
}

#[test]
fn test_convert_covariance_fixture_to_workspace_preserves_structure() {
    let json = include_str!("../../../../tests/fixtures/sl_covariance_three_bin.json");
    let spec: SimplifiedLikelihoodWorkspace =
        serde_json::from_str(json).expect("covariance fixture should deserialize");

    let ws =
        simplified_to_workspace(&spec).expect("covariance fixture should convert to workspace");

    assert_eq!(ws.channels.len(), 2);
    assert_eq!(ws.channels[0].name, "SR_low");
    assert_eq!(ws.channels[0].samples.len(), 2);
    assert_eq!(ws.channels[1].name, "SR_high");
    assert_eq!(ws.channels[1].samples.len(), 2);
    assert_eq!(ws.channels[0].samples[1].modifiers.len(), 3);
    assert_eq!(ws.channels[1].samples[1].modifiers.len(), 3);
}

#[test]
fn test_convert_basis_fixture_to_model_matches_templates_at_minus_one_zero_plus_one() {
    let json = include_str!("../../../../tests/fixtures/sl_basis_two_bin.json");
    let spec: SimplifiedLikelihoodWorkspace =
        serde_json::from_str(json).expect("simplified likelihood fixture should deserialize");

    let model = simplified_to_model(&spec).expect("basis fixture should convert to model");
    let mut params: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();
    let alpha_idx = model
        .parameters()
        .iter()
        .position(|p| p.name == "sl_np_000")
        .expect("reduced nuisance parameter should exist");

    params[alpha_idx] = -1.0;
    let minus_one =
        model.expected_data_pyhf_main(&params).expect("minus-one expected data should evaluate");
    assert_vec_close(&minus_one, &[11.7, 8.3]);

    params[alpha_idx] = 0.0;
    let nominal =
        model.expected_data_pyhf_main(&params).expect("nominal expected data should evaluate");
    assert_vec_close(&nominal, &[12.3, 8.7]);

    params[alpha_idx] = 1.0;
    let plus_one =
        model.expected_data_pyhf_main(&params).expect("plus-one expected data should evaluate");
    assert_vec_close(&plus_one, &[12.9, 9.1]);
}
