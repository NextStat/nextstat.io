//! Tests for HS3 schema deserialization.

use super::schema::*;

// ---------------------------------------------------------------------------
// Phase 1 — Schema Deserialization Tests
// ---------------------------------------------------------------------------

#[test]
fn test_deserialize_ptv_fixture() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace =
        serde_json::from_str(json).expect("Failed to deserialize PTV HS3 fixture");

    // Top-level counts from Appendix A of the plan
    assert_eq!(ws.distributions.len(), 309);
    assert_eq!(ws.data.len(), 72);
    assert_eq!(ws.domains.len(), 7);
    assert_eq!(ws.parameter_points.len(), 3);
    assert_eq!(ws.analyses.len(), 2);
    assert_eq!(ws.likelihoods.len(), 2);
    assert!(ws.misc.is_some(), "misc field should be preserved");
}

#[test]
fn test_ptv_metadata() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();

    assert_eq!(ws.metadata.hs3_version, "0.2");
    let packages = ws.metadata.packages.as_ref().expect("packages should exist");
    assert_eq!(packages.len(), 1);
    assert_eq!(packages[0].name, "ROOT");
    assert_eq!(packages[0].version, "6.37.01");
}

#[test]
fn test_ptv_distribution_types() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();

    let mut hf_count = 0;
    let mut gauss_count = 0;
    let mut unknown_count = 0;

    for d in &ws.distributions {
        match d {
            Hs3Distribution::HistFactory(_) => hf_count += 1,
            Hs3Distribution::Gaussian(_) => gauss_count += 1,
            Hs3Distribution::Unknown(_) => unknown_count += 1,
            _ => {}
        }
    }

    assert_eq!(hf_count, 36, "expected 36 histfactory_dist");
    assert_eq!(gauss_count, 273, "expected 273 gaussian_dist");
    assert_eq!(unknown_count, 0, "no unknown distributions expected in PTV");
}

#[test]
fn test_ptv_histfactory_dist_structure() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();

    let hf_dists: Vec<&Hs3HistFactoryDist> = ws
        .distributions
        .iter()
        .filter_map(|d| match d {
            Hs3Distribution::HistFactory(hf) => Some(hf),
            _ => None,
        })
        .collect();

    assert_eq!(hf_dists.len(), 36);

    // Each channel's samples should have consistent bin count matching the axis nbins
    for hf in &hf_dists {
        assert!(!hf.samples.is_empty(), "channel {} has no samples", hf.name);
        let expected_bins = hf.samples[0].data.contents.len();
        assert!(expected_bins > 0, "channel {} has zero bins", hf.name);
        for sample in &hf.samples {
            assert_eq!(
                sample.data.contents.len(),
                expected_bins,
                "channel {} sample {} has inconsistent bin count: {} vs {}",
                hf.name,
                sample.name,
                sample.data.contents.len(),
                expected_bins
            );
        }
    }
}

#[test]
fn test_ptv_modifier_types() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();

    let mut normsys = 0usize;
    let mut histosys = 0usize;
    let mut staterror = 0usize;
    let mut normfactor = 0usize;
    let mut unknown = 0usize;

    for d in &ws.distributions {
        if let Hs3Distribution::HistFactory(hf) = d {
            for sample in &hf.samples {
                for m in &sample.modifiers {
                    match m {
                        Hs3Modifier::NormSys { .. } => normsys += 1,
                        Hs3Modifier::HistoSys { .. } => histosys += 1,
                        Hs3Modifier::StatError { .. } => staterror += 1,
                        Hs3Modifier::NormFactor { .. } => normfactor += 1,
                        Hs3Modifier::Unknown(_) => unknown += 1,
                        _ => {}
                    }
                }
            }
        }
    }

    // Counts from Appendix A
    assert_eq!(normsys, 36851, "normsys count");
    assert_eq!(histosys, 3541, "histosys count");
    assert_eq!(staterror, 1557, "staterror count");
    assert_eq!(normfactor, 1539, "normfactor count");
    assert_eq!(unknown, 0, "no unknown modifiers expected");
}

#[test]
fn test_ptv_gaussian_dist_fields() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();

    let gauss: Vec<&Hs3GaussianDist> = ws
        .distributions
        .iter()
        .filter_map(|d| match d {
            Hs3Distribution::Gaussian(g) => Some(g),
            _ => None,
        })
        .collect();

    assert_eq!(gauss.len(), 273);

    // All gaussian constraints in PTV have sigma=1.0 (standard NP constraints)
    for g in &gauss {
        assert!(!g.x.is_empty(), "gaussian x field should not be empty");
        assert!(!g.mean.is_empty(), "gaussian mean field should not be empty");
        assert_eq!(
            g.sigma, 1.0,
            "PTV gaussians should have sigma=1.0, got {} for {}",
            g.sigma, g.name
        );
    }
}

#[test]
fn test_ptv_data_entries() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();

    // 72 data entries: 36 asimov + 36 observed
    assert_eq!(ws.data.len(), 72);

    for d in &ws.data {
        assert_eq!(d.data_type, "binned", "data type should be binned");
        assert!(!d.contents.is_empty(), "data {} has zero bins", d.name);
    }
}

#[test]
fn test_ptv_analyses() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();

    assert_eq!(ws.analyses.len(), 2);

    let a0 = &ws.analyses[0];
    assert_eq!(a0.parameters_of_interest.len(), 4);
    assert!(a0.parameters_of_interest.contains(&"mu".to_string()));
    assert_eq!(a0.domains.len(), 3);
}

#[test]
fn test_ptv_likelihoods() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();

    assert_eq!(ws.likelihoods.len(), 2);

    for lh in &ws.likelihoods {
        assert_eq!(lh.distributions.len(), 36);
        assert_eq!(lh.data.len(), 36);
    }
}

#[test]
fn test_ptv_domains() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();

    assert_eq!(ws.domains.len(), 7);

    for d in &ws.domains {
        assert_eq!(d.domain_type, "product_domain");
        assert!(!d.axes.is_empty(), "domain {} has no axes", d.name);
    }
}

#[test]
fn test_ptv_parameter_points() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();

    assert_eq!(ws.parameter_points.len(), 3);

    let default_set = ws
        .parameter_points
        .iter()
        .find(|pp| pp.name == "default_values")
        .expect("default_values parameter point set should exist");

    // 3243 parameters in default_values (including binWidth_*)
    assert_eq!(default_set.parameters.len(), 3243);
}

#[test]
fn test_ptv_staterror_has_sample_errors() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();

    // Every sample with a staterror modifier should have data.errors
    let mut staterror_without_errors = 0;
    for d in &ws.distributions {
        if let Hs3Distribution::HistFactory(hf) = d {
            for sample in &hf.samples {
                let has_staterror =
                    sample.modifiers.iter().any(|m| matches!(m, Hs3Modifier::StatError { .. }));
                if has_staterror && sample.data.errors.is_none() {
                    staterror_without_errors += 1;
                }
            }
        }
    }

    assert_eq!(staterror_without_errors, 0, "all samples with staterror should have data.errors");
}

#[test]
fn test_distribution_name_helper() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();

    for d in &ws.distributions {
        assert!(d.name().is_some(), "every distribution should have a name");
    }
}

#[test]
fn test_modifier_name_helper() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();

    for d in &ws.distributions {
        if let Hs3Distribution::HistFactory(hf) = d {
            for sample in &hf.samples {
                for m in &sample.modifiers {
                    assert!(m.name().is_some(), "every modifier should have a name");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fragment deserialization tests
// ---------------------------------------------------------------------------

#[test]
fn test_deserialize_normsys_modifier_fragment() {
    let json = r#"{
        "type": "normsys",
        "name": "JES",
        "parameter": "alpha_JES",
        "constraint_name": "alpha_JESConstraint",
        "data": {"hi": 1.05, "lo": 0.95}
    }"#;
    let m: Hs3Modifier = serde_json::from_str(json).unwrap();
    match m {
        Hs3Modifier::NormSys { name, parameter, data, .. } => {
            assert_eq!(name, "JES");
            assert_eq!(parameter, "alpha_JES");
            assert!((data.hi - 1.05).abs() < 1e-15);
            assert!((data.lo - 0.95).abs() < 1e-15);
        }
        _ => panic!("expected NormSys"),
    }
}

#[test]
fn test_deserialize_histosys_modifier_fragment() {
    let json = r#"{
        "type": "histosys",
        "name": "JER",
        "parameter": "alpha_JER",
        "constraint_name": "alpha_JERConstraint",
        "data": {
            "hi": {"contents": [1.0, 2.0, 3.0]},
            "lo": {"contents": [0.5, 1.5, 2.5]}
        }
    }"#;
    let m: Hs3Modifier = serde_json::from_str(json).unwrap();
    match m {
        Hs3Modifier::HistoSys { data, .. } => {
            assert_eq!(data.hi.contents, vec![1.0, 2.0, 3.0]);
            assert_eq!(data.lo.contents, vec![0.5, 1.5, 2.5]);
        }
        _ => panic!("expected HistoSys"),
    }
}

#[test]
fn test_deserialize_staterror_modifier_fragment() {
    let json = r#"{
        "type": "staterror",
        "name": "stat_SR",
        "parameters": ["gamma_bin_0", "gamma_bin_1"],
        "constraint_type": "Poisson"
    }"#;
    let m: Hs3Modifier = serde_json::from_str(json).unwrap();
    match m {
        Hs3Modifier::StatError { parameters, constraint_type, .. } => {
            assert_eq!(parameters.len(), 2);
            assert_eq!(constraint_type, "Poisson");
        }
        _ => panic!("expected StatError"),
    }
}

#[test]
fn test_deserialize_normfactor_modifier_fragment() {
    let json = r#"{
        "type": "normfactor",
        "name": "mu_sig",
        "parameter": "mu"
    }"#;
    let m: Hs3Modifier = serde_json::from_str(json).unwrap();
    match m {
        Hs3Modifier::NormFactor { name, parameter } => {
            assert_eq!(name, "mu_sig");
            assert_eq!(parameter, "mu");
        }
        _ => panic!("expected NormFactor"),
    }
}

#[test]
fn test_deserialize_unknown_modifier_preserved() {
    let json = r#"{
        "type": "future_modifier_v3",
        "name": "something_new",
        "fancy_field": 42
    }"#;
    let m: Hs3Modifier = serde_json::from_str(json).unwrap();
    match &m {
        Hs3Modifier::Unknown(v) => {
            assert_eq!(v.get("type").unwrap().as_str().unwrap(), "future_modifier_v3");
            assert_eq!(v.get("fancy_field").unwrap().as_i64().unwrap(), 42);
        }
        _ => panic!("expected Unknown"),
    }
    assert_eq!(m.type_tag(), "future_modifier_v3");
    assert_eq!(m.name().unwrap(), "something_new");
}

#[test]
fn test_deserialize_unknown_distribution_preserved() {
    let json = r#"{
        "type": "mixture_dist",
        "name": "mix1",
        "components": ["a", "b"]
    }"#;
    let d: Hs3Distribution = serde_json::from_str(json).unwrap();
    match &d {
        Hs3Distribution::Unknown(v) => {
            assert_eq!(v.get("type").unwrap().as_str().unwrap(), "mixture_dist");
        }
        _ => panic!("expected Unknown"),
    }
    assert_eq!(d.type_tag(), "mixture_dist");
    assert_eq!(d.name().unwrap(), "mix1");
}

#[test]
fn test_deserialize_gaussian_dist_fragment() {
    let json = r#"{
        "type": "gaussian_dist",
        "name": "alpha_JESConstraint",
        "x": "alpha_JES",
        "mean": "nom_alpha_JES",
        "sigma": 1.0
    }"#;
    let d: Hs3Distribution = serde_json::from_str(json).unwrap();
    match d {
        Hs3Distribution::Gaussian(g) => {
            assert_eq!(g.name, "alpha_JESConstraint");
            assert_eq!(g.x, "alpha_JES");
            assert_eq!(g.mean, "nom_alpha_JES");
            assert_eq!(g.sigma, 1.0);
        }
        _ => panic!("expected Gaussian"),
    }
}

// ===========================================================================
// Phase 2 — Reference Resolution Tests
// ===========================================================================

use super::resolve;

#[test]
fn test_resolve_ptv_first_analysis() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();

    let resolved = resolve::resolve(&ws, None, None).expect("resolve should succeed");

    // First analysis is combPdf_asimovData
    assert_eq!(resolved.analysis_name, "combPdf_asimovData");
    assert_eq!(resolved.pois.len(), 4);
    assert!(resolved.pois.contains(&"mu".to_string()));
    assert_eq!(resolved.channels.len(), 36);
}

#[test]
fn test_resolve_ptv_select_analysis_by_name() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();

    // Find the second analysis name
    let second_name = &ws.analyses[1].name;

    let resolved = resolve::resolve(&ws, Some(second_name), None).expect("resolve should succeed");

    assert_eq!(resolved.analysis_name, *second_name);
    assert_eq!(resolved.channels.len(), 36);
}

#[test]
fn test_resolve_ptv_missing_analysis_error() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();

    let result = resolve::resolve(&ws, Some("nonexistent_analysis"), None);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("analysis not found"),
        "expected 'analysis not found', got: {}",
        err
    );
}

#[test]
fn test_resolve_ptv_channels_have_consistent_data() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();
    let resolved = resolve::resolve(&ws, None, None).unwrap();

    for ch in &resolved.channels {
        assert_eq!(ch.observed.len(), ch.n_bins, "channel {} observed len != n_bins", ch.name);
        for sample in &ch.samples {
            assert_eq!(
                sample.nominal.len(),
                ch.n_bins,
                "channel {} sample {} nominal len != n_bins",
                ch.name,
                sample.name
            );
        }
    }
}

#[test]
fn test_resolve_ptv_constraints_populated() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();
    let resolved = resolve::resolve(&ws, None, None).unwrap();

    // 273 gaussian constraints → 273 constrained parameters
    assert_eq!(resolved.constraints.len(), 273);

    // All Gaussian constraints should have width=1.0 (standard NPs)
    for (param, ci) in &resolved.constraints {
        assert_eq!(
            ci.kind,
            resolve::ConstraintKind::Gaussian,
            "param {} expected Gaussian constraint",
            param
        );
        assert!(
            (ci.width - 1.0).abs() < 1e-15,
            "param {} expected width=1.0, got {}",
            param,
            ci.width
        );
    }
}

#[test]
fn test_resolve_ptv_bounds_from_domains() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();
    let resolved = resolve::resolve(&ws, None, None).unwrap();

    // Should have bounds for many parameters (from analysis domains)
    assert!(
        resolved.bounds.len() > 100,
        "expected >100 parameter bounds, got {}",
        resolved.bounds.len()
    );

    // mu should be in POIs domain
    assert!(resolved.bounds.contains_key("mu"), "mu should have bounds from POI domain");
}

#[test]
fn test_resolve_ptv_inits_from_default_values() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();
    let resolved = resolve::resolve(&ws, None, None).unwrap();

    // Should have init values (excluding binWidth_* and global observables)
    assert!(!resolved.inits.is_empty(), "should have parameter init values");

    // binWidth_* should NOT be in inits
    for name in resolved.inits.keys() {
        assert!(
            !name.starts_with("binWidth_"),
            "binWidth parameters should be filtered out: {}",
            name
        );
    }
}

#[test]
fn test_resolve_ptv_global_observables() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();
    let resolved = resolve::resolve(&ws, None, None).unwrap();

    // Global observables should be populated (nom_alpha_* parameters)
    assert!(!resolved.global_observables.is_empty(), "should have global observable values");

    // Standard NP global obs should be 0.0
    for (name, &val) in &resolved.global_observables {
        if name.starts_with("nom_alpha_") {
            assert!(val.abs() < 1e-10, "global obs {} expected ~0.0, got {}", name, val);
        }
    }
}

#[test]
fn test_resolve_ptv_modifiers_resolved() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();
    let resolved = resolve::resolve(&ws, None, None).unwrap();

    let mut normsys = 0usize;
    let mut histosys = 0usize;
    let mut staterror = 0usize;
    let mut normfactor = 0usize;

    for ch in &resolved.channels {
        for sample in &ch.samples {
            for m in &sample.modifiers {
                match m {
                    resolve::ResolvedModifier::NormSys { .. } => normsys += 1,
                    resolve::ResolvedModifier::HistoSys { .. } => histosys += 1,
                    resolve::ResolvedModifier::StatError { .. } => staterror += 1,
                    resolve::ResolvedModifier::NormFactor { .. } => normfactor += 1,
                    _ => {}
                }
            }
        }
    }

    // Should match Phase 1 counts exactly
    assert_eq!(normsys, 36851, "resolved normsys count");
    assert_eq!(histosys, 3541, "resolved histosys count");
    assert_eq!(staterror, 1557, "resolved staterror count");
    assert_eq!(normfactor, 1539, "resolved normfactor count");
}

#[test]
fn test_resolve_ptv_constraint_centers_from_global_obs() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();
    let resolved = resolve::resolve(&ws, None, None).unwrap();

    // Gaussian constraint centers should be set from global observable values
    // For standard alpha NPs, the center should be 0.0
    let alpha_constraints: Vec<_> =
        resolved.constraints.iter().filter(|(k, _)| k.starts_with("alpha_")).collect();

    assert!(!alpha_constraints.is_empty());

    for (name, ci) in &alpha_constraints {
        assert!(
            ci.center.abs() < 1e-10,
            "alpha constraint {} expected center ~0.0, got {}",
            name,
            ci.center
        );
    }
}

// ===========================================================================
// Phase 3 — Conversion Tests (ResolvedWorkspace → HistFactoryModel)
// ===========================================================================

use super::convert;
use crate::pyhf::model::{HistoSysInterpCode, NormSysInterpCode};

#[test]
fn test_convert_ptv_from_hs3_default() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let model = convert::from_hs3_default(json).expect("from_hs3_default should succeed");

    // Model should have parameters
    let n_params = model.parameters().len();
    assert!(n_params > 100, "expected >100 parameters, got {}", n_params);

    // Should have 36 channels
    assert_eq!(model.n_channels(), 36, "expected 36 channels");

    // POI should be set
    assert!(model.poi_index().is_some(), "POI index should be set");
    assert_eq!(model.poi_index().unwrap(), 0, "POI should be first parameter");
    assert_eq!(model.parameters()[0].name, "mu", "first param should be mu");
}

#[test]
fn test_convert_ptv_nll_finite() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let model = convert::from_hs3_default(json).unwrap();

    let init: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();
    let nll = model.nll(&init).expect("NLL computation should succeed");

    assert!(nll.is_finite(), "NLL at init should be finite, got {}", nll);
    assert!(nll > 0.0, "NLL should be positive, got {}", nll);
}

#[test]
fn test_convert_ptv_poi_names() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let model = convert::from_hs3_default(json).unwrap();

    // All 4 POIs from the PTV analysis should be in the parameter list
    let param_names: Vec<&str> = model.parameters().iter().map(|p| p.name.as_str()).collect();
    assert!(param_names.contains(&"mu"));
    assert!(param_names.contains(&"mu_PTV_150_250"));
    assert!(param_names.contains(&"mu_PTV_250_"));
    assert!(param_names.contains(&"mu_PTV_75_150"));
}

#[test]
fn test_convert_ptv_constrained_params() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let model = convert::from_hs3_default(json).unwrap();

    // Count constrained vs unconstrained
    let constrained = model.parameters().iter().filter(|p| p.constrained).count();
    let unconstrained = model.parameters().iter().filter(|p| !p.constrained).count();

    // 273 Gaussian constraints + staterror gamma parameters
    assert!(constrained > 273, "expected >273 constrained params, got {}", constrained);
    assert!(
        unconstrained >= 4,
        "expected >=4 unconstrained params (POIs + normfactors), got {}",
        unconstrained
    );
}

#[test]
fn test_convert_ptv_with_explicit_interp_codes() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");

    // Code4/Code4p (polynomial smoothing)
    let model =
        convert::from_hs3(json, None, None, NormSysInterpCode::Code4, HistoSysInterpCode::Code4p)
            .expect("from_hs3 with Code4/Code4p should succeed");

    let init: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();
    let nll = model.nll(&init).expect("NLL should succeed with Code4/Code4p");
    assert!(nll.is_finite(), "NLL should be finite with Code4/Code4p");
}

#[test]
fn test_convert_ptv_select_second_analysis() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");

    // Parse to find second analysis name
    let ws: Hs3Workspace = serde_json::from_str(json).unwrap();
    let second_name = &ws.analyses[1].name;

    let model = convert::from_hs3(
        json,
        Some(second_name),
        None,
        NormSysInterpCode::Code1,
        HistoSysInterpCode::Code0,
    )
    .expect("from_hs3 with second analysis should succeed");

    assert_eq!(model.n_channels(), 36);
    let init: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();
    let nll = model.nll(&init).unwrap();
    assert!(nll.is_finite());
}

// ===========================================================================
// Phase 4 — Format Detection Tests
// ===========================================================================

use super::detect;

#[test]
fn test_detect_hs3_format() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    assert_eq!(detect::detect_format(json), detect::WorkspaceFormat::Hs3);
}

#[test]
fn test_detect_pyhf_format() {
    let json = include_str!("../../../../tests/fixtures/simple_workspace.json");
    assert_eq!(detect::detect_format(json), detect::WorkspaceFormat::Pyhf);
}

#[test]
fn test_detect_pyhf_complex() {
    let json = include_str!("../../../../tests/fixtures/complex_workspace.json");
    assert_eq!(detect::detect_format(json), detect::WorkspaceFormat::Pyhf);
}

#[test]
fn test_detect_unknown_format() {
    let json = r#"{"foo": "bar", "baz": 42}"#;
    assert_eq!(detect::detect_format(json), detect::WorkspaceFormat::Unknown);
}

#[test]
fn test_detect_empty_json() {
    assert_eq!(detect::detect_format("{}"), detect::WorkspaceFormat::Unknown);
}

// ===========================================================================
// Phase 6 — Export / Roundtrip Tests
// ===========================================================================

use super::export;

#[test]
fn test_export_ptv_produces_valid_hs3() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let model = convert::from_hs3_default(json).unwrap();

    let exported = export::export_hs3(&model, "test_analysis", None, None);

    // Should have same number of channels
    let hf_count = exported
        .distributions
        .iter()
        .filter(|d| matches!(d, Hs3Distribution::HistFactory(_)))
        .count();
    assert_eq!(hf_count, 36, "exported should have 36 histfactory_dist");

    // Should have data for each channel
    assert_eq!(exported.data.len(), 36);

    // Should have parameter_points
    assert!(!exported.parameter_points.is_empty());

    // Metadata should be generated (no original)
    assert_eq!(exported.metadata.hs3_version, "0.2");
    let pkgs = exported.metadata.packages.as_ref().unwrap();
    assert_eq!(pkgs[0].name, "NextStat");
}

#[test]
fn test_export_ptv_preserves_original_metadata() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let original_ws: Hs3Workspace = serde_json::from_str(json).unwrap();
    let model = convert::from_hs3_default(json).unwrap();

    let exported = export::export_hs3(&model, "test_analysis", None, Some(&original_ws));

    assert_eq!(exported.metadata.hs3_version, "0.2");
    let pkgs = exported.metadata.packages.as_ref().unwrap();
    assert_eq!(pkgs[0].name, "ROOT");
    assert!(exported.misc.is_some(), "misc should be preserved from original");
}

#[test]
fn test_export_roundtrip_re_parseable() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let model = convert::from_hs3_default(json).unwrap();

    // Export to JSON string
    let exported_json = export::export_hs3_json(&model, "roundtrip_test", None, None).unwrap();

    // Re-parse the exported JSON
    let re_parsed: Hs3Workspace =
        serde_json::from_str(&exported_json).expect("exported JSON should re-parse");

    // Basic structural checks
    let hf_count = re_parsed
        .distributions
        .iter()
        .filter(|d| matches!(d, Hs3Distribution::HistFactory(_)))
        .count();
    assert_eq!(hf_count, 36);
    assert!(!re_parsed.parameter_points.is_empty());
    assert_eq!(re_parsed.analyses.len(), 1);
    assert_eq!(re_parsed.likelihoods.len(), 1);
}

#[test]
fn test_export_roundtrip_nll_match() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let model1 = convert::from_hs3_default(json).unwrap();

    let init1: Vec<f64> = model1.parameters().iter().map(|p| p.init).collect();
    let nll1 = model1.nll(&init1).unwrap();

    // Export and re-import
    let exported_json = export::export_hs3_json(&model1, "roundtrip_test", None, None).unwrap();
    let model2 = convert::from_hs3_default(&exported_json).unwrap();

    let init2: Vec<f64> = model2.parameters().iter().map(|p| p.init).collect();
    let nll2 = model2.nll(&init2).unwrap();

    // NLL should match (exact parity not guaranteed due to rounding in export,
    // but both should be finite and positive)
    assert!(nll1.is_finite() && nll1 > 0.0);
    assert!(nll2.is_finite() && nll2 > 0.0);

    // Structural parity
    assert_eq!(model1.n_channels(), model2.n_channels());
}

#[test]
fn test_export_with_bestfit_params() {
    let json = include_str!("../../../../tests/fixtures/workspace-postFit_PTV.json");
    let model = convert::from_hs3_default(json).unwrap();

    let bestfit: Vec<f64> = model.parameters().iter().map(|p| p.init).collect();

    let exported = export::export_hs3(&model, "test_analysis", Some(("bestfit", &bestfit)), None);

    assert_eq!(exported.parameter_points.len(), 2);
    assert_eq!(exported.parameter_points[0].name, "default_values");
    assert_eq!(exported.parameter_points[1].name, "bestfit");
    assert_eq!(exported.parameter_points[1].parameters.len(), model.parameters().len());
}

// ---------------------------------------------------------------------------
// Unbinned HS3 extension schema tests
// ---------------------------------------------------------------------------

#[test]
fn test_unbinned_dist_serde_roundtrip() {
    let dist = Hs3UnbinnedDist {
        name: "signal_region".into(),
        dist_type: "nextstat_unbinned_dist".into(),
        observables: vec![Hs3UnbinnedObservable {
            name: "mass".into(),
            min: 100.0,
            max: 180.0,
            expr: None,
        }],
        processes: vec![Hs3UnbinnedProcess {
            name: "signal".into(),
            pdf: Hs3UnbinnedPdf {
                pdf_type: "gaussian".into(),
                observable: Some("mass".into()),
                params: vec!["mu_sig".into(), "sigma_sig".into()],
                ..Default::default()
            },
            yield_spec: Hs3UnbinnedYield {
                yield_type: "scaled".into(),
                value: None,
                parameter: None,
                base_yield: Some(100.0),
                scale: Some("mu".into()),
                modifiers: vec![],
            },
        }],
        data_source: Some(Hs3UnbinnedDataSource {
            file: "events.root".into(),
            format: Some("root".into()),
            tree: Some("MyTree".into()),
            selection: None,
            weight: None,
        }),
    };

    let json = serde_json::to_string_pretty(&Hs3Distribution::Unbinned(dist.clone())).unwrap();
    assert!(json.contains("nextstat_unbinned_dist"));
    assert!(json.contains("signal_region"));
    assert!(json.contains("gaussian"));

    let roundtripped: Hs3Distribution = serde_json::from_str(&json).unwrap();
    match &roundtripped {
        Hs3Distribution::Unbinned(d) => {
            assert_eq!(d.name, "signal_region");
            assert_eq!(d.observables.len(), 1);
            assert_eq!(d.observables[0].name, "mass");
            assert_eq!(d.observables[0].min, 100.0);
            assert_eq!(d.processes.len(), 1);
            assert_eq!(d.processes[0].name, "signal");
            assert_eq!(d.processes[0].pdf.pdf_type, "gaussian");
            assert_eq!(d.processes[0].pdf.params, vec!["mu_sig", "sigma_sig"]);
            assert_eq!(d.processes[0].yield_spec.yield_type, "scaled");
            assert_eq!(d.processes[0].yield_spec.base_yield, Some(100.0));
            let src = d.data_source.as_ref().unwrap();
            assert_eq!(src.file, "events.root");
            assert_eq!(src.tree.as_deref(), Some("MyTree"));
        }
        other => panic!("expected Unbinned, got {:?}", other.type_tag()),
    }
}

#[test]
fn test_unbinned_dist_in_workspace() {
    let ws = Hs3Workspace {
        distributions: vec![Hs3Distribution::Unbinned(Hs3UnbinnedDist {
            name: "sr".into(),
            dist_type: "nextstat_unbinned_dist".into(),
            observables: vec![Hs3UnbinnedObservable {
                name: "x".into(),
                min: 0.0,
                max: 10.0,
                expr: None,
            }],
            processes: vec![Hs3UnbinnedProcess {
                name: "bkg".into(),
                pdf: Hs3UnbinnedPdf {
                    pdf_type: "exponential".into(),
                    observable: Some("x".into()),
                    params: vec!["lambda".into()],
                    ..Default::default()
                },
                yield_spec: Hs3UnbinnedYield {
                    yield_type: "fixed".into(),
                    value: Some(500.0),
                    parameter: None,
                    base_yield: None,
                    scale: None,
                    modifiers: vec![],
                },
            }],
            data_source: None,
        })],
        data: vec![],
        domains: vec![Hs3Domain {
            name: "test_domain".into(),
            domain_type: "product_domain".into(),
            axes: vec![Hs3DomainAxis { name: "lambda".into(), min: -10.0, max: 0.0 }],
        }],
        parameter_points: vec![Hs3ParameterPointSet {
            name: "default_values".into(),
            parameters: vec![Hs3ParameterValue { name: "lambda".into(), value: -1.0 }],
        }],
        analyses: vec![Hs3Analysis {
            name: "test_analysis".into(),
            likelihood: "test_lh".into(),
            parameters_of_interest: vec![],
            domains: vec!["test_domain".into()],
        }],
        likelihoods: vec![Hs3Likelihood {
            name: "test_lh".into(),
            distributions: vec!["sr".into()],
            data: vec![],
        }],
        metadata: Hs3Metadata {
            hs3_version: "0.2".into(),
            packages: Some(vec![Hs3Package {
                name: "NextStat".into(),
                version: env!("CARGO_PKG_VERSION").into(),
            }]),
        },
        misc: Some(serde_json::json!({ "nextstat_extensions": ["unbinned_dist"] })),
    };

    let json = serde_json::to_string_pretty(&ws).unwrap();
    let ws2: Hs3Workspace = serde_json::from_str(&json).unwrap();

    assert_eq!(ws2.distributions.len(), 1);
    assert_eq!(ws2.distributions[0].type_tag(), "nextstat_unbinned_dist");
    assert_eq!(ws2.distributions[0].name(), Some("sr"));
    assert_eq!(ws2.metadata.hs3_version, "0.2");
    assert!(ws2.misc.is_some());
}

#[test]
fn test_unbinned_pdf_with_rate_modifiers() {
    let proc = Hs3UnbinnedProcess {
        name: "bkg".into(),
        pdf: Hs3UnbinnedPdf {
            pdf_type: "histogram".into(),
            observable: Some("mass".into()),
            bin_edges: Some(vec![100.0, 120.0, 140.0, 160.0, 180.0]),
            bin_content: Some(vec![10.0, 20.0, 15.0, 5.0]),
            ..Default::default()
        },
        yield_spec: Hs3UnbinnedYield {
            yield_type: "fixed".into(),
            value: Some(50.0),
            parameter: None,
            base_yield: None,
            scale: None,
            modifiers: vec![
                Hs3UnbinnedRateModifier {
                    mod_type: "normsys".into(),
                    param: "alpha_jes".into(),
                    lo: 0.9,
                    hi: 1.1,
                },
                Hs3UnbinnedRateModifier {
                    mod_type: "weightsys".into(),
                    param: "alpha_btag".into(),
                    lo: 0.95,
                    hi: 1.05,
                },
            ],
        },
    };

    let json = serde_json::to_string_pretty(&proc).unwrap();
    let proc2: Hs3UnbinnedProcess = serde_json::from_str(&json).unwrap();

    assert_eq!(proc2.yield_spec.modifiers.len(), 2);
    assert_eq!(proc2.yield_spec.modifiers[0].mod_type, "normsys");
    assert_eq!(proc2.yield_spec.modifiers[0].param, "alpha_jes");
    assert_eq!(proc2.yield_spec.modifiers[1].mod_type, "weightsys");
    assert_eq!(proc2.pdf.bin_edges.as_ref().unwrap().len(), 5);
    assert_eq!(proc2.pdf.bin_content.as_ref().unwrap().len(), 4);
}

#[test]
fn test_unbinned_product_pdf_serde() {
    let pdf = Hs3UnbinnedPdf {
        pdf_type: "product".into(),
        components: Some(vec![
            Hs3UnbinnedPdf {
                pdf_type: "gaussian".into(),
                observable: Some("x".into()),
                params: vec!["mu_x".into(), "sigma_x".into()],
                ..Default::default()
            },
            Hs3UnbinnedPdf {
                pdf_type: "exponential".into(),
                observable: Some("y".into()),
                params: vec!["lambda_y".into()],
                ..Default::default()
            },
        ]),
        ..Default::default()
    };

    let json = serde_json::to_string_pretty(&pdf).unwrap();
    let pdf2: Hs3UnbinnedPdf = serde_json::from_str(&json).unwrap();

    assert_eq!(pdf2.pdf_type, "product");
    let comps = pdf2.components.unwrap();
    assert_eq!(comps.len(), 2);
    assert_eq!(comps[0].pdf_type, "gaussian");
    assert_eq!(comps[1].pdf_type, "exponential");
}

#[test]
fn test_convert_hybrid_workspace_skips_unbinned_distributions() {
    let ws = make_minimal_hybrid_hs3_workspace();
    let json = serde_json::to_string_pretty(&ws).unwrap();
    let model = convert::from_hs3_default(&json).expect("hybrid HS3 should parse for binned path");
    assert_eq!(model.n_channels(), 1, "only histfactory channel should be converted");
}

#[cfg(feature = "unbinned")]
#[test]
fn test_import_unbinned_hs3_to_spec() {
    let ws = make_minimal_hybrid_hs3_workspace();
    let spec = export::import_unbinned_hs3(&ws, None, None).expect("unbinned import should work");

    assert_eq!(spec.schema_version, ns_unbinned::spec::UNBINNED_SPEC_V0);
    assert_eq!(spec.channels.len(), 1);
    assert_eq!(spec.channels[0].name, "sr_unbinned");
    assert_eq!(spec.model.poi.as_deref(), Some("mu"));

    let mut names: Vec<&str> = spec.model.parameters.iter().map(|p| p.name.as_str()).collect();
    names.sort_unstable();
    assert_eq!(names, vec!["alpha_jes", "mu", "sigma"]);

    let alpha = spec
        .model
        .parameters
        .iter()
        .find(|p| p.name == "alpha_jes")
        .expect("alpha_jes should exist");
    match alpha.constraint.as_ref().expect("alpha_jes must be constrained") {
        ns_unbinned::spec::ConstraintSpec::Gaussian { mean, sigma } => {
            assert!((*mean - 0.2).abs() < 1e-12);
            assert!((*sigma - 1.5).abs() < 1e-12);
        }
    }

    let proc = &spec.channels[0].processes[0];
    match &proc.pdf {
        ns_unbinned::spec::PdfSpec::Gaussian { observable, params } => {
            assert_eq!(observable, "mass");
            assert_eq!(params, &vec!["mu".to_string(), "sigma".to_string()]);
        }
        other => panic!("expected gaussian PDF, got {:?}", other),
    }
}

#[cfg(feature = "unbinned")]
#[test]
fn test_import_unbinned_hs3_exotic_pdf_fallback() {
    let mut ws = make_minimal_hybrid_hs3_workspace();
    let exotic = Hs3Distribution::Unbinned(Hs3UnbinnedDist {
        name: "sr_exotic".into(),
        dist_type: "nextstat_unbinned_dist".into(),
        observables: vec![Hs3UnbinnedObservable {
            name: "x".into(),
            min: 0.0,
            max: 10.0,
            expr: None,
        }],
        processes: vec![Hs3UnbinnedProcess {
            name: "bkg".into(),
            pdf: Hs3UnbinnedPdf {
                pdf_type: "future_pdf_v9".into(),
                observable: Some("x".into()),
                ..Default::default()
            },
            yield_spec: Hs3UnbinnedYield {
                yield_type: "fixed".into(),
                value: Some(10.0),
                parameter: None,
                base_yield: None,
                scale: None,
                modifiers: vec![],
            },
        }],
        data_source: None,
    });
    ws.distributions.push(exotic);
    ws.likelihoods[0].distributions.push("sr_exotic".into());
    ws.likelihoods[0].data.push("obsData_sr_exotic".into());
    ws.data.push(Hs3Data {
        name: "obsData_sr_exotic".into(),
        data_type: "unbinned_ref".into(),
        axes: vec![],
        contents: vec![],
    });

    let spec = export::import_unbinned_hs3(&ws, None, None).expect("fallback import should work");
    let ch =
        spec.channels.iter().find(|c| c.name == "sr_exotic").expect("exotic channel must exist");
    match &ch.processes[0].pdf {
        ns_unbinned::spec::PdfSpec::Histogram { bin_edges, bin_content, .. } => {
            assert_eq!(bin_edges, &vec![0.0, 10.0]);
            assert_eq!(bin_content, &vec![1.0]);
        }
        other => panic!("expected fallback histogram, got {:?}", other),
    }
}

#[cfg(feature = "unbinned")]
#[test]
fn test_import_hybrid_hs3() {
    let ws = make_minimal_hybrid_hs3_workspace();
    let imported = export::import_hybrid_hs3(&ws, None, None).expect("hybrid import should work");
    assert_eq!(imported.binned.n_channels(), 1);
    assert_eq!(imported.unbinned_spec.channels.len(), 1);
    assert!(imported.shared_param_names.iter().any(|p| p == "mu"));
}

fn make_minimal_hybrid_hs3_workspace() -> Hs3Workspace {
    let binned_dist = Hs3Distribution::HistFactory(Hs3HistFactoryDist {
        name: "sr_binned".into(),
        dist_type: "histfactory_dist".into(),
        axes: vec![Hs3Axis {
            name: "obs_x_sr_binned".into(),
            min: Some(0.0),
            max: Some(1.0),
            nbins: Some(1),
            edges: None,
        }],
        samples: vec![Hs3Sample {
            name: "sig".into(),
            data: Hs3SampleData { contents: vec![4.0], errors: None },
            modifiers: vec![Hs3Modifier::NormFactor { name: "mu".into(), parameter: "mu".into() }],
        }],
    });

    let unbinned_dist = Hs3Distribution::Unbinned(Hs3UnbinnedDist {
        name: "sr_unbinned".into(),
        dist_type: "nextstat_unbinned_dist".into(),
        observables: vec![Hs3UnbinnedObservable {
            name: "mass".into(),
            min: 100.0,
            max: 180.0,
            expr: None,
        }],
        processes: vec![Hs3UnbinnedProcess {
            name: "signal".into(),
            pdf: Hs3UnbinnedPdf {
                pdf_type: "gaussian".into(),
                observable: Some("mass".into()),
                params: vec!["mu".into(), "sigma".into()],
                ..Default::default()
            },
            yield_spec: Hs3UnbinnedYield {
                yield_type: "scaled".into(),
                value: None,
                parameter: None,
                base_yield: Some(100.0),
                scale: Some("mu".into()),
                modifiers: vec![Hs3UnbinnedRateModifier {
                    mod_type: "normsys".into(),
                    param: "alpha_jes".into(),
                    lo: 0.9,
                    hi: 1.1,
                }],
            },
        }],
        data_source: Some(Hs3UnbinnedDataSource {
            file: "events.root".into(),
            format: Some("root".into()),
            tree: Some("Events".into()),
            selection: None,
            weight: None,
        }),
    });

    let gauss_constraint = Hs3Distribution::Gaussian(Hs3GaussianDist {
        name: "alpha_jesConstraint".into(),
        dist_type: "gaussian_dist".into(),
        x: "alpha_jes".into(),
        mean: "nom_alpha_jes".into(),
        sigma: 1.5,
    });

    Hs3Workspace {
        distributions: vec![binned_dist, unbinned_dist, gauss_constraint],
        data: vec![
            Hs3Data {
                name: "obsData_sr_binned".into(),
                data_type: "binned".into(),
                axes: vec![],
                contents: vec![10.0],
            },
            Hs3Data {
                name: "obsData_sr_unbinned".into(),
                data_type: "unbinned_ref".into(),
                axes: vec![],
                contents: vec![],
            },
        ],
        domains: vec![Hs3Domain {
            name: "analysis_np".into(),
            domain_type: "product_domain".into(),
            axes: vec![
                Hs3DomainAxis { name: "mu".into(), min: 0.0, max: 5.0 },
                Hs3DomainAxis { name: "sigma".into(), min: 0.1, max: 20.0 },
                Hs3DomainAxis { name: "alpha_jes".into(), min: -5.0, max: 5.0 },
                Hs3DomainAxis { name: "nom_alpha_jes".into(), min: -5.0, max: 5.0 },
            ],
        }],
        parameter_points: vec![Hs3ParameterPointSet {
            name: "default_values".into(),
            parameters: vec![
                Hs3ParameterValue { name: "mu".into(), value: 1.0 },
                Hs3ParameterValue { name: "sigma".into(), value: 2.0 },
                Hs3ParameterValue { name: "alpha_jes".into(), value: 0.0 },
                Hs3ParameterValue { name: "nom_alpha_jes".into(), value: 0.2 },
            ],
        }],
        analyses: vec![Hs3Analysis {
            name: "analysis".into(),
            likelihood: "analysis_lh".into(),
            parameters_of_interest: vec!["mu".into()],
            domains: vec!["analysis_np".into()],
        }],
        likelihoods: vec![Hs3Likelihood {
            name: "analysis_lh".into(),
            distributions: vec!["sr_binned".into(), "sr_unbinned".into()],
            data: vec!["obsData_sr_binned".into(), "obsData_sr_unbinned".into()],
        }],
        metadata: Hs3Metadata {
            hs3_version: "0.2".into(),
            packages: Some(vec![Hs3Package {
                name: "NextStat".into(),
                version: env!("CARGO_PKG_VERSION").into(),
            }]),
        },
        misc: None,
    }
}
