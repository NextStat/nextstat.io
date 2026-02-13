//! Integration tests for FlowPdf and the unbinned fit pipeline.
//!
//! These tests require:
//! - `--features neural` to compile
//! - Test ONNX fixtures under `tests/fixtures/flow_test` (regenerate via `scripts/neural/generate_test_fixtures.py`)
//!
//! Run: `cargo test -p ns-unbinned --features neural --test flow_integration`

#![cfg(feature = "neural")]

use std::path::PathBuf;
use std::sync::Arc;

use approx::assert_relative_eq;
use ns_inference::MaximumLikelihoodEstimator;
use ns_unbinned::event_store::{EventStore, ObservableSpec};
use ns_unbinned::normalize::{QuadratureGrid, QuadratureOrder, log_normalize_quadrature};
use ns_unbinned::pdf::UnbinnedPdf;
use ns_unbinned::{
    FlowPdf, GaussianPdf, Parameter, Process, UnbinnedChannel, UnbinnedModel, YieldExpr,
};
use rand::SeedableRng;

/// Path to the test flow fixtures directory.
fn fixture_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/
    p.pop(); // repo root
    p.push("tests/fixtures/flow_test");
    p
}

fn manifest_path() -> PathBuf {
    fixture_dir().join("flow_manifest.json")
}

fn has_fixtures() -> bool {
    manifest_path().exists()
}

fn make_flow_dataset(n_events: usize, seed: u64) -> EventStore {
    let flow = FlowPdf::from_manifest(&manifest_path(), &[]).unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    flow.sample(&[], n_events, &[(-6.0, 6.0)], &mut rng).unwrap()
}

// ═══════════════════════════════════════════════════════════════════════════
// D2: Flow PDF basic functionality
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_flow_pdf_loads_from_manifest() {
    if !has_fixtures() {
        eprintln!("SKIP: flow test fixtures not found under tests/fixtures/flow_test");
        return;
    }

    let flow = FlowPdf::from_manifest(&manifest_path(), &[]).unwrap();
    assert_eq!(flow.n_params(), 0); // unconditional
    assert_eq!(flow.observables(), &["x".to_string()]);
}

#[test]
fn test_flow_pdf_log_prob_matches_standard_normal() {
    if !has_fixtures() {
        eprintln!("SKIP: flow test fixtures not found.");
        return;
    }

    let flow = FlowPdf::from_manifest(&manifest_path(), &[]).unwrap();

    let obs = ObservableSpec::branch("x", (-6.0, 6.0));
    let xs = vec![0.0, 1.0, -1.0, 2.0, -2.0];
    let store =
        EventStore::from_columns(vec![obs], vec![("x".to_string(), xs.clone())], None).unwrap();

    let mut out = vec![0.0; xs.len()];
    flow.log_prob_batch(&store, &[], &mut out).unwrap();

    // Compare with analytical standard normal log-prob.
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    for (i, &x) in xs.iter().enumerate() {
        let expected = -0.5 * x * x - 0.5 * ln2pi;
        assert_relative_eq!(out[i], expected, epsilon = 1e-5);
    }
}

#[test]
fn test_flow_pdf_sample() {
    if !has_fixtures() {
        eprintln!("SKIP: flow test fixtures not found.");
        return;
    }

    let flow = FlowPdf::from_manifest(&manifest_path(), &[]).unwrap();

    let mut rng = rand::rng();
    let sampled = flow.sample(&[], 1000, &[(-6.0, 6.0)], &mut rng).unwrap();

    assert_eq!(sampled.n_events(), 1000);
    let col = sampled.column("x").unwrap();

    // For a standard normal, mean ≈ 0 and std ≈ 1 (with some tolerance for N=1000).
    let mean: f64 = col.iter().sum::<f64>() / col.len() as f64;
    let var: f64 = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / col.len() as f64;

    assert!(mean.abs() < 0.15, "sample mean should be ≈ 0, got {mean}");
    assert!((var.sqrt() - 1.0).abs() < 0.15, "sample std should be ≈ 1, got {}", var.sqrt());
}

// ═══════════════════════════════════════════════════════════════════════════
// D2: Flow PDF fit integration (unbinned MLE)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_flow_pdf_normalization_check() {
    if !has_fixtures() {
        eprintln!("SKIP: flow test fixtures not found.");
        return;
    }

    let flow = FlowPdf::from_manifest(&manifest_path(), &[]).unwrap();

    // Integrate flow over its support [-6, 6] via quadrature.
    let grid = QuadratureGrid::new_1d("x", (-6.0, 6.0), QuadratureOrder::N128).unwrap();
    let log_int = log_normalize_quadrature(&flow, &grid, &[]).unwrap();

    // Standard normal on [-6, 6] is essentially 1.0.
    assert_relative_eq!(log_int.exp(), 1.0, epsilon = 1e-4);
}

#[test]
fn test_flow_pdf_grad_batch_unconditional() {
    if !has_fixtures() {
        eprintln!("SKIP: flow test fixtures not found.");
        return;
    }

    let flow = FlowPdf::from_manifest(&manifest_path(), &[]).unwrap();

    let obs = ObservableSpec::branch("x", (-6.0, 6.0));
    let store =
        EventStore::from_columns(vec![obs], vec![("x".to_string(), vec![0.0, 1.0, -1.0])], None)
            .unwrap();

    let n = store.n_events();
    let mut logp = vec![0.0; n];
    // n_params = 0, so grad is empty.
    let mut grad = vec![];
    flow.log_prob_grad_batch(&store, &[], &mut logp, &mut grad).unwrap();

    // log_prob should still be correct.
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    assert_relative_eq!(logp[0], -0.5 * ln2pi, epsilon = 1e-5);
}

// ═══════════════════════════════════════════════════════════════════════════
// D3: Baseline compare — flow vs parametric on known distribution
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_flow_vs_gaussian_nll_surface() {
    if !has_fixtures() {
        eprintln!("SKIP: flow test fixtures not found.");
        return;
    }

    let flow = FlowPdf::from_manifest(&manifest_path(), &[]).unwrap();
    let gauss = GaussianPdf::new("x");

    // Create test data: samples from a standard normal (the flow's true distribution).
    let obs = ObservableSpec::branch("x", (-6.0, 6.0));
    let xs = vec![
        -1.5, -0.8, -0.3, 0.1, 0.4, 0.7, 1.2, 1.8, -0.6, 0.0, -2.1, 0.5, 1.0, -1.0, 0.2, -0.4, 0.9,
        -0.1, 1.5, -0.7,
    ];
    let n = xs.len();
    let store = EventStore::from_columns(vec![obs], vec![("x".to_string(), xs)], None).unwrap();

    // Flow NLL (no params).
    let mut flow_logp = vec![0.0; n];
    flow.log_prob_batch(&store, &[], &mut flow_logp).unwrap();
    let flow_nll: f64 = flow_logp.iter().map(|lp| -lp).sum();

    // Gaussian NLL at the true parameters (mu=0, sigma=1).
    let mut gauss_logp = vec![0.0; n];
    gauss.log_prob_batch(&store, &[0.0, 1.0], &mut gauss_logp).unwrap();
    let gauss_nll: f64 = gauss_logp.iter().map(|lp| -lp).sum();

    // The flow should produce very similar NLL to the Gaussian at the true params,
    // since the flow IS a standard normal.
    let nll_diff = (flow_nll - gauss_nll).abs();
    assert!(
        nll_diff < 0.5,
        "Flow NLL ({flow_nll:.4}) should be close to Gaussian NLL ({gauss_nll:.4}), diff={nll_diff:.4}"
    );
}

#[test]
fn test_flow_vs_gaussian_per_event_logprob() {
    if !has_fixtures() {
        eprintln!("SKIP: flow test fixtures not found.");
        return;
    }

    let flow = FlowPdf::from_manifest(&manifest_path(), &[]).unwrap();
    let gauss = GaussianPdf::new("x");

    let obs = ObservableSpec::branch("x", (-6.0, 6.0));
    let xs = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let n = xs.len();
    let store = EventStore::from_columns(vec![obs], vec![("x".to_string(), xs)], None).unwrap();

    let mut flow_logp = vec![0.0; n];
    let mut gauss_logp = vec![0.0; n];
    flow.log_prob_batch(&store, &[], &mut flow_logp).unwrap();
    gauss.log_prob_batch(&store, &[0.0, 1.0], &mut gauss_logp).unwrap();

    // Per-event log-prob should match closely (flow is standard normal,
    // Gaussian at mu=0, sigma=1 is also standard normal but truncated to [-6,6]).
    for i in 0..n {
        assert_relative_eq!(flow_logp[i], gauss_logp[i], epsilon = 0.01);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// D3: End-to-end fit + toys (FlowPdf vs parametric Gaussian)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_flowpdf_unbinned_mle_recovers_yield_parameter() {
    if !has_fixtures() {
        eprintln!("SKIP: flow test fixtures not found.");
        return;
    }

    let n_events = 500usize;
    let base_yield = n_events as f64; // ensures mu_hat ≈ 1 for a single-process extended model
    let data = Arc::new(make_flow_dataset(n_events, 0));

    let flow = Arc::new(FlowPdf::from_manifest(&manifest_path(), &[]).unwrap());

    // Parameter vector: [mu]
    let parameters =
        vec![Parameter { name: "mu".to_string(), init: 0.8, bounds: (0.0, 5.0), constraint: None }];

    let channel = UnbinnedChannel {
        name: "ch0".to_string(),
        include_in_fit: true,
        data,
        processes: vec![Process {
            name: "signal".to_string(),
            pdf: flow,
            shape_param_indices: vec![],
            yield_expr: YieldExpr::Scaled { base_yield, scale_index: 0 },
        }],
    };

    let model = UnbinnedModel::new(parameters, vec![channel], Some(0)).unwrap();

    let mle = MaximumLikelihoodEstimator::new();
    let opt = mle.fit_minimum(&model).unwrap();
    assert!(opt.converged, "MLE should converge, message={}", opt.message);
    assert_relative_eq!(opt.parameters[0], 1.0, epsilon = 5e-3);
}

#[test]
fn test_flowpdf_toys_fit_smoke_and_matches_gaussian_baseline() {
    if !has_fixtures() {
        eprintln!("SKIP: flow test fixtures not found.");
        return;
    }

    // Build two models on the same observed data:
    // - FlowPdf (unconditional standard normal)
    // - GaussianPdf with fixed (mu=0, sigma=1)
    //
    // Both should recover essentially the same yield scale parameter.
    let n_events = 300usize;
    let base_yield = n_events as f64;
    let observed = Arc::new(make_flow_dataset(n_events, 1));

    let mle = MaximumLikelihoodEstimator::new();

    // Flow model: params [mu]
    let flow_model = {
        let parameters = vec![Parameter {
            name: "mu".to_string(),
            init: 1.2,
            bounds: (0.0, 5.0),
            constraint: None,
        }];
        let channel = UnbinnedChannel {
            name: "ch0".to_string(),
            include_in_fit: true,
            data: observed.clone(),
            processes: vec![Process {
                name: "signal".to_string(),
                pdf: Arc::new(FlowPdf::from_manifest(&manifest_path(), &[]).unwrap()),
                shape_param_indices: vec![],
                yield_expr: YieldExpr::Scaled { base_yield, scale_index: 0 },
            }],
        };
        UnbinnedModel::new(parameters, vec![channel], Some(0)).unwrap()
    };

    // Gaussian baseline: params [mean (fixed), sigma (fixed), mu]
    let gauss_model = {
        let parameters = vec![
            Parameter { name: "mean".to_string(), init: 0.0, bounds: (0.0, 0.0), constraint: None },
            Parameter {
                name: "sigma".to_string(),
                init: 1.0,
                bounds: (1.0, 1.0),
                constraint: None,
            },
            Parameter { name: "mu".to_string(), init: 1.2, bounds: (0.0, 5.0), constraint: None },
        ];
        let channel = UnbinnedChannel {
            name: "ch0".to_string(),
            include_in_fit: true,
            data: observed,
            processes: vec![Process {
                name: "signal".to_string(),
                pdf: Arc::new(GaussianPdf::new("x")),
                shape_param_indices: vec![0, 1],
                yield_expr: YieldExpr::Scaled { base_yield, scale_index: 2 },
            }],
        };
        UnbinnedModel::new(parameters, vec![channel], Some(2)).unwrap()
    };

    let opt_flow = mle.fit_minimum(&flow_model).unwrap();
    let opt_gauss = mle.fit_minimum(&gauss_model).unwrap();
    assert!(opt_flow.converged, "flow MLE should converge, msg={}", opt_flow.message);
    assert!(opt_gauss.converged, "gauss MLE should converge, msg={}", opt_gauss.message);
    assert_relative_eq!(opt_flow.parameters[0], opt_gauss.parameters[2], epsilon = 5e-3);

    // Toys: sample + fit (FlowPdf path). This exercises ONNX sampling via the sample model.
    let toy_params = [1.0];
    for seed in 0..10u64 {
        let toy = flow_model.sample_poisson_toy(&toy_params, seed).unwrap();
        let opt = mle.fit_minimum(&toy).unwrap();
        assert!(opt.converged, "toy fit should converge (seed={seed}), msg={}", opt.message);
        assert!(
            (0.6..=1.4).contains(&opt.parameters[0]),
            "toy mu_hat out of expected range (seed={seed}): {}",
            opt.parameters[0]
        );
    }
}
