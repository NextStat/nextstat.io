//! G3 Integration tests: analytical gradient for conditional flows.
//!
//! Tests the analytical Jacobian model (`log_prob_grad`) against:
//! 1. Known closed-form derivatives (conditional Gaussian).
//! 2. Central finite-difference approximation of `log_prob_batch`.
//!
//! Requires `--features neural`. Run:
//! ```sh
//! cargo test -p ns-unbinned --features neural --test flow_grad_integration
//! ```

#![cfg(feature = "neural")]

use std::path::PathBuf;

use approx::assert_relative_eq;
use ns_unbinned::FlowPdf;
use ns_unbinned::event_store::{EventStore, ObservableSpec};
use ns_unbinned::pdf::UnbinnedPdf;

/// Path to the conditional-Gaussian flow test fixtures.
fn grad_fixture_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/
    p.pop(); // repo root
    p.push("tests/fixtures/flow_grad_test");
    p
}

fn grad_manifest_path() -> PathBuf {
    grad_fixture_dir().join("flow_manifest.json")
}

fn has_grad_fixtures() -> bool {
    grad_manifest_path().exists()
}

/// Build a small EventStore for observable "x".
fn make_events(xs: &[f64]) -> EventStore {
    let obs = ObservableSpec::branch("x", (-10.0, 10.0));
    EventStore::from_columns(vec![obs], vec![("x".to_string(), xs.to_vec())], None).unwrap()
}

// ═══════════════════════════════════════════════════════════════════════════
// G3-T1: Analytical gradient matches closed-form for conditional Gaussian
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_flow_grad_matches_closed_form() {
    if !has_grad_fixtures() {
        eprintln!("SKIP: flow_grad_test fixtures not found");
        return;
    }

    // context_param_indices: context[0]=mu=params[0], context[1]=sigma=params[1]
    let flow = FlowPdf::from_manifest(&grad_manifest_path(), &[0, 1]).unwrap();
    assert_eq!(flow.n_params(), 2);
    assert!(flow.has_analytical_grad());

    let xs = vec![0.0, 1.0, -1.0, 2.0, -2.0, 0.5];
    let n = xs.len();
    let store = make_events(&xs);

    let mu = 1.0_f64;
    let sigma = 2.0_f64;
    let params = [mu, sigma];

    let mut logp = vec![0.0; n];
    let mut grad = vec![0.0; n * 2]; // [event0_dmu, event0_dsigma, event1_dmu, ...]
    flow.log_prob_grad_batch(&store, &params, &mut logp, &mut grad).unwrap();

    let ln2pi = (2.0 * std::f64::consts::PI).ln();

    for (i, &x) in xs.iter().enumerate() {
        let z = (x - mu) / sigma;
        let expected_logp = -0.5 * z * z - sigma.ln() - 0.5 * ln2pi;
        let expected_dmu = z / sigma;
        let expected_dsigma = (x - mu).powi(2) / sigma.powi(3) - 1.0 / sigma;

        assert_relative_eq!(logp[i], expected_logp, epsilon = 1e-4,);
        assert_relative_eq!(grad[i * 2], expected_dmu, epsilon = 1e-4,);
        assert_relative_eq!(grad[i * 2 + 1], expected_dsigma, epsilon = 1e-4,);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G3-T2: Analytical gradient matches finite-difference approximation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_flow_grad_matches_finite_diff() {
    if !has_grad_fixtures() {
        eprintln!("SKIP: flow_grad_test fixtures not found");
        return;
    }

    let flow = FlowPdf::from_manifest(&grad_manifest_path(), &[0, 1]).unwrap();

    let xs = vec![0.0, 1.5, -0.5, 3.0, -2.5];
    let n = xs.len();
    let store = make_events(&xs);

    let mu = 0.5_f64;
    let sigma = 1.5_f64;
    let params = [mu, sigma];

    // Analytical gradient.
    let mut logp = vec![0.0; n];
    let mut grad = vec![0.0; n * 2];
    flow.log_prob_grad_batch(&store, &params, &mut logp, &mut grad).unwrap();

    // Finite-difference gradient.
    let eps = 1e-5;
    for j in 0..2 {
        let mut params_plus = params;
        let mut params_minus = params;
        params_plus[j] += eps;
        params_minus[j] -= eps;

        let mut logp_plus = vec![0.0; n];
        let mut logp_minus = vec![0.0; n];
        flow.log_prob_batch(&store, &params_plus, &mut logp_plus).unwrap();
        flow.log_prob_batch(&store, &params_minus, &mut logp_minus).unwrap();

        for i in 0..n {
            let fd = (logp_plus[i] - logp_minus[i]) / (2.0 * eps);
            let analytical = grad[i * 2 + j];
            let err = (analytical - fd).abs();
            // f32 ONNX models accumulate precision loss; 1e-2 is expected for cubic terms.
            // The closed-form test (G3-T1) validates correctness at 1e-4.
            assert!(
                err < 1e-2,
                "Finite-diff mismatch: param={j}, event={i}, analytical={analytical:.8}, fd={fd:.8}, err={err:.2e}"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// G3-T3: log_prob output consistency between log_prob and log_prob_grad
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_flow_logprob_consistent_between_models() {
    if !has_grad_fixtures() {
        eprintln!("SKIP: flow_grad_test fixtures not found");
        return;
    }

    let flow = FlowPdf::from_manifest(&grad_manifest_path(), &[0, 1]).unwrap();

    let xs = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
    let n = xs.len();
    let store = make_events(&xs);
    let params = [0.0, 1.0]; // standard normal

    // From log_prob model.
    let mut logp_plain = vec![0.0; n];
    flow.log_prob_batch(&store, &params, &mut logp_plain).unwrap();

    // From log_prob_grad model (output 0).
    let mut logp_grad = vec![0.0; n];
    let mut grad = vec![0.0; n * 2];
    flow.log_prob_grad_batch(&store, &params, &mut logp_grad, &mut grad).unwrap();

    for i in 0..n {
        assert_relative_eq!(logp_plain[i], logp_grad[i], epsilon = 1e-5,);
    }
}
