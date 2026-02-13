//! Integration tests for DcrSurrogate — neural DCR vs analytical reference.
//!
//! These tests require:
//! - `--features neural` to compile
//! - Test ONNX fixtures under `tests/fixtures/dcr_test` (regenerate via `scripts/neural/generate_dcr_test_fixtures.py`)
//!
//! Run: `cargo test -p ns-unbinned --features neural --test dcr_integration`

#![cfg(feature = "neural")]

use std::path::PathBuf;

use approx::assert_relative_eq;
use ns_unbinned::event_store::{EventStore, ObservableSpec};
use ns_unbinned::normalize::{QuadratureGrid, QuadratureOrder, log_normalize_quadrature};
use ns_unbinned::pdf::UnbinnedPdf;
use ns_unbinned::{DcrSurrogate, GaussianPdf};

/// DCR test fixture: Gaussian with systematic shift δ=0.5 per unit α.
const DELTA: f64 = 0.5;

fn fixture_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/
    p.pop(); // repo root
    p.push("tests/fixtures/dcr_test");
    p
}

fn manifest_path() -> PathBuf {
    fixture_dir().join("flow_manifest.json")
}

fn has_fixtures() -> bool {
    manifest_path().exists()
}

fn make_store(xs: Vec<f64>) -> EventStore {
    let obs = ObservableSpec::branch("mass", (-6.0, 6.0));
    EventStore::from_columns(vec![obs], vec![("mass".to_string(), xs)], None).unwrap()
}

// ═══════════════════════════════════════════════════════════════════════════
// C3.1: DcrSurrogate loads from manifest
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_dcr_loads_from_manifest() {
    if !has_fixtures() {
        eprintln!("SKIP: DCR test fixtures not found under tests/fixtures/dcr_test");
        return;
    }

    let dcr = DcrSurrogate::from_manifest(
        &manifest_path(),
        &[0], // systematic param index 0 in global params
        vec!["alpha_syst".to_string()],
        "background".to_string(),
    )
    .unwrap();

    assert_eq!(dcr.n_params(), 1);
    assert_eq!(dcr.observables(), &["mass".to_string()]);
    assert_eq!(dcr.process_name(), "background");
    assert_eq!(dcr.systematic_names(), &["alpha_syst".to_string()]);
}

// ═══════════════════════════════════════════════════════════════════════════
// C3.2: Nominal (α=0) matches standard normal
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_dcr_nominal_matches_standard_normal() {
    if !has_fixtures() {
        eprintln!("SKIP: DCR test fixtures not found.");
        return;
    }

    let dcr = DcrSurrogate::from_manifest(
        &manifest_path(),
        &[0],
        vec!["alpha_syst".to_string()],
        "background".to_string(),
    )
    .unwrap();

    let xs = vec![0.0, 1.0, -1.0, 2.0, -2.0];
    let store = make_store(xs.clone());
    let mut out = vec![0.0; xs.len()];

    // α = 0 → standard normal
    dcr.log_prob_batch(&store, &[0.0], &mut out).unwrap();

    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    for (i, &x) in xs.iter().enumerate() {
        let expected = -0.5 * x * x - 0.5 * ln2pi;
        assert_relative_eq!(out[i], expected, epsilon = 1e-5);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// C3.3: Shifted (α≠0) matches shifted Gaussian
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_dcr_shifted_matches_gaussian() {
    if !has_fixtures() {
        eprintln!("SKIP: DCR test fixtures not found.");
        return;
    }

    let dcr = DcrSurrogate::from_manifest(
        &manifest_path(),
        &[0],
        vec!["alpha_syst".to_string()],
        "background".to_string(),
    )
    .unwrap();

    let gauss = GaussianPdf::new("mass");

    for alpha in [-2.0, -1.0, 0.5, 1.0, 2.0] {
        let mu = DELTA * alpha;
        let xs = vec![-1.0, 0.0, 1.0, 2.0];
        let store = make_store(xs.clone());

        let mut dcr_logp = vec![0.0; xs.len()];
        let mut gauss_logp = vec![0.0; xs.len()];

        dcr.log_prob_batch(&store, &[alpha], &mut dcr_logp).unwrap();
        gauss.log_prob_batch(&store, &[mu, 1.0], &mut gauss_logp).unwrap();

        for i in 0..xs.len() {
            assert_relative_eq!(dcr_logp[i], gauss_logp[i], epsilon = 1e-5,);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// C3.4: Normalization holds at various α values
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_dcr_normalization_at_various_alpha() {
    if !has_fixtures() {
        eprintln!("SKIP: DCR test fixtures not found.");
        return;
    }

    let dcr = DcrSurrogate::from_manifest(
        &manifest_path(),
        &[0],
        vec!["alpha_syst".to_string()],
        "background".to_string(),
    )
    .unwrap();

    let grid = QuadratureGrid::new_1d("mass", (-6.0, 6.0), QuadratureOrder::N128).unwrap();

    for alpha in [-2.0, -1.0, 0.0, 1.0, 2.0] {
        let log_int = log_normalize_quadrature(&dcr, &grid, &[alpha]).unwrap();
        let integral = log_int.exp();
        assert!(
            (integral - 1.0).abs() < 0.01,
            "Normalization at α={alpha}: integral={integral:.6}, expected ≈ 1.0"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// C3.5: Gradient w.r.t. systematic parameter (finite-difference check)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_dcr_gradient_wrt_systematic() {
    if !has_fixtures() {
        eprintln!("SKIP: DCR test fixtures not found.");
        return;
    }

    let dcr = DcrSurrogate::from_manifest(
        &manifest_path(),
        &[0],
        vec!["alpha_syst".to_string()],
        "background".to_string(),
    )
    .unwrap();

    let xs = vec![0.0, 1.0, -1.0];
    let store = make_store(xs.clone());
    let n = xs.len();
    let alpha = 1.0;

    let mut logp = vec![0.0; n];
    let mut grad = vec![0.0; n]; // n_params=1 → grad has n entries
    dcr.log_prob_grad_batch(&store, &[alpha], &mut logp, &mut grad).unwrap();

    // Analytical gradient for Gaussian: d/dα log N(x; δα, 1) = δ(x - δα)
    for (i, &x) in xs.iter().enumerate() {
        let mu = DELTA * alpha;
        let expected_grad = DELTA * (x - mu);
        assert_relative_eq!(grad[i], expected_grad, epsilon = 0.05);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// C3.6: Sampling via DCR surrogate
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_dcr_sample_shifted() {
    if !has_fixtures() {
        eprintln!("SKIP: DCR test fixtures not found.");
        return;
    }

    let dcr = DcrSurrogate::from_manifest(
        &manifest_path(),
        &[0],
        vec!["alpha_syst".to_string()],
        "background".to_string(),
    )
    .unwrap();

    let alpha = 2.0;
    let expected_mu = DELTA * alpha; // 1.0

    let mut rng = rand::rng();
    let sampled = dcr.sample(&[alpha], 2000, &[(-6.0, 6.0)], &mut rng).unwrap();

    assert_eq!(sampled.n_events(), 2000);
    let col = sampled.column("mass").unwrap();

    let mean: f64 = col.iter().sum::<f64>() / col.len() as f64;
    let var: f64 = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / col.len() as f64;

    assert!(
        (mean - expected_mu).abs() < 0.15,
        "Sample mean at α={alpha} should be ≈ {expected_mu}, got {mean}"
    );
    assert!((var.sqrt() - 1.0).abs() < 0.15, "Sample std should be ≈ 1, got {}", var.sqrt());
}

// ═══════════════════════════════════════════════════════════════════════════
// C3.7: NLL surface — DCR vs Gaussian at various α
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_dcr_nll_surface_matches_gaussian() {
    if !has_fixtures() {
        eprintln!("SKIP: DCR test fixtures not found.");
        return;
    }

    let dcr = DcrSurrogate::from_manifest(
        &manifest_path(),
        &[0],
        vec!["alpha_syst".to_string()],
        "background".to_string(),
    )
    .unwrap();

    let gauss = GaussianPdf::new("mass");

    // Observed data (fixed)
    let xs = vec![-1.2, -0.5, 0.1, 0.7, 1.3, -0.8, 0.4, 1.0, -0.3, 0.6];
    let n = xs.len();
    let store = make_store(xs);

    // Scan α from -2 to 2 and compare NLL surfaces
    for alpha_x10 in -20..=20 {
        let alpha = alpha_x10 as f64 * 0.1;
        let mu = DELTA * alpha;

        let mut dcr_logp = vec![0.0; n];
        let mut gauss_logp = vec![0.0; n];

        dcr.log_prob_batch(&store, &[alpha], &mut dcr_logp).unwrap();
        gauss.log_prob_batch(&store, &[mu, 1.0], &mut gauss_logp).unwrap();

        let dcr_nll: f64 = dcr_logp.iter().map(|lp| -lp).sum();
        let gauss_nll: f64 = gauss_logp.iter().map(|lp| -lp).sum();

        assert!(
            (dcr_nll - gauss_nll).abs() < 0.1,
            "NLL mismatch at α={alpha:.1}: DCR={dcr_nll:.4}, Gauss={gauss_nll:.4}"
        );
    }
}
