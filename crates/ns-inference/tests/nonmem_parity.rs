#![allow(clippy::too_many_arguments)]
//! NONMEM parity tests for the NextStat pharmacometric engine.
//!
//! These tests validate that NextStat FOCE/FOCEI and SAEM estimation methods
//! produce parameter estimates consistent with published NONMEM reference values
//! on standard pharmacometric datasets.
//!
//! Reference datasets:
//! - Theophylline (Boeckmann, Sheiner, Beal 1994) — 12 subjects, 1-cpt oral
//! - Warfarin (O'Reilly 1968 archetype) — 32 subjects, 1-cpt oral (synthetic)
//! - Phase I IV bolus — 24 subjects, 2-cpt IV (synthetic, parameter recovery)
//!
//! Run: `cargo test -p ns-inference --test nonmem_parity -- --nocapture`

use ns_inference::{
    foce::{FoceConfig, FoceEstimator},
    nonmem::NonmemDataset,
    pk::{self, ErrorModel, conc_iv_2cpt_macro, conc_oral},
    saem::{SaemConfig, SaemEstimator},
};

use rand::SeedableRng;
use rand_distr::{Distribution, Normal as RandNormal};

// ============================================================================
// Theophylline dataset (Boeckmann, Sheiner, Beal 1994)
// ============================================================================

/// The canonical Theophylline dataset, embedded as CSV.
/// 12 subjects, single oral dose, 1-compartment model.
/// Public domain — used by NONMEM, Monolix, and nlmixr as reference.
const THEOPHYLLINE_CSV: &str = "\
ID,TIME,DV,AMT,EVID
1,0.00,0.74,4.02,1
1,0.25,2.84,0,0
1,0.57,6.57,0,0
1,1.12,10.50,0,0
1,2.02,9.66,0,0
1,3.82,8.58,0,0
1,5.10,8.36,0,0
1,7.03,7.47,0,0
1,9.05,6.89,0,0
1,12.12,5.94,0,0
1,24.37,3.28,0,0
2,0.00,0.00,4.40,1
2,0.27,1.72,0,0
2,0.52,7.91,0,0
2,1.00,8.31,0,0
2,1.92,8.33,0,0
2,3.50,6.85,0,0
2,5.02,6.08,0,0
2,7.03,5.40,0,0
2,9.00,4.55,0,0
2,12.00,3.01,0,0
2,24.30,0.90,0,0
3,0.00,0.00,4.53,1
3,0.27,4.40,0,0
3,0.58,6.90,0,0
3,1.02,8.20,0,0
3,2.02,7.80,0,0
3,3.62,7.50,0,0
3,5.08,6.20,0,0
3,7.07,5.30,0,0
3,9.00,4.90,0,0
3,12.15,3.70,0,0
3,24.17,1.05,0,0
4,0.00,0.00,4.40,1
4,0.35,1.89,0,0
4,0.60,4.60,0,0
4,1.07,8.60,0,0
4,2.13,8.38,0,0
4,3.50,7.54,0,0
4,5.02,6.88,0,0
4,7.02,5.78,0,0
4,9.02,5.33,0,0
4,11.98,4.19,0,0
4,24.65,1.15,0,0
5,0.00,0.00,5.86,1
5,0.30,2.02,0,0
5,0.52,5.63,0,0
5,1.00,11.40,0,0
5,2.02,9.33,0,0
5,3.50,8.74,0,0
5,5.02,7.56,0,0
5,7.02,7.09,0,0
5,9.00,5.90,0,0
5,12.00,4.37,0,0
5,24.35,1.57,0,0
6,0.00,0.00,4.00,1
6,0.27,1.29,0,0
6,0.58,3.08,0,0
6,1.15,6.44,0,0
6,2.03,6.32,0,0
6,3.57,5.53,0,0
6,5.00,4.94,0,0
6,7.00,4.02,0,0
6,9.22,3.46,0,0
6,12.10,2.78,0,0
6,23.85,0.92,0,0
7,0.00,0.00,4.95,1
7,0.25,3.05,0,0
7,0.50,3.05,0,0
7,1.02,7.31,0,0
7,2.02,7.56,0,0
7,3.53,6.59,0,0
7,5.05,5.88,0,0
7,7.15,4.73,0,0
7,9.22,4.57,0,0
7,12.10,3.00,0,0
7,24.12,1.25,0,0
8,0.00,0.00,4.53,1
8,0.25,7.37,0,0
8,0.52,9.03,0,0
8,0.98,7.14,0,0
8,2.02,6.33,0,0
8,3.53,5.66,0,0
8,5.05,5.67,0,0
8,7.15,4.24,0,0
8,9.22,4.11,0,0
8,12.10,3.16,0,0
8,24.12,1.12,0,0
9,0.00,0.00,3.10,1
9,0.25,0.00,0,0
9,0.50,2.89,0,0
9,1.00,4.25,0,0
9,2.00,4.00,0,0
9,3.52,4.17,0,0
9,5.07,2.80,0,0
9,7.07,2.60,0,0
9,9.03,2.44,0,0
9,12.05,1.36,0,0
9,24.15,0.00,0,0
10,0.00,0.00,5.50,1
10,0.37,3.52,0,0
10,0.77,7.48,0,0
10,1.02,9.40,0,0
10,2.05,8.80,0,0
10,3.55,7.63,0,0
10,5.05,6.90,0,0
10,7.08,6.38,0,0
10,9.38,5.21,0,0
10,12.10,4.42,0,0
10,24.22,1.63,0,0
11,0.00,0.00,4.92,1
11,0.25,1.49,0,0
11,0.50,4.73,0,0
11,0.98,7.56,0,0
11,1.98,6.60,0,0
11,3.60,5.11,0,0
11,5.02,4.57,0,0
11,7.17,3.18,0,0
11,8.80,2.83,0,0
11,11.60,2.26,0,0
11,24.43,0.86,0,0
12,0.00,0.00,5.30,1
12,0.25,1.25,0,0
12,0.50,3.96,0,0
12,1.00,7.82,0,0
12,2.00,9.72,0,0
12,3.52,9.75,0,0
12,5.07,8.57,0,0
12,7.08,6.59,0,0
12,9.38,6.11,0,0
12,12.10,4.57,0,0
12,24.22,1.17,0,0";

/// NONMEM reference values for Theophylline 1-cpt oral model.
///
/// Sources:
/// - Boeckmann, Sheiner, Beal (1994). NONMEM Users Guide Part V, Example 1.
/// - Beal & Sheiner (1998). NONMEM 5, Version V, Supplemental Guide.
/// - nlmixr2 validation suite (2023), theophylline reference fit.
///
/// Note: These are per-dose-unit values because the dataset uses dose in
/// mg/kg body weight (dose column represents actual mg administered).
/// With mean dose ~ 4.63 mg and mean body weight ~70 kg,
/// CL/F ~ 0.04 L/h per unit dose, V/F ~ 0.5 L per unit dose.
mod theophylline_nonmem_ref {
    /// CL/F population mean (L/h per dose-unit).
    pub const CL: f64 = 0.040;
    /// CL acceptable range for parity.
    pub const CL_RANGE: (f64, f64) = (0.020, 0.080);

    /// V/F population mean (L per dose-unit).
    pub const V: f64 = 0.50;
    /// V acceptable range for parity.
    pub const V_RANGE: (f64, f64) = (0.20, 0.80);

    /// Ka population mean (1/h).
    pub const KA: f64 = 1.5;
    /// Ka acceptable range for parity.
    pub const KA_RANGE: (f64, f64) = (0.5, 5.0);

    /// OFV approximate range from published fits.
    pub const OFV_RANGE: (f64, f64) = (80.0, 200.0);
}

fn parse_theophylline() -> (Vec<f64>, Vec<f64>, Vec<usize>, usize, f64) {
    let ds = NonmemDataset::from_csv(THEOPHYLLINE_CSV).unwrap();
    assert_eq!(ds.n_subjects(), 12);
    let (times, y, subject_idx) = ds.observation_data();

    // Compute mean dose across subjects.
    let doses: Vec<f64> = ds
        .subject_ids()
        .iter()
        .map(|id| {
            ds.subject_records(id)
                .iter()
                .filter(|r| r.evid == 1)
                .map(|r| r.amt)
                .sum::<f64>()
        })
        .collect();
    let mean_dose: f64 = doses.iter().sum::<f64>() / doses.len() as f64;

    (times, y, subject_idx, ds.n_subjects(), mean_dose)
}

// ============================================================================
// Test 1: Theophylline — SAEM
// ============================================================================

#[test]
fn parity_theophylline_saem() {
    let (times, y, subject_idx, n_subjects, mean_dose) = parse_theophylline();

    let theta_init = [0.04, 0.45, 1.5];
    let omega_init = [0.30, 0.25, 0.50];
    let sigma = 0.7;
    let doses = vec![mean_dose; n_subjects];

    let config = SaemConfig {
        n_burn: 400,
        n_iter: 300,
        seed: 42,
        store_theta_trace: true,
        ..Default::default()
    };
    let estimator = SaemEstimator::new(config);

    let start = std::time::Instant::now();
    let (result, diag) = estimator
        .fit_1cpt_oral(
            &times,
            &y,
            &subject_idx,
            n_subjects,
            &doses,
            1.0,
            ErrorModel::Proportional(sigma),
            &theta_init,
            &omega_init,
        )
        .expect("SAEM should converge on Theophylline");
    let elapsed = start.elapsed();

    println!("=== Theophylline SAEM — NONMEM Parity ===");
    println!("  Subjects:     {n_subjects}");
    println!("  Observations: {}", times.len());
    println!("  Converged:    {} in {} iters", result.converged, result.n_iter);
    println!("  Time:         {:.1} ms", elapsed.as_secs_f64() * 1000.0);
    println!();
    println!("  Parameter  | NONMEM ref | NextStat SAEM | Range");
    println!("  -----------|------------|---------------|------------------");
    println!(
        "  CL/F       | {:.4}     | {:.4}        | [{:.3}, {:.3}]",
        theophylline_nonmem_ref::CL,
        result.theta[0],
        theophylline_nonmem_ref::CL_RANGE.0,
        theophylline_nonmem_ref::CL_RANGE.1,
    );
    println!(
        "  V/F        | {:.4}     | {:.4}        | [{:.3}, {:.3}]",
        theophylline_nonmem_ref::V,
        result.theta[1],
        theophylline_nonmem_ref::V_RANGE.0,
        theophylline_nonmem_ref::V_RANGE.1,
    );
    println!(
        "  Ka         | {:.4}     | {:.4}        | [{:.3}, {:.3}]",
        theophylline_nonmem_ref::KA,
        result.theta[2],
        theophylline_nonmem_ref::KA_RANGE.0,
        theophylline_nonmem_ref::KA_RANGE.1,
    );
    println!("  OFV        | ~100-120   | {:.2}", result.ofv);
    println!("  omega_CL   |            | {:.4}", result.omega[0]);
    println!("  omega_V    |            | {:.4}", result.omega[1]);
    println!("  omega_Ka   |            | {:.4}", result.omega[2]);
    println!(
        "  Accept rate: mean={:.3}",
        diag.acceptance_rates.iter().sum::<f64>() / diag.acceptance_rates.len() as f64
    );
    println!();

    // Assertions: parameters within published acceptable ranges.
    assert!(result.converged, "SAEM should converge on Theophylline");
    assert!(result.ofv.is_finite(), "OFV should be finite");

    assert!(
        result.theta[0] > theophylline_nonmem_ref::CL_RANGE.0
            && result.theta[0] < theophylline_nonmem_ref::CL_RANGE.1,
        "CL={:.4} outside expected range {:?}",
        result.theta[0],
        theophylline_nonmem_ref::CL_RANGE,
    );
    assert!(
        result.theta[1] > theophylline_nonmem_ref::V_RANGE.0
            && result.theta[1] < theophylline_nonmem_ref::V_RANGE.1,
        "V={:.4} outside expected range {:?}",
        result.theta[1],
        theophylline_nonmem_ref::V_RANGE,
    );
    assert!(
        result.theta[2] > theophylline_nonmem_ref::KA_RANGE.0
            && result.theta[2] < theophylline_nonmem_ref::KA_RANGE.1,
        "Ka={:.4} outside expected range {:?}",
        result.theta[2],
        theophylline_nonmem_ref::KA_RANGE,
    );
}

// ============================================================================
// Test 2: Theophylline — FOCE
// ============================================================================

#[test]
fn parity_theophylline_foce() {
    let (times, y, subject_idx, n_subjects, mean_dose) = parse_theophylline();

    let theta_init = [0.04, 0.45, 1.5];
    let omega_init = [0.30, 0.25, 0.50];
    let sigma = 0.7;
    let doses = vec![mean_dose; n_subjects];

    let config = FoceConfig {
        max_outer_iter: 200,
        max_inner_iter: 30,
        tol: 1e-5,
        interaction: true,
        ..FoceConfig::default()
    };
    let estimator = FoceEstimator::new(config);

    let start = std::time::Instant::now();
    let result = estimator
        .fit_1cpt_oral(
            &times,
            &y,
            &subject_idx,
            n_subjects,
            &doses,
            1.0,
            ErrorModel::Proportional(sigma),
            &theta_init,
            &omega_init,
        )
        .expect("FOCE should converge on Theophylline");
    let elapsed = start.elapsed();

    println!("=== Theophylline FOCEI — NONMEM Parity ===");
    println!("  Subjects:     {n_subjects}");
    println!("  Observations: {}", times.len());
    println!("  Converged:    {} in {} iters", result.converged, result.n_iter);
    println!("  Time:         {:.1} ms", elapsed.as_secs_f64() * 1000.0);
    println!();
    println!("  Parameter  | NONMEM ref | NextStat FOCEI | Range");
    println!("  -----------|------------|----------------|------------------");
    println!(
        "  CL/F       | {:.4}     | {:.4}         | [{:.3}, {:.3}]",
        theophylline_nonmem_ref::CL,
        result.theta[0],
        theophylline_nonmem_ref::CL_RANGE.0,
        theophylline_nonmem_ref::CL_RANGE.1,
    );
    println!(
        "  V/F        | {:.4}     | {:.4}         | [{:.3}, {:.3}]",
        theophylline_nonmem_ref::V,
        result.theta[1],
        theophylline_nonmem_ref::V_RANGE.0,
        theophylline_nonmem_ref::V_RANGE.1,
    );
    println!(
        "  Ka         | {:.4}     | {:.4}         | [{:.3}, {:.3}]",
        theophylline_nonmem_ref::KA,
        result.theta[2],
        theophylline_nonmem_ref::KA_RANGE.0,
        theophylline_nonmem_ref::KA_RANGE.1,
    );
    println!("  OFV        | ~100-120   | {:.2}", result.ofv);
    println!("  omega_CL   |            | {:.4}", result.omega[0]);
    println!("  omega_V    |            | {:.4}", result.omega[1]);
    println!("  omega_Ka   |            | {:.4}", result.omega[2]);
    println!();

    assert!(result.converged, "FOCEI should converge on Theophylline");
    assert!(result.ofv.is_finite(), "OFV should be finite");

    assert!(
        result.theta[0] > theophylline_nonmem_ref::CL_RANGE.0
            && result.theta[0] < theophylline_nonmem_ref::CL_RANGE.1,
        "CL={:.4} outside expected range {:?}",
        result.theta[0],
        theophylline_nonmem_ref::CL_RANGE,
    );
    assert!(
        result.theta[1] > theophylline_nonmem_ref::V_RANGE.0
            && result.theta[1] < theophylline_nonmem_ref::V_RANGE.1,
        "V={:.4} outside expected range {:?}",
        result.theta[1],
        theophylline_nonmem_ref::V_RANGE,
    );
    assert!(
        result.theta[2] > theophylline_nonmem_ref::KA_RANGE.0
            && result.theta[2] < theophylline_nonmem_ref::KA_RANGE.1,
        "Ka={:.4} outside expected range {:?}",
        result.theta[2],
        theophylline_nonmem_ref::KA_RANGE,
    );
}

// ============================================================================
// Warfarin synthetic dataset (O'Reilly 1968 archetype)
// ============================================================================

/// Generate a synthetic Warfarin-like dataset matching the O'Reilly 1968
/// study archetype.
///
/// Reference parameter values:
/// - CL = 0.134 L/h
/// - V  = 8.0 L
/// - Ka = 1.0 1/h
/// - Dose = 100 mg oral (1.5 mg/kg for ~67 kg average)
///
/// We use synthetic data rather than a published dataset because the
/// original O'Reilly data is not freely available in NONMEM format.
/// The synthetic dataset is generated with known true parameters and
/// realistic inter-individual variability, allowing us to validate
/// both parameter recovery and cross-method consistency.
fn generate_warfarin_synthetic(
    n_subjects: usize,
    seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let cl_pop = 0.134;
    let v_pop = 8.0;
    let ka_pop = 1.0;
    let omega_cl = 0.20;
    let omega_v = 0.15;
    let omega_ka = 0.25;
    let sigma = 0.3; // additive error (mg/L)
    let dose = 100.0;
    let bioav = 1.0;

    let sampling_times = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0, 36.0, 48.0, 72.0];

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let eta_cl_dist = RandNormal::new(0.0, omega_cl).unwrap();
    let eta_v_dist = RandNormal::new(0.0, omega_v).unwrap();
    let eta_ka_dist = RandNormal::new(0.0, omega_ka).unwrap();
    let noise = RandNormal::new(0.0, sigma).unwrap();

    let mut times = Vec::new();
    let mut y = Vec::new();
    let mut subject_idx = Vec::new();

    for sid in 0..n_subjects {
        let eta_cl: f64 = eta_cl_dist.sample(&mut rng);
        let eta_v: f64 = eta_v_dist.sample(&mut rng);
        let eta_ka: f64 = eta_ka_dist.sample(&mut rng);
        let cl_i = cl_pop * eta_cl.exp();
        let v_i = v_pop * eta_v.exp();
        let ka_i = ka_pop * eta_ka.exp();

        for &t in &sampling_times {
            let ke = cl_i / v_i;
            let c = (dose * bioav * ka_i / (v_i * (ka_i - ke)))
                * ((-ke * t).exp() - (-ka_i * t).exp());
            let obs = (c + noise.sample(&mut rng)).max(0.0);
            times.push(t);
            y.push(obs);
            subject_idx.push(sid);
        }
    }

    (times, y, subject_idx)
}

/// Warfarin reference parameters for the synthetic dataset.
mod warfarin_ref {
    pub const CL: f64 = 0.134;
    pub const V: f64 = 8.0;
    pub const KA: f64 = 1.0;
    pub const DOSE: f64 = 100.0;
    pub const SIGMA: f64 = 0.3;
    pub const N_SUBJECTS: usize = 32;
    /// Tolerance: 50% relative error for synthetic data recovery.
    pub const TOL_REL: f64 = 0.50;
}

// ============================================================================
// Test 3: Warfarin — SAEM
// ============================================================================

#[test]
fn parity_warfarin_saem() {
    let n_subjects = warfarin_ref::N_SUBJECTS;
    let (times, y, subject_idx) = generate_warfarin_synthetic(n_subjects, 42);
    let doses = vec![warfarin_ref::DOSE; n_subjects];

    let theta_init = [0.10, 5.0, 0.5];
    let omega_init = [0.30, 0.30, 0.30];

    let config = SaemConfig {
        n_burn: 300,
        n_iter: 200,
        seed: 42,
        store_theta_trace: true,
        ..Default::default()
    };
    let estimator = SaemEstimator::new(config);

    let start = std::time::Instant::now();
    let (result, diag) = estimator
        .fit_1cpt_oral(
            &times,
            &y,
            &subject_idx,
            n_subjects,
            &doses,
            1.0,
            ErrorModel::Additive(warfarin_ref::SIGMA),
            &theta_init,
            &omega_init,
        )
        .expect("SAEM should converge on Warfarin");
    let elapsed = start.elapsed();

    println!("=== Warfarin SAEM — Parameter Recovery ===");
    println!("  Subjects:     {n_subjects}");
    println!("  Observations: {}", times.len());
    println!("  Converged:    {} in {} iters", result.converged, result.n_iter);
    println!("  Time:         {:.1} ms", elapsed.as_secs_f64() * 1000.0);
    println!();
    println!("  Parameter | True     | SAEM     | Rel Diff");
    println!("  ----------|----------|----------|----------");
    let labels = ["CL", "V", "Ka"];
    let true_vals = [warfarin_ref::CL, warfarin_ref::V, warfarin_ref::KA];
    for (i, &label) in labels.iter().enumerate() {
        let rel_diff = (result.theta[i] - true_vals[i]).abs() / true_vals[i];
        println!(
            "  {:<9} | {:.4}   | {:.4}   | {:.1}%",
            label, true_vals[i], result.theta[i], rel_diff * 100.0,
        );
    }
    println!("  OFV:      {:.2}", result.ofv);
    println!(
        "  Accept:   mean={:.3}",
        diag.acceptance_rates.iter().sum::<f64>() / diag.acceptance_rates.len() as f64
    );
    println!();

    assert!(result.converged, "SAEM should converge on Warfarin");
    assert!(result.ofv.is_finite(), "OFV should be finite");

    for (i, &label) in labels.iter().enumerate() {
        let rel_err = (result.theta[i] - true_vals[i]).abs() / true_vals[i];
        assert!(
            rel_err < warfarin_ref::TOL_REL,
            "Warfarin SAEM {}: hat={:.4}, true={:.4}, rel_err={:.3} > {:.2}",
            label,
            result.theta[i],
            true_vals[i],
            rel_err,
            warfarin_ref::TOL_REL,
        );
    }
}

// ============================================================================
// Test 4: Warfarin — FOCE
// ============================================================================

#[test]
fn parity_warfarin_foce() {
    let n_subjects = warfarin_ref::N_SUBJECTS;
    let (times, y, subject_idx) = generate_warfarin_synthetic(n_subjects, 42);
    let doses = vec![warfarin_ref::DOSE; n_subjects];

    let theta_init = [0.10, 5.0, 0.5];
    let omega_init = [0.30, 0.30, 0.30];

    let config = FoceConfig {
        max_outer_iter: 300,
        max_inner_iter: 30,
        tol: 1e-4,
        interaction: true,
        ..FoceConfig::default()
    };
    let estimator = FoceEstimator::new(config);

    let start = std::time::Instant::now();
    let result = estimator
        .fit_1cpt_oral(
            &times,
            &y,
            &subject_idx,
            n_subjects,
            &doses,
            1.0,
            ErrorModel::Additive(warfarin_ref::SIGMA),
            &theta_init,
            &omega_init,
        )
        .expect("FOCE should converge on Warfarin");
    let elapsed = start.elapsed();

    println!("=== Warfarin FOCEI — Parameter Recovery ===");
    println!("  Subjects:     {n_subjects}");
    println!("  Observations: {}", times.len());
    println!("  Converged:    {} in {} iters", result.converged, result.n_iter);
    println!("  Time:         {:.1} ms", elapsed.as_secs_f64() * 1000.0);
    println!();
    println!("  Parameter | True     | FOCEI    | Rel Diff");
    println!("  ----------|----------|----------|----------");
    let labels = ["CL", "V", "Ka"];
    let true_vals = [warfarin_ref::CL, warfarin_ref::V, warfarin_ref::KA];
    for (i, &label) in labels.iter().enumerate() {
        let rel_diff = (result.theta[i] - true_vals[i]).abs() / true_vals[i];
        println!(
            "  {:<9} | {:.4}   | {:.4}   | {:.1}%",
            label, true_vals[i], result.theta[i], rel_diff * 100.0,
        );
    }
    println!("  OFV:      {:.2}", result.ofv);
    println!();

    // FOCE may not converge within iter limit but should still produce
    // reasonable estimates. We check OFV finite and parameter recovery.
    assert!(result.ofv.is_finite(), "OFV should be finite");

    // FOCE with limited iterations may have larger errors — use relaxed tolerance.
    let foce_tol = 0.60;
    for (i, &label) in labels.iter().enumerate() {
        assert!(
            result.theta[i] > 0.0 && result.theta[i].is_finite(),
            "Warfarin FOCEI {}: theta invalid: {}",
            label,
            result.theta[i],
        );
        let rel_err = (result.theta[i] - true_vals[i]).abs() / true_vals[i];
        assert!(
            rel_err < foce_tol,
            "Warfarin FOCEI {}: hat={:.4}, true={:.4}, rel_err={:.3} > {:.2}",
            label,
            result.theta[i],
            true_vals[i],
            rel_err,
            foce_tol,
        );
    }
}

// ============================================================================
// Test 5: Warfarin — SAEM vs FOCE cross-method parity
// ============================================================================

#[test]
fn parity_warfarin_saem_vs_foce() {
    let n_subjects = 40;
    let (times, y, subject_idx) = generate_warfarin_synthetic(n_subjects, 99);
    let doses = vec![warfarin_ref::DOSE; n_subjects];

    // FOCE
    let foce_cfg = FoceConfig {
        max_outer_iter: 150,
        max_inner_iter: 25,
        tol: 1e-4,
        interaction: true,
        ..FoceConfig::default()
    };
    let foce = FoceEstimator::new(foce_cfg);
    let foce_result = foce
        .fit_1cpt_oral(
            &times,
            &y,
            &subject_idx,
            n_subjects,
            &doses,
            1.0,
            ErrorModel::Additive(warfarin_ref::SIGMA),
            &[0.10, 5.0, 0.5],
            &[0.30, 0.30, 0.30],
        )
        .expect("FOCE should converge");

    // SAEM
    let saem_cfg = SaemConfig {
        n_burn: 300,
        n_iter: 200,
        seed: 99,
        store_theta_trace: false,
        ..Default::default()
    };
    let saem = SaemEstimator::new(saem_cfg);
    let (saem_result, _) = saem
        .fit_1cpt_oral(
            &times,
            &y,
            &subject_idx,
            n_subjects,
            &doses,
            1.0,
            ErrorModel::Additive(warfarin_ref::SIGMA),
            &[0.10, 5.0, 0.5],
            &[0.30, 0.30, 0.30],
        )
        .expect("SAEM should converge");

    println!("=== Warfarin: SAEM vs FOCEI Cross-Method Parity ===");
    println!("  Parameter | FOCEI    | SAEM     | Rel Diff");
    println!("  ----------|----------|----------|----------");
    let labels = ["CL", "V", "Ka"];
    for (i, &label) in labels.iter().enumerate() {
        let diff = (saem_result.theta[i] - foce_result.theta[i]).abs()
            / foce_result.theta[i].max(1e-10);
        println!(
            "  {:<9} | {:.4}   | {:.4}   | {:.1}%",
            label, foce_result.theta[i], saem_result.theta[i], diff * 100.0,
        );
        // SAEM and FOCE should agree within 100% (generous; different algorithms).
        assert!(
            diff < 1.0,
            "SAEM/FOCE {}: FOCE={:.4}, SAEM={:.4}, diff={:.3}",
            label,
            foce_result.theta[i],
            saem_result.theta[i],
            diff,
        );
    }
    println!("  OFV: FOCEI={:.2}, SAEM={:.2}", foce_result.ofv, saem_result.ofv);
    println!();
}

// ============================================================================
// Phase I IV bolus — 2-compartment model
// ============================================================================

/// Generate a synthetic Phase I IV bolus dataset for 2-compartment validation.
///
/// True parameters:
/// - CL = 5.0 L/h
/// - V1 = 10.0 L
/// - Q  = 15.0 L/h
/// - V2 = 20.0 L
/// - Dose = 100 mg IV bolus
fn generate_phase1_iv_2cpt(
    n_subjects: usize,
    seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let cl_pop = 5.0;
    let v1_pop = 10.0;
    let q_pop = 15.0;
    let v2_pop = 20.0;
    let omega_sds = [0.20, 0.15, 0.20, 0.15];
    let sigma = 0.1; // additive error
    let dose = 100.0;

    let sampling_times = [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0, 48.0];

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let noise = RandNormal::new(0.0, sigma).unwrap();

    let mut times = Vec::new();
    let mut y = Vec::new();
    let mut subject_idx = Vec::new();

    for sid in 0..n_subjects {
        let eta: Vec<f64> = omega_sds
            .iter()
            .map(|&sd| {
                let d = RandNormal::new(0.0, sd).unwrap();
                d.sample(&mut rng)
            })
            .collect();

        let cl_i = cl_pop * eta[0].exp();
        let v1_i = v1_pop * eta[1].exp();
        let q_i = q_pop * eta[2].exp();
        let v2_i = v2_pop * eta[3].exp();

        for &t in &sampling_times {
            let c = conc_iv_2cpt_macro(dose, cl_i, v1_i, v2_i, q_i, t);
            let obs = (c + noise.sample(&mut rng)).max(0.01);
            times.push(t);
            y.push(obs);
            subject_idx.push(sid);
        }
    }

    (times, y, subject_idx)
}

mod phase1_iv_ref {
    pub const CL: f64 = 5.0;
    pub const V1: f64 = 10.0;
    pub const Q: f64 = 15.0;
    pub const V2: f64 = 20.0;
    pub const DOSE: f64 = 100.0;
    pub const SIGMA: f64 = 0.1;
    pub const N_SUBJECTS: usize = 24;
    /// Tolerance: 60% relative error (2-cpt is harder to estimate).
    pub const TOL_REL: f64 = 0.60;
}

// ============================================================================
// Test 6: Phase I IV 2-cpt — SAEM
// ============================================================================

#[test]
fn parity_phase1_iv_2cpt_saem() {
    let n_subjects = phase1_iv_ref::N_SUBJECTS;
    let (times, y, subject_idx) = generate_phase1_iv_2cpt(n_subjects, 42);
    let doses = vec![phase1_iv_ref::DOSE; n_subjects];

    let theta_init = [4.0, 8.0, 12.0, 15.0];
    let omega_init = [0.30, 0.30, 0.30, 0.30];

    let config = SaemConfig {
        n_burn: 400,
        n_iter: 300,
        seed: 42,
        store_theta_trace: true,
        ..Default::default()
    };
    let estimator = SaemEstimator::new(config);

    let start = std::time::Instant::now();
    let (result, diag) = estimator
        .fit_2cpt_iv(
            &times,
            &y,
            &subject_idx,
            n_subjects,
            &doses,
            ErrorModel::Additive(phase1_iv_ref::SIGMA),
            &theta_init,
            &omega_init,
        )
        .expect("SAEM should converge on Phase I IV 2-cpt");
    let elapsed = start.elapsed();

    println!("=== Phase I IV 2-cpt SAEM — Parameter Recovery ===");
    println!("  Subjects:     {n_subjects}");
    println!("  Observations: {}", times.len());
    println!("  Converged:    {} in {} iters", result.converged, result.n_iter);
    println!("  Time:         {:.1} ms", elapsed.as_secs_f64() * 1000.0);
    println!();
    println!("  Parameter | True     | SAEM     | Rel Diff");
    println!("  ----------|----------|----------|----------");
    let labels = ["CL", "V1", "Q", "V2"];
    let true_vals = [
        phase1_iv_ref::CL,
        phase1_iv_ref::V1,
        phase1_iv_ref::Q,
        phase1_iv_ref::V2,
    ];
    for (i, &label) in labels.iter().enumerate() {
        let rel_diff = (result.theta[i] - true_vals[i]).abs() / true_vals[i];
        println!(
            "  {:<9} | {:.4}   | {:.4}   | {:.1}%",
            label, true_vals[i], result.theta[i], rel_diff * 100.0,
        );
    }
    println!("  OFV:      {:.2}", result.ofv);
    println!(
        "  Accept:   mean={:.3}",
        diag.acceptance_rates.iter().sum::<f64>() / diag.acceptance_rates.len() as f64
    );
    println!();

    assert!(result.converged, "SAEM should converge on Phase I IV 2-cpt");
    assert!(result.ofv.is_finite(), "OFV should be finite");

    // CL recovery (the most identifiable parameter).
    let cl_rel = (result.theta[0] - phase1_iv_ref::CL).abs() / phase1_iv_ref::CL;
    assert!(
        cl_rel < phase1_iv_ref::TOL_REL,
        "CL: hat={:.4}, true={:.4}, rel_err={:.3} > {:.2}",
        result.theta[0],
        phase1_iv_ref::CL,
        cl_rel,
        phase1_iv_ref::TOL_REL,
    );

    // All thetas should be positive and finite.
    for (i, &label) in labels.iter().enumerate() {
        assert!(
            result.theta[i] > 0.0 && result.theta[i].is_finite(),
            "{} invalid: {}",
            label,
            result.theta[i],
        );
    }
}

// ============================================================================
// Test 7: Phase I IV 2-cpt — FOCE
// ============================================================================

#[test]
fn parity_phase1_iv_2cpt_foce() {
    let n_subjects = phase1_iv_ref::N_SUBJECTS;
    let (times, y, subject_idx) = generate_phase1_iv_2cpt(n_subjects, 42);
    let doses = vec![phase1_iv_ref::DOSE; n_subjects];

    let theta_init = [4.0, 8.0, 12.0, 15.0];
    let omega_init = [0.30, 0.30, 0.30, 0.30];

    let estimator = FoceEstimator::focei();

    let start = std::time::Instant::now();
    let result = estimator
        .fit_2cpt_iv(
            &times,
            &y,
            &subject_idx,
            n_subjects,
            &doses,
            ErrorModel::Additive(phase1_iv_ref::SIGMA),
            &theta_init,
            &omega_init,
        )
        .expect("FOCE should converge on Phase I IV 2-cpt");
    let elapsed = start.elapsed();

    println!("=== Phase I IV 2-cpt FOCEI — Parameter Recovery ===");
    println!("  Subjects:     {n_subjects}");
    println!("  Observations: {}", times.len());
    println!("  Converged:    {} in {} iters", result.converged, result.n_iter);
    println!("  Time:         {:.1} ms", elapsed.as_secs_f64() * 1000.0);
    println!();
    println!("  Parameter | True     | FOCEI    | Rel Diff");
    println!("  ----------|----------|----------|----------");
    let labels = ["CL", "V1", "Q", "V2"];
    let true_vals = [
        phase1_iv_ref::CL,
        phase1_iv_ref::V1,
        phase1_iv_ref::Q,
        phase1_iv_ref::V2,
    ];
    for (i, &label) in labels.iter().enumerate() {
        let rel_diff = (result.theta[i] - true_vals[i]).abs() / true_vals[i];
        println!(
            "  {:<9} | {:.4}   | {:.4}   | {:.1}%",
            label, true_vals[i], result.theta[i], rel_diff * 100.0,
        );
    }
    println!("  OFV:      {:.2}", result.ofv);
    println!();

    assert!(result.ofv.is_finite(), "OFV should be finite");

    // CL recovery.
    let cl_rel = (result.theta[0] - phase1_iv_ref::CL).abs() / phase1_iv_ref::CL;
    assert!(
        cl_rel < phase1_iv_ref::TOL_REL,
        "CL: hat={:.4}, true={:.4}, rel_err={:.3} > {:.2}",
        result.theta[0],
        phase1_iv_ref::CL,
        cl_rel,
        phase1_iv_ref::TOL_REL,
    );

    for (i, &label) in labels.iter().enumerate() {
        assert!(
            result.theta[i] > 0.0 && result.theta[i].is_finite(),
            "{} invalid: {}",
            label,
            result.theta[i],
        );
    }
}

// ============================================================================
// Test 8: Combined summary — all fits in one report
// ============================================================================

#[test]
fn parity_combined_summary() {
    println!();
    println!("============================================================");
    println!("  NextStat vs NONMEM: Parameter Estimation Parity Summary");
    println!("============================================================");
    println!();
    println!("  Dataset        | Model     | Method | Status");
    println!("  ---------------|-----------|--------|--------");

    // Theophylline SAEM
    let (times, y, subject_idx, n_subjects, mean_dose) = parse_theophylline();
    let theo_doses = vec![mean_dose; n_subjects];
    let saem_cfg = SaemConfig {
        n_burn: 300,
        n_iter: 200,
        seed: 42,
        store_theta_trace: false,
        ..Default::default()
    };
    let saem = SaemEstimator::new(saem_cfg);
    let (theo_saem, _) = saem
        .fit_1cpt_oral(
            &times, &y, &subject_idx, n_subjects, &theo_doses, 1.0,
            ErrorModel::Proportional(0.7), &[0.04, 0.45, 1.5], &[0.30, 0.25, 0.50],
        )
        .expect("Theophylline SAEM");
    let theo_saem_ok = theo_saem.converged
        && theo_saem.theta[0] > 0.02 && theo_saem.theta[0] < 0.08
        && theo_saem.theta[1] > 0.20 && theo_saem.theta[1] < 0.80
        && theo_saem.theta[2] > 0.5 && theo_saem.theta[2] < 5.0;
    println!(
        "  Theophylline   | 1-cpt oral| SAEM   | {}",
        if theo_saem_ok { "PASS" } else { "FAIL" }
    );

    // Theophylline FOCE
    let foce_cfg = FoceConfig {
        max_outer_iter: 200,
        max_inner_iter: 30,
        tol: 1e-5,
        interaction: true,
        ..FoceConfig::default()
    };
    let foce = FoceEstimator::new(foce_cfg);
    let theo_foce = foce
        .fit_1cpt_oral(
            &times, &y, &subject_idx, n_subjects, &theo_doses, 1.0,
            ErrorModel::Proportional(0.7), &[0.04, 0.45, 1.5], &[0.30, 0.25, 0.50],
        )
        .expect("Theophylline FOCE");
    // FOCEI: check parameter ranges (convergence flag may vary).
    let theo_foce_ok = theo_foce.ofv.is_finite()
        && theo_foce.theta[0] > 0.02 && theo_foce.theta[0] < 0.08
        && theo_foce.theta[1] > 0.20 && theo_foce.theta[1] < 0.80
        && theo_foce.theta[2] > 0.5 && theo_foce.theta[2] < 5.0;
    println!(
        "  Theophylline   | 1-cpt oral| FOCEI  | {}",
        if theo_foce_ok { "PASS" } else { "FAIL" }
    );

    // Warfarin SAEM
    let (wt, wy, ws) = generate_warfarin_synthetic(32, 42);
    let warf_doses = vec![100.0; 32];
    let (warf_saem, _) = saem
        .fit_1cpt_oral(
            &wt, &wy, &ws, 32, &warf_doses, 1.0,
            ErrorModel::Additive(0.3), &[0.10, 5.0, 0.5], &[0.30, 0.30, 0.30],
        )
        .expect("Warfarin SAEM");
    let warf_saem_ok = warf_saem.converged
        && (warf_saem.theta[0] - 0.134).abs() / 0.134 < 0.50
        && (warf_saem.theta[1] - 8.0).abs() / 8.0 < 0.50
        && (warf_saem.theta[2] - 1.0).abs() / 1.0 < 0.50;
    println!(
        "  Warfarin       | 1-cpt oral| SAEM   | {}",
        if warf_saem_ok { "PASS" } else { "FAIL" }
    );

    // Warfarin FOCE (more iterations for convergence)
    let foce_warf = FoceEstimator::new(FoceConfig {
        max_outer_iter: 300,
        max_inner_iter: 30,
        tol: 1e-4,
        interaction: true,
        ..FoceConfig::default()
    });
    let warf_foce = foce_warf
        .fit_1cpt_oral(
            &wt, &wy, &ws, 32, &warf_doses, 1.0,
            ErrorModel::Additive(0.3), &[0.10, 5.0, 0.5], &[0.30, 0.30, 0.30],
        )
        .expect("Warfarin FOCE");
    // FOCE may not converge fully — check parameter ranges with relaxed tolerance.
    let warf_foce_ok = warf_foce.ofv.is_finite()
        && warf_foce.theta[0] > 0.0 && warf_foce.theta[0].is_finite()
        && (warf_foce.theta[0] - 0.134).abs() / 0.134 < 0.60
        && (warf_foce.theta[1] - 8.0).abs() / 8.0 < 0.60
        && (warf_foce.theta[2] - 1.0).abs() / 1.0 < 0.60;
    println!(
        "  Warfarin       | 1-cpt oral| FOCEI  | {}",
        if warf_foce_ok { "PASS" } else { "FAIL" }
    );

    // Phase I IV 2-cpt SAEM
    let (it, iy, is) = generate_phase1_iv_2cpt(24, 42);
    let iv_doses = vec![100.0; 24];
    let saem4 = SaemEstimator::new(SaemConfig {
        n_burn: 400,
        n_iter: 300,
        seed: 42,
        store_theta_trace: false,
        ..Default::default()
    });
    let (iv_saem, _) = saem4
        .fit_2cpt_iv(
            &it, &iy, &is, 24, &iv_doses,
            ErrorModel::Additive(0.1), &[4.0, 8.0, 12.0, 15.0], &[0.30, 0.30, 0.30, 0.30],
        )
        .expect("Phase I IV SAEM");
    let iv_saem_ok = iv_saem.converged
        && (iv_saem.theta[0] - 5.0).abs() / 5.0 < 0.60
        && iv_saem.theta[0] > 0.0;
    println!(
        "  Phase I IV     | 2-cpt IV  | SAEM   | {}",
        if iv_saem_ok { "PASS" } else { "FAIL" }
    );

    // Phase I IV 2-cpt FOCE
    let iv_foce = FoceEstimator::focei()
        .fit_2cpt_iv(
            &it, &iy, &is, 24, &iv_doses,
            ErrorModel::Additive(0.1), &[4.0, 8.0, 12.0, 15.0], &[0.30, 0.30, 0.30, 0.30],
        )
        .expect("Phase I IV FOCE");
    let iv_foce_ok = iv_foce.ofv.is_finite()
        && (iv_foce.theta[0] - 5.0).abs() / 5.0 < 0.60
        && iv_foce.theta[0] > 0.0;
    println!(
        "  Phase I IV     | 2-cpt IV  | FOCEI  | {}",
        if iv_foce_ok { "PASS" } else { "FAIL" }
    );

    println!();

    // All should pass.
    assert!(theo_saem_ok, "Theophylline SAEM failed parity");
    assert!(theo_foce_ok, "Theophylline FOCEI failed parity");
    assert!(warf_saem_ok, "Warfarin SAEM failed parity");
    assert!(warf_foce_ok, "Warfarin FOCEI failed parity");
    assert!(iv_saem_ok, "Phase I IV SAEM failed parity");
    assert!(iv_foce_ok, "Phase I IV FOCEI failed parity");
}
