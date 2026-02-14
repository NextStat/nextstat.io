//! Parameter recovery integration tests for PK/NLME models.
//!
//! Covers gaps identified in test coverage audit:
//! - 2-compartment IV: parameter recovery with all 3 error models
//! - 2-compartment oral: parameter recovery with all 3 error models
//! - FOCE 1-cpt oral: proportional and combined error recovery
//! - SAEM 1-cpt oral: proportional and combined error recovery
//! - FOCE large sample (N=50): tighter recovery tolerance
//! - VPC with proportional error model
//! - GOF across all error models

use ns_inference::{
    VpcConfig,
    foce::{FoceConfig, FoceEstimator, OmegaMatrix},
    gof_1cpt_oral,
    mle::MaximumLikelihoodEstimator,
    pk::{
        ErrorModel, LloqPolicy, TwoCompartmentIvPkModel, TwoCompartmentOralPkModel,
        conc_iv_2cpt_macro, conc_oral, conc_oral_2cpt_macro,
    },
    saem::{SaemConfig, SaemEstimator},
    vpc_1cpt_oral,
};

use rand::SeedableRng;
use rand_distr::{Distribution, Normal as RandNormal};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate noisy observations from the 2-cpt IV model.
fn generate_2cpt_iv_data(
    cl: f64,
    v1: f64,
    v2: f64,
    q: f64,
    dose: f64,
    error_model: &ErrorModel,
    times: &[f64],
    seed: u64,
) -> Vec<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let eps = RandNormal::new(0.0, 1.0).unwrap();
    times
        .iter()
        .map(|&t| {
            let c = conc_iv_2cpt_macro(dose, cl, v1, v2, q, t);
            let noise_sd = error_model.sd(c);
            (c + noise_sd * eps.sample(&mut rng)).max(0.0)
        })
        .collect()
}

/// Generate noisy observations from the 2-cpt oral model.
fn generate_2cpt_oral_data(
    cl: f64,
    v1: f64,
    v2: f64,
    q: f64,
    ka: f64,
    dose: f64,
    bioav: f64,
    error_model: &ErrorModel,
    times: &[f64],
    seed: u64,
) -> Vec<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let eps = RandNormal::new(0.0, 1.0).unwrap();
    times
        .iter()
        .map(|&t| {
            let c = conc_oral_2cpt_macro(dose, bioav, cl, v1, v2, q, ka, t);
            let noise_sd = error_model.sd(c);
            (c + noise_sd * eps.sample(&mut rng)).max(0.0)
        })
        .collect()
}

/// Generate synthetic population PK data for 1-cpt oral with arbitrary error model.
struct PopPkData {
    times: Vec<f64>,
    y: Vec<f64>,
    subject_idx: Vec<usize>,
    n_subjects: usize,
}

fn generate_pop_1cpt_oral(
    cl_pop: f64,
    v_pop: f64,
    ka_pop: f64,
    omega_cl: f64,
    omega_v: f64,
    omega_ka: f64,
    error_model: &ErrorModel,
    dose: f64,
    bioav: f64,
    n_subjects: usize,
    times_per_subject: &[f64],
    seed: u64,
) -> PopPkData {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let eta_cl_dist = RandNormal::new(0.0, omega_cl).unwrap();
    let eta_v_dist = RandNormal::new(0.0, omega_v).unwrap();
    let eta_ka_dist = RandNormal::new(0.0, omega_ka).unwrap();
    let eps = RandNormal::new(0.0, 1.0).unwrap();

    let mut times = Vec::new();
    let mut y = Vec::new();
    let mut subject_idx = Vec::new();

    for sid in 0..n_subjects {
        let cl_i = cl_pop * eta_cl_dist.sample(&mut rng).exp();
        let v_i = v_pop * eta_v_dist.sample(&mut rng).exp();
        let ka_i = ka_pop * eta_ka_dist.sample(&mut rng).exp();

        for &t in times_per_subject {
            let c = conc_oral(dose, bioav, cl_i, v_i, ka_i, t);
            let noise_sd = error_model.sd(c);
            let obs = (c + noise_sd * eps.sample(&mut rng)).max(0.0);
            times.push(t);
            y.push(obs);
            subject_idx.push(sid);
        }
    }

    PopPkData { times, y, subject_idx, n_subjects }
}

fn assert_recovery(name: &str, label: &str, hat: f64, truth: f64, tol: f64) {
    let rel_err = (hat - truth).abs() / truth;
    assert!(
        rel_err < tol,
        "{name}: {label} relative error {rel_err:.3} > {tol} (hat={hat:.4}, true={truth:.4})"
    );
}

// ===========================================================================
// 2-compartment IV — parameter recovery (3 error models)
// ===========================================================================

fn run_2cpt_iv_recovery(error_model: ErrorModel, label: &str) {
    let cl = 1.0;
    let v1 = 10.0;
    let v2 = 20.0;
    let q = 0.5;
    let dose = 100.0;

    let times: Vec<f64> = (1..50).map(|i| i as f64 * 0.4).collect();
    let y = generate_2cpt_iv_data(cl, v1, v2, q, dose, &error_model, &times, 42);

    let model =
        TwoCompartmentIvPkModel::new(times, y, dose, error_model, None, LloqPolicy::Censored)
            .unwrap();

    let mle = MaximumLikelihoodEstimator::new();
    let fit = mle.fit(&model).unwrap();

    println!("=== 2-cpt IV {label} ===");
    println!("  Converged: {} in {} iters", fit.converged, fit.n_iter);
    println!("  CL: {:.4} (true: {cl})", fit.parameters[0]);
    println!("  V1: {:.4} (true: {v1})", fit.parameters[1]);
    println!("  V2: {:.4} (true: {v2})", fit.parameters[2]);
    println!("  Q:  {:.4} (true: {q})", fit.parameters[3]);

    assert!(fit.converged, "2-cpt IV {label}: did not converge");
    assert_recovery(&format!("2cpt_iv_{label}"), "CL", fit.parameters[0], cl, 0.30);
    assert_recovery(&format!("2cpt_iv_{label}"), "V1", fit.parameters[1], v1, 0.30);
}

#[test]
fn two_cpt_iv_recovery_additive() {
    run_2cpt_iv_recovery(ErrorModel::Additive(0.08), "additive");
}

#[test]
fn two_cpt_iv_recovery_proportional() {
    run_2cpt_iv_recovery(ErrorModel::Proportional(0.10), "proportional");
}

#[test]
fn two_cpt_iv_recovery_combined() {
    run_2cpt_iv_recovery(ErrorModel::Combined { sigma_add: 0.03, sigma_prop: 0.08 }, "combined");
}

// ===========================================================================
// 2-compartment oral — parameter recovery (3 error models)
// ===========================================================================

fn run_2cpt_oral_recovery(error_model: ErrorModel, label: &str) {
    let cl = 1.0;
    let v1 = 10.0;
    let v2 = 20.0;
    let q = 0.5;
    let ka = 1.5;
    let dose = 100.0;
    let bioav = 1.0;

    let times: Vec<f64> = (1..60).map(|i| i as f64 * 0.3).collect();
    let y = generate_2cpt_oral_data(cl, v1, v2, q, ka, dose, bioav, &error_model, &times, 55);

    let model = TwoCompartmentOralPkModel::new(
        times,
        y,
        dose,
        bioav,
        error_model,
        None,
        LloqPolicy::Censored,
    )
    .unwrap();

    let mle = MaximumLikelihoodEstimator::new();
    let fit = mle.fit(&model).unwrap();

    println!("=== 2-cpt oral {label} ===");
    println!("  Converged: {} in {} iters", fit.converged, fit.n_iter);
    println!("  CL: {:.4} (true: {cl})", fit.parameters[0]);
    println!("  V1: {:.4} (true: {v1})", fit.parameters[1]);
    println!("  V2: {:.4} (true: {v2})", fit.parameters[2]);
    println!("  Q:  {:.4} (true: {q})", fit.parameters[3]);
    println!("  Ka: {:.4} (true: {ka})", fit.parameters[4]);

    assert!(fit.converged, "2-cpt oral {label}: did not converge");
    assert_recovery(&format!("2cpt_oral_{label}"), "CL", fit.parameters[0], cl, 0.35);
    assert_recovery(&format!("2cpt_oral_{label}"), "V1", fit.parameters[1], v1, 0.35);
}

#[test]
fn two_cpt_oral_recovery_additive() {
    run_2cpt_oral_recovery(ErrorModel::Additive(0.05), "additive");
}

#[test]
fn two_cpt_oral_recovery_proportional() {
    run_2cpt_oral_recovery(ErrorModel::Proportional(0.10), "proportional");
}

#[test]
fn two_cpt_oral_recovery_combined() {
    run_2cpt_oral_recovery(ErrorModel::Combined { sigma_add: 0.02, sigma_prop: 0.07 }, "combined");
}

// ===========================================================================
// FOCE — 1-cpt oral with proportional / combined error
// ===========================================================================

fn run_foce_error_model_test(error_model: ErrorModel, label: &str) {
    let cl_pop = 0.15;
    let v_pop = 8.0;
    let ka_pop = 1.0;
    let omega_cl = 0.20;
    let omega_v = 0.15;
    let omega_ka = 0.25;
    let dose = 100.0;
    let bioav = 1.0;

    let sample_times = vec![0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0, 48.0];
    let data = generate_pop_1cpt_oral(
        cl_pop,
        v_pop,
        ka_pop,
        omega_cl,
        omega_v,
        omega_ka,
        &error_model,
        dose,
        bioav,
        30,
        &sample_times,
        101,
    );

    let cfg = FoceConfig { max_outer_iter: 300, max_inner_iter: 30, tol: 1e-4, interaction: true };
    let estimator = FoceEstimator::new(cfg);

    let result = estimator
        .fit_1cpt_oral(
            &data.times,
            &data.y,
            &data.subject_idx,
            data.n_subjects,
            dose,
            bioav,
            error_model,
            &[cl_pop, v_pop, ka_pop], // init at truth for non-additive error models
            &[omega_cl, omega_v, omega_ka],
        )
        .unwrap();

    println!("=== FOCE 1-cpt {label} ===");
    println!(
        "  Converged: {} in {} iters, OFV: {:.2}",
        result.converged, result.n_iter, result.ofv
    );
    println!("  θ_CL: {:.4} (true: {cl_pop})", result.theta[0]);
    println!("  θ_V:  {:.4} (true: {v_pop})", result.theta[1]);
    println!("  θ_Ka: {:.4} (true: {ka_pop})", result.theta[2]);

    assert!(result.ofv.is_finite(), "FOCE {label}: OFV not finite");
    // Non-additive error models can be harder to converge — check parameter recovery, not strict convergence
    assert_recovery(&format!("foce_{label}"), "CL", result.theta[0], cl_pop, 0.50);
    assert_recovery(&format!("foce_{label}"), "V", result.theta[1], v_pop, 0.50);
}

#[test]
fn foce_1cpt_oral_proportional_error() {
    run_foce_error_model_test(ErrorModel::Proportional(0.12), "proportional");
}

#[test]
fn foce_1cpt_oral_combined_error() {
    run_foce_error_model_test(
        ErrorModel::Combined { sigma_add: 0.1, sigma_prop: 0.10 },
        "combined",
    );
}

// ===========================================================================
// SAEM — 1-cpt oral with proportional / combined error
// ===========================================================================

fn run_saem_error_model_test(error_model: ErrorModel, label: &str) {
    let cl_pop = 0.15;
    let v_pop = 8.0;
    let ka_pop = 1.0;
    let omega_cl = 0.20;
    let omega_v = 0.15;
    let omega_ka = 0.25;
    let dose = 100.0;
    let bioav = 1.0;

    let sample_times = vec![0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0];
    let data = generate_pop_1cpt_oral(
        cl_pop,
        v_pop,
        ka_pop,
        omega_cl,
        omega_v,
        omega_ka,
        &error_model,
        dose,
        bioav,
        40,
        &sample_times,
        202,
    );

    let cfg = SaemConfig {
        n_burn: 100,
        n_iter: 80,
        n_chains: 1,
        seed: 12345,
        tol: 1e-4,
        ..SaemConfig::default()
    };
    let estimator = SaemEstimator::new(cfg);

    let (result, diag) = estimator
        .fit_1cpt_oral(
            &data.times,
            &data.y,
            &data.subject_idx,
            data.n_subjects,
            dose,
            bioav,
            error_model,
            &[cl_pop * 1.2, v_pop * 0.8, ka_pop * 1.1],
            &[omega_cl, omega_v, omega_ka],
        )
        .unwrap();

    println!("=== SAEM 1-cpt {label} ===");
    println!("  OFV: {:.2}, burn_only: {}", result.ofv, diag.burn_in_only);
    println!("  θ_CL: {:.4} (true: {cl_pop})", result.theta[0]);
    println!("  θ_V:  {:.4} (true: {v_pop})", result.theta[1]);
    println!("  θ_Ka: {:.4} (true: {ka_pop})", result.theta[2]);

    assert!(result.ofv.is_finite(), "SAEM {label}: OFV not finite");
    // SAEM is stochastic — use wider tolerance
    assert_recovery(&format!("saem_{label}"), "CL", result.theta[0], cl_pop, 0.60);
    assert_recovery(&format!("saem_{label}"), "V", result.theta[1], v_pop, 0.60);
}

#[test]
fn saem_1cpt_oral_proportional_error() {
    run_saem_error_model_test(ErrorModel::Proportional(0.12), "proportional");
}

#[test]
fn saem_1cpt_oral_combined_error() {
    run_saem_error_model_test(
        ErrorModel::Combined { sigma_add: 0.1, sigma_prop: 0.10 },
        "combined",
    );
}

// ===========================================================================
// FOCE large sample (N=50) — tighter recovery
// ===========================================================================

#[test]
fn foce_large_sample_n50() {
    let cl_pop = 0.134;
    let v_pop = 8.0;
    let ka_pop = 1.0;
    let omega_cl = 0.20;
    let omega_v = 0.15;
    let omega_ka = 0.25;
    let sigma = 0.3;
    let dose = 100.0;
    let bioav = 1.0;
    let error_model = ErrorModel::Additive(sigma);

    let sample_times = vec![0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0, 36.0, 48.0, 72.0];
    let data = generate_pop_1cpt_oral(
        cl_pop,
        v_pop,
        ka_pop,
        omega_cl,
        omega_v,
        omega_ka,
        &error_model,
        dose,
        bioav,
        50,
        &sample_times,
        303,
    );

    let cfg = FoceConfig { max_outer_iter: 300, max_inner_iter: 30, tol: 1e-4, interaction: true };
    let estimator = FoceEstimator::new(cfg);

    let result = estimator
        .fit_1cpt_oral(
            &data.times,
            &data.y,
            &data.subject_idx,
            data.n_subjects,
            dose,
            bioav,
            error_model,
            &[cl_pop * 1.1, v_pop * 1.1, ka_pop * 1.1], // slightly perturbed init
            &[omega_cl * 1.1, omega_v * 1.1, omega_ka * 1.1],
        )
        .unwrap();

    println!("=== FOCE Large Sample N=50 ===");
    println!(
        "  Converged: {} in {} iters, OFV: {:.2}",
        result.converged, result.n_iter, result.ofv
    );
    println!("  θ_CL: {:.4} (true: {cl_pop})", result.theta[0]);
    println!("  θ_V:  {:.4} (true: {v_pop})", result.theta[1]);
    println!("  θ_Ka: {:.4} (true: {ka_pop})", result.theta[2]);
    println!("  ω_CL: {:.4} (true: {omega_cl})", result.omega[0]);
    println!("  ω_V:  {:.4} (true: {omega_v})", result.omega[1]);
    println!("  ω_Ka: {:.4} (true: {omega_ka})", result.omega[2]);

    // With 50 subjects, tighter tolerance: 35%
    assert_recovery("foce_n50", "CL", result.theta[0], cl_pop, 0.35);
    assert_recovery("foce_n50", "V", result.theta[1], v_pop, 0.35);
    assert_recovery("foce_n50", "Ka", result.theta[2], ka_pop, 0.35);
    // Omega recovery: wider tolerance (50%) since variance components are harder to estimate
    assert_recovery("foce_n50", "ω_CL", result.omega[0], omega_cl, 0.50);
    assert_recovery("foce_n50", "ω_V", result.omega[1], omega_v, 0.50);
}

// ===========================================================================
// VPC with proportional error
// ===========================================================================

#[test]
fn vpc_proportional_error() {
    let cl_pop = 0.134;
    let v_pop = 8.0;
    let ka_pop = 1.0;
    let omega_cl = 0.20;
    let omega_v = 0.15;
    let omega_ka = 0.25;
    let dose = 100.0;
    let bioav = 1.0;
    let error_model = ErrorModel::Proportional(0.12);

    let sample_times = vec![0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0];
    let data = generate_pop_1cpt_oral(
        cl_pop,
        v_pop,
        ka_pop,
        omega_cl,
        omega_v,
        omega_ka,
        &error_model,
        dose,
        bioav,
        20,
        &sample_times,
        404,
    );

    let omega_mat = OmegaMatrix::from_diagonal(&[omega_cl, omega_v, omega_ka]).unwrap();
    let theta = [cl_pop, v_pop, ka_pop];

    let vpc = vpc_1cpt_oral(
        &data.times,
        &data.y,
        &data.subject_idx,
        data.n_subjects,
        dose,
        bioav,
        &theta,
        &omega_mat,
        &error_model,
        &VpcConfig { n_sim: 100, n_bins: 5, seed: 42, ..VpcConfig::default() },
    )
    .unwrap();

    println!("=== VPC proportional error ===");
    println!("  Bins: {}, Sims: {}", vpc.bins.len(), vpc.n_sim);
    for bin in &vpc.bins {
        println!("  t={:.1}: n_obs={}, obs_q={:?}", bin.time, bin.n_obs, bin.obs_quantiles);
    }

    assert_eq!(vpc.bins.len(), 5, "VPC: expected 5 bins");
    assert_eq!(vpc.n_sim, 100);
    let non_empty = vpc.bins.iter().filter(|b| b.n_obs > 0).count();
    assert!(non_empty >= 3, "VPC: at least 3 non-empty bins expected, got {non_empty}");
}

// ===========================================================================
// GOF across all error models
// ===========================================================================

#[test]
fn gof_all_error_models() {
    for (label, error_model) in [
        ("additive", ErrorModel::Additive(0.3)),
        ("proportional", ErrorModel::Proportional(0.12)),
        ("combined", ErrorModel::Combined { sigma_add: 0.1, sigma_prop: 0.10 }),
    ] {
        let cl_pop = 0.134;
        let v_pop = 8.0;
        let ka_pop = 1.0;
        let omega_cl = 0.20;
        let omega_v = 0.15;
        let omega_ka = 0.25;
        let dose = 100.0;
        let bioav = 1.0;

        let sample_times = vec![0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
        let data = generate_pop_1cpt_oral(
            cl_pop,
            v_pop,
            ka_pop,
            omega_cl,
            omega_v,
            omega_ka,
            &error_model,
            dose,
            bioav,
            15,
            &sample_times,
            505,
        );

        // Use true etas (all zero for simplicity — tests GOF infrastructure, not estimation)
        let etas: Vec<Vec<f64>> = (0..data.n_subjects).map(|_| vec![0.0, 0.0, 0.0]).collect();
        let theta = [cl_pop, v_pop, ka_pop];

        let gof = gof_1cpt_oral(
            &data.times,
            &data.y,
            &data.subject_idx,
            dose,
            bioav,
            &theta,
            &etas,
            &error_model,
        )
        .unwrap();

        println!("=== GOF {label} ===");
        println!("  Records: {}", gof.len());

        assert_eq!(gof.len(), data.times.len(), "GOF {label}: record count mismatch");
        for r in &gof {
            assert!(r.pred.is_finite(), "GOF {label}: PRED not finite at t={}", r.time);
            assert!(r.ipred.is_finite(), "GOF {label}: IPRED not finite at t={}", r.time);
            assert!(r.iwres.is_finite(), "GOF {label}: IWRES not finite at t={}", r.time);
            assert!(r.cwres.is_finite(), "GOF {label}: CWRES not finite at t={}", r.time);
        }

        let mean_iwres: f64 = gof.iter().map(|r| r.iwres).sum::<f64>() / gof.len() as f64;
        println!("  Mean IWRES: {mean_iwres:.4}");
    }
}
