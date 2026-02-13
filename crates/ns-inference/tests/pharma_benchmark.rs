//! Pharma benchmark suite: synthetic datasets mimicking classic pharmacometric
//! studies (Warfarin, Theophylline, Phenobarbital).
//!
//! Each benchmark generates a synthetic population dataset with known true
//! parameters, fits with FOCE/FOCEI, and validates parameter recovery.

use ns_inference::{
    ErrorModel, FoceConfig, FoceEstimator, OmegaMatrix, VpcConfig, gof_1cpt_oral, vpc_1cpt_oral,
};

use rand::SeedableRng;
use rand_distr::{Distribution, Normal as RandNormal};

// ---------------------------------------------------------------------------
// Helper: generate synthetic pop PK data
// ---------------------------------------------------------------------------

struct SyntheticPopPk {
    times: Vec<f64>,
    y: Vec<f64>,
    subject_idx: Vec<usize>,
    n_subjects: usize,
    true_theta: [f64; 3],
    true_omega_sds: [f64; 3],
    dose: f64,
    sigma: f64,
}

#[allow(clippy::too_many_arguments)]
fn generate_synthetic_1cpt_oral(
    cl_pop: f64,
    v_pop: f64,
    ka_pop: f64,
    omega_cl: f64,
    omega_v: f64,
    omega_ka: f64,
    sigma: f64,
    dose: f64,
    bioav: f64,
    n_subjects: usize,
    times_per_subject: &[f64],
    seed: u64,
) -> SyntheticPopPk {
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

        for &t in times_per_subject {
            let ke = cl_i / v_i;
            let c =
                (dose * bioav * ka_i / (v_i * (ka_i - ke))) * ((-ke * t).exp() - (-ka_i * t).exp());
            let obs = (c + noise.sample(&mut rng)).max(0.0);
            times.push(t);
            y.push(obs);
            subject_idx.push(sid);
        }
    }

    SyntheticPopPk {
        times,
        y,
        subject_idx,
        n_subjects,
        true_theta: [cl_pop, v_pop, ka_pop],
        true_omega_sds: [omega_cl, omega_v, omega_ka],
        dose,
        sigma,
    }
}

fn run_benchmark(
    name: &str,
    data: &SyntheticPopPk,
    theta_init: [f64; 3],
    omega_init: [f64; 3],
    tol_theta_rel: f64,
) {
    let bioav = 1.0;
    let em = ErrorModel::Additive(data.sigma);

    let cfg = FoceConfig { max_outer_iter: 150, max_inner_iter: 25, tol: 1e-4, interaction: true };
    let estimator = FoceEstimator::new(cfg);

    let start = std::time::Instant::now();
    let result = estimator
        .fit_1cpt_oral(
            &data.times,
            &data.y,
            &data.subject_idx,
            data.n_subjects,
            data.dose,
            bioav,
            em,
            &theta_init,
            &omega_init,
        )
        .unwrap();
    let elapsed = start.elapsed();

    println!("=== {name} Benchmark ===");
    println!("  Subjects: {}", data.n_subjects);
    println!("  Observations: {}", data.times.len());
    println!("  Converged: {} in {} iters", result.converged, result.n_iter);
    println!("  OFV: {:.2}", result.ofv);
    println!("  Time: {:.1} ms", elapsed.as_secs_f64() * 1000.0);
    println!(
        "  θ_CL: {:.4} (true: {:.4}, init: {:.4})",
        result.theta[0], data.true_theta[0], theta_init[0]
    );
    println!(
        "  θ_V:  {:.4} (true: {:.4}, init: {:.4})",
        result.theta[1], data.true_theta[1], theta_init[1]
    );
    println!(
        "  θ_Ka: {:.4} (true: {:.4}, init: {:.4})",
        result.theta[2], data.true_theta[2], theta_init[2]
    );
    println!("  ω_CL: {:.4} (true: {:.4})", result.omega[0], data.true_omega_sds[0]);
    println!("  ω_V:  {:.4} (true: {:.4})", result.omega[1], data.true_omega_sds[1]);
    println!("  ω_Ka: {:.4} (true: {:.4})", result.omega[2], data.true_omega_sds[2]);
    println!(
        "  Correlation matrix:\n    {:?}\n    {:?}\n    {:?}",
        result.correlation[0], result.correlation[1], result.correlation[2]
    );

    // Assertions.
    assert!(result.ofv.is_finite(), "{name}: OFV not finite: {}", result.ofv);
    assert_eq!(result.theta.len(), 3, "{name}: theta length");
    assert_eq!(result.eta.len(), data.n_subjects, "{name}: eta count");

    for (i, label) in ["CL", "V", "Ka"].iter().enumerate() {
        assert!(
            result.theta[i] > 0.0 && result.theta[i].is_finite(),
            "{name}: θ_{label} invalid: {}",
            result.theta[i]
        );
        let rel_err = (result.theta[i] - data.true_theta[i]).abs() / data.true_theta[i];
        assert!(
            rel_err < tol_theta_rel,
            "{name}: θ_{label} relative error {rel_err:.3} > {tol_theta_rel} (hat={:.4}, true={:.4})",
            result.theta[i],
            data.true_theta[i]
        );
    }

    // Run GOF diagnostics.
    let gof = gof_1cpt_oral(
        &data.times,
        &data.y,
        &data.subject_idx,
        data.dose,
        bioav,
        &result.theta,
        &result.eta,
        &em,
    )
    .unwrap();
    assert_eq!(gof.len(), data.times.len());
    let mean_iwres: f64 = gof.iter().map(|r| r.iwres).sum::<f64>() / gof.len() as f64;
    println!("  Mean IWRES: {mean_iwres:.4}");
    assert!(mean_iwres.abs() < 3.0, "{name}: mean IWRES = {mean_iwres:.4}, expected near 0");

    // Run VPC.
    let omega_mat = result.omega_matrix.clone();
    let vpc = vpc_1cpt_oral(
        &data.times,
        &data.y,
        &data.subject_idx,
        data.n_subjects,
        data.dose,
        bioav,
        &result.theta,
        &omega_mat,
        &em,
        &VpcConfig { n_sim: 50, n_bins: 5, seed: 42, ..VpcConfig::default() },
    )
    .unwrap();
    assert_eq!(vpc.bins.len(), 5, "{name}: VPC bins");
    println!("  VPC: {} bins, {} sims — OK", vpc.bins.len(), vpc.n_sim);
    println!();
}

// ---------------------------------------------------------------------------
// Warfarin benchmark
// ---------------------------------------------------------------------------
// Classic 1-cpt oral warfarin PK study (O'Reilly 1969 archetype).
// ~32 subjects, rich sampling.
// True params: CL=0.134 L/h, V=8.0 L, Ka=1.0 /h

#[test]
fn benchmark_warfarin() {
    let data = generate_synthetic_1cpt_oral(
        0.134, // CL (L/h)
        8.0,   // V (L)
        1.0,   // Ka (/h)
        0.20,  // ω_CL
        0.15,  // ω_V
        0.25,  // ω_Ka
        0.3,   // σ (mg/L additive)
        100.0, // dose (mg)
        1.0,   // bioav
        32,    // subjects
        &[0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0, 36.0, 48.0, 72.0],
        17,
    );
    run_benchmark("Warfarin", &data, [0.10, 5.0, 0.5], [0.3, 0.3, 0.3], 0.50);
}

// ---------------------------------------------------------------------------
// Theophylline benchmark
// ---------------------------------------------------------------------------
// Classic 1-cpt oral theophylline dataset (Boeckmann, Sheiner, Beal 1994).
// ~12 subjects, sparse sampling.
// True params: CL=0.04 L/h/kg (~2.8 L/h for 70 kg), V=0.5 L/kg (~35 L), Ka=1.5 /h

#[test]
fn benchmark_theophylline() {
    let data = generate_synthetic_1cpt_oral(
        2.8,   // CL (L/h) for 70 kg
        35.0,  // V (L) for 70 kg
        1.5,   // Ka (/h)
        0.20,  // ω_CL
        0.15,  // ω_V
        0.30,  // ω_Ka
        0.4,   // σ (mg/L additive)
        320.0, // dose (mg) typical oral theophylline
        1.0,
        12, // subjects
        &[0.25, 0.5, 1.0, 2.0, 3.5, 5.0, 7.0, 9.0, 12.0, 24.0],
        42,
    );
    run_benchmark(
        "Theophylline",
        &data,
        [2.0, 25.0, 1.0],
        [0.3, 0.3, 0.3],
        0.60, // more lenient: only 12 subjects
    );
}

// ---------------------------------------------------------------------------
// Phenobarbital benchmark
// ---------------------------------------------------------------------------
// Classic phenobarbital dataset (Grasela & Donn 1985 archetype).
// ~59 neonates, sparse IV dosing.
// Since our model is oral 1-cpt, we model it as 1-cpt oral with very fast Ka
// (essentially IV-like absorption).
// True params: CL=0.006 L/h/kg (~0.018 L/h for 3 kg neonate), V=0.9 L/kg (~2.7 L), Ka=10 /h

#[test]
fn benchmark_phenobarbital() {
    let data = generate_synthetic_1cpt_oral(
        0.018, // CL (L/h) for 3 kg neonate
        2.7,   // V (L) for 3 kg neonate
        10.0,  // Ka (/h) — very fast to simulate IV
        0.30,  // ω_CL (neonates have high variability)
        0.25,  // ω_V
        0.10,  // ω_Ka (low: rapid absorption)
        0.5,   // σ (mg/L additive)
        20.0,  // dose (mg) phenobarbital loading
        1.0,
        40, // subjects (neonatal ICU cohort)
        &[1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0, 72.0, 96.0],
        99,
    );
    run_benchmark("Phenobarbital", &data, [0.01, 2.0, 8.0], [0.3, 0.3, 0.3], 0.50);
}

// ---------------------------------------------------------------------------
// Full suite: all three benchmarks with correlated omega
// ---------------------------------------------------------------------------

#[test]
fn benchmark_warfarin_correlated_omega() {
    let cl_pop = 0.134;
    let v_pop = 8.0;
    let ka_pop = 1.0;
    let sigma = 0.3;
    let dose = 100.0;
    let bioav = 1.0;
    let n_subjects = 40;

    // True Ω with CL–V correlation of 0.5.
    let corr = vec![vec![1.0, 0.5, 0.0], vec![0.5, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
    let true_sds = [0.20, 0.15, 0.25];
    let true_omega = OmegaMatrix::from_correlation(&true_sds, &corr).unwrap();
    let chol = true_omega.cholesky();

    let times_per = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0];
    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    let std_normal = RandNormal::new(0.0_f64, 1.0).unwrap();
    let noise = RandNormal::new(0.0, sigma).unwrap();

    let mut times = Vec::new();
    let mut y = Vec::new();
    let mut subject_idx = Vec::new();

    for sid in 0..n_subjects {
        let z: Vec<f64> = (0..3).map(|_| std_normal.sample(&mut rng)).collect();
        let mut eta = [0.0; 3];
        for i in 0..3 {
            for j in 0..=i {
                eta[i] += chol[i][j] * z[j];
            }
        }
        let cl_i = cl_pop * eta[0].exp();
        let v_i = v_pop * eta[1].exp();
        let ka_i = ka_pop * eta[2].exp();

        for &t in &times_per {
            let ke = cl_i / v_i;
            let c =
                (dose * bioav * ka_i / (v_i * (ka_i - ke))) * ((-ke * t).exp() - (-ka_i * t).exp());
            times.push(t);
            y.push((c + noise.sample(&mut rng)).max(0.0));
            subject_idx.push(sid);
        }
    }

    let init_omega = OmegaMatrix::from_diagonal(&[0.3, 0.3, 0.3]).unwrap();
    let cfg = FoceConfig { max_outer_iter: 150, max_inner_iter: 25, tol: 1e-4, interaction: true };
    let estimator = FoceEstimator::new(cfg);

    let start = std::time::Instant::now();
    let result = estimator
        .fit_1cpt_oral_correlated(
            &times,
            &y,
            &subject_idx,
            n_subjects,
            dose,
            bioav,
            ErrorModel::Additive(sigma),
            &[0.10, 5.0, 0.5],
            init_omega,
        )
        .unwrap();
    let elapsed = start.elapsed();

    println!("=== Warfarin (Correlated Ω) Benchmark ===");
    println!("  Subjects: {n_subjects}");
    println!("  Converged: {} in {} iters", result.converged, result.n_iter);
    println!("  OFV: {:.2}", result.ofv);
    println!("  Time: {:.1} ms", elapsed.as_secs_f64() * 1000.0);
    println!("  θ: CL={:.4} V={:.4} Ka={:.4}", result.theta[0], result.theta[1], result.theta[2]);
    println!("  True: CL={cl_pop:.4} V={v_pop:.4} Ka={ka_pop:.4}");
    println!("  Estimated CL–V correlation: {:.3} (true: 0.500)", result.correlation[0][1]);
    println!();

    assert!(result.ofv.is_finite());
    for i in 0..3 {
        assert!(result.theta[i] > 0.0 && result.theta[i].is_finite());
    }
    assert!(result.correlation[0][1].is_finite());
}
