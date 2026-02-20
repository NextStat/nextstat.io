//! Phase 3 benchmark suite: SAEM, PD models, and adaptive ODE solvers.
//!
//! Run: `cargo test -p ns-inference --test phase3_benchmark -- --nocapture`

use ns_inference::ode_adaptive::{OdeOptions, OdeSystem, esdirk4, rk45};
use ns_inference::pd::{EmaxModel, IndirectResponseModel, IndirectResponseType, SigmoidEmaxModel};
use ns_inference::pk::ErrorModel;
use ns_inference::saem::{SaemConfig, SaemEstimator};

use rand::Rng;
use rand::SeedableRng;
use rand_distr::StandardNormal;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_warfarin_data(
    n_subjects: usize,
    n_obs: usize,
    theta: &[f64],
    omega_sds: &[f64],
    sigma: f64,
    dose: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut times = Vec::new();
    let mut y = Vec::new();
    let mut subj = Vec::new();

    for s in 0..n_subjects {
        let eta_cl: f64 = omega_sds[0] * rng.sample::<f64, _>(StandardNormal);
        let eta_v: f64 = omega_sds[1] * rng.sample::<f64, _>(StandardNormal);
        let eta_ka: f64 = omega_sds[2] * rng.sample::<f64, _>(StandardNormal);

        let cl_i = theta[0] * eta_cl.exp();
        let v_i = theta[1] * eta_v.exp();
        let ka_i = theta[2] * eta_ka.exp();

        for j in 0..n_obs {
            let t = (j + 1) as f64 * 24.0 / n_obs as f64;
            let c = ns_inference::pk::conc_oral(dose, 1.0, cl_i, v_i, ka_i, t);
            let noise: f64 = sigma * rng.sample::<f64, _>(StandardNormal);
            times.push(t);
            y.push((c + noise).max(0.01));
            subj.push(s);
        }
    }
    (times, y, subj)
}

// ---------------------------------------------------------------------------
// SAEM benchmarks
// ---------------------------------------------------------------------------

#[test]
fn benchmark_saem_warfarin() {
    let theta_true = [0.133, 8.0, 0.8];
    let omega_sds = [0.30, 0.25, 0.30];
    let sigma = 0.5;
    let dose = 100.0;
    let n_subjects = 32;
    let n_obs = 10;

    let (times, y, subj) =
        make_warfarin_data(n_subjects, n_obs, &theta_true, &omega_sds, sigma, dose, 42);
    let doses = vec![dose; n_subjects];

    let config = SaemConfig { n_burn: 200, n_iter: 100, seed: 42, ..Default::default() };
    let estimator = SaemEstimator::new(config);

    let (result, diag) = estimator
        .fit_1cpt_oral(
            &times,
            &y,
            &subj,
            n_subjects,
            &doses,
            1.0,
            ErrorModel::Additive(sigma),
            &theta_true,
            &omega_sds,
        )
        .expect("SAEM Warfarin should converge");

    println!("=== SAEM Warfarin ===");
    println!(
        "  theta: CL={:.4}, V={:.4}, Ka={:.4}",
        result.theta[0], result.theta[1], result.theta[2]
    );
    println!("  OFV: {:.2}", result.ofv);
    println!("  Converged: {}", result.converged);
    println!("  OFV trace len: {}", diag.ofv_trace.len());

    assert!(result.ofv.is_finite(), "OFV must be finite");

    for (k, name) in ["CL", "V", "Ka"].iter().enumerate() {
        let rel_err = (result.theta[k] - theta_true[k]).abs() / theta_true[k];
        assert!(
            rel_err < 3.0 * omega_sds[k],
            "SAEM {name}: fitted={:.4}, true={:.4}, rel_err={rel_err:.3}",
            result.theta[k],
            theta_true[k]
        );
    }
}

#[test]
fn benchmark_saem_vs_foce_parity() {
    let theta_true = [0.133, 8.0, 0.8];
    let omega_sds = [0.30, 0.25, 0.30];
    let sigma = 0.5;
    let dose = 100.0;
    let n_subjects = 40;
    let n_obs = 10;

    let (times, y, subj) =
        make_warfarin_data(n_subjects, n_obs, &theta_true, &omega_sds, sigma, dose, 99);
    let doses = vec![dose; n_subjects];

    // FOCE
    let foce = ns_inference::foce::FoceEstimator::focei();
    let foce_result = foce
        .fit_1cpt_oral(
            &times,
            &y,
            &subj,
            n_subjects,
            &doses,
            1.0,
            ErrorModel::Additive(sigma),
            &theta_true,
            &omega_sds,
        )
        .expect("FOCE should converge");

    // SAEM
    let saem_config = SaemConfig { n_burn: 200, n_iter: 100, seed: 99, ..Default::default() };
    let saem = SaemEstimator::new(saem_config);
    let (saem_result, _) = saem
        .fit_1cpt_oral(
            &times,
            &y,
            &subj,
            n_subjects,
            &doses,
            1.0,
            ErrorModel::Additive(sigma),
            &theta_true,
            &omega_sds,
        )
        .expect("SAEM should converge");

    println!("=== SAEM vs FOCE parity ===");
    for (k, name) in ["CL", "V", "Ka"].iter().enumerate() {
        let diff = (saem_result.theta[k] - foce_result.theta[k]).abs() / foce_result.theta[k];
        println!(
            "  {name}: FOCE={:.4}, SAEM={:.4}, rel_diff={:.3}",
            foce_result.theta[k], saem_result.theta[k], diff
        );
        // Both methods should agree within a generous tolerance
        // (they have different convergence properties)
        assert!(
            diff < 1.0,
            "SAEM/FOCE {name} diverged: FOCE={:.4}, SAEM={:.4}",
            foce_result.theta[k],
            saem_result.theta[k]
        );
    }
}

// ---------------------------------------------------------------------------
// PD model benchmarks
// ---------------------------------------------------------------------------

#[test]
fn benchmark_emax_dose_response() {
    let model = EmaxModel::new(0.0, 100.0, 10.0).unwrap();

    let doses = [0.0, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0];
    let predicted: Vec<f64> = doses.iter().map(|&d| model.predict(d)).collect();

    println!("=== Emax dose-response ===");
    for (d, p) in doses.iter().zip(predicted.iter()) {
        println!("  C={d:6.1} → E={p:.2}");
    }

    // Verify key pharmacological properties
    assert!((predicted[0] - 0.0).abs() < 1e-10, "E(0) = E0");
    assert!((predicted[4] - 50.0).abs() < 1e-10, "E(EC50) = Emax/2");
    assert!(predicted[7] > 90.0, "E(10×EC50) ≈ Emax");

    // Monotonicity
    for i in 1..predicted.len() {
        assert!(predicted[i] >= predicted[i - 1], "Emax must be monotonically increasing");
    }
}

#[test]
fn benchmark_sigmoid_emax_hill_coefficients() {
    let gammas = [0.5, 1.0, 2.0, 5.0];

    println!("=== Sigmoid Emax Hill coefficients ===");
    for &g in &gammas {
        let model = SigmoidEmaxModel::new(0.0, 100.0, 10.0, g).unwrap();
        let e_at_5 = model.predict(5.0);
        let e_at_10 = model.predict(10.0);
        let e_at_20 = model.predict(20.0);
        println!("  gamma={g:.1}: E(5)={e_at_5:.1}, E(10)={e_at_10:.1}, E(20)={e_at_20:.1}");

        // E(EC50) = Emax/2 regardless of gamma
        assert!((e_at_10 - 50.0).abs() < 1e-8, "E(EC50)=50 for gamma={g}");
    }

    // Higher gamma → steeper transition
    let shallow = SigmoidEmaxModel::new(0.0, 100.0, 10.0, 0.5).unwrap();
    let steep = SigmoidEmaxModel::new(0.0, 100.0, 10.0, 5.0).unwrap();
    assert!(
        steep.predict(20.0) > shallow.predict(20.0),
        "steeper gamma → faster approach to Emax above EC50"
    );
}

#[test]
fn benchmark_indirect_response_all_types() {
    let types = [
        (IndirectResponseType::InhibitProduction, "Type I: Inhibit Production"),
        (IndirectResponseType::InhibitLoss, "Type II: Inhibit Loss"),
        (IndirectResponseType::StimulateProduction, "Type III: Stimulate Production"),
        (IndirectResponseType::StimulateLoss, "Type IV: Stimulate Loss"),
    ];

    println!("=== Indirect Response Models ===");

    for (idr_type, name) in &types {
        let max_eff = match idr_type {
            IndirectResponseType::InhibitProduction | IndirectResponseType::InhibitLoss => 0.9,
            _ => 2.0,
        };

        let model = IndirectResponseModel::new(*idr_type, 1.0, 0.1, max_eff, 5.0).unwrap();

        let baseline = model.baseline();
        assert!((baseline - 10.0).abs() < 1e-10, "Baseline = kin/kout = 10");

        // Constant drug at high concentration
        let conc: Vec<(f64, f64)> = (0..=100).map(|i| (i as f64, 50.0)).collect();
        let times: Vec<f64> = vec![0.0, 24.0, 48.0, 72.0, 96.0];
        let response = model.simulate(&conc, &times, None, None).unwrap();

        println!("  {name}: baseline={baseline:.1}");
        for (t, r) in times.iter().zip(response.iter()) {
            println!("    t={t:5.0}h → R={r:.2}");
        }

        // Direction check
        let final_r = *response.last().unwrap();
        match idr_type {
            IndirectResponseType::InhibitProduction | IndirectResponseType::StimulateLoss => {
                assert!(final_r < baseline, "{name}: response should decrease, got {final_r:.2}");
            }
            IndirectResponseType::InhibitLoss | IndirectResponseType::StimulateProduction => {
                assert!(final_r > baseline, "{name}: response should increase, got {final_r:.2}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ODE solver benchmarks
// ---------------------------------------------------------------------------

struct TransitChainBench {
    n_transit: usize,
    ktr: f64,
    ka: f64,
    ke: f64,
}

impl OdeSystem for TransitChainBench {
    fn ndim(&self) -> usize {
        self.n_transit + 2
    }
    fn rhs(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        let n = self.n_transit;
        dydt[0] = -self.ktr * y[0];
        for i in 1..n {
            dydt[i] = self.ktr * (y[i - 1] - y[i]);
        }
        dydt[n] = self.ktr * y[n - 1] - self.ka * y[n];
        dydt[n + 1] = self.ka * y[n] - self.ke * y[n + 1];
    }
}

#[test]
fn benchmark_ode_transit_chain_rk45_vs_esdirk() {
    let sys = TransitChainBench { n_transit: 7, ktr: 10.0, ka: 1.0, ke: 0.1 };
    let mut y0 = vec![0.0; 9];
    y0[0] = 100.0;

    let opts = OdeOptions::default();

    let sol_rk = rk45(&sys, &y0, 0.0, 24.0, &opts).unwrap();
    let sol_es = esdirk4(&sys, &y0, 0.0, 24.0, &opts).unwrap();

    let c_rk = sol_rk.y.last().unwrap()[8];
    let c_es = sol_es.y.last().unwrap()[8];

    println!("=== Transit chain (7 compartments) ===");
    println!("  RK45:   C(24h)={c_rk:.4}, steps={}", sol_rk.t.len());
    println!("  ESDIRK: C(24h)={c_es:.4}, steps={}", sol_es.t.len());

    assert!(c_rk > 0.0, "RK45 central should have drug");
    assert!(c_es > 0.0, "ESDIRK central should have drug");

    let rel_diff = (c_rk - c_es).abs() / c_rk.max(1e-10);
    println!("  Relative diff: {rel_diff:.2e}");
    assert!(rel_diff < 0.05, "RK45 and ESDIRK should agree: rk={c_rk:.4}, es={c_es:.4}");
}

#[test]
fn benchmark_ode_stiff_transit_high_ktr() {
    // Stiff system: ktr=100 makes transit very fast vs slow ka/ke
    let sys = TransitChainBench { n_transit: 10, ktr: 100.0, ka: 1.0, ke: 0.1 };
    let mut y0 = vec![0.0; 12];
    y0[0] = 100.0;

    let opts = OdeOptions::default();

    let sol = esdirk4(&sys, &y0, 0.0, 48.0, &opts).unwrap();
    let c_final = sol.y.last().unwrap()[11];

    println!("=== Stiff transit (ktr=100, 10 compartments) ===");
    println!("  ESDIRK: C(48h)={c_final:.4}, steps={}", sol.t.len());

    assert!(c_final > 0.0, "central should have drug");
    // All transit compartments should be nearly empty
    for i in 0..10 {
        assert!(
            sol.y.last().unwrap()[i] < 0.1,
            "transit[{i}] should be empty: {}",
            sol.y.last().unwrap()[i]
        );
    }
}
