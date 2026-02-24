//! Acceptance gate: GLM n=5000 must converge with good diagnostics.
//!
//! Validates that LAPS produces well-mixed chains for a moderately large
//! logistic regression (n=5000, p=10). Runs 3 seeds and checks per-seed
//! and cross-seed stability thresholds.
//!
//! Run: `cargo test -p ns-inference --features cuda --release --test laps_glm_n5000_gate -- --ignored --nocapture`

#![cfg(any(feature = "cuda", feature = "metal"))]

use ns_inference::diagnostics::compute_diagnostics;
use ns_inference::laps::{LapsConfig, LapsModel, sample_laps};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, StandardNormal};

/// Generate synthetic logistic regression data (X ~ N(0,1), beta ~ N(0,0.5)).
fn generate_logistic_data(n: usize, p: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let beta_dist = Normal::new(0.0, 0.5).unwrap();
    let true_beta: Vec<f64> = (0..p).map(|_| beta_dist.sample(&mut rng)).collect();
    // X ~ N(0, 1), row-major [n x p]
    let x_data: Vec<f64> = (0..n * p).map(|_| StandardNormal.sample(&mut rng)).collect();
    // y ~ Bernoulli(sigmoid(X @ beta))
    let y_data: Vec<f64> = (0..n)
        .map(|i| {
            let eta: f64 = (0..p).map(|j| x_data[i * p + j] * true_beta[j]).sum();
            let prob = 1.0 / (1.0 + (-eta).exp());
            if rand::Rng::random::<f64>(&mut rng) < prob { 1.0 } else { 0.0 }
        })
        .collect();
    (x_data, y_data)
}

/// Multi-seed acceptance gate for GLM n=5000.
///
/// Per-seed thresholds (based on V100 benchmarks):
/// - max R-hat ≤ 1.2
/// - min ESS_bulk > 100
/// - divergence_rate < 0.05
///
/// Cross-seed stability:
/// - mean(max R-hat) ≤ 1.1
#[test]
#[ignore] // Requires GPU
fn test_glm_n5000_acceptance_gate() {
    let n = 5000;
    let p = 10;
    let seeds = [42u64, 123, 314];
    let mut max_rhats = Vec::new();

    for &seed in &seeds {
        let (x_data, y_data) = generate_logistic_data(n, p, seed);
        let model = LapsModel::GlmLogistic { x_data, y_data, n, p };
        let config = LapsConfig {
            n_chains: 4096,
            n_warmup: 500,
            n_samples: 1000,
            seed,
            report_chains: 256,
            ..Default::default()
        };
        let result = sample_laps(&model, config).expect("LAPS GLM n=5000 failed");
        let diag = compute_diagnostics(&result.sampler_result);

        // Per-seed gates
        let max_rhat =
            diag.r_hat.iter().copied().filter(|v| v.is_finite()).fold(f64::NEG_INFINITY, f64::max);
        let min_ess =
            diag.ess_bulk.iter().copied().filter(|v| v.is_finite()).fold(f64::INFINITY, f64::min);

        eprintln!(
            "seed {seed}: max_rhat={max_rhat:.4}, min_ess_bulk={min_ess:.0}, div_rate={:.4}",
            diag.divergence_rate
        );

        assert!(max_rhat <= 1.2, "seed {seed}: max R-hat {max_rhat:.3} > 1.2");
        assert!(min_ess > 100.0, "seed {seed}: min ESS_bulk {min_ess:.0} <= 100");
        assert!(
            diag.divergence_rate < 0.05,
            "seed {seed}: div rate {:.3} >= 5%",
            diag.divergence_rate
        );

        max_rhats.push(max_rhat);
    }

    // Cross-seed stability
    let mean_rhat: f64 = max_rhats.iter().sum::<f64>() / max_rhats.len() as f64;
    eprintln!("cross-seed: mean max_rhat={mean_rhat:.4}");
    assert!(mean_rhat <= 1.1, "mean max R-hat {mean_rhat:.3} > 1.1 across seeds");
}
