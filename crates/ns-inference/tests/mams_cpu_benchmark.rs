//! CPU MAMS benchmark: same 3 cases as LAPS (GPU) for direct comparison.
//!
//! Run with:
//!   cargo test -p ns-inference --release mams_cpu_benchmark -- --nocapture --ignored --test-threads=1

use ns_core::Result;
use ns_core::traits::{LogDensityModel, PreparedModelRef};
use ns_inference::diagnostics::compute_diagnostics;
use ns_inference::mams::{MamsConfig, sample_mams_multichain};

// ---------- Model: Standard Normal 10D ----------

struct StdNormal10D;

impl LogDensityModel for StdNormal10D {
    type Prepared<'a>
        = PreparedModelRef<'a, Self>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        10
    }
    fn parameter_names(&self) -> Vec<String> {
        (0..10).map(|i| format!("x[{}]", i)).collect()
    }
    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![(f64::NEG_INFINITY, f64::INFINITY); 10]
    }
    fn parameter_init(&self) -> Vec<f64> {
        vec![0.0; 10]
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        Ok(params.iter().map(|x| 0.5 * x * x).sum())
    }
    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        Ok(params.to_vec())
    }
    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

// ---------- Model: Neal's Funnel 10D ----------

struct NealFunnel10D;

impl LogDensityModel for NealFunnel10D {
    type Prepared<'a>
        = PreparedModelRef<'a, Self>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        10
    }
    fn parameter_names(&self) -> Vec<String> {
        let mut names = vec!["v".into()];
        for i in 1..10 {
            names.push(format!("x[{}]", i));
        }
        names
    }
    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![(f64::NEG_INFINITY, f64::INFINITY); 10]
    }
    fn parameter_init(&self) -> Vec<f64> {
        vec![0.0; 10]
    }

    fn nll(&self, params: &[f64]) -> Result<f64> {
        let v = params[0];
        let sum_x2: f64 = params[1..].iter().map(|x| x * x).sum();
        Ok(v * v / 18.0 + 9.0 * 0.5 * v + 0.5 * (-v).exp() * sum_x2)
    }

    fn grad_nll(&self, params: &[f64]) -> Result<Vec<f64>> {
        let v = params[0];
        let exp_neg_v = (-v).exp();
        let sum_x2: f64 = params[1..].iter().map(|x| x * x).sum();
        let mut grad = Vec::with_capacity(10);
        grad.push(v / 9.0 + 9.0 * 0.5 - 0.5 * exp_neg_v * sum_x2);
        for &xi in &params[1..] {
            grad.push(exp_neg_v * xi);
        }
        Ok(grad)
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

fn run_case(name: &str, model: &impl LogDensityModel, n_chains: usize) {
    let config = MamsConfig { n_warmup: 500, n_samples: 2000, ..Default::default() };
    let start = std::time::Instant::now();
    let result = sample_mams_multichain(model, n_chains, 42, config).unwrap();
    let wall = start.elapsed().as_secs_f64();

    let diag = compute_diagnostics(&result);
    let min_ess = diag.ess_bulk.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_rhat = diag.r_hat.iter().cloned().fold(0.0f64, f64::max);
    let n_total: usize = result.chains.len() * 2000;
    let accept: f64 =
        result.chains.iter().flat_map(|c| c.accept_probs.iter()).sum::<f64>() / n_total as f64;

    println!("{name} ({n_chains} chains, 500w + 2000s):");
    println!("  Wall:          {:.3}s", wall);
    println!("  Min ESS bulk:  {:.0}", min_ess);
    println!("  Max R-hat:     {:.4}", max_rhat);
    println!("  Accept rate:   {:.3}", accept);
    println!("  ESS/s:         {:.0}", min_ess / wall);
    println!("  Div rate:      {:.4}", diag.divergence_rate);
    println!();
}

// --- 4-chain benchmark (typical CPU usage) ---

#[test]
#[ignore = "CPU benchmark; run with --ignored --nocapture"]
fn mams_cpu_benchmark_std_normal_4ch() {
    run_case("std_normal_10d (CPU)", &StdNormal10D, 4);
}

#[test]
#[ignore = "CPU benchmark; run with --ignored --nocapture"]
fn mams_cpu_benchmark_eight_schools_4ch() {
    let model = ns_inference::eight_schools::EightSchoolsModel::new(
        vec![28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
        vec![15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
        5.0,
        5.0,
    )
    .unwrap();
    run_case("eight_schools (CPU)", &model, 4);
}

#[test]
#[ignore = "CPU benchmark; run with --ignored --nocapture"]
fn mams_cpu_benchmark_neal_funnel_4ch() {
    run_case("neal_funnel_10d (CPU)", &NealFunnel10D, 4);
}

// --- 16-chain benchmark (for fair ESS comparison with LAPS reporting 16 chains) ---

#[test]
#[ignore = "CPU benchmark; run with --ignored --nocapture"]
fn mams_cpu_benchmark_std_normal_16ch() {
    run_case("std_normal_10d (CPU 16ch)", &StdNormal10D, 16);
}

#[test]
#[ignore = "CPU benchmark; run with --ignored --nocapture"]
fn mams_cpu_benchmark_eight_schools_16ch() {
    let model = ns_inference::eight_schools::EightSchoolsModel::new(
        vec![28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
        vec![15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
        5.0,
        5.0,
    )
    .unwrap();
    run_case("eight_schools (CPU 16ch)", &model, 16);
}

#[test]
#[ignore = "CPU benchmark; run with --ignored --nocapture"]
fn mams_cpu_benchmark_neal_funnel_16ch() {
    run_case("neal_funnel_10d (CPU 16ch)", &NealFunnel10D, 16);
}

// --- 4096-chain benchmark (apples-to-apples with LAPS GPU) ---

#[test]
#[ignore = "CPU benchmark; run with --ignored --nocapture"]
fn mams_cpu_benchmark_std_normal_4096ch() {
    run_case("std_normal_10d (CPU 4096ch)", &StdNormal10D, 4096);
}

#[test]
#[ignore = "CPU benchmark; run with --ignored --nocapture"]
fn mams_cpu_benchmark_eight_schools_4096ch() {
    let model = ns_inference::eight_schools::EightSchoolsModel::new(
        vec![28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
        vec![15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
        5.0,
        5.0,
    )
    .unwrap();
    run_case("eight_schools (CPU 4096ch)", &model, 4096);
}

#[test]
#[ignore = "CPU benchmark; run with --ignored --nocapture"]
fn mams_cpu_benchmark_neal_funnel_4096ch() {
    run_case("neal_funnel_10d (CPU 4096ch)", &NealFunnel10D, 4096);
}
