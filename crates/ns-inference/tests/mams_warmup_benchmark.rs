//! MAMS warmup benchmark: Pathfinder vs Random initialization.
//!
//! Compares wall time, ESS/s, and convergence quality for both strategies
//! across 3 benchmark models (std_normal_10d, eight_schools, neal_funnel_10d).
//!
//! Run with:
//!   cargo test -p ns-inference --release mams_warmup_benchmark -- --nocapture --ignored --test-threads=1

use ns_core::Result;
use ns_core::traits::{LogDensityModel, PreparedModelRef};
use ns_inference::diagnostics::compute_diagnostics;
use ns_inference::mams::{MamsConfig, sample_mams_multichain};
use ns_inference::nuts::InitStrategy;

// ---------- Models ----------

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

// ---------- Benchmark runner ----------

struct BenchResult {
    wall_s: f64,
    min_ess_bulk: f64,
    max_rhat: f64,
    accept_rate: f64,
    div_rate: f64,
    ess_per_s: f64,
}

fn run_single(
    model: &impl LogDensityModel,
    n_chains: usize,
    n_warmup: usize,
    n_samples: usize,
    init_strategy: InitStrategy,
    seed: u64,
) -> BenchResult {
    let config = MamsConfig { n_warmup, n_samples, init_strategy, ..Default::default() };
    let start = std::time::Instant::now();
    let result = sample_mams_multichain(model, n_chains, seed, config).unwrap();
    let wall_s = start.elapsed().as_secs_f64();

    let diag = compute_diagnostics(&result);
    let min_ess_bulk = diag.ess_bulk.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_rhat = diag.r_hat.iter().cloned().fold(0.0f64, f64::max);
    let n_total = result.chains.len() * n_samples;
    let accept_rate: f64 =
        result.chains.iter().flat_map(|c| c.accept_probs.iter()).sum::<f64>() / n_total as f64;

    BenchResult {
        wall_s,
        min_ess_bulk,
        max_rhat,
        accept_rate,
        div_rate: diag.divergence_rate,
        ess_per_s: min_ess_bulk / wall_s,
    }
}

fn compare_case(name: &str, model: &impl LogDensityModel) {
    let n_chains = 4;
    let n_samples = 2000;

    // Run 3 seeds for each strategy and take median wall time
    let seeds = [42u64, 123, 456];

    println!("=== {} ===", name);
    println!();

    // Random init: n_warmup=1000 (old default)
    let mut random_results: Vec<BenchResult> = Vec::new();
    for &seed in &seeds {
        random_results.push(run_single(
            model,
            n_chains,
            1000,
            n_samples,
            InitStrategy::Random,
            seed,
        ));
    }

    // Pathfinder init: n_warmup=500 (opt-in, shorter warmup with Hessian metric)
    let mut pathfinder_results: Vec<BenchResult> = Vec::new();
    for &seed in &seeds {
        pathfinder_results.push(run_single(
            model,
            n_chains,
            500,
            n_samples,
            InitStrategy::Pathfinder,
            seed,
        ));
    }

    // Compute medians
    let median = |vals: &mut Vec<f64>| -> f64 {
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        vals[vals.len() / 2]
    };

    let r_wall = median(&mut random_results.iter().map(|r| r.wall_s).collect());
    let r_ess = median(&mut random_results.iter().map(|r| r.min_ess_bulk).collect());
    let r_ess_s = median(&mut random_results.iter().map(|r| r.ess_per_s).collect());
    let r_rhat = median(&mut random_results.iter().map(|r| r.max_rhat).collect());
    let r_accept = median(&mut random_results.iter().map(|r| r.accept_rate).collect());
    let r_div = median(&mut random_results.iter().map(|r| r.div_rate).collect());

    let p_wall = median(&mut pathfinder_results.iter().map(|r| r.wall_s).collect());
    let p_ess = median(&mut pathfinder_results.iter().map(|r| r.min_ess_bulk).collect());
    let p_ess_s = median(&mut pathfinder_results.iter().map(|r| r.ess_per_s).collect());
    let p_rhat = median(&mut pathfinder_results.iter().map(|r| r.max_rhat).collect());
    let p_accept = median(&mut pathfinder_results.iter().map(|r| r.accept_rate).collect());
    let p_div = median(&mut pathfinder_results.iter().map(|r| r.div_rate).collect());

    let speedup = r_wall / p_wall;
    let ess_s_ratio = p_ess_s / r_ess_s;

    println!("  {:20} {:>12} {:>12} {:>12}", "", "Random(w=1000)", "Pathfinder(w=500)", "Ratio");
    println!("  {:20} {:>12.3} {:>12.3} {:>12.2}x", "Wall (s)", r_wall, p_wall, speedup);
    println!("  {:20} {:>12.0} {:>12.0} {:>12.2}x", "Min ESS bulk", r_ess, p_ess, p_ess / r_ess);
    println!("  {:20} {:>12.0} {:>12.0} {:>12.2}x", "ESS/s", r_ess_s, p_ess_s, ess_s_ratio);
    println!("  {:20} {:>12.4} {:>12.4}", "Max R-hat", r_rhat, p_rhat);
    println!("  {:20} {:>12.3} {:>12.3}", "Accept rate", r_accept, p_accept);
    println!("  {:20} {:>12.4} {:>12.4}", "Div rate", r_div, p_div);
    println!();

    // Informational only — Pathfinder can fail on funnel-like geometries.
    if p_rhat > 1.1 {
        println!(
            "  WARNING: Pathfinder R-hat ({:.4}) exceeds 1.1 — init_strategy=\"pathfinder\" is NOT recommended for this model.",
            p_rhat
        );
        println!();
    }
}

// ---------- Benchmark tests ----------

#[test]
#[ignore = "warmup benchmark; run with --ignored --nocapture --test-threads=1"]
fn mams_warmup_benchmark_std_normal() {
    compare_case("std_normal_10d", &StdNormal10D);
}

#[test]
#[ignore = "warmup benchmark; run with --ignored --nocapture --test-threads=1"]
fn mams_warmup_benchmark_eight_schools() {
    let model = ns_inference::eight_schools::EightSchoolsModel::new(
        vec![28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
        vec![15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
        5.0,
        5.0,
    )
    .unwrap();
    compare_case("eight_schools", &model);
}

#[test]
#[ignore = "warmup benchmark; run with --ignored --nocapture --test-threads=1"]
fn mams_warmup_benchmark_neal_funnel() {
    compare_case("neal_funnel_10d", &NealFunnel10D);
}

#[test]
#[ignore = "warmup benchmark; run with --ignored --nocapture --test-threads=1"]
fn mams_warmup_benchmark_all() {
    println!("MAMS Warmup Benchmark: Pathfinder (w=500) vs Random (w=1000)");
    println!("4 chains × 2000 samples, 3 seeds per strategy, median reported");
    println!("================================================================\n");

    mams_warmup_benchmark_std_normal();
    mams_warmup_benchmark_eight_schools();
    mams_warmup_benchmark_neal_funnel();

    println!("================================================================");
    println!("DONE");
}
