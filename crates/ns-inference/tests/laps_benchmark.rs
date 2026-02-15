//! LAPS benchmark: 3 cases on GPU.
//!
//! Run with:
//!   cargo test -p ns-inference --features cuda --release laps_benchmark -- --nocapture --ignored

#[cfg(feature = "cuda")]
mod bench {
    use ns_inference::diagnostics::compute_diagnostics;
    use ns_inference::laps::{LapsConfig, LapsModel, sample_laps};

    fn run_case(name: &str, model: &LapsModel) {
        let config = LapsConfig {
            n_chains: 4096,
            n_warmup: 500,
            n_samples: 2000,
            seed: 42,
            ..Default::default()
        };
        let r = sample_laps(model, config).unwrap();
        let diag = compute_diagnostics(&r.sampler_result);
        let min_ess = diag.ess_bulk.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_rhat = diag.r_hat.iter().cloned().fold(0.0f64, f64::max);
        let n_total: usize = r.sampler_result.chains.len() * 2000;
        let accept: f64 =
            r.sampler_result.chains.iter().flat_map(|c| c.accept_probs.iter()).sum::<f64>()
                / n_total as f64;

        println!("{name} (4096 chains, 500w + 2000s):");
        println!("  Wall:          {:.3}s", r.wall_time_s);
        println!(
            "  Phases:        init={:.3}s  warmup={:.3}s  sampling={:.3}s",
            r.phase_times[0], r.phase_times[1], r.phase_times[2]
        );
        println!("  Kernels:       {}", r.n_kernel_launches);
        println!("  Min ESS bulk:  {:.0}", min_ess);
        println!("  Max R-hat:     {:.4}", max_rhat);
        println!("  Accept rate:   {:.3}", accept);
        println!("  ESS/s:         {:.0}", min_ess / r.wall_time_s);
        println!("  Div rate:      {:.4}", diag.divergence_rate);
        println!();
    }

    #[test]
    #[ignore = "GPU benchmark; run with --ignored --nocapture"]
    fn laps_benchmark_std_normal() {
        let model = LapsModel::StdNormal { dim: 10 };
        run_case("std_normal_10d", &model);
    }

    #[test]
    #[ignore = "GPU benchmark; run with --ignored --nocapture"]
    fn laps_benchmark_eight_schools() {
        let model = LapsModel::EightSchools {
            y: vec![28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
            sigma: vec![15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
            prior_mu_sigma: 5.0,
            prior_tau_scale: 5.0,
        };
        run_case("eight_schools", &model);
    }

    #[test]
    #[ignore = "GPU benchmark; run with --ignored --nocapture"]
    fn laps_benchmark_neal_funnel() {
        let model = LapsModel::NealFunnel { dim: 10 };
        run_case("neal_funnel_10d", &model);
    }
}
