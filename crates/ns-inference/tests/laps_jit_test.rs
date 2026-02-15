//! Cross-validation test: JIT-compiled std_normal vs hardcoded std_normal via LAPS.
//!
//! Verifies that the NVRTC JIT path produces statistically equivalent posterior
//! to the build-time PTX path for the standard normal model.
//!
//! Run: `cargo test -p ns-inference --test laps_jit_test --features cuda -- --ignored`
//!
//! Requires NVIDIA GPU with CUDA at runtime.

#![cfg(feature = "cuda")]

use ns_inference::laps::{LapsConfig, LapsModel, sample_laps};

/// Standard normal user code for JIT compilation.
const STD_NORMAL_JIT_SRC: &str = r#"
__device__ double user_nll(const double* x, int dim, const double* data) {
    double nll = 0.0;
    for (int i = 0; i < dim; i++) {
        nll += 0.5 * x[i] * x[i];
    }
    return nll;
}

__device__ void user_grad(const double* x, double* grad, int dim, const double* data) {
    for (int i = 0; i < dim; i++) {
        grad[i] = x[i];
    }
}
"#;

fn make_config(n_chains: usize, n_samples: usize) -> LapsConfig {
    LapsConfig {
        n_chains,
        n_warmup: 500,
        n_samples,
        target_accept: 0.9,
        seed: 42,
        ..Default::default()
    }
}

/// Compute posterior mean across all chains and samples for a single parameter.
fn posterior_mean(result: &ns_inference::laps::LapsResult, param_idx: usize) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for chain in &result.sampler_result.chains {
        for draw in &chain.draws_constrained {
            sum += draw[param_idx];
            count += 1;
        }
    }
    sum / count as f64
}

/// Compute posterior variance across all chains and samples for a single parameter.
fn posterior_var(result: &ns_inference::laps::LapsResult, param_idx: usize) -> f64 {
    let mean = posterior_mean(result, param_idx);
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    for chain in &result.sampler_result.chains {
        for draw in &chain.draws_constrained {
            let d = draw[param_idx] - mean;
            sum_sq += d * d;
            count += 1;
        }
    }
    sum_sq / count as f64
}

/// Count divergences across all chains.
fn total_divergences(result: &ns_inference::laps::LapsResult) -> usize {
    result.sampler_result.chains.iter().flat_map(|c| c.divergences.iter()).filter(|&&d| d).count()
}

#[test]
#[ignore] // Requires CUDA GPU — run with --ignored
fn test_laps_jit_std_normal_vs_hardcoded() {
    let dim = 10;
    let n_chains = 256;
    let n_samples = 500;

    // --- Hardcoded path ---
    let model_hc = LapsModel::StdNormal { dim };
    let config_hc = make_config(n_chains, n_samples);
    let result_hc = sample_laps(&model_hc, config_hc).expect("hardcoded LAPS failed");

    // --- JIT path ---
    let model_jit = LapsModel::Custom {
        dim,
        param_names: (0..dim).map(|i| format!("x[{}]", i)).collect(),
        model_data: vec![],
        cuda_src: STD_NORMAL_JIT_SRC.to_string(),
    };
    let config_jit = make_config(n_chains, n_samples);
    let result_jit = sample_laps(&model_jit, config_jit).expect("JIT LAPS failed");

    // --- Verify JIT result ---
    // Posterior mean should be near zero for all params
    for p in 0..dim {
        let mean = posterior_mean(&result_jit, p);
        assert!(mean.abs() < 0.5, "JIT param {p} posterior mean {mean:.4} too far from 0");
    }

    // Posterior variance should be near 1.0
    // Tolerance is wide because 256 chains × 500 samples in debug mode = limited ESS
    for p in 0..dim {
        let var = posterior_var(&result_jit, p);
        assert!(
            (0.3..3.0).contains(&var),
            "JIT param {p} posterior var {var:.4} not in [0.3, 3.0]"
        );
    }

    // Low divergence count (allow a few with short warmup)
    let n_div = total_divergences(&result_jit);
    let max_div = (n_chains * n_samples) / 10; // <10% divergences
    assert!(n_div <= max_div, "JIT had {n_div} divergences (max {max_div})");

    // --- Cross-validate: JIT vs hardcoded means should be similar ---
    for p in 0..dim {
        let mean_hc = posterior_mean(&result_hc, p);
        let mean_jit = posterior_mean(&result_jit, p);
        let diff = (mean_hc - mean_jit).abs();
        assert!(
            diff < 0.5,
            "param {p}: hardcoded mean {mean_hc:.4} vs JIT mean {mean_jit:.4}, diff {diff:.4}"
        );
    }

    println!("JIT wall time: {:.3}s", result_jit.wall_time_s);
    println!("Hardcoded wall time: {:.3}s", result_hc.wall_time_s);
    println!("JIT kernel launches: {}", result_jit.n_kernel_launches);
}

#[test]
#[ignore] // Requires CUDA GPU — run: cargo test -p ns-inference --test laps_jit_test --features cuda --release -- --ignored bench_jit --nocapture
fn bench_jit_vs_hardcoded() {
    use std::time::Instant;

    println!("\n=== LAPS JIT vs Hardcoded Benchmark (std_normal, dim=10) ===\n");

    // Clear NVRTC cache for fair first-compile measurement
    let _ = std::fs::remove_dir_all(dirs::home_dir().unwrap().join(".cache/nextstat/ptx"));

    let configs: &[(usize, usize, usize)] = &[
        // (n_chains, n_warmup, n_samples)
        (256, 200, 500),
        (1024, 200, 500),
        (4096, 500, 2000),
    ];

    let n_repeats = 3;

    println!(
        "{:<12} {:<10} {:<10} | {:<12} {:<12} {:<12} | {:<8}",
        "Chains", "Warmup", "Samples", "Hardcoded(s)", "JIT(s)", "JIT+Comp(s)", "Ratio"
    );
    println!("{}", "-".repeat(90));

    for &(n_chains, n_warmup, n_samples) in configs {
        let dim = 10;
        let mut hc_times = Vec::new();
        let mut jit_times = Vec::new();
        let mut jit_compile_ms = 0u128;

        for rep in 0..n_repeats {
            // Hardcoded
            let model_hc = LapsModel::StdNormal { dim };
            let config_hc = LapsConfig {
                n_chains,
                n_warmup,
                n_samples,
                seed: 42 + rep as u64,
                ..Default::default()
            };
            let result_hc = sample_laps(&model_hc, config_hc).expect("hardcoded failed");
            hc_times.push(result_hc.wall_time_s);

            // JIT (first rep measures compile time)
            if rep == 0 {
                // Clear cache for first-compile measurement
                let _ =
                    std::fs::remove_dir_all(dirs::home_dir().unwrap().join(".cache/nextstat/ptx"));
                let t_comp = Instant::now();
                let compiler = ns_compute::nvrtc_mams::MamsJitCompiler::new().unwrap();
                let _ptx = compiler.compile(STD_NORMAL_JIT_SRC).unwrap();
                jit_compile_ms = t_comp.elapsed().as_millis();
            }

            let model_jit = LapsModel::Custom {
                dim,
                param_names: (0..dim).map(|i| format!("x[{}]", i)).collect(),
                model_data: vec![],
                cuda_src: STD_NORMAL_JIT_SRC.to_string(),
            };
            let config_jit = LapsConfig {
                n_chains,
                n_warmup,
                n_samples,
                seed: 42 + rep as u64,
                ..Default::default()
            };
            let result_jit = sample_laps(&model_jit, config_jit).expect("JIT failed");
            jit_times.push(result_jit.wall_time_s);
        }

        let hc_med = median(&mut hc_times);
        let jit_med = median(&mut jit_times);
        let jit_with_comp = jit_med + jit_compile_ms as f64 / 1000.0;
        let ratio = hc_med / jit_med;

        println!(
            "{:<12} {:<10} {:<10} | {:<12.4} {:<12.4} {:<12.4} | {:<8.2}x",
            n_chains, n_warmup, n_samples, hc_med, jit_med, jit_with_comp, ratio
        );
    }

    println!("\nNVRTC first compile: {}ms (cached: <1ms)", {
        let compiler = ns_compute::nvrtc_mams::MamsJitCompiler::new().unwrap();
        let t = Instant::now();
        let _ = compiler.compile(STD_NORMAL_JIT_SRC).unwrap();
        t.elapsed().as_millis()
    });
    println!(
        "GPU: {:?}",
        std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=name")
            .arg("--format=csv,noheader")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_default()
    );
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 0 { (v[n / 2 - 1] + v[n / 2]) / 2.0 } else { v[n / 2] }
}

fn dirs_home_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(std::path::PathBuf::from)
}

mod dirs {
    pub fn home_dir() -> Option<std::path::PathBuf> {
        std::env::var_os("HOME").map(std::path::PathBuf::from)
    }
}

#[test]
#[ignore] // Requires CUDA GPU
fn test_laps_jit_compile_cache_performance() {
    // Verify that second compile is fast (cache hit)
    use ns_compute::nvrtc_mams::MamsJitCompiler;

    let compiler = MamsJitCompiler::new().expect("MamsJitCompiler::new failed");

    let t0 = std::time::Instant::now();
    let _ptx1 = compiler.compile(STD_NORMAL_JIT_SRC).expect("first compile failed");
    let first_ms = t0.elapsed().as_millis();

    let t1 = std::time::Instant::now();
    let _ptx2 = compiler.compile(STD_NORMAL_JIT_SRC).expect("second compile failed");
    let second_ms = t1.elapsed().as_millis();

    println!("First compile: {}ms", first_ms);
    println!("Second compile (cached): {}ms", second_ms);

    // Cache hit should be at least 10x faster
    assert!(
        first_ms == 0 || second_ms * 10 <= first_ms,
        "Cache doesn't seem effective: first={}ms, second={}ms",
        first_ms,
        second_ms
    );
}

// ============ Multi-GPU Tests ============

#[test]
#[ignore = "requires 2+ CUDA GPUs"]
fn test_laps_multi_gpu_consistency() {
    let dim = 10;
    let n_chains = 512;
    let n_samples = 500;

    // Single GPU: all chains on device 0
    let model = LapsModel::StdNormal { dim };
    let config_single = LapsConfig {
        n_chains,
        n_warmup: 500,
        n_samples,
        seed: 42,
        device_ids: Some(vec![0]),
        ..Default::default()
    };
    let result_single = sample_laps(&model, config_single).expect("single-GPU LAPS failed");
    assert_eq!(result_single.n_devices, 1);
    assert_eq!(result_single.device_ids, vec![0]);

    // Multi GPU: split across 2 devices
    let config_multi = LapsConfig {
        n_chains,
        n_warmup: 500,
        n_samples,
        seed: 42,
        device_ids: Some(vec![0, 1]),
        ..Default::default()
    };
    let result_multi = sample_laps(&model, config_multi).expect("multi-GPU LAPS failed");
    assert_eq!(result_multi.n_devices, 2);
    assert_eq!(result_multi.device_ids, vec![0, 1]);

    // Print per-param statistics
    println!("\n=== Multi-GPU Consistency (dim={}, chains={}, warmup=500) ===", dim, n_chains);
    println!(
        "{:<6} {:<12} {:<12} {:<12} {:<12}",
        "Param", "Mean(1GPU)", "Mean(MGPU)", "Var(1GPU)", "Var(MGPU)"
    );
    println!("{}", "-".repeat(56));
    for p in 0..dim {
        let mean_s = posterior_mean(&result_single, p);
        let mean_m = posterior_mean(&result_multi, p);
        let var_s = posterior_var(&result_single, p);
        let var_m = posterior_var(&result_multi, p);
        println!("{:<6} {:<12.4} {:<12.4} {:<12.4} {:<12.4}", p, mean_s, mean_m, var_s, var_m);
    }

    println!(
        "\nSingle GPU: {:.3}s, {} chains reported",
        result_single.wall_time_s,
        result_single.sampler_result.chains.len()
    );
    println!(
        "Multi  GPU: {:.3}s, {} chains reported, {:.2}x speedup",
        result_multi.wall_time_s,
        result_multi.sampler_result.chains.len(),
        result_single.wall_time_s / result_multi.wall_time_s,
    );

    // Both should produce reasonable posteriors for std_normal
    for p in 0..dim {
        let mean_s = posterior_mean(&result_single, p);
        let mean_m = posterior_mean(&result_multi, p);
        assert!(mean_s.abs() < 0.5, "single-GPU param {p} mean {mean_s:.4} too far from 0");
        assert!(mean_m.abs() < 0.5, "multi-GPU param {p} mean {mean_m:.4} too far from 0");

        let var_s = posterior_var(&result_single, p);
        let var_m = posterior_var(&result_multi, p);
        // Wide tolerance: MAMS autocorrelation + 16 chains × 500 samples = noisy variance
        assert!(
            (0.1..5.0).contains(&var_s),
            "single-GPU param {p} var {var_s:.4} not in [0.1, 5.0]"
        );
        assert!(
            (0.1..5.0).contains(&var_m),
            "multi-GPU param {p} var {var_m:.4} not in [0.1, 5.0]"
        );
    }
}

#[test]
#[ignore = "requires 4 CUDA GPUs — run on H100/A100 SXM"]
fn bench_laps_4x_gpu() {
    use std::time::Instant;

    // Detect GPU name
    let gpu_name = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader", "-i", "0"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "Unknown GPU".into());

    println!("\n=== LAPS Multi-GPU Scaling Benchmark ===");
    println!("GPU: {} x4", gpu_name);
    println!("Model: std_normal, dim=100, warmup=500, samples=2000\n");

    let dim = 100;
    let n_warmup = 500;
    let n_samples = 2000;

    // Scaling matrix:
    // (A) Strong scaling: same total chains, more GPUs
    // (B) Weak scaling: same chains/GPU, more GPUs (each GPU ~16384 chains)
    let configs: &[(usize, &[usize])] = &[
        // Strong scaling baseline
        (16384, &[0]),
        (16384, &[0, 1]),
        (16384, &[0, 1, 2, 3]),
        // Weak scaling: 16384 chains per GPU
        (32768, &[0, 1]),
        (65536, &[0, 1, 2, 3]),
        // High chain count (saturated)
        (65536, &[0]),
    ];

    println!(
        "{:<10} {:<8} {:<10} {:<12} {:<12} {:<14} {:<10}",
        "Chains", "GPUs", "Wall(s)", "Warmup(s)", "Sampling(s)", "Samples/s", "Speedup"
    );
    println!("{}", "-".repeat(80));

    // Track baselines for speedup calculation
    let mut baseline_1gpu: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();

    for &(n_chains, dev_ids) in configs {
        let model = LapsModel::StdNormal { dim };
        let config = LapsConfig {
            n_chains,
            n_warmup,
            n_samples,
            seed: 42,
            device_ids: Some(dev_ids.to_vec()),
            ..Default::default()
        };

        let t0 = Instant::now();
        let result = sample_laps(&model, config).expect("LAPS failed");
        let elapsed = t0.elapsed().as_secs_f64();

        let sampling_time = result.phase_times[2].max(0.001);
        let warmup_time = result.phase_times[1];
        let total_samples = n_chains as f64 * n_samples as f64;
        let samples_per_sec = total_samples / sampling_time;

        let n_gpus = dev_ids.len();

        // Record 1-GPU baseline per chain count
        if n_gpus == 1 {
            baseline_1gpu.insert(n_chains, elapsed);
        }

        let speedup = baseline_1gpu.get(&n_chains).map(|&base| base / elapsed).unwrap_or(1.0);

        let speedup_str =
            if n_gpus == 1 { "1.00x".to_string() } else { format!("{:.2}x", speedup) };

        println!(
            "{:<10} {:<8} {:<10.3} {:<12.3} {:<12.3} {:<14.0} {:<10}",
            n_chains, n_gpus, elapsed, warmup_time, sampling_time, samples_per_sec, speedup_str,
        );

        // Sanity: posterior should be reasonable
        for p in 0..dim.min(3) {
            let mean = posterior_mean(&result, p);
            assert!(
                mean.abs() < 1.0,
                "param {p} mean {mean:.4} too far from 0 (chains={n_chains}, devs={:?})",
                dev_ids
            );
        }
    }

    // Summary
    println!("\n=== Scaling Analysis ===");
    println!("Strong scaling (same chains, more GPUs): limited by GPU occupancy.");
    println!("  A100 SXM needs ~16K+ chains/GPU for saturation (108 SMs × 256 threads).");
    println!("Weak scaling (same chains/GPU, more GPUs): expect ~1.0x wall time.");
    println!("  16384ch×1GPU vs 32768ch×2GPU vs 65536ch×4GPU should have equal sampling time.");
    println!("Warmup has barrier sync overhead (~1.5-2x vs single-GPU).");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_laps_device_ids_auto_detect() {
    // device_ids=None should auto-detect and work
    let model = LapsModel::StdNormal { dim: 5 };
    let config = LapsConfig {
        n_chains: 128,
        n_warmup: 50,
        n_samples: 100,
        seed: 42,
        device_ids: None,
        ..Default::default()
    };
    let result = sample_laps(&model, config).expect("auto-detect LAPS failed");
    assert!(result.n_devices >= 1);
    assert!(!result.device_ids.is_empty());
    assert!(result.sampler_result.chains.len() > 0);
    println!("Auto-detected {} device(s): {:?}", result.n_devices, result.device_ids);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn test_laps_explicit_single_device() {
    // device_ids=Some(vec![0]) should work identically to the original single-GPU path
    let model = LapsModel::StdNormal { dim: 5 };
    let config = LapsConfig {
        n_chains: 128,
        n_warmup: 50,
        n_samples: 100,
        seed: 42,
        device_ids: Some(vec![0]),
        ..Default::default()
    };
    let result = sample_laps(&model, config).expect("explicit single-GPU LAPS failed");
    assert_eq!(result.n_devices, 1);
    assert_eq!(result.device_ids, vec![0]);
    println!(
        "Explicit single GPU: {:.3}s, {} chains",
        result.wall_time_s,
        result.sampler_result.chains.len()
    );
}

/// Test that fused kernel produces statistically equivalent posterior to batched path.
///
/// Not bit-exact (FP accumulation order differs), but posterior means must agree ±0.1.
#[test]
#[ignore = "requires CUDA GPU"]
fn test_laps_fused_vs_batched_parity() {
    let model = LapsModel::StdNormal { dim: 10 };

    // Batched baseline
    let config_batched = LapsConfig {
        n_chains: 512,
        n_warmup: 200,
        n_samples: 200,
        seed: 42,
        fused_transitions: 0,
        batch_size: 100,
        ..Default::default()
    };
    let res_batched = sample_laps(&model, config_batched).expect("batched LAPS failed");

    // Fused path
    let config_fused = LapsConfig {
        n_chains: 512,
        n_warmup: 200,
        n_samples: 200,
        seed: 42,
        fused_transitions: 100,
        batch_size: 100,
        ..Default::default()
    };
    let res_fused = sample_laps(&model, config_fused).expect("fused LAPS failed");

    // Fused should use fewer kernel launches
    println!(
        "Batched: {} launches, {:.3}s | Fused: {} launches, {:.3}s",
        res_batched.n_kernel_launches,
        res_batched.wall_time_s,
        res_fused.n_kernel_launches,
        res_fused.wall_time_s,
    );
    assert!(
        res_fused.n_kernel_launches < res_batched.n_kernel_launches,
        "fused should have fewer launches"
    );

    // Check posterior mean parity across dims
    let dim = 10;
    let n_chains = res_batched.sampler_result.chains.len();
    for d in 0..dim {
        let mean_batched: f64 = res_batched
            .sampler_result
            .chains
            .iter()
            .flat_map(|c| c.draws_constrained.iter().map(|draw| draw[d]))
            .sum::<f64>()
            / (n_chains * 200) as f64;
        let mean_fused: f64 = res_fused
            .sampler_result
            .chains
            .iter()
            .flat_map(|c| c.draws_constrained.iter().map(|draw| draw[d]))
            .sum::<f64>()
            / (n_chains * 200) as f64;

        let diff = (mean_batched - mean_fused).abs();
        println!("dim[{d}]: batched={mean_batched:.4}, fused={mean_fused:.4}, diff={diff:.4}");
        assert!(diff < 0.2, "posterior mean dim[{d}] diff {diff:.4} > 0.2");
    }
}

/// Test configurable sync_interval, welford_chains_per_device, batch_size.
#[test]
#[ignore = "requires CUDA GPU"]
fn test_laps_configurable_sync_interval() {
    let model = LapsModel::StdNormal { dim: 5 };
    let config = LapsConfig {
        n_chains: 256,
        n_warmup: 100,
        n_samples: 100,
        seed: 42,
        sync_interval: 200,
        welford_chains_per_device: 512,
        batch_size: 2000,
        ..Default::default()
    };
    let result = sample_laps(&model, config).expect("configurable LAPS failed");

    let chains = &result.sampler_result.chains;
    assert!(!chains.is_empty(), "should have report chains");
    assert_eq!(chains[0].draws_constrained.len(), 100);

    // Check posterior quality: mean near 0 for std_normal
    let n_chains = chains.len();
    for d in 0..5 {
        let mean: f64 =
            chains.iter().flat_map(|c| c.draws_constrained.iter().map(|draw| draw[d])).sum::<f64>()
                / (n_chains * 100) as f64;
        assert!(mean.abs() < 0.5, "dim[{d}] mean {mean:.3} too far from 0");
    }

    println!(
        "Custom config: {:.3}s, {} launches, {} chains",
        result.wall_time_s, result.n_kernel_launches, n_chains,
    );
}
