//! Integration test for the GPU flow session.
//!
//! Requires `--features cuda` and NVIDIA GPU at runtime.
//! Run: `cargo test -p ns-inference --features cuda --test gpu_flow_session_test`

#![cfg(feature = "cuda")]

use ns_compute::cuda_driver::{CudaContext, DevicePtr};
use ns_compute::cuda_flow_nll::{CudaFlowNllAccelerator, FlowNllConfig};
use ns_compute::unbinned_types::GpuUnbinnedGaussConstraintEntry;
use ns_inference::gpu_flow_session::{FlowProcessDesc, GpuFlowSession, GpuFlowSessionConfig};

fn has_cuda() -> bool {
    CudaFlowNllAccelerator::is_available()
}

// ── CudaFlowNllAccelerator unit tests ────────────────────────────────────

#[test]
fn test_cuda_flow_nll_single_process_standard_normal() {
    if !has_cuda() {
        eprintln!("SKIP: CUDA not available");
        return;
    }

    // Single process, no constraints. Events drawn from standard normal.
    // NLL = ν_tot - Σ log(ν * p(x_i)) = ν - Σ (log ν + log p(x_i))
    let n_events = 5;
    let n_procs = 1;
    let config = FlowNllConfig {
        n_events,
        n_procs,
        n_params: 0,
        n_context: 0,
        gauss_constraints: vec![],
        constraint_const: 0.0,
    };

    let mut accel = CudaFlowNllAccelerator::new(&config).unwrap();

    // log p for standard normal at x = {-1, 0, 1, 0.5, -0.5}
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    let xs = [-1.0, 0.0, 1.0, 0.5, -0.5];
    let logp: Vec<f64> = xs.iter().map(|&x| -0.5 * x * x - 0.5 * ln2pi).collect();

    let yields = [100.0]; // ν = 100
    let params: [f64; 0] = [];

    let nll = accel.nll(&logp, &yields, &params).unwrap();

    // Expected: ν_tot - Σ log(ν * p(x_i)) = 100 - Σ(log(100) + logp_i)
    let sum_logf: f64 = logp.iter().map(|lp| (100.0f64).ln() + lp).sum();
    let expected_nll = 100.0 - sum_logf;

    assert!((nll - expected_nll).abs() < 1e-6, "NLL mismatch: got {nll}, expected {expected_nll}");
}

#[test]
fn test_cuda_flow_nll_two_processes() {
    if !has_cuda() {
        eprintln!("SKIP: CUDA not available");
        return;
    }

    let n_events = 3;
    let n_procs = 2;
    let config = FlowNllConfig {
        n_events,
        n_procs,
        n_params: 0,
        n_context: 0,
        gauss_constraints: vec![],
        constraint_const: 0.0,
    };

    let mut accel = CudaFlowNllAccelerator::new(&config).unwrap();

    // Process 0: uniform on [-6, 6] → log p = -log(12) ≈ -2.485
    let logp_uniform = vec![-12.0f64.ln(); n_events];
    // Process 1: standard normal
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    let xs = [0.0, 1.0, -1.0];
    let logp_gauss: Vec<f64> = xs.iter().map(|&x| -0.5 * x * x - 0.5 * ln2pi).collect();

    // logp_flat is [n_procs × n_events] row-major
    let mut logp_flat = Vec::with_capacity(n_procs * n_events);
    logp_flat.extend_from_slice(&logp_uniform);
    logp_flat.extend_from_slice(&logp_gauss);

    let yields = [50.0, 100.0]; // signal=100, bkg=50
    let params: [f64; 0] = [];

    let nll = accel.nll(&logp_flat, &yields, &params).unwrap();

    // Compute expected NLL on CPU.
    let nu_tot = 150.0;
    let mut sum_logf = 0.0;
    for i in 0..n_events {
        let term0 = (50.0f64).ln() + logp_uniform[i];
        let term1 = (100.0f64).ln() + logp_gauss[i];
        let max_t = term0.max(term1);
        let lse = max_t + ((term0 - max_t).exp() + (term1 - max_t).exp()).ln();
        sum_logf += lse;
    }
    let expected_nll = nu_tot - sum_logf;

    assert!((nll - expected_nll).abs() < 1e-6, "NLL mismatch: got {nll}, expected {expected_nll}");
}

#[test]
fn test_cuda_flow_nll_with_gaussian_constraint() {
    if !has_cuda() {
        eprintln!("SKIP: CUDA not available");
        return;
    }

    let n_events = 2;
    let n_procs = 1;
    let config = FlowNllConfig {
        n_events,
        n_procs,
        n_params: 1,
        n_context: 0,
        gauss_constraints: vec![GpuUnbinnedGaussConstraintEntry {
            center: 0.0,
            inv_width: 1.0, // sigma = 1
            param_idx: 0,
            _pad: 0,
        }],
        constraint_const: 0.0,
    };

    let mut accel = CudaFlowNllAccelerator::new(&config).unwrap();

    let logp = vec![-1.0, -1.5]; // dummy log-prob values
    let yields = [10.0];
    let params = [2.0]; // parameter at 2σ from constraint center

    let nll = accel.nll(&logp, &yields, &params).unwrap();

    // Expected: 10 - Σ log(10 * exp(logp_i)) + 0.5*(2-0)^2/1^2
    // = 10 - (log10 + (-1)) - (log10 + (-1.5)) + 0.5*4
    // = 10 - (log10 - 1 + log10 - 1.5) + 2.0
    let sum_logf = (10.0f64.ln() - 1.0) + (10.0f64.ln() - 1.5);
    let expected_nll = 10.0 - sum_logf + 0.5 * 4.0;

    assert!((nll - expected_nll).abs() < 1e-6, "NLL mismatch: got {nll}, expected {expected_nll}");
}

// ── GpuFlowSession tests ─────────────────────────────────────────────────

#[test]
fn test_gpu_flow_session_yield_computation() {
    // This test doesn't need CUDA — just tests yield logic.
    let config = GpuFlowSessionConfig {
        processes: vec![
            FlowProcessDesc {
                process_index: 0,
                base_yield: 100.0,
                yield_param_idx: Some(0),
                yield_is_scaled: true,
                context_param_indices: vec![],
            },
            FlowProcessDesc {
                process_index: 1,
                base_yield: 50.0,
                yield_param_idx: None,
                yield_is_scaled: false,
                context_param_indices: vec![],
            },
        ],
        n_events: 10,
        n_params: 1,
        n_context: 0,
        gauss_constraints: vec![],
        constraint_const: 0.0,
    };

    // Can't construct GpuFlowSession without CUDA, but we can test yield logic
    // by calling the function directly if it were public. Instead, just verify
    // the config is well-formed.
    assert_eq!(config.processes.len(), 2);
    assert_eq!(config.processes[0].base_yield, 100.0);

    // With params = [1.5]:
    // Process 0: 100.0 * 1.5 = 150.0
    // Process 1: 50.0 (fixed)
    if !has_cuda() {
        eprintln!("SKIP: CUDA not available (GpuFlowSession needs GPU)");
        return;
    }

    let session = GpuFlowSession::new(config).unwrap();
    let yields = session.compute_yields(&[1.5]);
    assert_eq!(yields.len(), 2);
    assert!((yields[0] - 150.0).abs() < 1e-10);
    assert!((yields[1] - 50.0).abs() < 1e-10);
}

#[test]
fn test_gpu_flow_session_nll() {
    if !has_cuda() {
        eprintln!("SKIP: CUDA not available");
        return;
    }

    let config = GpuFlowSessionConfig {
        processes: vec![FlowProcessDesc {
            process_index: 0,
            base_yield: 50.0,
            yield_param_idx: Some(0),
            yield_is_scaled: true,
            context_param_indices: vec![],
        }],
        n_events: 3,
        n_params: 1,
        n_context: 0,
        gauss_constraints: vec![],
        constraint_const: 0.0,
    };

    let mut session = GpuFlowSession::new(config).unwrap();

    let logp = vec![-1.0, -1.5, -2.0]; // 1 proc × 3 events
    let params = [2.0]; // yield = 50 * 2 = 100

    let nll = session.nll(&logp, &params).unwrap();

    // Expected: ν_tot - Σ log(ν * p(x_i)) = 100 - Σ(log(100) + logp_i)
    let nu: f64 = 100.0;
    let sum_logf: f64 = logp.iter().map(|lp| nu.ln() + lp).sum();
    let expected_nll = nu - sum_logf;

    assert!((nll - expected_nll).abs() < 1e-6, "NLL mismatch: got {nll}, expected {expected_nll}");
}

// ── EP4: f32 NLL reducer numerical accuracy vs f64 path ──────────────────

#[test]
fn test_f32_nll_matches_f64_single_process() {
    if !has_cuda() {
        eprintln!("SKIP: CUDA not available");
        return;
    }

    let n_events = 100;
    let config = FlowNllConfig {
        n_events,
        n_procs: 1,
        n_params: 0,
        n_context: 0,
        gauss_constraints: vec![],
        constraint_const: 0.0,
    };

    let mut accel = CudaFlowNllAccelerator::new(&config).unwrap();

    // Standard normal log-prob for x_i = i/10 - 5 (spread across [-5, 5]).
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    let logp_f64: Vec<f64> = (0..n_events)
        .map(|i| {
            let x = i as f64 / 10.0 - 5.0;
            -0.5 * x * x - 0.5 * ln2pi
        })
        .collect();

    let yields = [200.0f64];
    let params: [f64; 0] = [];

    // f64 host path.
    let nll_f64 = accel.nll(&logp_f64, &yields, &params).unwrap();

    // f32 device path: upload f32 data to device manually via cudarc.
    let logp_f32: Vec<f32> = logp_f64.iter().map(|&v| v as f32).collect();
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let d_logp = stream.clone_htod(&logp_f32).unwrap();
    let (raw_ptr, _guard) = d_logp.device_ptr(&stream);
    let ptr = raw_ptr as u64;

    let nll_f32 = accel.nll_device_ptr_f32(ptr, &yields, &params).unwrap();

    let rel_err = (nll_f64 - nll_f32).abs() / nll_f64.abs().max(1.0);
    eprintln!("f32 vs f64 NLL: f64={nll_f64:.10}, f32={nll_f32:.10}, rel_err={rel_err:.2e}");
    assert!(rel_err < 1e-4, "f32 NLL deviates too much from f64: rel_err={rel_err:.2e}");
}

#[test]
fn test_f32_nll_matches_f64_two_processes_with_constraints() {
    if !has_cuda() {
        eprintln!("SKIP: CUDA not available");
        return;
    }

    let n_events = 50;
    let n_procs = 2;
    let config = FlowNllConfig {
        n_events,
        n_procs,
        n_params: 1,
        n_context: 0,
        gauss_constraints: vec![GpuUnbinnedGaussConstraintEntry {
            center: 0.0,
            inv_width: 1.0,
            param_idx: 0,
            _pad: 0,
        }],
        constraint_const: 0.5 * (2.0 * std::f64::consts::PI).ln(),
    };

    let mut accel = CudaFlowNllAccelerator::new(&config).unwrap();

    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    // Process 0: uniform logp
    let logp_uniform: Vec<f64> = vec![-12.0f64.ln(); n_events];
    // Process 1: standard normal
    let logp_gauss: Vec<f64> = (0..n_events)
        .map(|i| {
            let x = i as f64 / 25.0 - 1.0;
            -0.5 * x * x - 0.5 * ln2pi
        })
        .collect();

    let mut logp_f64 = Vec::with_capacity(n_procs * n_events);
    logp_f64.extend_from_slice(&logp_uniform);
    logp_f64.extend_from_slice(&logp_gauss);

    let yields = [30.0, 70.0];
    let params = [1.5]; // 1.5σ from constraint center

    // f64 host path.
    let nll_f64 = accel.nll(&logp_f64, &yields, &params).unwrap();

    // f32 device path.
    let logp_f32: Vec<f32> = logp_f64.iter().map(|&v| v as f32).collect();
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let d_logp = stream.clone_htod(&logp_f32).unwrap();
    let (raw_ptr, _guard) = d_logp.device_ptr(&stream);
    let ptr = raw_ptr as u64;

    let nll_f32 = accel.nll_device_ptr_f32(ptr, &yields, &params).unwrap();

    let rel_err = (nll_f64 - nll_f32).abs() / nll_f64.abs().max(1.0);
    eprintln!(
        "f32 vs f64 (2proc+constraint): f64={nll_f64:.10}, f32={nll_f32:.10}, rel_err={rel_err:.2e}"
    );
    assert!(rel_err < 1e-4, "f32 NLL deviates too much: rel_err={rel_err:.2e}");
}

#[test]
fn test_gpu_flow_session_nll_device_ptr_f32() {
    if !has_cuda() {
        eprintln!("SKIP: CUDA not available");
        return;
    }

    let config = GpuFlowSessionConfig {
        processes: vec![FlowProcessDesc {
            process_index: 0,
            base_yield: 50.0,
            yield_param_idx: Some(0),
            yield_is_scaled: true,
            context_param_indices: vec![],
        }],
        n_events: 5,
        n_params: 1,
        n_context: 0,
        gauss_constraints: vec![],
        constraint_const: 0.0,
    };

    let mut session = GpuFlowSession::new(config).unwrap();

    let logp_f64 = vec![-1.0, -1.5, -2.0, -0.5, -3.0];
    let params = [2.0]; // yield = 50 * 2 = 100

    // f64 host path via GpuFlowSession.
    let nll_f64 = session.nll(&logp_f64, &params).unwrap();

    // f32 device path via GpuFlowSession.
    let logp_f32: Vec<f32> = logp_f64.iter().map(|&v| v as f32).collect();
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let d_logp = stream.clone_htod(&logp_f32).unwrap();
    let (raw_ptr, _guard) = d_logp.device_ptr(&stream);
    let ptr = raw_ptr as u64;

    let nll_f32 = session.nll_device_ptr_f32(ptr, &params).unwrap();

    let rel_err = (nll_f64 - nll_f32).abs() / nll_f64.abs().max(1.0);
    eprintln!(
        "GpuFlowSession f32 vs f64: f64={nll_f64:.10}, f32={nll_f32:.10}, rel_err={rel_err:.2e}"
    );
    assert!(rel_err < 1e-4, "GpuFlowSession f32 NLL deviates too much: rel_err={rel_err:.2e}");
}

#[test]
fn test_f32_nll_extreme_logp_values() {
    if !has_cuda() {
        eprintln!("SKIP: CUDA not available");
        return;
    }

    let n_events = 10;
    let config = FlowNllConfig {
        n_events,
        n_procs: 1,
        n_params: 0,
        n_context: 0,
        gauss_constraints: vec![],
        constraint_const: 0.0,
    };

    let mut accel = CudaFlowNllAccelerator::new(&config).unwrap();

    // Extreme logp values: very negative (near tail of distribution).
    let logp_f64: Vec<f64> = (0..n_events)
        .map(|i| -10.0 - i as f64 * 2.0) // [-10, -12, ..., -28]
        .collect();

    let yields = [50.0f64];
    let params: [f64; 0] = [];

    let nll_f64 = accel.nll(&logp_f64, &yields, &params).unwrap();

    let logp_f32: Vec<f32> = logp_f64.iter().map(|&v| v as f32).collect();
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let d_logp = stream.clone_htod(&logp_f32).unwrap();
    let (raw_ptr, _guard) = d_logp.device_ptr(&stream);
    let ptr = raw_ptr as u64;

    let nll_f32 = accel.nll_device_ptr_f32(ptr, &yields, &params).unwrap();

    let rel_err = (nll_f64 - nll_f32).abs() / nll_f64.abs().max(1.0);
    eprintln!("Extreme logp: f64={nll_f64:.10}, f32={nll_f32:.10}, rel_err={rel_err:.2e}");
    // Larger tolerance for extreme values due to f32 precision limits.
    assert!(rel_err < 1e-3, "f32 NLL with extreme logp deviates too much: rel_err={rel_err:.2e}");
}
