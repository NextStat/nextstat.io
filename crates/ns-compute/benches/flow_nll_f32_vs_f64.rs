//! Benchmark: f32 device-pointer NLL vs f64 host-upload NLL for flow PDF reduction.
//!
//! Measures GPU kernel + transfer overhead for both paths across different event counts.
//!
//! Run: `cargo bench -p ns-compute --features cuda --bench flow_nll_f32_vs_f64`
//!
//! Requires NVIDIA GPU at runtime.

#![cfg(feature = "cuda")]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ns_compute::cuda_driver::{CudaContext, DevicePtr};
use ns_compute::cuda_flow_nll::{CudaFlowNllAccelerator, FlowNllConfig};
use ns_compute::unbinned_types::GpuUnbinnedGaussConstraintEntry;
use std::hint::black_box;

fn has_cuda() -> bool {
    CudaFlowNllAccelerator::is_available()
}

fn make_logp_f64(n_events: usize, n_procs: usize) -> Vec<f64> {
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    (0..n_procs * n_events)
        .map(|i| {
            let x = (i % n_events) as f64 / n_events as f64 * 10.0 - 5.0;
            -0.5 * x * x - 0.5 * ln2pi
        })
        .collect()
}

// ── Single process, no constraints ────────────────────────────────────────

fn bench_nll_f64_host(c: &mut Criterion) {
    if !has_cuda() {
        eprintln!("SKIP: CUDA not available");
        return;
    }

    let mut group = c.benchmark_group("flow_nll/f64_host");
    for &n_events in &[100, 1_000, 10_000, 100_000] {
        let config = FlowNllConfig {
            n_events,
            n_procs: 1,
            n_params: 0,
            n_context: 0,
            gauss_constraints: vec![],
            constraint_const: 0.0,
        };
        let mut accel = CudaFlowNllAccelerator::new(&config).unwrap();
        let logp = make_logp_f64(n_events, 1);
        let yields = [100.0f64];
        let params: [f64; 0] = [];

        group.bench_with_input(BenchmarkId::from_parameter(n_events), &n_events, |b, _| {
            b.iter(|| {
                black_box(accel.nll(&logp, &yields, &params).unwrap());
            });
        });
    }
    group.finish();
}

fn bench_nll_f32_device_ptr(c: &mut Criterion) {
    if !has_cuda() {
        eprintln!("SKIP: CUDA not available");
        return;
    }

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let mut group = c.benchmark_group("flow_nll/f32_device_ptr");
    for &n_events in &[100, 1_000, 10_000, 100_000] {
        let config = FlowNllConfig {
            n_events,
            n_procs: 1,
            n_params: 0,
            n_context: 0,
            gauss_constraints: vec![],
            constraint_const: 0.0,
        };
        let mut accel = CudaFlowNllAccelerator::new(&config).unwrap();

        let logp_f64 = make_logp_f64(n_events, 1);
        let logp_f32: Vec<f32> = logp_f64.iter().map(|&v| v as f32).collect();
        let d_logp = stream.clone_htod(&logp_f32).unwrap();
        let (raw_ptr, _guard) = d_logp.device_ptr(&stream);
        let ptr = raw_ptr as u64;

        let yields = [100.0f64];
        let params: [f64; 0] = [];

        group.bench_with_input(BenchmarkId::from_parameter(n_events), &n_events, |b, _| {
            b.iter(|| {
                black_box(accel.nll_device_ptr_f32(ptr, &yields, &params).unwrap());
            });
        });
    }
    group.finish();
}

// ── Two processes + Gaussian constraint ───────────────────────────────────

fn bench_nll_f64_host_2proc(c: &mut Criterion) {
    if !has_cuda() {
        return;
    }

    let mut group = c.benchmark_group("flow_nll_2proc/f64_host");
    for &n_events in &[1_000, 10_000, 100_000] {
        let config = FlowNllConfig {
            n_events,
            n_procs: 2,
            n_params: 1,
            n_context: 0,
            gauss_constraints: vec![GpuUnbinnedGaussConstraintEntry {
                center: 0.0,
                inv_width: 1.0,
                param_idx: 0,
                _pad: 0,
            }],
            constraint_const: 0.0,
        };
        let mut accel = CudaFlowNllAccelerator::new(&config).unwrap();
        let logp = make_logp_f64(n_events, 2);
        let yields = [50.0, 100.0];
        let params = [0.5];

        group.bench_with_input(BenchmarkId::from_parameter(n_events), &n_events, |b, _| {
            b.iter(|| {
                black_box(accel.nll(&logp, &yields, &params).unwrap());
            });
        });
    }
    group.finish();
}

fn bench_nll_f32_device_ptr_2proc(c: &mut Criterion) {
    if !has_cuda() {
        return;
    }

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let mut group = c.benchmark_group("flow_nll_2proc/f32_device_ptr");
    for &n_events in &[1_000, 10_000, 100_000] {
        let config = FlowNllConfig {
            n_events,
            n_procs: 2,
            n_params: 1,
            n_context: 0,
            gauss_constraints: vec![GpuUnbinnedGaussConstraintEntry {
                center: 0.0,
                inv_width: 1.0,
                param_idx: 0,
                _pad: 0,
            }],
            constraint_const: 0.0,
        };
        let mut accel = CudaFlowNllAccelerator::new(&config).unwrap();

        let logp_f64 = make_logp_f64(n_events, 2);
        let logp_f32: Vec<f32> = logp_f64.iter().map(|&v| v as f32).collect();
        let d_logp = stream.clone_htod(&logp_f32).unwrap();
        let (raw_ptr, _guard) = d_logp.device_ptr(&stream);
        let ptr = raw_ptr as u64;

        let yields = [50.0, 100.0];
        let params = [0.5];

        group.bench_with_input(BenchmarkId::from_parameter(n_events), &n_events, |b, _| {
            b.iter(|| {
                black_box(accel.nll_device_ptr_f32(ptr, &yields, &params).unwrap());
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_nll_f64_host,
    bench_nll_f32_device_ptr,
    bench_nll_f64_host_2proc,
    bench_nll_f32_device_ptr_2proc,
);
criterion_main!(benches);
