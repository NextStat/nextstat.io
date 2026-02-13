//! Benchmark: batch toy NLL for flow PDFs on CUDA.
//!
//! Measures `CudaFlowBatchNllAccelerator::batch_nll` and `batch_nll_grad` across
//! different toy counts and event sizes.
//!
//! Run: `cargo bench -p ns-compute --features cuda --bench flow_batch_nll`
//!
//! Requires NVIDIA GPU at runtime.

#![cfg(feature = "cuda")]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ns_compute::cuda_flow_batch::{
    CudaFlowBatchNllAccelerator, FlowBatchNllConfig, FlowBatchProcessDesc,
};
use std::hint::black_box;

fn has_cuda() -> bool {
    CudaFlowBatchNllAccelerator::is_available()
}

fn make_logp_flat(n_procs: usize, total_events: usize) -> Vec<f64> {
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    (0..n_procs * total_events)
        .map(|i| {
            let x = (i % total_events) as f64 / total_events as f64 * 10.0 - 5.0;
            -0.5 * x * x - 0.5 * ln2pi
        })
        .collect()
}

fn make_toy_offsets(n_toys: usize, events_per_toy: usize) -> Vec<u32> {
    (0..=n_toys).map(|t| (t * events_per_toy) as u32).collect()
}

// ── batch_nll: vary n_toys (fixed 1000 events/toy, 1 proc, 1 param) ─────

fn bench_batch_nll_vary_toys(c: &mut Criterion) {
    if !has_cuda() {
        eprintln!("SKIP: CUDA not available");
        return;
    }

    let events_per_toy = 1_000;
    let mut group = c.benchmark_group("flow_batch_nll/vary_toys");

    for &n_toys in &[10, 100, 1_000, 5_000] {
        let total_events = n_toys * events_per_toy;
        let toy_offsets = make_toy_offsets(n_toys, events_per_toy);
        let logp = make_logp_flat(1, total_events);

        let config = FlowBatchNllConfig {
            total_events,
            n_toys,
            toy_offsets,
            processes: vec![FlowBatchProcessDesc {
                base_yield: 100.0,
                yield_param_idx: Some(0),
                yield_is_scaled: true,
            }],
            n_params: 1,
            gauss_constraints: vec![],
            constraint_const: 0.0,
        };

        let mut accel = CudaFlowBatchNllAccelerator::new(&config, &logp, 0).unwrap();
        let params_flat: Vec<f64> = vec![1.0; n_toys];

        group.bench_with_input(BenchmarkId::from_parameter(n_toys), &n_toys, |b, _| {
            b.iter(|| {
                black_box(accel.batch_nll(&params_flat).unwrap());
            });
        });
    }
    group.finish();
}

// ── batch_nll: vary events_per_toy (fixed 100 toys, 1 proc, 1 param) ────

fn bench_batch_nll_vary_events(c: &mut Criterion) {
    if !has_cuda() {
        eprintln!("SKIP: CUDA not available");
        return;
    }

    let n_toys = 100;
    let mut group = c.benchmark_group("flow_batch_nll/vary_events");

    for &events_per_toy in &[100, 1_000, 10_000, 50_000] {
        let total_events = n_toys * events_per_toy;
        let toy_offsets = make_toy_offsets(n_toys, events_per_toy);
        let logp = make_logp_flat(1, total_events);

        let config = FlowBatchNllConfig {
            total_events,
            n_toys,
            toy_offsets,
            processes: vec![FlowBatchProcessDesc {
                base_yield: 100.0,
                yield_param_idx: Some(0),
                yield_is_scaled: true,
            }],
            n_params: 1,
            gauss_constraints: vec![],
            constraint_const: 0.0,
        };

        let mut accel = CudaFlowBatchNllAccelerator::new(&config, &logp, 0).unwrap();
        let params_flat: Vec<f64> = vec![1.0; n_toys];

        group.bench_with_input(
            BenchmarkId::from_parameter(events_per_toy),
            &events_per_toy,
            |b, _| {
                b.iter(|| {
                    black_box(accel.batch_nll(&params_flat).unwrap());
                });
            },
        );
    }
    group.finish();
}

// ── batch_nll_grad: 1000 toys × 1000 events, vary n_params ──────────────

fn bench_batch_nll_grad(c: &mut Criterion) {
    if !has_cuda() {
        eprintln!("SKIP: CUDA not available");
        return;
    }

    let n_toys = 1_000;
    let events_per_toy = 1_000;
    let total_events = n_toys * events_per_toy;
    let toy_offsets = make_toy_offsets(n_toys, events_per_toy);

    let mut group = c.benchmark_group("flow_batch_nll_grad/vary_params");

    for &n_params in &[1, 2, 5] {
        let n_procs = n_params; // 1 process per param for this benchmark
        let logp = make_logp_flat(n_procs, total_events);

        let processes: Vec<FlowBatchProcessDesc> = (0..n_procs)
            .map(|p| FlowBatchProcessDesc {
                base_yield: 100.0 / n_procs as f64,
                yield_param_idx: Some(p),
                yield_is_scaled: true,
            })
            .collect();

        let config = FlowBatchNllConfig {
            total_events,
            n_toys,
            toy_offsets: toy_offsets.clone(),
            processes,
            n_params,
            gauss_constraints: vec![],
            constraint_const: 0.0,
        };

        let mut accel = CudaFlowBatchNllAccelerator::new(&config, &logp, 0).unwrap();
        let params_flat: Vec<f64> = vec![1.0; n_toys * n_params];

        group.bench_with_input(BenchmarkId::from_parameter(n_params), &n_params, |b, _| {
            b.iter(|| {
                black_box(accel.batch_nll_grad(&params_flat).unwrap());
            });
        });
    }
    group.finish();
}

// ── 2-proc model: batch NLL, 500 toys × 5000 events ─────────────────────

fn bench_batch_nll_2proc(c: &mut Criterion) {
    if !has_cuda() {
        return;
    }

    let n_toys = 500;
    let events_per_toy = 5_000;
    let total_events = n_toys * events_per_toy;
    let toy_offsets = make_toy_offsets(n_toys, events_per_toy);
    let logp = make_logp_flat(2, total_events);

    let config = FlowBatchNllConfig {
        total_events,
        n_toys,
        toy_offsets,
        processes: vec![
            FlowBatchProcessDesc {
                base_yield: 50.0,
                yield_param_idx: Some(0),
                yield_is_scaled: true,
            },
            FlowBatchProcessDesc {
                base_yield: 200.0,
                yield_param_idx: None,
                yield_is_scaled: false,
            },
        ],
        n_params: 1,
        gauss_constraints: vec![],
        constraint_const: 0.0,
    };

    let mut accel = CudaFlowBatchNllAccelerator::new(&config, &logp, 0).unwrap();
    let params_flat: Vec<f64> = vec![1.0; n_toys];

    let mut group = c.benchmark_group("flow_batch_nll_2proc");
    group.bench_function("500toys_5kevents", |b| {
        b.iter(|| {
            black_box(accel.batch_nll(&params_flat).unwrap());
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_batch_nll_vary_toys,
    bench_batch_nll_vary_events,
    bench_batch_nll_grad,
    bench_batch_nll_2proc,
);
criterion_main!(benches);
