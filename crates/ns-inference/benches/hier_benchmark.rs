#![allow(clippy::needless_range_loop)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ns_core::traits::LogDensityModel;
use ns_inference::ModelBuilder;
use std::hint::black_box;

fn make_logistic_data(n: usize, p: usize) -> (Vec<Vec<f64>>, Vec<u8>) {
    let include_intercept = true;

    // Params: intercept + p weights.
    let mut beta = vec![0.0; p + if include_intercept { 1 } else { 0 }];
    beta[0] = 0.25;
    for j in 0..p {
        beta[1 + j] = (j as f64) * 0.01 - 0.02;
    }

    let mut x: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut y: Vec<u8> = Vec::with_capacity(n);

    for i in 0..n {
        let mut row = vec![0.0; p];
        for j in 0..p {
            // Deterministic pattern in [-1, 1].
            row[j] = (((i * 131 + j * 17) % 2000) as f64) / 1000.0 - 1.0;
        }

        let eta = beta[0] + row.iter().zip(beta[1..].iter()).map(|(&xj, &bj)| xj * bj).sum::<f64>();
        let prob = 1.0 / (1.0 + (-eta).exp());
        y.push(if prob > 0.5 { 1 } else { 0 });

        x.push(row);
    }

    (x, y)
}

fn bench_hier_logistic_random_intercept(c: &mut Criterion) {
    let n = 10_000usize;
    let p = 6usize;
    let include_intercept = true;
    let (x, y) = make_logistic_data(n, p);

    let mut group = c.benchmark_group("hier_logistic_random_intercept");

    for n_groups in [10usize, 100, 1_000] {
        let group_idx: Vec<usize> = (0..n).map(|i| i % n_groups).collect();
        let model = ModelBuilder::logistic_regression(x.clone(), y.clone(), include_intercept)
            .unwrap()
            .with_random_intercept(group_idx, n_groups)
            .unwrap()
            .with_random_intercept_non_centered(true)
            .unwrap()
            .build()
            .unwrap();

        let params = model.parameter_init();

        group.bench_with_input(BenchmarkId::new("nll/groups", n_groups), &params, |b, params| {
            b.iter(|| black_box(model.nll(black_box(params))).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("grad/groups", n_groups), &params, |b, params| {
            b.iter(|| black_box(model.grad_nll(black_box(params))).unwrap())
        });
    }

    group.finish();
}

fn bench_hier_logistic_correlated_intercept_slope(c: &mut Criterion) {
    let n = 2_000usize;
    let p = 6usize;
    let include_intercept = true;
    let feature_idx = 0usize;
    let (x, y) = make_logistic_data(n, p);

    let mut group = c.benchmark_group("hier_logistic_correlated_intercept_slope");

    for n_groups in [10usize, 100, 1_000] {
        let group_idx: Vec<usize> = (0..n).map(|i| i % n_groups).collect();
        let model = ModelBuilder::logistic_regression(x.clone(), y.clone(), include_intercept)
            .unwrap()
            .with_correlated_random_intercept_slope(feature_idx, group_idx, n_groups)
            .unwrap()
            .build()
            .unwrap();

        let params = model.parameter_init();

        group.bench_with_input(BenchmarkId::new("nll/groups", n_groups), &params, |b, params| {
            b.iter(|| black_box(model.nll(black_box(params))).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("grad/groups", n_groups), &params, |b, params| {
            b.iter(|| black_box(model.grad_nll(black_box(params))).unwrap())
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_hier_logistic_random_intercept,
    bench_hier_logistic_correlated_intercept_slope
);
criterion_main!(benches);
