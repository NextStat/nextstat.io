use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ns_inference::{LinearRegressionModel, LogisticRegressionModel, PoissonRegressionModel};
use serde::Deserialize;
use std::hint::black_box;

#[derive(Debug, Clone, Deserialize)]
struct Fixture {
    kind: String,
    include_intercept: bool,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    offset: Option<Vec<f64>>,
    beta_hat: Vec<f64>,
}

fn load_fixture(json: &'static str) -> Fixture {
    serde_json::from_str(json).unwrap()
}

fn bench_regression_small_fixtures(c: &mut Criterion) {
    let fx_ols = load_fixture(include_str!("../../../tests/fixtures/regression/ols_small.json"));
    let fx_log = load_fixture(include_str!("../../../tests/fixtures/regression/logistic_small.json"));
    let fx_pois = load_fixture(include_str!("../../../tests/fixtures/regression/poisson_small.json"));

    let m_ols =
        LinearRegressionModel::new(fx_ols.x.clone(), fx_ols.y.clone(), fx_ols.include_intercept)
            .unwrap();
    let beta_ols = fx_ols.beta_hat.clone();

    let y_log: Vec<u8> = fx_log.y.iter().map(|&v| if v >= 0.5 { 1 } else { 0 }).collect();
    let m_log =
        LogisticRegressionModel::new(fx_log.x.clone(), y_log, fx_log.include_intercept).unwrap();
    let beta_log = fx_log.beta_hat.clone();

    let y_pois: Vec<u64> = fx_pois.y.iter().map(|&v| v.round() as u64).collect();
    let m_pois = PoissonRegressionModel::new(
        fx_pois.x.clone(),
        y_pois,
        fx_pois.include_intercept,
        fx_pois.offset.clone(),
    )
    .unwrap();
    let beta_pois = fx_pois.beta_hat.clone();

    let mut group = c.benchmark_group("regression_small");

    group.bench_function("ols_nll", |b| {
        b.iter(|| black_box(m_ols.nll(black_box(&beta_ols))).unwrap())
    });
    group.bench_function("ols_grad", |b| {
        b.iter(|| black_box(m_ols.grad_nll(black_box(&beta_ols))).unwrap())
    });

    group.bench_function("logistic_nll", |b| {
        b.iter(|| black_box(m_log.nll(black_box(&beta_log))).unwrap())
    });
    group.bench_function("logistic_grad", |b| {
        b.iter(|| black_box(m_log.grad_nll(black_box(&beta_log))).unwrap())
    });

    group.bench_function("poisson_nll", |b| {
        b.iter(|| black_box(m_pois.nll(black_box(&beta_pois))).unwrap())
    });
    group.bench_function("poisson_grad", |b| {
        b.iter(|| black_box(m_pois.grad_nll(black_box(&beta_pois))).unwrap())
    });

    group.finish();
}

fn bench_regression_large_dense(c: &mut Criterion) {
    // Large-ish dense design matrices to benchmark the core O(n*p) loops.
    let n = 20_000usize;
    let p = 12usize;
    let include_intercept = true;

    // Deterministic pseudo-data without pulling in RNG overhead into the bench loop.
    let mut x: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut y_lin: Vec<f64> = Vec::with_capacity(n);
    let mut y_log: Vec<u8> = Vec::with_capacity(n);
    let mut y_pois: Vec<u64> = Vec::with_capacity(n);

    // Params: intercept + p weights.
    let mut beta = vec![0.0; p + 1];
    beta[0] = 0.3;
    for j in 0..p {
        beta[1 + j] = (j as f64) * 0.01 - 0.05;
    }

    for i in 0..n {
        let mut row = vec![0.0; p];
        for j in 0..p {
            // Simple, deterministic pattern in [-1, 1].
            row[j] = (((i * 131 + j * 17) % 2000) as f64) / 1000.0 - 1.0;
        }
        let eta = beta[0] + row.iter().zip(beta[1..].iter()).map(|(&xj, &bj)| xj * bj).sum::<f64>();
        x.push(row);

        // Linear: no noise, just to keep deterministic.
        y_lin.push(eta);

        // Logistic: y = 1 if p > 0.5 (deterministic).
        let prob = 1.0 / (1.0 + (-eta).exp());
        y_log.push(if prob > 0.5 { 1 } else { 0 });

        // Poisson: deterministic rounding of mean (not a statistical generator).
        y_pois.push(eta.exp().round().max(0.0) as u64);
    }

    let m_lin = LinearRegressionModel::new(x.clone(), y_lin, include_intercept).unwrap();
    let m_log = LogisticRegressionModel::new(x.clone(), y_log, include_intercept).unwrap();
    let m_pois = PoissonRegressionModel::new(x, y_pois, include_intercept, None).unwrap();

    let params = beta;

    let mut group = c.benchmark_group("regression_large");
    group.bench_with_input(BenchmarkId::new("nll", "linear"), &(), |b, _| {
        b.iter(|| black_box(m_lin.nll(black_box(&params))).unwrap())
    });
    group.bench_with_input(BenchmarkId::new("grad", "linear"), &(), |b, _| {
        b.iter(|| black_box(m_lin.grad_nll(black_box(&params))).unwrap())
    });

    group.bench_with_input(BenchmarkId::new("nll", "logistic"), &(), |b, _| {
        b.iter(|| black_box(m_log.nll(black_box(&params))).unwrap())
    });
    group.bench_with_input(BenchmarkId::new("grad", "logistic"), &(), |b, _| {
        b.iter(|| black_box(m_log.grad_nll(black_box(&params))).unwrap())
    });

    group.bench_with_input(BenchmarkId::new("nll", "poisson"), &(), |b, _| {
        b.iter(|| black_box(m_pois.nll(black_box(&params))).unwrap())
    });
    group.bench_with_input(BenchmarkId::new("grad", "poisson"), &(), |b, _| {
        b.iter(|| black_box(m_pois.grad_nll(black_box(&params))).unwrap())
    });
    group.finish();
}

criterion_group!(benches, bench_regression_small_fixtures, bench_regression_large_dense);
criterion_main!(benches);
