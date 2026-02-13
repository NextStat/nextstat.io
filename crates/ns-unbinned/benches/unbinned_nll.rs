//! Criterion benchmarks for unbinned NLL and gradient evaluation.
//!
//! Measures per-event throughput for common PDF types at various event counts.
//! These benchmarks establish NextStat's unbinned performance baseline for
//! comparison with RooFit, zfit, and MoreFit.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use std::sync::Arc;

use ns_core::traits::LogDensityModel;
use ns_unbinned::{
    Constraint, CrystalBallPdf, EventStore, ExponentialPdf, GaussianPdf, ObservableSpec, Parameter,
    Process, UnbinnedChannel, UnbinnedModel, YieldExpr,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_event_store(n_events: usize, lo: f64, hi: f64, seed: u64) -> EventStore {
    // Generate pseudo-uniform events using a simple LCG to avoid pulling in rand_distr
    let obs: Vec<f64> = {
        let mut state = seed;
        (0..n_events)
            .map(|_| {
                // xorshift64
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let u = (state as f64) / (u64::MAX as f64);
                lo + u * (hi - lo)
            })
            .collect()
    };

    EventStore::from_columns(
        vec![ObservableSpec::branch(String::from("x"), (lo, hi))],
        vec![(String::from("x"), obs)],
        None,
    )
    .unwrap()
}

/// Build a 1-channel, 2-process (signal Gaussian + background Exponential) model.
fn make_sig_bkg_model(n_events: usize) -> (UnbinnedModel, Vec<f64>) {
    let store = make_event_store(n_events, 100.0, 180.0, 42);

    let parameters = vec![
        Parameter { name: "mu".into(), init: 1.0, bounds: (0.0, 10.0), constraint: None },
        Parameter { name: "mu_sig".into(), init: 130.0, bounds: (100.0, 180.0), constraint: None },
        Parameter { name: "sigma_sig".into(), init: 5.0, bounds: (1.0, 30.0), constraint: None },
        Parameter { name: "lambda".into(), init: -0.02, bounds: (-1.0, 0.0), constraint: None },
        Parameter {
            name: "n_bkg".into(),
            init: n_events as f64 * 0.9,
            bounds: (0.0, n_events as f64 * 5.0),
            constraint: None,
        },
    ];

    let init: Vec<f64> = parameters.iter().map(|p| p.init).collect();

    let signal = Process {
        name: "signal".into(),
        pdf: Arc::new(GaussianPdf::new("x")),
        shape_param_indices: vec![1, 2],
        yield_expr: YieldExpr::Scaled { base_yield: n_events as f64 * 0.1, scale_index: 0 },
    };

    let background = Process {
        name: "background".into(),
        pdf: Arc::new(ExponentialPdf::new("x")),
        shape_param_indices: vec![3],
        yield_expr: YieldExpr::Parameter { index: 4 },
    };

    let channel = UnbinnedChannel {
        name: "sr".into(),
        include_in_fit: true,
        data: Arc::new(store),
        processes: vec![signal, background],
    };

    let model = UnbinnedModel::new(parameters, vec![channel], Some(0)).unwrap();
    (model, init)
}

/// Build a single-process Crystal Ball model.
fn make_crystal_ball_model(n_events: usize) -> (UnbinnedModel, Vec<f64>) {
    let store = make_event_store(n_events, 100.0, 180.0, 123);

    let parameters = vec![
        Parameter { name: "mu_cb".into(), init: 130.0, bounds: (100.0, 180.0), constraint: None },
        Parameter { name: "sigma_cb".into(), init: 5.0, bounds: (1.0, 30.0), constraint: None },
        Parameter { name: "alpha_cb".into(), init: 1.5, bounds: (0.1, 10.0), constraint: None },
        Parameter { name: "n_cb".into(), init: 3.0, bounds: (0.5, 50.0), constraint: None },
        Parameter {
            name: "n_events".into(),
            init: n_events as f64,
            bounds: (0.0, n_events as f64 * 5.0),
            constraint: None,
        },
    ];

    let init: Vec<f64> = parameters.iter().map(|p| p.init).collect();

    let proc = Process {
        name: "signal".into(),
        pdf: Arc::new(CrystalBallPdf::new("x")),
        shape_param_indices: vec![0, 1, 2, 3],
        yield_expr: YieldExpr::Parameter { index: 4 },
    };

    let channel = UnbinnedChannel {
        name: "sr".into(),
        include_in_fit: true,
        data: Arc::new(store),
        processes: vec![proc],
    };

    let model = UnbinnedModel::new(parameters, vec![channel], None).unwrap();
    (model, init)
}

/// Build a 1-channel, 2-process (signal CrystalBall + background Exponential) model.
fn make_cb_exp_model(n_events: usize) -> (UnbinnedModel, Vec<f64>) {
    let store = make_event_store(n_events, 100.0, 180.0, 55);

    let parameters = vec![
        Parameter { name: "mu".into(), init: 1.0, bounds: (0.0, 10.0), constraint: None },
        Parameter { name: "mu_cb".into(), init: 130.0, bounds: (100.0, 180.0), constraint: None },
        Parameter { name: "sigma_cb".into(), init: 5.0, bounds: (1.0, 30.0), constraint: None },
        Parameter { name: "alpha_cb".into(), init: 1.5, bounds: (0.1, 10.0), constraint: None },
        Parameter { name: "n_cb".into(), init: 5.0, bounds: (1.01, 50.0), constraint: None },
        Parameter { name: "lambda".into(), init: -0.02, bounds: (-1.0, 0.0), constraint: None },
        Parameter {
            name: "n_bkg".into(),
            init: n_events as f64 * 0.9,
            bounds: (0.0, n_events as f64 * 5.0),
            constraint: None,
        },
    ];

    let init: Vec<f64> = parameters.iter().map(|p| p.init).collect();

    let signal = Process {
        name: "signal".into(),
        pdf: Arc::new(CrystalBallPdf::new("x")),
        shape_param_indices: vec![1, 2, 3, 4],
        yield_expr: YieldExpr::Scaled { base_yield: n_events as f64 * 0.1, scale_index: 0 },
    };

    let background = Process {
        name: "background".into(),
        pdf: Arc::new(ExponentialPdf::new("x")),
        shape_param_indices: vec![5],
        yield_expr: YieldExpr::Parameter { index: 6 },
    };

    let channel = UnbinnedChannel {
        name: "sr".into(),
        include_in_fit: true,
        data: Arc::new(store),
        processes: vec![signal, background],
    };

    let model = UnbinnedModel::new(parameters, vec![channel], Some(0)).unwrap();
    (model, init)
}

/// Build a model with a constrained nuisance parameter.
fn make_constrained_model(n_events: usize) -> (UnbinnedModel, Vec<f64>) {
    let store = make_event_store(n_events, 100.0, 180.0, 77);

    let parameters = vec![
        Parameter { name: "mu".into(), init: 1.0, bounds: (0.0, 10.0), constraint: None },
        Parameter { name: "mu_sig".into(), init: 130.0, bounds: (100.0, 180.0), constraint: None },
        Parameter {
            name: "sigma_sig".into(),
            init: 5.0,
            bounds: (1.0, 30.0),
            constraint: Some(Constraint::Gaussian { mean: 5.0, sigma: 0.5 }),
        },
        Parameter {
            name: "lambda".into(),
            init: -0.02,
            bounds: (-1.0, 0.0),
            constraint: Some(Constraint::Gaussian { mean: -0.02, sigma: 0.005 }),
        },
        Parameter {
            name: "n_bkg".into(),
            init: n_events as f64 * 0.9,
            bounds: (0.0, n_events as f64 * 5.0),
            constraint: None,
        },
    ];

    let init: Vec<f64> = parameters.iter().map(|p| p.init).collect();

    let signal = Process {
        name: "signal".into(),
        pdf: Arc::new(GaussianPdf::new("x")),
        shape_param_indices: vec![1, 2],
        yield_expr: YieldExpr::Scaled { base_yield: n_events as f64 * 0.1, scale_index: 0 },
    };

    let background = Process {
        name: "background".into(),
        pdf: Arc::new(ExponentialPdf::new("x")),
        shape_param_indices: vec![3],
        yield_expr: YieldExpr::Parameter { index: 4 },
    };

    let channel = UnbinnedChannel {
        name: "sr".into(),
        include_in_fit: true,
        data: Arc::new(store),
        processes: vec![signal, background],
    };

    let model = UnbinnedModel::new(parameters, vec![channel], Some(0)).unwrap();
    (model, init)
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_nll(c: &mut Criterion) {
    let mut group = c.benchmark_group("unbinned_nll");
    group.sample_size(50);

    for &n in &[1_000, 10_000, 100_000] {
        let (model, params) = make_sig_bkg_model(n);

        group.bench_with_input(BenchmarkId::new("sig_bkg_gaussian_exp", n), &n, |b, _| {
            b.iter(|| black_box(model.nll(black_box(&params)).unwrap()))
        });

        group.bench_with_input(BenchmarkId::new("sig_bkg_gaussian_exp_generic", n), &n, |b, _| {
            b.iter(|| black_box(model.nll_generic(black_box(&params)).unwrap()))
        });
    }

    for &n in &[1_000, 10_000, 100_000] {
        let (model, params) = make_crystal_ball_model(n);

        group.bench_with_input(BenchmarkId::new("crystal_ball", n), &n, |b, _| {
            b.iter(|| black_box(model.nll(black_box(&params)).unwrap()))
        });
    }

    for &n in &[1_000, 10_000, 100_000] {
        let (model, params) = make_cb_exp_model(n);

        group.bench_with_input(BenchmarkId::new("cb_exp_fused", n), &n, |b, _| {
            b.iter(|| black_box(model.nll(black_box(&params)).unwrap()))
        });

        group.bench_with_input(BenchmarkId::new("cb_exp_generic", n), &n, |b, _| {
            b.iter(|| black_box(model.nll_generic(black_box(&params)).unwrap()))
        });
    }

    for &n in &[1_000, 10_000, 100_000] {
        let (model, params) = make_constrained_model(n);

        group.bench_with_input(BenchmarkId::new("constrained_sig_bkg", n), &n, |b, _| {
            b.iter(|| black_box(model.nll(black_box(&params)).unwrap()))
        });
    }

    group.finish();
}

fn bench_grad_nll(c: &mut Criterion) {
    let mut group = c.benchmark_group("unbinned_grad_nll");
    group.sample_size(50);

    for &n in &[1_000, 10_000, 100_000] {
        let (model, params) = make_sig_bkg_model(n);

        group.bench_with_input(BenchmarkId::new("sig_bkg_gaussian_exp", n), &n, |b, _| {
            b.iter(|| black_box(model.grad_nll(black_box(&params)).unwrap()))
        });

        group.bench_with_input(BenchmarkId::new("sig_bkg_gaussian_exp_generic", n), &n, |b, _| {
            b.iter(|| black_box(model.grad_nll_generic(black_box(&params)).unwrap()))
        });
    }

    for &n in &[1_000, 10_000, 100_000] {
        let (model, params) = make_crystal_ball_model(n);

        group.bench_with_input(BenchmarkId::new("crystal_ball", n), &n, |b, _| {
            b.iter(|| black_box(model.grad_nll(black_box(&params)).unwrap()))
        });
    }

    for &n in &[1_000, 10_000, 100_000] {
        let (model, params) = make_cb_exp_model(n);

        group.bench_with_input(BenchmarkId::new("cb_exp_fused", n), &n, |b, _| {
            b.iter(|| black_box(model.grad_nll(black_box(&params)).unwrap()))
        });

        group.bench_with_input(BenchmarkId::new("cb_exp_generic", n), &n, |b, _| {
            b.iter(|| black_box(model.grad_nll_generic(black_box(&params)).unwrap()))
        });
    }

    group.finish();
}

fn bench_mle_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("unbinned_mle_fit");
    group.sample_size(10);

    for &n in &[1_000, 10_000] {
        let (model, _) = make_sig_bkg_model(n);

        group.bench_with_input(BenchmarkId::new("sig_bkg_gaussian_exp", n), &n, |b, _| {
            let mle = ns_inference::MaximumLikelihoodEstimator::new();
            b.iter(|| {
                let result = mle.fit_minimum(black_box(&model)).unwrap();
                black_box(result)
            })
        });
    }

    for &n in &[1_000, 10_000] {
        let (model, _) = make_cb_exp_model(n);

        group.bench_with_input(BenchmarkId::new("cb_exp", n), &n, |b, _| {
            let mle = ns_inference::MaximumLikelihoodEstimator::new();
            b.iter(|| {
                let result = mle.fit_minimum(black_box(&model)).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_nll, bench_grad_nll, bench_mle_fit);
criterion_main!(benches);
