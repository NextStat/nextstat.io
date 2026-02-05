use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ns_inference::mle::MaximumLikelihoodEstimator;
use ns_translate::pyhf::{HistFactoryModel, Workspace};
use std::hint::black_box;

fn load_simple_model() -> HistFactoryModel {
    let json = include_str!("../../../tests/fixtures/simple_workspace.json");
    let ws: Workspace = serde_json::from_str(json).unwrap();
    HistFactoryModel::from_workspace(&ws).unwrap()
}

fn load_complex_model() -> HistFactoryModel {
    let json = include_str!("../../../tests/fixtures/complex_workspace.json");
    let ws: Workspace = serde_json::from_str(json).unwrap();
    HistFactoryModel::from_workspace(&ws).unwrap()
}

fn bench_mle_fit(c: &mut Criterion) {
    let mle = MaximumLikelihoodEstimator::new();

    let simple = load_simple_model();
    let complex = load_complex_model();

    let mut group = c.benchmark_group("mle_fit");

    group.bench_function("simple_workspace", |b| {
        b.iter(|| {
            let res = mle.fit(black_box(&simple)).unwrap();
            black_box(res.nll)
        })
    });

    group.bench_function("complex_workspace", |b| {
        b.iter(|| {
            let res = mle.fit(black_box(&complex)).unwrap();
            black_box(res.nll)
        })
    });

    group.finish();
}

fn bench_gradients(c: &mut Criterion) {
    let simple = load_simple_model();
    let complex = load_complex_model();

    let simple_params = vec![1.2, 1.0, 1.0];
    let complex_params: Vec<f64> = complex.parameters().iter().map(|p| p.init + 0.01).collect();

    let mut group = c.benchmark_group("model_gradients");

    group.bench_function("simple_gradient_reverse", |b| {
        b.iter(|| black_box(simple.gradient_reverse(black_box(&simple_params))).unwrap())
    });

    group.bench_function("simple_gradient_forward", |b| {
        b.iter(|| black_box(simple.gradient_ad(black_box(&simple_params))).unwrap())
    });

    group.bench_function("complex_gradient_reverse", |b| {
        b.iter(|| black_box(complex.gradient_reverse(black_box(&complex_params))).unwrap())
    });

    group.bench_function("complex_gradient_forward", |b| {
        b.iter(|| black_box(complex.gradient_ad(black_box(&complex_params))).unwrap())
    });

    group.finish();
}

fn bench_toys(c: &mut Criterion) {
    let mle = MaximumLikelihoodEstimator::new();
    let simple = load_simple_model();
    let params: Vec<f64> = simple.parameters().iter().map(|p| p.init).collect();

    let mut group = c.benchmark_group("fit_toys");
    for n_toys in [1usize, 5, 10] {
        group.bench_with_input(BenchmarkId::new("simple", n_toys), &n_toys, |b, &n| {
            b.iter(|| {
                let results = mle.fit_toys(black_box(&simple), black_box(&params), n, 123);
                black_box(results.len())
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_mle_fit, bench_gradients, bench_toys);
criterion_main!(benches);
