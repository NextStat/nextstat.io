use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ns_inference::{AsymptoticCLsContext, MaximumLikelihoodEstimator};
use ns_translate::pyhf::{HistFactoryModel, Workspace};
use std::hint::black_box;

fn load_simple_model() -> HistFactoryModel {
    let json = include_str!("../../../tests/fixtures/simple_workspace.json");
    let ws: Workspace = serde_json::from_str(json).unwrap();
    HistFactoryModel::from_workspace(&ws).unwrap()
}

fn bench_context_build(c: &mut Criterion) {
    let model = load_simple_model();
    let mle = MaximumLikelihoodEstimator::new();

    c.bench_function("hypotest_context_build_simple", |b| {
        b.iter(|| {
            let ctx = AsymptoticCLsContext::new(black_box(&mle), black_box(&model)).unwrap();
            drop(black_box(ctx));
        })
    });
}

fn bench_hypotest_single(c: &mut Criterion) {
    let model = load_simple_model();
    let mle = MaximumLikelihoodEstimator::new();
    let ctx = AsymptoticCLsContext::new(&mle, &model).unwrap();

    let mut group = c.benchmark_group("hypotest_qtilde");
    for mu in [0.0f64, 0.5, 1.0, 2.0] {
        group.bench_with_input(BenchmarkId::new("simple", mu), &mu, |b, &m| {
            b.iter(|| black_box(ctx.hypotest_qtilde(black_box(&mle), m)).unwrap())
        });
    }
    group.finish();
}

fn bench_upper_limit(c: &mut Criterion) {
    let model = load_simple_model();
    let mle = MaximumLikelihoodEstimator::new();
    let ctx = AsymptoticCLsContext::new(&mle, &model).unwrap();

    c.bench_function("upper_limit_qtilde_simple", |b| {
        b.iter(|| {
            let ul = ctx.upper_limit_qtilde(black_box(&mle), 0.05, 0.0, 5.0, 1e-4, 80).unwrap();
            black_box(ul)
        })
    });
}

criterion_group!(benches, bench_context_build, bench_hypotest_single, bench_upper_limit);
criterion_main!(benches);
