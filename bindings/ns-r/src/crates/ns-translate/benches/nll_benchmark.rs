use criterion::{Criterion, criterion_group, criterion_main};
use ns_translate::pyhf::{HistFactoryModel, Workspace};
use std::hint::black_box;

fn load_simple_workspace() -> Workspace {
    let json = include_str!("../../../tests/fixtures/simple_workspace.json");
    serde_json::from_str(json).unwrap()
}

fn benchmark_nll_computation(c: &mut Criterion) {
    let workspace = load_simple_workspace();
    let model = HistFactoryModel::from_workspace(&workspace).unwrap();
    let params = vec![1.0; model.n_params()];

    c.bench_function("nll_simple_workspace", |b| {
        b.iter(|| {
            let nll = model.nll(black_box(&params)).unwrap();
            black_box(nll)
        })
    });
}

fn benchmark_expected_data(c: &mut Criterion) {
    let workspace = load_simple_workspace();
    let model = HistFactoryModel::from_workspace(&workspace).unwrap();
    let params = vec![1.0; model.n_params()];

    c.bench_function("expected_data_simple_workspace", |b| {
        b.iter(|| {
            let expected = model.expected_data(black_box(&params)).unwrap();
            black_box(expected)
        })
    });
}

criterion_group!(benches, benchmark_nll_computation, benchmark_expected_data);
criterion_main!(benches);
