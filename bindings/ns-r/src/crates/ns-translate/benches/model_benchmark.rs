use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ns_translate::pyhf::schema as sch;
use ns_translate::pyhf::{HistFactoryModel, Workspace};
use std::hint::black_box;

fn load_fixture(name: &str) -> &'static str {
    match name {
        "simple" => include_str!("../../../tests/fixtures/simple_workspace.json"),
        "complex" => include_str!("../../../tests/fixtures/complex_workspace.json"),
        _ => panic!("unknown fixture"),
    }
}

fn model_from_fixture(name: &str) -> HistFactoryModel {
    let json = load_fixture(name);
    let ws: Workspace = serde_json::from_str(json).unwrap();
    HistFactoryModel::from_workspace(&ws).unwrap()
}

fn make_synthetic_workspace(n_bins: usize) -> sch::Workspace {
    // Single channel with 2 samples:
    // - signal scaled by POI `mu`
    // - background with shapesys (gamma per bin)
    let signal = sch::Sample {
        name: "signal".to_string(),
        data: vec![5.0; n_bins],
        modifiers: vec![sch::Modifier::NormFactor { name: "mu".to_string(), data: None }],
    };
    let bkg = sch::Sample {
        name: "background".to_string(),
        data: vec![50.0; n_bins],
        modifiers: vec![sch::Modifier::ShapeSys {
            name: "uncorr_bkguncrt".to_string(),
            data: vec![5.0; n_bins],
        }],
    };

    sch::Workspace {
        channels: vec![sch::Channel { name: "c".to_string(), samples: vec![signal, bkg] }],
        observations: vec![sch::Observation { name: "c".to_string(), data: vec![53.0; n_bins] }],
        measurements: vec![sch::Measurement {
            name: "m".to_string(),
            config: sch::MeasurementConfig { poi: "mu".to_string(), parameters: vec![] },
        }],
        version: Some("1.0.0".to_string()),
    }
}

fn bench_parse_and_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("translate_parse_build");

    for fixture in ["simple", "complex"] {
        let json = load_fixture(fixture);
        group.bench_function(BenchmarkId::new("parse_workspace", fixture), |b| {
            b.iter(|| {
                let ws: Workspace = serde_json::from_str(black_box(json)).unwrap();
                black_box(ws.channels.len())
            })
        });

        group.bench_function(BenchmarkId::new("build_model", fixture), |b| {
            b.iter(|| {
                let ws: Workspace = serde_json::from_str(black_box(json)).unwrap();
                let model = HistFactoryModel::from_workspace(black_box(&ws)).unwrap();
                black_box(model.n_params())
            })
        });
    }

    group.finish();
}

fn bench_model_eval_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("translate_model_eval");

    for fixture in ["simple", "complex"] {
        let model = model_from_fixture(fixture);
        let params = model.parameters().iter().map(|p| p.init).collect::<Vec<_>>();

        group.bench_function(BenchmarkId::new("expected_data", fixture), |b| {
            b.iter(|| black_box(model.expected_data(black_box(&params))).unwrap())
        });

        group.bench_function(BenchmarkId::new("nll_generic", fixture), |b| {
            b.iter(|| black_box(model.nll(black_box(&params))).unwrap())
        });

        group.bench_function(BenchmarkId::new("prepare", fixture), |b| {
            b.iter(|| {
                let prep = model.prepare();
                drop(black_box(prep));
            })
        });

        let prep = model.prepare();
        group.bench_function(BenchmarkId::new("nll_prepared", fixture), |b| {
            b.iter(|| black_box(prep.nll(black_box(&params))).unwrap())
        });

        group.bench_function(BenchmarkId::new("gradient_reverse", fixture), |b| {
            b.iter(|| black_box(model.gradient_reverse(black_box(&params))).unwrap())
        });
    }

    group.finish();
}

fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("translate_scaling");
    // Keep this moderate so `cargo bench -- --quick` stays fast.
    let big = std::env::var("NS_BENCH_BIG").ok().as_deref() == Some("1");
    let mut sizes: Vec<usize> = vec![2, 16, 64, 256, 1024];
    if big {
        sizes.extend_from_slice(&[4096, 10_000, 65_536]);
    }
    for n_bins in sizes {
        let ws = make_synthetic_workspace(n_bins);
        let model = HistFactoryModel::from_workspace(&ws).unwrap();
        let params = vec![1.0; model.n_params()];
        let prep = model.prepare();

        group.bench_with_input(BenchmarkId::new("nll_generic_bins", n_bins), &n_bins, |b, _| {
            b.iter(|| black_box(model.nll(black_box(&params))).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("nll_prepared_bins", n_bins), &n_bins, |b, _| {
            b.iter(|| black_box(prep.nll(black_box(&params))).unwrap())
        });
    }
    group.finish();
}

criterion_group!(benches, bench_parse_and_build, bench_model_eval_paths, bench_scaling);
criterion_main!(benches);
