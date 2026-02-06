use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::{DMatrix, DVector};
use ns_inference::timeseries::em::{KalmanEmConfig, kalman_em};
use ns_inference::timeseries::kalman::KalmanModel;
use ns_inference::timeseries::simulate::kalman_simulate;
use std::hint::black_box;

const TOL: f64 = 1e-300;
const MIN_DIAG: f64 = 1e-9;

fn make_decoupled_model(dim: usize, phi: f64, q: f64, r: f64) -> KalmanModel {
    let f = DMatrix::<f64>::identity(dim, dim) * phi;
    let q = DMatrix::<f64>::identity(dim, dim) * q;
    let h = DMatrix::<f64>::identity(dim, dim);
    let r = DMatrix::<f64>::identity(dim, dim) * r;
    let m0 = DVector::<f64>::zeros(dim);
    let p0 = DMatrix::<f64>::identity(dim, dim);
    KalmanModel::new(f, q, h, r, m0, p0).unwrap()
}

fn cfg_qr(max_iter: usize) -> KalmanEmConfig {
    KalmanEmConfig {
        max_iter,
        tol: TOL,
        estimate_q: true,
        estimate_r: true,
        estimate_f: false,
        estimate_h: false,
        min_diag: MIN_DIAG,
    }
}

fn bench_kalman_em_local_level_n_scaling(c: &mut Criterion) {
    let dim = 1usize;
    let true_model = make_decoupled_model(dim, 1.0, 0.1, 0.2);
    let init_model = make_decoupled_model(dim, 1.0, 0.5, 0.5);

    // We benchmark both per-iteration (max_iter=1) and "total" time (max_iter=5).
    let mut group = c.benchmark_group("timeseries/kalman_em/local_level_d=1/n_scaling");
    for n in [100usize, 1_000, 10_000] {
        let sim = kalman_simulate(&true_model, n, 123).unwrap();
        let ys = sim.ys;

        for iters in [1usize, 5] {
            group.bench_with_input(
                BenchmarkId::new(format!("iters={}", iters), n),
                &ys,
                |b, ys| {
                    b.iter(|| {
                        let res = kalman_em(black_box(&init_model), black_box(ys), cfg_qr(iters))
                            .unwrap();
                        black_box(res.n_iter);
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_kalman_em_local_level_dim_scaling(c: &mut Criterion) {
    let n = 1_000usize;
    let mut group = c.benchmark_group("timeseries/kalman_em/local_level_n=1000/dim_scaling");

    for dim in [1usize, 4, 16] {
        let true_model = make_decoupled_model(dim, 1.0, 0.1, 0.2);
        let init_model = make_decoupled_model(dim, 1.0, 0.5, 0.5);
        let sim = kalman_simulate(&true_model, n, 456).unwrap();
        let ys = sim.ys;

        for iters in [1usize, 5] {
            group.bench_with_input(
                BenchmarkId::new(format!("iters={}", iters), dim),
                &ys,
                |b, ys| {
                    b.iter(|| {
                        let res = kalman_em(black_box(&init_model), black_box(ys), cfg_qr(iters))
                            .unwrap();
                        black_box(res.n_iter);
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_kalman_em_ar1_n_scaling(c: &mut Criterion) {
    let dim = 1usize;
    let phi = 0.9;
    let true_model = make_decoupled_model(dim, phi, 0.1, 0.2);
    let init_model = make_decoupled_model(dim, phi, 0.5, 0.5);

    let mut group = c.benchmark_group("timeseries/kalman_em/ar1_d=1/n_scaling");
    for n in [100usize, 1_000, 10_000] {
        let sim = kalman_simulate(&true_model, n, 789).unwrap();
        let ys = sim.ys;

        for iters in [1usize, 5] {
            group.bench_with_input(
                BenchmarkId::new(format!("iters={}", iters), n),
                &ys,
                |b, ys| {
                    b.iter(|| {
                        let res = kalman_em(black_box(&init_model), black_box(ys), cfg_qr(iters))
                            .unwrap();
                        black_box(res.n_iter);
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_kalman_em_ar1_dim_scaling(c: &mut Criterion) {
    let n = 1_000usize;
    let phi = 0.9;
    let mut group = c.benchmark_group("timeseries/kalman_em/ar1_n=1000/dim_scaling");

    for dim in [1usize, 4, 16] {
        let true_model = make_decoupled_model(dim, phi, 0.1, 0.2);
        let init_model = make_decoupled_model(dim, phi, 0.5, 0.5);
        let sim = kalman_simulate(&true_model, n, 101112).unwrap();
        let ys = sim.ys;

        for iters in [1usize, 5] {
            group.bench_with_input(
                BenchmarkId::new(format!("iters={}", iters), dim),
                &ys,
                |b, ys| {
                    b.iter(|| {
                        let res = kalman_em(black_box(&init_model), black_box(ys), cfg_qr(iters))
                            .unwrap();
                        black_box(res.n_iter);
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_kalman_em_local_level_n_scaling,
    bench_kalman_em_local_level_dim_scaling,
    bench_kalman_em_ar1_n_scaling,
    bench_kalman_em_ar1_dim_scaling,
);
criterion_main!(benches);
