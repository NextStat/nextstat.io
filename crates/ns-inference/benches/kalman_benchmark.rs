use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nalgebra::{DMatrix, DVector};
use ns_inference::timeseries::forecast::{kalman_forecast, kalman_forecast_intervals};
use ns_inference::timeseries::kalman::{KalmanModel, kalman_filter, rts_smoother};
use std::hint::black_box;

fn make_ys_1d(n: usize) -> Vec<DVector<f64>> {
    let mut out = Vec::with_capacity(n);
    for t in 0..n {
        // Deterministic signal so runs are stable across machines.
        let y = (t as f64 * 0.01).sin() + (t as f64 * 0.001).cos();
        out.push(DVector::from_row_slice(&[y]));
    }
    out
}

fn make_ys_2d_partial_missing(n: usize) -> Vec<DVector<f64>> {
    let mut out = Vec::with_capacity(n);
    for t in 0..n {
        let mut y0 = (t as f64 * 0.01).sin();
        let mut y1 = (t as f64 * 0.02).cos();
        if t % 10 == 0 {
            y0 = f64::NAN;
        }
        if t % 15 == 0 {
            y1 = f64::NAN;
        }
        out.push(DVector::from_row_slice(&[y0, y1]));
    }
    out
}

fn bench_kalman_filter_1d(c: &mut Criterion) {
    let model = KalmanModel::local_level(0.1, 0.2, 0.0, 1.0).unwrap();

    let mut group = c.benchmark_group("timeseries/kalman_filter/1d_local_level");
    for n in [100usize, 1_000, 10_000] {
        let ys = make_ys_1d(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &ys, |b, ys| {
            b.iter(|| {
                let fr = kalman_filter(black_box(&model), black_box(ys)).unwrap();
                black_box(fr.log_likelihood);
            });
        });
    }
    group.finish();
}

fn bench_kalman_filter_2d_partial_missing(c: &mut Criterion) {
    // 2D fully decoupled model; missingness can then be reasoned about per-component.
    let model = KalmanModel::new(
        DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]), // F
        DMatrix::from_row_slice(2, 2, &[0.1, 0.0, 0.0, 0.2]), // Q
        DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]), // H
        DMatrix::from_row_slice(2, 2, &[0.3, 0.0, 0.0, 0.4]), // R
        DVector::from_row_slice(&[0.0, 0.0]),                 // m0
        DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]), // P0
    )
    .unwrap();

    let mut group = c.benchmark_group("timeseries/kalman_filter/2d_partial_missing");
    for n in [100usize, 1_000, 10_000] {
        let ys = make_ys_2d_partial_missing(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &ys, |b, ys| {
            b.iter(|| {
                let fr = kalman_filter(black_box(&model), black_box(ys)).unwrap();
                black_box(fr.log_likelihood);
            });
        });
    }
    group.finish();
}

fn bench_rts_smoother_1d(c: &mut Criterion) {
    let model = KalmanModel::local_level(0.1, 0.2, 0.0, 1.0).unwrap();
    let ys = make_ys_1d(10_000);
    let fr = kalman_filter(&model, &ys).unwrap();

    c.bench_function("timeseries/rts_smoother/1d_local_level_n=10000", |b| {
        b.iter(|| {
            let sr = rts_smoother(black_box(&model), black_box(&fr)).unwrap();
            black_box(&sr.smoothed_means);
        });
    });
}

fn bench_forecast_intervals_1d(c: &mut Criterion) {
    let model = KalmanModel::local_level(0.1, 0.2, 0.0, 1.0).unwrap();
    let ys = make_ys_1d(1_000);
    let fr = kalman_filter(&model, &ys).unwrap();

    c.bench_function("timeseries/forecast_intervals/1d_steps=100", |b| {
        b.iter(|| {
            let fc = kalman_forecast(black_box(&model), black_box(&fr), black_box(100)).unwrap();
            let iv = kalman_forecast_intervals(black_box(&fc), black_box(0.05)).unwrap();
            black_box((&iv.obs_lower, &iv.obs_upper));
        });
    });
}

criterion_group!(
    benches,
    bench_kalman_filter_1d,
    bench_kalman_filter_2d_partial_missing,
    bench_rts_smoother_1d,
    bench_forecast_intervals_1d
);
criterion_main!(benches);
