use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ns_inference::{LinearRegressionModel, LogisticRegressionModel, MaximumLikelihoodEstimator, PoissonRegressionModel};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, Poisson};
use std::hint::black_box;
use std::time::Duration;

fn dot(row: &[f64], beta: &[f64]) -> f64 {
    debug_assert_eq!(row.len(), beta.len());
    row.iter().zip(beta.iter()).map(|(&xj, &bj)| xj * bj).sum()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn predict_linear(x: &[Vec<f64>], params: &[f64], include_intercept: bool) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    if include_intercept {
        let b0 = params[0];
        let beta = &params[1..];
        for row in x {
            out.push(b0 + dot(row, beta));
        }
    } else {
        for row in x {
            out.push(dot(row, params));
        }
    }
    out
}

fn predict_logistic_proba(x: &[Vec<f64>], params: &[f64], include_intercept: bool) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    if include_intercept {
        let b0 = params[0];
        let beta = &params[1..];
        for row in x {
            out.push(sigmoid(b0 + dot(row, beta)));
        }
    } else {
        for row in x {
            out.push(sigmoid(dot(row, params)));
        }
    }
    out
}

fn predict_poisson_mean(
    x: &[Vec<f64>],
    params: &[f64],
    include_intercept: bool,
    offset: Option<&[f64]>,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    match (include_intercept, offset) {
        (true, Some(off)) => {
            let b0 = params[0];
            let beta = &params[1..];
            for (i, row) in x.iter().enumerate() {
                out.push((b0 + dot(row, beta) + off[i]).exp());
            }
        }
        (true, None) => {
            let b0 = params[0];
            let beta = &params[1..];
            for row in x {
                out.push((b0 + dot(row, beta)).exp());
            }
        }
        (false, Some(off)) => {
            for (i, row) in x.iter().enumerate() {
                out.push((dot(row, params) + off[i]).exp());
            }
        }
        (false, None) => {
            for row in x {
                out.push(dot(row, params).exp());
            }
        }
    }
    out
}

fn make_design(n: usize, p: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    let mut x = Vec::with_capacity(n);
    for _ in 0..n {
        let mut row = vec![0.0; p];
        for v in &mut row {
            *v = rng.gen_range(-1.0..=1.0);
        }
        x.push(row);
    }
    x
}

fn make_beta(p: usize) -> Vec<f64> {
    // intercept + p weights
    let mut beta = vec![0.0; p + 1];
    beta[0] = 0.1;
    for j in 0..p {
        beta[1 + j] = (j as f64) * 0.01 - 0.05;
    }
    beta
}

fn make_linear_dataset(n: usize, p: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>, bool) {
    let include_intercept = true;
    let mut rng = StdRng::seed_from_u64(seed);
    let x = make_design(n, p, &mut rng);
    let beta = make_beta(p);
    let noise = Normal::new(0.0, 0.2).unwrap();

    let mut y = Vec::with_capacity(n);
    for row in &x {
        let eta = beta[0] + dot(row, &beta[1..]);
        y.push(eta + noise.sample(&mut rng));
    }
    (x, y, include_intercept)
}

fn make_logistic_dataset(n: usize, p: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<u8>, bool) {
    let include_intercept = true;
    let mut rng = StdRng::seed_from_u64(seed);
    let x = make_design(n, p, &mut rng);
    let beta = make_beta(p);

    let mut y = Vec::with_capacity(n);
    for row in &x {
        let eta = beta[0] + dot(row, &beta[1..]);
        let p = sigmoid(eta).clamp(1e-6, 1.0 - 1e-6);
        y.push(if rng.gen_bool(p) { 1 } else { 0 });
    }
    (x, y, include_intercept)
}

fn make_poisson_dataset(
    n: usize,
    p: usize,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<u64>, bool, Option<Vec<f64>>) {
    let include_intercept = true;
    let mut rng = StdRng::seed_from_u64(seed);
    let x = make_design(n, p, &mut rng);
    let beta = make_beta(p);

    let mut y = Vec::with_capacity(n);
    for row in &x {
        let eta = beta[0] + dot(row, &beta[1..]);
        let lambda = eta.exp().clamp(1e-6, 50.0);
        let dist = Poisson::new(lambda).unwrap();
        y.push(dist.sample(&mut rng));
    }
    (x, y, include_intercept, None)
}

fn bench_fit_predict(c: &mut Criterion) {
    let mle = MaximumLikelihoodEstimator::new();

    // Keep it short: fits include finite-diff Hessian for covariance.
    let mut fit = c.benchmark_group("glm_fit");
    fit.sample_size(10);
    fit.measurement_time(Duration::from_secs(10));

    let sizes = [
        ("small", 200usize, 10usize),
        ("medium", 2_000usize, 20usize),
    ];

    for (label, n, p) in sizes {
        let (x, y, include_intercept) = make_linear_dataset(n, p, 1);
        let model = LinearRegressionModel::new(x, y, include_intercept).unwrap();
        fit.bench_with_input(BenchmarkId::new("linear", label), &(), |b, _| {
            b.iter(|| black_box(mle.fit(black_box(&model))).unwrap())
        });

        let (x, y, include_intercept) = make_logistic_dataset(n, p, 2);
        let model = LogisticRegressionModel::new(x, y, include_intercept).unwrap();
        fit.bench_with_input(BenchmarkId::new("logistic", label), &(), |b, _| {
            b.iter(|| black_box(mle.fit(black_box(&model))).unwrap())
        });

        let (x, y, include_intercept, offset) = make_poisson_dataset(n, p, 3);
        let model = PoissonRegressionModel::new(x, y, include_intercept, offset.clone()).unwrap();
        fit.bench_with_input(BenchmarkId::new("poisson", label), &(), |b, _| {
            b.iter(|| black_box(mle.fit(black_box(&model))).unwrap())
        });
    }
    fit.finish();

    // Predict-only benches (use a single fitted set of params per size).
    let mut predict = c.benchmark_group("glm_predict");
    predict.sample_size(50);
    predict.measurement_time(Duration::from_secs(10));

    let predict_sizes = [
        ("medium", 2_000usize, 20usize),
        ("large", 20_000usize, 20usize),
    ];
    for (label, n, p) in predict_sizes {
        let (x_lin, y_lin, include_intercept) = make_linear_dataset(n, p, 11);
        let m_lin = LinearRegressionModel::new(x_lin.clone(), y_lin, include_intercept).unwrap();
        let params_lin = mle.fit(&m_lin).unwrap().parameters;
        predict.bench_with_input(BenchmarkId::new("linear", label), &(), |b, _| {
            b.iter(|| {
                black_box(predict_linear(
                    black_box(&x_lin),
                    black_box(&params_lin),
                    include_intercept,
                ))
            })
        });

        let (x_log, y_log, include_intercept) = make_logistic_dataset(n, p, 12);
        let m_log = LogisticRegressionModel::new(x_log.clone(), y_log, include_intercept).unwrap();
        let params_log = mle.fit(&m_log).unwrap().parameters;
        predict.bench_with_input(BenchmarkId::new("logistic_proba", label), &(), |b, _| {
            b.iter(|| {
                black_box(predict_logistic_proba(
                    black_box(&x_log),
                    black_box(&params_log),
                    include_intercept,
                ))
            })
        });

        let (x_pois, y_pois, include_intercept, offset) = make_poisson_dataset(n, p, 13);
        let m_pois =
            PoissonRegressionModel::new(x_pois.clone(), y_pois, include_intercept, offset.clone())
                .unwrap();
        let params_pois = mle.fit(&m_pois).unwrap().parameters;
        predict.bench_with_input(BenchmarkId::new("poisson_mean", label), &(), |b, _| {
            b.iter(|| {
                black_box(predict_poisson_mean(
                    black_box(&x_pois),
                    black_box(&params_pois),
                    include_intercept,
                    offset.as_deref(),
                ))
            })
        });
    }
    predict.finish();
}

criterion_group!(benches, bench_fit_predict);
criterion_main!(benches);
