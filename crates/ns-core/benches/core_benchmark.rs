use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ns_core::FitResult;
use std::hint::black_box;

fn make_fit_result(n: usize) -> FitResult {
    // Build a simple positive definite covariance matrix: A*A^T + eps*I.
    // Keep it deterministic and cheap.
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = ((i + 1) as f64) * 1e-3 + ((j + 1) as f64) * 1e-4;
        }
    }

    let mut cov = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += a[i * n + k] * a[j * n + k];
            }
            cov[i * n + j] = s;
        }
        cov[i * n + i] += 1e-3; // eps*I
    }

    let params = vec![1.0; n];
    let uncs = (0..n).map(|i| cov[i * n + i].sqrt()).collect::<Vec<_>>();
    FitResult::with_covariance(params, uncs, cov, 1.0, true, 10)
}

fn bench_correlation_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("core_fitresult");

    for n in [4usize, 16, 64, 128] {
        let fr = make_fit_result(n);
        group.bench_with_input(BenchmarkId::new("correlation_full", n), &n, |b, &nn| {
            b.iter(|| {
                let mut acc = 0.0;
                for i in 0..nn {
                    for j in 0..nn {
                        acc += fr.correlation(i, j).unwrap();
                    }
                }
                black_box(acc)
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_correlation_matrix);
criterion_main!(benches);
