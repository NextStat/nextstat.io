//! Benchmarks for SIMD vs scalar Poisson NLL computation.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
#[cfg(all(feature = "accelerate", target_os = "macos"))]
use ns_compute::accelerate::poisson_nll_accelerate;
use ns_compute::simd::{poisson_nll_scalar, poisson_nll_simd, vec_mul_pairwise, vec_scale};
use statrs::function::gamma::ln_gamma;
use std::hint::black_box;

fn make_bench_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut expected = Vec::with_capacity(n);
    let mut observed = Vec::with_capacity(n);
    let mut ln_facts = Vec::with_capacity(n);
    let mut mask = Vec::with_capacity(n);

    for i in 0..n {
        let exp = 50.0 + (i as f64 * 7.3) % 100.0;
        let obs = (exp + (i as f64 * 3.1 - 40.0)).max(0.0).round();
        expected.push(exp.max(1e-10));
        observed.push(obs);
        ln_facts.push(ln_gamma(obs + 1.0));
        mask.push(if obs > 0.0 { 1.0 } else { 0.0 });
    }

    (expected, observed, ln_facts, mask)
}

fn bench_poisson_nll(c: &mut Criterion) {
    let mut group = c.benchmark_group("poisson_nll");

    for n in [4, 100, 1000, 10000] {
        let (exp, obs, lnf, msk) = make_bench_data(n);

        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| {
                let out = poisson_nll_scalar(
                    black_box(&exp),
                    black_box(&obs),
                    black_box(&lnf),
                    black_box(&msk),
                );
                black_box(out)
            })
        });

        group.bench_with_input(BenchmarkId::new("simd", n), &n, |b, _| {
            b.iter(|| {
                let out = poisson_nll_simd(
                    black_box(&exp),
                    black_box(&obs),
                    black_box(&lnf),
                    black_box(&msk),
                );
                black_box(out)
            })
        });

        #[cfg(all(feature = "accelerate", target_os = "macos"))]
        {
            // Pre-size thread-local scratch to avoid including first-call allocation in samples.
            let _ = poisson_nll_accelerate(&exp, &obs, &lnf, &msk);

            group.bench_with_input(BenchmarkId::new("accelerate", n), &n, |b, _| {
                b.iter(|| {
                    let out = poisson_nll_accelerate(
                        black_box(&exp),
                        black_box(&obs),
                        black_box(&lnf),
                        black_box(&msk),
                    );
                    black_box(out)
                })
            });
        }
    }

    group.finish();
}

fn bench_vector_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_kernels");

    for n in [4usize, 100, 1000, 10000] {
        let (exp, _obs, _lnf, _msk) = make_bench_data(n);
        let b = exp.clone();

        group.bench_with_input(BenchmarkId::new("vec_scale", n), &n, |bencher, _| {
            bencher.iter(|| {
                let mut x = exp.clone();
                vec_scale(black_box(&mut x), black_box(1.01));
                black_box(x[0])
            })
        });

        group.bench_with_input(BenchmarkId::new("vec_mul_pairwise", n), &n, |bencher, _| {
            bencher.iter(|| {
                let mut x = exp.clone();
                vec_mul_pairwise(black_box(&mut x), black_box(&b));
                black_box(x[0])
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_poisson_nll, bench_vector_kernels);
criterion_main!(benches);
