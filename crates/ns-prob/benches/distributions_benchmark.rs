use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn bench_scalar_distributions(c: &mut Criterion) {
    let xs: Vec<f64> = (0..10_000).map(|i| (i as f64) * 0.001 - 5.0).collect();

    c.bench_function("normal_logpdf_10k", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for &x in &xs {
                acc += ns_prob::normal::logpdf(x, 0.0, 1.3).unwrap();
            }
            black_box(acc)
        })
    });

    c.bench_function("student_t_logpdf_10k", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for &x in &xs {
                acc += ns_prob::student_t::logpdf(x, 0.0, 1.3, 7.0).unwrap();
            }
            black_box(acc)
        })
    });

    let ks: Vec<u64> = (0..10_000).map(|i| (i % 30) as u64).collect();
    c.bench_function("poisson_logpmf_10k", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for &k in &ks {
                acc += ns_prob::poisson::logpmf(k, 3.2).unwrap();
            }
            black_box(acc)
        })
    });

    c.bench_function("bernoulli_logpmf_logit_10k", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for (i, &x) in xs.iter().enumerate() {
                let y: u8 = if (i & 1) == 0 { 0 } else { 1 };
                acc += ns_prob::bernoulli::logpmf_logit(y, x).unwrap();
            }
            black_box(acc)
        })
    });

    let betas: Vec<f64> = (0..10_000).map(|i| ((i as f64) + 0.5) / 10_000.0).collect();
    c.bench_function("beta_logpdf_10k", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for &x in &betas {
                acc += ns_prob::beta::logpdf(x, 2.2, 3.3).unwrap();
            }
            black_box(acc)
        })
    });
}

criterion_group!(benches, bench_scalar_distributions);
criterion_main!(benches);
