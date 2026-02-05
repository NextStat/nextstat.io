//! Full Poisson NLL benchmark suite.
//!
//! Goal: isolate where time goes (log/ln vs arithmetic vs masking/branches) and
//! show the upper bound on SIMD speedups when `ln(exp)` is removed from the hot loop.
//!
//! Enable bigger sizes with: `NS_BENCH_BIG=1 cargo bench -p ns-compute --bench poisson_nll_full`

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ns_compute::simd::{
    poisson_nll_scalar, poisson_nll_scalar_sparse, poisson_nll_simd, poisson_nll_simd_sparse,
};
use statrs::function::gamma::ln_gamma;
use std::env;
use std::hint::black_box;
use wide::f64x4;

#[derive(Clone, Copy, Debug)]
enum ObsPattern {
    AllNonZero,
    HalfZero,
    AllZero,
}

struct BenchData {
    expected: Vec<f64>,
    ln_expected: Vec<f64>,
    observed: Vec<f64>,
    ln_factorials: Vec<f64>,
    mask: Vec<f64>,
}

fn make_bench_data(n: usize, pattern: ObsPattern) -> BenchData {
    let mut expected = Vec::with_capacity(n);
    let mut ln_expected = Vec::with_capacity(n);
    let mut observed = Vec::with_capacity(n);
    let mut ln_factorials = Vec::with_capacity(n);
    let mut mask = Vec::with_capacity(n);

    for i in 0..n {
        let exp = (50.0 + (i as f64 * 7.3) % 100.0).max(1e-10);
        let obs = match pattern {
            ObsPattern::AllNonZero => (exp + (i as f64 * 3.1 - 40.0)).max(1.0).round(),
            ObsPattern::HalfZero => {
                if i % 2 == 0 {
                    0.0
                } else {
                    (exp + (i as f64 * 3.1 - 40.0)).max(1.0).round()
                }
            }
            ObsPattern::AllZero => 0.0,
        };

        expected.push(exp);
        ln_expected.push(exp.ln());
        observed.push(obs);
        ln_factorials.push(ln_gamma(obs + 1.0));
        mask.push(if obs > 0.0 { 1.0 } else { 0.0 });
    }

    BenchData { expected, ln_expected, observed, ln_factorials, mask }
}

#[inline(always)]
fn poisson_nll_scalar_branchy(expected: &[f64], observed: &[f64], ln_factorials: &[f64]) -> f64 {
    let mut total = 0.0;
    for i in 0..expected.len() {
        let exp = expected[i];
        let obs = observed[i];
        if obs == 0.0 {
            total += exp;
        } else {
            total += exp - obs * exp.ln() + ln_factorials[i];
        }
    }
    total
}

#[inline(always)]
fn poisson_nll_scalar_preln(
    expected: &[f64],
    ln_expected: &[f64],
    observed: &[f64],
    ln_factorials: &[f64],
    mask: &[f64],
) -> f64 {
    let mut total = 0.0;
    for i in 0..expected.len() {
        total += expected[i] + mask[i] * (ln_factorials[i] - observed[i] * ln_expected[i]);
    }
    total
}

#[inline(always)]
fn ln_f64x4_lane_by_lane(v: f64x4) -> f64x4 {
    let arr: [f64; 4] = v.into();
    f64x4::from([arr[0].ln(), arr[1].ln(), arr[2].ln(), arr[3].ln()])
}

#[inline(always)]
fn poisson_nll_simd_preln(
    expected: &[f64],
    ln_expected: &[f64],
    observed: &[f64],
    ln_factorials: &[f64],
    mask: &[f64],
) -> f64 {
    let n = expected.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut acc = f64x4::ZERO;
    for i in 0..chunks {
        let offset = i * 4;
        let exp = f64x4::from(&expected[offset..offset + 4]);
        let ln_exp = f64x4::from(&ln_expected[offset..offset + 4]);
        let obs = f64x4::from(&observed[offset..offset + 4]);
        let lnf = f64x4::from(&ln_factorials[offset..offset + 4]);
        let m = f64x4::from(&mask[offset..offset + 4]);

        // nll = exp + mask * (lnf - obs * ln_exp)
        acc += exp + m * (lnf - obs * ln_exp);
    }

    let mut total: f64 = acc.reduce_add();
    let start = chunks * 4;
    for i in start..start + remainder {
        total += expected[i] + mask[i] * (ln_factorials[i] - observed[i] * ln_expected[i]);
    }
    total
}

#[inline(always)]
fn poisson_nll_simd_wide_ln(
    expected: &[f64],
    observed: &[f64],
    ln_factorials: &[f64],
    mask: &[f64],
) -> f64 {
    let n = expected.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut acc = f64x4::ZERO;
    for i in 0..chunks {
        let offset = i * 4;
        let exp = f64x4::from(&expected[offset..offset + 4]);
        let obs = f64x4::from(&observed[offset..offset + 4]);
        let lnf = f64x4::from(&ln_factorials[offset..offset + 4]);
        let m = f64x4::from(&mask[offset..offset + 4]);

        // NOTE: `wide::ln()` is fast but may be too inaccurate for parity.
        let ln_exp = exp.ln();
        acc += exp + m * (lnf - obs * ln_exp);
    }

    let mut total: f64 = acc.reduce_add();
    let start = chunks * 4;
    for i in start..start + remainder {
        let exp = expected[i];
        total += exp + mask[i] * (ln_factorials[i] - observed[i] * exp.ln());
    }
    total
}

fn bench_sizes() -> Vec<usize> {
    let big = env::var("NS_BENCH_BIG").ok().as_deref() == Some("1");
    if big {
        vec![4, 8, 16, 32, 64, 128, 256, 1024, 4096, 10_000, 65_536, 200_000]
    } else {
        vec![4, 16, 64, 256, 1024, 10_000]
    }
}

fn bench_poisson_nll_full(c: &mut Criterion) {
    for pattern in [ObsPattern::AllNonZero, ObsPattern::HalfZero, ObsPattern::AllZero] {
        let mut group = c.benchmark_group(format!("poisson_nll_full_{pattern:?}"));

        for n in bench_sizes() {
            let d = make_bench_data(n, pattern);

            group.bench_with_input(BenchmarkId::new("scalar_branchless", n), &n, |b, _| {
                b.iter(|| {
                    let out = poisson_nll_scalar(
                        black_box(&d.expected),
                        black_box(&d.observed),
                        black_box(&d.ln_factorials),
                        black_box(&d.mask),
                    );
                    black_box(out)
                })
            });

            group.bench_with_input(BenchmarkId::new("scalar_branchy", n), &n, |b, _| {
                b.iter(|| {
                    let out = poisson_nll_scalar_branchy(
                        black_box(&d.expected),
                        black_box(&d.observed),
                        black_box(&d.ln_factorials),
                    );
                    black_box(out)
                })
            });

            group.bench_with_input(BenchmarkId::new("simd_branchless", n), &n, |b, _| {
                b.iter(|| {
                    let out = poisson_nll_simd(
                        black_box(&d.expected),
                        black_box(&d.observed),
                        black_box(&d.ln_factorials),
                        black_box(&d.mask),
                    );
                    black_box(out)
                })
            });

            group.bench_with_input(BenchmarkId::new("scalar_sparse", n), &n, |b, _| {
                b.iter(|| {
                    let out = poisson_nll_scalar_sparse(
                        black_box(&d.expected),
                        black_box(&d.observed),
                        black_box(&d.ln_factorials),
                        black_box(&d.mask),
                    );
                    black_box(out)
                })
            });

            group.bench_with_input(BenchmarkId::new("simd_sparse", n), &n, |b, _| {
                b.iter(|| {
                    let out = poisson_nll_simd_sparse(
                        black_box(&d.expected),
                        black_box(&d.observed),
                        black_box(&d.ln_factorials),
                        black_box(&d.mask),
                    );
                    black_box(out)
                })
            });

            group.bench_with_input(BenchmarkId::new("scalar_preln", n), &n, |b, _| {
                b.iter(|| {
                    let out = poisson_nll_scalar_preln(
                        black_box(&d.expected),
                        black_box(&d.ln_expected),
                        black_box(&d.observed),
                        black_box(&d.ln_factorials),
                        black_box(&d.mask),
                    );
                    black_box(out)
                })
            });

            group.bench_with_input(BenchmarkId::new("simd_preln", n), &n, |b, _| {
                b.iter(|| {
                    let out = poisson_nll_simd_preln(
                        black_box(&d.expected),
                        black_box(&d.ln_expected),
                        black_box(&d.observed),
                        black_box(&d.ln_factorials),
                        black_box(&d.mask),
                    );
                    black_box(out)
                })
            });

            group.bench_with_input(BenchmarkId::new("simd_wide_ln", n), &n, |b, _| {
                b.iter(|| {
                    let out = poisson_nll_simd_wide_ln(
                        black_box(&d.expected),
                        black_box(&d.observed),
                        black_box(&d.ln_factorials),
                        black_box(&d.mask),
                    );
                    black_box(out)
                })
            });
        }

        group.finish();
    }
}

fn bench_log_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("log_only");

    for n in bench_sizes() {
        let d = make_bench_data(n, ObsPattern::AllNonZero);

        group.bench_with_input(BenchmarkId::new("scalar_f64_ln", n), &n, |b, _| {
            b.iter(|| {
                let mut s = 0.0;
                for &v in black_box(&d.expected) {
                    s += v.ln();
                }
                black_box(s)
            })
        });

        group.bench_with_input(BenchmarkId::new("simd_lane_by_lane", n), &n, |b, _| {
            b.iter(|| {
                let x = black_box(&d.expected);
                let chunks = x.len() / 4;
                let remainder = x.len() % 4;
                let mut acc = f64x4::ZERO;
                for i in 0..chunks {
                    let offset = i * 4;
                    let v = f64x4::from(&x[offset..offset + 4]);
                    acc += ln_f64x4_lane_by_lane(v);
                }
                let mut total = acc.reduce_add();
                for i in (chunks * 4)..(chunks * 4 + remainder) {
                    total += x[i].ln();
                }
                black_box(total)
            })
        });

        group.bench_with_input(BenchmarkId::new("simd_wide_ln", n), &n, |b, _| {
            b.iter(|| {
                let x = black_box(&d.expected);
                let chunks = x.len() / 4;
                let remainder = x.len() % 4;
                let mut acc = f64x4::ZERO;
                for i in 0..chunks {
                    let offset = i * 4;
                    let v = f64x4::from(&x[offset..offset + 4]);
                    acc += v.ln();
                }
                let mut total = acc.reduce_add();
                for i in (chunks * 4)..(chunks * 4 + remainder) {
                    total += x[i].ln();
                }
                black_box(total)
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_poisson_nll_full, bench_log_only);
criterion_main!(benches);
