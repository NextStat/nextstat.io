use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ns_ad::dual::Dual;
use ns_ad::tape::Tape;
use std::hint::black_box;

fn bench_tape_build_and_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("ad_tape");

    for n_vars in [4usize, 16, 64, 256, 1024] {
        group.bench_with_input(BenchmarkId::new("build_and_backward", n_vars), &n_vars, |b, &n| {
            b.iter(|| {
                let mut t = Tape::with_capacity(n * 6);
                let vars: Vec<_> = (0..n).map(|i| t.var(1.0 + (i as f64) * 1e-3)).collect();

                // f(x) = sum_i ln(x_i^2 + 1)
                let one = t.constant(1.0);
                let mut acc = t.constant(0.0);
                for &x in &vars {
                    let x2 = t.mul(x, x);
                    let x2p1 = t.add(x2, one);
                    let ln = t.ln(x2p1);
                    acc = t.add(acc, ln);
                }

                t.backward(acc);
                // Read a couple of adjoints to keep it "used".
                black_box((t.adjoint(vars[0]), t.adjoint(vars[n / 2])));
            })
        });

        group.bench_with_input(BenchmarkId::new("reuse_tape", n_vars), &n_vars, |b, &n| {
            let mut t = Tape::with_capacity(n * 6);
            b.iter(|| {
                t.clear();
                let vars: Vec<_> = (0..n).map(|i| t.var(1.0 + (i as f64) * 1e-3)).collect();

                let one = t.constant(1.0);
                let mut acc = t.constant(0.0);
                for &x in &vars {
                    let x2 = t.mul(x, x);
                    let x2p1 = t.add(x2, one);
                    let ln = t.ln(x2p1);
                    acc = t.add(acc, ln);
                }

                t.backward(acc);
                black_box(t.adjoint(vars[0]));
            })
        });
    }

    group.finish();
}

fn bench_dual_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("ad_dual");

    for n_vars in [4usize, 16, 64, 256] {
        group.bench_with_input(
            BenchmarkId::new("forward_grad_n_evals", n_vars),
            &n_vars,
            |b, &n| {
                // Forward-mode gradient: N evaluations, each with one seeded variable.
                let x0: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 1e-3).collect();

                b.iter(|| {
                    let mut grad = vec![0.0; n];
                    for seed in 0..n {
                        let mut xs: Vec<Dual> = x0.iter().copied().map(Dual::constant).collect();
                        xs[seed].dot = 1.0;

                        // f(x) = sum_i ln(x_i^2 + 1)
                        let mut acc = Dual::constant(0.0);
                        for x in xs {
                            let term = (x * x + 1.0).ln();
                            acc = acc + term;
                        }
                        grad[seed] = acc.dot;
                    }

                    black_box(grad[0]);
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_tape_build_and_backward, bench_dual_ops);
criterion_main!(benches);
