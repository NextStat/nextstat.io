use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ns_core::traits::{LogDensityModel, PreparedModelRef};
use ns_inference::{NutsConfig, sample_nuts};
use ns_translate::pyhf::{HistFactoryModel, Workspace};
use std::hint::black_box;
use std::time::Duration;

#[derive(Clone)]
struct NormalMeanModel {
    data: Vec<f64>,
    sigma: f64,
    prior_sigma: f64,
}

impl LogDensityModel for NormalMeanModel {
    type Prepared<'a>
        = PreparedModelRef<'a, Self>
    where
        Self: 'a;

    fn dim(&self) -> usize {
        1
    }

    fn parameter_names(&self) -> Vec<String> {
        vec!["mu".to_string()]
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        vec![(f64::NEG_INFINITY, f64::INFINITY)]
    }

    fn parameter_init(&self) -> Vec<f64> {
        vec![0.0]
    }

    fn nll(&self, params: &[f64]) -> ns_core::Result<f64> {
        let mu = params
            .first()
            .copied()
            .ok_or_else(|| ns_core::Error::Validation("expected 1 parameter (mu)".to_string()))?;
        if !self.sigma.is_finite() || self.sigma <= 0.0 {
            return Err(ns_core::Error::Validation(format!(
                "sigma must be finite and > 0, got {}",
                self.sigma
            )));
        }
        if !self.prior_sigma.is_finite() || self.prior_sigma <= 0.0 {
            return Err(ns_core::Error::Validation(format!(
                "prior_sigma must be finite and > 0, got {}",
                self.prior_sigma
            )));
        }

        let inv_var = 1.0 / (self.sigma * self.sigma);
        let mut nll = 0.0;
        for &x in &self.data {
            let r = x - mu;
            nll += 0.5 * r * r * inv_var;
        }

        // Gaussian prior (up to additive constants).
        nll += 0.5 * (mu / self.prior_sigma).powi(2);
        Ok(nll)
    }

    fn grad_nll(&self, params: &[f64]) -> ns_core::Result<Vec<f64>> {
        let mu = params
            .first()
            .copied()
            .ok_or_else(|| ns_core::Error::Validation("expected 1 parameter (mu)".to_string()))?;
        let inv_var = 1.0 / (self.sigma * self.sigma);

        let mut g = 0.0;
        for &x in &self.data {
            g += (mu - x) * inv_var;
        }
        g += mu / (self.prior_sigma * self.prior_sigma);
        Ok(vec![g])
    }

    fn prepared(&self) -> Self::Prepared<'_> {
        PreparedModelRef::new(self)
    }
}

fn load_simple_model() -> HistFactoryModel {
    let json = include_str!("../../../tests/fixtures/simple_workspace.json");
    let ws: Workspace = serde_json::from_str(json).unwrap();
    HistFactoryModel::from_workspace(&ws).unwrap()
}

fn bench_nuts_sampling(c: &mut Criterion) {
    let data: Vec<f64> = (0..200).map(|i| (((i * 17) % 2000) as f64) / 1000.0 - 1.0).collect();
    let normal = NormalMeanModel { data, sigma: 1.0, prior_sigma: 1.0 };
    let histfactory = load_simple_model();

    // Keep it short: NUTS runs include an MLE-based init step.
    let config = NutsConfig { max_treedepth: 6, target_accept: 0.8, ..Default::default() };

    let mut group = c.benchmark_group("nuts_sample");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for (w, s) in [(10usize, 10usize), (20, 20)] {
        group.bench_with_input(
            BenchmarkId::new("normal_mean", format!("w{}_s{}", w, s)),
            &(w, s),
            |b, &(w, s)| {
                b.iter(|| {
                    let chain = sample_nuts(black_box(&normal), w, s, 42, config.clone()).unwrap();
                    black_box(chain.draws_constrained.len())
                })
            },
        );
    }

    for (w, s) in [(5usize, 5usize), (10, 10)] {
        group.bench_with_input(
            BenchmarkId::new("histfactory_simple", format!("w{}_s{}", w, s)),
            &(w, s),
            |b, &(w, s)| {
                b.iter(|| {
                    let chain =
                        sample_nuts(black_box(&histfactory), w, s, 42, config.clone()).unwrap();
                    black_box(chain.draws_constrained.len())
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_nuts_sampling);
criterion_main!(benches);
