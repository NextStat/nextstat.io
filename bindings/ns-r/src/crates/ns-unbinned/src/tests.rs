use crate::HistoSysInterpCode;
use crate::event_store::{EventStore, ObservableSpec};
use crate::model::{Parameter, Process, RateModifier, UnbinnedChannel, UnbinnedModel, YieldExpr};
use crate::pdf::{
    ChebyshevPdf, CrystalBallPdf, DoubleCrystalBallPdf, ExponentialPdf, GaussianPdf, HistogramPdf,
    HistogramSystematic, HorizontalMorphingKdePdf, KdeHorizontalSystematic, KdePdf,
    KdeWeightSystematic, MorphingHistogramPdf, MorphingKdePdf, UnbinnedPdf,
};
use approx::assert_relative_eq;
use ns_core::traits::LogDensityModel;
use ns_inference::mle::MaximumLikelihoodEstimator;
use rand::prelude::*;
use rand_distr::Normal;
use std::sync::Arc;

fn finite_diff_grad_vec<F: Fn(&[f64]) -> Vec<f64>>(params: &[f64], eps: f64, eval: F) -> Vec<f64> {
    let n_params = params.len();
    let base = eval(params);
    let n_out = base.len();

    let mut grad = vec![0.0f64; n_out * n_params];
    for j in 0..n_params {
        let mut p_plus = params.to_vec();
        let mut p_minus = params.to_vec();
        p_plus[j] += eps;
        p_minus[j] -= eps;

        let y_plus = eval(&p_plus);
        let y_minus = eval(&p_minus);
        assert_eq!(y_plus.len(), n_out);
        assert_eq!(y_minus.len(), n_out);

        for i in 0..n_out {
            grad[i * n_params + j] = (y_plus[i] - y_minus[i]) / (2.0 * eps);
        }
    }
    grad
}

#[test]
fn test_gaussian_pdf_grad_matches_finite_difference() {
    let obs = ObservableSpec::branch("x", (0.0, 10.0));
    let xs = vec![1.0, 2.5, 7.7, 9.9];
    let store = EventStore::from_columns(vec![obs], vec![("x".to_string(), xs)], None).unwrap();

    let pdf = GaussianPdf::new("x");
    let params = [5.0, 1.4];
    let n = store.n_events();

    let mut logp = vec![0.0f64; n];
    let mut grad = vec![0.0f64; n * 2];
    pdf.log_prob_grad_batch(&store, &params, &mut logp, &mut grad).unwrap();

    let fd = finite_diff_grad_vec(&params, 1e-6, |p| {
        let mut lp = vec![0.0f64; n];
        pdf.log_prob_batch(&store, p, &mut lp).unwrap();
        lp
    });

    for i in 0..n {
        assert_relative_eq!(grad[i * 2], fd[i * 2], epsilon = 1e-5);
        assert_relative_eq!(grad[i * 2 + 1], fd[i * 2 + 1], epsilon = 1e-5);
    }
}

#[test]
fn test_exponential_pdf_grad_matches_finite_difference() {
    let obs = ObservableSpec::branch("x", (0.0, 10.0));
    let xs = vec![0.1, 1.0, 3.3, 9.8];
    let store = EventStore::from_columns(vec![obs], vec![("x".to_string(), xs)], None).unwrap();

    let pdf = ExponentialPdf::new("x");
    let params = [-0.2];
    let n = store.n_events();

    let mut logp = vec![0.0f64; n];
    let mut grad = vec![0.0f64; n];
    pdf.log_prob_grad_batch(&store, &params, &mut logp, &mut grad).unwrap();

    let fd = finite_diff_grad_vec(&params, 1e-6, |p| {
        let mut lp = vec![0.0f64; n];
        pdf.log_prob_batch(&store, p, &mut lp).unwrap();
        lp
    });

    for i in 0..n {
        assert_relative_eq!(grad[i], fd[i], epsilon = 1e-5);
    }
}

#[test]
fn test_crystal_ball_pdf_grad_matches_finite_difference() {
    let obs = ObservableSpec::branch("x", (0.0, 10.0));
    // Mix points in tail and core. Boundary at mu - alpha*sigma = 3.5.
    let xs = vec![1.0, 2.0, 4.0, 5.0, 8.0];
    let store = EventStore::from_columns(vec![obs], vec![("x".to_string(), xs)], None).unwrap();

    let pdf = CrystalBallPdf::new("x");
    let params = [5.0, 1.0, 1.5, 3.2];
    let n = store.n_events();

    let mut logp = vec![0.0f64; n];
    let mut grad = vec![0.0f64; n * 4];
    pdf.log_prob_grad_batch(&store, &params, &mut logp, &mut grad).unwrap();

    let fd = finite_diff_grad_vec(&params, 1e-6, |p| {
        let mut lp = vec![0.0f64; n];
        pdf.log_prob_batch(&store, p, &mut lp).unwrap();
        lp
    });

    for i in 0..n {
        for j in 0..4 {
            assert_relative_eq!(grad[i * 4 + j], fd[i * 4 + j], epsilon = 5e-5);
        }
    }
}

#[test]
fn test_double_crystal_ball_pdf_grad_matches_finite_difference() {
    let obs = ObservableSpec::branch("x", (0.0, 10.0));
    // Left tail boundary: 3.8, right tail boundary: 6.8.
    let xs = vec![1.0, 3.0, 4.5, 7.2, 9.0];
    let store = EventStore::from_columns(vec![obs], vec![("x".to_string(), xs)], None).unwrap();

    let pdf = DoubleCrystalBallPdf::new("x");
    let params = [5.0, 1.0, 1.2, 2.3, 1.8, 3.5];
    let n = store.n_events();

    let mut logp = vec![0.0f64; n];
    let mut grad = vec![0.0f64; n * 6];
    pdf.log_prob_grad_batch(&store, &params, &mut logp, &mut grad).unwrap();

    let fd = finite_diff_grad_vec(&params, 1e-6, |p| {
        let mut lp = vec![0.0f64; n];
        pdf.log_prob_batch(&store, p, &mut lp).unwrap();
        lp
    });

    for i in 0..n {
        for j in 0..6 {
            assert_relative_eq!(grad[i * 6 + j], fd[i * 6 + j], epsilon = 5e-5);
        }
    }
}

#[test]
fn test_chebyshev_pdf_grad_matches_finite_difference() {
    let obs = ObservableSpec::branch("x", (0.0, 1.0));
    let xs = vec![0.1, 0.5, 0.9];
    let store = EventStore::from_columns(vec![obs], vec![("x".to_string(), xs)], None).unwrap();

    let pdf = ChebyshevPdf::new("x", 3).unwrap();
    let params = [0.1, -0.05, 0.02];
    let n = store.n_events();

    let mut logp = vec![0.0f64; n];
    let mut grad = vec![0.0f64; n * 3];
    pdf.log_prob_grad_batch(&store, &params, &mut logp, &mut grad).unwrap();

    let fd = finite_diff_grad_vec(&params, 1e-6, |p| {
        let mut lp = vec![0.0f64; n];
        pdf.log_prob_batch(&store, p, &mut lp).unwrap();
        lp
    });

    for i in 0..n {
        for j in 0..3 {
            assert_relative_eq!(grad[i * 3 + j], fd[i * 3 + j], epsilon = 5e-5);
        }
    }
}

#[test]
fn test_histogram_pdf_uniform_log_prob() {
    let obs = ObservableSpec::branch("x", (0.0, 2.0));
    let xs = vec![0.1, 0.9, 1.1, 1.9, 2.0];
    let store = EventStore::from_columns(vec![obs], vec![("x".to_string(), xs)], None).unwrap();

    // Two equal bins -> uniform density 0.5 on [0,2].
    let pdf = HistogramPdf::from_edges_and_contents("x", vec![0.0, 1.0, 2.0], vec![1.0, 1.0], 0.0)
        .unwrap();

    let mut logp = vec![0.0f64; store.n_events()];
    pdf.log_prob_batch(&store, &[], &mut logp).unwrap();

    for &lp in &logp {
        assert_relative_eq!(lp, (0.5f64).ln(), epsilon = 1e-12);
    }

    // Gradient API: n_params=0 => empty grad buffer.
    let mut out_grad = Vec::new();
    pdf.log_prob_grad_batch(&store, &[], &mut logp, &mut out_grad).unwrap();
    assert!(out_grad.is_empty());
}

#[test]
fn test_kde_pdf_single_kernel_is_normalized_on_support() {
    let bounds = (0.0, 1.0);
    let obs = ObservableSpec::branch("x", bounds);

    // Evaluate on a fine grid to approximate integral.
    let n_grid = 2000usize;
    let dx = (bounds.1 - bounds.0) / (n_grid as f64);
    let xs: Vec<f64> = (0..n_grid).map(|i| bounds.0 + (i as f64 + 0.5) * dx).collect();
    let store = EventStore::from_columns(vec![obs], vec![("x".to_string(), xs)], None).unwrap();

    let pdf = KdePdf::from_samples("x", bounds, vec![0.3], None, 0.07).unwrap();
    let mut logp = vec![0.0f64; store.n_events()];
    pdf.log_prob_batch(&store, &[], &mut logp).unwrap();

    let mut integral = 0.0f64;
    for &lp in &logp {
        integral += lp.exp() * dx;
    }
    assert_relative_eq!(integral, 1.0, epsilon = 2e-3);
}

#[test]
fn test_unbinned_model_yield_grad_works_with_histogram_pdf() {
    let obs = ObservableSpec::branch("x", (0.0, 2.0));
    let xs = vec![0.1, 0.9, 1.1, 1.9, 2.0];
    let n = xs.len();
    let store =
        Arc::new(EventStore::from_columns(vec![obs], vec![("x".to_string(), xs)], None).unwrap());

    let params = vec![Parameter {
        name: "nu".into(),
        init: n as f64,
        bounds: (0.0, 100.0),
        constraint: None,
    }];
    let pdf: Arc<dyn UnbinnedPdf> = Arc::new(
        HistogramPdf::from_edges_and_contents("x", vec![0.0, 1.0, 2.0], vec![1.0, 1.0], 0.0)
            .unwrap(),
    );

    let channel = UnbinnedChannel {
        name: "SR".into(),
        include_in_fit: true,
        data: store,
        processes: vec![Process {
            name: "p".into(),
            pdf,
            shape_param_indices: vec![],
            yield_expr: YieldExpr::Parameter { index: 0 },
        }],
    };

    let model = UnbinnedModel::new(params, vec![channel], None).unwrap();

    let nu = n as f64;
    let logp = (0.5f64).ln();
    let expected = nu - (n as f64) * nu.ln() - (n as f64) * logp;
    let got = model.nll(&[nu]).unwrap();
    assert_relative_eq!(got, expected, epsilon = 1e-10);

    // For a single-process extended likelihood: d/dnu = 1 - N/nu => 0 at nu=N.
    let grad = model.grad_nll(&[nu]).unwrap();
    assert_eq!(grad.len(), 1);
    assert_relative_eq!(grad[0], 0.0, epsilon = 1e-10);
}

fn sample_bounded_exp<R: Rng>(rng: &mut R, lambda: f64, a: f64, b: f64) -> f64 {
    let u: f64 = rng.random();
    if lambda.abs() < 1e-12 {
        return a + (b - a) * u;
    }
    let ea = (lambda * a).exp();
    let eb = (lambda * b).exp();
    let x = (ea + u * (eb - ea)).ln() / lambda;
    x.clamp(a, b)
}

#[test]
fn test_unbinned_poisson_toy_reproducible() {
    let bounds = (0.0, 10.0);
    let obs = ObservableSpec::branch("x", bounds);
    let store = Arc::new(
        EventStore::from_columns(vec![obs], vec![("x".to_string(), Vec::new())], None).unwrap(),
    );

    let params = vec![
        Parameter { name: "gauss_mu".into(), init: 5.0, bounds: (5.0, 5.0), constraint: None },
        Parameter { name: "gauss_sigma".into(), init: 1.0, bounds: (1.0, 1.0), constraint: None },
    ];
    let pdf: Arc<dyn UnbinnedPdf> = Arc::new(GaussianPdf::new("x"));

    let channel = UnbinnedChannel {
        name: "SR".into(),
        include_in_fit: true,
        data: store,
        processes: vec![Process {
            name: "p".into(),
            pdf,
            shape_param_indices: vec![0, 1],
            yield_expr: YieldExpr::Fixed(100.0),
        }],
    };
    let model = UnbinnedModel::new(params, vec![channel], None).unwrap();

    let p = vec![5.0, 1.0];
    let toy1 = model.sample_poisson_toy(&p, 123).unwrap();
    let toy2 = model.sample_poisson_toy(&p, 123).unwrap();

    let x1 = toy1.channels()[0].data.column("x").unwrap().to_vec();
    let x2 = toy2.channels()[0].data.column("x").unwrap().to_vec();
    assert_eq!(x1, x2, "toy sampling must be deterministic for a fixed seed");

    // Basic bounds sanity.
    for &x in &x1 {
        assert!(x >= bounds.0 && x <= bounds.1);
    }
}

#[test]
fn test_normsys_yield_modifier_grad_matches_finite_difference() {
    let obs = ObservableSpec::branch("x", (0.0, 2.0));
    let xs = vec![0.1, 0.9, 1.1, 1.9, 2.0];
    let n = xs.len();
    let store =
        Arc::new(EventStore::from_columns(vec![obs], vec![("x".to_string(), xs)], None).unwrap());

    // Parameters: [nu0, alpha]
    let params = vec![
        Parameter { name: "nu0".into(), init: 50.0, bounds: (0.0, 1_000.0), constraint: None },
        Parameter {
            name: "alpha".into(),
            init: 0.3,
            bounds: (-5.0, 5.0),
            constraint: Some(crate::model::Constraint::Gaussian { mean: 0.0, sigma: 1.0 }),
        },
    ];

    let pdf: Arc<dyn UnbinnedPdf> = Arc::new(
        HistogramPdf::from_edges_and_contents("x", vec![0.0, 1.0, 2.0], vec![1.0, 1.0], 0.0)
            .unwrap(),
    );

    let channel = UnbinnedChannel {
        name: "SR".into(),
        include_in_fit: true,
        data: store,
        processes: vec![Process {
            name: "p".into(),
            pdf,
            shape_param_indices: vec![],
            yield_expr: YieldExpr::Modified {
                base: Box::new(YieldExpr::Parameter { index: 0 }),
                modifiers: vec![RateModifier::NormSys { alpha_index: 1, lo: 0.9, hi: 1.1 }],
            },
        }],
    };

    let model = UnbinnedModel::new(params, vec![channel], None).unwrap();

    let p0 = vec![50.0, 0.7];
    let grad = model.grad_nll(&p0).unwrap();

    let eps = 1e-6;
    let mut fd = vec![0.0f64; p0.len()];
    for j in 0..p0.len() {
        let mut p_plus = p0.clone();
        let mut p_minus = p0.clone();
        p_plus[j] += eps;
        p_minus[j] -= eps;
        let y_plus = model.nll(&p_plus).unwrap();
        let y_minus = model.nll(&p_minus).unwrap();
        fd[j] = (y_plus - y_minus) / (2.0 * eps);
    }

    assert_eq!(grad.len(), p0.len());
    for j in 0..p0.len() {
        assert_relative_eq!(grad[j], fd[j], epsilon = 5e-6);
    }

    // Spot-check behavior for alpha < 0 as well.
    let p1 = vec![50.0, -0.7];
    let grad1 = model.grad_nll(&p1).unwrap();
    for j in 0..p1.len() {
        let mut p_plus = p1.clone();
        let mut p_minus = p1.clone();
        p_plus[j] += eps;
        p_minus[j] -= eps;
        let y_plus = model.nll(&p_plus).unwrap();
        let y_minus = model.nll(&p_minus).unwrap();
        let fdj = (y_plus - y_minus) / (2.0 * eps);
        assert_relative_eq!(grad1[j], fdj, epsilon = 5e-6);
    }

    // Ensure the modified yield is positive (sanity).
    let nu0 = p0[0];
    let alpha = p0[1];
    let f = 1.1f64.powf(alpha);
    assert!(nu0 * f > n as f64);
}

#[test]
fn test_weightsys_yield_modifier_grad_matches_finite_difference() {
    let obs = ObservableSpec::branch("x", (0.0, 2.0));
    let xs = vec![0.1, 0.9, 1.1, 1.9, 2.0];
    let store =
        Arc::new(EventStore::from_columns(vec![obs], vec![("x".to_string(), xs)], None).unwrap());

    // Parameters: [nu0, alpha]
    let params = vec![
        Parameter { name: "nu0".into(), init: 50.0, bounds: (0.0, 1_000.0), constraint: None },
        Parameter {
            name: "alpha".into(),
            init: 0.3,
            bounds: (-5.0, 5.0),
            constraint: Some(crate::model::Constraint::Gaussian { mean: 0.0, sigma: 1.0 }),
        },
    ];

    let pdf: Arc<dyn UnbinnedPdf> = Arc::new(
        HistogramPdf::from_edges_and_contents("x", vec![0.0, 1.0, 2.0], vec![1.0, 1.0], 0.0)
            .unwrap(),
    );

    for interp_code in [HistoSysInterpCode::Code0, HistoSysInterpCode::Code4p] {
        let channel = UnbinnedChannel {
            name: "SR".into(),
            include_in_fit: true,
            data: store.clone(),
            processes: vec![Process {
                name: "p".into(),
                pdf: pdf.clone(),
                shape_param_indices: vec![],
                yield_expr: YieldExpr::Modified {
                    base: Box::new(YieldExpr::Parameter { index: 0 }),
                    modifiers: vec![RateModifier::WeightSys {
                        alpha_index: 1,
                        lo: 0.9,
                        hi: 1.1,
                        interp_code,
                    }],
                },
            }],
        };

        let model = UnbinnedModel::new(params.clone(), vec![channel], None).unwrap();

        let p0 = vec![50.0, 0.7];
        let grad = model.grad_nll(&p0).unwrap();

        let eps = 1e-6;
        let mut fd = vec![0.0f64; p0.len()];
        for j in 0..p0.len() {
            let mut p_plus = p0.clone();
            let mut p_minus = p0.clone();
            p_plus[j] += eps;
            p_minus[j] -= eps;
            let y_plus = model.nll(&p_plus).unwrap();
            let y_minus = model.nll(&p_minus).unwrap();
            fd[j] = (y_plus - y_minus) / (2.0 * eps);
        }

        assert_eq!(grad.len(), p0.len());
        for j in 0..p0.len() {
            assert_relative_eq!(grad[j], fd[j], epsilon = 5e-6);
        }
    }
}

#[test]
fn test_morphing_histogram_pdf_grad_matches_finite_difference() {
    let obs = ObservableSpec::branch("x", (0.0, 2.0));
    let xs = vec![0.1, 0.9, 1.1, 1.9, 2.0];
    let store = EventStore::from_columns(vec![obs], vec![("x".to_string(), xs)], None).unwrap();

    // Two bins with a single HistFactory-like systematic.
    let pdf = MorphingHistogramPdf::new(
        "x",
        vec![0.0, 1.0, 2.0],
        vec![10.0, 20.0],
        vec![HistogramSystematic {
            down: vec![9.0, 22.0],
            up: vec![12.0, 18.0],
            interp_code: HistoSysInterpCode::Code4p,
        }],
        0.0,
    )
    .unwrap();

    let params = [0.4];
    let n = store.n_events();

    let mut logp = vec![0.0f64; n];
    let mut grad = vec![0.0f64; n];
    pdf.log_prob_grad_batch(&store, &params, &mut logp, &mut grad).unwrap();

    let fd = finite_diff_grad_vec(&params, 1e-6, |p| {
        let mut lp = vec![0.0f64; n];
        pdf.log_prob_batch(&store, p, &mut lp).unwrap();
        lp
    });

    for i in 0..n {
        assert_relative_eq!(grad[i], fd[i], epsilon = 5e-6);
    }
}

#[test]
fn test_morphing_kde_pdf_grad_matches_finite_difference() {
    let bounds = (0.0, 1.0);
    let obs = ObservableSpec::branch("x", bounds);
    let xs = vec![0.05, 0.2, 0.5, 0.9, 1.0];
    let store = EventStore::from_columns(vec![obs], vec![("x".to_string(), xs)], None).unwrap();

    let pdf = MorphingKdePdf::new(
        "x",
        bounds,
        vec![0.3, 0.7],
        vec![1.0, 2.0],
        vec![KdeWeightSystematic {
            down: vec![0.9, 2.2],
            up: vec![1.3, 1.7],
            interp_code: HistoSysInterpCode::Code4p,
        }],
        0.08,
    )
    .unwrap();

    let params = [0.4];
    let n = store.n_events();

    let mut logp = vec![0.0f64; n];
    let mut grad = vec![0.0f64; n];
    pdf.log_prob_grad_batch(&store, &params, &mut logp, &mut grad).unwrap();

    let fd = finite_diff_grad_vec(&params, 1e-6, |p| {
        let mut lp = vec![0.0f64; n];
        pdf.log_prob_batch(&store, p, &mut lp).unwrap();
        lp
    });

    for i in 0..n {
        assert_relative_eq!(grad[i], fd[i], epsilon = 1e-5);
    }
}

#[test]
fn test_horizontal_morphing_kde_pdf_grad_matches_finite_difference() {
    let bounds = (0.0, 1.0);
    let obs = ObservableSpec::branch("x", bounds);
    let xs = vec![0.05, 0.2, 0.5, 0.9, 1.0];
    let store = EventStore::from_columns(vec![obs], vec![("x".to_string(), xs)], None).unwrap();

    let pdf = HorizontalMorphingKdePdf::new(
        "x",
        bounds,
        vec![0.3, 0.7],
        vec![1.0, 2.0],
        vec![],
        vec![KdeHorizontalSystematic {
            down: vec![0.28, 0.68],
            up: vec![0.32, 0.72],
            interp_code: HistoSysInterpCode::Code4p,
        }],
        0.08,
    )
    .unwrap();

    let params = [0.4];
    let n = store.n_events();

    let mut logp = vec![0.0f64; n];
    let mut grad = vec![0.0f64; n];
    pdf.log_prob_grad_batch(&store, &params, &mut logp, &mut grad).unwrap();

    let fd = finite_diff_grad_vec(&params, 1e-6, |p| {
        let mut lp = vec![0.0f64; n];
        pdf.log_prob_batch(&store, p, &mut lp).unwrap();
        lp
    });

    for i in 0..n {
        assert_relative_eq!(grad[i], fd[i], epsilon = 2e-5);
    }
}

#[test]
fn test_horizontal_morphing_kde_pdf_grad_matches_finite_difference_with_weightsys() {
    let bounds = (0.0, 1.0);
    let obs = ObservableSpec::branch("x", bounds);
    let xs = vec![0.05, 0.2, 0.5, 0.9, 1.0];
    let store = EventStore::from_columns(vec![obs], vec![("x".to_string(), xs)], None).unwrap();

    let pdf = HorizontalMorphingKdePdf::new(
        "x",
        bounds,
        vec![0.3, 0.7],
        vec![1.0, 2.0],
        vec![KdeWeightSystematic {
            down: vec![0.9, 2.2],
            up: vec![1.3, 1.7],
            interp_code: HistoSysInterpCode::Code4p,
        }],
        vec![KdeHorizontalSystematic {
            down: vec![0.28, 0.68],
            up: vec![0.32, 0.72],
            interp_code: HistoSysInterpCode::Code4p,
        }],
        0.08,
    )
    .unwrap();

    let params = [0.4, -0.3];
    let n = store.n_events();

    let mut logp = vec![0.0f64; n];
    let mut grad = vec![0.0f64; n * params.len()];
    pdf.log_prob_grad_batch(&store, &params, &mut logp, &mut grad).unwrap();

    let fd = finite_diff_grad_vec(&params, 1e-6, |p| {
        let mut lp = vec![0.0f64; n];
        pdf.log_prob_batch(&store, p, &mut lp).unwrap();
        lp
    });

    for i in 0..n {
        for j in 0..params.len() {
            assert_relative_eq!(
                grad[i * params.len() + j],
                fd[i * params.len() + j],
                epsilon = 2e-5
            );
        }
    }
}

#[test]
fn test_unbinned_mle_recovers_signal_strength() {
    let mut rng = StdRng::seed_from_u64(7);

    let bounds = (0.0, 10.0);
    let obs = ObservableSpec::branch("x", bounds);

    // Truth.
    let mu_true: f64 = 1.2;
    let s0: f64 = 50.0;
    let b_true: f64 = 200.0;
    let gauss_mu_true: f64 = 5.2;
    let gauss_sigma_true: f64 = 0.9;
    let lambda_bkg_true: f64 = -0.25;

    let n_sig = (mu_true * s0).round() as usize;
    let n_bkg = b_true.round() as usize;

    let normal = Normal::new(gauss_mu_true, gauss_sigma_true).unwrap();
    let mut xs = Vec::with_capacity(n_sig + n_bkg);
    for _ in 0..n_sig {
        // Rejection sample into bounds.
        let x = loop {
            let v = rng.sample(normal);
            if v >= bounds.0 && v <= bounds.1 {
                break v;
            }
        };
        xs.push(x);
    }
    for _ in 0..n_bkg {
        xs.push(sample_bounded_exp(&mut rng, lambda_bkg_true, bounds.0, bounds.1));
    }
    xs.shuffle(&mut rng);

    let store =
        Arc::new(EventStore::from_columns(vec![obs], vec![("x".to_string(), xs)], None).unwrap());

    // Parameters: [mu, gauss_mu, gauss_sigma, lambda_bkg, nu_bkg]
    let params = vec![
        Parameter { name: "mu".into(), init: 1.0, bounds: (0.0, 5.0), constraint: None },
        Parameter { name: "gauss_mu".into(), init: 5.0, bounds: (0.0, 10.0), constraint: None },
        Parameter { name: "gauss_sigma".into(), init: 1.2, bounds: (0.1, 5.0), constraint: None },
        Parameter { name: "lambda_bkg".into(), init: -0.1, bounds: (-2.0, 2.0), constraint: None },
        Parameter { name: "nu_bkg".into(), init: 150.0, bounds: (0.0, 500.0), constraint: None },
    ];

    let sig_pdf: Arc<dyn UnbinnedPdf> = Arc::new(GaussianPdf::new("x"));
    let bkg_pdf: Arc<dyn UnbinnedPdf> = Arc::new(ExponentialPdf::new("x"));

    let channel = UnbinnedChannel {
        name: "SR".into(),
        include_in_fit: true,
        data: store,
        processes: vec![
            Process {
                name: "signal".into(),
                pdf: sig_pdf,
                shape_param_indices: vec![1, 2],
                yield_expr: YieldExpr::Scaled { base_yield: s0, scale_index: 0 },
            },
            Process {
                name: "background".into(),
                pdf: bkg_pdf,
                shape_param_indices: vec![3],
                yield_expr: YieldExpr::Parameter { index: 4 },
            },
        ],
    };

    let model = UnbinnedModel::new(params, vec![channel], Some(0)).unwrap();
    let mle = MaximumLikelihoodEstimator::new();
    let fit = mle.fit(&model).unwrap();

    assert!(fit.converged, "fit did not converge: {:?}", fit);
    let mu_hat = fit.parameters[0];
    assert_relative_eq!(mu_hat, mu_true, max_relative = 0.25);
}

#[test]
fn test_baseline_compare_histogram_unbinned_matches_binned_histfactory() {
    use ns_core::traits::{FixedParamModel, PoiModel};
    use ns_translate::pyhf::HistFactoryModel;
    use ns_translate::pyhf::model::{
        HistoSysInterpCode as HfHistoSysInterpCode, NormSysInterpCode,
    };
    use ns_translate::pyhf::schema::{
        Channel as HfChannel, HistoSysData, Measurement as HfMeasurement, MeasurementConfig,
        Modifier as HfModifier, NormSysData, Observation as HfObservation, Sample as HfSample,
        Workspace as HfWorkspace,
    };

    // ── Define a tiny binned workspace with 1 channel and 2 samples ──────────
    let bin_edges = vec![0.0, 1.0, 2.0, 3.0];
    let obs_counts = vec![15.0, 25.0, 12.0];

    let sig_nominal = vec![5.0, 10.0, 3.0];
    let sig_down = vec![4.0, 12.0, 2.0];
    let sig_up = vec![6.0, 9.0, 3.0];

    let bkg_nominal = vec![10.0, 18.0, 9.0];
    let normsys_hi = 1.1;
    let normsys_lo = 0.9;

    let ws = HfWorkspace {
        channels: vec![HfChannel {
            name: "SR".into(),
            samples: vec![
                HfSample {
                    name: "signal".into(),
                    data: sig_nominal.clone(),
                    modifiers: vec![
                        HfModifier::NormFactor { name: "mu".into(), data: None },
                        HfModifier::HistoSys {
                            name: "sig_shape".into(),
                            data: HistoSysData {
                                hi_data: sig_up.clone(),
                                lo_data: sig_down.clone(),
                            },
                        },
                    ],
                },
                HfSample {
                    name: "background".into(),
                    data: bkg_nominal.clone(),
                    modifiers: vec![HfModifier::NormSys {
                        name: "bkg_norm".into(),
                        data: NormSysData { hi: normsys_hi, lo: normsys_lo },
                    }],
                },
            ],
        }],
        observations: vec![HfObservation { name: "SR".into(), data: obs_counts.clone() }],
        measurements: vec![HfMeasurement {
            name: "meas".into(),
            config: MeasurementConfig { poi: "mu".into(), parameters: vec![] },
        }],
        version: Some("1.0.0".into()),
    };

    // Match unbinned NormSys interpolation (hi^alpha / lo^{-alpha}).
    let hf = HistFactoryModel::from_workspace_with_settings(
        &ws,
        NormSysInterpCode::Code1,
        // Use code0 so that if up/down integrals match nominal, the sample normalization is stable
        // across alpha (binned and unbinned become equivalent up to constants).
        HfHistoSysInterpCode::Code0,
    )
    .unwrap();

    // ── Build an equivalent unbinned model using histogram-based PDFs ─────────
    let obs = ObservableSpec::branch("x", (0.0, 3.0));
    let centers = [0.5, 1.5, 2.5];
    let mut xs = Vec::<f64>::new();
    for (&n, &c) in obs_counts.iter().zip(centers.iter()) {
        xs.extend(std::iter::repeat_n(c, n as usize));
    }
    let store =
        Arc::new(EventStore::from_columns(vec![obs], vec![("x".to_string(), xs)], None).unwrap());

    let mu_idx = 0usize;
    let sig_shape_idx = 1usize;
    let bkg_norm_idx = 2usize;

    let params = vec![
        Parameter { name: "mu".into(), init: 1.0, bounds: (0.0, 10.0), constraint: None },
        Parameter {
            name: "sig_shape".into(),
            init: 0.0,
            bounds: (-5.0, 5.0),
            constraint: Some(crate::model::Constraint::Gaussian { mean: 0.0, sigma: 1.0 }),
        },
        Parameter {
            name: "bkg_norm".into(),
            init: 0.0,
            bounds: (-5.0, 5.0),
            constraint: Some(crate::model::Constraint::Gaussian { mean: 0.0, sigma: 1.0 }),
        },
    ];

    let sig_pdf: Arc<dyn UnbinnedPdf> = Arc::new(
        MorphingHistogramPdf::new(
            "x",
            bin_edges.clone(),
            sig_nominal.clone(),
            vec![HistogramSystematic {
                down: sig_down.clone(),
                up: sig_up.clone(),
                interp_code: HistoSysInterpCode::Code0,
            }],
            0.0,
        )
        .unwrap(),
    );

    let bkg_pdf: Arc<dyn UnbinnedPdf> = Arc::new(
        HistogramPdf::from_edges_and_contents("x", bin_edges.clone(), bkg_nominal.clone(), 0.0)
            .unwrap(),
    );

    let s0: f64 = sig_nominal.iter().sum();
    let b0: f64 = bkg_nominal.iter().sum();

    let channel = UnbinnedChannel {
        name: "SR".into(),
        include_in_fit: true,
        data: store,
        processes: vec![
            Process {
                name: "signal".into(),
                pdf: sig_pdf,
                shape_param_indices: vec![sig_shape_idx],
                yield_expr: YieldExpr::Scaled { base_yield: s0, scale_index: mu_idx },
            },
            Process {
                name: "background".into(),
                pdf: bkg_pdf,
                shape_param_indices: vec![],
                yield_expr: YieldExpr::Modified {
                    base: Box::new(YieldExpr::Fixed(b0)),
                    modifiers: vec![RateModifier::NormSys {
                        alpha_index: bkg_norm_idx,
                        lo: normsys_lo,
                        hi: normsys_hi,
                    }],
                },
            },
        ],
    };

    let unbinned = UnbinnedModel::new(params, vec![channel], Some(mu_idx)).unwrap();

    // ── Ranking compare (impact on POI) ──────────────────────────────────────
    let mle = MaximumLikelihoodEstimator::new();

    // Binned: built-in ranking.
    let binned_ranking = mle.ranking(&hf).unwrap();

    // Unbinned: same definition as `nextstat unbinned-ranking` (±1σ refits).
    let nominal = mle.fit(&unbinned).unwrap();
    assert!(nominal.converged, "unbinned nominal fit did not converge");
    let poi_idx = unbinned.poi_index().unwrap();
    assert_eq!(poi_idx, mu_idx, "unexpected POI index");
    let mu_hat = nominal.parameters[poi_idx];

    let mut unbinned_entries = Vec::<ns_inference::RankingEntry>::new();
    for (np_idx, p) in unbinned.parameters().iter().enumerate() {
        if np_idx == poi_idx {
            continue;
        }
        let Some(constraint) = p.constraint.clone() else { continue };
        let (center, sigma) = match constraint {
            crate::model::Constraint::Gaussian { mean, sigma } => (mean, sigma),
        };
        let (b_lo, b_hi) = p.bounds;
        let val_up = (center + sigma).min(b_hi);
        let val_down = (center - sigma).max(b_lo);

        // +1σ
        let m_up = unbinned.with_fixed_param(np_idx, val_up);
        let mut warm_up = nominal.parameters.clone();
        warm_up[np_idx] = val_up;
        let r_up = mle.fit_minimum_from(&m_up, &warm_up).unwrap();

        // -1σ
        let m_down = unbinned.with_fixed_param(np_idx, val_down);
        let mut warm_down = nominal.parameters.clone();
        warm_down[np_idx] = val_down;
        let r_down = mle.fit_minimum_from(&m_down, &warm_down).unwrap();

        unbinned_entries.push(ns_inference::RankingEntry {
            name: p.name.clone(),
            delta_mu_up: r_up.parameters[poi_idx] - mu_hat,
            delta_mu_down: r_down.parameters[poi_idx] - mu_hat,
            pull: (nominal.parameters[np_idx] - center) / sigma,
            constraint: nominal.uncertainties[np_idx] / sigma,
        });
    }

    // Compare binned vs unbinned deltas by NP name.
    for b in &binned_ranking {
        let u = unbinned_entries
            .iter()
            .find(|e| e.name == b.name)
            .unwrap_or_else(|| panic!("missing unbinned ranking entry for {}", b.name));
        assert_relative_eq!(u.delta_mu_up, b.delta_mu_up, epsilon = 1e-2);
        assert_relative_eq!(u.delta_mu_down, b.delta_mu_down, epsilon = 1e-2);
    }

    // ── Toy-based CLs compare (qtilde) ───────────────────────────────────────
    let mu_test = 1.0;
    let n_toys = 2000usize;
    let seed = 123u64;

    let binned = ns_inference::hypotest_qtilde_toys(&mle, &hf, mu_test, n_toys, seed).unwrap();
    let unbinned_toys = ns_inference::hypotest_qtilde_toys_with_sampler(
        &mle,
        &unbinned,
        mu_test,
        n_toys,
        seed,
        |m, params, s| m.sample_poisson_toy(params, s),
    )
    .unwrap();

    assert_relative_eq!(unbinned_toys.q_obs, binned.q_obs, epsilon = 1e-6);
    assert_relative_eq!(unbinned_toys.mu_hat, binned.mu_hat, epsilon = 1e-4);
    assert!(
        (unbinned_toys.cls - binned.cls).abs() < 0.1,
        "CLs mismatch: unbinned={} binned={}",
        unbinned_toys.cls,
        binned.cls
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Track A: Phase 3 tests
// ═══════════════════════════════════════════════════════════════════════════

use crate::normalize::{
    NormalizationCache, QuadratureGrid, QuadratureOrder, log_normalize_quadrature,
    log_normalize_quadrature_grad,
};
use crate::pdf::{ArgusPdf, KdeNdPdf, ProductPdf, SplinePdf, VoigtianPdf};

// ── A1: ProductPdf ──────────────────────────────────────────────────────

#[test]
fn test_product_pdf_two_gaussians() {
    let obs_x = ObservableSpec::branch("x", (0.0, 10.0));
    let obs_y = ObservableSpec::branch("y", (-5.0, 5.0));
    let xs = vec![2.0, 5.0, 8.0];
    let ys = vec![0.0, -1.0, 2.0];
    let store = EventStore::from_columns(
        vec![obs_x, obs_y],
        vec![("x".to_string(), xs.clone()), ("y".to_string(), ys.clone())],
        None,
    )
    .unwrap();

    let g1: Arc<dyn UnbinnedPdf> = Arc::new(GaussianPdf::new("x"));
    let g2: Arc<dyn UnbinnedPdf> = Arc::new(GaussianPdf::new("y"));
    let product = ProductPdf::new(vec![g1.clone(), g2.clone()]).unwrap();

    assert_eq!(product.n_params(), 4); // mu_x, sigma_x, mu_y, sigma_y
    assert_eq!(product.observables().len(), 2);

    // params: [mu_x=5, sigma_x=2, mu_y=0, sigma_y=1]
    let params = [5.0, 2.0, 0.0, 1.0];
    let n = store.n_events();
    let mut out = vec![0.0; n];
    product.log_prob_batch(&store, &params, &mut out).unwrap();

    // Verify: product log_prob = sum of individual log_probs.
    let store_x = EventStore::from_columns(
        vec![ObservableSpec::branch("x", (0.0, 10.0))],
        vec![("x".to_string(), xs)],
        None,
    )
    .unwrap();
    let store_y = EventStore::from_columns(
        vec![ObservableSpec::branch("y", (-5.0, 5.0))],
        vec![("y".to_string(), ys)],
        None,
    )
    .unwrap();

    let mut out_x = vec![0.0; n];
    let mut out_y = vec![0.0; n];
    g1.log_prob_batch(&store_x, &[5.0, 2.0], &mut out_x).unwrap();
    g2.log_prob_batch(&store_y, &[0.0, 1.0], &mut out_y).unwrap();

    for i in 0..n {
        assert_relative_eq!(out[i], out_x[i] + out_y[i], epsilon = 1e-12);
    }
}

#[test]
fn test_product_pdf_grad() {
    let obs_x = ObservableSpec::branch("x", (0.0, 10.0));
    let obs_y = ObservableSpec::branch("y", (-5.0, 5.0));
    let store = EventStore::from_columns(
        vec![obs_x, obs_y],
        vec![("x".to_string(), vec![3.0, 7.0]), ("y".to_string(), vec![1.0, -2.0])],
        None,
    )
    .unwrap();

    let g1: Arc<dyn UnbinnedPdf> = Arc::new(GaussianPdf::new("x"));
    let g2: Arc<dyn UnbinnedPdf> = Arc::new(GaussianPdf::new("y"));
    let product = ProductPdf::new(vec![g1, g2]).unwrap();

    let params = [5.0, 2.0, 0.0, 1.0];
    let n = store.n_events();
    let np = product.n_params();

    let mut logp = vec![0.0; n];
    let mut grad = vec![0.0; n * np];
    product.log_prob_grad_batch(&store, &params, &mut logp, &mut grad).unwrap();

    let fd = finite_diff_grad_vec(&params, 1e-6, |p| {
        let mut lp = vec![0.0; n];
        product.log_prob_batch(&store, p, &mut lp).unwrap();
        lp
    });

    for i in 0..n * np {
        assert_relative_eq!(grad[i], fd[i], epsilon = 5e-5);
    }
}

#[test]
fn test_product_pdf_rejects_duplicate_observables() {
    let g1: Arc<dyn UnbinnedPdf> = Arc::new(GaussianPdf::new("x"));
    let g2: Arc<dyn UnbinnedPdf> = Arc::new(GaussianPdf::new("x"));
    let result = ProductPdf::new(vec![g1, g2]);
    assert!(result.is_err());
}

// ── A2: SplinePdf ───────────────────────────────────────────────────────

#[test]
fn test_spline_pdf_uniform() {
    // Constant density y=1 on [0, 2] → density should be 0.5 everywhere.
    let knots_x = vec![0.0, 1.0, 2.0];
    let knots_y = vec![1.0, 1.0, 1.0];
    let pdf = SplinePdf::from_knots("x", knots_x, knots_y).unwrap();

    let obs = ObservableSpec::branch("x", (0.0, 2.0));
    let store =
        EventStore::from_columns(vec![obs], vec![("x".to_string(), vec![0.5, 1.0, 1.5])], None)
            .unwrap();

    let mut out = vec![0.0; 3];
    pdf.log_prob_batch(&store, &[], &mut out).unwrap();

    for &lp in &out {
        assert_relative_eq!(lp, (0.5f64).ln(), epsilon = 1e-10);
    }
}

#[test]
fn test_spline_pdf_normalization() {
    // Non-trivial shape; check that ∫ exp(logp) ≈ 1 via quadrature.
    let knots_x = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let knots_y = vec![0.5, 1.5, 2.0, 1.0, 0.3];
    let pdf = SplinePdf::from_knots("x", knots_x, knots_y).unwrap();

    let grid = QuadratureGrid::new_1d("x", (0.0, 1.0), QuadratureOrder::N64).unwrap();
    let log_int = log_normalize_quadrature(&pdf, &grid, &[]).unwrap();

    assert_relative_eq!(log_int.exp(), 1.0, epsilon = 1e-4);
}

// ── A3: KdeNdPdf ────────────────────────────────────────────────────────

#[test]
fn test_kde_nd_2d_construction() {
    let centers = vec![vec![0.5, 0.5], vec![0.3, 0.7], vec![0.7, 0.3], vec![0.6, 0.6]];
    let kde = KdeNdPdf::from_samples(
        vec!["x".to_string(), "y".to_string()],
        vec![(0.0, 1.0), (0.0, 1.0)],
        centers,
        None,
        None,
    )
    .unwrap();

    assert_eq!(kde.n_params(), 0);
    assert_eq!(kde.observables().len(), 2);
}

#[test]
fn test_kde_nd_2d_log_prob() {
    let centers = vec![vec![0.5, 0.5], vec![0.3, 0.7], vec![0.7, 0.3]];
    let kde = KdeNdPdf::from_samples(
        vec!["x".to_string(), "y".to_string()],
        vec![(0.0, 1.0), (0.0, 1.0)],
        centers,
        None,
        Some(vec![0.15, 0.15]),
    )
    .unwrap();

    let obs_x = ObservableSpec::branch("x", (0.0, 1.0));
    let obs_y = ObservableSpec::branch("y", (0.0, 1.0));
    let store = EventStore::from_columns(
        vec![obs_x, obs_y],
        vec![("x".to_string(), vec![0.5, 0.3]), ("y".to_string(), vec![0.5, 0.7])],
        None,
    )
    .unwrap();

    let mut out = vec![0.0; 2];
    kde.log_prob_batch(&store, &[], &mut out).unwrap();

    // Density at the center of mass should be higher than elsewhere.
    // At least check it's finite and reasonable.
    for &lp in &out {
        assert!(lp.is_finite(), "log_prob should be finite, got {lp}");
        assert!(lp > -20.0, "log_prob should not be extremely small, got {lp}");
    }
}

#[test]
fn test_kde_nd_rejects_1d() {
    let result = KdeNdPdf::from_samples(
        vec!["x".to_string()],
        vec![(0.0, 1.0)],
        vec![vec![0.5]],
        None,
        None,
    );
    assert!(result.is_err());
}

// ── A4: ArgusPdf ────────────────────────────────────────────────────────

#[test]
fn test_argus_pdf_log_prob() {
    let pdf = ArgusPdf::new("m");
    let obs = ObservableSpec::branch("m", (0.0, 5.29));
    let store = EventStore::from_columns(
        vec![obs],
        vec![("m".to_string(), vec![1.0, 2.5, 4.0, 5.0])],
        None,
    )
    .unwrap();

    // c = -2.0, p = 0.5 (classic ARGUS)
    let mut out = vec![0.0; 4];
    pdf.log_prob_batch(&store, &[-2.0, 0.5], &mut out).unwrap();

    for &lp in &out {
        assert!(lp.is_finite(), "log_prob should be finite, got {lp}");
    }
    // Density should decrease monotonically near the cutoff (x → m₀).
    // x=5.0 is very close to the cutoff 5.29, so density should be low there.
    assert!(out[0] < out[1], "density should increase from low x toward the peak");
}

#[test]
fn test_argus_pdf_grad() {
    let pdf = ArgusPdf::new("m");
    let obs = ObservableSpec::branch("m", (0.0, 5.29));
    let store =
        EventStore::from_columns(vec![obs], vec![("m".to_string(), vec![2.0, 3.5])], None).unwrap();

    let params = [-2.0, 0.5];
    let n = store.n_events();
    let np = pdf.n_params();

    let mut logp = vec![0.0; n];
    let mut grad = vec![0.0; n * np];
    pdf.log_prob_grad_batch(&store, &params, &mut logp, &mut grad).unwrap();

    let fd = finite_diff_grad_vec(&params, 1e-5, |p| {
        let mut lp = vec![0.0; n];
        pdf.log_prob_batch(&store, p, &mut lp).unwrap();
        lp
    });

    for i in 0..n * np {
        assert_relative_eq!(grad[i], fd[i], epsilon = 1e-3);
    }
}

// ── A4: VoigtianPdf ────────────────────────────────────────────────────

#[test]
fn test_voigtian_pdf_log_prob() {
    let pdf = VoigtianPdf::new("m");
    let obs = ObservableSpec::branch("m", (80.0, 100.0));
    let store =
        EventStore::from_columns(vec![obs], vec![("m".to_string(), vec![88.0, 91.2, 94.0])], None)
            .unwrap();

    // mu=91.2 (Z mass), sigma=2.0, gamma=2.5
    let mut out = vec![0.0; 3];
    pdf.log_prob_batch(&store, &[91.2, 2.0, 2.5], &mut out).unwrap();

    for &lp in &out {
        assert!(lp.is_finite(), "log_prob should be finite, got {lp}");
    }
    // Peak should be at the center.
    assert!(out[1] > out[0], "peak should be at mu");
    assert!(out[1] > out[2], "peak should be at mu");
}

#[test]
fn test_voigtian_pdf_grad() {
    let pdf = VoigtianPdf::new("m");
    let obs = ObservableSpec::branch("m", (80.0, 100.0));
    let store =
        EventStore::from_columns(vec![obs], vec![("m".to_string(), vec![89.0, 91.0])], None)
            .unwrap();

    let params = [91.2, 2.0, 2.5];
    let n = store.n_events();
    let np = pdf.n_params();

    let mut logp = vec![0.0; n];
    let mut grad = vec![0.0; n * np];
    pdf.log_prob_grad_batch(&store, &params, &mut logp, &mut grad).unwrap();

    let fd = finite_diff_grad_vec(&params, 1e-5, |p| {
        let mut lp = vec![0.0; n];
        pdf.log_prob_batch(&store, p, &mut lp).unwrap();
        lp
    });

    for i in 0..n * np {
        assert_relative_eq!(grad[i], fd[i], epsilon = 1e-3);
    }
}

// ── A5: Normalization quadrature gradient ───────────────────────────────

#[test]
fn test_normalize_quadrature_gaussian_integral_is_one() {
    let pdf = GaussianPdf::new("x");
    let grid = QuadratureGrid::new_1d("x", (0.0, 10.0), QuadratureOrder::N64).unwrap();
    let log_int = log_normalize_quadrature(&pdf, &grid, &[5.0, 1.0]).unwrap();
    // A properly normalized PDF should integrate to 1.
    assert_relative_eq!(log_int.exp(), 1.0, epsilon = 1e-8);
}

#[test]
fn test_normalize_quadrature_grad_gaussian() {
    let pdf = GaussianPdf::new("x");
    let grid = QuadratureGrid::new_1d("x", (0.0, 10.0), QuadratureOrder::N64).unwrap();
    let params = [5.0, 1.0];

    let (log_int, grad) = log_normalize_quadrature_grad(&pdf, &grid, &params).unwrap();

    // For a properly normalized PDF, log_int ≈ 0 and grad ≈ 0.
    assert_relative_eq!(log_int, 0.0, epsilon = 1e-8);

    // Verify gradient via finite differences.
    let eps = 1e-5;
    for k in 0..params.len() {
        let mut p_plus = params.to_vec();
        let mut p_minus = params.to_vec();
        p_plus[k] += eps;
        p_minus[k] -= eps;
        let li_plus = log_normalize_quadrature(&pdf, &grid, &p_plus).unwrap();
        let li_minus = log_normalize_quadrature(&pdf, &grid, &p_minus).unwrap();
        let fd = (li_plus - li_minus) / (2.0 * eps);
        assert_relative_eq!(grad[k], fd, epsilon = 1e-4);
    }
}

// ── A6: NormalizationCache ──────────────────────────────────────────────

#[test]
fn test_normalization_cache_hit() {
    let pdf = GaussianPdf::new("x");
    let grid = QuadratureGrid::new_1d("x", (0.0, 10.0), QuadratureOrder::N32).unwrap();
    let mut cache = NormalizationCache::with_default_precision(grid);

    let params = [5.0, 1.0];

    let v1 = cache.log_norm(&pdf, &params).unwrap();
    assert_eq!(cache.len(), 1);

    // Same params → cache hit.
    let v2 = cache.log_norm(&pdf, &params).unwrap();
    assert_eq!(cache.len(), 1);
    assert_eq!(v1, v2);

    // Different params → cache miss.
    let _v3 = cache.log_norm(&pdf, &[5.0, 2.0]).unwrap();
    assert_eq!(cache.len(), 2);
}

#[test]
fn test_normalization_cache_clear() {
    let pdf = GaussianPdf::new("x");
    let grid = QuadratureGrid::new_1d("x", (0.0, 10.0), QuadratureOrder::N32).unwrap();
    let mut cache = NormalizationCache::with_default_precision(grid);

    cache.log_norm(&pdf, &[5.0, 1.0]).unwrap();
    assert_eq!(cache.len(), 1);

    cache.clear();
    assert!(cache.is_empty());
}
