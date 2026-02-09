use crate::event_store::{EventStore, ObservableSpec};
use crate::model::{Parameter, Process, RateModifier, UnbinnedChannel, UnbinnedModel, YieldExpr};
use crate::pdf::{
    ChebyshevPdf, CrystalBallPdf, DoubleCrystalBallPdf, ExponentialPdf, GaussianPdf, HistogramPdf,
    KdePdf, UnbinnedPdf,
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
