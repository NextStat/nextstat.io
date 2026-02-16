#![cfg(any(feature = "metal", feature = "cuda"))]

use ns_compute::unbinned_types::*;
use ns_core::traits::LogDensityModel;
use ns_unbinned::{
    ChebyshevPdf, Constraint, CrystalBallPdf, DoubleCrystalBallPdf, EventStore, ExponentialPdf,
    GaussianPdf, HistogramPdf, ObservableSpec, Parameter, Process, RateModifier, UnbinnedChannel,
    UnbinnedModel, YieldExpr,
};
use std::sync::Arc;

fn lower_gauss_constraints(params: &[Parameter]) -> (Vec<GpuUnbinnedGaussConstraintEntry>, f64) {
    let mut entries = Vec::new();
    let mut constant = 0.0f64;
    for (param_idx, p) in params.iter().enumerate() {
        let Some(c) = &p.constraint else { continue };
        match c {
            Constraint::Gaussian { mean, sigma } => {
                let mean = *mean;
                let sigma = *sigma;
                entries.push(GpuUnbinnedGaussConstraintEntry {
                    center: mean,
                    inv_width: 1.0 / sigma,
                    param_idx: param_idx as u32,
                    _pad: 0,
                });
                constant += sigma.ln() + 0.5 * (2.0 * std::f64::consts::PI).ln();
            }
        }
    }
    (entries, constant)
}

fn build_reference_model_and_gpu_data() -> (UnbinnedModel, UnbinnedGpuModelData, Vec<f64>) {
    // Parameters (stable order)
    // 0: mu (POI) - signal strength (scale)
    // 1: nu_bkg - background yield parameter
    // 2: gauss_mu
    // 3: gauss_sigma
    // 4: exp_lambda (with Gaussian constraint)
    // 5: alpha_bkg (NormSys yield modifier nuisance)
    let params = vec![
        Parameter { name: "mu".into(), init: 1.0, bounds: (0.0, 5.0), constraint: None },
        Parameter { name: "nu_bkg".into(), init: 500.0, bounds: (0.0, 5000.0), constraint: None },
        Parameter { name: "gauss_mu".into(), init: 125.0, bounds: (0.0, 500.0), constraint: None },
        Parameter {
            name: "gauss_sigma".into(),
            init: 30.0,
            bounds: (0.1, 200.0),
            constraint: None,
        },
        Parameter {
            name: "exp_lambda".into(),
            init: -0.01,
            bounds: (-1.0, 1.0),
            constraint: Some(Constraint::Gaussian { mean: -0.01, sigma: 0.1 }),
        },
        Parameter {
            name: "alpha_bkg".into(),
            init: 0.0,
            bounds: (-5.0, 5.0),
            constraint: Some(Constraint::Gaussian { mean: 0.0, sigma: 1.0 }),
        },
    ];

    // Synthetic observed events (1D).
    let n_events = 256usize;
    let mut xs = Vec::with_capacity(n_events);
    for i in 0..n_events {
        let t = (i as f64 + 0.5) / (n_events as f64);
        xs.push(500.0 * t);
    }
    let obs = ObservableSpec::branch("x", (0.0, 500.0));
    let store = EventStore::from_columns(vec![obs], vec![("x".into(), xs.clone())], None).unwrap();
    let store = Arc::new(store);

    // Processes: signal Gaussian + background Exponential
    let sig_pdf = Arc::new(GaussianPdf::new("x"));
    let bkg_pdf = Arc::new(ExponentialPdf::new("x"));

    let sig = Process {
        name: "sig".into(),
        pdf: sig_pdf,
        shape_param_indices: vec![2, 3],
        yield_expr: YieldExpr::Scaled { base_yield: 200.0, scale_index: 0 },
    };
    let bkg = Process {
        name: "bkg".into(),
        pdf: bkg_pdf,
        shape_param_indices: vec![4],
        yield_expr: YieldExpr::Modified {
            base: Box::new(YieldExpr::Parameter { index: 1 }),
            modifiers: vec![RateModifier::NormSys { alpha_index: 5, lo: 0.9, hi: 1.1 }],
        },
    };

    let ch = UnbinnedChannel {
        name: "SR".into(),
        include_in_fit: true,
        data: store,
        processes: vec![sig, bkg],
    };

    let (gauss_constraints, constraint_const) = lower_gauss_constraints(&params);
    let model = UnbinnedModel::new(params, vec![ch], Some(0)).unwrap();
    let init = model.parameter_init();

    let gpu = UnbinnedGpuModelData {
        n_params: init.len(),
        n_obs: 1,
        n_events,
        obs_bounds: vec![(0.0, 500.0)],
        obs_soa: xs,
        event_weights: None,
        processes: vec![
            // Signal: Gaussian(mu=gauss_mu, sigma=gauss_sigma), yield = 200 * mu
            GpuUnbinnedProcessDesc {
                base_yield: 200.0,
                pdf_kind: pdf_kind::GAUSSIAN,
                yield_kind: yield_kind::SCALED,
                obs_index: 0,
                shape_param_offset: 0,
                n_shape_params: 2,
                yield_param_idx: 0,
                rate_mod_offset: 0,
                n_rate_mods: 0,
                pdf_aux_offset: 0,
                pdf_aux_len: 0,
            },
            // Background: Exponential(lambda=exp_lambda), yield = nu_bkg (free parameter)
            GpuUnbinnedProcessDesc {
                base_yield: 0.0,
                pdf_kind: pdf_kind::EXPONENTIAL,
                yield_kind: yield_kind::PARAMETER,
                obs_index: 0,
                shape_param_offset: 2,
                n_shape_params: 1,
                yield_param_idx: 1,
                rate_mod_offset: 0,
                n_rate_mods: 1,
                pdf_aux_offset: 0,
                pdf_aux_len: 0,
            },
        ],
        rate_modifiers: vec![GpuUnbinnedRateModifierDesc {
            kind: rate_modifier_kind::NORM_SYS,
            alpha_param_idx: 5,
            interp_code: 0,
            _pad: 0,
            lo: 0.9,
            hi: 1.1,
        }],
        shape_param_indices: vec![2, 3, 4],
        pdf_aux_f64: Vec::new(),
        gauss_constraints,
        constraint_const,
    };

    (model, gpu, init)
}

fn build_weighted_reference_model_and_gpu_data() -> (UnbinnedModel, UnbinnedGpuModelData, Vec<f64>)
{
    // Same as `build_reference_model_and_gpu_data`, but with per-event weights applied.
    let params = vec![
        Parameter { name: "mu".into(), init: 1.0, bounds: (0.0, 5.0), constraint: None },
        Parameter { name: "nu_bkg".into(), init: 500.0, bounds: (0.0, 5000.0), constraint: None },
        Parameter { name: "gauss_mu".into(), init: 125.0, bounds: (0.0, 500.0), constraint: None },
        Parameter {
            name: "gauss_sigma".into(),
            init: 30.0,
            bounds: (0.1, 200.0),
            constraint: None,
        },
        Parameter {
            name: "exp_lambda".into(),
            init: -0.01,
            bounds: (-1.0, 1.0),
            constraint: Some(Constraint::Gaussian { mean: -0.01, sigma: 0.1 }),
        },
        Parameter {
            name: "alpha_bkg".into(),
            init: 0.0,
            bounds: (-5.0, 5.0),
            constraint: Some(Constraint::Gaussian { mean: 0.0, sigma: 1.0 }),
        },
    ];

    let n_events = 256usize;
    let mut xs = Vec::with_capacity(n_events);
    let mut wts = Vec::with_capacity(n_events);
    for i in 0..n_events {
        let t = (i as f64 + 0.5) / (n_events as f64);
        xs.push(500.0 * t);
        wts.push(if (i % 7) == 0 { 0.5 } else { 1.2 });
    }
    let obs = ObservableSpec::branch("x", (0.0, 500.0));
    let store =
        EventStore::from_columns(vec![obs], vec![("x".into(), xs.clone())], Some(wts.clone()))
            .unwrap();
    let store = Arc::new(store);

    let sig_pdf = Arc::new(GaussianPdf::new("x"));
    let bkg_pdf = Arc::new(ExponentialPdf::new("x"));

    let sig = Process {
        name: "sig".into(),
        pdf: sig_pdf,
        shape_param_indices: vec![2, 3],
        yield_expr: YieldExpr::Scaled { base_yield: 200.0, scale_index: 0 },
    };
    let bkg = Process {
        name: "bkg".into(),
        pdf: bkg_pdf,
        shape_param_indices: vec![4],
        yield_expr: YieldExpr::Modified {
            base: Box::new(YieldExpr::Parameter { index: 1 }),
            modifiers: vec![RateModifier::NormSys { alpha_index: 5, lo: 0.9, hi: 1.1 }],
        },
    };

    let ch = UnbinnedChannel {
        name: "SR".into(),
        include_in_fit: true,
        data: store,
        processes: vec![sig, bkg],
    };

    let (gauss_constraints, constraint_const) = lower_gauss_constraints(&params);
    let model = UnbinnedModel::new(params, vec![ch], Some(0)).unwrap();
    let init = model.parameter_init();

    let gpu = UnbinnedGpuModelData {
        n_params: init.len(),
        n_obs: 1,
        n_events,
        obs_bounds: vec![(0.0, 500.0)],
        obs_soa: xs,
        event_weights: Some(wts),
        processes: vec![
            GpuUnbinnedProcessDesc {
                base_yield: 200.0,
                pdf_kind: pdf_kind::GAUSSIAN,
                yield_kind: yield_kind::SCALED,
                obs_index: 0,
                shape_param_offset: 0,
                n_shape_params: 2,
                yield_param_idx: 0,
                rate_mod_offset: 0,
                n_rate_mods: 0,
                pdf_aux_offset: 0,
                pdf_aux_len: 0,
            },
            GpuUnbinnedProcessDesc {
                base_yield: 0.0,
                pdf_kind: pdf_kind::EXPONENTIAL,
                yield_kind: yield_kind::PARAMETER,
                obs_index: 0,
                shape_param_offset: 2,
                n_shape_params: 1,
                yield_param_idx: 1,
                rate_mod_offset: 0,
                n_rate_mods: 1,
                pdf_aux_offset: 0,
                pdf_aux_len: 0,
            },
        ],
        rate_modifiers: vec![GpuUnbinnedRateModifierDesc {
            kind: rate_modifier_kind::NORM_SYS,
            alpha_param_idx: 5,
            interp_code: 0,
            _pad: 0,
            lo: 0.9,
            hi: 1.1,
        }],
        shape_param_indices: vec![2, 3, 4],
        pdf_aux_f64: Vec::new(),
        gauss_constraints,
        constraint_const,
    };

    (model, gpu, init)
}

fn build_crystal_ball_reference_model_and_gpu_data()
-> (UnbinnedModel, UnbinnedGpuModelData, Vec<f64>) {
    // 0: mu (POI)
    // 1: nu_bkg
    // 2..5: CrystalBall (mu, sigma, alpha, n)
    // 6: exp_lambda (Gaussian constraint)
    // 7: alpha_bkg (NormSys nuisance, Gaussian constraint)
    let params = vec![
        Parameter { name: "mu".into(), init: 1.0, bounds: (0.0, 5.0), constraint: None },
        Parameter { name: "nu_bkg".into(), init: 500.0, bounds: (0.0, 5000.0), constraint: None },
        Parameter { name: "cb_mu".into(), init: 125.0, bounds: (0.0, 500.0), constraint: None },
        Parameter { name: "cb_sigma".into(), init: 30.0, bounds: (0.1, 200.0), constraint: None },
        Parameter { name: "cb_alpha".into(), init: 1.5, bounds: (0.01, 10.0), constraint: None },
        Parameter { name: "cb_n".into(), init: 3.0, bounds: (1.01, 80.0), constraint: None },
        Parameter {
            name: "exp_lambda".into(),
            init: -0.01,
            bounds: (-1.0, 1.0),
            constraint: Some(Constraint::Gaussian { mean: -0.01, sigma: 0.1 }),
        },
        Parameter {
            name: "alpha_bkg".into(),
            init: 0.0,
            bounds: (-5.0, 5.0),
            constraint: Some(Constraint::Gaussian { mean: 0.0, sigma: 1.0 }),
        },
    ];

    let n_events = 256usize;
    let mut xs = Vec::with_capacity(n_events);
    for i in 0..n_events {
        let t = (i as f64 + 0.5) / (n_events as f64);
        xs.push(500.0 * t);
    }
    let obs = ObservableSpec::branch("x", (0.0, 500.0));
    let store = EventStore::from_columns(vec![obs], vec![("x".into(), xs.clone())], None).unwrap();
    let store = Arc::new(store);

    let sig_pdf = Arc::new(CrystalBallPdf::new("x"));
    let bkg_pdf = Arc::new(ExponentialPdf::new("x"));

    let sig = Process {
        name: "sig".into(),
        pdf: sig_pdf,
        shape_param_indices: vec![2, 3, 4, 5],
        yield_expr: YieldExpr::Scaled { base_yield: 200.0, scale_index: 0 },
    };
    let bkg = Process {
        name: "bkg".into(),
        pdf: bkg_pdf,
        shape_param_indices: vec![6],
        yield_expr: YieldExpr::Modified {
            base: Box::new(YieldExpr::Parameter { index: 1 }),
            modifiers: vec![RateModifier::NormSys { alpha_index: 7, lo: 0.9, hi: 1.1 }],
        },
    };

    let ch = UnbinnedChannel {
        name: "SR".into(),
        include_in_fit: true,
        data: store,
        processes: vec![sig, bkg],
    };

    let model = UnbinnedModel::new(params.clone(), vec![ch], Some(0)).unwrap();
    let init = model.parameter_init();

    let (gauss_constraints, constraint_const) = lower_gauss_constraints(&params);
    let gpu = UnbinnedGpuModelData {
        n_params: init.len(),
        n_obs: 1,
        n_events,
        obs_bounds: vec![(0.0, 500.0)],
        obs_soa: xs,
        event_weights: None,
        processes: vec![
            GpuUnbinnedProcessDesc {
                base_yield: 200.0,
                pdf_kind: pdf_kind::CRYSTAL_BALL,
                yield_kind: yield_kind::SCALED,
                obs_index: 0,
                shape_param_offset: 0,
                n_shape_params: 4,
                yield_param_idx: 0,
                rate_mod_offset: 0,
                n_rate_mods: 0,
                pdf_aux_offset: 0,
                pdf_aux_len: 0,
            },
            GpuUnbinnedProcessDesc {
                base_yield: 0.0,
                pdf_kind: pdf_kind::EXPONENTIAL,
                yield_kind: yield_kind::PARAMETER,
                obs_index: 0,
                shape_param_offset: 4,
                n_shape_params: 1,
                yield_param_idx: 1,
                rate_mod_offset: 0,
                n_rate_mods: 1,
                pdf_aux_offset: 0,
                pdf_aux_len: 0,
            },
        ],
        rate_modifiers: vec![GpuUnbinnedRateModifierDesc {
            kind: rate_modifier_kind::NORM_SYS,
            alpha_param_idx: 7,
            interp_code: 0,
            _pad: 0,
            lo: 0.9,
            hi: 1.1,
        }],
        shape_param_indices: vec![2, 3, 4, 5, 6],
        pdf_aux_f64: Vec::new(),
        gauss_constraints,
        constraint_const,
    };

    (model, gpu, init)
}

fn build_double_crystal_ball_reference_model_and_gpu_data()
-> (UnbinnedModel, UnbinnedGpuModelData, Vec<f64>) {
    // 0: mu (POI)
    // 1: nu_bkg
    // 2..7: DoubleCB (mu, sigma, alpha_l, n_l, alpha_r, n_r)
    // 8: exp_lambda (Gaussian constraint)
    // 9: alpha_bkg (NormSys nuisance, Gaussian constraint)
    let params = vec![
        Parameter { name: "mu".into(), init: 1.0, bounds: (0.0, 5.0), constraint: None },
        Parameter { name: "nu_bkg".into(), init: 500.0, bounds: (0.0, 5000.0), constraint: None },
        Parameter { name: "dcb_mu".into(), init: 125.0, bounds: (0.0, 500.0), constraint: None },
        Parameter { name: "dcb_sigma".into(), init: 30.0, bounds: (0.1, 200.0), constraint: None },
        Parameter { name: "dcb_alpha_l".into(), init: 1.5, bounds: (0.01, 10.0), constraint: None },
        Parameter { name: "dcb_n_l".into(), init: 3.0, bounds: (1.01, 80.0), constraint: None },
        Parameter { name: "dcb_alpha_r".into(), init: 2.0, bounds: (0.01, 10.0), constraint: None },
        Parameter { name: "dcb_n_r".into(), init: 4.0, bounds: (1.01, 80.0), constraint: None },
        Parameter {
            name: "exp_lambda".into(),
            init: -0.01,
            bounds: (-1.0, 1.0),
            constraint: Some(Constraint::Gaussian { mean: -0.01, sigma: 0.1 }),
        },
        Parameter {
            name: "alpha_bkg".into(),
            init: 0.0,
            bounds: (-5.0, 5.0),
            constraint: Some(Constraint::Gaussian { mean: 0.0, sigma: 1.0 }),
        },
    ];

    let n_events = 256usize;
    let mut xs = Vec::with_capacity(n_events);
    for i in 0..n_events {
        let t = (i as f64 + 0.5) / (n_events as f64);
        xs.push(500.0 * t);
    }
    let obs = ObservableSpec::branch("x", (0.0, 500.0));
    let store = EventStore::from_columns(vec![obs], vec![("x".into(), xs.clone())], None).unwrap();
    let store = Arc::new(store);

    let sig_pdf = Arc::new(DoubleCrystalBallPdf::new("x"));
    let bkg_pdf = Arc::new(ExponentialPdf::new("x"));

    let sig = Process {
        name: "sig".into(),
        pdf: sig_pdf,
        shape_param_indices: vec![2, 3, 4, 5, 6, 7],
        yield_expr: YieldExpr::Scaled { base_yield: 200.0, scale_index: 0 },
    };
    let bkg = Process {
        name: "bkg".into(),
        pdf: bkg_pdf,
        shape_param_indices: vec![8],
        yield_expr: YieldExpr::Modified {
            base: Box::new(YieldExpr::Parameter { index: 1 }),
            modifiers: vec![RateModifier::NormSys { alpha_index: 9, lo: 0.9, hi: 1.1 }],
        },
    };

    let ch = UnbinnedChannel {
        name: "SR".into(),
        include_in_fit: true,
        data: store,
        processes: vec![sig, bkg],
    };

    let model = UnbinnedModel::new(params.clone(), vec![ch], Some(0)).unwrap();
    let init = model.parameter_init();

    let (gauss_constraints, constraint_const) = lower_gauss_constraints(&params);
    let gpu = UnbinnedGpuModelData {
        n_params: init.len(),
        n_obs: 1,
        n_events,
        obs_bounds: vec![(0.0, 500.0)],
        obs_soa: xs,
        event_weights: None,
        processes: vec![
            GpuUnbinnedProcessDesc {
                base_yield: 200.0,
                pdf_kind: pdf_kind::DOUBLE_CRYSTAL_BALL,
                yield_kind: yield_kind::SCALED,
                obs_index: 0,
                shape_param_offset: 0,
                n_shape_params: 6,
                yield_param_idx: 0,
                rate_mod_offset: 0,
                n_rate_mods: 0,
                pdf_aux_offset: 0,
                pdf_aux_len: 0,
            },
            GpuUnbinnedProcessDesc {
                base_yield: 0.0,
                pdf_kind: pdf_kind::EXPONENTIAL,
                yield_kind: yield_kind::PARAMETER,
                obs_index: 0,
                shape_param_offset: 6,
                n_shape_params: 1,
                yield_param_idx: 1,
                rate_mod_offset: 0,
                n_rate_mods: 1,
                pdf_aux_offset: 0,
                pdf_aux_len: 0,
            },
        ],
        rate_modifiers: vec![GpuUnbinnedRateModifierDesc {
            kind: rate_modifier_kind::NORM_SYS,
            alpha_param_idx: 9,
            interp_code: 0,
            _pad: 0,
            lo: 0.9,
            hi: 1.1,
        }],
        shape_param_indices: vec![2, 3, 4, 5, 6, 7, 8],
        pdf_aux_f64: Vec::new(),
        gauss_constraints,
        constraint_const,
    };

    (model, gpu, init)
}

fn build_chebyshev_reference_model_and_gpu_data() -> (UnbinnedModel, UnbinnedGpuModelData, Vec<f64>)
{
    // 0: mu (POI)
    // 1: nu_bkg
    // 2..3: Gaussian (mu, sigma)
    // 4..6: Chebyshev (c1..c3)
    // 7: alpha_bkg (NormSys nuisance, Gaussian constraint)
    let params = vec![
        Parameter { name: "mu".into(), init: 1.0, bounds: (0.0, 5.0), constraint: None },
        Parameter { name: "nu_bkg".into(), init: 500.0, bounds: (0.0, 5000.0), constraint: None },
        Parameter { name: "gauss_mu".into(), init: 125.0, bounds: (0.0, 500.0), constraint: None },
        Parameter {
            name: "gauss_sigma".into(),
            init: 30.0,
            bounds: (0.1, 200.0),
            constraint: None,
        },
        Parameter { name: "c1".into(), init: 0.10, bounds: (-0.5, 0.5), constraint: None },
        Parameter { name: "c2".into(), init: -0.05, bounds: (-0.5, 0.5), constraint: None },
        Parameter { name: "c3".into(), init: 0.02, bounds: (-0.5, 0.5), constraint: None },
        Parameter {
            name: "alpha_bkg".into(),
            init: 0.0,
            bounds: (-5.0, 5.0),
            constraint: Some(Constraint::Gaussian { mean: 0.0, sigma: 1.0 }),
        },
    ];

    let n_events = 256usize;
    let mut xs = Vec::with_capacity(n_events);
    for i in 0..n_events {
        let t = (i as f64 + 0.5) / (n_events as f64);
        xs.push(500.0 * t);
    }
    let obs = ObservableSpec::branch("x", (0.0, 500.0));
    let store = EventStore::from_columns(vec![obs], vec![("x".into(), xs.clone())], None).unwrap();
    let store = Arc::new(store);

    let sig_pdf = Arc::new(GaussianPdf::new("x"));
    let bkg_pdf = Arc::new(ChebyshevPdf::new("x", 3).unwrap());

    let sig = Process {
        name: "sig".into(),
        pdf: sig_pdf,
        shape_param_indices: vec![2, 3],
        yield_expr: YieldExpr::Scaled { base_yield: 200.0, scale_index: 0 },
    };
    let bkg = Process {
        name: "bkg".into(),
        pdf: bkg_pdf,
        shape_param_indices: vec![4, 5, 6],
        yield_expr: YieldExpr::Modified {
            base: Box::new(YieldExpr::Parameter { index: 1 }),
            modifiers: vec![RateModifier::NormSys { alpha_index: 7, lo: 0.9, hi: 1.1 }],
        },
    };

    let ch = UnbinnedChannel {
        name: "SR".into(),
        include_in_fit: true,
        data: store,
        processes: vec![sig, bkg],
    };

    let model = UnbinnedModel::new(params.clone(), vec![ch], Some(0)).unwrap();
    let init = model.parameter_init();

    let (gauss_constraints, constraint_const) = lower_gauss_constraints(&params);
    let gpu = UnbinnedGpuModelData {
        n_params: init.len(),
        n_obs: 1,
        n_events,
        obs_bounds: vec![(0.0, 500.0)],
        obs_soa: xs,
        event_weights: None,
        processes: vec![
            GpuUnbinnedProcessDesc {
                base_yield: 200.0,
                pdf_kind: pdf_kind::GAUSSIAN,
                yield_kind: yield_kind::SCALED,
                obs_index: 0,
                shape_param_offset: 0,
                n_shape_params: 2,
                yield_param_idx: 0,
                rate_mod_offset: 0,
                n_rate_mods: 0,
                pdf_aux_offset: 0,
                pdf_aux_len: 0,
            },
            GpuUnbinnedProcessDesc {
                base_yield: 0.0,
                pdf_kind: pdf_kind::CHEBYSHEV,
                yield_kind: yield_kind::PARAMETER,
                obs_index: 0,
                shape_param_offset: 2,
                n_shape_params: 3,
                yield_param_idx: 1,
                rate_mod_offset: 0,
                n_rate_mods: 1,
                pdf_aux_offset: 0,
                pdf_aux_len: 0,
            },
        ],
        rate_modifiers: vec![GpuUnbinnedRateModifierDesc {
            kind: rate_modifier_kind::NORM_SYS,
            alpha_param_idx: 7,
            interp_code: 0,
            _pad: 0,
            lo: 0.9,
            hi: 1.1,
        }],
        shape_param_indices: vec![2, 3, 4, 5, 6],
        pdf_aux_f64: Vec::new(),
        gauss_constraints,
        constraint_const,
    };

    (model, gpu, init)
}

fn build_histogram_reference_model_and_gpu_data() -> (UnbinnedModel, UnbinnedGpuModelData, Vec<f64>)
{
    // 0: nu (free yield parameter)
    let params = vec![Parameter {
        name: "nu".into(),
        init: 1000.0,
        bounds: (0.0, 5000.0),
        constraint: None,
    }];

    // Synthetic observed events (1D) spanning the full support.
    let n_events = 256usize;
    let mut xs = Vec::with_capacity(n_events);
    for i in 0..n_events {
        let t = (i as f64 + 0.5) / (n_events as f64);
        xs.push(500.0 * t);
    }
    let obs = ObservableSpec::branch("x", (0.0, 500.0));
    let store = EventStore::from_columns(vec![obs], vec![("x".into(), xs.clone())], None).unwrap();
    let store = Arc::new(store);

    // Simple 5-bin histogram with pseudo-count smoothing.
    let bin_edges = vec![0.0, 100.0, 200.0, 300.0, 400.0, 500.0];
    let bin_content = vec![1.0, 2.0, 3.0, 2.0, 1.0];
    let pseudo_count = 0.5;
    let pdf = Arc::new(
        HistogramPdf::from_edges_and_contents(
            "x",
            bin_edges.clone(),
            bin_content.clone(),
            pseudo_count,
        )
        .unwrap(),
    );

    let p = Process {
        name: "p".into(),
        pdf,
        shape_param_indices: vec![],
        yield_expr: YieldExpr::Parameter { index: 0 },
    };
    let ch = UnbinnedChannel {
        name: "SR".into(),
        include_in_fit: true,
        data: store,
        processes: vec![p],
    };

    let model = UnbinnedModel::new(params.clone(), vec![ch], None).unwrap();
    let init = model.parameter_init();

    // Lower histogram aux buffer as: edges + log_density (matches HistogramPdf semantics).
    let mut total = 0.0f64;
    for &w in &bin_content {
        total += w + pseudo_count;
    }
    let log_total = total.ln();
    let mut log_density = Vec::with_capacity(bin_content.len());
    for i in 0..bin_content.len() {
        let w = bin_content[i] + pseudo_count;
        let width = bin_edges[i + 1] - bin_edges[i];
        log_density.push(w.ln() - log_total - width.ln());
    }
    let mut pdf_aux_f64 = Vec::<f64>::new();
    pdf_aux_f64.extend_from_slice(&bin_edges);
    pdf_aux_f64.extend_from_slice(&log_density);

    let gpu = UnbinnedGpuModelData {
        n_params: init.len(),
        n_obs: 1,
        n_events,
        obs_bounds: vec![(0.0, 500.0)],
        obs_soa: xs,
        event_weights: None,
        processes: vec![GpuUnbinnedProcessDesc {
            base_yield: 0.0,
            pdf_kind: pdf_kind::HISTOGRAM,
            yield_kind: yield_kind::PARAMETER,
            obs_index: 0,
            shape_param_offset: 0,
            n_shape_params: 0,
            yield_param_idx: 0,
            rate_mod_offset: 0,
            n_rate_mods: 0,
            pdf_aux_offset: 0,
            pdf_aux_len: pdf_aux_f64.len() as u32,
        }],
        rate_modifiers: Vec::new(),
        shape_param_indices: Vec::new(),
        pdf_aux_f64,
        gauss_constraints: Vec::new(),
        constraint_const: 0.0,
    };

    (model, gpu, init)
}

#[cfg(feature = "metal")]
#[test]
fn metal_unbinned_nll_grad_matches_cpu_on_synthetic_model() {
    if !ns_compute::metal_unbinned::MetalUnbinnedAccelerator::is_available() {
        // Allow building the `metal` feature on non-Metal machines.
        return;
    }

    // Metal path is f32 and uses atomic accumulation; keep tolerances relaxed.
    const ABS_TOL: f64 = 2e-2;
    const REL_TOL: f64 = 2e-6;

    let (model, gpu, params) = build_reference_model_and_gpu_data();

    let (cpu_nll, cpu_grad) = model.nll_grad_prepared(&model.prepared(), &params).unwrap();

    let mut accel =
        ns_compute::metal_unbinned::MetalUnbinnedAccelerator::from_unbinned_data(&gpu).unwrap();
    let (gpu_nll, gpu_grad) = accel.single_nll_grad(&params).unwrap();

    let nll_tol = ABS_TOL.max(REL_TOL * cpu_nll.abs());
    assert!(
        (gpu_nll - cpu_nll).abs() < nll_tol,
        "NLL mismatch: gpu={gpu_nll:.8} cpu={cpu_nll:.8} diff={:.3e}",
        (gpu_nll - cpu_nll).abs()
    );
    assert_eq!(gpu_grad.len(), cpu_grad.len());
    for i in 0..cpu_grad.len() {
        let diff = (gpu_grad[i] - cpu_grad[i]).abs();
        let grad_tol = ABS_TOL.max(REL_TOL * cpu_grad[i].abs());
        assert!(
            diff < grad_tol,
            "grad[{i}] mismatch: gpu={} cpu={} diff={}",
            gpu_grad[i],
            cpu_grad[i],
            diff
        );
    }
}

#[cfg(feature = "metal")]
#[test]
fn metal_unbinned_nll_grad_matches_cpu_on_weighted_synthetic_model() {
    let (model, gpu, params) = build_weighted_reference_model_and_gpu_data();
    assert_metal_unbinned_matches_cpu(&model, &gpu, &params);
}

#[cfg(feature = "metal")]
fn assert_metal_unbinned_matches_cpu(
    model: &UnbinnedModel,
    gpu: &UnbinnedGpuModelData,
    params: &[f64],
) {
    if !ns_compute::metal_unbinned::MetalUnbinnedAccelerator::is_available() {
        return;
    }

    // Metal path is f32 and uses atomic accumulation; keep tolerances relaxed.
    const ABS_TOL: f64 = 5e-2;
    const REL_TOL: f64 = 5e-6;

    let (cpu_nll, cpu_grad) = model.nll_grad_prepared(&model.prepared(), params).unwrap();

    let mut accel =
        ns_compute::metal_unbinned::MetalUnbinnedAccelerator::from_unbinned_data(gpu).unwrap();
    let (gpu_nll, gpu_grad) = accel.single_nll_grad(params).unwrap();

    let nll_tol = ABS_TOL.max(REL_TOL * cpu_nll.abs());
    assert!(
        (gpu_nll - cpu_nll).abs() < nll_tol,
        "NLL mismatch: gpu={gpu_nll:.8} cpu={cpu_nll:.8} diff={:.3e}",
        (gpu_nll - cpu_nll).abs()
    );
    assert_eq!(gpu_grad.len(), cpu_grad.len());
    for i in 0..cpu_grad.len() {
        let diff = (gpu_grad[i] - cpu_grad[i]).abs();
        let grad_tol = ABS_TOL.max(REL_TOL * cpu_grad[i].abs());
        assert!(
            diff < grad_tol,
            "grad[{i}] mismatch: gpu={} cpu={} diff={}",
            gpu_grad[i],
            cpu_grad[i],
            diff
        );
    }
}

#[cfg(feature = "metal")]
fn assert_metal_unbinned_batch_matches_single(gpu: &UnbinnedGpuModelData, params: &[f64]) {
    if !ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::is_available() {
        return;
    }

    const ABS_TOL: f64 = 5e-3;
    const REL_TOL: f64 = 1e-6;

    let mut accel_single =
        ns_compute::metal_unbinned::MetalUnbinnedAccelerator::from_unbinned_data(gpu).unwrap();
    let (nll_single, grad_single) = accel_single.single_nll_grad(params).unwrap();

    let toy_offsets = vec![0u32, gpu.n_events as u32];
    let obs_flat = gpu.obs_soa.clone();
    let mut accel_batch = ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::from_unbinned_static_and_toys(
        gpu,
        &toy_offsets,
        &obs_flat,
        1,
    )
    .unwrap();
    let (nlls, grads_flat) = accel_batch.batch_nll_grad(params).unwrap();

    assert_eq!(nlls.len(), 1);
    assert_eq!(grads_flat.len(), grad_single.len());

    let nll_tol = ABS_TOL.max(REL_TOL * nll_single.abs());
    assert!(
        (nlls[0] - nll_single).abs() < nll_tol,
        "batch vs single NLL mismatch: batch={} single={} diff={}",
        nlls[0],
        nll_single,
        (nlls[0] - nll_single).abs()
    );
    for i in 0..grad_single.len() {
        let diff = (grads_flat[i] - grad_single[i]).abs();
        let tol = ABS_TOL.max(REL_TOL * grad_single[i].abs());
        assert!(
            diff < tol,
            "batch vs single grad[{i}] mismatch: batch={} single={} diff={}",
            grads_flat[i],
            grad_single[i],
            diff
        );
    }
}

#[cfg(feature = "metal")]
#[test]
fn metal_unbinned_nll_grad_matches_cpu_on_crystal_ball_model() {
    let (model, gpu, params) = build_crystal_ball_reference_model_and_gpu_data();
    assert_metal_unbinned_matches_cpu(&model, &gpu, &params);
    assert_metal_unbinned_batch_matches_single(&gpu, &params);
}

#[cfg(feature = "metal")]
#[test]
fn metal_unbinned_nll_grad_matches_cpu_on_double_crystal_ball_model() {
    let (model, gpu, params) = build_double_crystal_ball_reference_model_and_gpu_data();
    assert_metal_unbinned_matches_cpu(&model, &gpu, &params);
    assert_metal_unbinned_batch_matches_single(&gpu, &params);
}

#[cfg(feature = "metal")]
#[test]
fn metal_unbinned_nll_grad_matches_cpu_on_chebyshev_model() {
    let (model, gpu, params) = build_chebyshev_reference_model_and_gpu_data();
    assert_metal_unbinned_matches_cpu(&model, &gpu, &params);
    assert_metal_unbinned_batch_matches_single(&gpu, &params);
}

#[cfg(feature = "metal")]
#[test]
fn metal_unbinned_nll_grad_matches_cpu_on_histogram_model() {
    let (model, gpu, params) = build_histogram_reference_model_and_gpu_data();
    assert_metal_unbinned_matches_cpu(&model, &gpu, &params);
    assert_metal_unbinned_batch_matches_single(&gpu, &params);
}

/// 3-process model (Gaussian signal + Exponential bkg + Chebyshev bkg2).
/// With n_procs=3 this hits the fused single-pass event loop pipeline
/// (`ENABLE_FUSED=1`, `use_fused = n_procs >= 3 && n_procs <= 4`).
fn build_three_proc_reference_model_and_gpu_data() -> (UnbinnedModel, UnbinnedGpuModelData, Vec<f64>)
{
    // 0: mu (POI)
    // 1: nu_bkg1 (exponential yield)
    // 2: nu_bkg2 (chebyshev yield)
    // 3: gauss_mu
    // 4: gauss_sigma
    // 5: exp_lambda (Gaussian constraint)
    // 6..8: chebyshev c1..c3
    // 9: alpha_bkg (NormSys nuisance, Gaussian constraint)
    let params = vec![
        Parameter { name: "mu".into(), init: 1.0, bounds: (0.0, 5.0), constraint: None },
        Parameter { name: "nu_bkg1".into(), init: 300.0, bounds: (0.0, 5000.0), constraint: None },
        Parameter { name: "nu_bkg2".into(), init: 200.0, bounds: (0.0, 5000.0), constraint: None },
        Parameter { name: "gauss_mu".into(), init: 125.0, bounds: (0.0, 500.0), constraint: None },
        Parameter {
            name: "gauss_sigma".into(),
            init: 30.0,
            bounds: (0.1, 200.0),
            constraint: None,
        },
        Parameter {
            name: "exp_lambda".into(),
            init: -0.01,
            bounds: (-1.0, 1.0),
            constraint: Some(Constraint::Gaussian { mean: -0.01, sigma: 0.1 }),
        },
        Parameter { name: "c1".into(), init: 0.10, bounds: (-0.5, 0.5), constraint: None },
        Parameter { name: "c2".into(), init: -0.05, bounds: (-0.5, 0.5), constraint: None },
        Parameter { name: "c3".into(), init: 0.02, bounds: (-0.5, 0.5), constraint: None },
        Parameter {
            name: "alpha_bkg".into(),
            init: 0.0,
            bounds: (-5.0, 5.0),
            constraint: Some(Constraint::Gaussian { mean: 0.0, sigma: 1.0 }),
        },
    ];

    let n_events = 256usize;
    let mut xs = Vec::with_capacity(n_events);
    for i in 0..n_events {
        let t = (i as f64 + 0.5) / (n_events as f64);
        xs.push(500.0 * t);
    }
    let obs = ObservableSpec::branch("x", (0.0, 500.0));
    let store = EventStore::from_columns(vec![obs], vec![("x".into(), xs.clone())], None).unwrap();
    let store = Arc::new(store);

    let sig_pdf = Arc::new(GaussianPdf::new("x"));
    let bkg1_pdf = Arc::new(ExponentialPdf::new("x"));
    let bkg2_pdf = Arc::new(ChebyshevPdf::new("x", 3).unwrap());

    let sig = Process {
        name: "sig".into(),
        pdf: sig_pdf,
        shape_param_indices: vec![3, 4],
        yield_expr: YieldExpr::Scaled { base_yield: 200.0, scale_index: 0 },
    };
    let bkg1 = Process {
        name: "bkg1".into(),
        pdf: bkg1_pdf,
        shape_param_indices: vec![5],
        yield_expr: YieldExpr::Modified {
            base: Box::new(YieldExpr::Parameter { index: 1 }),
            modifiers: vec![RateModifier::NormSys { alpha_index: 9, lo: 0.9, hi: 1.1 }],
        },
    };
    let bkg2 = Process {
        name: "bkg2".into(),
        pdf: bkg2_pdf,
        shape_param_indices: vec![6, 7, 8],
        yield_expr: YieldExpr::Parameter { index: 2 },
    };

    let ch = UnbinnedChannel {
        name: "SR".into(),
        include_in_fit: true,
        data: store,
        processes: vec![sig, bkg1, bkg2],
    };

    let model = UnbinnedModel::new(params.clone(), vec![ch], Some(0)).unwrap();
    let init = model.parameter_init();

    let (gauss_constraints, constraint_const) = lower_gauss_constraints(&params);
    let gpu = UnbinnedGpuModelData {
        n_params: init.len(),
        n_obs: 1,
        n_events,
        obs_bounds: vec![(0.0, 500.0)],
        obs_soa: xs,
        event_weights: None,
        processes: vec![
            // Signal: Gaussian(mu, sigma), yield = 200 * mu
            GpuUnbinnedProcessDesc {
                base_yield: 200.0,
                pdf_kind: pdf_kind::GAUSSIAN,
                yield_kind: yield_kind::SCALED,
                obs_index: 0,
                shape_param_offset: 0,
                n_shape_params: 2,
                yield_param_idx: 0,
                rate_mod_offset: 0,
                n_rate_mods: 0,
                pdf_aux_offset: 0,
                pdf_aux_len: 0,
            },
            // Bkg1: Exponential(lambda), yield = nu_bkg1 * NormSys(alpha_bkg)
            GpuUnbinnedProcessDesc {
                base_yield: 0.0,
                pdf_kind: pdf_kind::EXPONENTIAL,
                yield_kind: yield_kind::PARAMETER,
                obs_index: 0,
                shape_param_offset: 2,
                n_shape_params: 1,
                yield_param_idx: 1,
                rate_mod_offset: 0,
                n_rate_mods: 1,
                pdf_aux_offset: 0,
                pdf_aux_len: 0,
            },
            // Bkg2: Chebyshev(c1,c2,c3), yield = nu_bkg2
            GpuUnbinnedProcessDesc {
                base_yield: 0.0,
                pdf_kind: pdf_kind::CHEBYSHEV,
                yield_kind: yield_kind::PARAMETER,
                obs_index: 0,
                shape_param_offset: 3,
                n_shape_params: 3,
                yield_param_idx: 2,
                rate_mod_offset: 0,
                n_rate_mods: 0,
                pdf_aux_offset: 0,
                pdf_aux_len: 0,
            },
        ],
        rate_modifiers: vec![GpuUnbinnedRateModifierDesc {
            kind: rate_modifier_kind::NORM_SYS,
            alpha_param_idx: 9,
            interp_code: 0,
            _pad: 0,
            lo: 0.9,
            hi: 1.1,
        }],
        shape_param_indices: vec![3, 4, 5, 6, 7, 8],
        pdf_aux_f64: Vec::new(),
        gauss_constraints,
        constraint_const,
    };

    (model, gpu, init)
}

/// Tests the fused pipeline path (3 processes → use_fused=true).
#[cfg(feature = "metal")]
#[test]
fn metal_unbinned_fused_3proc_nll_grad_matches_cpu() {
    let (model, gpu, params) = build_three_proc_reference_model_and_gpu_data();
    assert_metal_unbinned_matches_cpu(&model, &gpu, &params);
    assert_metal_unbinned_batch_matches_single(&gpu, &params);
}

/// Perf: batch NLL+grad throughput on the 2-process model (2-pass path).
#[cfg(feature = "metal")]
#[test]
fn metal_unbinned_batch_perf_2proc() {
    if !ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::is_available() {
        return;
    }
    let (_model, gpu, init) = build_reference_model_and_gpu_data();
    let n_toys = 64;
    let toy_offsets: Vec<u32> = (0..=n_toys).map(|i| (i * gpu.n_events) as u32).collect();
    let obs_flat: Vec<f64> = (0..n_toys).flat_map(|_| gpu.obs_soa.iter().copied()).collect();
    let params_flat: Vec<f64> = (0..n_toys).flat_map(|_| init.iter().copied()).collect();

    let mut accel =
        ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::from_unbinned_static_and_toys(
            &gpu,
            &toy_offsets,
            &obs_flat,
            n_toys,
        )
        .unwrap();

    // Warmup
    let _ = accel.batch_nll_grad(&params_flat).unwrap();

    let n_iter = 50;
    let t0 = std::time::Instant::now();
    for _ in 0..n_iter {
        let _ = accel.batch_nll_grad(&params_flat).unwrap();
    }
    let elapsed = t0.elapsed();
    let us_per_call = elapsed.as_micros() as f64 / n_iter as f64;
    eprintln!(
        "[perf] 2-pass (2 procs, {} toys, {} events/toy, {} params): {:.0} us/call ({:.1} ms total for {} iters)",
        n_toys,
        gpu.n_events,
        gpu.n_params,
        us_per_call,
        elapsed.as_secs_f64() * 1e3,
        n_iter
    );
}

/// Perf: batch NLL+grad throughput on the 3-process model (fused path).
#[cfg(feature = "metal")]
#[test]
fn metal_unbinned_batch_perf_3proc_fused() {
    if !ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::is_available() {
        return;
    }
    let (_model, gpu, init) = build_three_proc_reference_model_and_gpu_data();
    let n_toys = 64;
    let toy_offsets: Vec<u32> = (0..=n_toys).map(|i| (i * gpu.n_events) as u32).collect();
    let obs_flat: Vec<f64> = (0..n_toys).flat_map(|_| gpu.obs_soa.iter().copied()).collect();
    let params_flat: Vec<f64> = (0..n_toys).flat_map(|_| init.iter().copied()).collect();

    let mut accel =
        ns_compute::metal_unbinned_batch::MetalUnbinnedBatchAccelerator::from_unbinned_static_and_toys(
            &gpu,
            &toy_offsets,
            &obs_flat,
            n_toys,
        )
        .unwrap();

    // Warmup
    let _ = accel.batch_nll_grad(&params_flat).unwrap();

    let n_iter = 50;
    let t0 = std::time::Instant::now();
    for _ in 0..n_iter {
        let _ = accel.batch_nll_grad(&params_flat).unwrap();
    }
    let elapsed = t0.elapsed();
    let us_per_call = elapsed.as_micros() as f64 / n_iter as f64;
    eprintln!(
        "[perf] fused (3 procs, {} toys, {} events/toy, {} params): {:.0} us/call ({:.1} ms total for {} iters)",
        n_toys,
        gpu.n_events,
        gpu.n_params,
        us_per_call,
        elapsed.as_secs_f64() * 1e3,
        n_iter
    );
}

#[cfg(feature = "cuda")]
fn assert_cuda_unbinned_matches_cpu(
    model: &UnbinnedModel,
    gpu: &UnbinnedGpuModelData,
    params: &[f64],
) {
    if !ns_compute::cuda_unbinned::CudaUnbinnedAccelerator::is_available() {
        return;
    }

    const ABS_TOL: f64 = 1e-8;
    const REL_TOL: f64 = 1e-10;

    let (cpu_nll, cpu_grad) = model.nll_grad_prepared(&model.prepared(), params).unwrap();

    let mut accel =
        ns_compute::cuda_unbinned::CudaUnbinnedAccelerator::from_unbinned_data(gpu).unwrap();
    let (gpu_nll, gpu_grad) = accel.single_nll_grad(params).unwrap();

    let nll_tol = ABS_TOL.max(REL_TOL * cpu_nll.abs());
    assert!(
        (gpu_nll - cpu_nll).abs() < nll_tol,
        "NLL mismatch: gpu={gpu_nll:.12} cpu={cpu_nll:.12} diff={:.3e}",
        (gpu_nll - cpu_nll).abs()
    );
    assert_eq!(gpu_grad.len(), cpu_grad.len());
    for i in 0..cpu_grad.len() {
        let diff = (gpu_grad[i] - cpu_grad[i]).abs();
        let grad_tol = ABS_TOL.max(REL_TOL * cpu_grad[i].abs());
        assert!(
            diff < grad_tol,
            "grad[{i}] mismatch: gpu={} cpu={} diff={}",
            gpu_grad[i],
            cpu_grad[i],
            diff
        );
    }
}

#[cfg(feature = "cuda")]
fn assert_cuda_unbinned_batch_matches_single(gpu: &UnbinnedGpuModelData, params: &[f64]) {
    if !ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::is_available() {
        return;
    }

    const ABS_TOL: f64 = 1e-8;
    const REL_TOL: f64 = 1e-10;

    let mut accel_single =
        ns_compute::cuda_unbinned::CudaUnbinnedAccelerator::from_unbinned_data(gpu).unwrap();
    let (nll_single, grad_single) = accel_single.single_nll_grad(params).unwrap();

    let toy_offsets = vec![0u32, gpu.n_events as u32];
    let obs_flat = gpu.obs_soa.clone();
    let mut accel_batch =
        ns_compute::cuda_unbinned_batch::CudaUnbinnedBatchAccelerator::from_unbinned_static_and_toys(
            gpu,
            &toy_offsets,
            &obs_flat,
            1,
        )
        .unwrap();
    let (nlls, grads_flat) = accel_batch.batch_nll_grad(params).unwrap();

    assert_eq!(nlls.len(), 1);
    assert_eq!(grads_flat.len(), grad_single.len());

    let nll_tol = ABS_TOL.max(REL_TOL * nll_single.abs());
    assert!(
        (nlls[0] - nll_single).abs() < nll_tol,
        "batch vs single NLL mismatch: batch={} single={} diff={}",
        nlls[0],
        nll_single,
        (nlls[0] - nll_single).abs()
    );
    for i in 0..grad_single.len() {
        let diff = (grads_flat[i] - grad_single[i]).abs();
        let tol = ABS_TOL.max(REL_TOL * grad_single[i].abs());
        assert!(
            diff < tol,
            "batch vs single grad[{i}] mismatch: batch={} single={} diff={}",
            grads_flat[i],
            grad_single[i],
            diff
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_unbinned_nll_grad_matches_cpu_on_histogram_model() {
    let (model, gpu, params) = build_histogram_reference_model_and_gpu_data();
    assert_cuda_unbinned_matches_cpu(&model, &gpu, &params);
    assert_cuda_unbinned_batch_matches_single(&gpu, &params);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_unbinned_nll_grad_matches_cpu_on_weighted_synthetic_model() {
    let (model, gpu, params) = build_weighted_reference_model_and_gpu_data();
    assert_cuda_unbinned_matches_cpu(&model, &gpu, &params);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_unbinned_nll_grad_matches_cpu_on_crystal_ball_model() {
    let (model, gpu, params) = build_crystal_ball_reference_model_and_gpu_data();
    assert_cuda_unbinned_matches_cpu(&model, &gpu, &params);
    assert_cuda_unbinned_batch_matches_single(&gpu, &params);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_unbinned_nll_grad_matches_cpu_on_double_crystal_ball_model() {
    let (model, gpu, params) = build_double_crystal_ball_reference_model_and_gpu_data();
    assert_cuda_unbinned_matches_cpu(&model, &gpu, &params);
    assert_cuda_unbinned_batch_matches_single(&gpu, &params);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_unbinned_nll_grad_matches_cpu_on_chebyshev_model() {
    let (model, gpu, params) = build_chebyshev_reference_model_and_gpu_data();
    assert_cuda_unbinned_matches_cpu(&model, &gpu, &params);
    assert_cuda_unbinned_batch_matches_single(&gpu, &params);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_unbinned_nll_grad_matches_cpu_on_synthetic_model() {
    if !ns_compute::cuda_unbinned::CudaUnbinnedAccelerator::is_available() {
        // Allow building the `cuda` feature on non-CUDA machines.
        return;
    }

    let (model, gpu, params) = build_reference_model_and_gpu_data();

    let (cpu_nll, cpu_grad) = model.nll_grad_prepared(&model.prepared(), &params).unwrap();

    let mut accel =
        ns_compute::cuda_unbinned::CudaUnbinnedAccelerator::from_unbinned_data(&gpu).unwrap();
    let (gpu_nll, gpu_grad) = accel.single_nll_grad(&params).unwrap();

    assert!(
        (gpu_nll - cpu_nll).abs() < 1e-6,
        "NLL mismatch: gpu={gpu_nll:.12} cpu={cpu_nll:.12} diff={:.3e}",
        (gpu_nll - cpu_nll).abs()
    );
    assert_eq!(gpu_grad.len(), cpu_grad.len());
    for i in 0..cpu_grad.len() {
        let diff = (gpu_grad[i] - cpu_grad[i]).abs();
        assert!(
            diff < 1e-6,
            "grad[{i}] mismatch: gpu={} cpu={} diff={}",
            gpu_grad[i],
            cpu_grad[i],
            diff
        );
    }
}
