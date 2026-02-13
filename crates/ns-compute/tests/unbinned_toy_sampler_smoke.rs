#![cfg(any(feature = "metal", feature = "cuda"))]

use ns_compute::unbinned_types::*;

fn build_single_proc_data(
    pdf_kind: u32,
    obs_bounds: (f64, f64),
    shape_param_indices: Vec<u32>,
    pdf_aux_f64: Vec<f64>,
    n_shape_params: u32,
    pdf_aux_len: u32,
) -> UnbinnedGpuModelData {
    UnbinnedGpuModelData {
        n_params: (shape_param_indices.iter().copied().max().unwrap_or(0) as usize) + 1,
        n_obs: 1,
        n_events: 0,
        obs_bounds: vec![obs_bounds],
        obs_soa: Vec::new(),
        event_weights: None,
        processes: vec![GpuUnbinnedProcessDesc {
            base_yield: 5.0,
            pdf_kind,
            yield_kind: yield_kind::FIXED,
            obs_index: 0,
            shape_param_offset: 0,
            n_shape_params,
            yield_param_idx: 0,
            rate_mod_offset: 0,
            n_rate_mods: 0,
            pdf_aux_offset: 0,
            pdf_aux_len,
        }],
        rate_modifiers: Vec::new(),
        shape_param_indices,
        pdf_aux_f64,
        gauss_constraints: Vec::new(),
        constraint_const: 0.0,
    }
}

fn assert_basic_samples(samples: &[f64], a: f64, b: f64) {
    assert!(!samples.is_empty(), "expected some samples");
    let mut minv = f64::INFINITY;
    let mut maxv = f64::NEG_INFINITY;
    for &x in samples {
        assert!(x.is_finite(), "non-finite sample: {x}");
        assert!(x >= a && x <= b, "sample out of bounds: x={x} not in [{a}, {b}]");
        minv = minv.min(x);
        maxv = maxv.max(x);
    }
    assert!((maxv - minv) > 1e-9, "samples look degenerate: min={minv}, max={maxv}");
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_toy_sampler_supports_more_pdfs() {
    use ns_compute::cuda_unbinned_toy::CudaUnbinnedToySampler;

    if !CudaUnbinnedToySampler::is_available() {
        return;
    }

    let ctx = match std::panic::catch_unwind(|| cudarc::driver::CudaContext::new(0)) {
        Ok(Ok(ctx)) => ctx,
        _ => return,
    };
    let stream = ctx.default_stream();

    // Gaussian
    {
        let data =
            build_single_proc_data(pdf_kind::GAUSSIAN, (0.0, 10.0), vec![0, 1], Vec::new(), 2, 0);
        let mut sampler =
            CudaUnbinnedToySampler::with_context(ctx.clone(), stream.clone(), &data).unwrap();
        let params = vec![5.0, 1.0];
        let (_offs, xs) = sampler.sample_toys_1d(&params, 64, 123).unwrap();
        assert_basic_samples(&xs, 0.0, 10.0);
    }

    // Exponential
    {
        let data =
            build_single_proc_data(pdf_kind::EXPONENTIAL, (0.0, 10.0), vec![0], Vec::new(), 1, 0);
        let mut sampler =
            CudaUnbinnedToySampler::with_context(ctx.clone(), stream.clone(), &data).unwrap();
        let params = vec![-0.4];
        let (_offs, xs) = sampler.sample_toys_1d(&params, 64, 456).unwrap();
        assert_basic_samples(&xs, 0.0, 10.0);
    }

    // CrystalBall: mu, sigma, alpha, n
    {
        let data = build_single_proc_data(
            pdf_kind::CRYSTAL_BALL,
            (0.0, 10.0),
            vec![0, 1, 2, 3],
            Vec::new(),
            4,
            0,
        );
        let mut sampler =
            CudaUnbinnedToySampler::with_context(ctx.clone(), stream.clone(), &data).unwrap();
        let params = vec![5.0, 1.0, 1.5, 3.0];
        let (_offs, xs) = sampler.sample_toys_1d(&params, 64, 789).unwrap();
        assert_basic_samples(&xs, 0.0, 10.0);
    }

    // DoubleCrystalBall: mu, sigma, alphaL, nL, alphaR, nR
    {
        let data = build_single_proc_data(
            pdf_kind::DOUBLE_CRYSTAL_BALL,
            (0.0, 10.0),
            vec![0, 1, 2, 3, 4, 5],
            Vec::new(),
            6,
            0,
        );
        let mut sampler =
            CudaUnbinnedToySampler::with_context(ctx.clone(), stream.clone(), &data).unwrap();
        let params = vec![5.0, 1.0, 1.3, 3.0, 1.8, 4.0];
        let (_offs, xs) = sampler.sample_toys_1d(&params, 64, 101112).unwrap();
        assert_basic_samples(&xs, 0.0, 10.0);
    }

    // Chebyshev: coefficients c1..cK
    {
        let data = build_single_proc_data(
            pdf_kind::CHEBYSHEV,
            (-1.0, 1.0),
            vec![0, 1, 2],
            Vec::new(),
            3,
            0,
        );
        let mut sampler =
            CudaUnbinnedToySampler::with_context(ctx.clone(), stream.clone(), &data).unwrap();
        let params = vec![0.10, -0.05, 0.02];
        let (_offs, xs) = sampler.sample_toys_1d(&params, 64, 131415).unwrap();
        assert_basic_samples(&xs, -1.0, 1.0);
    }

    // Histogram: edges[4] then logdens[3]
    {
        let pdf_aux_f64 = vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let data =
            build_single_proc_data(pdf_kind::HISTOGRAM, (0.0, 3.0), Vec::new(), pdf_aux_f64, 0, 7);
        let mut sampler =
            CudaUnbinnedToySampler::with_context(ctx.clone(), stream.clone(), &data).unwrap();
        let params = vec![0.0]; // unused
        let (_offs, xs) = sampler.sample_toys_1d(&params, 64, 161718).unwrap();
        assert_basic_samples(&xs, 0.0, 3.0);

        let mut bins = [0usize; 3];
        for &x in &xs {
            let b = if x < 1.0 {
                0
            } else if x < 2.0 {
                1
            } else {
                2
            };
            bins[b] += 1;
        }
        assert!(
            bins.iter().filter(|&&c| c > 0).count() >= 2,
            "histogram samples concentrated in one bin: {bins:?}"
        );
    }
}

#[cfg(feature = "metal")]
#[test]
#[ignore]
fn metal_toy_sampler_supports_more_pdfs() {
    use ns_compute::metal_unbinned_toy::MetalUnbinnedToySampler;

    if !MetalUnbinnedToySampler::is_available() {
        return;
    }

    // CrystalBall: mu, sigma, alpha, n
    {
        let data = build_single_proc_data(
            pdf_kind::CRYSTAL_BALL,
            (0.0, 10.0),
            vec![0, 1, 2, 3],
            Vec::new(),
            4,
            0,
        );
        let sampler = MetalUnbinnedToySampler::from_unbinned_static(&data).unwrap();
        let params = vec![5.0, 1.0, 1.5, 3.0];
        let (_offs, xs) = sampler.sample_toys_1d(&params, 64, 424242).unwrap();
        assert_basic_samples(&xs, 0.0, 10.0);
    }

    // Chebyshev: coefficients c1..cK
    {
        let data = build_single_proc_data(
            pdf_kind::CHEBYSHEV,
            (-1.0, 1.0),
            vec![0, 1, 2],
            Vec::new(),
            3,
            0,
        );
        let sampler = MetalUnbinnedToySampler::from_unbinned_static(&data).unwrap();
        let params = vec![0.10, -0.05, 0.02];
        let (_offs, xs) = sampler.sample_toys_1d(&params, 64, 434343).unwrap();
        assert_basic_samples(&xs, -1.0, 1.0);
    }

    // Histogram: edges[4] then logdens[3]
    {
        let pdf_aux_f64 = vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let data =
            build_single_proc_data(pdf_kind::HISTOGRAM, (0.0, 3.0), Vec::new(), pdf_aux_f64, 0, 7);
        let sampler = MetalUnbinnedToySampler::from_unbinned_static(&data).unwrap();
        let params = vec![0.0]; // unused
        let (_offs, xs) = sampler.sample_toys_1d(&params, 64, 444444).unwrap();
        assert_basic_samples(&xs, 0.0, 3.0);
    }
}
