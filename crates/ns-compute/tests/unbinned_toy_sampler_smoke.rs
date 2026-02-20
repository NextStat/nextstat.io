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

/// 3-process model for multi-process CDF dispatch testing.
/// Gaussian(mu=5, sigma=1) + Exponential(lambda=-0.4) + Chebyshev(c1,c2,c3).
/// No rate modifiers — tests pure CDF process selection.
fn build_three_proc_toy_data() -> (UnbinnedGpuModelData, Vec<f64>) {
    // Params: 0=gauss_mu, 1=gauss_sigma, 2=exp_lambda, 3=c1, 4=c2, 5=c3
    let data = UnbinnedGpuModelData {
        n_params: 6,
        n_obs: 1,
        n_events: 0,
        obs_bounds: vec![(0.0, 10.0)],
        obs_soa: Vec::new(),
        event_weights: None,
        processes: vec![
            // Gaussian: yield=100 (fixed)
            GpuUnbinnedProcessDesc {
                base_yield: 100.0,
                pdf_kind: pdf_kind::GAUSSIAN,
                yield_kind: yield_kind::FIXED,
                obs_index: 0,
                shape_param_offset: 0,
                n_shape_params: 2,
                yield_param_idx: 0,
                rate_mod_offset: 0,
                n_rate_mods: 0,
                pdf_aux_offset: 0,
                pdf_aux_len: 0,
            },
            // Exponential: yield=80 (fixed)
            GpuUnbinnedProcessDesc {
                base_yield: 80.0,
                pdf_kind: pdf_kind::EXPONENTIAL,
                yield_kind: yield_kind::FIXED,
                obs_index: 0,
                shape_param_offset: 2,
                n_shape_params: 1,
                yield_param_idx: 0,
                rate_mod_offset: 0,
                n_rate_mods: 0,
                pdf_aux_offset: 0,
                pdf_aux_len: 0,
            },
            // Chebyshev: yield=60 (fixed)
            GpuUnbinnedProcessDesc {
                base_yield: 60.0,
                pdf_kind: pdf_kind::CHEBYSHEV,
                yield_kind: yield_kind::FIXED,
                obs_index: 0,
                shape_param_offset: 3,
                n_shape_params: 3,
                yield_param_idx: 0,
                rate_mod_offset: 0,
                n_rate_mods: 0,
                pdf_aux_offset: 0,
                pdf_aux_len: 0,
            },
        ],
        rate_modifiers: Vec::new(),
        shape_param_indices: vec![0, 1, 2, 3, 4, 5],
        pdf_aux_f64: Vec::new(),
        gauss_constraints: Vec::new(),
        constraint_const: 0.0,
    };
    let params = vec![5.0, 1.0, -0.4, 0.10, -0.05, 0.02];
    (data, params)
}

#[cfg(feature = "metal")]
#[test]
fn metal_toy_sampler_multiproc_deterministic() {
    use ns_compute::metal_unbinned_toy::MetalUnbinnedToySampler;

    if !MetalUnbinnedToySampler::is_available() {
        return;
    }

    let (data, params) = build_three_proc_toy_data();
    let sampler = MetalUnbinnedToySampler::from_unbinned_static(&data).unwrap();

    let n_toys = 128;
    let seed = 42u64;

    // Run twice — must be bit-exact.
    let (offs1, xs1) = sampler.sample_toys_1d(&params, n_toys, seed).unwrap();
    let (offs2, xs2) = sampler.sample_toys_1d(&params, n_toys, seed).unwrap();

    assert_eq!(offs1, offs2, "toy offsets differ between identical runs");
    assert_eq!(xs1.len(), xs2.len(), "sample count differs between identical runs");
    for (i, (&a, &b)) in xs1.iter().zip(xs2.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "sample {i} not bit-exact: {a} vs {b}"
        );
    }

    // All samples in bounds.
    assert_basic_samples(&xs1, 0.0, 10.0);

    // Statistical: with yields 100:80:60, events should come from multiple processes.
    // A single-process degenerate model would cluster all samples similarly.
    // Check that at least 2 different "regions" have events (coarse check).
    let total = xs1.len();
    assert!(total > 0, "no events generated");
    let n_lower = xs1.iter().filter(|&&x| x < 3.0).count();
    let n_upper = xs1.iter().filter(|&&x| x >= 3.0).count();
    assert!(
        n_lower > 0 && n_upper > 0,
        "events not distributed across processes: n_lower={n_lower}, n_upper={n_upper}, total={total}"
    );

    println!(
        "multiproc deterministic: {n_toys} toys, {total} total events, \
         lower(<3)={n_lower}, upper(>=3)={n_upper}"
    );
}

#[cfg(feature = "metal")]
#[test]
fn metal_toy_sampler_fused_parity() {
    use ns_compute::metal_unbinned_toy::MetalUnbinnedToySampler;

    if !MetalUnbinnedToySampler::is_available() {
        return;
    }

    let (data, params) = build_three_proc_toy_data();
    let sampler = MetalUnbinnedToySampler::from_unbinned_static(&data).unwrap();

    let n_toys = 256;
    let seed = 99u64;

    // Fused path (default).
    let (offs_fused, xs_fused) = sampler.sample_toys_1d(&params, n_toys, seed).unwrap();
    // Two-pass path (forced).
    let (offs_two, xs_two) = sampler
        .sample_toys_1d_two_pass_for_test(&params, n_toys, seed)
        .unwrap();

    assert_eq!(
        offs_fused, offs_two,
        "toy offsets differ between fused and two-pass"
    );
    assert_eq!(
        xs_fused.len(),
        xs_two.len(),
        "sample count differs: fused={} vs two_pass={}",
        xs_fused.len(),
        xs_two.len()
    );
    for (i, (&a, &b)) in xs_fused.iter().zip(xs_two.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "sample {i} not bit-exact: fused={a} vs two_pass={b}"
        );
    }

    println!(
        "fused parity: {} toys, {} events, bit-exact OK",
        n_toys,
        xs_fused.len()
    );
}

#[cfg(feature = "metal")]
#[test]
fn metal_toy_sampler_fused_timing() {
    use ns_compute::metal_unbinned_toy::MetalUnbinnedToySampler;

    if !MetalUnbinnedToySampler::is_available() {
        return;
    }

    let (data, params) = build_three_proc_toy_data();
    let sampler = MetalUnbinnedToySampler::from_unbinned_static(&data).unwrap();

    let n_toys = 256;
    let n_iters = 10;

    // Warmup.
    let _ = sampler.sample_toys_1d(&params, n_toys, 0).unwrap();

    let mut total_ms = 0.0f64;
    for i in 0..n_iters {
        let (_offs, _xs, timing) = sampler
            .sample_toys_1d_timed(&params, n_toys, i as u64 + 1000)
            .unwrap();
        let wall = timing.counts_kernel_s + timing.counts_readback_s + timing.prefix_sum_s
            + timing.sample_kernel_s;
        total_ms += wall * 1000.0;
        if i == 0 {
            println!(
                "fused_timing iter=0: fused={}, counts_kernel={:.3}ms readback={:.3}ms prefix={:.3}ms sample={:.3}ms prepare={:.3}ms convert={:.3}ms",
                timing.fused,
                timing.counts_kernel_s * 1000.0,
                timing.counts_readback_s * 1000.0,
                timing.prefix_sum_s * 1000.0,
                timing.sample_kernel_s * 1000.0,
                timing.prepare_s * 1000.0,
                timing.host_convert_s * 1000.0,
            );
        }
    }
    println!(
        "fused_timing: {n_toys} toys, {n_iters} iters, avg={:.3}ms/call",
        total_ms / n_iters as f64
    );
}
