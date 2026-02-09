/**
 * Fused NLL + Analytical Gradient Metal Kernel for HistFactory models.
 *
 * Port of batch_nll_grad.cu to Metal Shading Language (MSL).
 * ALL computation in float (f32) — Apple Silicon has no hardware f64.
 *
 * Architecture: 1 threadgroup = 1 toy experiment.
 * Threads process bins via grid-stride loop.
 * Parameters cached in threadgroup memory (~1KB for 250 params).
 *
 * Supports all pyhf modifier types:
 *   0 = NormFactor    (scalar multiplicative)
 *   1 = ShapeSys      (per-bin multiplicative gamma)
 *   2 = ShapeFactor   (per-bin multiplicative free factor)
 *   3 = NormSys       (Code4 interpolation, scalar)
 *   4 = HistoSys      (Code4p additive delta, per-bin)
 *   5 = StatError     (per-bin multiplicative gamma)
 *   6 = Lumi          (scalar multiplicative)
 */

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

/* ---------- Modifier type constants (must match GpuModifierType in Rust) --- */
constant uint MOD_NORMFACTOR  = 0;
constant uint MOD_SHAPESYS    = 1;
constant uint MOD_SHAPEFACTOR = 2;
constant uint MOD_NORMSYS     = 3;
constant uint MOD_HISTOSYS    = 4;
constant uint MOD_STATERROR   = 5;
constant uint MOD_LUMI        = 6;

/* ---------- Struct mirrors of Rust #[repr(C)] types ---------------------- */

struct GpuSampleInfo {
    uint first_sample_bin;
    uint n_bins;
    uint main_bin_offset;
    uint n_modifiers;
};

struct GpuModifierDesc {
    uint param_idx;
    uchar modifier_type;
    uchar is_per_bin;
    ushort _pad;
    uint data_offset;
    uint n_bins;
};

struct MetalAuxPoissonEntry {
    ushort gamma_param_idx;
    ushort _pad;
    float tau;
    float observed_aux;
    float lgamma_obs;
};

struct MetalGaussConstraintEntry {
    ushort param_idx;
    ushort _pad;
    float center;
    float inv_width;
};

/* ---------- Scalar constants passed via set_bytes() ---------------------- */

struct ScalarArgs {
    uint n_params;
    uint n_main_bins;
    uint n_samples;
    uint n_aux_poisson;
    uint n_gauss;
    float constraint_const;
};

/* ---------- Helper: NormSys Code4 interpolation ------------------------- */

float normsys_code4_value(
    float alpha,
    const device float* mdata  /* 8 floats at data_offset */
) {
    float abs_alpha = fabs(alpha);
    if (abs_alpha >= 1.0f) {
        float ln_base = (alpha >= 0.0f) ? mdata[6] : mdata[7];
        return exp(abs_alpha * ln_base);
    }
    float a1 = alpha;
    float a2 = a1 * alpha;
    float a3 = a2 * alpha;
    float a4 = a3 * alpha;
    float a5 = a4 * alpha;
    float a6 = a5 * alpha;
    return 1.0f
        + mdata[0] * a1
        + mdata[1] * a2
        + mdata[2] * a3
        + mdata[3] * a4
        + mdata[4] * a5
        + mdata[5] * a6;
}

float normsys_code4_deriv_over_val(
    float alpha,
    float factor,
    const device float* mdata
) {
    float abs_alpha = fabs(alpha);
    if (abs_alpha >= 1.0f) {
        float ln_base = (alpha >= 0.0f) ? mdata[6] : mdata[7];
        float sign_a = (alpha >= 0.0f) ? 1.0f : -1.0f;
        return sign_a * ln_base;
    }
    float fprime = mdata[0]
        + 2.0f * mdata[1] * alpha
        + 3.0f * mdata[2] * alpha * alpha
        + 4.0f * mdata[3] * alpha * alpha * alpha
        + 5.0f * mdata[4] * alpha * alpha * alpha * alpha
        + 6.0f * mdata[5] * alpha * alpha * alpha * alpha * alpha;
    return fprime / factor;
}

/* ---------- Helper: HistoSys Code4p delta ------------------------------- */

float histosys_delta(float alpha, float delta_up, float delta_dn) {
    if (alpha > 1.0f) return delta_up * alpha;
    if (alpha < -1.0f) return delta_dn * alpha;

    float S = 0.5f * (delta_up + delta_dn);
    float A = 0.0625f * (delta_up - delta_dn);
    float a2 = alpha * alpha;
    float tmp = a2 * (a2 * (3.0f * a2 - 10.0f) + 15.0f);
    return alpha * S + tmp * A;
}

float histosys_delta_deriv(float alpha, float delta_up, float delta_dn) {
    if (alpha > 1.0f) return delta_up;
    if (alpha < -1.0f) return delta_dn;

    float S = 0.5f * (delta_up + delta_dn);
    float A = 0.0625f * (delta_up - delta_dn);
    float a2 = alpha * alpha;
    float a3 = a2 * alpha;
    float a5 = a3 * a2;
    return S + (18.0f * a5 - 40.0f * a3 + 30.0f * alpha) * A;
}

/* ---------- Main fused NLL + Gradient kernel ----------------------------- */

kernel void batch_nll_grad(
    /* Dynamic per-iteration buffers */
    const device float* g_params           [[buffer(0)]],   /* [n_toys × n_params] */
    const device float* g_observed         [[buffer(1)]],   /* [n_toys × n_main_bins] */
    const device float* g_ln_facts         [[buffer(2)]],   /* [n_toys × n_main_bins] */
    const device float* g_obs_mask         [[buffer(3)]],   /* [n_toys × n_main_bins] */
    /* Static model buffers */
    const device float* g_nominal          [[buffer(4)]],   /* [n_sample_bins] */
    const device GpuSampleInfo* g_samples  [[buffer(5)]],   /* [n_samples] */
    const device GpuModifierDesc* g_mod_descs [[buffer(6)]], /* [total_modifiers] */
    const device uint* g_mod_offsets       [[buffer(7)]],   /* [n_samples + 1] */
    const device uint* g_per_bin_pidx      [[buffer(8)]],   /* per-bin param indices */
    const device float* g_mod_data         [[buffer(9)]],   /* modifier extra data */
    const device MetalAuxPoissonEntry* g_aux_poisson [[buffer(10)]], /* [n_aux] */
    const device MetalGaussConstraintEntry* g_gauss  [[buffer(11)]], /* [n_gauss] */
    /* Output */
    device float* g_nll_out                [[buffer(12)]],  /* [n_toys] */
    device atomic_float* g_grad_out        [[buffer(13)]],  /* [n_toys × n_params] */
    /* Scalar args */
    constant ScalarArgs& args              [[buffer(14)]],
    /* Threadgroup / thread IDs */
    uint toy_idx       [[threadgroup_position_in_grid]],
    uint tid           [[thread_position_in_threadgroup]],
    uint block_size    [[threads_per_threadgroup]],
    /* Threadgroup memory: params[n_params] | nll_scratch[block_size] */
    threadgroup float* shared [[threadgroup(0)]]
) {
    uint n_params = args.n_params;
    uint n_main_bins = args.n_main_bins;
    uint n_samples = args.n_samples;
    uint n_aux_poisson = args.n_aux_poisson;
    uint n_gauss = args.n_gauss;
    float constraint_const = args.constraint_const;

    threadgroup float* s_params = shared;
    threadgroup float* s_nll = shared + n_params;

    /* ----- Phase 0: Load params into threadgroup memory ----- */
    const device float* my_params = g_params + toy_idx * n_params;
    for (uint i = tid; i < n_params; i += block_size) {
        s_params[i] = my_params[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* Toy-specific observed data pointers */
    const device float* my_obs = g_observed + toy_idx * n_main_bins;
    const device float* my_lnf = g_ln_facts + toy_idx * n_main_bins;
    const device float* my_mask = g_obs_mask + toy_idx * n_main_bins;
    device atomic_float* my_grad = g_grad_out + toy_idx * n_params;

    float local_nll = 0.0f;

    /* ----- Phase 1-3: Expected counts, Poisson NLL, gradient per bin ----- */
    for (uint bin = tid; bin < n_main_bins; bin += block_size) {
        float expected_bin = 0.0f;

        for (uint s = 0; s < n_samples; s++) {
            GpuSampleInfo sinfo = g_samples[s];
            if (bin < sinfo.main_bin_offset || bin >= sinfo.main_bin_offset + sinfo.n_bins) {
                continue;
            }
            uint local_bin = bin - sinfo.main_bin_offset;
            uint sample_bin = sinfo.first_sample_bin + local_bin;

            float nom = g_nominal[sample_bin];
            float delta = 0.0f;
            float factor = 1.0f;

            uint mod_start = g_mod_offsets[s];
            uint mod_end = g_mod_offsets[s + 1];

            for (uint m = mod_start; m < mod_end; m++) {
                GpuModifierDesc md = g_mod_descs[m];
                uint mtype = (uint)md.modifier_type;

                if (mtype == MOD_NORMFACTOR || mtype == MOD_LUMI) {
                    factor *= s_params[md.param_idx];
                }
                else if (mtype == MOD_SHAPESYS || mtype == MOD_STATERROR || mtype == MOD_SHAPEFACTOR) {
                    uint pidx = g_per_bin_pidx[md.data_offset + local_bin];
                    factor *= s_params[pidx];
                }
                else if (mtype == MOD_NORMSYS) {
                    float alpha = s_params[md.param_idx];
                    const device float* mdata = g_mod_data + md.data_offset;
                    float f = normsys_code4_value(alpha, mdata);
                    factor *= f;
                }
                else if (mtype == MOD_HISTOSYS) {
                    float alpha = s_params[md.param_idx];
                    const device float* mdata = g_mod_data + md.data_offset;
                    float d_up = mdata[2 * local_bin];
                    float d_dn = mdata[2 * local_bin + 1];
                    delta += histosys_delta(alpha, d_up, d_dn);
                }
            }

            expected_bin += (nom + delta) * factor;
        }

        /* Clamp expected >= 1e-10 */
        if (expected_bin < 1e-10f) expected_bin = 1e-10f;

        /* Phase 2: Poisson NLL for this bin */
        float obs = my_obs[bin];
        float mask = my_mask[bin];
        float lnfact = my_lnf[bin];
        float nll_bin = expected_bin + mask * (lnfact - obs * log(expected_bin));
        local_nll += nll_bin;

        /* Phase 3: Gradient accumulation */
        float w = 1.0f - obs / expected_bin;

        for (uint s = 0; s < n_samples; s++) {
            GpuSampleInfo sinfo = g_samples[s];
            if (bin < sinfo.main_bin_offset || bin >= sinfo.main_bin_offset + sinfo.n_bins) {
                continue;
            }
            uint local_bin2 = bin - sinfo.main_bin_offset;
            uint sample_bin2 = sinfo.first_sample_bin + local_bin2;

            float nom = g_nominal[sample_bin2];
            float delta = 0.0f;
            float factor = 1.0f;

            uint mod_start = g_mod_offsets[s];
            uint mod_end = g_mod_offsets[s + 1];

            for (uint m = mod_start; m < mod_end; m++) {
                GpuModifierDesc md = g_mod_descs[m];
                uint mtype = (uint)md.modifier_type;

                if (mtype == MOD_NORMFACTOR || mtype == MOD_LUMI) {
                    factor *= s_params[md.param_idx];
                }
                else if (mtype == MOD_SHAPESYS || mtype == MOD_STATERROR || mtype == MOD_SHAPEFACTOR) {
                    uint pidx = g_per_bin_pidx[md.data_offset + local_bin2];
                    factor *= s_params[pidx];
                }
                else if (mtype == MOD_NORMSYS) {
                    float alpha = s_params[md.param_idx];
                    const device float* mdata = g_mod_data + md.data_offset;
                    factor *= normsys_code4_value(alpha, mdata);
                }
                else if (mtype == MOD_HISTOSYS) {
                    float alpha = s_params[md.param_idx];
                    const device float* mdata = g_mod_data + md.data_offset;
                    float d_up = mdata[2 * local_bin2];
                    float d_dn = mdata[2 * local_bin2 + 1];
                    delta += histosys_delta(alpha, d_up, d_dn);
                }
            }

            float base = nom + delta;
            float mu_sample = base * factor;

            /* Gradient per modifier */
            for (uint m = mod_start; m < mod_end; m++) {
                GpuModifierDesc md = g_mod_descs[m];
                uint mtype = (uint)md.modifier_type;

                if (mtype == MOD_NORMFACTOR || mtype == MOD_LUMI) {
                    float p = s_params[md.param_idx];
                    if (fabs(p) > 1e-30f) {
                        float dmu_dp = mu_sample / p;
                        atomic_fetch_add_explicit(&my_grad[md.param_idx], w * dmu_dp, memory_order_relaxed);
                    }
                }
                else if (mtype == MOD_SHAPESYS || mtype == MOD_STATERROR || mtype == MOD_SHAPEFACTOR) {
                    uint pidx = g_per_bin_pidx[md.data_offset + local_bin2];
                    float gamma = s_params[pidx];
                    if (fabs(gamma) > 1e-30f) {
                        float dmu_dg = mu_sample / gamma;
                        atomic_fetch_add_explicit(&my_grad[pidx], w * dmu_dg, memory_order_relaxed);
                    }
                }
                else if (mtype == MOD_NORMSYS) {
                    float alpha = s_params[md.param_idx];
                    const device float* mdata = g_mod_data + md.data_offset;
                    float f = normsys_code4_value(alpha, mdata);
                    float deriv_ratio = normsys_code4_deriv_over_val(alpha, f, mdata);
                    atomic_fetch_add_explicit(&my_grad[md.param_idx], w * mu_sample * deriv_ratio, memory_order_relaxed);
                }
                else if (mtype == MOD_HISTOSYS) {
                    float alpha = s_params[md.param_idx];
                    const device float* mdata = g_mod_data + md.data_offset;
                    float d_up = mdata[2 * local_bin2];
                    float d_dn = mdata[2 * local_bin2 + 1];
                    float dprime = histosys_delta_deriv(alpha, d_up, d_dn);
                    atomic_fetch_add_explicit(&my_grad[md.param_idx], w * factor * dprime, memory_order_relaxed);
                }
            }
        }
    }

    /* ----- Phase 4: Barlow-Beeston auxiliary Poisson constraints ----- */
    for (uint a = tid; a < n_aux_poisson; a += block_size) {
        MetalAuxPoissonEntry aux = g_aux_poisson[a];
        uint gidx = (uint)aux.gamma_param_idx;
        float gamma = s_params[gidx];
        float tau = aux.tau;
        float obs_aux = aux.observed_aux;
        float exp_aux = gamma * tau;
        if (exp_aux < 1e-10f) exp_aux = 1e-10f;

        if (obs_aux > 0.0f) {
            /* lgamma precomputed on CPU (Metal has no lgamma) */
            local_nll += exp_aux - obs_aux * log(exp_aux) + aux.lgamma_obs;
        } else {
            local_nll += exp_aux;
        }

        float grad_aux = tau * (1.0f - obs_aux / exp_aux);
        atomic_fetch_add_explicit(&my_grad[gidx], grad_aux, memory_order_relaxed);
    }

    /* ----- Phase 5: Gaussian constraints ----- */
    for (uint g = tid; g < n_gauss; g += block_size) {
        MetalGaussConstraintEntry gc = g_gauss[g];
        uint pidx = (uint)gc.param_idx;
        float val = s_params[pidx];
        float pull = (val - gc.center) * gc.inv_width;
        local_nll += 0.5f * pull * pull;
        atomic_fetch_add_explicit(&my_grad[pidx], pull * gc.inv_width, memory_order_relaxed);
    }

    /* ----- Phase 6: Block-wide NLL reduction ----- */
    s_nll[tid] = local_nll;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_nll[tid] += s_nll[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        g_nll_out[toy_idx] = s_nll[0] + constraint_const;
    }
}

/* ---------- NLL-only kernel (no gradient) -------------------------------- */

kernel void batch_nll_only(
    const device float* g_params           [[buffer(0)]],
    const device float* g_observed         [[buffer(1)]],
    const device float* g_ln_facts         [[buffer(2)]],
    const device float* g_obs_mask         [[buffer(3)]],
    const device float* g_nominal          [[buffer(4)]],
    const device GpuSampleInfo* g_samples  [[buffer(5)]],
    const device GpuModifierDesc* g_mod_descs [[buffer(6)]],
    const device uint* g_mod_offsets       [[buffer(7)]],
    const device uint* g_per_bin_pidx      [[buffer(8)]],
    const device float* g_mod_data         [[buffer(9)]],
    const device MetalAuxPoissonEntry* g_aux_poisson [[buffer(10)]],
    const device MetalGaussConstraintEntry* g_gauss  [[buffer(11)]],
    device float* g_nll_out                [[buffer(12)]],
    constant ScalarArgs& args              [[buffer(14)]],
    uint toy_idx       [[threadgroup_position_in_grid]],
    uint tid           [[thread_position_in_threadgroup]],
    uint block_size    [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    uint n_params = args.n_params;
    uint n_main_bins = args.n_main_bins;
    uint n_samples = args.n_samples;
    uint n_aux_poisson = args.n_aux_poisson;
    uint n_gauss = args.n_gauss;
    float constraint_const = args.constraint_const;

    threadgroup float* s_params = shared;
    threadgroup float* s_nll = shared + n_params;

    const device float* my_params = g_params + toy_idx * n_params;
    for (uint i = tid; i < n_params; i += block_size) {
        s_params[i] = my_params[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device float* my_obs = g_observed + toy_idx * n_main_bins;
    const device float* my_lnf = g_ln_facts + toy_idx * n_main_bins;
    const device float* my_mask = g_obs_mask + toy_idx * n_main_bins;

    float local_nll = 0.0f;

    for (uint bin = tid; bin < n_main_bins; bin += block_size) {
        float expected_bin = 0.0f;

        for (uint s = 0; s < n_samples; s++) {
            GpuSampleInfo sinfo = g_samples[s];
            if (bin < sinfo.main_bin_offset || bin >= sinfo.main_bin_offset + sinfo.n_bins) {
                continue;
            }
            uint local_bin = bin - sinfo.main_bin_offset;
            uint sample_bin = sinfo.first_sample_bin + local_bin;

            float nom = g_nominal[sample_bin];
            float delta = 0.0f;
            float factor = 1.0f;

            uint mod_start = g_mod_offsets[s];
            uint mod_end = g_mod_offsets[s + 1];

            for (uint m = mod_start; m < mod_end; m++) {
                GpuModifierDesc md = g_mod_descs[m];
                uint mtype = (uint)md.modifier_type;

                if (mtype == MOD_NORMFACTOR || mtype == MOD_LUMI) {
                    factor *= s_params[md.param_idx];
                }
                else if (mtype == MOD_SHAPESYS || mtype == MOD_STATERROR || mtype == MOD_SHAPEFACTOR) {
                    uint pidx = g_per_bin_pidx[md.data_offset + local_bin];
                    factor *= s_params[pidx];
                }
                else if (mtype == MOD_NORMSYS) {
                    float alpha = s_params[md.param_idx];
                    const device float* mdata = g_mod_data + md.data_offset;
                    factor *= normsys_code4_value(alpha, mdata);
                }
                else if (mtype == MOD_HISTOSYS) {
                    float alpha = s_params[md.param_idx];
                    const device float* mdata = g_mod_data + md.data_offset;
                    float d_up = mdata[2 * local_bin];
                    float d_dn = mdata[2 * local_bin + 1];
                    delta += histosys_delta(alpha, d_up, d_dn);
                }
            }

            expected_bin += (nom + delta) * factor;
        }

        if (expected_bin < 1e-10f) expected_bin = 1e-10f;

        float obs = my_obs[bin];
        float mask = my_mask[bin];
        float lnfact = my_lnf[bin];
        local_nll += expected_bin + mask * (lnfact - obs * log(expected_bin));
    }

    /* Auxiliary Poisson */
    for (uint a = tid; a < n_aux_poisson; a += block_size) {
        MetalAuxPoissonEntry aux = g_aux_poisson[a];
        float gamma = s_params[(uint)aux.gamma_param_idx];
        float exp_aux = gamma * aux.tau;
        if (exp_aux < 1e-10f) exp_aux = 1e-10f;
        if (aux.observed_aux > 0.0f) {
            local_nll += exp_aux - aux.observed_aux * log(exp_aux) + aux.lgamma_obs;
        } else {
            local_nll += exp_aux;
        }
    }

    /* Gaussian constraints */
    for (uint g = tid; g < n_gauss; g += block_size) {
        MetalGaussConstraintEntry gc = g_gauss[g];
        float val = s_params[(uint)gc.param_idx];
        float pull = (val - gc.center) * gc.inv_width;
        local_nll += 0.5f * pull * pull;
    }

    /* Reduction */
    s_nll[tid] = local_nll;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_nll[tid] += s_nll[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        g_nll_out[toy_idx] = s_nll[0] + constraint_const;
    }
}
