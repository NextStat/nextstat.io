/**
 * Differentiable NLL + Signal Gradient Metal Kernel for profiled fitting.
 *
 * Port of differentiable_nll_grad.cu to Metal Shading Language (MSL).
 * ALL computation in float (f32) — Apple Silicon has no hardware f64.
 *
 * Key differences from batch_nll_grad.metal:
 *   1. Single model (1 threadgroup), not batch
 *   2. Reads signal sample nominal from an external buffer (uploaded from CPU)
 *   3. Writes ∂NLL/∂signal_bins into a separate output buffer
 *   4. Multi-channel signal: signal may appear in multiple channels
 *
 * Architecture: single threadgroup (1 model).
 * Threads process bins via grid-stride loop.
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

struct DiffScalarArgs {
    uint n_params;
    uint n_main_bins;
    uint n_samples;
    uint n_aux_poisson;
    uint n_gauss;
    uint n_signal_entries;
    float constraint_const;
};

/* ---------- Helper: NormSys Code4 interpolation ------------------------- */

float normsys_code4_value(
    float alpha,
    const device float* mdata
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

/* ---------- Main differentiable NLL + gradient kernel -------------------- */

kernel void differentiable_nll_grad(
    /* Nuisance parameters */
    const device float* g_params               [[buffer(0)]],   /* [n_params] */
    /* Observed data */
    const device float* g_observed             [[buffer(1)]],   /* [n_main_bins] */
    const device float* g_ln_facts             [[buffer(2)]],   /* [n_main_bins] */
    const device float* g_obs_mask             [[buffer(3)]],   /* [n_main_bins] */
    /* Static model buffers */
    const device float* g_nominal              [[buffer(4)]],   /* [n_sample_bins] */
    const device GpuSampleInfo* g_samples      [[buffer(5)]],   /* [n_samples] */
    const device GpuModifierDesc* g_mod_descs  [[buffer(6)]],
    const device uint* g_mod_offsets           [[buffer(7)]],
    const device uint* g_per_bin_pidx          [[buffer(8)]],
    const device float* g_mod_data             [[buffer(9)]],
    const device MetalAuxPoissonEntry* g_aux_poisson [[buffer(10)]],
    const device MetalGaussConstraintEntry* g_gauss  [[buffer(11)]],
    /* Output: NLL scalar + gradient w.r.t. nuisance params */
    device float* g_nll_out                    [[buffer(12)]],   /* [1] */
    device atomic_float* g_grad_params_out     [[buffer(13)]],   /* [n_params] */
    /* External signal buffer (uploaded from CPU, f32) */
    const device float* g_external_signal      [[buffer(14)]],   /* [total_signal_bins] */
    /* Signal gradient output */
    device atomic_float* g_grad_signal_out     [[buffer(15)]],   /* [total_signal_bins] */
    /* Multi-channel signal entry arrays */
    const device uint* g_signal_sample_indices [[buffer(16)]],   /* [n_signal_entries] */
    const device uint* g_signal_first_bins     [[buffer(17)]],   /* [n_signal_entries] */
    const device uint* g_signal_n_bins_arr     [[buffer(18)]],   /* [n_signal_entries] */
    /* Scalar args */
    constant DiffScalarArgs& args              [[buffer(19)]],
    /* Thread IDs */
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
    uint n_signal_entries = args.n_signal_entries;
    float constraint_const = args.constraint_const;

    threadgroup float* s_params = shared;
    threadgroup float* s_nll = shared + n_params;

    /* Load params into threadgroup memory */
    for (uint i = tid; i < n_params; i += block_size) {
        s_params[i] = g_params[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_nll = 0.0f;

    /* ----- Main bin loop ----- */
    for (uint bin = tid; bin < n_main_bins; bin += block_size) {
        float expected_bin = 0.0f;

        /* Track signal sample's factor for gradient computation */
        float signal_factor = 0.0f;
        bool has_signal = false;
        uint sig_local_bin = 0;

        for (uint s = 0; s < n_samples; s++) {
            GpuSampleInfo sinfo = g_samples[s];
            if (bin < sinfo.main_bin_offset || bin >= sinfo.main_bin_offset + sinfo.n_bins) {
                continue;
            }
            uint local_bin = bin - sinfo.main_bin_offset;
            uint sample_bin = sinfo.first_sample_bin + local_bin;

            /* Check if this sample+bin is in a signal entry */
            float nom = g_nominal[sample_bin];
            bool is_signal = false;
            uint sig_local_bin_candidate = 0;
            uint sig_global_offset = 0;
            for (uint se = 0; se < n_signal_entries; se++) {
                uint se_sidx = g_signal_sample_indices[se];
                uint se_first = g_signal_first_bins[se];
                uint se_nbins = g_signal_n_bins_arr[se];
                if (s == se_sidx && bin >= se_first && bin < se_first + se_nbins) {
                    nom = g_external_signal[sig_global_offset + (bin - se_first)];
                    is_signal = true;
                    sig_local_bin_candidate = sig_global_offset + (bin - se_first);
                    break;
                }
                sig_global_offset += se_nbins;
            }

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

            float sample_expected = (nom + delta) * factor;
            expected_bin += sample_expected;

            if (is_signal) {
                has_signal = true;
                sig_local_bin = sig_local_bin_candidate;
                signal_factor = factor;
            }
        }

        /* Clamp */
        if (expected_bin < 1e-10f) expected_bin = 1e-10f;

        /* Poisson NLL */
        float obs = g_observed[bin];
        float mask = g_obs_mask[bin];
        float lnfact = g_ln_facts[bin];
        float nll_bin = expected_bin + mask * (lnfact - obs * log(expected_bin));
        local_nll += nll_bin;

        /* Weight for gradient: dNLL/d(expected) = 1 - obs/expected */
        float w = 1.0f - obs / expected_bin;

        /* ----- Gradient w.r.t. signal bins ----- */
        if (has_signal) {
            atomic_fetch_add_explicit(&g_grad_signal_out[sig_local_bin], w * signal_factor, memory_order_relaxed);
        }

        /* ----- Gradient w.r.t. nuisance parameters ----- */
        for (uint s = 0; s < n_samples; s++) {
            GpuSampleInfo sinfo = g_samples[s];
            if (bin < sinfo.main_bin_offset || bin >= sinfo.main_bin_offset + sinfo.n_bins) {
                continue;
            }
            uint local_bin2 = bin - sinfo.main_bin_offset;
            uint sample_bin2 = sinfo.first_sample_bin + local_bin2;

            /* Re-read nominal (signal or standard) */
            float nom = g_nominal[sample_bin2];
            uint sig_global_offset2 = 0;
            for (uint se = 0; se < n_signal_entries; se++) {
                uint se_sidx = g_signal_sample_indices[se];
                uint se_first = g_signal_first_bins[se];
                uint se_nbins = g_signal_n_bins_arr[se];
                if (s == se_sidx && bin >= se_first && bin < se_first + se_nbins) {
                    nom = g_external_signal[sig_global_offset2 + (bin - se_first)];
                    break;
                }
                sig_global_offset2 += se_nbins;
            }

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

            /* Gradient for each modifier */
            for (uint m = mod_start; m < mod_end; m++) {
                GpuModifierDesc md = g_mod_descs[m];
                uint mtype = (uint)md.modifier_type;

                if (mtype == MOD_NORMFACTOR || mtype == MOD_LUMI) {
                    float p = s_params[md.param_idx];
                    if (fabs(p) > 1e-30f) {
                        atomic_fetch_add_explicit(&g_grad_params_out[md.param_idx], w * mu_sample / p, memory_order_relaxed);
                    }
                }
                else if (mtype == MOD_SHAPESYS || mtype == MOD_STATERROR || mtype == MOD_SHAPEFACTOR) {
                    uint pidx = g_per_bin_pidx[md.data_offset + local_bin2];
                    float gamma = s_params[pidx];
                    if (fabs(gamma) > 1e-30f) {
                        atomic_fetch_add_explicit(&g_grad_params_out[pidx], w * mu_sample / gamma, memory_order_relaxed);
                    }
                }
                else if (mtype == MOD_NORMSYS) {
                    float alpha = s_params[md.param_idx];
                    const device float* mdata = g_mod_data + md.data_offset;
                    float f = normsys_code4_value(alpha, mdata);
                    float dr = normsys_code4_deriv_over_val(alpha, f, mdata);
                    atomic_fetch_add_explicit(&g_grad_params_out[md.param_idx], w * mu_sample * dr, memory_order_relaxed);
                }
                else if (mtype == MOD_HISTOSYS) {
                    float alpha = s_params[md.param_idx];
                    const device float* mdata = g_mod_data + md.data_offset;
                    float d_up = mdata[2 * local_bin2];
                    float d_dn = mdata[2 * local_bin2 + 1];
                    float dprime = histosys_delta_deriv(alpha, d_up, d_dn);
                    atomic_fetch_add_explicit(&g_grad_params_out[md.param_idx], w * factor * dprime, memory_order_relaxed);
                }
            }
        }
    }

    /* ----- Auxiliary Poisson constraints ----- */
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
        atomic_fetch_add_explicit(&g_grad_params_out[gidx], tau * (1.0f - obs_aux / exp_aux), memory_order_relaxed);
    }

    /* ----- Gaussian constraints ----- */
    for (uint g = tid; g < n_gauss; g += block_size) {
        MetalGaussConstraintEntry gc = g_gauss[g];
        uint pidx = (uint)gc.param_idx;
        float val = s_params[pidx];
        float pull = (val - gc.center) * gc.inv_width;
        local_nll += 0.5f * pull * pull;
        atomic_fetch_add_explicit(&g_grad_params_out[pidx], pull * gc.inv_width, memory_order_relaxed);
    }

    /* ----- Block-wide NLL reduction ----- */
    s_nll[tid] = local_nll;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_nll[tid] += s_nll[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        g_nll_out[0] = s_nll[0] + constraint_const;
    }
}
