/**
 * Differentiable NLL + Signal Gradient CUDA Kernel for PyTorch integration.
 *
 * Based on batch_nll_grad.cu but with two key additions:
 *   1. Reads signal sample nominal from an external CUDA pointer (PyTorch tensor)
 *   2. Writes ∂NLL/∂signal_bins into an external CUDA pointer (PyTorch grad tensor)
 *
 * This enables zero-copy data flow between PyTorch and NextStat:
 *   - PyTorch allocates signal tensor on GPU
 *   - NextStat kernel reads signal bins directly (no H→D copy)
 *   - NextStat kernel writes gradient directly into PyTorch's grad buffer
 *
 * Architecture: single CUDA block (1 model, not batch).
 * Threads process bins via grid-stride loop.
 *
 * Multi-channel signal: the signal sample may appear in multiple channels.
 * Signal entries (arrays of sample_idx, first_bin, n_bins) describe each occurrence.
 * The external signal buffer is laid out as [ch0_bins..., ch1_bins..., ...].
 */

#include "common.cuh"

extern "C" __global__ void differentiable_nll_grad(
    /* Nuisance parameters (Host → Device by Rust) */
    const double* __restrict__ g_params,       /* [n_params] */
    /* Observed data (uploaded once) */
    const double* __restrict__ g_observed,     /* [n_main_bins] */
    const double* __restrict__ g_ln_facts,     /* [n_main_bins] */
    const double* __restrict__ g_obs_mask,     /* [n_main_bins] */
    /* Static model buffers */
    const double* __restrict__ g_nominal,      /* [n_sample_bins] */
    const struct GpuSampleInfo* __restrict__ g_samples,  /* [n_samples] */
    const struct GpuModifierDesc* __restrict__ g_mod_descs,
    const unsigned int* __restrict__ g_mod_offsets,
    const unsigned int* __restrict__ g_per_bin_pidx,
    const double* __restrict__ g_mod_data,
    const struct GpuAuxPoissonEntry* __restrict__ g_aux_poisson,
    const struct GpuGaussConstraintEntry* __restrict__ g_gauss,
    /* Output: NLL scalar + gradient w.r.t. nuisance params */
    double* __restrict__ g_nll_out,            /* [1] */
    double* __restrict__ g_grad_params_out,    /* [n_params] */
    /* === PyTorch zero-copy pointers === */
    const double* __restrict__ g_external_signal,  /* [total_signal_bins] — PyTorch signal tensor */
    double* __restrict__ g_grad_signal_out,        /* [total_signal_bins] — PyTorch grad tensor */
    /* Multi-channel signal entry arrays */
    const unsigned int* __restrict__ g_signal_sample_indices,  /* [n_signal_entries] */
    const unsigned int* __restrict__ g_signal_first_bins,      /* [n_signal_entries] */
    const unsigned int* __restrict__ g_signal_n_bins_arr,      /* [n_signal_entries] */
    unsigned int n_signal_entries,
    /* Scalar metadata */
    unsigned int n_params,
    unsigned int n_main_bins,
    unsigned int n_samples,
    unsigned int n_aux_poisson,
    unsigned int n_gauss,
    double constraint_const
) {
    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    /* Shared memory: params[n_params] | nll_scratch[block_size] */
    extern __shared__ double shared[];
    double* s_params = shared;
    double* s_nll = shared + n_params;

    /* Load params into shared memory */
    for (unsigned int i = tid; i < n_params; i += block_size) {
        s_params[i] = g_params[i];
    }
    __syncthreads();

    double local_nll = 0.0;

    /* ----- Main bin loop ----- */
    for (unsigned int bin = tid; bin < n_main_bins; bin += block_size) {
        double expected_bin = 0.0;

        /* Track signal sample's factor_product for gradient computation */
        double signal_factor = 0.0;
        int has_signal = 0;
        unsigned int sig_local_bin = 0;

        for (unsigned int s = 0; s < n_samples; s++) {
            struct GpuSampleInfo sinfo = g_samples[s];
            if (bin < sinfo.main_bin_offset || bin >= sinfo.main_bin_offset + sinfo.n_bins) {
                continue;
            }
            unsigned int local_bin = bin - sinfo.main_bin_offset;
            unsigned int sample_bin = sinfo.first_sample_bin + local_bin;

            /* Check if this sample+bin is in a signal entry */
            double nom = g_nominal[sample_bin];
            unsigned int sig_global_offset = 0;
            for (unsigned int se = 0; se < n_signal_entries; se++) {
                unsigned int se_sidx = g_signal_sample_indices[se];
                unsigned int se_first = g_signal_first_bins[se];
                unsigned int se_nbins = g_signal_n_bins_arr[se];
                if (s == se_sidx && bin >= se_first && bin < se_first + se_nbins) {
                    nom = g_external_signal[sig_global_offset + (bin - se_first)];
                    has_signal = 1;
                    sig_local_bin = sig_global_offset + (bin - se_first);
                    break;
                }
                sig_global_offset += se_nbins;
            }

            double delta = 0.0;
            double factor = 1.0;

            unsigned int mod_start = g_mod_offsets[s];
            unsigned int mod_end = g_mod_offsets[s + 1];

            for (unsigned int m = mod_start; m < mod_end; m++) {
                struct GpuModifierDesc md = g_mod_descs[m];
                unsigned char mtype = md.modifier_type;

                if (mtype == MOD_NORMFACTOR || mtype == MOD_LUMI) {
                    factor *= s_params[md.param_idx];
                }
                else if (mtype == MOD_SHAPESYS || mtype == MOD_STATERROR || mtype == MOD_SHAPEFACTOR) {
                    unsigned int pidx = g_per_bin_pidx[md.data_offset + local_bin];
                    factor *= s_params[pidx];
                }
                else if (mtype == MOD_NORMSYS) {
                    double alpha = s_params[md.param_idx];
                    const double* mdata = g_mod_data + md.data_offset;
                    factor *= normsys_code4_value(alpha, mdata);
                }
                else if (mtype == MOD_HISTOSYS) {
                    double alpha = s_params[md.param_idx];
                    const double* mdata = g_mod_data + md.data_offset;
                    double d_up = mdata[2 * local_bin];
                    double d_dn = mdata[2 * local_bin + 1];
                    delta += histosys_delta(alpha, d_up, d_dn);
                }
            }

            double sample_expected = (nom + delta) * factor;
            expected_bin += sample_expected;

            /* Remember signal factor for gradient (last match wins — only one per bin) */
            if (has_signal) {
                signal_factor = factor;
            }
        }

        /* Clamp */
        if (expected_bin < 1e-10) expected_bin = 1e-10;

        /* Poisson NLL */
        double obs = g_observed[bin];
        double mask = g_obs_mask[bin];
        double lnfact = g_ln_facts[bin];
        double nll_bin = expected_bin + mask * (lnfact - obs * log(expected_bin));
        local_nll += nll_bin;

        /* Weight for gradient: dNLL/d(expected) = 1 - obs/expected */
        double w = 1.0 - obs / expected_bin;

        /* ----- Gradient w.r.t. signal bins (zero-copy into PyTorch) ----- */
        if (has_signal) {
            atomicAdd(&g_grad_signal_out[sig_local_bin], w * signal_factor);
        }

        /* ----- Gradient w.r.t. nuisance parameters ----- */
        for (unsigned int s = 0; s < n_samples; s++) {
            struct GpuSampleInfo sinfo = g_samples[s];
            if (bin < sinfo.main_bin_offset || bin >= sinfo.main_bin_offset + sinfo.n_bins) {
                continue;
            }
            unsigned int local_bin2 = bin - sinfo.main_bin_offset;
            unsigned int sample_bin2 = sinfo.first_sample_bin + local_bin2;

            /* Re-read nominal (signal or standard) */
            double nom = g_nominal[sample_bin2];
            unsigned int sig_global_offset2 = 0;
            for (unsigned int se = 0; se < n_signal_entries; se++) {
                unsigned int se_sidx = g_signal_sample_indices[se];
                unsigned int se_first = g_signal_first_bins[se];
                unsigned int se_nbins = g_signal_n_bins_arr[se];
                if (s == se_sidx && bin >= se_first && bin < se_first + se_nbins) {
                    nom = g_external_signal[sig_global_offset2 + (bin - se_first)];
                    break;
                }
                sig_global_offset2 += se_nbins;
            }

            double delta = 0.0;
            double factor = 1.0;

            unsigned int mod_start = g_mod_offsets[s];
            unsigned int mod_end = g_mod_offsets[s + 1];

            for (unsigned int m = mod_start; m < mod_end; m++) {
                struct GpuModifierDesc md = g_mod_descs[m];
                unsigned char mtype = md.modifier_type;
                if (mtype == MOD_NORMFACTOR || mtype == MOD_LUMI) {
                    factor *= s_params[md.param_idx];
                }
                else if (mtype == MOD_SHAPESYS || mtype == MOD_STATERROR || mtype == MOD_SHAPEFACTOR) {
                    unsigned int pidx = g_per_bin_pidx[md.data_offset + local_bin2];
                    factor *= s_params[pidx];
                }
                else if (mtype == MOD_NORMSYS) {
                    double alpha = s_params[md.param_idx];
                    const double* mdata = g_mod_data + md.data_offset;
                    factor *= normsys_code4_value(alpha, mdata);
                }
                else if (mtype == MOD_HISTOSYS) {
                    double alpha = s_params[md.param_idx];
                    const double* mdata = g_mod_data + md.data_offset;
                    double d_up = mdata[2 * local_bin2];
                    double d_dn = mdata[2 * local_bin2 + 1];
                    delta += histosys_delta(alpha, d_up, d_dn);
                }
            }

            double base = nom + delta;
            double mu_sample = base * factor;

            /* Gradient for each modifier */
            for (unsigned int m = mod_start; m < mod_end; m++) {
                struct GpuModifierDesc md = g_mod_descs[m];
                unsigned char mtype = md.modifier_type;

                if (mtype == MOD_NORMFACTOR || mtype == MOD_LUMI) {
                    double p = s_params[md.param_idx];
                    if (fabs(p) > 1e-30) {
                        atomicAdd(&g_grad_params_out[md.param_idx], w * mu_sample / p);
                    }
                }
                else if (mtype == MOD_SHAPESYS || mtype == MOD_STATERROR || mtype == MOD_SHAPEFACTOR) {
                    unsigned int pidx = g_per_bin_pidx[md.data_offset + local_bin2];
                    double gamma = s_params[pidx];
                    if (fabs(gamma) > 1e-30) {
                        atomicAdd(&g_grad_params_out[pidx], w * mu_sample / gamma);
                    }
                }
                else if (mtype == MOD_NORMSYS) {
                    double alpha = s_params[md.param_idx];
                    const double* mdata = g_mod_data + md.data_offset;
                    double f = normsys_code4_value(alpha, mdata);
                    double dr = normsys_code4_deriv_over_val(alpha, f, mdata);
                    atomicAdd(&g_grad_params_out[md.param_idx], w * mu_sample * dr);
                }
                else if (mtype == MOD_HISTOSYS) {
                    double alpha = s_params[md.param_idx];
                    const double* mdata = g_mod_data + md.data_offset;
                    double d_up = mdata[2 * local_bin2];
                    double d_dn = mdata[2 * local_bin2 + 1];
                    double dprime = histosys_delta_deriv(alpha, d_up, d_dn);
                    atomicAdd(&g_grad_params_out[md.param_idx], w * factor * dprime);
                }
            }
        }
    }

    /* ----- Auxiliary Poisson constraints ----- */
    for (unsigned int a = tid; a < n_aux_poisson; a += block_size) {
        struct GpuAuxPoissonEntry aux = g_aux_poisson[a];
        unsigned int gidx = (unsigned int)aux.gamma_param_idx;
        double gamma = s_params[gidx];
        double tau = aux.tau;
        double obs_aux = aux.observed_aux;
        double exp_aux = gamma * tau;
        if (exp_aux < 1e-10) exp_aux = 1e-10;

        if (obs_aux > 0.0) {
            local_nll += exp_aux - obs_aux * log(exp_aux) + lgamma(obs_aux + 1.0);
        } else {
            local_nll += exp_aux;
        }
        atomicAdd(&g_grad_params_out[gidx], tau * (1.0 - obs_aux / exp_aux));
    }

    /* ----- Gaussian constraints ----- */
    for (unsigned int g = tid; g < n_gauss; g += block_size) {
        struct GpuGaussConstraintEntry gc = g_gauss[g];
        unsigned int pidx = (unsigned int)gc.param_idx;
        double val = s_params[pidx];
        double pull = (val - gc.center) * gc.inv_width;
        local_nll += 0.5 * pull * pull;
        atomicAdd(&g_grad_params_out[pidx], pull * gc.inv_width);
    }

    /* ----- Block-wide NLL reduction ----- */
    s_nll[tid] = local_nll;
    __syncthreads();
    for (unsigned int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_nll[tid] += s_nll[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_nll_out[0] = s_nll[0] + constraint_const;
    }
}
