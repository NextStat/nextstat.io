/**
 * Fused NLL + Analytical Gradient CUDA Kernel for HistFactory models.
 *
 * Architecture: 1 CUDA block = 1 toy experiment.
 * Threads in block process bins via grid-stride loop.
 * Parameters cached in shared memory (~2KB for 250 params).
 *
 * Supports all pyhf modifier types:
 *   0 = NormFactor    (scalar multiplicative)
 *   1 = ShapeSys      (per-bin multiplicative gamma, Barlow-Beeston)
 *   2 = ShapeFactor   (per-bin multiplicative free factor)
 *   3 = NormSys       (Code4 interpolation, scalar)
 *   4 = HistoSys      (Code4p additive delta, per-bin)
 *   5 = StatError     (per-bin multiplicative gamma)
 *   6 = Lumi          (scalar multiplicative)
 */

#include <math.h>

/* ---------- Modifier type constants (must match GpuModifierType in Rust) --- */
#define MOD_NORMFACTOR  0
#define MOD_SHAPESYS    1
#define MOD_SHAPEFACTOR 2
#define MOD_NORMSYS     3
#define MOD_HISTOSYS    4
#define MOD_STATERROR   5
#define MOD_LUMI        6

/* ---------- Struct mirrors of Rust #[repr(C)] types ---------------------- */

struct GpuSampleInfo {
    unsigned int first_sample_bin;
    unsigned int n_bins;
    unsigned int main_bin_offset;
    unsigned int n_modifiers;
};

struct GpuModifierDesc {
    unsigned int param_idx;
    unsigned char modifier_type;
    unsigned char is_per_bin;
    unsigned short _pad;
    unsigned int data_offset;
    unsigned int n_bins;
};

struct GpuAuxPoissonEntry {
    unsigned short gamma_param_idx;
    unsigned short _pad;
    float _pad2;
    double tau;
    double observed_aux;
};

struct GpuGaussConstraintEntry {
    unsigned short param_idx;
    unsigned short _pad;
    float _pad2;
    double center;
    double inv_width;
};

/* ---------- Device helper: NormSys Code4 interpolation ------------------- */

/**
 * normsys_code4(alpha, hi, lo) — matches pyhf code4 interpolation.
 *
 * modifier_data layout at data_offset:
 *   [0..5] = polynomial coefficients a[0..5]
 *   [6]    = ln(hi_factor)
 *   [7]    = ln(lo_factor)
 */
__device__ double normsys_code4_value(
    double alpha,
    const double* __restrict__ mdata  /* 8 doubles at data_offset */
) {
    double abs_alpha = fabs(alpha);
    if (abs_alpha >= 1.0) {
        /* Exponential regime: base^|alpha| */
        double ln_base = (alpha >= 0.0) ? mdata[6] : mdata[7];
        return exp(abs_alpha * ln_base);
    }
    /* Polynomial regime: 1 + sum_{k=1}^{6} a[k-1] * alpha^k */
    double a1 = alpha;
    double a2 = a1 * alpha;
    double a3 = a2 * alpha;
    double a4 = a3 * alpha;
    double a5 = a4 * alpha;
    double a6 = a5 * alpha;
    return 1.0
        + mdata[0] * a1
        + mdata[1] * a2
        + mdata[2] * a3
        + mdata[3] * a4
        + mdata[4] * a5
        + mdata[5] * a6;
}

/**
 * Derivative of normsys_code4 w.r.t. alpha.
 * Returns f'(alpha) / f(alpha) — the logarithmic derivative,
 * which when multiplied by mu_b gives d(mu_b)/d(alpha).
 */
__device__ double normsys_code4_deriv_over_val(
    double alpha,
    double factor,  /* normsys_code4_value result */
    const double* __restrict__ mdata
) {
    double abs_alpha = fabs(alpha);
    if (abs_alpha >= 1.0) {
        double ln_base = (alpha >= 0.0) ? mdata[6] : mdata[7];
        double sign_a = (alpha >= 0.0) ? 1.0 : -1.0;
        return sign_a * ln_base;  /* d/dalpha [base^|a|] / base^|a| = sign(a)*ln(base) */
    }
    /* Polynomial derivative: a1 + 2*a2*alpha + ... + 6*a6*alpha^5 */
    double fprime = mdata[0]
        + 2.0 * mdata[1] * alpha
        + 3.0 * mdata[2] * alpha * alpha
        + 4.0 * mdata[3] * alpha * alpha * alpha
        + 5.0 * mdata[4] * alpha * alpha * alpha * alpha
        + 6.0 * mdata[5] * alpha * alpha * alpha * alpha * alpha;
    return fprime / factor;
}

/* ---------- Device helper: HistoSys Code4p delta ------------------------- */

/**
 * histosys_code4p_delta(alpha, delta_up, delta_dn) — per-bin additive delta.
 *
 * modifier_data layout at data_offset + 2*bin_idx:
 *   [0] = delta_up = hi_template - nominal
 *   [1] = delta_dn = nominal - lo_template
 */
__device__ double histosys_delta(double alpha, double delta_up, double delta_dn) {
    if (alpha > 1.0) return delta_up * alpha;
    if (alpha < -1.0) return delta_dn * alpha;

    double S = 0.5 * (delta_up + delta_dn);
    double A = 0.0625 * (delta_up - delta_dn);
    double a2 = alpha * alpha;
    double tmp = a2 * (a2 * (3.0 * a2 - 10.0) + 15.0);
    return alpha * S + tmp * A;
}

/**
 * Derivative of histosys delta w.r.t. alpha.
 */
__device__ double histosys_delta_deriv(double alpha, double delta_up, double delta_dn) {
    if (alpha > 1.0) return delta_up;
    if (alpha < -1.0) return delta_dn;

    double S = 0.5 * (delta_up + delta_dn);
    double A = 0.0625 * (delta_up - delta_dn);
    /* d/dalpha [alpha*S + (3a^6 - 10a^4 + 15a^2)*A] =
     *   S + (18a^5 - 40a^3 + 30a) * A */
    double a2 = alpha * alpha;
    double a3 = a2 * alpha;
    double a5 = a3 * a2;
    return S + (18.0 * a5 - 40.0 * a3 + 30.0 * alpha) * A;
}

/* ---------- Main fused NLL + Gradient kernel ----------------------------- */

extern "C" __global__ void batch_nll_grad(
    /* Dynamic per-iteration buffers */
    const double* __restrict__ g_params,       /* [n_toys × n_params] */
    const double* __restrict__ g_observed,     /* [n_toys × n_main_bins] */
    const double* __restrict__ g_ln_facts,     /* [n_toys × n_main_bins] */
    const double* __restrict__ g_obs_mask,     /* [n_toys × n_main_bins] */
    /* Static model buffers */
    const double* __restrict__ g_nominal,      /* [n_sample_bins] */
    const struct GpuSampleInfo* __restrict__ g_samples,  /* [n_samples] */
    const struct GpuModifierDesc* __restrict__ g_mod_descs,  /* [total_modifiers] */
    const unsigned int* __restrict__ g_mod_offsets,  /* [n_samples + 1] */
    const unsigned int* __restrict__ g_per_bin_pidx, /* per-bin param indices */
    const double* __restrict__ g_mod_data,     /* modifier extra data */
    const struct GpuAuxPoissonEntry* __restrict__ g_aux_poisson, /* [n_aux] */
    const struct GpuGaussConstraintEntry* __restrict__ g_gauss,  /* [n_gauss] */
    /* Output */
    double* __restrict__ g_nll_out,            /* [n_toys] */
    double* __restrict__ g_grad_out,           /* [n_toys × n_params] */
    /* Scalar metadata */
    unsigned int n_params,
    unsigned int n_main_bins,
    unsigned int n_samples,
    unsigned int n_aux_poisson,
    unsigned int n_gauss,
    double constraint_const
) {
    unsigned int toy_idx = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    /* Shared memory layout: params[n_params] | nll_scratch[block_size] */
    extern __shared__ double shared[];
    double* s_params = shared;
    double* s_nll = shared + n_params;

    /* ----- Phase 0: Load params into shared memory ----- */
    const double* my_params = g_params + (size_t)toy_idx * n_params;
    for (unsigned int i = tid; i < n_params; i += block_size) {
        s_params[i] = my_params[i];
    }
    __syncthreads();

    /* Toy-specific observed data pointers */
    const double* my_obs = g_observed + (size_t)toy_idx * n_main_bins;
    const double* my_lnf = g_ln_facts + (size_t)toy_idx * n_main_bins;
    const double* my_mask = g_obs_mask + (size_t)toy_idx * n_main_bins;
    double* my_grad = g_grad_out + (size_t)toy_idx * n_params;

    /* ----- Phase 1: Compute expected counts (per-bin) ----- */
    /* We need a local array for expected[n_main_bins]. For models with
     * <1024 bins this fits in registers via the grid-stride loop.
     * For larger models we accumulate into global memory. */

    /* Since we cannot dynamically allocate per-thread expected array
     * on GPU, we process one bin at a time across all samples. */

    double local_nll = 0.0;

    for (unsigned int bin = tid; bin < n_main_bins; bin += block_size) {
        /* Accumulate expected for this main bin across all samples */
        double expected_bin = 0.0;

        for (unsigned int s = 0; s < n_samples; s++) {
            struct GpuSampleInfo sinfo = g_samples[s];
            if (bin < sinfo.main_bin_offset || bin >= sinfo.main_bin_offset + sinfo.n_bins) {
                continue;  /* This sample doesn't contribute to this bin */
            }
            unsigned int local_bin = bin - sinfo.main_bin_offset;
            unsigned int sample_bin = sinfo.first_sample_bin + local_bin;

            double nom = g_nominal[sample_bin];
            double delta = 0.0;
            double factor = 1.0;

            /* Apply modifiers for this sample */
            unsigned int mod_start = g_mod_offsets[s];
            unsigned int mod_end = g_mod_offsets[s + 1];

            for (unsigned int m = mod_start; m < mod_end; m++) {
                struct GpuModifierDesc md = g_mod_descs[m];
                unsigned char mtype = md.modifier_type;

                if (mtype == MOD_NORMFACTOR || mtype == MOD_LUMI) {
                    /* Scalar multiplicative */
                    factor *= s_params[md.param_idx];
                }
                else if (mtype == MOD_SHAPESYS || mtype == MOD_STATERROR || mtype == MOD_SHAPEFACTOR) {
                    /* Per-bin multiplicative gamma */
                    unsigned int pidx = g_per_bin_pidx[md.data_offset + local_bin];
                    factor *= s_params[pidx];
                }
                else if (mtype == MOD_NORMSYS) {
                    /* Code4 interpolation */
                    double alpha = s_params[md.param_idx];
                    const double* mdata = g_mod_data + md.data_offset;
                    double f = normsys_code4_value(alpha, mdata);
                    factor *= f;
                }
                else if (mtype == MOD_HISTOSYS) {
                    /* Code4p additive delta */
                    double alpha = s_params[md.param_idx];
                    const double* mdata = g_mod_data + md.data_offset;
                    double d_up = mdata[2 * local_bin];
                    double d_dn = mdata[2 * local_bin + 1];
                    delta += histosys_delta(alpha, d_up, d_dn);
                }
            }

            expected_bin += (nom + delta) * factor;
        }

        /* Clamp expected >= 1e-10 */
        if (expected_bin < 1e-10) expected_bin = 1e-10;

        /* ----- Phase 2: Poisson NLL for this bin ----- */
        double obs = my_obs[bin];
        double mask = my_mask[bin];
        double lnfact = my_lnf[bin];

        /* nll_bin = mu + mask * (lnfact - obs * ln(mu)) */
        double nll_bin = expected_bin + mask * (lnfact - obs * log(expected_bin));
        local_nll += nll_bin;

        /* ----- Phase 3: Gradient accumulation for this bin ----- */
        /* Weight w = 1 - obs / mu (= dNLL/dmu for Poisson) */
        double w = 1.0 - obs / expected_bin;

        /* Re-iterate samples to compute d(expected)/d(param) */
        for (unsigned int s = 0; s < n_samples; s++) {
            struct GpuSampleInfo sinfo = g_samples[s];
            if (bin < sinfo.main_bin_offset || bin >= sinfo.main_bin_offset + sinfo.n_bins) {
                continue;
            }
            unsigned int local_bin2 = bin - sinfo.main_bin_offset;
            unsigned int sample_bin2 = sinfo.first_sample_bin + local_bin2;

            double nom = g_nominal[sample_bin2];

            /* Recompute delta, factor, and per-modifier products for gradient */
            double delta = 0.0;
            double factor = 1.0;

            /* First pass: compute delta and factor (same as phase 1) */
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

            double base = nom + delta;  /* before factor */
            double mu_sample = base * factor;  /* this sample's contribution */

            /* Second pass: compute gradient for each modifier */
            for (unsigned int m = mod_start; m < mod_end; m++) {
                struct GpuModifierDesc md = g_mod_descs[m];
                unsigned char mtype = md.modifier_type;

                if (mtype == MOD_NORMFACTOR || mtype == MOD_LUMI) {
                    /* d(mu)/dp = mu_sample / p */
                    double p = s_params[md.param_idx];
                    if (fabs(p) > 1e-30) {
                        double dmu_dp = mu_sample / p;
                        atomicAdd(&my_grad[md.param_idx], w * dmu_dp);
                    }
                }
                else if (mtype == MOD_SHAPESYS || mtype == MOD_STATERROR || mtype == MOD_SHAPEFACTOR) {
                    /* d(mu)/d(gamma) = mu_sample / gamma */
                    unsigned int pidx = g_per_bin_pidx[md.data_offset + local_bin2];
                    double gamma = s_params[pidx];
                    if (fabs(gamma) > 1e-30) {
                        double dmu_dg = mu_sample / gamma;
                        atomicAdd(&my_grad[pidx], w * dmu_dg);
                    }
                }
                else if (mtype == MOD_NORMSYS) {
                    /* d(mu)/d(alpha) = mu_sample * f'(alpha)/f(alpha) */
                    double alpha = s_params[md.param_idx];
                    const double* mdata = g_mod_data + md.data_offset;
                    double f = normsys_code4_value(alpha, mdata);
                    double deriv_ratio = normsys_code4_deriv_over_val(alpha, f, mdata);
                    atomicAdd(&my_grad[md.param_idx], w * mu_sample * deriv_ratio);
                }
                else if (mtype == MOD_HISTOSYS) {
                    /* d(mu)/d(alpha) = factor * delta'(alpha) */
                    double alpha = s_params[md.param_idx];
                    const double* mdata = g_mod_data + md.data_offset;
                    double d_up = mdata[2 * local_bin2];
                    double d_dn = mdata[2 * local_bin2 + 1];
                    double dprime = histosys_delta_deriv(alpha, d_up, d_dn);
                    atomicAdd(&my_grad[md.param_idx], w * factor * dprime);
                }
            }
        }
    }

    /* ----- Phase 4: Barlow-Beeston auxiliary Poisson constraints ----- */
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

        /* d(NLL_aux)/d(gamma) = tau * (1 - obs_aux / (gamma * tau)) */
        double grad_aux = tau * (1.0 - obs_aux / exp_aux);
        atomicAdd(&my_grad[gidx], grad_aux);
    }

    /* ----- Phase 5: Gaussian constraints ----- */
    for (unsigned int g = tid; g < n_gauss; g += block_size) {
        struct GpuGaussConstraintEntry gc = g_gauss[g];
        unsigned int pidx = (unsigned int)gc.param_idx;
        double val = s_params[pidx];
        double pull = (val - gc.center) * gc.inv_width;

        local_nll += 0.5 * pull * pull;

        /* d(NLL_gauss)/dp = pull * inv_width */
        atomicAdd(&my_grad[pidx], pull * gc.inv_width);
    }

    /* ----- Phase 6: Block-wide NLL reduction ----- */
    s_nll[tid] = local_nll;
    __syncthreads();

    /* Standard parallel reduction */
    for (unsigned int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_nll[tid] += s_nll[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        g_nll_out[toy_idx] = s_nll[0] + constraint_const;
    }
}

/* ---------- NLL-only kernel (no gradient) -------------------------------- */

extern "C" __global__ void batch_nll_only(
    const double* __restrict__ g_params,
    const double* __restrict__ g_observed,
    const double* __restrict__ g_ln_facts,
    const double* __restrict__ g_obs_mask,
    const double* __restrict__ g_nominal,
    const struct GpuSampleInfo* __restrict__ g_samples,
    const struct GpuModifierDesc* __restrict__ g_mod_descs,
    const unsigned int* __restrict__ g_mod_offsets,
    const unsigned int* __restrict__ g_per_bin_pidx,
    const double* __restrict__ g_mod_data,
    const struct GpuAuxPoissonEntry* __restrict__ g_aux_poisson,
    const struct GpuGaussConstraintEntry* __restrict__ g_gauss,
    double* __restrict__ g_nll_out,
    unsigned int n_params,
    unsigned int n_main_bins,
    unsigned int n_samples,
    unsigned int n_aux_poisson,
    unsigned int n_gauss,
    double constraint_const
) {
    unsigned int toy_idx = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    extern __shared__ double shared[];
    double* s_params = shared;
    double* s_nll = shared + n_params;

    const double* my_params = g_params + (size_t)toy_idx * n_params;
    for (unsigned int i = tid; i < n_params; i += block_size) {
        s_params[i] = my_params[i];
    }
    __syncthreads();

    const double* my_obs = g_observed + (size_t)toy_idx * n_main_bins;
    const double* my_lnf = g_ln_facts + (size_t)toy_idx * n_main_bins;
    const double* my_mask = g_obs_mask + (size_t)toy_idx * n_main_bins;

    double local_nll = 0.0;

    for (unsigned int bin = tid; bin < n_main_bins; bin += block_size) {
        double expected_bin = 0.0;

        for (unsigned int s = 0; s < n_samples; s++) {
            struct GpuSampleInfo sinfo = g_samples[s];
            if (bin < sinfo.main_bin_offset || bin >= sinfo.main_bin_offset + sinfo.n_bins) {
                continue;
            }
            unsigned int local_bin = bin - sinfo.main_bin_offset;
            unsigned int sample_bin = sinfo.first_sample_bin + local_bin;

            double nom = g_nominal[sample_bin];
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

            expected_bin += (nom + delta) * factor;
        }

        if (expected_bin < 1e-10) expected_bin = 1e-10;

        double obs = my_obs[bin];
        double mask = my_mask[bin];
        double lnfact = my_lnf[bin];
        local_nll += expected_bin + mask * (lnfact - obs * log(expected_bin));
    }

    /* Auxiliary Poisson */
    for (unsigned int a = tid; a < n_aux_poisson; a += block_size) {
        struct GpuAuxPoissonEntry aux = g_aux_poisson[a];
        double gamma = s_params[(unsigned int)aux.gamma_param_idx];
        double exp_aux = gamma * aux.tau;
        if (exp_aux < 1e-10) exp_aux = 1e-10;
        if (aux.observed_aux > 0.0) {
            local_nll += exp_aux - aux.observed_aux * log(exp_aux) + lgamma(aux.observed_aux + 1.0);
        } else {
            local_nll += exp_aux;
        }
    }

    /* Gaussian constraints */
    for (unsigned int g = tid; g < n_gauss; g += block_size) {
        struct GpuGaussConstraintEntry gc = g_gauss[g];
        double val = s_params[(unsigned int)gc.param_idx];
        double pull = (val - gc.center) * gc.inv_width;
        local_nll += 0.5 * pull * pull;
    }

    /* Reduction */
    s_nll[tid] = local_nll;
    __syncthreads();
    for (unsigned int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_nll[tid] += s_nll[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        g_nll_out[toy_idx] = s_nll[0] + constraint_const;
    }
}
