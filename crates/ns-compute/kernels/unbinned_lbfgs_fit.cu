/**
 * GPU-native persistent L-BFGS optimizer for unbinned batch toy fitting.
 *
 * Architecture: 1 CUDA block = 1 toy = 1 complete L-BFGS optimization.
 * Single kernel launch runs the entire optimization loop on device.
 * Eliminates all host-device roundtrips per iteration.
 *
 * Memory layout:
 *   Shared: params[n_params] + scratch[block_size]
 *   Global per-toy: L-BFGS history, gradient, direction workspaces
 */

#include "unbinned_common.cuh"

#define MAX_LBFGS_M 16
#define ARMIJO_C1 1e-4

/* ========================================================================= */
/*  Device function: NLL + gradient for one toy (all threads participate)     */
/* ========================================================================= */

__device__ void compute_nll_and_grad(
    const double* __restrict__ s_params,
    const double* __restrict__ obs,
    unsigned int n_events,
    double obs_lo,
    double obs_hi,
    const struct GpuUnbinnedProcessDesc* __restrict__ g_procs,
    const struct GpuUnbinnedRateModifierDesc* __restrict__ g_rate_mods,
    const unsigned int* __restrict__ g_shape_pidx,
    const double* __restrict__ g_pdf_aux_f64,
    const struct GpuUnbinnedGaussConstraintEntry* __restrict__ g_gauss,
    double* __restrict__ nll_out,
    double* __restrict__ grad_out,
    double* __restrict__ s_scratch,
    unsigned int n_params,
    unsigned int n_procs,
    unsigned int total_rate_mods,
    unsigned int total_shape_params,
    unsigned int n_gauss,
    double constraint_const
) {
    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    double local_sum_logf = 0.0;
    double a = obs_lo;
    double b = obs_hi;

    for (unsigned int i = tid; i < n_events; i += block_size) {
        double x = obs[i];

        double max_term = -INFINITY;
        double sum_exp = 0.0;

        for (unsigned int p = 0; p < n_procs; p++) {
            struct GpuUnbinnedProcessDesc proc = g_procs[p];
            if (proc.obs_index != 0u) {
                continue;
            }

            double nu = 0.0;
            if (proc.yield_kind == YIELD_FIXED) {
                nu = proc.base_yield;
            } else if (proc.yield_kind == YIELD_PARAMETER) {
                nu = s_params[proc.yield_param_idx];
            } else if (proc.yield_kind == YIELD_SCALED) {
                nu = proc.base_yield * s_params[proc.yield_param_idx];
            } else {
                continue;
            }
            unsigned int mod_off = proc.rate_mod_offset;
            unsigned int nmods = proc.n_rate_mods;
            if (mod_off + nmods > total_rate_mods) {
                nmods = 0u;
            }
            for (unsigned int m = 0; m < nmods; m++) {
                double f, dlogf;
                rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dlogf);
                (void)dlogf;
                nu *= f;
            }
            if (!(nu > 0.0) || !isfinite(nu)) {
                continue;
            }

            double logp = -INFINITY;
            if (proc.pdf_kind == PDF_GAUSSIAN) {
                unsigned int off = proc.shape_param_offset;
                if (proc.n_shape_params != 2u || off + 1u >= total_shape_params) {
                    continue;
                }
                unsigned int mu_idx = g_shape_pidx[off];
                unsigned int sig_idx = g_shape_pidx[off + 1u];
                double mu = s_params[mu_idx];
                double sigma = s_params[sig_idx];
                logp = gaussian_logp_only(x, mu, sigma, a, b);
            } else if (proc.pdf_kind == PDF_EXPONENTIAL) {
                unsigned int off = proc.shape_param_offset;
                if (proc.n_shape_params != 1u || off >= total_shape_params) {
                    continue;
                }
                unsigned int lam_idx = g_shape_pidx[off];
                double lambda = s_params[lam_idx];
                logp = exponential_logp_only(x, lambda, a, b);
            } else if (proc.pdf_kind == PDF_CRYSTAL_BALL) {
                unsigned int off = proc.shape_param_offset;
                if (proc.n_shape_params != 4u || off + 3u >= total_shape_params) {
                    continue;
                }
                unsigned int mu_idx = g_shape_pidx[off];
                unsigned int sig_idx = g_shape_pidx[off + 1u];
                unsigned int alpha_idx = g_shape_pidx[off + 2u];
                unsigned int n_idx = g_shape_pidx[off + 3u];
                double mu = s_params[mu_idx];
                double sigma = s_params[sig_idx];
                double alpha = s_params[alpha_idx];
                double nn = s_params[n_idx];
                logp = crystal_ball_logp_only(x, mu, sigma, alpha, nn, a, b);
            } else if (proc.pdf_kind == PDF_DOUBLE_CRYSTAL_BALL) {
                unsigned int off = proc.shape_param_offset;
                if (proc.n_shape_params != 6u || off + 5u >= total_shape_params) {
                    continue;
                }
                unsigned int mu_idx = g_shape_pidx[off];
                unsigned int sig_idx = g_shape_pidx[off + 1u];
                unsigned int alpha_l_idx = g_shape_pidx[off + 2u];
                unsigned int n_l_idx = g_shape_pidx[off + 3u];
                unsigned int alpha_r_idx = g_shape_pidx[off + 4u];
                unsigned int n_r_idx = g_shape_pidx[off + 5u];
                double mu = s_params[mu_idx];
                double sigma = s_params[sig_idx];
                double alpha_l = s_params[alpha_l_idx];
                double n_l = s_params[n_l_idx];
                double alpha_r = s_params[alpha_r_idx];
                double n_r = s_params[n_r_idx];
                logp = double_crystal_ball_logp_only(x, mu, sigma, alpha_l, n_l, alpha_r, n_r, a, b);
            } else if (proc.pdf_kind == PDF_CHEBYSHEV) {
                unsigned int off = proc.shape_param_offset;
                unsigned int order = proc.n_shape_params;
                if (order == 0u || off + order - 1u >= total_shape_params) {
                    continue;
                }
                logp = chebyshev_logp_only(x, g_shape_pidx, off, order, s_params, a, b);
            } else if (proc.pdf_kind == PDF_HISTOGRAM) {
                if (proc.n_shape_params != 0u) {
                    continue;
                }
                logp = histogram_logp_only(x, g_pdf_aux_f64, proc.pdf_aux_offset, proc.pdf_aux_len);
            } else {
                continue;
            }

            double term = log(nu) + logp;
            if (!isfinite(term)) {
                continue;
            }

            if (term > max_term) {
                sum_exp = sum_exp * exp(max_term - term) + 1.0;
                max_term = term;
            } else {
                sum_exp += exp(term - max_term);
            }
        }

        double logf = max_term + log(sum_exp);
        if (!isfinite(logf)) {
            logf = log(DBL_MIN);
        }
        local_sum_logf += logf;

        /* Gradient contributions per event */
        for (unsigned int p = 0; p < n_procs; p++) {
            struct GpuUnbinnedProcessDesc proc = g_procs[p];
            if (proc.obs_index != 0u) {
                continue;
            }

            double nu = 0.0;
            double dnu = 0.0;
            unsigned int y_idx = proc.yield_param_idx;
            unsigned int has_yield_param = 0u;
            if (proc.yield_kind == YIELD_FIXED) {
                nu = proc.base_yield;
            } else if (proc.yield_kind == YIELD_PARAMETER) {
                nu = s_params[y_idx];
                dnu = 1.0;
                has_yield_param = 1u;
            } else if (proc.yield_kind == YIELD_SCALED) {
                nu = proc.base_yield * s_params[y_idx];
                dnu = proc.base_yield;
                has_yield_param = 1u;
            } else {
                continue;
            }
            unsigned int mod_off = proc.rate_mod_offset;
            unsigned int nmods = proc.n_rate_mods;
            if (mod_off + nmods > total_rate_mods) {
                nmods = 0u;
            }
            for (unsigned int m = 0; m < nmods; m++) {
                double f, dl;
                rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dl);
                (void)dl;
                nu *= f;
                dnu *= f;
            }
            if (!(nu > 0.0) || !isfinite(nu)) {
                continue;
            }

            double logp_val = -INFINITY;
            if (proc.pdf_kind == PDF_GAUSSIAN) {
                unsigned int off = proc.shape_param_offset;
                if (proc.n_shape_params != 2u || off + 1u >= total_shape_params) {
                    continue;
                }
                unsigned int mu_idx = g_shape_pidx[off];
                unsigned int sig_idx = g_shape_pidx[off + 1u];
                double mu = s_params[mu_idx];
                double sigma = s_params[sig_idx];
                double dmu, dsig;
                gaussian_logp_grad(x, mu, sigma, a, b, &logp_val, &dmu, &dsig);
                if (!isfinite(logp_val)) {
                    continue;
                }
                double p_over_f = exp(logp_val - logf);
                if (!(p_over_f > 0.0) || !isfinite(p_over_f)) {
                    continue;
                }
                double r = nu * p_over_f;
                atomicAdd(&grad_out[mu_idx], -r * dmu);
                atomicAdd(&grad_out[sig_idx], -r * dsig);
                if (has_yield_param) {
                    atomicAdd(&grad_out[y_idx], -dnu * p_over_f);
                }
            } else if (proc.pdf_kind == PDF_EXPONENTIAL) {
                unsigned int off = proc.shape_param_offset;
                if (proc.n_shape_params != 1u || off >= total_shape_params) {
                    continue;
                }
                unsigned int lam_idx = g_shape_pidx[off];
                double lambda = s_params[lam_idx];
                double dlam;
                exponential_logp_grad(x, lambda, a, b, &logp_val, &dlam);
                if (!isfinite(logp_val)) {
                    continue;
                }
                double p_over_f = exp(logp_val - logf);
                if (!(p_over_f > 0.0) || !isfinite(p_over_f)) {
                    continue;
                }
                double r = nu * p_over_f;
                atomicAdd(&grad_out[lam_idx], -r * dlam);
                if (has_yield_param) {
                    atomicAdd(&grad_out[y_idx], -dnu * p_over_f);
                }
            } else if (proc.pdf_kind == PDF_CRYSTAL_BALL) {
                unsigned int off = proc.shape_param_offset;
                if (proc.n_shape_params != 4u || off + 3u >= total_shape_params) {
                    continue;
                }
                unsigned int mu_idx = g_shape_pidx[off];
                unsigned int sig_idx = g_shape_pidx[off + 1u];
                unsigned int alpha_idx = g_shape_pidx[off + 2u];
                unsigned int n_idx = g_shape_pidx[off + 3u];
                double mu = s_params[mu_idx];
                double sigma = s_params[sig_idx];
                double alpha = s_params[alpha_idx];
                double nn = s_params[n_idx];
                double dmu, dsig, dalpha, dn;
                crystal_ball_logp_grad(x, mu, sigma, alpha, nn, a, b,
                    &logp_val, &dmu, &dsig, &dalpha, &dn);
                if (!isfinite(logp_val)) {
                    continue;
                }
                double p_over_f = exp(logp_val - logf);
                if (!(p_over_f > 0.0) || !isfinite(p_over_f)) {
                    continue;
                }
                double r = nu * p_over_f;
                atomicAdd(&grad_out[mu_idx], -r * dmu);
                atomicAdd(&grad_out[sig_idx], -r * dsig);
                atomicAdd(&grad_out[alpha_idx], -r * dalpha);
                atomicAdd(&grad_out[n_idx], -r * dn);
                if (has_yield_param) {
                    atomicAdd(&grad_out[y_idx], -dnu * p_over_f);
                }
            } else if (proc.pdf_kind == PDF_DOUBLE_CRYSTAL_BALL) {
                unsigned int off = proc.shape_param_offset;
                if (proc.n_shape_params != 6u || off + 5u >= total_shape_params) {
                    continue;
                }
                unsigned int mu_idx = g_shape_pidx[off];
                unsigned int sig_idx = g_shape_pidx[off + 1u];
                unsigned int alpha_l_idx = g_shape_pidx[off + 2u];
                unsigned int n_l_idx = g_shape_pidx[off + 3u];
                unsigned int alpha_r_idx = g_shape_pidx[off + 4u];
                unsigned int n_r_idx = g_shape_pidx[off + 5u];
                double mu = s_params[mu_idx];
                double sigma = s_params[sig_idx];
                double alpha_l = s_params[alpha_l_idx];
                double n_l = s_params[n_l_idx];
                double alpha_r = s_params[alpha_r_idx];
                double n_r = s_params[n_r_idx];
                double dmu, dsig, dal, dnl, dar, dnr;
                double_crystal_ball_logp_grad(x, mu, sigma, alpha_l, n_l, alpha_r, n_r, a, b,
                    &logp_val, &dmu, &dsig, &dal, &dnl, &dar, &dnr);
                if (!isfinite(logp_val)) {
                    continue;
                }
                double p_over_f = exp(logp_val - logf);
                if (!(p_over_f > 0.0) || !isfinite(p_over_f)) {
                    continue;
                }
                double r = nu * p_over_f;
                atomicAdd(&grad_out[mu_idx], -r * dmu);
                atomicAdd(&grad_out[sig_idx], -r * dsig);
                atomicAdd(&grad_out[alpha_l_idx], -r * dal);
                atomicAdd(&grad_out[n_l_idx], -r * dnl);
                atomicAdd(&grad_out[alpha_r_idx], -r * dar);
                atomicAdd(&grad_out[n_r_idx], -r * dnr);
                if (has_yield_param) {
                    atomicAdd(&grad_out[y_idx], -dnu * p_over_f);
                }
            } else if (proc.pdf_kind == PDF_CHEBYSHEV) {
                unsigned int off = proc.shape_param_offset;
                unsigned int order = proc.n_shape_params;
                if (order == 0u || off + order - 1u >= total_shape_params) {
                    continue;
                }
                /* Chebyshev logp_only + inline gradient */
                double w = b - a;
                double i0 = w;
                for (unsigned int j = 0; j < order; j++) {
                    unsigned int c_idx = g_shape_pidx[off + j];
                    double c = s_params[c_idx];
                    unsigned int k = j + 1u;
                    if ((k & 1u) == 0u) {
                        double denom = 1.0 - (double)k * (double)k;
                        i0 += w * c / denom;
                    }
                }
                if (!isfinite(i0) || !(i0 > 0.0)) {
                    continue;
                }
                double log_i = log(i0);
                double xp = chebyshev_xprime(x, a, b);
                double f0 = 1.0;
                double tkm1 = 1.0;
                double tk = xp;
                for (unsigned int j = 0; j < order; j++) {
                    unsigned int k = j + 1u;
                    double tval = 0.0;
                    if (k == 1u) {
                        tval = tk;
                    } else {
                        double tkp1 = 2.0 * xp * tk - tkm1;
                        tval = tkp1;
                        tkm1 = tk;
                        tk = tkp1;
                    }
                    unsigned int c_idx = g_shape_pidx[off + j];
                    double c = s_params[c_idx];
                    f0 += c * tval;
                }
                if (!isfinite(f0) || !(f0 > 0.0)) {
                    continue;
                }
                logp_val = log(f0) - log_i;
                if (!isfinite(logp_val)) {
                    continue;
                }
                double p_over_f = exp(logp_val - logf);
                if (!(p_over_f > 0.0) || !isfinite(p_over_f)) {
                    continue;
                }
                if (has_yield_param) {
                    atomicAdd(&grad_out[y_idx], -dnu * p_over_f);
                }
                double r = nu * p_over_f;
                double inv_f0 = 1.0 / f0;
                double inv_i0 = 1.0 / i0;
                tkm1 = 1.0;
                tk = xp;
                for (unsigned int j = 0; j < order; j++) {
                    unsigned int k = j + 1u;
                    double tval = 0.0;
                    if (k == 1u) {
                        tval = tk;
                    } else {
                        double tkp1 = 2.0 * xp * tk - tkm1;
                        tval = tkp1;
                        tkm1 = tk;
                        tk = tkp1;
                    }
                    double dlogi = 0.0;
                    if ((k & 1u) == 0u) {
                        double denom = 1.0 - (double)k * (double)k;
                        double di_dc = w / denom;
                        dlogi = di_dc * inv_i0;
                    }
                    double dlogp_dc = tval * inv_f0 - dlogi;
                    unsigned int c_idx = g_shape_pidx[off + j];
                    atomicAdd(&grad_out[c_idx], -r * dlogp_dc);
                }
            } else if (proc.pdf_kind == PDF_HISTOGRAM) {
                if (proc.n_shape_params != 0u) {
                    continue;
                }
                logp_val = histogram_logp_only(x, g_pdf_aux_f64, proc.pdf_aux_offset, proc.pdf_aux_len);
                if (!isfinite(logp_val)) {
                    continue;
                }
                double p_over_f = exp(logp_val - logf);
                if (!(p_over_f > 0.0) || !isfinite(p_over_f)) {
                    continue;
                }
                if (has_yield_param) {
                    atomicAdd(&grad_out[y_idx], -dnu * p_over_f);
                }
            }

            /* Rate modifier gradient contributions */
            for (unsigned int rm = 0; rm < nmods; rm++) {
                struct GpuUnbinnedRateModifierDesc rmd = g_rate_mods[mod_off + rm];
                unsigned int aidx = rmd.alpha_param_idx;
                if (aidx >= n_params) {
                    continue;
                }
                double f, dl;
                rate_modifier_factor_dlogf(g_rate_mods, mod_off + rm, s_params, &f, &dl);
                (void)f;
                double dnu_m = nu * dl;
                if (isfinite(dnu_m) && dnu_m != 0.0) {
                    double p_over_f2 = exp(logp_val - logf);
                    if (isfinite(p_over_f2) && p_over_f2 > 0.0) {
                        atomicAdd(&grad_out[aidx], -dnu_m * p_over_f2);
                    }
                }
            }
        }
    }

    /* Block reduction of sum_logf */
    s_scratch[tid] = local_sum_logf;
    __syncthreads();
    for (unsigned int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] += s_scratch[tid + stride];
        }
        __syncthreads();
    }

    /* Thread 0: finalize NLL and gradient nu_tot terms */
    if (tid == 0) {
        double sum_logf = s_scratch[0];

        double nu_tot = 0.0;
        for (unsigned int p = 0; p < n_procs; p++) {
            struct GpuUnbinnedProcessDesc proc = g_procs[p];
            double nu = 0.0;
            double dnu = 0.0;
            unsigned int y_idx = proc.yield_param_idx;
            unsigned int has_yield_param = 0u;
            if (proc.yield_kind == YIELD_FIXED) {
                nu = proc.base_yield;
            } else if (proc.yield_kind == YIELD_PARAMETER) {
                nu = s_params[y_idx];
                dnu = 1.0;
                has_yield_param = 1u;
            } else if (proc.yield_kind == YIELD_SCALED) {
                nu = proc.base_yield * s_params[y_idx];
                dnu = proc.base_yield;
                has_yield_param = 1u;
            } else {
                continue;
            }
            unsigned int mod_off = proc.rate_mod_offset;
            unsigned int nmods = proc.n_rate_mods;
            if (mod_off + nmods > total_rate_mods) {
                nmods = 0u;
            }
            double mod_factor = 1.0;
            for (unsigned int m = 0; m < nmods; m++) {
                double f, dl;
                rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dl);
                (void)dl;
                mod_factor *= f;
            }
            nu *= mod_factor;
            dnu *= mod_factor;
            if (isfinite(nu) && nu >= 0.0) {
                nu_tot += nu;
                if (has_yield_param && isfinite(dnu) && dnu != 0.0) {
                    grad_out[y_idx] += dnu;
                }
                for (unsigned int m = 0; m < nmods; m++) {
                    struct GpuUnbinnedRateModifierDesc rm = g_rate_mods[mod_off + m];
                    unsigned int aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    double f, dl;
                    rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dl);
                    (void)f;
                    double dnu_m = nu * dl;
                    if (isfinite(dnu_m) && dnu_m != 0.0) {
                        grad_out[aidx] += dnu_m;
                    }
                }
            }
        }

        double nll = nu_tot - sum_logf;

        for (unsigned int k = 0; k < n_gauss; k++) {
            struct GpuUnbinnedGaussConstraintEntry gc = g_gauss[k];
            unsigned int idx = gc.param_idx;
            if (idx >= n_params) {
                continue;
            }
            double x0 = s_params[idx];
            double diff = x0 - gc.center;
            double z = diff * gc.inv_width;
            nll += 0.5 * z * z;
            grad_out[idx] += z * gc.inv_width;
        }

        nll += constraint_const;
        *nll_out = nll;
    }
}

/* ========================================================================= */
/*  Device function: NLL only for one toy (line search, no gradient)         */
/* ========================================================================= */

__device__ double compute_nll_only(
    const double* __restrict__ s_params,
    const double* __restrict__ obs,
    unsigned int n_events,
    double obs_lo,
    double obs_hi,
    const struct GpuUnbinnedProcessDesc* __restrict__ g_procs,
    const struct GpuUnbinnedRateModifierDesc* __restrict__ g_rate_mods,
    const unsigned int* __restrict__ g_shape_pidx,
    const double* __restrict__ g_pdf_aux_f64,
    const struct GpuUnbinnedGaussConstraintEntry* __restrict__ g_gauss,
    double* __restrict__ s_scratch,
    unsigned int n_params,
    unsigned int n_procs,
    unsigned int total_rate_mods,
    unsigned int total_shape_params,
    unsigned int n_gauss,
    double constraint_const
) {
    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    double local_sum_logf = 0.0;
    double a = obs_lo;
    double b = obs_hi;

    for (unsigned int i = tid; i < n_events; i += block_size) {
        double x = obs[i];

        double max_term = -INFINITY;
        double sum_exp = 0.0;

        for (unsigned int p = 0; p < n_procs; p++) {
            struct GpuUnbinnedProcessDesc proc = g_procs[p];
            if (proc.obs_index != 0u) {
                continue;
            }

            double nu = 0.0;
            if (proc.yield_kind == YIELD_FIXED) {
                nu = proc.base_yield;
            } else if (proc.yield_kind == YIELD_PARAMETER) {
                nu = s_params[proc.yield_param_idx];
            } else if (proc.yield_kind == YIELD_SCALED) {
                nu = proc.base_yield * s_params[proc.yield_param_idx];
            } else {
                continue;
            }
            unsigned int mod_off = proc.rate_mod_offset;
            unsigned int nmods = proc.n_rate_mods;
            if (mod_off + nmods > total_rate_mods) {
                nmods = 0u;
            }
            for (unsigned int m = 0; m < nmods; m++) {
                double f, dl;
                rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dl);
                (void)dl;
                nu *= f;
            }
            if (!(nu > 0.0) || !isfinite(nu)) {
                continue;
            }

            double logp = -INFINITY;
            if (proc.pdf_kind == PDF_GAUSSIAN) {
                unsigned int off = proc.shape_param_offset;
                if (proc.n_shape_params != 2u || off + 1u >= total_shape_params) {
                    continue;
                }
                double mu = s_params[g_shape_pidx[off]];
                double sigma = s_params[g_shape_pidx[off + 1u]];
                logp = gaussian_logp_only(x, mu, sigma, a, b);
            } else if (proc.pdf_kind == PDF_EXPONENTIAL) {
                unsigned int off = proc.shape_param_offset;
                if (proc.n_shape_params != 1u || off >= total_shape_params) {
                    continue;
                }
                double lambda = s_params[g_shape_pidx[off]];
                logp = exponential_logp_only(x, lambda, a, b);
            } else if (proc.pdf_kind == PDF_CRYSTAL_BALL) {
                unsigned int off = proc.shape_param_offset;
                if (proc.n_shape_params != 4u || off + 3u >= total_shape_params) {
                    continue;
                }
                double mu = s_params[g_shape_pidx[off]];
                double sigma = s_params[g_shape_pidx[off + 1u]];
                double alpha = s_params[g_shape_pidx[off + 2u]];
                double nn = s_params[g_shape_pidx[off + 3u]];
                logp = crystal_ball_logp_only(x, mu, sigma, alpha, nn, a, b);
            } else if (proc.pdf_kind == PDF_DOUBLE_CRYSTAL_BALL) {
                unsigned int off = proc.shape_param_offset;
                if (proc.n_shape_params != 6u || off + 5u >= total_shape_params) {
                    continue;
                }
                double mu = s_params[g_shape_pidx[off]];
                double sigma = s_params[g_shape_pidx[off + 1u]];
                double alpha_l = s_params[g_shape_pidx[off + 2u]];
                double n_l = s_params[g_shape_pidx[off + 3u]];
                double alpha_r = s_params[g_shape_pidx[off + 4u]];
                double n_r = s_params[g_shape_pidx[off + 5u]];
                logp = double_crystal_ball_logp_only(x, mu, sigma, alpha_l, n_l, alpha_r, n_r, a, b);
            } else if (proc.pdf_kind == PDF_CHEBYSHEV) {
                unsigned int off = proc.shape_param_offset;
                unsigned int order = proc.n_shape_params;
                if (order == 0u || off + order - 1u >= total_shape_params) {
                    continue;
                }
                logp = chebyshev_logp_only(x, g_shape_pidx, off, order, s_params, a, b);
            } else if (proc.pdf_kind == PDF_HISTOGRAM) {
                if (proc.n_shape_params != 0u) {
                    continue;
                }
                logp = histogram_logp_only(x, g_pdf_aux_f64, proc.pdf_aux_offset, proc.pdf_aux_len);
            } else {
                continue;
            }

            double term = log(nu) + logp;
            if (!isfinite(term)) {
                continue;
            }
            if (term > max_term) {
                sum_exp = sum_exp * exp(max_term - term) + 1.0;
                max_term = term;
            } else {
                sum_exp += exp(term - max_term);
            }
        }

        double logf = max_term + log(sum_exp);
        if (!isfinite(logf)) {
            logf = log(DBL_MIN);
        }
        local_sum_logf += logf;
    }

    /* Block reduction */
    s_scratch[tid] = local_sum_logf;
    __syncthreads();
    for (unsigned int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] += s_scratch[tid + stride];
        }
        __syncthreads();
    }

    /* Thread 0: finalize NLL (no gradient) */
    double nll = 0.0;
    if (tid == 0) {
        double sum_logf = s_scratch[0];
        double nu_tot = 0.0;
        for (unsigned int p = 0; p < n_procs; p++) {
            struct GpuUnbinnedProcessDesc proc = g_procs[p];
            double nu = 0.0;
            if (proc.yield_kind == YIELD_FIXED) {
                nu = proc.base_yield;
            } else if (proc.yield_kind == YIELD_PARAMETER) {
                nu = s_params[proc.yield_param_idx];
            } else if (proc.yield_kind == YIELD_SCALED) {
                nu = proc.base_yield * s_params[proc.yield_param_idx];
            } else {
                continue;
            }
            unsigned int mod_off = proc.rate_mod_offset;
            unsigned int nmods = proc.n_rate_mods;
            if (mod_off + nmods > total_rate_mods) {
                nmods = 0u;
            }
            for (unsigned int m = 0; m < nmods; m++) {
                double f, dl;
                rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dl);
                (void)dl;
                nu *= f;
            }
            if (isfinite(nu) && nu > 0.0) {
                nu_tot += nu;
            }
        }
        nll = nu_tot - sum_logf;
        for (unsigned int k = 0; k < n_gauss; k++) {
            struct GpuUnbinnedGaussConstraintEntry gc = g_gauss[k];
            double x0 = s_params[gc.param_idx];
            double diff = x0 - gc.center;
            double z = diff * gc.inv_width;
            nll += 0.5 * z * z;
        }
        nll += constraint_const;
    }
    return nll;
}

/* ========================================================================= */
/*  Persistent mega-kernel: full L-BFGS optimization, 1 block = 1 toy       */
/* ========================================================================= */

extern "C" __global__ void unbinned_batch_lbfgs_fit(
    /* L-BFGS state buffers (device global, per-toy) */
    double* __restrict__ g_x,              /* [n_toys × n_params] current params */
    double* __restrict__ g_prev_x,         /* [n_toys × n_params] */
    double* __restrict__ g_prev_grad,      /* [n_toys × n_params] */
    double* __restrict__ g_s_hist,         /* [n_toys × lbfgs_m × n_params] */
    double* __restrict__ g_y_hist,         /* [n_toys × lbfgs_m × n_params] */
    double* __restrict__ g_rho_hist,       /* [n_toys × lbfgs_m] */
    double* __restrict__ g_grad,           /* [n_toys × n_params] gradient workspace */
    double* __restrict__ g_direction,      /* [n_toys × n_params] search direction */
    /* Parameter bounds */
    const double* __restrict__ g_bounds_lo, /* [n_params] */
    const double* __restrict__ g_bounds_hi, /* [n_params] */
    /* Per-channel descriptors (device pointers to channel-local buffers) */
    const struct GpuChannelDesc* __restrict__ g_channels,
    /* Output */
    double* __restrict__ g_nll_out,        /* [n_toys] final NLL */
    unsigned int* __restrict__ g_status,   /* [n_toys] 0=max_iter, 1=converged, 2=failed */
    unsigned int* __restrict__ g_iters,    /* [n_toys] iterations used */
    unsigned int* __restrict__ g_line_search_exhaust, /* [n_toys] line-search exhaustion count */
    /* Scalar configuration */
    unsigned int n_params,
    unsigned int n_channels,
    unsigned int n_toys,
    unsigned int max_iter,
    unsigned int lbfgs_m,
    double tol,
    unsigned int max_backtracks
) {
    unsigned int toy = blockIdx.x;
    if (toy >= n_toys) {
        return;
    }

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    /* Per-toy pointers */
    double* x         = g_x         + (size_t)toy * n_params;
    double* prev_x    = g_prev_x    + (size_t)toy * n_params;
    double* prev_grad = g_prev_grad + (size_t)toy * n_params;
    double* grad      = g_grad      + (size_t)toy * n_params;
    double* dir       = g_direction + (size_t)toy * n_params;
    double* s_hist    = g_s_hist    + (size_t)toy * lbfgs_m * n_params;
    double* y_hist    = g_y_hist    + (size_t)toy * lbfgs_m * n_params;
    double* rho_hist  = g_rho_hist  + (size_t)toy * lbfgs_m;

    /* Shared memory layout: params[n_params] + grad[n_params] + scratch[block_size] + 2 int flags */
    extern __shared__ double shared[];
    double* s_params  = shared;
    double* s_grad    = shared + n_params;
    double* s_scratch = s_grad + n_params;
    /* Borrow 2 ints (8 bytes = 1 double) at the end of scratch for flags */
    volatile int* s_flag = (volatile int*)(s_scratch + block_size);
    /* s_flag[0] = converged/exit, s_flag[1] = accepted (line search) */

    if (tid == 0) {
        s_flag[0] = 0;
        s_flag[1] = 0;
    }
    __syncthreads();

    /* Thread 0 local state */
    unsigned int n_stored = 0;
    unsigned int line_search_exhaust = 0;
    int has_prev = 0;
    double fval = INFINITY;
    double prev_fval = INFINITY;
    // Directional derivative for Armijo uses the actual projected step delta = trial - prev_x.
    double g_dot_d = 0.0;
    double step_val = 1.0;

    for (unsigned int iter = 0; iter < max_iter; iter++) {
        /* ---- Phase A: Load params into shared memory ---- */
        for (unsigned int i = tid; i < n_params; i += block_size) {
            s_params[i] = x[i];
        }
        __syncthreads();

        /* ---- Phase B: Zero gradient (shared) ---- */
        for (unsigned int i = tid; i < n_params; i += block_size) {
            s_grad[i] = 0.0;
        }
        __syncthreads();

        /* ---- Phase C: Compute NLL + gradient across all channels ---- */
        double nll_tmp = 0.0;
        for (unsigned int ch = 0; ch < n_channels; ch++) {
            struct GpuChannelDesc cd = g_channels[ch];
            unsigned int ev_start = cd.toy_offsets[toy];
            unsigned int ev_end   = cd.toy_offsets[toy + 1u];
            unsigned int n_events_ch = (ev_end > ev_start) ? (ev_end - ev_start) : 0u;
            const double* obs_ch = cd.obs_flat + ev_start;

            double nll_ch;
            compute_nll_and_grad(
                s_params, obs_ch, n_events_ch, cd.obs_lo, cd.obs_hi,
                cd.procs,
                cd.rate_mods,
                cd.shape_pidx,
                cd.pdf_aux_f64,
                cd.gauss,
                &nll_ch, s_grad, s_scratch,
                n_params, cd.n_procs, cd.total_rate_mods, cd.total_shape_params,
                cd.n_gauss, cd.constraint_const
            );
            __syncthreads();

            if (tid == 0) {
                nll_tmp += nll_ch;
            }
        }

        /* ---- Phase D: L-BFGS step (thread 0 only) ---- */
        if (tid == 0) {
            fval = nll_tmp;

            if (!isfinite(fval)) {
                g_nll_out[toy] = fval;
                g_status[toy] = 2u;
                g_iters[toy] = iter;
                s_flag[0] = 1;
            }

            /* Check convergence: projected gradient infinity norm */
            double pg_norm = 0.0;
            int grad_is_finite = 1;
            for (unsigned int i = 0; i < n_params; i++) {
                double gi = s_grad[i];
                double xi = x[i];
                if (!isfinite(gi) || !isfinite(xi)) {
                    grad_is_finite = 0;
                    break;
                }
                double pg;
                if (gi < 0.0) {
                    pg = fmin(-gi, g_bounds_hi[i] - xi);
                } else {
                    pg = fmin(gi, xi - g_bounds_lo[i]);
                }
                if (fabs(pg) > pg_norm) {
                    pg_norm = fabs(pg);
                }
            }
            if (!grad_is_finite) {
                g_nll_out[toy] = fval;
                g_status[toy] = 2u;
                g_iters[toy] = iter;
                s_flag[0] = 1;
            }

            // Large-N scaling: for sum-of-events NLLs, |NLL| is O(N), and the gradient norm at
            // statistically-relevant solutions typically scales like O(sqrt(N)). Using a fixed
            // absolute tolerance can therefore lead to max-iter behavior for huge event counts.
            double tol_eff = tol * sqrt(fmax(fabs(fval), 1.0));
            // Standard L-BFGS-B style relative objective decrease criterion.
            double rel_obj = INFINITY;
            if (isfinite(prev_fval)) {
                rel_obj = fabs(prev_fval - fval) / fmax(fmax(fabs(prev_fval), fabs(fval)), 1.0);
            }
            if (s_flag[0] == 0 && (pg_norm < tol_eff || (iter >= 3 && rel_obj < tol))) {
                g_nll_out[toy] = fval;
                g_status[toy] = 1u;
                g_iters[toy] = iter;
                s_flag[0] = 1;
            }
            prev_fval = fval;
        }
        __syncthreads();

        /* Early exit if converged — all threads see the shared flag */
        if (s_flag[0]) {
            if (tid == 0) {
                g_line_search_exhaust[toy] = line_search_exhaust;
            }
            return;
        }

        /* ---- Phase D continued: L-BFGS direction (thread 0) ---- */
        if (tid == 0) {
            /* Update L-BFGS history from previous iteration */
            if (has_prev) {
                unsigned int write_slot = n_stored % lbfgs_m;
                double ys = 0.0;
                for (unsigned int i = 0; i < n_params; i++) {
                    double si = x[i] - prev_x[i];
                    double yi = s_grad[i] - prev_grad[i];
                    s_hist[write_slot * n_params + i] = si;
                    y_hist[write_slot * n_params + i] = yi;
                    ys += yi * si;
                }
                if (ys > 1e-20) {
                    rho_hist[write_slot] = 1.0 / ys;
                    n_stored++;
                }
            }

            /* Save current x and grad for next history update */
            for (unsigned int i = 0; i < n_params; i++) {
                prev_x[i] = x[i];
                prev_grad[i] = s_grad[i];
            }
            has_prev = 1;

            /* L-BFGS two-loop recursion */
            unsigned int count = (n_stored < lbfgs_m) ? n_stored : lbfgs_m;

            /* q = -grad */
            for (unsigned int i = 0; i < n_params; i++) {
                dir[i] = -s_grad[i];
            }

            double alpha_tmp[MAX_LBFGS_M];

            if (count > 0) {
                /* Backward pass: most recent to oldest */
                for (unsigned int j = 0; j < count; j++) {
                    unsigned int slot = (n_stored - 1 - j) % lbfgs_m;
                    double dot = 0.0;
                    for (unsigned int i = 0; i < n_params; i++) {
                        dot += s_hist[slot * n_params + i] * dir[i];
                    }
                    double aj = rho_hist[slot] * dot;
                    alpha_tmp[j] = aj;
                    for (unsigned int i = 0; i < n_params; i++) {
                        dir[i] -= aj * y_hist[slot * n_params + i];
                    }
                }

                /* Scale by gamma = (s^T y) / (y^T y) from most recent pair */
                unsigned int newest = (n_stored - 1) % lbfgs_m;
                double ys_g = 0.0, yy_g = 0.0;
                for (unsigned int i = 0; i < n_params; i++) {
                    double yi = y_hist[newest * n_params + i];
                    ys_g += s_hist[newest * n_params + i] * yi;
                    yy_g += yi * yi;
                }
                double gamma = (yy_g > 0.0) ? (ys_g / yy_g) : 1.0;
                for (unsigned int i = 0; i < n_params; i++) {
                    dir[i] *= gamma;
                }

                /* Forward pass: oldest to most recent */
                for (unsigned int j = 0; j < count; j++) {
                    unsigned int slot = (n_stored - count + j) % lbfgs_m;
                    double dot = 0.0;
                    for (unsigned int i = 0; i < n_params; i++) {
                        dot += y_hist[slot * n_params + i] * dir[i];
                    }
                    double beta = rho_hist[slot] * dot;
                    for (unsigned int i = 0; i < n_params; i++) {
                        dir[i] += (alpha_tmp[count - 1 - j] - beta) * s_hist[slot * n_params + i];
                    }
                }
            }

            /* Initial trial: prev_x + 1.0 * dir, clamped to bounds */
            step_val = 1.0;
            for (unsigned int i = 0; i < n_params; i++) {
                double trial = prev_x[i] + dir[i];
                trial = fmax(trial, g_bounds_lo[i]);
                trial = fmin(trial, g_bounds_hi[i]);
                s_params[i] = trial;
            }
            // Armijo uses actual projected step delta, not the unclamped direction.
            g_dot_d = 0.0;
            for (unsigned int i = 0; i < n_params; i++) {
                g_dot_d += s_grad[i] * (s_params[i] - prev_x[i]);
            }
            // If the projected step is not a descent direction, fall back to steepest descent
            // and drop L-BFGS history (it produced a bad direction).
            if (!isfinite(g_dot_d) || g_dot_d >= 0.0) {
                n_stored = 0;
                step_val = 1.0;
                for (unsigned int i = 0; i < n_params; i++) {
                    dir[i] = -s_grad[i];
                    double trial = prev_x[i] + dir[i];
                    trial = fmax(trial, g_bounds_lo[i]);
                    trial = fmin(trial, g_bounds_hi[i]);
                    s_params[i] = trial;
                }
                g_dot_d = 0.0;
                for (unsigned int i = 0; i < n_params; i++) {
                    g_dot_d += s_grad[i] * (s_params[i] - prev_x[i]);
                }
            }
            s_flag[1] = 0;
        }
        __syncthreads();

        /* ---- Phase E: Line search (Armijo backtracking) ---- */
        for (unsigned int bt = 0; bt < max_backtracks; bt++) {
            /* All threads: compute NLL at trial point across all channels */
            double trial_nll = 0.0;
            for (unsigned int ch = 0; ch < n_channels; ch++) {
                struct GpuChannelDesc cd = g_channels[ch];
                unsigned int ev_s = cd.toy_offsets[toy];
                unsigned int ev_e = cd.toy_offsets[toy + 1u];
                unsigned int n_ev = (ev_e > ev_s) ? (ev_e - ev_s) : 0u;
                const double* obs_ch = cd.obs_flat + ev_s;

                double nll_ch = compute_nll_only(
                    s_params, obs_ch, n_ev, cd.obs_lo, cd.obs_hi,
                    cd.procs,
                    cd.rate_mods,
                    cd.shape_pidx,
                    cd.pdf_aux_f64,
                    cd.gauss,
                    s_scratch,
                    n_params, cd.n_procs, cd.total_rate_mods, cd.total_shape_params,
                    cd.n_gauss, cd.constraint_const
                );
                __syncthreads();

                trial_nll += nll_ch;
            }

            if (tid == 0) {
                if (trial_nll <= fval + ARMIJO_C1 * g_dot_d) {
                    s_flag[1] = 1;
                    for (unsigned int i = 0; i < n_params; i++) {
                        x[i] = s_params[i];
                    }
                    fval = trial_nll;
                } else {
                    step_val *= 0.5;
                    for (unsigned int i = 0; i < n_params; i++) {
                        double trial = prev_x[i] + step_val * dir[i];
                        trial = fmax(trial, g_bounds_lo[i]);
                        trial = fmin(trial, g_bounds_hi[i]);
                        s_params[i] = trial;
                    }
                    g_dot_d = 0.0;
                    for (unsigned int i = 0; i < n_params; i++) {
                        g_dot_d += s_grad[i] * (s_params[i] - prev_x[i]);
                    }
                }
            }
            __syncthreads();

            if (s_flag[1]) {
                break;
            }
        }

        /* If not accepted after all backtracks, take the last trial */
        if (tid == 0 && !s_flag[1]) {
            line_search_exhaust += 1u;
            for (unsigned int i = 0; i < n_params; i++) {
                x[i] = s_params[i];
            }
        }
        __syncthreads();
    }

    /* Reached max_iter without convergence */
    if (tid == 0) {
        g_nll_out[toy] = fval;
        g_status[toy] = 0u;
        g_iters[toy] = max_iter;
        g_line_search_exhaust[toy] = line_search_exhaust;
    }
}
