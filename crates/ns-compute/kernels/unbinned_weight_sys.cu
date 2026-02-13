/**
 * Unbinned helper CUDA kernel: per-event WeightSys morphing.
 *
 * Computes morphed weights and their derivatives w.r.t. nuisance parameters α
 * using HistFactory-like WeightSys interpolation codes:
 *   - Code0  (piecewise linear)
 *   - Code4p (smooth polynomial on [-1,1], linear extrap outside)
 *
 * This is a building block for unbinned non-parametric PDFs (e.g. KDE / template
 * shapes) where weights are defined per event/kernel and can be morphed by
 * systematic variations.
 */

#include <math.h>
#include <float.h>

// WeightSys interp codes (must match ns-compute/unbinned_types.rs).
#define INTERP_CODE0  0u
#define INTERP_CODE4P 1u

__device__ inline void histosys_interp(
    double alpha,
    double down,
    double nominal,
    double up,
    unsigned int code,
    double* __restrict__ out_val,
    double* __restrict__ out_der
) {
    if (!(isfinite(alpha) && isfinite(down) && isfinite(nominal) && isfinite(up))) {
        *out_val = nominal;
        *out_der = 0.0;
        return;
    }

    if (code == INTERP_CODE0) {
        // Code0: piecewise linear with linear extrapolation.
        if (alpha >= 0.0) {
            double der = up - nominal;
            *out_val = nominal + der * alpha;
            *out_der = der;
        } else {
            double der = nominal - down;
            *out_val = nominal + der * alpha;
            *out_der = der;
        }
        return;
    }

    // Default: Code4p
    double delta_up = up - nominal;
    double delta_dn = nominal - down;

    if (alpha > 1.0) {
        *out_val = nominal + delta_up * alpha;
        *out_der = delta_up;
        return;
    }
    if (alpha < -1.0) {
        *out_val = nominal + delta_dn * alpha;
        *out_der = delta_dn;
        return;
    }

    double s = 0.5 * (delta_up + delta_dn);
    double a = 0.0625 * (delta_up - delta_dn);

    double a2 = alpha * alpha;
    double a3 = a2 * alpha;
    double a4 = a2 * a2;
    double a5 = a4 * alpha;
    double a6 = a3 * a3;

    double tmp3 = (3.0 * a6) - (10.0 * a4) + (15.0 * a2);
    double dtmp3 = (18.0 * a5) - (40.0 * a3) + (30.0 * alpha);

    double delta = alpha * s + tmp3 * a;
    double ddelta = s + dtmp3 * a;

    *out_val = nominal + delta;
    *out_der = ddelta;
}

extern "C" __global__ void unbinned_weight_sys_apply(
    const double* __restrict__ g_params,      /* [n_params] */
    const double* __restrict__ g_w_nom,       /* [n_events] */
    const double* __restrict__ g_w_down,      /* [n_sys × n_events] (row-major by sys) */
    const double* __restrict__ g_w_up,        /* [n_sys × n_events] (row-major by sys) */
    const unsigned int* __restrict__ g_alpha_param_idx, /* [n_sys] */
    const unsigned int* __restrict__ g_interp_code,     /* [n_sys] */
    double* __restrict__ g_w_out,             /* [n_events] */
    double* __restrict__ g_dw_out,            /* [n_sys × n_events] */
    unsigned int n_params,
    unsigned int n_events,
    unsigned int n_sys
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_events) {
        return;
    }

    const double eps_w = DBL_MIN;

    double nom = g_w_nom[i];
    if (!isfinite(nom)) {
        nom = 1.0;
    }

    double w = nom;
    for (unsigned int s = 0; s < n_sys; s++) {
        unsigned int aidx = g_alpha_param_idx[s];
        double alpha = (aidx < n_params) ? g_params[aidx] : 0.0;

        double down = g_w_down[s * n_events + i];
        double up = g_w_up[s * n_events + i];

        double val = nom;
        double der = 0.0;
        histosys_interp(alpha, down, nom, up, g_interp_code[s], &val, &der);

        // Additive around nominal (matches CPU morphing_kde baseline).
        w += (val - nom);
        g_dw_out[s * n_events + i] = der;
    }

    if (!isfinite(w) || w <= 0.0) {
        w = eps_w;
        for (unsigned int s = 0; s < n_sys; s++) {
            g_dw_out[s * n_events + i] = 0.0;
        }
    }

    g_w_out[i] = w;
}

