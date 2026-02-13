/**
 * Shared device helper functions for HistFactory CUDA kernels.
 *
 * Extracted from batch_nll_grad.cu for reuse across:
 *   - batch_nll_grad.cu (batch toy fitting)
 *   - differentiable_nll_grad.cu (PyTorch differentiable layer)
 */

#ifndef NEXTSTAT_COMMON_CUH
#define NEXTSTAT_COMMON_CUH

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
__device__ inline double normsys_code4_value(
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
__device__ inline double normsys_code4_deriv_over_val(
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
__device__ inline double histosys_delta(double alpha, double delta_up, double delta_dn) {
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
__device__ inline double histosys_delta_deriv(double alpha, double delta_up, double delta_dn) {
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

#endif /* NEXTSTAT_COMMON_CUH */
