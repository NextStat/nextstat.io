/**
 * Shared types, constants, and device helper functions for unbinned CUDA kernels.
 *
 * Extracted from unbinned_nll_grad.cu for reuse across:
 *   - unbinned_nll_grad.cu (batch toy fitting, single-dataset NLL, toy sampling)
 *   - unbinned_lbfgs_fit.cu (persistent GPU-native L-BFGS optimizer)
 */

#ifndef NEXTSTAT_UNBINNED_COMMON_CUH
#define NEXTSTAT_UNBINNED_COMMON_CUH

#include <math.h>
#include <float.h>

/* ---------- Constants (must match ns-compute/unbinned_types.rs) ----------- */
#define PDF_GAUSSIAN    0u
#define PDF_EXPONENTIAL 1u
#define PDF_CRYSTAL_BALL 2u
#define PDF_DOUBLE_CRYSTAL_BALL 3u
#define PDF_CHEBYSHEV 4u
#define PDF_HISTOGRAM 5u

#define YIELD_FIXED     0u
#define YIELD_PARAMETER 1u
#define YIELD_SCALED    2u

// Rate modifier kinds.
#define RATE_NORM_SYS   0u
#define RATE_WEIGHT_SYS 1u

// WeightSys interp codes.
#define INTERP_CODE0  0u
#define INTERP_CODE4P 1u

/* ---------- Struct mirrors of Rust #[repr(C)] types ---------------------- */

struct GpuUnbinnedProcessDesc {
    double base_yield;
    unsigned int pdf_kind;
    unsigned int yield_kind;
    unsigned int obs_index;
    unsigned int shape_param_offset;
    unsigned int n_shape_params;
    unsigned int yield_param_idx;
    unsigned int rate_mod_offset;
    unsigned int n_rate_mods;
    unsigned int pdf_aux_offset;
    unsigned int pdf_aux_len;
};

struct GpuUnbinnedGaussConstraintEntry {
    double center;
    double inv_width;
    unsigned int param_idx;
    unsigned int _pad;
};

struct GpuUnbinnedRateModifierDesc {
    unsigned int kind;
    unsigned int alpha_param_idx;
    unsigned int interp_code;
    unsigned int _pad;
    double lo;
    double hi;
};

/**
 * Per-channel descriptor for multi-channel GPU-native L-BFGS.
 *
 * Holds device pointers to channel-local static buffers. This avoids any
 * host-side download/re-upload when launching the persistent optimizer in
 * multi-channel mode.
 *
 * Gaussian constraints and constraint_const should be assigned to only one
 * channel (typically channel 0) to avoid double-counting.
 */
struct GpuChannelDesc {
    const double* obs_flat;                           /* flattened events for this channel */
    const unsigned int* toy_offsets;                  /* [n_toys+1] prefix sums */
    const struct GpuUnbinnedProcessDesc* procs;        /* [n_procs] */
    const struct GpuUnbinnedRateModifierDesc* rate_mods; /* [total_rate_mods] */
    const unsigned int* shape_pidx;                   /* [total_shape_params] */
    const double* pdf_aux_f64;                        /* [pdf_aux_len] */
    const struct GpuUnbinnedGaussConstraintEntry* gauss; /* [n_gauss] (channel 0 only) */
    unsigned int n_procs;
    unsigned int total_rate_mods;
    unsigned int total_shape_params;
    unsigned int n_gauss;
    double obs_lo;
    double obs_hi;
    double constraint_const;
};

/* ---------- Helpers: standard normal ------------------------------------ */

__device__ inline double stdnorm_logpdf(double z) {
    // -0.5*z^2 - log(sqrt(2*pi))
    return -0.5 * z * z - 0.91893853320467274178032973640562;
}

__device__ inline double stdnorm_pdf(double z) {
    return exp(stdnorm_logpdf(z));
}

__device__ inline double stdnorm_cdf(double z) {
    // Phi(z) = 0.5 * erfc(-z / sqrt(2))
    const double inv_sqrt2 = 0.70710678118654752440084436210485;
    return 0.5 * erfc(-z * inv_sqrt2);
}

/* ---------- Helpers: PDFs (logp + dlogp) --------------------------------- */

__device__ inline double gaussian_logp_only(
    double x,
    double mu,
    double sigma,
    double a,
    double b
) {
    if (!isfinite(mu) || !isfinite(sigma) || sigma <= 0.0) {
        return -INFINITY;
    }
    double inv_sigma = 1.0 / sigma;
    double z_a = (a - mu) * inv_sigma;
    double z_b = (b - mu) * inv_sigma;
    double z_x = (x - mu) * inv_sigma;

    double z = stdnorm_cdf(z_b) - stdnorm_cdf(z_a);
    if (!isfinite(z) || z <= 0.0) {
        z = DBL_MIN;
    }
    double log_z = log(z);
    return stdnorm_logpdf(z_x) - log(sigma) - log_z;
}

__device__ inline void gaussian_logp_grad(
    double x,
    double mu,
    double sigma,
    double a,
    double b,
    double* __restrict__ out_logp,
    double* __restrict__ out_dmu,
    double* __restrict__ out_dsigma
) {
    if (!isfinite(mu) || !isfinite(sigma) || sigma <= 0.0) {
        *out_logp = -INFINITY;
        *out_dmu = 0.0;
        *out_dsigma = 0.0;
        return;
    }

    double inv_sigma = 1.0 / sigma;
    double z_a = (a - mu) * inv_sigma;
    double z_b = (b - mu) * inv_sigma;
    double z_x = (x - mu) * inv_sigma;

    double z = stdnorm_cdf(z_b) - stdnorm_cdf(z_a);
    if (!isfinite(z) || z <= 0.0) {
        z = DBL_MIN;
    }
    double log_z = log(z);

    // Derivatives of logZ.
    double phi_a = stdnorm_pdf(z_a);
    double phi_b = stdnorm_pdf(z_b);
    double dlogz_dmu = (phi_a - phi_b) * inv_sigma / z;
    double dlogz_dsigma = (z_a * phi_a - z_b * phi_b) * inv_sigma / z;

    *out_logp = stdnorm_logpdf(z_x) - log(sigma) - log_z;
    *out_dmu = z_x * inv_sigma - dlogz_dmu;
    *out_dsigma = (z_x * z_x - 1.0) * inv_sigma - dlogz_dsigma;
}

__device__ inline void exp_logz_ex(
    double lambda,
    double a,
    double b,
    double* __restrict__ out_logz,
    double* __restrict__ out_ex
) {
    // Matches ns-unbinned ExponentialPdf::logz_and_ex.
    if (!isfinite(lambda) || !isfinite(a) || !isfinite(b) || !(a < b)) {
        *out_logz = INFINITY;
        *out_ex = 0.5 * (a + b);
        return;
    }

    if (fabs(lambda) < 1e-12) {
        double z = b - a;
        if (!(isfinite(z) && z > 0.0)) {
            *out_logz = INFINITY;
            *out_ex = 0.5 * (a + b);
            return;
        }
        *out_logz = log(z);
        *out_ex = 0.5 * (a + b);
        return;
    }

    double t_a = lambda * a;
    double t_b = lambda * b;
    double hi_t = (t_b >= t_a) ? t_b : t_a;
    double lo_t = (t_b >= t_a) ? t_a : t_b;

    // r = exp(lo-hi) in (0,1)
    double r = exp(lo_t - hi_t);
    double log_num = hi_t + log1p(-r);
    double log_z = log_num - log(fabs(lambda));

    double denom = 1.0 - r;
    if (!(denom > 0.0) || !isfinite(denom)) {
        // Numerically indistinguishable from uniform.
        *out_logz = log(b - a);
        *out_ex = 0.5 * (a + b);
        return;
    }

    double x_hi = (t_b >= t_a) ? b : a;
    double x_lo = (t_b >= t_a) ? a : b;
    double ratio = (x_hi - x_lo * r) / denom;
    double ex = ratio - 1.0 / lambda;

    *out_logz = log_z;
    *out_ex = ex;
}

__device__ inline double exponential_logp_only(
    double x,
    double lambda,
    double a,
    double b
) {
    double logz, ex;
    exp_logz_ex(lambda, a, b, &logz, &ex);
    return lambda * x - logz;
}

__device__ inline void exponential_logp_grad(
    double x,
    double lambda,
    double a,
    double b,
    double* __restrict__ out_logp,
    double* __restrict__ out_dlambda
) {
    double logz, ex;
    exp_logz_ex(lambda, a, b, &logz, &ex);
    *out_logp = lambda * x - logz;
    *out_dlambda = x - ex;
}

/* ---------- Helpers: Histogram PDF (bin_edges + log_density) ------------ */

__device__ inline double histogram_logp_only(
    double x,
    const double* __restrict__ aux_f64,
    unsigned int aux_offset,
    unsigned int aux_len
) {
    // Layout in aux_f64:
    //   edges[0..n_bins] (length n_bins+1)
    //   log_density[0..n_bins-1] (length n_bins)
    // Total length = 2*n_bins + 1.
    if (aux_len < 3u) {
        return -INFINITY;
    }
    unsigned int n_bins = (aux_len - 1u) / 2u;
    if (n_bins == 0u) {
        return -INFINITY;
    }

    const double* __restrict__ edges = aux_f64 + (size_t)aux_offset;
    const double* __restrict__ logdens = edges + (size_t)(n_bins + 1u);

    double x_min = edges[0];
    double x_max = edges[n_bins];
    if (!isfinite(x) || !isfinite(x_min) || !isfinite(x_max)) {
        return -INFINITY;
    }
    if (x < x_min || x > x_max) {
        return -INFINITY;
    }
    if (x >= x_max) {
        return logdens[n_bins - 1u];
    }

    // Find the largest k such that edges[k] <= x, with k in [0, n_bins-1].
    unsigned int lo = 0u;
    unsigned int hi = n_bins;
    while (lo + 1u < hi) {
        unsigned int mid = (lo + hi) >> 1;
        double e = edges[mid];
        if (e <= x) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return logdens[lo];
}

/* ---------- Helpers: Crystal Ball / Chebyshev PDFs ---------------------- */

__device__ inline double gauss_logf(double t) {
    return -0.5 * t * t;
}

__device__ inline double gauss_integral(double t1, double t2) {
    // ∫ exp(-0.5 t^2) dt = sqrt(2π) * (Φ(t2) - Φ(t1))
    const double sqrt_2pi = 2.5066282746310005024157652848110;
    return sqrt_2pi * (stdnorm_cdf(t2) - stdnorm_cdf(t1));
}

__device__ inline int cb_tail_init(double alpha, double n, double* __restrict__ out_log_a, double* __restrict__ out_b) {
    if (!(isfinite(alpha) && alpha > 0.0 && isfinite(n) && n > 1.0)) {
        *out_log_a = 0.0;
        *out_b = 0.0;
        return 0;
    }
    double log_a = n * log(n / alpha) - 0.5 * alpha * alpha;
    double b = n / alpha - alpha;
    if (!isfinite(log_a) || !isfinite(b)) {
        *out_log_a = 0.0;
        *out_b = 0.0;
        return 0;
    }
    *out_log_a = log_a;
    *out_b = b;
    return 1;
}

__device__ inline double cb_logf_left(double t, double log_a, double b, double n) {
    return log_a - n * log(b - t);
}

__device__ inline double cb_logf_right(double t, double log_a, double b, double n) {
    return log_a - n * log(b + t);
}

__device__ inline double cb_dlogf_dt_left(double t, double b, double n) {
    return n / (b - t);
}

__device__ inline double cb_dlogf_dt_right(double t, double b, double n) {
    return -n / (b + t);
}

__device__ inline double cb_dlogf_dalpha_left(double t, double alpha, double n, double b) {
    double dln_a = -(n / alpha + alpha);
    double db = -(n / (alpha * alpha) + 1.0);
    return dln_a - n * db / (b - t);
}

__device__ inline double cb_dlogf_dn_left(double t, double alpha, double n, double b) {
    double dln_a = 1.0 + log(n / alpha);
    double db = 1.0 / alpha;
    return dln_a - log(b - t) - n * db / (b - t);
}

__device__ inline double cb_dlogf_dalpha_right(double t, double alpha, double n, double b) {
    double dln_a = -(n / alpha + alpha);
    double db = -(n / (alpha * alpha) + 1.0);
    return dln_a - n * db / (b + t);
}

__device__ inline double cb_dlogf_dn_right(double t, double alpha, double n, double b) {
    double dln_a = 1.0 + log(n / alpha);
    double db = 1.0 / alpha;
    return dln_a - log(b + t) - n * db / (b + t);
}

__device__ inline int cb_integral_left_only(
    double t1,
    double t2,
    double n,
    double log_a,
    double b,
    double* __restrict__ out_i
) {
    // Tail integral on the left side for t in [t1, t2] where t2 <= -alpha.
    double m = n - 1.0;
    double a = exp(log_a);
    double u2 = pow(b - t2, -m);
    double u1 = pow(b - t1, -m);
    double i = a / m * (u2 - u1);
    if (!isfinite(i) || !(i > 0.0)) {
        *out_i = 0.0;
        return 0;
    }
    *out_i = i;
    return 1;
}

__device__ inline int cb_integral_right_only(
    double t1,
    double t2,
    double n,
    double log_a,
    double b,
    double* __restrict__ out_i
) {
    // Tail integral on the right side for t in [t1, t2] where t1 >= +alpha.
    double m = n - 1.0;
    double a = exp(log_a);
    double u1 = pow(b + t1, -m);
    double u2 = pow(b + t2, -m);
    double i = a / m * (u1 - u2);
    if (!isfinite(i) || !(i > 0.0)) {
        *out_i = 0.0;
        return 0;
    }
    *out_i = i;
    return 1;
}

__device__ inline int cb_integral_and_derivs_left(
    double t1,
    double t2,
    double alpha,
    double n,
    double log_a,
    double b,
    double* __restrict__ out_i,
    double* __restrict__ out_di_dalpha,
    double* __restrict__ out_di_dn
) {
    double m = n - 1.0;
    double a = exp(log_a);

    double b1 = b - t1;
    double b2 = b - t2;
    double u1 = pow(b1, -m);
    double u2 = pow(b2, -m);
    double i = a / m * (u2 - u1);
    if (!isfinite(i) || !(i > 0.0)) {
        *out_i = 0.0;
        *out_di_dalpha = 0.0;
        *out_di_dn = 0.0;
        return 0;
    }

    double dln_a_dalpha = -(n / alpha + alpha);
    double dln_a_dn = 1.0 + log(n / alpha);
    double db_dalpha = -(n / (alpha * alpha) + 1.0);
    double db_dn = 1.0 / alpha;

    double v1 = pow(b1, -n);
    double v2 = pow(b2, -n);

    double di_dalpha = i * dln_a_dalpha - a * db_dalpha * (v2 - v1);

    double du1_dn = u1 * (-log(b1) - m * db_dn / b1);
    double du2_dn = u2 * (-log(b2) - m * db_dn / b2);
    double di_dn = i * dln_a_dn - i / m + (a / m) * (du2_dn - du1_dn);

    *out_i = i;
    *out_di_dalpha = di_dalpha;
    *out_di_dn = di_dn;
    return 1;
}

__device__ inline int cb_integral_and_derivs_right(
    double t1,
    double t2,
    double alpha,
    double n,
    double log_a,
    double b,
    double* __restrict__ out_i,
    double* __restrict__ out_di_dalpha,
    double* __restrict__ out_di_dn
) {
    double m = n - 1.0;
    double a = exp(log_a);

    double b1 = b + t1;
    double b2 = b + t2;
    double u1 = pow(b1, -m);
    double u2 = pow(b2, -m);
    double i = a / m * (u1 - u2);
    if (!isfinite(i) || !(i > 0.0)) {
        *out_i = 0.0;
        *out_di_dalpha = 0.0;
        *out_di_dn = 0.0;
        return 0;
    }

    double dln_a_dalpha = -(n / alpha + alpha);
    double dln_a_dn = 1.0 + log(n / alpha);
    double db_dalpha = -(n / (alpha * alpha) + 1.0);
    double db_dn = 1.0 / alpha;

    double v1 = pow(b1, -n);
    double v2 = pow(b2, -n);

    double di_dalpha = i * dln_a_dalpha - a * db_dalpha * (v1 - v2);

    double du1_dn = u1 * (-log(b1) - m * db_dn / b1);
    double du2_dn = u2 * (-log(b2) - m * db_dn / b2);
    double di_dn = i * dln_a_dn - i / m + (a / m) * (du1_dn - du2_dn);

    *out_i = i;
    *out_di_dalpha = di_dalpha;
    *out_di_dn = di_dn;
    return 1;
}

__device__ inline double crystal_ball_logp_only(
    double x,
    double mu,
    double sigma,
    double alpha,
    double n,
    double a,
    double b
) {
    if (!isfinite(mu) || !isfinite(sigma) || sigma <= 0.0) {
        return -INFINITY;
    }
    double log_a = 0.0;
    double bt = 0.0;
    if (!cb_tail_init(alpha, n, &log_a, &bt)) {
        return -INFINITY;
    }

    double inv_sigma = 1.0 / sigma;
    double t_a = (a - mu) * inv_sigma;
    double t_b = (b - mu) * inv_sigma;
    double t = (x - mu) * inv_sigma;
    double t0 = -alpha;

    double i = 0.0;
    if (t_b <= t0) {
        if (!cb_integral_left_only(t_a, t_b, n, log_a, bt, &i)) {
            return -INFINITY;
        }
    } else if (t_a >= t0) {
        i = gauss_integral(t_a, t_b);
    } else {
        double it = 0.0;
        if (!cb_integral_left_only(t_a, t0, n, log_a, bt, &it)) {
            return -INFINITY;
        }
        i = it + gauss_integral(t0, t_b);
    }
    if (!isfinite(i) || !(i > 0.0)) {
        return -INFINITY;
    }
    double log_i = log(i);

    double logf = (t > t0) ? gauss_logf(t) : cb_logf_left(t, log_a, bt, n);
    return logf - log(sigma) - log_i;
}

__device__ inline void crystal_ball_logp_grad(
    double x,
    double mu,
    double sigma,
    double alpha,
    double n,
    double a,
    double b,
    double* __restrict__ out_logp,
    double* __restrict__ out_dmu,
    double* __restrict__ out_dsigma,
    double* __restrict__ out_dalpha,
    double* __restrict__ out_dn
) {
    if (!isfinite(mu) || !isfinite(sigma) || sigma <= 0.0) {
        *out_logp = -INFINITY;
        *out_dmu = 0.0;
        *out_dsigma = 0.0;
        *out_dalpha = 0.0;
        *out_dn = 0.0;
        return;
    }
    double log_a = 0.0;
    double bt = 0.0;
    if (!cb_tail_init(alpha, n, &log_a, &bt)) {
        *out_logp = -INFINITY;
        *out_dmu = 0.0;
        *out_dsigma = 0.0;
        *out_dalpha = 0.0;
        *out_dn = 0.0;
        return;
    }

    double inv_sigma = 1.0 / sigma;
    double t_a = (a - mu) * inv_sigma;
    double t_b = (b - mu) * inv_sigma;
    double t = (x - mu) * inv_sigma;
    double t0 = -alpha;

    double i = 0.0;
    double di_dalpha = 0.0;
    double di_dn = 0.0;
    if (t_b <= t0) {
        if (!cb_integral_and_derivs_left(t_a, t_b, alpha, n, log_a, bt, &i, &di_dalpha, &di_dn)) {
            *out_logp = -INFINITY;
            *out_dmu = 0.0;
            *out_dsigma = 0.0;
            *out_dalpha = 0.0;
            *out_dn = 0.0;
            return;
        }
    } else if (t_a >= t0) {
        i = gauss_integral(t_a, t_b);
        di_dalpha = 0.0;
        di_dn = 0.0;
    } else {
        double it = 0.0;
        double dit_da = 0.0;
        double dit_dn = 0.0;
        if (!cb_integral_and_derivs_left(t_a, t0, alpha, n, log_a, bt, &it, &dit_da, &dit_dn)) {
            *out_logp = -INFINITY;
            *out_dmu = 0.0;
            *out_dsigma = 0.0;
            *out_dalpha = 0.0;
            *out_dn = 0.0;
            return;
        }
        i = it + gauss_integral(t0, t_b);
        di_dalpha = dit_da;
        di_dn = dit_dn;
    }
    if (!isfinite(i) || !(i > 0.0)) {
        *out_logp = -INFINITY;
        *out_dmu = 0.0;
        *out_dsigma = 0.0;
        *out_dalpha = 0.0;
        *out_dn = 0.0;
        return;
    }

    double log_i = log(i);
    double dlogi_dalpha = di_dalpha / i;
    double dlogi_dn = di_dn / i;

    double logf_a = (t_a > t0) ? gauss_logf(t_a) : cb_logf_left(t_a, log_a, bt, n);
    double logf_b = (t_b > t0) ? gauss_logf(t_b) : cb_logf_left(t_b, log_a, bt, n);
    double f_a = exp(logf_a);
    double f_b = exp(logf_b);

    double dlogi_dmu = (f_a - f_b) * inv_sigma / i;
    double dlogi_dsigma = (f_a * t_a - f_b * t_b) * inv_sigma / i;

    int is_gauss = (t > t0);
    double logf = 0.0;
    double dlogf_dt = 0.0;
    double dlogf_dalpha = 0.0;
    double dlogf_dn = 0.0;
    if (is_gauss) {
        logf = gauss_logf(t);
        dlogf_dt = -t;
        dlogf_dalpha = 0.0;
        dlogf_dn = 0.0;
    } else {
        logf = cb_logf_left(t, log_a, bt, n);
        dlogf_dt = cb_dlogf_dt_left(t, bt, n);
        dlogf_dalpha = cb_dlogf_dalpha_left(t, alpha, n, bt);
        dlogf_dn = cb_dlogf_dn_left(t, alpha, n, bt);
    }

    *out_logp = logf - log(sigma) - log_i;

    *out_dmu = -inv_sigma * dlogf_dt - dlogi_dmu;
    *out_dsigma = -t * inv_sigma * dlogf_dt - inv_sigma - dlogi_dsigma;
    *out_dalpha = dlogf_dalpha - dlogi_dalpha;
    *out_dn = dlogf_dn - dlogi_dn;
}

__device__ inline double double_crystal_ball_logp_only(
    double x,
    double mu,
    double sigma,
    double alpha_l,
    double n_l,
    double alpha_r,
    double n_r,
    double a,
    double b
) {
    if (!isfinite(mu) || !isfinite(sigma) || sigma <= 0.0) {
        return -INFINITY;
    }
    double log_a_l = 0.0, b_l = 0.0;
    double log_a_r = 0.0, b_r = 0.0;
    if (!cb_tail_init(alpha_l, n_l, &log_a_l, &b_l)) {
        return -INFINITY;
    }
    if (!cb_tail_init(alpha_r, n_r, &log_a_r, &b_r)) {
        return -INFINITY;
    }

    double inv_sigma = 1.0 / sigma;
    double t_a = (a - mu) * inv_sigma;
    double t_b = (b - mu) * inv_sigma;
    double t = (x - mu) * inv_sigma;

    double t_l = -alpha_l;
    double t_r = alpha_r;

    double i = 0.0;
    if (t_a < t_l) {
        double t2 = (t_b < t_l) ? t_b : t_l;
        double it = 0.0;
        if (!cb_integral_left_only(t_a, t2, n_l, log_a_l, b_l, &it)) {
            return -INFINITY;
        }
        i += it;
    }
    double core_lo = (t_a > t_l) ? t_a : t_l;
    double core_hi = (t_b < t_r) ? t_b : t_r;
    if (core_hi > core_lo) {
        i += gauss_integral(core_lo, core_hi);
    }
    if (t_b > t_r) {
        double t1 = (t_a > t_r) ? t_a : t_r;
        double it = 0.0;
        if (!cb_integral_right_only(t1, t_b, n_r, log_a_r, b_r, &it)) {
            return -INFINITY;
        }
        i += it;
    }
    if (!isfinite(i) || !(i > 0.0)) {
        return -INFINITY;
    }
    double log_i = log(i);

    double logf = 0.0;
    if (t < t_l) {
        logf = cb_logf_left(t, log_a_l, b_l, n_l);
    } else if (t > t_r) {
        logf = cb_logf_right(t, log_a_r, b_r, n_r);
    } else {
        logf = gauss_logf(t);
    }
    return logf - log(sigma) - log_i;
}

__device__ inline void double_crystal_ball_logp_grad(
    double x,
    double mu,
    double sigma,
    double alpha_l,
    double n_l,
    double alpha_r,
    double n_r,
    double a,
    double b,
    double* __restrict__ out_logp,
    double* __restrict__ out_dmu,
    double* __restrict__ out_dsigma,
    double* __restrict__ out_dalpha_l,
    double* __restrict__ out_dn_l,
    double* __restrict__ out_dalpha_r,
    double* __restrict__ out_dn_r
) {
    if (!isfinite(mu) || !isfinite(sigma) || sigma <= 0.0) {
        *out_logp = -INFINITY;
        *out_dmu = 0.0;
        *out_dsigma = 0.0;
        *out_dalpha_l = 0.0;
        *out_dn_l = 0.0;
        *out_dalpha_r = 0.0;
        *out_dn_r = 0.0;
        return;
    }
    double log_a_l = 0.0, b_l = 0.0;
    double log_a_r = 0.0, b_r = 0.0;
    if (!cb_tail_init(alpha_l, n_l, &log_a_l, &b_l) || !cb_tail_init(alpha_r, n_r, &log_a_r, &b_r)) {
        *out_logp = -INFINITY;
        *out_dmu = 0.0;
        *out_dsigma = 0.0;
        *out_dalpha_l = 0.0;
        *out_dn_l = 0.0;
        *out_dalpha_r = 0.0;
        *out_dn_r = 0.0;
        return;
    }

    double inv_sigma = 1.0 / sigma;
    double t_a = (a - mu) * inv_sigma;
    double t_b = (b - mu) * inv_sigma;
    double t = (x - mu) * inv_sigma;

    double t_l = -alpha_l;
    double t_r = alpha_r;

    double i = 0.0;
    double di_dalpha_l = 0.0;
    double di_dn_l = 0.0;
    double di_dalpha_r = 0.0;
    double di_dn_r = 0.0;

    if (t_a < t_l) {
        double t2 = (t_b < t_l) ? t_b : t_l;
        double it = 0.0, dit_da = 0.0, dit_dn = 0.0;
        if (!cb_integral_and_derivs_left(t_a, t2, alpha_l, n_l, log_a_l, b_l, &it, &dit_da, &dit_dn)) {
            *out_logp = -INFINITY;
            *out_dmu = 0.0;
            *out_dsigma = 0.0;
            *out_dalpha_l = 0.0;
            *out_dn_l = 0.0;
            *out_dalpha_r = 0.0;
            *out_dn_r = 0.0;
            return;
        }
        i += it;
        di_dalpha_l += dit_da;
        di_dn_l += dit_dn;
    }

    double core_lo = (t_a > t_l) ? t_a : t_l;
    double core_hi = (t_b < t_r) ? t_b : t_r;
    if (core_hi > core_lo) {
        i += gauss_integral(core_lo, core_hi);
    }

    if (t_b > t_r) {
        double t1 = (t_a > t_r) ? t_a : t_r;
        double it = 0.0, dit_da = 0.0, dit_dn = 0.0;
        if (!cb_integral_and_derivs_right(t1, t_b, alpha_r, n_r, log_a_r, b_r, &it, &dit_da, &dit_dn)) {
            *out_logp = -INFINITY;
            *out_dmu = 0.0;
            *out_dsigma = 0.0;
            *out_dalpha_l = 0.0;
            *out_dn_l = 0.0;
            *out_dalpha_r = 0.0;
            *out_dn_r = 0.0;
            return;
        }
        i += it;
        di_dalpha_r += dit_da;
        di_dn_r += dit_dn;
    }

    if (!isfinite(i) || !(i > 0.0)) {
        *out_logp = -INFINITY;
        *out_dmu = 0.0;
        *out_dsigma = 0.0;
        *out_dalpha_l = 0.0;
        *out_dn_l = 0.0;
        *out_dalpha_r = 0.0;
        *out_dn_r = 0.0;
        return;
    }

    double log_i = log(i);
    double dlogi_dalpha_l = di_dalpha_l / i;
    double dlogi_dn_l = di_dn_l / i;
    double dlogi_dalpha_r = di_dalpha_r / i;
    double dlogi_dn_r = di_dn_r / i;

    double logf_a = 0.0;
    if (t_a < t_l) {
        logf_a = cb_logf_left(t_a, log_a_l, b_l, n_l);
    } else if (t_a > t_r) {
        logf_a = cb_logf_right(t_a, log_a_r, b_r, n_r);
    } else {
        logf_a = gauss_logf(t_a);
    }
    double logf_b = 0.0;
    if (t_b < t_l) {
        logf_b = cb_logf_left(t_b, log_a_l, b_l, n_l);
    } else if (t_b > t_r) {
        logf_b = cb_logf_right(t_b, log_a_r, b_r, n_r);
    } else {
        logf_b = gauss_logf(t_b);
    }
    double f_a = exp(logf_a);
    double f_b = exp(logf_b);

    double dlogi_dmu = (f_a - f_b) * inv_sigma / i;
    double dlogi_dsigma = (f_a * t_a - f_b * t_b) * inv_sigma / i;

    double logf = 0.0;
    double dlogf_dt = 0.0;
    double dlogf_dalpha_l = 0.0;
    double dlogf_dn_l = 0.0;
    double dlogf_dalpha_r = 0.0;
    double dlogf_dn_r = 0.0;

    if (t < t_l) {
        logf = cb_logf_left(t, log_a_l, b_l, n_l);
        dlogf_dt = cb_dlogf_dt_left(t, b_l, n_l);
        dlogf_dalpha_l = cb_dlogf_dalpha_left(t, alpha_l, n_l, b_l);
        dlogf_dn_l = cb_dlogf_dn_left(t, alpha_l, n_l, b_l);
        dlogf_dalpha_r = 0.0;
        dlogf_dn_r = 0.0;
    } else if (t > t_r) {
        logf = cb_logf_right(t, log_a_r, b_r, n_r);
        dlogf_dt = cb_dlogf_dt_right(t, b_r, n_r);
        dlogf_dalpha_l = 0.0;
        dlogf_dn_l = 0.0;
        dlogf_dalpha_r = cb_dlogf_dalpha_right(t, alpha_r, n_r, b_r);
        dlogf_dn_r = cb_dlogf_dn_right(t, alpha_r, n_r, b_r);
    } else {
        logf = gauss_logf(t);
        dlogf_dt = -t;
        dlogf_dalpha_l = 0.0;
        dlogf_dn_l = 0.0;
        dlogf_dalpha_r = 0.0;
        dlogf_dn_r = 0.0;
    }

    *out_logp = logf - log(sigma) - log_i;

    *out_dmu = -inv_sigma * dlogf_dt - dlogi_dmu;
    *out_dsigma = -t * inv_sigma * dlogf_dt - inv_sigma - dlogi_dsigma;
    *out_dalpha_l = dlogf_dalpha_l - dlogi_dalpha_l;
    *out_dn_l = dlogf_dn_l - dlogi_dn_l;
    *out_dalpha_r = dlogf_dalpha_r - dlogi_dalpha_r;
    *out_dn_r = dlogf_dn_r - dlogi_dn_r;
}

__device__ inline double chebyshev_xprime(double x, double a, double b) {
    double w = b - a;
    double xp = (2.0 * x - (a + b)) / w;
    if (xp < -1.0) {
        xp = -1.0;
    }
    if (xp > 1.0) {
        xp = 1.0;
    }
    return xp;
}

__device__ inline double chebyshev_logp_only(
    double x,
    const unsigned int* __restrict__ shape_pidx,
    unsigned int shape_param_offset,
    unsigned int order,
    const double* __restrict__ params,
    double a,
    double b
) {
    if (order == 0u) {
        return -INFINITY;
    }
    double w = b - a;
    if (!isfinite(w) || !(w > 0.0)) {
        return -INFINITY;
    }

    double i = w;
    for (unsigned int j = 0; j < order; j++) {
        unsigned int c_idx = shape_pidx[shape_param_offset + j];
        double c = params[c_idx];
        unsigned int k = j + 1u;
        if ((k & 1u) == 0u) {
            double denom = 1.0 - (double)k * (double)k;
            i += w * c / denom;
        }
    }
    if (!isfinite(i) || !(i > 0.0)) {
        return -INFINITY;
    }
    double log_i = log(i);

    double xp = chebyshev_xprime(x, a, b);
    double f = 1.0;
    double tkm1 = 1.0; // T0
    double tk = xp;    // T1
    for (unsigned int j = 0; j < order; j++) {
        unsigned int c_idx = shape_pidx[shape_param_offset + j];
        double c = params[c_idx];
        unsigned int k = j + 1u;
        double t = 0.0;
        if (k == 1u) {
            t = tk;
        } else {
            double tkp1 = 2.0 * xp * tk - tkm1;
            t = tkp1;
            tkm1 = tk;
            tk = tkp1;
        }
        f += c * t;
    }
    if (!isfinite(f) || !(f > 0.0)) {
        return -INFINITY;
    }
    return log(f) - log_i;
}

/* ---------- Helpers: Rate modifiers (yield systematics) ------------------ */

__device__ inline void histosys_interp(
    double alpha,
    double down,
    double nominal,
    double up,
    unsigned int code,
    double* __restrict__ out_val,
    double* __restrict__ out_der
) {
    // Matches ns-unbinned/src/interp.rs for Code0 and Code4p.
    if (code == INTERP_CODE0) {
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

    // Code4p: smooth polynomial in [-1,1], linear extrapolation outside.
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

__device__ inline void rate_modifier_factor_dlogf(
    const struct GpuUnbinnedRateModifierDesc* __restrict__ mods,
    unsigned int midx,
    const double* __restrict__ params,
    double* __restrict__ out_f,
    double* __restrict__ out_dlogf
) {
    struct GpuUnbinnedRateModifierDesc m = mods[midx];
    unsigned int aidx = m.alpha_param_idx;
    double alpha = params[aidx];

    if (m.kind == RATE_NORM_SYS) {
        double lo = m.lo;
        double hi = m.hi;
        if (!(isfinite(lo) && lo > 0.0 && isfinite(hi) && hi > 0.0 && isfinite(alpha))) {
            *out_f = 1.0;
            *out_dlogf = 0.0;
            return;
        }
        double log_hi = log(hi);
        double log_lo = log(lo);
        if (alpha >= 0.0) {
            *out_f = exp(alpha * log_hi);
            *out_dlogf = log_hi;
        } else {
            *out_f = exp(-alpha * log_lo);
            *out_dlogf = -log_lo;
        }
        return;
    }

    if (m.kind == RATE_WEIGHT_SYS) {
        double lo = m.lo;
        double hi = m.hi;
        if (!(isfinite(lo) && lo > 0.0 && isfinite(hi) && hi > 0.0 && isfinite(alpha))) {
            *out_f = 1.0;
            *out_dlogf = 0.0;
            return;
        }
        double val = 1.0;
        double der = 0.0;
        histosys_interp(alpha, lo, 1.0, hi, m.interp_code, &val, &der);
        if (!isfinite(val) || val <= 0.0) {
            val = DBL_MIN;
            der = 0.0;
        }
        *out_f = val;
        *out_dlogf = (val > 0.0) ? (der / val) : 0.0;
        return;
    }

    *out_f = 1.0;
    *out_dlogf = 0.0;
}

#endif /* NEXTSTAT_UNBINNED_COMMON_CUH */
