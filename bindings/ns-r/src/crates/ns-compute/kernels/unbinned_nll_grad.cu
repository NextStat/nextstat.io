/**
 * Unbinned (event-level) extended likelihood CUDA kernels.
 *
 * Computes:
 *   NLL(params) = nu_tot(params) - sum_{i in events} log( sum_p nu_p(params) * p_p(x_i | params) )
 *                + gaussian_constraints(params)
 *
 * plus the analytical gradient w.r.t. the global parameter vector.
 *
 * Phase 1 GPU supports a closed set of PDFs:
 *   - Truncated/normalized Gaussian on [a,b]
 *   - Bounded exponential family p(x) ∝ exp(lambda * x) on [a,b]
 *
 * Architecture: 1 CUDA block = 1 NLL evaluation on one dataset.
 * Threads in the block process events via grid-stride loop.
 *
 * Gradient is accumulated via atomic adds (same pattern as HistFactory GPU kernels).
 */

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

/* ---------- Kernels ------------------------------------------------------ */

extern "C" __global__ void unbinned_nll_grad(
    /* Dynamic per-iteration buffers */
    const double* __restrict__ g_params,    /* [n_params] */
    /* Static model + observed buffers */
    const double* __restrict__ g_obs_soa,   /* [n_obs × n_events] */
    const double* __restrict__ g_obs_lo,    /* [n_obs] */
    const double* __restrict__ g_obs_hi,    /* [n_obs] */
    const double* __restrict__ g_evt_wts,   /* [n_events] (optional; ignored when has_evt_wts=0) */
    unsigned int has_evt_wts,
    const struct GpuUnbinnedProcessDesc* __restrict__ g_procs, /* [n_procs] */
    const struct GpuUnbinnedRateModifierDesc* __restrict__ g_rate_mods, /* [total_rate_mods] */
    const unsigned int* __restrict__ g_shape_pidx, /* [total_shape_params] */
    const double* __restrict__ g_pdf_aux_f64, /* [total_pdf_aux_f64] */
    const struct GpuUnbinnedGaussConstraintEntry* __restrict__ g_gauss, /* [n_gauss] */
    /* Output */
    double* __restrict__ g_nll_out,         /* [1] */
    double* __restrict__ g_grad_out,        /* [n_params] */
    /* Scalar metadata */
    unsigned int n_params,
    unsigned int n_obs,
    unsigned int n_events,
    unsigned int n_procs,
    unsigned int total_rate_mods,
    unsigned int total_shape_params,
    unsigned int n_gauss,
    double constraint_const
) {
    (void)total_shape_params; // not needed in kernel; included for API completeness.

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    /* Shared memory layout: params[n_params] | scratch[block_size] */
    extern __shared__ double shared[];
    double* s_params = shared;
    double* s_scratch = shared + n_params;

    /* Load params into shared memory */
    for (unsigned int i = tid; i < n_params; i += block_size) {
        s_params[i] = g_params[i];
    }
    __syncthreads();

    /* Local partial sum of log f(x_i) */
    double local_sum_logf = 0.0;

    for (unsigned int i = tid; i < n_events; i += block_size) {
        double evt_w = has_evt_wts ? g_evt_wts[i] : 1.0;
        if (evt_w == 0.0) {
            continue;
        }
        // Online logsumexp over processes.
        double max_term = -INFINITY;
        double sum_exp = 0.0;

        for (unsigned int p = 0; p < n_procs; p++) {
            struct GpuUnbinnedProcessDesc proc = g_procs[p];
            unsigned int obs_idx = proc.obs_index;
            if (obs_idx >= n_obs) {
                continue;
            }
            double x = g_obs_soa[(size_t)obs_idx * n_events + i];
            double a = g_obs_lo[obs_idx];
            double b = g_obs_hi[obs_idx];

            // Yield value nu_p(params)
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

            // Apply multiplicative yield modifiers.
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
            // Important: allow `nu == 0` in the gradient path so yield-parameter derivatives
            // remain non-zero at the boundary (e.g. mu=0 for scaled yields). The log-likelihood
            // term already excludes `nu == 0` from f(x), but df/dtheta can still be non-zero.
            if (!isfinite(nu) || nu < 0.0) {
                continue;
            }

            // log p(x|theta)
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

        // log f(x_i) = logsumexp(terms)
        double logf = max_term + log(sum_exp);
        if (!isfinite(logf)) {
            // Degenerate (e.g. all nu<=0). Keep finite to avoid NaNs.
            logf = log(DBL_MIN);
        }
        local_sum_logf += evt_w * logf;

        // Gradient contributions from this event.
        for (unsigned int p = 0; p < n_procs; p++) {
            struct GpuUnbinnedProcessDesc proc = g_procs[p];
            unsigned int obs_idx = proc.obs_index;
            if (obs_idx >= n_obs) {
                continue;
            }
            double x = g_obs_soa[(size_t)obs_idx * n_events + i];
            double a = g_obs_lo[obs_idx];
            double b = g_obs_hi[obs_idx];

            // Yield value and derivative dnu/d(alpha) for yield-param kinds.
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

            // Apply multiplicative yield modifiers.
            unsigned int mod_off = proc.rate_mod_offset;
            unsigned int nmods = proc.n_rate_mods;
            if (mod_off + nmods > total_rate_mods) {
                nmods = 0u;
            }
            double mod_factor = 1.0;
            for (unsigned int m = 0; m < nmods; m++) {
                double f, dlogf;
                rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dlogf);
                (void)dlogf;
                mod_factor *= f;
            }
            nu *= mod_factor;
            dnu *= mod_factor;
            if (!(nu > 0.0) || !isfinite(nu)) {
                continue;
            }

            if (proc.pdf_kind == PDF_GAUSSIAN) {
                unsigned int off = proc.shape_param_offset;
                if (proc.n_shape_params != 2u || off + 1u >= total_shape_params) {
                    continue;
                }
                unsigned int mu_idx = g_shape_pidx[off];
                unsigned int sig_idx = g_shape_pidx[off + 1u];
                double mu = s_params[mu_idx];
                double sigma = s_params[sig_idx];

                double logp, dmu, ds;
                gaussian_logp_grad(x, mu, sigma, a, b, &logp, &dmu, &ds);
                if (!isfinite(logp)) {
                    continue;
                }

                // p(x)/f(x) = exp(logp - logf)
                double p_over_f = evt_w * exp(logp - logf);
                if (!(p_over_f > 0.0) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    atomicAdd(&g_grad_out[y_idx], -dnu * p_over_f);
                }
                for (unsigned int m = 0; m < nmods; m++) {
                    struct GpuUnbinnedRateModifierDesc rm = g_rate_mods[mod_off + m];
                    unsigned int aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    double f, dlogf;
                    rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dlogf);
                    (void)f;
                    double dnu_m = nu * dlogf;
                    if (isfinite(dnu_m) && dnu_m != 0.0) {
                        atomicAdd(&g_grad_out[aidx], -dnu_m * p_over_f);
                    }
                }

                // r = nu * p/f
                double r = nu * p_over_f;
                atomicAdd(&g_grad_out[mu_idx], -r * dmu);
                atomicAdd(&g_grad_out[sig_idx], -r * ds);
            }
            else if (proc.pdf_kind == PDF_EXPONENTIAL) {
                unsigned int off = proc.shape_param_offset;
                if (proc.n_shape_params != 1u || off >= total_shape_params) {
                    continue;
                }
                unsigned int lam_idx = g_shape_pidx[off];
                double lambda = s_params[lam_idx];

                double logp, dl;
                exponential_logp_grad(x, lambda, a, b, &logp, &dl);
                if (!isfinite(logp)) {
                    continue;
                }

                double p_over_f = evt_w * exp(logp - logf);
                if (!(p_over_f > 0.0) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    atomicAdd(&g_grad_out[y_idx], -dnu * p_over_f);
                }
                for (unsigned int m = 0; m < nmods; m++) {
                    struct GpuUnbinnedRateModifierDesc rm = g_rate_mods[mod_off + m];
                    unsigned int aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    double f, dlogf;
                    rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dlogf);
                    (void)f;
                    double dnu_m = nu * dlogf;
                    if (isfinite(dnu_m) && dnu_m != 0.0) {
                        atomicAdd(&g_grad_out[aidx], -dnu_m * p_over_f);
                    }
                }

                double r = nu * p_over_f;
                atomicAdd(&g_grad_out[lam_idx], -r * dl);
            }
            else if (proc.pdf_kind == PDF_CRYSTAL_BALL) {
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

                double logp, dmu, ds, da, dn;
                crystal_ball_logp_grad(x, mu, sigma, alpha, nn, a, b, &logp, &dmu, &ds, &da, &dn);
                if (!isfinite(logp)) {
                    continue;
                }
                double p_over_f = evt_w * exp(logp - logf);
                if (!(p_over_f > 0.0) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    atomicAdd(&g_grad_out[y_idx], -dnu * p_over_f);
                }
                for (unsigned int m = 0; m < nmods; m++) {
                    struct GpuUnbinnedRateModifierDesc rm = g_rate_mods[mod_off + m];
                    unsigned int aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    double f, dlogf;
                    rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dlogf);
                    (void)f;
                    double dnu_m = nu * dlogf;
                    if (isfinite(dnu_m) && dnu_m != 0.0) {
                        atomicAdd(&g_grad_out[aidx], -dnu_m * p_over_f);
                    }
                }

                double r = nu * p_over_f;
                atomicAdd(&g_grad_out[mu_idx], -r * dmu);
                atomicAdd(&g_grad_out[sig_idx], -r * ds);
                atomicAdd(&g_grad_out[alpha_idx], -r * da);
                atomicAdd(&g_grad_out[n_idx], -r * dn);
            }
            else if (proc.pdf_kind == PDF_DOUBLE_CRYSTAL_BALL) {
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

                double logp, dmu, ds, da_l, dn_l, da_r, dn_r;
                double_crystal_ball_logp_grad(
                    x, mu, sigma, alpha_l, n_l, alpha_r, n_r, a, b,
                    &logp, &dmu, &ds, &da_l, &dn_l, &da_r, &dn_r
                );
                if (!isfinite(logp)) {
                    continue;
                }
                double p_over_f = evt_w * exp(logp - logf);
                if (!(p_over_f > 0.0) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    atomicAdd(&g_grad_out[y_idx], -dnu * p_over_f);
                }
                for (unsigned int m = 0; m < nmods; m++) {
                    struct GpuUnbinnedRateModifierDesc rm = g_rate_mods[mod_off + m];
                    unsigned int aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    double f, dlogf;
                    rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dlogf);
                    (void)f;
                    double dnu_m = nu * dlogf;
                    if (isfinite(dnu_m) && dnu_m != 0.0) {
                        atomicAdd(&g_grad_out[aidx], -dnu_m * p_over_f);
                    }
                }

                double r = nu * p_over_f;
                atomicAdd(&g_grad_out[mu_idx], -r * dmu);
                atomicAdd(&g_grad_out[sig_idx], -r * ds);
                atomicAdd(&g_grad_out[alpha_l_idx], -r * da_l);
                atomicAdd(&g_grad_out[n_l_idx], -r * dn_l);
                atomicAdd(&g_grad_out[alpha_r_idx], -r * da_r);
                atomicAdd(&g_grad_out[n_r_idx], -r * dn_r);
            }
            else if (proc.pdf_kind == PDF_CHEBYSHEV) {
                unsigned int off = proc.shape_param_offset;
                unsigned int order = proc.n_shape_params;
                if (order == 0u || off + order - 1u >= total_shape_params) {
                    continue;
                }

                double w = b - a;
                if (!isfinite(w) || !(w > 0.0)) {
                    continue;
                }
                double i0 = w;
                for (unsigned int j = 0; j < order; j++) {
                    unsigned int k = j + 1u;
                    if ((k & 1u) == 0u) {
                        unsigned int c_idx = g_shape_pidx[off + j];
                        double c = s_params[c_idx];
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
                double logp = log(f0) - log_i;
                if (!isfinite(logp)) {
                    continue;
                }
                double p_over_f = evt_w * exp(logp - logf);
                if (!(p_over_f > 0.0) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    atomicAdd(&g_grad_out[y_idx], -dnu * p_over_f);
                }
                for (unsigned int m = 0; m < nmods; m++) {
                    struct GpuUnbinnedRateModifierDesc rm = g_rate_mods[mod_off + m];
                    unsigned int aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    double f, dlogf;
                    rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dlogf);
                    (void)f;
                    double dnu_m = nu * dlogf;
                    if (isfinite(dnu_m) && dnu_m != 0.0) {
                        atomicAdd(&g_grad_out[aidx], -dnu_m * p_over_f);
                    }
                }

                double r = nu * p_over_f;
                double inv_f0 = 1.0 / f0;
                double inv_i0 = 1.0 / i0;
                // Second pass: accumulate coefficient gradients.
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
                    atomicAdd(&g_grad_out[c_idx], -r * dlogp_dc);
                }
            }
            else if (proc.pdf_kind == PDF_HISTOGRAM) {
                if (proc.n_shape_params != 0u) {
                    continue;
                }
                double logp = histogram_logp_only(x, g_pdf_aux_f64, proc.pdf_aux_offset, proc.pdf_aux_len);
                if (!isfinite(logp)) {
                    continue;
                }
                double p_over_f = evt_w * exp(logp - logf);
                if (!(p_over_f > 0.0) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    atomicAdd(&g_grad_out[y_idx], -dnu * p_over_f);
                }
                for (unsigned int m = 0; m < nmods; m++) {
                    struct GpuUnbinnedRateModifierDesc rm = g_rate_mods[mod_off + m];
                    unsigned int aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    double f, dlogf;
                    rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dlogf);
                    (void)f;
                    double dnu_m = nu * dlogf;
                    if (isfinite(dnu_m) && dnu_m != 0.0) {
                        atomicAdd(&g_grad_out[aidx], -dnu_m * p_over_f);
                    }
                }
            }
        }
    }

    // Reduce sum_logf across threads (power-of-two block size).
    s_scratch[tid] = local_sum_logf;
    __syncthreads();
    for (unsigned int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] += s_scratch[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        double sum_logf = s_scratch[0];

        // nu_tot and constant yield-gradient (+dnu)
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

            // Apply multiplicative yield modifiers.
            unsigned int mod_off = proc.rate_mod_offset;
            unsigned int nmods = proc.n_rate_mods;
            if (mod_off + nmods > total_rate_mods) {
                nmods = 0u;
            }
            double mod_factor = 1.0;
            for (unsigned int m = 0; m < nmods; m++) {
                double f, dlogf;
                rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dlogf);
                (void)dlogf;
                mod_factor *= f;
            }
            nu *= mod_factor;
            dnu *= mod_factor;
            if (isfinite(nu) && nu >= 0.0) {
                nu_tot += nu;
                if (has_yield_param && isfinite(dnu) && dnu != 0.0) {
                    g_grad_out[y_idx] += dnu;
                }
                for (unsigned int m = 0; m < nmods; m++) {
                    struct GpuUnbinnedRateModifierDesc rm = g_rate_mods[mod_off + m];
                    unsigned int aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    double f, dlogf;
                    rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dlogf);
                    (void)f;
                    double dnu_m = nu * dlogf;
                    if (isfinite(dnu_m) && dnu_m != 0.0) {
                        g_grad_out[aidx] += dnu_m;
                    }
                }
            }
        }

        double nll = nu_tot - sum_logf;

        // Gaussian constraints.
        for (unsigned int k = 0; k < n_gauss; k++) {
            struct GpuUnbinnedGaussConstraintEntry gc = g_gauss[k];
            unsigned int idx = gc.param_idx;
            if (idx >= n_params) {
                continue;
            }
            double x = s_params[idx];
            double diff = x - gc.center;
            double z = diff * gc.inv_width;
            nll += 0.5 * z * z;
            g_grad_out[idx] += z * gc.inv_width;
        }

        nll += constraint_const;
        g_nll_out[0] = nll;
    }
}

extern "C" __global__ void unbinned_nll_only(
    /* Dynamic per-iteration buffers */
    const double* __restrict__ g_params,    /* [n_params] */
    /* Static model + observed buffers */
    const double* __restrict__ g_obs_soa,   /* [n_obs × n_events] */
    const double* __restrict__ g_obs_lo,    /* [n_obs] */
    const double* __restrict__ g_obs_hi,    /* [n_obs] */
    const double* __restrict__ g_evt_wts,   /* [n_events] (optional; ignored when has_evt_wts=0) */
    unsigned int has_evt_wts,
    const struct GpuUnbinnedProcessDesc* __restrict__ g_procs, /* [n_procs] */
    const struct GpuUnbinnedRateModifierDesc* __restrict__ g_rate_mods, /* [total_rate_mods] */
    const unsigned int* __restrict__ g_shape_pidx, /* [total_shape_params] */
    const double* __restrict__ g_pdf_aux_f64, /* [total_pdf_aux_f64] */
    const struct GpuUnbinnedGaussConstraintEntry* __restrict__ g_gauss, /* [n_gauss] */
    /* Output */
    double* __restrict__ g_nll_out,         /* [1] */
    /* Scalar metadata */
    unsigned int n_params,
    unsigned int n_obs,
    unsigned int n_events,
    unsigned int n_procs,
    unsigned int total_rate_mods,
    unsigned int total_shape_params,
    unsigned int n_gauss,
    double constraint_const
) {
    (void)n_params;
    (void)total_shape_params;

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    /* Shared memory layout: params[n_params] | scratch[block_size] */
    extern __shared__ double shared[];
    double* s_params = shared;
    double* s_scratch = shared + n_params;

    /* Load params into shared memory */
    for (unsigned int i = tid; i < n_params; i += block_size) {
        s_params[i] = g_params[i];
    }
    __syncthreads();

    double local_sum_logf = 0.0;

    for (unsigned int i = tid; i < n_events; i += block_size) {
        double evt_w = has_evt_wts ? g_evt_wts[i] : 1.0;
        if (evt_w == 0.0) {
            continue;
        }
        double max_term = -INFINITY;
        double sum_exp = 0.0;

        for (unsigned int p = 0; p < n_procs; p++) {
            struct GpuUnbinnedProcessDesc proc = g_procs[p];
            unsigned int obs_idx = proc.obs_index;
            if (obs_idx >= n_obs) {
                continue;
            }
            double x = g_obs_soa[(size_t)obs_idx * n_events + i];
            double a = g_obs_lo[obs_idx];
            double b = g_obs_hi[obs_idx];

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
            // See note above: allow `nu == 0` so yield derivatives remain defined.
            if (!isfinite(nu) || nu < 0.0) {
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
        local_sum_logf += evt_w * logf;
    }

    // Reduce sum_logf across threads.
    s_scratch[tid] = local_sum_logf;
    __syncthreads();
    for (unsigned int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] += s_scratch[tid + stride];
        }
        __syncthreads();
    }

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
                double f, dlogf;
                rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dlogf);
                (void)dlogf;
                nu *= f;
            }
            if (isfinite(nu) && nu > 0.0) {
                nu_tot += nu;
            }
        }

        double nll = nu_tot - sum_logf;

        for (unsigned int k = 0; k < n_gauss; k++) {
            struct GpuUnbinnedGaussConstraintEntry gc = g_gauss[k];
            double x = s_params[gc.param_idx];
            double diff = x - gc.center;
            double z = diff * gc.inv_width;
            nll += 0.5 * z * z;
        }

        nll += constraint_const;
        g_nll_out[0] = nll;
    }
}

/* ---------- Batch kernels: 1 block = 1 toy dataset ------------------------ */

extern "C" __global__ void unbinned_batch_nll_grad(
    /* Dynamic per-iteration buffers */
    const double* __restrict__ g_params_flat, /* [n_toys × n_params] */
    /* Toy observed data (Phase 1: 1 observable) */
    const double* __restrict__ g_obs_flat,    /* [total_events] */
    const unsigned int* __restrict__ g_toy_offsets, /* [n_toys + 1] */
    /* Observable bounds */
    const double* __restrict__ g_obs_lo,      /* [1] */
    const double* __restrict__ g_obs_hi,      /* [1] */
    /* Static model buffers */
    const struct GpuUnbinnedProcessDesc* __restrict__ g_procs, /* [n_procs] */
    const struct GpuUnbinnedRateModifierDesc* __restrict__ g_rate_mods, /* [total_rate_mods] */
    const unsigned int* __restrict__ g_shape_pidx, /* [total_shape_params] */
    const double* __restrict__ g_pdf_aux_f64, /* [total_pdf_aux_f64] */
    const struct GpuUnbinnedGaussConstraintEntry* __restrict__ g_gauss, /* [n_gauss] */
    /* Output */
    double* __restrict__ g_nll_out,           /* [n_toys] */
    double* __restrict__ g_grad_out,          /* [n_toys × n_params] */
    /* Scalar metadata */
    unsigned int n_params,
    unsigned int n_procs,
    unsigned int total_rate_mods,
    unsigned int total_shape_params,
    unsigned int n_gauss,
    double constraint_const,
    unsigned int n_toys
) {
    unsigned int toy = blockIdx.x;
    if (toy >= n_toys) {
        return;
    }

    unsigned int start = g_toy_offsets[toy];
    unsigned int end = g_toy_offsets[toy + 1u];
    if (end < start) {
        return;
    }
    unsigned int n_events = end - start;

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    const double* __restrict__ g_params = g_params_flat + (size_t)toy * n_params;
    double* __restrict__ grad_out = g_grad_out + (size_t)toy * n_params;

    /* Shared memory layout: params[n_params] | scratch[block_size] */
    extern __shared__ double shared[];
    double* s_params = shared;
    double* s_scratch = shared + n_params;

    for (unsigned int i = tid; i < n_params; i += block_size) {
        s_params[i] = g_params[i];
    }
    __syncthreads();

    double local_sum_logf = 0.0;

    // Phase 1 batch toys: exactly one observable dimension.
    double a = g_obs_lo[0];
    double b = g_obs_hi[0];

    for (unsigned int i = tid; i < n_events; i += block_size) {
        double x = g_obs_flat[(size_t)start + i];

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
            double mod_factor = 1.0;
            for (unsigned int m = 0; m < nmods; m++) {
                double f, dlogf;
                rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dlogf);
                (void)dlogf;
                mod_factor *= f;
            }
            nu *= mod_factor;
            dnu *= mod_factor;
            if (!(nu > 0.0) || !isfinite(nu)) {
                continue;
            }

            if (proc.pdf_kind == PDF_GAUSSIAN) {
                unsigned int off = proc.shape_param_offset;
                if (proc.n_shape_params != 2u || off + 1u >= total_shape_params) {
                    continue;
                }
                unsigned int mu_idx = g_shape_pidx[off];
                unsigned int sig_idx = g_shape_pidx[off + 1u];
                double mu = s_params[mu_idx];
                double sigma = s_params[sig_idx];

                double logp, dmu, ds;
                gaussian_logp_grad(x, mu, sigma, a, b, &logp, &dmu, &ds);
                if (!isfinite(logp)) {
                    continue;
                }
                double p_over_f = exp(logp - logf);
                if (!(p_over_f > 0.0) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    atomicAdd(&grad_out[y_idx], -dnu * p_over_f);
                }
                for (unsigned int m = 0; m < nmods; m++) {
                    struct GpuUnbinnedRateModifierDesc rm = g_rate_mods[mod_off + m];
                    unsigned int aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    double f, dlogf;
                    rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dlogf);
                    (void)f;
                    double dnu_m = nu * dlogf;
                    if (isfinite(dnu_m) && dnu_m != 0.0) {
                        atomicAdd(&grad_out[aidx], -dnu_m * p_over_f);
                    }
                }

                double r = nu * p_over_f;
                atomicAdd(&grad_out[mu_idx], -r * dmu);
                atomicAdd(&grad_out[sig_idx], -r * ds);
            } else if (proc.pdf_kind == PDF_EXPONENTIAL) {
                unsigned int off = proc.shape_param_offset;
                if (proc.n_shape_params != 1u || off >= total_shape_params) {
                    continue;
                }
                unsigned int lam_idx = g_shape_pidx[off];
                double lambda = s_params[lam_idx];

                double logp, dl;
                exponential_logp_grad(x, lambda, a, b, &logp, &dl);
                if (!isfinite(logp)) {
                    continue;
                }
                double p_over_f = exp(logp - logf);
                if (!(p_over_f > 0.0) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    atomicAdd(&grad_out[y_idx], -dnu * p_over_f);
                }
                for (unsigned int m = 0; m < nmods; m++) {
                    struct GpuUnbinnedRateModifierDesc rm = g_rate_mods[mod_off + m];
                    unsigned int aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    double f, dlogf;
                    rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dlogf);
                    (void)f;
                    double dnu_m = nu * dlogf;
                    if (isfinite(dnu_m) && dnu_m != 0.0) {
                        atomicAdd(&grad_out[aidx], -dnu_m * p_over_f);
                    }
                }

                double r = nu * p_over_f;
                atomicAdd(&grad_out[lam_idx], -r * dl);
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

                double logp, dmu, ds, da, dn;
                crystal_ball_logp_grad(x, mu, sigma, alpha, nn, a, b, &logp, &dmu, &ds, &da, &dn);
                if (!isfinite(logp)) {
                    continue;
                }
                double p_over_f = exp(logp - logf);
                if (!(p_over_f > 0.0) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    atomicAdd(&grad_out[y_idx], -dnu * p_over_f);
                }
                for (unsigned int m = 0; m < nmods; m++) {
                    struct GpuUnbinnedRateModifierDesc rm = g_rate_mods[mod_off + m];
                    unsigned int aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    double f, dlogf;
                    rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dlogf);
                    (void)f;
                    double dnu_m = nu * dlogf;
                    if (isfinite(dnu_m) && dnu_m != 0.0) {
                        atomicAdd(&grad_out[aidx], -dnu_m * p_over_f);
                    }
                }

                double r = nu * p_over_f;
                atomicAdd(&grad_out[mu_idx], -r * dmu);
                atomicAdd(&grad_out[sig_idx], -r * ds);
                atomicAdd(&grad_out[alpha_idx], -r * da);
                atomicAdd(&grad_out[n_idx], -r * dn);
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

                double logp, dmu, ds, da_l, dn_l, da_r, dn_r;
                double_crystal_ball_logp_grad(
                    x, mu, sigma, alpha_l, n_l, alpha_r, n_r, a, b,
                    &logp, &dmu, &ds, &da_l, &dn_l, &da_r, &dn_r
                );
                if (!isfinite(logp)) {
                    continue;
                }
                double p_over_f = exp(logp - logf);
                if (!(p_over_f > 0.0) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    atomicAdd(&grad_out[y_idx], -dnu * p_over_f);
                }
                for (unsigned int m = 0; m < nmods; m++) {
                    struct GpuUnbinnedRateModifierDesc rm = g_rate_mods[mod_off + m];
                    unsigned int aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    double f, dlogf;
                    rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dlogf);
                    (void)f;
                    double dnu_m = nu * dlogf;
                    if (isfinite(dnu_m) && dnu_m != 0.0) {
                        atomicAdd(&grad_out[aidx], -dnu_m * p_over_f);
                    }
                }

                double r = nu * p_over_f;
                atomicAdd(&grad_out[mu_idx], -r * dmu);
                atomicAdd(&grad_out[sig_idx], -r * ds);
                atomicAdd(&grad_out[alpha_l_idx], -r * da_l);
                atomicAdd(&grad_out[n_l_idx], -r * dn_l);
                atomicAdd(&grad_out[alpha_r_idx], -r * da_r);
                atomicAdd(&grad_out[n_r_idx], -r * dn_r);
            } else if (proc.pdf_kind == PDF_CHEBYSHEV) {
                unsigned int off = proc.shape_param_offset;
                unsigned int order = proc.n_shape_params;
                if (order == 0u || off + order - 1u >= total_shape_params) {
                    continue;
                }

                double w = b - a;
                if (!isfinite(w) || !(w > 0.0)) {
                    continue;
                }
                double i0 = w;
                for (unsigned int j = 0; j < order; j++) {
                    unsigned int k = j + 1u;
                    if ((k & 1u) == 0u) {
                        unsigned int c_idx = g_shape_pidx[off + j];
                        double c = s_params[c_idx];
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
                double logp = log(f0) - log_i;
                if (!isfinite(logp)) {
                    continue;
                }
                double p_over_f = exp(logp - logf);
                if (!(p_over_f > 0.0) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    atomicAdd(&grad_out[y_idx], -dnu * p_over_f);
                }
                for (unsigned int m = 0; m < nmods; m++) {
                    struct GpuUnbinnedRateModifierDesc rm = g_rate_mods[mod_off + m];
                    unsigned int aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    double f, dlogf;
                    rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dlogf);
                    (void)f;
                    double dnu_m = nu * dlogf;
                    if (isfinite(dnu_m) && dnu_m != 0.0) {
                        atomicAdd(&grad_out[aidx], -dnu_m * p_over_f);
                    }
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
            }
            else if (proc.pdf_kind == PDF_HISTOGRAM) {
                if (proc.n_shape_params != 0u) {
                    continue;
                }
                double logp = histogram_logp_only(x, g_pdf_aux_f64, proc.pdf_aux_offset, proc.pdf_aux_len);
                if (!isfinite(logp)) {
                    continue;
                }
                double p_over_f = exp(logp - logf);
                if (!(p_over_f > 0.0) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    atomicAdd(&grad_out[y_idx], -dnu * p_over_f);
                }
                for (unsigned int m = 0; m < nmods; m++) {
                    struct GpuUnbinnedRateModifierDesc rm = g_rate_mods[mod_off + m];
                    unsigned int aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    double f, dlogf;
                    rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dlogf);
                    (void)f;
                    double dnu_m = nu * dlogf;
                    if (isfinite(dnu_m) && dnu_m != 0.0) {
                        atomicAdd(&grad_out[aidx], -dnu_m * p_over_f);
                    }
                }
            }
        }
    }

    // Reduce sum_logf across threads.
    s_scratch[tid] = local_sum_logf;
    __syncthreads();
    for (unsigned int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] += s_scratch[tid + stride];
        }
        __syncthreads();
    }

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
                double f, dlogf;
                rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dlogf);
                (void)dlogf;
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
                    double f, dlogf;
                    rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dlogf);
                    (void)f;
                    double dnu_m = nu * dlogf;
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
        g_nll_out[toy] = nll;
    }
}

extern "C" __global__ void unbinned_batch_nll_only(
    const double* __restrict__ g_params_flat, /* [n_toys × n_params] */
    const double* __restrict__ g_obs_flat,    /* [total_events] */
    const unsigned int* __restrict__ g_toy_offsets, /* [n_toys + 1] */
    const double* __restrict__ g_obs_lo,      /* [1] */
    const double* __restrict__ g_obs_hi,      /* [1] */
    const struct GpuUnbinnedProcessDesc* __restrict__ g_procs, /* [n_procs] */
    const struct GpuUnbinnedRateModifierDesc* __restrict__ g_rate_mods, /* [total_rate_mods] */
    const unsigned int* __restrict__ g_shape_pidx, /* [total_shape_params] */
    const double* __restrict__ g_pdf_aux_f64, /* [total_pdf_aux_f64] */
    const struct GpuUnbinnedGaussConstraintEntry* __restrict__ g_gauss, /* [n_gauss] */
    double* __restrict__ g_nll_out,           /* [n_toys] */
    unsigned int n_params,
    unsigned int n_procs,
    unsigned int total_rate_mods,
    unsigned int total_shape_params,
    unsigned int n_gauss,
    double constraint_const,
    unsigned int n_toys
) {
    unsigned int toy = blockIdx.x;
    if (toy >= n_toys) {
        return;
    }

    unsigned int start = g_toy_offsets[toy];
    unsigned int end = g_toy_offsets[toy + 1u];
    if (end < start) {
        return;
    }
    unsigned int n_events = end - start;

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    const double* __restrict__ g_params = g_params_flat + (size_t)toy * n_params;

    extern __shared__ double shared[];
    double* s_params = shared;
    double* s_scratch = shared + n_params;

    for (unsigned int i = tid; i < n_params; i += block_size) {
        s_params[i] = g_params[i];
    }
    __syncthreads();

    double local_sum_logf = 0.0;

    double a = g_obs_lo[0];
    double b = g_obs_hi[0];

    for (unsigned int i = tid; i < n_events; i += block_size) {
        double x = g_obs_flat[(size_t)start + i];

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
    }

    s_scratch[tid] = local_sum_logf;
    __syncthreads();
    for (unsigned int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] += s_scratch[tid + stride];
        }
        __syncthreads();
    }

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
                double f, dlogf;
                rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, &f, &dlogf);
                (void)dlogf;
                nu *= f;
            }
            if (isfinite(nu) && nu > 0.0) {
                nu_tot += nu;
            }
        }

        double nll = nu_tot - sum_logf;

        for (unsigned int k = 0; k < n_gauss; k++) {
            struct GpuUnbinnedGaussConstraintEntry gc = g_gauss[k];
            double x0 = s_params[gc.param_idx];
            double diff = x0 - gc.center;
            double z = diff * gc.inv_width;
            nll += 0.5 * z * z;
        }

        nll += constraint_const;
        g_nll_out[toy] = nll;
    }
}

/* ---------- Toy sampling (CUDA) ----------------------------------------- */

__device__ inline unsigned long long splitmix64_next(unsigned long long* state) {
    unsigned long long z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__device__ inline double u01_open(unsigned long long* state) {
    // Uniform in (0,1), 53-bit precision.
    unsigned long long x = splitmix64_next(state);
    unsigned long long mant = x >> 11; // top 53 bits
    return (mant + 0.5) * (1.0 / 9007199254740992.0); // 2^53
}

__device__ inline double norminv_acklam(double p) {
    // Rational approximation for inverse normal CDF (Acklam), with one Halley refinement step.
    // p must be in (0,1).
    const double a1 = -3.969683028665376e+01;
    const double a2 = 2.209460984245205e+02;
    const double a3 = -2.759285104469687e+02;
    const double a4 = 1.383577518672690e+02;
    const double a5 = -3.066479806614716e+01;
    const double a6 = 2.506628277459239e+00;

    const double b1 = -5.447609879822406e+01;
    const double b2 = 1.615858368580409e+02;
    const double b3 = -1.556989798598866e+02;
    const double b4 = 6.680131188771972e+01;
    const double b5 = -1.328068155288572e+01;

    const double c1 = -7.784894002430293e-03;
    const double c2 = -3.223964580411365e-01;
    const double c3 = -2.400758277161838e+00;
    const double c4 = -2.549732539343734e+00;
    const double c5 = 4.374664141464968e+00;
    const double c6 = 2.938163982698783e+00;

    const double d1 = 7.784695709041462e-03;
    const double d2 = 3.224671290700398e-01;
    const double d3 = 2.445134137142996e+00;
    const double d4 = 3.754408661907416e+00;

    const double p_low = 0.02425;
    const double p_high = 1.0 - p_low;

    double x;
    if (p < p_low) {
        double q = sqrt(-2.0 * log(p));
        x = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
        x = -x;
    } else if (p <= p_high) {
        double q = p - 0.5;
        double r = q * q;
        x = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
    } else {
        double q = sqrt(-2.0 * log(1.0 - p));
        x = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }

    // One Halley refinement step.
    double e = stdnorm_cdf(x) - p;
    double u = e / stdnorm_pdf(x);
    x = x - u / (1.0 + 0.5 * x * u);
    return x;
}

__device__ inline double gaussian_sample_trunc(
    double mu,
    double sigma,
    double a,
    double b,
    unsigned long long* state
) {
    if (!isfinite(mu) || !isfinite(sigma) || sigma <= 0.0) {
        return 0.5 * (a + b);
    }
    double z_a = (a - mu) / sigma;
    double z_b = (b - mu) / sigma;
    double u_lo = stdnorm_cdf(z_a);
    double u_hi = stdnorm_cdf(z_b);
    const double eps = 1e-15;
    if (!isfinite(u_lo)) u_lo = eps;
    if (!isfinite(u_hi)) u_hi = 1.0 - eps;
    u_lo = fmax(u_lo, eps);
    u_hi = fmin(u_hi, 1.0 - eps);
    if (!(u_lo < u_hi)) {
        double x0 = mu;
        if (x0 < a) x0 = a;
        if (x0 > b) x0 = b;
        return x0;
    }
    double u = u_lo + (u_hi - u_lo) * u01_open(state);
    double z = norminv_acklam(u);
    double x = mu + sigma * z;
    if (x < a) x = a;
    if (x > b) x = b;
    return x;
}

__device__ inline double exponential_sample_trunc(
    double lambda,
    double a,
    double b,
    unsigned long long* state
) {
    double u = u01_open(state);
    if (!isfinite(lambda) || fabs(lambda) < 1e-12) {
        return a + (b - a) * u;
    }

    double t_a = lambda * a;
    double t_b = lambda * b;
    double hi_t = (t_b >= t_a) ? t_b : t_a;
    double lo_t = (t_b >= t_a) ? t_a : t_b;
    double r = exp(lo_t - hi_t); // in (0,1)
    double one_minus_r = 1.0 - r;
    double yfac = (t_b >= t_a) ? (r + u * one_minus_r) : (1.0 - u * one_minus_r);
    // Guard (should not trigger with u in (0,1)).
    yfac = fmin(fmax(yfac, DBL_MIN), 1.0);

    double x = (hi_t + log(yfac)) / lambda;
    if (x < a) x = a;
    if (x > b) x = b;
    return x;
}

__device__ inline unsigned int poisson_knuth(double lambda, unsigned long long* state) {
    // O(lambda) inversion, used for small lambda only.
    double L = exp(-lambda);
    unsigned int k = 0u;
    double p = 1.0;
    do {
        k += 1u;
        p *= u01_open(state);
    } while (p > L);
    return k - 1u;
}

__device__ inline unsigned int poisson_ptrs(double lambda, unsigned long long* state) {
    // PTRS: Poisson Transformed Rejection with Squeeze (Hörmann, 1993).
    double sl = sqrt(lambda);
    double b = 0.931 + 2.53 * sl;
    double a = -0.059 + 0.02483 * b;
    double inv_alpha = 1.1239 + 1.1328 / (b - 3.4);
    double v_r = 0.9277 - 3.6224 / (b - 2.0);

    while (1) {
        double u = u01_open(state) - 0.5; // (-0.5,0.5)
        double v = u01_open(state);
        double us = 0.5 - fabs(u);
        if (!(us > 0.0)) {
            continue;
        }
        int k = (int)floor((2.0 * a / us + b) * u + lambda + 0.43);

        if (us >= 0.07 && v <= v_r) {
            return (k < 0) ? 0u : (unsigned int)k;
        }
        if (k < 0) {
            continue;
        }
        if (us < 0.013 && v > us) {
            continue;
        }

        double lhs = log(v) + log(inv_alpha) - log(a / (us * us) + b);
        double rhs = -lambda + k * log(lambda) - lgamma((double)k + 1.0);
        if (lhs <= rhs) {
            return (unsigned int)k;
        }
    }
}

__device__ inline unsigned int poisson_sample(double lambda, unsigned long long* state) {
    if (!(lambda > 0.0) || !isfinite(lambda)) {
        return 0u;
    }
    if (lambda < 10.0) {
        return poisson_knuth(lambda, state);
    }
    return poisson_ptrs(lambda, state);
}

__device__ inline double proc_yield(
    const struct GpuUnbinnedProcessDesc proc,
    const struct GpuUnbinnedRateModifierDesc* __restrict__ mods,
    unsigned int total_rate_mods,
    const double* params
) {
    double nu = 0.0;
    if (proc.yield_kind == YIELD_FIXED) {
        nu = proc.base_yield;
    } else if (proc.yield_kind == YIELD_PARAMETER) {
        nu = params[proc.yield_param_idx];
    } else if (proc.yield_kind == YIELD_SCALED) {
        nu = proc.base_yield * params[proc.yield_param_idx];
    } else {
        nu = 0.0;
    }
    unsigned int mod_off = proc.rate_mod_offset;
    unsigned int nmods = proc.n_rate_mods;
    if (mod_off + nmods > total_rate_mods) {
        nmods = 0u;
    }
    for (unsigned int m = 0; m < nmods; m++) {
        double f, dlogf;
        rate_modifier_factor_dlogf(mods, mod_off + m, params, &f, &dlogf);
        (void)dlogf;
        nu *= f;
    }
    if (!isfinite(nu) || nu < 0.0) {
        nu = 0.0;
    }
    return nu;
}

/// Sample total event counts per toy: N ~ Poisson(nu_tot(params)).
extern "C" __global__ void unbinned_toy_counts(
    const double* __restrict__ g_params, /* [n_params] */
    const struct GpuUnbinnedProcessDesc* __restrict__ g_procs, /* [n_procs] */
    const struct GpuUnbinnedRateModifierDesc* __restrict__ g_rate_mods, /* [total_rate_mods] */
    const unsigned int n_procs,
    const unsigned int total_rate_mods,
    const unsigned int n_toys,
    const unsigned long long seed,
    unsigned int* __restrict__ g_counts /* [n_toys] */
) {
    unsigned int toy = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (toy >= n_toys) {
        return;
    }

    double nu_tot = 0.0;
    for (unsigned int p = 0; p < n_procs; p++) {
        nu_tot += proc_yield(g_procs[p], g_rate_mods, total_rate_mods, g_params);
    }

    unsigned long long state = (unsigned long long)(seed + (unsigned long long)toy) ^ 0x6a09e667f3bcc909ULL;
    unsigned int n = poisson_sample(nu_tot, &state);
    g_counts[toy] = n;
}

/// Sample 1D toy observables for many toys into a flattened array.
///
/// Requires `toy_offsets` prefix sums (length `n_toys+1`) computed on the host.
extern "C" __global__ void unbinned_toy_sample_obs_1d(
    const double* __restrict__ g_params, /* [n_params] */
    const double* __restrict__ g_obs_lo, /* [1] */
    const double* __restrict__ g_obs_hi, /* [1] */
    const struct GpuUnbinnedProcessDesc* __restrict__ g_procs, /* [n_procs] */
    const struct GpuUnbinnedRateModifierDesc* __restrict__ g_rate_mods, /* [total_rate_mods] */
    const unsigned int* __restrict__ g_shape_pidx, /* [total_shape_params] */
    const double* __restrict__ g_pdf_aux_f64, /* [total_pdf_aux_f64] */
    const unsigned int n_procs,
    const unsigned int total_rate_mods,
    const unsigned int total_shape_params,
    const unsigned int total_pdf_aux_f64,
    const unsigned int* __restrict__ g_toy_offsets, /* [n_toys+1] */
    const unsigned int n_toys,
    const unsigned long long seed,
    double* __restrict__ g_obs_flat_out /* [toy_offsets[n_toys]] */
) {
    unsigned int toy = (unsigned int)blockIdx.x;
    if (toy >= n_toys) {
        return;
    }

    unsigned int tid = (unsigned int)threadIdx.x;
    unsigned int start = g_toy_offsets[toy];
    unsigned int end = g_toy_offsets[toy + 1u];
    unsigned int n_events = end - start;

    double a = g_obs_lo[0];
    double b = g_obs_hi[0];

    // Compute nu_tot once per toy.
    double nu_tot = 0.0;
    for (unsigned int p = 0; p < n_procs; p++) {
        nu_tot += proc_yield(g_procs[p], g_rate_mods, total_rate_mods, g_params);
    }
    if (!(nu_tot > 0.0) || !isfinite(nu_tot)) {
        // No events expected; nothing to do (offsets should reflect this).
        return;
    }

    for (unsigned int i = tid; i < n_events; i += (unsigned int)blockDim.x) {
        unsigned long long state =
            (unsigned long long)(seed + (unsigned long long)toy) ^
            (0x9e3779b97f4a7c15ULL * ((unsigned long long)i + 1ULL));

        // Choose process by yield weights.
        double u = u01_open(&state) * nu_tot;
        double cum = 0.0;
        unsigned int chosen = 0u;
        for (unsigned int p = 0; p < n_procs; p++) {
            cum += proc_yield(g_procs[p], g_rate_mods, total_rate_mods, g_params);
            if (u <= cum) {
                chosen = p;
                break;
            }
        }

        struct GpuUnbinnedProcessDesc proc = g_procs[chosen];
        double x = 0.5 * (a + b);

        if (proc.pdf_kind == PDF_GAUSSIAN) {
            unsigned int off = proc.shape_param_offset;
            if (proc.n_shape_params == 2u && off + 1u < total_shape_params) {
                unsigned int mu_idx = g_shape_pidx[off];
                unsigned int sig_idx = g_shape_pidx[off + 1u];
                double mu = g_params[mu_idx];
                double sigma = g_params[sig_idx];
                x = gaussian_sample_trunc(mu, sigma, a, b, &state);
            }
        } else if (proc.pdf_kind == PDF_EXPONENTIAL) {
            unsigned int off = proc.shape_param_offset;
            if (proc.n_shape_params == 1u && off < total_shape_params) {
                unsigned int lam_idx = g_shape_pidx[off];
                double lambda = g_params[lam_idx];
                x = exponential_sample_trunc(lambda, a, b, &state);
            }
        } else if (proc.pdf_kind == PDF_CRYSTAL_BALL) {
            unsigned int off = proc.shape_param_offset;
            if (proc.n_shape_params == 4u && off + 3u < total_shape_params) {
                unsigned int mu_idx = g_shape_pidx[off];
                unsigned int sig_idx = g_shape_pidx[off + 1u];
                unsigned int alpha_idx = g_shape_pidx[off + 2u];
                unsigned int n_idx = g_shape_pidx[off + 3u];
                double mu = g_params[mu_idx];
                double sigma = g_params[sig_idx];
                double alpha = g_params[alpha_idx];
                double nn = g_params[n_idx];

                // Rejection sampling on [a,b] using logp at the mode as envelope.
                double x_peak = mu;
                if (x_peak < a) x_peak = a;
                if (x_peak > b) x_peak = b;
                double logp_max = crystal_ball_logp_only(x_peak, mu, sigma, alpha, nn, a, b);
                if (isfinite(logp_max)) {
                    for (unsigned int it = 0; it < 1024u; it++) {
                        double xr = a + (b - a) * u01_open(&state);
                        double logp = crystal_ball_logp_only(xr, mu, sigma, alpha, nn, a, b);
                        if (!isfinite(logp)) {
                            continue;
                        }
                        double lu = log(u01_open(&state));
                        if (lu <= (logp - logp_max)) {
                            x = xr;
                            break;
                        }
                    }
                }
            }
        } else if (proc.pdf_kind == PDF_DOUBLE_CRYSTAL_BALL) {
            unsigned int off = proc.shape_param_offset;
            if (proc.n_shape_params == 6u && off + 5u < total_shape_params) {
                unsigned int mu_idx = g_shape_pidx[off];
                unsigned int sig_idx = g_shape_pidx[off + 1u];
                unsigned int alpha_l_idx = g_shape_pidx[off + 2u];
                unsigned int n_l_idx = g_shape_pidx[off + 3u];
                unsigned int alpha_r_idx = g_shape_pidx[off + 4u];
                unsigned int n_r_idx = g_shape_pidx[off + 5u];
                double mu = g_params[mu_idx];
                double sigma = g_params[sig_idx];
                double alpha_l = g_params[alpha_l_idx];
                double n_l = g_params[n_l_idx];
                double alpha_r = g_params[alpha_r_idx];
                double n_r = g_params[n_r_idx];

                double x_peak = mu;
                if (x_peak < a) x_peak = a;
                if (x_peak > b) x_peak = b;
                double logp_max =
                    double_crystal_ball_logp_only(x_peak, mu, sigma, alpha_l, n_l, alpha_r, n_r, a, b);
                if (isfinite(logp_max)) {
                    for (unsigned int it = 0; it < 1024u; it++) {
                        double xr = a + (b - a) * u01_open(&state);
                        double logp =
                            double_crystal_ball_logp_only(xr, mu, sigma, alpha_l, n_l, alpha_r, n_r, a, b);
                        if (!isfinite(logp)) {
                            continue;
                        }
                        double lu = log(u01_open(&state));
                        if (lu <= (logp - logp_max)) {
                            x = xr;
                            break;
                        }
                    }
                }
            }
        } else if (proc.pdf_kind == PDF_CHEBYSHEV) {
            unsigned int off = proc.shape_param_offset;
            unsigned int order = proc.n_shape_params;
            if (order > 0u && off + order - 1u < total_shape_params) {
                // Rejection sampling on [a,b] with envelope f(x') <= 1 + Σ|c_k|.
                double sum_abs = 1.0;
                for (unsigned int j = 0; j < order; j++) {
                    unsigned int c_idx = g_shape_pidx[off + j];
                    double c = g_params[c_idx];
                    if (isfinite(c)) {
                        sum_abs += fabs(c);
                    }
                }
                if (isfinite(sum_abs) && sum_abs > 0.0) {
                    for (unsigned int it = 0; it < 4096u; it++) {
                        double xr = a + (b - a) * u01_open(&state);
                        double xp = chebyshev_xprime(xr, a, b);

                        // Compute f0(x') = 1 + Σ c_k T_k(x').
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
                            double c = g_params[c_idx];
                            f0 += c * tval;
                        }
                        if (!isfinite(f0) || !(f0 > 0.0)) {
                            continue;
                        }
                        double uacc = u01_open(&state) * sum_abs;
                        if (uacc <= f0) {
                            x = xr;
                            break;
                        }
                    }
                }
            }
        } else if (proc.pdf_kind == PDF_HISTOGRAM) {
            // Inverse-CDF over bins (O(n_bins) per sample). Aux layout:
            // edges[0..n_bins] then log_density[0..n_bins-1].
            unsigned int off = proc.pdf_aux_offset;
            unsigned int len = proc.pdf_aux_len;
            if (proc.n_shape_params == 0u && len >= 3u && off + len <= total_pdf_aux_f64) {
                unsigned int n_bins = (len - 1u) / 2u;
                const double* __restrict__ edges = g_pdf_aux_f64 + (size_t)off;
                const double* __restrict__ logdens = edges + (size_t)(n_bins + 1u);
                double total = 0.0;
                for (unsigned int k = 0; k < n_bins; k++) {
                    double w = edges[k + 1u] - edges[k];
                    double ld = logdens[k];
                    if (!isfinite(w) || !(w > 0.0) || !isfinite(ld)) {
                        continue;
                    }
                    total += exp(ld) * w;
                }
                if (isfinite(total) && total > 0.0) {
                    double u_mass = u01_open(&state) * total;
                    double cum_mass = 0.0;
                    unsigned int chosen_bin = n_bins - 1u;
                    for (unsigned int k = 0; k < n_bins; k++) {
                        double w = edges[k + 1u] - edges[k];
                        double ld = logdens[k];
                        if (!isfinite(w) || !(w > 0.0) || !isfinite(ld)) {
                            continue;
                        }
                        cum_mass += exp(ld) * w;
                        if (u_mass <= cum_mass) {
                            chosen_bin = k;
                            break;
                        }
                    }
                    double lo = edges[chosen_bin];
                    double hi = edges[chosen_bin + 1u];
                    x = lo + (hi - lo) * u01_open(&state);
                }
            }
        }

        unsigned int out_idx = start + i;
        g_obs_flat_out[out_idx] = x;
    }
}
