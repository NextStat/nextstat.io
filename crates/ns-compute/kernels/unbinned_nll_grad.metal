/**
 * Unbinned (event-level) extended likelihood Metal kernels.
 *
 * Port of `unbinned_nll_grad.cu` to Metal Shading Language (MSL).
 * All computation in float (f32) — Apple Silicon has no hardware f64.
 *
 * Architecture: 1 threadgroup = 1 NLL evaluation on one dataset.
 * Threads in the threadgroup process events via grid-stride loop.
 *
 * Gradient is accumulated via atomic adds (device atomic_float), matching the
 * existing HistFactory Metal kernels.
 */

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

/* ---------- Constants (must match ns-compute/unbinned_types.rs) ----------- */
constant uint PDF_GAUSSIAN            = 0;
constant uint PDF_EXPONENTIAL         = 1;
constant uint PDF_CRYSTAL_BALL        = 2;
constant uint PDF_DOUBLE_CRYSTAL_BALL = 3;
constant uint PDF_CHEBYSHEV           = 4;
constant uint PDF_HISTOGRAM           = 5;

constant uint YIELD_FIXED     = 0;
constant uint YIELD_PARAMETER = 1;
constant uint YIELD_SCALED    = 2;

constant uint RATE_NORM_SYS   = 0;
constant uint RATE_WEIGHT_SYS = 1;

constant uint INTERP_CODE0  = 0;
constant uint INTERP_CODE4P = 1;
constant uint BATCH_LOCAL_GRAD_CAP = 24;
constant uint BATCH_PROC_CACHE_CAP = 256;
constant uint BATCH_RATE_MOD_DNU_CAP = 256;
/* Preprocessor guard: ENABLE_FUSED is set to 0 or 1 by the Rust host
   prepending `#define ENABLE_FUSED 1` before MSL compilation.
   When 0 (default), the fused single-pass event loop is physically absent
   from the compiled shader, eliminating register pressure from the fused
   arrays (c_logp, c_grad, c_pidx, c_ng) that cause a catastrophic GPU
   occupancy cliff at >300 threadgroups on small models. */
#ifndef ENABLE_FUSED
#define ENABLE_FUSED 0
#endif

#if ENABLE_FUSED
constant uint FUSED_PROC_CAP = 4;
constant uint FUSED_GRAD_STRIDE = 8;
#endif

/* ---------- Struct mirrors of Rust #[repr(C)] types ---------------------- */

struct MetalUnbinnedProcessDesc {
    float base_yield;
    uint pdf_kind;
    uint yield_kind;
    uint obs_index;
    uint shape_param_offset;
    uint n_shape_params;
    uint yield_param_idx;
    uint rate_mod_offset;
    uint n_rate_mods;
    uint pdf_aux_offset;
    uint pdf_aux_len;
};

struct MetalUnbinnedGaussConstraintEntry {
    float center;
    float inv_width;
    uint param_idx;
    uint _pad;
};

struct MetalUnbinnedRateModifierDesc {
    uint kind;
    uint alpha_param_idx;
    uint interp_code;
    uint _pad;
    float lo;
    float hi;
};

struct ScalarArgs {
    uint n_params;
    uint n_obs;
    uint n_events;
    uint has_evt_wts;
    uint n_procs;
    uint total_rate_mods;
    uint total_shape_params;
    uint n_gauss;
    float constraint_const;
};

struct BatchScalarArgs {
    uint n_params;
    uint n_procs;
    uint total_rate_mods;
    uint total_shape_params;
    uint n_gauss;
    uint n_toys;
    uint local_grad_cols;
    float constraint_const;
    uint toy_offset;  /* chunk offset: toy = tg_pos.x + toy_offset */
};

/* ---------- Helpers: standard normal ------------------------------------ */

inline float stdnorm_logpdf(float z) {
    // -0.5*z^2 - log(sqrt(2*pi))
    return -0.5f * z * z - 0.91893853320467274f;
}

inline float stdnorm_pdf(float z) {
    return exp(stdnorm_logpdf(z));
}

inline float erf_approx(float x) {
    // Approximation from Abramowitz & Stegun 7.1.26.
    // Max error ~1.5e-7 (float), sufficient for our f32 Metal backend.
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;
    const float p = 0.3275911f;

    float sign = (x >= 0.0f) ? 1.0f : -1.0f;
    float ax = fabs(x);
    float t = 1.0f / (1.0f + p * ax);
    float y = 1.0f - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * exp(-ax * ax);
    return sign * y;
}

inline float stdnorm_cdf(float z) {
    // Phi(z) = 0.5 * (1 + erf(z / sqrt(2))).
    //
    // Metal does not provide `erf`/`erfc` on all targets, so we use an approximation.
    const float inv_sqrt2 = 0.7071067811865475f;
    return 0.5f * (1.0f + erf_approx(z * inv_sqrt2));
}

/// Fused CDF + PDF: computes both Phi(z) and phi(z) with a single exp(-z²/2),
/// saving one exp call vs separate stdnorm_cdf + stdnorm_pdf.
inline void stdnorm_cdf_pdf(float z, thread float& out_cdf, thread float& out_pdf) {
    const float inv_sqrt2 = 0.7071067811865475f;
    const float inv_sqrt_2pi = 0.3989422804014327f;
    const float p = 0.3275911f;
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;

    float ax = fabs(z * inv_sqrt2);
    float t = 1.0f / (1.0f + p * ax);
    float exp_neg_z2_half = exp(-0.5f * z * z);
    float y = 1.0f - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * exp_neg_z2_half;
    float sign = (z >= 0.0f) ? 1.0f : -1.0f;
    out_cdf = 0.5f * (1.0f + sign * y);
    out_pdf = exp_neg_z2_half * inv_sqrt_2pi;
}

/* ---------- Helpers: PDFs (logp + dlogp) --------------------------------- */

inline float gaussian_logp_only(float x, float mu, float sigma, float a, float b) {
    if (!isfinite(mu) || !isfinite(sigma) || sigma <= 0.0f) {
        return -INFINITY;
    }
    float inv_sigma = 1.0f / sigma;
    float z_a = (a - mu) * inv_sigma;
    float z_b = (b - mu) * inv_sigma;
    float z_x = (x - mu) * inv_sigma;

    float z = stdnorm_cdf(z_b) - stdnorm_cdf(z_a);
    if (!isfinite(z) || z <= 0.0f) {
        z = FLT_MIN;
    }
    float log_z = log(z);
    return stdnorm_logpdf(z_x) - log(sigma) - log_z;
}

inline void gaussian_logp_grad(
    float x,
    float mu,
    float sigma,
    float a,
    float b,
    thread float& out_logp,
    thread float& out_dmu,
    thread float& out_dsigma
) {
    if (!isfinite(mu) || !isfinite(sigma) || sigma <= 0.0f) {
        out_logp = -INFINITY;
        out_dmu = 0.0f;
        out_dsigma = 0.0f;
        return;
    }

    float inv_sigma = 1.0f / sigma;
    float z_a = (a - mu) * inv_sigma;
    float z_b = (b - mu) * inv_sigma;
    float z_x = (x - mu) * inv_sigma;

    float cdf_a, phi_a;
    float cdf_b, phi_b;
    stdnorm_cdf_pdf(z_a, cdf_a, phi_a);
    stdnorm_cdf_pdf(z_b, cdf_b, phi_b);

    float z = cdf_b - cdf_a;
    if (!isfinite(z) || z <= 0.0f) {
        z = FLT_MIN;
    }
    float log_z = log(z);

    float dlogz_dmu = (phi_a - phi_b) * inv_sigma / z;
    float dlogz_dsigma = (z_a * phi_a - z_b * phi_b) * inv_sigma / z;

    out_logp = stdnorm_logpdf(z_x) - log(sigma) - log_z;
    out_dmu = z_x * inv_sigma - dlogz_dmu;
    out_dsigma = (z_x * z_x - 1.0f) * inv_sigma - dlogz_dsigma;
}

inline void exp_logz_ex(float lambda, float a, float b, thread float& out_logz, thread float& out_ex) {
    // Matches ns-unbinned ExponentialPdf::logz_and_ex (f32).
    if (!isfinite(lambda) || !isfinite(a) || !isfinite(b) || !(a < b)) {
        out_logz = INFINITY;
        out_ex = 0.5f * (a + b);
        return;
    }

    if (fabs(lambda) < 1e-12f) {
        float z = b - a;
        if (!isfinite(z) || z <= 0.0f) {
            out_logz = INFINITY;
            out_ex = 0.5f * (a + b);
            return;
        }
        out_logz = log(z);
        out_ex = 0.5f * (a + b);
        return;
    }

    float t_a = lambda * a;
    float t_b = lambda * b;
    float hi_t = (t_b >= t_a) ? t_b : t_a;
    float lo_t = (t_b >= t_a) ? t_a : t_b;

    float r = exp(lo_t - hi_t);
    // Metal lacks `log1p`; for our r = exp(lo-hi) in (0,1], log(1-r) is OK.
    float log_num = hi_t + log(1.0f - r);
    float log_z = log_num - log(fabs(lambda));

    float denom = 1.0f - r;
    if (!(denom > 0.0f) || !isfinite(denom)) {
        out_logz = log(b - a);
        out_ex = 0.5f * (a + b);
        return;
    }

    float x_hi = (t_b >= t_a) ? b : a;
    float x_lo = (t_b >= t_a) ? a : b;
    float ratio = (x_hi - x_lo * r) / denom;
    float ex = ratio - 1.0f / lambda;

    out_logz = log_z;
    out_ex = ex;
}

inline float exponential_logp_only(float x, float lambda, float a, float b) {
    float logz, ex;
    exp_logz_ex(lambda, a, b, logz, ex);
    return lambda * x - logz;
}

inline void exponential_logp_grad(
    float x,
    float lambda,
    float a,
    float b,
    thread float& out_logp,
    thread float& out_dlambda
) {
    float logz, ex;
    exp_logz_ex(lambda, a, b, logz, ex);
    out_logp = lambda * x - logz;
    out_dlambda = x - ex;
}

/* ---------- Helpers: Histogram PDF (bin_edges + log_density) ------------ */

inline float histogram_logp_only(
    float x,
    const device float* aux_f32,
    uint aux_offset,
    uint aux_len
) {
    // Layout in aux_f32:
    //   edges[0..n_bins] (length n_bins+1)
    //   log_density[0..n_bins-1] (length n_bins)
    // Total length = 2*n_bins + 1.
    if (aux_len < 3u) {
        return -INFINITY;
    }
    uint n_bins = (aux_len - 1u) / 2u;
    if (n_bins == 0u) {
        return -INFINITY;
    }

    const device float* edges = aux_f32 + aux_offset;
    const device float* logdens = edges + (n_bins + 1u);

    float x_min = edges[0];
    float x_max = edges[n_bins];
    if (!isfinite(x) || !isfinite(x_min) || !isfinite(x_max)) {
        return -INFINITY;
    }
    if (x < x_min || x > x_max) {
        return -INFINITY;
    }
    if (x >= x_max) {
        return logdens[n_bins - 1u];
    }

    uint lo = 0u;
    uint hi = n_bins;
    while (lo + 1u < hi) {
        uint mid = (lo + hi) >> 1;
        float e = edges[mid];
        if (e <= x) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return logdens[lo];
}

/* ---------- Helpers: Crystal Ball / Chebyshev PDFs ---------------------- */

inline float gauss_logf(float t) {
    return -0.5f * t * t;
}

inline float gauss_integral(float t1, float t2) {
    // ∫ exp(-0.5 t^2) dt = sqrt(2π) * (Φ(t2) - Φ(t1))
    const float sqrt_2pi = 2.5066282746310005f;
    return sqrt_2pi * (stdnorm_cdf(t2) - stdnorm_cdf(t1));
}

inline bool cb_tail_init(float alpha, float n, thread float& out_log_a, thread float& out_b) {
    if (!(isfinite(alpha) && alpha > 0.0f && isfinite(n) && n > 1.0f)) {
        out_log_a = 0.0f;
        out_b = 0.0f;
        return false;
    }
    float log_a = n * log(n / alpha) - 0.5f * alpha * alpha;
    float b = n / alpha - alpha;
    if (!isfinite(log_a) || !isfinite(b)) {
        out_log_a = 0.0f;
        out_b = 0.0f;
        return false;
    }
    out_log_a = log_a;
    out_b = b;
    return true;
}

inline float cb_logf_left(float t, float log_a, float b, float n) {
    return log_a - n * log(b - t);
}

inline float cb_logf_right(float t, float log_a, float b, float n) {
    return log_a - n * log(b + t);
}

inline float cb_dlogf_dt_left(float t, float b, float n) {
    return n / (b - t);
}

inline float cb_dlogf_dt_right(float t, float b, float n) {
    return -n / (b + t);
}

inline float cb_dlogf_dalpha_left(float t, float alpha, float n, float b) {
    float dln_a = -(n / alpha + alpha);
    float db = -(n / (alpha * alpha) + 1.0f);
    return dln_a - n * db / (b - t);
}

inline float cb_dlogf_dn_left(float t, float alpha, float n, float b) {
    float dln_a = 1.0f + log(n / alpha);
    float db = 1.0f / alpha;
    return dln_a - log(b - t) - n * db / (b - t);
}

inline float cb_dlogf_dalpha_right(float t, float alpha, float n, float b) {
    float dln_a = -(n / alpha + alpha);
    float db = -(n / (alpha * alpha) + 1.0f);
    return dln_a - n * db / (b + t);
}

inline float cb_dlogf_dn_right(float t, float alpha, float n, float b) {
    float dln_a = 1.0f + log(n / alpha);
    float db = 1.0f / alpha;
    return dln_a - log(b + t) - n * db / (b + t);
}

inline bool cb_integral_left_only(float t1, float t2, float n, float log_a, float b, thread float& out_i) {
    float m = n - 1.0f;
    float a0 = exp(log_a);
    float u2 = pow(b - t2, -m);
    float u1 = pow(b - t1, -m);
    float i = a0 / m * (u2 - u1);
    if (!isfinite(i) || !(i > 0.0f)) {
        out_i = 0.0f;
        return false;
    }
    out_i = i;
    return true;
}

inline bool cb_integral_right_only(float t1, float t2, float n, float log_a, float b, thread float& out_i) {
    float m = n - 1.0f;
    float a0 = exp(log_a);
    float u1 = pow(b + t1, -m);
    float u2 = pow(b + t2, -m);
    float i = a0 / m * (u1 - u2);
    if (!isfinite(i) || !(i > 0.0f)) {
        out_i = 0.0f;
        return false;
    }
    out_i = i;
    return true;
}

inline bool cb_integral_and_derivs_left(
    float t1,
    float t2,
    float alpha,
    float n,
    float log_a,
    float b,
    thread float& out_i,
    thread float& out_di_dalpha,
    thread float& out_di_dn
) {
    float m = n - 1.0f;
    float a0 = exp(log_a);

    float b1 = b - t1;
    float b2 = b - t2;
    float u1 = pow(b1, -m);
    float u2 = pow(b2, -m);
    float i = a0 / m * (u2 - u1);
    if (!isfinite(i) || !(i > 0.0f)) {
        out_i = 0.0f;
        out_di_dalpha = 0.0f;
        out_di_dn = 0.0f;
        return false;
    }

    float dln_a_dalpha = -(n / alpha + alpha);
    float dln_a_dn = 1.0f + log(n / alpha);
    float db_dalpha = -(n / (alpha * alpha) + 1.0f);
    float db_dn = 1.0f / alpha;

    float v1 = pow(b1, -n);
    float v2 = pow(b2, -n);

    float di_dalpha = i * dln_a_dalpha - a0 * db_dalpha * (v2 - v1);

    float du1_dn = u1 * (-log(b1) - m * db_dn / b1);
    float du2_dn = u2 * (-log(b2) - m * db_dn / b2);
    float di_dn = i * dln_a_dn - i / m + (a0 / m) * (du2_dn - du1_dn);

    out_i = i;
    out_di_dalpha = di_dalpha;
    out_di_dn = di_dn;
    return true;
}

inline bool cb_integral_and_derivs_right(
    float t1,
    float t2,
    float alpha,
    float n,
    float log_a,
    float b,
    thread float& out_i,
    thread float& out_di_dalpha,
    thread float& out_di_dn
) {
    float m = n - 1.0f;
    float a0 = exp(log_a);

    float b1 = b + t1;
    float b2 = b + t2;
    float u1 = pow(b1, -m);
    float u2 = pow(b2, -m);
    float i = a0 / m * (u1 - u2);
    if (!isfinite(i) || !(i > 0.0f)) {
        out_i = 0.0f;
        out_di_dalpha = 0.0f;
        out_di_dn = 0.0f;
        return false;
    }

    float dln_a_dalpha = -(n / alpha + alpha);
    float dln_a_dn = 1.0f + log(n / alpha);
    float db_dalpha = -(n / (alpha * alpha) + 1.0f);
    float db_dn = 1.0f / alpha;

    float v1 = pow(b1, -n);
    float v2 = pow(b2, -n);

    float di_dalpha = i * dln_a_dalpha - a0 * db_dalpha * (v1 - v2);

    float du1_dn = u1 * (-log(b1) - m * db_dn / b1);
    float du2_dn = u2 * (-log(b2) - m * db_dn / b2);
    float di_dn = i * dln_a_dn - i / m + (a0 / m) * (du1_dn - du2_dn);

    out_i = i;
    out_di_dalpha = di_dalpha;
    out_di_dn = di_dn;
    return true;
}

inline float crystal_ball_logp_only(float x, float mu, float sigma, float alpha, float n, float a, float b) {
    if (!isfinite(mu) || !isfinite(sigma) || sigma <= 0.0f) {
        return -INFINITY;
    }
    float log_a = 0.0f;
    float bt = 0.0f;
    if (!cb_tail_init(alpha, n, log_a, bt)) {
        return -INFINITY;
    }

    float inv_sigma = 1.0f / sigma;
    float t_a = (a - mu) * inv_sigma;
    float t_b = (b - mu) * inv_sigma;
    float t = (x - mu) * inv_sigma;
    float t0 = -alpha;

    float i = 0.0f;
    if (t_b <= t0) {
        if (!cb_integral_left_only(t_a, t_b, n, log_a, bt, i)) {
            return -INFINITY;
        }
    } else if (t_a >= t0) {
        i = gauss_integral(t_a, t_b);
    } else {
        float it = 0.0f;
        if (!cb_integral_left_only(t_a, t0, n, log_a, bt, it)) {
            return -INFINITY;
        }
        i = it + gauss_integral(t0, t_b);
    }
    if (!isfinite(i) || !(i > 0.0f)) {
        return -INFINITY;
    }
    float log_i = log(i);

    float logf = (t > t0) ? gauss_logf(t) : cb_logf_left(t, log_a, bt, n);
    return logf - log(sigma) - log_i;
}

inline void crystal_ball_logp_grad(
    float x,
    float mu,
    float sigma,
    float alpha,
    float n,
    float a,
    float b,
    thread float& out_logp,
    thread float& out_dmu,
    thread float& out_dsigma,
    thread float& out_dalpha,
    thread float& out_dn
) {
    if (!isfinite(mu) || !isfinite(sigma) || sigma <= 0.0f) {
        out_logp = -INFINITY;
        out_dmu = 0.0f;
        out_dsigma = 0.0f;
        out_dalpha = 0.0f;
        out_dn = 0.0f;
        return;
    }
    float log_a = 0.0f;
    float bt = 0.0f;
    if (!cb_tail_init(alpha, n, log_a, bt)) {
        out_logp = -INFINITY;
        out_dmu = 0.0f;
        out_dsigma = 0.0f;
        out_dalpha = 0.0f;
        out_dn = 0.0f;
        return;
    }

    float inv_sigma = 1.0f / sigma;
    float t_a = (a - mu) * inv_sigma;
    float t_b = (b - mu) * inv_sigma;
    float t = (x - mu) * inv_sigma;
    float t0 = -alpha;

    float i = 0.0f;
    float di_dalpha = 0.0f;
    float di_dn = 0.0f;
    if (t_b <= t0) {
        if (!cb_integral_and_derivs_left(t_a, t_b, alpha, n, log_a, bt, i, di_dalpha, di_dn)) {
            out_logp = -INFINITY;
            out_dmu = 0.0f;
            out_dsigma = 0.0f;
            out_dalpha = 0.0f;
            out_dn = 0.0f;
            return;
        }
    } else if (t_a >= t0) {
        i = gauss_integral(t_a, t_b);
        di_dalpha = 0.0f;
        di_dn = 0.0f;
    } else {
        float it = 0.0f;
        float dit_da = 0.0f;
        float dit_dn = 0.0f;
        if (!cb_integral_and_derivs_left(t_a, t0, alpha, n, log_a, bt, it, dit_da, dit_dn)) {
            out_logp = -INFINITY;
            out_dmu = 0.0f;
            out_dsigma = 0.0f;
            out_dalpha = 0.0f;
            out_dn = 0.0f;
            return;
        }
        i = it + gauss_integral(t0, t_b);
        di_dalpha = dit_da;
        di_dn = dit_dn;
    }
    if (!isfinite(i) || !(i > 0.0f)) {
        out_logp = -INFINITY;
        out_dmu = 0.0f;
        out_dsigma = 0.0f;
        out_dalpha = 0.0f;
        out_dn = 0.0f;
        return;
    }

    float log_i = log(i);
    float dlogi_dalpha = di_dalpha / i;
    float dlogi_dn = di_dn / i;

    float logf_a = (t_a > t0) ? gauss_logf(t_a) : cb_logf_left(t_a, log_a, bt, n);
    float logf_b = (t_b > t0) ? gauss_logf(t_b) : cb_logf_left(t_b, log_a, bt, n);
    float f_a = exp(logf_a);
    float f_b = exp(logf_b);

    float dlogi_dmu = (f_a - f_b) * inv_sigma / i;
    float dlogi_dsigma = (f_a * t_a - f_b * t_b) * inv_sigma / i;

    bool is_gauss = (t > t0);
    float logf = 0.0f;
    float dlogf_dt = 0.0f;
    float dlogf_dalpha = 0.0f;
    float dlogf_dn = 0.0f;
    if (is_gauss) {
        logf = gauss_logf(t);
        dlogf_dt = -t;
        dlogf_dalpha = 0.0f;
        dlogf_dn = 0.0f;
    } else {
        logf = cb_logf_left(t, log_a, bt, n);
        dlogf_dt = cb_dlogf_dt_left(t, bt, n);
        dlogf_dalpha = cb_dlogf_dalpha_left(t, alpha, n, bt);
        dlogf_dn = cb_dlogf_dn_left(t, alpha, n, bt);
    }

    out_logp = logf - log(sigma) - log_i;

    out_dmu = -inv_sigma * dlogf_dt - dlogi_dmu;
    out_dsigma = -t * inv_sigma * dlogf_dt - inv_sigma - dlogi_dsigma;
    out_dalpha = dlogf_dalpha - dlogi_dalpha;
    out_dn = dlogf_dn - dlogi_dn;
}

inline float double_crystal_ball_logp_only(
    float x,
    float mu,
    float sigma,
    float alpha_l,
    float n_l,
    float alpha_r,
    float n_r,
    float a,
    float b
) {
    if (!isfinite(mu) || !isfinite(sigma) || sigma <= 0.0f) {
        return -INFINITY;
    }
    float log_a_l = 0.0f, b_l = 0.0f;
    float log_a_r = 0.0f, b_r = 0.0f;
    if (!cb_tail_init(alpha_l, n_l, log_a_l, b_l)) {
        return -INFINITY;
    }
    if (!cb_tail_init(alpha_r, n_r, log_a_r, b_r)) {
        return -INFINITY;
    }

    float inv_sigma = 1.0f / sigma;
    float t_a = (a - mu) * inv_sigma;
    float t_b = (b - mu) * inv_sigma;
    float t = (x - mu) * inv_sigma;

    float t_l = -alpha_l;
    float t_r = alpha_r;

    float i = 0.0f;
    if (t_a < t_l) {
        float t2 = (t_b < t_l) ? t_b : t_l;
        float it = 0.0f;
        if (!cb_integral_left_only(t_a, t2, n_l, log_a_l, b_l, it)) {
            return -INFINITY;
        }
        i += it;
    }
    float core_lo = (t_a > t_l) ? t_a : t_l;
    float core_hi = (t_b < t_r) ? t_b : t_r;
    if (core_hi > core_lo) {
        i += gauss_integral(core_lo, core_hi);
    }
    if (t_b > t_r) {
        float t1 = (t_a > t_r) ? t_a : t_r;
        float it = 0.0f;
        if (!cb_integral_right_only(t1, t_b, n_r, log_a_r, b_r, it)) {
            return -INFINITY;
        }
        i += it;
    }
    if (!isfinite(i) || !(i > 0.0f)) {
        return -INFINITY;
    }
    float log_i = log(i);

    float logf = 0.0f;
    if (t < t_l) {
        logf = cb_logf_left(t, log_a_l, b_l, n_l);
    } else if (t > t_r) {
        logf = cb_logf_right(t, log_a_r, b_r, n_r);
    } else {
        logf = gauss_logf(t);
    }
    return logf - log(sigma) - log_i;
}

inline void double_crystal_ball_logp_grad(
    float x,
    float mu,
    float sigma,
    float alpha_l,
    float n_l,
    float alpha_r,
    float n_r,
    float a,
    float b,
    thread float& out_logp,
    thread float& out_dmu,
    thread float& out_dsigma,
    thread float& out_dalpha_l,
    thread float& out_dn_l,
    thread float& out_dalpha_r,
    thread float& out_dn_r
) {
    if (!isfinite(mu) || !isfinite(sigma) || sigma <= 0.0f) {
        out_logp = -INFINITY;
        out_dmu = 0.0f;
        out_dsigma = 0.0f;
        out_dalpha_l = 0.0f;
        out_dn_l = 0.0f;
        out_dalpha_r = 0.0f;
        out_dn_r = 0.0f;
        return;
    }
    float log_a_l = 0.0f, b_l = 0.0f;
    float log_a_r = 0.0f, b_r = 0.0f;
    if (!cb_tail_init(alpha_l, n_l, log_a_l, b_l) || !cb_tail_init(alpha_r, n_r, log_a_r, b_r)) {
        out_logp = -INFINITY;
        out_dmu = 0.0f;
        out_dsigma = 0.0f;
        out_dalpha_l = 0.0f;
        out_dn_l = 0.0f;
        out_dalpha_r = 0.0f;
        out_dn_r = 0.0f;
        return;
    }

    float inv_sigma = 1.0f / sigma;
    float t_a = (a - mu) * inv_sigma;
    float t_b = (b - mu) * inv_sigma;
    float t = (x - mu) * inv_sigma;

    float t_l = -alpha_l;
    float t_r = alpha_r;

    float i = 0.0f;
    float di_dalpha_l = 0.0f;
    float di_dn_l = 0.0f;
    float di_dalpha_r = 0.0f;
    float di_dn_r = 0.0f;

    if (t_a < t_l) {
        float t2 = (t_b < t_l) ? t_b : t_l;
        float it = 0.0f, dit_da = 0.0f, dit_dn = 0.0f;
        if (!cb_integral_and_derivs_left(t_a, t2, alpha_l, n_l, log_a_l, b_l, it, dit_da, dit_dn)) {
            out_logp = -INFINITY;
            out_dmu = 0.0f;
            out_dsigma = 0.0f;
            out_dalpha_l = 0.0f;
            out_dn_l = 0.0f;
            out_dalpha_r = 0.0f;
            out_dn_r = 0.0f;
            return;
        }
        i += it;
        di_dalpha_l += dit_da;
        di_dn_l += dit_dn;
    }

    float core_lo = (t_a > t_l) ? t_a : t_l;
    float core_hi = (t_b < t_r) ? t_b : t_r;
    if (core_hi > core_lo) {
        i += gauss_integral(core_lo, core_hi);
    }

    if (t_b > t_r) {
        float t1 = (t_a > t_r) ? t_a : t_r;
        float it = 0.0f, dit_da = 0.0f, dit_dn = 0.0f;
        if (!cb_integral_and_derivs_right(t1, t_b, alpha_r, n_r, log_a_r, b_r, it, dit_da, dit_dn)) {
            out_logp = -INFINITY;
            out_dmu = 0.0f;
            out_dsigma = 0.0f;
            out_dalpha_l = 0.0f;
            out_dn_l = 0.0f;
            out_dalpha_r = 0.0f;
            out_dn_r = 0.0f;
            return;
        }
        i += it;
        di_dalpha_r += dit_da;
        di_dn_r += dit_dn;
    }

    if (!isfinite(i) || !(i > 0.0f)) {
        out_logp = -INFINITY;
        out_dmu = 0.0f;
        out_dsigma = 0.0f;
        out_dalpha_l = 0.0f;
        out_dn_l = 0.0f;
        out_dalpha_r = 0.0f;
        out_dn_r = 0.0f;
        return;
    }

    float log_i = log(i);
    float dlogi_dalpha_l = di_dalpha_l / i;
    float dlogi_dn_l = di_dn_l / i;
    float dlogi_dalpha_r = di_dalpha_r / i;
    float dlogi_dn_r = di_dn_r / i;

    float logf_a = 0.0f;
    if (t_a < t_l) {
        logf_a = cb_logf_left(t_a, log_a_l, b_l, n_l);
    } else if (t_a > t_r) {
        logf_a = cb_logf_right(t_a, log_a_r, b_r, n_r);
    } else {
        logf_a = gauss_logf(t_a);
    }
    float logf_b = 0.0f;
    if (t_b < t_l) {
        logf_b = cb_logf_left(t_b, log_a_l, b_l, n_l);
    } else if (t_b > t_r) {
        logf_b = cb_logf_right(t_b, log_a_r, b_r, n_r);
    } else {
        logf_b = gauss_logf(t_b);
    }
    float f_a = exp(logf_a);
    float f_b = exp(logf_b);

    float dlogi_dmu = (f_a - f_b) * inv_sigma / i;
    float dlogi_dsigma = (f_a * t_a - f_b * t_b) * inv_sigma / i;

    float logf = 0.0f;
    float dlogf_dt = 0.0f;
    float dlogf_dalpha_l = 0.0f;
    float dlogf_dn_l = 0.0f;
    float dlogf_dalpha_r = 0.0f;
    float dlogf_dn_r = 0.0f;

    if (t < t_l) {
        logf = cb_logf_left(t, log_a_l, b_l, n_l);
        dlogf_dt = cb_dlogf_dt_left(t, b_l, n_l);
        dlogf_dalpha_l = cb_dlogf_dalpha_left(t, alpha_l, n_l, b_l);
        dlogf_dn_l = cb_dlogf_dn_left(t, alpha_l, n_l, b_l);
        dlogf_dalpha_r = 0.0f;
        dlogf_dn_r = 0.0f;
    } else if (t > t_r) {
        logf = cb_logf_right(t, log_a_r, b_r, n_r);
        dlogf_dt = cb_dlogf_dt_right(t, b_r, n_r);
        dlogf_dalpha_l = 0.0f;
        dlogf_dn_l = 0.0f;
        dlogf_dalpha_r = cb_dlogf_dalpha_right(t, alpha_r, n_r, b_r);
        dlogf_dn_r = cb_dlogf_dn_right(t, alpha_r, n_r, b_r);
    } else {
        logf = gauss_logf(t);
        dlogf_dt = -t;
        dlogf_dalpha_l = 0.0f;
        dlogf_dn_l = 0.0f;
        dlogf_dalpha_r = 0.0f;
        dlogf_dn_r = 0.0f;
    }

    out_logp = logf - log(sigma) - log_i;

    out_dmu = -inv_sigma * dlogf_dt - dlogi_dmu;
    out_dsigma = -t * inv_sigma * dlogf_dt - inv_sigma - dlogi_dsigma;
    out_dalpha_l = dlogf_dalpha_l - dlogi_dalpha_l;
    out_dn_l = dlogf_dn_l - dlogi_dn_l;
    out_dalpha_r = dlogf_dalpha_r - dlogi_dalpha_r;
    out_dn_r = dlogf_dn_r - dlogi_dn_r;
}

inline float chebyshev_xprime(float x, float a, float b) {
    float w = b - a;
    float xp = (2.0f * x - (a + b)) / w;
    xp = clamp(xp, -1.0f, 1.0f);
    return xp;
}

/* ---------- Helpers: Rate modifiers (yield systematics) ------------------ */

inline void histosys_interp(
    float alpha,
    float down,
    float nominal,
    float up,
    uint code,
    thread float& out_val,
    thread float& out_der
) {
    // Matches ns-unbinned/src/interp.rs for Code0 and Code4p.
    if (code == INTERP_CODE0) {
        if (alpha >= 0.0f) {
            float der = up - nominal;
            out_val = nominal + der * alpha;
            out_der = der;
        } else {
            float der = nominal - down;
            out_val = nominal + der * alpha;
            out_der = der;
        }
        return;
    }

    // Code4p: smooth polynomial in [-1,1], linear extrapolation outside.
    float delta_up = up - nominal;
    float delta_dn = nominal - down;

    if (alpha > 1.0f) {
        out_val = nominal + delta_up * alpha;
        out_der = delta_up;
        return;
    }
    if (alpha < -1.0f) {
        out_val = nominal + delta_dn * alpha;
        out_der = delta_dn;
        return;
    }

    float s = 0.5f * (delta_up + delta_dn);
    float a = 0.0625f * (delta_up - delta_dn);

    float a2 = alpha * alpha;
    float a3 = a2 * alpha;
    float a4 = a2 * a2;
    float a5 = a4 * alpha;
    float a6 = a3 * a3;

    float tmp3 = (3.0f * a6) - (10.0f * a4) + (15.0f * a2);
    float dtmp3 = (18.0f * a5) - (40.0f * a3) + (30.0f * alpha);

    float delta = alpha * s + tmp3 * a;
    float ddelta = s + dtmp3 * a;

    out_val = nominal + delta;
    out_der = ddelta;
}

inline void rate_modifier_factor_dlogf(
    const device MetalUnbinnedRateModifierDesc* mods,
    uint midx,
    const threadgroup float* params,
    thread float& out_f,
    thread float& out_dlogf
) {
    MetalUnbinnedRateModifierDesc m = mods[midx];
    uint aidx = m.alpha_param_idx;
    float alpha = params[aidx];

    if (m.kind == RATE_NORM_SYS) {
        float lo = m.lo;
        float hi = m.hi;
        if (!(isfinite(lo) && lo > 0.0f && isfinite(hi) && hi > 0.0f && isfinite(alpha))) {
            out_f = 1.0f;
            out_dlogf = 0.0f;
            return;
        }
        float log_hi = log(hi);
        float log_lo = log(lo);
        if (alpha >= 0.0f) {
            out_f = exp(alpha * log_hi);
            out_dlogf = log_hi;
        } else {
            out_f = exp(-alpha * log_lo);
            out_dlogf = -log_lo;
        }
        return;
    }

    if (m.kind == RATE_WEIGHT_SYS) {
        float lo = m.lo;
        float hi = m.hi;
        if (!(isfinite(lo) && lo > 0.0f && isfinite(hi) && hi > 0.0f && isfinite(alpha))) {
            out_f = 1.0f;
            out_dlogf = 0.0f;
            return;
        }
        float val = 1.0f;
        float der = 0.0f;
        histosys_interp(alpha, lo, 1.0f, hi, m.interp_code, val, der);
        if (!isfinite(val) || val <= 0.0f) {
            val = FLT_MIN;
            der = 0.0f;
        }
        out_f = val;
        out_dlogf = (val > 0.0f) ? (der / val) : 0.0f;
        return;
    }

    out_f = 1.0f;
    out_dlogf = 0.0f;
}

inline float rate_modifier_factor_only_tg(
    const device MetalUnbinnedRateModifierDesc* mods,
    uint midx,
    const threadgroup float* params
) {
    float f = 1.0f;
    float dlogf = 0.0f;
    rate_modifier_factor_dlogf(mods, midx, params, f, dlogf);
    (void)dlogf;
    return f;
}

/* ---------- Kernels ------------------------------------------------------ */

kernel void unbinned_nll_grad(
    const device float* g_params                    [[buffer(0)]],  /* [n_params] */
    const device float* g_obs_soa                   [[buffer(1)]],  /* [n_obs × n_events] */
    const device float* g_obs_lo                    [[buffer(2)]],  /* [n_obs] */
    const device float* g_obs_hi                    [[buffer(3)]],  /* [n_obs] */
    const device float* g_evt_wts                   [[buffer(4)]],  /* [n_events] (optional; ignored when args.has_evt_wts=0) */
    const device MetalUnbinnedProcessDesc* g_procs  [[buffer(5)]],  /* [n_procs] */
    const device MetalUnbinnedRateModifierDesc* g_rate_mods [[buffer(6)]], /* [total_rate_mods] */
    const device uint* g_shape_pidx                 [[buffer(7)]],  /* [total_shape_params] */
    const device float* g_pdf_aux_f32               [[buffer(8)]],  /* [total_pdf_aux_f32] */
    const device MetalUnbinnedGaussConstraintEntry* g_gauss [[buffer(9)]], /* [n_gauss] */
    device float* g_nll_out                         [[buffer(10)]], /* [1] */
    device atomic_float* g_grad_out                 [[buffer(11)]], /* [n_params] */
    constant ScalarArgs& args                       [[buffer(12)]],
    uint tid                                        [[thread_position_in_threadgroup]],
    uint block_size                                 [[threads_per_threadgroup]],
    threadgroup float* shared                       [[threadgroup(0)]]
) {
    uint n_params = args.n_params;
    uint n_obs = args.n_obs;
    uint n_events = args.n_events;
    uint n_procs = args.n_procs;
    uint total_rate_mods = args.total_rate_mods;
    uint total_shape_params = args.total_shape_params;
    uint n_gauss = args.n_gauss;
    float constraint_const = args.constraint_const;

    threadgroup float* s_params = shared;
    threadgroup float* s_scratch = shared + n_params;

    // Load params into threadgroup memory.
    for (uint i = tid; i < n_params; i += block_size) {
        s_params[i] = g_params[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum_logf = 0.0f;

    for (uint i = tid; i < n_events; i += block_size) {
        float evt_w = (args.has_evt_wts != 0u) ? g_evt_wts[i] : 1.0f;
        if (evt_w == 0.0f) {
            continue;
        }
        float max_term = -INFINITY;
        float sum_exp = 0.0f;

        for (uint p = 0; p < n_procs; p++) {
            MetalUnbinnedProcessDesc proc = g_procs[p];
            uint obs_idx = proc.obs_index;
            if (obs_idx >= n_obs) {
                continue;
            }
            float x = g_obs_soa[obs_idx * n_events + i];
            float a = g_obs_lo[obs_idx];
            float b = g_obs_hi[obs_idx];

            float nu = 0.0f;
            if (proc.yield_kind == YIELD_FIXED) {
                nu = proc.base_yield;
            } else if (proc.yield_kind == YIELD_PARAMETER) {
                nu = s_params[proc.yield_param_idx];
            } else if (proc.yield_kind == YIELD_SCALED) {
                nu = proc.base_yield * s_params[proc.yield_param_idx];
            } else {
                continue;
            }
            uint mod_off = proc.rate_mod_offset;
            uint nmods = proc.n_rate_mods;
            if (mod_off + nmods > total_rate_mods) {
                nmods = 0u;
            }
            for (uint m = 0; m < nmods; m++) {
                nu *= rate_modifier_factor_only_tg(g_rate_mods, mod_off + m, s_params);
            }
            // Important: allow `nu == 0` in the gradient path so yield-parameter derivatives
            // remain non-zero at the boundary (e.g. mu=0 for scaled yields). The log-likelihood
            // term already excludes `nu == 0` from f(x), but df/dtheta can still be non-zero.
            if (!isfinite(nu) || nu < 0.0f) {
                continue;
            }

            float logp = -INFINITY;
            if (proc.pdf_kind == PDF_GAUSSIAN) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 2u || off + 1u >= total_shape_params) {
                    continue;
                }
                uint mu_idx = g_shape_pidx[off];
                uint sig_idx = g_shape_pidx[off + 1u];
                float mu = s_params[mu_idx];
                float sigma = s_params[sig_idx];
                logp = gaussian_logp_only(x, mu, sigma, a, b);
            } else if (proc.pdf_kind == PDF_EXPONENTIAL) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 1u || off >= total_shape_params) {
                    continue;
                }
                uint lam_idx = g_shape_pidx[off];
                float lambda = s_params[lam_idx];
                logp = exponential_logp_only(x, lambda, a, b);
            } else if (proc.pdf_kind == PDF_CRYSTAL_BALL) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 4u || off + 3u >= total_shape_params) {
                    continue;
                }
                uint mu_idx = g_shape_pidx[off];
                uint sig_idx = g_shape_pidx[off + 1u];
                uint alpha_idx = g_shape_pidx[off + 2u];
                uint n_idx = g_shape_pidx[off + 3u];
                float mu = s_params[mu_idx];
                float sigma = s_params[sig_idx];
                float alpha = s_params[alpha_idx];
                float nn = s_params[n_idx];
                logp = crystal_ball_logp_only(x, mu, sigma, alpha, nn, a, b);
            } else if (proc.pdf_kind == PDF_DOUBLE_CRYSTAL_BALL) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 6u || off + 5u >= total_shape_params) {
                    continue;
                }
                uint mu_idx = g_shape_pidx[off];
                uint sig_idx = g_shape_pidx[off + 1u];
                uint alpha_l_idx = g_shape_pidx[off + 2u];
                uint n_l_idx = g_shape_pidx[off + 3u];
                uint alpha_r_idx = g_shape_pidx[off + 4u];
                uint n_r_idx = g_shape_pidx[off + 5u];
                float mu = s_params[mu_idx];
                float sigma = s_params[sig_idx];
                float alpha_l = s_params[alpha_l_idx];
                float n_l = s_params[n_l_idx];
                float alpha_r = s_params[alpha_r_idx];
                float n_r = s_params[n_r_idx];
                logp = double_crystal_ball_logp_only(x, mu, sigma, alpha_l, n_l, alpha_r, n_r, a, b);
            } else if (proc.pdf_kind == PDF_CHEBYSHEV) {
                uint off = proc.shape_param_offset;
                uint order = proc.n_shape_params;
                if (order == 0u || off + order - 1u >= total_shape_params) {
                    continue;
                }
                float w = b - a;
                if (!isfinite(w) || !(w > 0.0f)) {
                    continue;
                }
                float i0 = w;
                for (uint j = 0; j < order; j++) {
                    uint k = j + 1u;
                    if ((k & 1u) == 0u) {
                        uint c_idx = g_shape_pidx[off + j];
                        float c = s_params[c_idx];
                        float denom = 1.0f - (float)k * (float)k;
                        i0 += w * c / denom;
                    }
                }
                if (!isfinite(i0) || !(i0 > 0.0f)) {
                    continue;
                }
                float log_i = log(i0);
                float xp = chebyshev_xprime(x, a, b);
                float f0 = 1.0f;
                float tkm1 = 1.0f;
                float tk = xp;
                for (uint j = 0; j < order; j++) {
                    uint k = j + 1u;
                    float tval = 0.0f;
                    if (k == 1u) {
                        tval = tk;
                    } else {
                        float tkp1 = 2.0f * xp * tk - tkm1;
                        tval = tkp1;
                        tkm1 = tk;
                        tk = tkp1;
                    }
                    uint c_idx = g_shape_pidx[off + j];
                    float c = s_params[c_idx];
                    f0 += c * tval;
                }
                if (!isfinite(f0) || !(f0 > 0.0f)) {
                    continue;
                }
                logp = log(f0) - log_i;
            } else if (proc.pdf_kind == PDF_HISTOGRAM) {
                if (proc.n_shape_params != 0u) {
                    continue;
                }
                logp = histogram_logp_only(x, g_pdf_aux_f32, proc.pdf_aux_offset, proc.pdf_aux_len);
            } else {
                continue;
            }

            float term = log(nu) + logp;
            if (!isfinite(term)) {
                continue;
            }

            if (term > max_term) {
                sum_exp = sum_exp * exp(max_term - term) + 1.0f;
                max_term = term;
            } else {
                sum_exp += exp(term - max_term);
            }
        }

        float logf = max_term + log(sum_exp);
        if (!isfinite(logf)) {
            logf = log(FLT_MIN);
        }
        local_sum_logf += evt_w * logf;

        // Gradient contributions from this event.
        for (uint p = 0; p < n_procs; p++) {
            MetalUnbinnedProcessDesc proc = g_procs[p];
            uint obs_idx = proc.obs_index;
            if (obs_idx >= n_obs) {
                continue;
            }
            float x = g_obs_soa[obs_idx * n_events + i];
            float a = g_obs_lo[obs_idx];
            float b = g_obs_hi[obs_idx];

            float nu = 0.0f;
            float dnu = 0.0f;
            uint y_idx = proc.yield_param_idx;
            bool has_yield_param = false;
            if (proc.yield_kind == YIELD_FIXED) {
                nu = proc.base_yield;
            } else if (proc.yield_kind == YIELD_PARAMETER) {
                nu = s_params[y_idx];
                dnu = 1.0f;
                has_yield_param = true;
            } else if (proc.yield_kind == YIELD_SCALED) {
                nu = proc.base_yield * s_params[y_idx];
                dnu = proc.base_yield;
                has_yield_param = true;
            } else {
                continue;
            }
            uint mod_off = proc.rate_mod_offset;
            uint nmods = proc.n_rate_mods;
            if (mod_off + nmods > total_rate_mods) {
                nmods = 0u;
            }
            float mod_factor = 1.0f;
            for (uint m = 0; m < nmods; m++) {
                mod_factor *= rate_modifier_factor_only_tg(g_rate_mods, mod_off + m, s_params);
            }
            nu *= mod_factor;
            dnu *= mod_factor;
            if (!(nu > 0.0f) || !isfinite(nu)) {
                continue;
            }

            if (proc.pdf_kind == PDF_GAUSSIAN) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 2u || off + 1u >= total_shape_params) {
                    continue;
                }
                uint mu_idx = g_shape_pidx[off];
                uint sig_idx = g_shape_pidx[off + 1u];
                float mu = s_params[mu_idx];
                float sigma = s_params[sig_idx];

                float logp, dmu, ds;
                gaussian_logp_grad(x, mu, sigma, a, b, logp, dmu, ds);
                if (!isfinite(logp)) {
                    continue;
                }

                float p_over_f = evt_w * exp(logp - logf);
                if (!(p_over_f > 0.0f) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    atomic_fetch_add_explicit(&g_grad_out[y_idx], -dnu * p_over_f, memory_order_relaxed);
                }
                for (uint m = 0; m < nmods; m++) {
                    MetalUnbinnedRateModifierDesc rm = g_rate_mods[mod_off + m];
                    uint aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    float f, dlogf;
                    rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, f, dlogf);
                    (void)f;
                    float dnu_m = nu * dlogf;
                    if (isfinite(dnu_m) && dnu_m != 0.0f) {
                        atomic_fetch_add_explicit(&g_grad_out[aidx], -dnu_m * p_over_f, memory_order_relaxed);
                    }
                }

                float r = nu * p_over_f;
                atomic_fetch_add_explicit(&g_grad_out[mu_idx], -r * dmu, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_grad_out[sig_idx], -r * ds, memory_order_relaxed);
            } else if (proc.pdf_kind == PDF_EXPONENTIAL) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 1u || off >= total_shape_params) {
                    continue;
                }
                uint lam_idx = g_shape_pidx[off];
                float lambda = s_params[lam_idx];

                float logp, dl;
                exponential_logp_grad(x, lambda, a, b, logp, dl);
                if (!isfinite(logp)) {
                    continue;
                }

                float p_over_f = evt_w * exp(logp - logf);
                if (!(p_over_f > 0.0f) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    atomic_fetch_add_explicit(&g_grad_out[y_idx], -dnu * p_over_f, memory_order_relaxed);
                }
                for (uint m = 0; m < nmods; m++) {
                    MetalUnbinnedRateModifierDesc rm = g_rate_mods[mod_off + m];
                    uint aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    float f, dlogf;
                    rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, f, dlogf);
                    (void)f;
                    float dnu_m = nu * dlogf;
                    if (isfinite(dnu_m) && dnu_m != 0.0f) {
                        atomic_fetch_add_explicit(&g_grad_out[aidx], -dnu_m * p_over_f, memory_order_relaxed);
                    }
                }

                float r = nu * p_over_f;
                atomic_fetch_add_explicit(&g_grad_out[lam_idx], -r * dl, memory_order_relaxed);
            } else if (proc.pdf_kind == PDF_CRYSTAL_BALL) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 4u || off + 3u >= total_shape_params) {
                    continue;
                }
                uint mu_idx = g_shape_pidx[off];
                uint sig_idx = g_shape_pidx[off + 1u];
                uint alpha_idx = g_shape_pidx[off + 2u];
                uint n_idx = g_shape_pidx[off + 3u];
                float mu = s_params[mu_idx];
                float sigma = s_params[sig_idx];
                float alpha = s_params[alpha_idx];
                float nn = s_params[n_idx];

                float logp, dmu, ds, da, dn;
                crystal_ball_logp_grad(x, mu, sigma, alpha, nn, a, b, logp, dmu, ds, da, dn);
                if (!isfinite(logp)) {
                    continue;
                }
                float p_over_f = evt_w * exp(logp - logf);
                if (!(p_over_f > 0.0f) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    atomic_fetch_add_explicit(&g_grad_out[y_idx], -dnu * p_over_f, memory_order_relaxed);
                }
                for (uint m = 0; m < nmods; m++) {
                    MetalUnbinnedRateModifierDesc rm = g_rate_mods[mod_off + m];
                    uint aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    float f, dlogf;
                    rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, f, dlogf);
                    (void)f;
                    float dnu_m = nu * dlogf;
                    if (isfinite(dnu_m) && dnu_m != 0.0f) {
                        atomic_fetch_add_explicit(&g_grad_out[aidx], -dnu_m * p_over_f, memory_order_relaxed);
                    }
                }

                float r = nu * p_over_f;
                atomic_fetch_add_explicit(&g_grad_out[mu_idx], -r * dmu, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_grad_out[sig_idx], -r * ds, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_grad_out[alpha_idx], -r * da, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_grad_out[n_idx], -r * dn, memory_order_relaxed);
            } else if (proc.pdf_kind == PDF_DOUBLE_CRYSTAL_BALL) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 6u || off + 5u >= total_shape_params) {
                    continue;
                }
                uint mu_idx = g_shape_pidx[off];
                uint sig_idx = g_shape_pidx[off + 1u];
                uint alpha_l_idx = g_shape_pidx[off + 2u];
                uint n_l_idx = g_shape_pidx[off + 3u];
                uint alpha_r_idx = g_shape_pidx[off + 4u];
                uint n_r_idx = g_shape_pidx[off + 5u];
                float mu = s_params[mu_idx];
                float sigma = s_params[sig_idx];
                float alpha_l = s_params[alpha_l_idx];
                float n_l = s_params[n_l_idx];
                float alpha_r = s_params[alpha_r_idx];
                float n_r = s_params[n_r_idx];

                float logp, dmu, ds, da_l, dn_l, da_r, dn_r;
                double_crystal_ball_logp_grad(
                    x, mu, sigma, alpha_l, n_l, alpha_r, n_r, a, b,
                    logp, dmu, ds, da_l, dn_l, da_r, dn_r
                );
                if (!isfinite(logp)) {
                    continue;
                }
                float p_over_f = evt_w * exp(logp - logf);
                if (!(p_over_f > 0.0f) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    atomic_fetch_add_explicit(&g_grad_out[y_idx], -dnu * p_over_f, memory_order_relaxed);
                }
                for (uint m = 0; m < nmods; m++) {
                    MetalUnbinnedRateModifierDesc rm = g_rate_mods[mod_off + m];
                    uint aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    float f, dlogf;
                    rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, f, dlogf);
                    (void)f;
                    float dnu_m = nu * dlogf;
                    if (isfinite(dnu_m) && dnu_m != 0.0f) {
                        atomic_fetch_add_explicit(&g_grad_out[aidx], -dnu_m * p_over_f, memory_order_relaxed);
                    }
                }

                float r = nu * p_over_f;
                atomic_fetch_add_explicit(&g_grad_out[mu_idx], -r * dmu, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_grad_out[sig_idx], -r * ds, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_grad_out[alpha_l_idx], -r * da_l, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_grad_out[n_l_idx], -r * dn_l, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_grad_out[alpha_r_idx], -r * da_r, memory_order_relaxed);
                atomic_fetch_add_explicit(&g_grad_out[n_r_idx], -r * dn_r, memory_order_relaxed);
            } else if (proc.pdf_kind == PDF_CHEBYSHEV) {
                uint off = proc.shape_param_offset;
                uint order = proc.n_shape_params;
                if (order == 0u || off + order - 1u >= total_shape_params) {
                    continue;
                }
                float w = b - a;
                if (!isfinite(w) || !(w > 0.0f)) {
                    continue;
                }
                float i0 = w;
                for (uint j = 0; j < order; j++) {
                    uint k = j + 1u;
                    if ((k & 1u) == 0u) {
                        uint c_idx = g_shape_pidx[off + j];
                        float c = s_params[c_idx];
                        float denom = 1.0f - (float)k * (float)k;
                        i0 += w * c / denom;
                    }
                }
                if (!isfinite(i0) || !(i0 > 0.0f)) {
                    continue;
                }
                float log_i = log(i0);

                float xp = chebyshev_xprime(x, a, b);
                float f0 = 1.0f;
                float tkm1 = 1.0f;
                float tk = xp;
                for (uint j = 0; j < order; j++) {
                    uint k = j + 1u;
                    float tval = 0.0f;
                    if (k == 1u) {
                        tval = tk;
                    } else {
                        float tkp1 = 2.0f * xp * tk - tkm1;
                        tval = tkp1;
                        tkm1 = tk;
                        tk = tkp1;
                    }
                    uint c_idx = g_shape_pidx[off + j];
                    float c = s_params[c_idx];
                    f0 += c * tval;
                }
                if (!isfinite(f0) || !(f0 > 0.0f)) {
                    continue;
                }
                float logp = log(f0) - log_i;
                if (!isfinite(logp)) {
                    continue;
                }
                float p_over_f = evt_w * exp(logp - logf);
                if (!(p_over_f > 0.0f) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    atomic_fetch_add_explicit(&g_grad_out[y_idx], -dnu * p_over_f, memory_order_relaxed);
                }
                for (uint m = 0; m < nmods; m++) {
                    MetalUnbinnedRateModifierDesc rm = g_rate_mods[mod_off + m];
                    uint aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    float f, dlogf;
                    rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, f, dlogf);
                    (void)f;
                    float dnu_m = nu * dlogf;
                    if (isfinite(dnu_m) && dnu_m != 0.0f) {
                        atomic_fetch_add_explicit(&g_grad_out[aidx], -dnu_m * p_over_f, memory_order_relaxed);
                    }
                }

                float r = nu * p_over_f;
                float inv_f0 = 1.0f / f0;
                float inv_i0 = 1.0f / i0;
                tkm1 = 1.0f;
                tk = xp;
                for (uint j = 0; j < order; j++) {
                    uint k = j + 1u;
                    float tval = 0.0f;
                    if (k == 1u) {
                        tval = tk;
                    } else {
                        float tkp1 = 2.0f * xp * tk - tkm1;
                        tval = tkp1;
                        tkm1 = tk;
                        tk = tkp1;
                    }
                    float dlogi = 0.0f;
                    if ((k & 1u) == 0u) {
                        float denom = 1.0f - (float)k * (float)k;
                        float di_dc = w / denom;
                        dlogi = di_dc * inv_i0;
                    }
                    float dlogp_dc = tval * inv_f0 - dlogi;
                    uint c_idx = g_shape_pidx[off + j];
                    atomic_fetch_add_explicit(&g_grad_out[c_idx], -r * dlogp_dc, memory_order_relaxed);
                }
            }
            else if (proc.pdf_kind == PDF_HISTOGRAM) {
                if (proc.n_shape_params != 0u) {
                    continue;
                }
                float logp = histogram_logp_only(x, g_pdf_aux_f32, proc.pdf_aux_offset, proc.pdf_aux_len);
                if (!isfinite(logp)) {
                    continue;
                }
                float p_over_f = evt_w * exp(logp - logf);
                if (!(p_over_f > 0.0f) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    atomic_fetch_add_explicit(&g_grad_out[y_idx], -dnu * p_over_f, memory_order_relaxed);
                }
                for (uint m = 0; m < nmods; m++) {
                    MetalUnbinnedRateModifierDesc rm = g_rate_mods[mod_off + m];
                    uint aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    float f, dlogf;
                    rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, f, dlogf);
                    (void)f;
                    float dnu_m = nu * dlogf;
                    if (isfinite(dnu_m) && dnu_m != 0.0f) {
                        atomic_fetch_add_explicit(&g_grad_out[aidx], -dnu_m * p_over_f, memory_order_relaxed);
                    }
                }
            }
        }
    }

    // Reduce sum_logf across threads (power-of-two block_size).
    s_scratch[tid] = local_sum_logf;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] += s_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float sum_logf = s_scratch[0];

        float nu_tot = 0.0f;
        for (uint p = 0; p < n_procs; p++) {
            MetalUnbinnedProcessDesc proc = g_procs[p];
            float nu = 0.0f;
            float dnu = 0.0f;
            uint y_idx = proc.yield_param_idx;
            bool has_yield_param = false;
            if (proc.yield_kind == YIELD_FIXED) {
                nu = proc.base_yield;
            } else if (proc.yield_kind == YIELD_PARAMETER) {
                nu = s_params[y_idx];
                dnu = 1.0f;
                has_yield_param = true;
            } else if (proc.yield_kind == YIELD_SCALED) {
                nu = proc.base_yield * s_params[y_idx];
                dnu = proc.base_yield;
                has_yield_param = true;
            } else {
                continue;
            }
            uint mod_off = proc.rate_mod_offset;
            uint nmods = proc.n_rate_mods;
            if (mod_off + nmods > total_rate_mods) {
                nmods = 0u;
            }
            float mod_factor = 1.0f;
            for (uint m = 0; m < nmods; m++) {
                mod_factor *= rate_modifier_factor_only_tg(g_rate_mods, mod_off + m, s_params);
            }
            nu *= mod_factor;
            dnu *= mod_factor;
            if (isfinite(nu) && nu >= 0.0f) {
                nu_tot += nu;
                if (has_yield_param && isfinite(dnu) && dnu != 0.0f) {
                    atomic_fetch_add_explicit(&g_grad_out[y_idx], dnu, memory_order_relaxed);
                }
                for (uint m = 0; m < nmods; m++) {
                    MetalUnbinnedRateModifierDesc rm = g_rate_mods[mod_off + m];
                    uint aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    float f, dlogf;
                    rate_modifier_factor_dlogf(g_rate_mods, mod_off + m, s_params, f, dlogf);
                    (void)f;
                    float dnu_m = nu * dlogf;
                    if (isfinite(dnu_m) && dnu_m != 0.0f) {
                        atomic_fetch_add_explicit(&g_grad_out[aidx], dnu_m, memory_order_relaxed);
                    }
                }
            }
        }

        float nll = nu_tot - sum_logf;

        for (uint k = 0; k < n_gauss; k++) {
            MetalUnbinnedGaussConstraintEntry gc = g_gauss[k];
            uint idx = gc.param_idx;
            if (idx >= n_params) {
                continue;
            }
            float x = s_params[idx];
            float diff = x - gc.center;
            float z = diff * gc.inv_width;
            nll += 0.5f * z * z;
            atomic_fetch_add_explicit(&g_grad_out[idx], z * gc.inv_width, memory_order_relaxed);
        }

        nll += constraint_const;
        g_nll_out[0] = nll;
    }
}

kernel void unbinned_nll_only(
    const device float* g_params                    [[buffer(0)]],  /* [n_params] */
    const device float* g_obs_soa                   [[buffer(1)]],  /* [n_obs × n_events] */
    const device float* g_obs_lo                    [[buffer(2)]],  /* [n_obs] */
    const device float* g_obs_hi                    [[buffer(3)]],  /* [n_obs] */
    const device float* g_evt_wts                   [[buffer(4)]],  /* [n_events] (optional; ignored when args.has_evt_wts=0) */
    const device MetalUnbinnedProcessDesc* g_procs  [[buffer(5)]],  /* [n_procs] */
    const device MetalUnbinnedRateModifierDesc* g_rate_mods [[buffer(6)]], /* [total_rate_mods] */
    const device uint* g_shape_pidx                 [[buffer(7)]],  /* [total_shape_params] */
    const device float* g_pdf_aux_f32               [[buffer(8)]],  /* [total_pdf_aux_f32] */
    const device MetalUnbinnedGaussConstraintEntry* g_gauss [[buffer(9)]], /* [n_gauss] */
    device float* g_nll_out                         [[buffer(10)]], /* [1] */
    constant ScalarArgs& args                       [[buffer(11)]],
    uint tid                                        [[thread_position_in_threadgroup]],
    uint block_size                                 [[threads_per_threadgroup]],
    threadgroup float* shared                       [[threadgroup(0)]]
) {
    uint n_params = args.n_params;
    uint n_obs = args.n_obs;
    uint n_events = args.n_events;
    uint n_procs = args.n_procs;
    uint total_rate_mods = args.total_rate_mods;
    uint total_shape_params = args.total_shape_params;
    uint n_gauss = args.n_gauss;
    float constraint_const = args.constraint_const;

    threadgroup float* s_params = shared;
    threadgroup float* s_scratch = shared + n_params;

    for (uint i = tid; i < n_params; i += block_size) {
        s_params[i] = g_params[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum_logf = 0.0f;

    for (uint i = tid; i < n_events; i += block_size) {
        float evt_w = (args.has_evt_wts != 0u) ? g_evt_wts[i] : 1.0f;
        if (evt_w == 0.0f) {
            continue;
        }
        float max_term = -INFINITY;
        float sum_exp = 0.0f;

        for (uint p = 0; p < n_procs; p++) {
            MetalUnbinnedProcessDesc proc = g_procs[p];
            uint obs_idx = proc.obs_index;
            if (obs_idx >= n_obs) {
                continue;
            }
            float x = g_obs_soa[obs_idx * n_events + i];
            float a = g_obs_lo[obs_idx];
            float b = g_obs_hi[obs_idx];

            float nu = 0.0f;
            if (proc.yield_kind == YIELD_FIXED) {
                nu = proc.base_yield;
            } else if (proc.yield_kind == YIELD_PARAMETER) {
                nu = s_params[proc.yield_param_idx];
            } else if (proc.yield_kind == YIELD_SCALED) {
                nu = proc.base_yield * s_params[proc.yield_param_idx];
            } else {
                continue;
            }
            uint mod_off = proc.rate_mod_offset;
            uint nmods = proc.n_rate_mods;
            if (mod_off + nmods > total_rate_mods) {
                nmods = 0u;
            }
            for (uint m = 0; m < nmods; m++) {
                nu *= rate_modifier_factor_only_tg(g_rate_mods, mod_off + m, s_params);
            }
            // See note above: allow `nu == 0` so yield derivatives remain defined.
            if (!isfinite(nu) || nu < 0.0f) {
                continue;
            }

            float logp = -INFINITY;
            if (proc.pdf_kind == PDF_GAUSSIAN) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 2u || off + 1u >= total_shape_params) {
                    continue;
                }
                uint mu_idx = g_shape_pidx[off];
                uint sig_idx = g_shape_pidx[off + 1u];
                float mu = s_params[mu_idx];
                float sigma = s_params[sig_idx];
                logp = gaussian_logp_only(x, mu, sigma, a, b);
            } else if (proc.pdf_kind == PDF_EXPONENTIAL) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 1u || off >= total_shape_params) {
                    continue;
                }
                uint lam_idx = g_shape_pidx[off];
                float lambda = s_params[lam_idx];
                logp = exponential_logp_only(x, lambda, a, b);
            } else if (proc.pdf_kind == PDF_CRYSTAL_BALL) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 4u || off + 3u >= total_shape_params) {
                    continue;
                }
                uint mu_idx = g_shape_pidx[off];
                uint sig_idx = g_shape_pidx[off + 1u];
                uint alpha_idx = g_shape_pidx[off + 2u];
                uint n_idx = g_shape_pidx[off + 3u];
                float mu = s_params[mu_idx];
                float sigma = s_params[sig_idx];
                float alpha = s_params[alpha_idx];
                float nn = s_params[n_idx];
                logp = crystal_ball_logp_only(x, mu, sigma, alpha, nn, a, b);
            } else if (proc.pdf_kind == PDF_DOUBLE_CRYSTAL_BALL) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 6u || off + 5u >= total_shape_params) {
                    continue;
                }
                uint mu_idx = g_shape_pidx[off];
                uint sig_idx = g_shape_pidx[off + 1u];
                uint alpha_l_idx = g_shape_pidx[off + 2u];
                uint n_l_idx = g_shape_pidx[off + 3u];
                uint alpha_r_idx = g_shape_pidx[off + 4u];
                uint n_r_idx = g_shape_pidx[off + 5u];
                float mu = s_params[mu_idx];
                float sigma = s_params[sig_idx];
                float alpha_l = s_params[alpha_l_idx];
                float n_l = s_params[n_l_idx];
                float alpha_r = s_params[alpha_r_idx];
                float n_r = s_params[n_r_idx];
                logp = double_crystal_ball_logp_only(x, mu, sigma, alpha_l, n_l, alpha_r, n_r, a, b);
            } else if (proc.pdf_kind == PDF_CHEBYSHEV) {
                uint off = proc.shape_param_offset;
                uint order = proc.n_shape_params;
                if (order == 0u || off + order - 1u >= total_shape_params) {
                    continue;
                }
                float w = b - a;
                if (!isfinite(w) || !(w > 0.0f)) {
                    continue;
                }
                float i0 = w;
                for (uint j = 0; j < order; j++) {
                    uint k = j + 1u;
                    if ((k & 1u) == 0u) {
                        uint c_idx = g_shape_pidx[off + j];
                        float c = s_params[c_idx];
                        float denom = 1.0f - (float)k * (float)k;
                        i0 += w * c / denom;
                    }
                }
                if (!isfinite(i0) || !(i0 > 0.0f)) {
                    continue;
                }
                float log_i = log(i0);
                float xp = chebyshev_xprime(x, a, b);
                float f0 = 1.0f;
                float tkm1 = 1.0f;
                float tk = xp;
                for (uint j = 0; j < order; j++) {
                    uint k = j + 1u;
                    float tval = 0.0f;
                    if (k == 1u) {
                        tval = tk;
                    } else {
                        float tkp1 = 2.0f * xp * tk - tkm1;
                        tval = tkp1;
                        tkm1 = tk;
                        tk = tkp1;
                    }
                    uint c_idx = g_shape_pidx[off + j];
                    float c = s_params[c_idx];
                    f0 += c * tval;
                }
                if (!isfinite(f0) || !(f0 > 0.0f)) {
                    continue;
                }
                logp = log(f0) - log_i;
            } else if (proc.pdf_kind == PDF_HISTOGRAM) {
                if (proc.n_shape_params != 0u) {
                    continue;
                }
                logp = histogram_logp_only(x, g_pdf_aux_f32, proc.pdf_aux_offset, proc.pdf_aux_len);
            } else {
                continue;
            }

            float term = log(nu) + logp;
            if (!isfinite(term)) {
                continue;
            }

            if (term > max_term) {
                sum_exp = sum_exp * exp(max_term - term) + 1.0f;
                max_term = term;
            } else {
                sum_exp += exp(term - max_term);
            }
        }

        float logf = max_term + log(sum_exp);
        if (!isfinite(logf)) {
            logf = log(FLT_MIN);
        }
        local_sum_logf += evt_w * logf;
    }

    s_scratch[tid] = local_sum_logf;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] += s_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float sum_logf = s_scratch[0];

        float nu_tot = 0.0f;
        for (uint p = 0; p < n_procs; p++) {
            MetalUnbinnedProcessDesc proc = g_procs[p];
            float nu = 0.0f;
            if (proc.yield_kind == YIELD_FIXED) {
                nu = proc.base_yield;
            } else if (proc.yield_kind == YIELD_PARAMETER) {
                nu = s_params[proc.yield_param_idx];
            } else if (proc.yield_kind == YIELD_SCALED) {
                nu = proc.base_yield * s_params[proc.yield_param_idx];
            } else {
                continue;
            }
            uint mod_off = proc.rate_mod_offset;
            uint nmods = proc.n_rate_mods;
            if (mod_off + nmods > total_rate_mods) {
                nmods = 0u;
            }
            for (uint m = 0; m < nmods; m++) {
                nu *= rate_modifier_factor_only_tg(g_rate_mods, mod_off + m, s_params);
            }
            if (isfinite(nu) && nu > 0.0f) {
                nu_tot += nu;
            }
        }

        float nll = nu_tot - sum_logf;
        for (uint k = 0; k < n_gauss; k++) {
            MetalUnbinnedGaussConstraintEntry gc = g_gauss[k];
            float x = s_params[gc.param_idx];
            float diff = x - gc.center;
            float z = diff * gc.inv_width;
            nll += 0.5f * z * z;
        }

        nll += constraint_const;
        g_nll_out[0] = nll;
    }
}

/* ---------- Batch kernels: 1 threadgroup = 1 toy dataset ------------------ */

inline void batch_grad_add(
    threadgroup float* local_grad,
    uint local_grad_cols,
    device atomic_float* grad_out,
    uint idx,
    uint n_params,
    float value
) {
    if (idx >= n_params || !isfinite(value) || value == 0.0f) {
        return;
    }
    if (idx < local_grad_cols) {
        local_grad[idx] += value;
    } else {
        atomic_fetch_add_explicit(&grad_out[idx], value, memory_order_relaxed);
    }
}

inline float batch_rate_mod_dnu(
    const device MetalUnbinnedRateModifierDesc* mods,
    const threadgroup float* params,
    const threadgroup float* cached_dnu,
    bool use_cache,
    uint midx,
    float nu
) {
    if (use_cache) {
        return cached_dnu[midx];
    }
    float f, dlogf;
    rate_modifier_factor_dlogf(mods, midx, params, f, dlogf);
    (void)f;
    float dnu_m = nu * dlogf;
    if (!isfinite(dnu_m)) {
        return 0.0f;
    }
    return dnu_m;
}

kernel void unbinned_batch_nll_grad(
    const device float* g_params_flat              [[buffer(0)]],  /* [n_toys × n_params] */
    const device float* g_obs_flat                 [[buffer(1)]],  /* [total_events] */
    const device uint* g_toy_offsets               [[buffer(2)]],  /* [n_toys + 1] */
    const device float* g_obs_lo                   [[buffer(3)]],  /* [1] */
    const device float* g_obs_hi                   [[buffer(4)]],  /* [1] */
    const device MetalUnbinnedProcessDesc* g_procs [[buffer(5)]],  /* [n_procs] */
    const device MetalUnbinnedRateModifierDesc* g_rate_mods [[buffer(6)]], /* [total_rate_mods] */
    const device uint* g_shape_pidx                [[buffer(7)]],  /* [total_shape_params] */
    const device float* g_pdf_aux_f32              [[buffer(8)]],  /* [total_pdf_aux_f32] */
    const device MetalUnbinnedGaussConstraintEntry* g_gauss [[buffer(9)]], /* [n_gauss] */
    device float* g_nll_out                        [[buffer(10)]], /* [n_toys] */
    device atomic_float* g_grad_out                [[buffer(11)]], /* [n_toys × n_params] */
    constant BatchScalarArgs& args                 [[buffer(12)]],
    uint3 tid3                                     [[thread_position_in_threadgroup]],
    uint3 tptg                                     [[threads_per_threadgroup]],
    uint3 tg_pos                                   [[threadgroup_position_in_grid]],
    threadgroup float* shared                      [[threadgroup(0)]]
) {
    uint tid = tid3.x;
    uint block_size = tptg.x;
    uint toy = tg_pos.x + args.toy_offset;
    if (toy >= args.n_toys) {
        return;
    }

    uint n_params = args.n_params;
    uint n_procs = args.n_procs;
    uint total_rate_mods = args.total_rate_mods;
    uint total_shape_params = args.total_shape_params;
    uint n_gauss = args.n_gauss;
    uint local_grad_cols = min(min(args.local_grad_cols, n_params), BATCH_LOCAL_GRAD_CAP);
    float constraint_const = args.constraint_const;

    uint start = g_toy_offsets[toy];
    uint end = g_toy_offsets[toy + 1];
    if (end < start) {
        return;
    }
    uint n_events = end - start;

    threadgroup float* s_params = shared;
    threadgroup float* s_scratch = shared + n_params;
    threadgroup float* s_grad_partial = s_scratch + block_size;
    threadgroup float* s_proc_nu = s_grad_partial + block_size * local_grad_cols;
    threadgroup float* s_proc_log_nu = s_proc_nu + BATCH_PROC_CACHE_CAP;
    threadgroup float* s_proc_dnu = s_proc_log_nu + BATCH_PROC_CACHE_CAP;
    threadgroup float* s_rate_mod_dnu = s_proc_dnu + BATCH_PROC_CACHE_CAP;
    threadgroup uint* s_proc_mod_off =
        reinterpret_cast<threadgroup uint*>(s_rate_mod_dnu + BATCH_RATE_MOD_DNU_CAP);
    threadgroup uint* s_proc_nmods = s_proc_mod_off + BATCH_PROC_CACHE_CAP;
    bool use_proc_cache = (n_procs <= BATCH_PROC_CACHE_CAP);
    bool use_rate_mod_dnu_cache = use_proc_cache && (total_rate_mods <= BATCH_RATE_MOD_DNU_CAP);

    const device float* params = g_params_flat + toy * n_params;
    device atomic_float* grad_out = g_grad_out + toy * n_params;
    threadgroup float* local_grad = s_grad_partial + tid * local_grad_cols;

    for (uint i = tid; i < n_params; i += block_size) {
        s_params[i] = params[i];
    }
    for (uint i = 0; i < local_grad_cols; i++) {
        local_grad[i] = 0.0f;
    }
    if (use_proc_cache) {
        for (uint p = tid; p < n_procs; p += block_size) {
            MetalUnbinnedProcessDesc proc = g_procs[p];
            float nu = 0.0f;
            float dnu = 0.0f;
            uint mod_off = proc.rate_mod_offset;
            uint nmods = proc.n_rate_mods;
            if (mod_off + nmods > total_rate_mods) {
                nmods = 0u;
            }
            s_proc_mod_off[p] = mod_off;
            s_proc_nmods[p] = nmods;
            if (proc.obs_index == 0u) {
                if (proc.yield_kind == YIELD_FIXED) {
                    nu = proc.base_yield;
                } else if (proc.yield_kind == YIELD_PARAMETER) {
                    nu = s_params[proc.yield_param_idx];
                    dnu = 1.0f;
                } else if (proc.yield_kind == YIELD_SCALED) {
                    nu = proc.base_yield * s_params[proc.yield_param_idx];
                    dnu = proc.base_yield;
                }
                float mod_factor = 1.0f;
                for (uint m = 0; m < nmods; m++) {
                    mod_factor *= rate_modifier_factor_only_tg(g_rate_mods, mod_off + m, s_params);
                }
                nu *= mod_factor;
                dnu *= mod_factor;
            }
            if (isfinite(nu) && nu > 0.0f) {
                s_proc_nu[p] = nu;
                s_proc_log_nu[p] = log(nu);
                s_proc_dnu[p] = dnu;
            } else {
                s_proc_nu[p] = 0.0f;
                s_proc_log_nu[p] = -INFINITY;
                s_proc_dnu[p] = 0.0f;
            }
        }
    }
    if (use_rate_mod_dnu_cache) {
        for (uint midx = tid; midx < total_rate_mods; midx += block_size) {
            s_rate_mod_dnu[midx] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p = tid; p < n_procs; p += block_size) {
            float nu = s_proc_nu[p];
            if (!(nu > 0.0f) || !isfinite(nu)) {
                continue;
            }
            uint mod_off = s_proc_mod_off[p];
            uint nmods = s_proc_nmods[p];
            for (uint m = 0; m < nmods; m++) {
                uint midx = mod_off + m;
                float f, dlogf;
                rate_modifier_factor_dlogf(g_rate_mods, midx, s_params, f, dlogf);
                (void)f;
                float dnu_m = nu * dlogf;
                if (!isfinite(dnu_m)) {
                    dnu_m = 0.0f;
                }
                s_rate_mod_dnu[midx] = dnu_m;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum_logf = 0.0f;

    float a = g_obs_lo[0];
    float b = g_obs_hi[0];

#if ENABLE_FUSED
    /* -------------------------------------------------------------------
     * Fused single-pass event loop: computes logp + gradients together,
     * caches per-process results, and accumulates gradients from cache.
     * Eliminates redundant PDF evaluations — ~50% fewer exp() calls for
     * Gaussian/Exponential PDFs.  Falls back to 2-pass when n_procs >
     * FUSED_PROC_CAP or any Chebyshev order exceeds FUSED_GRAD_STRIDE.
     * ----------------------------------------------------------------- */
    bool can_fuse = (n_procs <= FUSED_PROC_CAP);
    if (can_fuse) {
        for (uint fp = 0; fp < n_procs; fp++) {
            MetalUnbinnedProcessDesc fproc = g_procs[fp];
            if (fproc.pdf_kind == PDF_CHEBYSHEV && fproc.n_shape_params > FUSED_GRAD_STRIDE) {
                can_fuse = false;
                break;
            }
        }
    }
    if (can_fuse) {
        float c_logp[FUSED_PROC_CAP];
        float c_grad[FUSED_PROC_CAP * FUSED_GRAD_STRIDE];
        uint  c_pidx[FUSED_PROC_CAP * FUSED_GRAD_STRIDE];
        uint  c_ng[FUSED_PROC_CAP];

        // Precompute shape param indices (constant across events).
        for (uint fp = 0; fp < n_procs; fp++) {
            MetalUnbinnedProcessDesc fproc = g_procs[fp];
            uint foff = fproc.shape_param_offset;
            uint fgb = fp * FUSED_GRAD_STRIDE;
            uint fns = fproc.n_shape_params;
            for (uint fj = 0; fj < fns && fj < FUSED_GRAD_STRIDE; fj++) {
                c_pidx[fgb + fj] = (foff + fj < total_shape_params) ? g_shape_pidx[foff + fj] : 0u;
            }
        }

        for (uint i = tid; i < n_events; i += block_size) {
            float x = g_obs_flat[start + i];
            float max_term = -INFINITY;
            float sum_exp = 0.0f;

            // Phase 1: compute logp + grads per process, accumulate logsumexp.
            for (uint p = 0; p < n_procs; p++) {
                MetalUnbinnedProcessDesc proc = g_procs[p];
                if (proc.obs_index != 0u) {
                    c_logp[p] = -INFINITY; c_ng[p] = 0; continue;
                }
                float log_nu = s_proc_log_nu[p];
                if (!isfinite(log_nu)) {
                    c_logp[p] = -INFINITY; c_ng[p] = 0; continue;
                }

                float logp = -INFINITY;
                uint ng = 0;
                uint gbase = p * FUSED_GRAD_STRIDE;

                if (proc.pdf_kind == PDF_GAUSSIAN) {
                    if (proc.n_shape_params != 2u) { c_logp[p] = -INFINITY; c_ng[p] = 0; continue; }
                    float dmu, ds;
                    gaussian_logp_grad(x, s_params[c_pidx[gbase]], s_params[c_pidx[gbase + 1u]],
                                       a, b, logp, dmu, ds);
                    c_grad[gbase] = dmu; c_grad[gbase + 1] = ds;
                    ng = 2;
                } else if (proc.pdf_kind == PDF_EXPONENTIAL) {
                    if (proc.n_shape_params != 1u) { c_logp[p] = -INFINITY; c_ng[p] = 0; continue; }
                    float dl;
                    exponential_logp_grad(x, s_params[c_pidx[gbase]], a, b, logp, dl);
                    c_grad[gbase] = dl;
                    ng = 1;
                } else if (proc.pdf_kind == PDF_CRYSTAL_BALL) {
                    if (proc.n_shape_params != 4u) { c_logp[p] = -INFINITY; c_ng[p] = 0; continue; }
                    float dmu, ds, da, dn;
                    crystal_ball_logp_grad(x,
                        s_params[c_pidx[gbase]], s_params[c_pidx[gbase + 1u]],
                        s_params[c_pidx[gbase + 2u]], s_params[c_pidx[gbase + 3u]],
                        a, b, logp, dmu, ds, da, dn);
                    c_grad[gbase] = dmu; c_grad[gbase + 1] = ds;
                    c_grad[gbase + 2] = da; c_grad[gbase + 3] = dn;
                    ng = 4;
                } else if (proc.pdf_kind == PDF_DOUBLE_CRYSTAL_BALL) {
                    if (proc.n_shape_params != 6u) { c_logp[p] = -INFINITY; c_ng[p] = 0; continue; }
                    float dmu, ds, da_l, dn_l, da_r, dn_r;
                    double_crystal_ball_logp_grad(x,
                        s_params[c_pidx[gbase]], s_params[c_pidx[gbase + 1u]],
                        s_params[c_pidx[gbase + 2u]], s_params[c_pidx[gbase + 3u]],
                        s_params[c_pidx[gbase + 4u]], s_params[c_pidx[gbase + 5u]],
                        a, b, logp, dmu, ds, da_l, dn_l, da_r, dn_r);
                    c_grad[gbase] = dmu; c_grad[gbase + 1] = ds;
                    c_grad[gbase + 2] = da_l; c_grad[gbase + 3] = dn_l;
                    c_grad[gbase + 4] = da_r; c_grad[gbase + 5] = dn_r;
                    ng = 6;
                } else if (proc.pdf_kind == PDF_CHEBYSHEV) {
                    uint order = proc.n_shape_params;
                    if (order == 0u) { c_logp[p] = -INFINITY; c_ng[p] = 0; continue; }
                    float w = b - a;
                    if (!isfinite(w) || !(w > 0.0f)) { c_logp[p] = -INFINITY; c_ng[p] = 0; continue; }
                    float i0 = w;
                    for (uint j = 0; j < order; j++) {
                        uint k = j + 1u;
                        if ((k & 1u) == 0u) {
                            float c = s_params[c_pidx[gbase + j]];
                            float denom = 1.0f - (float)k * (float)k;
                            i0 += w * c / denom;
                        }
                    }
                    if (!isfinite(i0) || !(i0 > 0.0f)) { c_logp[p] = -INFINITY; c_ng[p] = 0; continue; }
                    float xp = chebyshev_xprime(x, a, b);
                    float f0 = 1.0f;
                    float tkm1 = 1.0f, tk = xp;
                    for (uint j = 0; j < order; j++) {
                        uint k = j + 1u;
                        float tval;
                        if (k == 1u) { tval = tk; }
                        else { float tkp1 = 2.0f * xp * tk - tkm1; tval = tkp1; tkm1 = tk; tk = tkp1; }
                        f0 += s_params[c_pidx[gbase + j]] * tval;
                    }
                    if (!isfinite(f0) || !(f0 > 0.0f)) { c_logp[p] = -INFINITY; c_ng[p] = 0; continue; }
                    logp = log(f0) - log(i0);
                    float inv_f0 = 1.0f / f0, inv_i0 = 1.0f / i0;
                    tkm1 = 1.0f; tk = xp;
                    for (uint j = 0; j < order; j++) {
                        uint k = j + 1u;
                        float tval;
                        if (k == 1u) { tval = tk; }
                        else { float tkp1 = 2.0f * xp * tk - tkm1; tval = tkp1; tkm1 = tk; tk = tkp1; }
                        float dlogi = 0.0f;
                        if ((k & 1u) == 0u) {
                            float denom = 1.0f - (float)k * (float)k;
                            dlogi = (w / denom) * inv_i0;
                        }
                        c_grad[gbase + j] = tval * inv_f0 - dlogi;
                    }
                    ng = order;
                } else if (proc.pdf_kind == PDF_HISTOGRAM) {
                    logp = histogram_logp_only(x, g_pdf_aux_f32, proc.pdf_aux_offset, proc.pdf_aux_len);
                    ng = 0;
                } else {
                    c_logp[p] = -INFINITY; c_ng[p] = 0; continue;
                }

                c_logp[p] = logp;
                c_ng[p] = ng;

                float term = log_nu + logp;
                if (!isfinite(term)) { continue; }
                if (term > max_term) {
                    sum_exp = sum_exp * exp(max_term - term) + 1.0f;
                    max_term = term;
                } else {
                    sum_exp += exp(term - max_term);
                }
            }

            float logf = max_term + log(sum_exp);
            if (!isfinite(logf)) { logf = log(FLT_MIN); }
            local_sum_logf += logf;

            // Phase 2: gradient accumulation from cached logp + grads.
            for (uint p = 0; p < n_procs; p++) {
                float logp = c_logp[p];
                if (!isfinite(logp)) { continue; }
                float p_over_f = exp(logp - logf);
                if (!(p_over_f > 0.0f) || !isfinite(p_over_f)) { continue; }

                float nu = s_proc_nu[p];
                float dnu = s_proc_dnu[p];
                uint y_idx = g_procs[p].yield_param_idx;
                if (dnu != 0.0f) {
                    batch_grad_add(local_grad, local_grad_cols, grad_out, y_idx, n_params, -dnu * p_over_f);
                }

                uint mod_off = s_proc_mod_off[p];
                uint nmods = s_proc_nmods[p];
                for (uint m = 0; m < nmods; m++) {
                    uint midx = mod_off + m;
                    MetalUnbinnedRateModifierDesc rm = g_rate_mods[midx];
                    uint aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) { continue; }
                    float dnu_m = batch_rate_mod_dnu(
                        g_rate_mods, s_params, s_rate_mod_dnu,
                        use_rate_mod_dnu_cache, midx, nu
                    );
                    if (isfinite(dnu_m) && dnu_m != 0.0f) {
                        batch_grad_add(local_grad, local_grad_cols, grad_out, aidx, n_params, -dnu_m * p_over_f);
                    }
                }

                float r = nu * p_over_f;
                uint gbase = p * FUSED_GRAD_STRIDE;
                uint ng = c_ng[p];
                for (uint g = 0; g < ng; g++) {
                    batch_grad_add(local_grad, local_grad_cols, grad_out, c_pidx[gbase + g], n_params, -r * c_grad[gbase + g]);
                }
            }
        }
    } /* end fused single-pass */

    if (!can_fuse) {
#endif
    /* --- Original 2-pass event loop (fallback for large n_procs). --- */
    for (uint i = tid; i < n_events; i += block_size) {
        float x = g_obs_flat[start + i];

        float max_term = -INFINITY;
        float sum_exp = 0.0f;

        for (uint p = 0; p < n_procs; p++) {
            MetalUnbinnedProcessDesc proc = g_procs[p];
            if (proc.obs_index != 0u) {
                continue;
            }

            float nu = 0.0f;
            float log_nu = -INFINITY;
            if (use_proc_cache) {
                nu = s_proc_nu[p];
                log_nu = s_proc_log_nu[p];
                if (!isfinite(log_nu)) {
                    continue;
                }
            } else {
                if (proc.yield_kind == YIELD_FIXED) {
                    nu = proc.base_yield;
                } else if (proc.yield_kind == YIELD_PARAMETER) {
                    nu = s_params[proc.yield_param_idx];
                } else if (proc.yield_kind == YIELD_SCALED) {
                    nu = proc.base_yield * s_params[proc.yield_param_idx];
                } else {
                    continue;
                }
                uint mod_off = proc.rate_mod_offset;
                uint nmods = proc.n_rate_mods;
                if (mod_off + nmods > total_rate_mods) {
                    nmods = 0u;
                }
                for (uint m = 0; m < nmods; m++) {
                    nu *= rate_modifier_factor_only_tg(g_rate_mods, mod_off + m, s_params);
                }
                if (!(nu > 0.0f) || !isfinite(nu)) {
                    continue;
                }
                log_nu = log(nu);
            }

            float logp = -INFINITY;
            if (proc.pdf_kind == PDF_GAUSSIAN) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 2u || off + 1u >= total_shape_params) {
                    continue;
                }
                uint mu_idx = g_shape_pidx[off];
                uint sig_idx = g_shape_pidx[off + 1u];
                float mu = s_params[mu_idx];
                float sigma = s_params[sig_idx];
                logp = gaussian_logp_only(x, mu, sigma, a, b);
            } else if (proc.pdf_kind == PDF_EXPONENTIAL) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 1u || off >= total_shape_params) {
                    continue;
                }
                uint lam_idx = g_shape_pidx[off];
                float lambda = s_params[lam_idx];
                logp = exponential_logp_only(x, lambda, a, b);
            } else if (proc.pdf_kind == PDF_CRYSTAL_BALL) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 4u || off + 3u >= total_shape_params) {
                    continue;
                }
                uint mu_idx = g_shape_pidx[off];
                uint sig_idx = g_shape_pidx[off + 1u];
                uint alpha_idx = g_shape_pidx[off + 2u];
                uint n_idx = g_shape_pidx[off + 3u];
                float mu = s_params[mu_idx];
                float sigma = s_params[sig_idx];
                float alpha = s_params[alpha_idx];
                float nn = s_params[n_idx];
                logp = crystal_ball_logp_only(x, mu, sigma, alpha, nn, a, b);
            } else if (proc.pdf_kind == PDF_DOUBLE_CRYSTAL_BALL) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 6u || off + 5u >= total_shape_params) {
                    continue;
                }
                uint mu_idx = g_shape_pidx[off];
                uint sig_idx = g_shape_pidx[off + 1u];
                uint alpha_l_idx = g_shape_pidx[off + 2u];
                uint n_l_idx = g_shape_pidx[off + 3u];
                uint alpha_r_idx = g_shape_pidx[off + 4u];
                uint n_r_idx = g_shape_pidx[off + 5u];
                float mu = s_params[mu_idx];
                float sigma = s_params[sig_idx];
                float alpha_l = s_params[alpha_l_idx];
                float n_l = s_params[n_l_idx];
                float alpha_r = s_params[alpha_r_idx];
                float n_r = s_params[n_r_idx];
                logp = double_crystal_ball_logp_only(x, mu, sigma, alpha_l, n_l, alpha_r, n_r, a, b);
            } else if (proc.pdf_kind == PDF_CHEBYSHEV) {
                uint off = proc.shape_param_offset;
                uint order = proc.n_shape_params;
                if (order == 0u || off + order - 1u >= total_shape_params) {
                    continue;
                }
                float w = b - a;
                if (!isfinite(w) || !(w > 0.0f)) {
                    continue;
                }
                float i0 = w;
                for (uint j = 0; j < order; j++) {
                    uint k = j + 1u;
                    if ((k & 1u) == 0u) {
                        uint c_idx = g_shape_pidx[off + j];
                        float c = s_params[c_idx];
                        float denom = 1.0f - (float)k * (float)k;
                        i0 += w * c / denom;
                    }
                }
                if (!isfinite(i0) || !(i0 > 0.0f)) {
                    continue;
                }
                float log_i = log(i0);
                float xp = chebyshev_xprime(x, a, b);
                float f0 = 1.0f;
                float tkm1 = 1.0f;
                float tk = xp;
                for (uint j = 0; j < order; j++) {
                    uint k = j + 1u;
                    float tval = 0.0f;
                    if (k == 1u) {
                        tval = tk;
                    } else {
                        float tkp1 = 2.0f * xp * tk - tkm1;
                        tval = tkp1;
                        tkm1 = tk;
                        tk = tkp1;
                    }
                    uint c_idx = g_shape_pidx[off + j];
                    float c = s_params[c_idx];
                    f0 += c * tval;
                }
                if (!isfinite(f0) || !(f0 > 0.0f)) {
                    continue;
                }
                logp = log(f0) - log_i;
            } else if (proc.pdf_kind == PDF_HISTOGRAM) {
                if (proc.n_shape_params != 0u) {
                    continue;
                }
                logp = histogram_logp_only(x, g_pdf_aux_f32, proc.pdf_aux_offset, proc.pdf_aux_len);
            } else {
                continue;
            }

            float term = log_nu + logp;
            if (!isfinite(term)) {
                continue;
            }

            if (term > max_term) {
                sum_exp = sum_exp * exp(max_term - term) + 1.0f;
                max_term = term;
            } else {
                sum_exp += exp(term - max_term);
            }
        }

        float logf = max_term + log(sum_exp);
        if (!isfinite(logf)) {
            logf = log(FLT_MIN);
        }
        local_sum_logf += logf;

        for (uint p = 0; p < n_procs; p++) {
            MetalUnbinnedProcessDesc proc = g_procs[p];
            if (proc.obs_index != 0u) {
                continue;
            }

            float nu = 0.0f;
            float dnu = 0.0f;
            uint y_idx = proc.yield_param_idx;
            uint has_yield_param = 0u;
            uint mod_off = 0u;
            uint nmods = 0u;
            if (use_proc_cache) {
                mod_off = s_proc_mod_off[p];
                nmods = s_proc_nmods[p];
                nu = s_proc_nu[p];
                dnu = s_proc_dnu[p];
                has_yield_param = (dnu != 0.0f) ? 1u : 0u;
            } else {
                mod_off = proc.rate_mod_offset;
                nmods = proc.n_rate_mods;
                if (proc.yield_kind == YIELD_FIXED) {
                    nu = proc.base_yield;
                } else if (proc.yield_kind == YIELD_PARAMETER) {
                    nu = s_params[y_idx];
                    dnu = 1.0f;
                    has_yield_param = 1u;
                } else if (proc.yield_kind == YIELD_SCALED) {
                    nu = proc.base_yield * s_params[y_idx];
                    dnu = proc.base_yield;
                    has_yield_param = 1u;
                } else {
                    continue;
                }
                if (mod_off + nmods > total_rate_mods) {
                    nmods = 0u;
                }
                float mod_factor = 1.0f;
                for (uint m = 0; m < nmods; m++) {
                    mod_factor *= rate_modifier_factor_only_tg(g_rate_mods, mod_off + m, s_params);
                }
                nu *= mod_factor;
                dnu *= mod_factor;
            }
            if (!(nu > 0.0f) || !isfinite(nu)) {
                continue;
            }

            if (proc.pdf_kind == PDF_GAUSSIAN) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 2u || off + 1u >= total_shape_params) {
                    continue;
                }
                uint mu_idx = g_shape_pidx[off];
                uint sig_idx = g_shape_pidx[off + 1u];
                float mu = s_params[mu_idx];
                float sigma = s_params[sig_idx];

                float logp, dmu, ds;
                gaussian_logp_grad(x, mu, sigma, a, b, logp, dmu, ds);
                if (!isfinite(logp)) {
                    continue;
                }

                float p_over_f = exp(logp - logf);
                if (!(p_over_f > 0.0f) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    batch_grad_add(local_grad, local_grad_cols, grad_out, y_idx, n_params, -dnu * p_over_f);
                }
                for (uint m = 0; m < nmods; m++) {
                    uint midx = mod_off + m;
                    MetalUnbinnedRateModifierDesc rm = g_rate_mods[midx];
                    uint aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    float dnu_m = batch_rate_mod_dnu(
                        g_rate_mods, s_params, s_rate_mod_dnu,
                        use_rate_mod_dnu_cache, midx, nu
                    );
                    if (isfinite(dnu_m) && dnu_m != 0.0f) {
                        batch_grad_add(local_grad, local_grad_cols, grad_out, aidx, n_params, -dnu_m * p_over_f);
                    }
                }

                float r = nu * p_over_f;
                batch_grad_add(local_grad, local_grad_cols, grad_out, mu_idx, n_params, -r * dmu);
                batch_grad_add(local_grad, local_grad_cols, grad_out, sig_idx, n_params, -r * ds);
            } else if (proc.pdf_kind == PDF_EXPONENTIAL) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 1u || off >= total_shape_params) {
                    continue;
                }
                uint lam_idx = g_shape_pidx[off];
                float lambda = s_params[lam_idx];

                float logp, dl;
                exponential_logp_grad(x, lambda, a, b, logp, dl);
                if (!isfinite(logp)) {
                    continue;
                }

                float p_over_f = exp(logp - logf);
                if (!(p_over_f > 0.0f) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    batch_grad_add(local_grad, local_grad_cols, grad_out, y_idx, n_params, -dnu * p_over_f);
                }
                for (uint m = 0; m < nmods; m++) {
                    uint midx = mod_off + m;
                    MetalUnbinnedRateModifierDesc rm = g_rate_mods[midx];
                    uint aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    float dnu_m = batch_rate_mod_dnu(
                        g_rate_mods, s_params, s_rate_mod_dnu,
                        use_rate_mod_dnu_cache, midx, nu
                    );
                    if (isfinite(dnu_m) && dnu_m != 0.0f) {
                        batch_grad_add(local_grad, local_grad_cols, grad_out, aidx, n_params, -dnu_m * p_over_f);
                    }
                }

                float r = nu * p_over_f;
                batch_grad_add(local_grad, local_grad_cols, grad_out, lam_idx, n_params, -r * dl);
            } else if (proc.pdf_kind == PDF_CRYSTAL_BALL) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 4u || off + 3u >= total_shape_params) {
                    continue;
                }
                uint mu_idx = g_shape_pidx[off];
                uint sig_idx = g_shape_pidx[off + 1u];
                uint alpha_idx = g_shape_pidx[off + 2u];
                uint n_idx = g_shape_pidx[off + 3u];
                float mu = s_params[mu_idx];
                float sigma = s_params[sig_idx];
                float alpha = s_params[alpha_idx];
                float nn = s_params[n_idx];

                float logp, dmu, ds, da, dn;
                crystal_ball_logp_grad(x, mu, sigma, alpha, nn, a, b, logp, dmu, ds, da, dn);
                if (!isfinite(logp)) {
                    continue;
                }
                float p_over_f = exp(logp - logf);
                if (!(p_over_f > 0.0f) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    batch_grad_add(local_grad, local_grad_cols, grad_out, y_idx, n_params, -dnu * p_over_f);
                }
                for (uint m = 0; m < nmods; m++) {
                    uint midx = mod_off + m;
                    MetalUnbinnedRateModifierDesc rm = g_rate_mods[midx];
                    uint aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    float dnu_m = batch_rate_mod_dnu(
                        g_rate_mods, s_params, s_rate_mod_dnu,
                        use_rate_mod_dnu_cache, midx, nu
                    );
                    if (isfinite(dnu_m) && dnu_m != 0.0f) {
                        batch_grad_add(local_grad, local_grad_cols, grad_out, aidx, n_params, -dnu_m * p_over_f);
                    }
                }

                float r = nu * p_over_f;
                batch_grad_add(local_grad, local_grad_cols, grad_out, mu_idx, n_params, -r * dmu);
                batch_grad_add(local_grad, local_grad_cols, grad_out, sig_idx, n_params, -r * ds);
                batch_grad_add(local_grad, local_grad_cols, grad_out, alpha_idx, n_params, -r * da);
                batch_grad_add(local_grad, local_grad_cols, grad_out, n_idx, n_params, -r * dn);
            } else if (proc.pdf_kind == PDF_DOUBLE_CRYSTAL_BALL) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 6u || off + 5u >= total_shape_params) {
                    continue;
                }
                uint mu_idx = g_shape_pidx[off];
                uint sig_idx = g_shape_pidx[off + 1u];
                uint alpha_l_idx = g_shape_pidx[off + 2u];
                uint n_l_idx = g_shape_pidx[off + 3u];
                uint alpha_r_idx = g_shape_pidx[off + 4u];
                uint n_r_idx = g_shape_pidx[off + 5u];
                float mu = s_params[mu_idx];
                float sigma = s_params[sig_idx];
                float alpha_l = s_params[alpha_l_idx];
                float n_l = s_params[n_l_idx];
                float alpha_r = s_params[alpha_r_idx];
                float n_r = s_params[n_r_idx];

                float logp, dmu, ds, da_l, dn_l, da_r, dn_r;
                double_crystal_ball_logp_grad(
                    x, mu, sigma, alpha_l, n_l, alpha_r, n_r, a, b,
                    logp, dmu, ds, da_l, dn_l, da_r, dn_r
                );
                if (!isfinite(logp)) {
                    continue;
                }
                float p_over_f = exp(logp - logf);
                if (!(p_over_f > 0.0f) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    batch_grad_add(local_grad, local_grad_cols, grad_out, y_idx, n_params, -dnu * p_over_f);
                }
                for (uint m = 0; m < nmods; m++) {
                    uint midx = mod_off + m;
                    MetalUnbinnedRateModifierDesc rm = g_rate_mods[midx];
                    uint aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    float dnu_m = batch_rate_mod_dnu(
                        g_rate_mods, s_params, s_rate_mod_dnu,
                        use_rate_mod_dnu_cache, midx, nu
                    );
                    if (isfinite(dnu_m) && dnu_m != 0.0f) {
                        batch_grad_add(local_grad, local_grad_cols, grad_out, aidx, n_params, -dnu_m * p_over_f);
                    }
                }

                float r = nu * p_over_f;
                batch_grad_add(local_grad, local_grad_cols, grad_out, mu_idx, n_params, -r * dmu);
                batch_grad_add(local_grad, local_grad_cols, grad_out, sig_idx, n_params, -r * ds);
                batch_grad_add(local_grad, local_grad_cols, grad_out, alpha_l_idx, n_params, -r * da_l);
                batch_grad_add(local_grad, local_grad_cols, grad_out, n_l_idx, n_params, -r * dn_l);
                batch_grad_add(local_grad, local_grad_cols, grad_out, alpha_r_idx, n_params, -r * da_r);
                batch_grad_add(local_grad, local_grad_cols, grad_out, n_r_idx, n_params, -r * dn_r);
            } else if (proc.pdf_kind == PDF_CHEBYSHEV) {
                uint off = proc.shape_param_offset;
                uint order = proc.n_shape_params;
                if (order == 0u || off + order - 1u >= total_shape_params) {
                    continue;
                }
                float w = b - a;
                if (!isfinite(w) || !(w > 0.0f)) {
                    continue;
                }
                float i0 = w;
                for (uint j = 0; j < order; j++) {
                    uint k = j + 1u;
                    if ((k & 1u) == 0u) {
                        uint c_idx = g_shape_pidx[off + j];
                        float c = s_params[c_idx];
                        float denom = 1.0f - (float)k * (float)k;
                        i0 += w * c / denom;
                    }
                }
                if (!isfinite(i0) || !(i0 > 0.0f)) {
                    continue;
                }
                float log_i = log(i0);

                float xp = chebyshev_xprime(x, a, b);
                float f0 = 1.0f;
                float tkm1 = 1.0f;
                float tk = xp;
                for (uint j = 0; j < order; j++) {
                    uint k = j + 1u;
                    float tval = 0.0f;
                    if (k == 1u) {
                        tval = tk;
                    } else {
                        float tkp1 = 2.0f * xp * tk - tkm1;
                        tval = tkp1;
                        tkm1 = tk;
                        tk = tkp1;
                    }
                    uint c_idx = g_shape_pidx[off + j];
                    float c = s_params[c_idx];
                    f0 += c * tval;
                }
                if (!isfinite(f0) || !(f0 > 0.0f)) {
                    continue;
                }
                float logp = log(f0) - log_i;
                if (!isfinite(logp)) {
                    continue;
                }
                float p_over_f = exp(logp - logf);
                if (!(p_over_f > 0.0f) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    batch_grad_add(local_grad, local_grad_cols, grad_out, y_idx, n_params, -dnu * p_over_f);
                }
                for (uint m = 0; m < nmods; m++) {
                    uint midx = mod_off + m;
                    MetalUnbinnedRateModifierDesc rm = g_rate_mods[midx];
                    uint aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    float dnu_m = batch_rate_mod_dnu(
                        g_rate_mods, s_params, s_rate_mod_dnu,
                        use_rate_mod_dnu_cache, midx, nu
                    );
                    if (isfinite(dnu_m) && dnu_m != 0.0f) {
                        batch_grad_add(local_grad, local_grad_cols, grad_out, aidx, n_params, -dnu_m * p_over_f);
                    }
                }

                float r = nu * p_over_f;
                float inv_f0 = 1.0f / f0;
                float inv_i0 = 1.0f / i0;
                tkm1 = 1.0f;
                tk = xp;
                for (uint j = 0; j < order; j++) {
                    uint k = j + 1u;
                    float tval = 0.0f;
                    if (k == 1u) {
                        tval = tk;
                    } else {
                        float tkp1 = 2.0f * xp * tk - tkm1;
                        tval = tkp1;
                        tkm1 = tk;
                        tk = tkp1;
                    }
                    float dlogi = 0.0f;
                    if ((k & 1u) == 0u) {
                        float denom = 1.0f - (float)k * (float)k;
                        float di_dc = w / denom;
                        dlogi = di_dc * inv_i0;
                    }
                    float dlogp_dc = tval * inv_f0 - dlogi;
                    uint c_idx = g_shape_pidx[off + j];
                    batch_grad_add(local_grad, local_grad_cols, grad_out, c_idx, n_params, -r * dlogp_dc);
                }
            }
            else if (proc.pdf_kind == PDF_HISTOGRAM) {
                if (proc.n_shape_params != 0u) {
                    continue;
                }
                float logp = histogram_logp_only(x, g_pdf_aux_f32, proc.pdf_aux_offset, proc.pdf_aux_len);
                if (!isfinite(logp)) {
                    continue;
                }
                float p_over_f = exp(logp - logf);
                if (!(p_over_f > 0.0f) || !isfinite(p_over_f)) {
                    continue;
                }

                if (has_yield_param) {
                    batch_grad_add(local_grad, local_grad_cols, grad_out, y_idx, n_params, -dnu * p_over_f);
                }
                for (uint m = 0; m < nmods; m++) {
                    uint midx = mod_off + m;
                    MetalUnbinnedRateModifierDesc rm = g_rate_mods[midx];
                    uint aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    float dnu_m = batch_rate_mod_dnu(
                        g_rate_mods, s_params, s_rate_mod_dnu,
                        use_rate_mod_dnu_cache, midx, nu
                    );
                    if (isfinite(dnu_m) && dnu_m != 0.0f) {
                        batch_grad_add(local_grad, local_grad_cols, grad_out, aidx, n_params, -dnu_m * p_over_f);
                    }
                }
            }
        }
    }
#if ENABLE_FUSED
    } /* end if (!can_fuse) */
#endif

    /* Gradient reduction: tree reduction across threads (O(log2(block_size))). */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = 0; p < local_grad_cols; p++) {
        s_scratch[tid] = s_grad_partial[tid * local_grad_cols + p];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_scratch[tid] += s_scratch[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            float acc = s_scratch[0];
            if (isfinite(acc) && acc != 0.0f) {
                atomic_fetch_add_explicit(&grad_out[p], acc, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    s_scratch[tid] = local_sum_logf;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] += s_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float sum_logf = s_scratch[0];

        float nu_tot = 0.0f;
        for (uint p = 0; p < n_procs; p++) {
            MetalUnbinnedProcessDesc proc = g_procs[p];
            float nu = 0.0f;
            float dnu = 0.0f;
            uint y_idx = proc.yield_param_idx;
            uint has_yield_param = 0u;
            uint mod_off = 0u;
            uint nmods = 0u;
            if (use_proc_cache) {
                mod_off = s_proc_mod_off[p];
                nmods = s_proc_nmods[p];
                nu = s_proc_nu[p];
                dnu = s_proc_dnu[p];
                has_yield_param = (dnu != 0.0f) ? 1u : 0u;
            } else {
                mod_off = proc.rate_mod_offset;
                nmods = proc.n_rate_mods;
                if (proc.yield_kind == YIELD_FIXED) {
                    nu = proc.base_yield;
                } else if (proc.yield_kind == YIELD_PARAMETER) {
                    nu = s_params[y_idx];
                    dnu = 1.0f;
                    has_yield_param = 1u;
                } else if (proc.yield_kind == YIELD_SCALED) {
                    nu = proc.base_yield * s_params[y_idx];
                    dnu = proc.base_yield;
                    has_yield_param = 1u;
                } else {
                    continue;
                }
                if (mod_off + nmods > total_rate_mods) {
                    nmods = 0u;
                }
                float mod_factor = 1.0f;
                for (uint m = 0; m < nmods; m++) {
                    mod_factor *= rate_modifier_factor_only_tg(g_rate_mods, mod_off + m, s_params);
                }
                nu *= mod_factor;
                dnu *= mod_factor;
            }
            if (isfinite(nu) && nu >= 0.0f) {
                nu_tot += nu;
                if (has_yield_param && isfinite(dnu) && dnu != 0.0f) {
                    atomic_fetch_add_explicit(&grad_out[y_idx], dnu, memory_order_relaxed);
                }
                for (uint m = 0; m < nmods; m++) {
                    uint midx = mod_off + m;
                    MetalUnbinnedRateModifierDesc rm = g_rate_mods[midx];
                    uint aidx = rm.alpha_param_idx;
                    if (aidx >= n_params) {
                        continue;
                    }
                    float dnu_m = batch_rate_mod_dnu(
                        g_rate_mods, s_params, s_rate_mod_dnu,
                        use_rate_mod_dnu_cache, midx, nu
                    );
                    if (isfinite(dnu_m) && dnu_m != 0.0f) {
                        atomic_fetch_add_explicit(&grad_out[aidx], dnu_m, memory_order_relaxed);
                    }
                }
            }
        }

        float nll = nu_tot - sum_logf;
        for (uint k = 0; k < n_gauss; k++) {
            MetalUnbinnedGaussConstraintEntry gc = g_gauss[k];
            uint idx = gc.param_idx;
            if (idx >= n_params) {
                continue;
            }
            float x0 = s_params[idx];
            float diff = x0 - gc.center;
            float z = diff * gc.inv_width;
            nll += 0.5f * z * z;
            atomic_fetch_add_explicit(&grad_out[idx], z * gc.inv_width, memory_order_relaxed);
        }

        nll += constraint_const;
        g_nll_out[toy] = nll;
    }
}

/* ---------- Toy sampling (Metal) ---------------------------------------- */

struct ToyScalarArgs {
    uint n_procs;
    uint total_rate_mods;
    uint total_shape_params;
    uint n_toys;
    ulong seed;
    uint total_pdf_aux_f32;
    uint _pad0;
    uint _pad1;
};

constant uint TOY_MAX_PROC_CACHE = 256u;

constant float TOY_LOG_FACTORIAL_SMALL[21] = {
    0.0f,
    0.0f,
    0.6931471805599453f,
    1.791759469228055f,
    3.1780538303479458f,
    4.787491742782046f,
    6.579251212010101f,
    8.525161361065415f,
    10.60460290274525f,
    12.80182748008147f,
    15.104412573075516f,
    17.502307845873887f,
    19.987214495661885f,
    22.552163853123425f,
    25.191221182738683f,
    27.899271383840894f,
    30.671860106080675f,
    33.50507345013689f,
    36.39544520803305f,
    39.339884187199495f,
    42.335616460753485f,
};

inline ulong splitmix64_next(thread ulong* state) {
    ulong z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

inline float u01_open(thread ulong* state) {
    // Uniform in (0,1) with ~24 bits of mantissa (sufficient for f32 backend).
    ulong x = splitmix64_next(state);
    uint mant = (uint)(x >> 40); // top 24 bits
    return ((float)mant + 0.5f) * (1.0f / 16777216.0f); // 2^24
}

inline float norminv_acklam(float p) {
    // Rational approximation for inverse normal CDF (Acklam), with one Halley refinement step.
    // p must be in (0,1).
    const float a1 = -3.9696830e+01f;
    const float a2 = 2.2094610e+02f;
    const float a3 = -2.7592852e+02f;
    const float a4 = 1.3835775e+02f;
    const float a5 = -3.0664797e+01f;
    const float a6 = 2.5066283e+00f;

    const float b1 = -5.4476097e+01f;
    const float b2 = 1.6158583e+02f;
    const float b3 = -1.5569899e+02f;
    const float b4 = 6.6801311e+01f;
    const float b5 = -1.3280681e+01f;

    const float c1 = -7.7848940e-03f;
    const float c2 = -3.2239646e-01f;
    const float c3 = -2.4007583e+00f;
    const float c4 = -2.5497324e+00f;
    const float c5 = 4.3746643e+00f;
    const float c6 = 2.9381640e+00f;

    const float d1 = 7.7846959e-03f;
    const float d2 = 3.2246712e-01f;
    const float d3 = 2.4451342e+00f;
    const float d4 = 3.7544086e+00f;

    const float p_low = 0.02425f;
    const float p_high = 1.0f - p_low;

    float x;
    if (p < p_low) {
        float q = sqrt(-2.0f * log(p));
        x = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0f);
        x = -x;
    } else if (p <= p_high) {
        float q = p - 0.5f;
        float r = q * q;
        x = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0f);
    } else {
        float q = sqrt(-2.0f * log(1.0f - p));
        x = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0f);
    }

    // One Halley refinement step.
    float e = stdnorm_cdf(x) - p;
    float u = e / stdnorm_pdf(x);
    x = x - u / (1.0f + 0.5f * x * u);
    return x;
}

inline float gaussian_sample_trunc(float mu, float sigma, float a, float b, thread ulong* state) {
    if (!isfinite(mu) || !isfinite(sigma) || sigma <= 0.0f) {
        return 0.5f * (a + b);
    }
    float z_a = (a - mu) / sigma;
    float z_b = (b - mu) / sigma;
    float u_lo = stdnorm_cdf(z_a);
    float u_hi = stdnorm_cdf(z_b);
    const float eps = 1e-7f;
    if (!isfinite(u_lo)) u_lo = eps;
    if (!isfinite(u_hi)) u_hi = 1.0f - eps;
    u_lo = fmax(u_lo, eps);
    u_hi = fmin(u_hi, 1.0f - eps);
    if (!(u_lo < u_hi)) {
        float x0 = mu;
        if (x0 < a) x0 = a;
        if (x0 > b) x0 = b;
        return x0;
    }
    float u = u_lo + (u_hi - u_lo) * u01_open(state);
    float z = norminv_acklam(u);
    float x = mu + sigma * z;
    if (x < a) x = a;
    if (x > b) x = b;
    return x;
}

inline float exponential_sample_trunc(float lambda, float a, float b, thread ulong* state) {
    float u = u01_open(state);
    if (!isfinite(lambda) || fabs(lambda) < 1e-12f) {
        return a + (b - a) * u;
    }

    float t_a = lambda * a;
    float t_b = lambda * b;
    float hi_t = (t_b >= t_a) ? t_b : t_a;
    float lo_t = (t_b >= t_a) ? t_a : t_b;
    float r = exp(lo_t - hi_t); // in (0,1)
    float one_minus_r = 1.0f - r;
    float yfac = (t_b >= t_a) ? (r + u * one_minus_r) : (1.0f - u * one_minus_r);
    yfac = fmin(fmax(yfac, FLT_MIN), 1.0f);

    float x = (hi_t + log(yfac)) / lambda;
    if (x < a) x = a;
    if (x > b) x = b;
    return x;
}

inline float log_factorial_approx(uint k) {
    // Exact small-k table, Stirling approximation for larger k.
    // Accuracy is sufficient for PTRS acceptance test.
    if (k <= 20u) {
        return TOY_LOG_FACTORIAL_SMALL[k];
    }
    float x = (float)k;
    const float half_log_2pi = 0.91893853320467274f; // 0.5*log(2*pi)
    float inv_x = 1.0f / x;
    float inv_x2 = inv_x * inv_x;
    float corr = inv_x / 12.0f - inv_x2 * inv_x / 360.0f;
    return (x + 0.5f) * log(x) - x + half_log_2pi + corr;
}

inline uint poisson_knuth(float lambda, thread ulong* state) {
    float L = exp(-lambda);
    uint k = 0u;
    float p = 1.0f;
    do {
        k += 1u;
        p *= u01_open(state);
    } while (p > L);
    return k - 1u;
}

inline uint poisson_ptrs(float lambda, thread ulong* state) {
    // PTRS: Poisson Transformed Rejection with Squeeze (Hörmann, 1993).
    float sl = sqrt(lambda);
    float b = 0.931f + 2.53f * sl;
    float a = -0.059f + 0.02483f * b;
    float inv_alpha = 1.1239f + 1.1328f / (b - 3.4f);
    float v_r = 0.9277f - 3.6224f / (b - 2.0f);

    while (true) {
        float u = u01_open(state) - 0.5f; // (-0.5,0.5)
        float v = u01_open(state);
        float us = 0.5f - fabs(u);
        if (!(us > 0.0f)) {
            continue;
        }
        int k = (int)floor((2.0f * a / us + b) * u + lambda + 0.43f);

        if (us >= 0.07f && v <= v_r) {
            return (k < 0) ? 0u : (uint)k;
        }
        if (k < 0) {
            continue;
        }
        if (us < 0.013f && v > us) {
            continue;
        }

        float lhs = log(v) + log(inv_alpha) - log(a / (us * us) + b);
        float rhs = -lambda + ((float)k) * log(lambda) - log_factorial_approx((uint)k);
        if (lhs <= rhs) {
            return (uint)k;
        }
    }
}

inline uint poisson_sample(float lambda, thread ulong* state) {
    if (!(lambda > 0.0f) || !isfinite(lambda)) {
        return 0u;
    }
    if (lambda < 10.0f) {
        return poisson_knuth(lambda, state);
    }
    return poisson_ptrs(lambda, state);
}

inline float rate_modifier_factor_only(
    const device MetalUnbinnedRateModifierDesc* mods,
    uint midx,
    const device float* params
) {
    MetalUnbinnedRateModifierDesc m = mods[midx];
    uint aidx = m.alpha_param_idx;
    float alpha = params[aidx];

    if (m.kind == RATE_NORM_SYS) {
        float lo = m.lo;
        float hi = m.hi;
        if (!(isfinite(lo) && lo > 0.0f && isfinite(hi) && hi > 0.0f && isfinite(alpha))) {
            return 1.0f;
        }
        float log_hi = log(hi);
        float log_lo = log(lo);
        if (alpha >= 0.0f) {
            return exp(alpha * log_hi);
        }
        return exp(-alpha * log_lo);
    }

    if (m.kind == RATE_WEIGHT_SYS) {
        float lo = m.lo;
        float hi = m.hi;
        if (!(isfinite(lo) && lo > 0.0f && isfinite(hi) && hi > 0.0f && isfinite(alpha))) {
            return 1.0f;
        }
        float val = 1.0f;
        float der = 0.0f;
        histosys_interp(alpha, lo, 1.0f, hi, m.interp_code, val, der);
        (void)der;
        if (!isfinite(val) || val <= 0.0f) {
            val = FLT_MIN;
        }
        return val;
    }

    return 1.0f;
}

inline float proc_yield(
    MetalUnbinnedProcessDesc proc,
    const device MetalUnbinnedRateModifierDesc* mods,
    uint total_rate_mods,
    const device float* params
) {
    float nu = 0.0f;
    if (proc.yield_kind == YIELD_FIXED) {
        nu = proc.base_yield;
    } else if (proc.yield_kind == YIELD_PARAMETER) {
        nu = params[proc.yield_param_idx];
    } else if (proc.yield_kind == YIELD_SCALED) {
        nu = proc.base_yield * params[proc.yield_param_idx];
    } else {
        nu = 0.0f;
    }
    uint mod_off = proc.rate_mod_offset;
    uint nmods = proc.n_rate_mods;
    if (mod_off + nmods > total_rate_mods) {
        nmods = 0u;
    }
    for (uint m = 0; m < nmods; m++) {
        nu *= rate_modifier_factor_only(mods, mod_off + m, params);
    }
    if (!isfinite(nu) || nu < 0.0f) {
        nu = 0.0f;
    }
    return nu;
}

/// Sample total event counts per toy: N ~ Poisson(nu_tot(params)).
kernel void unbinned_toy_counts(
    const device float* g_params [[buffer(0)]],
    const device MetalUnbinnedProcessDesc* g_procs [[buffer(1)]],
    const device MetalUnbinnedRateModifierDesc* g_rate_mods [[buffer(2)]],
    constant ToyScalarArgs& s [[buffer(3)]],
    device uint* g_counts [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint toy = gid;
    if (toy >= s.n_toys) {
        return;
    }

    float nu_tot = 0.0f;
    for (uint p = 0; p < s.n_procs; p++) {
        nu_tot += proc_yield(g_procs[p], g_rate_mods, s.total_rate_mods, g_params);
    }

    ulong state = (ulong)(s.seed + (ulong)toy) ^ 0x6a09e667f3bcc909ULL;
    uint n = poisson_sample(nu_tot, &state);
    g_counts[toy] = n;
}

/// Sample 1D toy observables for many toys into a flattened array.
kernel void unbinned_toy_sample_obs_1d(
    const device float* g_params [[buffer(0)]],
    const device float* g_obs_lo [[buffer(1)]],
    const device float* g_obs_hi [[buffer(2)]],
    const device MetalUnbinnedProcessDesc* g_procs [[buffer(3)]],
    const device MetalUnbinnedRateModifierDesc* g_rate_mods [[buffer(4)]],
    const device uint* g_shape_pidx [[buffer(5)]],
    const device uint* g_toy_offsets [[buffer(6)]],
    const device float* g_pdf_aux_f32 [[buffer(7)]],
    constant ToyScalarArgs& s [[buffer(8)]],
    device float* g_obs_flat_out [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    uint toy = tg_pos.x;
    if (toy >= s.n_toys) {
        return;
    }

    uint start = g_toy_offsets[toy];
    uint end = g_toy_offsets[toy + 1u];
    uint n_events = end - start;

    float a = g_obs_lo[0];
    float b = g_obs_hi[0];

    threadgroup float s_proc_cdf[TOY_MAX_PROC_CACHE];
    threadgroup float s_nu_tot;
    threadgroup uint s_cached_n_procs;

    if (tid == 0u) {
        float cum = 0.0f;
        if (s.n_procs <= TOY_MAX_PROC_CACHE) {
            uint cached_n = s.n_procs;
            for (uint p = 0u; p < cached_n; p++) {
                float nu = proc_yield(g_procs[p], g_rate_mods, s.total_rate_mods, g_params);
                if (!isfinite(nu) || nu < 0.0f) {
                    nu = 0.0f;
                }
                cum += nu;
                s_proc_cdf[p] = cum;
            }
            s_cached_n_procs = cached_n;
        } else {
            for (uint p = 0u; p < s.n_procs; p++) {
                float nu = proc_yield(g_procs[p], g_rate_mods, s.total_rate_mods, g_params);
                if (isfinite(nu) && nu > 0.0f) {
                    cum += nu;
                }
            }
            s_cached_n_procs = 0u;
        }
        s_nu_tot = cum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float nu_tot = s_nu_tot;
    if (!(nu_tot > 0.0f) || !isfinite(nu_tot)) {
        return;
    }

    uint stride = tg_size.x;
    for (uint i = tid; i < n_events; i += stride) {
        ulong state =
            (ulong)(s.seed + (ulong)toy) ^
            (0x9e3779b97f4a7c15ULL * ((ulong)i + 1ULL));

        float u = u01_open(&state) * nu_tot;
        uint chosen = 0u;
        if (s_cached_n_procs == s.n_procs && s_cached_n_procs > 0u) {
            chosen = s_cached_n_procs - 1u;
            for (uint p = 0u; p < s_cached_n_procs; p++) {
                if (u <= s_proc_cdf[p]) {
                    chosen = p;
                    break;
                }
            }
        } else {
            float cum = 0.0f;
            chosen = (s.n_procs > 0u) ? (s.n_procs - 1u) : 0u;
            for (uint p = 0u; p < s.n_procs; p++) {
                cum += proc_yield(g_procs[p], g_rate_mods, s.total_rate_mods, g_params);
                if (u <= cum) {
                    chosen = p;
                    break;
                }
            }
        }

        MetalUnbinnedProcessDesc proc = g_procs[chosen];
        float x = 0.5f * (a + b);

        if (proc.pdf_kind == PDF_GAUSSIAN) {
            uint off = proc.shape_param_offset;
            if (proc.n_shape_params == 2u && off + 1u < s.total_shape_params) {
                uint mu_idx = g_shape_pidx[off];
                uint sig_idx = g_shape_pidx[off + 1u];
                float mu = g_params[mu_idx];
                float sigma = g_params[sig_idx];
                x = gaussian_sample_trunc(mu, sigma, a, b, &state);
            }
        } else if (proc.pdf_kind == PDF_EXPONENTIAL) {
            uint off = proc.shape_param_offset;
            if (proc.n_shape_params == 1u && off < s.total_shape_params) {
                uint lam_idx = g_shape_pidx[off];
                float lambda = g_params[lam_idx];
                x = exponential_sample_trunc(lambda, a, b, &state);
            }
        } else if (proc.pdf_kind == PDF_CRYSTAL_BALL) {
            uint off = proc.shape_param_offset;
            if (proc.n_shape_params == 4u && off + 3u < s.total_shape_params) {
                uint mu_idx = g_shape_pidx[off];
                uint sig_idx = g_shape_pidx[off + 1u];
                uint alpha_idx = g_shape_pidx[off + 2u];
                uint n_idx = g_shape_pidx[off + 3u];
                float mu = g_params[mu_idx];
                float sigma = g_params[sig_idx];
                float alpha = g_params[alpha_idx];
                float nn = g_params[n_idx];

                float x_peak = clamp(mu, a, b);
                float logp_max = crystal_ball_logp_only(x_peak, mu, sigma, alpha, nn, a, b);
                if (isfinite(logp_max)) {
                    for (uint it = 0; it < 1024u; it++) {
                        float xr = a + (b - a) * u01_open(&state);
                        float logp = crystal_ball_logp_only(xr, mu, sigma, alpha, nn, a, b);
                        if (!isfinite(logp)) {
                            continue;
                        }
                        float lu = log(u01_open(&state));
                        if (lu <= (logp - logp_max)) {
                            x = xr;
                            break;
                        }
                    }
                }
            }
        } else if (proc.pdf_kind == PDF_DOUBLE_CRYSTAL_BALL) {
            uint off = proc.shape_param_offset;
            if (proc.n_shape_params == 6u && off + 5u < s.total_shape_params) {
                uint mu_idx = g_shape_pidx[off];
                uint sig_idx = g_shape_pidx[off + 1u];
                uint alpha_l_idx = g_shape_pidx[off + 2u];
                uint n_l_idx = g_shape_pidx[off + 3u];
                uint alpha_r_idx = g_shape_pidx[off + 4u];
                uint n_r_idx = g_shape_pidx[off + 5u];
                float mu = g_params[mu_idx];
                float sigma = g_params[sig_idx];
                float alpha_l = g_params[alpha_l_idx];
                float n_l = g_params[n_l_idx];
                float alpha_r = g_params[alpha_r_idx];
                float n_r = g_params[n_r_idx];

                float x_peak = clamp(mu, a, b);
                float logp_max = double_crystal_ball_logp_only(x_peak, mu, sigma, alpha_l, n_l, alpha_r, n_r, a, b);
                if (isfinite(logp_max)) {
                    for (uint it = 0; it < 1024u; it++) {
                        float xr = a + (b - a) * u01_open(&state);
                        float logp = double_crystal_ball_logp_only(xr, mu, sigma, alpha_l, n_l, alpha_r, n_r, a, b);
                        if (!isfinite(logp)) {
                            continue;
                        }
                        float lu = log(u01_open(&state));
                        if (lu <= (logp - logp_max)) {
                            x = xr;
                            break;
                        }
                    }
                }
            }
        } else if (proc.pdf_kind == PDF_CHEBYSHEV) {
            uint off = proc.shape_param_offset;
            uint order = proc.n_shape_params;
            if (order > 0u && off + order - 1u < s.total_shape_params) {
                float sum_abs = 1.0f;
                for (uint j = 0; j < order; j++) {
                    uint c_idx = g_shape_pidx[off + j];
                    float c = g_params[c_idx];
                    if (isfinite(c)) {
                        sum_abs += fabs(c);
                    }
                }
                if (isfinite(sum_abs) && sum_abs > 0.0f) {
                    for (uint it = 0; it < 4096u; it++) {
                        float xr = a + (b - a) * u01_open(&state);
                        float xp = chebyshev_xprime(xr, a, b);

                        float f0 = 1.0f;
                        float tkm1 = 1.0f;
                        float tk = xp;
                        for (uint j = 0; j < order; j++) {
                            uint k = j + 1u;
                            float tval = 0.0f;
                            if (k == 1u) {
                                tval = tk;
                            } else {
                                float tkp1 = 2.0f * xp * tk - tkm1;
                                tval = tkp1;
                                tkm1 = tk;
                                tk = tkp1;
                            }
                            uint c_idx = g_shape_pidx[off + j];
                            float c = g_params[c_idx];
                            f0 += c * tval;
                        }
                        if (!isfinite(f0) || !(f0 > 0.0f)) {
                            continue;
                        }
                        float uacc = u01_open(&state) * sum_abs;
                        if (uacc <= f0) {
                            x = xr;
                            break;
                        }
                    }
                }
            }
        } else if (proc.pdf_kind == PDF_HISTOGRAM) {
            uint off = proc.pdf_aux_offset;
            uint len = proc.pdf_aux_len;
            if (proc.n_shape_params == 0u && len >= 3u && off + len <= s.total_pdf_aux_f32) {
                uint n_bins = (len - 1u) / 2u;
                const device float* edges = g_pdf_aux_f32 + off;
                const device float* logdens = edges + (n_bins + 1u);

                float total = 0.0f;
                for (uint k = 0; k < n_bins; k++) {
                    float w = edges[k + 1u] - edges[k];
                    float ld = logdens[k];
                    if (!isfinite(w) || !(w > 0.0f) || !isfinite(ld)) {
                        continue;
                    }
                    total += exp(ld) * w;
                }
                if (isfinite(total) && total > 0.0f) {
                    float u_mass = u01_open(&state) * total;
                    float cum_mass = 0.0f;
                    uint chosen_bin = n_bins - 1u;
                    for (uint k = 0; k < n_bins; k++) {
                        float w = edges[k + 1u] - edges[k];
                        float ld = logdens[k];
                        if (!isfinite(w) || !(w > 0.0f) || !isfinite(ld)) {
                            continue;
                        }
                        cum_mass += exp(ld) * w;
                        if (u_mass <= cum_mass) {
                            chosen_bin = k;
                            break;
                        }
                    }
                    float lo = edges[chosen_bin];
                    float hi = edges[chosen_bin + 1u];
                    x = lo + (hi - lo) * u01_open(&state);
                }
            }
        }

        uint out_idx = start + i;
        g_obs_flat_out[out_idx] = x;
    }
}

kernel void unbinned_batch_nll_only(
    const device float* g_params_flat              [[buffer(0)]],  /* [n_toys × n_params] */
    const device float* g_obs_flat                 [[buffer(1)]],  /* [total_events] */
    const device uint* g_toy_offsets               [[buffer(2)]],  /* [n_toys + 1] */
    const device float* g_obs_lo                   [[buffer(3)]],  /* [1] */
    const device float* g_obs_hi                   [[buffer(4)]],  /* [1] */
    const device MetalUnbinnedProcessDesc* g_procs [[buffer(5)]],  /* [n_procs] */
    const device MetalUnbinnedRateModifierDesc* g_rate_mods [[buffer(6)]], /* [total_rate_mods] */
    const device uint* g_shape_pidx                [[buffer(7)]],  /* [total_shape_params] */
    const device float* g_pdf_aux_f32              [[buffer(8)]],  /* [total_pdf_aux_f32] */
    const device MetalUnbinnedGaussConstraintEntry* g_gauss [[buffer(9)]], /* [n_gauss] */
    device float* g_nll_out                        [[buffer(10)]], /* [n_toys] */
    constant BatchScalarArgs& args                 [[buffer(11)]],
    uint3 tid3                                     [[thread_position_in_threadgroup]],
    uint3 tptg                                     [[threads_per_threadgroup]],
    uint3 tg_pos                                   [[threadgroup_position_in_grid]],
    threadgroup float* shared                      [[threadgroup(0)]]
) {
    uint tid = tid3.x;
    uint block_size = tptg.x;
    uint toy = tg_pos.x + args.toy_offset;
    if (toy >= args.n_toys) {
        return;
    }

    uint n_params = args.n_params;
    uint n_procs = args.n_procs;
    uint total_rate_mods = args.total_rate_mods;
    uint total_shape_params = args.total_shape_params;
    uint n_gauss = args.n_gauss;
    float constraint_const = args.constraint_const;

    uint start = g_toy_offsets[toy];
    uint end = g_toy_offsets[toy + 1];
    if (end < start) {
        return;
    }
    uint n_events = end - start;

    threadgroup float* s_params = shared;
    threadgroup float* s_scratch = shared + n_params;
    threadgroup float* s_proc_nu = s_scratch + block_size;
    threadgroup float* s_proc_log_nu = s_proc_nu + BATCH_PROC_CACHE_CAP;
    bool use_proc_cache = (n_procs <= BATCH_PROC_CACHE_CAP);

    const device float* params = g_params_flat + toy * n_params;
    for (uint i = tid; i < n_params; i += block_size) {
        s_params[i] = params[i];
    }
    if (use_proc_cache) {
        for (uint p = tid; p < n_procs; p += block_size) {
            MetalUnbinnedProcessDesc proc = g_procs[p];
            float nu = 0.0f;
            if (proc.obs_index == 0u) {
                if (proc.yield_kind == YIELD_FIXED) {
                    nu = proc.base_yield;
                } else if (proc.yield_kind == YIELD_PARAMETER) {
                    nu = s_params[proc.yield_param_idx];
                } else if (proc.yield_kind == YIELD_SCALED) {
                    nu = proc.base_yield * s_params[proc.yield_param_idx];
                }
                uint mod_off = proc.rate_mod_offset;
                uint nmods = proc.n_rate_mods;
                if (mod_off + nmods > total_rate_mods) {
                    nmods = 0u;
                }
                for (uint m = 0; m < nmods; m++) {
                    nu *= rate_modifier_factor_only_tg(g_rate_mods, mod_off + m, s_params);
                }
            }
            if (isfinite(nu) && nu > 0.0f) {
                s_proc_nu[p] = nu;
                s_proc_log_nu[p] = log(nu);
            } else {
                s_proc_nu[p] = 0.0f;
                s_proc_log_nu[p] = -INFINITY;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum_logf = 0.0f;

    float a = g_obs_lo[0];
    float b = g_obs_hi[0];

    for (uint i = tid; i < n_events; i += block_size) {
        float x = g_obs_flat[start + i];

        float max_term = -INFINITY;
        float sum_exp = 0.0f;

        for (uint p = 0; p < n_procs; p++) {
            MetalUnbinnedProcessDesc proc = g_procs[p];
            if (proc.obs_index != 0u) {
                continue;
            }

            float nu = 0.0f;
            float log_nu = -INFINITY;
            if (use_proc_cache) {
                nu = s_proc_nu[p];
                log_nu = s_proc_log_nu[p];
                if (!isfinite(log_nu)) {
                    continue;
                }
            } else {
                if (proc.yield_kind == YIELD_FIXED) {
                    nu = proc.base_yield;
                } else if (proc.yield_kind == YIELD_PARAMETER) {
                    nu = s_params[proc.yield_param_idx];
                } else if (proc.yield_kind == YIELD_SCALED) {
                    nu = proc.base_yield * s_params[proc.yield_param_idx];
                } else {
                    continue;
                }
                uint mod_off = proc.rate_mod_offset;
                uint nmods = proc.n_rate_mods;
                if (mod_off + nmods > total_rate_mods) {
                    nmods = 0u;
                }
                for (uint m = 0; m < nmods; m++) {
                    nu *= rate_modifier_factor_only_tg(g_rate_mods, mod_off + m, s_params);
                }
                if (!(nu > 0.0f) || !isfinite(nu)) {
                    continue;
                }
                log_nu = log(nu);
            }

            float logp = -INFINITY;
            if (proc.pdf_kind == PDF_GAUSSIAN) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 2u || off + 1u >= total_shape_params) {
                    continue;
                }
                uint mu_idx = g_shape_pidx[off];
                uint sig_idx = g_shape_pidx[off + 1u];
                float mu = s_params[mu_idx];
                float sigma = s_params[sig_idx];
                logp = gaussian_logp_only(x, mu, sigma, a, b);
            } else if (proc.pdf_kind == PDF_EXPONENTIAL) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 1u || off >= total_shape_params) {
                    continue;
                }
                uint lam_idx = g_shape_pidx[off];
                float lambda = s_params[lam_idx];
                logp = exponential_logp_only(x, lambda, a, b);
            } else if (proc.pdf_kind == PDF_CRYSTAL_BALL) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 4u || off + 3u >= total_shape_params) {
                    continue;
                }
                uint mu_idx = g_shape_pidx[off];
                uint sig_idx = g_shape_pidx[off + 1u];
                uint alpha_idx = g_shape_pidx[off + 2u];
                uint n_idx = g_shape_pidx[off + 3u];
                float mu = s_params[mu_idx];
                float sigma = s_params[sig_idx];
                float alpha = s_params[alpha_idx];
                float nn = s_params[n_idx];
                logp = crystal_ball_logp_only(x, mu, sigma, alpha, nn, a, b);
            } else if (proc.pdf_kind == PDF_DOUBLE_CRYSTAL_BALL) {
                uint off = proc.shape_param_offset;
                if (proc.n_shape_params != 6u || off + 5u >= total_shape_params) {
                    continue;
                }
                uint mu_idx = g_shape_pidx[off];
                uint sig_idx = g_shape_pidx[off + 1u];
                uint alpha_l_idx = g_shape_pidx[off + 2u];
                uint n_l_idx = g_shape_pidx[off + 3u];
                uint alpha_r_idx = g_shape_pidx[off + 4u];
                uint n_r_idx = g_shape_pidx[off + 5u];
                float mu = s_params[mu_idx];
                float sigma = s_params[sig_idx];
                float alpha_l = s_params[alpha_l_idx];
                float n_l = s_params[n_l_idx];
                float alpha_r = s_params[alpha_r_idx];
                float n_r = s_params[n_r_idx];
                logp = double_crystal_ball_logp_only(x, mu, sigma, alpha_l, n_l, alpha_r, n_r, a, b);
            } else if (proc.pdf_kind == PDF_CHEBYSHEV) {
                uint off = proc.shape_param_offset;
                uint order = proc.n_shape_params;
                if (order == 0u || off + order - 1u >= total_shape_params) {
                    continue;
                }
                float w = b - a;
                if (!isfinite(w) || !(w > 0.0f)) {
                    continue;
                }
                float i0 = w;
                for (uint j = 0; j < order; j++) {
                    uint k = j + 1u;
                    if ((k & 1u) == 0u) {
                        uint c_idx = g_shape_pidx[off + j];
                        float c = s_params[c_idx];
                        float denom = 1.0f - (float)k * (float)k;
                        i0 += w * c / denom;
                    }
                }
                if (!isfinite(i0) || !(i0 > 0.0f)) {
                    continue;
                }
                float log_i = log(i0);
                float xp = chebyshev_xprime(x, a, b);
                float f0 = 1.0f;
                float tkm1 = 1.0f;
                float tk = xp;
                for (uint j = 0; j < order; j++) {
                    uint k = j + 1u;
                    float tval = 0.0f;
                    if (k == 1u) {
                        tval = tk;
                    } else {
                        float tkp1 = 2.0f * xp * tk - tkm1;
                        tval = tkp1;
                        tkm1 = tk;
                        tk = tkp1;
                    }
                    uint c_idx = g_shape_pidx[off + j];
                    float c = s_params[c_idx];
                    f0 += c * tval;
                }
                if (!isfinite(f0) || !(f0 > 0.0f)) {
                    continue;
                }
                logp = log(f0) - log_i;
            } else if (proc.pdf_kind == PDF_HISTOGRAM) {
                if (proc.n_shape_params != 0u) {
                    continue;
                }
                logp = histogram_logp_only(x, g_pdf_aux_f32, proc.pdf_aux_offset, proc.pdf_aux_len);
            } else {
                continue;
            }

            float term = log_nu + logp;
            if (!isfinite(term)) {
                continue;
            }

            if (term > max_term) {
                sum_exp = sum_exp * exp(max_term - term) + 1.0f;
                max_term = term;
            } else {
                sum_exp += exp(term - max_term);
            }
        }

        float logf = max_term + log(sum_exp);
        if (!isfinite(logf)) {
            logf = log(FLT_MIN);
        }
        local_sum_logf += logf;
    }

    s_scratch[tid] = local_sum_logf;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_scratch[tid] += s_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float sum_logf = s_scratch[0];

        float nu_tot = 0.0f;
        for (uint p = 0; p < n_procs; p++) {
            MetalUnbinnedProcessDesc proc = g_procs[p];
            float nu = 0.0f;
            if (use_proc_cache) {
                nu = s_proc_nu[p];
            } else {
                if (proc.yield_kind == YIELD_FIXED) {
                    nu = proc.base_yield;
                } else if (proc.yield_kind == YIELD_PARAMETER) {
                    nu = s_params[proc.yield_param_idx];
                } else if (proc.yield_kind == YIELD_SCALED) {
                    nu = proc.base_yield * s_params[proc.yield_param_idx];
                } else {
                    continue;
                }
                uint mod_off = proc.rate_mod_offset;
                uint nmods = proc.n_rate_mods;
                if (mod_off + nmods > total_rate_mods) {
                    nmods = 0u;
                }
                for (uint m = 0; m < nmods; m++) {
                    nu *= rate_modifier_factor_only_tg(g_rate_mods, mod_off + m, s_params);
                }
            }
            if (isfinite(nu) && nu > 0.0f) {
                nu_tot += nu;
            }
        }

        float nll = nu_tot - sum_logf;
        for (uint k = 0; k < n_gauss; k++) {
            MetalUnbinnedGaussConstraintEntry gc = g_gauss[k];
            float x0 = s_params[gc.param_idx];
            float diff = x0 - gc.center;
            float z = diff * gc.inv_width;
            nll += 0.5f * z * z;
        }

        nll += constraint_const;
        g_nll_out[toy] = nll;
    }
}
