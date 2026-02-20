/**
 * MAMS Leapfrog Metal Kernel — built-in models for LAPS on Apple Silicon.
 *
 * Architecture: 1 thread = 1 chain (scalar kernel), or
 *               32 threads (1 SIMD group) = 1 chain (simdgroup kernel).
 * All computation in float (f32) — Apple Silicon has no hardware f64.
 *
 * Precision: NO fast math. MAMS energy conservation requires precise
 * exp/log/sqrt for MH detailed balance.
 *
 * Models:
 *   0 = StdNormal
 *   1 = EightSchools (non-centered)
 *   2 = NealFunnel
 *   3 = GlmLogistic
 *   4 = NealFunnelNcp (non-centered parameterization)
 */

#include <metal_stdlib>
using namespace metal;

constant int MAX_DIM = 128;
constant int MAX_DIM_SIMD = 32;

/* ---------- Philox-4x32 inline RNG --------------------------------------- */

constant uint PHILOX_M0 = 0xD2511F53u;
constant uint PHILOX_M1 = 0xCD9E8D57u;
constant uint PHILOX_W0 = 0x9E3779B9u;
constant uint PHILOX_W1 = 0xBB67AE85u;

struct PhiloxState {
    uint counter[4];
    uint key[2];
    int draw_idx;
};

inline uint philox_mulhi(uint a, uint b) {
    return (uint)(((ulong)a * (ulong)b) >> 32);
}

inline void philox_round(thread uint* ctr, thread uint* key) {
    uint lo0 = ctr[0];
    uint lo1 = ctr[2];
    uint hi0 = philox_mulhi(PHILOX_M0, lo0);
    uint hi1 = philox_mulhi(PHILOX_M1, lo1);
    ctr[0] = hi1 ^ ctr[1] ^ key[0];
    ctr[1] = lo1;
    ctr[2] = hi0 ^ ctr[3] ^ key[1];
    ctr[3] = lo0;
    key[0] += PHILOX_W0;
    key[1] += PHILOX_W1;
}

inline void philox4x32(thread PhiloxState& s) {
    philox_round(s.counter, s.key);
    philox_round(s.counter, s.key);
    philox_round(s.counter, s.key);
    philox_round(s.counter, s.key);
}

inline void philox_init(thread PhiloxState& s, ulong seed, int chain) {
    s.counter[0] = (uint)(chain & 0xFFFFFFFF);
    s.counter[1] = (uint)((ulong)chain >> 32);
    s.counter[2] = 0;
    s.counter[3] = 0;
    s.key[0] = (uint)(seed & 0xFFFFFFFF);
    s.key[1] = (uint)(seed >> 32);
    s.draw_idx = 4; // force generate on first use
}

inline float philox_uniform(thread PhiloxState& s) {
    if (s.draw_idx >= 4) {
        s.counter[2]++;
        philox4x32(s);
        s.draw_idx = 0;
    }
    uint u = s.counter[s.draw_idx++];
    return float(u >> 1) * (1.0f / 2147483648.0f);
}

inline float philox_normal(thread PhiloxState& s) {
    float u1 = philox_uniform(s);
    float u2 = philox_uniform(s);
    float r = sqrt(-2.0f * log(max(u1, 1e-20f)));
    return r * cos(2.0f * 3.14159265f * u2);
}

/* ---------- Model functions (f32) ---------------------------------------- */

// Model 0: Standard normal in d dimensions.
// NLL = 0.5 * sum(x_i^2), grad = x_i
inline void grad_std_normal(
    thread const float* x, thread float* grad, int dim,
    device const float* /*model_data*/)
{
    for (int i = 0; i < dim; i++) {
        grad[i] = x[i];
    }
}

inline float nll_std_normal(
    thread const float* x, int dim,
    device const float* /*model_data*/)
{
    float nll = 0.0f;
    for (int i = 0; i < dim; i++) {
        nll += 0.5f * x[i] * x[i];
    }
    return nll;
}

// Model 1: Eight Schools (non-centered).
// params: [mu, tau, theta_raw_0..J-1]  (dim = 2+J)
// model_data: [J, y_0..y_{J-1}, inv_var_0..inv_var_{J-1}, prior_mu_sigma, prior_tau_scale]
inline void grad_eight_schools(
    thread const float* x, thread float* grad, int dim,
    device const float* model_data)
{
    int J = (int)model_data[0];
    float mu = x[0];
    float tau = x[1];
    float prior_mu_sigma = model_data[1 + 2 * J];
    float prior_tau_scale = model_data[1 + 2 * J + 1];

    float d_mu = 0.0f;
    float d_tau = 0.0f;

    for (int i = 0; i < J; i++) {
        float y_i = model_data[1 + i];
        float inv_var_i = model_data[1 + J + i];
        float theta_raw = x[2 + i];
        float theta = mu + tau * theta_raw;
        float r = y_i - theta;
        grad[2 + i] = -r * tau * inv_var_i + theta_raw;
        d_mu -= r * inv_var_i;
        d_tau -= r * theta_raw * inv_var_i;
    }

    d_mu += mu / (prior_mu_sigma * prior_mu_sigma);
    grad[0] = d_mu;

    float s2 = prior_tau_scale * prior_tau_scale;
    d_tau += 2.0f * tau / (s2 + tau * tau);
    grad[1] = d_tau;
}

inline float nll_eight_schools(
    thread const float* x, int dim,
    device const float* model_data)
{
    int J = (int)model_data[0];
    float mu = x[0];
    float tau = x[1];
    float prior_mu_sigma = model_data[1 + 2 * J];
    float prior_tau_scale = model_data[1 + 2 * J + 1];

    float nll = 0.0f;
    for (int i = 0; i < J; i++) {
        float y_i = model_data[1 + i];
        float inv_var_i = model_data[1 + J + i];
        float theta_raw = x[2 + i];
        float theta = mu + tau * theta_raw;
        float r = y_i - theta;
        nll += 0.5f * r * r * inv_var_i;
        nll += 0.5f * theta_raw * theta_raw;
    }
    nll += 0.5f * (mu / prior_mu_sigma) * (mu / prior_mu_sigma);
    float t = tau / prior_tau_scale;
    nll += log(1.0f + t * t);

    return nll;
}

// Model 2: Neal's funnel (dim = d).
// params: [v, x_1, ..., x_{d-1}]
// v ~ N(0,9), x_i|v ~ N(0, exp(v))
// NLL = v^2/18 + (d-1)/2 * v + 0.5 * exp(-v) * sum(x_i^2)
inline void grad_neal_funnel(
    thread const float* x, thread float* grad, int dim,
    device const float* /*model_data*/)
{
    float v = x[0];
    // Clamp v for f32 stability: exp(-v) with v in [-80,80]
    float v_clamped = clamp(v, -80.0f, 80.0f);
    float exp_neg_v = exp(-v_clamped);
    float sum_x2 = 0.0f;
    for (int i = 1; i < dim; i++) {
        sum_x2 += x[i] * x[i];
    }

    grad[0] = v / 9.0f + (dim - 1) * 0.5f - 0.5f * exp_neg_v * sum_x2;

    for (int i = 1; i < dim; i++) {
        grad[i] = exp_neg_v * x[i];
    }
}

inline float nll_neal_funnel(
    thread const float* x, int dim,
    device const float* /*model_data*/)
{
    float v = x[0];
    float v_clamped = clamp(v, -80.0f, 80.0f);
    float sum_x2 = 0.0f;
    for (int i = 1; i < dim; i++) {
        sum_x2 += x[i] * x[i];
    }
    return v * v / 18.0f + (dim - 1) * 0.5f * v + 0.5f * exp(-v_clamped) * sum_x2;
}

// Model 3: GLM logistic regression with N(0,1) prior on beta.
// params: [beta_0, ..., beta_{p-1}]
// model_data: [n, p, X_row(n*p), y(n), X_col(n*p)]
inline float sigmoid_f(float x) {
    if (x >= 0.0f) {
        float e = exp(-x);
        return 1.0f / (1.0f + e);
    } else {
        float e = exp(x);
        return e / (1.0f + e);
    }
}

inline void grad_glm_logistic(
    thread const float* beta, thread float* grad, int dim,
    device const float* model_data)
{
    int n = (int)model_data[0];
    int p = (int)model_data[1];
    device const float* X = model_data + 2;
    device const float* y = model_data + 2 + n * p;

    for (int j = 0; j < p; j++) {
        grad[j] = beta[j]; // prior gradient
    }

    for (int i = 0; i < n; i++) {
        float eta = 0.0f;
        for (int j = 0; j < p; j++) {
            eta += X[i * p + j] * beta[j];
        }
        float prob = sigmoid_f(eta);
        float diff = prob - y[i];
        for (int j = 0; j < p; j++) {
            grad[j] += diff * X[i * p + j];
        }
    }
}

inline float nll_glm_logistic(
    thread const float* beta, int dim,
    device const float* model_data)
{
    int n = (int)model_data[0];
    int p = (int)model_data[1];
    device const float* X = model_data + 2;
    device const float* y = model_data + 2 + n * p;

    float nll = 0.0f;
    for (int j = 0; j < p; j++) {
        nll += 0.5f * beta[j] * beta[j];
    }

    for (int i = 0; i < n; i++) {
        float eta = 0.0f;
        for (int j = 0; j < p; j++) {
            eta += X[i * p + j] * beta[j];
        }
        float abs_eta = abs(eta);
        nll += max(eta, 0.0f) + log(1.0f + exp(-abs_eta)) - y[i] * eta;
    }
    return nll;
}

// Model 4: NealFunnel — non-centered parameterization.
// params: [v, z_1, ..., z_{d-1}]
// Potential: U(v,z) = v^2/18 + sum(z_i^2)/2
// Transform back: x_i = z_i * exp(v/2)
inline void grad_neal_funnel_ncp(
    thread const float* x, thread float* grad, int dim,
    device const float* /*model_data*/)
{
    grad[0] = x[0] / 9.0f;
    for (int i = 1; i < dim; i++) {
        grad[i] = x[i];
    }
}

inline float nll_neal_funnel_ncp(
    thread const float* x, int dim,
    device const float* /*model_data*/)
{
    float nll = x[0] * x[0] / 18.0f;
    for (int i = 1; i < dim; i++) {
        nll += x[i] * x[i] * 0.5f;
    }
    return nll;
}

// Model 5: NealFunnel — Riemannian MAMS with Fisher metric preconditioning.
// G = diag(1/9, exp(-v), ..., exp(-v))
// NLL is IDENTICAL to the standard centered funnel (model_id=2):
//   NLL = v²/18 + (d-1)/2*v + exp(-v)*Σx²/2
// The Riemannian metric only affects the B-step and A-step (preconditioning),
// not the target distribution. MH compares the original NLL values.
inline void grad_neal_funnel_riemannian(
    thread const float* x, thread float* grad, int dim,
    device const float* /*model_data*/)
{
    float v = x[0];
    float v_clamped = clamp(v, -20.0f, 20.0f);
    float exp_neg_v = exp(-v_clamped);
    float sum_x2 = 0.0f;
    for (int i = 1; i < dim; i++) {
        sum_x2 += x[i] * x[i];
    }
    // Full NLL gradient (same as centered funnel) — correct target distribution
    grad[0] = v / 9.0f + (dim - 1) * 0.5f - 0.5f * exp_neg_v * sum_x2;
    for (int i = 1; i < dim; i++) {
        grad[i] = exp_neg_v * x[i];
    }
}

inline float nll_neal_funnel_riemannian(
    thread const float* x, int dim,
    device const float* /*model_data*/)
{
    float v = x[0];
    float v_clamped = clamp(v, -20.0f, 20.0f);
    float sum_x2 = 0.0f;
    for (int i = 1; i < dim; i++) {
        sum_x2 += x[i] * x[i];
    }
    // Full NLL (same as centered funnel) — Riemannian metric only affects dynamics
    return v * v / 18.0f + (dim - 1) * 0.5f * v + 0.5f * exp(-v_clamped) * sum_x2;
}

// Hybrid Riemannian B-step for Neal Funnel:
//   v (i=0): standard preconditioning via inv_mass[0] from Welford adaptation
//   x_i (i>0): position-dependent metric exp(v/2) matching funnel geometry
// This avoids the large constant gradient issue from (d-1)/2 in grad[0] while
// providing correct position-dependent scaling for x_i.
inline float b_step_riemannian_funnel(
    thread float* u, thread const float* grad, thread const float* x,
    device const float* inv_mass, float half_eps, int dim, float dm1)
{
    float v = x[0];
    float v_clamped = clamp(v, -20.0f, 20.0f);
    float ev2 = exp(v_clamped * 0.5f);

    // Hybrid preconditioned gradient:
    //   g_tilde[0] = sqrt(inv_mass[0]) * grad[0]  (standard)
    //   g_tilde[i] = exp(v/2) * grad[i]           (Riemannian)
    float g_norm_sq = 0.0f;
    float sqrt_m0 = sqrt(inv_mass[0]);
    float gt0 = sqrt_m0 * grad[0];
    g_norm_sq += gt0 * gt0;
    for (int i = 1; i < dim; i++) {
        float gti = ev2 * grad[i];
        g_norm_sq += gti * gti;
    }
    float g_norm = sqrt(g_norm_sq);
    if (g_norm < 1e-20f) return 0.0f;

    // e = g_tilde / |g_tilde|, compute e_dot_u
    float inv_g = 1.0f / g_norm;
    float e0 = gt0 * inv_g;
    float e_dot_u = e0 * u[0];
    for (int i = 1; i < dim; i++) {
        float ei = ev2 * grad[i] * inv_g;
        e_dot_u += ei * u[i];
    }
    e_dot_u = clamp(e_dot_u, -1.0f, 1.0f);

    float delta = half_eps * g_norm / dm1;
    float clamped_delta = min(delta, 40.0f);
    float zeta_m1 = exp(-2.0f * clamped_delta) - 1.0f;
    float c_u = 2.0f + zeta_m1;
    float c_e = -zeta_m1;

    // u_new = u * c_u - e * c_e, then renormalize
    float u_norm_sq = 0.0f;
    u[0] = u[0] * c_u - e0 * c_e;
    u_norm_sq += u[0] * u[0];
    for (int i = 1; i < dim; i++) {
        float ei = ev2 * grad[i] * inv_g;
        u[i] = u[i] * c_u - ei * c_e;
        u_norm_sq += u[i] * u[i];
    }
    float u_norm = sqrt(u_norm_sq);
    if (u_norm > 1e-12f) {
        float inv = 1.0f / u_norm;
        for (int i = 0; i < dim; i++) u[i] *= inv;
    } else {
        u[0] = -e0;
        for (int i = 1; i < dim; i++) {
            u[i] = -(ev2 * grad[i] * inv_g);
        }
    }

    float arg = 0.5f * zeta_m1 * (1.0f + e_dot_u);
    if (arg < -1.0f + 1e-20f) arg = -1.0f + 1e-20f;
    return dm1 * (delta + log(1.0f + arg));
}

// Hybrid Riemannian A-step for Neal Funnel:
//   v (i=0): standard scaling via inv_mass[0]
//   x_i (i>0): position-dependent exp(v/2), sub-stepped to track v changes
inline void a_step_riemannian_funnel(
    thread float* x, thread const float* u,
    device const float* inv_mass, float eps, int dim)
{
    float sqrt_m0 = sqrt(inv_mass[0]);
    constexpr int K_SUB = 4;
    float sub_eps = eps / float(K_SUB);
    for (int k = 0; k < K_SUB; k++) {
        float v = x[0];
        float v_clamped = clamp(v, -20.0f, 20.0f);
        float ev2 = exp(v_clamped * 0.5f);
        x[0] += sub_eps * sqrt_m0 * u[0];
        for (int i = 1; i < dim; i++) {
            x[i] += sub_eps * ev2 * u[i];
        }
    }
}

/* ---------- Model dispatch ----------------------------------------------- */

inline float model_nll(
    thread const float* x, int dim,
    device const float* model_data, int model_id)
{
    switch (model_id) {
        case 0: return nll_std_normal(x, dim, model_data);
        case 1: return nll_eight_schools(x, dim, model_data);
        case 2: return nll_neal_funnel(x, dim, model_data);
        case 3: return nll_glm_logistic(x, dim, model_data);
        case 4: return nll_neal_funnel_ncp(x, dim, model_data);
        case 5: return nll_neal_funnel_riemannian(x, dim, model_data);
        default: return 1e30f;
    }
}

inline void model_grad(
    thread const float* x, thread float* grad, int dim,
    device const float* model_data, int model_id)
{
    switch (model_id) {
        case 0: grad_std_normal(x, grad, dim, model_data); break;
        case 1: grad_eight_schools(x, grad, dim, model_data); break;
        case 2: grad_neal_funnel(x, grad, dim, model_data); break;
        case 3: grad_glm_logistic(x, grad, dim, model_data); break;
        case 4: grad_neal_funnel_ncp(x, grad, dim, model_data); break;
        case 5: grad_neal_funnel_riemannian(x, grad, dim, model_data); break;
    }
}

/* ---------- Integrator steps --------------------------------------------- */

// B-step: half-step velocity update on unit sphere.
// Returns delta_k contribution.
inline float b_step(
    thread float* u, thread const float* grad, device const float* inv_mass,
    float half_eps, int dim, float dm1)
{
    // Preconditioned gradient: g_tilde_i = sqrt(inv_mass_i) * grad_i
    float g_norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        float gi = sqrt(inv_mass[i]) * grad[i];
        g_norm_sq += gi * gi;
    }
    float g_norm = sqrt(g_norm_sq);
    if (g_norm < 1e-20f) return 0.0f;

    // e = g_tilde / |g_tilde|, compute e_dot_u
    float e_dot_u = 0.0f;
    for (int i = 0; i < dim; i++) {
        float ei = sqrt(inv_mass[i]) * grad[i] / g_norm;
        e_dot_u += ei * u[i];
    }
    e_dot_u = clamp(e_dot_u, -1.0f, 1.0f);

    float delta = half_eps * g_norm / dm1;

    // zeta_m1 = exp(-2*delta) - 1
    // Clamp delta for f32: avoid exp underflow at large delta
    float clamped_delta = min(delta, 40.0f);
    float zeta_m1 = exp(-2.0f * clamped_delta) - 1.0f;
    float c_u = 2.0f + zeta_m1;   // proportional to cosh(delta)
    float c_e = -zeta_m1;         // proportional to sinh(delta)

    // u_new = u * c_u - e * c_e, then renormalize
    float u_norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        float ei = sqrt(inv_mass[i]) * grad[i] / g_norm;
        u[i] = u[i] * c_u - ei * c_e;
        u_norm_sq += u[i] * u[i];
    }
    float u_norm = sqrt(u_norm_sq);
    if (u_norm > 1e-12f) {
        float inv = 1.0f / u_norm;
        for (int i = 0; i < dim; i++) u[i] *= inv;
    } else {
        for (int i = 0; i < dim; i++) {
            u[i] = -sqrt(inv_mass[i]) * grad[i] / g_norm;
        }
    }

    // delta_k = dm1 * (delta + ln(1 + 0.5 * zeta_m1 * (1 + e_dot_u)))
    float arg = 0.5f * zeta_m1 * (1.0f + e_dot_u);
    if (arg < -1.0f + 1e-20f) arg = -1.0f + 1e-20f;
    return dm1 * (delta + log(1.0f + arg));
}

// A-step: full-step position update.
inline void a_step(
    thread float* x, thread const float* u, device const float* inv_mass,
    float eps, int dim)
{
    for (int i = 0; i < dim; i++) {
        x[i] += eps * sqrt(inv_mass[i]) * u[i];
    }
}

/* ---------- Scalar args buffer ------------------------------------------- */

struct MamsArgs {
    float l;
    int max_leapfrog;
    int dim;
    int enable_mh;
    int model_id;
    uint seed_lo;
    uint seed_hi;
    int iteration;
    int n_chains;
    int store_idx;
    int n_report;
    int n_obs;
    int n_feat;
    int riemannian;
    float divergence_threshold;
};

/* ---------- Main kernel: mams_transition --------------------------------- */

kernel void mams_transition(
    // Per-chain state [K * dim]
    device float* g_x              [[buffer(0)]],
    device float* g_u              [[buffer(1)]],
    device float* g_potential      [[buffer(2)]],
    device float* g_grad           [[buffer(3)]],
    // Per-chain step sizes
    device const float* d_eps      [[buffer(4)]],
    // Inverse mass matrix
    device const float* inv_mass   [[buffer(5)]],
    // Model data
    device const float* model_data [[buffer(6)]],
    // Output
    device int* accepted           [[buffer(7)]],
    device float* energy_error     [[buffer(8)]],
    // Accumulation buffers
    device float* g_sample_buf     [[buffer(9)]],
    device float* g_accum_potential [[buffer(10)]],
    device int* g_accum_accepted   [[buffer(11)]],
    device float* g_accum_energy   [[buffer(12)]],
    // Scalar args
    constant MamsArgs& args        [[buffer(13)]],
    uint tid [[thread_position_in_grid]]
) {
    int chain = (int)tid;
    if (chain >= args.n_chains) return;

    int dim = args.dim;
    int model_id = args.model_id;
    float l = args.l;

    // Load per-chain step size
    float eps = d_eps[chain];
    int n_steps = (int)(l / eps + 0.5f);
    if (n_steps < 1) n_steps = 1;
    if (n_steps > args.max_leapfrog) n_steps = args.max_leapfrog;

    float dm1 = (float)(dim - 1);
    if (dm1 < 0.5f) dm1 = 1.0f;

    // Load chain state into thread-local arrays
    float x[MAX_DIM], u[MAX_DIM], grad[MAX_DIM];
    int off = chain * dim;
    for (int i = 0; i < dim; i++) {
        x[i] = g_x[off + i];
        u[i] = g_u[off + i];
        grad[i] = g_grad[off + i];
    }
    float potential = g_potential[chain];

    // Initialize Philox RNG
    ulong seed = ((ulong)args.seed_hi << 32) | (ulong)args.seed_lo;
    PhiloxState rng;
    philox_init(rng, seed ^ ((ulong)args.iteration * 0x9E3779B97F4A7C15ULL), chain);

    // ---------- 1. Partial velocity refresh (Gram-Schmidt) ----------
    float angle = eps / l;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    float z[MAX_DIM];
    float u_dot_z = 0.0f;
    for (int i = 0; i < dim; i++) {
        z[i] = philox_normal(rng);
        u_dot_z += u[i] * z[i];
    }
    float z_perp_norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        z[i] -= u_dot_z * u[i];
        z_perp_norm_sq += z[i] * z[i];
    }
    float z_perp_norm = sqrt(z_perp_norm_sq);
    if (z_perp_norm > 1e-12f) {
        float inv_norm = 1.0f / z_perp_norm;
        float u_new_norm_sq = 0.0f;
        for (int i = 0; i < dim; i++) {
            u[i] = u[i] * cos_a + z[i] * inv_norm * sin_a;
            u_new_norm_sq += u[i] * u[i];
        }
        float u_new_norm = sqrt(u_new_norm_sq);
        float inv = 1.0f / u_new_norm;
        for (int i = 0; i < dim; i++) u[i] *= inv;
    }

    // ---------- 2. Save pre-trajectory state for MH ----------
    float x_old[MAX_DIM], u_old[MAX_DIM], grad_old[MAX_DIM];
    float potential_old = potential;
    for (int i = 0; i < dim; i++) {
        x_old[i] = x[i];
        u_old[i] = u[i];
        grad_old[i] = grad[i];
    }

    // ---------- 3. Isokinetic leapfrog trajectory ----------
    float total_delta_k = 0.0f;
    int divergent = 0;
    int is_riemannian = args.riemannian;

    for (int s = 0; s < n_steps; s++) {
        if (is_riemannian && model_id == 5) {
            total_delta_k += b_step_riemannian_funnel(u, grad, x, inv_mass, eps * 0.5f, dim, dm1);
            a_step_riemannian_funnel(x, u, inv_mass, eps, dim);
        } else {
            total_delta_k += b_step(u, grad, inv_mass, eps * 0.5f, dim, dm1);
            a_step(x, u, inv_mass, eps, dim);
        }
        model_grad(x, grad, dim, model_data, model_id);
        potential = model_nll(x, dim, model_data, model_id);
        if (is_riemannian && model_id == 5) {
            total_delta_k += b_step_riemannian_funnel(u, grad, x, inv_mass, eps * 0.5f, dim, dm1);
        } else {
            total_delta_k += b_step(u, grad, inv_mass, eps * 0.5f, dim, dm1);
        }

        float current_w = (potential - potential_old) + total_delta_k;
        if (isfinite(potential_old) && (!isfinite(current_w) || current_w > args.divergence_threshold)) {
            divergent = 1;
            break;
        }
    }

    for (int i = 0; i < dim; i++) {
        if (!isfinite(x[i])) { divergent = 1; break; }
    }

    // ---------- 4. MH accept/reject ----------
    int acc = 0;
    float w = 0.0f;

    int first_call = !isfinite(potential_old);

    if (divergent && !first_call) {
        for (int i = 0; i < dim; i++) {
            x[i] = x_old[i];
            u[i] = -u_old[i];
            grad[i] = grad_old[i];
        }
        potential = potential_old;
        w = INFINITY;
    } else if (first_call) {
        acc = 1;
        w = 0.0f;
    } else {
        float delta_v = potential - potential_old;
        w = delta_v + total_delta_k;

        if (args.enable_mh) {
            // Clamp w for f32: exp(-w) underflows at w > 88
            float w_clamped = clamp(w, -80.0f, 80.0f);
            if (isfinite(w) && (w <= 0.0f || philox_uniform(rng) < exp(-w_clamped))) {
                acc = 1;
            } else {
                for (int i = 0; i < dim; i++) {
                    x[i] = x_old[i];
                    u[i] = -u_old[i];
                    grad[i] = grad_old[i];
                }
                potential = potential_old;
            }
        } else {
            acc = 1;
        }
    }

    // ---------- 5. Store state back ----------
    for (int i = 0; i < dim; i++) {
        g_x[off + i] = x[i];
        g_u[off + i] = u[i];
        g_grad[off + i] = grad[i];
    }
    g_potential[chain] = potential;
    accepted[chain] = acc;
    energy_error[chain] = w;

    // ---------- 6. Accumulate samples for batch download ----------
    if (args.n_report > 0 && chain < args.n_report) {
        int slot = args.store_idx * args.n_report + chain;
        g_accum_potential[slot] = potential;
        g_accum_accepted[slot] = acc;
        g_accum_energy[slot] = w;
        int pos_off = slot * dim;
        for (int i = 0; i < dim; i++) {
            g_sample_buf[pos_off + i] = x[i];
        }
    }
}

/* ---------- Fused multi-step kernel -------------------------------------- */

kernel void mams_transition_fused(
    device float* g_x              [[buffer(0)]],
    device float* g_u              [[buffer(1)]],
    device float* g_potential      [[buffer(2)]],
    device float* g_grad           [[buffer(3)]],
    device const float* d_eps      [[buffer(4)]],
    device const float* inv_mass   [[buffer(5)]],
    device const float* model_data [[buffer(6)]],
    device int* accepted_out       [[buffer(7)]],
    device float* energy_error_out [[buffer(8)]],
    device float* g_sample_buf     [[buffer(9)]],
    device float* g_accum_potential [[buffer(10)]],
    device int* g_accum_accepted   [[buffer(11)]],
    device float* g_accum_energy   [[buffer(12)]],
    constant MamsArgs& args        [[buffer(13)]],
    constant int& n_transitions    [[buffer(14)]],
    uint tid [[thread_position_in_grid]]
) {
    int chain = (int)tid;
    if (chain >= args.n_chains) return;

    int dim = args.dim;
    int model_id = args.model_id;
    float l = args.l;

    float eps = d_eps[chain];
    int n_steps = (int)(l / eps + 0.5f);
    if (n_steps < 1) n_steps = 1;
    if (n_steps > args.max_leapfrog) n_steps = args.max_leapfrog;

    float dm1 = (float)(dim - 1);
    if (dm1 < 0.5f) dm1 = 1.0f;

    // Load chain state into registers (ONE read for all transitions)
    float x[MAX_DIM], u[MAX_DIM], grad_reg[MAX_DIM];
    int off = chain * dim;
    for (int i = 0; i < dim; i++) {
        x[i] = g_x[off + i];
        u[i] = g_u[off + i];
        grad_reg[i] = g_grad[off + i];
    }
    float potential = g_potential[chain];

    int last_acc = 0;
    float last_w = 0.0f;

    ulong seed = ((ulong)args.seed_hi << 32) | (ulong)args.seed_lo;

    for (int t = 0; t < n_transitions; t++) {
        int iteration = args.iteration + t;

        PhiloxState rng;
        philox_init(rng, seed ^ ((ulong)iteration * 0x9E3779B97F4A7C15ULL), chain);

        // --- 1. Partial velocity refresh ---
        float angle = eps / l;
        float cos_a = cos(angle);
        float sin_a = sin(angle);

        float z[MAX_DIM];
        float u_dot_z = 0.0f;
        for (int i = 0; i < dim; i++) {
            z[i] = philox_normal(rng);
            u_dot_z += u[i] * z[i];
        }
        float z_perp_norm_sq = 0.0f;
        for (int i = 0; i < dim; i++) {
            z[i] -= u_dot_z * u[i];
            z_perp_norm_sq += z[i] * z[i];
        }
        float z_perp_norm = sqrt(z_perp_norm_sq);
        if (z_perp_norm > 1e-12f) {
            float inv_norm = 1.0f / z_perp_norm;
            float u_new_norm_sq = 0.0f;
            for (int i = 0; i < dim; i++) {
                u[i] = u[i] * cos_a + z[i] * inv_norm * sin_a;
                u_new_norm_sq += u[i] * u[i];
            }
            float u_new_norm = sqrt(u_new_norm_sq);
            float inv = 1.0f / u_new_norm;
            for (int i = 0; i < dim; i++) u[i] *= inv;
        }

        // --- 2. Save pre-trajectory state ---
        float x_old[MAX_DIM], u_old[MAX_DIM], grad_old[MAX_DIM];
        float potential_old = potential;
        for (int i = 0; i < dim; i++) {
            x_old[i] = x[i];
            u_old[i] = u[i];
            grad_old[i] = grad_reg[i];
        }

        // --- 3. Isokinetic leapfrog ---
        float total_delta_k = 0.0f;
        int divergent = 0;
        int is_riemannian = args.riemannian;

        for (int s = 0; s < n_steps; s++) {
            if (is_riemannian && model_id == 5) {
                total_delta_k += b_step_riemannian_funnel(u, grad_reg, x, inv_mass, eps * 0.5f, dim, dm1);
                a_step_riemannian_funnel(x, u, inv_mass, eps, dim);
            } else {
                total_delta_k += b_step(u, grad_reg, inv_mass, eps * 0.5f, dim, dm1);
                a_step(x, u, inv_mass, eps, dim);
            }
            model_grad(x, grad_reg, dim, model_data, model_id);
            potential = model_nll(x, dim, model_data, model_id);
            if (is_riemannian && model_id == 5) {
                total_delta_k += b_step_riemannian_funnel(u, grad_reg, x, inv_mass, eps * 0.5f, dim, dm1);
            } else {
                total_delta_k += b_step(u, grad_reg, inv_mass, eps * 0.5f, dim, dm1);
            }

            float current_w = (potential - potential_old) + total_delta_k;
            if (isfinite(potential_old) && (!isfinite(current_w) || current_w > args.divergence_threshold)) {
                divergent = 1;
                break;
            }
        }

        for (int i = 0; i < dim; i++) {
            if (!isfinite(x[i])) { divergent = 1; break; }
        }

        // --- 4. MH accept/reject ---
        int acc = 0;
        float w = 0.0f;
        int first_call = !isfinite(potential_old);

        if (divergent && !first_call) {
            for (int i = 0; i < dim; i++) {
                x[i] = x_old[i];
                u[i] = -u_old[i];
                grad_reg[i] = grad_old[i];
            }
            potential = potential_old;
            w = INFINITY;
        } else if (first_call) {
            acc = 1;
            w = 0.0f;
        } else {
            float delta_v = potential - potential_old;
            w = delta_v + total_delta_k;

            if (args.enable_mh) {
                float w_clamped = clamp(w, -80.0f, 80.0f);
                if (isfinite(w) && (w <= 0.0f || philox_uniform(rng) < exp(-w_clamped))) {
                    acc = 1;
                } else {
                    for (int i = 0; i < dim; i++) {
                        x[i] = x_old[i];
                        u[i] = -u_old[i];
                        grad_reg[i] = grad_old[i];
                    }
                    potential = potential_old;
                }
            } else {
                acc = 1;
            }
        }

        last_acc = acc;
        last_w = w;

        // --- 5. Accumulate sample ---
        if (args.n_report > 0 && chain < args.n_report) {
            int slot = t * args.n_report + chain;
            g_accum_potential[slot] = potential;
            g_accum_accepted[slot] = acc;
            g_accum_energy[slot] = w;
            int pos_off = slot * dim;
            for (int i = 0; i < dim; i++) {
                g_sample_buf[pos_off + i] = x[i];
            }
        }
    }

    // Store state back (ONE write for all transitions)
    for (int i = 0; i < dim; i++) {
        g_x[off + i] = x[i];
        g_u[off + i] = u[i];
        g_grad[off + i] = grad_reg[i];
    }
    g_potential[chain] = potential;
    accepted_out[chain] = last_acc;
    energy_error_out[chain] = last_w;
}

/* ---------- SIMD group kernel (1 SIMD group = 1 chain) ------------------- */
/*
 * For data-heavy models (GLM logistic: N observations).
 * 32 threads (1 SIMD group) cooperate on a single chain's grad/NLL.
 * Lanes distribute the observation loop; simd_sum reduces.
 *
 * Shared memory: X_col [p × n] + y [n] in threadgroup memory.
 * Falls back to global memory when n*p > 8000.
 */

// SIMD-cooperative GLM gradient + NLL (fused, threadgroup memory)
inline float grad_nll_glm_simd_tg(
    thread const float* beta, thread float* grad, int dim,
    threadgroup const float* s_X_col, threadgroup const float* s_y,
    int n, int p, uint lane_id)
{
    for (int j = 0; j < p; j++) grad[j] = 0.0f;
    float nll = 0.0f;

    for (int i = (int)lane_id; i < n; i += 32) {
        float eta = 0.0f;
        for (int j = 0; j < p; j++) {
            eta += s_X_col[j * n + i] * beta[j];
        }

        float prob = sigmoid_f(eta);
        float abs_eta = abs(eta);
        nll += max(eta, 0.0f) + log(1.0f + exp(-abs_eta)) - s_y[i] * eta;

        float diff = prob - s_y[i];
        for (int j = 0; j < p; j++) {
            grad[j] += diff * s_X_col[j * n + i];
        }
    }

    // SIMD group reduction + broadcast
    nll = simd_sum(nll);

    for (int j = 0; j < p; j++) {
        grad[j] = simd_sum(grad[j]) + beta[j]; // + prior
    }

    float prior = 0.0f;
    for (int j = 0; j < p; j++) {
        prior += 0.5f * beta[j] * beta[j];
    }
    return nll + prior;
}

// SIMD-cooperative GLM gradient + NLL — compile-time unrolled for known p.
// Metal has no C++ templates, but the compiler fully unrolls when P is a
// compile-time constant passed to an always-inlined function.

#define DEFINE_GRAD_NLL_GLM_SIMD_GLOBAL_P(P_VAL)                               \
inline float grad_nll_glm_simd_global_p##P_VAL(                                \
    thread const float* beta, thread float* grad,                              \
    device const float* X_col, device const float* y,                          \
    int n, uint lane_id)                                                       \
{                                                                              \
    for (int j = 0; j < P_VAL; j++) grad[j] = 0.0f;                           \
    float nll = 0.0f;                                                          \
                                                                               \
    for (int i = (int)lane_id; i < n; i += 32) {                               \
        float eta = 0.0f;                                                      \
        for (int j = 0; j < P_VAL; j++) {                                      \
            eta += X_col[j * n + i] * beta[j];                                \
        }                                                                      \
        float prob = sigmoid_f(eta);                                           \
        float abs_eta = abs(eta);                                              \
        nll += max(eta, 0.0f) + log(1.0f + exp(-abs_eta)) - y[i] * eta;       \
        float diff = prob - y[i];                                              \
        for (int j = 0; j < P_VAL; j++) {                                      \
            grad[j] += diff * X_col[j * n + i];                               \
        }                                                                      \
    }                                                                          \
                                                                               \
    nll = simd_sum(nll);                                                       \
    for (int j = 0; j < P_VAL; j++) {                                          \
        grad[j] = simd_sum(grad[j]) + beta[j];                                \
    }                                                                          \
    float prior = 0.0f;                                                        \
    for (int j = 0; j < P_VAL; j++) {                                          \
        prior += 0.5f * beta[j] * beta[j];                                    \
    }                                                                          \
    return nll + prior;                                                        \
}

DEFINE_GRAD_NLL_GLM_SIMD_GLOBAL_P(6)
DEFINE_GRAD_NLL_GLM_SIMD_GLOBAL_P(20)

// SIMD-cooperative GLM gradient + NLL (fused, global memory)
// Dispatches to compile-time-unrolled specializations for p=6, p=20.
inline float grad_nll_glm_simd_global(
    thread const float* beta, thread float* grad, int dim,
    device const float* model_data, int n, int p, uint lane_id)
{
    device const float* X_col = model_data + 2 + n * p + n;
    device const float* y = model_data + 2 + n * p;

    switch (p) {
        case 6:
            return grad_nll_glm_simd_global_p6(beta, grad, X_col, y, n, lane_id);
        case 20:
            return grad_nll_glm_simd_global_p20(beta, grad, X_col, y, n, lane_id);
        default:
            break;
    }

    // Generic fallback for other p values
    for (int j = 0; j < p; j++) grad[j] = 0.0f;
    float nll = 0.0f;

    for (int i = (int)lane_id; i < n; i += 32) {
        float eta = 0.0f;
        for (int j = 0; j < p; j++) {
            eta += X_col[j * n + i] * beta[j];
        }

        float prob = sigmoid_f(eta);
        float abs_eta = abs(eta);
        nll += max(eta, 0.0f) + log(1.0f + exp(-abs_eta)) - y[i] * eta;

        float diff = prob - y[i];
        for (int j = 0; j < p; j++) {
            grad[j] += diff * X_col[j * n + i];
        }
    }

    nll = simd_sum(nll);

    for (int j = 0; j < p; j++) {
        grad[j] = simd_sum(grad[j]) + beta[j];
    }

    float prior = 0.0f;
    for (int j = 0; j < p; j++) {
        prior += 0.5f * beta[j] * beta[j];
    }
    return nll + prior;
}

kernel void mams_transition_simdgroup(
    device float* g_x              [[buffer(0)]],
    device float* g_u              [[buffer(1)]],
    device float* g_potential      [[buffer(2)]],
    device float* g_grad           [[buffer(3)]],
    device const float* d_eps      [[buffer(4)]],
    device const float* inv_mass   [[buffer(5)]],
    device const float* model_data [[buffer(6)]],
    device int* accepted           [[buffer(7)]],
    device float* energy_error     [[buffer(8)]],
    device float* g_sample_buf     [[buffer(9)]],
    device float* g_accum_potential [[buffer(10)]],
    device int* g_accum_accepted   [[buffer(11)]],
    device float* g_accum_energy   [[buffer(12)]],
    constant MamsArgs& args        [[buffer(13)]],
    threadgroup float* tg_mem      [[threadgroup(0)]],
    uint tid [[thread_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    int chain = (int)(tid / 32);
    if (chain >= args.n_chains) return;

    int dim = args.dim;
    int model_id = args.model_id;
    float l = args.l;
    int n_obs = args.n_obs;
    int n_feat = args.n_feat;

    // Cooperative load of model data into threadgroup memory (GLM only)
    bool use_tg = (model_id == 3 && n_obs > 0 && n_feat > 0 &&
                   n_obs * n_feat <= 8000);
    threadgroup float* s_X_col = tg_mem;
    threadgroup float* s_y = tg_mem + n_obs * n_feat;

    if (use_tg) {
        device const float* X_col = model_data + 2 + n_obs * n_feat + n_obs;
        device const float* y_src = model_data + 2 + n_obs * n_feat;

        int n_total = n_obs * n_feat;
        // Use all threads in the SIMD group for cooperative load
        for (int idx = (int)lane_id; idx < n_total; idx += 32) {
            s_X_col[idx] = X_col[idx];
        }
        for (int idx = (int)lane_id; idx < n_obs; idx += 32) {
            s_y[idx] = y_src[idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Load per-chain step size (all lanes load same chain)
    float eps = d_eps[chain];
    int n_steps = (int)(l / eps + 0.5f);
    if (n_steps < 1) n_steps = 1;
    if (n_steps > args.max_leapfrog) n_steps = args.max_leapfrog;

    float dm1 = (float)(dim - 1);
    if (dm1 < 0.5f) dm1 = 1.0f;

    // All lanes load the same chain state (replicated in registers)
    float x[MAX_DIM_SIMD], u[MAX_DIM_SIMD], grad_reg[MAX_DIM_SIMD];
    int off = chain * dim;
    for (int i = 0; i < dim; i++) {
        x[i] = g_x[off + i];
        u[i] = g_u[off + i];
        grad_reg[i] = g_grad[off + i];
    }
    float potential = g_potential[chain];

    // All lanes get identical Philox state
    ulong seed = ((ulong)args.seed_hi << 32) | (ulong)args.seed_lo;
    PhiloxState rng;
    philox_init(rng, seed ^ ((ulong)args.iteration * 0x9E3779B97F4A7C15ULL), chain);

    // ---------- 1. Partial velocity refresh (identical across lanes) ----------
    float angle = eps / l;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    float z[MAX_DIM_SIMD];
    float u_dot_z = 0.0f;
    for (int i = 0; i < dim; i++) {
        z[i] = philox_normal(rng);
        u_dot_z += u[i] * z[i];
    }
    float z_perp_norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        z[i] -= u_dot_z * u[i];
        z_perp_norm_sq += z[i] * z[i];
    }
    float z_perp_norm = sqrt(z_perp_norm_sq);
    if (z_perp_norm > 1e-12f) {
        float inv_norm = 1.0f / z_perp_norm;
        float u_new_norm_sq = 0.0f;
        for (int i = 0; i < dim; i++) {
            u[i] = u[i] * cos_a + z[i] * inv_norm * sin_a;
            u_new_norm_sq += u[i] * u[i];
        }
        float u_new_norm = sqrt(u_new_norm_sq);
        float inv = 1.0f / u_new_norm;
        for (int i = 0; i < dim; i++) u[i] *= inv;
    }

    // ---------- 2. Save pre-trajectory state ----------
    float x_old[MAX_DIM_SIMD], u_old[MAX_DIM_SIMD], grad_old[MAX_DIM_SIMD];
    float potential_old = potential;
    for (int i = 0; i < dim; i++) {
        x_old[i] = x[i];
        u_old[i] = u[i];
        grad_old[i] = grad_reg[i];
    }

    // ---------- 3. Isokinetic leapfrog (SIMD-cooperative grad/nll) ----------
    float total_delta_k = 0.0f;
    int divergent = 0;

    for (int s = 0; s < n_steps; s++) {
        // b_step: identical across all lanes (deterministic, O(dim))
        total_delta_k += b_step(u, grad_reg, inv_mass, eps * 0.5f, dim, dm1);
        a_step(x, u, inv_mass, eps, dim);

        // SIMD-cooperative grad + NLL
        if (model_id == 3) {
            if (use_tg) {
                potential = grad_nll_glm_simd_tg(
                    x, grad_reg, dim, s_X_col, s_y, n_obs, n_feat, lane_id);
            } else {
                potential = grad_nll_glm_simd_global(
                    x, grad_reg, dim, model_data, n_obs, n_feat, lane_id);
            }
        } else {
            model_grad(x, grad_reg, dim, model_data, model_id);
            potential = model_nll(x, dim, model_data, model_id);
        }

        total_delta_k += b_step(u, grad_reg, inv_mass, eps * 0.5f, dim, dm1);

        float current_w = (potential - potential_old) + total_delta_k;
        if (isfinite(potential_old) && (!isfinite(current_w) || current_w > 1000.0f)) {
            divergent = 1;
            break;
        }
    }

    for (int i = 0; i < dim; i++) {
        if (!isfinite(x[i])) { divergent = 1; break; }
    }

    // ---------- 4. MH accept/reject (RNG identical across lanes) ----------
    int acc = 0;
    float w = 0.0f;
    int first_call = !isfinite(potential_old);

    if (divergent && !first_call) {
        for (int i = 0; i < dim; i++) {
            x[i] = x_old[i];
            u[i] = -u_old[i];
            grad_reg[i] = grad_old[i];
        }
        potential = potential_old;
        w = INFINITY;
    } else if (first_call) {
        acc = 1;
        w = 0.0f;
    } else {
        float delta_v = potential - potential_old;
        w = delta_v + total_delta_k;

        if (args.enable_mh) {
            float w_clamped = clamp(w, -80.0f, 80.0f);
            if (isfinite(w) && (w <= 0.0f || philox_uniform(rng) < exp(-w_clamped))) {
                acc = 1;
            } else {
                for (int i = 0; i < dim; i++) {
                    x[i] = x_old[i];
                    u[i] = -u_old[i];
                    grad_reg[i] = grad_old[i];
                }
                potential = potential_old;
            }
        } else {
            acc = 1;
        }
    }

    // ---------- 5. Store state back (ONLY lane 0 writes) ----------
    if (lane_id == 0) {
        for (int i = 0; i < dim; i++) {
            g_x[off + i] = x[i];
            g_u[off + i] = u[i];
            g_grad[off + i] = grad_reg[i];
        }
        g_potential[chain] = potential;
        accepted[chain] = acc;
        energy_error[chain] = w;

        if (args.n_report > 0 && chain < args.n_report) {
            int slot = args.store_idx * args.n_report + chain;
            g_accum_potential[slot] = potential;
            g_accum_accepted[slot] = acc;
            g_accum_energy[slot] = w;
            int pos_off = slot * dim;
            for (int i = 0; i < dim; i++) {
                g_sample_buf[pos_off + i] = x[i];
            }
        }
    }
}
