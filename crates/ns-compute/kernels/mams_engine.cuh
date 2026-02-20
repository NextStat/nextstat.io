/**
 * MAMS Engine Header — model-independent code for LAPS GPU sampler.
 *
 * This header contains:
 *   - Philox-4x32 RNG (inline, no cuRAND dependency)
 *   - Box-Muller normal sampling
 *   - B-step (velocity half-step on unit sphere)
 *   - A-step (position full-step with preconditioning)
 *   - mams_transition kernel
 *
 * CONTRACT: Before including this header, define:
 *   __device__ double user_nll(const double* x, int dim, const double* model_data);
 *   __device__ void   user_grad(const double* x, double* grad, int dim, const double* model_data);
 *
 * Optional fast path:
 *   #define MAMS_FUSED_GRAD_NLL_DEFINED
 *   __device__ double user_grad_nll(const double* x, double* grad, int dim, const double* model_data);
 *
 * The kernel calls user_nll() and user_grad() for model evaluation.
 * Build-time (nvcc) and JIT (NVRTC) paths both use this header.
 *
 * For the build-time path with multiple models, user_nll/user_grad can
 * dispatch on __mams_model_id (a __device__ variable set by the kernel).
 * JIT models define user_nll/user_grad directly, ignoring __mams_model_id.
 */

#ifndef MAMS_ENGINE_CUH
#define MAMS_ENGINE_CUH

#define MAX_DIM 128

/* __device__ variable for model dispatch (build-time multi-model path).
 * Set by mams_transition at kernel entry from the model_id argument.
 * JIT user models ignore this — they define user_nll/user_grad directly.
 *
 * If the including TU already defines __mams_model_id (via MAMS_MODEL_ID_DEFINED),
 * we skip the declaration here to avoid redefinition. */
#ifndef MAMS_MODEL_ID_DEFINED
static __device__ int __mams_model_id;
#endif

/* ---------- Philox-4x32 inline RNG --------------------------------------- */

#define PHILOX_M0 0xD2511F53u
#define PHILOX_M1 0xCD9E8D57u
#define PHILOX_W0 0x9E3779B9u
#define PHILOX_W1 0xBB67AE85u

struct PhiloxState {
    unsigned int counter[4];
    unsigned int key[2];
    int draw_idx;   // which of the 4 u32s we consumed
};

__device__ __forceinline__ void philox_round(unsigned int* ctr, unsigned int* key) {
    unsigned int lo0 = ctr[0], hi0;
    unsigned int lo1 = ctr[2], hi1;
    hi0 = __umulhi(PHILOX_M0, lo0);
    hi1 = __umulhi(PHILOX_M1, lo1);
    ctr[0] = hi1 ^ ctr[1] ^ key[0];
    ctr[1] = lo1;
    ctr[2] = hi0 ^ ctr[3] ^ key[1];
    ctr[3] = lo0;
    key[0] += PHILOX_W0;
    key[1] += PHILOX_W1;
}

__device__ __forceinline__ void philox4x32(PhiloxState* s) {
    philox_round(s->counter, s->key);
    philox_round(s->counter, s->key);
    philox_round(s->counter, s->key);
    philox_round(s->counter, s->key);
}

__device__ __forceinline__ void philox_init(PhiloxState* s, unsigned long long seed, int chain) {
    s->counter[0] = (unsigned int)(chain & 0xFFFFFFFF);
    s->counter[1] = (unsigned int)((unsigned long long)chain >> 32);
    s->counter[2] = 0;
    s->counter[3] = 0;
    s->key[0] = (unsigned int)(seed & 0xFFFFFFFF);
    s->key[1] = (unsigned int)(seed >> 32);
    s->draw_idx = 4; // force generate on first use
}

__device__ __forceinline__ double philox_uniform(PhiloxState* s) {
    if (s->draw_idx >= 4) {
        // Bump sub-counter and regenerate
        s->counter[2]++;
        // Reset key for determinism
        philox4x32(s);
        s->draw_idx = 0;
    }
    unsigned int u = s->counter[s->draw_idx++];
    return (double)(u >> 1) * (1.0 / 2147483648.0);
}

__device__ __forceinline__ double philox_normal(PhiloxState* s) {
    double u1 = philox_uniform(s);
    double u2 = philox_uniform(s);
    double r = sqrt(-2.0 * log(fmax(u1, 1e-30)));
    return r * cos(2.0 * 3.14159265358979323846 * u2);
}

// Funnel geometries need stronger momentum refresh than angle=eps/l alone.
// This improves cross-energy mixing (E-BFMI) in short-budget runs.
__device__ __forceinline__ int mams_is_funnel_model(int model_id) {
    return (model_id == 2) || (model_id == 4) || (model_id == 5);
}

__device__ __forceinline__ double mams_refresh_angle(double eps, double l, int model_id) {
    double angle = eps / l;
    if (model_id == 0) {
        // StdNormal benefits from stronger refresh to reduce long-range
        // autocorrelation in short-budget massively parallel runs.
        const double kStdNormalAngleFloor = 0.35;
        if (angle < kStdNormalAngleFloor) angle = kStdNormalAngleFloor;
    }
    if (mams_is_funnel_model(model_id)) {
        // Empirical floor for funnel-like geometries: avoid near-identity refresh.
        const double kFunnelAngleFloor = 0.20;
        if (angle < kFunnelAngleFloor) angle = kFunnelAngleFloor;
    }
    return angle;
}

__device__ __forceinline__ void mams_partial_refresh(
    double* u,
    int dim,
    double cos_a,
    double sin_a,
    PhiloxState* rng,
    double* z_buf)
{
    double u_dot_z = 0.0;
    for (int i = 0; i < dim; i++) {
        z_buf[i] = philox_normal(rng);
        u_dot_z += u[i] * z_buf[i];
    }

    double z_perp_norm_sq = 0.0;
    for (int i = 0; i < dim; i++) {
        z_buf[i] -= u_dot_z * u[i];
        z_perp_norm_sq += z_buf[i] * z_buf[i];
    }

    double z_perp_norm = sqrt(z_perp_norm_sq);
    if (z_perp_norm <= 1e-12) return;

    double inv_norm = 1.0 / z_perp_norm;
    double u_new_norm_sq = 0.0;
    for (int i = 0; i < dim; i++) {
        u[i] = u[i] * cos_a + z_buf[i] * inv_norm * sin_a;
        u_new_norm_sq += u[i] * u[i];
    }

    double u_new_norm = sqrt(u_new_norm_sq);
    if (u_new_norm <= 1e-12) return;

    double inv = 1.0 / u_new_norm;
    for (int i = 0; i < dim; i++) {
        u[i] *= inv;
    }
}

/* ---------- Integrator steps --------------------------------------------- */

// B-step with precomputed sqrt(inv_mass): half-step velocity update on unit sphere.
// Returns delta_k contribution.
__device__ __forceinline__ double b_step_precond(
    double* u, const double* grad, const double* sqrt_inv_mass,
    double half_eps, int dim, double dm1)
{
    // Preconditioned gradient: g_tilde_i = sqrt(inv_mass_i) * grad_i
    double g_norm_sq = 0.0;
    for (int i = 0; i < dim; i++) {
        double gi = sqrt_inv_mass[i] * grad[i];
        g_norm_sq += gi * gi;
    }
    double g_norm = sqrt(g_norm_sq);
    if (g_norm < 1e-30) return 0.0;

    // e = g_tilde / |g_tilde|, compute e_dot_u
    double e_dot_u = 0.0;
    for (int i = 0; i < dim; i++) {
        double ei = sqrt_inv_mass[i] * grad[i] / g_norm;
        e_dot_u += ei * u[i];
    }
    if (e_dot_u > 1.0) e_dot_u = 1.0;
    if (e_dot_u < -1.0) e_dot_u = -1.0;

    double delta = half_eps * g_norm / dm1;

    // zeta_m1 = exp(-2*delta) - 1
    double zeta_m1 = expm1(-2.0 * delta);
    double c_u = 2.0 + zeta_m1;   // proportional to cosh(delta)
    double c_e = -zeta_m1;         // proportional to sinh(delta)

    // u_new = u * c_u - e * c_e, then renormalize
    double u_norm_sq = 0.0;
    for (int i = 0; i < dim; i++) {
        double ei = sqrt_inv_mass[i] * grad[i] / g_norm;
        u[i] = u[i] * c_u - ei * c_e;
        u_norm_sq += u[i] * u[i];
    }
    double u_norm = sqrt(u_norm_sq);
    if (u_norm > 1e-12) {
        double inv = 1.0 / u_norm;
        for (int i = 0; i < dim; i++) u[i] *= inv;
    } else {
        for (int i = 0; i < dim; i++) {
            u[i] = -sqrt_inv_mass[i] * grad[i] / g_norm;
        }
    }

    // delta_k = dm1 * (delta + ln(1 + 0.5 * zeta_m1 * (1 + e_dot_u)))
    double arg = 0.5 * zeta_m1 * (1.0 + e_dot_u);
    if (arg < -1.0 + 1e-50) arg = -1.0 + 1e-50;
    return dm1 * (delta + log1p(arg));
}

// Backward-compatible wrapper when precomputed sqrt(inv_mass) is unavailable.
__device__ __forceinline__ double b_step(
    double* u, const double* grad, const double* inv_mass,
    double half_eps, int dim, double dm1)
{
    double sqrt_inv_mass[MAX_DIM];
    for (int i = 0; i < dim; i++) {
        sqrt_inv_mass[i] = sqrt(inv_mass[i]);
    }
    return b_step_precond(u, grad, sqrt_inv_mass, half_eps, dim, dm1);
}

// A-step: full-step position update.
__device__ void a_step(
    double* x, const double* u, const double* inv_mass,
    double eps, int dim)
{
    for (int i = 0; i < dim; i++) {
        x[i] += eps * sqrt(inv_mass[i]) * u[i];
    }
}

__device__ __forceinline__ double mams_eval_grad_nll(
    const double* x, double* grad, int dim, const double* model_data)
{
#ifdef MAMS_FUSED_GRAD_NLL_DEFINED
    return user_grad_nll(x, grad, dim, model_data);
#else
    user_grad(x, grad, dim, model_data);
    return user_nll(x, dim, model_data);
#endif
}

/* ---------- Riemannian steps for Neal Funnel (model_id=5) --------------- */
/*
 * Position-dependent metric: G = diag(1/9, exp(-v), ..., exp(-v))
 * sqrt(G^{-1}) = diag(3, exp(v/2), ..., exp(v/2))
 *
 * These replace b_step/a_step when model_id==5 and riemannian flag is set.
 */

// Riemannian B-step for Neal Funnel.
// Hybrid Riemannian B-step for Neal Funnel:
//   v (i=0): standard preconditioning via inv_mass[0] from Welford adaptation
//   x_i (i>0): position-dependent metric exp(v/2) matching funnel geometry
__device__ double b_step_riemannian_funnel(
    double* u, const double* grad, const double* x,
    const double* inv_mass, double half_eps, int dim, double dm1)
{
    double v = x[0];
    double v_clamped = fmax(fmin(v, 20.0), -20.0);
    double ev2 = exp(v_clamped * 0.5);

    // Hybrid preconditioned gradient:
    //   g_tilde[0] = sqrt(inv_mass[0]) * grad[0]  (standard)
    //   g_tilde[i] = exp(v/2) * grad[i]           (Riemannian)
    double g_norm_sq = 0.0;
    double sqrt_ginv[MAX_DIM];
    sqrt_ginv[0] = sqrt(inv_mass[0]);
    for (int i = 1; i < dim; i++) sqrt_ginv[i] = ev2;

    for (int i = 0; i < dim; i++) {
        double gi = sqrt_ginv[i] * grad[i];
        g_norm_sq += gi * gi;
    }
    double g_norm = sqrt(g_norm_sq);
    if (g_norm < 1e-30) return 0.0;

    double e_dot_u = 0.0;
    for (int i = 0; i < dim; i++) {
        double ei = sqrt_ginv[i] * grad[i] / g_norm;
        e_dot_u += ei * u[i];
    }
    if (e_dot_u > 1.0) e_dot_u = 1.0;
    if (e_dot_u < -1.0) e_dot_u = -1.0;

    double delta = half_eps * g_norm / dm1;
    double zeta_m1 = expm1(-2.0 * delta);
    double c_u = 2.0 + zeta_m1;
    double c_e = -zeta_m1;

    double u_norm_sq = 0.0;
    for (int i = 0; i < dim; i++) {
        double ei = sqrt_ginv[i] * grad[i] / g_norm;
        u[i] = u[i] * c_u - ei * c_e;
        u_norm_sq += u[i] * u[i];
    }
    double u_norm = sqrt(u_norm_sq);
    if (u_norm > 1e-12) {
        double inv = 1.0 / u_norm;
        for (int i = 0; i < dim; i++) u[i] *= inv;
    } else {
        for (int i = 0; i < dim; i++) {
            u[i] = -sqrt_ginv[i] * grad[i] / g_norm;
        }
    }

    double arg = 0.5 * zeta_m1 * (1.0 + e_dot_u);
    if (arg < -1.0 + 1e-50) arg = -1.0 + 1e-50;
    return dm1 * (delta + log1p(arg));
}

// Hybrid Riemannian A-step for Neal Funnel:
//   v (i=0): standard scaling via inv_mass[0]
//   x_i (i>0): position-dependent exp(v/2), sub-stepped to track v changes
__device__ void a_step_riemannian_funnel(
    double* x, const double* u, const double* inv_mass, double eps, int dim)
{
    double sqrt_m0 = sqrt(inv_mass[0]);
    const int K_SUB = 4;
    double sub_eps = eps / (double)K_SUB;
    for (int k = 0; k < K_SUB; k++) {
        double v = x[0];
        double v_clamped = fmax(fmin(v, 20.0), -20.0);
        double ev2 = exp(v_clamped * 0.5);
        x[0] += sub_eps * sqrt_m0 * u[0];
        for (int i = 1; i < dim; i++) {
            x[i] += sub_eps * ev2 * u[i];
        }
    }
}

/* ---------- Main kernel -------------------------------------------------- */

extern "C" __global__ void mams_transition(
    // Per-chain state [K * dim] — persistent across iterations
    double* __restrict__ g_x,           // positions
    double* __restrict__ g_u,           // unit velocities
    double* __restrict__ g_potential,    // [K] potentials
    double* __restrict__ g_grad,        // [K * dim] gradients
    // Sampler params — per-chain step size
    const double* __restrict__ d_eps,   // [K] per-chain step sizes
    double l,
    int max_leapfrog,                   // cap on per-chain n_steps
    int dim,
    const double* __restrict__ inv_mass,   // [dim]
    int enable_mh,
    // Model data
    const double* __restrict__ model_data,
    int model_id,
    // RNG
    unsigned long long seed,
    int iteration,        // bumped each call for fresh RNG draws
    // Chain count
    int n_chains,
    // Output (per-chain, overwritten each call)
    int* __restrict__ accepted,        // [K]
    double* __restrict__ energy_error, // [K]
    // Accumulation buffers for batch download (pass n_report=0 to disable)
    double* __restrict__ g_sample_buf,      // [batch_stride * n_report * dim]
    double* __restrict__ g_accum_potential,  // [batch_stride * n_report]
    int*    __restrict__ g_accum_accepted,   // [batch_stride * n_report]
    double* __restrict__ g_accum_energy,     // [batch_stride * n_report]
    int store_idx,                           // slot in ring buffer (0..batch_stride-1)
    int n_report,                            // chains to accumulate (0 = disabled)
    double divergence_threshold              // energy error threshold for divergence detection
) {
    int chain = blockIdx.x * blockDim.x + threadIdx.x;
    if (chain >= n_chains) return;

    // Publish model_id so user_nll/user_grad dispatch can read it (build-time path)
    __mams_model_id = model_id;

    // Load per-chain step size
    double eps = d_eps[chain];
    int n_steps = (int)round(l / eps);
    if (n_steps < 1) n_steps = 1;
    if (n_steps > max_leapfrog) n_steps = max_leapfrog;

    double dm1 = (double)(dim - 1);
    if (dm1 < 0.5) dm1 = 1.0;  // safety for dim=1

    // Load chain state into registers
    double x[MAX_DIM], u[MAX_DIM], grad[MAX_DIM];
    size_t off = (size_t)chain * dim;
    for (int i = 0; i < dim; i++) {
        x[i] = g_x[off + i];
        u[i] = g_u[off + i];
        grad[i] = g_grad[off + i];
    }
    double potential = g_potential[chain];

    // Initialize Philox RNG with (seed, chain, iteration) for uniqueness
    PhiloxState rng;
    philox_init(&rng, seed ^ ((unsigned long long)iteration * 0x9E3779B97F4A7C15ULL), chain);

    // ---------- 1. Partial velocity refresh (Gram-Schmidt) ----------
    double angle = mams_refresh_angle(eps, l, model_id);
    double cos_a = cos(angle);
    double sin_a = sin(angle);

    double z[MAX_DIM];
    mams_partial_refresh(u, dim, cos_a, sin_a, &rng, z);

    // ---------- 2. Save pre-trajectory state for MH ----------
    double x_old[MAX_DIM], u_old[MAX_DIM], grad_old[MAX_DIM];
    double potential_old = potential;
    for (int i = 0; i < dim; i++) {
        x_old[i] = x[i];
        u_old[i] = u[i];
        grad_old[i] = grad[i];
    }

    // ---------- 3. Isokinetic leapfrog trajectory ----------
    double total_delta_k = 0.0;
    int divergent = 0;
    int is_riemannian = (model_id == 5);
    double sqrt_inv_mass_local[MAX_DIM];
    for (int i = 0; i < dim; i++) {
        sqrt_inv_mass_local[i] = sqrt(inv_mass[i]);
    }

    for (int s = 0; s < n_steps; s++) {
        if (is_riemannian) {
            total_delta_k += b_step_riemannian_funnel(u, grad, x, inv_mass, eps * 0.5, dim, dm1);
            a_step_riemannian_funnel(x, u, inv_mass, eps, dim);
        } else {
            total_delta_k += b_step_precond(u, grad, sqrt_inv_mass_local, eps * 0.5, dim, dm1);
            a_step(x, u, inv_mass, eps, dim);
        }
        // Recompute gradient + potential (fused path when available).
        potential = mams_eval_grad_nll(x, grad, dim, model_data);
        if (is_riemannian) {
            total_delta_k += b_step_riemannian_funnel(u, grad, x, inv_mass, eps * 0.5, dim, dm1);
        } else {
            total_delta_k += b_step_precond(u, grad, sqrt_inv_mass_local, eps * 0.5, dim, dm1);
        }

        // Early termination check (skip on first call when potential_old = NaN)
        double current_w = (potential - potential_old) + total_delta_k;
        if (isfinite(potential_old) && (!isfinite(current_w) || current_w > divergence_threshold)) {
            divergent = 1;
            break;
        }
    }

    // Check for non-finite position
    for (int i = 0; i < dim; i++) {
        if (!isfinite(x[i])) { divergent = 1; break; }
    }

    // ---------- 4. MH accept/reject ----------
    int acc = 0;
    double w = 0.0;

    // First transition (potential_old = NaN): always accept to initialize
    // potential. The divergence check was already skipped above.
    int first_call = !isfinite(potential_old);

    if (divergent && !first_call) {
        // Reject: revert to pre-trajectory state with negated velocity
        for (int i = 0; i < dim; i++) {
            x[i] = x_old[i];
            u[i] = -u_old[i];
            grad[i] = grad_old[i];
        }
        potential = potential_old;
        w = 1.0 / 0.0;  // +inf
    } else if (first_call) {
        // Accept unconditionally — initializes potential + gradient
        acc = 1;
        w = 0.0;
    } else {
        double delta_v = potential - potential_old;
        w = delta_v + total_delta_k;

        if (enable_mh) {
            // Metropolis check
            if (isfinite(w) && (w <= 0.0 || philox_uniform(&rng) < exp(-w))) {
                acc = 1;
            } else {
                // Reject
                for (int i = 0; i < dim; i++) {
                    x[i] = x_old[i];
                    u[i] = -u_old[i];
                    grad[i] = grad_old[i];
                }
                potential = potential_old;
            }
        } else {
            // Phase 1: always accept (unadjusted)
            acc = 1;
        }
    }

    // ---------- 5. Store state back to global memory ----------
    for (int i = 0; i < dim; i++) {
        g_x[off + i] = x[i];
        g_u[off + i] = u[i];
        g_grad[off + i] = grad[i];
    }
    g_potential[chain] = potential;
    accepted[chain] = acc;
    energy_error[chain] = w;

    // ---------- 6. Accumulate samples for batch download ----------
    if (n_report > 0 && chain < n_report) {
        size_t slot = (size_t)store_idx * n_report + chain;
        g_accum_potential[slot] = potential;
        g_accum_accepted[slot] = acc;
        g_accum_energy[slot] = w;
        size_t pos_off = slot * dim;
        for (int i = 0; i < dim; i++) {
            g_sample_buf[pos_off + i] = x[i];
        }
    }
}

/* ---------- Fused multi-step kernel -------------------------------------- */
/*
 * Executes n_transitions MAMS transitions in a single kernel launch.
 * Chain state (x, u, grad, potential) lives in registers for the entire
 * duration, eliminating per-transition global memory round-trips.
 *
 * Achieves ~N× reduction in kernel launch overhead for fast models.
 * PhiloxState is re-initialized each transition for bit-exact determinism.
 */

extern "C" __global__ void mams_transition_fused(
    // Per-chain state [K * dim] — persistent across iterations
    double* __restrict__ g_x,
    double* __restrict__ g_u,
    double* __restrict__ g_potential,
    double* __restrict__ g_grad,
    // Sampler params — per-chain step size
    const double* __restrict__ d_eps,   // [K] per-chain step sizes
    double l,
    int max_leapfrog,                   // cap on per-chain n_steps
    int dim,
    const double* __restrict__ inv_mass,
    int enable_mh,
    // Model data
    const double* __restrict__ model_data,
    int model_id,
    // RNG
    unsigned long long seed,
    int iteration_start,      // first iteration index
    // Chain count
    int n_chains,
    // Output (per-chain, last transition's values)
    int* __restrict__ accepted_out,
    double* __restrict__ energy_error_out,
    // Accumulation buffers for batch download
    double* __restrict__ g_sample_buf,
    double* __restrict__ g_accum_potential,
    int*    __restrict__ g_accum_accepted,
    double* __restrict__ g_accum_energy,
    int n_report,
    // Fused-specific
    int n_transitions,        // number of transitions to execute
    double divergence_threshold
) {
    int chain = blockIdx.x * blockDim.x + threadIdx.x;
    if (chain >= n_chains) return;

    __mams_model_id = model_id;

    // Load per-chain step size
    double eps = d_eps[chain];
    int n_steps = (int)round(l / eps);
    if (n_steps < 1) n_steps = 1;
    if (n_steps > max_leapfrog) n_steps = max_leapfrog;

    double dm1 = (double)(dim - 1);
    if (dm1 < 0.5) dm1 = 1.0;

    // Load chain state into registers (ONE read for all transitions)
    double x[MAX_DIM], u[MAX_DIM], grad_reg[MAX_DIM];
    size_t off = (size_t)chain * dim;
    for (int i = 0; i < dim; i++) {
        x[i] = g_x[off + i];
        u[i] = g_u[off + i];
        grad_reg[i] = g_grad[off + i];
    }
    double potential = g_potential[chain];

    int last_acc = 0;
    double last_w = 0.0;

    for (int t = 0; t < n_transitions; t++) {
        int iteration = iteration_start + t;

        // Fresh RNG per transition (deterministic, matches non-fused path)
        PhiloxState rng;
        philox_init(&rng, seed ^ ((unsigned long long)iteration * 0x9E3779B97F4A7C15ULL), chain);

        // --- 1. Partial velocity refresh ---
        double angle = mams_refresh_angle(eps, l, model_id);
        double cos_a = cos(angle);
        double sin_a = sin(angle);

        double z[MAX_DIM];
        mams_partial_refresh(u, dim, cos_a, sin_a, &rng, z);

        // --- 2. Save pre-trajectory state ---
        double x_old[MAX_DIM], u_old[MAX_DIM], grad_old[MAX_DIM];
        double potential_old = potential;
        for (int i = 0; i < dim; i++) {
            x_old[i] = x[i];
            u_old[i] = u[i];
            grad_old[i] = grad_reg[i];
        }

        // --- 3. Isokinetic leapfrog ---
        double total_delta_k = 0.0;
        int divergent = 0;
        int is_riemannian = (model_id == 5);
        double sqrt_inv_mass_local[MAX_DIM];
        for (int i = 0; i < dim; i++) {
            sqrt_inv_mass_local[i] = sqrt(inv_mass[i]);
        }

        for (int s = 0; s < n_steps; s++) {
            if (is_riemannian) {
                total_delta_k += b_step_riemannian_funnel(u, grad_reg, x, inv_mass, eps * 0.5, dim, dm1);
                a_step_riemannian_funnel(x, u, inv_mass, eps, dim);
            } else {
                total_delta_k +=
                    b_step_precond(u, grad_reg, sqrt_inv_mass_local, eps * 0.5, dim, dm1);
                a_step(x, u, inv_mass, eps, dim);
            }
            potential = mams_eval_grad_nll(x, grad_reg, dim, model_data);
            if (is_riemannian) {
                total_delta_k += b_step_riemannian_funnel(u, grad_reg, x, inv_mass, eps * 0.5, dim, dm1);
            } else {
                total_delta_k +=
                    b_step_precond(u, grad_reg, sqrt_inv_mass_local, eps * 0.5, dim, dm1);
            }

            double current_w = (potential - potential_old) + total_delta_k;
            if (isfinite(potential_old) && (!isfinite(current_w) || current_w > divergence_threshold)) {
                divergent = 1;
                break;
            }
        }

        for (int i = 0; i < dim; i++) {
            if (!isfinite(x[i])) { divergent = 1; break; }
        }

        // --- 4. MH accept/reject ---
        int acc = 0;
        double w = 0.0;
        int first_call = !isfinite(potential_old);

        if (divergent && !first_call) {
            for (int i = 0; i < dim; i++) {
                x[i] = x_old[i];
                u[i] = -u_old[i];
                grad_reg[i] = grad_old[i];
            }
            potential = potential_old;
            w = 1.0 / 0.0;
        } else if (first_call) {
            acc = 1;
            w = 0.0;
        } else {
            double delta_v = potential - potential_old;
            w = delta_v + total_delta_k;

            if (enable_mh) {
                if (isfinite(w) && (w <= 0.0 || philox_uniform(&rng) < exp(-w))) {
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

        // --- 5. Accumulate sample for batch download ---
        if (n_report > 0 && chain < n_report) {
            size_t slot = (size_t)t * n_report + chain;
            g_accum_potential[slot] = potential;
            g_accum_accepted[slot] = acc;
            g_accum_energy[slot] = w;
            size_t pos_off = slot * dim;
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

/* ---------- Warp-cooperative kernel (1 warp = 1 chain) ------------------ */
/*
 * For data-heavy models where the observation loop dominates runtime.
 * 32 CUDA threads (1 warp) cooperate on a single chain's grad/NLL computation.
 *
 * Architecture:
 *   - All 32 lanes replicate chain state (x, u, grad) — deterministic ops
 *     produce identical results across lanes (same inputs, same SIMT instructions)
 *   - user_grad_warp / user_nll_warp distribute observations across lanes
 *   - __shfl_down_sync reduces partial results, __shfl_sync broadcasts
 *   - Only lane 0 writes results to global memory
 *
 * Shared memory:
 *   - Cooperative load of model data matrix in column-major layout
 *   - Host passes shared_mem_bytes = (n*p + n) * sizeof(double)
 *   - Copy: precomputed column-major global → column-major shared
 *
 * Launch config: total_threads = n_chains * 32, block_dim = 256 (8 chains/block)
 */

#ifndef MAMS_WARP_DEFINED
/* Default serial warp functions when caller doesn't provide cooperative versions */
__device__ double user_nll_warp(
    const double* x, int dim, const double* model_data,
    const double* /*s_X_col*/, const double* /*s_y*/,
    int /*n_obs*/, int /*n_feat*/, int /*lane_id*/, int /*use_shmem*/)
{
    return user_nll(x, dim, model_data);
}
__device__ void user_grad_warp(
    const double* x, double* grad, int dim, const double* model_data,
    const double* /*s_X_col*/, const double* /*s_y*/,
    int /*n_obs*/, int /*n_feat*/, int /*lane_id*/, int /*use_shmem*/)
{
    user_grad(x, grad, dim, model_data);
}
#endif

/* Optional fused warp callback:
 * compute grad and potential in one pass.
 * Default fallback calls separate callbacks to preserve behavior.
 */
#ifndef MAMS_WARP_FUSED_DEFINED
__device__ double user_grad_nll_warp(
    const double* x, double* grad, int dim, const double* model_data,
    const double* s_X_col, const double* s_y,
    int n_obs, int n_feat, int lane_id, int use_shmem)
{
    user_grad_warp(x, grad, dim, model_data, s_X_col, s_y, n_obs, n_feat, lane_id, use_shmem);
    return user_nll_warp(x, dim, model_data, s_X_col, s_y, n_obs, n_feat, lane_id, use_shmem);
}
#endif

/* Maximum parameter dimension for warp kernel.
 * The warp kernel targets low-dimensional, data-heavy models (e.g. GLM with
 * dim=6 and N=200 obs). Using MAX_DIM=128 here would put 7*128=896 doubles
 * into local memory per thread, and with 32 lanes × 4096 chains = 131k threads,
 * that's ~940 MB of local memory → cache thrashing, slower than scalar.
 * With MAX_DIM_WARP=32, all arrays fit in registers (~112 per thread). */
#define MAX_DIM_WARP 32

extern "C" __global__ void mams_transition_warp(
    // Per-chain state [K * dim] — persistent across iterations
    double* __restrict__ g_x,
    double* __restrict__ g_u,
    double* __restrict__ g_potential,
    double* __restrict__ g_grad,
    // Sampler params — per-chain step size
    const double* __restrict__ d_eps,   // [K] per-chain step sizes
    double l,
    int max_leapfrog,
    int dim,
    const double* __restrict__ inv_mass,
    int enable_mh,
    // Model data
    const double* __restrict__ model_data,
    int model_id,
    // RNG
    unsigned long long seed,
    int iteration,
    // Chain count
    int n_chains,
    // Output
    int* __restrict__ accepted,
    double* __restrict__ energy_error,
    // Accumulation buffers
    double* __restrict__ g_sample_buf,
    double* __restrict__ g_accum_potential,
    int*    __restrict__ g_accum_accepted,
    double* __restrict__ g_accum_energy,
    int store_idx,
    int n_report,
    // Warp-specific: data dimensions for shared memory layout
    int n_obs,    // number of observations (rows of X)
    int n_feat,   // number of features (columns of X)
    int warp_use_shmem, // whether to use shared-memory transpose
    double divergence_threshold
) {
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int chain = global_thread / 32;
    int lane_id = threadIdx.x % 32;

    // --- Optional cooperative shared memory load ---
    // For GLM, host layout is [n, p, X_row(n*p), y(n), X_col(n*p)].
    // Shared memory receives X_col directly (no runtime transpose).
    extern __shared__ double s_data[];
    double* s_X_col = s_data;               // [n_feat × n_obs] column-major
    double* s_y = s_data + n_obs * n_feat;  // [n_obs]

    if (warp_use_shmem == 1 && n_obs > 0 && n_feat > 0) {
        const double* X_col = model_data + 2 + n_obs * n_feat + n_obs;
        const double* y_src = model_data + 2 + n_obs * n_feat;

        // Cooperative copy: all threads in block participate
        int n_total = n_obs * n_feat;
        for (int idx = threadIdx.x; idx < n_total; idx += blockDim.x) {
            s_X_col[idx] = X_col[idx];
        }
        for (int idx = threadIdx.x; idx < n_obs; idx += blockDim.x) {
            s_y[idx] = y_src[idx];
        }
        __syncthreads();
    }

    if (chain >= n_chains) return;

    __mams_model_id = model_id;

    // Load per-chain step size (lane 0 loads, broadcast to all lanes via __shfl_sync)
    double eps;
    if (lane_id == 0) {
        eps = d_eps[chain];
    }
    eps = __shfl_sync(0xFFFFFFFF, eps, 0);
    int n_steps = (int)round(l / eps);
    if (n_steps < 1) n_steps = 1;
    if (n_steps > max_leapfrog) n_steps = max_leapfrog;

    double dm1 = (double)(dim - 1);
    if (dm1 < 0.5) dm1 = 1.0;

    // All lanes load the same chain state (replicated in registers)
    double x[MAX_DIM_WARP], u[MAX_DIM_WARP], grad_reg[MAX_DIM_WARP];
    size_t off = (size_t)chain * dim;
    for (int i = 0; i < dim; i++) {
        x[i] = g_x[off + i];
        u[i] = g_u[off + i];
        grad_reg[i] = g_grad[off + i];
    }
    double potential = g_potential[chain];

    // All lanes get identical Philox state (same chain → same sequence)
    PhiloxState rng;
    philox_init(&rng, seed ^ ((unsigned long long)iteration * 0x9E3779B97F4A7C15ULL), chain);

    // ---------- 1. Partial velocity refresh (identical across all lanes) ----------
    double angle = mams_refresh_angle(eps, l, model_id);
    double cos_a = cos(angle);
    double sin_a = sin(angle);

    double z[MAX_DIM_WARP];
    mams_partial_refresh(u, dim, cos_a, sin_a, &rng, z);

    // ---------- 2. Save pre-trajectory state ----------
    double x_old[MAX_DIM_WARP], u_old[MAX_DIM_WARP], grad_old[MAX_DIM_WARP];
    double potential_old = potential;
    for (int i = 0; i < dim; i++) {
        x_old[i] = x[i];
        u_old[i] = u[i];
        grad_old[i] = grad_reg[i];
    }

    // ---------- 3. Isokinetic leapfrog (warp-cooperative grad/nll) ----------
    double total_delta_k = 0.0;
    int divergent = 0;
    int is_riemannian = (model_id == 5);
    double sqrt_inv_mass_local[MAX_DIM];
    for (int i = 0; i < dim; i++) {
        sqrt_inv_mass_local[i] = sqrt(inv_mass[i]);
    }

    for (int s = 0; s < n_steps; s++) {
        if (is_riemannian) {
            total_delta_k += b_step_riemannian_funnel(u, grad_reg, x, inv_mass, eps * 0.5, dim, dm1);
            a_step_riemannian_funnel(x, u, inv_mass, eps, dim);
        } else {
            total_delta_k += b_step_precond(u, grad_reg, sqrt_inv_mass_local, eps * 0.5, dim, dm1);
            a_step(x, u, inv_mass, eps, dim);
        }
        // Warp-cooperative grad + nll (single fused callback)
        potential = user_grad_nll_warp(
            x, grad_reg, dim, model_data, s_X_col, s_y, n_obs, n_feat, lane_id, warp_use_shmem);
        if (is_riemannian) {
            total_delta_k += b_step_riemannian_funnel(u, grad_reg, x, inv_mass, eps * 0.5, dim, dm1);
        } else {
            total_delta_k += b_step_precond(u, grad_reg, sqrt_inv_mass_local, eps * 0.5, dim, dm1);
        }

        double current_w = (potential - potential_old) + total_delta_k;
        if (isfinite(potential_old) && (!isfinite(current_w) || current_w > divergence_threshold)) {
            divergent = 1;
            break;
        }
    }

    for (int i = 0; i < dim; i++) {
        if (!isfinite(x[i])) { divergent = 1; break; }
    }

    // ---------- 4. MH accept/reject (RNG identical across lanes) ----------
    int acc = 0;
    double w = 0.0;
    int first_call = !isfinite(potential_old);

    if (divergent && !first_call) {
        for (int i = 0; i < dim; i++) {
            x[i] = x_old[i];
            u[i] = -u_old[i];
            grad_reg[i] = grad_old[i];
        }
        potential = potential_old;
        w = 1.0 / 0.0;
    } else if (first_call) {
        acc = 1;
        w = 0.0;
    } else {
        double delta_v = potential - potential_old;
        w = delta_v + total_delta_k;

        if (enable_mh) {
            if (isfinite(w) && (w <= 0.0 || philox_uniform(&rng) < exp(-w))) {
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

        if (n_report > 0 && chain < n_report) {
            size_t slot = (size_t)store_idx * n_report + chain;
            g_accum_potential[slot] = potential;
            g_accum_accepted[slot] = acc;
            g_accum_energy[slot] = w;
            size_t pos_off = slot * dim;
            for (int i = 0; i < dim; i++) {
                g_sample_buf[pos_off + i] = x[i];
            }
        }
    }
}

#endif /* MAMS_ENGINE_CUH */
