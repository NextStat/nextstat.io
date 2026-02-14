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

/* ---------- Integrator steps --------------------------------------------- */

// B-step: half-step velocity update on unit sphere.
// Returns delta_k contribution.
__device__ double b_step(
    double* u, const double* grad, const double* inv_mass,
    double half_eps, int dim, double dm1)
{
    // Preconditioned gradient: g_tilde_i = sqrt(inv_mass_i) * grad_i
    double g_norm_sq = 0.0;
    for (int i = 0; i < dim; i++) {
        double gi = sqrt(inv_mass[i]) * grad[i];
        g_norm_sq += gi * gi;
    }
    double g_norm = sqrt(g_norm_sq);
    if (g_norm < 1e-30) return 0.0;

    // e = g_tilde / |g_tilde|, compute e_dot_u
    double e_dot_u = 0.0;
    for (int i = 0; i < dim; i++) {
        double ei = sqrt(inv_mass[i]) * grad[i] / g_norm;
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
        double ei = sqrt(inv_mass[i]) * grad[i] / g_norm;
        u[i] = u[i] * c_u - ei * c_e;
        u_norm_sq += u[i] * u[i];
    }
    double u_norm = sqrt(u_norm_sq);
    if (u_norm > 1e-12) {
        double inv = 1.0 / u_norm;
        for (int i = 0; i < dim; i++) u[i] *= inv;
    } else {
        for (int i = 0; i < dim; i++) {
            u[i] = -sqrt(inv_mass[i]) * grad[i] / g_norm;
        }
    }

    // delta_k = dm1 * (delta + ln(1 + 0.5 * zeta_m1 * (1 + e_dot_u)))
    double arg = 0.5 * zeta_m1 * (1.0 + e_dot_u);
    if (arg < -1.0 + 1e-50) arg = -1.0 + 1e-50;
    return dm1 * (delta + log1p(arg));
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

/* ---------- Main kernel -------------------------------------------------- */

extern "C" __global__ void mams_transition(
    // Per-chain state [K * dim] — persistent across iterations
    double* __restrict__ g_x,           // positions
    double* __restrict__ g_u,           // unit velocities
    double* __restrict__ g_potential,    // [K] potentials
    double* __restrict__ g_grad,        // [K * dim] gradients
    // Sampler params (uniform across chains)
    double eps,
    double l,
    int n_steps,
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
    int n_report                             // chains to accumulate (0 = disabled)
) {
    int chain = blockIdx.x * blockDim.x + threadIdx.x;
    if (chain >= n_chains) return;

    // Publish model_id so user_nll/user_grad dispatch can read it (build-time path)
    __mams_model_id = model_id;

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
    double angle = eps / l;
    double cos_a = cos(angle);
    double sin_a = sin(angle);

    // Sample z ~ N(0,I) and project out u-component
    double z[MAX_DIM];
    double u_dot_z = 0.0;
    for (int i = 0; i < dim; i++) {
        z[i] = philox_normal(&rng);
        u_dot_z += u[i] * z[i];
    }
    double z_perp_norm_sq = 0.0;
    for (int i = 0; i < dim; i++) {
        z[i] -= u_dot_z * u[i];
        z_perp_norm_sq += z[i] * z[i];
    }
    double z_perp_norm = sqrt(z_perp_norm_sq);
    if (z_perp_norm > 1e-12) {
        double inv_norm = 1.0 / z_perp_norm;
        double u_new_norm_sq = 0.0;
        for (int i = 0; i < dim; i++) {
            u[i] = u[i] * cos_a + z[i] * inv_norm * sin_a;
            u_new_norm_sq += u[i] * u[i];
        }
        double u_new_norm = sqrt(u_new_norm_sq);
        double inv = 1.0 / u_new_norm;
        for (int i = 0; i < dim; i++) u[i] *= inv;
    }

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

    for (int s = 0; s < n_steps; s++) {
        // B(eps/2)
        total_delta_k += b_step(u, grad, inv_mass, eps * 0.5, dim, dm1);
        // A(eps)
        a_step(x, u, inv_mass, eps, dim);
        // Recompute gradient + potential
        user_grad(x, grad, dim, model_data);
        potential = user_nll(x, dim, model_data);
        // B(eps/2)
        total_delta_k += b_step(u, grad, inv_mass, eps * 0.5, dim, dm1);

        // Early termination check
        double current_w = (potential - potential_old) + total_delta_k;
        if (!isfinite(current_w) || current_w > 1000.0) {
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

    if (divergent) {
        // Reject: revert to pre-trajectory state with negated velocity
        for (int i = 0; i < dim; i++) {
            x[i] = x_old[i];
            u[i] = -u_old[i];
            grad[i] = grad_old[i];
        }
        potential = potential_old;
        w = 1.0 / 0.0;  // +inf
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

#endif /* MAMS_ENGINE_CUH */
