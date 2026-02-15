/**
 * Monte Carlo Fault-Tree Simulation Metal Kernel.
 *
 * Architecture: 1 thread = 1 scenario, grid-stride via thread_position_in_grid.
 * Inline Philox-4x32 RNG (no external RNG library).
 * Box-Muller for normal sampling (BernoulliUncertain Z).
 * Atomic counters for TOP failure + component importance.
 *
 * All computation in float (f32) — Apple Silicon has no hardware f64.
 *
 * Component failure modes:
 *   0 = Bernoulli(p)              — params: [p, 0, 0]
 *   1 = BernoulliUncertain(mu,s)  — params: [mu, sigma, 0]
 *   2 = WeibullMission(k,lam,T)   — params: [k, lambda, mission_time]
 *
 * Node types:
 *   0 = Component (data = component_index)
 *   1 = AND gate
 *   2 = OR gate
 */

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

/* ---------- Philox-4x32 inline RNG --------------------------------------- */

constant uint PHILOX_M0 = 0xD2511F53u;
constant uint PHILOX_M1 = 0xCD9E8D57u;
constant uint PHILOX_W0 = 0x9E3779B9u;
constant uint PHILOX_W1 = 0xBB67AE85u;

struct PhiloxState {
    uint counter[4];
    uint key[2];
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

inline float u32_to_uniform(uint x) {
    return float(x >> 1) * (1.0f / 2147483648.0f);
}

inline float box_muller(float u1, float u2) {
    float r = sqrt(-2.0f * log(max(u1, 1e-20f)));
    float theta = 2.0f * 3.14159265f * u2;
    return r * cos(theta);
}

inline float sigmoid_f(float x) {
    if (x >= 0.0f) {
        float e = exp(-x);
        return 1.0f / (1.0f + e);
    } else {
        float e = exp(x);
        return e / (1.0f + e);
    }
}

/* ---------- Kernel ------------------------------------------------------- */

kernel void fault_tree_mc_kernel(
    device const int* comp_types       [[buffer(0)]],  // [n_comp]
    device const float* comp_params    [[buffer(1)]],  // [3 * n_comp]
    device const int* node_types       [[buffer(2)]],  // [n_nodes]
    device const int* node_data        [[buffer(3)]],  // [n_nodes]
    device const int* children_offsets [[buffer(4)]],  // [n_nodes+1]
    device const int* children_flat    [[buffer(5)]],  // [total_children]
    device atomic_uint* top_fail_count [[buffer(6)]],  // [1]
    device atomic_uint* comp_fail_counts [[buffer(7)]], // [n_comp]
    constant int& n_components         [[buffer(8)]],
    constant int& n_nodes              [[buffer(9)]],
    constant int& top_node             [[buffer(10)]],
    constant uint& seed_lo             [[buffer(11)]],
    constant uint& seed_hi             [[buffer(12)]],
    constant int& n_scenarios          [[buffer(13)]],
    uint tid [[thread_position_in_grid]]
) {
    if ((int)tid >= n_scenarios) return;

    int sid = (int)tid;

    // RNG: xoshiro128**-inspired. High-quality, fast, no lambda issues.
    uint rng_s0 = seed_lo ^ ((uint)sid * 0x9E3779B9u + 1u);
    uint rng_s1 = seed_hi ^ ((uint)sid * 0x517CC1B7u + 1u);
    uint rng_s2 = seed_lo ^ (seed_hi + (uint)sid + 0x6C62272Eu);
    uint rng_s3 = (uint)sid ^ (seed_lo * 0x61C88647u + seed_hi);
    // Warm up
    for (int _w = 0; _w < 4; _w++) {
        uint t = rng_s1 << 9;
        rng_s2 ^= rng_s0; rng_s3 ^= rng_s1; rng_s1 ^= rng_s2; rng_s0 ^= rng_s3;
        rng_s2 ^= t; rng_s3 = (rng_s3 << 11) | (rng_s3 >> 21);
    }
    #define NEXT_UNIFORM(result) { \
        uint _r = rng_s1 * 5; \
        _r = ((_r << 7) | (_r >> 25)) * 9; \
        uint _t = rng_s1 << 9; \
        rng_s2 ^= rng_s0; rng_s3 ^= rng_s1; rng_s1 ^= rng_s2; rng_s0 ^= rng_s3; \
        rng_s2 ^= _t; rng_s3 = (rng_s3 << 11) | (rng_s3 >> 21); \
        result = u32_to_uniform(_r); \
    }

    // Draw Z for epistemic uncertainty.
    float u_z1, u_z2;
    NEXT_UNIFORM(u_z1);
    NEXT_UNIFORM(u_z2);
    float z_epistemic = box_muller(u_z1, u_z2);

    // Evaluate component failures (bitmask, up to 64 components in fast path).
    ulong comp_mask = 0;
    ulong comp_mask_hi = 0;

    for (int c = 0; c < n_components; c++) {
        float u;
        NEXT_UNIFORM(u);
        int ctype = comp_types[c];
        bool failed = false;

        if (ctype == 0) {
            failed = (u < comp_params[3 * c]);
        } else if (ctype == 1) {
            float mu = comp_params[3 * c];
            float sigma = comp_params[3 * c + 1];
            float p = sigmoid_f(mu + sigma * z_epistemic);
            failed = (u < p);
        } else {
            float k = comp_params[3 * c];
            float lambda = comp_params[3 * c + 1];
            float mission_time = comp_params[3 * c + 2];
            float t_sample = lambda * pow(-log(max(u, 1e-20f)), 1.0f / k);
            failed = (t_sample <= mission_time);
        }

        if (failed) {
            if (c < 64) {
                comp_mask |= (1UL << c);
            } else {
                comp_mask_hi |= (1UL << (c - 64));
            }
        }
    }

    // Evaluate fault tree bottom-up.
    // Fast path: bitmask for ≤64 nodes; general path: thread-local bool array for >64.
    bool top_failed = false;

    if (n_nodes <= 64) {
        ulong node_mask = 0;
        for (int ni = 0; ni < n_nodes; ni++) {
            int ntype = node_types[ni];
            bool state;
            if (ntype == 0) {
                int ci = node_data[ni];
                state = (ci < 64) ? ((comp_mask >> ci) & 1) : ((comp_mask_hi >> (ci - 64)) & 1);
            } else {
                int start = children_offsets[ni];
                int end = children_offsets[ni + 1];
                if (ntype == 1) {
                    state = true;
                    for (int j = start; j < end; j++) {
                        if (!((node_mask >> children_flat[j]) & 1)) {
                            state = false;
                            break;
                        }
                    }
                } else {
                    state = false;
                    for (int j = start; j < end; j++) {
                        if ((node_mask >> children_flat[j]) & 1) {
                            state = true;
                            break;
                        }
                    }
                }
            }
            if (state) {
                node_mask |= (1UL << ni);
            }
        }
        top_failed = (node_mask >> top_node) & 1;
    } else {
        // General path: thread-local bool array (supports up to 512 nodes).
        bool node_states[512];
        int nn = min(n_nodes, 512);
        for (int ni = 0; ni < nn; ni++) {
            int ntype = node_types[ni];
            bool state;
            if (ntype == 0) {
                int ci = node_data[ni];
                state = (ci < 64) ? ((comp_mask >> ci) & 1) : ((comp_mask_hi >> (ci - 64)) & 1);
            } else {
                int start = children_offsets[ni];
                int end = children_offsets[ni + 1];
                if (ntype == 1) {
                    state = true;
                    for (int j = start; j < end; j++) {
                        if (!node_states[children_flat[j]]) {
                            state = false;
                            break;
                        }
                    }
                } else {
                    state = false;
                    for (int j = start; j < end; j++) {
                        if (node_states[children_flat[j]]) {
                            state = true;
                            break;
                        }
                    }
                }
            }
            node_states[ni] = state;
        }
        top_failed = node_states[top_node];
    }

    if (top_failed) {
        atomic_fetch_add_explicit(top_fail_count, 1u, memory_order_relaxed);
        for (int c = 0; c < n_components && c < 128; c++) {
            bool cf = (c < 64) ? ((comp_mask >> c) & 1) : ((comp_mask_hi >> (c - 64)) & 1);
            if (cf) {
                atomic_fetch_add_explicit(&comp_fail_counts[c], 1u, memory_order_relaxed);
            }
        }
    }
}
