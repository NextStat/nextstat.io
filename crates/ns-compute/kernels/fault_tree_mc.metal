/**
 * Monte Carlo Fault-Tree Simulation Metal Kernel.
 *
 * Architecture: 1 thread = 1 scenario, grid-stride via thread_position_in_grid.
 * Counter-based RNG stream (no external RNG library).
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

/* ---------- Counter-based RNG (SplitMix64 hash stream) -------------------- */

inline ulong splitmix64(ulong x) {
    x += 0x9E3779B97F4A7C15ul;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ul;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBul;
    return x ^ (x >> 31);
}

inline uint counter_u32(uint seed_lo, uint seed_hi, ulong global_sid, ulong draw_idx) {
    ulong seed64 = ((ulong)seed_hi << 32) | (ulong)seed_lo;
    ulong x =
        seed64
        ^ (global_sid * 0xD2B74407B1CE6E93ul)
        ^ (draw_idx * 0x9E3779B97F4A7C15ul);
    return (uint)splitmix64(x);
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
    constant uint& scenario_offset_lo  [[buffer(13)]],
    constant uint& scenario_offset_hi  [[buffer(14)]],
    constant int& n_scenarios          [[buffer(15)]],
    uint tid [[thread_position_in_grid]]
) {
    if ((int)tid >= n_scenarios) return;

    ulong scenario_offset = ((ulong)scenario_offset_hi << 32) | (ulong)scenario_offset_lo;
    ulong global_sid = scenario_offset + (ulong)tid;
    ulong draw_idx = 0ul;
    #define NEXT_UNIFORM(result) { \
        uint _r = counter_u32(seed_lo, seed_hi, global_sid, draw_idx++); \
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
