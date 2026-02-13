/**
 * NLL reduction kernel for externally-computed log-prob values (flow PDFs).
 *
 * Unlike unbinned_nll_grad.cu which fuses PDF evaluation + NLL reduction,
 * this kernel takes pre-computed log p(x|θ) per process per event and only
 * performs the extended unbinned likelihood reduction:
 *
 *   NLL = ν_tot - Σ_events log(Σ_procs ν_p * p_p(x_i)) + constraints
 *
 * This enables:
 *   - Flow PDFs evaluated on CPU (FlowPdf) or CUDA EP (ONNX Runtime)
 *   - Mixed models: some processes use flows, others use parametric PDFs
 *   - The log_prob buffer can come from host upload or GPU-resident ONNX output
 *
 * Architecture: 1 CUDA block, threads cooperate on event-level reduction.
 */

#include <math.h>
#include <float.h>

struct FlowGaussConstraintEntry {
    double center;
    double inv_width;
    unsigned int param_idx;
    unsigned int _pad;
};

/**
 * NLL reduction from pre-computed log-prob values.
 *
 * @param g_logp_flat  [n_procs × n_events] row-major: logp[p * n_events + i]
 * @param g_yields     [n_procs] per-process yields ν_p (already includes rate modifiers)
 * @param g_gauss      [n_gauss] Gaussian constraint entries
 * @param g_params     [n_params] current parameter values (for constraints)
 * @param g_nll_out    [1] output NLL scalar
 * @param n_events     number of events
 * @param n_procs      number of processes
 * @param n_gauss      number of Gaussian constraints
 * @param constraint_const  constant term from constraints
 */
extern "C" __global__ void flow_nll_reduce(
    const double* __restrict__ g_logp_flat,   /* [n_procs × n_events] */
    const double* __restrict__ g_yields,      /* [n_procs] */
    const struct FlowGaussConstraintEntry* __restrict__ g_gauss, /* [n_gauss] */
    const double* __restrict__ g_params,      /* [n_params] */
    double* __restrict__ g_nll_out,           /* [1] */
    unsigned int n_events,
    unsigned int n_procs,
    unsigned int n_gauss,
    double constraint_const
) {
    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    /* Shared memory for parallel reduction. */
    extern __shared__ double s_scratch[];

    /* Load yields into registers (n_procs is small, typically 2-10). */
    double local_sum_logf = 0.0;

    for (unsigned int i = tid; i < n_events; i += block_size) {
        /* Online logsumexp over processes: log(Σ_p ν_p * p_p(x_i)) */
        double max_term = -INFINITY;
        double sum_exp = 0.0;

        for (unsigned int p = 0; p < n_procs; p++) {
            double nu = g_yields[p];
            if (!(nu > 0.0) || !isfinite(nu)) {
                continue;
            }
            double logp = g_logp_flat[(size_t)p * n_events + i];
            if (!isfinite(logp)) {
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

    /* Block-level parallel reduction of local_sum_logf. */
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

        /* ν_tot = Σ_p ν_p */
        double nu_tot = 0.0;
        for (unsigned int p = 0; p < n_procs; p++) {
            nu_tot += g_yields[p];
        }

        /* Gaussian constraints: 0.5 * Σ ((p - center) * inv_width)^2 */
        double constraint_sum = 0.0;
        for (unsigned int g = 0; g < n_gauss; g++) {
            double center = g_gauss[g].center;
            double inv_w = g_gauss[g].inv_width;
            unsigned int pidx = g_gauss[g].param_idx;
            double delta = (g_params[pidx] - center) * inv_w;
            constraint_sum += 0.5 * delta * delta;
        }

        /* Extended unbinned NLL:
         * NLL = ν_tot - Σ log f(x_i) + constraints + constraint_const */
        double nll = nu_tot - sum_logf + constraint_sum + constraint_const;
        g_nll_out[0] = nll;
    }
}

/**
 * Same as flow_nll_reduce, but consumes log-prob as f32.
 *
 * This is the preferred path for CUDA EP I/O binding, since ONNX Runtime
 * commonly produces `float` log_prob outputs.
 */
extern "C" __global__ void flow_nll_reduce_f32(
    const float* __restrict__ g_logp_flat,    /* [n_procs × n_events] */
    const double* __restrict__ g_yields,      /* [n_procs] */
    const struct FlowGaussConstraintEntry* __restrict__ g_gauss, /* [n_gauss] */
    const double* __restrict__ g_params,      /* [n_params] */
    double* __restrict__ g_nll_out,           /* [1] */
    unsigned int n_events,
    unsigned int n_procs,
    unsigned int n_gauss,
    double constraint_const
) {
    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    extern __shared__ double s_scratch[];

    double local_sum_logf = 0.0;

    for (unsigned int i = tid; i < n_events; i += block_size) {
        double max_term = -INFINITY;
        double sum_exp = 0.0;

        for (unsigned int p = 0; p < n_procs; p++) {
            double nu = g_yields[p];
            if (!(nu > 0.0) || !isfinite(nu)) {
                continue;
            }
            double logp = (double)g_logp_flat[(size_t)p * n_events + i];
            if (!isfinite(logp)) {
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
            nu_tot += g_yields[p];
        }

        double constraint_sum = 0.0;
        for (unsigned int g = 0; g < n_gauss; g++) {
            double center = g_gauss[g].center;
            double inv_w = g_gauss[g].inv_width;
            unsigned int pidx = g_gauss[g].param_idx;
            double delta = (g_params[pidx] - center) * inv_w;
            constraint_sum += 0.5 * delta * delta;
        }

        double nll = nu_tot - sum_logf + constraint_sum + constraint_const;
        g_nll_out[0] = nll;
    }
}

/**
 * Fused NLL + analytical gradient intermediates from pre-computed log-prob and Jacobian.
 *
 * Computes in a single kernel launch:
 *   1. NLL = ν_tot − Σᵢ log f(xᵢ) + constraints + const
 *   2. sum_r[p] = Σᵢ exp(logpₚ(xᵢ)) / f(xᵢ)  (per-process responsibility sum)
 *   3. sum_jr[p * n_context + k] = Σᵢ [exp(logpₚ(xᵢ)) / f(xᵢ)] · ∂logpₚ/∂ctx_k
 *
 * Host-side gradient assembly (cheap, O(n_params)):
 *   ∂NLL/∂θⱼ (yield)   = ∂νₚ/∂θⱼ · (1 − sum_r[p])
 *   ∂NLL/∂θⱼ (context) = − Σₚ νₚ · sum_jr[p * n_ctx + k]  where ctx_k maps to θⱼ
 *   ∂NLL/∂θⱼ (constr)  = (θⱼ − center) · inv_width²
 *
 * Shared memory layout (dynamic):
 *   s_nll    [block_size]                          — NLL reduction scratch
 *   s_sum_r  [block_size × n_procs]                — per-process responsibility
 *   s_sum_jr [block_size × n_procs × n_context]    — per-process Jacobian-weighted responsibility
 *
 * @param g_logp_flat   [n_procs × n_events] row-major: logp[p * n_events + i]
 * @param g_jac_flat    [n_procs × n_events × n_context] row-major: jac[p * n_events * n_context + i * n_context + k]
 *                      May be NULL if n_context == 0.
 * @param g_yields      [n_procs] per-process yields
 * @param g_gauss       [n_gauss] Gaussian constraint entries
 * @param g_params      [n_params] current parameter values
 * @param g_nll_out     [1] output NLL scalar
 * @param g_sum_r_out   [n_procs] output per-process responsibility sums
 * @param g_sum_jr_out  [n_procs × n_context] output Jacobian-weighted responsibility sums
 * @param n_events      number of events
 * @param n_procs       number of processes
 * @param n_context     number of context parameters per process (0 for unconditional)
 * @param n_gauss       number of Gaussian constraints
 * @param constraint_const  constant term from constraints
 */
extern "C" __global__ void flow_nll_grad_reduce(
    const double* __restrict__ g_logp_flat,    /* [n_procs × n_events] */
    const double* __restrict__ g_jac_flat,     /* [n_procs × n_events × n_context] or NULL */
    const double* __restrict__ g_yields,       /* [n_procs] */
    const struct FlowGaussConstraintEntry* __restrict__ g_gauss, /* [n_gauss] */
    const double* __restrict__ g_params,       /* [n_params] */
    double* __restrict__ g_nll_out,            /* [1] */
    double* __restrict__ g_sum_r_out,          /* [n_procs] */
    double* __restrict__ g_sum_jr_out,         /* [n_procs × n_context] */
    unsigned int n_events,
    unsigned int n_procs,
    unsigned int n_context,
    unsigned int n_gauss,
    double constraint_const
) {
    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    /* Dynamic shared memory partitioning:
     *   s_nll    [block_size]
     *   s_sum_r  [block_size × n_procs]
     *   s_sum_jr [block_size × n_procs × n_context]
     */
    extern __shared__ double s_mem[];
    double* s_nll    = s_mem;
    double* s_sum_r  = s_nll + block_size;
    double* s_sum_jr = s_sum_r + (size_t)block_size * n_procs;

    /* Thread-local accumulators. */
    double local_sum_logf = 0.0;

    /* Zero per-process responsibility accumulators in shared memory. */
    for (unsigned int p = 0; p < n_procs; p++) {
        s_sum_r[tid * n_procs + p] = 0.0;
    }
    for (unsigned int idx = 0; idx < n_procs * n_context; idx++) {
        s_sum_jr[tid * n_procs * n_context + idx] = 0.0;
    }

    /* Main event loop: each thread processes a strided subset of events. */
    for (unsigned int i = tid; i < n_events; i += block_size) {
        /* Compute f(xᵢ) = Σₚ νₚ · exp(logpₚ) via online logsumexp. */
        double max_term = -INFINITY;
        double sum_exp = 0.0;

        for (unsigned int p = 0; p < n_procs; p++) {
            double nu = g_yields[p];
            if (!(nu > 0.0) || !isfinite(nu)) {
                continue;
            }
            double logp = g_logp_flat[(size_t)p * n_events + i];
            if (!isfinite(logp)) {
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

        double inv_f = 1.0 / exp(logf);

        /* Accumulate per-process responsibilities and Jacobian-weighted responsibilities. */
        for (unsigned int p = 0; p < n_procs; p++) {
            double nu = g_yields[p];
            if (!(nu > 0.0) || !isfinite(nu)) {
                continue;
            }
            double logp = g_logp_flat[(size_t)p * n_events + i];
            if (!isfinite(logp)) {
                continue;
            }

            double exp_logp = exp(logp);
            double r_pi = exp_logp * inv_f;  /* exp(logpₚ) / f(xᵢ) */

            s_sum_r[tid * n_procs + p] += r_pi;

            /* Jacobian-weighted responsibility: r_pi · ∂logpₚ/∂ctx_k */
            if (n_context > 0 && g_jac_flat != 0) {
                size_t jac_base = (size_t)p * n_events * n_context + (size_t)i * n_context;
                size_t jr_base = tid * n_procs * n_context + (size_t)p * n_context;
                for (unsigned int k = 0; k < n_context; k++) {
                    double djac = g_jac_flat[jac_base + k];
                    s_sum_jr[jr_base + k] += r_pi * djac;
                }
            }
        }
    }

    /* Store NLL local sum for reduction. */
    s_nll[tid] = local_sum_logf;
    __syncthreads();

    /* Block-level parallel reduction for all accumulators. */
    for (unsigned int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_nll[tid] += s_nll[tid + stride];
            for (unsigned int p = 0; p < n_procs; p++) {
                s_sum_r[tid * n_procs + p] += s_sum_r[(tid + stride) * n_procs + p];
            }
            unsigned int jr_count = n_procs * n_context;
            for (unsigned int idx = 0; idx < jr_count; idx++) {
                s_sum_jr[tid * jr_count + idx] += s_sum_jr[(tid + stride) * jr_count + idx];
            }
        }
        __syncthreads();
    }

    /* Thread 0 writes final outputs. */
    if (tid == 0) {
        double sum_logf = s_nll[0];

        /* ν_tot */
        double nu_tot = 0.0;
        for (unsigned int p = 0; p < n_procs; p++) {
            nu_tot += g_yields[p];
        }

        /* Gaussian constraints */
        double constraint_sum = 0.0;
        for (unsigned int g = 0; g < n_gauss; g++) {
            double center = g_gauss[g].center;
            double inv_w = g_gauss[g].inv_width;
            unsigned int pidx = g_gauss[g].param_idx;
            double delta = (g_params[pidx] - center) * inv_w;
            constraint_sum += 0.5 * delta * delta;
        }

        g_nll_out[0] = nu_tot - sum_logf + constraint_sum + constraint_const;

        /* Write per-process responsibility sums. */
        for (unsigned int p = 0; p < n_procs; p++) {
            g_sum_r_out[p] = s_sum_r[p];
        }

        /* Write Jacobian-weighted responsibility sums. */
        unsigned int jr_count = n_procs * n_context;
        for (unsigned int idx = 0; idx < jr_count; idx++) {
            g_sum_jr_out[idx] = s_sum_jr[idx];
        }
    }
}

/**
 * Same as flow_nll_grad_reduce, but consumes log-prob as f32 and Jacobian as f32.
 *
 * Preferred for CUDA EP I/O binding path where ONNX Runtime produces float outputs.
 */
extern "C" __global__ void flow_nll_grad_reduce_f32(
    const float* __restrict__ g_logp_flat,     /* [n_procs × n_events] */
    const float* __restrict__ g_jac_flat,      /* [n_procs × n_events × n_context] or NULL */
    const double* __restrict__ g_yields,       /* [n_procs] */
    const struct FlowGaussConstraintEntry* __restrict__ g_gauss, /* [n_gauss] */
    const double* __restrict__ g_params,       /* [n_params] */
    double* __restrict__ g_nll_out,            /* [1] */
    double* __restrict__ g_sum_r_out,          /* [n_procs] */
    double* __restrict__ g_sum_jr_out,         /* [n_procs × n_context] */
    unsigned int n_events,
    unsigned int n_procs,
    unsigned int n_context,
    unsigned int n_gauss,
    double constraint_const
) {
    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    extern __shared__ double s_mem[];
    double* s_nll    = s_mem;
    double* s_sum_r  = s_nll + block_size;
    double* s_sum_jr = s_sum_r + (size_t)block_size * n_procs;

    double local_sum_logf = 0.0;

    for (unsigned int p = 0; p < n_procs; p++) {
        s_sum_r[tid * n_procs + p] = 0.0;
    }
    for (unsigned int idx = 0; idx < n_procs * n_context; idx++) {
        s_sum_jr[tid * n_procs * n_context + idx] = 0.0;
    }

    for (unsigned int i = tid; i < n_events; i += block_size) {
        double max_term = -INFINITY;
        double sum_exp = 0.0;

        for (unsigned int p = 0; p < n_procs; p++) {
            double nu = g_yields[p];
            if (!(nu > 0.0) || !isfinite(nu)) {
                continue;
            }
            double logp = (double)g_logp_flat[(size_t)p * n_events + i];
            if (!isfinite(logp)) {
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

        double inv_f = 1.0 / exp(logf);

        for (unsigned int p = 0; p < n_procs; p++) {
            double nu = g_yields[p];
            if (!(nu > 0.0) || !isfinite(nu)) {
                continue;
            }
            double logp = (double)g_logp_flat[(size_t)p * n_events + i];
            if (!isfinite(logp)) {
                continue;
            }

            double exp_logp = exp(logp);
            double r_pi = exp_logp * inv_f;

            s_sum_r[tid * n_procs + p] += r_pi;

            if (n_context > 0 && g_jac_flat != 0) {
                size_t jac_base = (size_t)p * n_events * n_context + (size_t)i * n_context;
                size_t jr_base = tid * n_procs * n_context + (size_t)p * n_context;
                for (unsigned int k = 0; k < n_context; k++) {
                    double djac = (double)g_jac_flat[jac_base + k];
                    s_sum_jr[jr_base + k] += r_pi * djac;
                }
            }
        }
    }

    s_nll[tid] = local_sum_logf;
    __syncthreads();

    for (unsigned int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_nll[tid] += s_nll[tid + stride];
            for (unsigned int p = 0; p < n_procs; p++) {
                s_sum_r[tid * n_procs + p] += s_sum_r[(tid + stride) * n_procs + p];
            }
            unsigned int jr_count = n_procs * n_context;
            for (unsigned int idx = 0; idx < jr_count; idx++) {
                s_sum_jr[tid * jr_count + idx] += s_sum_jr[(tid + stride) * jr_count + idx];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        double sum_logf = s_nll[0];

        double nu_tot = 0.0;
        for (unsigned int p = 0; p < n_procs; p++) {
            nu_tot += g_yields[p];
        }

        double constraint_sum = 0.0;
        for (unsigned int g = 0; g < n_gauss; g++) {
            double center = g_gauss[g].center;
            double inv_w = g_gauss[g].inv_width;
            unsigned int pidx = g_gauss[g].param_idx;
            double delta = (g_params[pidx] - center) * inv_w;
            constraint_sum += 0.5 * delta * delta;
        }

        g_nll_out[0] = nu_tot - sum_logf + constraint_sum + constraint_const;

        for (unsigned int p = 0; p < n_procs; p++) {
            g_sum_r_out[p] = s_sum_r[p];
        }

        unsigned int jr_count = n_procs * n_context;
        for (unsigned int idx = 0; idx < jr_count; idx++) {
            g_sum_jr_out[idx] = s_sum_jr[idx];
        }
    }
}

/**
 * Batch version: 1 block = 1 toy dataset.
 *
 * @param g_logp_flat      [n_procs × total_events] row-major
 * @param g_toy_offsets    [n_toys + 1] prefix sums of per-toy event counts
 * @param g_yields_flat    [n_toys × n_procs] row-major: yields per toy per process
 * @param g_gauss          [n_gauss] Gaussian constraints
 * @param g_params_flat    [n_toys × n_params] row-major
 * @param g_nll_out        [n_toys] output NLL per toy
 * @param n_procs          number of processes
 * @param n_gauss          number of Gaussian constraints
 * @param constraint_const constant term from constraints
 * @param n_toys           number of toys (= gridDim.x)
 */
extern "C" __global__ void flow_batch_nll_reduce(
    const double* __restrict__ g_logp_flat,     /* [n_procs × total_events] */
    const unsigned int* __restrict__ g_toy_offsets, /* [n_toys + 1] */
    const double* __restrict__ g_yields_flat,   /* [n_toys × n_procs] */
    const struct FlowGaussConstraintEntry* __restrict__ g_gauss, /* [n_gauss] */
    const double* __restrict__ g_params_flat,   /* [n_toys × n_params] */
    double* __restrict__ g_nll_out,             /* [n_toys] */
    unsigned int n_procs,
    unsigned int n_params,
    unsigned int n_gauss,
    double constraint_const,
    unsigned int n_toys
) {
    unsigned int toy_idx = blockIdx.x;
    if (toy_idx >= n_toys) {
        return;
    }

    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;

    extern __shared__ double s_scratch[];

    unsigned int ev_start = g_toy_offsets[toy_idx];
    unsigned int ev_end = g_toy_offsets[toy_idx + 1];
    unsigned int n_events = ev_end - ev_start;

    const double* yields = &g_yields_flat[(size_t)toy_idx * n_procs];
    const double* params = &g_params_flat[(size_t)toy_idx * n_params];

    double local_sum_logf = 0.0;

    for (unsigned int i = tid; i < n_events; i += block_size) {
        unsigned int global_ev = ev_start + i;
        double max_term = -INFINITY;
        double sum_exp = 0.0;

        for (unsigned int p = 0; p < n_procs; p++) {
            double nu = yields[p];
            if (!(nu > 0.0) || !isfinite(nu)) {
                continue;
            }
            double logp = g_logp_flat[(size_t)p * (ev_end - g_toy_offsets[0]) + global_ev - g_toy_offsets[0]];
            if (!isfinite(logp)) {
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
            nu_tot += yields[p];
        }

        double constraint_sum = 0.0;
        for (unsigned int g = 0; g < n_gauss; g++) {
            double center = g_gauss[g].center;
            double inv_w = g_gauss[g].inv_width;
            unsigned int pidx = g_gauss[g].param_idx;
            double delta = (params[pidx] - center) * inv_w;
            constraint_sum += 0.5 * delta * delta;
        }

        g_nll_out[toy_idx] = nu_tot - sum_logf + constraint_sum + constraint_const;
    }
}
