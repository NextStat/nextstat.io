/**
 * Auxiliary kernels for cuBLAS-based batched GLM evaluation.
 *
 * These kernels are intentionally small and simple:
 *  - `glm_*_diff_nll` computes `diff`, family-specific data NLL, and any
 *    trailing scalar likelihood gradients that are not recoverable from the
 *    shared `X^T @ diff` GEMM.
 *  - `glm_add_prior` adds N(0,1) prior contributions to the full packed
 *    parameter vector for each chain.
 */

__device__ inline double device_digamma(double x) {
    double result = 0.0;
    while (x < 8.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    double inv_x = 1.0 / x;
    double inv_x2 = inv_x * inv_x;
    result += log(x)
        - 0.5 * inv_x
        - inv_x2
            * (1.0 / 12.0
                - inv_x2
                    * (1.0 / 120.0
                        - inv_x2
                            * (1.0 / 252.0 - inv_x2 * (1.0 / 240.0 - inv_x2 * (1.0 / 132.0)))));
    return result;
}

extern "C" __global__ void glm_logistic_diff_nll(
    const double* __restrict__ eta,       // [n_chains * n]
    const double* __restrict__ y,         // [n]
    const double* __restrict__ offset,    // [n]
    const double* __restrict__ params,    // [n_chains * param_dim]
    double* __restrict__ diff,            // [n_chains * n]
    double* __restrict__ grad,            // [n_chains * param_dim]
    double* __restrict__ nll_out,         // [n_chains]
    int n,
    int beta_dim,
    int param_dim,
    int n_chains
) {
    (void)params;
    (void)grad;
    (void)beta_dim;
    (void)param_dim;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n_chains;
    if (idx >= total) return;

    int chain = idx / n;
    int i = idx - chain * n;
    double e = eta[idx] + offset[i];
    double yi = y[i];

    double p;
    if (e >= 0.0) {
        double z = exp(-e);
        p = 1.0 / (1.0 + z);
    } else {
        double z = exp(e);
        p = z / (1.0 + z);
    }
    diff[idx] = p - yi;

    double abs_e = fabs(e);
    double ll_i = fmax(e, 0.0) + log(1.0 + exp(-abs_e)) - yi * e;
    atomicAdd(&nll_out[chain], ll_i);
}

extern "C" __global__ void glm_linear_diff_nll(
    const double* __restrict__ eta,       // [n_chains * n]
    const double* __restrict__ y,         // [n]
    const double* __restrict__ offset,    // [n]
    const double* __restrict__ params,    // [n_chains * param_dim]
    double* __restrict__ diff,            // [n_chains * n]
    double* __restrict__ grad,            // [n_chains * param_dim]
    double* __restrict__ nll_out,         // [n_chains]
    int n,
    int beta_dim,
    int param_dim,
    int n_chains
) {
    (void)offset;
    (void)params;
    (void)grad;
    (void)beta_dim;
    (void)param_dim;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n_chains;
    if (idx >= total) return;

    int chain = idx / n;
    int i = idx - chain * n;
    double residual = eta[idx] - y[i];
    diff[idx] = residual;
    atomicAdd(&nll_out[chain], 0.5 * residual * residual);
}

extern "C" __global__ void glm_poisson_diff_nll(
    const double* __restrict__ eta,       // [n_chains * n]
    const double* __restrict__ y,         // [n]
    const double* __restrict__ offset,    // [n]
    const double* __restrict__ params,    // [n_chains * param_dim]
    double* __restrict__ diff,            // [n_chains * n]
    double* __restrict__ grad,            // [n_chains * param_dim]
    double* __restrict__ nll_out,         // [n_chains]
    int n,
    int beta_dim,
    int param_dim,
    int n_chains
) {
    (void)params;
    (void)grad;
    (void)beta_dim;
    (void)param_dim;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n_chains;
    if (idx >= total) return;

    int chain = idx / n;
    int i = idx - chain * n;
    double e = eta[idx] + offset[i];
    double yi = y[i];
    double e_clamped = fmin(fmax(e, -50.0), 50.0);
    double mu = exp(e_clamped);

    diff[idx] = mu - yi;
    atomicAdd(&nll_out[chain], mu - yi * e_clamped);
}

extern "C" __global__ void glm_negbin_diff_nll(
    const double* __restrict__ eta,       // [n_chains * n]
    const double* __restrict__ y,         // [n]
    const double* __restrict__ offset,    // [n]
    const double* __restrict__ params,    // [n_chains * param_dim]
    double* __restrict__ diff,            // [n_chains * n]
    double* __restrict__ grad,            // [n_chains * param_dim]
    double* __restrict__ nll_out,         // [n_chains]
    int n,
    int beta_dim,
    int param_dim,
    int n_chains
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n_chains;
    if (idx >= total) return;

    int chain = idx / n;
    int i = idx - chain * n;

    double log_alpha_raw = params[chain * param_dim + beta_dim];
    double log_alpha = fmin(fmax(log_alpha_raw, -10.0), 8.0);
    double alpha = exp(log_alpha);
    double theta = 1.0 / alpha;

    double e = eta[idx] + offset[i];
    double yi = y[i];
    double e_clamped = fmin(fmax(e, -50.0), 50.0);
    double mu = exp(e_clamped);
    double denom = theta + mu;

    diff[idx] = mu * (theta + yi) / denom - yi;

    double ll_i = -(lgamma(yi + theta) - lgamma(theta)
        + theta * log(theta / denom)
        + yi * log(mu / denom));
    atomicAdd(&nll_out[chain], ll_i);

    if (log_alpha_raw > -10.0 && log_alpha_raw < 8.0) {
        double psi_yi_theta = device_digamma(yi + theta);
        double psi_theta = device_digamma(theta);
        double d_theta = -(psi_yi_theta - psi_theta + log(theta / denom) + 1.0 - (theta + yi) / denom);
        atomicAdd(&grad[chain * param_dim + beta_dim], d_theta * (-theta));
    }
}

extern "C" __global__ void glm_add_prior(
    const double* __restrict__ params,    // [n_chains * param_dim]
    double* __restrict__ grad,            // [n_chains * param_dim]
    double* __restrict__ nll_out,         // [n_chains]
    int param_dim,
    int n_chains
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = param_dim * n_chains;
    if (idx >= total) return;

    int chain = idx / param_dim;
    double b = params[idx];
    grad[idx] += b;
    atomicAdd(&nll_out[chain], 0.5 * b * b);
}
