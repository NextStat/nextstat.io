/**
 * Auxiliary kernels for cuBLAS-based batched GLM logistic evaluation.
 *
 * These kernels are intentionally small and simple:
 *  - glm_logistic_diff_nll: computes diff = sigmoid(eta)-y and accumulates
 *    data NLL per chain.
 *  - glm_add_prior: adds N(0,1) prior contributions to grad and NLL.
 */

extern "C" __global__ void glm_logistic_diff_nll(
    const double* __restrict__ eta,      // [n_chains * n]
    const double* __restrict__ y,        // [n]
    double* __restrict__ diff,           // [n_chains * n]
    double* __restrict__ nll_out,        // [n_chains]
    int n,
    int n_chains
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n_chains;
    if (idx >= total) return;

    int chain = idx / n;
    int i = idx - chain * n;
    double e = eta[idx];
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

extern "C" __global__ void glm_add_prior(
    const double* __restrict__ beta,     // [n_chains * p]
    double* __restrict__ grad,           // [n_chains * p]
    double* __restrict__ nll_out,        // [n_chains]
    int p,
    int n_chains
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = p * n_chains;
    if (idx >= total) return;

    int chain = idx / p;
    double b = beta[idx];
    grad[idx] += b;
    atomicAdd(&nll_out[chain], 0.5 * b * b);
}

