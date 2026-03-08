/**
 * Auxiliary kernels for cuBLAS-based interval-censored Weibull AFT evaluation.
 *
 * The shared flow is:
 *  - eta = X @ beta on cuBLAS
 *  - this kernel computes per-observation data NLL, d(nll)/d(log_lambda_i),
 *    and d(nll)/d(log_k)
 *  - grad_beta = X^T @ diff on cuBLAS
 *  - scatter beta gradients back into packed [log_k, beta...] storage
 *  - add the standard normal prior on all packed parameters
 */

#define CENSOR_EXACT 0
#define CENSOR_RIGHT 1
#define CENSOR_LEFT 2
#define CENSOR_INTERVAL 3

__device__ inline double device_log_diff_exp(double a, double b) {
    return a + log1p(-exp(b - a));
}

extern "C" __global__ void weibull_aft_diff_nll(
    const double* __restrict__ eta,             // [n_chains * n]
    const double* __restrict__ time_lower,      // [n]
    const double* __restrict__ time_upper,      // [n]
    const double* __restrict__ ln_time_lower,   // [n]
    const double* __restrict__ ln_time_upper,   // [n]
    const unsigned char* __restrict__ censor_code, // [n]
    const double* __restrict__ params,          // [n_chains * param_dim], [log_k, beta...]
    double* __restrict__ diff_log_lambda,       // [n_chains * n]
    double* __restrict__ grad,                  // [n_chains * param_dim]
    double* __restrict__ nll_out,               // [n_chains]
    int n,
    int beta_dim,
    int param_dim,
    int n_chains
) {
    (void)beta_dim;

    const double surv_diff_floor = 2e-131;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n_chains;
    if (idx >= total) return;

    int chain = idx / n;
    int i = idx - chain * n;

    double log_k_raw = params[chain * param_dim];
    double log_k = fmin(fmax(log_k_raw, -10.0), 10.0);
    double k = exp(log_k);
    double log_lam = eta[idx];
    double diff = 0.0;
    double grad_log_k = 0.0;
    double nll_i = 0.0;

    unsigned char code = censor_code[i];
    if (code == CENSOR_EXACT) {
        double ln_z = ln_time_lower[i] - log_lam;
        double a = exp(k * ln_z);
        nll_i = -log_k + log_lam - (k - 1.0) * ln_z + a;
        grad_log_k = -(1.0 + k * ln_z * (1.0 - a));
        diff = k * (1.0 - a);
    } else if (code == CENSOR_RIGHT) {
        if (time_lower[i] != 0.0) {
            double ln_z = ln_time_lower[i] - log_lam;
            double a = exp(k * ln_z);
            nll_i = a;
            grad_log_k = a * k * ln_z;
            diff = -a * k;
        }
    } else if (code == CENSOR_LEFT) {
        double ln_z = ln_time_upper[i] - log_lam;
        double a = exp(k * ln_z);
        double s = exp(-a);
        double log_f = fmax(log1p(-exp(-a)), -300.0);
        double ratio = (s > 1.0 - 1e-15) ? a : (s / (1.0 - s));
        nll_i = -log_f;
        grad_log_k = -ratio * a * k * ln_z;
        diff = ratio * a * k;
    } else if (code == CENSOR_INTERVAL) {
        double ln_z_u = ln_time_upper[i] - log_lam;
        double a_u = exp(k * ln_z_u);
        double s_u = exp(-a_u);
        if (time_lower[i] == 0.0) {
            double diff_surv = 1.0 - s_u;
            double log_f = fmax(log1p(-exp(-a_u)), -300.0);
            nll_i = -log_f;
            if (diff_surv > surv_diff_floor) {
                grad_log_k = -(s_u * a_u * k * ln_z_u) / diff_surv;
                diff = (s_u * a_u * k) / diff_surv;
            }
        } else {
            double ln_z_l = ln_time_lower[i] - log_lam;
            double a_l = exp(k * ln_z_l);
            double s_l = exp(-a_l);
            double diff_surv = s_l - s_u;
            double log_f = fmax(device_log_diff_exp(-a_l, -a_u), -300.0);
            nll_i = -log_f;
            if (diff_surv > surv_diff_floor) {
                grad_log_k = (s_l * a_l * k * ln_z_l - s_u * a_u * k * ln_z_u) / diff_surv;
                diff = (-s_l * a_l * k + s_u * a_u * k) / diff_surv;
            }
        }
    }

    diff_log_lambda[idx] = diff;
    atomicAdd(&grad[chain * param_dim], grad_log_k);
    atomicAdd(&nll_out[chain], nll_i);
}

extern "C" __global__ void weibull_aft_scatter_beta_grad(
    const double* __restrict__ beta_grad,   // [n_chains * beta_dim]
    double* __restrict__ grad,              // [n_chains * param_dim]
    int beta_dim,
    int param_dim,
    int n_chains
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = beta_dim * n_chains;
    if (idx >= total) return;

    int chain = idx / beta_dim;
    int j = idx - chain * beta_dim;
    grad[chain * param_dim + 1 + j] += beta_grad[idx];
}

extern "C" __global__ void weibull_aft_add_prior(
    const double* __restrict__ params,      // [n_chains * param_dim]
    double* __restrict__ grad,              // [n_chains * param_dim]
    double* __restrict__ nll_out,           // [n_chains]
    int param_dim,
    int n_chains
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = param_dim * n_chains;
    if (idx >= total) return;

    int chain = idx / param_dim;
    double value = params[idx];
    grad[idx] += value;
    atomicAdd(&nll_out[chain], 0.5 * value * value);
}
