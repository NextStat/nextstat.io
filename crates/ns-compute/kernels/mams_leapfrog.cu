/**
 * MAMS Leapfrog CUDA Kernel — hardcoded benchmark models for LAPS.
 *
 * This file defines 4 built-in models and dispatches to them via __mams_model_id.
 * The engine (RNG, integrator, kernel) lives in mams_engine.cuh.
 *
 * For user-defined models, the JIT path (NVRTC) concatenates user code with
 * mams_engine.cuh directly, bypassing this file entirely.
 *
 * Precision: NO --use_fast_math. MAMS energy conservation requires precise
 * exp/log/sqrt for MH detailed balance.
 */

/* ---------- Model gradient functions ------------------------------------- */

// Model 0: Standard normal in d dimensions.
// NLL = 0.5 * sum(x_i^2), grad = x_i
__device__ void grad_std_normal(
    const double* x, double* grad, int dim,
    const double* /*model_data*/)
{
    for (int i = 0; i < dim; i++) {
        grad[i] = x[i];
    }
}

__device__ double nll_std_normal(
    const double* x, int dim,
    const double* /*model_data*/)
{
    double nll = 0.0;
    for (int i = 0; i < dim; i++) {
        nll += 0.5 * x[i] * x[i];
    }
    return nll;
}

// Model 1: Eight Schools (non-centered).
// params: [mu, tau, theta_raw_0..J-1]  (dim = 2+J)
// model_data: [J, y_0..y_{J-1}, inv_var_0..inv_var_{J-1}, prior_mu_sigma, prior_tau_scale]
__device__ void grad_eight_schools(
    const double* x, double* grad, int dim,
    const double* model_data)
{
    int J = (int)model_data[0];
    const double* y = model_data + 1;
    const double* inv_var = model_data + 1 + J;
    double prior_mu_sigma = model_data[1 + 2 * J];
    double prior_tau_scale = model_data[1 + 2 * J + 1];

    double mu = x[0];
    double tau = x[1];

    double d_mu = 0.0;
    double d_tau = 0.0;

    for (int i = 0; i < J; i++) {
        double theta_raw = x[2 + i];
        double theta = mu + tau * theta_raw;
        double r = y[i] - theta;
        // d/d(theta_raw)
        grad[2 + i] = -r * tau * inv_var[i] + theta_raw;
        // accumulate d/d(mu) and d/d(tau)
        d_mu -= r * inv_var[i];
        d_tau -= r * theta_raw * inv_var[i];
    }

    // Prior on mu: N(0, prior_mu_sigma)
    d_mu += mu / (prior_mu_sigma * prior_mu_sigma);
    grad[0] = d_mu;

    // Prior on tau: HalfCauchy(0, prior_tau_scale)
    double s2 = prior_tau_scale * prior_tau_scale;
    d_tau += 2.0 * tau / (s2 + tau * tau);
    grad[1] = d_tau;
}

__device__ double nll_eight_schools(
    const double* x, int dim,
    const double* model_data)
{
    int J = (int)model_data[0];
    const double* y = model_data + 1;
    const double* inv_var = model_data + 1 + J;
    double prior_mu_sigma = model_data[1 + 2 * J];
    double prior_tau_scale = model_data[1 + 2 * J + 1];

    double mu = x[0];
    double tau = x[1];
    double nll = 0.0;

    for (int i = 0; i < J; i++) {
        double theta_raw = x[2 + i];
        double theta = mu + tau * theta_raw;
        double r = y[i] - theta;
        nll += 0.5 * r * r * inv_var[i];
        nll += 0.5 * theta_raw * theta_raw;
    }
    nll += 0.5 * (mu / prior_mu_sigma) * (mu / prior_mu_sigma);
    double t = tau / prior_tau_scale;
    nll += log(1.0 + t * t);

    return nll;
}

// Model 2: Neal's funnel (dim = d).
// params: [v, x_1, ..., x_{d-1}]
// v ~ N(0, 9),  x_i | v ~ N(0, exp(v))
// NLL = v^2/18 + (d-1)/2 * v + 0.5 * exp(-v) * sum(x_i^2)
__device__ void grad_neal_funnel(
    const double* x, double* grad, int dim,
    const double* /*model_data*/)
{
    double v = x[0];
    double exp_neg_v = exp(-v);
    double sum_x2 = 0.0;
    for (int i = 1; i < dim; i++) {
        sum_x2 += x[i] * x[i];
    }

    // d/dv = v/9 + (d-1)/2 - 0.5 * exp(-v) * sum(x_i^2)
    grad[0] = v / 9.0 + (dim - 1) * 0.5 - 0.5 * exp_neg_v * sum_x2;

    // d/dx_i = exp(-v) * x_i
    for (int i = 1; i < dim; i++) {
        grad[i] = exp_neg_v * x[i];
    }
}

__device__ double nll_neal_funnel(
    const double* x, int dim,
    const double* /*model_data*/)
{
    double v = x[0];
    double sum_x2 = 0.0;
    for (int i = 1; i < dim; i++) {
        sum_x2 += x[i] * x[i];
    }
    return v * v / 18.0 + (dim - 1) * 0.5 * v + 0.5 * exp(-v) * sum_x2;
}

// Model 3: GLM logistic regression.
// params: [beta_0, beta_1, ..., beta_{p-1}]
// model_data: [n, p, X_{n*p row-major}, y_n]
// NLL = sum_i [ log(1 + exp(eta_i)) - y_i * eta_i ]  where eta_i = X_i . beta
__device__ void grad_glm_logistic(
    const double* beta, double* grad, int dim,
    const double* model_data)
{
    int n = (int)model_data[0];
    int p = (int)model_data[1];
    const double* X = model_data + 2;
    const double* y = model_data + 2 + n * p;

    for (int j = 0; j < p; j++) {
        grad[j] = 0.0;
    }

    for (int i = 0; i < n; i++) {
        // Compute eta = X[i,:] . beta
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            eta += X[(size_t)i * p + j] * beta[j];
        }
        // Stable sigmoid: prob = 1/(1+exp(-eta))
        double prob;
        if (eta >= 0.0) {
            double e = exp(-eta);
            prob = 1.0 / (1.0 + e);
        } else {
            double e = exp(eta);
            prob = e / (1.0 + e);
        }
        // grad_j += (prob - y_i) * X[i,j]
        double diff = prob - y[i];
        for (int j = 0; j < p; j++) {
            grad[j] += diff * X[(size_t)i * p + j];
        }
    }
}

__device__ double nll_glm_logistic(
    const double* beta, int dim,
    const double* model_data)
{
    int n = (int)model_data[0];
    int p = (int)model_data[1];
    const double* X = model_data + 2;
    const double* y = model_data + 2 + n * p;
    double nll = 0.0;

    for (int i = 0; i < n; i++) {
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            eta += X[(size_t)i * p + j] * beta[j];
        }
        // Stable: log(1+exp(eta)) = max(eta,0) + log(1+exp(-|eta|))
        double abs_eta = fabs(eta);
        nll += fmax(eta, 0.0) + log(1.0 + exp(-abs_eta)) - y[i] * eta;
    }
    return nll;
}

/* ---------- Model dispatch → user_nll / user_grad ----------------------- */

/* The engine header (mams_engine.cuh) calls user_nll/user_grad with a fixed
 * 3-argument signature. For the build-time path, we dispatch on __mams_model_id
 * (set by mams_transition at kernel entry from the model_id argument).
 * We define it here before #include to make it visible to dispatch functions. */
#define MAMS_MODEL_ID_DEFINED
static __device__ int __mams_model_id;

__device__ double user_nll(const double* x, int dim, const double* model_data) {
    switch (__mams_model_id) {
        case 0: return nll_std_normal(x, dim, model_data);
        case 1: return nll_eight_schools(x, dim, model_data);
        case 2: return nll_neal_funnel(x, dim, model_data);
        case 3: return nll_glm_logistic(x, dim, model_data);
        default: return 1e30;
    }
}

__device__ void user_grad(const double* x, double* grad, int dim, const double* model_data) {
    switch (__mams_model_id) {
        case 0: grad_std_normal(x, grad, dim, model_data); break;
        case 1: grad_eight_schools(x, grad, dim, model_data); break;
        case 2: grad_neal_funnel(x, grad, dim, model_data); break;
        case 3: grad_glm_logistic(x, grad, dim, model_data); break;
    }
}

/* ---------- Engine (RNG + integrator + kernel) --------------------------- */

/* __mams_model_id is declared in the engine header as a forward reference.
 * We defined user_nll/user_grad above, which the engine kernel calls. */
#include "mams_engine.cuh"
