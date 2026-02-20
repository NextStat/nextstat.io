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

__device__ double grad_nll_std_normal(
    const double* x, double* grad, int dim,
    const double* /*model_data*/)
{
    double nll = 0.0;
    for (int i = 0; i < dim; i++) {
        double xi = x[i];
        grad[i] = xi;
        nll += 0.5 * xi * xi;
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

// Model 3: GLM logistic regression with N(0,1) prior on beta.
// params: [beta_0, beta_1, ..., beta_{p-1}]
// model_data: [n, p, X_row(n*p), y(n), X_col(n*p)]
// NLL = sum_i [ log(1 + exp(eta_i)) - y_i * eta_i ] + 0.5 * sum_j(beta_j^2)
__device__ void grad_glm_logistic(
    const double* beta, double* grad, int dim,
    const double* model_data)
{
    int n = (int)model_data[0];
    int p = (int)model_data[1];
    const double* X = model_data + 2;
    const double* y = model_data + 2 + n * p;

    // Prior gradient: d/d(beta_j) [0.5 * beta_j^2] = beta_j
    for (int j = 0; j < p; j++) {
        grad[j] = beta[j];
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

    // Prior: 0.5 * sum(beta_j^2) — N(0,1)
    double nll = 0.0;
    for (int j = 0; j < p; j++) {
        nll += 0.5 * beta[j] * beta[j];
    }

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

// Model 6: GLM linear regression with N(0,1) prior on beta, σ=1 fixed.
// params: [beta_0, ..., beta_{p-1}]
// model_data: [n, p, X_row(n*p), y(n), X_col(n*p)]
// NLL = 0.5 * sum_i(eta_i - y_i)^2 + 0.5 * sum_j(beta_j^2)
__device__ void grad_glm_linear(
    const double* beta, double* grad, int dim,
    const double* model_data)
{
    int n = (int)model_data[0];
    int p = (int)model_data[1];
    const double* X = model_data + 2;
    const double* y = model_data + 2 + n * p;

    for (int j = 0; j < p; j++) {
        grad[j] = beta[j]; // prior
    }

    for (int i = 0; i < n; i++) {
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            eta += X[(size_t)i * p + j] * beta[j];
        }
        double residual = eta - y[i];
        for (int j = 0; j < p; j++) {
            grad[j] += residual * X[(size_t)i * p + j];
        }
    }
}

__device__ double nll_glm_linear(
    const double* beta, int dim,
    const double* model_data)
{
    int n = (int)model_data[0];
    int p = (int)model_data[1];
    const double* X = model_data + 2;
    const double* y = model_data + 2 + n * p;

    double nll = 0.0;
    for (int j = 0; j < p; j++) {
        nll += 0.5 * beta[j] * beta[j];
    }

    for (int i = 0; i < n; i++) {
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            eta += X[(size_t)i * p + j] * beta[j];
        }
        double r = eta - y[i];
        nll += 0.5 * r * r;
    }
    return nll;
}

// Model 7: GLM Poisson regression with N(0,1) prior on beta, log link.
// params: [beta_0, ..., beta_{p-1}]
// model_data: [n, p, offset_flag, X_row(n*p), y(n), X_col(n*p), offset(n if flag=1)]
// NLL = sum_i[exp(eta_i) - y_i * eta_i] + 0.5 * sum_j(beta_j^2)
__device__ void grad_glm_poisson(
    const double* beta, double* grad, int dim,
    const double* model_data)
{
    int n = (int)model_data[0];
    int p = (int)model_data[1];
    int offset_flag = (int)model_data[2];
    const double* X = model_data + 3;
    const double* y = model_data + 3 + n * p;
    const double* X_col = model_data + 3 + n * p + n;
    const double* offset = (offset_flag == 1) ? (model_data + 3 + n * p + n + n * p) : 0;

    for (int j = 0; j < p; j++) {
        grad[j] = beta[j]; // prior
    }

    for (int i = 0; i < n; i++) {
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            eta += X[(size_t)i * p + j] * beta[j];
        }
        if (offset_flag == 1) eta += offset[i];
        // Clamp eta to prevent overflow in exp()
        if (eta > 50.0) eta = 50.0;
        if (eta < -50.0) eta = -50.0;
        double mu = exp(eta);
        double diff = mu - y[i];
        for (int j = 0; j < p; j++) {
            grad[j] += diff * X[(size_t)i * p + j];
        }
    }
}

__device__ double nll_glm_poisson(
    const double* beta, int dim,
    const double* model_data)
{
    int n = (int)model_data[0];
    int p = (int)model_data[1];
    int offset_flag = (int)model_data[2];
    const double* X = model_data + 3;
    const double* y = model_data + 3 + n * p;
    const double* offset = (offset_flag == 1) ? (model_data + 3 + n * p + n + n * p) : 0;

    double nll = 0.0;
    for (int j = 0; j < p; j++) {
        nll += 0.5 * beta[j] * beta[j];
    }

    for (int i = 0; i < n; i++) {
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            eta += X[(size_t)i * p + j] * beta[j];
        }
        if (offset_flag == 1) eta += offset[i];
        if (eta > 50.0) eta = 50.0;
        if (eta < -50.0) eta = -50.0;
        nll += exp(eta) - y[i] * eta;
    }
    return nll;
}

// Digamma function: asymptotic expansion with recursion for x < 8.
// Accuracy ~1e-12 for x > 0. Uses ψ(x+1) = ψ(x) + 1/x recurrence.
__device__ double device_digamma(double x) {
    double result = 0.0;
    // Shift x up to ≥ 8 using recurrence ψ(x) = ψ(x+1) - 1/x
    while (x < 8.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic expansion for x ≥ 8
    double inv_x = 1.0 / x;
    double inv_x2 = inv_x * inv_x;
    result += log(x) - 0.5 * inv_x
        - inv_x2 * (1.0/12.0 - inv_x2 * (1.0/120.0 - inv_x2 * (1.0/252.0
        - inv_x2 * (1.0/240.0 - inv_x2 * (1.0/132.0)))));
    return result;
}

// Model 8: GLM Negative Binomial regression with N(0,1) prior on beta and log_alpha.
// params: [beta_0, ..., beta_{p-1}, log_alpha]  (dim = p+1)
// model_data: [n, p, offset_flag, X_row(n*p), y(n), X_col(n*p), offset(n if flag=1)]
// alpha = exp(clamp(log_alpha, -10, 8)), theta = 1/alpha
// NLL = -sum_i[lgamma(y_i+theta) - lgamma(theta) + theta*ln(theta/(theta+mu))
//        + y_i*ln(mu/(theta+mu))] + 0.5*sum(beta_j^2) + 0.5*log_alpha^2
__device__ void grad_glm_negbin(
    const double* x, double* grad, int dim,
    const double* model_data)
{
    int n = (int)model_data[0];
    int p = (int)model_data[1];
    int offset_flag = (int)model_data[2];
    const double* X = model_data + 3;
    const double* y = model_data + 3 + n * p;
    const double* offset = (offset_flag == 1) ? (model_data + 3 + n * p + n + n * p) : 0;

    const double* beta = x;
    double log_alpha = x[p];
    double log_alpha_c = fmax(fmin(log_alpha, 8.0), -10.0);
    double alpha = exp(log_alpha_c);
    double theta = 1.0 / alpha;

    // Prior gradients
    for (int j = 0; j < p; j++) {
        grad[j] = beta[j];
    }
    grad[p] = log_alpha; // prior on log_alpha

    double d_log_alpha = 0.0;

    for (int i = 0; i < n; i++) {
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            eta += X[(size_t)i * p + j] * beta[j];
        }
        if (offset_flag == 1) eta += offset[i];
        if (eta > 50.0) eta = 50.0;
        if (eta < -50.0) eta = -50.0;

        double mu = exp(eta);
        double yi = y[i];
        double denom = theta + mu;

        // Gradient w.r.t. beta_j: d(NLL)/d(beta_j) = sum_i (mu*(yi+theta)/denom - yi) * X_ij / denom * mu ...
        // Simplified: d(NLL)/d(eta_i) = mu * (1 - (yi + theta)/denom) = mu * (mu - yi) / denom
        // No — correct form:
        // NLL_i = -[lgamma(yi+theta) - lgamma(theta) + theta*ln(theta/denom) + yi*ln(mu/denom)]
        // d(NLL_i)/d(eta_i) = -(yi * mu/denom - yi*mu/denom + ...)
        // = -(yi/mu * mu - yi*mu/denom) ... let's derive carefully:
        // d(NLL_i)/d(mu) = -(theta * (-1/denom) + yi * (1/mu - 1/denom)) * (dmu/deta)  — wait,
        // d(NLL_i)/d(eta_i) = d(NLL_i)/d(mu_i) * d(mu_i)/d(eta_i) = d(NLL_i)/d(mu_i) * mu_i
        // d(NLL_i)/d(mu_i) = -(theta*(-1/denom) + yi*(1/mu - 1/denom))
        //                   = -(- theta/denom + yi/mu - yi/denom)
        //                   = theta/denom - yi/mu + yi/denom
        //                   = (theta + yi)/denom - yi/mu
        // d(NLL_i)/d(eta_i) = mu * ((theta + yi)/denom - yi/mu) = mu*(theta+yi)/denom - yi
        double d_eta = mu * (theta + yi) / denom - yi;
        for (int j = 0; j < p; j++) {
            grad[j] += d_eta * X[(size_t)i * p + j];
        }

        // Gradient w.r.t. log_alpha: chain rule through alpha → theta
        // d(theta)/d(log_alpha) = d(1/alpha)/d(alpha) * d(alpha)/d(log_alpha) = -1/alpha^2 * alpha = -theta
        // d(NLL_i)/d(theta) = -(digamma(yi+theta) - digamma(theta) + ln(theta/denom) + theta*(1/theta - 1/denom) - yi/denom)
        //                   = -(digamma(yi+theta) - digamma(theta) + ln(theta/denom) + 1 - theta/denom - yi/denom)
        //                   = -(digamma(yi+theta) - digamma(theta) + ln(theta/denom) + 1 - (theta+yi)/denom)
        double psi_yi_theta = device_digamma(yi + theta);
        double psi_theta = device_digamma(theta);
        double d_theta = -(psi_yi_theta - psi_theta + log(theta / denom) + 1.0 - (theta + yi) / denom);
        // d(NLL_i)/d(log_alpha) = d(NLL_i)/d(theta) * d(theta)/d(log_alpha) = d_theta * (-theta)
        d_log_alpha += d_theta * (-theta);
    }

    // When log_alpha is outside the clamp range [-10, 8], the likelihood NLL
    // is flat (uses clamped value), so the true likelihood gradient is zero.
    // Only the prior gradient (log_alpha) remains.  Match CPU reference.
    if (log_alpha <= -10.0 || log_alpha >= 8.0) {
        d_log_alpha = 0.0;
    }

    grad[p] += d_log_alpha;
}

__device__ double nll_glm_negbin(
    const double* x, int dim,
    const double* model_data)
{
    int n = (int)model_data[0];
    int p = (int)model_data[1];
    int offset_flag = (int)model_data[2];
    const double* X = model_data + 3;
    const double* y = model_data + 3 + n * p;
    const double* offset = (offset_flag == 1) ? (model_data + 3 + n * p + n + n * p) : 0;

    const double* beta = x;
    double log_alpha = x[p];
    double log_alpha_c = fmax(fmin(log_alpha, 8.0), -10.0);
    double alpha = exp(log_alpha_c);
    double theta = 1.0 / alpha;

    double nll = 0.0;
    // Prior
    for (int j = 0; j < p; j++) {
        nll += 0.5 * beta[j] * beta[j];
    }
    nll += 0.5 * log_alpha * log_alpha;

    for (int i = 0; i < n; i++) {
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            eta += X[(size_t)i * p + j] * beta[j];
        }
        if (offset_flag == 1) eta += offset[i];
        if (eta > 50.0) eta = 50.0;
        if (eta < -50.0) eta = -50.0;

        double mu = exp(eta);
        double yi = y[i];
        double denom = theta + mu;

        // -[lgamma(yi+theta) - lgamma(theta) + theta*ln(theta/denom) + yi*ln(mu/denom)]
        nll -= lgamma(yi + theta) - lgamma(theta) + theta * log(theta / denom) + yi * log(mu / denom);
    }
    return nll;
}

// Model 9: Composed GLM logistic with random intercept (NCP).
// params: [beta_0..beta_{p-1}, z_0..z_{G-1}], dim = p + G
// model_data: [n, p, G, X_row(n*p), y(n), group_idx(n), X_col(n*p), re_prior_sigma]
// eta_i = X[i,:].beta + re_scale * z_{group_idx[i]}
// NLL = logistic_likelihood(eta, y) + 0.5*sum(beta_j^2) + 0.5*sum(z_j^2)
__device__ void grad_glm_composed_logistic(
    const double* x, double* grad, int dim,
    const double* model_data)
{
    int n = (int)model_data[0];
    int p = (int)model_data[1];
    int G = (int)model_data[2];
    const double* X = model_data + 3;
    const double* y = model_data + 3 + n * p;
    const double* grp = model_data + 3 + n * p + n;
    // re_prior_sigma is at end: model_data + 3 + n*p + n + n + n*p
    double re_scale = model_data[3 + n * p + n + n + n * p];

    const double* beta = x;
    const double* z_re = x + p;

    // Prior gradient: beta_j, z_j
    for (int j = 0; j < p; j++) {
        grad[j] = beta[j];
    }
    for (int j = 0; j < G; j++) {
        grad[p + j] = z_re[j]; // N(0,1) prior on z
    }

    for (int i = 0; i < n; i++) {
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            eta += X[(size_t)i * p + j] * beta[j];
        }
        int gi = (int)grp[i];
        if (gi >= 0 && gi < G) {
            eta += re_scale * z_re[gi];
        }

        double prob;
        if (eta >= 0.0) {
            double e = exp(-eta);
            prob = 1.0 / (1.0 + e);
        } else {
            double e = exp(eta);
            prob = e / (1.0 + e);
        }
        double diff = prob - y[i];
        for (int j = 0; j < p; j++) {
            grad[j] += diff * X[(size_t)i * p + j];
        }
        if (gi >= 0 && gi < G) {
            grad[p + gi] += diff * re_scale;
        }
    }
}

__device__ double nll_glm_composed_logistic(
    const double* x, int dim,
    const double* model_data)
{
    int n = (int)model_data[0];
    int p = (int)model_data[1];
    int G = (int)model_data[2];
    const double* X = model_data + 3;
    const double* y = model_data + 3 + n * p;
    const double* grp = model_data + 3 + n * p + n;
    double re_scale = model_data[3 + n * p + n + n + n * p];

    const double* beta = x;
    const double* z_re = x + p;

    double nll = 0.0;
    // Prior: 0.5 * sum(beta_j^2) + 0.5 * sum(z_j^2)
    for (int j = 0; j < p; j++) {
        nll += 0.5 * beta[j] * beta[j];
    }
    for (int j = 0; j < G; j++) {
        nll += 0.5 * z_re[j] * z_re[j];
    }

    for (int i = 0; i < n; i++) {
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            eta += X[(size_t)i * p + j] * beta[j];
        }
        int gi = (int)grp[i];
        if (gi >= 0 && gi < G) {
            eta += re_scale * z_re[gi];
        }
        double abs_eta = fabs(eta);
        nll += fmax(eta, 0.0) + log(1.0 + exp(-abs_eta)) - y[i] * eta;
    }
    return nll;
}

// Model 4: Neal's funnel non-centered parameterization.
// params: [v, z_1, ..., z_{d-1}]
// v ~ N(0, 9),  x_i = exp(v/2) * z_i, z_i ~ N(0,1)
// NLL = v^2/18 + 0.5 * sum(z_i^2)
__device__ void grad_neal_funnel_ncp(
    const double* x, double* grad, int dim,
    const double* /*model_data*/)
{
    grad[0] = x[0] / 9.0;
    for (int i = 1; i < dim; i++) {
        grad[i] = x[i];
    }
}

__device__ double nll_neal_funnel_ncp(
    const double* x, int dim,
    const double* /*model_data*/)
{
    double nll = x[0] * x[0] / 18.0;
    for (int i = 1; i < dim; i++) {
        nll += 0.5 * x[i] * x[i];
    }
    return nll;
}

// Model 5: Neal's funnel — Riemannian MAMS with Fisher metric.
// Uses effective potential U_eff = v^2/18 + 0.5*exp(-v)*sum(x_i^2).
// The (d-1)*v/2 terms from U and 0.5*ln|det G| cancel.
// Gradient of U_eff: d/dv = v/9 - 0.5*exp(-v)*sum(x_i^2), d/dx_i = exp(-v)*x_i.
__device__ void grad_neal_funnel_riemannian(
    const double* x, double* grad, int dim,
    const double* /*model_data*/)
{
    double v = x[0];
    double v_clamped = fmax(fmin(v, 20.0), -20.0);
    double exp_neg_v = exp(-v_clamped);
    double sum_x2 = 0.0;
    for (int i = 1; i < dim; i++) {
        sum_x2 += x[i] * x[i];
    }
    // Full NLL gradient (same as centered funnel) — correct target distribution
    grad[0] = v / 9.0 + (dim - 1) * 0.5 - 0.5 * exp_neg_v * sum_x2;
    for (int i = 1; i < dim; i++) {
        grad[i] = exp_neg_v * x[i];
    }
}

__device__ double nll_neal_funnel_riemannian(
    const double* x, int dim,
    const double* /*model_data*/)
{
    double v = x[0];
    double v_clamped = fmax(fmin(v, 20.0), -20.0);
    double sum_x2 = 0.0;
    for (int i = 1; i < dim; i++) {
        sum_x2 += x[i] * x[i];
    }
    // Full NLL (same as centered funnel) — Riemannian metric only affects dynamics
    return v * v / 18.0 + (dim - 1) * 0.5 * v + 0.5 * exp(-v_clamped) * sum_x2;
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
        case 4: return nll_neal_funnel_ncp(x, dim, model_data);
        case 5: return nll_neal_funnel_riemannian(x, dim, model_data);
        case 6: return nll_glm_linear(x, dim, model_data);
        case 7: return nll_glm_poisson(x, dim, model_data);
        case 8: return nll_glm_negbin(x, dim, model_data);
        case 9: return nll_glm_composed_logistic(x, dim, model_data);
        default: return 1e30;
    }
}

__device__ void user_grad(const double* x, double* grad, int dim, const double* model_data) {
    switch (__mams_model_id) {
        case 0: grad_std_normal(x, grad, dim, model_data); break;
        case 1: grad_eight_schools(x, grad, dim, model_data); break;
        case 2: grad_neal_funnel(x, grad, dim, model_data); break;
        case 3: grad_glm_logistic(x, grad, dim, model_data); break;
        case 4: grad_neal_funnel_ncp(x, grad, dim, model_data); break;
        case 5: grad_neal_funnel_riemannian(x, grad, dim, model_data); break;
        case 6: grad_glm_linear(x, grad, dim, model_data); break;
        case 7: grad_glm_poisson(x, grad, dim, model_data); break;
        case 8: grad_glm_negbin(x, grad, dim, model_data); break;
        case 9: grad_glm_composed_logistic(x, grad, dim, model_data); break;
    }
}

// Optional fused grad+nll callback for non-warp paths in mams_engine.cuh.
#define MAMS_FUSED_GRAD_NLL_DEFINED
__device__ double user_grad_nll(
    const double* x, double* grad, int dim, const double* model_data)
{
    switch (__mams_model_id) {
        case 0:
            return grad_nll_std_normal(x, grad, dim, model_data);
        default:
            user_grad(x, grad, dim, model_data);
            return user_nll(x, dim, model_data);
    }
}

/* ---------- Warp-cooperative model functions (1 warp = 1 chain) ---------- */

/*
 * For data-heavy models (GLM logistic: N=200 obs), the serial observation
 * loop is the bottleneck. These functions distribute it across 32 warp lanes:
 *   - Each lane processes observations [lane_id, lane_id+32, lane_id+64, ...]
 *   - __shfl_down_sync reduces partial sums within the warp (5 clock cycles)
 *   - __shfl_sync broadcasts the result to all lanes
 *
 * The shared memory variant loads X in column-major for coalesced access.
 * Layout: s_data = [X_col_major (p × n doubles), y (n doubles)]
 * The host must pass shared_mem_bytes = (n*p + n) * sizeof(double).
 */
// Warp-cooperative GLM logistic gradient (shared memory, column-major X).
// Includes N(0,1) prior on beta: grad_j += beta_j.
// s_X_col: column-major X [p][n], s_y: response [n], loaded by caller.
__device__ void grad_glm_logistic_warp_shmem(
    const double* beta, double* grad, int dim,
    const double* s_X_col, const double* s_y,
    int n, int p, int lane_id)
{
    // Prior gradient: beta_j (only lane 0 initializes to avoid double-counting)
    for (int j = 0; j < p; j++) grad[j] = 0.0;

    for (int i = lane_id; i < n; i += 32) {
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            eta += s_X_col[j * n + i] * beta[j];   // coalesced: stride-1 across lanes
        }
        double prob;
        if (eta >= 0.0) {
            double e = exp(-eta);
            prob = 1.0 / (1.0 + e);
        } else {
            double e = exp(eta);
            prob = e / (1.0 + e);
        }
        double diff = prob - s_y[i];                // coalesced
        for (int j = 0; j < p; j++) {
            grad[j] += diff * s_X_col[j * n + i];  // coalesced
        }
    }

    // Warp reduction + broadcast (register-only, no shared memory needed)
    for (int j = 0; j < p; j++) {
        double g = grad[j];
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            g += __shfl_down_sync(0xffffffff, g, offset);
        }
        // After reduction, add prior gradient (only once, post-reduction)
        g = __shfl_sync(0xffffffff, g, 0) + beta[j];  // broadcast + prior
        grad[j] = g;
    }
}

// Warp-cooperative GLM logistic NLL (shared memory, column-major X).
// Includes N(0,1) prior: NLL += 0.5 * sum(beta_j^2).
__device__ double nll_glm_logistic_warp_shmem(
    const double* beta, int dim,
    const double* s_X_col, const double* s_y,
    int n, int p, int lane_id)
{
    double nll = 0.0;

    for (int i = lane_id; i < n; i += 32) {
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            eta += s_X_col[j * n + i] * beta[j];
        }
        double abs_eta = fabs(eta);
        nll += fmax(eta, 0.0) + log(1.0 + exp(-abs_eta)) - s_y[i] * eta;
    }

    // Warp reduction + broadcast
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        nll += __shfl_down_sync(0xffffffff, nll, offset);
    }
    nll = __shfl_sync(0xffffffff, nll, 0);

    // Prior: 0.5 * sum(beta_j^2) — add once post-reduction (all lanes same)
    double prior = 0.0;
    for (int j = 0; j < p; j++) {
        prior += 0.5 * beta[j] * beta[j];
    }
    return nll + prior;
}

// Warp-cooperative GLM logistic gradient (global memory column-major X).
// For large n*p when shared memory is unavailable.
// model_data: [n, p, X_row(n*p), y(n), X_col(n*p)]
__device__ void grad_glm_logistic_warp_global(
    const double* beta, double* grad, int dim,
    const double* model_data, int n, int p, int lane_id)
{
    for (int j = 0; j < p; j++) {
        grad[j] = 0.0;
    }

    const double* X_col = model_data + 2 + n * p + n;
    const double* y = model_data + 2 + n * p;

    for (int i = lane_id; i < n; i += 32) {
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            eta += X_col[(size_t)j * n + i] * beta[j];
        }

        double prob;
        if (eta >= 0.0) {
            double e = exp(-eta);
            prob = 1.0 / (1.0 + e);
        } else {
            double e = exp(eta);
            prob = e / (1.0 + e);
        }

        double diff = prob - y[i];
        for (int j = 0; j < p; j++) {
            grad[j] += diff * X_col[(size_t)j * n + i];
        }
    }

    // Warp reduction + broadcast + prior
    for (int j = 0; j < p; j++) {
        double g = grad[j];
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            g += __shfl_down_sync(0xffffffff, g, offset);
        }
        // After reduction, add prior gradient (all lanes same at lane 0)
        g = __shfl_sync(0xffffffff, g, 0) + beta[j];
        grad[j] = g;
    }
}

// Warp-cooperative GLM logistic NLL (global memory column-major X).
// model_data: [n, p, X_row(n*p), y(n), X_col(n*p)]
__device__ double nll_glm_logistic_warp_global(
    const double* beta, int dim,
    const double* model_data, int n, int p, int lane_id)
{
    (void)dim;

    double nll = 0.0;
    const double* X_col = model_data + 2 + n * p + n;
    const double* y = model_data + 2 + n * p;

    for (int i = lane_id; i < n; i += 32) {
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            eta += X_col[(size_t)j * n + i] * beta[j];
        }

        double abs_eta = fabs(eta);
        nll += fmax(eta, 0.0) + log(1.0 + exp(-abs_eta)) - y[i] * eta;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        nll += __shfl_down_sync(0xffffffff, nll, offset);
    }
    nll = __shfl_sync(0xffffffff, nll, 0);

    double prior = 0.0;
    for (int j = 0; j < p; j++) {
        prior += 0.5 * beta[j] * beta[j];
    }
    return nll + prior;
}

// Fused warp-cooperative GLM logistic: compute gradient and NLL in one pass.
// Shared-memory path (column-major X).
__device__ double grad_nll_glm_logistic_warp_shmem(
    const double* beta, double* grad, int dim,
    const double* s_X_col, const double* s_y,
    int n, int p, int lane_id)
{
    (void)dim;

    for (int j = 0; j < p; j++) {
        grad[j] = 0.0;
    }
    double nll = 0.0;

    for (int i = lane_id; i < n; i += 32) {
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            eta += s_X_col[j * n + i] * beta[j];
        }

        double prob, log1pexp;
        if (eta >= 0.0) {
            double e = exp(-eta);
            prob = 1.0 / (1.0 + e);
            log1pexp = eta + log1p(e);
        } else {
            double e = exp(eta);
            prob = e / (1.0 + e);
            log1pexp = log1p(e);
        }
        nll += log1pexp - s_y[i] * eta;
        double diff = prob - s_y[i];
        for (int j = 0; j < p; j++) {
            grad[j] += diff * s_X_col[j * n + i];
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        nll += __shfl_down_sync(0xffffffff, nll, offset);
    }
    nll = __shfl_sync(0xffffffff, nll, 0);

    for (int j = 0; j < p; j++) {
        double g = grad[j];
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            g += __shfl_down_sync(0xffffffff, g, offset);
        }
        grad[j] = __shfl_sync(0xffffffff, g, 0) + beta[j];
    }

    double prior = 0.0;
    for (int j = 0; j < p; j++) {
        prior += 0.5 * beta[j] * beta[j];
    }
    return nll + prior;
}

// Fused warp-cooperative GLM logistic: compute gradient and NLL in one pass.
// Global-memory path (precomputed column-major X in model_data).
template <int P>
__device__ double grad_nll_glm_logistic_warp_global_p(
    const double* beta, double* grad,
    const double* X_col, const double* y,
    int n, int lane_id)
{
    #pragma unroll
    for (int j = 0; j < P; j++) {
        grad[j] = 0.0;
    }
    double nll = 0.0;

    for (int i = lane_id; i < n; i += 32) {
        double eta = 0.0;
        #pragma unroll
        for (int j = 0; j < P; j++) {
            eta += X_col[(size_t)j * n + i] * beta[j];
        }

        double prob, log1pexp;
        if (eta >= 0.0) {
            double e = exp(-eta);
            prob = 1.0 / (1.0 + e);
            log1pexp = eta + log1p(e);
        } else {
            double e = exp(eta);
            prob = e / (1.0 + e);
            log1pexp = log1p(e);
        }
        nll += log1pexp - y[i] * eta;
        double diff = prob - y[i];
        #pragma unroll
        for (int j = 0; j < P; j++) {
            grad[j] += diff * X_col[(size_t)j * n + i];
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        nll += __shfl_down_sync(0xffffffff, nll, offset);
    }
    nll = __shfl_sync(0xffffffff, nll, 0);

    #pragma unroll
    for (int j = 0; j < P; j++) {
        double g = grad[j];
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            g += __shfl_down_sync(0xffffffff, g, offset);
        }
        grad[j] = __shfl_sync(0xffffffff, g, 0) + beta[j];
    }

    double prior = 0.0;
    #pragma unroll
    for (int j = 0; j < P; j++) {
        prior += 0.5 * beta[j] * beta[j];
    }
    return nll + prior;
}

__device__ double grad_nll_glm_logistic_warp_global(
    const double* beta, double* grad, int dim,
    const double* model_data, int n, int p, int lane_id)
{
    (void)dim;
    const double* X_col = model_data + 2 + n * p + n;
    const double* y = model_data + 2 + n * p;

    switch (p) {
        case 6:
            return grad_nll_glm_logistic_warp_global_p<6>(beta, grad, X_col, y, n, lane_id);
        case 20:
            return grad_nll_glm_logistic_warp_global_p<20>(beta, grad, X_col, y, n, lane_id);
        default:
            break;
    }

    for (int j = 0; j < p; j++) {
        grad[j] = 0.0;
    }
    double nll = 0.0;

    for (int i = lane_id; i < n; i += 32) {
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            eta += X_col[(size_t)j * n + i] * beta[j];
        }

        double prob, log1pexp;
        if (eta >= 0.0) {
            double e = exp(-eta);
            prob = 1.0 / (1.0 + e);
            log1pexp = eta + log1p(e);
        } else {
            double e = exp(eta);
            prob = e / (1.0 + e);
            log1pexp = log1p(e);
        }
        nll += log1pexp - y[i] * eta;
        double diff = prob - y[i];
        for (int j = 0; j < p; j++) {
            grad[j] += diff * X_col[(size_t)j * n + i];
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        nll += __shfl_down_sync(0xffffffff, nll, offset);
    }
    nll = __shfl_sync(0xffffffff, nll, 0);

    for (int j = 0; j < p; j++) {
        double g = grad[j];
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            g += __shfl_down_sync(0xffffffff, g, offset);
        }
        grad[j] = __shfl_sync(0xffffffff, g, 0) + beta[j];
    }

    double prior = 0.0;
    for (int j = 0; j < p; j++) {
        prior += 0.5 * beta[j] * beta[j];
    }
    return nll + prior;
}

/* ---- Warp-cooperative kernels for GLM Linear (model 6) ---- */
// Fused grad+NLL, global memory. model_data: [n, p, X_row(n*p), y(n), X_col(n*p)]
__device__ double grad_nll_glm_linear_warp_global(
    const double* beta, double* grad, int dim,
    const double* model_data, int n, int p, int lane_id)
{
    (void)dim;
    const double* X_col = model_data + 2 + n * p + n;
    const double* y = model_data + 2 + n * p;

    for (int j = 0; j < p; j++) grad[j] = 0.0;
    double nll = 0.0;

    for (int i = lane_id; i < n; i += 32) {
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            eta += X_col[(size_t)j * n + i] * beta[j];
        }
        double residual = eta - y[i];
        nll += 0.5 * residual * residual;
        for (int j = 0; j < p; j++) {
            grad[j] += residual * X_col[(size_t)j * n + i];
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        nll += __shfl_down_sync(0xffffffff, nll, offset);
    }
    nll = __shfl_sync(0xffffffff, nll, 0);

    for (int j = 0; j < p; j++) {
        double g = grad[j];
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            g += __shfl_down_sync(0xffffffff, g, offset);
        }
        grad[j] = __shfl_sync(0xffffffff, g, 0) + beta[j];
    }
    double prior = 0.0;
    for (int j = 0; j < p; j++) prior += 0.5 * beta[j] * beta[j];
    return nll + prior;
}

// Fused grad+NLL, shared memory.
__device__ double grad_nll_glm_linear_warp_shmem(
    const double* beta, double* grad, int dim,
    const double* s_X_col, const double* s_y,
    int n, int p, int lane_id)
{
    (void)dim;
    for (int j = 0; j < p; j++) grad[j] = 0.0;
    double nll = 0.0;

    for (int i = lane_id; i < n; i += 32) {
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            eta += s_X_col[j * n + i] * beta[j];
        }
        double residual = eta - s_y[i];
        nll += 0.5 * residual * residual;
        for (int j = 0; j < p; j++) {
            grad[j] += residual * s_X_col[j * n + i];
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        nll += __shfl_down_sync(0xffffffff, nll, offset);
    }
    nll = __shfl_sync(0xffffffff, nll, 0);

    for (int j = 0; j < p; j++) {
        double g = grad[j];
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            g += __shfl_down_sync(0xffffffff, g, offset);
        }
        grad[j] = __shfl_sync(0xffffffff, g, 0) + beta[j];
    }
    double prior = 0.0;
    for (int j = 0; j < p; j++) prior += 0.5 * beta[j] * beta[j];
    return nll + prior;
}

/* ---- Warp-cooperative kernels for GLM Poisson (model 7) ---- */
// Fused grad+NLL, global memory.
// model_data: [n, p, offset_flag, X_row(n*p), y(n), X_col(n*p), offset(n if flag)]
__device__ double grad_nll_glm_poisson_warp_global(
    const double* beta, double* grad, int dim,
    const double* model_data, int n, int p, int lane_id)
{
    (void)dim;
    int offset_flag = (int)model_data[2];
    const double* X_col = model_data + 3 + n * p + n;
    const double* y = model_data + 3 + n * p;
    const double* offset_arr = (offset_flag == 1) ? (model_data + 3 + n * p + n + n * p) : NULL;

    for (int j = 0; j < p; j++) grad[j] = 0.0;
    double nll = 0.0;

    for (int i = lane_id; i < n; i += 32) {
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            eta += X_col[(size_t)j * n + i] * beta[j];
        }
        if (offset_arr) eta += offset_arr[i];
        if (eta > 50.0) eta = 50.0; else if (eta < -50.0) eta = -50.0;

        double exp_eta = exp(eta);
        nll += exp_eta - y[i] * eta;
        double diff = exp_eta - y[i];
        for (int j = 0; j < p; j++) {
            grad[j] += diff * X_col[(size_t)j * n + i];
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        nll += __shfl_down_sync(0xffffffff, nll, offset);
    }
    nll = __shfl_sync(0xffffffff, nll, 0);

    for (int j = 0; j < p; j++) {
        double g = grad[j];
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            g += __shfl_down_sync(0xffffffff, g, offset);
        }
        grad[j] = __shfl_sync(0xffffffff, g, 0) + beta[j];
    }
    double prior = 0.0;
    for (int j = 0; j < p; j++) prior += 0.5 * beta[j] * beta[j];
    return nll + prior;
}

/* ---- Warp-cooperative kernels for GLM NegBin (model 8) ---- */
// Fused grad+NLL, global memory.
// model_data: [n, p, offset_flag, X_row(n*p), y(n), X_col(n*p), offset(n if flag)]
// x = [beta_0..p-1, log_alpha], dim = p+1
__device__ double grad_nll_glm_negbin_warp_global(
    const double* x, double* grad, int dim,
    const double* model_data, int n, int p, int lane_id)
{
    int offset_flag = (int)model_data[2];
    const double* X_col = model_data + 3 + n * p + n;
    const double* y = model_data + 3 + n * p;
    const double* offset_arr = (offset_flag == 1) ? (model_data + 3 + n * p + n + n * p) : NULL;

    const double* beta = x;
    double log_alpha = x[p];
    double log_alpha_c = log_alpha;
    if (log_alpha_c < -10.0) log_alpha_c = -10.0;
    if (log_alpha_c > 8.0) log_alpha_c = 8.0;
    double alpha = exp(log_alpha_c);
    double theta = 1.0 / alpha;

    for (int j = 0; j < dim; j++) grad[j] = 0.0;
    double nll = 0.0;

    for (int i = lane_id; i < n; i += 32) {
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            eta += X_col[(size_t)j * n + i] * beta[j];
        }
        if (offset_arr) eta += offset_arr[i];
        if (eta > 50.0) eta = 50.0; else if (eta < -50.0) eta = -50.0;

        double mu = exp(eta);
        double yi = y[i];
        double denom = theta + mu;

        nll -= lgamma(yi + theta) - lgamma(theta) + theta * log(theta / denom) + yi * log(mu / denom);
        // d(NLL_i)/d(eta_i) = mu*(theta+yi)/denom - yi  (same as scalar kernel)
        double d_eta = mu * (theta + yi) / denom - yi;
        for (int j = 0; j < p; j++) {
            grad[j] += d_eta * X_col[(size_t)j * n + i];
        }
        // grad wrt log_alpha: chain rule through alpha → theta
        double psi_yi_theta = device_digamma(yi + theta);
        double psi_theta = device_digamma(theta);
        double dnll_dtheta = -(psi_yi_theta - psi_theta + log(theta / denom) + 1.0 - (theta + yi) / denom);
        grad[p] += dnll_dtheta * (-theta);
    }

    // Warp reduction for NLL
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        nll += __shfl_down_sync(0xffffffff, nll, offset);
    }
    nll = __shfl_sync(0xffffffff, nll, 0);

    // Warp reduction for grad[0..p-1] + prior
    for (int j = 0; j < p; j++) {
        double g = grad[j];
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            g += __shfl_down_sync(0xffffffff, g, offset);
        }
        grad[j] = __shfl_sync(0xffffffff, g, 0) + beta[j];
    }
    // Warp reduction for grad[p] (log_alpha) + prior
    {
        double g = grad[p];
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            g += __shfl_down_sync(0xffffffff, g, offset);
        }
        double likelihood_grad = __shfl_sync(0xffffffff, g, 0);
        // When log_alpha is outside clamp range, likelihood is flat → zero gradient.
        if (log_alpha <= -10.0 || log_alpha >= 8.0) {
            likelihood_grad = 0.0;
        }
        grad[p] = likelihood_grad + log_alpha;
    }

    double prior = 0.0;
    for (int j = 0; j < p; j++) prior += 0.5 * beta[j] * beta[j];
    prior += 0.5 * log_alpha * log_alpha;
    return nll + prior;
}

/* ---- Warp-cooperative kernels for GLM ComposedLogistic (model 9) ---- */
// Fused grad+NLL, global memory.
// model_data: [n, p, G, X_row(n*p), y(n), group_idx(n), X_col(n*p), re_prior_sigma]
// x = [beta_0..p-1, z_0..z_{G-1}], dim = p + G
__device__ double grad_nll_glm_composed_logistic_warp_global(
    const double* x, double* grad, int dim,
    const double* model_data, int n, int p, int lane_id)
{
    int G = (int)model_data[2];
    const double* X_col = model_data + 3 + n * p + n + n;
    const double* y = model_data + 3 + n * p;
    const double* group_idx_f = model_data + 3 + n * p + n;
    double re_scale = model_data[3 + n * p + n + n + n * p];

    const double* beta = x;
    const double* z = x + p;

    for (int j = 0; j < dim; j++) grad[j] = 0.0;
    double nll = 0.0;

    for (int i = lane_id; i < n; i += 32) {
        double eta = 0.0;
        for (int j = 0; j < p; j++) {
            eta += X_col[(size_t)j * n + i] * beta[j];
        }
        int gi = (int)group_idx_f[i];
        eta += re_scale * z[gi];

        double prob;
        if (eta >= 0.0) {
            double e = exp(-eta);
            prob = 1.0 / (1.0 + e);
        } else {
            double e = exp(eta);
            prob = e / (1.0 + e);
        }

        double abs_eta = fabs(eta);
        nll += fmax(eta, 0.0) + log(1.0 + exp(-abs_eta)) - y[i] * eta;

        double diff = prob - y[i];
        for (int j = 0; j < p; j++) {
            grad[j] += diff * X_col[(size_t)j * n + i];
        }
        grad[p + gi] += diff * re_scale;
    }

    // Warp reduction for NLL
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        nll += __shfl_down_sync(0xffffffff, nll, offset);
    }
    nll = __shfl_sync(0xffffffff, nll, 0);

    // Warp reduction for grad[0..dim-1] + prior
    for (int j = 0; j < dim; j++) {
        double g = grad[j];
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            g += __shfl_down_sync(0xffffffff, g, offset);
        }
        double prior_g = (j < p) ? beta[j] : z[j - p];  // N(0,1) prior on all params
        grad[j] = __shfl_sync(0xffffffff, g, 0) + prior_g;
    }

    double prior = 0.0;
    for (int j = 0; j < p; j++) prior += 0.5 * beta[j] * beta[j];
    for (int j = 0; j < G; j++) prior += 0.5 * z[j] * z[j];
    return nll + prior;
}

// Warp dispatch: uses shared memory for GLM, serial fallback for others.
// s_X_col / s_y pointers are only valid when model_id == 3.
#define MAMS_WARP_DEFINED

__device__ double user_nll_warp(
    const double* x, int dim, const double* model_data,
    const double* s_X_col, const double* s_y,
    int n_obs, int n_feat, int lane_id, int use_shmem)
{
    // For models 6-9, use fused grad_nll warp (one pass) and discard gradient.
    // This avoids duplicating the observation loop for NLL-only.
    double dummy_grad[128];
    switch (__mams_model_id) {
        case 3:
            if (use_shmem == 1) {
                return nll_glm_logistic_warp_shmem(x, dim, s_X_col, s_y, n_obs, n_feat, lane_id);
            }
            return nll_glm_logistic_warp_global(x, dim, model_data, n_obs, n_feat, lane_id);
        case 6:
            if (use_shmem == 1)
                return grad_nll_glm_linear_warp_shmem(x, dummy_grad, dim, s_X_col, s_y, n_obs, n_feat, lane_id);
            return grad_nll_glm_linear_warp_global(x, dummy_grad, dim, model_data, n_obs, n_feat, lane_id);
        case 7:
            return grad_nll_glm_poisson_warp_global(x, dummy_grad, dim, model_data, n_obs, n_feat, lane_id);
        case 8:
            return grad_nll_glm_negbin_warp_global(x, dummy_grad, dim, model_data, n_obs, n_feat, lane_id);
        case 9:
            return grad_nll_glm_composed_logistic_warp_global(x, dummy_grad, dim, model_data, n_obs, n_feat, lane_id);
        default: return user_nll(x, dim, model_data);
    }
}

__device__ void user_grad_warp(
    const double* x, double* grad, int dim, const double* model_data,
    const double* s_X_col, const double* s_y,
    int n_obs, int n_feat, int lane_id, int use_shmem)
{
    switch (__mams_model_id) {
        case 3:
            if (use_shmem == 1) {
                grad_glm_logistic_warp_shmem(x, grad, dim, s_X_col, s_y, n_obs, n_feat, lane_id);
            } else {
                grad_glm_logistic_warp_global(x, grad, dim, model_data, n_obs, n_feat, lane_id);
            }
            break;
        case 6:
            if (use_shmem == 1) {
                grad_nll_glm_linear_warp_shmem(x, grad, dim, s_X_col, s_y, n_obs, n_feat, lane_id);
            } else {
                grad_nll_glm_linear_warp_global(x, grad, dim, model_data, n_obs, n_feat, lane_id);
            }
            break;
        case 7:
            grad_nll_glm_poisson_warp_global(x, grad, dim, model_data, n_obs, n_feat, lane_id);
            break;
        case 8:
            grad_nll_glm_negbin_warp_global(x, grad, dim, model_data, n_obs, n_feat, lane_id);
            break;
        case 9:
            grad_nll_glm_composed_logistic_warp_global(x, grad, dim, model_data, n_obs, n_feat, lane_id);
            break;
        default: user_grad(x, grad, dim, model_data); break;
    }
}

#define MAMS_WARP_FUSED_DEFINED
__device__ double user_grad_nll_warp(
    const double* x, double* grad, int dim, const double* model_data,
    const double* s_X_col, const double* s_y,
    int n_obs, int n_feat, int lane_id, int use_shmem)
{
    switch (__mams_model_id) {
        case 3:
            if (use_shmem == 1) {
                return grad_nll_glm_logistic_warp_shmem(
                    x, grad, dim, s_X_col, s_y, n_obs, n_feat, lane_id);
            }
            return grad_nll_glm_logistic_warp_global(
                x, grad, dim, model_data, n_obs, n_feat, lane_id);
        case 6:
            if (use_shmem == 1) {
                return grad_nll_glm_linear_warp_shmem(
                    x, grad, dim, s_X_col, s_y, n_obs, n_feat, lane_id);
            }
            return grad_nll_glm_linear_warp_global(
                x, grad, dim, model_data, n_obs, n_feat, lane_id);
        case 7:
            return grad_nll_glm_poisson_warp_global(
                x, grad, dim, model_data, n_obs, n_feat, lane_id);
        case 8:
            return grad_nll_glm_negbin_warp_global(
                x, grad, dim, model_data, n_obs, n_feat, lane_id);
        case 9:
            return grad_nll_glm_composed_logistic_warp_global(
                x, grad, dim, model_data, n_obs, n_feat, lane_id);
        default:
            user_grad(x, grad, dim, model_data);
            return user_nll(x, dim, model_data);
    }
}

/* ---------- Engine (RNG + integrator + kernel) --------------------------- */

/* __mams_model_id is declared in the engine header as a forward reference.
 * We defined user_nll/user_grad above, which the engine kernel calls. */
#include "mams_engine.cuh"
