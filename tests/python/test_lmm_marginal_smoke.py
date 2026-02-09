from __future__ import annotations

import math

import numpy as np


def _fit_lmm(model):
    import nextstat as ns

    mle = ns.MaximumLikelihoodEstimator()
    r = mle.fit(model)
    assert r.converged
    assert math.isfinite(r.nll)
    assert len(r.parameters) == model.n_params()
    return r


def _param_map(model, params):
    names = model.parameter_names()
    assert len(names) == len(params)
    return dict(zip(names, params, strict=True))


def test_lmm_marginal_random_intercept_recovers_synthetic_params():
    import nextstat as ns

    rng = np.random.default_rng(1234)
    n_groups = 12
    m_per = 6
    n = n_groups * m_per

    beta0_true = 1.25
    beta1_true = -0.7
    sigma_y_true = 0.35
    tau_alpha_true = 0.9

    group_idx = np.repeat(np.arange(n_groups, dtype=int), m_per)
    x1 = rng.normal(size=n)
    alpha = rng.normal(scale=tau_alpha_true, size=n_groups)
    eps = rng.normal(scale=sigma_y_true, size=n)
    y = beta0_true + beta1_true * x1 + alpha[group_idx] + eps

    x = x1.reshape(-1, 1).tolist()
    model = ns.LmmMarginalModel(
        x,
        y.tolist(),
        include_intercept=True,
        group_idx=group_idx.tolist(),
        n_groups=n_groups,
    )
    r = _fit_lmm(model)
    pm = _param_map(model, r.parameters)

    beta0_hat = pm["intercept"]
    beta1_hat = pm["beta1"]
    sigma_y_hat = math.exp(pm["log_sigma_y"])
    tau_alpha_hat = math.exp(pm["log_tau_alpha"])

    assert abs(beta0_hat - beta0_true) < 0.35
    assert abs(beta1_hat - beta1_true) < 0.25
    assert abs(sigma_y_hat - sigma_y_true) < 0.25
    assert abs(tau_alpha_hat - tau_alpha_true) < 0.35


def test_lmm_marginal_random_intercept_slope_recovers_synthetic_params():
    import nextstat as ns

    rng = np.random.default_rng(4321)
    n_groups = 10
    m_per = 8
    n = n_groups * m_per

    beta0_true = -0.3
    beta1_true = 0.9
    sigma_y_true = 0.25
    tau_alpha_true = 0.6
    tau_u_true = 0.5

    group_idx = np.repeat(np.arange(n_groups, dtype=int), m_per)
    x1 = rng.normal(size=n)
    alpha = rng.normal(scale=tau_alpha_true, size=n_groups)
    u = rng.normal(scale=tau_u_true, size=n_groups)
    eps = rng.normal(scale=sigma_y_true, size=n)
    y = beta0_true + beta1_true * x1 + alpha[group_idx] + u[group_idx] * x1 + eps

    x = x1.reshape(-1, 1).tolist()
    model = ns.LmmMarginalModel(
        x,
        y.tolist(),
        include_intercept=True,
        group_idx=group_idx.tolist(),
        n_groups=n_groups,
        random_slope_feature_idx=0,
    )
    r = _fit_lmm(model)
    pm = _param_map(model, r.parameters)

    beta0_hat = pm["intercept"]
    beta1_hat = pm["beta1"]
    sigma_y_hat = math.exp(pm["log_sigma_y"])
    tau_alpha_hat = math.exp(pm["log_tau_alpha"])
    tau_u_hat = math.exp(pm["log_tau_u_beta1"])

    assert abs(beta0_hat - beta0_true) < 0.4
    assert abs(beta1_hat - beta1_true) < 0.3
    assert abs(sigma_y_hat - sigma_y_true) < 0.25
    assert abs(tau_alpha_hat - tau_alpha_true) < 0.4
    assert abs(tau_u_hat - tau_u_true) < 0.4

