#!/usr/bin/env python3
"""Generate golden LMM fixtures for parity testing.

Each fixture contains:
- dgp_params: true data-generating parameters
- data: X, y, group_idx, n_groups
- model_config: include_intercept, random_slope_feature_idx
- expected: NLL at truth (from nextstat), fitted params + NLL (from nextstat MLE)

Usage:
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/generate_golden_lmm.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _generate_random_intercept_fixture(seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)

    n_groups = 15
    m_per = 8
    n = n_groups * m_per

    # True DGP parameters
    beta0_true = 2.0
    beta1_true = -0.5
    sigma_y_true = 0.4
    tau_alpha_true = 0.8

    group_idx = np.repeat(np.arange(n_groups, dtype=int), m_per)
    x1 = rng.normal(size=n)
    alpha = rng.normal(scale=tau_alpha_true, size=n_groups)
    eps = rng.normal(scale=sigma_y_true, size=n)
    y = beta0_true + beta1_true * x1 + alpha[group_idx] + eps

    import nextstat

    x_list = x1.reshape(-1, 1).tolist()
    y_list = y.tolist()
    g_list = group_idx.tolist()

    model = nextstat.LmmMarginalModel(
        x_list,
        y_list,
        include_intercept=True,
        group_idx=g_list,
        n_groups=n_groups,
    )

    # NLL at truth
    log_sigma_y = math.log(sigma_y_true)
    log_tau_alpha = math.log(tau_alpha_true)
    truth_params = [beta0_true, beta1_true, log_sigma_y, log_tau_alpha]
    nll_at_truth = float(model.nll(truth_params))

    # MLE fit
    mle = nextstat.MaximumLikelihoodEstimator()
    r = mle.fit(model)

    param_names = model.parameter_names()
    fitted = dict(zip(param_names, [float(v) for v in r.parameters]))

    return {
        "name": "random_intercept_gaussian",
        "seed": seed,
        "dgp_params": {
            "beta0": beta0_true,
            "beta1": beta1_true,
            "sigma_y": sigma_y_true,
            "tau_alpha": tau_alpha_true,
        },
        "data": {
            "x": x_list,
            "y": y_list,
            "group_idx": g_list,
            "n_groups": n_groups,
        },
        "model_config": {
            "include_intercept": True,
            "random_slope_feature_idx": None,
        },
        "expected": {
            "n_params": model.n_params(),
            "parameter_names": param_names,
            "nll_at_truth": nll_at_truth,
            "mle_nll": float(r.nll),
            "mle_converged": bool(r.converged),
            "mle_params": fitted,
            "truth_params_vector": truth_params,
        },
    }


def _generate_random_intercept_slope_fixture(seed: int = 99) -> dict:
    rng = np.random.default_rng(seed)

    n_groups = 12
    m_per = 10
    n = n_groups * m_per

    beta0_true = -0.3
    beta1_true = 1.2
    sigma_y_true = 0.3
    tau_alpha_true = 0.7
    tau_u_true = 0.4

    group_idx = np.repeat(np.arange(n_groups, dtype=int), m_per)
    x1 = rng.normal(size=n)
    alpha = rng.normal(scale=tau_alpha_true, size=n_groups)
    u = rng.normal(scale=tau_u_true, size=n_groups)
    eps = rng.normal(scale=sigma_y_true, size=n)
    y = beta0_true + beta1_true * x1 + alpha[group_idx] + u[group_idx] * x1 + eps

    import nextstat

    x_list = x1.reshape(-1, 1).tolist()
    y_list = y.tolist()
    g_list = group_idx.tolist()

    model = nextstat.LmmMarginalModel(
        x_list,
        y_list,
        include_intercept=True,
        group_idx=g_list,
        n_groups=n_groups,
        random_slope_feature_idx=0,
    )

    log_sigma_y = math.log(sigma_y_true)
    log_tau_alpha = math.log(tau_alpha_true)
    log_tau_u = math.log(tau_u_true)
    truth_params = [beta0_true, beta1_true, log_sigma_y, log_tau_alpha, log_tau_u]
    nll_at_truth = float(model.nll(truth_params))

    mle = nextstat.MaximumLikelihoodEstimator()
    r = mle.fit(model)

    param_names = model.parameter_names()
    fitted = dict(zip(param_names, [float(v) for v in r.parameters]))

    return {
        "name": "random_intercept_slope_gaussian",
        "seed": seed,
        "dgp_params": {
            "beta0": beta0_true,
            "beta1": beta1_true,
            "sigma_y": sigma_y_true,
            "tau_alpha": tau_alpha_true,
            "tau_u": tau_u_true,
        },
        "data": {
            "x": x_list,
            "y": y_list,
            "group_idx": g_list,
            "n_groups": n_groups,
        },
        "model_config": {
            "include_intercept": True,
            "random_slope_feature_idx": 0,
        },
        "expected": {
            "n_params": model.n_params(),
            "parameter_names": param_names,
            "nll_at_truth": nll_at_truth,
            "mle_nll": float(r.nll),
            "mle_converged": bool(r.converged),
            "mle_params": fitted,
            "truth_params_vector": truth_params,
        },
    }


def main() -> int:
    fixtures = [
        ("lmm_random_intercept.json", _generate_random_intercept_fixture),
        ("lmm_random_intercept_slope.json", _generate_random_intercept_slope_fixture),
    ]
    for fname, gen_fn in fixtures:
        path = FIXTURES_DIR / fname
        data = gen_fn()
        path.write_text(json.dumps(data, indent=2))
        print(f"Wrote: {path}")
        print(f"  NLL at truth: {data['expected']['nll_at_truth']:.6f}")
        print(f"  MLE NLL:      {data['expected']['mle_nll']:.6f}")
        print(f"  Converged:    {data['expected']['mle_converged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
