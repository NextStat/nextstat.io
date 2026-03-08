"""Shared SBC helpers and sampler-quality gates for Bayesian samplers."""

from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Sequence, Tuple

import pytest

import nextstat


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _var(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mu = _mean(xs)
    return float(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _sample_normal(rng: random.Random, mu: float, sigma: float) -> float:
    return float(mu + sigma * rng.gauss(0.0, 1.0))


def _sample_lognormal(rng: random.Random, m: float, s: float) -> float:
    return float(math.exp(m + s * rng.gauss(0.0, 1.0)))


def _flatten_chains(posterior: Dict[str, List[List[float]]], name: str) -> List[float]:
    out: List[float] = []
    for chain in posterior[name]:
        out.extend(float(v) for v in chain)
    return out


def _ranks_u01(draws: Sequence[float], truth: float) -> float:
    n = len(draws)
    if n == 0:
        return float("nan")
    r = sum(1 for v in draws if float(v) < float(truth))
    return float(r / n)


def _assert_sbc_u01(samples_u: Sequence[float], *, max_mean_delta: float, max_var_delta: float) -> None:
    mu = _mean(samples_u)
    v = _var(samples_u)
    assert abs(mu - 0.5) <= max_mean_delta, f"SBC mean(u)={mu} too far from 0.5"
    assert abs(v - (1.0 / 12.0)) <= max_var_delta, f"SBC var(u)={v} too far from 1/12"


def require_slow() -> Tuple[int, int, int, int]:
    if os.environ.get("NS_RUN_SLOW") != "1":
        pytest.skip("Set NS_RUN_SLOW=1 to run slow SBC tests.")
    if os.environ.get("NS_RUN_SBC") != "1":
        pytest.skip("Set NS_RUN_SBC=1 to run SBC tests (they are very slow).")
    n_runs = int(os.environ.get("NS_SBC_RUNS", "20"))
    n_warmup = int(os.environ.get("NS_SBC_WARMUP", "200"))
    n_samples = int(os.environ.get("NS_SBC_SAMPLES", "200"))
    seed = int(os.environ.get("NS_SBC_SEED", "0"))
    if n_runs < 10:
        pytest.skip("SBC needs NS_SBC_RUNS >= 10 to be meaningful/stable.")
    if n_warmup < 100 or n_samples < 100:
        pytest.skip("SBC needs NS_SBC_WARMUP >= 100 and NS_SBC_SAMPLES >= 100.")
    return n_runs, n_warmup, n_samples, seed


def sample_posterior_u(
    method: str,
    model,
    truth_by_name: Dict[str, float],
    *,
    seed: int,
    n_warmup: int,
    n_samples: int,
    target_accept: float = 0.8,
    rhat_max: float,
    divergence_rate_max: float,
) -> Dict[str, float]:
    result = nextstat.sample(
        model,
        method=method,
        n_chains=2,
        n_warmup=n_warmup,
        n_samples=n_samples,
        seed=seed,
        init_jitter_rel=0.10,
        target_accept=target_accept,
    )
    diag = result["diagnostics"]
    assert float(diag["divergence_rate"]) <= divergence_rate_max, (
        f"{method} divergence_rate={diag['divergence_rate']} > {divergence_rate_max}"
    )
    for name, value in diag["r_hat"].items():
        assert float(value) < rhat_max, f"{method} R-hat({name})={value} >= {rhat_max}"

    posterior = result["posterior"]
    out: Dict[str, float] = {}
    for name, truth in truth_by_name.items():
        draws = _flatten_chains(posterior, name)
        out[name] = _ranks_u01(draws, truth)
    return out


def run_sbc_linear_regression_1d_mean_only(method: str) -> None:
    n_runs, n_warmup, n_samples, seed0 = require_slow()

    n = 25
    x = [[1.0] for _ in range(n)]
    coef_prior_mu = 0.0
    coef_prior_sigma = 1.0

    u_by_param: Dict[str, List[float]] = {"beta1": []}
    for run in range(n_runs):
        rng = random.Random(seed0 + run)
        beta1 = _sample_normal(rng, coef_prior_mu, coef_prior_sigma)
        y = [_sample_normal(rng, beta1, 1.0) for _ in range(n)]

        model = nextstat.ComposedGlmModel.linear_regression(
            x,
            y,
            include_intercept=False,
            coef_prior_mu=coef_prior_mu,
            coef_prior_sigma=coef_prior_sigma,
        )

        u = sample_posterior_u(
            method,
            model,
            {"beta1": beta1},
            seed=seed0 + 10_000 + run,
            n_warmup=n_warmup,
            n_samples=n_samples,
            rhat_max=1.20,
            divergence_rate_max=0.02,
        )
        u_by_param["beta1"].append(u["beta1"])

    tol = 0.25 if n_runs < 20 else 0.12
    _assert_sbc_u01(u_by_param["beta1"], max_mean_delta=tol, max_var_delta=tol)


def run_sbc_linear_regression_2d(method: str) -> None:
    n_runs, n_warmup, n_samples, seed0 = require_slow()

    n = 30
    xs = [(-1.0 + 2.0 * i / (n - 1)) for i in range(n)]
    x = [[1.0, float(v)] for v in xs]
    coef_prior_mu = 0.0
    coef_prior_sigma = 1.0

    u_by_param: Dict[str, List[float]] = {"beta1": [], "beta2": []}
    for run in range(n_runs):
        rng = random.Random(seed0 + 100 + run)
        beta1 = _sample_normal(rng, coef_prior_mu, coef_prior_sigma)
        beta2 = _sample_normal(rng, coef_prior_mu, coef_prior_sigma)
        y = [_sample_normal(rng, beta1 + beta2 * float(v), 1.0) for v in xs]

        model = nextstat.ComposedGlmModel.linear_regression(
            x,
            y,
            include_intercept=False,
            coef_prior_mu=coef_prior_mu,
            coef_prior_sigma=coef_prior_sigma,
        )

        u = sample_posterior_u(
            method,
            model,
            {"beta1": beta1, "beta2": beta2},
            seed=seed0 + 20_000 + run,
            n_warmup=n_warmup,
            n_samples=n_samples,
            rhat_max=1.20,
            divergence_rate_max=0.02,
        )
        u_by_param["beta1"].append(u["beta1"])
        u_by_param["beta2"].append(u["beta2"])

    tol = 0.25 if n_runs < 20 else 0.12
    _assert_sbc_u01(u_by_param["beta1"], max_mean_delta=tol, max_var_delta=tol)
    _assert_sbc_u01(u_by_param["beta2"], max_mean_delta=tol, max_var_delta=tol)


def run_sbc_random_intercept_gaussian_smoke(method: str) -> None:
    n_runs, n_warmup, n_samples, seed0 = require_slow()

    mu_prior = (0.0, 1.0)
    logsigma_prior = (0.0, 0.5)
    coef_prior_mu = 0.0
    coef_prior_sigma = 1.0

    n_groups = 4
    n_per = 6
    n = n_groups * n_per
    group_idx = [g for g in range(n_groups) for _ in range(n_per)]
    z = [(-1.0 + 2.0 * i / (n - 1)) for i in range(n)]
    x = [[float(v)] for v in z]

    u_mu_alpha: List[float] = []
    u_sigma_alpha: List[float] = []

    for run in range(n_runs):
        rng = random.Random(seed0 + 1_000 + run)
        beta1 = _sample_normal(rng, coef_prior_mu, coef_prior_sigma)
        mu_alpha = _sample_normal(rng, mu_prior[0], mu_prior[1])
        sigma_alpha = _sample_lognormal(rng, logsigma_prior[0], logsigma_prior[1])
        alphas = [_sample_normal(rng, mu_alpha, sigma_alpha) for _ in range(n_groups)]

        y: List[float] = []
        for i in range(n):
            eta = beta1 * float(z[i]) + alphas[group_idx[i]]
            y.append(_sample_normal(rng, eta, 1.0))

        model = nextstat.ComposedGlmModel.linear_regression(
            x,
            y,
            include_intercept=False,
            group_idx=group_idx,
            n_groups=n_groups,
            coef_prior_mu=coef_prior_mu,
            coef_prior_sigma=coef_prior_sigma,
        )

        u = sample_posterior_u(
            method,
            model,
            {"beta1": beta1, "mu_alpha": mu_alpha, "sigma_alpha": sigma_alpha},
            seed=seed0 + 30_000 + run,
            n_warmup=n_warmup,
            n_samples=n_samples,
            target_accept=0.90,
            rhat_max=1.50,
            divergence_rate_max=0.05,
        )
        u_mu_alpha.append(u["mu_alpha"])
        u_sigma_alpha.append(u["sigma_alpha"])

    tol_mu = 0.30 if n_runs < 20 else 0.15
    tol_sig = 0.35 if n_runs < 20 else 0.20
    _assert_sbc_u01(u_mu_alpha, max_mean_delta=tol_mu, max_var_delta=tol_mu)
    _assert_sbc_u01(u_sigma_alpha, max_mean_delta=tol_sig, max_var_delta=tol_sig)


def run_sbc_random_intercept_bernoulli_smoke(method: str) -> None:
    n_runs, n_warmup, n_samples, seed0 = require_slow()

    mu_prior = (0.0, 1.0)
    logsigma_prior = (0.0, 0.5)
    coef_prior_mu = 0.0
    coef_prior_sigma = 1.0

    n_groups = 4
    n_per = 12
    n = n_groups * n_per
    group_idx = [g for g in range(n_groups) for _ in range(n_per)]
    z = [(-1.0 + 2.0 * i / (n - 1)) for i in range(n)]
    x = [[float(v)] for v in z]

    u_mu_alpha: List[float] = []
    u_sigma_alpha: List[float] = []

    for run in range(n_runs):
        rng = random.Random(seed0 + 2_000 + run)
        beta1 = _sample_normal(rng, coef_prior_mu, coef_prior_sigma)
        mu_alpha = _sample_normal(rng, mu_prior[0], mu_prior[1])
        sigma_alpha = _sample_lognormal(rng, logsigma_prior[0], logsigma_prior[1])
        alphas = [_sample_normal(rng, mu_alpha, sigma_alpha) for _ in range(n_groups)]

        y: List[int] = []
        for i in range(n):
            eta = beta1 * float(z[i]) + alphas[group_idx[i]]
            p = 1.0 / (1.0 + math.exp(-eta))
            y.append(1 if rng.random() < p else 0)

        model = nextstat.ComposedGlmModel.logistic_regression(
            x,
            y,
            include_intercept=False,
            group_idx=group_idx,
            n_groups=n_groups,
            coef_prior_mu=coef_prior_mu,
            coef_prior_sigma=coef_prior_sigma,
        )

        u = sample_posterior_u(
            method,
            model,
            {"beta1": beta1, "mu_alpha": mu_alpha, "sigma_alpha": sigma_alpha},
            seed=seed0 + 40_000 + run,
            n_warmup=n_warmup,
            n_samples=n_samples,
            target_accept=0.90,
            rhat_max=1.60,
            divergence_rate_max=0.05,
        )
        u_mu_alpha.append(u["mu_alpha"])
        u_sigma_alpha.append(u["sigma_alpha"])

    tol_mu = 0.35 if n_runs < 20 else 0.18
    tol_sig = 0.40 if n_runs < 20 else 0.22
    _assert_sbc_u01(u_mu_alpha, max_mean_delta=tol_mu, max_var_delta=tol_mu)
    _assert_sbc_u01(u_sigma_alpha, max_mean_delta=tol_sig, max_var_delta=tol_sig)


def run_quality_gate_glm_strict(method: str) -> None:
    if os.environ.get("NS_RUN_SLOW") != "1" or os.environ.get("NS_SBC_STRICT") != "1":
        pytest.skip("Set NS_RUN_SLOW=1 and NS_SBC_STRICT=1 to run strict sampler quality gate.")

    n_warmup = int(os.environ.get("NS_SBC_WARMUP", "400"))
    n_samples = int(os.environ.get("NS_SBC_SAMPLES", "400"))
    seed = int(os.environ.get("NS_SBC_SEED", "0")) + 999
    if n_warmup < 300 or n_samples < 300:
        pytest.skip("Strict gate needs NS_SBC_WARMUP >= 300 and NS_SBC_SAMPLES >= 300.")

    n = 40
    xs = [(-1.0 + 2.0 * i / (n - 1)) for i in range(n)]
    x = [[1.0, float(v)] for v in xs]
    y = [0.25 + 0.9 * float(v) + 0.1 * math.sin(3.0 * float(v)) for v in xs]

    model = nextstat.ComposedGlmModel.linear_regression(
        x,
        y,
        include_intercept=False,
        coef_prior_mu=0.0,
        coef_prior_sigma=1.0,
    )
    result = nextstat.sample(
        model,
        method=method,
        n_chains=2,
        n_warmup=n_warmup,
        n_samples=n_samples,
        seed=seed,
        init_jitter_rel=0.10,
        target_accept=0.90,
    )

    diag = result["diagnostics"]
    assert float(diag["divergence_rate"]) < 0.01, f"{method} divergence_rate={diag['divergence_rate']}"
    for name, value in diag["r_hat"].items():
        assert float(value) < 1.02, f"{method} R-hat({name})={value}"


def run_moments_golden_gaussian_strict(method: str) -> None:
    if os.environ.get("NS_RUN_SLOW") != "1" or os.environ.get("NS_SBC_STRICT") != "1":
        pytest.skip("Set NS_RUN_SLOW=1 and NS_SBC_STRICT=1 to run strict golden sampler moments test.")

    n_warmup = int(os.environ.get("NS_SBC_WARMUP", "400"))
    n_samples = int(os.environ.get("NS_SBC_SAMPLES", "400"))
    seed = int(os.environ.get("NS_SBC_SEED", "0")) + 1234
    if n_warmup < 800 or n_samples < 800:
        pytest.skip("Golden moments needs NS_SBC_WARMUP >= 800 and NS_SBC_SAMPLES >= 800.")

    x = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    y = [0.0, 0.0, 0.0, 0.0]

    model = nextstat.ComposedGlmModel.linear_regression(
        x,
        y,
        include_intercept=False,
        coef_prior_mu=0.0,
        coef_prior_sigma=1e6,
    )
    result = nextstat.sample(
        model,
        method=method,
        n_chains=2,
        n_warmup=n_warmup,
        n_samples=n_samples,
        seed=seed,
        init_jitter_rel=0.10,
        target_accept=0.90,
    )

    diag = result["diagnostics"]
    assert float(diag["divergence_rate"]) < 0.01, f"{method} divergence_rate={diag['divergence_rate']}"
    for name, value in diag["r_hat"].items():
        assert float(value) < 1.02, f"{method} R-hat({name})={value}"

    names = result["param_names"]
    assert names == ["beta1", "beta2", "beta3", "beta4"]
    draws = [_flatten_chains(result["posterior"], name) for name in names]
    n = len(draws[0])
    assert n >= 200

    means = [_mean(xs) for xs in draws]
    for i, mu in enumerate(means):
        assert abs(mu) < 0.10, f"{method} mean[{i}]={mu} too far from 0"

    cov: List[List[float]] = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            s = 0.0
            mi = means[i]
            mj = means[j]
            for k in range(n):
                s += (draws[i][k] - mi) * (draws[j][k] - mj)
            cov[i][j] = s / float(n - 1)

    for i in range(4):
        assert abs(cov[i][i] - 1.0) < 0.15, f"{method} var[{i}]={cov[i][i]} not ~1"
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            assert abs(cov[i][j]) < 0.15, f"{method} cov[{i},{j}]={cov[i][j]} not ~0"
