"""Contract tests for PPC utilities (Phase 7.4)."""

from __future__ import annotations

import math
import random

import nextstat


def test_ppc_glm_from_sample_logistic_random_intercept_smoke():
    x = [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]]
    y = [0, 1, 0, 1, 0, 1]
    group_idx = [0, 0, 1, 1, 0, 1]

    spec = nextstat.data.GlmSpec.logistic_regression(
        x=x,
        y=y,
        include_intercept=False,
        group_idx=group_idx,
        n_groups=2,
        coef_prior_mu=0.0,
        coef_prior_sigma=1.0,
    )
    model = spec.build()

    raw = nextstat.sample(
        model,
        n_chains=1,
        n_warmup=30,
        n_samples=20,
        seed=123,
        init_jitter_rel=0.1,
    )

    out = nextstat.ppc.ppc_glm_from_sample(spec, raw, n_draws=5, seed=0)
    assert isinstance(out.observed, dict)
    assert isinstance(out.replicated, list)
    assert len(out.replicated) == 5
    assert "mean" in out.observed
    assert "mean" in out.replicated[0]


def test_ppc_replicate_glm_linear_smoke():
    x = [[0.0], [1.0], [2.0], [3.0]]
    y = [0.2, 1.1, 2.0, 3.2]
    group_idx = [0, 0, 1, 1]
    spec = nextstat.data.GlmSpec.linear_regression(
        x=x,
        y=y,
        include_intercept=False,
        group_idx=group_idx,
        n_groups=2,
        coef_prior_mu=0.0,
        coef_prior_sigma=1.0,
    )
    # Minimal centered random-intercept draw.
    draw = {
        "beta1": 0.5,
        "mu_alpha": 0.0,
        "sigma_alpha": 1.0,
        "alpha1": 0.1,
        "alpha2": -0.2,
    }
    y_rep = nextstat.ppc.replicate_glm(spec, draw, seed=1)
    assert isinstance(y_rep, list)
    assert len(y_rep) == len(y)


def test_ppc_replicate_glm_linear_random_slope_centered_matches_manual_eta():
    # This locks in that PPC accounts for varying slopes (critical for Phase 7).
    x = [[1.0], [1.0], [2.0], [2.0]]
    y = [0.0, 0.0, 0.0, 0.0]
    group_idx = [0, 1, 0, 1]
    spec = nextstat.data.GlmSpec(
        kind="linear",
        x=x,
        y=y,
        include_intercept=False,
        group_idx=group_idx,
        n_groups=2,
        random_slope_feature_idx=0,
        random_slope_non_centered=False,
    )

    draw = {
        "beta1": 0.1,
        "mu_alpha": 0.0,
        "sigma_alpha": 1.0,
        "alpha1": 0.0,
        "alpha2": 0.0,
        "mu_u_beta1": 0.0,
        "sigma_u_beta1": 1.0,
        "u_beta1_1": 10.0,
        "u_beta1_2": -10.0,
    }

    got = nextstat.ppc.replicate_glm(spec, draw, seed=123)

    # Manual replicate with the expected eta (same RNG behavior).
    rng = random.Random(123)
    expected = []
    for xr, g in zip(x, group_idx):
        eta = draw["beta1"] * xr[0] + draw[f"alpha{g+1}"] + draw[f"u_beta1_{g+1}"] * xr[0]
        expected.append(float(rng.gauss(float(eta), 1.0)))

    assert got == expected


def test_ppc_replicate_glm_linear_correlated_intercept_slope_matches_manual_eta():
    x = [[1.0], [1.0], [2.0], [2.0]]
    y = [0.0, 0.0, 0.0, 0.0]
    group_idx = [0, 1, 0, 1]
    spec = nextstat.data.GlmSpec(
        kind="linear",
        x=x,
        y=y,
        include_intercept=False,
        group_idx=group_idx,
        n_groups=2,
        correlated_feature_idx=0,
    )

    draw = {
        "beta1": 0.0,
        "mu_alpha": 1.0,
        "mu_u_beta1": 2.0,
        "tau_alpha": 0.5,
        "tau_u_beta1": 1.0,
        "rho_alpha_u_beta1": 0.25,
        "z_alpha_g1": 1.0,
        "z_u_beta1_g1": 2.0,
        "z_alpha_g2": -1.0,
        "z_u_beta1_g2": -2.0,
    }

    got = nextstat.ppc.replicate_glm(spec, draw, seed=7)

    rng = random.Random(7)
    expected = []
    rho = float(draw["rho_alpha_u_beta1"])
    s = math.sqrt(1.0 - rho * rho)
    for xr, g in zip(x, group_idx):
        z1 = float(draw[f"z_alpha_g{g+1}"])
        z2 = float(draw[f"z_u_beta1_g{g+1}"])
        alpha_g = float(draw["mu_alpha"]) + float(draw["tau_alpha"]) * z1
        u_g = float(draw["mu_u_beta1"]) + float(draw["tau_u_beta1"]) * (rho * z1 + s * z2)
        eta = alpha_g + u_g * xr[0]
        expected.append(float(rng.gauss(float(eta), 1.0)))

    assert got == expected


def test_ppc_replicate_glm_poisson_smoke():
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = [0, 0, 0, 0]
    group_idx = [0, 0, 1, 1]
    offset = [0.0, 0.0, 0.0, 0.0]
    spec = nextstat.data.GlmSpec(
        kind="poisson",
        x=x,
        y=y,
        include_intercept=True,
        group_idx=group_idx,
        n_groups=2,
        offset=offset,
    )

    draw = {
        "intercept": math.log(3.0),
        "beta1": 0.0,
        "mu_alpha": 0.0,
        "sigma_alpha": 1.0,
        "alpha1": 0.0,
        "alpha2": 0.0,
    }
    y_rep = nextstat.ppc.replicate_glm(spec, draw, seed=1)
    assert len(y_rep) == len(y)
    assert all(float(v) >= 0.0 for v in y_rep)
    assert all(float(v).is_integer() for v in y_rep)
