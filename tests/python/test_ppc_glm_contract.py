"""Contract tests for PPC utilities (Phase 7.4)."""

from __future__ import annotations

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

