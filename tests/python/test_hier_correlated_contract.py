"""Contract tests for correlated random effects (Phase 7.3 Python exposure)."""

from __future__ import annotations

import pytest

import nextstat


def test_composed_glm_correlated_intercept_slope_build_fit_sample_smoke():
    x = [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]]
    y = [0, 1, 0, 1, 0, 1]
    group_idx = [0, 0, 1, 1, 0, 1]

    m = nextstat.ComposedGlmModel.logistic_regression(
        x,
        y,
        include_intercept=False,
        group_idx=group_idx,
        n_groups=2,
        coef_prior_mu=0.0,
        coef_prior_sigma=1.0,
        correlated_feature_idx=0,
        lkj_eta=2.0,
    )
    names = m.parameter_names()
    assert "tau_alpha" in names
    assert "rho_alpha_u_beta1" in names

    fit = nextstat.fit(m)
    assert len(fit.bestfit) == m.n_params()

    raw = nextstat.sample(
        m,
        n_chains=1,
        n_warmup=30,
        n_samples=15,
        seed=7,
        init_jitter_rel=0.1,
    )
    assert isinstance(raw, dict)
    assert {"posterior", "sample_stats", "diagnostics"} <= set(raw.keys())


def test_correlated_requires_group_idx():
    x = [[0.0], [1.0]]
    y = [0, 1]
    with pytest.raises(ValueError, match="require group_idx"):
        nextstat.ComposedGlmModel.logistic_regression(
            x,
            y,
            include_intercept=False,
            correlated_feature_idx=0,
        )


def test_correlated_cannot_mix_with_random_slope_args():
    x = [[0.0], [1.0], [0.0], [1.0]]
    y = [0, 1, 0, 1]
    group_idx = [0, 0, 1, 1]
    with pytest.raises(ValueError, match="cannot be combined"):
        nextstat.ComposedGlmModel.logistic_regression(
            x,
            y,
            include_intercept=False,
            group_idx=group_idx,
            n_groups=2,
            correlated_feature_idx=0,
            random_slope_feature_idx=0,
        )

