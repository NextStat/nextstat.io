"""Contract tests for random slopes (Phase 7.1 core + Python exposure)."""

from __future__ import annotations

import nextstat


def test_composed_glm_random_slope_build_fit_smoke():
    x = [[0.0], [1.0], [2.0], [3.0]]
    y = [0.0, 1.0, 2.0, 3.0]
    group_idx = [0, 0, 1, 1]

    m = nextstat.ComposedGlmModel.linear_regression(
        x,
        y,
        include_intercept=False,
        group_idx=group_idx,
        n_groups=2,
        coef_prior_mu=0.0,
        coef_prior_sigma=1.0,
        random_slope_feature_idx=0,
        random_slope_non_centered=True,
        random_intercept_non_centered=True,
    )
    names = m.parameter_names()
    assert any(n.startswith("u_beta1_") or n.startswith("z_u_beta1_") for n in names)

    fit = nextstat.fit(m)
    assert len(fit.bestfit) == m.n_params()

