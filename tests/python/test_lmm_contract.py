"""Contract tests for linear mixed models (Phase 9 Pack B).

These tests validate that the marginal-likelihood LMM surface is usable from
Python and integrates with generic inference APIs (fit).
"""

from __future__ import annotations

import math

import pytest

import nextstat


def _is_finite(x: float) -> bool:
    return math.isfinite(float(x))


def test_lmm_marginal_random_intercept_fit_smoke() -> None:
    x = [[1.0], [2.0], [3.0], [1.5], [2.5], [3.5]]
    y = [1.0, 2.1, 2.9, 1.4, 2.7, 3.6]
    group_idx = [0, 0, 0, 1, 1, 1]

    m = nextstat.LmmMarginalModel(
        x,
        y,
        include_intercept=False,
        group_idx=group_idx,
        n_groups=2,
    )

    assert m.n_params() == 3
    assert m.parameter_names() == ["beta1", "log_sigma_y", "log_tau_alpha"]
    assert len(m.suggested_init()) == m.n_params()

    nll0 = float(m.nll(m.suggested_init()))
    assert _is_finite(nll0)
    g0 = m.grad_nll(m.suggested_init())
    assert len(g0) == m.n_params()
    assert all(_is_finite(float(v)) for v in g0)

    r = nextstat.fit(m)
    assert len(r.bestfit) == m.n_params()
    assert _is_finite(float(r.nll))


def test_lmm_marginal_random_intercept_slope_fit_smoke() -> None:
    x = [
        [1.0, 0.1],
        [1.0, 0.2],
        [1.0, 0.3],
        [1.0, 0.1],
        [1.0, 0.2],
        [1.0, 0.3],
    ]
    y = [1.0, 1.1, 0.9, 1.4, 1.6, 1.3]
    group_idx = [0, 0, 0, 1, 1, 1]

    m = nextstat.LmmMarginalModel(
        x,
        y,
        include_intercept=True,
        group_idx=group_idx,
        n_groups=2,
        random_slope_feature_idx=1,
    )

    assert m.n_params() == 6
    assert m.parameter_names() == [
        "intercept",
        "beta1",
        "beta2",
        "log_sigma_y",
        "log_tau_alpha",
        "log_tau_u_beta2",
    ]

    nll0 = float(m.nll(m.suggested_init()))
    assert _is_finite(nll0)

    r = nextstat.fit(m)
    assert len(r.bestfit) == m.n_params()
    assert _is_finite(float(r.nll))


def test_lmm_marginal_group_idx_length_mismatch_raises() -> None:
    x = [[1.0], [2.0]]
    y = [1.0, 2.0]
    with pytest.raises(ValueError):
        nextstat.LmmMarginalModel(x, y, group_idx=[0])  # wrong length
