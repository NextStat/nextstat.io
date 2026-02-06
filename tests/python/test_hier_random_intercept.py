"""Contract tests for Phase 7 hierarchical helpers."""

from __future__ import annotations

import pytest

import nextstat


def test_logistic_random_intercept_builds_and_fits_smoke():
    # Two groups, simple separable-ish signal with group offsets.
    x = [[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]]
    y = [0, 1, 0, 1, 0, 1]
    group_idx = [0, 0, 1, 1, 0, 1]

    m = nextstat.hier.logistic_random_intercept(x=x, y=y, group_idx=group_idx, n_groups=2)
    assert m.n_params() == 2 + 2 + 2  # (intercept + beta) + (mu,sigma) + (alpha per group)

    fit = nextstat.fit(m)
    assert isinstance(fit.bestfit, list)
    assert len(fit.bestfit) == m.n_params()


def test_random_intercept_group_idx_length_mismatch_raises():
    x = [[0.0], [1.0], [0.0]]
    y = [0, 1, 0]
    group_idx = [0, 1]  # wrong length

    with pytest.raises(ValueError):
        nextstat.hier.logistic_random_intercept(x=x, y=y, group_idx=group_idx, n_groups=2)


def test_random_intercept_group_out_of_range_raises():
    x = [[0.0], [1.0], [0.0]]
    y = [0, 1, 0]
    group_idx = [0, 2, 0]  # 2 is out of range for n_groups=2

    with pytest.raises(ValueError):
        nextstat.hier.logistic_random_intercept(x=x, y=y, group_idx=group_idx, n_groups=2)


def test_logistic_random_intercept_sampling_smoke():
    x = [[0.0], [1.0], [0.0], [1.0]]
    y = [0, 1, 0, 1]
    group_idx = [0, 0, 1, 1]

    m = nextstat.hier.logistic_random_intercept(x=x, y=y, group_idx=group_idx, n_groups=2)
    raw = nextstat.sample(
        m,
        n_chains=1,
        n_warmup=30,
        n_samples=15,
        seed=123,
        init_jitter_rel=0.1,
    )
    assert isinstance(raw, dict)
    assert {"posterior", "sample_stats", "diagnostics"} <= set(raw.keys())

