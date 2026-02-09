"""Contract tests for the composed model builder (Phase 5).

These tests are intentionally independent from pyhf. They validate that the
builder surface is usable from Python and integrates with generic inference
APIs (fit).
"""

from __future__ import annotations

import math

import nextstat


def _is_finite(x: float) -> bool:
    return math.isfinite(float(x))


def test_composed_glm_linear_regression_build_and_fit_smoke():
    x = [[0.0], [1.0], [2.0], [3.0]]
    y = [1.0, 3.0, 5.0, 7.0]  # y = 1 + 2x

    m = nextstat.ComposedGlmModel.linear_regression(x, y, include_intercept=True)
    assert m.n_params() >= 2
    assert len(m.parameter_names()) == m.n_params()
    assert len(m.suggested_init()) == m.n_params()
    assert len(m.suggested_bounds()) == m.n_params()

    nll0 = float(m.nll(m.suggested_init()))
    assert _is_finite(nll0)

    mle = nextstat.MaximumLikelihoodEstimator()
    r1 = mle.fit(m)
    r2 = nextstat.fit(m)
    for r in (r1, r2):
        assert isinstance(r.bestfit, list)
        assert len(r.bestfit) == m.n_params()
        assert _is_finite(float(r.nll))
        assert isinstance(r.success, bool)


def test_composed_glm_logistic_regression_build_smoke():
    # Simple separable dataset.
    x = [[-2.0], [-1.0], [1.0], [2.0]]
    y = [0, 0, 1, 1]

    m = nextstat.ComposedGlmModel.logistic_regression(x, y, include_intercept=True)
    assert m.n_params() >= 2
    nll0 = float(m.nll(m.suggested_init()))
    assert _is_finite(nll0)

