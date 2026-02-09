"""Smoke tests for causal helpers: propensity scores + diagnostics (Phase 9.C)."""

from __future__ import annotations

import math
import random


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def test_propensity_ipw_diagnostics_smoke_improves_balance():
    import nextstat

    rng = random.Random(0)
    n = 1200
    x = []
    t = []

    for _ in range(n):
        x1 = rng.gauss(0.0, 1.0)
        x2 = rng.gauss(0.0, 1.0)
        p = _sigmoid(1.4 * x1 - 1.0 * x2)
        ti = 1 if rng.random() < p else 0
        x.append([x1, x2])
        t.append(ti)

    ps = nextstat.causal.propensity.fit(x, t, l2=1.0, clip_eps=1e-6)
    assert len(ps.propensity_scores) == n
    assert all(0.0 < float(v) < 1.0 for v in ps.propensity_scores)

    w = nextstat.causal.propensity.ipw_weights(
        t,
        ps.propensity_scores,
        estimand="ate",
        stabilized=True,
        max_weight=50.0,
    )
    assert len(w) == n
    assert all(math.isfinite(float(v)) and float(v) > 0.0 for v in w)

    d0 = nextstat.causal.propensity.diagnostics(x, t, ps.propensity_scores)
    dw = nextstat.causal.propensity.diagnostics(x, t, ps.propensity_scores, weights=w)

    assert math.isfinite(float(d0.max_abs_smd_unweighted))
    assert dw.max_abs_smd_weighted is not None
    assert math.isfinite(float(dw.max_abs_smd_weighted))

    # Weighting should reduce imbalance in this synthetic confounded dataset.
    assert float(dw.max_abs_smd_weighted) < float(d0.max_abs_smd_unweighted)

    assert dw.ess_treated is not None and dw.ess_control is not None
    assert float(dw.ess_treated) > 0.0 and float(dw.ess_control) > 0.0


def test_propensity_rejects_non_binary_treatment():
    import nextstat
    import pytest

    x = [[0.0], [1.0], [2.0]]
    t = [0, 2, 1]
    with pytest.raises(ValueError):
        nextstat.causal.propensity.fit(x, t)

