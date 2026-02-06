"""Contract tests for ordered probit regression (Phase 9.C)."""

from __future__ import annotations

import math
import random


def _normal_cdf(x: float) -> float:
    # Phi(x) = 0.5 * erfc(-x / sqrt(2))
    return 0.5 * math.erfc(-float(x) / math.sqrt(2.0))


def _randn(rng: random.Random) -> float:
    # Box-Muller (deterministic with rng).
    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _sample_ordered_probit_y(rng: random.Random, *, eta: float, cuts: list[float]) -> int:
    if len(cuts) < 1:
        raise ValueError("cuts must contain at least 1 cutpoint")
    if any(not math.isfinite(c) for c in cuts):
        raise ValueError("cuts must be finite")

    cdf = [_normal_cdf(float(cuts[0]) - float(eta))]
    for c in cuts[1:]:
        cdf.append(_normal_cdf(float(c) - float(eta)))
    probs = [cdf[0]]
    for j in range(1, len(cdf)):
        probs.append(cdf[j] - cdf[j - 1])
    probs.append(1.0 - cdf[-1])

    u = rng.random()
    acc = 0.0
    for k, p in enumerate(probs):
        acc += float(p)
        if u <= acc:
            return int(k)
    return int(len(probs) - 1)


def test_ordered_probit_fit_recovers_sign_and_predict_proba_shapes():
    import nextstat

    rng = random.Random(0)

    beta_true = 1.0
    cuts_true = [-0.4, 0.6]  # 3 levels

    n = 700
    x = []
    y = []
    for _ in range(n):
        xi = _randn(rng)
        eta = beta_true * xi
        yi = _sample_ordered_probit_y(rng, eta=eta, cuts=cuts_true)
        x.append([xi])
        y.append(yi)

    fit = nextstat.ordinal.ordered_probit.fit(x, y, n_levels=3)
    assert math.isfinite(float(fit.nll))
    assert int(fit.n_levels) == 3
    assert len(fit.coef) == 1
    assert len(fit.cutpoints) == 2
    assert float(fit.cutpoints[0]) < float(fit.cutpoints[1])

    assert bool(fit.converged)
    assert float(fit.coef[0]) > 0.0

    ps = fit.predict_proba([[-2.0], [0.0], [2.0]])
    assert len(ps) == 3
    for row in ps:
        assert len(row) == 3
        s = sum(float(v) for v in row)
        assert abs(s - 1.0) < 1e-12
        assert all(float(v) >= 0.0 for v in row)

    assert float(ps[2][2]) > float(ps[0][2])


def test_ordered_probit_optional_parity_vs_statsmodels_predict_proba():
    # Optional parity check: only runs when statsmodels is installed.
    import pytest

    pytest.importorskip("statsmodels")
    pytest.importorskip("numpy")

    import numpy as np
    import nextstat
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    rng = random.Random(1)
    beta_true = 0.7
    cuts_true = [-0.2, 0.8]

    n = 900
    x = []
    y = []
    for _ in range(n):
        xi = _randn(rng)
        eta = beta_true * xi
        yi = _sample_ordered_probit_y(rng, eta=eta, cuts=cuts_true)
        x.append([xi])
        y.append(yi)

    ns_fit = nextstat.ordinal.ordered_probit.fit(x, y, n_levels=3)
    assert bool(ns_fit.converged)

    X = np.asarray(x, dtype=float)
    Y = np.asarray(y, dtype=int)
    sm = OrderedModel(Y, X, distr="probit")
    sm_res = sm.fit(method="bfgs", disp=False, maxiter=200)

    grid = [[-2.0], [-1.0], [0.0], [1.0], [2.0]]
    ns_p = ns_fit.predict_proba(grid)
    sm_p = sm.model.predict(sm_res.params, exog=np.asarray(grid, dtype=float))

    sm_p = np.asarray(sm_p, dtype=float)
    assert sm_p.shape == (len(grid), 3)
    max_abs = float(np.max(np.abs(sm_p - np.asarray(ns_p, dtype=float))))
    assert max_abs < 0.10

