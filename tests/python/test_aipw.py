from __future__ import annotations

import math
import random

import pytest

import nextstat


def _randn(rng: random.Random) -> float:
    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


def _make_confounding_data(n: int, *, seed: int, tau: float) -> tuple[list[list[float]], list[float], list[int]]:
    rng = random.Random(seed)
    x: list[list[float]] = []
    y: list[float] = []
    t: list[int] = []

    for _ in range(n):
        x1 = _randn(rng)
        p = _sigmoid(-0.2 + 1.2 * x1)
        ti = 1 if rng.random() < p else 0
        ui = 0.2 * _randn(rng)
        yi = tau * float(ti) + 0.7 * x1 + ui
        x.append([x1])
        t.append(ti)
        y.append(yi)

    if sum(t) == 0 or sum(t) == n:
        raise RuntimeError("degenerate treatment assignment in fixture")
    return x, y, t


def test_aipw_ate_recovers_effect():
    x, y, t = _make_confounding_data(800, seed=0, tau=2.0)
    fit = nextstat.causal.aipw.aipw_fit(x, y, t, estimand="ate", trim_eps=1e-6)
    assert abs(float(fit.estimate) - 2.0) < 0.2
    assert float(fit.standard_error) > 0.0
    assert fit.propensity_diagnostics.n == 800


def test_aipw_att_recovers_effect():
    x, y, t = _make_confounding_data(800, seed=1, tau=1.5)
    fit = nextstat.causal.aipw.aipw_fit(x, y, t, estimand="att", trim_eps=1e-6)
    assert abs(float(fit.estimate) - 1.5) < 0.25
    assert float(fit.standard_error) > 0.0


def test_aipw_hooks_propensity_and_mu_smoke():
    x, y, t = _make_confounding_data(300, seed=2, tau=1.0)
    base = nextstat.causal.aipw.aipw_fit(x, y, t, estimand="ate")

    hooked = nextstat.causal.aipw.aipw_fit(
        x,
        y,
        t,
        estimand="ate",
        propensity_scores=base.propensity_scores,
        mu0=base.mu0,
        mu1=base.mu1,
    )
    assert abs(float(hooked.estimate) - float(base.estimate)) < 1e-9


def test_e_value_rr_basic():
    assert nextstat.causal.aipw.e_value_rr(1.0) == 1.0
    assert nextstat.causal.aipw.e_value_rr(2.0) > 2.0
    assert nextstat.causal.aipw.e_value_rr(0.5) == pytest.approx(nextstat.causal.aipw.e_value_rr(2.0))

