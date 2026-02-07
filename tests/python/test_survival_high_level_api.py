from __future__ import annotations

import math

import pytest

import nextstat


def test_survival_high_level_parametric_fit_smoke() -> None:
    times = [0.5, 1.2, 0.7, 2.0, 0.9]
    events = [True, False, True, False, True]

    r = nextstat.survival.exponential.fit(times, events)
    assert r.model == "exponential"
    assert len(r.params) == 1
    assert len(r.se) == 1
    assert math.isfinite(float(r.nll))


def test_survival_high_level_cox_fit_smoke_and_ci() -> None:
    times = [2.0, 1.0, 1.0, 0.5, 0.5, 0.2]
    events = [True, True, False, True, False, False]
    x = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, -1.0],
        [0.0, -1.0],
        [0.5, 0.5],
    ]

    fit = nextstat.survival.cox_ph.fit(times, events, x, ties="efron", robust=True)
    assert isinstance(fit.coef, list) and len(fit.coef) == 2
    assert fit.se is not None and len(fit.se) == 2
    assert fit.robust_se is not None and len(fit.robust_se) == 2
    assert all(math.isfinite(float(v)) for v in fit.se)
    assert all(math.isfinite(float(v)) for v in fit.robust_se)

    cis = fit.confint(level=0.95, robust=False)
    rcis = fit.confint(level=0.95, robust=True)
    assert len(cis) == 2
    assert len(rcis) == 2
    for (lo, hi), b in zip(cis, fit.coef):
        assert lo <= float(b) <= hi

    hr = fit.hazard_ratios()
    assert len(hr) == 2
    assert all(float(v) > 0.0 for v in hr)

    hr_cis = fit.hazard_ratio_confint(level=0.95, robust=True)
    assert len(hr_cis) == 2
    for (lo, hi), b in zip(hr_cis, fit.coef):
        assert lo <= math.exp(float(b)) <= hi


def test_survival_high_level_cox_builder_is_callable() -> None:
    times = [2.0, 1.0, 1.0, 0.5]
    events = [True, True, False, True]
    x = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]]

    m = nextstat.survival.cox_ph(times, events, x, ties="breslow")
    assert m.n_params() == 2

