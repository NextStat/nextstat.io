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

    fit = nextstat.survival.cox_ph.fit(times, events, x, ties="efron", robust=True, compute_baseline=True)
    assert isinstance(fit.coef, list) and len(fit.coef) == 2
    assert fit.se is not None and len(fit.se) == 2
    assert fit.robust_se is not None and len(fit.robust_se) == 2
    assert all(math.isfinite(float(v)) for v in fit.se)
    assert all(math.isfinite(float(v)) for v in fit.robust_se)
    assert fit.robust_kind in (None, "hc0", "cluster")

    # Baseline survival predictions are step functions in time; smoke-check shape and bounds.
    s = fit.predict_survival([[0.0, 0.0]])
    assert len(s) == 1
    assert len(s[0]) == len(fit.baseline_times)
    assert all(0.0 < float(v) <= 1.0 for v in s[0])
    assert all(float(s[0][i + 1]) <= float(s[0][i]) for i in range(len(s[0]) - 1))

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


def test_survival_cox_cluster_robust_unique_clusters_matches_hc0() -> None:
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
    groups = list(range(len(times)))  # each row is its own cluster

    # With no small-sample correction, cluster-robust with unique clusters should match HC0.
    fit_hc0 = nextstat.survival.cox_ph.fit(
        times, events, x, ties="efron", robust=True, groups=None
    )
    fit_cl = nextstat.survival.cox_ph.fit(
        times, events, x, ties="efron", robust=True, groups=groups, cluster_correction=False
    )

    assert fit_hc0.robust_kind == "hc0"
    assert fit_cl.robust_kind == "cluster"
    assert fit_hc0.robust_se is not None and fit_cl.robust_se is not None
    assert [float(v) for v in fit_cl.robust_se] == pytest.approx(
        [float(v) for v in fit_hc0.robust_se], rel=0.0, abs=1e-10
    )


def test_survival_cox_cluster_correction_scales_cov() -> None:
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
    groups = list(range(len(times)))  # each row is its own cluster

    fit_unc = nextstat.survival.cox_ph.fit(
        times, events, x, ties="efron", robust=True, groups=groups, cluster_correction=False
    )
    fit_cor = nextstat.survival.cox_ph.fit(
        times, events, x, ties="efron", robust=True, groups=groups, cluster_correction=True
    )

    assert fit_unc.robust_se is not None and fit_cor.robust_se is not None
    factor = float(len(groups)) / float(len(groups) - 1)  # G/(G-1)
    # SE should scale by sqrt(factor) when covariance is scaled by factor.
    assert [float(v) for v in fit_cor.robust_se] == pytest.approx(
        [math.sqrt(factor) * float(v) for v in fit_unc.robust_se], rel=0.0, abs=1e-10
    )


def test_survival_cox_schoenfeld_sum_matches_score() -> None:
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

    fit = nextstat.survival.cox_ph.fit(times, events, x, ties="efron", robust=False, compute_cov=False)
    beta = [float(v) for v in fit.coef]

    # Score identity: sum Schoenfeld residuals == grad_ll == -grad_nll.
    sr = nextstat.survival.cox_ph_schoenfeld(times, events, x, ties="efron", coef=beta)
    assert len(sr.event_times) == sum(1 for v in events if v)
    assert len(sr.residuals) == len(sr.event_times)

    score = [0.0, 0.0]
    for r in sr.residuals:
        score[0] += float(r[0])
        score[1] += float(r[1])

    m = nextstat.CoxPhModel(times, events, x, ties="efron")
    g = [float(v) for v in m.grad_nll(beta)]
    assert score[0] == pytest.approx(-g[0], rel=0.0, abs=1e-10)
    assert score[1] == pytest.approx(-g[1], rel=0.0, abs=1e-10)


def test_survival_cox_ph_test_smoke() -> None:
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
    fit = nextstat.survival.cox_ph.fit(times, events, x, ties="efron", robust=False, compute_cov=False)
    tests = nextstat.survival.cox_ph_ph_test(times, events, x, ties="efron", coef=fit.coef)
    assert isinstance(tests, list) and len(tests) == 2
    for row in tests:
        assert 0.0 <= float(row["p"]) <= 1.0
