from __future__ import annotations

import math

import nextstat


def _make_staggered_noiseless():
    entity = []
    time = []
    cohort = []
    y = []
    treat = []

    alpha = {"a": 1.0, "b": -2.0, "c": 0.5}
    gamma = {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0}
    g_by_e = {"a": 2, "b": 3, "c": None}
    tau = 2.0

    for e in ("a", "b", "c"):
        for t in (0, 1, 2, 3, 4):
            g = g_by_e[e]
            tr = int(g is not None and t >= g)
            entity.append(e)
            time.append(t)
            cohort.append(g)
            treat.append(tr)
            y.append(alpha[e] + gamma[t] + tau * tr)
    return {"y": y, "entity": entity, "time": time, "cohort": cohort, "treat": treat}


def test_staggered_did_recovers_noiseless_att():
    d = _make_staggered_noiseless()
    fit = nextstat.econometrics.staggered_did_fit(
        d["y"],
        entity=d["entity"],
        time=d["time"],
        cohort=d["cohort"],
        control_group="not_yet_treated",
    )
    assert abs(float(fit.att) - 2.0) <= 1e-12
    assert math.isfinite(float(fit.att_se))
    assert fit.n_entities == 3
    assert len(fit.cells) > 0
    for a in fit.event_att:
        assert abs(float(a) - 2.0) <= 1e-12


def test_staggered_did_from_formula_smoke_with_treat():
    d = _make_staggered_noiseless()
    data = {
        "y": d["y"],
        "entity": d["entity"],
        "t": d["time"],
        "treated": d["treat"],
    }
    fit = nextstat.econometrics.staggered_did_from_formula(
        "y ~ 1",
        data,
        entity="entity",
        time="t",
        treat="treated",
    )
    assert math.isfinite(float(fit.att))
    assert fit.control_group == "not_yet_treated"
