from __future__ import annotations

import math

import pytest

import nextstat


def test_did_twfe_recovers_att_no_noise():
    # Two entities, four periods. One treated entity, policy at t>=2.
    entity = ["a", "a", "a", "a", "b", "b", "b", "b"]
    t = [0, 1, 2, 3, 0, 1, 2, 3]
    treat = [1, 1, 1, 1, 0, 0, 0, 0]
    post = [0, 0, 1, 1, 0, 0, 1, 1]

    att = 5.0

    # Add entity and time effects to ensure FE matter.
    alpha = {"a": 10.0, "b": -3.0}
    gamma = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0}
    y = []
    for e, ti, tr, po in zip(entity, t, treat, post):
        y.append(alpha[e] + gamma[int(ti)] + att * float(tr) * float(po))

    out = nextstat.econometrics.did_twfe_fit(
        None,
        y,
        treat=treat,
        post=post,
        entity=entity,
        time=t,
        cluster="entity",
    )
    assert abs(float(out.att) - att) <= 1e-12
    assert math.isfinite(float(out.att_se))


def test_did_twfe_from_formula_smoke_and_names():
    data = {
        "y": [10.0, 11.0, 17.0, 18.0, -3.0, -2.0, 2.0, 3.0],
        "x1": [0.0, 0.5, 1.0, 1.5, 0.0, 0.5, 1.0, 1.5],
        "entity": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "t": [0, 1, 2, 3, 0, 1, 2, 3],
        "treated": [1, 1, 1, 1, 0, 0, 0, 0],
        "post": [0, 0, 1, 1, 0, 0, 1, 1],
    }
    out = nextstat.econometrics.did_twfe_from_formula(
        "y ~ 1 + x1",
        data,
        entity="entity",
        time="t",
        treat="treated",
        post="post",
        cluster="entity",
    )
    assert out.twfe.column_names[0] == "treat_post"
    # x1 is perfectly absorbed by time FE in this constructed example and may be dropped.
    assert "treat_post" in out.twfe.column_names


def test_event_study_twfe_recovers_step_effect_no_noise():
    # Event time at 2, window (-1,2), reference=-1 => estimate k=0,1,2.
    entity = ["a", "a", "a", "a", "b", "b", "b", "b"]
    t = [0, 1, 2, 3, 0, 1, 2, 3]
    treat = [1, 1, 1, 1, 0, 0, 0, 0]
    event_time = 2

    # Effect: 0 pre, then +5 at k>=0 (k=0,1). Keep k=2 also +5.
    eff = {0: 5.0, 1: 5.0, 2: 5.0}

    alpha = {"a": 10.0, "b": -3.0}
    gamma = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0}
    y = []
    for e, ti, tr in zip(entity, t, treat):
        rt = int(ti) - int(event_time)
        add = 0.0
        if tr and rt in eff:
            add = eff[rt]
        y.append(alpha[e] + gamma[int(ti)] + add)

    es = nextstat.econometrics.event_study_twfe_fit(
        y,
        treat=treat,
        time=t,
        event_time=event_time,
        entity=entity,
        window=(-1, 2),
        reference=-1,
        cluster="entity",
    )
    # rel_time=2 does not exist in this sample (t only goes to 3), so that bin is dropped.
    assert es.rel_times == [0, 1]
    for b in es.coef:
        assert abs(float(b) - 5.0) <= 1e-12


def test_event_study_twfe_from_formula_requires_time_column_for_cluster_time():
    data = {
        "y": [1.0, 2.0, 3.0, 4.0],
        "entity": ["a", "a", "b", "b"],
        "t": [0, 1, 0, 1],
        "treated": [1, 1, 0, 0],
    }
    es = nextstat.econometrics.event_study_twfe_from_formula(
        "y ~ 1",
        data,
        entity="entity",
        time="t",
        treat="treated",
        event_time=0,
        cluster="time",
    )
    assert es.n_obs == 4
