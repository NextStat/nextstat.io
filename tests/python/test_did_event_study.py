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


def test_did_twfe_supports_two_way_clustering():
    entity = ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"]
    t = [0, 1, 2, 3] * 3
    treat = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    post = [0, 0, 1, 1] * 3
    y = [10.0, 11.0, 16.0, 17.0, -1.0, 0.0, 1.0, 2.0, -2.0, -1.0, 0.0, 1.0]

    out = nextstat.econometrics.did_twfe_fit(
        None,
        y,
        treat=treat,
        post=post,
        entity=entity,
        time=t,
        cluster="two_way",
    )
    assert math.isfinite(float(out.att))
    assert math.isfinite(float(out.att_se))
    assert out.twfe.cluster == "two_way"


def test_two_way_demean_unbalanced_zeroes_entity_and_time_means():
    entity = ["a", "a", "a", "b", "b", "c", "c", "c", "c"]
    t = [0, 1, 3, 0, 2, 0, 1, 2, 3]  # unbalanced missing cells
    x = [[0.2], [0.5], [1.2], [0.1], [0.7], [0.3], [0.9], [1.4], [1.8]]
    alpha = {"a": 1.0, "b": -0.5, "c": 0.2}
    gamma = {0: 0.0, 1: 0.2, 2: 0.5, 3: 0.9}
    y = [alpha[e] + gamma[int(tt)] + 2.0 * row[0] for e, tt, row in zip(entity, t, x)]

    y_dd, x_dd, _ne, _nt = nextstat.econometrics._two_way_demean(y, x, entity, t)  # type: ignore[attr-defined]
    for e in sorted(set(entity)):
        idx = [i for i, ee in enumerate(entity) if ee == e]
        assert abs(sum(float(y_dd[i]) for i in idx) / float(len(idx))) < 1e-8
        assert abs(sum(float(x_dd[i][0]) for i in idx) / float(len(idx))) < 1e-8
    for tt in sorted(set(t)):
        idx = [i for i, tt2 in enumerate(t) if tt2 == tt]
        assert abs(sum(float(y_dd[i]) for i in idx) / float(len(idx))) < 1e-8
        assert abs(sum(float(x_dd[i][0]) for i in idx) / float(len(idx))) < 1e-8


def test_did_twfe_recovers_att_on_unbalanced_panel():
    entity = ["a", "a", "a", "b", "b", "c", "c", "c", "c", "d", "d", "d"]
    t = [0, 1, 3, 0, 2, 0, 1, 2, 3, 0, 2, 3]
    treat = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    post = [0 if tt < 2 else 1 for tt in t]
    att = 3.0
    alpha = {"a": 1.0, "b": 1.3, "c": -0.7, "d": 0.5}
    gamma = {0: 0.0, 1: 0.5, 2: 0.9, 3: 1.2}
    y = [alpha[e] + gamma[int(tt)] + att * float(tr) * float(po) for e, tt, tr, po in zip(entity, t, treat, post)]

    out = nextstat.econometrics.did_twfe_fit(
        None,
        y,
        treat=treat,
        post=post,
        entity=entity,
        time=t,
        cluster="entity",
    )
    assert abs(float(out.att) - att) < 1e-8


def test_did_twfe_wild_cluster_bootstrap_smoke():
    entity = []
    t = []
    treat = []
    post = []
    y = []
    alpha = {e: float(i) * 0.2 for i, e in enumerate("abcdefghijkl")}
    gamma = {k: float(k) * 0.3 for k in range(6)}
    for e in "abcdefghijkl":
        for tt in range(6):
            tr = 1 if e in "abcdef" else 0
            po = 1 if tt >= 3 else 0
            entity.append(e)
            t.append(tt)
            treat.append(tr)
            post.append(po)
            y.append(alpha[e] + gamma[tt] + 2.0 * float(tr) * float(po))

    wb = nextstat.econometrics.did_twfe_wild_cluster_bootstrap(
        None,
        y,
        treat=treat,
        post=post,
        entity=entity,
        time=t,
        cluster_on="entity",
        n_boot=199,
        seed=7,
        weight_dist="webb6",
    )
    assert math.isfinite(float(wb.att))
    assert math.isfinite(float(wb.att_se_bootstrap))
    assert 0.0 <= float(wb.p_value) <= 1.0
    assert wb.n_boot == 199


def test_did_twfe_wild_cluster_bootstrap_webb_requires_6_clusters():
    entity = ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"]
    t = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    treat = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    post = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    y = [1.0 + 0.5 * tt + 2.0 * tr * po for tt, tr, po in zip(t, treat, post)]
    with pytest.raises(ValueError, match="at least 6 clusters"):
        nextstat.econometrics.did_twfe_wild_cluster_bootstrap(
            None,
            y,
            treat=treat,
            post=post,
            entity=entity,
            time=t,
            cluster_on="entity",
            n_boot=99,
            weight_dist="webb6",
        )
