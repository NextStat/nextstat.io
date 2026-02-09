from __future__ import annotations

import pytest

import nextstat


def test_panel_fixed_effects_within_estimator_recovers_beta():
    # Two entities, three periods each.
    # Construct noise orthogonal to within-x so beta is exact, but residuals are non-zero.
    x = [[0.0], [1.0], [2.0], [0.0], [1.0], [2.0]]
    y = [1.0, 0.0, 5.0, 0.0, 5.0, 4.0]  # alpha_B=1, beta=2, noise orthogonal
    entity = ["A", "A", "A", "B", "B", "B"]
    time = [0, 1, 2, 0, 1, 2]

    r = nextstat.panel.fit_fixed_effects(x, y, entity=entity, time=time, cluster="entity")
    assert r.coef == pytest.approx([2.0])
    assert r.n_obs == 6
    assert r.n_entities == 2
    assert r.cluster_kind == "entity"
    assert r.n_clusters == 2
    assert len(r.standard_errors) == 1
    assert len(r.covariance) == 1
    assert len(r.covariance[0]) == 1


def test_panel_fixed_effects_cluster_by_time():
    x = [[0.0], [1.0], [2.0], [0.0], [1.0], [2.0]]
    y = [1.0, 0.0, 5.0, 0.0, 5.0, 4.0]
    entity = ["A", "A", "A", "B", "B", "B"]
    time = [0, 1, 2, 0, 1, 2]

    r = nextstat.panel.fit_fixed_effects(x, y, entity=entity, time=time, cluster="time")
    assert r.cluster_kind == "time"
    assert r.n_clusters == 3


def test_panel_fixed_effects_cluster_time_requires_time():
    x = [[0.0], [1.0], [2.0], [0.0], [1.0], [2.0]]
    y = [1.0, 0.0, 5.0, 0.0, 5.0, 4.0]
    entity = ["A", "A", "A", "B", "B", "B"]

    with pytest.raises(ValueError, match="requires time"):
        nextstat.panel.fit_fixed_effects(x, y, entity=entity, cluster="time")

