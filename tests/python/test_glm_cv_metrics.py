from __future__ import annotations


def test_metrics_smoke():
    from nextstat.glm import metrics

    assert metrics.rmse([0.0, 1.0], [0.0, 2.0]) > 0.0
    assert metrics.log_loss([0, 1], [0.1, 0.9]) < metrics.log_loss([0, 1], [0.5, 0.5])
    assert metrics.poisson_deviance([0, 1, 2], [1.0, 1.0, 2.0]) >= 0.0


def test_kfold_indices_deterministic():
    from nextstat.glm import cv

    s1 = cv.kfold_indices(10, 3, shuffle=True, seed=123)
    s2 = cv.kfold_indices(10, 3, shuffle=True, seed=123)
    assert s1 == s2

    # All indices appear exactly once across each fold's test set.
    all_test = sorted(ix for _train, test in s1 for ix in test)
    assert all_test == list(range(10))


def test_cross_val_score_linear_smoke():
    import nextstat
    from nextstat.glm import cv

    # y = 1 + 2x
    x = [[float(i)] for i in range(10)]
    y = [1.0 + 2.0 * float(i) for i in range(10)]

    r = cv.cross_val_score("linear", x, y, k=5, seed=0, metric="rmse")
    assert r.k == 5
    assert len(r.scores) == 5
    assert r.mean >= 0.0

