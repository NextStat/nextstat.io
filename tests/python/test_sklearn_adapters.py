from __future__ import annotations

import nextstat


def test_sklearn_linear_regression_fit_predict_smoke():
    x = [[0.0], [1.0], [2.0], [3.0]]
    y = [0.2, 1.1, 2.0, 3.2]

    m = nextstat.sklearn.NextStatLinearRegression(include_intercept=True)
    m.fit(x, y)
    pred = m.predict(x)
    assert isinstance(pred, list)
    assert len(pred) == len(y)
    assert isinstance(m.coef_, list)
    assert isinstance(m.intercept_, float)


def test_sklearn_logistic_regression_fit_predict_smoke():
    x = [[-2.0], [-1.0], [0.0], [1.0], [2.0]]
    y = [0, 0, 0, 1, 1]

    m = nextstat.sklearn.NextStatLogisticRegression(include_intercept=True, l2=10.0)
    m.fit(x, y)
    pred = m.predict(x)
    proba = m.predict_proba(x)
    assert isinstance(pred, list)
    assert len(pred) == len(y)
    assert isinstance(proba, list)
    assert len(proba) == len(y)
    assert len(proba[0]) == 2


def test_sklearn_poisson_regressor_fit_predict_smoke():
    x = [[0.0], [1.0], [2.0], [3.0], [4.0]]
    y = [1, 1, 2, 4, 7]

    m = nextstat.sklearn.NextStatPoissonRegressor(include_intercept=True)
    m.fit(x, y)
    mu = m.predict(x)
    assert isinstance(mu, list)
    assert len(mu) == len(y)

