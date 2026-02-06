from __future__ import annotations

import math

import pytest

import nextstat


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


def test_ols_hc_covariance_matches_numpy():
    np = pytest.importorskip("numpy")

    x = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [0.2, 1.3, 2.1, 2.9, 4.2, 5.1]

    fit = nextstat.glm.linear.fit(x, y, include_intercept=True)
    n = len(y)

    X = np.column_stack([np.ones(n), np.array([r[0] for r in x])])
    beta = np.array(fit.coef, dtype=float)
    resid = np.array(y, dtype=float) - X.dot(beta)

    xtx_inv = np.linalg.inv(X.T.dot(X))
    h = np.sum(X * (X.dot(xtx_inv)), axis=1)

    def cov_expected(kind: str):
        if kind == "HC0":
            w = resid**2
        elif kind == "HC1":
            w = resid**2 * (float(n) / float(n - X.shape[1]))
        elif kind == "HC2":
            w = resid**2 / (1.0 - h)
        elif kind == "HC3":
            w = resid**2 / ((1.0 - h) ** 2)
        else:
            raise AssertionError(kind)
        meat = X.T.dot(w[:, None] * X)
        cov = xtx_inv.dot(meat).dot(xtx_inv)
        return cov

    for kind in ("HC0", "HC1", "HC2", "HC3"):
        cov_ns = nextstat.robust.ols_hc_covariance(
            x,
            resid.tolist(),
            include_intercept=True,
            kind=kind,  # type: ignore[arg-type]
        )
        cov_np = cov_expected(kind)
        for row_ns, row_np in zip(cov_ns, cov_np.tolist()):
            assert row_ns == pytest.approx(row_np, rel=1e-10, abs=1e-10)


def test_ols_cluster_covariance_matches_numpy():
    np = pytest.importorskip("numpy")

    x = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [0.2, 1.3, 2.1, 2.9, 4.2, 5.1]
    cluster = [0, 0, 1, 1, 2, 2]

    fit = nextstat.glm.linear.fit(x, y, include_intercept=True)
    pred = fit.predict(x)
    resid = [obs - p for obs, p in zip(y, pred)]
    n = len(y)

    X = np.column_stack([np.ones(n), np.array([r[0] for r in x])])
    xtx_inv = np.linalg.inv(X.T.dot(X))

    meat = np.zeros((X.shape[1], X.shape[1]), dtype=float)
    for g in sorted(set(cluster)):
        idx = np.array([i for i, cg in enumerate(cluster) if cg == g], dtype=int)
        Xg = X[idx, :]
        ug = resid
        rg = np.array([ug[i] for i in idx], dtype=float).reshape(-1, 1)
        sg = Xg.T.dot(rg)
        meat += sg.dot(sg.T)

    g = len(set(cluster))
    k = X.shape[1]
    scale = (float(g) / float(g - 1)) * ((float(n) - 1.0) / float(n - k))
    cov_np = xtx_inv.dot(meat * scale).dot(xtx_inv)

    cov_ns = nextstat.robust.ols_cluster_covariance(
        x,
        resid,
        cluster,
        include_intercept=True,
        df_correction=True,
    )
    for row_ns, row_np in zip(cov_ns, cov_np.tolist()):
        assert row_ns == pytest.approx(row_np, rel=1e-8, abs=1e-8)


def test_logistic_sandwich_covariance_matches_numpy():
    np = pytest.importorskip("numpy")

    x = [[-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]]
    y = [0, 0, 0, 1, 1, 1]

    fit = nextstat.glm.logistic.fit(x, y, include_intercept=True)
    cov_ns, _se_ns = nextstat.robust.logistic_sandwich_from_fit(fit, x, y)

    n = len(y)
    X = np.column_stack([np.ones(n), np.array([r[0] for r in x])])
    beta = np.array(fit.coef, dtype=float)
    eta = X.dot(beta)
    mu = np.array([_sigmoid(float(v)) for v in eta], dtype=float)
    w = mu * (1.0 - mu)

    H = X.T.dot(w[:, None] * X)
    inv_h = np.linalg.inv(H)
    resid = np.array(y, dtype=float) - mu
    meat = X.T.dot((resid**2)[:, None] * X)
    cov_np = inv_h.dot(meat).dot(inv_h)

    for row_ns, row_np in zip(cov_ns, cov_np.tolist()):
        assert row_ns == pytest.approx(row_np, rel=1e-8, abs=1e-8)


def test_poisson_sandwich_covariance_matches_numpy():
    np = pytest.importorskip("numpy")

    x = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [1, 1, 2, 4, 7, 12]

    fit = nextstat.glm.poisson.fit(x, y, include_intercept=True)
    cov_ns, _se_ns = nextstat.robust.poisson_sandwich_from_fit(fit, x, y)

    n = len(y)
    X = np.column_stack([np.ones(n), np.array([r[0] for r in x])])
    beta = np.array(fit.coef, dtype=float)
    mu = np.exp(X.dot(beta))
    w = mu

    H = X.T.dot(w[:, None] * X)
    inv_h = np.linalg.inv(H)
    resid = np.array(y, dtype=float) - mu
    meat = X.T.dot((resid**2)[:, None] * X)
    cov_np = inv_h.dot(meat).dot(inv_h)

    for row_ns, row_np in zip(cov_ns, cov_np.tolist()):
        assert row_ns == pytest.approx(row_np, rel=1e-10, abs=1e-10)
