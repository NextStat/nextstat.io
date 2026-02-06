"""Robust covariance estimators (sandwich / heteroskedasticity / cluster).

This module is intentionally dependency-light (no numpy required). It provides
baseline robust covariance estimators for small/medium p using pure-Python
linear algebra helpers.

Notes
- OLS HC0-HC3 match the classic White/MacKinnon formulas.
- Cluster-robust is 1-way (CR0 with optional finite-sample correction).
- For GLMs we provide a baseline sandwich estimator using canonical-link score
  contributions and the Fisher information approximation (X' W X).
  This is a starting point and does not cover penalized/MAP fits rigorously.
"""

from __future__ import annotations

import math
from typing import Any, List, Literal, Optional, Sequence, Tuple

from .glm._linalg import add_intercept, as_2d_float_list, mat_inv, mat_mul, mat_t


Matrix = List[List[float]]
Vector = List[float]

HcKind = Literal["HC0", "HC1", "HC2", "HC3"]


def cov_to_se(cov: Sequence[Sequence[float]]) -> List[float]:
    """Extract standard errors (sqrt(diag(cov)))."""
    out: List[float] = []
    for i, row in enumerate(cov):
        if i >= len(row):
            raise ValueError("cov must be square")
        v = float(row[i])
        out.append(math.sqrt(v) if v > 0.0 else float("inf"))
    if len(out) != len(cov):
        raise ValueError("cov must be square")
    return out


def _outer(x: Vector) -> Matrix:
    return [[xi * xj for xj in x] for xi in x]


def _mat_add_inplace(a: Matrix, b: Matrix) -> None:
    if len(a) != len(b):
        raise ValueError("matrix shape mismatch")
    for i in range(len(a)):
        if len(a[i]) != len(b[i]):
            raise ValueError("matrix shape mismatch")
        for j in range(len(a[i])):
            a[i][j] += b[i][j]


def _mat_scale_inplace(a: Matrix, s: float) -> None:
    fs = float(s)
    for i in range(len(a)):
        for j in range(len(a[i])):
            a[i][j] *= fs


def _design_matrix(x: Sequence[Sequence[float]], *, include_intercept: bool) -> Matrix:
    x2 = as_2d_float_list(x)
    return add_intercept(x2) if include_intercept else x2


def _validate_lengths(n: int, *seqs: Sequence[Any]) -> None:
    for s in seqs:
        if len(s) != n:
            raise ValueError("length mismatch between inputs")


def _weighted_xtx(xd: Matrix, w: Sequence[float]) -> Matrix:
    n = len(xd)
    if n == 0:
        raise ValueError("x must have at least 1 row")
    k = len(xd[0]) if xd else 0
    if any(len(row) != k for row in xd):
        raise ValueError("x must be rectangular")
    _validate_lengths(n, w)

    out: Matrix = [[0.0] * k for _ in range(k)]
    for wi, xi in zip(w, xd):
        oi = _outer([float(v) for v in xi])
        _mat_scale_inplace(oi, float(wi))
        _mat_add_inplace(out, oi)
    return out


def ols_hc_covariance(
    x: Sequence[Sequence[float]],
    residuals: Sequence[float],
    *,
    include_intercept: bool,
    kind: HcKind = "HC1",
) -> Matrix:
    """Compute OLS heteroskedasticity-consistent covariance (HC0-HC3).

    Inputs follow NextStat GLM conventions:
    - `x` excludes the intercept column; set `include_intercept=True` to add it.
    - `residuals` is length-n with the same observation order as `x`.
    """
    xd = _design_matrix(x, include_intercept=include_intercept)
    n = len(xd)
    if n == 0:
        raise ValueError("x must have at least 1 row")
    u = [float(v) for v in residuals]
    _validate_lengths(n, u)

    k = len(xd[0]) if xd else 0
    if any(len(row) != k for row in xd):
        raise ValueError("x must be rectangular")
    if k == 0:
        raise ValueError("x must have at least 1 column (add intercept or features)")

    if kind != "HC0" and n <= k:
        raise ValueError("Need n > n_params for HC1/HC2/HC3")

    xt = mat_t(xd)
    xtx_inv = mat_inv(mat_mul(xt, xd))

    # Leverage (diag of hat matrix) for HC2/HC3.
    h: Optional[List[float]] = None
    if kind in ("HC2", "HC3"):
        h = []
        for xi in xd:
            # h_i = x_i' (X'X)^-1 x_i
            # compute v = (X'X)^-1 x_i then dot(x_i, v)
            v = [sum(xtx_inv[r][c] * xi[c] for c in range(k)) for r in range(k)]
            h.append(sum(xi[j] * v[j] for j in range(k)))

    meat: Matrix = [[0.0] * k for _ in range(k)]
    for i, xi in enumerate(xd):
        ui = float(u[i])
        if kind == "HC0" or kind == "HC1":
            wi = ui * ui
        else:
            assert h is not None
            denom = 1.0 - float(h[i])
            if not (denom > 0.0):
                raise ValueError("invalid leverage (1 - h_i) <= 0; design may be singular")
            if kind == "HC2":
                wi = (ui * ui) / denom
            else:
                wi = (ui * ui) / (denom * denom)

        oi = _outer([float(v) for v in xi])
        _mat_scale_inplace(oi, wi)
        _mat_add_inplace(meat, oi)

    if kind == "HC1":
        _mat_scale_inplace(meat, float(n) / float(n - k))

    cov = mat_mul(mat_mul(xtx_inv, meat), xtx_inv)
    return cov


def ols_cluster_covariance(
    x: Sequence[Sequence[float]],
    residuals: Sequence[float],
    cluster: Sequence[Any],
    *,
    include_intercept: bool,
    df_correction: bool = True,
) -> Matrix:
    """Compute 1-way cluster-robust covariance for OLS.

    This is a baseline CR0 estimator with an optional finite-sample correction:
    - multiplies by (G/(G-1)) * ((n-1)/(n-k))
    """
    xd = _design_matrix(x, include_intercept=include_intercept)
    n = len(xd)
    if n == 0:
        raise ValueError("x must have at least 1 row")
    u = [float(v) for v in residuals]
    _validate_lengths(n, u, cluster)

    k = len(xd[0]) if xd else 0
    if any(len(row) != k for row in xd):
        raise ValueError("x must be rectangular")
    if k == 0:
        raise ValueError("x must have at least 1 column (add intercept or features)")
    if n <= k:
        raise ValueError("Need n > n_params for cluster-robust covariance")

    xt = mat_t(xd)
    xtx_inv = mat_inv(mat_mul(xt, xd))

    # s_g = sum_{i in g} x_i * u_i (k-vector)
    by_g: dict[Any, Vector] = {}
    for xi, ui, gi in zip(xd, u, cluster):
        s = by_g.get(gi)
        if s is None:
            by_g[gi] = [float(v) * float(ui) for v in xi]
        else:
            for j in range(k):
                s[j] += float(xi[j]) * float(ui)

    g = len(by_g)
    if g < 2:
        raise ValueError("cluster must have at least 2 distinct groups")

    meat: Matrix = [[0.0] * k for _ in range(k)]
    for s in by_g.values():
        _mat_add_inplace(meat, _outer(s))

    if df_correction:
        scale = (float(g) / float(g - 1)) * ((float(n) - 1.0) / float(n - k))
        _mat_scale_inplace(meat, scale)

    return mat_mul(mat_mul(xtx_inv, meat), xtx_inv)


def ols_hc_from_fit(
    fit: Any,
    x: Sequence[Sequence[float]],
    y: Sequence[float],
    *,
    kind: HcKind = "HC1",
) -> Tuple[Matrix, List[float]]:
    """Convenience wrapper: compute OLS HC covariance/SE from a LinearFit-like object."""
    if not hasattr(fit, "predict") or not hasattr(fit, "include_intercept"):
        raise TypeError("fit must look like nextstat.glm.linear.LinearFit")
    y2 = [float(v) for v in y]
    pred = list(getattr(fit, "predict")(x))
    _validate_lengths(len(y2), pred)
    resid = [obs - p for obs, p in zip(y2, pred)]

    cov = ols_hc_covariance(x, resid, include_intercept=bool(getattr(fit, "include_intercept")), kind=kind)
    return cov, cov_to_se(cov)


def ols_cluster_from_fit(
    fit: Any,
    x: Sequence[Sequence[float]],
    y: Sequence[float],
    cluster: Sequence[Any],
    *,
    df_correction: bool = True,
) -> Tuple[Matrix, List[float]]:
    """Convenience wrapper: compute OLS cluster covariance/SE from a LinearFit-like object."""
    if not hasattr(fit, "predict") or not hasattr(fit, "include_intercept"):
        raise TypeError("fit must look like nextstat.glm.linear.LinearFit")
    y2 = [float(v) for v in y]
    pred = list(getattr(fit, "predict")(x))
    _validate_lengths(len(y2), pred, cluster)
    resid = [obs - p for obs, p in zip(y2, pred)]

    cov = ols_cluster_covariance(
        x,
        resid,
        cluster,
        include_intercept=bool(getattr(fit, "include_intercept")),
        df_correction=df_correction,
    )
    return cov, cov_to_se(cov)


def logistic_sandwich_covariance(
    x: Sequence[Sequence[float]],
    y: Sequence[int],
    mu: Sequence[float],
    *,
    include_intercept: bool,
    cluster: Optional[Sequence[Any]] = None,
    df_correction: bool = False,
) -> Matrix:
    """Baseline sandwich covariance for Bernoulli-logit GLM.

    Uses:
    - score_i = x_i * (y_i - mu_i)
    - info ~= X' W X with W_i = mu_i * (1 - mu_i)

    This is a pragmatic baseline. For penalized/MAP fits, this does not account
    for the prior.
    """
    xd = _design_matrix(x, include_intercept=include_intercept)
    n = len(xd)
    if n == 0:
        raise ValueError("x must have at least 1 row")
    _validate_lengths(n, y, mu)

    k = len(xd[0]) if xd else 0
    if any(len(row) != k for row in xd):
        raise ValueError("x must be rectangular")
    if k == 0:
        raise ValueError("x must have at least 1 column (add intercept or features)")

    w = [float(p) * (1.0 - float(p)) for p in mu]
    inv_h = mat_inv(_weighted_xtx(xd, w))

    if cluster is None:
        meat = _weighted_xtx(xd, [(float(yi) - float(pi)) ** 2 for yi, pi in zip(y, mu)])
        if df_correction and n > k:
            _mat_scale_inplace(meat, float(n) / float(n - k))
    else:
        _validate_lengths(n, cluster)
        by_g: dict[Any, Vector] = {}
        for xi, yi, pi, gi in zip(xd, y, mu, cluster):
            r = float(yi) - float(pi)
            s = by_g.get(gi)
            if s is None:
                by_g[gi] = [float(v) * r for v in xi]
            else:
                for j in range(k):
                    s[j] += float(xi[j]) * r

        g = len(by_g)
        if g < 2:
            raise ValueError("cluster must have at least 2 distinct groups")

        meat = [[0.0] * k for _ in range(k)]
        for s in by_g.values():
            _mat_add_inplace(meat, _outer(s))

        if df_correction and n > k:
            scale = (float(g) / float(g - 1)) * ((float(n) - 1.0) / float(n - k))
            _mat_scale_inplace(meat, scale)

    return mat_mul(mat_mul(inv_h, meat), inv_h)


def poisson_sandwich_covariance(
    x: Sequence[Sequence[float]],
    y: Sequence[int],
    mu: Sequence[float],
    *,
    include_intercept: bool,
    cluster: Optional[Sequence[Any]] = None,
    df_correction: bool = False,
) -> Matrix:
    """Baseline sandwich covariance for Poisson (log link) GLM.

    Uses:
    - score_i = x_i * (y_i - mu_i)
    - info ~= X' W X with W_i = mu_i
    """
    xd = _design_matrix(x, include_intercept=include_intercept)
    n = len(xd)
    if n == 0:
        raise ValueError("x must have at least 1 row")
    _validate_lengths(n, y, mu)

    k = len(xd[0]) if xd else 0
    if any(len(row) != k for row in xd):
        raise ValueError("x must be rectangular")
    if k == 0:
        raise ValueError("x must have at least 1 column (add intercept or features)")

    inv_h = mat_inv(_weighted_xtx(xd, [float(m) for m in mu]))

    if cluster is None:
        meat = _weighted_xtx(xd, [(float(yi) - float(mi)) ** 2 for yi, mi in zip(y, mu)])
        if df_correction and n > k:
            _mat_scale_inplace(meat, float(n) / float(n - k))
    else:
        _validate_lengths(n, cluster)
        by_g: dict[Any, Vector] = {}
        for xi, yi, mi, gi in zip(xd, y, mu, cluster):
            r = float(yi) - float(mi)
            s = by_g.get(gi)
            if s is None:
                by_g[gi] = [float(v) * r for v in xi]
            else:
                for j in range(k):
                    s[j] += float(xi[j]) * r

        g = len(by_g)
        if g < 2:
            raise ValueError("cluster must have at least 2 distinct groups")

        meat = [[0.0] * k for _ in range(k)]
        for s in by_g.values():
            _mat_add_inplace(meat, _outer(s))

        if df_correction and n > k:
            scale = (float(g) / float(g - 1)) * ((float(n) - 1.0) / float(n - k))
            _mat_scale_inplace(meat, scale)

    return mat_mul(mat_mul(inv_h, meat), inv_h)


def logistic_sandwich_from_fit(
    fit: Any,
    x: Sequence[Sequence[float]],
    y: Sequence[int],
    *,
    cluster: Optional[Sequence[Any]] = None,
    df_correction: bool = False,
) -> Tuple[Matrix, List[float]]:
    """Convenience wrapper for nextstat.glm.logistic.LogisticFit."""
    if not hasattr(fit, "predict_proba") or not hasattr(fit, "include_intercept"):
        raise TypeError("fit must look like nextstat.glm.logistic.LogisticFit")
    mu = list(getattr(fit, "predict_proba")(x))
    cov = logistic_sandwich_covariance(
        x,
        [int(v) for v in y],
        mu,
        include_intercept=bool(getattr(fit, "include_intercept")),
        cluster=cluster,
        df_correction=df_correction,
    )
    return cov, cov_to_se(cov)


def poisson_sandwich_from_fit(
    fit: Any,
    x: Sequence[Sequence[float]],
    y: Sequence[int],
    *,
    cluster: Optional[Sequence[Any]] = None,
    df_correction: bool = False,
) -> Tuple[Matrix, List[float]]:
    """Convenience wrapper for nextstat.glm.poisson.PoissonFit."""
    if not hasattr(fit, "predict_mean") or not hasattr(fit, "include_intercept"):
        raise TypeError("fit must look like nextstat.glm.poisson.PoissonFit")
    mu = list(getattr(fit, "predict_mean")(x))
    cov = poisson_sandwich_covariance(
        x,
        [int(v) for v in y],
        mu,
        include_intercept=bool(getattr(fit, "include_intercept")),
        cluster=cluster,
        df_correction=df_correction,
    )
    return cov, cov_to_se(cov)


__all__ = [
    "HcKind",
    "cov_to_se",
    "ols_hc_covariance",
    "ols_cluster_covariance",
    "ols_hc_from_fit",
    "ols_cluster_from_fit",
    "logistic_sandwich_covariance",
    "poisson_sandwich_covariance",
    "logistic_sandwich_from_fit",
    "poisson_sandwich_from_fit",
]
