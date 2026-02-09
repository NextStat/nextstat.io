"""Robust covariance estimators (Phase 11.4).

Supports:
- OLS HC0–HC3
- 1-way cluster-robust (sandwich) baseline
- GLM sandwich (logit/poisson/negbin) via score approximation

Design goals:
- No numpy dependency (small problems, deterministic).
- Explicit limitations (Wald / sandwich asymptotics).
"""

from __future__ import annotations

import math
from typing import Any, List, Literal, Optional, Sequence

from ._linalg import add_intercept, as_2d_float_list, mat_inv, mat_mul, mat_t, mat_vec_mul


Matrix = List[List[float]]
Vector = List[float]

HcKind = Literal["hc0", "hc1", "hc2", "hc3"]


def _outer(x: Vector) -> Matrix:
    return [[xi * xj for xj in x] for xi in x]


def _mat_add(a: Matrix, b: Matrix) -> Matrix:
    return [[ai + bi for ai, bi in zip(ar, br)] for ar, br in zip(a, b)]


def _mat_scale(a: Matrix, s: float) -> Matrix:
    return [[s * v for v in row] for row in a]


def _diag_from_vec(v: Vector) -> Matrix:
    n = len(v)
    return [[(v[i] if i == j else 0.0) for j in range(n)] for i in range(n)]


def _check_xy(x: Sequence[Sequence[float]], y: Sequence[float]) -> tuple[Matrix, Vector]:
    x2 = as_2d_float_list(x)
    y2 = [float(v) for v in y]
    if not x2 or not x2[0]:
        raise ValueError("X must be non-empty")
    if len(x2) != len(y2):
        raise ValueError("X and y must have the same length")
    return x2, y2


def _encode_clusters(cluster: Sequence[Any]) -> list[int]:
    labels = [("None" if v is None else str(v)) for v in cluster]
    levels = sorted(set(labels))
    idx = {lvl: i for i, lvl in enumerate(levels)}
    return [idx[v] for v in labels]


def ols_hc_covariance(
    x: Sequence[Sequence[float]],
    y: Sequence[float],
    coef: Sequence[float],
    *,
    include_intercept: bool = True,
    kind: HcKind = "hc1",
) -> Matrix:
    """OLS heteroskedasticity-consistent covariance (HC0–HC3)."""
    x2, y2 = _check_xy(x, y)
    xd = add_intercept(x2) if include_intercept else x2
    b = [float(v) for v in coef]
    if len(b) != len(xd[0]):
        raise ValueError("coef has wrong length for X design")

    n = len(y2)
    k = len(b)
    if n <= k:
        raise ValueError("Need n > n_params")

    yhat = mat_vec_mul(xd, b)
    resid = [obs - pred for obs, pred in zip(y2, yhat)]

    xt = mat_t(xd)
    xtx = mat_mul(xt, xd)
    bread = mat_inv(xtx)

    # leverage h_ii = x_i^T (X'X)^-1 x_i
    h: list[float] = []
    for row in xd:
        # tmp = bread * x_i
        tmp = [sum(bread[r][c] * row[c] for c in range(k)) for r in range(k)]
        h.append(sum(row[c] * tmp[c] for c in range(k)))

    w: list[float] = []
    for i, e in enumerate(resid):
        if kind == "hc0" or kind == "hc1":
            w.append(e * e)
        elif kind == "hc2":
            den = max(1e-12, 1.0 - h[i])
            w.append((e * e) / den)
        elif kind == "hc3":
            den = max(1e-12, 1.0 - h[i])
            w.append((e * e) / (den * den))
        else:
            raise ValueError("kind must be one of: hc0,hc1,hc2,hc3")

    # meat = X' diag(w) X
    meat = mat_mul(mat_mul(xt, _diag_from_vec(w)), xd)
    cov = mat_mul(mat_mul(bread, meat), bread)

    if kind == "hc1":
        cov = _mat_scale(cov, float(n) / float(n - k))

    return cov


def ols_cluster_covariance(
    x: Sequence[Sequence[float]],
    y: Sequence[float],
    coef: Sequence[float],
    cluster: Sequence[Any],
    *,
    include_intercept: bool = True,
) -> Matrix:
    """OLS 1-way cluster-robust covariance (sandwich)."""
    x2, y2 = _check_xy(x, y)
    if len(cluster) != len(y2):
        raise ValueError("cluster must have length n")

    xd = add_intercept(x2) if include_intercept else x2
    b = [float(v) for v in coef]
    if len(b) != len(xd[0]):
        raise ValueError("coef has wrong length for X design")

    xt = mat_t(xd)
    xtx = mat_mul(xt, xd)
    bread = mat_inv(xtx)

    yhat = mat_vec_mul(xd, b)
    resid = [obs - pred for obs, pred in zip(y2, yhat)]

    cl = _encode_clusters(cluster)
    g_max = (max(cl) + 1) if cl else 0
    k = len(b)
    sums: list[Vector] = [[0.0] * k for _ in range(g_max)]
    for i, g in enumerate(cl):
        for j in range(k):
            sums[g][j] += xd[i][j] * resid[i]

    meat: Matrix = [[0.0] * k for _ in range(k)]
    for s in sums:
        meat = _mat_add(meat, _outer(s))

    return mat_mul(mat_mul(bread, meat), bread)


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


def _glm_sandwich(
    xd: Matrix,
    y: Vector,
    mu: Vector,
    w: Vector,
    *,
    cluster: Optional[Sequence[Any]] = None,
    hc1: bool = True,
) -> Matrix:
    n = len(y)
    k = len(xd[0]) if xd else 0
    if n == 0 or k == 0:
        raise ValueError("empty design")

    xt = mat_t(xd)
    xtwx = mat_mul(mat_mul(xt, _diag_from_vec(w)), xd)
    bread = mat_inv(xtwx)

    if cluster is None:
        # meat = sum u_i u_i^T, u_i = x_i * (y_i - mu_i)
        meat: Matrix = [[0.0] * k for _ in range(k)]
        for i in range(n):
            ui = [xd[i][j] * (y[i] - mu[i]) for j in range(k)]
            meat = _mat_add(meat, _outer(ui))
        cov = mat_mul(mat_mul(bread, meat), bread)
        if hc1:
            cov = _mat_scale(cov, float(n) / float(max(1, n - k)))
        return cov

    if len(cluster) != n:
        raise ValueError("cluster must have length n")

    cl = _encode_clusters(cluster)
    g_max = (max(cl) + 1) if cl else 0
    sums: list[Vector] = [[0.0] * k for _ in range(g_max)]
    for i, g in enumerate(cl):
        for j in range(k):
            sums[g][j] += xd[i][j] * (y[i] - mu[i])

    meat2: Matrix = [[0.0] * k for _ in range(k)]
    for s in sums:
        meat2 = _mat_add(meat2, _outer(s))
    return mat_mul(mat_mul(bread, meat2), bread)


def logistic_sandwich_covariance(
    x: Sequence[Sequence[float]],
    y: Sequence[int],
    coef: Sequence[float],
    *,
    include_intercept: bool = True,
    cluster: Optional[Sequence[Any]] = None,
    hc1: bool = True,
) -> Matrix:
    x2 = as_2d_float_list(x)
    y2 = [int(v) for v in y]
    if len(x2) != len(y2):
        raise ValueError("X and y must have the same length")
    xd = add_intercept(x2) if include_intercept else x2
    b = [float(v) for v in coef]
    if len(b) != len(xd[0]):
        raise ValueError("coef has wrong length for X design")

    eta = mat_vec_mul(xd, b)
    mu = [_sigmoid(v) for v in eta]
    w = [max(1e-12, m * (1.0 - m)) for m in mu]
    yy = [float(v) for v in y2]
    return _glm_sandwich(xd, yy, mu, w, cluster=cluster, hc1=hc1)


def poisson_sandwich_covariance(
    x: Sequence[Sequence[float]],
    y: Sequence[int],
    coef: Sequence[float],
    *,
    include_intercept: bool = True,
    offset: Optional[Sequence[float]] = None,
    exposure: Optional[Sequence[float]] = None,
    cluster: Optional[Sequence[Any]] = None,
    hc1: bool = True,
) -> Matrix:
    if offset is not None and exposure is not None:
        raise ValueError("Specify only one of offset= or exposure=")

    x2 = as_2d_float_list(x)
    y2 = [int(v) for v in y]
    if len(x2) != len(y2):
        raise ValueError("X and y must have the same length")
    xd = add_intercept(x2) if include_intercept else x2
    b = [float(v) for v in coef]
    if len(b) != len(xd[0]):
        raise ValueError("coef has wrong length for X design")

    off: Optional[List[float]] = None
    if exposure is not None:
        off = []
        for v in exposure:
            ev = float(v)
            if not (ev > 0.0) or not math.isfinite(ev):
                raise ValueError("exposure must be finite and > 0")
            off.append(math.log(ev))
    elif offset is not None:
        off = [float(v) for v in offset]

    if off is not None and len(off) != len(y2):
        raise ValueError("offset/exposure must have length n")

    eta = mat_vec_mul(xd, b)
    if off is not None:
        eta = [e + o for e, o in zip(eta, off)]
    mu = [math.exp(v) for v in eta]
    w = [max(1e-12, m) for m in mu]
    yy = [float(v) for v in y2]
    return _glm_sandwich(xd, yy, mu, w, cluster=cluster, hc1=hc1)


def negbin_sandwich_covariance(
    x: Sequence[Sequence[float]],
    y: Sequence[int],
    coef: Sequence[float],
    *,
    alpha: float,
    include_intercept: bool = True,
    offset: Optional[Sequence[float]] = None,
    exposure: Optional[Sequence[float]] = None,
    cluster: Optional[Sequence[Any]] = None,
    hc1: bool = True,
) -> Matrix:
    """NB2 sandwich covariance via quasi-score approximation.

    Uses:
    - mu = exp(eta + offset)
    - Var = mu + alpha * mu^2
    - W = (dmu/deta)^2 / Var = mu / (1 + alpha*mu)
    - score contribution ~ x_i * (y_i - mu_i) / (1 + alpha*mu_i)
    """
    if not (float(alpha) > 0.0) or not math.isfinite(float(alpha)):
        raise ValueError("alpha must be finite and > 0")

    if offset is not None and exposure is not None:
        raise ValueError("Specify only one of offset= or exposure=")

    x2 = as_2d_float_list(x)
    y2 = [int(v) for v in y]
    if len(x2) != len(y2):
        raise ValueError("X and y must have the same length")
    xd = add_intercept(x2) if include_intercept else x2
    b = [float(v) for v in coef]
    if len(b) != len(xd[0]):
        raise ValueError("coef has wrong length for X design")

    off: Optional[List[float]] = None
    if exposure is not None:
        off = []
        for v in exposure:
            ev = float(v)
            if not (ev > 0.0) or not math.isfinite(ev):
                raise ValueError("exposure must be finite and > 0")
            off.append(math.log(ev))
    elif offset is not None:
        off = [float(v) for v in offset]

    if off is not None and len(off) != len(y2):
        raise ValueError("offset/exposure must have length n")

    eta = mat_vec_mul(xd, b)
    if off is not None:
        eta = [e + o for e, o in zip(eta, off)]
    mu = [math.exp(v) for v in eta]

    a = float(alpha)
    w = [max(1e-12, m / (1.0 + a * m)) for m in mu]

    # Adjust residuals in the score for quasi-score form.
    yy = [float(v) for v in y2]
    mu_adj = mu[:]  # still use mu in (y-mu)
    if cluster is None:
        n = len(yy)
        k = len(xd[0])
        xt = mat_t(xd)
        xtwx = mat_mul(mat_mul(xt, _diag_from_vec(w)), xd)
        bread = mat_inv(xtwx)

        meat: Matrix = [[0.0] * k for _ in range(k)]
        for i in range(n):
            den = 1.0 + a * mu[i]
            ui = [xd[i][j] * ((yy[i] - mu_adj[i]) / den) for j in range(k)]
            meat = _mat_add(meat, _outer(ui))
        cov = mat_mul(mat_mul(bread, meat), bread)
        if hc1:
            cov = _mat_scale(cov, float(n) / float(max(1, n - k)))
        return cov

    if len(cluster) != len(yy):
        raise ValueError("cluster must have length n")
    cl = _encode_clusters(cluster)
    g_max = (max(cl) + 1) if cl else 0
    k = len(xd[0])
    sums: list[Vector] = [[0.0] * k for _ in range(g_max)]
    for i, g in enumerate(cl):
        den = 1.0 + a * mu[i]
        for j in range(k):
            sums[g][j] += xd[i][j] * ((yy[i] - mu_adj[i]) / den)

    xt = mat_t(xd)
    xtwx = mat_mul(mat_mul(xt, _diag_from_vec(w)), xd)
    bread = mat_inv(xtwx)
    meat2: Matrix = [[0.0] * k for _ in range(k)]
    for s in sums:
        meat2 = _mat_add(meat2, _outer(s))
    return mat_mul(mat_mul(bread, meat2), bread)


def standard_errors(cov: Matrix) -> Vector:
    """Return sqrt(diag(cov))."""
    out: Vector = []
    for i, row in enumerate(cov):
        v = float(row[i])
        out.append(math.sqrt(v) if v > 0.0 else float("inf"))
    return out


__all__ = [
    "ols_hc_covariance",
    "ols_cluster_covariance",
    "logistic_sandwich_covariance",
    "poisson_sandwich_covariance",
    "negbin_sandwich_covariance",
    "standard_errors",
]

