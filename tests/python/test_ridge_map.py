from __future__ import annotations

import math


def _mat_t(a):
    return [list(col) for col in zip(*a)]


def _mat_mul(a, b):
    bt = _mat_t(b)
    return [[sum(ai * bj for ai, bj in zip(row, col)) for col in bt] for row in a]


def _mat_vec_mul(a, x):
    return [sum(ai * xi for ai, xi in zip(row, x)) for row in a]


def _solve_linear(a, b):
    n = len(a)
    m = [row[:] + [bi] for row, bi in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-15:
            raise ValueError("singular")
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
        piv = m[col][col]
        for r in range(col + 1, n):
            f = m[r][col] / piv
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / m[i][i]
    return x


def test_ridge_linear_matches_closed_form():
    import nextstat

    x = [[-1.0, 0.2], [-0.5, -0.1], [0.3, 0.4], [1.2, -0.7], [2.0, 0.1], [0.7, 1.1]]
    y = [0.1, -0.2, 0.6, 0.9, 1.2, 0.8]
    lam = 2.5

    r = nextstat.glm.linear.fit(x, y, include_intercept=True, l2=lam, penalize_intercept=False)

    xd = [[1.0] + row[:] for row in x]
    xt = _mat_t(xd)
    xtx = _mat_mul(xt, xd)
    xty = _mat_vec_mul(xt, y)
    k = len(xty)
    for i in range(k):
        if i == 0:
            continue  # no intercept penalty
        xtx[i][i] += lam
    beta = _solve_linear(xtx, xty)

    assert len(r.coef) == len(beta)
    for ai, bi in zip(r.coef, beta):
        assert abs(float(ai) - float(bi)) < 1e-10


def test_ridge_logistic_shrinks_coefficients():
    import nextstat

    # Strongly separable toy dataset.
    x = [[-3.0], [-2.0], [-1.5], [-1.0], [1.0], [1.5], [2.0], [3.0]]
    y = [0, 0, 0, 0, 1, 1, 1, 1]

    unreg = nextstat.glm.logistic.fit(x, y, include_intercept=True)
    reg = nextstat.glm.logistic.fit(x, y, include_intercept=True, l2=10.0, penalize_intercept=False)

    un = max(abs(v) for v in unreg.coef)
    rr = max(abs(v) for v in reg.coef)
    assert math.isfinite(rr)
    assert rr < un


def test_ridge_poisson_smoke():
    import nextstat

    x = [[-1.0], [0.0], [1.0], [2.0], [3.0]]
    y = [1, 1, 2, 4, 7]

    r = nextstat.glm.poisson.fit(x, y, include_intercept=True, l2=5.0)
    assert r.converged
    rate = r.predict_mean(x)
    assert all(v > 0.0 and math.isfinite(v) for v in rate)

