"""Linear algebra helpers — numpy-accelerated with pure-Python fallback.

The public interface uses List[List[float]] / List[float] for backward
compatibility with callers that expect plain Python lists.  Internally
everything runs through numpy (BLAS) when available.
"""

from __future__ import annotations

from typing import List, Sequence

try:
    import numpy as _np
    _HAS_NP = True
except ImportError:
    _HAS_NP = False

Vector = List[float]
Matrix = List[List[float]]


def mat_t(a: Matrix) -> Matrix:
    if _HAS_NP:
        return _np.asarray(a, dtype=_np.float64).T.tolist()
    return [list(col) for col in zip(*a)]


def mat_mul(a: Matrix, b: Matrix) -> Matrix:
    if _HAS_NP:
        return (_np.asarray(a, dtype=_np.float64) @ _np.asarray(b, dtype=_np.float64)).tolist()
    bt = mat_t(b)
    return [[sum(ai * bj for ai, bj in zip(row, col)) for col in bt] for row in a]


def mat_vec_mul(a: Matrix, x: Vector) -> Vector:
    if _HAS_NP:
        return (_np.asarray(a, dtype=_np.float64) @ _np.asarray(x, dtype=_np.float64)).tolist()
    return [sum(ai * xi for ai, xi in zip(row, x)) for row in a]


def solve_linear(a: Matrix, b: Vector) -> Vector:
    """Solve a x = b."""
    if _HAS_NP:
        return _np.linalg.solve(
            _np.asarray(a, dtype=_np.float64),
            _np.asarray(b, dtype=_np.float64),
        ).tolist()

    n = len(a)
    if n == 0:
        return []
    if any(len(row) != n for row in a):
        raise ValueError("a must be square")
    if len(b) != n:
        raise ValueError("b has wrong length")

    m = [row[:] + [bi] for row, bi in zip(a, b)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-15:
            raise ValueError("singular matrix")
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]

        piv = m[col][col]
        for r in range(col + 1, n):
            f = m[r][col] / piv
            if f == 0.0:
                continue
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / m[i][i]
    return x


def mat_inv(a: Matrix) -> Matrix:
    if _HAS_NP:
        return _np.linalg.inv(_np.asarray(a, dtype=_np.float64)).tolist()

    n = len(a)
    if n == 0:
        return []
    if any(len(row) != n for row in a):
        raise ValueError("a must be square")

    inv: Matrix = []
    for i in range(n):
        e = [0.0] * n
        e[i] = 1.0
        col = solve_linear(a, e)
        inv.append(col)
    return mat_t(inv)  # columns -> rows


def add_intercept(x: Sequence[Sequence[float]]) -> Matrix:
    if _HAS_NP:
        arr = _np.asarray(x, dtype=_np.float64)
        return _np.column_stack([_np.ones(arr.shape[0]), arr]).tolist()
    return [[1.0] + [float(v) for v in row] for row in x]


def as_2d_float_list(x: Sequence[Sequence[float]]) -> Matrix:
    if _HAS_NP:
        return _np.asarray(x, dtype=_np.float64).tolist()
    return [[float(v) for v in row] for row in x]
