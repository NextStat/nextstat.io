"""Tiny linear algebra helpers for small problems (no numpy required).

These routines are intentionally minimal and intended for small p (<= ~50).
"""

from __future__ import annotations

from typing import List, Sequence


Vector = List[float]
Matrix = List[List[float]]


def mat_t(a: Matrix) -> Matrix:
    return [list(col) for col in zip(*a)]


def mat_mul(a: Matrix, b: Matrix) -> Matrix:
    bt = mat_t(b)
    return [[sum(ai * bj for ai, bj in zip(row, col)) for col in bt] for row in a]


def mat_vec_mul(a: Matrix, x: Vector) -> Vector:
    return [sum(ai * xi for ai, xi in zip(row, x)) for row in a]


def solve_linear(a: Matrix, b: Vector) -> Vector:
    """Solve a x = b by Gaussian elimination with partial pivoting."""

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
    return [[1.0] + [float(v) for v in row] for row in x]


def as_2d_float_list(x: Sequence[Sequence[float]]) -> Matrix:
    return [[float(v) for v in row] for row in x]

