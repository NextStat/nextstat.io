"""Linear regression (Gaussian).

This is a high-level Python surface that wraps:
- `_core.ols_fit` for coefficients
- classic OLS standard errors via sigma^2 * (X'X)^-1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from ._linalg import add_intercept, as_2d_float_list, mat_inv, mat_mul, mat_t, mat_vec_mul


@dataclass(frozen=True)
class LinearFit:
    coef: List[float]
    standard_errors: List[float]
    covariance: List[List[float]]
    sigma2_hat: float
    include_intercept: bool

    def predict(self, x: Sequence[Sequence[float]]) -> List[float]:
        x2 = as_2d_float_list(x)
        xd = add_intercept(x2) if self.include_intercept else x2
        return mat_vec_mul(xd, self.coef)


def fit(
    x: Sequence[Sequence[float]],
    y: Sequence[float],
    *,
    include_intercept: bool = True,
) -> LinearFit:
    import nextstat

    x2 = as_2d_float_list(x)
    y2 = [float(v) for v in y]
    if not x2 or not x2[0]:
        raise ValueError("X must be non-empty")
    if len(x2) != len(y2):
        raise ValueError("X and y must have the same length")

    coef = nextstat._core.ols_fit(x2, y2, include_intercept=include_intercept)

    xd = add_intercept(x2) if include_intercept else x2
    n = len(y2)
    k = len(coef)
    if n <= k:
        raise ValueError("Need n > n_params to compute sigma2_hat")

    resid = [pred - obs for pred, obs in zip(mat_vec_mul(xd, coef), y2)]
    sse = sum(r * r for r in resid)
    sigma2_hat = sse / float(n - k)

    xt = mat_t(xd)
    xtx = mat_mul(xt, xd)
    xtx_inv = mat_inv(xtx)
    cov = [[sigma2_hat * v for v in row] for row in xtx_inv]
    se = [(cov[i][i] ** 0.5) if cov[i][i] > 0.0 else float("inf") for i in range(k)]

    return LinearFit(
        coef=list(coef),
        standard_errors=se,
        covariance=cov,
        sigma2_hat=sigma2_hat,
        include_intercept=include_intercept,
    )


__all__ = ["LinearFit", "fit"]

