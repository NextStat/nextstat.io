"""Linear regression (Gaussian).

This is a high-level Python surface that wraps:
- `_core.ols_fit` for coefficients
- classic OLS standard errors via sigma^2 * (X'X)^-1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from ._linalg import add_intercept, as_2d_float_list, mat_inv, mat_mul, mat_t, mat_vec_mul, solve_linear


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
    l2: Optional[float] = None,
    penalize_intercept: bool = False,
) -> LinearFit:
    import nextstat

    x2 = as_2d_float_list(x)
    y2 = [float(v) for v in y]
    if not x2 or not x2[0]:
        raise ValueError("X must be non-empty")
    if len(x2) != len(y2):
        raise ValueError("X and y must have the same length")

    xd = add_intercept(x2) if include_intercept else x2
    n = len(y2)
    k = (len(xd[0]) if xd else 0)
    if n <= k:
        raise ValueError("Need n > n_params to compute sigma2_hat")

    if l2 is None or float(l2) <= 0.0:
        coef = nextstat._core.ols_fit(x2, y2, include_intercept=include_intercept)
    else:
        lam = float(l2)
        xt = mat_t(xd)
        xtx = mat_mul(xt, xd)
        xty = mat_vec_mul(xt, y2)
        for i in range(k):
            if include_intercept and not penalize_intercept and i == 0:
                continue
            xtx[i][i] += lam
        coef = solve_linear(xtx, xty)

    resid = [pred - obs for pred, obs in zip(mat_vec_mul(xd, coef), y2)]
    sse = sum(r * r for r in resid)
    sigma2_hat = sse / float(n - k)

    xt = mat_t(xd)
    xtx = mat_mul(xt, xd)
    if l2 is not None and float(l2) > 0.0:
        lam = float(l2)
        for i in range(k):
            if include_intercept and not penalize_intercept and i == 0:
                continue
            xtx[i][i] += lam
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

def from_formula(
    formula: str,
    data: Any,
    *,
    categorical: Optional[Sequence[str]] = None,
    l2: Optional[float] = None,
    penalize_intercept: bool = False,
) -> Tuple[LinearFit, List[str]]:
    """Fit linear regression from a minimal formula and tabular data.

    Returns `(fit, column_names)` where `column_names` matches the coefficient order.
    """
    import nextstat

    y_name, terms, include_intercept = nextstat.formula.parse_formula(formula)
    cols = nextstat.formula.to_columnar(data, [y_name] + terms)
    y, x_full, names_full = nextstat.formula.design_matrices(formula, cols, categorical=categorical)

    # Map design-matrix output to the `fit(...)` convention (X excludes intercept column).
    if names_full and names_full[0] == "Intercept":
        x = [row[1:] for row in x_full]
        feature_names = list(names_full[1:])
        include_intercept = True
    else:
        x = x_full
        feature_names = list(names_full)
        include_intercept = False

    # Intercept-only model: compute closed-form fit directly.
    if include_intercept and not feature_names:
        n = len(y)
        if n < 2:
            raise ValueError("Need at least 2 observations for intercept-only fit")
        mu = sum(y) / float(n)
        resid = [v - mu for v in y]
        sse = sum(r * r for r in resid)
        sigma2_hat = sse / float(n - 1)
        se = [(sigma2_hat / float(n)) ** 0.5]
        cov = [[sigma2_hat / float(n)]]
        return (
            LinearFit(
                coef=[mu],
                standard_errors=se,
                covariance=cov,
                sigma2_hat=sigma2_hat,
                include_intercept=True,
            ),
            ["Intercept"],
        )

    r = fit(x, y, include_intercept=include_intercept, l2=l2, penalize_intercept=penalize_intercept)
    colnames = (["Intercept"] if include_intercept else []) + feature_names
    return r, colnames


__all__ = ["LinearFit", "fit", "from_formula"]
