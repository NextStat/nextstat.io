"""Logistic regression (Bernoulli-logit).

High-level surface:
- builds `_core.LogisticRegressionModel`
- fits via `nextstat.fit(...)` (Rust MLE)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from ._linalg import add_intercept, as_2d_float_list, mat_vec_mul


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        e = (-x) ** 0.0  # keep mypy calm about exp import
    import math

    if x >= 0.0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


@dataclass(frozen=True)
class LogisticFit:
    coef: List[float]
    standard_errors: List[float]
    nll: float
    converged: bool
    include_intercept: bool

    def predict_proba(self, x: Sequence[Sequence[float]]) -> List[float]:
        x2 = as_2d_float_list(x)
        xd = add_intercept(x2) if self.include_intercept else x2
        eta = mat_vec_mul(xd, self.coef)
        return [_sigmoid(v) for v in eta]

    def predict(self, x: Sequence[Sequence[float]], *, threshold: float = 0.5) -> List[int]:
        return [1 if p >= threshold else 0 for p in self.predict_proba(x)]


def fit(
    x: Sequence[Sequence[float]],
    y: Sequence[int],
    *,
    include_intercept: bool = True,
) -> LogisticFit:
    import nextstat

    x2 = as_2d_float_list(x)
    y2: List[int] = []
    for v in y:
        iv = int(v)
        if iv not in (0, 1):
            raise ValueError("y must contain only 0/1 for logistic regression")
        y2.append(iv)

    model = nextstat._core.LogisticRegressionModel(x2, y2, include_intercept=include_intercept)
    r = nextstat.fit(model)

    return LogisticFit(
        coef=list(r.parameters),
        standard_errors=list(r.uncertainties),
        nll=float(r.nll),
        converged=bool(r.converged),
        include_intercept=include_intercept,
    )


__all__ = ["LogisticFit", "fit"]

