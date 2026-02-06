"""Logistic regression (Bernoulli-logit).

High-level surface:
- builds `_core.LogisticRegressionModel`
- fits via `nextstat.fit(...)` (Rust MLE)

Robustness notes:
- Perfect separation can make the unregularized MLE ill-defined (coefficients diverge).
  Use `l2=...` (ridge/MAP) to stabilize.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, List, Optional, Sequence, Tuple

from ._linalg import add_intercept, as_2d_float_list, mat_vec_mul


def _sigmoid(x: float) -> float:
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
    warnings: List[str]

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
    l2: Optional[float] = None,
    penalize_intercept: bool = False,
) -> LogisticFit:
    import nextstat

    x2 = as_2d_float_list(x)
    y2: List[int] = []
    for v in y:
        iv = int(v)
        if iv not in (0, 1):
            raise ValueError("y must contain only 0/1 for logistic regression")
        y2.append(iv)

    warnings: List[str] = []

    if l2 is None or float(l2) <= 0.0:
        model = nextstat._core.LogisticRegressionModel(x2, y2, include_intercept=include_intercept)
        r = nextstat.fit(model)
    else:
        lam = float(l2)
        sigma = 1.0 / math.sqrt(lam)
        model = nextstat._core.ComposedGlmModel.logistic_regression(
            x2,
            y2,
            include_intercept=include_intercept,
            coef_prior_mu=0.0,
            coef_prior_sigma=sigma,
            penalize_intercept=penalize_intercept,
        )
        r = nextstat.fit(model)

    if not bool(r.converged):
        warnings.append("not_converged")
        if l2 is None or float(l2) <= 0.0:
            warnings.append("possible_separation_use_l2")
    else:
        if l2 is None or float(l2) <= 0.0:
            # Heuristic: separation often manifests as very large coefficients and/or
            # extremely ill-conditioned curvature (huge SE). Thresholds are intentionally
            # conservative to avoid spurious warnings on well-scaled problems.
            max_abs_coef = max(abs(float(v)) for v in r.parameters) if r.parameters else 0.0
            max_se = max(abs(float(v)) for v in r.uncertainties) if r.uncertainties else 0.0
            if (max_abs_coef > 15.0) or (max_se > 1e3):
                warnings.append("large_coefficients_possible_separation")

    return LogisticFit(
        coef=list(r.parameters),
        standard_errors=list(r.uncertainties),
        nll=float(r.nll),
        converged=bool(r.converged),
        include_intercept=include_intercept,
        warnings=warnings,
    )

def from_formula(
    formula: str,
    data: Any,
    *,
    categorical: Optional[Sequence[str]] = None,
    l2: Optional[float] = None,
    penalize_intercept: bool = False,
) -> Tuple[LogisticFit, List[str]]:
    """Fit logistic regression from a minimal formula and tabular data.

    Returns `(fit, column_names)` where `column_names` matches the coefficient order.
    """
    import nextstat

    y_name, terms, _include_intercept = nextstat.formula.parse_formula(formula)
    cols = nextstat.formula.to_columnar(data, [y_name] + terms)
    y_float, x_full, names_full = nextstat.formula.design_matrices(formula, cols, categorical=categorical)

    y = [int(v) for v in y_float]
    if names_full and names_full[0] == "Intercept":
        x = [row[1:] for row in x_full]
        feature_names = list(names_full[1:])
        include_intercept = True
    else:
        x = x_full
        feature_names = list(names_full)
        include_intercept = False

    r = fit(x, y, include_intercept=include_intercept, l2=l2, penalize_intercept=penalize_intercept)
    colnames = (["Intercept"] if include_intercept else []) + feature_names
    return r, colnames


__all__ = ["LogisticFit", "fit", "from_formula"]
