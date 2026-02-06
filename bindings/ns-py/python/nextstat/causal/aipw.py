"""Doubly-robust AIPW estimators (Phase 12).

This module provides minimal **augmented inverse probability weighting (AIPW)**
estimators for binary treatments:
- ATE (average treatment effect)
- ATT (average treatment effect on the treated)

Nuisance models (baseline):
- propensity score e(x) via logistic regression
- outcome regressions mu0(x), mu1(x) via linear regression fit separately per arm

Sensitivity analysis
- `e_value_rr(...)` is implemented for risk-ratio scale effects.
- Rosenbaum-style sensitivity is provided as a placeholder (API only).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, List, Literal, Optional, Sequence

from nextstat.causal import propensity as ps
from nextstat.glm import linear as ns_linear
from nextstat.glm._linalg import as_2d_float_list


Estimand = Literal["ate", "att"]


def _as_binary_list(t: Sequence[int]) -> List[int]:
    out: List[int] = []
    for v in t:
        iv = int(v)
        if iv not in (0, 1):
            raise ValueError("treatment indicator must contain only 0/1 values")
        out.append(iv)
    return out


def _clip_prob(p: float, eps: float) -> float:
    if not (0.0 < eps < 0.5):
        raise ValueError("trim_eps must satisfy 0 < trim_eps < 0.5")
    return float(min(max(float(p), eps), 1.0 - eps))


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    return float(sum(float(v) for v in xs) / float(len(xs)))


def _sample_var(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mu = _mean(xs)
    s = 0.0
    for v in xs:
        dv = float(v) - mu
        s += dv * dv
    return float(s / float(n - 1))


@dataclass(frozen=True)
class AipwFit:
    estimand: Estimand
    estimate: float
    standard_error: float
    influence: List[float]
    n_obs: int
    trim_eps: float
    propensity_scores: List[float]
    mu0: List[float]
    mu1: Optional[List[float]]
    propensity_diagnostics: ps.PropensityDiagnostics


def _fit_outcome_mu(
    x: List[List[float]],
    y: List[float],
    t: List[int],
    *,
    l2: Optional[float] = None,
    penalize_intercept: bool = False,
) -> tuple[List[float], List[float]]:
    idx0 = [i for i, ti in enumerate(t) if ti == 0]
    idx1 = [i for i, ti in enumerate(t) if ti == 1]
    if not idx0 or not idx1:
        raise ValueError("both treatment groups must be non-empty")

    x0 = [x[i] for i in idx0]
    y0 = [y[i] for i in idx0]
    x1 = [x[i] for i in idx1]
    y1 = [y[i] for i in idx1]

    m0 = ns_linear.fit(x0, y0, include_intercept=True, l2=l2, penalize_intercept=penalize_intercept)
    m1 = ns_linear.fit(x1, y1, include_intercept=True, l2=l2, penalize_intercept=penalize_intercept)
    mu0 = list(m0.predict(x))
    mu1 = list(m1.predict(x))
    return mu0, mu1


def aipw_fit(
    x: Sequence[Sequence[float]],
    y: Sequence[float],
    t: Sequence[int],
    *,
    estimand: Estimand = "ate",
    trim_eps: float = 1e-6,
    propensity_scores: Optional[Sequence[float]] = None,
    mu0: Optional[Sequence[float]] = None,
    mu1: Optional[Sequence[float]] = None,
    propensity_l2: float = 1.0,
    propensity_penalize_intercept: bool = False,
    outcome_l2: Optional[float] = None,
    outcome_penalize_intercept: bool = False,
) -> AipwFit:
    """Fit AIPW for ATE/ATT.

    Hooks:
    - pass `propensity_scores` to bypass fitting the propensity model
    - pass `mu0`/`mu1` to bypass fitting outcome regressions
    """
    x2 = as_2d_float_list(x)
    y2 = [float(v) for v in y]
    t2 = _as_binary_list(t)
    n = len(y2)
    if len(x2) != n or len(t2) != n:
        raise ValueError("x/y/t length mismatch")
    if n == 0:
        raise ValueError("need at least 1 observation")

    est = str(estimand).lower()
    if est not in ("ate", "att"):
        raise ValueError("estimand must be one of: ate, att")

    # Propensity scores.
    if propensity_scores is None:
        pfit = ps.fit(
            x2,
            t2,
            include_intercept=True,
            l2=float(propensity_l2),
            penalize_intercept=bool(propensity_penalize_intercept),
            clip_eps=float(trim_eps),
        )
        e = list(pfit.propensity_scores)
    else:
        if len(propensity_scores) != n:
            raise ValueError("propensity_scores length mismatch")
        e = [_clip_prob(float(v), float(trim_eps)) for v in propensity_scores]

    # Outcome regression.
    if mu0 is None or mu1 is None:
        mu0_f, mu1_f = _fit_outcome_mu(
            x2,
            y2,
            t2,
            l2=outcome_l2,
            penalize_intercept=bool(outcome_penalize_intercept),
        )
        mu0_used = mu0_f if mu0 is None else [float(v) for v in mu0]
        mu1_used = mu1_f if mu1 is None else [float(v) for v in mu1]
    else:
        if len(mu0) != n or len(mu1) != n:
            raise ValueError("mu0/mu1 length mismatch")
        mu0_used = [float(v) for v in mu0]
        mu1_used = [float(v) for v in mu1]

    # Overlap + balance diagnostics.
    if est == "ate":
        w = ps.ipw_weights(t2, e, estimand="ate", stabilized=True)
    else:
        w = ps.ipw_weights(t2, e, estimand="att", stabilized=False)
    diag = ps.diagnostics(x2, t2, e, weights=w)

    if est == "ate":
        psi: List[float] = []
        for yi, ti, ei, m0, m1 in zip(y2, t2, e, mu0_used, mu1_used):
            if ti == 1:
                psi.append(float(m1) + (float(yi) - float(m1)) / float(ei) - float(m0))
            else:
                psi.append(float(m1) - float(m0) - (float(yi) - float(m0)) / float(1.0 - float(ei)))
        theta = _mean(psi)
    else:
        p_hat = float(sum(t2) / len(t2))
        if not (p_hat > 0.0):
            raise ValueError("need at least one treated observation for ATT")
        psi = []
        for yi, ti, ei, m0 in zip(y2, t2, e, mu0_used):
            if ti == 1:
                psi.append((float(yi) - float(m0)) / p_hat)
            else:
                psi.append(-(float(ei) / float(1.0 - float(ei))) * (float(yi) - float(m0)) / p_hat)
        theta = _mean(psi)

    infl = [float(v) - float(theta) for v in psi]
    se = math.sqrt(_sample_var(infl) / float(n)) if n > 1 else 0.0

    return AipwFit(
        estimand=est,  # type: ignore[arg-type]
        estimate=float(theta),
        standard_error=float(se),
        influence=infl,
        n_obs=int(n),
        trim_eps=float(trim_eps),
        propensity_scores=list(e),
        mu0=list(mu0_used),
        mu1=(list(mu1_used) if mu1_used is not None else None),
        propensity_diagnostics=diag,
    )


def e_value_rr(rr: float) -> float:
    """E-value for a risk ratio (RR) effect estimate."""
    r = float(rr)
    if not math.isfinite(r) or r <= 0.0:
        raise ValueError("rr must be finite and > 0")
    if r < 1.0:
        r = 1.0 / r
    if r == 1.0:
        return 1.0
    return float(r + math.sqrt(r * (r - 1.0)))


@dataclass(frozen=True)
class RosenbaumSensitivityPlaceholder:
    gamma_grid: List[float]
    message: str


def rosenbaum_sensitivity_placeholder(*, gamma_grid: Optional[Sequence[float]] = None) -> RosenbaumSensitivityPlaceholder:
    """Placeholder for Rosenbaum-style sensitivity analysis."""
    gg = [1.0, 1.25, 1.5, 2.0] if gamma_grid is None else [float(v) for v in gamma_grid]
    if any((not math.isfinite(v)) or (v < 1.0) for v in gg):
        raise ValueError("gamma_grid must be finite and >= 1")
    return RosenbaumSensitivityPlaceholder(
        gamma_grid=gg,
        message="Rosenbaum sensitivity bounds are design-specific; this is a placeholder hook.",
    )


__all__ = [
    "AipwFit",
    "Estimand",
    "RosenbaumSensitivityPlaceholder",
    "aipw_fit",
    "e_value_rr",
    "rosenbaum_sensitivity_placeholder",
]

