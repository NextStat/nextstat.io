"""Propensity score estimation and diagnostics (Phase 9.C).

This module provides:
- propensity score estimation via `nextstat.glm.logistic.fit`
- inverse-probability weights (IPW)
- overlap + balance diagnostics (standardized mean differences)

Important: these tools do **not** create causality. They only help make common
applied workflows easier to reproduce and review.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Literal, Optional, Sequence, Tuple

from nextstat.glm import logistic as ns_logistic
from nextstat.glm._linalg import as_2d_float_list


Estimand = Literal["ate", "att", "atc"]


def _as_binary_list(t: Sequence[int]) -> List[int]:
    out: List[int] = []
    for v in t:
        iv = int(v)
        if iv not in (0, 1):
            raise ValueError("treatment indicator must contain only 0/1 values")
        out.append(iv)
    return out


def _clip_prob(p: float, eps: float) -> float:
    # Avoid 0/1 probabilities that create infinite weights.
    if not (0.0 < eps < 0.5):
        raise ValueError("eps must satisfy 0 < eps < 0.5")
    return float(min(max(float(p), eps), 1.0 - eps))


def _weighted_mean(xs: Sequence[float], ws: Sequence[float]) -> float:
    s = 0.0
    wsum = 0.0
    for x, w in zip(xs, ws):
        wf = float(w)
        s += float(x) * wf
        wsum += wf
    if not (wsum > 0.0) or not math.isfinite(wsum):
        raise ValueError("invalid weights: sum(weights) must be finite and > 0")
    return float(s / wsum)


def _weighted_var_pop(xs: Sequence[float], ws: Sequence[float], mu: float) -> float:
    # Population variance (no Bessel correction). This is sufficient for balance diagnostics.
    s = 0.0
    wsum = 0.0
    for x, w in zip(xs, ws):
        wf = float(w)
        dx = float(x) - float(mu)
        s += wf * dx * dx
        wsum += wf
    if not (wsum > 0.0) or not math.isfinite(wsum):
        raise ValueError("invalid weights: sum(weights) must be finite and > 0")
    return float(s / wsum)


def _ess(ws: Sequence[float]) -> float:
    # Effective sample size under weights.
    s1 = 0.0
    s2 = 0.0
    for w in ws:
        wf = float(w)
        s1 += wf
        s2 += wf * wf
    if not (s2 > 0.0) or not (math.isfinite(s1) and math.isfinite(s2)):
        return float("nan")
    return float((s1 * s1) / s2)


def standardized_mean_differences(
    x: Sequence[Sequence[float]],
    t: Sequence[int],
    *,
    weights: Optional[Sequence[float]] = None,
) -> List[float]:
    """Compute standardized mean differences (SMD) per covariate.

    For each column j:
      SMD_j = (mean_treated - mean_control) / sqrt(0.5*(var_treated + var_control))

    If `weights` is provided, uses weighted means/variances within each group.
    """

    x2 = as_2d_float_list(x)
    t2 = _as_binary_list(t)
    if len(x2) != len(t2):
        raise ValueError("x/t length mismatch")
    if not x2:
        return []

    p = len(x2[0])
    if any(len(row) != p for row in x2):
        raise ValueError("x must be a rectangular matrix")

    if weights is None:
        w2 = [1.0] * len(t2)
    else:
        if len(weights) != len(t2):
            raise ValueError("weights length mismatch")
        w2 = [float(w) for w in weights]
        if any((not math.isfinite(w)) or (w < 0.0) for w in w2):
            raise ValueError("weights must be finite and >= 0")

    smd: List[float] = []
    for j in range(p):
        xj = [float(row[j]) for row in x2]

        xt = [x for x, tj in zip(xj, t2) if tj == 1]
        xc = [x for x, tj in zip(xj, t2) if tj == 0]
        wt = [w for w, tj in zip(w2, t2) if tj == 1]
        wc = [w for w, tj in zip(w2, t2) if tj == 0]

        if not xt or not xc:
            raise ValueError("both treatment groups must be non-empty")

        mt = _weighted_mean(xt, wt)
        mc = _weighted_mean(xc, wc)
        vt = _weighted_var_pop(xt, wt, mt)
        vc = _weighted_var_pop(xc, wc, mc)

        denom = math.sqrt(0.5 * (vt + vc))
        if denom <= 0.0 or not math.isfinite(denom):
            smd.append(0.0)
        else:
            smd.append(float((mt - mc) / denom))
    return smd


def ipw_weights(
    t: Sequence[int],
    p: Sequence[float],
    *,
    estimand: Estimand = "ate",
    stabilized: bool = True,
    max_weight: Optional[float] = None,
) -> List[float]:
    """Inverse-probability weights for common estimands.

    - ATE: w_i = 1/p for treated, 1/(1-p) for control (optionally stabilized)
    - ATT: w_i = 1 for treated, p/(1-p) for control (not stabilized)
    - ATC: w_i = (1-p)/p for treated, 1 for control (not stabilized)
    """

    t2 = _as_binary_list(t)
    if len(p) != len(t2):
        raise ValueError("p/t length mismatch")
    if not t2:
        return []

    p2 = [float(v) for v in p]
    if any((not math.isfinite(v)) or (v <= 0.0) or (v >= 1.0) for v in p2):
        raise ValueError("p must be finite and strictly between 0 and 1")

    pt = float(sum(t2) / len(t2))
    pc = 1.0 - pt
    if not (0.0 < pt < 1.0):
        raise ValueError("t must contain both 0 and 1 values")

    out: List[float] = []
    for ti, pi in zip(t2, p2):
        if estimand == "ate":
            if stabilized:
                w = (pt / pi) if ti == 1 else (pc / (1.0 - pi))
            else:
                w = (1.0 / pi) if ti == 1 else (1.0 / (1.0 - pi))
        elif estimand == "att":
            if stabilized:
                raise ValueError("stabilized weights are only supported for estimand='ate'")
            w = 1.0 if ti == 1 else (pi / (1.0 - pi))
        elif estimand == "atc":
            if stabilized:
                raise ValueError("stabilized weights are only supported for estimand='ate'")
            w = ((1.0 - pi) / pi) if ti == 1 else 1.0
        else:
            raise ValueError("unknown estimand")

        if max_weight is not None:
            mw = float(max_weight)
            if not (mw > 0.0) or not math.isfinite(mw):
                raise ValueError("max_weight must be finite and > 0")
            w = min(w, mw)

        out.append(float(w))
    return out


@dataclass(frozen=True)
class PropensityScoreFit:
    model: ns_logistic.LogisticFit
    propensity_scores: List[float]
    treatment_rate: float
    clip_eps: float

    def predict_proba(self, x: Sequence[Sequence[float]]) -> List[float]:
        ps = self.model.predict_proba(x)
        return [_clip_prob(v, self.clip_eps) for v in ps]


@dataclass(frozen=True)
class PropensityDiagnostics:
    n: int
    p: int
    treatment_rate: float
    propensity_min: float
    propensity_max: float
    propensity_min_treated: float
    propensity_max_treated: float
    propensity_min_control: float
    propensity_max_control: float
    smd_unweighted: List[float]
    smd_weighted: Optional[List[float]]
    ess_treated: Optional[float]
    ess_control: Optional[float]
    warnings: List[str]

    @property
    def max_abs_smd_unweighted(self) -> float:
        return float(max(abs(v) for v in self.smd_unweighted)) if self.smd_unweighted else 0.0

    @property
    def max_abs_smd_weighted(self) -> Optional[float]:
        if self.smd_weighted is None:
            return None
        return float(max(abs(v) for v in self.smd_weighted)) if self.smd_weighted else 0.0


def diagnostics(
    x: Sequence[Sequence[float]],
    t: Sequence[int],
    propensity_scores: Sequence[float],
    *,
    weights: Optional[Sequence[float]] = None,
    overlap_eps: float = 0.01,
) -> PropensityDiagnostics:
    x2 = as_2d_float_list(x)
    t2 = _as_binary_list(t)
    if len(x2) != len(t2) or len(propensity_scores) != len(t2):
        raise ValueError("x/t/p length mismatch")
    if not x2:
        raise ValueError("x must be non-empty")
    p = len(x2[0])
    if any(len(row) != p for row in x2):
        raise ValueError("x must be a rectangular matrix")

    ps = [float(v) for v in propensity_scores]
    if any((not math.isfinite(v)) or (v <= 0.0) or (v >= 1.0) for v in ps):
        raise ValueError("propensity_scores must be finite and strictly between 0 and 1")

    wt = [float(w) for w, ti in zip(weights or [1.0] * len(t2), t2) if ti == 1]
    wc = [float(w) for w, ti in zip(weights or [1.0] * len(t2), t2) if ti == 0]
    if weights is not None and (len(wt) + len(wc) != len(t2)):
        raise ValueError("weights length mismatch")
    if weights is not None and any((not math.isfinite(w)) or (w < 0.0) for w in (wt + wc)):
        raise ValueError("weights must be finite and >= 0")

    treat_rate = float(sum(t2) / len(t2))

    p_all_min = float(min(ps))
    p_all_max = float(max(ps))

    p_t = [p for p, ti in zip(ps, t2) if ti == 1]
    p_c = [p for p, ti in zip(ps, t2) if ti == 0]
    if not p_t or not p_c:
        raise ValueError("both treatment groups must be non-empty")

    warnings: List[str] = []
    if p_all_min < overlap_eps or p_all_max > 1.0 - overlap_eps:
        warnings.append("overlap_risk_extreme_propensity")

    smd0 = standardized_mean_differences(x2, t2, weights=None)
    smd1 = standardized_mean_differences(x2, t2, weights=weights) if weights is not None else None

    ess_t = _ess(wt) if weights is not None else None
    ess_c = _ess(wc) if weights is not None else None
    if weights is not None:
        if (ess_t is not None and ess_t < 0.5 * len(p_t)) or (ess_c is not None and ess_c < 0.5 * len(p_c)):
            warnings.append("low_effective_sample_size")

    return PropensityDiagnostics(
        n=len(t2),
        p=p,
        treatment_rate=treat_rate,
        propensity_min=p_all_min,
        propensity_max=p_all_max,
        propensity_min_treated=float(min(p_t)),
        propensity_max_treated=float(max(p_t)),
        propensity_min_control=float(min(p_c)),
        propensity_max_control=float(max(p_c)),
        smd_unweighted=smd0,
        smd_weighted=smd1,
        ess_treated=ess_t,
        ess_control=ess_c,
        warnings=warnings,
    )


def fit(
    x: Sequence[Sequence[float]],
    t: Sequence[int],
    *,
    include_intercept: bool = True,
    l2: float = 1.0,
    penalize_intercept: bool = False,
    clip_eps: float = 1e-6,
) -> PropensityScoreFit:
    """Fit a propensity score model and return clipped propensity scores."""

    x2 = as_2d_float_list(x)
    t2 = _as_binary_list(t)
    if len(x2) != len(t2):
        raise ValueError("x/t length mismatch")
    if not x2:
        raise ValueError("x must be non-empty")

    # Default to a small ridge penalty: it improves stability on practical datasets
    # (rare treatments, separability) and is rarely harmful for propensity modeling.
    lam = float(l2)
    if not (lam > 0.0) or not math.isfinite(lam):
        raise ValueError("l2 must be finite and > 0")

    model = ns_logistic.fit(
        x2,
        t2,
        include_intercept=include_intercept,
        l2=lam,
        penalize_intercept=penalize_intercept,
    )
    ps = [_clip_prob(v, clip_eps) for v in model.predict_proba(x2)]
    tr = float(sum(t2) / len(t2))
    return PropensityScoreFit(
        model=model,
        propensity_scores=ps,
        treatment_rate=tr,
        clip_eps=float(clip_eps),
    )


__all__ = [
    "Estimand",
    "PropensityDiagnostics",
    "PropensityScoreFit",
    "diagnostics",
    "fit",
    "ipw_weights",
    "standardized_mean_differences",
]

