"""Ordered logistic regression (proportional odds) (Phase 9.C).

This module is intentionally minimal and dependency-light.

Model notes:
- Outcome levels must be encoded as integers in {0, 1, ..., K-1}.
- Do not include an explicit intercept term: thresholds/cutpoints play that role.
  (If you include a constant column in X, the model becomes non-identifiable.)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Sequence

from nextstat.glm._linalg import as_2d_float_list, mat_vec_mul


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


def _softplus(x: float) -> float:
    # Stable log(1 + exp(x))
    if x > 0.0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def _cutpoints_from_raw(raw: Sequence[float]) -> List[float]:
    if not raw:
        return []
    c: List[float] = [float(raw[0])]
    for r in raw[1:]:
        c.append(float(c[-1] + _softplus(float(r))))
    return c


@dataclass(frozen=True)
class OrderedLogitFit:
    coef: List[float]
    cut_raw: List[float]
    cutpoints: List[float]
    n_levels: int
    nll: float
    converged: bool

    def predict_proba(self, x: Sequence[Sequence[float]]) -> List[List[float]]:
        x2 = as_2d_float_list(x)
        if int(self.n_levels) < 2:
            raise ValueError("n_levels must be >= 2")
        if len(self.cutpoints) != int(self.n_levels) - 1:
            raise ValueError("cutpoints length mismatch")

        eta = mat_vec_mul(x2, self.coef)
        out: List[List[float]] = []
        for e in eta:
            # P(y <= k) = sigmoid(c_{k+1} - eta)
            cdf = [_sigmoid(self.cutpoints[0] - float(e))]
            for ck in self.cutpoints[1:]:
                cdf.append(_sigmoid(float(ck) - float(e)))

            # Convert CDF -> PMF.
            probs: List[float] = []
            probs.append(float(cdf[0]))
            for j in range(1, len(cdf)):
                probs.append(float(cdf[j] - cdf[j - 1]))
            probs.append(float(1.0 - cdf[-1]))

            # Guard against tiny negative values from float subtraction.
            probs = [0.0 if p < 0.0 and p > -1e-12 else float(p) for p in probs]
            s = sum(probs)
            if not (math.isfinite(s) and s > 0.0):
                raise ValueError("predict_proba produced invalid probabilities")
            probs = [p / s for p in probs]
            out.append(probs)
        return out

    def predict(self, x: Sequence[Sequence[float]]) -> List[int]:
        ps = self.predict_proba(x)
        out: List[int] = []
        for row in ps:
            k = max(range(len(row)), key=lambda i: float(row[i]))
            out.append(int(k))
        return out


def fit(
    x: Sequence[Sequence[float]],
    y: Sequence[int],
    *,
    n_levels: int,
) -> OrderedLogitFit:
    import nextstat

    x2 = as_2d_float_list(x)
    k = int(n_levels)
    if k < 2:
        raise ValueError("n_levels must be >= 2")

    y2: List[int] = []
    for v in y:
        iv = int(v)
        if not (0 <= iv < k):
            raise ValueError(f"y must be in [0, n_levels), got {iv}")
        y2.append(iv)

    model = nextstat._core.OrderedLogitModel(x2, y2, n_levels=k)
    r = nextstat.fit(model)

    p = list(r.parameters)
    beta = p[: len(x2[0])]
    cut_raw = p[len(beta) :]
    cutpoints = _cutpoints_from_raw(cut_raw)

    return OrderedLogitFit(
        coef=[float(v) for v in beta],
        cut_raw=[float(v) for v in cut_raw],
        cutpoints=cutpoints,
        n_levels=k,
        nll=float(r.nll),
        converged=bool(r.converged),
    )


__all__ = ["OrderedLogitFit", "fit"]

