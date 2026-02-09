"""Regression/GLM metrics (Phase 6.4.2).

These helpers are intentionally dependency-free (no numpy required).
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence


def _as_floats(x: Iterable[float]) -> List[float]:
    return [float(v) for v in x]


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    yt = _as_floats(y_true)
    yp = _as_floats(y_pred)
    if len(yt) != len(yp):
        raise ValueError("y_true and y_pred must have the same length")
    if not yt:
        raise ValueError("need at least 1 sample")
    s = 0.0
    for a, b in zip(yt, yp):
        d = b - a
        s += d * d
    return math.sqrt(s / float(len(yt)))


def log_loss(y_true: Sequence[int], y_prob: Sequence[float], *, eps: float = 1e-15) -> float:
    """Binary log-loss (cross-entropy)."""
    yt = [int(v) for v in y_true]
    yp = _as_floats(y_prob)
    if len(yt) != len(yp):
        raise ValueError("y_true and y_prob must have the same length")
    if not yt:
        raise ValueError("need at least 1 sample")
    if not (0.0 < float(eps) < 0.5):
        raise ValueError("eps must be in (0, 0.5)")

    loss = 0.0
    for y, p in zip(yt, yp):
        if y not in (0, 1):
            raise ValueError("y_true must contain only 0/1")
        p = min(max(p, eps), 1.0 - eps)
        loss -= y * math.log(p) + (1 - y) * math.log(1.0 - p)
    return loss / float(len(yt))


def poisson_deviance(y_true: Sequence[int], mu: Sequence[float]) -> float:
    """Poisson deviance: 2 * sum(y*log(y/mu) - (y - mu))."""
    yt = [int(v) for v in y_true]
    mu2 = _as_floats(mu)
    if len(yt) != len(mu2):
        raise ValueError("y_true and mu must have the same length")
    if not yt:
        raise ValueError("need at least 1 sample")
    dev = 0.0
    for y, m in zip(yt, mu2):
        if y < 0:
            raise ValueError("y_true must be non-negative for Poisson deviance")
        if not (m > 0.0) or not math.isfinite(m):
            raise ValueError("mu must be finite and > 0")
        if y == 0:
            dev += 2.0 * (0.0 - (0.0 - m))
        else:
            dev += 2.0 * (y * math.log(float(y) / m) - (y - m))
    return dev


def mean_poisson_deviance(y_true: Sequence[int], mu: Sequence[float]) -> float:
    yt = [int(v) for v in y_true]
    if not yt:
        raise ValueError("need at least 1 sample")
    return poisson_deviance(yt, mu) / float(len(yt))


__all__ = [
    "rmse",
    "log_loss",
    "poisson_deviance",
    "mean_poisson_deviance",
]

