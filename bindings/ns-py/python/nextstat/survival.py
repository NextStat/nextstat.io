"""High-level survival surface (Phase 9 Pack A).

Baseline scope:
- intercept-only parametric models
- right-censoring via (t, event)
- Cox proportional hazards (partial likelihood)
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple


def _tolist(x: Any) -> Any:
    tolist = getattr(x, "tolist", None)
    if callable(tolist):
        return tolist()
    return x


def _as_1d_float_list(x: Any, *, name: str) -> List[float]:
    x = _tolist(x)
    if not isinstance(x, Sequence) or isinstance(x, (bytes, str)):
        raise TypeError(f"{name} must be a 1D sequence (or numpy array).")
    return [float(v) for v in x]


def _as_1d_bool_list(x: Any, *, name: str) -> List[bool]:
    x = _tolist(x)
    if not isinstance(x, Sequence) or isinstance(x, (bytes, str)):
        raise TypeError(f"{name} must be a 1D sequence (or numpy array).")
    return [bool(v) for v in x]

def _as_2d_float_list(x: Any, *, name: str) -> List[List[float]]:
    x = _tolist(x)
    if not isinstance(x, Sequence) or isinstance(x, (bytes, str)):
        raise TypeError(f"{name} must be a 2D sequence (or numpy array).")
    out: List[List[float]] = []
    for i, row in enumerate(x):
        if not isinstance(row, Sequence) or isinstance(row, (bytes, str)):
            raise TypeError(f"{name}[{i}] must be a 1D sequence.")
        out.append([float(v) for v in row])
    return out


def exponential(times: Any, events: Any):
    """Build an exponential survival model (right-censoring)."""
    import nextstat

    return nextstat.ExponentialSurvivalModel(
        _as_1d_float_list(times, name="times"),
        _as_1d_bool_list(events, name="events"),
    )


def weibull(times: Any, events: Any):
    """Build a Weibull survival model (right-censoring)."""
    import nextstat

    return nextstat.WeibullSurvivalModel(
        _as_1d_float_list(times, name="times"),
        _as_1d_bool_list(events, name="events"),
    )


def lognormal_aft(times: Any, events: Any):
    """Build a log-normal AFT survival model (right-censoring)."""
    import nextstat

    return nextstat.LogNormalAftModel(
        _as_1d_float_list(times, name="times"),
        _as_1d_bool_list(events, name="events"),
    )

def cox_ph(times: Any, events: Any, x: Any, *, ties: str = "efron"):
    """Build a Cox proportional hazards model (partial likelihood)."""
    import nextstat

    return nextstat.CoxPhModel(
        _as_1d_float_list(times, name="times"),
        _as_1d_bool_list(events, name="events"),
        _as_2d_float_list(x, name="x"),
        ties=ties,
    )


__all__ = [
    "exponential",
    "weibull",
    "lognormal_aft",
    "cox_ph",
]
