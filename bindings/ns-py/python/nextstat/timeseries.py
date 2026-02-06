"""Time series helpers (Phase 8).

Currently provides a baseline linear-Gaussian Kalman filter + RTS smoother.

The heavy lifting lives in Rust (`ns-inference`). This Python layer keeps a
stable, user-facing surface.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def kalman_filter(model, ys: Sequence[Sequence[float]]) -> Mapping[str, Any]:
    """Run Kalman filtering.

    Parameters
    - model: `nextstat._core.KalmanModel`
    - ys: list of observation vectors (shape: T x n_obs)

    Returns
    - dict with keys:
      - log_likelihood
      - predicted_means, predicted_covs
      - filtered_means, filtered_covs
    """
    from . import _core

    return _core.kalman_filter(model, [list(y) for y in ys])


def kalman_smooth(model, ys: Sequence[Sequence[float]]) -> Mapping[str, Any]:
    """Run Kalman filtering + RTS smoothing.

    Returns
    - dict with keys:
      - log_likelihood
      - filtered_means, filtered_covs
      - smoothed_means, smoothed_covs
    """
    from . import _core

    return _core.kalman_smooth(model, [list(y) for y in ys])


__all__ = [
    "kalman_filter",
    "kalman_smooth",
]

