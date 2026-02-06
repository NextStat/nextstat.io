"""Time series helpers (Phase 8).

Currently provides a baseline linear-Gaussian Kalman filter + RTS smoother.

The heavy lifting lives in Rust (`ns-inference`). This Python layer keeps a
stable, user-facing surface.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def kalman_filter(model, ys: Sequence[Sequence[float | None]]) -> Mapping[str, Any]:
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


def kalman_smooth(model, ys: Sequence[Sequence[float | None]]) -> Mapping[str, Any]:
    """Run Kalman filtering + RTS smoothing.

    Returns
    - dict with keys:
      - log_likelihood
      - filtered_means, filtered_covs
      - smoothed_means, smoothed_covs
    """
    from . import _core

    return _core.kalman_smooth(model, [list(y) for y in ys])


def kalman_em(
    model,
    ys: Sequence[Sequence[float | None]],
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    estimate_q: bool = True,
    estimate_r: bool = True,
    min_diag: float = 1e-12,
) -> Mapping[str, Any]:
    """Fit Q/R with EM while keeping F/H/m0/P0 fixed."""
    from . import _core

    return _core.kalman_em(
        model,
        [list(y) for y in ys],
        max_iter=max_iter,
        tol=tol,
        estimate_q=estimate_q,
        estimate_r=estimate_r,
        min_diag=min_diag,
    )


def kalman_forecast(model, ys: Sequence[Sequence[float | None]], *, steps: int = 1) -> Mapping[str, Any]:
    """Forecast future states/observations after ingesting `ys`."""
    from . import _core

    return _core.kalman_forecast(model, [list(y) for y in ys], steps=steps)


def kalman_simulate(model, *, t_max: int, seed: int = 42) -> Mapping[str, Any]:
    """Simulate (xs, ys) from the model."""
    from . import _core

    return _core.kalman_simulate(model, t_max=t_max, seed=seed)


__all__ = [
    "kalman_filter",
    "kalman_smooth",
    "kalman_em",
    "kalman_forecast",
    "kalman_simulate",
]
