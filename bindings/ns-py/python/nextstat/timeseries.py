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
    estimate_f: bool = False,
    estimate_h: bool = False,
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
        estimate_f=estimate_f,
        estimate_h=estimate_h,
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

def local_level_model(*, q: float, r: float, m0: float = 0.0, p0: float = 1.0):
    """Construct a 1D local level (random walk) Kalman model."""
    from . import _core

    return _core.KalmanModel([[1.0]], [[float(q)]], [[1.0]], [[float(r)]], [float(m0)], [[float(p0)]])


def local_linear_trend_model(
    *,
    q_level: float,
    q_slope: float,
    r: float,
    level0: float = 0.0,
    slope0: float = 0.0,
    p0_level: float = 1.0,
    p0_slope: float = 1.0,
):
    """Construct a 2D local linear trend (level+slope) Kalman model."""
    from . import _core

    f = [[1.0, 1.0], [0.0, 1.0]]
    q = [[float(q_level), 0.0], [0.0, float(q_slope)]]
    h = [[1.0, 0.0]]
    rr = [[float(r)]]
    m0 = [float(level0), float(slope0)]
    p0 = [[float(p0_level), 0.0], [0.0, float(p0_slope)]]
    return _core.KalmanModel(f, q, h, rr, m0, p0)

def ar1_model(*, phi: float, q: float, r: float, m0: float = 0.0, p0: float = 1.0):
    """Construct a 1D AR(1) Kalman model."""
    from . import _core

    return _core.KalmanModel([[float(phi)]], [[float(q)]], [[1.0]], [[float(r)]], [float(m0)], [[float(p0)]])

def local_level_seasonal_model(
    *,
    period: int,
    q_level: float,
    q_season: float,
    r: float,
    level0: float = 0.0,
    p0_level: float = 1.0,
    p0_season: float = 1.0,
):
    """Construct a local level + seasonal (dummy seasonal) Kalman model."""
    from . import _core

    if int(period) < 2:
        raise ValueError("period must be >= 2")

    sdim = int(period) - 1
    dim = 1 + sdim

    # F: level random walk + seasonal dummy transition
    f = [[0.0 for _ in range(dim)] for _ in range(dim)]
    f[0][0] = 1.0
    for j in range(sdim):
        f[1][1 + j] = -1.0
    for i in range(1, sdim):
        f[1 + i][1 + (i - 1)] = 1.0

    q = [[0.0 for _ in range(dim)] for _ in range(dim)]
    q[0][0] = float(q_level)
    for j in range(sdim):
        q[1 + j][1 + j] = float(q_season)

    h = [[0.0 for _ in range(dim)]]
    h[0][0] = 1.0
    h[0][1] = 1.0

    rr = [[float(r)]]
    m0 = [float(level0)] + [0.0 for _ in range(sdim)]

    p0 = [[0.0 for _ in range(dim)] for _ in range(dim)]
    p0[0][0] = float(p0_level)
    for j in range(sdim):
        p0[1 + j][1 + j] = float(p0_season)

    return _core.KalmanModel(f, q, h, rr, m0, p0)

def local_linear_trend_seasonal_model(
    *,
    period: int,
    q_level: float,
    q_slope: float,
    q_season: float,
    r: float,
    level0: float = 0.0,
    slope0: float = 0.0,
    p0_level: float = 1.0,
    p0_slope: float = 1.0,
    p0_season: float = 1.0,
):
    """Construct a local linear trend + seasonal (dummy seasonal) Kalman model."""
    from . import _core

    if int(period) < 2:
        raise ValueError("period must be >= 2")

    sdim = int(period) - 1
    dim = 2 + sdim

    f = [[0.0 for _ in range(dim)] for _ in range(dim)]
    f[0][0] = 1.0
    f[0][1] = 1.0
    f[1][1] = 1.0
    for j in range(sdim):
        f[2][2 + j] = -1.0
    for i in range(1, sdim):
        f[2 + i][2 + (i - 1)] = 1.0

    q = [[0.0 for _ in range(dim)] for _ in range(dim)]
    q[0][0] = float(q_level)
    q[1][1] = float(q_slope)
    for j in range(sdim):
        q[2 + j][2 + j] = float(q_season)

    h = [[0.0 for _ in range(dim)]]
    h[0][0] = 1.0
    h[0][2] = 1.0

    rr = [[float(r)]]
    m0 = [float(level0), float(slope0)] + [0.0 for _ in range(sdim)]

    p0 = [[0.0 for _ in range(dim)] for _ in range(dim)]
    p0[0][0] = float(p0_level)
    p0[1][1] = float(p0_slope)
    for j in range(sdim):
        p0[2 + j][2 + j] = float(p0_season)

    return _core.KalmanModel(f, q, h, rr, m0, p0)


__all__ = [
    "kalman_filter",
    "kalman_smooth",
    "kalman_em",
    "kalman_forecast",
    "kalman_simulate",
    "local_level_model",
    "local_linear_trend_model",
    "ar1_model",
    "local_level_seasonal_model",
    "local_linear_trend_seasonal_model",
]
