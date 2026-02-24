"""ODE helpers (Phase 13 baseline).

This module provides a deterministic baseline integrator (RK4) for simple systems.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence


def rk4_linear(
    a: Sequence[Sequence[float]],
    y0: Sequence[float],
    t0: float,
    t1: float,
    dt: float,
    *,
    max_steps: int = 100_000,
) -> Dict[str, Any]:
    """Integrate `dy/dt = A y` using fixed-step RK4.

    Returns a dict: {"t": [...], "y": [[...], ...]}.
    """
    from . import _core

    a_list: List[List[float]] = [list(map(float, row)) for row in a]
    y0_list: List[float] = [float(v) for v in y0]
    return _core.rk4_linear(a_list, y0_list, float(t0), float(t1), float(dt), max_steps=int(max_steps))

def rk4_linear_dde(
    a: Sequence[Sequence[float]],
    b: Sequence[Sequence[float]],
    y0: Sequence[float],
    y_history: Sequence[float],
    t0: float,
    t1: float,
    tau: float,
    dt: float,
    *,
    max_steps: int = 100_000,
) -> Dict[str, Any]:
    """Integrate linear DDE `dy/dt = A y(t) + B y(t-tau)` with fixed delay.

    Uses method-of-steps + RK4 and constant pre-history `y(t<=t0)=y_history`.
    Returns a dict: {"t": [...], "y": [[...], ...]}.
    """
    from . import _core

    a_list: List[List[float]] = [list(map(float, row)) for row in a]
    b_list: List[List[float]] = [list(map(float, row)) for row in b]
    y0_list: List[float] = [float(v) for v in y0]
    yh_list: List[float] = [float(v) for v in y_history]
    return _core.rk4_linear_dde(
        a_list,
        b_list,
        y0_list,
        yh_list,
        float(t0),
        float(t1),
        float(tau),
        float(dt),
        max_steps=int(max_steps),
    )


def lsoda(
    a: Sequence[Sequence[float]] | Callable[[float, Sequence[float]], Sequence[float]],
    y0: Sequence[float],
    t0: float,
    t1: float,
    *,
    jac: Callable[[float, Sequence[float]], Sequence[Sequence[float]]] | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    h0: float = 0.0,
    h_min: float = 1e-14,
    h_max: float = float("inf"),
    max_steps: int = 100_000,
    dense_output: bool = True,
) -> Dict[str, Any]:
    """Integrate with LSODA-style stiff auto-switching.

    Supports:
    - Linear matrix form: `dy/dt = A y` with `a` as a 2D matrix.
    - Callback form: `a` is `rhs(t, y) -> dy_dt`, optional `jac(t, y)`.
    """
    from . import _core

    y0_list: List[float] = [float(v) for v in y0]
    if callable(a):
        return _core.lsoda_callback(
            a,
            y0_list,
            float(t0),
            float(t1),
            jac=jac,
            rtol=float(rtol),
            atol=float(atol),
            h0=float(h0),
            h_min=float(h_min),
            h_max=float(h_max),
            max_steps=int(max_steps),
            dense_output=bool(dense_output),
        )
    if jac is not None:
        raise ValueError("jac is only supported when lsoda() is used with a callback rhs")
    a_list: List[List[float]] = [list(map(float, row)) for row in a]
    return _core.lsoda(
        a_list,
        y0_list,
        float(t0),
        float(t1),
        rtol=float(rtol),
        atol=float(atol),
        h0=float(h0),
        h_min=float(h_min),
        h_max=float(h_max),
        max_steps=int(max_steps),
        dense_output=bool(dense_output),
    )


def forward_sensitivity_solve(
    a_params: Sequence[Sequence[Sequence[float]]]
    | Callable[[float, Sequence[float], Sequence[float]], Sequence[float]],
    params: Sequence[float],
    y0: Sequence[float],
    t0: float,
    t1: float,
    *,
    jac_y: Callable[[float, Sequence[float], Sequence[float]], Sequence[Sequence[float]]] | None = None,
    jac_params: Callable[[float, Sequence[float], Sequence[float]], Sequence[Sequence[float]]] | None = None,
    solver: str = "lsoda",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    h0: float = 0.0,
    h_min: float = 1e-14,
    h_max: float = float("inf"),
    max_steps: int = 100_000,
    dense_output: bool = True,
) -> Dict[str, Any]:
    """Forward sensitivities for parametric ODEs.

    Supports:
    - Linear matrix form: `dy/dt = (sum_p params[p] * A_p) y`
    - Callback form: `rhs(t, y, params) -> dy_dt`, with optional
      `jac_y(t, y, params)` and `jac_params(t, y, params)`.
    """
    from . import _core

    params_list: List[float] = [float(v) for v in params]
    y0_list: List[float] = [float(v) for v in y0]
    if callable(a_params):
        return _core.forward_sensitivity_solve_callback(
            a_params,
            params_list,
            y0_list,
            float(t0),
            float(t1),
            jac_y=jac_y,
            jac_params=jac_params,
            solver=str(solver),
            rtol=float(rtol),
            atol=float(atol),
            h0=float(h0),
            h_min=float(h_min),
            h_max=float(h_max),
            max_steps=int(max_steps),
            dense_output=bool(dense_output),
        )
    if jac_y is not None or jac_params is not None:
        raise ValueError(
            "jac_y/jac_params are only supported when forward_sensitivity_solve() is used with a callback rhs"
        )
    a_params_list: List[List[List[float]]] = [
        [list(map(float, row)) for row in ap] for ap in a_params
    ]
    return _core.forward_sensitivity_solve(
        a_params_list,
        params_list,
        y0_list,
        float(t0),
        float(t1),
        solver=str(solver),
        rtol=float(rtol),
        atol=float(atol),
        h0=float(h0),
        h_min=float(h_min),
        h_max=float(h_max),
        max_steps=int(max_steps),
        dense_output=bool(dense_output),
    )


__all__ = [
    "rk4_linear",
    "rk4_linear_dde",
    "lsoda",
    "forward_sensitivity_solve",
]
