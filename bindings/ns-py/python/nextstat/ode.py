"""ODE helpers (Phase 13 baseline).

This module provides a deterministic baseline integrator (RK4) for simple systems.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


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


__all__ = ["rk4_linear"]

