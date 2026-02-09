"""Smoothing algorithms for histogram shape systematics.

Implements 353QH,twice (ROOT TH1::Smooth equivalent) and Gaussian kernel
smoothing. Works on **deltas** (variation - nominal) to preserve the nominal
shape.

This module is intentionally dependency-free (no numpy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

SmoothMethod = Literal["353qh_twice", "gaussian"]


@dataclass(frozen=True)
class SmoothResult:
    up: list[float]
    down: list[float]
    method: str
    max_delta_before_up: float
    max_delta_after_up: float
    max_delta_before_down: float
    max_delta_after_down: float
    maxvariation_applied: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _running_median(x: list[float], k: int) -> list[float]:
    """Running median with window size *k* (3 or 5). Edges: repeat boundary."""
    n = len(x)
    if n == 0:
        return []
    half = k // 2
    out: list[float] = []
    for i in range(n):
        window: list[float] = []
        for j in range(i - half, i + half + 1):
            idx = max(0, min(n - 1, j))
            window.append(x[idx])
        window.sort()
        out.append(window[len(window) // 2])
    return out


def _hanning(x: list[float]) -> list[float]:
    """Hanning smoothing: out[i] = 0.25*x[i-1] + 0.5*x[i] + 0.25*x[i+1].

    Edges: repeat boundary values.
    """
    n = len(x)
    if n == 0:
        return []
    if n == 1:
        return [x[0]]
    out: list[float] = []
    for i in range(n):
        left = x[max(0, i - 1)]
        right = x[min(n - 1, i + 1)]
        out.append(0.25 * left + 0.5 * x[i] + 0.25 * right)
    return out


def _353qh_once(delta: list[float]) -> list[float]:
    """One pass of 353QH: median3 → median5 → median3 → hanning, then smooth residual."""
    m3 = _running_median(delta, 3)
    m5 = _running_median(m3, 5)
    m3b = _running_median(m5, 3)
    h = _hanning(m3b)
    residual = [d - hv for d, hv in zip(delta, h)]
    r3 = _running_median(residual, 3)
    r5 = _running_median(r3, 5)
    r3b = _running_median(r5, 3)
    rh = _hanning(r3b)
    return [hv + rv for hv, rv in zip(h, rh)]


def smooth_353qh_twice(delta: list[float]) -> list[float]:
    """Apply 353QH once, then again on residuals (353QH,twice)."""
    if len(delta) <= 2:
        return list(delta)
    pass1 = _353qh_once(delta)
    residual = [d - p for d, p in zip(delta, pass1)]
    pass2 = _353qh_once(residual)
    return [p1 + p2 for p1, p2 in zip(pass1, pass2)]


def smooth_gaussian_kernel(delta: list[float], sigma: float = 1.5) -> list[float]:
    """Gaussian kernel smooth on delta. *sigma* in units of bins.

    Truncates kernel at 3*sigma. For sigma <= 0 returns the input unchanged.
    """
    n = len(delta)
    if n == 0:
        return []
    if sigma <= 0.0:
        return list(delta)

    half_width = max(1, int(math.ceil(3.0 * sigma)))
    out: list[float] = []
    for i in range(n):
        w_sum = 0.0
        v_sum = 0.0
        for j in range(max(0, i - half_width), min(n, i + half_width + 1)):
            w = math.exp(-0.5 * ((j - i) / sigma) ** 2)
            w_sum += w
            v_sum += w * delta[j]
        out.append(v_sum / w_sum if w_sum > 0.0 else delta[i])
    return out


def apply_maxvariation_cap(smoothed_delta: list[float], original_delta: list[float]) -> list[float]:
    """Cap |smoothed[i]| <= max(|original|)."""
    if not original_delta:
        return list(smoothed_delta)
    max_orig = max(abs(d) for d in original_delta)
    return [max(-max_orig, min(max_orig, d)) for d in smoothed_delta]


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------


def _max_abs(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return max(abs(v) for v in xs)


def smooth_variation(
    nominal: Sequence[float],
    up: Sequence[float],
    down: Sequence[float],
    *,
    method: SmoothMethod = "353qh_twice",
    sigma: float = 1.5,
    apply_maxvariation: bool = True,
) -> SmoothResult:
    """Smooth up/down variations around nominal.

    Works on deltas (variation - nominal), smooths each independently,
    optionally applies MAXVARIATION cap, then reconstructs.
    """
    nom = [float(v) for v in nominal]
    up_f = [float(v) for v in up]
    down_f = [float(v) for v in down]

    delta_up = [u - n for u, n in zip(up_f, nom)]
    delta_down = [d - n for d, n in zip(down_f, nom)]

    max_delta_before_up = _max_abs(delta_up)
    max_delta_before_down = _max_abs(delta_down)

    if method == "353qh_twice":
        sm_up = smooth_353qh_twice(delta_up)
        sm_down = smooth_353qh_twice(delta_down)
    elif method == "gaussian":
        sm_up = smooth_gaussian_kernel(delta_up, sigma=sigma)
        sm_down = smooth_gaussian_kernel(delta_down, sigma=sigma)
    else:
        raise ValueError(f"unknown smooth method: {method!r}")

    maxvar_applied = False
    if apply_maxvariation:
        sm_up_capped = apply_maxvariation_cap(sm_up, delta_up)
        sm_down_capped = apply_maxvariation_cap(sm_down, delta_down)
        if sm_up_capped != sm_up or sm_down_capped != sm_down:
            maxvar_applied = True
        sm_up = sm_up_capped
        sm_down = sm_down_capped

    max_delta_after_up = _max_abs(sm_up)
    max_delta_after_down = _max_abs(sm_down)

    smoothed_up = [n + d for n, d in zip(nom, sm_up)]
    smoothed_down = [n + d for n, d in zip(nom, sm_down)]

    return SmoothResult(
        up=smoothed_up,
        down=smoothed_down,
        method=method,
        max_delta_before_up=max_delta_before_up,
        max_delta_after_up=max_delta_after_up,
        max_delta_before_down=max_delta_before_down,
        max_delta_after_down=max_delta_after_down,
        maxvariation_applied=maxvar_applied,
    )


__all__ = [
    "SmoothMethod",
    "SmoothResult",
    "apply_maxvariation_cap",
    "smooth_353qh_twice",
    "smooth_gaussian_kernel",
    "smooth_variation",
]
