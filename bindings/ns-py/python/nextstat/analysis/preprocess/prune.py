"""Pruning criteria for removing negligible systematics from workspaces.

Implements shape, norm, and overall (norm+shape decomposition) pruning
following TRExFitter conventions.

This module is intentionally dependency-free (no numpy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

PruneMethod = Literal["shape", "norm", "overall"]


@dataclass(frozen=True)
class PruneDecision:
    should_prune: bool
    reason: str
    max_rel_delta_up: float
    max_rel_delta_down: float
    norm_effect_up: float
    norm_effect_down: float


def _max_rel_delta(nominal: Sequence[float], variation: Sequence[float]) -> float:
    """Max |var[i] - nom[i]| / nom[i] over bins where nom[i] > 0."""
    best = 0.0
    for n, v in zip(nominal, variation):
        nf = float(n)
        if nf <= 0.0:
            continue
        rel = abs(float(v) - nf) / nf
        if rel > best:
            best = rel
    return best


def should_prune_histosys_shape(
    nominal: Sequence[float],
    hi_data: Sequence[float],
    lo_data: Sequence[float],
    *,
    threshold: float = 0.005,
) -> PruneDecision:
    """Prune if max |delta/nominal| < threshold for both up and down."""
    max_rel_up = _max_rel_delta(nominal, hi_data)
    max_rel_down = _max_rel_delta(nominal, lo_data)
    max_overall = max(max_rel_up, max_rel_down)
    prune = max_overall < threshold
    reason = f"max_rel_delta={max_overall:.6g} {'<' if prune else '>='} threshold={threshold}"
    return PruneDecision(
        should_prune=prune,
        reason=reason,
        max_rel_delta_up=max_rel_up,
        max_rel_delta_down=max_rel_down,
        norm_effect_up=0.0,
        norm_effect_down=0.0,
    )


def should_prune_normsys(
    hi: float,
    lo: float,
    *,
    threshold: float = 0.005,
) -> PruneDecision:
    """Prune if |hi - 1| < threshold AND |lo - 1| < threshold."""
    delta_hi = abs(float(hi) - 1.0)
    delta_lo = abs(float(lo) - 1.0)
    max_delta = max(delta_hi, delta_lo)
    prune = max_delta < threshold
    reason = f"max_norm_delta={max_delta:.6g} {'<' if prune else '>='} threshold={threshold}"
    return PruneDecision(
        should_prune=prune,
        reason=reason,
        max_rel_delta_up=delta_hi,
        max_rel_delta_down=delta_lo,
        norm_effect_up=float(hi),
        norm_effect_down=float(lo),
    )


def should_prune_histosys_overall(
    nominal: Sequence[float],
    hi_data: Sequence[float],
    lo_data: Sequence[float],
    *,
    norm_threshold: float = 0.005,
    shape_threshold: float = 0.005,
) -> PruneDecision:
    """Decompose histosys into norm (integral ratio) + shape (residual).

    norm_effect = sum(var) / sum(nom)
    shape_var = var * (sum(nom) / sum(var))  — rescaled to same integral
    Then apply shape criterion on the residual.
    Prune if *both* norm and shape effects are below their thresholds.
    """
    nom_f = [float(v) for v in nominal]
    hi_f = [float(v) for v in hi_data]
    lo_f = [float(v) for v in lo_data]

    sum_nom = sum(nom_f)

    # Norm effects
    sum_hi = sum(hi_f)
    sum_lo = sum(lo_f)

    norm_up = sum_hi / sum_nom if sum_nom > 0.0 else 1.0
    norm_down = sum_lo / sum_nom if sum_nom > 0.0 else 1.0

    norm_delta_up = abs(norm_up - 1.0)
    norm_delta_down = abs(norm_down - 1.0)

    # Shape effects: rescale variation to same integral as nominal
    if sum_hi > 0.0 and sum_nom > 0.0:
        scale_hi = sum_nom / sum_hi
        shape_hi = [v * scale_hi for v in hi_f]
    else:
        shape_hi = list(hi_f)

    if sum_lo > 0.0 and sum_nom > 0.0:
        scale_lo = sum_nom / sum_lo
        shape_lo = [v * scale_lo for v in lo_f]
    else:
        shape_lo = list(lo_f)

    max_shape_up = _max_rel_delta(nom_f, shape_hi)
    max_shape_down = _max_rel_delta(nom_f, shape_lo)

    norm_small = max(norm_delta_up, norm_delta_down) < norm_threshold
    shape_small = max(max_shape_up, max_shape_down) < shape_threshold

    prune = norm_small and shape_small

    parts: list[str] = []
    parts.append(f"norm_delta=({norm_delta_up:.6g},{norm_delta_down:.6g}) {'<' if norm_small else '>='} {norm_threshold}")
    parts.append(f"shape_delta=({max_shape_up:.6g},{max_shape_down:.6g}) {'<' if shape_small else '>='} {shape_threshold}")
    reason = "; ".join(parts)

    return PruneDecision(
        should_prune=prune,
        reason=reason,
        max_rel_delta_up=max(norm_delta_up, max_shape_up),
        max_rel_delta_down=max(norm_delta_down, max_shape_down),
        norm_effect_up=norm_up,
        norm_effect_down=norm_down,
    )


__all__ = [
    "PruneDecision",
    "PruneMethod",
    "should_prune_histosys_overall",
    "should_prune_histosys_shape",
    "should_prune_normsys",
]
