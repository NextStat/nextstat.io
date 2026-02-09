"""Hygiene policies for negative bins / integrals in histogram templates.

TREx-like workflows often require explicit handling of negative bins that can
otherwise break fits or produce misleading plots.

This module is intentionally dependency-free (no numpy).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Literal, Sequence

NegativeBinsPolicy = Literal["error", "clamp_renorm", "keep_warn"]


@dataclass(frozen=True)
class NegativeBinsResult:
    bins: list[float]
    policy: str
    changed: bool
    n_negative: int
    min_bin: float
    sum_before: float
    sum_after: float
    scale: float | None
    warnings: list[str]


def _as_list(xs: Sequence[float] | Iterable[float]) -> list[float]:
    return [float(x) for x in xs]


def _validate_finite(name: str, xs: Sequence[float]) -> None:
    for i, v in enumerate(xs):
        if not isfinite(float(v)):
            raise ValueError(f"{name}[{i}] must be finite, got {v!r}")


def apply_negative_bins_policy(
    bins: Sequence[float] | Iterable[float],
    *,
    policy: NegativeBinsPolicy = "error",
    tol: float = 1e-12,
    renorm: bool = True,
) -> NegativeBinsResult:
    """Apply an explicit negative-bin policy to a 1D template.

    Policies
    - "error": raise if any bin < -tol.
    - "keep_warn": keep bins unchanged and return warnings (no I/O side effects).
    - "clamp_renorm": clamp bins < 0 to 0 and (optionally) rescale all bins to preserve
      the original integral when possible.
    """
    xs = _as_list(bins)
    _validate_finite("bins", xs)

    t = float(tol)
    neg_idx = [i for i, v in enumerate(xs) if float(v) < -t]
    n_negative = len(neg_idx)
    min_bin = min(xs) if xs else 0.0
    sum_before = float(sum(xs))

    if policy not in ("error", "keep_warn", "clamp_renorm"):
        raise ValueError(f"unknown policy: {policy!r}")

    if n_negative == 0:
        return NegativeBinsResult(
            bins=xs,
            policy=policy,
            changed=False,
            n_negative=0,
            min_bin=min_bin,
            sum_before=sum_before,
            sum_after=sum_before,
            scale=None,
            warnings=[],
        )

    if policy == "error":
        raise ValueError(f"negative bins found (n={n_negative}, min={min_bin}): idx={neg_idx}")

    if policy == "keep_warn":
        return NegativeBinsResult(
            bins=xs,
            policy=policy,
            changed=False,
            n_negative=n_negative,
            min_bin=min_bin,
            sum_before=sum_before,
            sum_after=sum_before,
            scale=None,
            warnings=[f"negative bins kept (n={n_negative}, min={min_bin})"],
        )

    # clamp_renorm
    clamped = [0.0 if float(v) < 0.0 else float(v) for v in xs]
    sum_after = float(sum(clamped))
    scale: float | None = None
    warnings: list[str] = []

    out = clamped
    if renorm:
        if sum_before > t and sum_after > t:
            scale = sum_before / sum_after
            out = [float(v) * scale for v in clamped]
            sum_after = float(sum(out))
        else:
            warnings.append("renorm skipped (non-positive integral before/after clamping)")

    return NegativeBinsResult(
        bins=out,
        policy=policy,
        changed=True,
        n_negative=n_negative,
        min_bin=min_bin,
        sum_before=sum_before,
        sum_after=sum_after,
        scale=scale,
        warnings=warnings,
    )


__all__ = [
    "NegativeBinsPolicy",
    "NegativeBinsResult",
    "apply_negative_bins_policy",
]

