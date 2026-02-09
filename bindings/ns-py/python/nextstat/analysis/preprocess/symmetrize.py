"""Symmetrization policies for histogram shape systematics.

This module is intentionally dependency-free (no numpy).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Literal, Sequence

SymmetrizeMethod = Literal["onesided", "twosided", "absmean", "max"]
NegativePolicy = Literal["error", "clamp"]


@dataclass(frozen=True)
class SymmetrizeResult:
    up: list[float]
    down: list[float]
    method: str
    negative_policy: str


def _as_list(x: Sequence[float] | Iterable[float]) -> list[float]:
    return list(x)


def _validate_vec(name: str, xs: Sequence[float]) -> None:
    for i, v in enumerate(xs):
        fv = float(v)
        if not isfinite(fv):
            raise ValueError(f"{name}[{i}] must be finite, got {v!r}")


def _validate_lengths(nominal: Sequence[float], up: Sequence[float] | None, down: Sequence[float] | None) -> None:
    n = len(nominal)
    if up is not None and len(up) != n:
        raise ValueError(f"up length mismatch: len(up)={len(up)} len(nominal)={n}")
    if down is not None and len(down) != n:
        raise ValueError(f"down length mismatch: len(down)={len(down)} len(nominal)={n}")


def _max_abs_delta(nominal: Sequence[float], var: Sequence[float]) -> float:
    best = 0.0
    for n, v in zip(nominal, var):
        d = abs(float(v) - float(n))
        if d > best:
            best = d
    return best


def _apply_negative_policy(xs: list[float], policy: NegativePolicy) -> list[float]:
    if policy == "error":
        for i, v in enumerate(xs):
            if v < 0.0:
                raise ValueError(f"negative bin after symmetrization at index {i}: {v}")
        return xs
    if policy == "clamp":
        return [0.0 if v < 0.0 else v for v in xs]
    raise ValueError(f"unknown negative_policy: {policy!r}")


def symmetrize_shapes(
    nominal: Sequence[float] | Iterable[float],
    *,
    up: Sequence[float] | Iterable[float] | None = None,
    down: Sequence[float] | Iterable[float] | None = None,
    method: SymmetrizeMethod = "absmean",
    negative_policy: NegativePolicy = "error",
) -> SymmetrizeResult:
    """Symmetrize shape variations around a nominal template.

    Parameters
    - nominal: nominal template bins.
    - up/down: variation templates. For `method="onesided"`, if only one side is provided,
      the other is reflected: `other = 2*nominal - provided`.
    - method:
      - "onesided": reflect one side around nominal (chooses the side with larger max |delta|
        if both provided).
      - "twosided": symmetric slope method: `delta = 0.5*(up - down)`, then `nom ± delta`.
      - "absmean": magnitude = 0.5*(|up-nom| + |down-nom|), then `nom ± magnitude`.
      - "max": magnitude = max(|up-nom|, |down-nom|), then `nom ± magnitude`.
    - negative_policy:
      - "error": raise if any output bin becomes negative.
      - "clamp": clamp negative output bins to 0.
    """
    nominal_v = _as_list(nominal)
    up_v = None if up is None else _as_list(up)
    down_v = None if down is None else _as_list(down)

    _validate_vec("nominal", nominal_v)
    if up_v is not None:
        _validate_vec("up", up_v)
    if down_v is not None:
        _validate_vec("down", down_v)
    _validate_lengths(nominal_v, up_v, down_v)

    if method not in ("onesided", "twosided", "absmean", "max"):
        raise ValueError(f"unknown method: {method!r}")

    n = len(nominal_v)
    if n == 0:
        return SymmetrizeResult(up=[], down=[], method=method, negative_policy=negative_policy)

    if method == "onesided":
        if up_v is None and down_v is None:
            raise ValueError("onesided symmetrization requires up or down")
        if up_v is None:
            ref = down_v  # type: ignore[assignment]
        elif down_v is None:
            ref = up_v
        else:
            ref = up_v if _max_abs_delta(nominal_v, up_v) >= _max_abs_delta(nominal_v, down_v) else down_v
        assert ref is not None
        other = [(2.0 * float(n0)) - float(v0) for n0, v0 in zip(nominal_v, ref)]
        up_out = [float(v) for v in ref]
        down_out = other
        up_out = _apply_negative_policy(up_out, negative_policy)
        down_out = _apply_negative_policy(down_out, negative_policy)
        return SymmetrizeResult(up=up_out, down=down_out, method=method, negative_policy=negative_policy)

    if up_v is None or down_v is None:
        raise ValueError(f"{method} symmetrization requires both up and down")

    if method == "twosided":
        delta = [0.5 * (float(u) - float(d)) for u, d in zip(up_v, down_v)]
        up_out = [float(n0) + float(dt) for n0, dt in zip(nominal_v, delta)]
        down_out = [float(n0) - float(dt) for n0, dt in zip(nominal_v, delta)]
        up_out = _apply_negative_policy(up_out, negative_policy)
        down_out = _apply_negative_policy(down_out, negative_policy)
        return SymmetrizeResult(up=up_out, down=down_out, method=method, negative_policy=negative_policy)

    # absmean / max
    mag: list[float] = []
    if method == "absmean":
        for n0, u, d in zip(nominal_v, up_v, down_v):
            du = abs(float(u) - float(n0))
            dd = abs(float(d) - float(n0))
            mag.append(0.5 * (du + dd))
    elif method == "max":
        for n0, u, d in zip(nominal_v, up_v, down_v):
            du = abs(float(u) - float(n0))
            dd = abs(float(d) - float(n0))
            mag.append(du if du >= dd else dd)

    up_out = [float(n0) + float(m) for n0, m in zip(nominal_v, mag)]
    down_out = [float(n0) - float(m) for n0, m in zip(nominal_v, mag)]
    up_out = _apply_negative_policy(up_out, negative_policy)
    down_out = _apply_negative_policy(down_out, negative_policy)
    return SymmetrizeResult(up=up_out, down=down_out, method=method, negative_policy=negative_policy)


__all__ = [
    "NegativePolicy",
    "SymmetrizeMethod",
    "SymmetrizeResult",
    "symmetrize_shapes",
]
