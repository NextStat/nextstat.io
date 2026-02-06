"""Python-first data adapters (Phase 5 / Sprint 5.3).

This module provides small helpers for:
- building regression models from `X`, `y`, and optional `group_idx`
- serializing a minimal JSON spec for test reproduction

No pandas dependency (NumPy is optional; arrays are accepted via `.tolist()`).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, List, Literal, Optional, Sequence, Union, cast


def _tolist(x: Any) -> Any:
    """Best-effort conversion for numpy arrays without importing numpy."""

    tolist = getattr(x, "tolist", None)
    if callable(tolist):
        return tolist()
    return x


def _as_2d_float_list(x: Any) -> List[List[float]]:
    x = _tolist(x)
    if not isinstance(x, Sequence) or isinstance(x, (bytes, str)):
        raise TypeError("X must be a 2D sequence (or numpy array).")
    out: List[List[float]] = []
    for row in x:
        row = _tolist(row)
        if not isinstance(row, Sequence) or isinstance(row, (bytes, str)):
            raise TypeError("X must be a 2D sequence (or numpy array).")
        out.append([float(v) for v in row])
    return out


def _as_1d_float_list(y: Any) -> List[float]:
    y = _tolist(y)
    if not isinstance(y, Sequence) or isinstance(y, (bytes, str)):
        raise TypeError("y must be a 1D sequence (or numpy array).")
    return [float(v) for v in y]


def _as_1d_u8_list(y: Any) -> List[int]:
    y = _tolist(y)
    if not isinstance(y, Sequence) or isinstance(y, (bytes, str)):
        raise TypeError("y must be a 1D sequence (or numpy array).")
    out: List[int] = []
    for v in y:
        iv = int(v)
        if iv not in (0, 1):
            raise ValueError("logistic y must contain only 0/1")
        out.append(iv)
    return out


def _as_1d_u64_list(y: Any) -> List[int]:
    y = _tolist(y)
    if not isinstance(y, Sequence) or isinstance(y, (bytes, str)):
        raise TypeError("y must be a 1D sequence (or numpy array).")
    out: List[int] = []
    for v in y:
        iv = int(v)
        if iv < 0:
            raise ValueError("poisson y must be >= 0")
        out.append(iv)
    return out


def _as_group_idx(group_idx: Any) -> Optional[List[int]]:
    if group_idx is None:
        return None
    group_idx = _tolist(group_idx)
    if not isinstance(group_idx, Sequence) or isinstance(group_idx, (bytes, str)):
        raise TypeError("group_idx must be a 1D sequence (or numpy array).")
    return [int(v) for v in group_idx]


Kind = Literal["linear", "logistic", "poisson"]


@dataclass(frozen=True)
class GlmSpec:
    """Minimal spec for building a GLM model from data."""

    kind: Kind
    x: List[List[float]]
    y: Union[List[float], List[int]]
    include_intercept: bool = True
    group_idx: Optional[List[int]] = None
    n_groups: Optional[int] = None
    offset: Optional[List[float]] = None
    coef_prior_mu: float = 0.0
    coef_prior_sigma: float = 10.0

    @staticmethod
    def linear_regression(
        *,
        x: Any,
        y: Any,
        include_intercept: bool = True,
        group_idx: Any = None,
        n_groups: Optional[int] = None,
        coef_prior_mu: float = 0.0,
        coef_prior_sigma: float = 10.0,
    ) -> "GlmSpec":
        return GlmSpec(
            kind="linear",
            x=_as_2d_float_list(x),
            y=_as_1d_float_list(y),
            include_intercept=bool(include_intercept),
            group_idx=_as_group_idx(group_idx),
            n_groups=n_groups,
            coef_prior_mu=float(coef_prior_mu),
            coef_prior_sigma=float(coef_prior_sigma),
        )

    @staticmethod
    def logistic_regression(
        *,
        x: Any,
        y: Any,
        include_intercept: bool = True,
        group_idx: Any = None,
        n_groups: Optional[int] = None,
        coef_prior_mu: float = 0.0,
        coef_prior_sigma: float = 10.0,
    ) -> "GlmSpec":
        return GlmSpec(
            kind="logistic",
            x=_as_2d_float_list(x),
            y=_as_1d_u8_list(y),
            include_intercept=bool(include_intercept),
            group_idx=_as_group_idx(group_idx),
            n_groups=n_groups,
            offset=None,
            coef_prior_mu=float(coef_prior_mu),
            coef_prior_sigma=float(coef_prior_sigma),
        )

    @staticmethod
    def poisson_regression(
        *,
        x: Any,
        y: Any,
        include_intercept: bool = True,
        offset: Any = None,
        group_idx: Any = None,
        n_groups: Optional[int] = None,
        coef_prior_mu: float = 0.0,
        coef_prior_sigma: float = 10.0,
    ) -> "GlmSpec":
        return GlmSpec(
            kind="poisson",
            x=_as_2d_float_list(x),
            y=_as_1d_u64_list(y),
            include_intercept=bool(include_intercept),
            group_idx=_as_group_idx(group_idx),
            n_groups=n_groups,
            offset=(None if offset is None else _as_1d_float_list(offset)),
            coef_prior_mu=float(coef_prior_mu),
            coef_prior_sigma=float(coef_prior_sigma),
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "kind": self.kind,
                "include_intercept": self.include_intercept,
                "x": self.x,
                "y": self.y,
                "group_idx": self.group_idx,
                "n_groups": self.n_groups,
                "offset": self.offset,
                "coef_prior_mu": self.coef_prior_mu,
                "coef_prior_sigma": self.coef_prior_sigma,
            }
        )

    @staticmethod
    def from_json(s: str) -> "GlmSpec":
        d = json.loads(s)
        if not isinstance(d, dict):
            raise TypeError("spec JSON must decode to an object")
        kind = cast(Kind, d.get("kind"))
        if kind not in ("linear", "logistic", "poisson"):
            raise ValueError(f"Unsupported kind: {kind!r}")
        return GlmSpec(
            kind=kind,
            include_intercept=bool(d.get("include_intercept", True)),
            x=_as_2d_float_list(d.get("x")),
            y=(
                _as_1d_float_list(d.get("y"))
                if kind == "linear"
                else (_as_1d_u8_list(d.get("y")) if kind == "logistic" else _as_1d_u64_list(d.get("y")))
            ),
            group_idx=_as_group_idx(d.get("group_idx")),
            n_groups=(None if d.get("n_groups") is None else int(d.get("n_groups"))),
            offset=(None if d.get("offset") is None else _as_1d_float_list(d.get("offset"))),
            coef_prior_mu=float(d.get("coef_prior_mu", 0.0)),
            coef_prior_sigma=float(d.get("coef_prior_sigma", 10.0)),
        )

    def build(self) -> Any:
        from . import _core  # local import

        if self.group_idx is None and self.n_groups is not None:
            raise ValueError("n_groups requires group_idx")
        if self.group_idx is not None and not self.group_idx:
            raise ValueError("group_idx must be non-empty if provided")
        if self.kind != "poisson" and self.offset is not None:
            raise ValueError("offset is only supported for poisson")

        ng = self.n_groups
        if self.group_idx is not None and ng is None:
            ng = max(self.group_idx) + 1

        if self.kind == "linear":
            return _core.ComposedGlmModel.linear_regression(
                self.x,
                cast(List[float], self.y),
                include_intercept=self.include_intercept,
                group_idx=self.group_idx,
                n_groups=ng,
                coef_prior_mu=self.coef_prior_mu,
                coef_prior_sigma=self.coef_prior_sigma,
            )
        if self.kind == "logistic":
            return _core.ComposedGlmModel.logistic_regression(
                self.x,
                cast(List[int], self.y),
                include_intercept=self.include_intercept,
                group_idx=self.group_idx,
                n_groups=ng,
                coef_prior_mu=self.coef_prior_mu,
                coef_prior_sigma=self.coef_prior_sigma,
            )
        return _core.ComposedGlmModel.poisson_regression(
            self.x,
            cast(List[int], self.y),
            include_intercept=self.include_intercept,
            offset=self.offset,
            group_idx=self.group_idx,
            n_groups=ng,
            coef_prior_mu=self.coef_prior_mu,
            coef_prior_sigma=self.coef_prior_sigma,
        )


__all__ = ["GlmSpec"]
