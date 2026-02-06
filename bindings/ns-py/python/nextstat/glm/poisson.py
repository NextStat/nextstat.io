from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, List, Optional, Sequence


def _tolist(x: Any) -> Any:
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


def _as_1d_u64_list(y: Any) -> List[int]:
    y = _tolist(y)
    if not isinstance(y, Sequence) or isinstance(y, (bytes, str)):
        raise TypeError("y must be a 1D sequence (or numpy array).")
    out: List[int] = []
    for v in y:
        iv = int(v)
        if iv < 0:
            raise ValueError("poisson y must be non-negative")
        out.append(iv)
    return out


def _as_offset(offset: Any, *, n: int) -> Optional[List[float]]:
    if offset is None:
        return None
    offset = _tolist(offset)
    if not isinstance(offset, Sequence) or isinstance(offset, (bytes, str)):
        raise TypeError("offset must be a 1D sequence (or numpy array).")
    out = [float(v) for v in offset]
    if len(out) != n:
        raise ValueError(f"offset has wrong length: expected {n}, got {len(out)}")
    return out


def _offset_from_exposure(exposure: Any, *, n: int) -> List[float]:
    exposure = _tolist(exposure)
    if not isinstance(exposure, Sequence) or isinstance(exposure, (bytes, str)):
        raise TypeError("exposure must be a 1D sequence (or numpy array).")
    out: List[float] = []
    for v in exposure:
        ev = float(v)
        if not (ev > 0.0) or not math.isfinite(ev):
            raise ValueError("exposure must be finite and > 0")
        out.append(math.log(ev))
    if len(out) != n:
        raise ValueError(f"exposure has wrong length: expected {n}, got {len(out)}")
    return out


def _eta(x: List[List[float]], params: List[float], *, include_intercept: bool) -> List[float]:
    if include_intercept:
        b0 = float(params[0])
        beta = params[1:]
    else:
        b0 = 0.0
        beta = params
    out: List[float] = []
    for row in x:
        if len(row) != len(beta):
            raise ValueError("X has wrong number of columns for fitted parameters.")
        out.append(b0 + sum(float(a) * float(b) for a, b in zip(row, beta)))
    return out


@dataclass(frozen=True)
class FittedPoissonRegression:
    model: Any
    result: Any
    include_intercept: bool
    offset: Optional[List[float]]

    @property
    def params_(self) -> List[float]:
        return list(self.result.parameters)

    @property
    def intercept_(self) -> float:
        return float(self.params_[0]) if self.include_intercept else 0.0

    @property
    def coef_(self) -> List[float]:
        return self.params_[1:] if self.include_intercept else self.params_

    def predict_mean(
        self,
        x: Any,
        *,
        offset: Any = None,
        exposure: Any = None,
    ) -> List[float]:
        x2 = _as_2d_float_list(x)
        n = len(x2)
        off = None
        if exposure is not None:
            off = _offset_from_exposure(exposure, n=n)
        elif offset is not None:
            off = _as_offset(offset, n=n)
        else:
            off = self.offset

        eta = _eta(x2, self.params_, include_intercept=self.include_intercept)
        if off is not None:
            if len(off) != len(eta):
                raise ValueError("offset length mismatch")
            eta = [e + o for e, o in zip(eta, off)]
        return [math.exp(e) for e in eta]


def fit(
    x: Any,
    y: Any,
    *,
    include_intercept: bool = True,
    offset: Any = None,
    exposure: Any = None,
) -> FittedPoissonRegression:
    from .. import _core  # local import (compiled module)

    x2 = _as_2d_float_list(x)
    y1 = _as_1d_u64_list(y)
    n = len(x2)
    off = None
    if exposure is not None:
        off = _offset_from_exposure(exposure, n=n)
    elif offset is not None:
        off = _as_offset(offset, n=n)
    model = _core.PoissonRegressionModel(x2, y1, include_intercept=bool(include_intercept), offset=off)
    result = _core.fit(model)
    return FittedPoissonRegression(
        model=model,
        result=result,
        include_intercept=bool(include_intercept),
        offset=off,
    )

