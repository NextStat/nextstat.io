from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, List, Sequence


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


def _as_1d_binary_list(y: Any) -> List[int]:
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


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


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
class FittedLogisticRegression:
    model: Any
    result: Any
    include_intercept: bool

    @property
    def params_(self) -> List[float]:
        return list(self.result.parameters)

    @property
    def intercept_(self) -> float:
        return float(self.params_[0]) if self.include_intercept else 0.0

    @property
    def coef_(self) -> List[float]:
        return self.params_[1:] if self.include_intercept else self.params_

    def predict_logit(self, x: Any) -> List[float]:
        x2 = _as_2d_float_list(x)
        return _eta(x2, self.params_, include_intercept=self.include_intercept)

    def predict_proba(self, x: Any) -> List[float]:
        return [_sigmoid(v) for v in self.predict_logit(x)]

    def predict(self, x: Any, *, threshold: float = 0.5) -> List[int]:
        t = float(threshold)
        return [int(p >= t) for p in self.predict_proba(x)]


def fit(x: Any, y: Any, *, include_intercept: bool = True) -> FittedLogisticRegression:
    from .. import _core  # local import (compiled module)

    x2 = _as_2d_float_list(x)
    y1 = _as_1d_binary_list(y)
    model = _core.LogisticRegressionModel(x2, y1, include_intercept=bool(include_intercept))
    result = _core.fit(model)
    return FittedLogisticRegression(model=model, result=result, include_intercept=bool(include_intercept))

