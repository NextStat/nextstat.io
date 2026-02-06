from __future__ import annotations

from dataclasses import dataclass
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


def _as_1d_float_list(y: Any) -> List[float]:
    y = _tolist(y)
    if not isinstance(y, Sequence) or isinstance(y, (bytes, str)):
        raise TypeError("y must be a 1D sequence (or numpy array).")
    return [float(v) for v in y]


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
class FittedLinearRegression:
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

    def predict(self, x: Any) -> List[float]:
        x2 = _as_2d_float_list(x)
        return _eta(x2, self.params_, include_intercept=self.include_intercept)


def fit(x: Any, y: Any, *, include_intercept: bool = True) -> FittedLinearRegression:
    from .. import _core  # local import (compiled module)

    x2 = _as_2d_float_list(x)
    y1 = _as_1d_float_list(y)
    model = _core.LinearRegressionModel(x2, y1, include_intercept=bool(include_intercept))
    result = _core.fit(model)
    return FittedLinearRegression(model=model, result=result, include_intercept=bool(include_intercept))


def fit_ols(x: Any, y: Any, *, include_intercept: bool = True) -> List[float]:
    """Closed-form OLS coefficients (sigma fixed to 1).

    Returns parameter vector in NextStat order:
    - with intercept: [intercept, beta1, beta2, ...]
    - no intercept:   [beta1, beta2, ...]
    """

    from .. import _core  # local import (compiled module)

    x2 = _as_2d_float_list(x)
    y1 = _as_1d_float_list(y)
    return list(_core.ols_fit(x2, y1, include_intercept=bool(include_intercept)))

