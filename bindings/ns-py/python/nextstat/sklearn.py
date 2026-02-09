"""scikit-learn style adapters (Phase 11.5).

These wrappers are intentionally lightweight:
- If scikit-learn is installed, classes inherit BaseEstimator and the relevant mixins.
- If not installed, the classes still work as plain Python estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from .glm import linear as _linear
from .glm import logistic as _logistic
from .glm import poisson as _poisson

try:  # pragma: no cover - optional dependency
    from sklearn.base import BaseEstimator as _BaseEstimator  # type: ignore
    from sklearn.base import ClassifierMixin as _ClassifierMixin  # type: ignore
    from sklearn.base import RegressorMixin as _RegressorMixin  # type: ignore
except Exception:  # pragma: no cover

    class _BaseEstimator:
        def get_params(self, deep: bool = True) -> dict[str, Any]:
            return dict(self.__dict__)

        def set_params(self, **params: Any) -> "_BaseEstimator":
            for k, v in params.items():
                setattr(self, k, v)
            return self

    class _RegressorMixin:
        pass

    class _ClassifierMixin:
        pass


@dataclass
class NextStatLinearRegression(_BaseEstimator, _RegressorMixin):
    include_intercept: bool = True
    l2: Optional[float] = None
    penalize_intercept: bool = False

    fit_result_: Optional[_linear.LinearFit] = None
    coef_: Optional[List[float]] = None
    intercept_: Optional[float] = None

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> "NextStatLinearRegression":
        r = _linear.fit(
            X,
            y,
            include_intercept=bool(self.include_intercept),
            l2=self.l2,
            penalize_intercept=bool(self.penalize_intercept),
        )
        self.fit_result_ = r
        if r.include_intercept:
            self.intercept_ = float(r.coef[0]) if r.coef else 0.0
            self.coef_ = [float(v) for v in r.coef[1:]]
        else:
            self.intercept_ = 0.0
            self.coef_ = [float(v) for v in r.coef]
        return self

    def predict(self, X: Sequence[Sequence[float]]) -> List[float]:
        if self.fit_result_ is None:
            raise RuntimeError("estimator is not fitted")
        return self.fit_result_.predict(X)


@dataclass
class NextStatLogisticRegression(_BaseEstimator, _ClassifierMixin):
    include_intercept: bool = True
    l2: Optional[float] = None
    penalize_intercept: bool = False
    threshold: float = 0.5

    fit_result_: Optional[_logistic.LogisticFit] = None
    coef_: Optional[List[float]] = None
    intercept_: Optional[float] = None
    classes_: Optional[List[int]] = None

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> "NextStatLogisticRegression":
        r = _logistic.fit(
            X,
            y,
            include_intercept=bool(self.include_intercept),
            l2=self.l2,
            penalize_intercept=bool(self.penalize_intercept),
        )
        self.fit_result_ = r
        self.classes_ = [0, 1]
        if r.include_intercept:
            self.intercept_ = float(r.coef[0]) if r.coef else 0.0
            self.coef_ = [float(v) for v in r.coef[1:]]
        else:
            self.intercept_ = 0.0
            self.coef_ = [float(v) for v in r.coef]
        return self

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[List[float]]:
        if self.fit_result_ is None:
            raise RuntimeError("estimator is not fitted")
        p1 = self.fit_result_.predict_proba(X)
        return [[1.0 - float(p), float(p)] for p in p1]

    def predict(self, X: Sequence[Sequence[float]]) -> List[int]:
        if self.fit_result_ is None:
            raise RuntimeError("estimator is not fitted")
        return self.fit_result_.predict(X, threshold=float(self.threshold))


@dataclass
class NextStatPoissonRegressor(_BaseEstimator, _RegressorMixin):
    include_intercept: bool = True
    offset: Optional[Sequence[float]] = None
    exposure: Optional[Sequence[float]] = None
    l2: Optional[float] = None
    penalize_intercept: bool = False

    fit_result_: Optional[_poisson.PoissonFit] = None
    coef_: Optional[List[float]] = None
    intercept_: Optional[float] = None

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> "NextStatPoissonRegressor":
        r = _poisson.fit(
            X,
            y,
            include_intercept=bool(self.include_intercept),
            offset=self.offset,
            exposure=self.exposure,
            l2=self.l2,
            penalize_intercept=bool(self.penalize_intercept),
        )
        self.fit_result_ = r
        if r.include_intercept:
            self.intercept_ = float(r.coef[0]) if r.coef else 0.0
            self.coef_ = [float(v) for v in r.coef[1:]]
        else:
            self.intercept_ = 0.0
            self.coef_ = [float(v) for v in r.coef]
        return self

    def predict(self, X: Sequence[Sequence[float]]) -> List[float]:
        if self.fit_result_ is None:
            raise RuntimeError("estimator is not fitted")
        return self.fit_result_.predict_mean(X, offset=self.offset, exposure=self.exposure)


__all__ = [
    "NextStatLinearRegression",
    "NextStatLogisticRegression",
    "NextStatPoissonRegressor",
]

