"""High-level regression/GLM surface (Phase 6).

This package provides small, Python-first wrappers around the compiled models in
`nextstat._core`, plus common convenience methods like predict / predict_proba.
"""

from .linear import FittedLinearRegression, fit as fit_linear, fit_ols  # noqa: F401
from .logistic import FittedLogisticRegression, fit as fit_logistic  # noqa: F401
from .poisson import FittedPoissonRegression, fit as fit_poisson  # noqa: F401

__all__ = [
    "FittedLinearRegression",
    "FittedLogisticRegression",
    "FittedPoissonRegression",
    "fit_linear",
    "fit_logistic",
    "fit_poisson",
    "fit_ols",
]

