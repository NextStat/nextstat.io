"""High-level regression/GLM surface (Phase 6).

Primary API:
- `nextstat.glm.linear.fit(...)`
- `nextstat.glm.logistic.fit(...)`
- `nextstat.glm.poisson.fit(...)`

Back-compat re-exports:
- `nextstat.glm.fit_linear(...)`, `nextstat.glm.fit_logistic(...)`, `nextstat.glm.fit_poisson(...)`
"""

from __future__ import annotations

from . import linear as linear
from . import logistic as logistic
from . import poisson as poisson
from . import negbin as negbin
from . import metrics as metrics
from . import cv as cv

from .linear import LinearFit, fit as fit_linear  # noqa: F401
from .logistic import LogisticFit, fit as fit_logistic  # noqa: F401
from .poisson import PoissonFit, fit as fit_poisson  # noqa: F401
from .negbin import NegativeBinomialFit, fit as fit_negbin  # noqa: F401
from .metrics import log_loss, mean_poisson_deviance, poisson_deviance, rmse  # noqa: F401
from .cv import CvResult, cross_val_score, kfold_indices  # noqa: F401

__all__ = [
    "linear",
    "logistic",
    "poisson",
    "negbin",
    "metrics",
    "cv",
    "LinearFit",
    "LogisticFit",
    "PoissonFit",
    "NegativeBinomialFit",
    "rmse",
    "log_loss",
    "poisson_deviance",
    "mean_poisson_deviance",
    "CvResult",
    "kfold_indices",
    "cross_val_score",
    "fit_linear",
    "fit_logistic",
    "fit_poisson",
    "fit_negbin",
]
