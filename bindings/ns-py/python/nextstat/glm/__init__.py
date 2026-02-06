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

from .linear import LinearFit, fit as fit_linear  # noqa: F401
from .logistic import LogisticFit, fit as fit_logistic  # noqa: F401
from .poisson import PoissonFit, fit as fit_poisson  # noqa: F401

__all__ = [
    "linear",
    "logistic",
    "poisson",
    "LinearFit",
    "LogisticFit",
    "PoissonFit",
    "fit_linear",
    "fit_logistic",
    "fit_poisson",
]
