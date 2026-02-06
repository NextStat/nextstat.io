"""Generalized Linear Models (Phase 6).

Public API:
- `nextstat.glm.linear.fit(...)`
- `nextstat.glm.logistic.fit(...)`
- `nextstat.glm.poisson.fit(...)`
"""

from __future__ import annotations

from . import linear as linear
from . import logistic as logistic
from . import poisson as poisson

__all__ = ["linear", "logistic", "poisson"]

