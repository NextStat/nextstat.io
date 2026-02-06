"""Causal convenience helpers (Phase 9.C).

These helpers are intentionally pragmatic and minimal:
- propensity score estimation (via logistic regression)
- weighting helpers (IPW)
- balance/overlap diagnostics

They do **not** make causal identification claims. Correct causal inference depends on
study design and assumptions (e.g., exchangeability/unconfoundedness, positivity, and
correct model specification).

Keep imports lazy to avoid circular-import issues when importing `nextstat`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["propensity"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

