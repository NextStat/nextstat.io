"""Ordinal models (Phase 9.C).

Keep imports lazy to avoid circular-import issues when importing `nextstat`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["ordered_logit", "ordered_probit"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
