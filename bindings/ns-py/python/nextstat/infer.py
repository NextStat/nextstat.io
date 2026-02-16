"""Frequentist inference helpers (Phase 3.1).

This module intentionally mirrors the pyhf surface where practical:

- `hypotest(poi_test, model, ...)` -> CLs
- `upper_limit(model, ...)` -> observed CLs upper limit
- `profile_scan(model, mu_values, ...)` -> q_mu scan
"""

from __future__ import annotations

from ._core import hypotest, hypotest_toys, profile_scan, upper_limit, upper_limits  # type: ignore

__all__ = [
    "hypotest",
    "hypotest_toys",
    "profile_scan",
    "upper_limit",
    "upper_limits",
]
