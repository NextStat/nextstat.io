"""NextStat Python package.

The compiled extension is exposed as `nextstat._core` (built via PyO3/maturin).
"""

from __future__ import annotations

try:
    from ._core import (  # type: ignore
        __version__,
        fit,
        hypotest,
        HistFactoryModel,
        MaximumLikelihoodEstimator,
        FitResult,
        from_pyhf,
        profile_scan,
        upper_limit,
    )
except ImportError:  # pragma: no cover
    __version__ = "0.0.0"
    fit = None  # type: ignore
    hypotest = None  # type: ignore
    HistFactoryModel = None  # type: ignore
    MaximumLikelihoodEstimator = None  # type: ignore
    FitResult = None  # type: ignore
    from_pyhf = None  # type: ignore
    profile_scan = None  # type: ignore
    upper_limit = None  # type: ignore

# Aliases used throughout docs/plans.
PyModel = HistFactoryModel
PyFitResult = FitResult

__all__ = [
    "__version__",
    "fit",
    "hypotest",
    "HistFactoryModel",
    "MaximumLikelihoodEstimator",
    "FitResult",
    "from_pyhf",
    "profile_scan",
    "upper_limit",
    "PyModel",
    "PyFitResult",
]
