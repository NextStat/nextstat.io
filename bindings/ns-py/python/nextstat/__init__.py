"""NextStat Python package.

The compiled extension is exposed as `nextstat._core` (built via PyO3/maturin).
"""

from __future__ import annotations

try:
    from ._core import (  # type: ignore
        __version__,
        fit,
        hypotest,
        sample,
        HistFactoryModel,
        MaximumLikelihoodEstimator,
        FitResult,
        from_pyhf,
        profile_scan,
        upper_limit,
        upper_limits,
        upper_limits_root,
    )
except ImportError:  # pragma: no cover
    __version__ = "0.0.0"
    fit = None  # type: ignore
    hypotest = None  # type: ignore
    sample = None  # type: ignore
    HistFactoryModel = None  # type: ignore
    MaximumLikelihoodEstimator = None  # type: ignore
    FitResult = None  # type: ignore
    from_pyhf = None  # type: ignore
    profile_scan = None  # type: ignore
    upper_limit = None  # type: ignore
    upper_limits = None  # type: ignore
    upper_limits_root = None  # type: ignore

# PyO3 renamed class: `from ... import` fails but attribute access works.
import nextstat._core as _core  # type: ignore  # noqa: E402

GaussianMeanModel = getattr(_core, "GaussianMeanModel", None)  # type: ignore
LinearRegressionModel = getattr(_core, "LinearRegressionModel", None)  # type: ignore
LogisticRegressionModel = getattr(_core, "LogisticRegressionModel", None)  # type: ignore
PoissonRegressionModel = getattr(_core, "PoissonRegressionModel", None)  # type: ignore
ComposedGlmModel = getattr(_core, "ComposedGlmModel", None)  # type: ignore
ols_fit = getattr(_core, "ols_fit", None)  # type: ignore

# Optional convenience wrappers (use optional deps like arviz).
from . import bayes as bayes  # noqa: E402
from . import viz as viz  # noqa: E402
from . import data as data  # noqa: E402

# Back-compat alias: make the sampler intent explicit without breaking `sample`.
sample_nuts = sample

# Aliases used throughout docs/plans.
PyModel = HistFactoryModel
PyFitResult = FitResult

__all__ = [
    "__version__",
    "fit",
    "hypotest",
    "sample",
    "sample_nuts",
    "bayes",
    "viz",
    "data",
    "HistFactoryModel",
    "GaussianMeanModel",
    "LinearRegressionModel",
    "LogisticRegressionModel",
    "PoissonRegressionModel",
    "ComposedGlmModel",
    "MaximumLikelihoodEstimator",
    "FitResult",
    "ols_fit",
    "from_pyhf",
    "profile_scan",
    "upper_limit",
    "upper_limits",
    "upper_limits_root",
    "PyModel",
    "PyFitResult",
]
