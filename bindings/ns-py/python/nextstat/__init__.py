"""NextStat Python package.

The compiled extension is exposed as `nextstat._core` (built via PyO3/maturin).
"""

from __future__ import annotations

from pathlib import Path

try:
    import nextstat._core as _core  # type: ignore  # noqa: E402
except ImportError:  # pragma: no cover
    _core = None  # type: ignore


def _get(name: str, default=None):
    if _core is None:
        return default
    return getattr(_core, name, default)


__version__ = _get("__version__", "0.0.0")

fit = _get("fit")
map_fit = _get("map_fit")
fit_batch = _get("fit_batch")
hypotest = _get("hypotest")
sample = _get("sample")
from_pyhf = _get("from_pyhf")
profile_scan = _get("profile_scan")
upper_limit = _get("upper_limit")
upper_limits = _get("upper_limits")
upper_limits_root = _get("upper_limits_root")

HistFactoryModel = _get("HistFactoryModel")
MaximumLikelihoodEstimator = _get("MaximumLikelihoodEstimator")
FitResult = _get("FitResult")
Posterior = _get("Posterior")

GaussianMeanModel = _get("GaussianMeanModel")
LinearRegressionModel = _get("LinearRegressionModel")
LogisticRegressionModel = _get("LogisticRegressionModel")
PoissonRegressionModel = _get("PoissonRegressionModel")
OrderedLogitModel = _get("OrderedLogitModel")
OrderedProbitModel = _get("OrderedProbitModel")
NegativeBinomialRegressionModel = _get("NegativeBinomialRegressionModel")
ComposedGlmModel = _get("ComposedGlmModel")
LmmMarginalModel = _get("LmmMarginalModel")
KalmanModel = _get("KalmanModel")
ExponentialSurvivalModel = _get("ExponentialSurvivalModel")
WeibullSurvivalModel = _get("WeibullSurvivalModel")
LogNormalAftModel = _get("LogNormalAftModel")
CoxPhModel = _get("CoxPhModel")
OneCompartmentOralPkModel = _get("OneCompartmentOralPkModel")
OneCompartmentOralPkNlmeModel = _get("OneCompartmentOralPkNlmeModel")
ols_fit = _get("ols_fit")
fit_toys = _get("fit_toys")
asimov_data = _get("asimov_data")
poisson_toys = _get("poisson_toys")
ranking = _get("ranking")
rk4_linear = _get("rk4_linear")

# Optional convenience wrappers (use optional deps like arviz).
from . import bayes as bayes  # noqa: E402
from . import viz as viz  # noqa: E402
from . import data as data  # noqa: E402
from . import glm as glm  # noqa: E402
from . import timeseries as timeseries  # noqa: E402
from . import hier as hier  # noqa: E402
from . import ppc as ppc  # noqa: E402
from . import survival as survival  # noqa: E402
from . import ordinal as ordinal  # noqa: E402
from . import causal as causal  # noqa: E402
from . import missing as missing  # noqa: E402
from . import ode as ode  # noqa: E402
from . import formula as formula  # noqa: E402
from . import summary as summary  # noqa: E402
from . import robust as robust  # noqa: E402
from . import sklearn as sklearn  # noqa: E402
from . import panel as panel  # noqa: E402
from . import econometrics as econometrics  # noqa: E402

# Back-compat alias: make the sampler intent explicit without breaking `sample`.
sample_nuts = sample

# Aliases used throughout docs/plans.
PyModel = HistFactoryModel
PyFitResult = FitResult

def from_histfactory_xml(xml_path: str | Path) -> HistFactoryModel:
    """Create a `HistFactoryModel` from a HistFactory export (`combination.xml` + referenced ROOT hists)."""
    if _core is None:  # pragma: no cover
        raise ImportError("nextstat._core is not available (native extension not built/installed).")

    xml_path = Path(xml_path).resolve()
    return _core.from_histfactory(str(xml_path))

__all__ = [
    "__version__",
    "fit",
    "map_fit",
    "fit_batch",
    "hypotest",
    "sample",
    "sample_nuts",
    "bayes",
    "viz",
    "data",
    "glm",
    "timeseries",
    "hier",
    "ppc",
    "survival",
    "ordinal",
    "causal",
    "missing",
    "ode",
    "formula",
    "summary",
    "robust",
    "sklearn",
    "panel",
    "econometrics",
    "HistFactoryModel",
    "GaussianMeanModel",
    "LinearRegressionModel",
    "LogisticRegressionModel",
    "PoissonRegressionModel",
    "OrderedLogitModel",
    "OrderedProbitModel",
    "NegativeBinomialRegressionModel",
    "ComposedGlmModel",
    "LmmMarginalModel",
    "KalmanModel",
    "ExponentialSurvivalModel",
    "WeibullSurvivalModel",
    "LogNormalAftModel",
    "CoxPhModel",
    "OneCompartmentOralPkModel",
    "OneCompartmentOralPkNlmeModel",
    "MaximumLikelihoodEstimator",
    "FitResult",
    "Posterior",
    "ols_fit",
    "fit_toys",
    "asimov_data",
    "poisson_toys",
    "ranking",
    "rk4_linear",
    "from_pyhf",
    "from_histfactory_xml",
    "profile_scan",
    "upper_limit",
    "upper_limits",
    "upper_limits_root",
    "PyModel",
    "PyFitResult",
    "ExponentialSurvivalModel",
    "WeibullSurvivalModel",
    "LogNormalAftModel",
    "CoxPhModel",
    "OneCompartmentOralPkModel",
    "OneCompartmentOralPkNlmeModel",
]
