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
hypotest_toys = _get("hypotest_toys")
sample = _get("sample")
from_pyhf = _get("from_pyhf")
apply_patchset = _get("apply_patchset")
profile_scan = _get("profile_scan")
upper_limit = _get("upper_limit")
upper_limits = _get("upper_limits")
upper_limits_root = _get("upper_limits_root")

HistFactoryModel = _get("HistFactoryModel")
MaximumLikelihoodEstimator = _get("MaximumLikelihoodEstimator")
FitResult = _get("FitResult")
Posterior = _get("Posterior")

GaussianMeanModel = _get("GaussianMeanModel")
FunnelModel = _get("FunnelModel")
StdNormalModel = _get("StdNormalModel")
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
fit_toys_batch = _get("fit_toys_batch")
fit_toys_batch_gpu = _get("fit_toys_batch_gpu")
set_eval_mode = _get("set_eval_mode")
get_eval_mode = _get("get_eval_mode")
has_accelerate = _get("has_accelerate")
has_cuda = _get("has_cuda")
has_metal = _get("has_metal")
DifferentiableSession = _get("DifferentiableSession")
ProfiledDifferentiableSession = _get("ProfiledDifferentiableSession")
asimov_data = _get("asimov_data")
poisson_toys = _get("poisson_toys")
ranking = _get("ranking")
rk4_linear = _get("rk4_linear")
read_root_histogram = _get("read_root_histogram")
workspace_audit = _get("workspace_audit")
cls_curve = _get("cls_curve")
profile_curve = _get("profile_curve")
kalman_filter = _get("kalman_filter")
kalman_smooth = _get("kalman_smooth")
kalman_em = _get("kalman_em")
kalman_forecast = _get("kalman_forecast")
kalman_simulate = _get("kalman_simulate")
from_arrow_ipc = _get("from_arrow_ipc")
from_parquet = _get("from_parquet")
to_arrow_yields_ipc = _get("to_arrow_yields_ipc")
to_arrow_params_ipc = _get("to_arrow_params_ipc")

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
from . import econometrics as econometrics
from . import mlops as mlops  # noqa: E402
from . import interpret as interpret  # noqa: E402
from . import tools as tools  # noqa: E402
from . import distill as distill  # noqa: E402
from . import remote as remote  # noqa: E402

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


def histfactory_bin_edges_by_channel(xml_path: str | Path) -> dict[str, list[float]]:
    """Return `{channel_name: bin_edges}` from a HistFactory export (`combination.xml`)."""
    if _core is None:  # pragma: no cover
        raise ImportError("nextstat._core is not available (native extension not built/installed).")

    xml_path = Path(xml_path).resolve()
    return _core.histfactory_bin_edges_by_channel(str(xml_path))


def from_arrow(table, *, poi: str = "mu", observations: dict | None = None) -> HistFactoryModel:
    """Create a HistFactoryModel from a PyArrow Table or RecordBatch.

    The table must have columns: ``channel`` (Utf8), ``sample`` (Utf8),
    ``yields`` (List<Float64>), optionally ``stat_error`` (List<Float64>).

    Args:
        table: ``pyarrow.Table`` or ``pyarrow.RecordBatch``.
        poi: parameter of interest name (default ``"mu"``).
        observations: optional ``{channel_name: [obs_counts]}``.
            If ``None``, Asimov data (sum of yields) is used.

    Returns:
        :class:`HistFactoryModel`
    """
    import pyarrow as pa

    if isinstance(table, pa.RecordBatch):
        table = pa.Table.from_batches([table])

    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, table.schema)
    for batch in table.to_batches():
        writer.write_batch(batch)
    writer.close()
    ipc_bytes = sink.getvalue().to_pybytes()

    return from_arrow_ipc(ipc_bytes, poi=poi, observations=observations)


def to_arrow(model: HistFactoryModel, *, params: list[float] | None = None, what: str = "yields"):
    """Export model data as a PyArrow Table.

    Args:
        model: a :class:`HistFactoryModel`.
        params: parameter values (default: model init).
        what: ``"yields"`` (expected yields per channel) or ``"params"``
            (parameter metadata).

    Returns:
        ``pyarrow.Table``
    """
    import pyarrow as pa

    if what == "yields":
        ipc_bytes = to_arrow_yields_ipc(model, params=params)
    elif what == "params":
        ipc_bytes = to_arrow_params_ipc(model, params=params)
    else:
        raise ValueError(f"Unknown export type: {what!r}. Use 'yields' or 'params'.")

    reader = pa.ipc.open_stream(ipc_bytes)
    return reader.read_all()


__all__ = [
    "__version__",
    "fit",
    "map_fit",
    "fit_batch",
    "hypotest",
    "hypotest_toys",
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
    "FunnelModel",
    "StdNormalModel",
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
    "fit_toys_batch",
    "fit_toys_batch_gpu",
    "set_eval_mode",
    "get_eval_mode",
    "has_accelerate",
    "has_cuda",
    "DifferentiableSession",
    "ProfiledDifferentiableSession",
    "asimov_data",
    "poisson_toys",
    "ranking",
    "rk4_linear",
    "has_metal",
    "read_root_histogram",
    "workspace_audit",
    "cls_curve",
    "profile_curve",
    "kalman_filter",
    "kalman_smooth",
    "kalman_em",
    "kalman_forecast",
    "kalman_simulate",
    "from_arrow_ipc",
    "from_parquet",
    "to_arrow_yields_ipc",
    "to_arrow_params_ipc",
    "from_arrow",
    "to_arrow",
    "from_pyhf",
    "apply_patchset",
    "from_histfactory_xml",
    "histfactory_bin_edges_by_channel",
    "profile_scan",
    "upper_limit",
    "upper_limits",
    "upper_limits_root",
    "mlops",
    "interpret",
    "tools",
    "distill",
    "remote",
    "PyModel",
    "PyFitResult",
]
