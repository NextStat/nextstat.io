"""NextStat Python package.

The compiled extension is exposed as `nextstat._core` (built via PyO3/maturin).
"""

from __future__ import annotations

import importlib
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
unbinned_hypotest = _get("unbinned_hypotest")
unbinned_hypotest_toys = _get("unbinned_hypotest_toys")
unbinned_profile_scan = _get("unbinned_profile_scan")
unbinned_fit_toys = _get("unbinned_fit_toys")

HistFactoryModel = _get("HistFactoryModel")
UnbinnedModel = _get("UnbinnedModel")
HybridModel = _get("HybridModel")
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
GammaRegressionModel = _get("GammaRegressionModel")
TweedieRegressionModel = _get("TweedieRegressionModel")
GevModel = _get("GevModel")
GpdModel = _get("GpdModel")
EightSchoolsModel = _get("EightSchoolsModel")
ols_fit = _get("ols_fit")
fit_toys = _get("fit_toys")
fit_toys_batch = _get("fit_toys_batch")
fit_toys_batch_gpu = _get("fit_toys_batch_gpu")
set_eval_mode = _get("set_eval_mode")
set_threads = _get("set_threads")
get_eval_mode = _get("get_eval_mode")
has_accelerate = _get("has_accelerate")
has_cuda = _get("has_cuda")
has_metal = _get("has_metal")
DifferentiableSession = _get("DifferentiableSession")
ProfiledDifferentiableSession = _get("ProfiledDifferentiableSession")
MetalProfiledDifferentiableSession = _get("MetalProfiledDifferentiableSession")
asimov_data = _get("asimov_data")
poisson_toys = _get("poisson_toys")
ranking = _get("ranking")
unbinned_ranking = _get("unbinned_ranking")
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
meta_fixed = _get("meta_fixed")
meta_random = _get("meta_random")
chain_ladder = _get("chain_ladder")
mack_chain_ladder = _get("mack_chain_ladder")
kaplan_meier = _get("kaplan_meier")
log_rank_test = _get("log_rank_test")
rosenbaum_bounds = _get("rosenbaum_bounds")
churn_generate_data = _get("churn_generate_data")
churn_retention = _get("churn_retention")
churn_risk_model = _get("churn_risk_model")
churn_uplift = _get("churn_uplift")
churn_diagnostics = _get("churn_diagnostics")
churn_cohort_matrix = _get("churn_cohort_matrix")
churn_compare = _get("churn_compare")
churn_uplift_survival = _get("churn_uplift_survival")
churn_bootstrap_hr = _get("churn_bootstrap_hr")
churn_ingest = _get("churn_ingest")
from_arrow_ipc = _get("from_arrow_ipc")
from_parquet = _get("from_parquet")
from_parquet_with_modifiers = _get("from_parquet_with_modifiers")
to_arrow_yields_ipc = _get("to_arrow_yields_ipc")
to_arrow_params_ipc = _get("to_arrow_params_ipc")

# Optional native classes (feature-gated in Rust; present only when built with those features).
FlowPdf = _get("FlowPdf")
DcrSurrogate = _get("DcrSurrogate")

_LAZY_SUBMODULES = {
    # Optional convenience wrappers (may require optional deps like numpy/arviz/matplotlib).
    "unbinned",
    "arrow_io",
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
    "mlops",
    "interpret",
    "tools",
    "distill",
    "remote",
    "audit",
    "report",
    "validation_report",
    "analysis",
    "trex_config",
    "torch",
    "gym",
    "volatility",
}


def __getattr__(name: str):
    # Keep `import nextstat` lightweight: submodules are importable but loaded on demand.
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
    "_core",
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
    "unbinned",
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
    "volatility",
    "torch",
    "gym",
    "HistFactoryModel",
    "UnbinnedModel",
    "HybridModel",
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
    "GammaRegressionModel",
    "TweedieRegressionModel",
    "GevModel",
    "GpdModel",
    "EightSchoolsModel",
    "MaximumLikelihoodEstimator",
    "FitResult",
    "Posterior",
    "asimov_data",
    "poisson_toys",
    "ranking",
    "unbinned_ranking",
    "rk4_linear",
    "ols_fit",
    "fit_toys",
    "unbinned_fit_toys",
    "fit_toys_batch",
    "fit_toys_batch_gpu",
    "set_eval_mode",
    "set_threads",
    "get_eval_mode",
    "has_accelerate",
    "has_cuda",
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
    "meta_fixed",
    "meta_random",
    "chain_ladder",
    "mack_chain_ladder",
    "kaplan_meier",
    "log_rank_test",
    "churn_generate_data",
    "churn_retention",
    "churn_risk_model",
    "churn_uplift",
    "churn_diagnostics",
    "churn_cohort_matrix",
    "churn_compare",
    "churn_uplift_survival",
    "churn_bootstrap_hr",
    "churn_ingest",
    "from_arrow_ipc",
    "from_parquet",
    "from_parquet_with_modifiers",
    "to_arrow_yields_ipc",
    "to_arrow_params_ipc",
    "FlowPdf",
    "DcrSurrogate",
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
    "unbinned_hypotest",
    "unbinned_hypotest_toys",
    "unbinned_profile_scan",
    "mlops",
    "interpret",
    "tools",
    "distill",
    "remote",
    "arrow_io",
    "PyModel",
    "PyFitResult",
]
