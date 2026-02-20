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
_sample_nuts = _get("sample")
_sample_mams = _get("sample_mams")
_sample_laps = _get("sample_laps")  # None if not built with cuda
RawCudaModel = _get("RawCudaModel")  # None if not built with cuda
from_pyhf = _get("from_pyhf")
apply_patchset = _get("apply_patchset")
profile_scan = _get("profile_scan")
upper_limit = _get("upper_limit")
upper_limits = _get("upper_limits")

HistFactoryModel = _get("HistFactoryModel")
UnbinnedModel = _get("UnbinnedModel")
HybridModel = _get("HybridModel")
MaximumLikelihoodEstimator = _get("MaximumLikelihoodEstimator")
FitResult = _get("FitResult")
Posterior = _get("Posterior")

GaussianMeanModel = _get("GaussianMeanModel")
FunnelModel = _get("FunnelModel")
FunnelNcpModel = _get("FunnelNcpModel")
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
IntervalCensoredWeibullModel = _get("IntervalCensoredWeibullModel")
IntervalCensoredExponentialModel = _get("IntervalCensoredExponentialModel")
IntervalCensoredLogNormalModel = _get("IntervalCensoredLogNormalModel")
OneCompartmentOralPkModel = _get("OneCompartmentOralPkModel")
OneCompartmentOralPkNlmeModel = _get("OneCompartmentOralPkNlmeModel")
TwoCompartmentIvPkModel = _get("TwoCompartmentIvPkModel")
TwoCompartmentOralPkModel = _get("TwoCompartmentOralPkModel")
nlme_foce = _get("nlme_foce")
nlme_saem = _get("nlme_saem")
pk_vpc = _get("pk_vpc")
pk_gof = _get("pk_gof")
read_nonmem = _get("read_nonmem")
GammaRegressionModel = _get("GammaRegressionModel")
TweedieRegressionModel = _get("TweedieRegressionModel")
GevModel = _get("GevModel")
GpdModel = _get("GpdModel")
EightSchoolsModel = _get("EightSchoolsModel")
ols_fit = _get("ols_fit")
fit_toys = _get("fit_toys")
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
rk4_linear = _get("rk4_linear")
read_root_histogram = _get("read_root_histogram")
workspace_audit = _get("workspace_audit")
cls_curve = _get("cls_curve")
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
fault_tree_mc = _get("fault_tree_mc")
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
    "viz_render",
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

def sample(
    model,
    *,
    method: str = "nuts",
    return_idata: bool = False,
    out: str | Path | None = None,
    out_format: str = "json",
    **kwargs,
):
    """Unified sampling interface for all MCMC methods.

    Args:
        model: A NextStat model instance, or a string model name for LAPS
            (``"std_normal"``, ``"eight_schools"``, ``"neal_funnel"``, ``"glm_logistic"``).
        method: Sampling algorithm — ``"nuts"`` (default), ``"mams"``, or ``"laps"`` (GPU).
        return_idata: If ``True``, return an ArviZ ``InferenceData`` object instead
            of a raw dict. Requires ``arviz`` and ``numpy``.
        out: Optional path to save results (JSON or NetCDF).
        out_format: ``"json"`` (default) or ``"netcdf"`` — used when *out* is set.
        **kwargs: Method-specific parameters forwarded to the underlying sampler.

    Common kwargs (all methods):
        n_chains (int): Number of parallel chains. Default: 4 (NUTS/MAMS), 4096 (LAPS).
        n_warmup (int): Warmup iterations. Default: 500.
        n_samples (int): Sampling iterations. Default: 1000 (NUTS/MAMS), 2000 (LAPS).
        seed (int): Random seed. Default: 42.
        target_accept (float): Target acceptance rate. Default: 0.8 (NUTS), 0.9 (MAMS/LAPS).

    NUTS-specific kwargs:
        max_treedepth (int): Max tree depth. Default: 10.
        init_strategy (str): ``"random"``, ``"mle"``, or ``"pathfinder"``. Default: ``"random"``.
        metric (str): ``"diagonal"``, ``"dense"``, or ``"auto"``. Default: ``"diagonal"``.
        stepsize_jitter (float): Step size jitter. Default: 0.0.

    MAMS-specific kwargs:
        init_strategy (str): ``"random"``, ``"mle"``, or ``"pathfinder"``. Default: ``"random"``.
        metric (str): ``"diagonal"`` or ``"dense"``. Default: ``"diagonal"``.
        init_step_size (float): Initial step size. Default: 0.0 (auto).
        init_l (float): Initial trajectory length. Default: 0.0 (auto).
        max_leapfrog (int): Max leapfrog steps. Default: 1024.
        diagonal_precond (bool): Use diagonal preconditioning. Default: True.

    LAPS-specific kwargs (GPU, requires CUDA build):
        model_data (dict): Model-specific data (e.g. ``{"y": [...], "sigma": [...]}``)
        init_step_size (float): Initial step size. Default: 0.0 (auto).
        init_l (float): Initial trajectory length. Default: 0.0 (auto).
        max_leapfrog (int): Max leapfrog steps. Default: 8192.
        report_chains (int): Number of chains retained for diagnostics. Default: 256.

    Returns:
        dict or InferenceData: Sampling results. When ``return_idata=False`` (default),
        returns a dict with keys ``"posterior"``, ``"sample_stats"``,
        ``"diagnostics"``, ``"param_names"``, ``"n_chains"``, ``"n_warmup"``,
        ``"n_samples"``. When ``return_idata=True``, returns an ArviZ ``InferenceData``.

    Examples:
        >>> import nextstat as ns
        >>> model = ns.EightSchoolsModel([28,8,-3,7,-1,1,18,12], [15,10,16,11,9,11,10,18])
        >>> result = ns.sample(model, method="nuts", n_samples=2000)
        >>> idata = ns.sample(model, method="mams", n_samples=2000, return_idata=True)
    """
    if _core is None:
        raise ImportError("nextstat._core is not available (native extension not built/installed).")

    if method == "nuts":
        if _sample_nuts is None:
            raise RuntimeError("NUTS sampler not available in this build.")
        raw = _sample_nuts(model, **kwargs)
    elif method == "mams":
        if _sample_mams is None:
            raise RuntimeError("MAMS sampler not available in this build.")
        raw = _sample_mams(model, **kwargs)
    elif method == "laps":
        if _sample_laps is None:
            _has = has_cuda() if has_cuda is not None else False
            if not _has:
                raise RuntimeError(
                    "LAPS requires CUDA. Build with `maturin develop --features cuda` "
                    "or use method='mams' for CPU sampling."
                )
            raise RuntimeError("LAPS sampler not available in this build.")
        raw = _sample_laps(model, **kwargs)
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Use 'nuts', 'mams', or 'laps'."
        )

    if not return_idata:
        if out is not None:
            import json as _json
            Path(out).write_text(_json.dumps(raw, indent=2, sort_keys=True) + "\n")
        return raw

    from .bayes import to_inferencedata, save as _save_idata

    idata = to_inferencedata(raw)
    if out is not None:
        _save_idata(idata, out, format=out_format)
    return idata


# Explicit per-method aliases for direct access.
sample_nuts = _sample_nuts
sample_mams = _sample_mams
sample_laps = _sample_laps

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
    # Inference functions (unified API)
    "fit",
    "map_fit",
    "fit_batch",
    "fit_toys",
    "hypotest",
    "hypotest_toys",
    "profile_scan",
    "upper_limit",
    "upper_limits",
    "ranking",
    "sample",
    "cls_curve",
    # Model classes
    "HistFactoryModel",
    "UnbinnedModel",
    "HybridModel",
    "GaussianMeanModel",
    "FunnelModel",
    "FunnelNcpModel",
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
    "IntervalCensoredWeibullModel",
    "IntervalCensoredExponentialModel",
    "IntervalCensoredLogNormalModel",
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
    "RawCudaModel",
    "FlowPdf",
    "DcrSurrogate",
    # Utility functions
    "asimov_data",
    "poisson_toys",
    "rk4_linear",
    "ols_fit",
    "set_eval_mode",
    "set_threads",
    "get_eval_mode",
    "has_accelerate",
    "has_cuda",
    "has_metal",
    "read_root_histogram",
    "workspace_audit",
    # Time series
    "kalman_filter",
    "kalman_smooth",
    "kalman_em",
    "kalman_forecast",
    "kalman_simulate",
    # Meta-analysis & actuarial
    "meta_fixed",
    "meta_random",
    "chain_ladder",
    "mack_chain_ladder",
    # Survival
    "kaplan_meier",
    "log_rank_test",
    # Churn
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
    # Arrow/Parquet I/O
    "from_arrow_ipc",
    "from_parquet",
    "from_parquet_with_modifiers",
    "to_arrow_yields_ipc",
    "to_arrow_params_ipc",
    "from_arrow",
    "to_arrow",
    # Factory functions
    "from_pyhf",
    "apply_patchset",
    "from_histfactory_xml",
    "histfactory_bin_edges_by_channel",
    # Submodules
    "bayes",
    "viz",
    "viz_render",
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
    "mlops",
    "interpret",
    "tools",
    "distill",
    "remote",
    "arrow_io",
    # Back-compat aliases
    "PyModel",
    "PyFitResult",
]
