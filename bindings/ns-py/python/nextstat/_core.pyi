"""Type stubs for the compiled extension module `nextstat._core`.

These stubs are intentionally lightweight and cover the public API exposed by
the PyO3 module in `bindings/ns-py/src/lib.rs`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, Union, overload

__version__: str


# ---------------------------------------------------------------------------
# Structured return types for sampling results
# ---------------------------------------------------------------------------

class QualitySummary(TypedDict):
    status: str
    enabled: bool
    warnings: List[str]
    failures: List[str]
    total_draws: int
    max_r_hat: float
    min_ess_bulk: float
    min_ess_tail: float
    min_ebfmi: float

class SampleStats(TypedDict):
    diverging: List[List[bool]]
    tree_depth: List[List[int]]
    accept_prob: List[List[float]]
    energy: List[List[float]]
    step_size: List[float]
    n_leapfrog: List[List[int]]

class Diagnostics(TypedDict):
    r_hat: Dict[str, float]
    ess_bulk: Dict[str, float]
    ess_tail: Dict[str, float]
    divergence_rate: float
    max_treedepth_rate: float
    ebfmi: List[float]
    quality: QualitySummary

class SamplerResult(TypedDict):
    posterior: Dict[str, List[List[float]]]
    sample_stats: SampleStats
    diagnostics: Diagnostics
    param_names: List[str]
    n_chains: int
    n_warmup: int
    n_samples: int

# ---------------------------------------------------------------------------
# Structured return types for inference results
# ---------------------------------------------------------------------------

class RankingEntry(TypedDict):
    name: str
    delta_mu_up: float
    delta_mu_down: float
    pull: float
    constraint: float

class HypotestResult(TypedDict):
    cls: float
    clsb: float
    clb: float

class HypotestToysMetaResult(TypedDict):
    mu_test: float
    cls: float
    clsb: float
    clb: float
    q_obs: float
    mu_hat: float
    n_toys_b: int
    n_toys_sb: int
    n_error_b: int
    n_error_sb: int
    n_nonconverged_b: int
    n_nonconverged_sb: int
    expected: Optional[List[float]]

class ProfileScanPoint(TypedDict, total=False):
    mu: float
    q_mu: float
    nll_mu: float
    converged: bool
    n_iter: int
    params: Optional[List[float]]
    message: Optional[str]
    n_fev: Optional[int]
    n_gev: Optional[int]
    initial_cost: Optional[float]
    grad_l2: Optional[float]

class ProfileScanResult(TypedDict):
    poi_index: int
    mu_hat: float
    nll_hat: float
    points: List[ProfileScanPoint]

class ProfileCurveResult(TypedDict):
    poi_index: int
    mu_hat: float
    nll_hat: float
    mu_values: List[float]
    q_mu_values: List[float]
    twice_delta_nll: List[float]
    points: List[ProfileScanPoint]

class UnbinnedHypotestResult(TypedDict):
    input_schema_version: str
    poi_index: int
    mu_test: float
    mu_hat: float
    nll_hat: float
    nll_mu: float
    q_mu: float
    q0: Optional[float]
    nll_mu0: Optional[float]
    converged_hat: bool
    converged_mu: bool
    n_iter_hat: int
    n_iter_mu: int

class WorkspaceAuditResult(TypedDict):
    n_channels: int
    n_samples: int
    n_parameters: int
    n_bins_total: int
    modifiers: Dict[str, int]
    unsupported: List[str]

# ---------------------------------------------------------------------------
# Structured return types for econometrics
# ---------------------------------------------------------------------------

class PanelFeResult(TypedDict):
    coefficients: List[float]
    se_ols: List[float]
    se_cluster: Optional[List[float]]
    r_squared_within: float
    n_obs: int
    n_entities: int
    n_time_periods: int
    n_regressors: int
    df_absorbed: int
    rss: float

class DidResult(TypedDict):
    att: float
    se: float
    se_cluster: Optional[float]
    t_stat: float
    mean_treated_post: float
    mean_treated_pre: float
    mean_control_post: float
    mean_control_pre: float
    n_obs: int

class EventStudyResult(TypedDict):
    coefficients: List[float]
    se_cluster: List[float]
    lag_labels: List[int]
    n_obs: int

class Iv2slsResult(TypedDict):
    coefficients: List[float]
    se: List[float]
    se_cluster: Optional[List[float]]
    names: List[str]
    n_obs: int

# ---------------------------------------------------------------------------
# Structured return types for time series
# ---------------------------------------------------------------------------

class KalmanFilterResult(TypedDict):
    filtered_states: List[List[float]]
    filtered_covs: List[List[List[float]]]
    log_likelihood: float

class KalmanSmoothResult(TypedDict):
    smoothed_states: List[List[float]]
    smoothed_covs: List[List[List[float]]]
    log_likelihood: float

class KalmanEmResult(TypedDict):
    model: Dict[str, Any]
    log_likelihood: float
    n_iter: int
    converged: bool

class KalmanForecastResult(TypedDict):
    states: List[List[float]]
    observations: List[List[float]]
    intervals: Optional[Dict[str, Any]]

class KalmanSimulateResult(TypedDict):
    states: List[List[float]]
    observations: List[List[float]]

# ---------------------------------------------------------------------------
# Structured return types for meta-analysis & actuarial
# ---------------------------------------------------------------------------

class MetaAnalysisResult(TypedDict):
    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    weights: List[float]
    z: float
    p_value: float
    q_statistic: Optional[float]
    i_squared: Optional[float]
    tau_squared: Optional[float]

class ChainLadderResult(TypedDict):
    development_factors: List[float]
    full_triangle: List[List[float]]
    reserves: List[float]
    total_reserve: float

class MackChainLadderResult(TypedDict):
    development_factors: List[float]
    full_triangle: List[List[float]]
    reserves: List[float]
    total_reserve: float
    se_reserves: List[float]
    se_total: float
    ci_lower: List[float]
    ci_upper: List[float]

# ---------------------------------------------------------------------------
# Structured return types for survival
# ---------------------------------------------------------------------------

class KaplanMeierResult(TypedDict):
    times: List[float]
    survival: List[float]
    ci_lower: List[float]
    ci_upper: List[float]
    at_risk: List[int]
    events: List[int]

class LogRankTestResult(TypedDict):
    statistic: float
    p_value: float
    df: int

# ---------------------------------------------------------------------------
# Structured return types for CLs curve
# ---------------------------------------------------------------------------

class ClsCurvePoint(TypedDict):
    mu: float
    cls: float
    expected: List[float]

class ClsCurveResult(TypedDict):
    alpha: float
    nsigma_order: List[float]
    obs_limit: float
    exp_limits: List[float]
    mu_values: List[float]
    cls_obs: List[float]
    cls_exp: List[List[float]]
    points: List[ClsCurvePoint]

# ---------------------------------------------------------------------------
# Structured return types for causal inference
# ---------------------------------------------------------------------------

class AipwAteResult(TypedDict):
    ate: float
    se: float
    ci_lower: float
    ci_upper: float
    n_trimmed: int

class RosenbaumBoundsResult(TypedDict):
    gammas: List[float]
    p_values_upper: List[float]
    p_values_lower: List[float]

# `HistFactoryModel` accepts Python sequences (list/tuple/array('d')) and also
# buffer-protocol objects for performance. Type stubs stay conservative and
# describe the common supported surfaces without using `Any`.
ParamsLike = Union[Sequence[float], memoryview]


class HistFactoryModel:
    @staticmethod
    def from_workspace(json_str: str) -> HistFactoryModel: ...
    @staticmethod
    def from_xml(xml_path: str) -> HistFactoryModel: ...
    @staticmethod
    def from_hs3(
        json_str: str,
        analysis: Optional[str] = ...,
        param_points: Optional[str] = ...,
    ) -> HistFactoryModel: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: ParamsLike) -> float: ...
    def grad_nll(self, params: ParamsLike) -> List[float]: ...
    def expected_data(self, params: ParamsLike, *, include_auxdata: bool = ...) -> List[float]: ...
    def with_observed_main(self, observed_main: List[float]) -> HistFactoryModel: ...
    def set_sample_nominal(self, *, channel: str, sample: str, nominal: ParamsLike) -> None: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...
    def poi_index(self) -> Optional[int]: ...
    def observed_main_by_channel(self) -> List[Dict[str, Any]]: ...
    def expected_main_by_channel_sample(self, params: ParamsLike) -> List[Dict[str, Any]]: ...


class GammaRegressionModel:
    def __init__(self, x: List[List[float]], y: List[float], *, include_intercept: bool = ...) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class TweedieRegressionModel:
    def __init__(
        self,
        x: List[List[float]],
        y: List[float],
        *,
        p: float = ...,
        include_intercept: bool = ...,
    ) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def power(self) -> float: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class GevModel:
    def __init__(self, data: List[float]) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...

    @staticmethod
    def return_level(params: List[float], return_period: float) -> float: ...


class GpdModel:
    def __init__(self, exceedances: List[float]) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...

    @staticmethod
    def quantile(params: List[float], p: float) -> float: ...


class EightSchoolsModel:
    def __init__(
        self,
        y: List[float],
        sigma: List[float],
        *,
        prior_mu_sigma: float = ...,
        prior_tau_scale: float = ...,
    ) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class UnbinnedModel:
    @staticmethod
    def from_config(path: str) -> UnbinnedModel: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def schema_version(self) -> str: ...
    def nll(self, params: ParamsLike) -> float: ...
    def grad_nll(self, params: ParamsLike) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...
    def poi_index(self) -> Optional[int]: ...
    def with_fixed_param(self, param_idx: int, value: float) -> UnbinnedModel: ...


class HybridModel:
    @staticmethod
    def from_models(
        binned: HistFactoryModel,
        unbinned: UnbinnedModel,
        poi_from: str = ...,
    ) -> HybridModel: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: ParamsLike) -> float: ...
    def grad_nll(self, params: ParamsLike) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...
    def poi_index(self) -> Optional[int]: ...
    def n_shared(self) -> int: ...
    def with_fixed_param(self, param_idx: int, value: float) -> HybridModel: ...


class GaussianMeanModel:
    def __init__(self, y: List[float], sigma: float) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class FunnelModel:
    def __init__(self, dim: int = 2) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class FunnelNcpModel:
    """Neal's funnel in non-centered parameterization (NCP).

    Parameters are (v, z_1, ..., z_{d-1}) where v ~ N(0,9), z_i ~ N(0,1).
    Original parameters: x_i = exp(v/2) * z_i.
    """

    def __init__(self, dim: int = 10) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class StdNormalModel:
    def __init__(self, dim: int = ...) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class LinearRegressionModel:
    def __init__(self, x: List[List[float]], y: List[float], *, include_intercept: bool = ...) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class LogisticRegressionModel:
    def __init__(self, x: List[List[float]], y: List[int], *, include_intercept: bool = ...) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class OrderedLogitModel:
    def __init__(self, x: List[List[float]], y: List[int], *, n_levels: int) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class OrderedProbitModel:
    def __init__(self, x: List[List[float]], y: List[int], *, n_levels: int) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class PoissonRegressionModel:
    def __init__(
        self,
        x: List[List[float]],
        y: List[int],
        *,
        include_intercept: bool = ...,
        offset: Optional[List[float]] = ...,
    ) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class ExponentialSurvivalModel:
    def __init__(self, times: List[float], events: List[bool]) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class WeibullSurvivalModel:
    def __init__(self, times: List[float], events: List[bool]) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class LogNormalAftModel:
    def __init__(self, times: List[float], events: List[bool]) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class CoxPhModel:
    def __init__(
        self,
        times: List[float],
        events: List[bool],
        x: List[List[float]],
        *,
        ties: Literal["efron", "breslow"] = ...,
    ) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class IntervalCensoredWeibullModel:
    def __init__(
        self,
        time_lower: List[float],
        time_upper: List[float],
        censor_type: List[str],
    ) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class IntervalCensoredWeibullAftModel:
    def __init__(
        self,
        time_lower: List[float],
        time_upper: List[float],
        censor_type: List[str],
        covariates: List[List[float]],
    ) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class IntervalCensoredExponentialModel:
    def __init__(
        self,
        time_lower: List[float],
        time_upper: List[float],
        censor_type: List[str],
    ) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class IntervalCensoredLogNormalModel:
    def __init__(
        self,
        time_lower: List[float],
        time_upper: List[float],
        censor_type: List[str],
    ) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class OneCompartmentOralPkModel:
    def __init__(
        self,
        times: List[float],
        y: List[float],
        *,
        dose: float,
        bioavailability: float = ...,
        sigma: float = ...,
        lloq: Optional[float] = ...,
        lloq_policy: Literal["ignore", "replace_half", "censored"] = ...,
    ) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...
    def predict(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class OneCompartmentOralPkNlmeModel:
    def __init__(
        self,
        times: List[float],
        y: List[float],
        subject_idx: List[int],
        n_subjects: int,
        *,
        dose: float,
        bioavailability: float = ...,
        sigma: float = ...,
        lloq: Optional[float] = ...,
        lloq_policy: Literal["ignore", "replace_half", "censored"] = ...,
    ) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class TwoCompartmentIvPkModel:
    def __init__(
        self,
        times: List[float],
        y: List[float],
        *,
        dose: float,
        error_model: Literal["additive", "proportional", "combined"] = ...,
        sigma: float = ...,
        sigma_add: Optional[float] = ...,
        lloq: Optional[float] = ...,
        lloq_policy: Literal["ignore", "replace_half", "censored"] = ...,
    ) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...
    def predict(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class TwoCompartmentOralPkModel:
    def __init__(
        self,
        times: List[float],
        y: List[float],
        *,
        dose: float,
        bioavailability: float = ...,
        error_model: Literal["additive", "proportional", "combined"] = ...,
        sigma: float = ...,
        sigma_add: Optional[float] = ...,
        lloq: Optional[float] = ...,
        lloq_policy: Literal["ignore", "replace_half", "censored"] = ...,
    ) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...
    def predict(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class NegativeBinomialRegressionModel:
    def __init__(
        self,
        x: List[List[float]],
        y: List[int],
        *,
        include_intercept: bool = ...,
        offset: Optional[List[float]] = ...,
    ) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class ComposedGlmModel:
    @staticmethod
    def linear_regression(
        x: List[List[float]],
        y: List[float],
        *,
        include_intercept: bool = ...,
        group_idx: Optional[List[int]] = ...,
        n_groups: Optional[int] = ...,
        coef_prior_mu: float = ...,
        coef_prior_sigma: float = ...,
        penalize_intercept: bool = ...,
        obs_sigma_prior_m: Optional[float] = ...,
        obs_sigma_prior_s: Optional[float] = ...,
        random_intercept_non_centered: bool = ...,
        random_slope_feature_idx: Optional[int] = ...,
        random_slope_non_centered: bool = ...,
        correlated_feature_idx: Optional[int] = ...,
        lkj_eta: float = ...,
    ) -> ComposedGlmModel: ...

    @staticmethod
    def logistic_regression(
        x: List[List[float]],
        y: List[int],
        *,
        include_intercept: bool = ...,
        group_idx: Optional[List[int]] = ...,
        n_groups: Optional[int] = ...,
        coef_prior_mu: float = ...,
        coef_prior_sigma: float = ...,
        penalize_intercept: bool = ...,
        random_intercept_non_centered: bool = ...,
        random_slope_feature_idx: Optional[int] = ...,
        random_slope_non_centered: bool = ...,
        correlated_feature_idx: Optional[int] = ...,
        lkj_eta: float = ...,
    ) -> ComposedGlmModel: ...

    @staticmethod
    def poisson_regression(
        x: List[List[float]],
        y: List[int],
        *,
        include_intercept: bool = ...,
        offset: Optional[List[float]] = ...,
        group_idx: Optional[List[int]] = ...,
        n_groups: Optional[int] = ...,
        coef_prior_mu: float = ...,
        coef_prior_sigma: float = ...,
        penalize_intercept: bool = ...,
        random_intercept_non_centered: bool = ...,
        random_slope_feature_idx: Optional[int] = ...,
        random_slope_non_centered: bool = ...,
        correlated_feature_idx: Optional[int] = ...,
        lkj_eta: float = ...,
    ) -> ComposedGlmModel: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...
    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class LmmMarginalModel:
    def __init__(
        self,
        x: List[List[float]],
        y: List[float],
        *,
        include_intercept: bool = ...,
        group_idx: List[int],
        n_groups: Optional[int] = ...,
        random_slope_feature_idx: Optional[int] = ...,
    ) -> None: ...

    def n_params(self) -> int: ...
    def dim(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class KalmanModel:
    def __init__(
        self,
        f: List[List[float]],
        q: List[List[float]],
        h: List[List[float]],
        r: List[List[float]],
        m0: List[float],
        p0: List[List[float]],
    ) -> None: ...

    def n_state(self) -> int: ...
    def n_obs(self) -> int: ...


class FitResult:
    parameters: List[float]
    uncertainties: List[float]
    nll: float
    converged: bool
    n_evaluations: int
    n_iter: int
    n_fev: int
    n_gev: int
    termination_reason: str
    final_grad_norm: float
    initial_nll: float
    n_active_bounds: int
    edm: float
    warnings: List[str]

    @property
    def bestfit(self) -> List[float]: ...

    @property
    def twice_nll(self) -> float: ...

    @property
    def success(self) -> bool: ...


class FitMinimumResult:
    parameters: List[float]
    nll: float
    converged: bool
    n_iter: int
    n_fev: int
    n_gev: int
    message: str
    initial_nll: float
    final_gradient: Optional[List[float]]
    edm: float

    @property
    def bestfit(self) -> List[float]: ...

    @property
    def twice_nll(self) -> float: ...

    @property
    def success(self) -> bool: ...


class Posterior:
    def __init__(
        self,
        model: Union[
            HistFactoryModel,
            GaussianMeanModel,
            LinearRegressionModel,
            LogisticRegressionModel,
            OrderedLogitModel,
            OrderedProbitModel,
            PoissonRegressionModel,
            NegativeBinomialRegressionModel,
            GammaRegressionModel,
            TweedieRegressionModel,
            ComposedGlmModel,
            LmmMarginalModel,
            ExponentialSurvivalModel,
            WeibullSurvivalModel,
            LogNormalAftModel,
            CoxPhModel,
            IntervalCensoredWeibullModel,
            IntervalCensoredExponentialModel,
            IntervalCensoredLogNormalModel,
            OneCompartmentOralPkModel,
            OneCompartmentOralPkNlmeModel,
            TwoCompartmentIvPkModel,
            TwoCompartmentOralPkModel,
            GevModel,
            GpdModel,
        ],
    ) -> None: ...

    def dim(self) -> int: ...
    def parameter_names(self) -> List[str]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...
    def suggested_init(self) -> List[float]: ...

    def clear_priors(self) -> None: ...
    def set_prior_flat(self, name: str) -> None: ...
    def set_prior_normal(self, name: str, center: float, width: float) -> None: ...
    def priors(self) -> Dict[str, Dict[str, Any]]: ...

    def logpdf(self, theta: ParamsLike) -> float: ...
    def grad(self, theta: ParamsLike) -> List[float]: ...

    def to_unconstrained(self, theta: ParamsLike) -> List[float]: ...
    def to_constrained(self, z: ParamsLike) -> List[float]: ...
    def logpdf_unconstrained(self, z: ParamsLike) -> float: ...
    def grad_unconstrained(self, z: ParamsLike) -> List[float]: ...


class MaximumLikelihoodEstimator:
    def __init__(self, *, max_iter: int = ..., tol: float = ..., m: int = ..., smooth_bounds: bool = ...) -> None: ...
    @overload
    def fit(self, model: HistFactoryModel, *, data: Optional[List[float]] = ..., init_pars: Optional[List[float]] = ...) -> FitResult: ...
    @overload
    def fit(
        self,
        model: Union[
            GaussianMeanModel,
            LinearRegressionModel,
            LogisticRegressionModel,
            OrderedLogitModel,
            OrderedProbitModel,
            PoissonRegressionModel,
            NegativeBinomialRegressionModel,
            ComposedGlmModel,
            LmmMarginalModel,
            ExponentialSurvivalModel,
            WeibullSurvivalModel,
            LogNormalAftModel,
            CoxPhModel,
            IntervalCensoredWeibullModel,
            IntervalCensoredExponentialModel,
            IntervalCensoredLogNormalModel,
            OneCompartmentOralPkModel,
            OneCompartmentOralPkNlmeModel,
            TwoCompartmentIvPkModel,
            TwoCompartmentOralPkModel,
            GammaRegressionModel,
            TweedieRegressionModel,
            GevModel,
            GpdModel,
        ],
        *,
        data: Literal[None] = ...,
        init_pars: Optional[List[float]] = ...,
    ) -> FitResult: ...
    @overload
    def fit_minimum(
        self,
        model: HistFactoryModel,
        *,
        data: Optional[List[float]] = ...,
        init_pars: Optional[List[float]] = ...,
        bounds: Optional[List[Tuple[float, float]]] = ...,
    ) -> FitMinimumResult: ...
    @overload
    def fit_minimum(
        self,
        model: Union[
            GaussianMeanModel,
            LinearRegressionModel,
            LogisticRegressionModel,
            OrderedLogitModel,
            OrderedProbitModel,
            PoissonRegressionModel,
            NegativeBinomialRegressionModel,
            ComposedGlmModel,
            LmmMarginalModel,
            ExponentialSurvivalModel,
            WeibullSurvivalModel,
            LogNormalAftModel,
            CoxPhModel,
            IntervalCensoredWeibullModel,
            IntervalCensoredExponentialModel,
            IntervalCensoredLogNormalModel,
            OneCompartmentOralPkModel,
            OneCompartmentOralPkNlmeModel,
            TwoCompartmentIvPkModel,
            TwoCompartmentOralPkModel,
        ],
        *,
        data: Literal[None] = ...,
        init_pars: Optional[List[float]] = ...,
        bounds: Literal[None] = ...,
    ) -> FitMinimumResult: ...
    @overload
    def fit_batch(
        self,
        models_or_model: List[HistFactoryModel],
        datasets: Literal[None] = ...,
    ) -> List[FitResult]: ...
    @overload
    def fit_batch(
        self,
        models_or_model: List[GaussianMeanModel],
        datasets: Literal[None] = ...,
    ) -> List[FitResult]: ...
    @overload
    def fit_batch(
        self,
        models_or_model: List[LinearRegressionModel],
        datasets: Literal[None] = ...,
    ) -> List[FitResult]: ...
    @overload
    def fit_batch(
        self,
        models_or_model: List[LogisticRegressionModel],
        datasets: Literal[None] = ...,
    ) -> List[FitResult]: ...
    @overload
    def fit_batch(
        self,
        models_or_model: List[OrderedLogitModel],
        datasets: Literal[None] = ...,
    ) -> List[FitResult]: ...
    @overload
    def fit_batch(
        self,
        models_or_model: List[OrderedProbitModel],
        datasets: Literal[None] = ...,
    ) -> List[FitResult]: ...
    @overload
    def fit_batch(
        self,
        models_or_model: List[PoissonRegressionModel],
        datasets: Literal[None] = ...,
    ) -> List[FitResult]: ...
    @overload
    def fit_batch(
        self,
        models_or_model: List[NegativeBinomialRegressionModel],
        datasets: Literal[None] = ...,
    ) -> List[FitResult]: ...
    @overload
    def fit_batch(
        self,
        models_or_model: List[ComposedGlmModel],
        datasets: Literal[None] = ...,
    ) -> List[FitResult]: ...
    @overload
    def fit_batch(
        self,
        models_or_model: List[LmmMarginalModel],
        datasets: Literal[None] = ...,
    ) -> List[FitResult]: ...
    @overload
    def fit_batch(
        self,
        models_or_model: List[ExponentialSurvivalModel],
        datasets: Literal[None] = ...,
    ) -> List[FitResult]: ...
    @overload
    def fit_batch(
        self,
        models_or_model: List[WeibullSurvivalModel],
        datasets: Literal[None] = ...,
    ) -> List[FitResult]: ...
    @overload
    def fit_batch(
        self,
        models_or_model: List[LogNormalAftModel],
        datasets: Literal[None] = ...,
    ) -> List[FitResult]: ...
    @overload
    def fit_batch(
        self,
        models_or_model: List[CoxPhModel],
        datasets: Literal[None] = ...,
    ) -> List[FitResult]: ...
    @overload
    def fit_batch(
        self,
        models_or_model: List[OneCompartmentOralPkModel],
        datasets: Literal[None] = ...,
    ) -> List[FitResult]: ...
    @overload
    def fit_batch(
        self,
        models_or_model: List[OneCompartmentOralPkNlmeModel],
        datasets: Literal[None] = ...,
    ) -> List[FitResult]: ...
    @overload
    def fit_batch(
        self,
        models_or_model: List[TwoCompartmentIvPkModel],
        datasets: Literal[None] = ...,
    ) -> List[FitResult]: ...
    @overload
    def fit_batch(
        self,
        models_or_model: List[TwoCompartmentOralPkModel],
        datasets: Literal[None] = ...,
    ) -> List[FitResult]: ...
    @overload
    def fit_batch(
        self,
        models_or_model: HistFactoryModel,
        datasets: List[List[float]],
    ) -> List[FitResult]: ...
    def fit_toys(
        self,
        model: HistFactoryModel,
        params: List[float],
        *,
        n_toys: int = ...,
        seed: int = ...,
    ) -> List[FitResult]: ...
    def ranking(
        self,
        model: HistFactoryModel,
    ) -> List[RankingEntry]: ...
    def q0_like_loss_and_grad_nominal(
        self,
        model: HistFactoryModel,
        *,
        channel: str,
        sample: str,
        nominal: List[float],
    ) -> Tuple[float, List[float]]: ...
    def qmu_like_loss_and_grad_nominal(
        self,
        model: HistFactoryModel,
        *,
        mu_test: float,
        channel: str,
        sample: str,
        nominal: List[float],
    ) -> Tuple[float, List[float]]: ...


def from_pyhf(json_str: str) -> HistFactoryModel: ...
def from_histfactory(xml_path: str) -> HistFactoryModel: ...
def histfactory_bin_edges_by_channel(xml_path: str) -> Dict[str, List[float]]: ...
def apply_patchset(
    workspace_json: str,
    patchset_json: str,
    *,
    patch_name: Optional[str] = ...,
) -> str: ...
def workspace_combine(
    workspace_json_1: str,
    workspace_json_2: str,
    *,
    join: str = "none",
) -> str: ...
def workspace_prune(
    workspace_json: str,
    *,
    channels: List[str] = ...,
    samples: List[str] = ...,
    modifiers: List[str] = ...,
    measurements: List[str] = ...,
) -> str: ...
def workspace_rename(
    workspace_json: str,
    *,
    channels: Optional[Dict[str, str]] = ...,
    samples: Optional[Dict[str, str]] = ...,
    modifiers: Optional[Dict[str, str]] = ...,
    measurements: Optional[Dict[str, str]] = ...,
) -> str: ...
def workspace_sorted(workspace_json: str) -> str: ...
def workspace_digest(workspace_json: str) -> str: ...
def workspace_to_xml(
    workspace_json: str,
    output_prefix: str = "output",
) -> List[Tuple[str, str]]: ...
def simplemodel_uncorrelated(
    signal: List[float],
    bkg: List[float],
    bkg_uncertainty: List[float],
) -> str: ...
def simplemodel_correlated(
    signal: List[float],
    bkg: List[float],
    bkg_up: List[float],
    bkg_down: List[float],
) -> str: ...
def meta_fixed(
    estimates: List[float],
    standard_errors: List[float],
    *,
    labels: Optional[List[str]] = ...,
    conf_level: float = ...,
) -> MetaAnalysisResult: ...
def meta_random(
    estimates: List[float],
    standard_errors: List[float],
    *,
    labels: Optional[List[str]] = ...,
    conf_level: float = ...,
) -> MetaAnalysisResult: ...
def chain_ladder(
    triangle: List[List[float]],
) -> ChainLadderResult: ...
def mack_chain_ladder(
    triangle: List[List[float]],
    *,
    conf_level: float = ...,
) -> MackChainLadderResult: ...
def hypotest_toys(
    poi_test: float,
    model: Union[HistFactoryModel, UnbinnedModel],
    *,
    n_toys: int = ...,
    seed: int = ...,
    expected_set: bool = ...,
    data: Optional[List[float]] = ...,
    return_tail_probs: bool = ...,
    return_meta: bool = ...,
) -> Union[float, Tuple[float, List[float]], HypotestToysMetaResult]: ...

def read_root_histogram(root_path: str, hist_path: str) -> Dict[str, Any]: ...
@overload
def fit(
    model: HistFactoryModel,
    *,
    data: Optional[List[float]] = ...,
    init_pars: Optional[List[float]] = ...,
    device: str = "cpu",
) -> FitResult: ...
@overload
def fit(
    model: Union[
        UnbinnedModel,
        GaussianMeanModel,
        LinearRegressionModel,
        LogisticRegressionModel,
        OrderedLogitModel,
        OrderedProbitModel,
        PoissonRegressionModel,
        NegativeBinomialRegressionModel,
        GammaRegressionModel,
        TweedieRegressionModel,
        ComposedGlmModel,
        LmmMarginalModel,
        ExponentialSurvivalModel,
        WeibullSurvivalModel,
        LogNormalAftModel,
        CoxPhModel,
        OneCompartmentOralPkModel,
        OneCompartmentOralPkNlmeModel,
        GevModel,
        GpdModel,
    ],
    *,
    data: Literal[None] = ...,
    init_pars: Optional[List[float]] = ...,
    device: str = "cpu",
) -> FitResult: ...

def map_fit(posterior: Posterior) -> FitResult: ...
@overload
def fit_batch(models_or_model: List[HistFactoryModel], datasets: Literal[None] = ...) -> List[FitResult]: ...
@overload
def fit_batch(models_or_model: List[UnbinnedModel], datasets: Literal[None] = ...) -> List[FitResult]: ...
@overload
def fit_batch(models_or_model: List[GaussianMeanModel], datasets: Literal[None] = ...) -> List[FitResult]: ...
@overload
def fit_batch(models_or_model: List[LinearRegressionModel], datasets: Literal[None] = ...) -> List[FitResult]: ...
@overload
def fit_batch(models_or_model: List[LogisticRegressionModel], datasets: Literal[None] = ...) -> List[FitResult]: ...
@overload
def fit_batch(models_or_model: List[OrderedLogitModel], datasets: Literal[None] = ...) -> List[FitResult]: ...
@overload
def fit_batch(models_or_model: List[OrderedProbitModel], datasets: Literal[None] = ...) -> List[FitResult]: ...
@overload
def fit_batch(models_or_model: List[PoissonRegressionModel], datasets: Literal[None] = ...) -> List[FitResult]: ...
@overload
def fit_batch(models_or_model: List[NegativeBinomialRegressionModel], datasets: Literal[None] = ...) -> List[FitResult]: ...
@overload
def fit_batch(models_or_model: List[ComposedGlmModel], datasets: Literal[None] = ...) -> List[FitResult]: ...
@overload
def fit_batch(models_or_model: List[LmmMarginalModel], datasets: Literal[None] = ...) -> List[FitResult]: ...
@overload
def fit_batch(models_or_model: List[ExponentialSurvivalModel], datasets: Literal[None] = ...) -> List[FitResult]: ...
@overload
def fit_batch(models_or_model: List[WeibullSurvivalModel], datasets: Literal[None] = ...) -> List[FitResult]: ...
@overload
def fit_batch(models_or_model: List[LogNormalAftModel], datasets: Literal[None] = ...) -> List[FitResult]: ...
@overload
def fit_batch(models_or_model: List[CoxPhModel], datasets: Literal[None] = ...) -> List[FitResult]: ...
@overload
def fit_batch(
    models_or_model: List[OneCompartmentOralPkModel],
    datasets: Literal[None] = ...,
) -> List[FitResult]: ...
@overload
def fit_batch(
    models_or_model: List[OneCompartmentOralPkNlmeModel],
    datasets: Literal[None] = ...,
) -> List[FitResult]: ...
@overload
def fit_batch(
    models_or_model: List[TwoCompartmentIvPkModel],
    datasets: Literal[None] = ...,
) -> List[FitResult]: ...
@overload
def fit_batch(
    models_or_model: List[TwoCompartmentOralPkModel],
    datasets: Literal[None] = ...,
) -> List[FitResult]: ...
@overload
def fit_batch(models_or_model: HistFactoryModel, datasets: List[List[float]]) -> List[FitResult]: ...
def fit_toys(
    model: Union[HistFactoryModel, UnbinnedModel],
    params: List[float],
    *,
    n_toys: int = ...,
    seed: int = ...,
    device: str = "cpu",
    batch: bool = ...,
    compute_hessian: bool = ...,
    max_retries: int = ...,
    max_iter: int = ...,
    init_params: Optional[List[float]] = ...,
) -> List[FitResult]: ...
def set_eval_mode(mode: str) -> None: ...
def set_threads(threads: int) -> bool: ...
def get_eval_mode() -> str: ...
def has_accelerate() -> bool: ...
def has_cuda() -> bool: ...
def has_metal() -> bool: ...
def workspace_audit(json_str: str) -> WorkspaceAuditResult: ...
def asimov_data(model: HistFactoryModel, params: List[float]) -> List[float]: ...
def poisson_toys(
    model: HistFactoryModel,
    params: List[float],
    *,
    n_toys: int = ...,
    seed: int = ...,
) -> List[List[float]]: ...
def ranking(
    model: Union[HistFactoryModel, UnbinnedModel],
    *,
    device: str = "cpu",
) -> List[RankingEntry]: ...


def rk4_linear(
    a: List[List[float]],
    y0: List[float],
    t0: float,
    t1: float,
    dt: float,
    *,
    max_steps: int = ...,
) -> Dict[str, Any]: ...
def ols_fit(x: List[List[float]], y: List[float], *, include_intercept: bool = ...) -> List[float]: ...


def hypotest(
    poi_test: float,
    model: Union[HistFactoryModel, UnbinnedModel],
    *,
    data: Optional[List[float]] = ...,
    return_tail_probs: bool = ...,
) -> Union[float, Tuple[float, List[float]], UnbinnedHypotestResult]: ...


def profile_scan(
    model: Union[HistFactoryModel, UnbinnedModel],
    mu_values: List[float],
    *,
    data: Optional[List[float]] = ...,
    device: str = "cpu",
    return_params: bool = ...,
    return_curve: bool = ...,
) -> Union[ProfileScanResult, ProfileCurveResult]: ...


def upper_limit(
    model: HistFactoryModel,
    *,
    method: str = "bisect",
    alpha: float = ...,
    lo: float = ...,
    hi: Optional[float] = ...,
    rtol: float = ...,
    max_iter: int = ...,
    data: Optional[List[float]] = ...,
) -> Union[float, Tuple[float, List[float]]]: ...


def upper_limits(
    model: HistFactoryModel,
    scan: List[float],
    *,
    alpha: float = ...,
    data: Optional[List[float]] = ...,
) -> Tuple[float, List[float]]: ...




@overload
def sample(
    model: HistFactoryModel,
    *,
    n_chains: int = ...,
    n_warmup: int = ...,
    n_samples: int = ...,
    seed: int = ...,
    max_treedepth: int = ...,
    target_accept: float = ...,
    init_strategy: str = ...,
    init_jitter: float = ...,
    init_jitter_rel: Optional[float] = ...,
    init_overdispersed_rel: Optional[float] = ...,
    data: Optional[List[float]] = ...,
) -> SamplerResult: ...
@overload
def sample(
    model: Posterior,
    *,
    n_chains: int = ...,
    n_warmup: int = ...,
    n_samples: int = ...,
    seed: int = ...,
    max_treedepth: int = ...,
    target_accept: float = ...,
    init_strategy: str = ...,
    init_jitter: float = ...,
    init_jitter_rel: Optional[float] = ...,
    init_overdispersed_rel: Optional[float] = ...,
    data: Literal[None] = ...,
) -> SamplerResult: ...
@overload
def sample(
    model: Union[
        GaussianMeanModel,
        FunnelModel,
        FunnelNcpModel,
        StdNormalModel,
        LinearRegressionModel,
        LogisticRegressionModel,
        OrderedLogitModel,
        OrderedProbitModel,
        PoissonRegressionModel,
        NegativeBinomialRegressionModel,
        GammaRegressionModel,
        TweedieRegressionModel,
        ComposedGlmModel,
        LmmMarginalModel,
        ExponentialSurvivalModel,
        WeibullSurvivalModel,
        LogNormalAftModel,
        CoxPhModel,
        OneCompartmentOralPkModel,
        OneCompartmentOralPkNlmeModel,
        GevModel,
        GpdModel,
        EightSchoolsModel,
    ],
    *,
    n_chains: int = ...,
    n_warmup: int = ...,
    n_samples: int = ...,
    seed: int = ...,
    max_treedepth: int = ...,
    target_accept: float = ...,
    init_strategy: str = ...,
    init_jitter: float = ...,
    init_jitter_rel: Optional[float] = ...,
    init_overdispersed_rel: Optional[float] = ...,
    data: Literal[None] = ...,
) -> SamplerResult: ...


@overload
def sample_mams(
    model: HistFactoryModel,
    *,
    n_chains: int = ...,
    n_warmup: int = ...,
    n_samples: int = ...,
    seed: int = ...,
    target_accept: float = ...,
    init_strategy: str = ...,
    metric: str = ...,
    init_step_size: float = ...,
    init_l: float = ...,
    max_leapfrog: int = ...,
    diagonal_precond: bool = ...,
    eps_jitter: float = ...,
    data: Optional[List[float]] = ...,
) -> SamplerResult: ...
@overload
def sample_mams(
    model: Posterior,
    *,
    n_chains: int = ...,
    n_warmup: int = ...,
    n_samples: int = ...,
    seed: int = ...,
    target_accept: float = ...,
    init_strategy: str = ...,
    metric: str = ...,
    init_step_size: float = ...,
    init_l: float = ...,
    max_leapfrog: int = ...,
    diagonal_precond: bool = ...,
    eps_jitter: float = ...,
    data: Literal[None] = ...,
) -> SamplerResult: ...
@overload
def sample_mams(
    model: Union[
        GaussianMeanModel,
        FunnelModel,
        FunnelNcpModel,
        StdNormalModel,
        LinearRegressionModel,
        LogisticRegressionModel,
        OrderedLogitModel,
        OrderedProbitModel,
        PoissonRegressionModel,
        NegativeBinomialRegressionModel,
        GammaRegressionModel,
        TweedieRegressionModel,
        ComposedGlmModel,
        LmmMarginalModel,
        ExponentialSurvivalModel,
        WeibullSurvivalModel,
        LogNormalAftModel,
        CoxPhModel,
        OneCompartmentOralPkModel,
        OneCompartmentOralPkNlmeModel,
        GevModel,
        GpdModel,
        EightSchoolsModel,
    ],
    *,
    n_chains: int = ...,
    n_warmup: int = ...,
    n_samples: int = ...,
    seed: int = ...,
    target_accept: float = ...,
    init_strategy: str = ...,
    metric: str = ...,
    init_step_size: float = ...,
    init_l: float = ...,
    max_leapfrog: int = ...,
    diagonal_precond: bool = ...,
    eps_jitter: float = ...,
    data: Literal[None] = ...,
) -> SamplerResult: ...


class RawCudaModel:
    """User-defined CUDA model for GPU LAPS sampling via NVRTC JIT compilation.

    The ``cuda_src`` must define ``user_nll()`` and ``user_grad()`` device functions.
    """
    dim: int
    cuda_src: str
    def __init__(
        self,
        dim: int,
        cuda_src: str,
        *,
        data: Optional[List[float]] = ...,
        param_names: Optional[List[str]] = ...,
    ) -> None: ...

def sample_laps(
    model: Union[str, RawCudaModel],
    *,
    model_data: Optional[Dict[str, Any]] = ...,
    n_chains: int = ...,
    n_warmup: int = ...,
    n_samples: int = ...,
    seed: int = ...,
    target_accept: float = ...,
    init_step_size: float = ...,
    init_l: float = ...,
    max_leapfrog: int = ...,
    device_ids: Optional[List[int]] = ...,
    sync_interval: int = ...,
    welford_chains: int = ...,
    batch_size: int = ...,
    fused_transitions: int = ...,
    report_chains: int = ...,
    diagonal_precond: bool = ...,
    divergence_threshold: float = ...,
) -> SamplerResult: ...


def cls_curve(
    model: HistFactoryModel,
    scan: List[float],
    *,
    alpha: float = ...,
    data: Optional[List[float]] = ...,
) -> ClsCurveResult: ...




def kalman_filter(
    model: KalmanModel,
    ys: List[List[Optional[float]]],
) -> KalmanFilterResult: ...


def kalman_smooth(
    model: KalmanModel,
    ys: List[List[Optional[float]]],
) -> KalmanSmoothResult: ...


def kalman_em(
    model: KalmanModel,
    ys: List[List[Optional[float]]],
    *,
    max_iter: int = ...,
    tol: float = ...,
    estimate_q: bool = ...,
    estimate_r: bool = ...,
    estimate_f: bool = ...,
    estimate_h: bool = ...,
    min_diag: float = ...,
) -> KalmanEmResult: ...


def kalman_forecast(
    model: KalmanModel,
    ys: List[List[Optional[float]]],
    *,
    steps: int = ...,
    alpha: Optional[float] = ...,
) -> KalmanForecastResult: ...


def kalman_simulate(
    model: KalmanModel,
    *,
    t_max: int,
    seed: int = ...,
    init: str = ...,
    x0: Optional[List[float]] = ...,
) -> KalmanSimulateResult: ...


def garch11_fit(
    ys: List[float],
    *,
    max_iter: int = ...,
    tol: float = ...,
    alpha_beta_max: float = ...,
    min_var: float = ...,
) -> Dict[str, Any]: ...


def sv_logchi2_fit(
    ys: List[float],
    *,
    max_iter: int = ...,
    tol: float = ...,
    log_eps: float = ...,
) -> Dict[str, Any]: ...


def panel_fe(
    y: List[float],
    x: List[float],
    entity_ids: List[int],
    p: int,
    *,
    time_ids: Optional[List[int]] = ...,
    cluster_ids: Optional[List[int]] = ...,
) -> PanelFeResult: ...


def did(
    y: List[float],
    treat: List[int],
    post: List[int],
    cluster_ids: List[int],
) -> DidResult: ...


def event_study(
    y: List[float],
    entity_ids: List[int],
    time_ids: List[int],
    relative_time: List[int],
    min_lag: int,
    max_lag: int,
    reference_period: int,
    cluster_ids: List[int],
) -> EventStudyResult: ...


def iv_2sls(
    y: List[float],
    x_exog: List[float],
    k_exog: int,
    x_endog: List[float],
    k_endog: int,
    z: List[float],
    m: int,
    *,
    exog_names: Optional[List[str]] = ...,
    endog_names: Optional[List[str]] = ...,
    cluster_ids: Optional[List[int]] = ...,
) -> Dict[str, Any]: ...


def aipw_ate(
    y: List[float],
    treat: List[int],
    propensity: List[float],
    mu1: List[float],
    mu0: List[float],
    *,
    trim: float = ...,
) -> AipwAteResult: ...


def rosenbaum_bounds(
    y_treated: List[float],
    y_control: List[float],
    gammas: List[float],
) -> RosenbaumBoundsResult: ...


def nlme_foce(
    times: List[float],
    y: List[float],
    subject_idx: List[int],
    n_subjects: int,
    *,
    dose: float,
    bioavailability: float = ...,
    error_model: Literal["additive", "proportional", "combined"] = ...,
    sigma: float = ...,
    sigma_add: Optional[float] = ...,
    theta_init: List[float],
    omega_init: List[float],
    max_outer_iter: int = ...,
    max_inner_iter: int = ...,
    tol: float = ...,
    interaction: bool = ...,
) -> Dict[str, Any]: ...


def nlme_saem(
    times: List[float],
    y: List[float],
    subject_idx: List[int],
    n_subjects: int,
    *,
    dose: float,
    bioavailability: float = ...,
    error_model: Literal["additive", "proportional", "combined"] = ...,
    sigma: float = ...,
    sigma_add: Optional[float] = ...,
    theta_init: List[float],
    omega_init: List[float],
    n_burn: int = ...,
    n_iter: int = ...,
    n_chains: int = ...,
    seed: int = ...,
    tol: float = ...,
) -> Dict[str, Any]: ...


def pk_vpc(
    times: List[float],
    y: List[float],
    subject_idx: List[int],
    n_subjects: int,
    *,
    dose: float,
    bioavailability: float = ...,
    theta: List[float],
    omega_matrix: List[List[float]],
    error_model: Literal["additive", "proportional", "combined"] = ...,
    sigma: float = ...,
    sigma_add: Optional[float] = ...,
    n_sim: int = ...,
    quantiles: Optional[List[float]] = ...,
    n_bins: int = ...,
    seed: int = ...,
    pi_level: float = ...,
) -> Dict[str, Any]: ...


def pk_gof(
    times: List[float],
    y: List[float],
    subject_idx: List[int],
    *,
    dose: float,
    bioavailability: float = ...,
    theta: List[float],
    eta: List[List[float]],
    error_model: Literal["additive", "proportional", "combined"] = ...,
    sigma: float = ...,
    sigma_add: Optional[float] = ...,
) -> List[Dict[str, Any]]: ...


def read_nonmem(csv_text: str) -> Dict[str, Any]: ...


def kaplan_meier(
    times: List[float],
    events: List[bool],
    *,
    conf_level: float = ...,
) -> KaplanMeierResult: ...


def log_rank_test(
    times: List[float],
    events: List[bool],
    groups: List[int],
) -> LogRankTestResult: ...


def fault_tree_mc(
    spec: Dict[str, Any],
    n_scenarios: int,
    *,
    seed: int = ...,
    device: str = ...,
    chunk_size: int = ...,
) -> Dict[str, Any]: ...


def fault_tree_mc_ce_is(
    spec: Dict[str, Any],
    *,
    n_per_level: int = ...,
    elite_fraction: float = ...,
    max_levels: int = ...,
    q_max: float = ...,
    seed: int = ...,
) -> Dict[str, Any]: ...


def profile_ci_py(
    model: Any,
    fit_result: FitResult,
    *,
    param_idx: Optional[int] = ...,
    chi2_level: float = ...,
    tol: float = ...,
) -> Any: ...


def churn_generate_data(
    *,
    n_customers: int = ...,
    n_cohorts: int = ...,
    max_time: float = ...,
    treatment_fraction: float = ...,
    seed: int = ...,
) -> Dict[str, Any]: ...


def churn_retention(
    times: List[float],
    events: List[bool],
    groups: List[int],
    *,
    conf_level: float = ...,
) -> Dict[str, Any]: ...


def churn_risk_model(
    times: List[float],
    events: List[bool],
    covariates: List[List[float]],
    names: List[str],
    *,
    conf_level: float = ...,
) -> Dict[str, Any]: ...


def churn_uplift(
    times: List[float],
    events: List[bool],
    treated: List[int],
    covariates: List[List[float]],
    *,
    horizon: float = ...,
) -> Dict[str, Any]: ...


def churn_diagnostics(
    times: List[float],
    events: List[bool],
    groups: List[int],
    *,
    treated: List[int] = ...,
    covariates: List[List[float]] = ...,
    covariate_names: List[str] = ...,
    trim: float = ...,
) -> Dict[str, Any]: ...


def churn_cohort_matrix(
    times: List[float],
    events: List[bool],
    groups: List[int],
    period_boundaries: List[float],
) -> Dict[str, Any]: ...


def churn_compare(
    times: List[float],
    events: List[bool],
    groups: List[int],
    *,
    conf_level: float = ...,
    correction: str = ...,
    alpha: float = ...,
) -> Dict[str, Any]: ...


def churn_uplift_survival(
    times: List[float],
    events: List[bool],
    treated: List[int],
    *,
    covariates: List[List[float]] = ...,
    horizon: float = ...,
    eval_horizons: List[float] = ...,
    trim: float = ...,
) -> Dict[str, Any]: ...


def churn_bootstrap_hr(
    times: List[float],
    events: List[bool],
    covariates: List[List[float]],
    names: List[str],
    *,
    n_bootstrap: int = ...,
    seed: int = ...,
    conf_level: float = ...,
) -> Dict[str, Any]: ...


def churn_ingest(
    times: List[float],
    events: List[bool],
    *,
    groups: Optional[List[int]] = ...,
    treated: Optional[List[int]] = ...,
    covariates: List[List[float]] = ...,
    covariate_names: List[str] = ...,
    observation_end: Optional[float] = ...,
) -> Dict[str, Any]: ...


def from_arrow_ipc(
    ipc_bytes: bytes,
    poi: str = "mu",
    observations: Optional[Dict[str, List[float]]] = ...,
) -> HistFactoryModel: ...


def from_parquet(
    path: str,
    poi: str = "mu",
    observations: Optional[Dict[str, List[float]]] = ...,
) -> HistFactoryModel: ...

def from_parquet_with_modifiers(
    yields_path: str,
    modifiers_path: str,
    poi: str = "mu",
    observations: Optional[Dict[str, List[float]]] = ...,
) -> HistFactoryModel: ...

def to_arrow_yields_ipc(
    model: HistFactoryModel,
    params: Optional[List[float]] = ...,
) -> bytes: ...

def to_arrow_params_ipc(
    model: HistFactoryModel,
    params: Optional[List[float]] = ...,
) -> bytes: ...


class DifferentiableSession:
    def __init__(self, model: HistFactoryModel, signal_sample_name: str) -> None: ...
    def nll_grad_signal(
        self,
        params: List[float],
        signal_ptr: int,
        grad_signal_ptr: int,
    ) -> float: ...
    def signal_n_bins(self) -> int: ...
    def n_params(self) -> int: ...
    def parameter_init(self) -> List[float]: ...


class ProfiledDifferentiableSession:
    def __init__(self, model: HistFactoryModel, signal_sample_name: str) -> None: ...
    def profiled_q0_and_grad(self, signal_ptr: int) -> Tuple[float, List[float]]: ...
    def profiled_qmu_and_grad(
        self,
        mu_test: float,
        signal_ptr: int,
    ) -> Tuple[float, List[float]]: ...
    def batch_profiled_qmu(self, mu_values: List[float]) -> List[Tuple[float, List[float]]]: ...
    def signal_n_bins(self) -> int: ...
    def n_params(self) -> int: ...
    def parameter_init(self) -> List[float]: ...


class MetalProfiledDifferentiableSession:
    def __init__(self, model: HistFactoryModel, signal_sample_name: str) -> None: ...
    def upload_signal(self, signal: List[float]) -> None: ...
    def profiled_q0_and_grad(self) -> Tuple[float, List[float]]: ...
    def profiled_qmu_and_grad(self, mu_test: float) -> Tuple[float, List[float]]: ...
    def batch_profiled_qmu(self, mu_values: List[float]) -> List[Tuple[float, List[float]]]: ...
    def signal_n_bins(self) -> int: ...
    def n_params(self) -> int: ...
    def parameter_init(self) -> List[float]: ...


class FlowPdf:
    @staticmethod
    def from_manifest(manifest_path: str, context_param_indices: List[int]) -> FlowPdf: ...
    def n_context(self) -> int: ...
    def has_sample_model(self) -> bool: ...
    def has_analytical_grad(self) -> bool: ...
    def has_gpu_logprob(self) -> bool: ...
    def has_gpu_grad(self) -> bool: ...
    def gpu_ep_kind(self) -> Optional[str]: ...
    def capabilities(self) -> Dict[str, Any]: ...
    def observable_names(self) -> List[str]: ...
    def log_norm_correction(self) -> float: ...
    def update_normalization(self, params: ParamsLike) -> None: ...
    def log_prob_batch(
        self,
        events: Dict[str, List[float]],
        bounds: Dict[str, Tuple[float, float]],
        params: ParamsLike,
    ) -> List[float]: ...


class DcrSurrogate:
    @staticmethod
    def from_manifest(
        manifest_path: str,
        systematic_param_indices: List[int],
        systematic_names: List[str],
        process_name: str,
    ) -> DcrSurrogate: ...
    def process_name(self) -> str: ...
    def systematic_names(self) -> List[str]: ...
    def norm_tolerance(self) -> float: ...
    def update_normalization(self, params: ParamsLike) -> None: ...
    def validate_nominal_normalization(self, params: ParamsLike) -> Tuple[float, float]: ...
    def log_prob_batch(
        self,
        events: Dict[str, List[float]],
        bounds: Dict[str, Tuple[float, float]],
        params: ParamsLike,
    ) -> List[float]: ...


class GpuFlowSession:
    def __init__(
        self,
        n_events: int,
        n_params: int,
        processes: List[Dict[str, Any]],
        gauss_constraints: Optional[List[Dict[str, Any]]] = ...,
        constraint_const: float = ...,
    ) -> None: ...
    def nll(self, logp_flat: List[float], params: List[float]) -> float: ...
    def nll_device_ptr_f32(self, d_logp_flat_ptr: int, params: List[float]) -> float: ...
    def compute_yields(self, params: List[float]) -> List[float]: ...
    def n_events(self) -> int: ...
    def n_procs(self) -> int: ...
    def n_params(self) -> int: ...

# ── Native Rust visualization renderer ──────────────────────────────
def render_viz(
    artifact_json: str,
    kind: str,
    format: str = "svg",
    *,
    config_yaml: Optional[str] = ...,
    dpi: Optional[int] = ...,
) -> bytes:
    """Render a viz artifact JSON to bytes (SVG/PDF/PNG) using the native Rust renderer."""
    ...
