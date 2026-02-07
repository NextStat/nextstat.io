"""Type stubs for the compiled extension module `nextstat._core`.

These stubs are intentionally lightweight and cover the public API exposed by
the PyO3 module in `bindings/ns-py/src/lib.rs`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

__version__: str

# `HistFactoryModel` accepts Python sequences (list/tuple/array('d')) and also
# buffer-protocol objects for performance. Type stubs stay conservative and
# describe the common supported surfaces without using `Any`.
ParamsLike = Union[Sequence[float], memoryview]


class HistFactoryModel:
    @staticmethod
    def from_workspace(json_str: str) -> HistFactoryModel: ...
    @staticmethod
    def from_xml(xml_path: str) -> HistFactoryModel: ...

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
    def __init__(self) -> None: ...

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
            ComposedGlmModel,
            LmmMarginalModel,
            ExponentialSurvivalModel,
            WeibullSurvivalModel,
            LogNormalAftModel,
            CoxPhModel,
            OneCompartmentOralPkModel,
            OneCompartmentOralPkNlmeModel,
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
    def __init__(self, *, max_iter: int = ..., tol: float = ..., m: int = ...) -> None: ...
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
            OneCompartmentOralPkModel,
            OneCompartmentOralPkNlmeModel,
        ],
        *,
        data: Literal[None] = ...,
        init_pars: Optional[List[float]] = ...,
    ) -> FitResult: ...
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
    ) -> List[Dict[str, Any]]: ...


def from_pyhf(json_str: str) -> HistFactoryModel: ...
def from_histfactory(xml_path: str) -> HistFactoryModel: ...
def histfactory_bin_edges_by_channel(xml_path: str) -> Dict[str, List[float]]: ...

def read_root_histogram(root_path: str, hist_path: str) -> Dict[str, Any]: ...
@overload
def fit(model: HistFactoryModel, *, data: Optional[List[float]] = ..., init_pars: Optional[List[float]] = ...) -> FitResult: ...
@overload
def fit(
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
        OneCompartmentOralPkModel,
        OneCompartmentOralPkNlmeModel,
    ],
    *,
    data: Literal[None] = ...,
    init_pars: Optional[List[float]] = ...,
) -> FitResult: ...

def map_fit(posterior: Posterior) -> FitResult: ...
@overload
def fit_batch(models_or_model: List[HistFactoryModel], datasets: Literal[None] = ...) -> List[FitResult]: ...
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
def fit_batch(models_or_model: HistFactoryModel, datasets: List[List[float]]) -> List[FitResult]: ...
def fit_toys(
    model: HistFactoryModel,
    params: List[float],
    *,
    n_toys: int = ...,
    seed: int = ...,
) -> List[FitResult]: ...
def fit_toys_batch(
    model: HistFactoryModel,
    params: List[float],
    *,
    n_toys: int = ...,
    seed: int = ...,
) -> List[FitResult]: ...
def fit_toys_batch_gpu(
    model: HistFactoryModel,
    params: List[float],
    *,
    n_toys: int = ...,
    seed: int = ...,
    device: str = ...,
) -> List[FitResult]: ...
def set_eval_mode(mode: str) -> None: ...
def get_eval_mode() -> str: ...
def has_accelerate() -> bool: ...
def has_cuda() -> bool: ...
def asimov_data(model: HistFactoryModel, params: List[float]) -> List[float]: ...
def poisson_toys(
    model: HistFactoryModel,
    params: List[float],
    *,
    n_toys: int = ...,
    seed: int = ...,
) -> List[List[float]]: ...
def ranking(
    model: HistFactoryModel,
) -> List[Dict[str, Any]]: ...
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
    model: HistFactoryModel,
    *,
    data: Optional[List[float]] = ...,
    return_tail_probs: bool = ...,
) -> Union[float, Tuple[float, List[float]]]: ...


def profile_scan(
    model: HistFactoryModel,
    mu_values: List[float],
    *,
    data: Optional[List[float]] = ...,
) -> Dict[str, Any]: ...


def upper_limit(
    model: HistFactoryModel,
    *,
    alpha: float = ...,
    lo: float = ...,
    hi: Optional[float] = ...,
    rtol: float = ...,
    max_iter: int = ...,
    data: Optional[List[float]] = ...,
) -> float: ...


def upper_limits(
    model: HistFactoryModel,
    scan: List[float],
    *,
    alpha: float = ...,
    data: Optional[List[float]] = ...,
) -> Tuple[float, List[float]]: ...


def upper_limits_root(
    model: HistFactoryModel,
    *,
    alpha: float = ...,
    lo: float = ...,
    hi: Optional[float] = ...,
    rtol: float = ...,
    max_iter: int = ...,
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
    init_jitter: float = ...,
    init_jitter_rel: Optional[float] = ...,
    init_overdispersed_rel: Optional[float] = ...,
    data: Optional[List[float]] = ...,
) -> Dict[str, Any]: ...
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
    init_jitter: float = ...,
    init_jitter_rel: Optional[float] = ...,
    init_overdispersed_rel: Optional[float] = ...,
    data: Literal[None] = ...,
) -> Dict[str, Any]: ...
@overload
def sample(
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
        OneCompartmentOralPkModel,
        OneCompartmentOralPkNlmeModel,
    ],
    *,
    n_chains: int = ...,
    n_warmup: int = ...,
    n_samples: int = ...,
    seed: int = ...,
    max_treedepth: int = ...,
    target_accept: float = ...,
    init_jitter: float = ...,
    init_jitter_rel: Optional[float] = ...,
    init_overdispersed_rel: Optional[float] = ...,
    data: Literal[None] = ...,
) -> Dict[str, Any]: ...


def cls_curve(
    model: HistFactoryModel,
    scan: List[float],
    *,
    alpha: float = ...,
    data: Optional[List[float]] = ...,
) -> Dict[str, Any]: ...


def profile_curve(
    model: HistFactoryModel,
    mu_values: List[float],
    *,
    data: Optional[List[float]] = ...,
) -> Dict[str, Any]: ...


def kalman_filter(
    model: KalmanModel,
    ys: List[List[Optional[float]]],
) -> Dict[str, Any]: ...


def kalman_smooth(
    model: KalmanModel,
    ys: List[List[Optional[float]]],
) -> Dict[str, Any]: ...


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
) -> Dict[str, Any]: ...


def kalman_forecast(
    model: KalmanModel,
    ys: List[List[Optional[float]]],
    *,
    steps: int = ...,
    alpha: Optional[float] = ...,
) -> Dict[str, Any]: ...


def kalman_simulate(
    model: KalmanModel,
    *,
    t_max: int,
    seed: int = ...,
    init: str = ...,
    x0: Optional[List[float]] = ...,
) -> Dict[str, Any]: ...
