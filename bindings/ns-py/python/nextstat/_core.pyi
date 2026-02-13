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
    termination_reason: str
    final_grad_norm: float
    initial_nll: float
    n_active_bounds: int

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
            OneCompartmentOralPkModel,
            OneCompartmentOralPkNlmeModel,
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
            OneCompartmentOralPkModel,
            OneCompartmentOralPkNlmeModel,
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
            OneCompartmentOralPkModel,
            OneCompartmentOralPkNlmeModel,
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
def meta_fixed(
    estimates: List[float],
    standard_errors: List[float],
    *,
    labels: Optional[List[str]] = ...,
    conf_level: float = ...,
) -> Dict[str, Any]: ...
def meta_random(
    estimates: List[float],
    standard_errors: List[float],
    *,
    labels: Optional[List[str]] = ...,
    conf_level: float = ...,
) -> Dict[str, Any]: ...
def chain_ladder(
    triangle: List[List[float]],
) -> Dict[str, Any]: ...
def mack_chain_ladder(
    triangle: List[List[float]],
    *,
    conf_level: float = ...,
) -> Dict[str, Any]: ...
def hypotest_toys(
    poi_test: float,
    model: HistFactoryModel,
    *,
    n_toys: int = ...,
    seed: int = ...,
    expected_set: bool = ...,
    data: Optional[List[float]] = ...,
    return_tail_probs: bool = ...,
    return_meta: bool = ...,
) -> Any: ...
def ranking_gpu(model: HistFactoryModel) -> List[Dict[str, Any]]: ...
def ranking_metal(model: HistFactoryModel) -> List[Dict[str, Any]]: ...

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
def fit_batch(models_or_model: HistFactoryModel, datasets: List[List[float]]) -> List[FitResult]: ...
def fit_toys(
    model: HistFactoryModel,
    params: List[float],
    *,
    n_toys: int = ...,
    seed: int = ...,
) -> List[FitResult]: ...
def unbinned_fit_toys(
    model: UnbinnedModel,
    params: List[float],
    *,
    n_toys: int = ...,
    seed: int = ...,
    init_params: Optional[List[float]] = ...,
    max_retries: int = ...,
    max_iter: int = ...,
    compute_hessian: bool = ...,
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
    device: str = "cpu",
) -> List[FitResult]: ...
def set_eval_mode(mode: str) -> None: ...
def set_threads(threads: int) -> bool: ...
def get_eval_mode() -> str: ...
def has_accelerate() -> bool: ...
def has_cuda() -> bool: ...
def has_metal() -> bool: ...
def workspace_audit(json_str: str) -> Dict[str, Any]: ...
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

def unbinned_ranking(
    model: UnbinnedModel,
) -> List[Dict[str, Any]]: ...

def unbinned_profile_scan(
    model: UnbinnedModel,
    mu_values: List[float],
) -> Dict[str, Any]: ...

def unbinned_hypotest(
    mu_test: float,
    model: UnbinnedModel,
) -> Dict[str, Any]: ...

def unbinned_hypotest_toys(
    poi_test: float,
    model: UnbinnedModel,
    *,
    n_toys: int = ...,
    seed: int = ...,
    expected_set: bool = ...,
    return_tail_probs: bool = ...,
    return_meta: bool = ...,
) -> Any: ...

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
    device: str = "cpu",
    return_params: bool = ...,
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
    entity_ids: List[int],
    x: List[float],
    y: List[float],
    p: int,
    *,
    cluster_ids: Optional[List[int]] = ...,
) -> Dict[str, Any]: ...


def did(
    y: List[float],
    treat: List[int],
    post: List[int],
    cluster_ids: List[int],
) -> Dict[str, Any]: ...


def event_study(
    y: List[float],
    entity_ids: List[int],
    time_ids: List[int],
    relative_time: List[int],
    min_lag: int,
    max_lag: int,
    reference_period: int,
    cluster_ids: List[int],
) -> Dict[str, Any]: ...


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
) -> Dict[str, Any]: ...


def rosenbaum_bounds(
    y_treated: List[float],
    y_control: List[float],
    gammas: List[float],
) -> Dict[str, Any]: ...


def kaplan_meier(
    times: List[float],
    events: List[bool],
    *,
    conf_level: float = ...,
) -> Dict[str, Any]: ...


def log_rank_test(
    times: List[float],
    events: List[bool],
    groups: List[int],
) -> Dict[str, Any]: ...


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
