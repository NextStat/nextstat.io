"""Type stubs for the compiled extension module `nextstat._core`.

These stubs are intentionally lightweight and cover the public API exposed by
the PyO3 module in `bindings/ns-py/src/lib.rs`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, overload

__version__: str


class HistFactoryModel:
    @staticmethod
    def from_workspace(json_str: str) -> HistFactoryModel: ...

    def n_params(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...
    def expected_data(self, params: List[float], *, include_auxdata: bool = ...) -> List[float]: ...
    def with_observed_main(self, observed_main: List[float]) -> HistFactoryModel: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...
    def poi_index(self) -> Optional[int]: ...


class GaussianMeanModel:
    def __init__(self, y: List[float], sigma: float) -> None: ...

    def n_params(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class LinearRegressionModel:
    def __init__(self, x: List[List[float]], y: List[float], *, include_intercept: bool = ...) -> None: ...

    def n_params(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
    def grad_nll(self, params: List[float]) -> List[float]: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class LogisticRegressionModel:
    def __init__(self, x: List[List[float]], y: List[int], *, include_intercept: bool = ...) -> None: ...

    def n_params(self) -> int: ...
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
    ) -> ComposedGlmModel: ...

    def n_params(self) -> int: ...
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


class MaximumLikelihoodEstimator:
    def __init__(self) -> None: ...
    @overload
    def fit(self, model: HistFactoryModel, *, data: Optional[List[float]] = ...) -> FitResult: ...
    @overload
    def fit(
        self,
        model: Union[
            GaussianMeanModel,
            LinearRegressionModel,
            LogisticRegressionModel,
            PoissonRegressionModel,
            NegativeBinomialRegressionModel,
            ComposedGlmModel,
        ],
        *,
        data: Literal[None] = ...,
    ) -> FitResult: ...
    def fit_batch(
        self,
        models_or_model: Union[List[HistFactoryModel], HistFactoryModel],
        datasets: Optional[List[List[float]]] = ...,
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
@overload
def fit(model: HistFactoryModel, *, data: Optional[List[float]] = ...) -> FitResult: ...
@overload
def fit(
    model: Union[
        GaussianMeanModel,
        LinearRegressionModel,
        LogisticRegressionModel,
        PoissonRegressionModel,
        NegativeBinomialRegressionModel,
        ComposedGlmModel,
    ],
    *,
    data: Literal[None] = ...,
) -> FitResult: ...
def fit_toys(
    model: HistFactoryModel,
    params: List[float],
    *,
    n_toys: int = ...,
    seed: int = ...,
) -> List[FitResult]: ...
def ranking(
    model: HistFactoryModel,
) -> List[Dict[str, Any]]: ...
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
    model: Union[
        GaussianMeanModel,
        LinearRegressionModel,
        LogisticRegressionModel,
        PoissonRegressionModel,
        NegativeBinomialRegressionModel,
        ComposedGlmModel,
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
    ys: List[List[float]],
) -> Dict[str, Any]: ...


def kalman_smooth(
    model: KalmanModel,
    ys: List[List[float]],
) -> Dict[str, Any]: ...


def kalman_em(
    model: KalmanModel,
    ys: List[List[float]],
    *,
    max_iter: int = ...,
    tol: float = ...,
    estimate_q: bool = ...,
    estimate_r: bool = ...,
    min_diag: float = ...,
) -> Dict[str, Any]: ...
