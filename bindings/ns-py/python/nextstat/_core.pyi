"""Type stubs for the compiled extension module `nextstat._core`.

These stubs are intentionally lightweight and cover the public API exposed by
the PyO3 module in `bindings/ns-py/src/lib.rs`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

__version__: str


class HistFactoryModel:
    @staticmethod
    def from_workspace(json_str: str) -> HistFactoryModel: ...

    def n_params(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...
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

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class LinearRegressionModel:
    def __init__(self, x: List[List[float]], y: List[float], *, include_intercept: bool = ...) -> None: ...

    def n_params(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


class LogisticRegressionModel:
    def __init__(self, x: List[List[float]], y: List[int], *, include_intercept: bool = ...) -> None: ...

    def n_params(self) -> int: ...
    def nll(self, params: List[float]) -> float: ...

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
    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...


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
    def fit(
        self,
        model: Union[
            HistFactoryModel,
            GaussianMeanModel,
            LinearRegressionModel,
            LogisticRegressionModel,
            PoissonRegressionModel,
            NegativeBinomialRegressionModel,
            ComposedGlmModel,
        ],
        *,
        data: Optional[List[float]] = ...,
    ) -> FitResult: ...


def from_pyhf(json_str: str) -> HistFactoryModel: ...
def fit(
    model: Union[
        HistFactoryModel,
        GaussianMeanModel,
        LinearRegressionModel,
        LogisticRegressionModel,
        PoissonRegressionModel,
        NegativeBinomialRegressionModel,
        ComposedGlmModel,
    ],
    *,
    data: Optional[List[float]] = ...,
) -> FitResult: ...
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


def sample(
    model: Union[
        HistFactoryModel,
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
    data: Optional[List[float]] = ...,
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
