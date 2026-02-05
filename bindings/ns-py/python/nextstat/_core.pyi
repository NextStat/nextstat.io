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
    def with_observed_main(self, observed_main: List[float]) -> HistFactoryModel: ...

    def parameter_names(self) -> List[str]: ...
    def suggested_init(self) -> List[float]: ...
    def suggested_bounds(self) -> List[Tuple[float, float]]: ...
    def poi_index(self) -> Optional[int]: ...


class FitResult:
    parameters: List[float]
    uncertainties: List[float]
    nll: float
    converged: bool
    n_evaluations: int

    @property
    def bestfit(self) -> List[float]: ...

    @property
    def twice_nll(self) -> float: ...

    @property
    def success(self) -> bool: ...


class MaximumLikelihoodEstimator:
    def __init__(self) -> None: ...
    def fit(self, model: HistFactoryModel, *, data: Optional[List[float]] = ...) -> FitResult: ...


def from_pyhf(json_str: str) -> HistFactoryModel: ...
def fit(model: HistFactoryModel, *, data: Optional[List[float]] = ...) -> FitResult: ...


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
    model: HistFactoryModel,
    *,
    n_chains: int = ...,
    n_warmup: int = ...,
    n_samples: int = ...,
    seed: int = ...,
    max_treedepth: int = ...,
    target_accept: float = ...,
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
