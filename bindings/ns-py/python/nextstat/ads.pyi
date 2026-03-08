from __future__ import annotations

from typing import List, Optional, Sequence, TypedDict

from ._core import (
    BetaBinomialModel,
    CovariateTiming,
    CupedAdjustmentResult,
    CureAdjustmentResult,
    DelayCorrectionModel,
)


class CovariateProvenance(TypedDict):
    name: str
    timing: CovariateTiming
    source_dataset: Optional[str]


def hill(x: float, ec: float, slope: float) -> float: ...


def adstock_geometric(spend: Sequence[float], decay: float) -> List[float]: ...


def cuped_adjust(
    control_outcomes: Sequence[float],
    control_covariates: Sequence[float],
    variant_outcomes: Sequence[float],
    variant_covariates: Sequence[float],
    *,
    covariate_name: Optional[str] = ...,
    covariate_provenance: Optional[CovariateProvenance] = ...,
    pre_treatment_only: bool = ...,
) -> CupedAdjustmentResult: ...


def cure_adjust(
    control_outcomes: Sequence[float],
    control_covariates: Sequence[Sequence[float]],
    variant_outcomes: Sequence[float],
    variant_covariates: Sequence[Sequence[float]],
    *,
    covariate_names: Optional[Sequence[str]] = ...,
    covariate_provenance: Optional[Sequence[CovariateProvenance]] = ...,
    pre_treatment_only: bool = ...,
) -> CureAdjustmentResult: ...


__all__: List[str]
