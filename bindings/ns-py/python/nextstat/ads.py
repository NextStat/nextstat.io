"""Ads-native observation and response helpers.

This module keeps the public Python surface stable while delegating the heavy
lifting to the Rust backend in ``nextstat._core``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import _core

BetaBinomialModel = getattr(_core, "BetaBinomialModel", None)
DelayCorrectionModel = getattr(_core, "DelayCorrectionModel", None)
_ads_hill = getattr(_core, "ads_hill", None)
_ads_adstock_geometric = getattr(_core, "ads_adstock_geometric", None)
_ads_cuped_adjust = getattr(_core, "ads_cuped_adjust", None)
_ads_cure_adjust = getattr(_core, "ads_cure_adjust", None)


def _require_backend(name: str, obj):
    if obj is None:
        raise NotImplementedError(f"{name} requires the compiled nextstat ads backend")
    return obj


def _normalize_single_covariate_provenance(
    covariate_name: str | None,
    covariate_provenance: Mapping[str, Any] | None,
) -> tuple[str | None, str | None, str | None]:
    if covariate_provenance is None:
        return covariate_name, None, None

    provenance_name = covariate_provenance.get("name")
    if provenance_name is not None:
        provenance_name = str(provenance_name)
    resolved_name = covariate_name or provenance_name
    if not resolved_name:
        raise ValueError("covariate_provenance requires a non-empty 'name'")
    if covariate_name is not None and provenance_name is not None and covariate_name != provenance_name:
        raise ValueError("covariate_name must match covariate_provenance['name']")

    timing = covariate_provenance.get("timing")
    if timing is None:
        raise ValueError("covariate_provenance['timing'] is required")
    source_dataset = covariate_provenance.get("source_dataset")
    return (
        resolved_name,
        str(timing),
        None if source_dataset is None else str(source_dataset),
    )


def _normalize_multi_covariate_provenance(
    covariate_names: Sequence[str] | None,
    covariate_provenance: Sequence[Mapping[str, Any]] | None,
) -> tuple[list[str] | None, list[str] | None, list[str] | None]:
    names = None if covariate_names is None else [str(value) for value in covariate_names]
    if covariate_provenance is None:
        return names, None, None

    items = list(covariate_provenance)
    if names is not None and len(items) != len(names):
        raise ValueError("covariate_provenance length must match covariate_names length")

    resolved_names: list[str] = []
    timings: list[str] = []
    source_datasets: list[str] = []
    for idx, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"covariate_provenance[{idx}] must be a mapping")
        item_name = item.get("name")
        if item_name is not None:
            item_name = str(item_name)
        expected_name = None if names is None else names[idx]
        resolved_name = expected_name or item_name
        if not resolved_name:
            raise ValueError(f"covariate_provenance[{idx}] requires a non-empty 'name'")
        if expected_name is not None and item_name is not None and expected_name != item_name:
            raise ValueError(
                f"covariate_names[{idx}] must match covariate_provenance[{idx}]['name']"
            )
        timing = item.get("timing")
        if timing is None:
            raise ValueError(f"covariate_provenance[{idx}]['timing'] is required")
        source_dataset = item.get("source_dataset")

        resolved_names.append(resolved_name)
        timings.append(str(timing))
        if source_dataset is not None:
            source_datasets.append(str(source_dataset))

    return (
        resolved_names,
        timings,
        source_datasets if len(source_datasets) == len(items) else None,
    )


def hill(x: float, ec: float, slope: float) -> float:
    """Evaluate the Hill saturation curve."""
    fn = _require_backend("nextstat.ads.hill", _ads_hill)
    return float(fn(float(x), float(ec), float(slope)))


def adstock_geometric(spend: Sequence[float], decay: float) -> list[float]:
    """Apply geometric adstock to a spend series."""
    fn = _require_backend("nextstat.ads.adstock_geometric", _ads_adstock_geometric)
    return list(fn([float(value) for value in spend], float(decay)))


def cuped_adjust(
    control_outcomes: Sequence[float],
    control_covariates: Sequence[float],
    variant_outcomes: Sequence[float],
    variant_covariates: Sequence[float],
    *,
    covariate_name: str | None = None,
    covariate_provenance: Mapping[str, Any] | None = None,
    pre_treatment_only: bool = True,
) -> dict:
    """Apply one-covariate CUPED adjustment.

    CUPED is treated as the one-covariate case of the shared CURE layer.
    The returned dict includes method/solver labels plus variance-reduction
    diagnostics suitable for logging or artifact surfaces. Optional
    ``covariate_provenance`` may provide ``name``, ``timing``, and
    ``source_dataset`` for fail-fast leakage validation.
    """

    fn = _require_backend("nextstat.ads.cuped_adjust", _ads_cuped_adjust)
    resolved_name, resolved_timing, resolved_source_dataset = _normalize_single_covariate_provenance(
        covariate_name,
        covariate_provenance,
    )
    return dict(
        fn(
            [float(value) for value in control_outcomes],
            [float(value) for value in control_covariates],
            [float(value) for value in variant_outcomes],
            [float(value) for value in variant_covariates],
            covariate_name=resolved_name,
            covariate_timing=resolved_timing,
            covariate_source_dataset=resolved_source_dataset,
            pre_treatment_only=bool(pre_treatment_only),
        )
    )


def cure_adjust(
    control_outcomes: Sequence[float],
    control_covariates: Sequence[Sequence[float]],
    variant_outcomes: Sequence[float],
    variant_covariates: Sequence[Sequence[float]],
    *,
    covariate_names: Sequence[str] | None = None,
    covariate_provenance: Sequence[Mapping[str, Any]] | None = None,
    pre_treatment_only: bool = True,
) -> dict:
    """Apply multi-covariate CURE adjustment with collinearity guardrails.

    ``covariate_provenance`` may provide one mapping per covariate with
    ``name``, ``timing``, and ``source_dataset`` for fail-fast leakage
    validation.
    """

    fn = _require_backend("nextstat.ads.cure_adjust", _ads_cure_adjust)
    resolved_names, resolved_timings, resolved_source_datasets = _normalize_multi_covariate_provenance(
        covariate_names,
        covariate_provenance,
    )
    return dict(
        fn(
            [float(value) for value in control_outcomes],
            [[float(value) for value in row] for row in control_covariates],
            [float(value) for value in variant_outcomes],
            [[float(value) for value in row] for row in variant_covariates],
            covariate_names=resolved_names,
            covariate_timings=resolved_timings,
            covariate_source_datasets=resolved_source_datasets,
            pre_treatment_only=bool(pre_treatment_only),
        )
    )


__all__ = [
    "BetaBinomialModel",
    "DelayCorrectionModel",
    "cuped_adjust",
    "cure_adjust",
    "hill",
    "adstock_geometric",
]
