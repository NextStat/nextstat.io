"""Hierarchical / multilevel modeling helpers (Phase 7).

The core random-intercept implementation lives in the Rust `ModelBuilder` and is
exposed via `nextstat._core.ComposedGlmModel.*` constructors. This module adds a
Python-friendly surface and keeps optional deps out of the compiled extension.
"""

from __future__ import annotations

from typing import Any, Optional

from .data import GlmSpec


def linear_random_intercept(
    *,
    x: Any,
    y: Any,
    group_idx: Any,
    include_intercept: bool = True,
    n_groups: Optional[int] = None,
    coef_prior_mu: float = 0.0,
    coef_prior_sigma: float = 10.0,
) -> Any:
    """Build a Gaussian linear regression with a group-indexed random intercept."""

    return GlmSpec.linear_regression(
        x=x,
        y=y,
        include_intercept=include_intercept,
        group_idx=group_idx,
        n_groups=n_groups,
        coef_prior_mu=coef_prior_mu,
        coef_prior_sigma=coef_prior_sigma,
    ).build()


def linear_random_slope(
    *,
    x: Any,
    y: Any,
    group_idx: Any,
    random_slope_feature_idx: int,
    include_intercept: bool = True,
    n_groups: Optional[int] = None,
    coef_prior_mu: float = 0.0,
    coef_prior_sigma: float = 10.0,
    random_intercept_non_centered: bool = False,
    random_slope_non_centered: bool = False,
) -> Any:
    """Build a Gaussian linear regression with random intercept + one random slope."""

    base = GlmSpec.linear_regression(
        x=x,
        y=y,
        include_intercept=include_intercept,
        group_idx=group_idx,
        n_groups=n_groups,
        coef_prior_mu=coef_prior_mu,
        coef_prior_sigma=coef_prior_sigma,
    )
    return GlmSpec(
        kind=base.kind,
        x=base.x,
        y=base.y,
        include_intercept=base.include_intercept,
        group_idx=base.group_idx,
        n_groups=base.n_groups,
        offset=base.offset,
        coef_prior_mu=base.coef_prior_mu,
        coef_prior_sigma=base.coef_prior_sigma,
        random_intercept_non_centered=bool(random_intercept_non_centered),
        random_slope_feature_idx=int(random_slope_feature_idx),
        random_slope_non_centered=bool(random_slope_non_centered),
    ).build()


def logistic_random_intercept(
    *,
    x: Any,
    y: Any,
    group_idx: Any,
    include_intercept: bool = True,
    n_groups: Optional[int] = None,
    coef_prior_mu: float = 0.0,
    coef_prior_sigma: float = 10.0,
) -> Any:
    """Build a Bernoulli-logit regression with a group-indexed random intercept."""

    return GlmSpec.logistic_regression(
        x=x,
        y=y,
        include_intercept=include_intercept,
        group_idx=group_idx,
        n_groups=n_groups,
        coef_prior_mu=coef_prior_mu,
        coef_prior_sigma=coef_prior_sigma,
    ).build()


def logistic_random_slope(
    *,
    x: Any,
    y: Any,
    group_idx: Any,
    random_slope_feature_idx: int,
    include_intercept: bool = True,
    n_groups: Optional[int] = None,
    coef_prior_mu: float = 0.0,
    coef_prior_sigma: float = 10.0,
    random_intercept_non_centered: bool = False,
    random_slope_non_centered: bool = False,
) -> Any:
    """Build a Bernoulli-logit regression with random intercept + one random slope."""

    base = GlmSpec.logistic_regression(
        x=x,
        y=y,
        include_intercept=include_intercept,
        group_idx=group_idx,
        n_groups=n_groups,
        coef_prior_mu=coef_prior_mu,
        coef_prior_sigma=coef_prior_sigma,
    )
    return GlmSpec(
        kind=base.kind,
        x=base.x,
        y=base.y,
        include_intercept=base.include_intercept,
        group_idx=base.group_idx,
        n_groups=base.n_groups,
        offset=base.offset,
        coef_prior_mu=base.coef_prior_mu,
        coef_prior_sigma=base.coef_prior_sigma,
        random_intercept_non_centered=bool(random_intercept_non_centered),
        random_slope_feature_idx=int(random_slope_feature_idx),
        random_slope_non_centered=bool(random_slope_non_centered),
    ).build()


def poisson_random_intercept(
    *,
    x: Any,
    y: Any,
    group_idx: Any,
    include_intercept: bool = True,
    offset: Any = None,
    n_groups: Optional[int] = None,
    coef_prior_mu: float = 0.0,
    coef_prior_sigma: float = 10.0,
) -> Any:
    """Build a Poisson regression (log link) with a group-indexed random intercept."""

    return GlmSpec.poisson_regression(
        x=x,
        y=y,
        include_intercept=include_intercept,
        offset=offset,
        group_idx=group_idx,
        n_groups=n_groups,
        coef_prior_mu=coef_prior_mu,
        coef_prior_sigma=coef_prior_sigma,
    ).build()


def logistic_correlated_intercept_slope(
    *,
    x: Any,
    y: Any,
    group_idx: Any,
    correlated_feature_idx: int,
    include_intercept: bool = True,
    n_groups: Optional[int] = None,
    coef_prior_mu: float = 0.0,
    coef_prior_sigma: float = 10.0,
    lkj_eta: float = 1.0,
) -> Any:
    """Build a Bernoulli-logit regression with correlated random intercept + slope."""

    base = GlmSpec.logistic_regression(
        x=x,
        y=y,
        include_intercept=include_intercept,
        group_idx=group_idx,
        n_groups=n_groups,
        coef_prior_mu=coef_prior_mu,
        coef_prior_sigma=coef_prior_sigma,
    )
    return GlmSpec(
        kind=base.kind,
        x=base.x,
        y=base.y,
        include_intercept=base.include_intercept,
        group_idx=base.group_idx,
        n_groups=base.n_groups,
        offset=base.offset,
        coef_prior_mu=base.coef_prior_mu,
        coef_prior_sigma=base.coef_prior_sigma,
        correlated_feature_idx=int(correlated_feature_idx),
        lkj_eta=float(lkj_eta),
    ).build()


__all__ = [
    "linear_random_intercept",
    "linear_random_slope",
    "logistic_random_intercept",
    "logistic_random_slope",
    "logistic_correlated_intercept_slope",
    "poisson_random_intercept",
]
