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


__all__ = [
    "linear_random_intercept",
    "logistic_random_intercept",
    "poisson_random_intercept",
]

