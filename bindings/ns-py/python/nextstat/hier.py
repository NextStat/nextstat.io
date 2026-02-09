"""Hierarchical / multilevel modeling helpers (Phase 7).

The core random-intercept implementation lives in the Rust `ModelBuilder` and is
exposed via `nextstat._core.ComposedGlmModel.*` constructors. This module adds a
Python-friendly surface and keeps optional deps out of the compiled extension.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .data import GlmSpec


def _encode_group_idx(group: Sequence[Any]) -> tuple[list[int], int]:
    # If already integer-like, keep as-is (after int cast).
    try:
        ints = [int(v) for v in group]
        # If conversion doesn't change values (e.g. "1" -> 1 is fine), accept.
        if all(isinstance(v, (int, bool)) or str(v).strip().lstrip("+-").isdigit() for v in group):
            n_groups = (max(ints) + 1) if ints else 0
            return ints, n_groups
    except Exception:
        pass

    # Otherwise encode categorical group labels deterministically.
    labels = [("None" if v is None else str(v)) for v in group]
    levels = sorted(set(labels))
    idx = {lvl: i for i, lvl in enumerate(levels)}
    return [idx[v] for v in labels], len(levels)


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


def linear_random_intercept_from_formula(
    formula: str,
    data: Any,
    *,
    group: str,
    categorical: Optional[Sequence[str]] = None,
    n_groups: Optional[int] = None,
    coef_prior_mu: float = 0.0,
    coef_prior_sigma: float = 10.0,
) -> Any:
    """Build a Gaussian linear random-intercept model from a formula and tabular data."""
    import nextstat

    y_name, terms, _ = nextstat.formula.parse_formula(formula)
    cols = nextstat.formula.to_columnar(data, [y_name] + terms + [group])
    y, x_full, names_full = nextstat.formula.design_matrices(formula, cols, categorical=categorical)

    if names_full and names_full[0] == "Intercept":
        x = [row[1:] for row in x_full]
        include_intercept = True
    else:
        x = x_full
        include_intercept = False

    group_idx, n_g = _encode_group_idx(cols[group])
    return linear_random_intercept(
        x=x,
        y=y,
        group_idx=group_idx,
        include_intercept=include_intercept,
        n_groups=(n_groups if n_groups is not None else n_g),
        coef_prior_mu=coef_prior_mu,
        coef_prior_sigma=coef_prior_sigma,
    )

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


def logistic_random_intercept_from_formula(
    formula: str,
    data: Any,
    *,
    group: str,
    categorical: Optional[Sequence[str]] = None,
    n_groups: Optional[int] = None,
    coef_prior_mu: float = 0.0,
    coef_prior_sigma: float = 10.0,
) -> Any:
    """Build a logistic random-intercept model from a formula and tabular data."""
    import nextstat

    y_name, terms, _ = nextstat.formula.parse_formula(formula)
    cols = nextstat.formula.to_columnar(data, [y_name] + terms + [group])
    y_float, x_full, names_full = nextstat.formula.design_matrices(formula, cols, categorical=categorical)
    y = [int(v) for v in y_float]

    if names_full and names_full[0] == "Intercept":
        x = [row[1:] for row in x_full]
        include_intercept = True
    else:
        x = x_full
        include_intercept = False

    group_idx, n_g = _encode_group_idx(cols[group])
    return logistic_random_intercept(
        x=x,
        y=y,
        group_idx=group_idx,
        include_intercept=include_intercept,
        n_groups=(n_groups if n_groups is not None else n_g),
        coef_prior_mu=coef_prior_mu,
        coef_prior_sigma=coef_prior_sigma,
    )

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


def poisson_random_intercept_from_formula(
    formula: str,
    data: Any,
    *,
    group: str,
    categorical: Optional[Sequence[str]] = None,
    offset: Any = None,
    n_groups: Optional[int] = None,
    coef_prior_mu: float = 0.0,
    coef_prior_sigma: float = 10.0,
) -> Any:
    """Build a Poisson random-intercept model from a formula and tabular data."""
    import nextstat

    y_name, terms, _ = nextstat.formula.parse_formula(formula)
    extra_cols: list[str] = [group]
    if isinstance(offset, str):
        extra_cols.append(offset)
    cols = nextstat.formula.to_columnar(data, [y_name] + terms + extra_cols)

    y_float, x_full, names_full = nextstat.formula.design_matrices(formula, cols, categorical=categorical)
    y = [int(v) for v in y_float]

    if names_full and names_full[0] == "Intercept":
        x = [row[1:] for row in x_full]
        include_intercept = True
    else:
        x = x_full
        include_intercept = False

    group_idx, n_g = _encode_group_idx(cols[group])
    off = cols[offset] if isinstance(offset, str) else offset

    return poisson_random_intercept(
        x=x,
        y=y,
        group_idx=group_idx,
        include_intercept=include_intercept,
        offset=off,
        n_groups=(n_groups if n_groups is not None else n_g),
        coef_prior_mu=coef_prior_mu,
        coef_prior_sigma=coef_prior_sigma,
    )

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
    "linear_random_intercept_from_formula",
    "linear_random_slope",
    "logistic_random_intercept",
    "logistic_random_intercept_from_formula",
    "logistic_random_slope",
    "logistic_correlated_intercept_slope",
    "poisson_random_intercept",
    "poisson_random_intercept_from_formula",
]
