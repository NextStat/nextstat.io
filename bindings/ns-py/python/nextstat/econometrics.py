"""Econometrics helpers (Phase 12).

Current baseline:
- Panel linear regression with 1-way fixed effects (within estimator)
- 1-way cluster-robust standard errors (entity or time)

Notes / limitations:
- This is a minimal baseline intended for small/medium problems.
- FE absorbs the intercept and any time-invariant regressors within each entity.
- Cluster-robust SE is 1-way (no 2-way clustering yet).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Mapping, Optional, Sequence, Tuple

import nextstat

from .glm._linalg import as_2d_float_list, mat_vec_mul


ClusterKind = Literal["entity", "time", "none"]


def _validate_lengths(n: int, *seqs: Sequence[Any]) -> None:
    for s in seqs:
        if len(s) != n:
            raise ValueError("length mismatch between inputs")


def _encode_groups(values: Sequence[Any]) -> Tuple[List[int], int]:
    # Deterministic label encoding for non-int identifiers.
    labels = [("None" if v is None else str(v)) for v in values]
    levels = sorted(set(labels))
    idx = {lvl: i for i, lvl in enumerate(levels)}
    return [idx[v] for v in labels], len(levels)


def _within_demean(y: List[float], x: List[List[float]], entity: Sequence[Any]) -> Tuple[List[float], List[List[float]]]:
    n = len(y)
    if n == 0:
        raise ValueError("need at least 1 observation")
    k = len(x[0]) if x else 0
    if k == 0:
        raise ValueError("X must have at least 1 column")
    if any(len(row) != k for row in x):
        raise ValueError("X must be rectangular")
    _validate_lengths(n, x, entity)

    ent_idx, _n_ent = _encode_groups(entity)

    # Means per entity.
    sum_y: dict[int, float] = {}
    sum_x: dict[int, List[float]] = {}
    cnt: dict[int, int] = {}

    for yi, xi, gi in zip(y, x, ent_idx):
        sum_y[gi] = sum_y.get(gi, 0.0) + float(yi)
        if gi not in sum_x:
            sum_x[gi] = [0.0] * k
        sx = sum_x[gi]
        for j in range(k):
            sx[j] += float(xi[j])
        cnt[gi] = cnt.get(gi, 0) + 1

    mean_y = {g: sum_y[g] / float(cnt[g]) for g in cnt}
    mean_x = {g: [sxj / float(cnt[g]) for sxj in sum_x[g]] for g in cnt}

    y_star: List[float] = []
    x_star: List[List[float]] = []
    for yi, xi, gi in zip(y, x, ent_idx):
        y_star.append(float(yi) - float(mean_y[gi]))
        mx = mean_x[gi]
        x_star.append([float(xi[j]) - float(mx[j]) for j in range(k)])

    return y_star, x_star


@dataclass(frozen=True)
class PanelFeFit:
    coef: List[float]
    standard_errors: List[float]
    covariance: List[List[float]]
    column_names: List[str]
    n_obs: int
    n_entities: int
    cluster: ClusterKind


def panel_fe_fit(
    x: Sequence[Sequence[float]],
    y: Sequence[float],
    *,
    entity: Sequence[Any],
    time: Optional[Sequence[Any]] = None,
    cluster: ClusterKind = "entity",
) -> PanelFeFit:
    """Fit panel linear regression with 1-way entity fixed effects (within estimator)."""
    x2 = as_2d_float_list(x)
    y2 = [float(v) for v in y]
    n = len(y2)
    if n == 0:
        raise ValueError("need at least 1 observation")
    if not x2 or not x2[0]:
        raise ValueError("X must be non-empty")
    if len(x2) != n:
        raise ValueError("X and y must have the same length")
    _validate_lengths(n, entity)
    if time is not None:
        _validate_lengths(n, time)

    y_star, x_star = _within_demean(y2, x2, entity)

    # Fit OLS on transformed data (no intercept; absorbed by FE).
    fit = nextstat.glm.linear.fit(x_star, y_star, include_intercept=False)
    coef = [float(v) for v in fit.coef]

    # Residuals for robust covariance.
    yhat = mat_vec_mul(x_star, coef)
    resid = [obs - pred for obs, pred in zip(y_star, yhat)]

    cluster = str(cluster).lower()  # type: ignore[assignment]
    if cluster not in ("entity", "time", "none"):
        raise ValueError("cluster must be one of: entity, time, none")

    if cluster == "none":
        cov = fit.covariance
        se = fit.standard_errors
        ent_idx, n_ent = _encode_groups(entity)
        _ = ent_idx
    else:
        if cluster == "entity":
            groups = entity
            _ent_idx, n_ent = _encode_groups(entity)
        else:
            if time is None:
                raise ValueError("time must be provided when cluster='time'")
            groups = time
            _ent_idx, n_ent = _encode_groups(entity)

        cov = nextstat.robust.ols_cluster_covariance(
            x_star,
            resid,
            groups,
            include_intercept=False,
            df_correction=True,
        )
        se = nextstat.robust.cov_to_se(cov)

    return PanelFeFit(
        coef=coef,
        standard_errors=[float(v) for v in se],
        covariance=[[float(v) for v in row] for row in cov],
        column_names=[f"x{i}" for i in range(len(coef))],
        n_obs=n,
        n_entities=n_ent,
        cluster=cluster,  # type: ignore[arg-type]
    )


def panel_fe_from_formula(
    formula: str,
    data: Any,
    *,
    entity: str,
    time: Optional[str] = None,
    categorical: Optional[Sequence[str]] = None,
    cluster: ClusterKind = "entity",
) -> PanelFeFit:
    """Fit a panel FE model from a minimal formula and tabular data."""
    y_name, terms, _ = nextstat.formula.parse_formula(formula)
    extra = [entity] + ([time] if time is not None else [])
    cols: Mapping[str, Sequence[Any]] = nextstat.formula.to_columnar(data, [y_name] + terms + extra)
    y, x_full, names_full = nextstat.formula.design_matrices(formula, cols, categorical=categorical)

    # Drop intercept if present (absorbed by FE).
    if names_full and names_full[0] == "Intercept":
        x = [row[1:] for row in x_full]
        names = list(names_full[1:])
    else:
        x = x_full
        names = list(names_full)

    fit = panel_fe_fit(
        x,
        y,
        entity=cols[entity],
        time=(None if time is None else cols[time]),
        cluster=cluster,
    )
    return PanelFeFit(
        coef=fit.coef,
        standard_errors=fit.standard_errors,
        covariance=fit.covariance,
        column_names=names,
        n_obs=fit.n_obs,
        n_entities=fit.n_entities,
        cluster=fit.cluster,
    )


__all__ = ["PanelFeFit", "panel_fe_fit", "panel_fe_from_formula"]

