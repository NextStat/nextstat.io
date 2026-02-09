"""Panel-data estimators (Phase 12).

Currently includes a baseline 1-way fixed effects (within) estimator for linear
regression with 1-way cluster-robust standard errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

from .glm._linalg import as_2d_float_list, mat_mul, mat_t, mat_vec_mul, solve_linear
from .robust import cov_to_se, ols_cluster_covariance


@dataclass(frozen=True)
class PanelFixedEffectsFit:
    coef: List[float]
    standard_errors: List[float]
    covariance: List[List[float]]
    n_obs: int
    n_entities: int
    cluster_kind: str
    n_clusters: int


def _encode_levels(values: Sequence[Any]) -> tuple[list[int], int]:
    idx: dict[Any, int] = {}
    out: list[int] = []
    for v in values:
        if v not in idx:
            idx[v] = len(idx)
        out.append(idx[v])
    return out, len(idx)


def fit_fixed_effects(
    x: Sequence[Sequence[float]],
    y: Sequence[float],
    *,
    entity: Sequence[Any],
    time: Optional[Sequence[Any]] = None,
    cluster: Union[str, Sequence[Any]] = "entity",
    df_correction: bool = True,
) -> PanelFixedEffectsFit:
    """Fit a 1-way entity fixed effects model via the within estimator.

    Model:
        y_it = alpha_i + x_it' beta + e_it

    Args
    - x: feature matrix (no intercept column).
    - entity: entity/group labels, length n.
    - time: optional time labels, length n (only needed for cluster='time').
    - cluster:
        - 'entity' (default): cluster-robust SE by entity
        - 'time': cluster-robust SE by time (requires time=...)
        - sequence: explicit cluster labels, length n
    """
    x2 = as_2d_float_list(x)
    y2 = [float(v) for v in y]
    n = len(y2)
    if n == 0:
        raise ValueError("need at least 1 observation")
    if len(x2) != n or len(entity) != n:
        raise ValueError("x, y, and entity must have the same length")

    p = len(x2[0]) if x2 else 0
    if any(len(row) != p for row in x2):
        raise ValueError("x must be rectangular")
    if p == 0:
        raise ValueError("x must have at least 1 feature column")

    ent_idx, n_ent = _encode_levels(entity)
    if n_ent < 2:
        raise ValueError("need at least 2 entities for fixed effects")

    sum_y = [0.0] * n_ent
    sum_x = [[0.0] * p for _ in range(n_ent)]
    count = [0] * n_ent

    for i, g in enumerate(ent_idx):
        sum_y[g] += y2[i]
        for j in range(p):
            sum_x[g][j] += x2[i][j]
        count[g] += 1

    mean_y = [sum_y[g] / float(count[g]) for g in range(n_ent)]
    mean_x = [[sum_x[g][j] / float(count[g]) for j in range(p)] for g in range(n_ent)]

    y_tilde: list[float] = []
    x_tilde: list[list[float]] = []
    for i, g in enumerate(ent_idx):
        y_tilde.append(y2[i] - mean_y[g])
        x_tilde.append([x2[i][j] - mean_x[g][j] for j in range(p)])

    # OLS on transformed data (no intercept).
    xt = mat_t(x_tilde)
    xtx = mat_mul(xt, x_tilde)
    xty = mat_vec_mul(xt, y_tilde)
    beta = solve_linear(xtx, xty)

    y_hat = mat_vec_mul(x_tilde, beta)
    resid = [obs - pred for obs, pred in zip(y_tilde, y_hat)]

    if isinstance(cluster, str):
        if cluster == "entity":
            cl = ent_idx
            kind = "entity"
        elif cluster == "time":
            if time is None:
                raise ValueError("cluster='time' requires time=")
            cl, _ = _encode_levels(time)
            kind = "time"
        else:
            raise ValueError("cluster must be 'entity', 'time', or a sequence")
    else:
        if len(cluster) != n:
            raise ValueError("cluster labels must have length n")
        cl, _ = _encode_levels(cluster)
        kind = "custom"

    n_clusters = len(set(cl))
    cov = ols_cluster_covariance(x_tilde, resid, cl, include_intercept=False, df_correction=bool(df_correction))
    se = cov_to_se(cov)

    return PanelFixedEffectsFit(
        coef=list(beta),
        standard_errors=se,
        covariance=cov,
        n_obs=n,
        n_entities=n_ent,
        cluster_kind=kind,
        n_clusters=n_clusters,
    )


__all__ = ["PanelFixedEffectsFit", "fit_fixed_effects"]

