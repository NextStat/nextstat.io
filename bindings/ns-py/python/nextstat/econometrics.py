"""Econometrics helpers (Phase 12).

Current baseline:
- Panel linear regression with 1-way fixed effects (within estimator)
- Robust covariance options:
  - 1-way and 2-way cluster-robust standard errors
  - HAC / Newey-West for IV-2SLS
- Difference-in-Differences (DiD) + event-study helpers (TWFE baseline)
- Wild cluster bootstrap inference for DiD TWFE (Webb 6-point / Rademacher)
- Staggered-adoption DiD (group-time ATT, not-yet-treated controls)

Notes / limitations:
- This is a minimal baseline intended for small/medium problems.
- FE absorbs the intercept and any time-invariant regressors within each entity.
- DiD/event-study use a two-way fixed effects (TWFE) within transformation.
- Staggered DiD here is a transparent baseline (group-time ATT aggregation), not
  a full doubly-robust estimator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

import nextstat


ClusterKind = Literal["entity", "time", "two_way", "none"]
IvCovKind = Literal["homoskedastic", "hc1", "cluster", "hac"]
_GS_TOL = 1e-12
_TWOWAY_DEFAULT_TOL = 1e-10
_TWOWAY_DEFAULT_MAX_ITER = 5000


def _validate_lengths(n: int, *seqs: Sequence[Any]) -> None:
    for s in seqs:
        if len(s) != n:
            raise ValueError("length mismatch between inputs")


def _encode_groups(values: Sequence[Any]) -> Tuple[List[int], int]:
    # Fast path for integer-typed sequences via numpy.
    try:
        arr = np.asarray(values)
        if np.issubdtype(arr.dtype, np.integer):
            unique, inverse = np.unique(arr, return_inverse=True)
            return inverse.tolist(), int(len(unique))
    except (ValueError, TypeError):
        pass
    # General path for arbitrary hashable values.
    labels = [("None" if v is None else str(v)) for v in values]
    levels = sorted(set(labels))
    idx = {lvl: i for i, lvl in enumerate(levels)}
    return [idx[v] for v in labels], len(levels)


def _within_demean(
    y: Any, x: Any, entity: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = len(y)
    if n == 0:
        raise ValueError("need at least 1 observation")
    k = x.shape[1] if x.ndim == 2 else 0
    if k == 0:
        raise ValueError("X must have at least 1 column")
    if x.shape[0] != n:
        raise ValueError("X must be rectangular")
    _validate_lengths(n, entity)

    ent_idx, _n_ent = _encode_groups(entity)

    g = np.asarray(ent_idx, dtype=np.intp)
    cnt = np.bincount(g).astype(np.float64)
    y_mean_g = np.bincount(g, weights=y) / cnt
    x_mean_g = np.zeros((_n_ent, k), dtype=np.float64)
    for j in range(k):
        x_mean_g[:, j] = np.bincount(g, weights=x[:, j]) / cnt

    return y - y_mean_g[g], x - x_mean_g[g]


def _two_way_demean(
    y: Any, x: Any, entity: Sequence[Any], time: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = len(y)
    if n == 0:
        raise ValueError("need at least 1 observation")
    k = x.shape[1] if x.ndim == 2 else 0
    if k == 0:
        raise ValueError("X must have at least 1 column")
    if x.shape[0] != n:
        raise ValueError("X must be rectangular")
    _validate_lengths(n, entity, time)

    ent_idx, n_ent = _encode_groups(entity)
    time_idx, n_time = _encode_groups(time)

    g = np.asarray(ent_idx, dtype=np.intp)
    t = np.asarray(time_idx, dtype=np.intp)

    # Fast exact path for fully balanced panel; iterative alternating projections
    # for unbalanced panels (or duplicated cells).
    unique_cells = len(set(zip(g.tolist(), t.tolist())))
    is_balanced = (unique_cells == n) and (n == n_ent * n_time)

    if is_balanced:
        cnt_g = np.bincount(g).astype(np.float64)
        cnt_t = np.bincount(t).astype(np.float64)

        y_mean_g = np.bincount(g, weights=y) / cnt_g
        y_mean_t = np.bincount(t, weights=y) / cnt_t
        y_mean_all = y.mean()

        x_mean_g = np.zeros((n_ent, k), dtype=np.float64)
        x_mean_t = np.zeros((n_time, k), dtype=np.float64)
        for j in range(k):
            x_mean_g[:, j] = np.bincount(g, weights=x[:, j]) / cnt_g
            x_mean_t[:, j] = np.bincount(t, weights=x[:, j]) / cnt_t
        x_mean_all = x.mean(axis=0)

        y_dd = y - y_mean_g[g] - y_mean_t[t] + y_mean_all
        x_dd = x - x_mean_g[g] - x_mean_t[t] + x_mean_all
        return y_dd, x_dd, n_ent, n_time

    cnt_g = np.bincount(g, minlength=n_ent).astype(np.float64)
    cnt_t = np.bincount(t, minlength=n_time).astype(np.float64)
    if np.any(cnt_g <= 0.0) or np.any(cnt_t <= 0.0):
        raise ValueError("entity/time groups must be non-empty")

    y_dd = y.copy()
    x_dd = np.asarray(x, dtype=np.float64).copy()
    max_delta = float("inf")
    it = 0
    while it < _TWOWAY_DEFAULT_MAX_ITER and max_delta > _TWOWAY_DEFAULT_TOL:
        y_prev = y_dd.copy()
        x_prev = x_dd.copy()

        # Remove entity means.
        y_g = np.bincount(g, weights=y_dd, minlength=n_ent) / cnt_g
        y_dd = y_dd - y_g[g]
        for j in range(k):
            x_g = np.bincount(g, weights=x_dd[:, j], minlength=n_ent) / cnt_g
            x_dd[:, j] = x_dd[:, j] - x_g[g]

        # Remove time means.
        y_t = np.bincount(t, weights=y_dd, minlength=n_time) / cnt_t
        y_dd = y_dd - y_t[t]
        for j in range(k):
            x_t = np.bincount(t, weights=x_dd[:, j], minlength=n_time) / cnt_t
            x_dd[:, j] = x_dd[:, j] - x_t[t]

        dy = float(np.max(np.abs(y_dd - y_prev)))
        dx = float(np.max(np.abs(x_dd - x_prev)))
        max_delta = max(dy, dx)
        it += 1

    if not np.isfinite(max_delta) or max_delta > _TWOWAY_DEFAULT_TOL * 10.0:
        raise ValueError("two-way demeaning did not converge; check panel structure")

    return y_dd, x_dd, n_ent, n_time


def _select_independent_columns(
    x: Any,
    names: List[str],
    *,
    mandatory: Optional[Sequence[int]] = None,
    tol: float = _GS_TOL,
) -> Tuple[np.ndarray, List[str]]:
    """Select a numerically linearly independent subset of columns from X.

    Uses QR decomposition (LAPACK) for fast rank detection. Mandatory columns
    are placed first so they are never dropped in favour of later columns.
    """
    if x.size == 0:
        raise ValueError("X must be non-empty")
    k = x.shape[1]
    if k == 0:
        raise ValueError("X must have at least 1 column")
    if len(names) != k:
        raise ValueError("column_names length mismatch")

    mand = sorted(set(int(i) for i in (mandatory or []) if 0 <= int(i) < k))
    order = mand + [j for j in range(k) if j not in set(mand)]

    xa = np.ascontiguousarray(x[:, order], dtype=np.float64)
    _Q, R = np.linalg.qr(xa, mode='reduced')
    diag = np.abs(np.diag(R))

    kept_in_order = [order[j] for j in range(k) if diag[j] > tol]
    kept_sorted = sorted(kept_in_order)
    return np.ascontiguousarray(x[:, kept_sorted], dtype=np.float64), [names[j] for j in kept_sorted]


@dataclass(frozen=True)
class PanelFeFit:
    coef: List[float]
    standard_errors: List[float]
    covariance: List[List[float]]
    column_names: List[str]
    n_obs: int
    n_entities: int
    cluster: ClusterKind


@dataclass(frozen=True)
class TwfeFit:
    coef: List[float]
    standard_errors: List[float]
    covariance: List[List[float]]
    column_names: List[str]
    n_obs: int
    n_entities: int
    n_times: int
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
    x_np = np.asarray(x, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64)
    n = len(y_np)
    if n == 0:
        raise ValueError("need at least 1 observation")
    if x_np.ndim != 2 or x_np.shape[1] == 0:
        raise ValueError("X must be non-empty")
    if x_np.shape[0] != n:
        raise ValueError("X and y must have the same length")
    _validate_lengths(n, entity)
    if time is not None:
        _validate_lengths(n, time)

    y_star, x_star = _within_demean(y_np, x_np, entity)

    # Fast OLS via numpy — avoids PyO3 Vec<Vec<f64>> deserialization overhead.
    coef = np.linalg.lstsq(x_star, y_star, rcond=None)[0]
    resid = y_star - x_star @ coef

    cluster = str(cluster).lower()  # type: ignore[assignment]
    if cluster not in ("entity", "time", "two_way", "none"):
        raise ValueError("cluster must be one of: entity, time, two_way, none")

    if cluster == "none":
        k = x_star.shape[1]
        sigma2 = float(resid @ resid) / float(n - k)
        xtx_inv = np.linalg.inv(x_star.T @ x_star)
        cov_np = sigma2 * xtx_inv
        cov = cov_np.tolist()
        se = np.sqrt(np.diag(cov_np)).tolist()
        _ent_idx, n_ent = _encode_groups(entity)
    else:
        if cluster == "entity":
            groups = entity
            _ent_idx, n_ent = _encode_groups(entity)
            cov = nextstat.robust.ols_cluster_covariance(
                x_star,
                resid,
                groups,
                include_intercept=False,
                df_correction=True,
            )
        elif cluster == "time":
            if time is None:
                raise ValueError("time must be provided when cluster='time'")
            _ent_idx, n_ent = _encode_groups(entity)
            cov = nextstat.robust.ols_cluster_covariance(
                x_star,
                resid,
                time,
                include_intercept=False,
                df_correction=True,
            )
        else:
            if time is None:
                raise ValueError("time must be provided when cluster='two_way'")
            _ent_idx, n_ent = _encode_groups(entity)
            cov = nextstat.robust.ols_two_way_cluster_covariance(
                x_star,
                resid,
                entity,
                time,
                include_intercept=False,
                df_correction=True,
            )
        se = nextstat.robust.cov_to_se(cov)

    return PanelFeFit(
        coef=coef.tolist(),
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


def _twfe_fit_ols(
    x: np.ndarray,
    y: np.ndarray,
    *,
    entity: Sequence[Any],
    time: Sequence[Any],
    cluster: ClusterKind,
    column_names: List[str],
) -> TwfeFit:
    n = len(y)
    if n == 0:
        raise ValueError("need at least 1 observation")

    y_dd, x_dd, n_ent, n_time = _two_way_demean(y, x, entity, time)

    mandatory_idx: List[int] = []
    if "treat_post" in column_names:
        mandatory_idx = [column_names.index("treat_post")]

    x_sel, names_sel = _select_independent_columns(x_dd, list(column_names), mandatory=mandatory_idx)
    if mandatory_idx and "treat_post" not in names_sel:
        raise ValueError("treat_post is not identifiable (absorbed by FE or has no variation)")
    if n <= len(names_sel):
        raise ValueError("Need n > n_params after TWFE transformation; reduce controls or narrow event-study window")

    # Fast OLS via numpy — avoids PyO3 Vec<Vec<f64>> deserialization overhead.
    coef = np.linalg.lstsq(x_sel, y_dd, rcond=None)[0]
    resid = y_dd - x_sel @ coef

    cluster = str(cluster).lower()  # type: ignore[assignment]
    if cluster not in ("entity", "time", "two_way", "none"):
        raise ValueError("cluster must be one of: entity, time, two_way, none")

    if cluster == "none":
        k = x_sel.shape[1]
        sigma2 = float(resid @ resid) / float(n - k)
        xtx_inv = np.linalg.inv(x_sel.T @ x_sel)
        cov_np = sigma2 * xtx_inv
        cov = cov_np.tolist()
        se = np.sqrt(np.diag(cov_np)).tolist()
    else:
        if cluster == "entity":
            cov = nextstat.robust.ols_cluster_covariance(
                x_sel,
                resid,
                entity,
                include_intercept=False,
                df_correction=True,
            )
        elif cluster == "time":
            cov = nextstat.robust.ols_cluster_covariance(
                x_sel,
                resid,
                time,
                include_intercept=False,
                df_correction=True,
            )
        else:
            cov = nextstat.robust.ols_two_way_cluster_covariance(
                x_sel,
                resid,
                entity,
                time,
                include_intercept=False,
                df_correction=True,
            )
        se = nextstat.robust.cov_to_se(cov)

    return TwfeFit(
        coef=coef.tolist(),
        standard_errors=[float(v) for v in se],
        covariance=[[float(v) for v in row] for row in cov],
        column_names=list(names_sel),
        n_obs=n,
        n_entities=n_ent,
        n_times=n_time,
        cluster=cluster,  # type: ignore[arg-type]
    )


def did_regressor(treat: Sequence[Any], post: Sequence[Any]) -> List[float]:
    """Return the standard DiD regressor: 1[treat] * 1[post]."""
    _validate_lengths(len(treat), post)
    out: List[float] = []
    for a, b in zip(treat, post):
        out.append(float(bool(a)) * float(bool(b)))
    return out


@dataclass(frozen=True)
class DidTwfeFit:
    att: float
    att_se: float
    twfe: TwfeFit


def did_twfe_fit(
    x: Optional[Sequence[Sequence[float]]],
    y: Sequence[float],
    *,
    treat: Sequence[Any],
    post: Sequence[Any],
    entity: Sequence[Any],
    time: Sequence[Any],
    cluster: ClusterKind = "entity",
) -> DidTwfeFit:
    """Difference-in-Differences via TWFE (two-way fixed effects) baseline.

    Fits: y_it ~ alpha_i + gamma_t + ATT * (treat_i * post_t) + controls
    """
    y_np = np.asarray(y, dtype=np.float64)
    n = len(y_np)
    if n == 0:
        raise ValueError("need at least 1 observation")
    _validate_lengths(n, treat, post, entity, time)

    d = np.array([float(bool(a)) * float(bool(b)) for a, b in zip(treat, post)],
                 dtype=np.float64)
    if x is None:
        x_np = d.reshape(-1, 1)
        names = ["treat_post"]
    else:
        x_raw = np.asarray(x, dtype=np.float64)
        if x_raw.shape[0] != n:
            raise ValueError("X and y must have the same length")
        x_np = np.column_stack([d, x_raw])
        names = ["treat_post"] + [f"x{i}" for i in range(x_raw.shape[1])]

    twfe = _twfe_fit_ols(x_np, y_np, entity=entity, time=time, cluster=cluster, column_names=names)
    att = float(twfe.coef[0]) if twfe.coef else float("nan")
    att_se = float(twfe.standard_errors[0]) if twfe.standard_errors else float("nan")
    return DidTwfeFit(att=att, att_se=att_se, twfe=twfe)


def did_twfe_from_formula(
    formula: str,
    data: Any,
    *,
    entity: str,
    time: str,
    treat: str,
    post: str,
    categorical: Optional[Sequence[str]] = None,
    cluster: ClusterKind = "entity",
) -> DidTwfeFit:
    """DiD TWFE from a minimal formula plus tabular data.

    The formula provides outcome + optional controls. Intercept is ignored/absorbed by FE.
    """
    y_name, terms, _ = nextstat.formula.parse_formula(formula)
    cols: Mapping[str, Sequence[Any]] = nextstat.formula.to_columnar(
        data, [y_name] + terms + [entity, time, treat, post]
    )
    y_vals, x_full, names_full = nextstat.formula.design_matrices(formula, cols, categorical=categorical)

    # Drop intercept column (absorbed by FE).
    if names_full and names_full[0] == "Intercept":
        x_controls = [row[1:] for row in x_full]
        control_names = list(names_full[1:])
    else:
        x_controls = x_full
        control_names = list(names_full)

    x_arg: Optional[List[List[float]]] = None
    if control_names:
        x_arg = x_controls

    did = did_twfe_fit(
        x_arg,
        y_vals,
        treat=cols[treat],
        post=cols[post],
        entity=cols[entity],
        time=cols[time],
        cluster=cluster,
    )
    return did


WildWeightKind = Literal["webb6", "rademacher"]


@dataclass(frozen=True)
class DidTwfeWildBootstrap:
    att: float
    att_se_analytic: float
    att_se_bootstrap: float
    p_value: float
    ci_lower: float
    ci_upper: float
    n_boot: int
    cluster_on: Literal["entity", "time"]
    weight_dist: WildWeightKind


def _draw_wild_weights(rng: np.random.Generator, n_groups: int, weight_dist: WildWeightKind) -> np.ndarray:
    if weight_dist == "webb6":
        vals = np.array(
            [
                -math.sqrt(1.5),
                -1.0,
                -math.sqrt(0.5),
                math.sqrt(0.5),
                1.0,
                math.sqrt(1.5),
            ],
            dtype=np.float64,
        )
        return vals[rng.integers(0, len(vals), size=n_groups)]
    if weight_dist == "rademacher":
        return np.where(rng.random(size=n_groups) < 0.5, -1.0, 1.0).astype(np.float64)
    raise ValueError("weight_dist must be one of: webb6, rademacher")


def did_twfe_wild_cluster_bootstrap(
    x: Optional[Sequence[Sequence[float]]],
    y: Sequence[float],
    *,
    treat: Sequence[Any],
    post: Sequence[Any],
    entity: Sequence[Any],
    time: Sequence[Any],
    cluster_on: Literal["entity", "time"] = "entity",
    n_boot: int = 999,
    seed: int = 0,
    alpha: float = 0.05,
    weight_dist: WildWeightKind = "webb6",
) -> DidTwfeWildBootstrap:
    """Wild cluster bootstrap for DiD TWFE ATT (Webb 6-point or Rademacher).

    Uses:
    - unrestricted bootstrap distribution for bootstrap SE and percentile CI
    - restricted (ATT=0) bootstrap distribution for p-value
    """
    y_np = np.asarray(y, dtype=np.float64)
    n = len(y_np)
    if n == 0:
        raise ValueError("need at least 1 observation")
    _validate_lengths(n, treat, post, entity, time)
    if int(n_boot) < 50:
        raise ValueError("n_boot must be >= 50 for stable inference")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    d = np.array([float(bool(a)) * float(bool(b)) for a, b in zip(treat, post)], dtype=np.float64)
    if x is None:
        x_np = d.reshape(-1, 1)
        names = ["treat_post"]
    else:
        x_raw = np.asarray(x, dtype=np.float64)
        if x_raw.shape[0] != n:
            raise ValueError("X and y must have the same length")
        x_np = np.column_stack([d, x_raw])
        names = ["treat_post"] + [f"x{i}" for i in range(x_raw.shape[1])]

    y_dd, x_dd, _n_ent, _n_time = _two_way_demean(y_np, x_np, entity, time)
    x_sel, names_sel = _select_independent_columns(x_dd, names, mandatory=[0])
    if "treat_post" not in names_sel:
        raise ValueError("treat_post is not identifiable (absorbed by FE or has no variation)")
    att_idx = int(names_sel.index("treat_post"))

    coef_hat = np.linalg.lstsq(x_sel, y_dd, rcond=None)[0]
    y_hat = x_sel @ coef_hat
    resid = y_dd - y_hat
    att_hat = float(coef_hat[att_idx])

    cluster_on = str(cluster_on).lower()  # type: ignore[assignment]
    if cluster_on not in ("entity", "time"):
        raise ValueError("cluster_on must be one of: entity, time")
    groups = entity if cluster_on == "entity" else time
    g_idx, n_groups = _encode_groups(groups)
    if n_groups < 6 and weight_dist == "webb6":
        raise ValueError("Webb 6-point weights require at least 6 clusters")

    # Restricted model for bootstrap p-value under H0: ATT = 0.
    keep = [j for j in range(x_sel.shape[1]) if j != att_idx]
    if keep:
        x_restricted = x_sel[:, keep]
        beta_r = np.linalg.lstsq(x_restricted, y_dd, rcond=None)[0]
        y_hat_null = x_restricted @ beta_r
    else:
        y_hat_null = np.zeros_like(y_dd)
    resid_null = y_dd - y_hat_null

    rng = np.random.default_rng(int(seed))
    att_unres = np.zeros(int(n_boot), dtype=np.float64)
    att_null = np.zeros(int(n_boot), dtype=np.float64)
    g_arr = np.asarray(g_idx, dtype=np.intp)
    for b in range(int(n_boot)):
        w = _draw_wild_weights(rng, n_groups, weight_dist)[g_arr]

        y_star = y_hat + resid * w
        beta_star = np.linalg.lstsq(x_sel, y_star, rcond=None)[0]
        att_unres[b] = float(beta_star[att_idx])

        y_star0 = y_hat_null + resid_null * w
        beta_star0 = np.linalg.lstsq(x_sel, y_star0, rcond=None)[0]
        att_null[b] = float(beta_star0[att_idx])

    did = did_twfe_fit(
        x,
        y,
        treat=treat,
        post=post,
        entity=entity,
        time=time,
        cluster=cluster_on,  # type: ignore[arg-type]
    )
    se_boot = float(np.std(att_unres, ddof=1))
    p_val = float(np.mean(np.abs(att_null) >= abs(att_hat)))
    lo = float(np.quantile(att_unres, float(alpha) / 2.0))
    hi = float(np.quantile(att_unres, 1.0 - (float(alpha) / 2.0)))

    return DidTwfeWildBootstrap(
        att=att_hat,
        att_se_analytic=float(did.att_se),
        att_se_bootstrap=se_boot,
        p_value=p_val,
        ci_lower=lo,
        ci_upper=hi,
        n_boot=int(n_boot),
        cluster_on=cluster_on,  # type: ignore[arg-type]
        weight_dist=weight_dist,
    )


ControlGroupKind = Literal["not_yet_treated", "never_treated"]


@dataclass(frozen=True)
class StaggeredDidCell:
    cohort_time: int
    time: int
    rel_time: int
    att: float
    standard_error: float
    n_treated: int
    n_control: int


@dataclass(frozen=True)
class StaggeredDidFit:
    att: float
    att_se: float
    event_times: List[int]
    event_att: List[float]
    event_se: List[float]
    cells: List[StaggeredDidCell]
    n_obs: int
    n_entities: int
    n_times: int
    control_group: ControlGroupKind


def _mean_and_se_diff(a: List[float], b: List[float]) -> Tuple[float, float]:
    if not a or not b:
        return float("nan"), float("nan")
    ma = float(sum(a) / float(len(a)))
    mb = float(sum(b) / float(len(b)))
    va = float(np.var(np.asarray(a, dtype=np.float64), ddof=1)) if len(a) > 1 else 0.0
    vb = float(np.var(np.asarray(b, dtype=np.float64), ddof=1)) if len(b) > 1 else 0.0
    se = math.sqrt(max(0.0, (va / float(len(a))) + (vb / float(len(b)))))
    return ma - mb, se


def _stable_unique_sorted(values: Sequence[Any]) -> List[Any]:
    levels = list(set(values))
    try:
        return sorted(levels)
    except TypeError:
        return sorted(levels, key=lambda v: str(v))


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(np.isnan(v))
    except Exception:
        return False


def staggered_did_fit(
    y: Sequence[float],
    *,
    entity: Sequence[Any],
    time: Sequence[Any],
    treat: Optional[Sequence[Any]] = None,
    cohort: Optional[Sequence[Any]] = None,
    control_group: ControlGroupKind = "not_yet_treated",
) -> StaggeredDidFit:
    """Staggered-adoption DiD via group-time ATT (baseline Callaway-Sant'Anna style).

    Exactly one of `treat` or `cohort` must be provided.
    - `treat`: binary per-observation indicator (cohort inferred as first treated period per entity)
    - `cohort`: first-treated period per observation/entity; use None/NaN for never-treated
    """
    y_np = np.asarray(y, dtype=np.float64)
    n = len(y_np)
    if n == 0:
        raise ValueError("need at least 1 observation")
    _validate_lengths(n, entity, time)
    if (treat is None) == (cohort is None):
        raise ValueError("provide exactly one of: treat or cohort")

    if control_group not in ("not_yet_treated", "never_treated"):
        raise ValueError("control_group must be one of: not_yet_treated, never_treated")

    t_levels = _stable_unique_sorted(time)
    t_to_idx = {tv: i for i, tv in enumerate(t_levels)}
    t_idx = [int(t_to_idx[tv]) for tv in time]
    n_t = len(t_levels)

    # Build panel map and reject duplicates (entity,time).
    y_by_entity: dict[Any, dict[int, float]] = {}
    for yi, ei, ti in zip(y_np, entity, t_idx):
        by_t = y_by_entity.get(ei)
        if by_t is None:
            by_t = {}
            y_by_entity[ei] = by_t
        if ti in by_t:
            raise ValueError("duplicate observations for the same (entity, time)")
        by_t[ti] = float(yi)

    entities = list(y_by_entity.keys())
    n_entities = len(entities)

    # Infer / validate first-treatment cohort per entity in index-time scale.
    g_by_e: dict[Any, int] = {}
    if treat is not None:
        _validate_lengths(n, treat)
        for ei in entities:
            g = n_t  # never treated
            for tr, e_obs, tt in zip(treat, entity, t_idx):
                if e_obs != ei:
                    continue
                if bool(tr):
                    g = min(g, int(tt))
            g_by_e[ei] = g
    else:
        assert cohort is not None
        _validate_lengths(n, cohort)
        tmp: dict[Any, int] = {}
        for e_obs, c_obs in zip(entity, cohort):
            if _is_missing(c_obs):
                g_val = n_t
            else:
                if c_obs not in t_to_idx:
                    raise ValueError("cohort contains time values not present in time")
                g_val = int(t_to_idx[c_obs])
            prev = tmp.get(e_obs)
            if prev is None:
                tmp[e_obs] = g_val
            elif prev != g_val:
                raise ValueError("cohort must be constant within each entity")
        g_by_e = tmp

    cohorts = sorted({g for g in g_by_e.values() if g < n_t})
    cells: List[StaggeredDidCell] = []
    for g in cohorts:
        pre_t = g - 1
        if pre_t < 0:
            continue
        for t_cur in range(g, n_t):
            treated_deltas: List[float] = []
            control_deltas: List[float] = []
            for ei in entities:
                yi_map = y_by_entity[ei]
                if pre_t not in yi_map or t_cur not in yi_map:
                    continue
                delta = float(yi_map[t_cur] - yi_map[pre_t])
                g_i = int(g_by_e[ei])
                if g_i == g:
                    treated_deltas.append(delta)
                    continue
                if control_group == "not_yet_treated":
                    if g_i > t_cur:
                        control_deltas.append(delta)
                else:
                    if g_i >= n_t:
                        control_deltas.append(delta)

            if not treated_deltas or not control_deltas:
                continue
            att_gt, se_gt = _mean_and_se_diff(treated_deltas, control_deltas)
            cells.append(
                StaggeredDidCell(
                    cohort_time=int(g),
                    time=int(t_cur),
                    rel_time=int(t_cur - g),
                    att=float(att_gt),
                    standard_error=float(se_gt),
                    n_treated=len(treated_deltas),
                    n_control=len(control_deltas),
                )
            )

    if not cells:
        raise ValueError("no identifiable staggered DiD cells; check treatment timing and controls")

    w = np.asarray([float(c.n_treated) for c in cells], dtype=np.float64)
    w_sum = float(np.sum(w))
    att = float(np.sum(w * np.asarray([c.att for c in cells], dtype=np.float64)) / w_sum)
    att_se = float(
        math.sqrt(
            max(
                0.0,
                float(np.sum(((w / w_sum) ** 2) * np.asarray([c.standard_error ** 2 for c in cells], dtype=np.float64))),
            )
        )
    )

    rel_levels = sorted({c.rel_time for c in cells})
    event_att: List[float] = []
    event_se: List[float] = []
    for rel in rel_levels:
        rel_cells = [c for c in cells if c.rel_time == rel]
        wr = np.asarray([float(c.n_treated) for c in rel_cells], dtype=np.float64)
        wr_sum = float(np.sum(wr))
        a = float(np.sum(wr * np.asarray([c.att for c in rel_cells], dtype=np.float64)) / wr_sum)
        s = float(
            math.sqrt(
                max(
                    0.0,
                    float(np.sum(((wr / wr_sum) ** 2) * np.asarray([c.standard_error ** 2 for c in rel_cells], dtype=np.float64))),
                )
            )
        )
        event_att.append(a)
        event_se.append(s)

    return StaggeredDidFit(
        att=att,
        att_se=att_se,
        event_times=rel_levels,
        event_att=event_att,
        event_se=event_se,
        cells=cells,
        n_obs=n,
        n_entities=n_entities,
        n_times=n_t,
        control_group=control_group,
    )


def staggered_did_from_formula(
    formula: str,
    data: Any,
    *,
    entity: str,
    time: str,
    treat: Optional[str] = None,
    cohort: Optional[str] = None,
    control_group: ControlGroupKind = "not_yet_treated",
) -> StaggeredDidFit:
    """Staggered-adoption DiD from formula + tabular data."""
    y_name, terms, _ = nextstat.formula.parse_formula(formula)
    if terms and not (len(terms) == 1 and terms[0] == "1"):
        raise ValueError("staggered_did_from_formula currently supports outcome-only formula: 'y ~ 1'")
    if (treat is None) == (cohort is None):
        raise ValueError("provide exactly one of: treat or cohort")

    cols_needed = [y_name, entity, time]
    if treat is not None:
        cols_needed.append(treat)
    if cohort is not None:
        cols_needed.append(cohort)
    cols: Mapping[str, Sequence[Any]] = nextstat.formula.to_columnar(data, cols_needed)

    return staggered_did_fit(
        cols[y_name],
        entity=cols[entity],
        time=cols[time],
        treat=(None if treat is None else cols[treat]),
        cohort=(None if cohort is None else cols[cohort]),
        control_group=control_group,
    )


def relative_time(time: Sequence[Any], event_time: Union[int, float, Sequence[Any]]) -> List[int]:
    """Compute integer relative time = t - t_event."""
    if isinstance(event_time, (int, float)):
        out: List[int] = []
        for t in time:
            out.append(int(t) - int(event_time))
        return out

    _validate_lengths(len(time), event_time)
    out2: List[int] = []
    for t, e in zip(time, event_time):
        out2.append(int(t) - int(e))
    return out2


def event_study_regressors(
    treat: Sequence[Any],
    rel_time: Sequence[int],
    *,
    window: Tuple[int, int] = (-4, 4),
    reference: int = -1,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Build event-study regressors: treat * 1[rel_time == k] for k in window, excluding reference."""
    lo, hi = int(window[0]), int(window[1])
    if lo > hi:
        raise ValueError("window must satisfy lo <= hi")

    ks = [k for k in range(lo, hi + 1) if k != int(reference)]
    _validate_lengths(len(treat), rel_time)

    n = len(treat)
    treat_arr = np.array([float(bool(a)) for a in treat], dtype=np.float64)
    rt_arr = np.asarray(rel_time, dtype=np.intp)

    # Vectorised dummy creation.
    x = np.zeros((n, len(ks)), dtype=np.float64)
    for j, k in enumerate(ks):
        x[:, j] = treat_arr * (rt_arr == k).astype(np.float64)

    # Drop bins with no support to avoid singular designs.
    col_has_support = x.any(axis=0)
    keep = np.where(col_has_support)[0]

    x2 = x[:, keep]
    ks2 = [ks[int(j)] for j in keep]
    names = [f"event[{k}]" for k in ks2]
    return x2, names, ks2


@dataclass(frozen=True)
class EventStudyTwfeFit:
    rel_times: List[int]
    coef: List[float]
    standard_errors: List[float]
    covariance: List[List[float]]
    reference: int
    n_obs: int
    n_entities: int
    n_times: int
    cluster: ClusterKind


def event_study_twfe_fit(
    y: Sequence[float],
    *,
    treat: Sequence[Any],
    time: Sequence[Any],
    event_time: Union[int, float, Sequence[Any]],
    entity: Sequence[Any],
    window: Tuple[int, int] = (-4, 4),
    reference: int = -1,
    x: Optional[Sequence[Sequence[float]]] = None,
    cluster: ClusterKind = "entity",
) -> EventStudyTwfeFit:
    """Event-study via TWFE baseline.

    Fits: y_it ~ alpha_i + gamma_t + sum_k beta_k * treat_i * 1[rel_time==k] + controls
    """
    y_np = np.asarray(y, dtype=np.float64)
    n = len(y_np)
    if n == 0:
        raise ValueError("need at least 1 observation")
    _validate_lengths(n, treat, time, entity)

    rel = relative_time(time, event_time)
    x_ev, names_ev, ks = event_study_regressors(treat, rel, window=window, reference=reference)
    if not ks:
        raise ValueError("no supported event-study bins in the requested window")

    if x is None:
        x_all = x_ev
        names = names_ev
    else:
        x_raw = np.asarray(x, dtype=np.float64)
        if x_raw.shape[0] != n:
            raise ValueError("X and y must have the same length")
        x_all = np.column_stack([x_ev, x_raw])
        names = names_ev + [f"x{i}" for i in range(x_raw.shape[1])]

    twfe = _twfe_fit_ols(x_all, y_np, entity=entity, time=time, cluster=cluster, column_names=names)
    n_ev = 0
    for nm in twfe.column_names:
        if nm.startswith("event[") and nm.endswith("]"):
            n_ev += 1
        else:
            break
    rel_kept: List[int] = []
    for nm in twfe.column_names[:n_ev]:
        rel_kept.append(int(nm[len("event[") : -1]))
    return EventStudyTwfeFit(
        rel_times=rel_kept,
        coef=list(twfe.coef[:n_ev]),
        standard_errors=list(twfe.standard_errors[:n_ev]),
        covariance=[list(r[:n_ev]) for r in twfe.covariance[:n_ev]],
        reference=int(reference),
        n_obs=twfe.n_obs,
        n_entities=twfe.n_entities,
        n_times=twfe.n_times,
        cluster=twfe.cluster,
    )


def event_study_twfe_from_formula(
    formula: str,
    data: Any,
    *,
    entity: str,
    time: str,
    treat: str,
    event_time: Union[int, float, str],
    window: Tuple[int, int] = (-4, 4),
    reference: int = -1,
    categorical: Optional[Sequence[str]] = None,
    cluster: ClusterKind = "entity",
) -> EventStudyTwfeFit:
    """Event-study TWFE from a formula plus tabular data.

    The formula provides outcome + optional controls. Intercept is ignored/absorbed by FE.
    `event_time` may be a scalar or a column name in `data`.
    """
    y_name, terms, _ = nextstat.formula.parse_formula(formula)
    extra_cols: List[str] = [entity, time, treat]
    if isinstance(event_time, str):
        extra_cols.append(event_time)
    cols: Mapping[str, Sequence[Any]] = nextstat.formula.to_columnar(data, [y_name] + terms + extra_cols)
    y_vals, x_full, names_full = nextstat.formula.design_matrices(formula, cols, categorical=categorical)

    if names_full and names_full[0] == "Intercept":
        x_controls = [row[1:] for row in x_full]
    else:
        x_controls = x_full

    x_arg: Optional[List[List[float]]] = None
    if x_controls and (len(x_controls[0]) > 0):
        x_arg = x_controls

    et: Union[int, float, Sequence[Any]] = cols[event_time] if isinstance(event_time, str) else event_time

    return event_study_twfe_fit(
        y_vals,
        treat=cols[treat],
        time=cols[time],
        event_time=et,
        entity=cols[entity],
        window=window,
        reference=reference,
        x=x_arg,
        cluster=cluster,
    )


@dataclass(frozen=True)
class WeakIvDiagnostics:
    excluded_instruments: List[str]
    first_stage_f: List[float]
    first_stage_partial_r2: List[float]
    n_obs: int


@dataclass(frozen=True)
class Iv2slsFit:
    coef: List[float]
    standard_errors: List[float]
    covariance: List[List[float]]
    column_names: List[str]
    n_obs: int
    df_resid: int
    diagnostics: WeakIvDiagnostics


def _require_full_rank(x: np.ndarray, names: List[str], what: str) -> None:
    if x.size == 0:
        raise ValueError(f"{what} must be non-empty")
    k = x.shape[1]
    xs, _ns = _select_independent_columns(x, names, mandatory=list(range(k)))
    if xs.shape[1] != k:
        raise ValueError(f"{what} is rank-deficient / collinear")


def iv_2sls_fit(
    y: Sequence[float],
    *,
    endog: Sequence[Sequence[float]],
    instruments: Sequence[Sequence[float]],
    exog: Optional[Sequence[Sequence[float]]] = None,
    endog_names: Optional[Sequence[str]] = None,
    exog_names: Optional[Sequence[str]] = None,
    instrument_names: Optional[Sequence[str]] = None,
    cov: IvCovKind = "hc1",
    cluster: Optional[Sequence[Any]] = None,
    time_index: Optional[Sequence[Any]] = None,
    max_lag: Optional[int] = None,
    df_correction: bool = True,
) -> Iv2slsFit:
    """Baseline IV / 2SLS for linear models (Phase 12).

    Structural equation: y = X_endog * beta + X_exog * gamma + u
    Instruments: Z = [instruments, X_exog]
    """
    y_np = np.asarray(y, dtype=np.float64)
    n = len(y_np)
    if n == 0:
        raise ValueError("need at least 1 observation")

    x_endog = np.asarray(endog, dtype=np.float64)
    z_excl = np.asarray(instruments, dtype=np.float64)
    if x_endog.shape[0] != n or z_excl.shape[0] != n:
        raise ValueError("length mismatch between y/endog/instruments")

    p_endog = x_endog.shape[1] if x_endog.ndim == 2 else 0
    if p_endog == 0:
        raise ValueError("endog must have at least 1 column")

    q = z_excl.shape[1] if z_excl.ndim == 2 else 0
    if q == 0:
        raise ValueError("instruments must have at least 1 column")
    if q < p_endog:
        raise ValueError("underidentified: need at least as many excluded instruments as endogenous regressors")

    if exog is None:
        x_exog = np.zeros((n, 0), dtype=np.float64)
        p_exog = 0
    else:
        x_exog = np.asarray(exog, dtype=np.float64)
        if x_exog.shape[0] != n:
            raise ValueError("length mismatch between y and exog")
        p_exog = x_exog.shape[1] if x_exog.ndim == 2 else 0
        if p_exog == 0:
            raise ValueError("exog must have at least 1 column when provided")

    endog_names2 = [f"endog{i}" for i in range(p_endog)] if endog_names is None else [str(s) for s in endog_names]
    if len(endog_names2) != p_endog:
        raise ValueError("endog_names length mismatch")

    exog_names2 = [f"x{i}" for i in range(p_exog)] if exog_names is None else [str(s) for s in exog_names]
    if len(exog_names2) != p_exog:
        raise ValueError("exog_names length mismatch")

    instr_names2 = [f"z{i}" for i in range(q)] if instrument_names is None else [str(s) for s in instrument_names]
    if len(instr_names2) != q:
        raise ValueError("instrument_names length mismatch")

    # Construct full X = [endog | exog] and Z = [instruments | exog].
    x_np = np.column_stack([x_endog, x_exog]) if p_exog > 0 else x_endog.copy()
    z_full = np.column_stack([z_excl, x_exog]) if p_exog > 0 else z_excl.copy()

    x_names = endog_names2 + exog_names2
    z_names = instr_names2 + exog_names2

    _require_full_rank(x_np, x_names, "X")

    exog_idx = list(range(q, q + p_exog))
    z_sel, z_names_sel = _select_independent_columns(z_full, z_names, mandatory=exog_idx)
    if p_exog > 0 and any(nm not in z_names_sel for nm in exog_names2):
        raise ValueError("exog columns are collinear in Z")

    excluded_kept = [nm for nm in z_names_sel if nm in instr_names2]

    kx = x_np.shape[1]
    kz = z_sel.shape[1]
    if n <= kx:
        raise ValueError("Need n > n_params to estimate sigma2_hat")
    if kz < kx:
        raise ValueError("underidentified after dropping collinear instruments")

    # --- 2SLS: all numpy, zero intermediate list conversions ---
    ztz_inv = np.linalg.inv(z_sel.T @ z_sel)
    xtz = x_np.T @ z_sel           # (kx, kz)
    xz_inv = xtz @ ztz_inv          # (kx, kz)
    a = xz_inv @ xtz.T              # (kx, kx)
    a_inv = np.linalg.inv(a)

    zty = z_sel.T @ y_np            # (kz,)
    rhs = xz_inv @ zty              # (kx,)
    beta = a_inv @ rhs              # (kx,)

    yhat = x_np @ beta
    resid = y_np - yhat

    cov = str(cov).lower()  # type: ignore[assignment]
    if cov not in ("homoskedastic", "hc1", "cluster", "hac"):
        raise ValueError("cov must be one of: homoskedastic, hc1, cluster, hac")

    if cov == "homoskedastic":
        sigma2 = float(resid @ resid) / float(n - kx)
        cov_beta = (sigma2 * a_inv).tolist()
    else:
        if cov == "hc1":
            zu = z_sel * resid[:, None]        # (n, kz)
            meat = zu.T @ zu                    # (kz, kz)
            if df_correction and n > kx:
                meat *= float(n) / float(n - kx)
        elif cov == "cluster":
            if cluster is None:
                raise ValueError("cluster must be provided when cov='cluster'")
            _validate_lengths(n, cluster)
            cl_idx, n_cl = _encode_groups(cluster)
            if n_cl < 2:
                raise ValueError("cluster must have at least 2 distinct groups")
            ga = np.asarray(cl_idx, dtype=np.intp)
            zu = z_sel * resid[:, None]
            sg = np.zeros((n_cl, kz), dtype=np.float64)
            for j in range(kz):
                sg[:, j] = np.bincount(ga, weights=zu[:, j], minlength=n_cl)
            meat = sg.T @ sg
            if df_correction and n > kx:
                scale = (float(n_cl) / float(n_cl - 1)) * ((float(n) - 1.0) / float(n - kx))
                meat *= scale
        else:
            zu = z_sel * resid[:, None]  # (n, kz)
            if time_index is not None:
                _validate_lengths(n, time_index)
                try:
                    ord_idx = sorted(range(n), key=lambda i: (time_index[i], i))
                except TypeError:
                    ord_idx = sorted(range(n), key=lambda i: (str(time_index[i]), i))
                zu = zu[np.asarray(ord_idx, dtype=np.intp), :]

            lag = int(math.floor(4.0 * (float(n) / 100.0) ** (2.0 / 9.0))) if max_lag is None else int(max_lag)
            lag = max(0, min(lag, n - 1))

            meat = zu.T @ zu
            for ell in range(1, lag + 1):
                w = 1.0 - (float(ell) / float(lag + 1))
                gamma = zu[ell:, :].T @ zu[:-ell, :]
                meat += w * (gamma + gamma.T)
            if df_correction and n > kx:
                meat *= float(n) / float(n - kx)

        # Sandwich: cov_beta = a_inv @ (xz_inv @ meat @ ztz_inv @ xtz.T) @ a_inv
        b = xz_inv @ meat @ ztz_inv @ xtz.T
        cov_beta = (a_inv @ b @ a_inv).tolist()

    se = nextstat.robust.cov_to_se(cov_beta)

    # First-stage diagnostics.
    exog_sel_idx = [i for i, nm in enumerate(z_names_sel) if nm in exog_names2]
    z_exog_only = z_sel[:, exog_sel_idx] if exog_sel_idx else None

    f_stats: List[float] = []
    partial_r2s: List[float] = []
    excl_idx = [i for i, nm in enumerate(z_names_sel) if nm in instr_names2]
    q_kept = len(excl_idx)
    k_ur = kz
    df2 = n - k_ur

    for j in range(p_endog):
        dcol_np = x_endog[:, j]
        if q_kept == 0 or df2 <= 0:
            f_stats.append(float("nan"))
            partial_r2s.append(float("nan"))
            continue

        try:
            beta_ur = np.linalg.lstsq(z_sel, dcol_np, rcond=None)[0]
            diff_ur = dcol_np - z_sel @ beta_ur
            ssr_ur = float(diff_ur @ diff_ur)
        except Exception:
            f_stats.append(float("nan"))
            partial_r2s.append(float("nan"))
            continue

        if z_exog_only is None or z_exog_only.shape[1] == 0:
            ssr_r = float(dcol_np @ dcol_np)
        else:
            try:
                beta_r = np.linalg.lstsq(z_exog_only, dcol_np, rcond=None)[0]
                diff_r = dcol_np - z_exog_only @ beta_r
                ssr_r = float(diff_r @ diff_r)
            except Exception:
                ssr_r = float("nan")

        if not (math.isfinite(ssr_r) and math.isfinite(ssr_ur)) or not (ssr_ur > 0.0) or not (ssr_r >= ssr_ur):
            f_stats.append(float("nan"))
            partial_r2s.append(float("nan"))
            continue

        num = (ssr_r - ssr_ur) / float(q_kept)
        den = ssr_ur / float(df2) if df2 > 0 else float("nan")
        f_stats.append(float(num / den) if (den > 0.0) else float("nan"))
        partial_r2s.append(float(1.0 - (ssr_ur / ssr_r)) if (ssr_r > 0.0) else float("nan"))

    diag = WeakIvDiagnostics(
        excluded_instruments=list(excluded_kept),
        first_stage_f=f_stats,
        first_stage_partial_r2=partial_r2s,
        n_obs=n,
    )

    return Iv2slsFit(
        coef=beta.tolist(),
        standard_errors=[float(v) for v in se],
        covariance=[[float(v) for v in row] for row in cov_beta],
        column_names=list(x_names),
        n_obs=n,
        df_resid=int(n - kx),
        diagnostics=diag,
    )


def iv_2sls_from_formula(
    formula: str,
    data: Any,
    *,
    endog: Union[str, Sequence[str]],
    instruments: Sequence[str],
    categorical: Optional[Sequence[str]] = None,
    cluster: Optional[Union[str, Sequence[Any]]] = None,
    time_index: Optional[Union[str, Sequence[Any]]] = None,
    cov: IvCovKind = "hc1",
    max_lag: Optional[int] = None,
    df_correction: bool = True,
) -> Iv2slsFit:
    """2SLS from a minimal exogenous formula plus tabular data."""
    endog_names = [str(endog)] if isinstance(endog, str) else [str(s) for s in endog]
    instr_names = [str(s) for s in instruments]
    if not instr_names:
        raise ValueError("instruments must be non-empty")

    y_name, terms, _ = nextstat.formula.parse_formula(formula)
    cols_needed = [y_name] + terms + endog_names + instr_names
    if isinstance(cluster, str):
        cols_needed.append(cluster)
    if isinstance(time_index, str):
        cols_needed.append(time_index)
    cols: Mapping[str, Sequence[Any]] = nextstat.formula.to_columnar(data, cols_needed)

    y_vals, x_exog, exog_names = nextstat.formula.design_matrices(formula, cols, categorical=categorical)
    n = len(y_vals)

    # Vectorised column extraction via numpy.
    x_endog = np.column_stack([np.asarray(cols[nm], dtype=np.float64) for nm in endog_names])
    z_excl = np.column_stack([np.asarray(cols[nm], dtype=np.float64) for nm in instr_names])

    cluster_vals: Optional[Sequence[Any]] = None
    if cov == "cluster":
        if cluster is None:
            raise ValueError("cluster must be provided when cov='cluster'")
        cluster_vals = cols[cluster] if isinstance(cluster, str) else cluster

    time_vals: Optional[Sequence[Any]] = None
    if cov == "hac":
        if time_index is not None:
            time_vals = cols[time_index] if isinstance(time_index, str) else time_index

    return iv_2sls_fit(
        y_vals,
        endog=x_endog,
        instruments=z_excl,
        exog=x_exog,
        endog_names=endog_names,
        exog_names=exog_names,
        instrument_names=instr_names,
        cov=cov,
        cluster=cluster_vals,
        time_index=time_vals,
        max_lag=max_lag,
        df_correction=df_correction,
    )


__all__ = [
    "PanelFeFit",
    "panel_fe_fit",
    "panel_fe_from_formula",
    "TwfeFit",
    "did_regressor",
    "DidTwfeFit",
    "did_twfe_fit",
    "did_twfe_from_formula",
    "WildWeightKind",
    "DidTwfeWildBootstrap",
    "did_twfe_wild_cluster_bootstrap",
    "ControlGroupKind",
    "StaggeredDidCell",
    "StaggeredDidFit",
    "staggered_did_fit",
    "staggered_did_from_formula",
    "relative_time",
    "event_study_regressors",
    "EventStudyTwfeFit",
    "event_study_twfe_fit",
    "event_study_twfe_from_formula",
    "WeakIvDiagnostics",
    "Iv2slsFit",
    "IvCovKind",
    "iv_2sls_fit",
    "iv_2sls_from_formula",
]
