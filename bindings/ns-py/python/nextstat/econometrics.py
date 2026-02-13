"""Econometrics helpers (Phase 12).

Current baseline:
- Panel linear regression with 1-way fixed effects (within estimator)
- 1-way cluster-robust standard errors (entity or time)
- Difference-in-Differences (DiD) + event-study helpers (TWFE baseline)

Notes / limitations:
- This is a minimal baseline intended for small/medium problems.
- FE absorbs the intercept and any time-invariant regressors within each entity.
- Cluster-robust SE is 1-way (no 2-way clustering yet).
- DiD/event-study use a two-way fixed effects (TWFE) within transformation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

import nextstat


ClusterKind = Literal["entity", "time", "none"]
_GS_TOL = 1e-12


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
    if cluster not in ("entity", "time", "none"):
        raise ValueError("cluster must be one of: entity, time, none")

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
    if cluster not in ("entity", "time", "none"):
        raise ValueError("cluster must be one of: entity, time, none")

    if cluster == "none":
        k = x_sel.shape[1]
        sigma2 = float(resid @ resid) / float(n - k)
        xtx_inv = np.linalg.inv(x_sel.T @ x_sel)
        cov_np = sigma2 * xtx_inv
        cov = cov_np.tolist()
        se = np.sqrt(np.diag(cov_np)).tolist()
    else:
        groups = entity if cluster == "entity" else time
        cov = nextstat.robust.ols_cluster_covariance(
            x_sel,
            resid,
            groups,
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
    cov: Literal["homoskedastic", "hc1", "cluster"] = "hc1",
    cluster: Optional[Sequence[Any]] = None,
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
    if cov not in ("homoskedastic", "hc1", "cluster"):
        raise ValueError("cov must be one of: homoskedastic, hc1, cluster")

    if cov == "homoskedastic":
        sigma2 = float(resid @ resid) / float(n - kx)
        cov_beta = (sigma2 * a_inv).tolist()
    else:
        if cov == "hc1":
            zu = z_sel * resid[:, None]        # (n, kz)
            meat = zu.T @ zu                    # (kz, kz)
            if df_correction and n > kx:
                meat *= float(n) / float(n - kx)
        else:
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
    cov: Literal["homoskedastic", "hc1", "cluster"] = "hc1",
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
    "relative_time",
    "event_study_regressors",
    "EventStudyTwfeFit",
    "event_study_twfe_fit",
    "event_study_twfe_from_formula",
    "WeakIvDiagnostics",
    "Iv2slsFit",
    "iv_2sls_fit",
    "iv_2sls_from_formula",
]
