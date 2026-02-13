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

from .glm._linalg import as_2d_float_list, mat_inv, mat_mul, mat_t, mat_vec_mul


ClusterKind = Literal["entity", "time", "none"]
_GS_TOL = 1e-12


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

    # Vectorised group-mean subtraction via numpy.
    g = np.asarray(ent_idx, dtype=np.intp)
    ya = np.asarray(y, dtype=np.float64)
    xa = np.asarray(x, dtype=np.float64)

    cnt = np.bincount(g).astype(np.float64)  # (n_groups,)
    # Group sums → group means.
    y_mean_g = np.bincount(g, weights=ya) / cnt
    x_mean_g = np.zeros((_n_ent, k), dtype=np.float64)
    for j in range(k):
        x_mean_g[:, j] = np.bincount(g, weights=xa[:, j]) / cnt

    y_star = (ya - y_mean_g[g]).tolist()
    x_star = (xa - x_mean_g[g]).tolist()
    return y_star, x_star


def _two_way_demean(y: List[float], x: List[List[float]], entity: Sequence[Any], time: Sequence[Any]) -> Tuple[List[float], List[List[float]], int, int]:
    n = len(y)
    if n == 0:
        raise ValueError("need at least 1 observation")
    k = len(x[0]) if x else 0
    if k == 0:
        raise ValueError("X must have at least 1 column")
    if any(len(row) != k for row in x):
        raise ValueError("X must be rectangular")
    _validate_lengths(n, x, entity, time)

    ent_idx, n_ent = _encode_groups(entity)
    time_idx, n_time = _encode_groups(time)

    # Vectorised two-way demeaning via numpy.
    g = np.asarray(ent_idx, dtype=np.intp)
    t = np.asarray(time_idx, dtype=np.intp)
    ya = np.asarray(y, dtype=np.float64)
    xa = np.asarray(x, dtype=np.float64)

    cnt_g = np.bincount(g).astype(np.float64)
    cnt_t = np.bincount(t).astype(np.float64)

    # y means by entity, time, and overall.
    y_mean_g = np.bincount(g, weights=ya) / cnt_g
    y_mean_t = np.bincount(t, weights=ya) / cnt_t
    y_mean_all = ya.mean()

    # x means by entity, time, and overall.
    x_mean_g = np.zeros((n_ent, k), dtype=np.float64)
    x_mean_t = np.zeros((n_time, k), dtype=np.float64)
    for j in range(k):
        x_mean_g[:, j] = np.bincount(g, weights=xa[:, j]) / cnt_g
        x_mean_t[:, j] = np.bincount(t, weights=xa[:, j]) / cnt_t
    x_mean_all = xa.mean(axis=0)

    # Two-way demeaning: y_it - y_bar_i - y_bar_t + y_bar
    y_dd = (ya - y_mean_g[g] - y_mean_t[t] + y_mean_all).tolist()
    x_dd = (xa - x_mean_g[g] - x_mean_t[t] + x_mean_all).tolist()

    return y_dd, x_dd, n_ent, n_time


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("length mismatch")
    return float(np.dot(a, b))

def _outer(v: Sequence[float]) -> List[List[float]]:
    va = np.asarray(v, dtype=np.float64)
    return np.outer(va, va).tolist()


def _mat_add_inplace(a: List[List[float]], b: List[List[float]]) -> None:
    if len(a) != len(b):
        raise ValueError("matrix shape mismatch")
    for i in range(len(a)):
        if len(a[i]) != len(b[i]):
            raise ValueError("matrix shape mismatch")
        for j in range(len(a[i])):
            a[i][j] += float(b[i][j])


def _mat_scale_inplace(a: List[List[float]], s: float) -> None:
    fs = float(s)
    for i in range(len(a)):
        for j in range(len(a[i])):
            a[i][j] *= fs


def _col_as_vec(x: Sequence[Sequence[float]], j: int) -> List[float]:
    return [float(row[j]) for row in x]


def _select_independent_columns(
    x: List[List[float]],
    names: List[str],
    *,
    mandatory: Optional[Sequence[int]] = None,
    tol: float = _GS_TOL,
) -> Tuple[List[List[float]], List[str]]:
    """Select a numerically linearly independent subset of columns from X.

    This prevents singular XtX in small samples and drops columns absorbed by FE.
    Uses modified Gram-Schmidt via numpy for performance.
    """
    if not x:
        raise ValueError("X must be non-empty")
    k = len(x[0])
    if k == 0:
        raise ValueError("X must have at least 1 column")
    if len(names) != k:
        raise ValueError("column_names length mismatch")
    if any(len(row) != k for row in x):
        raise ValueError("X must be rectangular")

    mand = sorted(set(int(i) for i in (mandatory or []) if 0 <= int(i) < k))
    order = mand + [j for j in range(k) if j not in set(mand)]

    xa = np.asarray(x, dtype=np.float64)
    basis: list[np.ndarray] = []
    kept: List[int] = []
    tol2 = float(tol) * float(tol)

    for j in order:
        v = xa[:, j].copy()
        for b in basis:
            proj = float(v @ b)
            if proj != 0.0:
                v -= proj * b
        norm2 = float(v @ v)
        if norm2 > tol2:
            v /= np.sqrt(norm2)
            basis.append(v)
            kept.append(j)

    kept_sorted = sorted(kept)
    x2 = xa[:, kept_sorted].tolist()
    names2 = [names[j] for j in kept_sorted]
    return x2, names2


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

def _twfe_fit_ols(
    x: List[List[float]],
    y: List[float],
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

    fit = nextstat.glm.linear.fit(x_sel, y_dd, include_intercept=False)
    coef = [float(v) for v in fit.coef]

    yhat = mat_vec_mul(x_sel, coef)
    resid = [obs - pred for obs, pred in zip(y_dd, yhat)]

    cluster = str(cluster).lower()  # type: ignore[assignment]
    if cluster not in ("entity", "time", "none"):
        raise ValueError("cluster must be one of: entity, time, none")

    if cluster == "none":
        cov = fit.covariance
        se = fit.standard_errors
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
        coef=coef,
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
    y2 = [float(v) for v in y]
    n = len(y2)
    if n == 0:
        raise ValueError("need at least 1 observation")
    _validate_lengths(n, treat, post, entity, time)

    d = did_regressor(treat, post)
    if x is None:
        x2: List[List[float]] = [[di] for di in d]
        names = ["treat_post"]
    else:
        x_raw = as_2d_float_list(x)
        if len(x_raw) != n:
            raise ValueError("X and y must have the same length")
        x2 = [[di] + [float(v) for v in row] for di, row in zip(d, x_raw)]
        names = ["treat_post"] + [f"x{i}" for i in range(len(x_raw[0]))]

    twfe = _twfe_fit_ols(x2, y2, entity=entity, time=time, cluster=cluster, column_names=names)
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
) -> Tuple[List[List[float]], List[str], List[int]]:
    """Build event-study regressors: treat * 1[rel_time == k] for k in window, excluding reference."""
    lo, hi = int(window[0]), int(window[1])
    if lo > hi:
        raise ValueError("window must satisfy lo <= hi")

    ks = [k for k in range(lo, hi + 1) if k != int(reference)]
    _validate_lengths(len(treat), rel_time)

    cols: List[List[float]] = [[] for _ in ks]
    for a, rt in zip(treat, rel_time):
        ta = float(bool(a))
        for j, k in enumerate(ks):
            cols[j].append(ta * (1.0 if int(rt) == int(k) else 0.0))

    # Drop bins with no support to avoid singular designs in small samples.
    keep: List[int] = []
    for j, col in enumerate(cols):
        if any(float(v) != 0.0 for v in col):
            keep.append(j)

    cols2 = [cols[j] for j in keep]
    ks2 = [ks[j] for j in keep]
    x = [[cols2[j][i] for j in range(len(ks2))] for i in range(len(rel_time))]
    names = [f"event[{k}]" for k in ks2]
    return x, names, ks2


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
    y2 = [float(v) for v in y]
    n = len(y2)
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
        x_raw = as_2d_float_list(x)
        if len(x_raw) != n:
            raise ValueError("X and y must have the same length")
        x_all = [row_ev + row_ctrl for row_ev, row_ctrl in zip(x_ev, x_raw)]
        names = names_ev + [f"x{i}" for i in range(len(x_raw[0]))]

    twfe = _twfe_fit_ols(x_all, y2, entity=entity, time=time, cluster=cluster, column_names=names)
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


def _require_full_rank(x: List[List[float]], names: List[str], what: str) -> None:
    if not x:
        raise ValueError(f"{what} must be non-empty")
    k = len(x[0])
    xs, _ns = _select_independent_columns(x, names, mandatory=list(range(k)))
    if len(xs[0]) != k:
        raise ValueError(f"{what} is rank-deficient / collinear")


def _sum_sq(v: Sequence[float]) -> float:
    va = np.asarray(v, dtype=np.float64)
    return float(va @ va)


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
    y2 = [float(v) for v in y]
    n = len(y2)
    if n == 0:
        raise ValueError("need at least 1 observation")

    x_endog = as_2d_float_list(endog)
    z_excl = as_2d_float_list(instruments)
    if len(x_endog) != n or len(z_excl) != n:
        raise ValueError("length mismatch between y/endog/instruments")

    p_endog = len(x_endog[0]) if x_endog else 0
    if p_endog == 0:
        raise ValueError("endog must have at least 1 column")
    if any(len(row) != p_endog for row in x_endog):
        raise ValueError("endog must be rectangular")

    q = len(z_excl[0]) if z_excl else 0
    if q == 0:
        raise ValueError("instruments must have at least 1 column")
    if any(len(row) != q for row in z_excl):
        raise ValueError("instruments must be rectangular")
    if q < p_endog:
        raise ValueError("underidentified: need at least as many excluded instruments as endogenous regressors")

    x_exog: List[List[float]]
    p_exog: int
    if exog is None:
        x_exog = [[] for _ in range(n)]
        p_exog = 0
    else:
        x_exog = as_2d_float_list(exog)
        if len(x_exog) != n:
            raise ValueError("length mismatch between y and exog")
        p_exog = len(x_exog[0]) if x_exog else 0
        if p_exog == 0:
            raise ValueError("exog must have at least 1 column when provided")
        if any(len(row) != p_exog for row in x_exog):
            raise ValueError("exog must be rectangular")

    endog_names2 = [f"endog{i}" for i in range(p_endog)] if endog_names is None else [str(s) for s in endog_names]
    if len(endog_names2) != p_endog:
        raise ValueError("endog_names length mismatch")

    exog_names2 = [f"x{i}" for i in range(p_exog)] if exog_names is None else [str(s) for s in exog_names]
    if len(exog_names2) != p_exog:
        raise ValueError("exog_names length mismatch")

    instr_names2 = [f"z{i}" for i in range(q)] if instrument_names is None else [str(s) for s in instrument_names]
    if len(instr_names2) != q:
        raise ValueError("instrument_names length mismatch")

    x = [xe + xx for xe, xx in zip(x_endog, x_exog)]
    x_names = endog_names2 + exog_names2
    _require_full_rank(x, x_names, "X")

    z = [zi + xx for zi, xx in zip(z_excl, x_exog)]
    z_names = instr_names2 + exog_names2

    exog_idx = list(range(q, q + p_exog))
    z_sel, z_names_sel = _select_independent_columns(z, z_names, mandatory=exog_idx)
    if p_exog > 0 and any(nm not in z_names_sel for nm in exog_names2):
        raise ValueError("exog columns are collinear in Z")

    excluded_kept = [nm for nm in z_names_sel if nm in instr_names2]

    kx = len(x[0])
    kz = len(z_sel[0])
    if n <= kx:
        raise ValueError("Need n > n_params to estimate sigma2_hat")
    if kz < kx:
        raise ValueError("underidentified after dropping collinear instruments")

    xt = mat_t(x)
    zt = mat_t(z_sel)
    ztz_inv = mat_inv(mat_mul(zt, z_sel))

    xtz = mat_mul(xt, z_sel)  # (kx x kz)
    xz_inv = mat_mul(xtz, ztz_inv)  # (kx x kz)
    ztx = mat_t(xtz)  # (kz x kx)
    a = mat_mul(xz_inv, ztx)  # (kx x kx)
    a_inv = mat_inv(a)

    zty = mat_vec_mul(zt, y2)  # (kz)
    rhs = mat_vec_mul(xz_inv, zty)  # (kx)
    beta = mat_vec_mul(a_inv, rhs)  # (kx)

    yhat = mat_vec_mul(x, beta)
    resid = [obs - pred for obs, pred in zip(y2, yhat)]

    cov = str(cov).lower()  # type: ignore[assignment]
    if cov not in ("homoskedastic", "hc1", "cluster"):
        raise ValueError("cov must be one of: homoskedastic, hc1, cluster")

    if cov == "homoskedastic":
        sigma2 = _sum_sq(resid) / float(n - kx)
        cov_beta = [[sigma2 * float(v) for v in row] for row in a_inv]
    else:
        # Vectorised sandwich meat computation via numpy.
        za = np.asarray(z_sel, dtype=np.float64)   # (n, kz)
        ua = np.asarray(resid, dtype=np.float64)    # (n,)

        if cov == "hc1":
            zu = za * ua[:, None]                   # (n, kz) element-wise
            meat_np = zu.T @ zu                     # (kz, kz)
            if df_correction and n > kx:
                meat_np *= float(n) / float(n - kx)
        else:
            if cluster is None:
                raise ValueError("cluster must be provided when cov='cluster'")
            _validate_lengths(n, cluster)
            cl_idx, n_cl = _encode_groups(cluster)
            if n_cl < 2:
                raise ValueError("cluster must have at least 2 distinct groups")
            ga = np.asarray(cl_idx, dtype=np.intp)
            # Score per observation, then sum within cluster.
            zu = za * ua[:, None]                   # (n, kz)
            sg = np.zeros((n_cl, kz), dtype=np.float64)
            for j in range(kz):
                sg[:, j] = np.bincount(ga, weights=zu[:, j], minlength=n_cl)
            meat_np = sg.T @ sg                     # (kz, kz)
            if df_correction and n > kx:
                scale = (float(n_cl) / float(n_cl - 1)) * ((float(n) - 1.0) / float(n - kx))
                meat_np *= scale

        meat = meat_np.tolist()
        b = mat_mul(mat_mul(xz_inv, mat_mul(meat, ztz_inv)), ztx)
        cov_beta = mat_mul(mat_mul(a_inv, b), a_inv)

    se = nextstat.robust.cov_to_se(cov_beta)

    exog_sel_idx = [i for i, nm in enumerate(z_names_sel) if nm in exog_names2]
    z_exog_only = [[row[i] for i in exog_sel_idx] for row in z_sel] if exog_sel_idx else None

    f_stats: List[float] = []
    partial_r2s: List[float] = []
    excl_idx = [i for i, nm in enumerate(z_names_sel) if nm in instr_names2]
    q_kept = len(excl_idx)
    k_ur = kz
    df2 = n - k_ur

    for j in range(p_endog):
        dcol = _col_as_vec(x_endog, j)
        if q_kept == 0 or df2 <= 0:
            f_stats.append(float("nan"))
            partial_r2s.append(float("nan"))
            continue

        try:
            fs_ur = nextstat.glm.linear.fit(z_sel, dcol, include_intercept=False)
            d_hat = list(fs_ur.predict(z_sel))
            ssr_ur = _sum_sq([a - b for a, b in zip(dcol, d_hat)])
        except Exception:
            f_stats.append(float("nan"))
            partial_r2s.append(float("nan"))
            continue

        if z_exog_only is None or (z_exog_only and len(z_exog_only[0]) == 0):
            ssr_r = _sum_sq(dcol)
        else:
            try:
                fs_r = nextstat.glm.linear.fit(z_exog_only, dcol, include_intercept=False)
                d_hat_r = list(fs_r.predict(z_exog_only))
                ssr_r = _sum_sq([a - b for a, b in zip(dcol, d_hat_r)])
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
        coef=[float(v) for v in beta],
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

    x_endog: List[List[float]] = []
    for i in range(n):
        row: List[float] = []
        for nm in endog_names:
            row.append(float(cols[nm][i]))
        x_endog.append(row)

    z_excl: List[List[float]] = []
    for i in range(n):
        row2: List[float] = []
        for nm in instr_names:
            row2.append(float(cols[nm][i]))
        z_excl.append(row2)

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
