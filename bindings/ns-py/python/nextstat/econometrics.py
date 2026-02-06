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

from dataclasses import dataclass
from typing import Any, List, Literal, Mapping, Optional, Sequence, Tuple, Union

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

    # Means by entity, by time, and overall.
    cnt_ent: dict[int, int] = {}
    cnt_time: dict[int, int] = {}
    sum_y_ent: dict[int, float] = {}
    sum_y_time: dict[int, float] = {}
    sum_x_ent: dict[int, List[float]] = {}
    sum_x_time: dict[int, List[float]] = {}

    sum_y_all = 0.0
    sum_x_all = [0.0] * k

    for yi, xi, gi, ti in zip(y, x, ent_idx, time_idx):
        fy = float(yi)
        sum_y_all += fy
        sum_y_ent[gi] = sum_y_ent.get(gi, 0.0) + fy
        sum_y_time[ti] = sum_y_time.get(ti, 0.0) + fy

        if gi not in sum_x_ent:
            sum_x_ent[gi] = [0.0] * k
        if ti not in sum_x_time:
            sum_x_time[ti] = [0.0] * k

        sxg = sum_x_ent[gi]
        sxt = sum_x_time[ti]
        for j in range(k):
            fx = float(xi[j])
            sum_x_all[j] += fx
            sxg[j] += fx
            sxt[j] += fx

        cnt_ent[gi] = cnt_ent.get(gi, 0) + 1
        cnt_time[ti] = cnt_time.get(ti, 0) + 1

    mean_y_all = sum_y_all / float(n)
    mean_x_all = [sx / float(n) for sx in sum_x_all]

    mean_y_ent = {g: sum_y_ent[g] / float(cnt_ent[g]) for g in cnt_ent}
    mean_y_time = {t: sum_y_time[t] / float(cnt_time[t]) for t in cnt_time}
    mean_x_ent = {g: [sxj / float(cnt_ent[g]) for sxj in sum_x_ent[g]] for g in cnt_ent}
    mean_x_time = {t: [sxj / float(cnt_time[t]) for sxj in sum_x_time[t]] for t in cnt_time}

    y_dd: List[float] = []
    x_dd: List[List[float]] = []
    for yi, xi, gi, ti in zip(y, x, ent_idx, time_idx):
        y_dd.append(float(yi) - float(mean_y_ent[gi]) - float(mean_y_time[ti]) + float(mean_y_all))
        mxg = mean_x_ent[gi]
        mxt = mean_x_time[ti]
        x_dd.append([float(xi[j]) - float(mxg[j]) - float(mxt[j]) + float(mean_x_all[j]) for j in range(k)])

    return y_dd, x_dd, n_ent, n_time


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

    fit = nextstat.glm.linear.fit(x_dd, y_dd, include_intercept=False)
    coef = [float(v) for v in fit.coef]

    yhat = mat_vec_mul(x_dd, coef)
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
            x_dd,
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
        column_names=list(column_names),
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
    # Keep nicer names in returned fit.
    twfe = TwfeFit(
        coef=did.twfe.coef,
        standard_errors=did.twfe.standard_errors,
        covariance=did.twfe.covariance,
        column_names=["treat_post"] + control_names,
        n_obs=did.twfe.n_obs,
        n_entities=did.twfe.n_entities,
        n_times=did.twfe.n_times,
        cluster=did.twfe.cluster,
    )
    return DidTwfeFit(att=did.att, att_se=did.att_se, twfe=twfe)


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

    x = [[cols[j][i] for j in range(len(ks))] for i in range(len(rel_time))]
    names = [f"event[{k}]" for k in ks]
    return x, names, ks


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
    return EventStudyTwfeFit(
        rel_times=list(ks),
        coef=list(twfe.coef[: len(ks)]),
        standard_errors=list(twfe.standard_errors[: len(ks)]),
        covariance=[list(r[: len(ks)]) for r in twfe.covariance[: len(ks)]],
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
]
