"""High-level survival surface (Phase 9 Pack A).

Baseline scope:
- intercept-only parametric models
- right-censoring via (t, event)
- Cox proportional hazards (partial likelihood)
"""

from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
import math
from statistics import NormalDist
from typing import Any, Callable, List, Optional, Sequence


def _tolist(x: Any) -> Any:
    tolist = getattr(x, "tolist", None)
    if callable(tolist):
        return tolist()
    return x


def _as_1d_float_list(x: Any, *, name: str) -> List[float]:
    x = _tolist(x)
    if not isinstance(x, Sequence) or isinstance(x, (bytes, str)):
        raise TypeError(f"{name} must be a 1D sequence (or numpy array).")
    return [float(v) for v in x]


def _as_1d_bool_list(x: Any, *, name: str) -> List[bool]:
    x = _tolist(x)
    if not isinstance(x, Sequence) or isinstance(x, (bytes, str)):
        raise TypeError(f"{name} must be a 1D sequence (or numpy array).")
    return [bool(v) for v in x]

def _as_1d_int_list(x: Any, *, name: str) -> List[int]:
    x = _tolist(x)
    if not isinstance(x, Sequence) or isinstance(x, (bytes, str)):
        raise TypeError(f"{name} must be a 1D sequence (or numpy array).")
    return [int(v) for v in x]

def _as_2d_float_list(x: Any, *, name: str) -> List[List[float]]:
    x = _tolist(x)
    if not isinstance(x, Sequence) or isinstance(x, (bytes, str)):
        raise TypeError(f"{name} must be a 2D sequence (or numpy array).")
    out: List[List[float]] = []
    for i, row in enumerate(x):
        if not isinstance(row, Sequence) or isinstance(row, (bytes, str)):
            raise TypeError(f"{name}[{i}] must be a 1D sequence.")
        out.append([float(v) for v in row])
    return out


def _exp_clamped(x: float) -> float:
    # Mirror ns_prob::math::exp_clamped (~exp with clamp at ~700 to avoid inf).
    if x > 700.0:
        x = 700.0
    if x < -700.0:
        x = -700.0
    return math.exp(float(x))


def _z_for_level(level: float) -> float:
    if not (0.0 < float(level) < 1.0):
        raise ValueError("level must be in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + 0.5 * float(level))
    if not (math.isfinite(float(z)) and float(z) > 0.0):
        raise ValueError("invalid level (z-score was not finite)")
    return float(z)


def _mat_t(a: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*a)]


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    bt = _mat_t(b)
    return [[sum(ai * bj for ai, bj in zip(row, col)) for col in bt] for row in a]


def _mat_inv(a: list[list[float]]) -> list[list[float]]:
    # Reuse GLM tiny linalg (no numpy) for small matrices.
    from nextstat.glm._linalg import mat_inv as _inv

    return _inv(a)


def _hessian_from_grad(grad_fn: Callable[[list[float]], list[float]], beta: list[float]) -> list[list[float]]:
    p = len(beta)
    if p == 0:
        return []
    base = [float(v) for v in beta]
    h: list[list[float]] = [[0.0] * p for _ in range(p)]

    for j in range(p):
        eps = 1e-5 * (abs(float(base[j])) + 1.0)
        bp = base[:]
        bm = base[:]
        bp[j] += eps
        bm[j] -= eps
        gp = [float(v) for v in grad_fn(bp)]
        gm = [float(v) for v in grad_fn(bm)]
        for i in range(p):
            h[i][j] = (gp[i] - gm[i]) / (2.0 * eps)

    # Symmetrize to reduce finite-difference noise.
    for i in range(p):
        for j in range(i + 1, p):
            v = 0.5 * (float(h[i][j]) + float(h[j][i]))
            h[i][j] = v
            h[j][i] = v
    return h


def _cox_score_residual_outer(
    *,
    times: list[float],
    events: list[bool],
    x: list[list[float]],
    beta: list[float],
    ties: str,
    groups: Optional[list[int]] = None,
) -> list[list[float]]:
    # Score residual outer products for the Cox partial likelihood at beta_hat.
    #
    # In counting-process form, U(beta) = sum_i ∫ (x_i - xbar(beta,t)) dN_i(t),
    # which reduces to a sum over event times. Censored individuals have no events,
    # so their contributions are 0.
    #
    # For tied events:
    # - Breslow: xbar is computed once at the event time.
    # - Efron: use the average over the m Efron sub-steps at that event time.
    n = len(times)
    if n == 0:
        return []
    p = len(x[0]) if x else 0
    if p == 0:
        return []

    order = sorted(range(n), key=lambda i: float(times[i]), reverse=True)
    times_s = [float(times[i]) for i in order]
    events_s = [bool(events[i]) for i in order]
    x_s = [[float(v) for v in x[i]] for i in order]
    groups_s = None if groups is None else [int(groups[i]) for i in order]

    group_starts = [0]
    for i in range(1, n):
        if times_s[i] != times_s[i - 1]:
            group_starts.append(i)

    eta = [sum(float(xij) * float(bj) for xij, bj in zip(xi, beta)) for xi in x_s]
    w = [_exp_clamped(e) for e in eta]

    # risk0/risk1 are cumulative over the risk set as we sweep time descending.
    risk0 = 0.0
    risk1 = [0.0] * p

    b: list[list[float]] = [[0.0] * p for _ in range(p)]
    min_tail = 1e-300
    cluster_sums: Optional[dict[int, list[float]]] = {} if groups_s is not None else None

    for g, start in enumerate(group_starts):
        end = group_starts[g + 1] if g + 1 < len(group_starts) else n

        # Add this time-slice into the risk set.
        for i in range(start, end):
            wi = float(w[i])
            risk0 += wi
            for j in range(p):
                risk1[j] += wi * float(x_s[i][j])

        # Collect event rows at this time.
        event_idx: list[int] = []
        d0 = 0.0
        d1 = [0.0] * p
        for i in range(start, end):
            if not events_s[i]:
                continue
            event_idx.append(i)
            wi = float(w[i])
            d0 += wi
            for j in range(p):
                d1[j] += wi * float(x_s[i][j])

        m = len(event_idx)
        if m == 0:
            continue

        if ties == "breslow":
            denom = max(float(risk0), min_tail)
            xbar = [float(risk1[j]) / denom for j in range(p)]
        elif ties == "efron":
            mf = float(m)
            sum_xbar = [0.0] * p
            for r in range(m):
                frac = float(r) / mf
                denom = max(float(risk0) - frac * float(d0), min_tail)
                for j in range(p):
                    num = float(risk1[j]) - frac * float(d1[j])
                    sum_xbar[j] += num / denom
            xbar = [float(v) / mf for v in sum_xbar]
        else:
            raise ValueError("ties must be 'breslow' or 'efron'")

        # Each event gets residual r_i = x_i - xbar (xbar is per-time, tie-adjusted).
        for i in event_idx:
            r = [float(x_s[i][j]) - float(xbar[j]) for j in range(p)]
            if cluster_sums is not None and groups_s is not None:
                gid = int(groups_s[i])
                acc = cluster_sums.get(gid)
                if acc is None:
                    cluster_sums[gid] = r
                else:
                    for j in range(p):
                        acc[j] += float(r[j])
            else:
                for a in range(p):
                    for c in range(p):
                        b[a][c] += float(r[a]) * float(r[c])

    if cluster_sums is None:
        return b

    # Cluster-robust B = sum_g (sum_{i in g} r_i)(sum_{i in g} r_i)^T
    b2: list[list[float]] = [[0.0] * p for _ in range(p)]
    for v in cluster_sums.values():
        for a in range(p):
            for c in range(p):
                b2[a][c] += float(v[a]) * float(v[c])
    return b2


def _cox_baseline_cumhaz(
    *,
    times: list[float],
    events: list[bool],
    x: list[list[float]],
    beta: list[float],
    ties: str,
) -> tuple[list[float], list[float]]:
    # Baseline cumulative hazard H0(t) for the Cox model using Breslow/Efron increments.
    # Returns (event_times_sorted_asc, cumhaz_sorted_asc). Only event times (with >=1 event) are included.
    n = len(times)
    if n == 0:
        return [], []
    p = len(x[0]) if x else 0
    if p == 0:
        return [], []

    order = sorted(range(n), key=lambda i: float(times[i]), reverse=True)
    times_s = [float(times[i]) for i in order]
    events_s = [bool(events[i]) for i in order]
    x_s = [[float(v) for v in x[i]] for i in order]

    group_starts = [0]
    for i in range(1, n):
        if times_s[i] != times_s[i - 1]:
            group_starts.append(i)

    eta = [sum(float(xij) * float(bj) for xij, bj in zip(xi, beta)) for xi in x_s]
    w = [_exp_clamped(e) for e in eta]

    risk0 = 0.0
    # risk1 not needed for baseline hazard, only risk0 and d0.
    min_tail = 1e-300

    deltas_by_time: dict[float, float] = {}

    for g, start in enumerate(group_starts):
        end = group_starts[g + 1] if g + 1 < len(group_starts) else n

        for i in range(start, end):
            risk0 += float(w[i])

        m = 0
        d0 = 0.0
        for i in range(start, end):
            if events_s[i]:
                m += 1
                d0 += float(w[i])
        if m == 0:
            continue

        t0 = float(times_s[start])
        if ties == "breslow":
            deltas_by_time[t0] = deltas_by_time.get(t0, 0.0) + float(m) / max(float(risk0), min_tail)
        elif ties == "efron":
            mf = float(m)
            inc = 0.0
            for r in range(m):
                frac = float(r) / mf
                denom = max(float(risk0) - frac * float(d0), min_tail)
                inc += 1.0 / denom
            deltas_by_time[t0] = deltas_by_time.get(t0, 0.0) + float(inc)
        else:
            raise ValueError("ties must be 'breslow' or 'efron'")

    if not deltas_by_time:
        return [], []

    ts = sorted(deltas_by_time.keys())
    cum = 0.0
    hs: list[float] = []
    for t0 in ts:
        cum += float(deltas_by_time[t0])
        hs.append(float(cum))
    return ts, hs


@dataclass(frozen=True)
class CoxPhFit:
    coef: List[float]
    nll: float
    converged: bool
    ties: str
    cov: Optional[List[List[float]]]
    se: Optional[List[float]]
    robust_cov: Optional[List[List[float]]]
    robust_se: Optional[List[float]]
    robust_kind: Optional[str]
    baseline_times: Optional[List[float]]
    baseline_cumhaz: Optional[List[float]]

    def hazard_ratios(self) -> List[float]:
        return [math.exp(float(b)) for b in self.coef]

    def confint(self, *, level: float = 0.95, robust: bool = False) -> List[tuple[float, float]]:
        z = _z_for_level(level)
        ses = self.robust_se if robust else self.se
        if ses is None:
            raise ValueError("SE not available (fit was called with compute_cov=False)")
        out: List[tuple[float, float]] = []
        for b, s in zip(self.coef, ses):
            lo = float(b) - float(z) * float(s)
            hi = float(b) + float(z) * float(s)
            out.append((lo, hi))
        return out

    def hazard_ratio_confint(self, *, level: float = 0.95, robust: bool = False) -> List[tuple[float, float]]:
        cis = self.confint(level=level, robust=robust)
        return [(math.exp(lo), math.exp(hi)) for lo, hi in cis]

    def predict_cumhaz(
        self,
        x: Sequence[Sequence[float]],
        *,
        times: Optional[Sequence[float]] = None,
    ) -> List[List[float]]:
        if self.baseline_times is None or self.baseline_cumhaz is None:
            raise ValueError("baseline cumulative hazard not available (fit was called with compute_baseline=False)")

        x2 = _as_2d_float_list(x, name="x")
        bt = [float(v) for v in self.baseline_times]
        bh = [float(v) for v in self.baseline_cumhaz]
        if len(bt) != len(bh):
            raise ValueError("baseline arrays length mismatch")

        # Evaluate H0(t) as a right-continuous step function.
        if times is None:
            grid = bt
        else:
            grid = [float(v) for v in times]

        out: List[List[float]] = []
        for xr in x2:
            eta = sum(float(a) * float(b) for a, b in zip(xr, self.coef))
            mult = math.exp(float(eta))
            row: List[float] = []
            for t0 in grid:
                k = bisect_right(bt, float(t0)) - 1
                h0 = 0.0 if k < 0 else float(bh[k])
                row.append(float(h0) * float(mult))
            out.append(row)
        return out

    def predict_survival(
        self,
        x: Sequence[Sequence[float]],
        *,
        times: Optional[Sequence[float]] = None,
    ) -> List[List[float]]:
        hs = self.predict_cumhaz(x, times=times)
        return [[math.exp(-float(h)) for h in row] for row in hs]


@dataclass(frozen=True)
class ParametricSurvivalFit:
    model: str
    params: List[float]
    se: List[float]
    nll: float
    converged: bool

    def confint(self, *, level: float = 0.95) -> List[tuple[float, float]]:
        z = _z_for_level(level)
        out: List[tuple[float, float]] = []
        for p, s in zip(self.params, self.se):
            out.append((float(p) - float(z) * float(s), float(p) + float(z) * float(s)))
        return out


def _build_exponential(times: Any, events: Any):
    """Build an exponential survival model (right-censoring)."""
    import nextstat

    return nextstat.ExponentialSurvivalModel(
        _as_1d_float_list(times, name="times"),
        _as_1d_bool_list(events, name="events"),
    )


def _build_weibull(times: Any, events: Any):
    """Build a Weibull survival model (right-censoring)."""
    import nextstat

    return nextstat.WeibullSurvivalModel(
        _as_1d_float_list(times, name="times"),
        _as_1d_bool_list(events, name="events"),
    )


def _build_lognormal_aft(times: Any, events: Any):
    """Build a log-normal AFT survival model (right-censoring)."""
    import nextstat

    return nextstat.LogNormalAftModel(
        _as_1d_float_list(times, name="times"),
        _as_1d_bool_list(events, name="events"),
    )

def _build_cox_ph(times: Any, events: Any, x: Any, *, ties: str = "efron"):
    """Build a Cox proportional hazards model (partial likelihood)."""
    import nextstat

    return nextstat.CoxPhModel(
        _as_1d_float_list(times, name="times"),
        _as_1d_bool_list(events, name="events"),
        _as_2d_float_list(x, name="x"),
        ties=ties,
    )

class _Builder:
    def __init__(self, build: Callable[..., Any], fit_fn: Callable[..., Any]):
        self._build = build
        self.fit = fit_fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._build(*args, **kwargs)


def _fit_parametric(*, model_name: str, build_fn: Callable[..., Any], times: Any, events: Any) -> ParametricSurvivalFit:
    import nextstat

    m = build_fn(times, events)
    r = nextstat.fit(m)
    params = [float(v) for v in r.bestfit]
    se = [float(v) for v in r.uncertainties]
    return ParametricSurvivalFit(
        model=str(model_name),
        params=params,
        se=se,
        nll=float(r.nll),
        converged=bool(r.converged),
    )


def _fit_cox_ph(
    times: Any,
    events: Any,
    x: Any,
    *,
    ties: str = "efron",
    robust: bool = True,
    compute_cov: bool = True,
    groups: Optional[Any] = None,
    compute_baseline: bool = True,
) -> CoxPhFit:
    import nextstat

    t = _as_1d_float_list(times, name="times")
    e = _as_1d_bool_list(events, name="events")
    xx = _as_2d_float_list(x, name="x")
    g = None if groups is None else _as_1d_int_list(groups, name="groups")
    if g is not None and len(g) != len(t):
        raise ValueError("groups must have length n")
    m = nextstat.CoxPhModel(t, e, xx, ties=str(ties))
    r = nextstat.fit(m)
    beta = [float(v) for v in r.bestfit]

    cov = None
    se = None
    robust_cov = None
    robust_se = None
    robust_kind = None
    baseline_times = None
    baseline_cumhaz = None

    if compute_cov:
        h = _hessian_from_grad(lambda b: [float(v) for v in m.grad_nll(b)], beta)
        cov = _mat_inv(h)
        se = [math.sqrt(max(float(cov[i][i]), 0.0)) for i in range(len(beta))]

        if robust:
            bmat = _cox_score_residual_outer(
                times=t, events=e, x=xx, beta=beta, ties=str(ties), groups=g
            )
            # cov_robust = I^{-1} B I^{-1}
            robust_cov = _mat_mul(_mat_mul(cov, bmat), cov)
            robust_se = [math.sqrt(max(float(robust_cov[i][i]), 0.0)) for i in range(len(beta))]
            robust_kind = "cluster" if g is not None else "hc0"

    if compute_baseline:
        baseline_times, baseline_cumhaz = _cox_baseline_cumhaz(
            times=t, events=e, x=xx, beta=beta, ties=str(ties)
        )

    return CoxPhFit(
        coef=beta,
        nll=float(r.nll),
        converged=bool(r.converged),
        ties=str(ties),
        cov=cov,
        se=se,
        robust_cov=robust_cov,
        robust_se=robust_se,
        robust_kind=robust_kind,
        baseline_times=baseline_times,
        baseline_cumhaz=baseline_cumhaz,
    )


exponential = _Builder(
    _build_exponential,
    lambda times, events: _fit_parametric(
        model_name="exponential", build_fn=_build_exponential, times=times, events=events
    ),
)
weibull = _Builder(
    _build_weibull,
    lambda times, events: _fit_parametric(
        model_name="weibull", build_fn=_build_weibull, times=times, events=events
    ),
)
lognormal_aft = _Builder(
    _build_lognormal_aft,
    lambda times, events: _fit_parametric(
        model_name="lognormal_aft", build_fn=_build_lognormal_aft, times=times, events=events
    ),
)
cox_ph = _Builder(_build_cox_ph, _fit_cox_ph)


__all__ = [
    "exponential",
    "weibull",
    "lognormal_aft",
    "cox_ph",
    "CoxPhFit",
    "ParametricSurvivalFit",
]
