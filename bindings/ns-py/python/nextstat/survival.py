"""High-level survival surface (Phase 9 Pack A).

Baseline scope:
- intercept-only parametric models
- right-censoring via (t, event)
- Cox proportional hazards (partial likelihood)
"""

from __future__ import annotations

from dataclasses import dataclass
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
            for a in range(p):
                for c in range(p):
                    b[a][c] += float(r[a]) * float(r[c])

    return b


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
) -> CoxPhFit:
    import nextstat

    t = _as_1d_float_list(times, name="times")
    e = _as_1d_bool_list(events, name="events")
    xx = _as_2d_float_list(x, name="x")
    m = nextstat.CoxPhModel(t, e, xx, ties=str(ties))
    r = nextstat.fit(m)
    beta = [float(v) for v in r.bestfit]

    cov = None
    se = None
    robust_cov = None
    robust_se = None

    if compute_cov:
        h = _hessian_from_grad(lambda b: [float(v) for v in m.grad_nll(b)], beta)
        cov = _mat_inv(h)
        se = [math.sqrt(max(float(cov[i][i]), 0.0)) for i in range(len(beta))]

        if robust:
            bmat = _cox_score_residual_outer(times=t, events=e, x=xx, beta=beta, ties=str(ties))
            # cov_robust = I^{-1} B I^{-1}
            robust_cov = _mat_mul(_mat_mul(cov, bmat), cov)
            robust_se = [math.sqrt(max(float(robust_cov[i][i]), 0.0)) for i in range(len(beta))]

    return CoxPhFit(
        coef=beta,
        nll=float(r.nll),
        converged=bool(r.converged),
        ties=str(ties),
        cov=cov,
        se=se,
        robust_cov=robust_cov,
        robust_se=robust_se,
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
