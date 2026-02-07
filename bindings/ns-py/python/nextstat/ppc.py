"""Posterior predictive checks (PPC) utilities (Phase 7.4).

This module is intentionally lightweight:
- no numpy/pandas dependency
- works directly with the raw dict returned by `nextstat.sample(...)`

Currently supports PPC for `ComposedGlmModel`-style regression specs built via
`nextstat.data.GlmSpec`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .data import GlmSpec


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def _softplus(x: float) -> float:
    # Stable log(1 + exp(x))
    if x > 0.0:
        return float(x) + math.log1p(math.exp(-float(x)))
    return math.log1p(math.exp(float(x)))


def _normal_cdf(x: float) -> float:
    # Phi(x) = 0.5 * erfc(-x / sqrt(2))
    return 0.5 * math.erfc(-float(x) / math.sqrt(2.0))


def _cutpoints_from_raw(raw: Sequence[float]) -> List[float]:
    if not raw:
        return []
    c: List[float] = [float(raw[0])]
    for r in raw[1:]:
        c.append(float(c[-1] + _softplus(float(r))))
    return c


def _flatten_posterior(
    posterior: Mapping[str, Sequence[Sequence[float]]],
    *,
    param_names: Sequence[str],
) -> List[List[float]]:
    if not param_names:
        raise ValueError("param_names must be non-empty")
    first = param_names[0]
    if first not in posterior:
        raise KeyError(f"posterior missing param {first!r}")
    n_chains = len(posterior[first])
    if n_chains == 0:
        raise ValueError("posterior must contain at least 1 chain")
    n_draws = len(posterior[first][0])
    if n_draws == 0:
        raise ValueError("posterior must contain at least 1 draw")

    out: List[List[float]] = []
    for c in range(n_chains):
        for d in range(n_draws):
            out.append([float(posterior[name][c][d]) for name in param_names])
    return out


def _effective_group_intercepts(
    draw: Mapping[str, float], *, n_groups: int
) -> Optional[List[float]]:
    # Centered random intercept: alpha1..alphaG
    alpha1 = draw.get("alpha1")
    if alpha1 is not None:
        return [float(draw[f"alpha{g+1}"]) for g in range(n_groups)]

    # Non-centered random intercept: z_alpha1..z_alphaG, with mu_alpha + sigma_alpha
    z1 = draw.get("z_alpha1")
    if z1 is not None:
        mu = float(draw["mu_alpha"])
        sigma = float(draw["sigma_alpha"])
        return [mu + sigma * float(draw[f"z_alpha{g+1}"]) for g in range(n_groups)]

    return None


def _effective_group_slopes(
    draw: Mapping[str, float], *, feature_idx: int, n_groups: int
) -> Optional[List[float]]:
    # Random slope naming is tied to ComposedGlmModel parameter_names:
    # - centered: u_beta{k}_{g}
    # - non-centered: z_u_beta{k}_{g} + hyper (mu_u_beta{k}, sigma_u_beta{k})
    k = int(feature_idx) + 1

    u1 = draw.get(f"u_beta{k}_1")
    if u1 is not None:
        return [float(draw[f"u_beta{k}_{g+1}"]) for g in range(n_groups)]

    z1 = draw.get(f"z_u_beta{k}_1")
    if z1 is not None:
        mu = float(draw[f"mu_u_beta{k}"])
        sigma = float(draw[f"sigma_u_beta{k}"])
        return [mu + sigma * float(draw[f"z_u_beta{k}_{g+1}"]) for g in range(n_groups)]

    return None


def _effective_correlated_intercept_slope(
    draw: Mapping[str, float], *, feature_idx: int, n_groups: int
) -> Optional[tuple[List[float], List[float]]]:
    # Correlated naming from ComposedGlmModel parameter_names:
    # - mu_alpha
    # - mu_u_beta{k}
    # - tau_alpha > 0
    # - tau_u_beta{k} > 0
    # - rho_alpha_u_beta{k} in [-1,1]
    # - per-group z_alpha_g{g}, z_u_beta{k}_g{g}
    k = int(feature_idx) + 1
    if f"rho_alpha_u_beta{k}" not in draw:
        return None

    mu_alpha = float(draw["mu_alpha"])
    mu_u = float(draw[f"mu_u_beta{k}"])
    tau_alpha = float(draw["tau_alpha"])
    tau_u = float(draw[f"tau_u_beta{k}"])
    rho = float(draw[f"rho_alpha_u_beta{k}"])

    if not (tau_alpha > 0.0 and math.isfinite(tau_alpha)):
        raise ValueError(f"tau_alpha must be finite and > 0, got {tau_alpha}")
    if not (tau_u > 0.0 and math.isfinite(tau_u)):
        raise ValueError(f"tau_u_beta{k} must be finite and > 0, got {tau_u}")
    if not (math.isfinite(rho) and -1.0 <= rho <= 1.0):
        raise ValueError(f"rho_alpha_u_beta{k} must be in [-1,1], got {rho}")

    s = 1.0 - rho * rho
    if s < 0.0:
        # Numeric jitter could put rho slightly outside [-1,1]. Keep this strict in PPC.
        raise ValueError(f"invalid rho_alpha_u_beta{k}={rho}: 1-rho^2 < 0")
    s = math.sqrt(s)

    alphas: List[float] = []
    slopes: List[float] = []
    for g in range(n_groups):
        z1 = float(draw[f"z_alpha_g{g+1}"])
        z2 = float(draw[f"z_u_beta{k}_g{g+1}"])
        alpha_g = mu_alpha + tau_alpha * z1
        u_g = mu_u + tau_u * (rho * z1 + s * z2)
        alphas.append(float(alpha_g))
        slopes.append(float(u_g))
    return alphas, slopes


def _poisson_sample(rng: random.Random, lam: float) -> int:
    if not (lam >= 0.0) or not math.isfinite(lam):
        raise ValueError(f"poisson rate must be finite and >= 0, got {lam}")
    if lam == 0.0:
        return 0

    # Exact: Knuth for small lambda, PTRS (Hoermann) for larger lambda.
    if lam < 10.0:
        l = math.exp(-lam)
        k = 0
        p = 1.0
        while p > l:
            k += 1
            p *= rng.random()
        return k - 1

    slam = math.sqrt(lam)
    loglam = math.log(lam)
    b = 0.931 + 2.53 * slam
    a = -0.059 + 0.02483 * b
    inv_alpha = 1.1239 + 1.1328 / (b - 3.4)
    v_r = 0.9277 - 3.6224 / (b - 2.0)

    while True:
        u = rng.random() - 0.5
        v = rng.random()
        us = 0.5 - abs(u)
        k = int(math.floor((2.0 * a / us + b) * u + lam + 0.43))

        if k < 0:
            continue
        if us >= 0.07 and v <= v_r:
            return k
        if us < 0.013 and v > us:
            continue

        # Acceptance test in log-space.
        lhs = math.log(v * inv_alpha / (a / (us * us) + b))
        rhs = -lam + k * loglam - math.lgamma(k + 1.0)
        if lhs <= rhs:
            return k


def _negbin_sample_nb2(rng: random.Random, *, mu: float, alpha: float) -> int:
    # NB2 parameterization: Var(Y) = mu + alpha * mu^2, alpha > 0.
    # Represent as Poisson-Gamma mixture:
    # lambda ~ Gamma(shape=1/alpha, scale=alpha*mu), y ~ Poisson(lambda).
    if not (mu >= 0.0) or not math.isfinite(mu):
        raise ValueError(f"negbin mean must be finite and >= 0, got {mu}")
    if not (alpha > 0.0) or not math.isfinite(alpha):
        raise ValueError(f"negbin alpha must be finite and > 0, got {alpha}")
    if mu == 0.0:
        return 0

    # Very small alpha approximates Poisson(mu).
    if alpha < 1e-12:
        return _poisson_sample(rng, mu)

    shape = 1.0 / float(alpha)
    scale = float(alpha) * float(mu)
    lam = rng.gammavariate(shape, scale)
    return _poisson_sample(rng, float(lam))


def _categorical_sample(rng: random.Random, probs: Sequence[float]) -> int:
    s = sum(float(p) for p in probs)
    if not (math.isfinite(s) and s > 0.0):
        raise ValueError("categorical probs must sum to a finite positive value")
    u = rng.random() * s
    acc = 0.0
    for i, p in enumerate(probs):
        acc += float(p)
        if u <= acc:
            return int(i)
    return int(len(probs) - 1)


def _eta_glm(
    *,
    x_row: Sequence[float],
    draw: Mapping[str, float],
    include_intercept: bool,
    group: Optional[int],
    group_intercepts: Optional[Sequence[float]],
    slope_feature_idx: Optional[int],
    group_slopes: Optional[Sequence[float]],
    offset: Optional[float],
) -> float:
    eta = 0.0
    if include_intercept:
        eta += float(draw.get("intercept", 0.0))
    for j, xj in enumerate(x_row):
        eta += float(draw.get(f"beta{j+1}", 0.0)) * float(xj)
    if group is not None and group_intercepts is not None:
        eta += float(group_intercepts[int(group)])
    if group is not None and slope_feature_idx is not None and group_slopes is not None:
        eta += float(group_slopes[int(group)]) * float(x_row[int(slope_feature_idx)])
    if offset is not None:
        eta += float(offset)
    return float(eta)


@dataclass(frozen=True)
class PpcStats:
    observed: Dict[str, float]
    replicated: List[Dict[str, float]]


def default_stats(kind: str, y: Sequence[float]) -> Dict[str, float]:
    ys = [float(v) for v in y]
    if not ys:
        return {"n": 0.0}
    mu = sum(ys) / len(ys)
    out: Dict[str, float] = {"n": float(len(ys)), "mean": float(mu)}
    if kind in ("linear", "poisson", "negbin"):
        v = sum((v - mu) ** 2 for v in ys) / max(1.0, float(len(ys) - 1))
        out["var"] = float(v)
    if kind == "ordinal_ordered":
        # Treat y as 0..K-1 integers; report per-level proportions.
        counts: Dict[int, int] = {}
        for v in ys:
            iv = int(v)
            counts[iv] = counts.get(iv, 0) + 1
        n = float(len(ys))
        for k in sorted(counts.keys()):
            out[f"p_level{k}"] = float(counts[k]) / n
    return out


def replicate_glm(
    spec: GlmSpec,
    draw: Mapping[str, float],
    *,
    seed: int,
) -> List[float]:
    rng = random.Random(int(seed))
    n = len(spec.x)
    ng = spec.n_groups
    group_intercepts: Optional[Sequence[float]] = None
    group_slopes: Optional[Sequence[float]] = None
    slope_feature_idx: Optional[int] = None

    if (spec.random_slope_feature_idx is not None or spec.correlated_feature_idx is not None) and spec.group_idx is None:
        raise ValueError("random slopes / correlated effects require spec.group_idx")

    if spec.group_idx is not None:
        if ng is None:
            raise ValueError("spec.n_groups must be set when spec.group_idx is present")
        n_groups = int(ng)

        if spec.correlated_feature_idx is not None:
            slope_feature_idx = int(spec.correlated_feature_idx)
            eff = _effective_correlated_intercept_slope(
                draw, feature_idx=slope_feature_idx, n_groups=n_groups
            )
            if eff is None:
                raise ValueError("draw does not contain correlated random-effect parameters")
            group_intercepts, group_slopes = eff
        else:
            group_intercepts = _effective_group_intercepts(draw, n_groups=n_groups)
            if group_intercepts is None:
                raise ValueError("draw does not contain random-intercept parameters")

            if spec.random_slope_feature_idx is not None:
                slope_feature_idx = int(spec.random_slope_feature_idx)
                group_slopes = _effective_group_slopes(
                    draw, feature_idx=slope_feature_idx, n_groups=n_groups
                )
                if group_slopes is None:
                    raise ValueError("draw does not contain random-slope parameters")

    y_rep: List[float] = []
    for i in range(n):
        group = None if spec.group_idx is None else int(spec.group_idx[i])
        offset_i = None
        if spec.offset is not None:
            offset_i = float(spec.offset[i])
        eta = _eta_glm(
            x_row=spec.x[i],
            draw=draw,
            include_intercept=bool(spec.include_intercept),
            group=group,
            group_intercepts=group_intercepts,
            slope_feature_idx=slope_feature_idx,
            group_slopes=group_slopes,
            offset=offset_i,
        )

        if spec.kind == "linear":
            y_rep.append(float(rng.gauss(eta, 1.0)))
        elif spec.kind == "logistic":
            p = _sigmoid(eta)
            y_rep.append(1.0 if rng.random() < p else 0.0)
        elif spec.kind == "poisson":
            lam = math.exp(eta)
            y_rep.append(float(_poisson_sample(rng, lam)))
        else:
            raise NotImplementedError(
                "PPC currently supports kind in {'linear','logistic','poisson'}"
            )

    return y_rep


@dataclass(frozen=True)
class NegBinomSpec:
    """Minimal spec for PPC of NB2 (mean/dispersion) regression with log link."""

    x: List[List[float]]
    y: List[int]
    include_intercept: bool = True
    offset: Optional[List[float]] = None


def replicate_negbin(
    spec: NegBinomSpec,
    draw: Mapping[str, float],
    *,
    seed: int,
) -> List[float]:
    rng = random.Random(int(seed))
    if spec.offset is not None and len(spec.offset) != len(spec.x):
        raise ValueError("spec.offset must have length n")

    log_alpha = float(draw.get("log_alpha", 0.0))
    alpha = math.exp(log_alpha)

    y_rep: List[float] = []
    for i, x_row in enumerate(spec.x):
        offset_i = None if spec.offset is None else float(spec.offset[i])
        eta = _eta_glm(
            x_row=x_row,
            draw=draw,
            include_intercept=bool(spec.include_intercept),
            group=None,
            group_intercepts=None,
            slope_feature_idx=None,
            group_slopes=None,
            offset=offset_i,
        )
        mu = math.exp(float(eta))
        y_rep.append(float(_negbin_sample_nb2(rng, mu=mu, alpha=alpha)))
    return y_rep


def ppc_negbin_from_sample(
    spec: NegBinomSpec,
    sample_raw: Mapping[str, Any],
    *,
    param_names: Optional[Sequence[str]] = None,
    n_draws: int = 50,
    seed: int = 0,
    stats_fn: Optional[Any] = None,
) -> PpcStats:
    posterior = sample_raw.get("posterior")
    if not isinstance(posterior, Mapping):
        raise ValueError("sample_raw must contain a 'posterior' mapping")

    if param_names is None:
        pn = sample_raw.get("param_names")
        if not isinstance(pn, list) or not pn:
            raise ValueError("sample_raw must contain non-empty 'param_names' or pass param_names=")
        param_names = [str(s) for s in pn]

    flat = _flatten_posterior(posterior, param_names=param_names)
    if n_draws <= 0:
        raise ValueError("n_draws must be > 0")
    draws = flat if n_draws >= len(flat) else random.Random(seed).sample(flat, k=int(n_draws))

    def mk_draw(vs: Sequence[float]) -> Dict[str, float]:
        return {str(name): float(v) for name, v in zip(param_names, vs)}

    if stats_fn is None:
        stats_fn = lambda kind, y: default_stats(kind, y)

    observed = stats_fn("negbin", spec.y)  # type: ignore[arg-type]
    replicated: List[Dict[str, float]] = []
    for i, vs in enumerate(draws):
        d = mk_draw(vs)
        y_rep = replicate_negbin(spec, d, seed=int(seed) + 20_000 + i)
        replicated.append(stats_fn("negbin", y_rep))

    return PpcStats(observed=observed, replicated=replicated)


@dataclass(frozen=True)
class OrderedSpec:
    """Minimal spec for PPC of ordered outcomes (logit/probit)."""

    x: List[List[float]]
    y: List[int]
    n_levels: int
    link: str = "logit"  # "logit" | "probit"


def _ordered_probs(
    *,
    eta: float,
    cutpoints: Sequence[float],
    link: str,
) -> List[float]:
    if link == "logit":
        cdf = [_sigmoid(float(cutpoints[0]) - float(eta))]
        for ck in cutpoints[1:]:
            cdf.append(_sigmoid(float(ck) - float(eta)))
    elif link == "probit":
        cdf = [_normal_cdf(float(cutpoints[0]) - float(eta))]
        for ck in cutpoints[1:]:
            cdf.append(_normal_cdf(float(ck) - float(eta)))
    else:
        raise ValueError(f"unsupported ordered link: {link!r}")

    probs: List[float] = []
    probs.append(float(cdf[0]))
    for j in range(1, len(cdf)):
        probs.append(float(cdf[j] - cdf[j - 1]))
    probs.append(float(1.0 - cdf[-1]))

    # Guard against tiny negative values from float subtraction.
    probs = [0.0 if p < 0.0 and p > -1e-12 else float(p) for p in probs]
    s = sum(probs)
    if not (math.isfinite(s) and s > 0.0):
        raise ValueError("ordered model produced invalid probabilities")
    return [p / s for p in probs]


def replicate_ordered(
    spec: OrderedSpec,
    draw: Mapping[str, float],
    *,
    seed: int,
) -> List[float]:
    rng = random.Random(int(seed))
    k = int(spec.n_levels)
    if k < 2:
        raise ValueError("spec.n_levels must be >= 2")
    if any(int(v) < 0 or int(v) >= k for v in spec.y):
        raise ValueError("spec.y must be in [0, n_levels)")

    p = len(spec.x[0]) if spec.x else 0
    beta = [float(draw.get(f"beta{j+1}", 0.0)) for j in range(p)]
    cut_raw = [float(draw.get(f"cut_raw{j+1}", 0.0)) for j in range(k - 1)]
    cutpoints = _cutpoints_from_raw(cut_raw)
    if len(cutpoints) != k - 1:
        raise ValueError("cutpoints length mismatch")

    y_rep: List[float] = []
    for x_row in spec.x:
        eta = 0.0
        for bj, xj in zip(beta, x_row):
            eta += float(bj) * float(xj)
        probs = _ordered_probs(eta=float(eta), cutpoints=cutpoints, link=str(spec.link))
        y_rep.append(float(_categorical_sample(rng, probs)))
    return y_rep


def ppc_ordered_from_sample(
    spec: OrderedSpec,
    sample_raw: Mapping[str, Any],
    *,
    param_names: Optional[Sequence[str]] = None,
    n_draws: int = 50,
    seed: int = 0,
    stats_fn: Optional[Any] = None,
) -> PpcStats:
    posterior = sample_raw.get("posterior")
    if not isinstance(posterior, Mapping):
        raise ValueError("sample_raw must contain a 'posterior' mapping")

    if param_names is None:
        pn = sample_raw.get("param_names")
        if not isinstance(pn, list) or not pn:
            raise ValueError("sample_raw must contain non-empty 'param_names' or pass param_names=")
        param_names = [str(s) for s in pn]

    flat = _flatten_posterior(posterior, param_names=param_names)
    if n_draws <= 0:
        raise ValueError("n_draws must be > 0")
    draws = flat if n_draws >= len(flat) else random.Random(seed).sample(flat, k=int(n_draws))

    def mk_draw(vs: Sequence[float]) -> Dict[str, float]:
        return {str(name): float(v) for name, v in zip(param_names, vs)}

    if stats_fn is None:
        stats_fn = lambda kind, y: default_stats(kind, y)

    observed = stats_fn("ordinal_ordered", spec.y)  # type: ignore[arg-type]
    replicated: List[Dict[str, float]] = []
    for i, vs in enumerate(draws):
        d = mk_draw(vs)
        y_rep = replicate_ordered(spec, d, seed=int(seed) + 30_000 + i)
        replicated.append(stats_fn("ordinal_ordered", y_rep))

    return PpcStats(observed=observed, replicated=replicated)


def ppc_glm_from_sample(
    spec: GlmSpec,
    sample_raw: Mapping[str, Any],
    *,
    param_names: Optional[Sequence[str]] = None,
    n_draws: int = 50,
    seed: int = 0,
    stats_fn: Optional[Any] = None,
) -> PpcStats:
    """Compute simple PPC stats from a raw `nextstat.sample(...)` dict.

    `stats_fn(kind, y) -> dict[str,float]` can be supplied; defaults to `default_stats`.
    """

    posterior = sample_raw.get("posterior")
    if not isinstance(posterior, Mapping):
        raise ValueError("sample_raw must contain a 'posterior' mapping")

    if param_names is None:
        # Use the stable order returned by the sampler.
        pn = sample_raw.get("param_names")
        if not isinstance(pn, list) or not pn:
            raise ValueError("sample_raw must contain non-empty 'param_names' or pass param_names=")
        param_names = [str(s) for s in pn]

    flat = _flatten_posterior(posterior, param_names=param_names)
    if n_draws <= 0:
        raise ValueError("n_draws must be > 0")
    draws = flat if n_draws >= len(flat) else random.Random(seed).sample(flat, k=int(n_draws))

    def mk_draw(vs: Sequence[float]) -> Dict[str, float]:
        return {str(name): float(v) for name, v in zip(param_names, vs)}

    if stats_fn is None:
        stats_fn = lambda kind, y: default_stats(kind, y)

    observed = stats_fn(spec.kind, spec.y)  # type: ignore[arg-type]
    replicated: List[Dict[str, float]] = []
    for i, vs in enumerate(draws):
        d = mk_draw(vs)
        y_rep = replicate_glm(spec, d, seed=int(seed) + 10_000 + i)
        replicated.append(stats_fn(spec.kind, y_rep))

    return PpcStats(observed=observed, replicated=replicated)


__all__ = [
    "PpcStats",
    "NegBinomSpec",
    "OrderedSpec",
    "default_stats",
    "replicate_glm",
    "replicate_negbin",
    "replicate_ordered",
    "ppc_glm_from_sample",
    "ppc_negbin_from_sample",
    "ppc_ordered_from_sample",
]
