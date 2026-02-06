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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .data import GlmSpec


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


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


def _eta_glm(
    *,
    x_row: Sequence[float],
    draw: Mapping[str, float],
    include_intercept: bool,
    group: Optional[int],
    group_intercepts: Optional[Sequence[float]],
) -> float:
    eta = 0.0
    if include_intercept:
        eta += float(draw.get("intercept", 0.0))
    for j, xj in enumerate(x_row):
        eta += float(draw.get(f"beta{j+1}", 0.0)) * float(xj)
    if group is not None and group_intercepts is not None:
        eta += float(group_intercepts[int(group)])
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
    if kind == "linear":
        v = sum((v - mu) ** 2 for v in ys) / max(1.0, float(len(ys) - 1))
        out["var"] = float(v)
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
    group_intercepts = None
    if spec.group_idx is not None:
        if ng is None:
            raise ValueError("spec.n_groups must be set when spec.group_idx is present")
        group_intercepts = _effective_group_intercepts(draw, n_groups=int(ng))

    y_rep: List[float] = []
    for i in range(n):
        group = None if spec.group_idx is None else int(spec.group_idx[i])
        eta = _eta_glm(
            x_row=spec.x[i],
            draw=draw,
            include_intercept=bool(spec.include_intercept),
            group=group,
            group_intercepts=group_intercepts,
        )

        if spec.kind == "linear":
            y_rep.append(float(rng.gauss(eta, 1.0)))
        elif spec.kind == "logistic":
            p = _sigmoid(eta)
            y_rep.append(1.0 if rng.random() < p else 0.0)
        else:
            raise NotImplementedError("PPC currently supports kind in {'linear','logistic'}")

    return y_rep


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
    "default_stats",
    "replicate_glm",
    "ppc_glm_from_sample",
]

