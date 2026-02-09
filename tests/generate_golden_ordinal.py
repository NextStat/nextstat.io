#!/usr/bin/env python3
"""
Generate deterministic "golden" fixtures for ordinal ordered outcomes models.

These fixtures are used by pure-Python tests at runtime
(`tests/python/test_golden_ordinal_fixtures.py`) and can optionally be enriched
with external references (Stan/PyMC) via merge scripts under `tests/external/`.

Usage:
  PYTHONPATH=bindings/ns-py/python python3 tests/generate_golden_ordinal.py
"""

from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, List, Sequence


ROOT = Path(__file__).resolve().parents[0]
FIXTURES_DIR = ROOT / "fixtures" / "ordinal"


def _randn(rng) -> float:
    u1 = max(rng.random(), 1e-12)
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _sample_logistic(rng) -> float:
    # Inverse CDF of standard logistic.
    u = min(max(rng.random(), 1e-12), 1.0 - 1e-12)
    return math.log(u) - math.log1p(-u)


def _cutpoints_from_raw(raw: Sequence[float]) -> List[float]:
    def softplus(x: float) -> float:
        if x > 0.0:
            return x + math.log1p(math.exp(-x))
        return math.log1p(math.exp(x))

    if not raw:
        return []
    c = [float(raw[0])]
    for r in raw[1:]:
        c.append(float(c[-1] + softplus(float(r))))
    return c


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _Phi(x: float) -> float:
    return 0.5 * math.erfc(-float(x) / math.sqrt(2.0))


def _predict_proba_ordered(
    *,
    link: str,
    coef: Sequence[float],
    cutpoints: Sequence[float],
    x: Sequence[Sequence[float]],
) -> List[List[float]]:
    out: List[List[float]] = []
    for row in x:
        eta = sum(float(a) * float(b) for a, b in zip(row, coef))
        if link == "logit":
            cdf = [_sigmoid(float(cutpoints[0]) - eta)]
            for ck in cutpoints[1:]:
                cdf.append(_sigmoid(float(ck) - eta))
        elif link == "probit":
            cdf = [_Phi(float(cutpoints[0]) - eta)]
            for ck in cutpoints[1:]:
                cdf.append(_Phi(float(ck) - eta))
        else:
            raise ValueError("unknown link")

        probs: List[float] = [float(cdf[0])]
        for j in range(1, len(cdf)):
            probs.append(float(cdf[j] - cdf[j - 1]))
        probs.append(float(1.0 - cdf[-1]))

        # Guard tiny negative roundoff.
        probs = [0.0 if p < 0.0 and p > -1e-12 else float(p) for p in probs]
        s = sum(probs)
        probs = [p / s for p in probs]
        out.append(probs)
    return out


def _fit_fixture(*, link: str, x: list[list[float]], y: list[int], n_levels: int) -> dict[str, Any]:
    try:
        import nextstat as ns  # type: ignore
    except Exception as e:
        raise SystemExit(
            "Unable to import nextstat. Run with:\n"
            "  PYTHONPATH=bindings/ns-py/python python3 tests/generate_golden_ordinal.py"
        ) from e

    if link == "logit":
        fit = ns.ordinal.ordered_logit.fit(x, y, n_levels=n_levels)
        core = ns._core.OrderedLogitModel(x, y, n_levels=n_levels)
    elif link == "probit":
        fit = ns.ordinal.ordered_probit.fit(x, y, n_levels=n_levels)
        core = ns._core.OrderedProbitModel(x, y, n_levels=n_levels)
    else:
        raise ValueError("unknown link")

    params_hat = [float(v) for v in (list(fit.coef) + list(fit.cut_raw))]
    if len(params_hat) != int(core.n_params()):
        raise SystemExit("params length mismatch vs core model")

    # Basic grid for parity comparisons (external tools should report predictions here).
    grid_x = [[-2.0], [-1.0], [0.0], [1.0], [2.0]]
    grid_pred = fit.predict_proba(grid_x)

    return {
        "kind": "ordinal_ordered",
        "link": link,
        "n_levels": int(n_levels),
        "n": len(y),
        "p": len(x[0]) if x else 0,
        "x": x,
        "y": y,
        "grid_x": grid_x,
        "parameter_names": list(core.parameter_names()),
        "params_hat": params_hat,
        "nll_at_hat": float(fit.nll),
        "derived": {
            "coef": [float(v) for v in fit.coef],
            "cut_raw": [float(v) for v in fit.cut_raw],
            "cutpoints": [float(v) for v in fit.cutpoints],
            "grid_pred_proba": grid_pred,
        },
        "source": {
            "tool": "nextstat",
            "version": ns.__version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }


def main() -> int:
    import random

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Deterministic small problem.
    rng = random.Random(0)
    n = 300
    x: list[list[float]] = []
    y_logit: list[int] = []
    y_probit: list[int] = []

    beta_true = 1.0
    cutpoints_true = [-0.3, 0.8]  # 3 levels

    for _ in range(n):
        xi = _randn(rng)
        eta = beta_true * xi

        z_logit = eta + _sample_logistic(rng)
        k = 0
        for c in cutpoints_true:
            if z_logit <= c:
                break
            k += 1
        y_logit.append(int(k))

        z_probit = eta + _randn(rng)
        k = 0
        for c in cutpoints_true:
            if z_probit <= c:
                break
            k += 1
        y_probit.append(int(k))

        x.append([float(xi)])

    fx_logit = _fit_fixture(link="logit", x=x, y=y_logit, n_levels=3)
    fx_probit = _fit_fixture(link="probit", x=x, y=y_probit, n_levels=3)

    (FIXTURES_DIR / "ordered_logit_small.json").write_text(json.dumps(fx_logit, indent=2, sort_keys=True) + "\n")
    (FIXTURES_DIR / "ordered_probit_small.json").write_text(
        json.dumps(fx_probit, indent=2, sort_keys=True) + "\n"
    )

    # Self-check: derived predictions can be recomputed without nextstat.
    for fx in (fx_logit, fx_probit):
        raw = fx["derived"]["cut_raw"]
        cutpoints = _cutpoints_from_raw(raw)
        grid_pred = _predict_proba_ordered(
            link=fx["link"],
            coef=fx["derived"]["coef"],
            cutpoints=cutpoints,
            x=fx["grid_x"],
        )
        # Compare a couple of entries loosely (float noise only).
        if abs(grid_pred[0][0] - fx["derived"]["grid_pred_proba"][0][0]) > 1e-10:
            raise SystemExit("grid_pred self-check failed")

    print("Wrote:")
    print(" -", FIXTURES_DIR / "ordered_logit_small.json")
    print(" -", FIXTURES_DIR / "ordered_probit_small.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

