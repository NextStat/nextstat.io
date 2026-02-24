#!/usr/bin/env python3
"""Gamma / Tweedie GLM benchmark runner.

Insurance pricing (Gamma) and healthcare costs (Tweedie) are prime use cases.

Cases:
- gamma_{10k,100k}: Gamma GLM with log link, p=20
- tweedie_{10k,100k}: Tweedie GLM (p=1.5), p=20

Baselines: statsmodels (Gamma/Tweedie families), glum, sklearn (Gamma only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from nextstat._core import GammaRegressionModel, TweedieRegressionModel, fit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.bench_env import collect_environment


def sha256_json_obj(obj: dict) -> str:
    b = (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _pctl(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * float(p)
    i = int(k)
    j = min(i + 1, len(xs) - 1)
    return xs[i] * (1.0 - (k - i)) + xs[j] * (k - i)


def _summary(values: list[float]) -> dict[str, float]:
    return {"min": min(values) if values else 0.0, "median": _pctl(values, 0.5), "p95": _pctl(values, 0.95)}


def _max_abs_rel_diff(a: list[float], b: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not a or not b or len(a) != len(b):
        return None, None
    abs_d = rel_d = 0.0
    for x, y in zip(a, b):
        d = abs(float(x) - float(y))
        abs_d = max(abs_d, d)
        denom = max(abs(float(x)), abs(float(y)), 1.0)
        rel_d = max(rel_d, d / denom)
    return float(abs_d), float(rel_d)


def _scalar_rel_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    denom = max(abs(float(a)), abs(float(b)), 1e-15)
    return abs(float(a) - float(b)) / denom


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def gen_gamma_data(n: int, p: int, seed: int) -> dict[str, Any]:
    """Generate Gamma-distributed response with log link."""
    rng = np.random.default_rng(int(seed))
    x = rng.standard_normal((n, p))
    beta = np.linspace(0.1, 0.3, num=p)
    intercept = 1.0
    eta = intercept + x @ beta
    mu = np.exp(eta)
    shape = 5.0
    y = rng.gamma(shape, scale=mu / shape, size=n)
    y = np.maximum(y, 1e-10)  # Ensure positive
    return {"x": x.tolist(), "y": y.tolist(), "beta_true": beta.tolist(), "intercept_true": intercept, "shape_true": shape}


def gen_tweedie_data(n: int, p: int, seed: int, *, tweedie_p: float = 1.5) -> dict[str, Any]:
    """Generate Tweedie-distributed response (compound Poisson-Gamma)."""
    rng = np.random.default_rng(int(seed))
    x = rng.standard_normal((n, p))
    beta = np.linspace(0.1, 0.3, num=p)
    intercept = 1.0
    eta = intercept + x @ beta
    mu = np.exp(eta)
    # Approximate Tweedie via Poisson-Gamma mixture
    phi = 1.0
    lam = mu ** (2.0 - tweedie_p) / (phi * (2.0 - tweedie_p))
    alpha_param = (2.0 - tweedie_p) / (tweedie_p - 1.0)
    beta_param = phi * (tweedie_p - 1.0) * mu ** (tweedie_p - 1.0)
    y = np.zeros(n)
    for i in range(n):
        n_claims = rng.poisson(max(lam[i], 0.01))
        if n_claims > 0:
            claims = rng.gamma(alpha_param, scale=beta_param[i], size=n_claims)
            y[i] = float(np.sum(claims))
    return {"x": x.tolist(), "y": y.tolist(), "beta_true": beta.tolist(), "intercept_true": intercept, "tweedie_p": tweedie_p}


# ---------------------------------------------------------------------------
# NextStat fits
# ---------------------------------------------------------------------------


def _fit_ns(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    x = data["x"]  # List[List[float]] — 2D row-major covariate matrix
    y = data["y"]  # List[float] — positive response values

    if kind == "gamma":
        model = GammaRegressionModel(x, y, include_intercept=True)
    elif kind == "tweedie":
        tweedie_p = data.get("tweedie_p", 1.5)
        model = TweedieRegressionModel(x, y, p=tweedie_p, include_intercept=True)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    result = fit(model)
    return {
        "params": list(result.parameters),
        "nll": float(result.nll),
        "converged": bool(result.converged),
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def _baseline_statsmodels(kind: str, data: dict[str, Any]) -> Optional[dict[str, Any]]:
    try:
        import statsmodels.api as sm
    except ImportError:
        return None

    X = np.asarray(data["x"], dtype=float)
    X = sm.add_constant(X)
    y = np.asarray(data["y"], dtype=float)

    if kind == "gamma":
        family = sm.families.Gamma(link=sm.families.links.Log())
    elif kind == "tweedie":
        tweedie_p = data.get("tweedie_p", 1.5)
        family = sm.families.Tweedie(link=sm.families.links.Log(), var_power=tweedie_p)
    else:
        return None

    mod = sm.GLM(y, X, family=family)
    res = mod.fit()
    return {
        "params": res.params.tolist(),
        "nll": float(-res.llf) if res.llf is not None else None,
        "converged": bool(res.converged),
    }


def _baseline_glum(kind: str, data: dict[str, Any]) -> Optional[dict[str, Any]]:
    try:
        from glum import GeneralizedLinearRegressor
    except ImportError:
        return None

    X = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)

    if kind == "gamma":
        mod = GeneralizedLinearRegressor(family="gamma", link="log", fit_intercept=True, alpha=0)
    elif kind == "tweedie":
        tweedie_p = data.get("tweedie_p", 1.5)
        mod = GeneralizedLinearRegressor(family=f"tweedie", link="log", fit_intercept=True, alpha=0, power=tweedie_p)
    else:
        return None

    mod.fit(X, y)
    coef = [float(mod.intercept_)] + mod.coef_.tolist()
    return {"params": coef, "nll": None, "converged": True}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--kind", required=True, choices=["gamma", "tweedie"])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--p", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kind = str(args.kind)
    n = int(args.n)
    p_cov = int(args.p)
    seed = int(args.seed)
    repeat = int(args.repeat)

    spec = {"kind": kind, "n": n, "p": p_cov, "seed": seed}
    if kind == "gamma":
        data = gen_gamma_data(n, p_cov, seed)
    else:
        data = gen_tweedie_data(n, p_cov, seed)
    dataset = {"id": f"generated:gamma_tweedie:{args.case}", "sha256": sha256_json_obj(spec), "spec": spec}

    status = "ok"
    reason: Optional[str] = None

    try:
        ns_result = _fit_ns(kind, data)
    except Exception as e:
        obj = {
            "schema_version": "nextstat.gamma_tweedie_benchmark_result.v1",
            "suite": "gamma_tweedie",
            "case": str(args.case),
            "environment": collect_environment(),
            "status": "failed",
            "reason": f"nextstat_error:{type(e).__name__}:{e}",
            "dataset": dataset,
            "config": {"kind": kind, "n": n, "p": p_cov},
            "parity": {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}},
            "timing": {"wall_time_s": {"min": 0.0, "median": 0.0, "p95": 0.0}, "raw": {"repeat": repeat, "runs_s": []}},
            "results": {"nextstat": {}, "baseline": None},
        }
        out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        return 2

    # Timing
    runs_s: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _fit_ns(kind, data)
        runs_s.append(float(time.perf_counter() - t0))
    timing = {"wall_time_s": _summary(runs_s), "raw": {"repeat": repeat, "policy": "median", "runs_s": runs_s}}

    # Parity
    baseline_obj: Optional[dict[str, Any]] = None
    parity: dict[str, Any] = {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}}
    timing_baseline: dict[str, Any] = {}

    if not args.skip_baselines:
        try:
            # Primary: statsmodels
            bl = _baseline_statsmodels(kind, data)
            if bl is not None:
                baseline_obj = bl
                runs_bl: list[float] = []
                for _ in range(max(1, repeat // 5)):
                    t0 = time.perf_counter()
                    _baseline_statsmodels(kind, data)
                    runs_bl.append(float(time.perf_counter() - t0))
                timing_baseline = {"name": "statsmodels", "wall_time_s": _summary(runs_bl), "raw": {"repeat": len(runs_bl), "runs_s": runs_bl}}

                coef_abs, coef_rel = _max_abs_rel_diff(ns_result["params"], bl["params"])
                nll_rel = _scalar_rel_diff(ns_result.get("nll"), bl.get("nll"))
                parity = {
                    "status": "ok",
                    "reference": {"name": "statsmodels", "version": ""},
                    "metrics": {"coef_max_abs_diff": coef_abs, "coef_max_rel_diff": coef_rel, "nll_rel_diff": nll_rel},
                }
        except Exception as e:
            status = "warn"
            reason = f"baseline_error:{type(e).__name__}:{e}"

    obj = {
        "schema_version": "nextstat.gamma_tweedie_benchmark_result.v1",
        "suite": "gamma_tweedie",
        "case": str(args.case),
        "environment": collect_environment(),
        "status": status,
        "reason": reason,
        "dataset": dataset,
        "config": {"kind": kind, "n": n, "p": p_cov},
        "parity": parity,
        "timing": timing,
        "timing_baseline": timing_baseline,
        "results": {"nextstat": ns_result, "baseline": baseline_obj},
    }
    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return 0 if status != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
