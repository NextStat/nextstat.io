#!/usr/bin/env python3
"""Ordinal regression benchmark runner.

Cases:
- ordered_logit_{1k,10k,100k}: Ordered Logit, K=5 categories, p=10
- ordered_probit_{1k,10k,100k}: Ordered Probit, K=5 categories, p=10

Baseline: statsmodels.miscmodels.ordinal_model.OrderedModel
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

import nextstat

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


def _scalar_rel_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    denom = max(abs(float(a)), abs(float(b)), 1e-15)
    return abs(float(a) - float(b)) / denom


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


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def gen_ordinal_data(
    n: int,
    p: int,
    n_levels: int,
    seed: int,
) -> dict[str, Any]:
    """Generate ordinal regression data with known thresholds."""
    rng = np.random.default_rng(int(seed))
    x = rng.standard_normal((n, p))
    beta = np.linspace(0.3, 0.8, num=p)
    eta = x @ beta

    # Evenly spaced thresholds
    thresholds = np.linspace(-1.0, 1.0, num=n_levels - 1)

    # Generate ordinal outcomes
    y = np.zeros(n, dtype=int)
    for i in range(n):
        probs = []
        for k in range(n_levels):
            if k == 0:
                probs.append(1.0 / (1.0 + np.exp(-(thresholds[0] - eta[i]))))
            elif k == n_levels - 1:
                probs.append(1.0 - 1.0 / (1.0 + np.exp(-(thresholds[-1] - eta[i]))))
            else:
                p_upper = 1.0 / (1.0 + np.exp(-(thresholds[k] - eta[i])))
                p_lower = 1.0 / (1.0 + np.exp(-(thresholds[k - 1] - eta[i])))
                probs.append(p_upper - p_lower)
        probs = np.maximum(probs, 1e-10)
        probs = np.array(probs) / sum(probs)
        y[i] = rng.choice(n_levels, p=probs)

    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "n_levels": n_levels,
        "beta_true": beta.tolist(),
        "thresholds_true": thresholds.tolist(),
    }


# ---------------------------------------------------------------------------
# NextStat fits
# ---------------------------------------------------------------------------


def _fit_ns(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    x = data["x"]
    y = data["y"]
    n_levels = data["n_levels"]

    if kind == "ordered_logit":
        model = nextstat.OrderedLogitModel(x, y, n_levels=n_levels)
    elif kind == "ordered_probit":
        model = nextstat.OrderedProbitModel(x, y, n_levels=n_levels)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    result = nextstat.fit(model)
    params = list(result.parameters)
    names = model.parameter_names()
    return {
        "params": {str(nm): float(p) for nm, p in zip(names, params)},
        "nll": float(result.nll),
        "converged": bool(result.converged),
    }


# ---------------------------------------------------------------------------
# Baseline: statsmodels OrderedModel
# ---------------------------------------------------------------------------


def _baseline_statsmodels(kind: str, data: dict[str, Any]) -> Optional[dict[str, Any]]:
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
    except ImportError:
        return None

    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=int)
    distr = "logit" if kind == "ordered_logit" else "probit"

    mod = OrderedModel(y, x, distr=distr)
    res = mod.fit(disp=False, method="bfgs")

    return {
        "params": res.params.tolist(),
        "nll": float(-res.llf),
        "converged": bool(res.mle_retvals.get("converged", False)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--kind", required=True, choices=["ordered_logit", "ordered_probit"])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--p", type=int, default=10)
    ap.add_argument("--n-levels", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kind = str(args.kind)
    n = int(args.n)
    p = int(args.p)
    n_levels = int(args.n_levels)
    seed = int(args.seed)
    repeat = int(args.repeat)

    spec = {"kind": kind, "n": n, "p": p, "n_levels": n_levels, "seed": seed}
    data = gen_ordinal_data(n, p, n_levels, seed)
    dataset = {"id": f"generated:ordinal:{args.case}", "sha256": sha256_json_obj(spec), "spec": spec}

    status = "ok"
    reason: Optional[str] = None

    try:
        ns_result = _fit_ns(kind, data)
    except Exception as e:
        obj = {
            "schema_version": "nextstat.ordinal_benchmark_result.v1",
            "suite": "ordinal",
            "case": str(args.case),
            "environment": collect_environment(),
            "status": "failed",
            "reason": f"nextstat_error:{type(e).__name__}:{e}",
            "dataset": dataset,
            "config": {"kind": kind, "n": n, "p": p, "n_levels": n_levels},
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
            bl = _baseline_statsmodels(kind, data)
            if bl is not None:
                baseline_obj = bl
                runs_bl: list[float] = []
                for _ in range(max(1, repeat // 5)):
                    t0 = time.perf_counter()
                    _baseline_statsmodels(kind, data)
                    runs_bl.append(float(time.perf_counter() - t0))
                timing_baseline = {"name": "statsmodels", "wall_time_s": _summary(runs_bl), "raw": {"repeat": len(runs_bl), "runs_s": runs_bl}}

                nll_diff = abs(ns_result["nll"] - bl["nll"]) if bl.get("nll") is not None else None
                nll_rel = _scalar_rel_diff(ns_result.get("nll"), bl.get("nll"))
                parity = {
                    "status": "ok",
                    "reference": {"name": "statsmodels.OrderedModel", "version": ""},
                    "metrics": {"nll_abs_diff": nll_diff, "nll_rel_diff": nll_rel},
                }
        except Exception as e:
            status = "warn"
            reason = f"baseline_error:{type(e).__name__}:{e}"

    obj = {
        "schema_version": "nextstat.ordinal_benchmark_result.v1",
        "suite": "ordinal",
        "case": str(args.case),
        "environment": collect_environment(),
        "status": status,
        "reason": reason,
        "dataset": dataset,
        "config": {"kind": kind, "n": n, "p": p, "n_levels": n_levels},
        "parity": parity,
        "timing": timing,
        "timing_baseline": timing_baseline,
        "results": {"nextstat": ns_result, "baseline": baseline_obj},
    }
    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return 0 if status != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
