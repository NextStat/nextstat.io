#!/usr/bin/env python3
"""Interval-censored survival benchmark runner.

Unique capability: lifelines does NOT support interval censoring.
scikit-survival has minimal support. NS has 3 IC models with analytical gradients.

Cases:
- ic_weibull_1k / ic_weibull_10k: Interval-censored Weibull
- ic_exponential_1k / ic_exponential_10k: Interval-censored Exponential
- ic_lognormal_1k / ic_lognormal_10k: Interval-censored LogNormal

Baseline: scipy MLE (manual NLL + minimize) for parity; no Python package
does this natively, which is precisely the point.
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
    a = xs[i]
    b = xs[j]
    t = k - i
    return a * (1.0 - t) + b * t


def _summary(values: list[float]) -> dict[str, float]:
    return {"min": min(values) if values else 0.0, "median": _pctl(values, 0.5), "p95": _pctl(values, 0.95)}


def _scalar_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return abs(float(a) - float(b))


def _scalar_rel_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    denom = max(abs(float(a)), abs(float(b)), 1e-15)
    return abs(float(a) - float(b)) / denom


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def gen_ic_data(
    n: int,
    seed: int,
    distribution: str = "weibull",
    *,
    shape: float = 1.5,
    scale: float = 10.0,
    inspection_rate: float = 0.3,
) -> dict[str, Any]:
    """Generate interval-censored survival data.

    Each subject has a true event time drawn from the specified distribution.
    Inspection times are generated from a Poisson process. The event is
    observed between the last inspection before and first inspection after
    the true event time.
    """
    rng = np.random.default_rng(int(seed))

    # True event times
    if distribution == "weibull":
        true_times = scale * rng.weibull(shape, size=n)
    elif distribution == "exponential":
        true_times = rng.exponential(scale, size=n)
    elif distribution == "lognormal":
        true_times = rng.lognormal(mean=np.log(scale), sigma=0.5, size=n)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    time_lower = []
    time_upper = []
    censor_type = []

    max_followup = float(np.percentile(true_times, 95)) * 1.5

    for i in range(n):
        t_true = float(true_times[i])
        # Generate inspection times (Poisson process)
        inspections = []
        t = 0.0
        while t < max_followup:
            gap = rng.exponential(1.0 / inspection_rate)
            t += gap
            if t < max_followup:
                inspections.append(t)

        if not inspections:
            inspections = [max_followup]

        # Find interval containing true event time
        prev = 0.0
        found = False
        for insp in sorted(inspections):
            if t_true <= insp:
                time_lower.append(prev)
                time_upper.append(insp)
                censor_type.append("interval")
                found = True
                break
            prev = insp

        if not found:
            # Right-censored: event after last inspection
            time_lower.append(inspections[-1])
            time_upper.append(inspections[-1])  # ignored for right-censored; must be finite
            censor_type.append("right")

    return {
        "time_lower": time_lower,
        "time_upper": time_upper,
        "censor_type": censor_type,
        "true_shape": shape,
        "true_scale": scale,
    }


# ---------------------------------------------------------------------------
# NextStat fits
# ---------------------------------------------------------------------------


def _fit_ns(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    """Fit an interval-censored model via NextStat."""
    tl = data["time_lower"]
    tu = data["time_upper"]
    ct = data["censor_type"]

    if kind == "ic_weibull":
        model = nextstat.IntervalCensoredWeibullModel(tl, tu, ct)
    elif kind == "ic_exponential":
        model = nextstat.IntervalCensoredExponentialModel(tl, tu, ct)
    elif kind == "ic_lognormal":
        model = nextstat.IntervalCensoredLogNormalModel(tl, tu, ct)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    result = nextstat.fit(model)
    params = list(result.parameters)
    names = model.parameter_names()
    nll = float(result.nll)

    return {
        "params": {str(n): float(p) for n, p in zip(names, params)},
        "nll": nll,
        "converged": bool(result.converged),
    }


# ---------------------------------------------------------------------------
# Baseline: scipy manual MLE (for parity only)
# ---------------------------------------------------------------------------


def _baseline_scipy(kind: str, data: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Manual MLE via scipy for parity comparison."""
    try:
        from scipy.optimize import minimize
        from scipy.stats import weibull_min, expon, lognorm
    except ImportError:
        return None

    tl = np.asarray(data["time_lower"], dtype=float)
    tu = np.asarray(data["time_upper"], dtype=float)
    ct = data["censor_type"]

    def nll_ic(params: np.ndarray) -> float:
        """Negative log-likelihood for interval-censored data."""
        total = 0.0
        for i in range(len(tl)):
            if kind == "ic_weibull":
                shape, scale = max(params[0], 1e-10), max(params[1], 1e-10)
                if ct[i] == "interval":
                    p = weibull_min.cdf(tu[i], shape, scale=scale) - weibull_min.cdf(tl[i], shape, scale=scale)
                else:
                    p = 1.0 - weibull_min.cdf(tl[i], shape, scale=scale)
            elif kind == "ic_exponential":
                rate = max(params[0], 1e-10)
                if ct[i] == "interval":
                    p = expon.cdf(tu[i], scale=1.0 / rate) - expon.cdf(tl[i], scale=1.0 / rate)
                else:
                    p = 1.0 - expon.cdf(tl[i], scale=1.0 / rate)
            elif kind == "ic_lognormal":
                mu, sigma = params[0], max(params[1], 1e-10)
                if ct[i] == "interval":
                    p = lognorm.cdf(tu[i], s=sigma, scale=np.exp(mu)) - lognorm.cdf(tl[i], s=sigma, scale=np.exp(mu))
                else:
                    p = 1.0 - lognorm.cdf(tl[i], s=sigma, scale=np.exp(mu))
            else:
                p = 1.0
            total -= np.log(max(p, 1e-300))
        return float(total)

    if kind == "ic_weibull":
        x0 = np.array([1.5, 10.0])
        bounds = [(0.01, 50.0), (0.01, 1000.0)]
    elif kind == "ic_exponential":
        x0 = np.array([0.1])
        bounds = [(1e-6, 100.0)]
    elif kind == "ic_lognormal":
        x0 = np.array([2.0, 0.5])
        bounds = [(-10.0, 10.0), (0.01, 10.0)]
    else:
        return None

    res = minimize(nll_ic, x0, method="L-BFGS-B", bounds=bounds)
    return {
        "params": res.x.tolist(),
        "nll": float(res.fun),
        "converged": bool(res.success),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--kind", required=True, choices=["ic_weibull", "ic_exponential", "ic_lognormal"])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kind = str(args.kind)
    n = int(args.n)
    seed = int(args.seed)
    repeat = int(args.repeat)

    # Distribution mapping
    dist_map = {"ic_weibull": "weibull", "ic_exponential": "exponential", "ic_lognormal": "lognormal"}
    distribution = dist_map[kind]
    spec = {"kind": kind, "n": n, "seed": seed, "distribution": distribution}
    data = gen_ic_data(n, seed, distribution=distribution)
    dataset = {"id": f"generated:ic_survival:{args.case}", "sha256": sha256_json_obj(spec), "spec": spec}

    status = "ok"
    reason: Optional[str] = None

    # --- NextStat fit ---
    try:
        ns_result = _fit_ns(kind, data)
    except Exception as e:
        obj = {
            "schema_version": "nextstat.ic_survival_benchmark_result.v1",
            "suite": "ic_survival",
            "case": str(args.case),
            "environment": collect_environment(),
            "status": "failed",
            "reason": f"nextstat_error:{type(e).__name__}:{e}",
            "dataset": dataset,
            "config": {"kind": kind, "n": n},
            "parity": {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}},
            "timing": {"wall_time_s": {"min": 0.0, "median": 0.0, "p95": 0.0}, "raw": {"repeat": repeat, "runs_s": []}},
            "results": {"nextstat": {}, "baseline": None},
        }
        out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        return 2

    # --- Timing ---
    runs_s: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _fit_ns(kind, data)
        runs_s.append(float(time.perf_counter() - t0))

    timing = {"wall_time_s": _summary(runs_s), "raw": {"repeat": repeat, "policy": "median", "runs_s": runs_s}}

    # --- Parity via scipy ---
    baseline_obj: Optional[dict[str, Any]] = None
    parity: dict[str, Any] = {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}}
    timing_baseline: dict[str, Any] = {}

    if not args.skip_baselines:
        try:
            bl = _baseline_scipy(kind, data)
            if bl is not None:
                baseline_obj = bl
                # Timing baseline
                runs_bl: list[float] = []
                for _ in range(max(1, repeat // 5)):
                    t0 = time.perf_counter()
                    _baseline_scipy(kind, data)
                    runs_bl.append(float(time.perf_counter() - t0))
                timing_baseline = {"name": "scipy_manual", "wall_time_s": _summary(runs_bl), "raw": {"repeat": len(runs_bl), "runs_s": runs_bl}}

                nll_diff = _scalar_diff(ns_result.get("nll"), bl.get("nll"))
                nll_rel = _scalar_rel_diff(ns_result.get("nll"), bl.get("nll"))
                parity = {
                    "status": "ok",
                    "reference": {"name": "scipy_manual_mle", "version": ""},
                    "metrics": {"nll_abs_diff": nll_diff, "nll_rel_diff": nll_rel},
                }
        except Exception as e:
            status = "warn"
            reason = f"baseline_error:{type(e).__name__}:{e}"

    obj = {
        "schema_version": "nextstat.ic_survival_benchmark_result.v1",
        "suite": "ic_survival",
        "case": str(args.case),
        "environment": collect_environment(),
        "status": status,
        "reason": reason,
        "dataset": dataset,
        "config": {"kind": kind, "n": n},
        "parity": parity,
        "timing": timing,
        "timing_baseline": timing_baseline,
        "results": {"nextstat": ns_result, "baseline": baseline_obj},
    }
    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return 0 if status != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
