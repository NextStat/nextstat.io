#!/usr/bin/env python3
"""Linear Mixed Models benchmark runner.

Cases:
- lmm_ri_{g100_n10,g100_n50,g1000_n10,g1000_n50}: Random intercept
- lmm_rs_{g100_n10,g100_n50,g1000_n10,g1000_n50}: Random intercept + slope

Baseline: statsmodels.regression.mixed_linear_model.MixedLM
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


def gen_lmm_data(
    n_groups: int,
    n_per_group: int,
    n_x: int,
    seed: int,
    *,
    random_slope: bool = False,
) -> dict[str, Any]:
    """Generate balanced LMM data with random intercepts (and optionally slopes)."""
    rng = np.random.default_rng(int(seed))
    n = n_groups * n_per_group

    group_idx = np.repeat(np.arange(n_groups), n_per_group)
    x = rng.standard_normal((n, n_x))
    beta = np.linspace(0.5, 1.0, num=n_x)

    # Random intercepts
    sigma_u = 1.5
    u = rng.normal(0, sigma_u, size=n_groups)

    # Random slopes on first covariate
    sigma_v = 0.5 if random_slope else 0.0
    v = rng.normal(0, sigma_v, size=n_groups) if random_slope else np.zeros(n_groups)

    # Residuals
    sigma_e = 1.0
    eps = rng.normal(0, sigma_e, size=n)

    y = x @ beta + u[group_idx] + v[group_idx] * x[:, 0] + eps

    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "group_idx": group_idx.tolist(),
        "n_groups": n_groups,
        "n_per_group": n_per_group,
        "n_x": n_x,
        "random_slope": random_slope,
        "beta_true": beta.tolist(),
        "sigma_u_true": sigma_u,
        "sigma_v_true": sigma_v,
        "sigma_e_true": sigma_e,
    }


# ---------------------------------------------------------------------------
# NextStat fits
# ---------------------------------------------------------------------------


def _fit_ns(data: dict[str, Any]) -> dict[str, Any]:
    x = data["x"]
    y = data["y"]
    group_idx = data["group_idx"]
    n_groups = data["n_groups"]
    random_slope = data["random_slope"]

    kwargs: dict[str, Any] = {
        "include_intercept": True,
        "group_idx": group_idx,
        "n_groups": n_groups,
    }
    if random_slope:
        kwargs["random_slope_feature_idx"] = 0

    model = nextstat.LmmMarginalModel(x, y, **kwargs)
    result = nextstat.fit(model)
    return {
        "params": list(result.parameters),
        "nll": float(result.nll),
        "converged": bool(result.converged),
        "parameter_names": model.parameter_names(),
    }


# ---------------------------------------------------------------------------
# Baseline: statsmodels MixedLM
# ---------------------------------------------------------------------------


def _baseline_statsmodels(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    try:
        import statsmodels.api as sm
        from statsmodels.regression.mixed_linear_model import MixedLM
    except ImportError:
        return None

    X = np.asarray(data["x"], dtype=float)
    X = sm.add_constant(X)
    y = np.asarray(data["y"], dtype=float)
    groups = np.asarray(data["group_idx"], dtype=int)

    if data["random_slope"]:
        # Random intercept + random slope on first covariate
        exog_re = X[:, [0, 1]]  # intercept + first covariate
    else:
        exog_re = None  # Random intercept only (default)

    mod = MixedLM(y, X, groups=groups, exog_re=exog_re)
    res = mod.fit(reml=False)

    return {
        "params": res.fe_params.tolist(),
        "nll": float(-res.llf),
        "converged": bool(res.converged),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--n-groups", type=int, required=True)
    ap.add_argument("--n-per-group", type=int, required=True)
    ap.add_argument("--n-x", type=int, default=5)
    ap.add_argument("--random-slope", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_groups = int(args.n_groups)
    n_per_group = int(args.n_per_group)
    n_x = int(args.n_x)
    seed = int(args.seed)
    repeat = int(args.repeat)
    random_slope = bool(args.random_slope)

    spec = {"n_groups": n_groups, "n_per_group": n_per_group, "n_x": n_x, "random_slope": random_slope, "seed": seed}
    data = gen_lmm_data(n_groups, n_per_group, n_x, seed, random_slope=random_slope)
    dataset = {"id": f"generated:lmm:{args.case}", "sha256": sha256_json_obj(spec), "spec": spec}

    status = "ok"
    reason: Optional[str] = None

    try:
        ns_result = _fit_ns(data)
    except Exception as e:
        obj = {
            "schema_version": "nextstat.lmm_benchmark_result.v1",
            "suite": "lmm",
            "case": str(args.case),
            "environment": collect_environment(),
            "status": "failed",
            "reason": f"nextstat_error:{type(e).__name__}:{e}",
            "dataset": dataset,
            "config": {"n_groups": n_groups, "n_per_group": n_per_group, "n_x": n_x, "random_slope": random_slope},
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
        _fit_ns(data)
        runs_s.append(float(time.perf_counter() - t0))
    timing = {"wall_time_s": _summary(runs_s), "raw": {"repeat": repeat, "policy": "median", "runs_s": runs_s}}

    # Parity
    baseline_obj: Optional[dict[str, Any]] = None
    parity: dict[str, Any] = {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}}
    timing_baseline: dict[str, Any] = {}

    if not args.skip_baselines:
        try:
            bl = _baseline_statsmodels(data)
            if bl is not None:
                baseline_obj = bl
                runs_bl: list[float] = []
                for _ in range(max(1, repeat // 5)):
                    t0 = time.perf_counter()
                    _baseline_statsmodels(data)
                    runs_bl.append(float(time.perf_counter() - t0))
                timing_baseline = {"name": "statsmodels.MixedLM", "wall_time_s": _summary(runs_bl), "raw": {"repeat": len(runs_bl), "runs_s": runs_bl}}

                # Fixed effects parity
                ns_fe = ns_result["params"][:len(bl["params"])]
                coef_abs, coef_rel = _max_abs_rel_diff(ns_fe, bl["params"])
                nll_rel = _scalar_rel_diff(ns_result.get("nll"), bl.get("nll"))
                parity = {
                    "status": "ok",
                    "reference": {"name": "statsmodels.MixedLM", "version": ""},
                    "metrics": {"fe_coef_max_abs_diff": coef_abs, "fe_coef_max_rel_diff": coef_rel, "nll_rel_diff": nll_rel},
                }
        except Exception as e:
            status = "warn"
            reason = f"baseline_error:{type(e).__name__}:{e}"

    obj = {
        "schema_version": "nextstat.lmm_benchmark_result.v1",
        "suite": "lmm",
        "case": str(args.case),
        "environment": collect_environment(),
        "status": status,
        "reason": reason,
        "dataset": dataset,
        "config": {"n_groups": n_groups, "n_per_group": n_per_group, "n_x": n_x, "random_slope": random_slope},
        "parity": parity,
        "timing": timing,
        "timing_baseline": timing_baseline,
        "results": {"nextstat": ns_result, "baseline": baseline_obj},
    }
    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return 0 if status != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
