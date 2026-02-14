#!/usr/bin/env python3
"""Meta-analysis benchmark: fixed-effects and random-effects pooling.

This is intentionally small and self-contained so outsiders can rerun it.

Cases:
- fixed_effects_10: 10-study inverse-variance fixed-effects meta-analysis
- random_effects_10: 10-study DerSimonian-Laird random-effects meta-analysis
- random_effects_50: 50-study DerSimonian-Laird random-effects meta-analysis
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


def _maybe_import(name: str) -> tuple[bool, Optional[str]]:
    try:
        __import__(name)
        mod = sys.modules.get(name)
        return True, str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return False, None


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _gen_meta(n_studies: int, *, seed: int, true_effect: float = 0.5, tau2: float = 0.0) -> dict[str, Any]:
    """Generate synthetic meta-analysis data.

    For fixed-effects (tau2=0), all true study effects equal true_effect.
    For random-effects (tau2>0), study-level effects are drawn from
    N(true_effect, tau2), then within-study noise is added.
    """
    rng = np.random.default_rng(int(seed))
    n = int(n_studies)

    if tau2 > 0:
        effects = rng.normal(true_effect, np.sqrt(tau2), n).astype(float)
    else:
        effects = np.full(n, true_effect, dtype=float)

    ses = rng.uniform(0.1, 0.5, n).astype(float)
    # Add within-study sampling noise
    effects = effects + rng.normal(0, ses)

    return {
        "estimates": effects.tolist(),
        "standard_errors": ses.tolist(),
        "true_effect": float(true_effect),
        "tau2": float(tau2),
        "n_studies": int(n),
    }


# ---------------------------------------------------------------------------
# NextStat runners
# ---------------------------------------------------------------------------

def _fit_fixed(data: dict[str, Any]) -> dict[str, Any]:
    result = nextstat.meta_fixed(
        estimates=data["estimates"],
        standard_errors=data["standard_errors"],
    )
    return dict(result)


def _fit_random(data: dict[str, Any]) -> dict[str, Any]:
    result = nextstat.meta_random(
        estimates=data["estimates"],
        standard_errors=data["standard_errors"],
    )
    return dict(result)


# ---------------------------------------------------------------------------
# Baseline: pymare
# ---------------------------------------------------------------------------

def _baseline_pymare_fixed(data: dict[str, Any]) -> dict[str, Any]:
    from pymare import Dataset  # type: ignore[import-not-found]
    from pymare.estimators import WeightedLeastSquares  # type: ignore[import-not-found]

    y = np.asarray(data["estimates"], dtype=float)
    ses = np.asarray(data["standard_errors"], dtype=float)
    v = ses ** 2

    ds = Dataset(y=y, v=v)
    est = WeightedLeastSquares()
    est.fit_dataset(ds)
    summary = est.summary()
    fe = summary.get_fe_stats()

    pooled = float(fe["est"][0][0])
    pooled_se = float(fe["se"][0][0])
    ci_lower = float(fe["ci_l"][0][0])
    ci_upper = float(fe["ci_u"][0][0])

    return {
        "pooled_estimate": pooled,
        "pooled_se": pooled_se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def _baseline_pymare_random(data: dict[str, Any]) -> dict[str, Any]:
    from pymare import Dataset  # type: ignore[import-not-found]
    from pymare.estimators import DerSimonianLaird  # type: ignore[import-not-found]

    y = np.asarray(data["estimates"], dtype=float)
    ses = np.asarray(data["standard_errors"], dtype=float)
    v = ses ** 2

    ds = Dataset(y=y, v=v)
    est = DerSimonianLaird()
    est.fit_dataset(ds)
    summary = est.summary()
    fe = summary.get_fe_stats()

    pooled = float(fe["est"][0][0])
    pooled_se = float(fe["se"][0][0])
    ci_lower = float(fe["ci_l"][0][0])
    ci_upper = float(fe["ci_u"][0][0])
    tau2 = float(np.asarray(summary.tau2, dtype=float).reshape(-1)[0])

    return {
        "pooled_estimate": pooled,
        "pooled_se": pooled_se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "tau2": tau2,
    }


# ---------------------------------------------------------------------------
# Parity comparison
# ---------------------------------------------------------------------------

def _scalar_diffs(ns_val: float, baseline_val: float) -> tuple[float, float]:
    """Return (abs_diff, rel_diff) between two scalar values."""
    abs_d = abs(ns_val - baseline_val)
    denom = max(abs(ns_val), abs(baseline_val), 1.0)
    rel_d = abs_d / denom
    return float(abs_d), float(rel_d)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--kind", required=True, choices=["fixed", "random"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=100)
    ap.add_argument("--n-studies", type=int, default=10)
    ap.add_argument("--true-effect", type=float, default=0.5)
    ap.add_argument("--tau2", type=float, default=0.0)
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kind = str(args.kind)
    seed = int(args.seed)
    repeat = int(args.repeat)
    n_studies = int(args.n_studies)
    true_effect = float(args.true_effect)
    tau2 = float(args.tau2)

    # Build deterministic dataset spec
    spec = {
        "kind": kind,
        "n_studies": n_studies,
        "true_effect": true_effect,
        "tau2": tau2,
        "seed": seed,
    }
    data = _gen_meta(n_studies, seed=seed, true_effect=true_effect, tau2=tau2)
    dataset = {"id": f"generated:meta_analysis:{args.case}", "sha256": sha256_json_obj(spec), "spec": spec}

    has_pymare, pymare_v = _maybe_import("pymare")

    status = "ok"
    reason: Optional[str] = None

    # Select runner
    if kind == "fixed":
        run_fit = _fit_fixed
    else:
        run_fit = _fit_random

    try:
        ns_result = run_fit(data)
    except Exception as e:
        obj = {
            "schema_version": "nextstat.meta_analysis_benchmark_result.v1",
            "suite": "meta_analysis",
            "case": str(args.case),
            "deterministic": bool(args.deterministic),
            "status": "failed",
            "reason": f"nextstat_error:{type(e).__name__}:{e}",
            "meta": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "nextstat_version": nextstat.__version__,
                "numpy_version": np.__version__,
                "pymare_version": pymare_v,
            },
            "dataset": dataset,
            "config": {"kind": kind, "n_studies": n_studies},
            "parity": {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}},
            "timing": {
                "wall_time_s": {"min": 0.0, "median": 0.0, "p95": 0.0},
                "raw": {"repeat": repeat, "policy": "median", "runs_s": []},
            },
            "timing_baseline": {
                "wall_time_s": {"min": 0.0, "median": 0.0, "p95": 0.0},
                "raw": {"repeat": repeat, "policy": "median", "runs_s": []},
            },
            "results": {"nextstat": {}, "baseline": None},
        }
        out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        return 2

    # Timing: NextStat
    runs_ns_s: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = run_fit(data)
        runs_ns_s.append(float(time.perf_counter() - t0))

    timing_ns = {
        "wall_time_s": _summary(runs_ns_s),
        "raw": {"repeat": repeat, "policy": "median", "runs_s": [float(x) for x in runs_ns_s]},
    }

    baseline_obj: Optional[dict[str, Any]] = None
    parity: dict[str, Any] = {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}}
    timing_baseline: dict[str, Any] = {
        "wall_time_s": {"min": 0.0, "median": 0.0, "p95": 0.0},
        "raw": {"repeat": repeat, "policy": "median", "runs_s": []},
    }

    try:
        if not has_pymare:
            raise RuntimeError("missing_dependency:pymare")

        if kind == "fixed":
            baseline_run = _baseline_pymare_fixed
        else:
            baseline_run = _baseline_pymare_random

        baseline_result = baseline_run(data)

        # Timing: pymare
        runs_base_s: list[float] = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            _ = baseline_run(data)
            runs_base_s.append(float(time.perf_counter() - t0))

        timing_baseline = {
            "wall_time_s": _summary(runs_base_s),
            "raw": {"repeat": repeat, "policy": "median", "runs_s": [float(x) for x in runs_base_s]},
        }

        baseline_obj = {
            "pooled_estimate": baseline_result["pooled_estimate"],
            "pooled_se": baseline_result["pooled_se"],
            "extra": {"backend": "pymare"},
        }
        if "tau2" in baseline_result:
            baseline_obj["tau2"] = baseline_result["tau2"]

        # Parity metrics
        pe_abs, pe_rel = _scalar_diffs(
            float(ns_result["estimate"]),
            float(baseline_result["pooled_estimate"]),
        )
        se_abs, se_rel = _scalar_diffs(
            float(ns_result["se"]),
            float(baseline_result["pooled_se"]),
        )

        metrics: dict[str, Any] = {
            "pooled_estimate_abs_diff": pe_abs,
            "pooled_estimate_rel_diff": pe_rel,
            "pooled_se_abs_diff": se_abs,
            "pooled_se_rel_diff": se_rel,
        }

        ns_tau2 = ns_result.get("heterogeneity", {}).get("tau_squared")
        bl_tau2 = baseline_result.get("tau2")
        if kind == "random" and ns_tau2 is not None and bl_tau2 is not None:
            tau2_abs, _ = _scalar_diffs(float(ns_tau2), float(bl_tau2))
            metrics["tau2_abs_diff"] = tau2_abs

        parity = {
            "status": "ok",
            "reference": {"name": "pymare", "version": str(pymare_v or "")},
            "metrics": metrics,
        }
    except Exception as e:
        status = "warn"
        reason = f"baseline_unavailable:{type(e).__name__}:{e}"
        parity = {"status": "warn", "reference": {"name": "", "version": ""}, "metrics": {}}

    cfg: dict[str, Any] = {
        "kind": kind,
        "n_studies": n_studies,
        "true_effect": true_effect,
    }
    if kind == "random":
        cfg["tau2_true"] = tau2

    ns_out: dict[str, Any] = {
        "pooled_estimate": ns_result.get("estimate"),
        "pooled_se": ns_result.get("se"),
        "ci_lower": ns_result.get("ci_lower"),
        "ci_upper": ns_result.get("ci_upper"),
        "z": ns_result.get("z"),
        "p_value": ns_result.get("p_value"),
    }
    if kind == "random":
        ns_out["tau2"] = ns_result.get("heterogeneity", {}).get("tau_squared")

    obj = {
        "schema_version": "nextstat.meta_analysis_benchmark_result.v1",
        "suite": "meta_analysis",
        "case": str(args.case),
        "deterministic": bool(args.deterministic),
        "status": status,
        "reason": reason,
        "meta": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "nextstat_version": nextstat.__version__,
            "numpy_version": np.__version__,
            "pymare_version": pymare_v,
        },
        "dataset": dataset,
        "config": cfg,
        "parity": parity,
        "timing": timing_ns,
        "timing_baseline": timing_baseline,
        "results": {"nextstat": ns_out, "baseline": baseline_obj},
    }
    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return 0 if status != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
