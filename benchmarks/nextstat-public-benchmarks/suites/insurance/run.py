#!/usr/bin/env python3
"""Insurance Chain Ladder benchmark: deterministic CL + Mack stochastic CL.

Backends:
- `nextstat` (chain_ladder / mack_chain_ladder via PyO3 bindings)
- `chainladder-python` (optional parity baseline)

Cases:
- chain_ladder_10x10: classic 10x10 Taylor-Ashe triangle
- mack_10x10: Mack stochastic CL on Taylor-Ashe
- chain_ladder_20x20: larger synthetic 20x20 triangle
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_json_obj(obj: dict[str, Any]) -> str:
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
    return {
        "min": min(values) if values else 0.0,
        "median": _pctl(values, 0.5),
        "p95": _pctl(values, 0.95),
    }


def _maybe_import(name: str) -> tuple[bool, Optional[str]]:
    try:
        __import__(name)
        mod = sys.modules.get(name)
        return True, str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return False, None


def _max_abs_rel_diff(a: list[float], b: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not a or not b or len(a) != len(b):
        return None, None
    abs_d = 0.0
    rel_d = 0.0
    for x, y in zip(a, b):
        d = abs(float(x) - float(y))
        abs_d = max(abs_d, d)
        denom = max(abs(float(x)), abs(float(y)), 1.0)
        rel_d = max(rel_d, d / denom)
    return float(abs_d), float(rel_d)


def _scalar_abs_rel_diff(a: float, b: float) -> tuple[float, float]:
    d = abs(float(a) - float(b))
    denom = max(abs(float(a)), abs(float(b)), 1.0)
    return float(d), float(d / denom)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Classic Taylor-Ashe 10x10 cumulative triangle (origin x development).
TAYLOR_ASHE = [
    [357848, 1124788, 1735330, 2218270, 2745596, 3319994, 3466336, 3606286, 3833515, 3901463],
    [352118, 1236139, 2170033, 3353322, 3799067, 4120063, 4647867, 4914039, 5339085],
    [290507, 1292306, 2218525, 3235179, 3985995, 4132918, 4628910, 4909315],
    [310608, 1418858, 2195047, 3757447, 4029929, 4381982, 4588268],
    [443160, 1136350, 2128333, 2897821, 3402672, 3873311],
    [396132, 1333217, 2180715, 2985752, 3691712],
    [440832, 1288463, 2419861, 3483130],
    [359480, 1421128, 2864498],
    [376686, 1363294],
    [344014],
]


def _cumulative_to_incremental(triangle: list[list[float]]) -> list[list[float]]:
    """Convert cumulative upper-left triangle to incremental form."""
    inc = []
    for row in triangle:
        inc_row = [float(row[0])]
        for j in range(1, len(row)):
            inc_row.append(float(row[j]) - float(row[j - 1]))
        inc.append(inc_row)
    return inc


def gen_triangle(n: int, seed: int, base_claims: float = 500000.0) -> list[list[float]]:
    """Generate a synthetic n x n cumulative upper-left triangle."""
    rng = np.random.default_rng(int(seed))
    # Typical cumulative development factors (ratios decay toward 1).
    factors = [3.0, 1.6, 1.3, 1.15, 1.08, 1.05, 1.03, 1.02, 1.01] + [1.005] * max(n - 10, 0)
    factors = factors[: n - 1]
    triangle: list[list[float]] = []
    for i in range(n):
        first = float(base_claims) * (1.0 + 0.05 * float(i)) + rng.normal(0, float(base_claims) * 0.1)
        first = max(first, 1000.0)
        row = [float(first)]
        for j in range(n - i - 1):
            f = float(factors[j]) if j < len(factors) else 1.002
            next_val = row[-1] * f * (1.0 + rng.normal(0, 0.02))
            row.append(max(float(next_val), row[-1]))  # cumulative: non-decreasing
        triangle.append(row)
    return triangle


def _get_triangle(kind: str, n: int, seed: int) -> tuple[list[list[float]], dict[str, Any]]:
    """Return (triangle, spec) for a given case configuration."""
    if n == 10 and kind in ("chain_ladder", "mack"):
        triangle = [[float(v) for v in row] for row in TAYLOR_ASHE]
        spec = {"source": "taylor_ashe", "n": 10, "kind": kind}
    else:
        triangle = gen_triangle(n, seed)
        spec = {"source": "synthetic", "n": int(n), "seed": int(seed), "kind": kind}
    return triangle, spec


# ---------------------------------------------------------------------------
# NextStat backend
# ---------------------------------------------------------------------------

def _run_nextstat_chain_ladder(triangle: list[list[float]]) -> dict[str, Any]:
    import nextstat
    return dict(nextstat.chain_ladder(triangle=triangle))


def _run_nextstat_mack(triangle: list[list[float]], conf_level: float) -> dict[str, Any]:
    import nextstat
    return dict(nextstat.mack_chain_ladder(triangle=triangle, conf_level=conf_level))


# ---------------------------------------------------------------------------
# chainladder-python baseline
# ---------------------------------------------------------------------------

def _build_cl_triangle(triangle: list[list[float]]):
    """Build a chainladder Triangle object from upper-left cumulative triangle."""
    import chainladder as cl  # type: ignore[import-not-found]
    import pandas as pd  # type: ignore[import-not-found]
    from datetime import datetime

    n = len(triangle)
    origins: list[datetime] = []
    devs: list[datetime] = []
    values: list[float] = []
    for i in range(n):
        for j in range(len(triangle[i])):
            origins.append(datetime(2000 + i, 1, 1))
            devs.append(datetime(2000 + i + j + 1, 1, 1))
            values.append(float(triangle[i][j]))
    df = pd.DataFrame({"origin": origins, "development": devs, "values": values})
    tri = cl.Triangle(df, origin="origin", development="development", columns="values", cumulative=True)
    return tri


def _run_cl_chain_ladder(triangle: list[list[float]]) -> dict[str, Any]:
    import chainladder as cl  # type: ignore[import-not-found]

    tri = _build_cl_triangle(triangle)
    model = cl.Chainladder().fit(tri)
    ultimates = [float(v) for v in np.asarray(model.ultimate_.values).flatten() if np.isfinite(v)]
    ibnr = [float(v) for v in np.asarray(model.ibnr_.values).flatten() if np.isfinite(v)]
    ldf = [float(v) for v in np.asarray(model.ldf_.values).flatten() if np.isfinite(v)]
    total_ibnr = float(sum(ibnr))
    return {
        "development_factors": ldf,
        "ultimates": ultimates,
        "ibnr": ibnr,
        "total_ibnr": total_ibnr,
    }


def _run_cl_mack(triangle: list[list[float]]) -> dict[str, Any]:
    import chainladder as cl  # type: ignore[import-not-found]

    tri = _build_cl_triangle(triangle)
    mack = cl.MackChainladder().fit(tri)
    ultimates = [float(v) for v in np.asarray(mack.ultimate_.values).flatten() if np.isfinite(v)]
    ibnr = [float(v) for v in np.asarray(mack.ibnr_.values).flatten() if np.isfinite(v)]
    ldf = [float(v) for v in np.asarray(mack.ldf_.values).flatten() if np.isfinite(v)]
    # Standard errors -- chainladder-python exposes full_std_err_ on MackChainladder
    se: list[float] = []
    try:
        se = [float(v) for v in np.asarray(mack.full_std_err_.values).flatten() if np.isfinite(v)]
    except Exception:
        pass
    total_ibnr = float(sum(ibnr))
    return {
        "development_factors": ldf,
        "ultimates": ultimates,
        "ibnr": ibnr,
        "total_ibnr": total_ibnr,
        "se": se,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Insurance Chain Ladder benchmark runner.")
    ap.add_argument("--case", required=True, help="Case id for reporting.")
    ap.add_argument("--kind", required=True, choices=["chain_ladder", "mack"],
                    help="Which method to benchmark.")
    ap.add_argument("--n", type=int, default=10, help="Triangle size (n x n). Default 10.")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    ap.add_argument("--seed", type=int, default=42, help="Seed for synthetic data generation.")
    ap.add_argument("--repeat", type=int, default=50, help="Number of timing repeats.")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--conf-level", type=float, default=0.95,
                    help="Confidence level for Mack method. Default 0.95.")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kind = str(args.kind)
    n = int(args.n)
    seed = int(args.seed)
    repeat = int(args.repeat)
    conf_level = float(args.conf_level)

    # -- Dataset --
    triangle, spec = _get_triangle(kind, n, seed)
    dataset_id = f"{'taylor_ashe' if spec['source'] == 'taylor_ashe' else 'synthetic'}:{kind}:n{n}:seed{seed}"
    dataset_sha = sha256_json_obj(spec)
    dataset = {"id": dataset_id, "sha256": dataset_sha, "spec": spec}

    # -- Dependency detection --
    has_ns, ns_v = _maybe_import("nextstat")
    has_cl, cl_v = _maybe_import("chainladder")

    meta: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "nextstat_version": ns_v,
        "chainladder_version": cl_v,
    }

    status = "ok"
    reason: Optional[str] = None

    # -- NextStat run --
    ns_result: Optional[dict[str, Any]] = None
    ns_runs: list[float] = []

    if has_ns:
        try:
            if kind == "chain_ladder":
                # Warmup
                _run_nextstat_chain_ladder(triangle)
                for _ in range(repeat):
                    t0 = time.perf_counter()
                    ns_result = _run_nextstat_chain_ladder(triangle)
                    ns_runs.append(float(time.perf_counter() - t0))
            else:
                # Mack
                _run_nextstat_mack(triangle, conf_level)
                for _ in range(repeat):
                    t0 = time.perf_counter()
                    ns_result = _run_nextstat_mack(triangle, conf_level)
                    ns_runs.append(float(time.perf_counter() - t0))
        except Exception as e:
            status = "failed"
            reason = f"nextstat_error:{type(e).__name__}:{e}"
    else:
        status = "failed"
        reason = "nextstat_not_available"

    # -- chainladder-python baseline --
    cl_result: Optional[dict[str, Any]] = None
    cl_runs: list[float] = []

    if has_cl:
        try:
            if kind == "chain_ladder":
                _run_cl_chain_ladder(triangle)
                for _ in range(repeat):
                    t0 = time.perf_counter()
                    cl_result = _run_cl_chain_ladder(triangle)
                    cl_runs.append(float(time.perf_counter() - t0))
            else:
                _run_cl_mack(triangle)
                for _ in range(repeat):
                    t0 = time.perf_counter()
                    cl_result = _run_cl_mack(triangle)
                    cl_runs.append(float(time.perf_counter() - t0))
        except Exception as e:
            cl_result = None
            cl_runs = []
            # Non-fatal: baseline is optional.

    # -- Parity --
    parity: dict[str, Any] = {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}}

    if ns_result is not None and cl_result is not None:
        metrics: dict[str, Any] = {}

        # Development factors
        ns_ldf = [float(v) for v in ns_result.get("development_factors", [])]
        cl_ldf = [float(v) for v in cl_result.get("development_factors", [])]
        ldf_abs, ldf_rel = _max_abs_rel_diff(ns_ldf, cl_ldf)
        metrics["dev_factors_max_abs_diff"] = ldf_abs
        metrics["dev_factors_max_rel_diff"] = ldf_rel

        # Ultimates
        ns_ult = [float(v) for v in ns_result.get("ultimates", [])]
        cl_ult = [float(v) for v in cl_result.get("ultimates", [])]
        ult_abs, ult_rel = _max_abs_rel_diff(ns_ult, cl_ult)
        metrics["ultimates_max_abs_diff"] = ult_abs
        metrics["ultimates_max_rel_diff"] = ult_rel

        # Total IBNR
        ns_total = float(ns_result.get("total_ibnr", 0.0))
        cl_total = float(cl_result.get("total_ibnr", 0.0))
        ibnr_abs, ibnr_rel = _scalar_abs_rel_diff(ns_total, cl_total)
        metrics["total_ibnr_abs_diff"] = float(ibnr_abs)
        metrics["total_ibnr_rel_diff"] = float(ibnr_rel)

        # Mack SE (if available)
        if kind == "mack":
            ns_se = [float(v) for v in ns_result.get("se", [])]
            cl_se = [float(v) for v in cl_result.get("se", [])]
            se_abs, se_rel = _max_abs_rel_diff(ns_se, cl_se)
            metrics["se_max_abs_diff"] = se_abs
            metrics["se_max_rel_diff"] = se_rel

        parity = {
            "status": "ok" if ult_abs is not None else "warn",
            "reference": {"name": "chainladder-python", "version": str(cl_v or "")},
            "metrics": metrics,
        }

    # -- Timing --
    timing: dict[str, Any] = {
        "wall_time_s": {
            "nextstat": _summary(ns_runs) if ns_runs else {"min": 0.0, "median": 0.0, "p95": 0.0},
        },
        "raw": {
            "repeat": repeat,
            "policy": "median",
            "runs_s": {
                "nextstat": [float(x) for x in ns_runs],
            },
        },
    }
    if cl_runs:
        timing["wall_time_s"]["chainladder"] = _summary(cl_runs)
        timing["raw"]["runs_s"]["chainladder"] = [float(x) for x in cl_runs]
        # Speedup ratio (median)
        ns_med = _pctl(ns_runs, 0.5) if ns_runs else 0.0
        cl_med = _pctl(cl_runs, 0.5) if cl_runs else 0.0
        if ns_med > 0.0 and cl_med > 0.0:
            timing["speedup_vs_chainladder"] = float(cl_med / ns_med)

    # -- Config --
    config: dict[str, Any] = {
        "kind": kind,
        "n": n,
        "seed": seed,
        "repeat": repeat,
    }
    if kind == "mack":
        config["conf_level"] = conf_level

    # -- Results --
    results: dict[str, Any] = {
        "nextstat": ns_result,
        "chainladder": cl_result,
    }

    # -- Output document --
    doc: dict[str, Any] = {
        "schema_version": "nextstat.insurance_benchmark_result.v1",
        "suite": "insurance",
        "case": str(args.case),
        "deterministic": bool(args.deterministic),
        "status": status,
        "reason": reason,
        "meta": meta,
        "dataset": dataset,
        "config": config,
        "parity": parity,
        "timing": timing,
        "results": results,
    }

    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return 0 if status != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
