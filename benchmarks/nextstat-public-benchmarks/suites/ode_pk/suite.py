#!/usr/bin/env python3
"""ODE PK benchmark suite runner.

Writes:
- per-case JSON files (nextstat.ode_pk_benchmark_result.v1)
- a suite index JSON (nextstat.ode_pk_benchmark_suite_result.v1)

Cases:
- transit_1cpt_sparse:  Transit absorption, 20 time points, single dose
- transit_1cpt_dense:   Transit absorption, 200 time points, multiple doses
- mm_1cpt_sparse:       Michaelis-Menten 1-cpt, 20 time points
- mm_1cpt_dense:        Michaelis-Menten 1-cpt, 200 time points
- tmdd_sparse:          TMDD, 30 time points
- tmdd_dense:           TMDD, 300 time points, multiple doses
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import nextstat
    _NS_VERSION = str(nextstat.__version__)
except ImportError:
    _NS_VERSION = "unavailable"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _maybe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

_CASES = [
    {"case_id": "transit_1cpt_sparse", "case_key": "transit_1cpt_sparse", "repeat": 50, "baseline_repeat": 5},
    {"case_id": "transit_1cpt_dense", "case_key": "transit_1cpt_dense", "repeat": 30, "baseline_repeat": 5},
    {"case_id": "mm_1cpt_sparse", "case_key": "mm_1cpt_sparse", "repeat": 50, "baseline_repeat": 5},
    {"case_id": "mm_1cpt_dense", "case_key": "mm_1cpt_dense", "repeat": 30, "baseline_repeat": 5},
    {"case_id": "tmdd_sparse", "case_key": "tmdd_sparse", "repeat": 50, "baseline_repeat": 5},
    {"case_id": "tmdd_dense", "case_key": "tmdd_dense", "repeat": 30, "baseline_repeat": 5},
]

_SMOKE_CASES = [
    {"case_id": "transit_1cpt_sparse", "case_key": "transit_1cpt_sparse", "repeat": 10, "baseline_repeat": 2},
    {"case_id": "mm_1cpt_sparse", "case_key": "mm_1cpt_sparse", "repeat": 10, "baseline_repeat": 2},
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="ODE PK benchmark suite")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="Fast mode: fewer cases, lower repeat, skip baselines.")
    ap.add_argument("--skip-baselines", action="store_true", help="Skip scipy baseline timing.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    run_py = Path(__file__).resolve().parent / "run.py"
    suite_cases = _SMOKE_CASES if args.smoke else _CASES

    index_cases: list[dict[str, Any]] = []
    n_ok = n_warn = n_failed = n_skipped = 0
    worst_case = "none"
    worst_wall = -1.0

    for c in suite_cases:
        case_id = str(c["case_id"])
        case_key = str(c["case_key"])
        repeat = int(c["repeat"])
        baseline_repeat = int(c["baseline_repeat"])
        out_path = cases_dir / f"{case_id}.json"

        # Build command
        cmd = [
            sys.executable, str(run_py),
            "--case", case_id,
            "--case-key", case_key,
            "--out", str(out_path),
            "--seed", str(int(args.seed)),
            "--repeat", str(repeat),
            "--baseline-repeat", str(baseline_repeat),
        ]

        if args.skip_baselines or args.smoke:
            cmd.append("--skip-baselines")

        print(f"  [{case_id}] case_key={case_key} repeat={repeat} ...", flush=True)
        p = subprocess.run(cmd)

        # Parse result
        try:
            obj = json.loads(out_path.read_text())
        except Exception:
            obj = {}

        case_status = str(obj.get("status") or ("failed" if p.returncode != 0 else "warn"))
        if case_status == "ok":
            n_ok += 1
        elif case_status == "skipped":
            n_skipped += 1
        elif case_status == "warn":
            n_warn += 1
        else:
            n_failed += 1

        sha = sha256_file(out_path) if out_path.exists() else "0" * 64
        cfg = obj.get("config", {}) if isinstance(obj, dict) else {}
        parity_status = str((obj.get("parity", {}) or {}).get("status", "skipped"))

        # Extract median wall time (solve)
        timing = obj.get("timing", {}) or {}
        ns_solve_wall = (timing.get("wall_time_s", {}) or {}).get("nextstat_solve", {})
        median = _maybe_float(ns_solve_wall.get("median") if isinstance(ns_solve_wall, dict) else None)
        median = float(median) if median is not None else 0.0
        if median >= worst_wall:
            worst_wall = median
            worst_case = case_id

        # Extract speedups if present
        solve_speedup = _maybe_float(timing.get("speedup_solve_vs_scipy"))
        nll_speedup = _maybe_float(timing.get("speedup_nll_vs_scipy"))

        entry: dict[str, Any] = {
            "case": case_id,
            "path": os.path.relpath(out_path, out_dir),
            "sha256": sha,
            "status": case_status,
            "model_type": str(cfg.get("model_type", "")),
            "n_times": int(cfg.get("n_times", 0)) if isinstance(cfg, dict) else 0,
            "n_doses": int(cfg.get("n_doses", 0)) if isinstance(cfg, dict) else 0,
            "wall_time_median_s": median,
            "parity_status": parity_status if parity_status in ("ok", "warn", "skipped") else "warn",
        }
        if solve_speedup is not None:
            entry["speedup_solve_vs_scipy"] = float(solve_speedup)
        if nll_speedup is not None:
            entry["speedup_nll_vs_scipy"] = float(nll_speedup)

        index_cases.append(entry)

        extra = ""
        if solve_speedup is not None:
            extra += f" solve_speedup={solve_speedup:.1f}x"
        if nll_speedup is not None:
            extra += f" nll_speedup={nll_speedup:.1f}x"
        print(f"  [{case_id}] {case_status.upper()} median={median:.6f}s parity={parity_status}{extra}", flush=True)

    # -- Suite index ----------------------------------------------------------
    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "nextstat_version": _NS_VERSION,
    }
    index = {
        "schema_version": "nextstat.ode_pk_benchmark_suite_result.v1",
        "suite": "ode_pk",
        "meta": meta,
        "cases": index_cases,
        "summary": {
            "n_cases": len(index_cases),
            "n_ok": n_ok,
            "n_warn": n_warn,
            "n_failed": n_failed,
            "n_skipped": n_skipped,
            "worst_case": worst_case,
        },
    }
    suite_path = out_dir / "ode_pk_suite.json"
    suite_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    print(f"\n  {n_ok} ok / {n_warn} warn / {n_failed} failed / {n_skipped} skipped ({len(index_cases)} total)")
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
