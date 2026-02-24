#!/usr/bin/env python3
"""PD models suite runner.

Runs all PD benchmark cases (Emax, Sigmoid Emax, IDR) and writes:
- per-case JSON files (pd_benchmark_result.v1)
- a suite index JSON (pd_benchmark_suite_result.v1)

Cases:
- emax_20conc / emax_100conc:             Emax dose-response fit
- sigmoid_emax_20conc / sigmoid_emax_100conc: Sigmoid Emax (Hill) fit
- idr_type1_50t / idr_type1_200t:         IDR Type I (InhibitProduction) simulation
- idr_type3_50t / idr_type3_200t:         IDR Type III (StimulateProduction) simulation
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
    {"case_id": "emax_20conc", "kind": "emax", "n_conc": 20, "repeat": 50},
    {"case_id": "emax_100conc", "kind": "emax", "n_conc": 100, "repeat": 30},
    {"case_id": "sigmoid_emax_20conc", "kind": "sigmoid_emax", "n_conc": 20, "repeat": 50},
    {"case_id": "sigmoid_emax_100conc", "kind": "sigmoid_emax", "n_conc": 100, "repeat": 30},
    {"case_id": "idr_type1_50t", "kind": "idr_type1", "n_times": 50, "repeat": 100},
    {"case_id": "idr_type3_50t", "kind": "idr_type3", "n_times": 50, "repeat": 100},
    {"case_id": "idr_type1_200t", "kind": "idr_type1", "n_times": 200, "repeat": 50},
    {"case_id": "idr_type3_200t", "kind": "idr_type3", "n_times": 200, "repeat": 50},
]

_SMOKE_CASES = [
    {"case_id": "emax_20conc", "kind": "emax", "n_conc": 20, "repeat": 10},
    {"case_id": "idr_type1_50t", "kind": "idr_type1", "n_times": 50, "repeat": 10},
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="PD models benchmark suite")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="Reduced set for CI speed.")
    ap.add_argument("--skip-baselines", action="store_true", help="Skip pure-Python baseline fits.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    run_py = Path(__file__).resolve().parent / "run.py"
    suite_cases = _SMOKE_CASES if args.smoke else _CASES

    index_cases: list[dict[str, Any]] = []
    n_ok = n_warn = n_failed = 0
    worst_case = "none"
    worst_wall = -1.0

    for c in suite_cases:
        case_id = str(c["case_id"])
        kind = str(c["kind"])
        out_path = cases_dir / f"{case_id}.json"

        # Build command
        cmd = [
            sys.executable, str(run_py),
            "--case", case_id,
            "--kind", kind,
            "--out", str(out_path),
            "--seed", str(int(args.seed)),
            "--repeat", str(int(c["repeat"])),
        ]

        # Emax / Sigmoid-Emax sizing
        if "n_conc" in c:
            cmd.extend(["--n-conc", str(int(c["n_conc"]))])
        # IDR sizing
        if "n_times" in c:
            cmd.extend(["--n-times", str(int(c["n_times"]))])

        if args.skip_baselines:
            cmd.append("--skip-baselines")

        print(f"  [{case_id}] kind={kind} ...", flush=True)
        p = subprocess.run(cmd)

        # Parse result
        try:
            obj = json.loads(out_path.read_text())
        except Exception:
            obj = {}

        case_status = str(obj.get("status") or ("failed" if p.returncode != 0 else "warn"))
        if case_status == "ok":
            n_ok += 1
        elif case_status == "warn":
            n_warn += 1
        else:
            n_failed += 1

        sha = sha256_file(out_path) if out_path.exists() else "0" * 64
        parity_status = str((obj.get("parity", {}) or {}).get("status", "skipped"))

        # Extract median wall time
        timing = obj.get("timing", {}) or {}
        ns_wall = (timing.get("wall_time_s", {}) or {}).get("nextstat", {})
        median = _maybe_float(ns_wall.get("median") if isinstance(ns_wall, dict) else None)
        median = float(median) if median is not None else 0.0
        if median >= worst_wall:
            worst_wall = median
            worst_case = case_id

        # Extract speedup if present
        speedup = _maybe_float(timing.get("speedup_vs_baseline"))

        entry: dict[str, Any] = {
            "case": case_id,
            "path": os.path.relpath(out_path, out_dir),
            "sha256": sha,
            "status": case_status,
            "kind": kind,
            "wall_time_median_s": median,
            "parity_status": parity_status,
        }

        # Add kind-specific metadata
        if "n_conc" in c:
            entry["n_conc"] = int(c["n_conc"])
        if "n_times" in c:
            entry["n_times"] = int(c["n_times"])
        if speedup is not None:
            entry["speedup_vs_baseline"] = float(speedup)

        index_cases.append(entry)
        extra = f" speedup={speedup:.1f}x" if speedup is not None else ""
        print(f"  [{case_id}] {case_status.upper()} median={median:.6f}s parity={parity_status}{extra}", flush=True)

    # -- Suite index --------------------------------------------------------
    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "nextstat_version": _NS_VERSION,
    }
    index = {
        "schema_version": "nextstat.pd_benchmark_suite_result.v1",
        "suite": "pd_models",
        "meta": meta,
        "cases": index_cases,
        "summary": {
            "n_cases": len(index_cases),
            "n_ok": n_ok,
            "n_warn": n_warn,
            "n_failed": n_failed,
            "worst_case": worst_case,
        },
    }
    suite_path = out_dir / "pd_models_suite.json"
    suite_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    print(f"\n  {n_ok} ok / {n_warn} warn / {n_failed} failed ({len(index_cases)} total)")
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
