#!/usr/bin/env python3
"""Time Series suite runner.

Writes:
- per-case JSON files (nextstat.timeseries_benchmark_result.v1)
- a suite index JSON (nextstat.timeseries_benchmark_suite_result.v1)
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

import nextstat


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _maybe_float(x):
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--baseline-repeat", type=int, default=1)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="Fast mode: fewer cases, lower repeat, skip baselines.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    run_py = Path(__file__).resolve().parent / "run.py"

    suite_cases: list[dict]
    if args.smoke:
        suite_cases = [
            {"case_id": "kalman_local_level_500", "kind": "kalman_local_level", "n": 500, "skip_baselines": True},
            {"case_id": "garch11_1000", "kind": "garch11", "n": 1000, "skip_baselines": True},
        ]
    else:
        # Baselines can be much slower than NextStat (e.g. statsmodels Kalman).
        # We keep parity checks on small cases, and use large-N cases for throughput only.
        suite_cases = [
            {"case_id": "kalman_local_level_500", "kind": "kalman_local_level", "n": 500, "skip_baselines": False},
            {"case_id": "kalman_local_level_5000", "kind": "kalman_local_level", "n": 5000, "skip_baselines": True},
            {"case_id": "garch11_1000", "kind": "garch11", "n": 1000, "skip_baselines": False},
            {"case_id": "garch11_5000", "kind": "garch11", "n": 5000, "skip_baselines": True},
        ]

    index_cases = []
    n_ok = 0
    n_warn = 0
    n_failed = 0
    worst_case = "none"
    worst_wall = -1.0

    for c in suite_cases:
        case_id = c["case_id"]
        kind = c["kind"]
        n = c["n"]
        skip_case_baselines = bool(c.get("skip_baselines", False))
        out_path = cases_dir / f"{case_id}.json"
        repeat = 3 if args.smoke else int(args.repeat)
        cmd = [
            sys.executable,
            str(run_py),
            "--case",
            case_id,
            "--kind",
            kind,
            "--n",
            str(int(n)),
            "--out",
            str(out_path),
            "--seed",
            str(int(args.seed)),
            "--repeat",
            str(int(repeat)),
            "--baseline-repeat",
            str(int(args.baseline_repeat)),
        ]
        if args.deterministic:
            cmd.append("--deterministic")
        if args.skip_baselines or args.smoke or skip_case_baselines:
            cmd.append("--skip-baselines")

        p = subprocess.run(cmd)
        try:
            obj = json.loads(out_path.read_text())
        except Exception:
            obj = {}

        status = str(obj.get("status") or ("failed" if int(p.returncode) != 0 else "warn"))
        if status == "ok":
            n_ok += 1
        elif status == "warn":
            n_warn += 1
        else:
            n_failed += 1

        sha = sha256_file(out_path) if out_path.exists() else "0" * 64
        cfg = obj.get("config", {}) if isinstance(obj, dict) else {}
        n_val = int(cfg.get("n", 0)) if isinstance(cfg, dict) else 0
        parity_status = str((obj.get("parity", {}) or {}).get("status", "skipped"))

        median = _maybe_float((obj.get("timing", {}) or {}).get("wall_time_s", {}).get("median"))
        median = float(median) if median is not None else 0.0
        if median >= worst_wall:
            worst_wall = median
            worst_case = case_id

        index_cases.append(
            {
                "case": case_id,
                "path": os.path.relpath(out_path, out_dir),
                "sha256": sha,
                "status": status,
                "kind": str(kind),
                "n": int(n_val),
                "wall_time_median_s": float(median),
                "parity_status": parity_status if parity_status in ("ok", "warn", "skipped") else "warn",
            }
        )

    meta = {"python": sys.version.split()[0], "platform": platform.platform(), "nextstat_version": nextstat.__version__}
    index = {
        "schema_version": "nextstat.timeseries_benchmark_suite_result.v1",
        "suite": "timeseries",
        "deterministic": bool(args.deterministic),
        "meta": meta,
        "cases": index_cases,
        "summary": {"n_cases": len(index_cases), "n_ok": n_ok, "n_warn": n_warn, "n_failed": n_failed, "worst_case": worst_case},
    }
    (out_dir / "timeseries_suite.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
