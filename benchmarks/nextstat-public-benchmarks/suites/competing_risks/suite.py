#!/usr/bin/env python3
"""Competing Risks (survival analysis) suite runner.

Writes:
- per-case JSON files (nextstat.competing_risks_benchmark_result.v1)
- a suite index JSON (nextstat.competing_risks_benchmark_suite_result.v1)

Cases:
- cif_1k / cif_10k:               Cumulative Incidence Function estimation
- gray_1k / gray_10k:             Gray's test for group comparison
- fine_gray_1k_p5 / fine_gray_10k_p10: Fine-Gray subdistribution hazard regression

Smoke (2 cases): cif_1k, fine_gray_1k_p5
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
    {"case_id": "cif_1k",             "kind": "cif",       "n": 1000,  "p": 0,  "repeat": 50},
    {"case_id": "cif_10k",            "kind": "cif",       "n": 10000, "p": 0,  "repeat": 30},
    {"case_id": "gray_1k",            "kind": "gray",      "n": 1000,  "p": 0,  "repeat": 50},
    {"case_id": "gray_10k",           "kind": "gray",      "n": 10000, "p": 0,  "repeat": 30},
    {"case_id": "fine_gray_1k_p5",    "kind": "fine_gray", "n": 1000,  "p": 5,  "repeat": 30},
    # Fine-Gray is substantially heavier than CIF/Gray; keep repeats low to make
    # multi-seed runs practical on EPYC while still measuring stable medians.
    {"case_id": "fine_gray_10k_p10",  "kind": "fine_gray", "n": 10000, "p": 10, "repeat": 5},
]

_SMOKE_CASES = [
    {"case_id": "cif_1k",             "kind": "cif",       "n": 1000,  "p": 0,  "repeat": 10},
    {"case_id": "fine_gray_1k_p5",    "kind": "fine_gray", "n": 1000,  "p": 5,  "repeat": 10},
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Competing Risks benchmark suite")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--baseline-repeat", type=int, default=1)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="Fast mode: fewer cases, lower repeat, skip baselines.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    run_py = Path(__file__).resolve().parent / "run.py"

    suite_cases = _SMOKE_CASES if args.smoke else _CASES

    index_cases: list[dict[str, Any]] = []
    n_ok = 0
    n_warn = 0
    n_failed = 0
    n_skipped = 0
    worst_case = "none"
    worst_wall = -1.0

    for c in suite_cases:
        case_id = str(c["case_id"])
        kind = str(c["kind"])
        n = int(c["n"])
        p = int(c["p"])
        repeat = int(c["repeat"])
        out_path = cases_dir / f"{case_id}.json"

        if args.smoke:
            repeat = c.get("repeat", 10)

        cmd = [
            sys.executable,
            str(run_py),
            "--case", case_id,
            "--kind", kind,
            "--n", str(n),
            "--out", str(out_path),
            "--seed", str(int(args.seed)),
            "--repeat", str(int(repeat)),
            "--baseline-repeat", str(int(args.baseline_repeat)),
        ]

        # Fine-Gray covariates
        if p > 0:
            cmd.extend(["--p", str(p)])

        if args.deterministic:
            cmd.append("--deterministic")
        if args.skip_baselines or args.smoke:
            cmd.append("--skip-baselines")

        print(f"  Running: {case_id} (kind={kind}, n={n}{f', p={p}' if p > 0 else ''}, repeat={repeat})")
        proc = subprocess.run(cmd)

        try:
            obj = json.loads(out_path.read_text())
        except Exception:
            obj = {}

        raw_status = str(obj.get("status") or ("failed" if int(proc.returncode) != 0 else "warn"))
        if raw_status == "ok":
            n_ok += 1
        elif raw_status == "skipped":
            n_skipped += 1
        elif raw_status == "warn":
            n_warn += 1
        else:
            n_failed += 1

        sha = sha256_file(out_path) if out_path.exists() else "0" * 64
        cfg = obj.get("config", {}) if isinstance(obj, dict) else {}
        n_val = int(cfg.get("n", 0)) if isinstance(cfg, dict) else 0
        parity_status = str((obj.get("parity", {}) or {}).get("status", "skipped"))

        # Extract median wall time
        timing = obj.get("timing", {}) or {}
        wall_time = timing.get("wall_time_s", {})
        median = _maybe_float(wall_time.get("median") if isinstance(wall_time, dict) else None)
        median = float(median) if median is not None else 0.0
        if median >= worst_wall:
            worst_wall = median
            worst_case = case_id

        # Extract baseline speedup if present
        bl_timing = obj.get("timing_baseline", {}) or {}
        bl_median = _maybe_float(
            (bl_timing.get("wall_time_s", {}) or {}).get("median")
            if isinstance(bl_timing, dict) else None
        )
        speedup: float | None = None
        if bl_median is not None and bl_median > 0 and median > 0:
            speedup = float(bl_median) / float(median)

        entry: dict[str, Any] = {
            "case": case_id,
            "path": os.path.relpath(out_path, out_dir),
            "sha256": sha,
            "status": raw_status,
            "kind": kind,
            "n": n_val,
            "wall_time_median_s": float(median),
            "parity_status": parity_status if parity_status in ("ok", "warn", "skipped") else "warn",
        }
        if p > 0:
            entry["p"] = p
        if speedup is not None:
            entry["speedup_vs_baseline"] = float(speedup)

        index_cases.append(entry)
        extra = f" speedup={speedup:.1f}x" if speedup is not None else ""
        print(f"  [{case_id}] {raw_status.upper()} median={median:.6f}s parity={parity_status}{extra}", flush=True)

    # -- Suite index -----------------------------------------------------------
    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "nextstat_version": _NS_VERSION,
    }
    index = {
        "schema_version": "nextstat.competing_risks_benchmark_suite_result.v1",
        "suite": "competing_risks",
        "deterministic": bool(args.deterministic),
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
    suite_path = out_dir / "competing_risks_suite.json"
    suite_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    print(f"\n  {n_ok} ok / {n_warn} warn / {n_failed} failed / {n_skipped} skipped ({len(index_cases)} total)")
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
