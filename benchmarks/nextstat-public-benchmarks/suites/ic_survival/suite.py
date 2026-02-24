#!/usr/bin/env python3
"""Interval-Censored Survival suite runner.

Writes:
- per-case JSON files (nextstat.ic_survival_benchmark_result.v1)
- a suite index JSON (nextstat.ic_survival_benchmark_suite_result.v1)
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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    run_py = Path(__file__).resolve().parent / "run.py"

    if args.smoke:
        suite_cases = [
            {"case_id": "ic_weibull_1k", "kind": "ic_weibull", "n": 1000},
            {"case_id": "ic_exponential_1k", "kind": "ic_exponential", "n": 1000},
        ]
    else:
        suite_cases = [
            {"case_id": "ic_weibull_1k", "kind": "ic_weibull", "n": 1000},
            {"case_id": "ic_weibull_10k", "kind": "ic_weibull", "n": 10000},
            {"case_id": "ic_exponential_1k", "kind": "ic_exponential", "n": 1000},
            {"case_id": "ic_exponential_10k", "kind": "ic_exponential", "n": 10000},
            {"case_id": "ic_lognormal_1k", "kind": "ic_lognormal", "n": 1000},
            {"case_id": "ic_lognormal_10k", "kind": "ic_lognormal", "n": 10000},
        ]

    index_cases = []
    n_ok = n_warn = n_failed = 0
    worst_case = "none"
    worst_wall = -1.0

    for c in suite_cases:
        case_id = c["case_id"]
        out_path = cases_dir / f"{case_id}.json"
        repeat = 5 if args.smoke else int(args.repeat)
        cmd = [
            sys.executable, str(run_py),
            "--case", case_id,
            "--kind", c["kind"],
            "--n", str(c["n"]),
            "--out", str(out_path),
            "--seed", str(args.seed),
            "--repeat", str(repeat),
        ]
        if args.skip_baselines or args.smoke:
            cmd.append("--skip-baselines")

        print(f"  [{case_id}] kind={c['kind']} n={c['n']} ...", flush=True)
        p = subprocess.run(cmd)

        try:
            obj = json.loads(out_path.read_text())
        except Exception:
            obj = {}

        status = str(obj.get("status") or ("failed" if p.returncode != 0 else "warn"))
        if status == "ok":
            n_ok += 1
        elif status == "warn":
            n_warn += 1
        else:
            n_failed += 1

        sha = sha256_file(out_path) if out_path.exists() else "0" * 64
        parity_status = str((obj.get("parity", {}) or {}).get("status", "skipped"))
        median = _maybe_float((obj.get("timing", {}) or {}).get("wall_time_s", {}).get("median"))
        median = float(median) if median is not None else 0.0
        if median >= worst_wall:
            worst_wall = median
            worst_case = case_id

        index_cases.append({
            "case": case_id,
            "path": os.path.relpath(out_path, out_dir),
            "sha256": sha,
            "status": status,
            "kind": c["kind"],
            "n": c["n"],
            "wall_time_median_s": median,
            "parity_status": parity_status,
        })
        print(f"  [{case_id}] {status.upper()} median={median:.6f}s parity={parity_status}", flush=True)

    meta = {"python": sys.version.split()[0], "platform": platform.platform(), "nextstat_version": nextstat.__version__}
    index = {
        "schema_version": "nextstat.ic_survival_benchmark_suite_result.v1",
        "suite": "ic_survival",
        "meta": meta,
        "cases": index_cases,
        "summary": {"n_cases": len(index_cases), "n_ok": n_ok, "n_warn": n_warn, "n_failed": n_failed, "worst_case": worst_case},
    }
    (out_dir / "ic_survival_suite.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"\n  {n_ok} ok / {n_warn} warn / {n_failed} failed ({len(index_cases)} total)")
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
