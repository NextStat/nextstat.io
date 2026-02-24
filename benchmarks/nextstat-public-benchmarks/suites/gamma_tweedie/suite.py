#!/usr/bin/env python3
"""Gamma/Tweedie GLM suite runner."""

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


_CASES = [
    {"case_id": "gamma_10k", "kind": "gamma", "n": 10000, "repeat": 20},
    {"case_id": "gamma_100k", "kind": "gamma", "n": 100000, "repeat": 5},
    {"case_id": "tweedie_10k", "kind": "tweedie", "n": 10000, "repeat": 20},
    {"case_id": "tweedie_100k", "kind": "tweedie", "n": 100000, "repeat": 5},
]

_SMOKE_CASES = [
    {"case_id": "gamma_10k", "kind": "gamma", "n": 10000, "repeat": 5},
    {"case_id": "tweedie_10k", "kind": "tweedie", "n": 10000, "repeat": 5},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p", type=int, default=20)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    run_py = Path(__file__).resolve().parent / "run.py"
    suite_cases = _SMOKE_CASES if args.smoke else _CASES

    index_cases = []
    n_ok = n_warn = n_failed = 0
    worst_case = "none"
    worst_wall = -1.0

    for c in suite_cases:
        case_id = c["case_id"]
        out_path = cases_dir / f"{case_id}.json"
        cmd = [
            sys.executable, str(run_py),
            "--case", case_id,
            "--kind", c["kind"],
            "--n", str(c["n"]),
            "--p", str(args.p),
            "--out", str(out_path),
            "--seed", str(args.seed),
            "--repeat", str(c["repeat"]),
        ]
        if args.skip_baselines or args.smoke:
            cmd.append("--skip-baselines")

        print(f"  [{case_id}] kind={c['kind']} n={c['n']} p={args.p} ...", flush=True)
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
        "schema_version": "nextstat.gamma_tweedie_benchmark_suite_result.v1",
        "suite": "gamma_tweedie",
        "meta": meta,
        "cases": index_cases,
        "summary": {"n_cases": len(index_cases), "n_ok": n_ok, "n_warn": n_warn, "n_failed": n_failed, "worst_case": worst_case},
    }
    (out_dir / "gamma_tweedie_suite.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"\n  {n_ok} ok / {n_warn} warn / {n_failed} failed ({len(index_cases)} total)")
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
