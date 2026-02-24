#!/usr/bin/env python3
"""GARCH / Stochastic Volatility suite runner.

Writes:
- per-case JSON files (nextstat.garch_family_benchmark_result.v1)
- a suite index JSON (nextstat.garch_family_benchmark_suite_result.v1)
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


_CASES = [
    {"case_id": "garch11_1000", "kind": "garch11", "n": 1000, "repeat": 50},
    {"case_id": "garch11_5000", "kind": "garch11", "n": 5000, "repeat": 30},
    {"case_id": "garch11_10000", "kind": "garch11", "n": 10000, "repeat": 20},
    {"case_id": "sv_logchi2_1000", "kind": "sv_logchi2", "n": 1000, "repeat": 50},
    {"case_id": "sv_logchi2_5000", "kind": "sv_logchi2", "n": 5000, "repeat": 30},
    {"case_id": "egarch11_1000", "kind": "egarch11", "n": 1000, "repeat": 50},
    {"case_id": "egarch11_5000", "kind": "egarch11", "n": 5000, "repeat": 30},
    {"case_id": "gjr_garch11_1000", "kind": "gjr_garch11", "n": 1000, "repeat": 50},
    {"case_id": "gjr_garch11_5000", "kind": "gjr_garch11", "n": 5000, "repeat": 30},
]

_SMOKE_CASES = [
    {"case_id": "garch11_1000", "kind": "garch11", "n": 1000, "repeat": 10},
    {"case_id": "sv_logchi2_1000", "kind": "sv_logchi2", "n": 1000, "repeat": 10},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline-repeat", type=int, default=1)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="Fast mode: fewer cases, lower repeat, skip baselines.")
    ap.add_argument("--run-all-baselines", action="store_true",
                    help="Run baselines even on large-N cases (for speedup data).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    run_py = Path(__file__).resolve().parent / "run.py"

    if args.smoke:
        suite_cases = _SMOKE_CASES
    else:
        suite_cases = _CASES

    index_cases = []
    n_ok = 0
    n_warn = 0
    n_failed = 0
    n_skipped = 0
    worst_case = "none"
    worst_wall = -1.0

    for c in suite_cases:
        case_id = c["case_id"]
        kind = c["kind"]
        n = c["n"]
        repeat = c["repeat"]
        out_path = cases_dir / f"{case_id}.json"

        if args.smoke:
            repeat = c.get("repeat", 10)

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
        if args.skip_baselines or args.smoke:
            cmd.append("--skip-baselines")

        print(f"  Running: {case_id} (kind={kind}, n={n}, repeat={repeat})")
        p = subprocess.run(cmd)

        try:
            obj = json.loads(out_path.read_text())
        except Exception:
            obj = {}

        raw_status = str(obj.get("status") or ("failed" if int(p.returncode) != 0 else "warn"))
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

        median = _maybe_float((obj.get("timing", {}) or {}).get("wall_time_s", {}).get("median"))
        median = float(median) if median is not None else 0.0
        if median >= worst_wall:
            worst_wall = median
            worst_case = case_id

        # --- per-case summary line ---
        speedup_str = ""
        spd = _maybe_float((obj.get("timing", {}) or {}).get("speedup"))
        if spd is not None:
            speedup_str = f" speedup={float(spd):.1f}x"
        tag = raw_status.upper()
        print(f"  [{case_id}] {tag} median={median:.6f}s parity={parity_status}{speedup_str}")

        index_cases.append(
            {
                "case": case_id,
                "path": os.path.relpath(out_path, out_dir),
                "sha256": sha,
                "status": raw_status,
                "kind": str(kind),
                "n": int(n_val),
                "wall_time_median_s": float(median),
                "parity_status": parity_status if parity_status in ("ok", "warn", "skipped") else "warn",
            }
        )

    meta = {"python": sys.version.split()[0], "platform": platform.platform(), "nextstat_version": nextstat.__version__}
    index = {
        "schema_version": "nextstat.garch_family_benchmark_suite_result.v1",
        "suite": "garch_family",
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
    (out_dir / "garch_family_suite.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
