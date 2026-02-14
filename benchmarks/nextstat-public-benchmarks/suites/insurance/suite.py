#!/usr/bin/env python3
"""Insurance suite runner.

Runs all Chain Ladder benchmark cases and writes:
- per-case JSON files (insurance_benchmark_result_v1)
- a suite index JSON (insurance_benchmark_suite_result_v1)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _maybe_import(name: str) -> tuple[bool, Optional[str]]:
    try:
        __import__(name)
        mod = sys.modules.get(name)
        return True, str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return False, None


def _maybe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Insurance Chain Ladder benchmark suite.")
    ap.add_argument("--out-dir", required=True, help="Output directory for results.")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=42, help="Seed for synthetic data. Default 42.")
    ap.add_argument("--repeat", type=int, default=50, help="Timing repeats per case. Default 50.")
    ap.add_argument("--conf-level", type=float, default=0.95,
                    help="Confidence level for Mack method. Default 0.95.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    run_py = Path(__file__).resolve().parent / "run.py"

    suite_cases: list[dict[str, Any]] = [
        {
            "case_id": "chain_ladder_10x10",
            "kind": "chain_ladder",
            "args": ["--n", "10"],
        },
        {
            "case_id": "mack_10x10",
            "kind": "mack",
            "args": ["--n", "10", "--conf-level", str(float(args.conf_level))],
        },
        {
            "case_id": "chain_ladder_20x20",
            "kind": "chain_ladder",
            "args": ["--n", "20"],
        },
    ]

    index_cases: list[dict[str, Any]] = []
    n_ok = 0
    n_warn = 0
    n_failed = 0
    worst_case = "none"
    worst_wall = -1.0

    for c in suite_cases:
        case_id = str(c["case_id"])
        kind = str(c["kind"])
        out_path = cases_dir / f"{case_id}.json"
        cmd = [
            sys.executable,
            str(run_py),
            "--case",
            case_id,
            "--kind",
            kind,
            "--out",
            str(out_path),
            "--seed",
            str(int(args.seed)),
            "--repeat",
            str(int(args.repeat)),
            *c["args"],
        ]
        if args.deterministic:
            cmd.append("--deterministic")

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
        n_tri = int(cfg.get("n", 0)) if isinstance(cfg, dict) else 0
        parity_status = str((obj.get("parity", {}) or {}).get("status", "skipped"))

        # Extract median wall time for nextstat.
        ns_timing = (obj.get("timing", {}) or {}).get("wall_time_s", {}).get("nextstat", {})
        median = _maybe_float(ns_timing.get("median") if isinstance(ns_timing, dict) else None)
        median = float(median) if median is not None else 0.0
        if median >= worst_wall:
            worst_wall = float(median)
            worst_case = case_id

        # Speedup.
        speedup = _maybe_float((obj.get("timing", {}) or {}).get("speedup_vs_chainladder"))

        # Parity metrics summary.
        parity_metrics = (obj.get("parity", {}) or {}).get("metrics", {})
        ult_rel = _maybe_float(parity_metrics.get("ultimates_max_rel_diff"))
        ibnr_rel = _maybe_float(parity_metrics.get("total_ibnr_rel_diff"))

        index_cases.append(
            {
                "case": case_id,
                "path": os.path.relpath(out_path, out_dir),
                "sha256": sha,
                "status": status,
                "kind": str(kind),
                "triangle_size": int(n_tri),
                "wall_time_median_s_nextstat": float(median),
                "speedup_vs_chainladder": float(speedup) if speedup is not None else None,
                "parity_status": parity_status if parity_status in ("ok", "warn", "skipped") else "warn",
                "ultimates_max_rel_diff": float(ult_rel) if ult_rel is not None else None,
                "total_ibnr_rel_diff": float(ibnr_rel) if ibnr_rel is not None else None,
            }
        )

    # -- Meta --
    has_ns, ns_v = _maybe_import("nextstat")
    has_cl, cl_v = _maybe_import("chainladder")
    meta: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "nextstat_version": ns_v,
        "chainladder_version": cl_v,
    }

    index: dict[str, Any] = {
        "schema_version": "nextstat.insurance_benchmark_suite_result.v1",
        "suite": "insurance",
        "deterministic": bool(args.deterministic),
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
    (out_dir / "insurance_suite.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
