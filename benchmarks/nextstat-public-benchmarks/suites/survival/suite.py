#!/usr/bin/env python3
"""Survival analysis suite runner (seed).

Writes:
- per-case JSON files (survival_benchmark_result_v1)
- a suite index JSON (survival_benchmark_suite_result_v1)
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

import nextstat


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _maybe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--smoke", action="store_true", help="Fast mode: skip truth-recovery cases.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    run_py = Path(__file__).resolve().parent / "run.py"

    suite_cases: list[dict] = [
        {
            "case_id": "cox_ph_1k_5p",
            "kind": "cox_ph",
            "args": ["--n", "1000", "--p", "5"],
        },
        {
            "case_id": "cox_ph_10k_10p",
            "kind": "cox_ph",
            "args": ["--n", "10000", "--p", "10"],
        },
        {
            "case_id": "kaplan_meier_1k",
            "kind": "kaplan_meier",
            "args": ["--n", "1000"],
        },
        {
            "case_id": "weibull_aft_1k",
            "kind": "weibull_aft",
            "args": ["--n", "1000"],
        },
    ]
    if not args.smoke:
        suite_cases.extend(
            [
                # Truth-recovery Weibull
                *[
                    {
                        "case_id": f"truth_weibull_n{nv}",
                        "kind": "truth_weibull",
                        "args": ["--n", str(nv), "--n-replicates", "200", "--censoring-rates", "0.1,0.3,0.5,0.7"],
                    }
                    for nv in [100, 500, 2000, 10000]
                ],
                # Truth-recovery Exponential
                *[
                    {
                        "case_id": f"truth_exponential_n{nv}",
                        "kind": "truth_exponential",
                        "args": ["--n", str(nv), "--n-replicates", "200", "--censoring-rates", "0.1,0.3,0.5,0.7"],
                    }
                    for nv in [100, 500, 2000, 10000]
                ],
                # Truth-recovery Cox
                *[
                    {
                        "case_id": f"truth_cox_n{nv}",
                        "kind": "truth_cox",
                        "args": ["--n", str(nv), "--p", "5", "--n-replicates", "200", "--censoring-rates", "0.1,0.3,0.5,0.7"],
                    }
                    for nv in [100, 500, 2000, 10000]
                ],
                # Truth-recovery IC Weibull
                *[
                    {
                        "case_id": f"truth_ic_weibull_n{nv}",
                        "kind": "truth_ic_weibull",
                        "args": ["--n", str(nv), "--n-replicates", "200", "--censoring-rates", "0.1,0.3,0.5,0.7"],
                    }
                    for nv in [100, 500, 2000, 10000]
                ],
                # Truth-recovery IC LogNormal
                *[
                    {
                        "case_id": f"truth_ic_lognormal_n{nv}",
                        "kind": "truth_ic_lognormal",
                        "args": ["--n", str(nv), "--n-replicates", "200", "--censoring-rates", "0.1,0.3,0.5,0.7"],
                    }
                    for nv in [500, 2000]
                ],
                # Truth-recovery IC Exponential
                *[
                    {
                        "case_id": f"truth_ic_exponential_n{nv}",
                        "kind": "truth_ic_exponential",
                        "args": ["--n", str(nv), "--n-replicates", "200", "--censoring-rates", "0.1,0.3,0.5,0.7"],
                    }
                    for nv in [500, 2000]
                ],
            ]
        )

    index_cases = []
    n_ok = 0
    n_warn = 0
    n_failed = 0
    worst_case = "none"
    worst_wall = -1.0

    for c in suite_cases:
        case_id = c["case_id"]
        kind = c["kind"]
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
        n_subjects = int(cfg.get("n", 0)) if isinstance(cfg, dict) else 0
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
                "n_subjects": int(n_subjects),
                "wall_time_median_s": float(median),
                "parity_status": parity_status if parity_status in ("ok", "warn", "skipped") else "warn",
            }
        )

    meta = {"python": sys.version.split()[0], "platform": platform.platform(), "nextstat_version": nextstat.__version__}
    index = {
        "schema_version": "nextstat.survival_benchmark_suite_result.v1",
        "suite": "survival",
        "deterministic": bool(args.deterministic),
        "meta": meta,
        "cases": index_cases,
        "summary": {"n_cases": len(index_cases), "n_ok": n_ok, "n_warn": n_warn, "n_failed": n_failed, "worst_case": worst_case},
    }
    (out_dir / "survival_suite.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
