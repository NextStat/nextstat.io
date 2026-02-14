#!/usr/bin/env python3
"""EVT suite runner (seed).

Runs GEV and GPD benchmark cases across sample sizes and writes:
- per-case JSON files (evt_benchmark_result_v1)
- a suite index JSON (evt_benchmark_suite_result_v1)
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
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--mu", type=float, default=100.0, help="GEV location parameter.")
    ap.add_argument("--sigma-gev", type=float, default=10.0, help="GEV scale parameter.")
    ap.add_argument("--xi-gev", type=float, default=0.1, help="GEV shape parameter.")
    ap.add_argument("--sigma-gpd", type=float, default=2.0, help="GPD scale parameter.")
    ap.add_argument("--xi-gpd", type=float, default=0.2, help="GPD shape parameter.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    run_py = Path(__file__).resolve().parent / "run.py"

    suite_cases: list[dict] = [
        {
            "case_id": "gev_block_maxima_500",
            "args": [
                "--model", "gev", "--n", "500",
                "--mu", str(float(args.mu)),
                "--sigma", str(float(args.sigma_gev)),
                "--xi", str(float(args.xi_gev)),
            ],
        },
        {
            "case_id": "gev_block_maxima_5000",
            "args": [
                "--model", "gev", "--n", "5000",
                "--mu", str(float(args.mu)),
                "--sigma", str(float(args.sigma_gev)),
                "--xi", str(float(args.xi_gev)),
            ],
        },
        {
            "case_id": "gpd_threshold_500",
            "args": [
                "--model", "gpd", "--n", "500",
                "--sigma", str(float(args.sigma_gpd)),
                "--xi", str(float(args.xi_gpd)),
            ],
        },
        {
            "case_id": "gpd_threshold_5000",
            "args": [
                "--model", "gpd", "--n", "5000",
                "--sigma", str(float(args.sigma_gpd)),
                "--xi", str(float(args.xi_gpd)),
            ],
        },
    ]

    index_cases: list[dict] = []
    n_ok = 0
    n_warn = 0
    n_failed = 0
    worst_case = "none"
    worst_wall = -1.0

    for c in suite_cases:
        case_id = c["case_id"]
        out_path = cases_dir / f"{case_id}.json"
        cmd = [
            sys.executable,
            str(run_py),
            "--case", case_id,
            "--out", str(out_path),
            "--seed", str(int(args.seed)),
            "--repeat", str(int(args.repeat)),
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
        model_type = str(cfg.get("model", "")) if isinstance(cfg, dict) else ""
        n_samples = int(cfg.get("n", 0)) if isinstance(cfg, dict) else 0
        parity_status = str((obj.get("parity", {}) or {}).get("status", "skipped"))

        # Extract timing median for nextstat
        timing = obj.get("timing", {}) or {}
        fit_time = timing.get("fit_time_s", {}) or {}
        ns_timing = fit_time.get("nextstat", {}) if isinstance(fit_time, dict) else {}
        median = _maybe_float(ns_timing.get("median") if isinstance(ns_timing, dict) else None)
        median = float(median) if median is not None else 0.0

        speedup = _maybe_float(timing.get("speedup_scipy_over_nextstat"))

        if median >= worst_wall:
            worst_wall = median
            worst_case = case_id

        # Extract parity metrics
        parity_metrics = (obj.get("parity", {}) or {}).get("metrics", {}) or {}
        param_max_abs = _maybe_float(parity_metrics.get("param_max_abs_diff"))
        nll_abs = _maybe_float(parity_metrics.get("nll_abs_diff"))

        index_cases.append(
            {
                "case": case_id,
                "path": os.path.relpath(out_path, out_dir),
                "sha256": sha,
                "status": status,
                "model": model_type,
                "n": int(n_samples),
                "fit_time_median_s_nextstat": float(median),
                "speedup_scipy_over_nextstat": float(speedup) if speedup is not None else None,
                "parity_status": parity_status if parity_status in ("ok", "warn", "skipped") else "warn",
                "param_max_abs_diff": float(param_max_abs) if param_max_abs is not None else None,
                "nll_abs_diff": float(nll_abs) if nll_abs is not None else None,
            }
        )

    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "nextstat_version": nextstat.__version__,
    }
    index = {
        "schema_version": "nextstat.evt_benchmark_suite_result.v1",
        "suite": "evt",
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
    (out_dir / "evt_suite.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
