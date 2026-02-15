#!/usr/bin/env python3
"""Econometrics suite runner (seed).

Writes:
- per-case JSON files (econometrics_benchmark_result_v1)
- a suite index JSON (econometrics_benchmark_suite_result_v1)
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
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--n-entities", type=int, default=2000)
    ap.add_argument("--n-times", type=int, default=8)
    ap.add_argument("--n-obs", type=int, default=5000)
    ap.add_argument("--n-x", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    run_py = Path(__file__).resolve().parent / "run.py"

    suite_cases: list[dict] = [
        {
            "case_id": "panel_fe_balanced",
            "kind": "panel_fe",
            "args": ["--n-entities", str(int(args.n_entities)), "--n-times", str(int(args.n_times))],
        },
        {
            "case_id": "did_twfe_balanced",
            "kind": "did_twfe",
            "args": ["--n-entities", str(int(args.n_entities)), "--n-times", str(int(args.n_times)), "--treat-frac", "0.5"],
        },
        {
            "case_id": "did_twfe_wild_bootstrap_balanced",
            "kind": "did_twfe_wild_bootstrap",
            "args": [
                "--n-entities",
                str(int(args.n_entities)),
                "--n-times",
                str(int(args.n_times)),
                "--treat-frac",
                "0.5",
                "--n-boot",
                "199",
            ],
        },
        {
            "case_id": "did_staggered_balanced",
            "kind": "did_staggered",
            "args": ["--n-entities", str(int(args.n_entities)), "--n-times", str(int(args.n_times))],
        },
        {
            "case_id": "event_study_twfe_balanced",
            "kind": "event_study_twfe",
            "args": ["--n-entities", str(int(args.n_entities)), "--n-times", str(int(args.n_times))],
        },
        {
            "case_id": "iv_2sls_linear",
            "kind": "iv_2sls",
            "args": ["--n-obs", str(int(args.n_obs))],
        },
        {
            "case_id": "iv_2sls_hac_linear",
            "kind": "iv_2sls_hac",
            "args": ["--n-obs", str(int(args.n_obs))],
        },
        {
            "case_id": "aipw_ate",
            "kind": "aipw",
            "args": ["--n-obs", str(int(args.n_obs))],
        },
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
            "--n-x",
            str(int(args.n_x)),
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
        n_obs = int(cfg.get("n_obs", 0)) if isinstance(cfg, dict) else 0
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
                "n_obs": int(n_obs),
                "wall_time_median_s": float(median),
                "parity_status": parity_status if parity_status in ("ok", "warn", "skipped") else "warn",
            }
        )

    meta = {"python": sys.version.split()[0], "platform": platform.platform(), "nextstat_version": nextstat.__version__}
    index = {
        "schema_version": "nextstat.econometrics_benchmark_suite_result.v1",
        "suite": "econometrics",
        "deterministic": bool(args.deterministic),
        "meta": meta,
        "cases": index_cases,
        "summary": {"n_cases": len(index_cases), "n_ok": n_ok, "n_warn": n_warn, "n_failed": n_failed, "worst_case": worst_case},
    }
    (out_dir / "econometrics_suite.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
