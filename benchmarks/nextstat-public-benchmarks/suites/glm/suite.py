#!/usr/bin/env python3
"""GLM suite runner (orchestrator).

Iterates all 12 cases (4 families x 3 sizes) and calls run.py for each.

Writes:
- per-case JSON files (nextstat.glm_benchmark_result.v1)
- a suite index JSON (nextstat.glm_benchmark_suite_result.v1)
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


# ---------------------------------------------------------------------------
# Case definitions: 4 families x 3 sizes = 12 cases
# ---------------------------------------------------------------------------

_FAMILIES = ["linear", "logistic", "poisson", "negbin"]

_SIZES = [
    ("1k", 1_000, 100),    # (suffix, n_obs, timing_repeat)
    ("10k", 10_000, 50),
    ("100k", 100_000, 10),
]


def _build_cases() -> list[dict]:
    cases = []
    for family in _FAMILIES:
        for suffix, n, repeat in _SIZES:
            case_id = f"{family}_{suffix}"
            cases.append({
                "case_id": case_id,
                "family": family,
                "n": n,
                "repeat": repeat,
            })
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description="GLM suite orchestrator")
    ap.add_argument("--out-dir", required=True, help="Output directory for results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p", type=int, default=10, help="Number of covariates")
    ap.add_argument("--repeat-override", type=int, default=0,
                    help="Override per-case repeat count (0 = use defaults)")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    run_py = Path(__file__).resolve().parent / "run.py"
    suite_cases = _build_cases()

    index_cases = []
    n_ok = 0
    n_warn = 0
    n_failed = 0
    worst_case = "none"
    worst_wall = -1.0

    for c in suite_cases:
        case_id = str(c["case_id"])
        family = str(c["family"])
        n = int(c["n"])
        repeat = int(args.repeat_override) if int(args.repeat_override) > 0 else int(c["repeat"])

        out_path = cases_dir / f"{case_id}.json"
        cmd = [
            sys.executable,
            str(run_py),
            "--case", case_id,
            "--family", family,
            "--n", str(n),
            "--p", str(int(args.p)),
            "--out", str(out_path),
            "--seed", str(int(args.seed)),
            "--repeat", str(repeat),
        ]
        if args.deterministic:
            cmd.append("--deterministic")

        print(f"  [{case_id}] family={family} n={n} p={args.p} repeat={repeat} ...", flush=True)
        proc = subprocess.run(cmd)

        try:
            obj = json.loads(out_path.read_text())
        except Exception:
            obj = {}

        status = str(obj.get("status") or ("failed" if int(proc.returncode) != 0 else "warn"))
        case_reason = str(obj.get("reason") or "")

        sha = sha256_file(out_path) if out_path.exists() else "0" * 64
        cfg = obj.get("config", {}) if isinstance(obj, dict) else {}
        n_obs = int(cfg.get("n", 0)) if isinstance(cfg, dict) else 0

        # Collect parity status across all competitors
        parity_obj = obj.get("parity", {}) if isinstance(obj, dict) else {}
        parity_statuses = []
        if isinstance(parity_obj, dict):
            for v in parity_obj.values():
                if isinstance(v, dict):
                    parity_statuses.append(str(v.get("status", "skipped")))
        if "warn" in parity_statuses:
            parity_agg = "warn"
        elif "ok" in parity_statuses:
            parity_agg = "ok"
        else:
            parity_agg = "skipped"

        median = _maybe_float((obj.get("timing", {}) or {}).get("wall_time_s", {}).get("median"))
        median = float(median) if median is not None else 0.0
        if median >= worst_wall:
            worst_wall = median
            worst_case = case_id

        converged = False
        n_evals = None
        results = obj.get("results", {})
        if isinstance(results, dict):
            ns_res = results.get("nextstat", {})
            if isinstance(ns_res, dict):
                converged = bool(ns_res.get("converged", False))
                extra = ns_res.get("extra", {})
                if isinstance(extra, dict):
                    raw_evals = extra.get("n_evaluations")
                    try:
                        n_evals = int(raw_evals) if raw_evals is not None else None
                    except Exception:
                        n_evals = None

        # Regression gate: deterministic NB runs must not hit first-step non-convergence.
        if family == "negbin":
            gate_broken = (not converged) or (n_evals is not None and n_evals <= 1)
            if gate_broken:
                gate_reason = f"gate:negbin_nonconvergence(converged={converged},n_evaluations={n_evals})"
                status = "failed"
                case_reason = f"{case_reason};{gate_reason}" if case_reason else gate_reason

        if status == "ok":
            n_ok += 1
        elif status == "warn":
            n_warn += 1
        else:
            n_failed += 1

        index_cases.append({
            "case": case_id,
            "path": os.path.relpath(out_path, out_dir),
            "sha256": sha,
            "status": status,
            "reason": case_reason or None,
            "family": family,
            "n": n_obs,
            "wall_time_median_s": float(median),
            "parity_status": parity_agg,
            "converged": converged,
            "n_evaluations": n_evals,
        })

        status_icon = "OK" if status == "ok" else ("WARN" if status == "warn" else "FAIL")
        reason_tail = f" reason={case_reason}" if case_reason else ""
        print(f"  [{case_id}] {status_icon} median={median:.6f}s parity={parity_agg}{reason_tail}", flush=True)

    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "nextstat_version": nextstat.__version__,
    }
    index = {
        "schema_version": "nextstat.glm_benchmark_suite_result.v1",
        "suite": "glm",
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
    suite_path = out_dir / "glm_suite.json"
    suite_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"\nSuite index: {suite_path}")
    print(f"  {n_ok} ok / {n_warn} warn / {n_failed} failed ({len(index_cases)} total)")
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
