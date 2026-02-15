#!/usr/bin/env python3
"""Monte Carlo safety benchmark suite runner.

Orchestrates multiple fault-tree MC cases across workloads and scenario counts.

Usage:
    python suite.py --out-dir /tmp/mc_safety --deterministic --seed 42
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Fast mode: reduce scenario counts for quick correctness/throughput checks.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    # Device-specific case outputs so CPU and CUDA runs can share the same out-dir
    # without overwriting each other.
    cases_dir = out_dir / f"cases_{args.device}"
    cases_dir.mkdir(parents=True, exist_ok=True)

    run_py = Path(__file__).resolve().parent / "run.py"

    suite_cases = [
        {
            "case_id": "ft_bernoulli_small",
            "workload": "bernoulli_fixed",
            "n_components": 16,
            "n_scenarios": 10_000_000,
        },
        {
            "case_id": "ft_bernoulli_medium",
            "workload": "bernoulli_fixed",
            "n_components": 32,
            "n_scenarios": 10_000_000,
        },
        {
            "case_id": "ft_bernoulli_50m",
            "workload": "bernoulli_fixed",
            "n_components": 16,
            "n_scenarios": 50_000_000,
        },
        {
            "case_id": "ft_uncertain_32",
            "workload": "bernoulli_uncertain",
            "n_components": 32,
            "n_scenarios": 10_000_000,
        },
        {
            "case_id": "ft_weibull_32",
            "workload": "weibull_mission",
            "n_components": 32,
            "n_scenarios": 10_000_000,
        },
    ]
    if args.smoke:
        for c in suite_cases:
            # Keep the relative ordering of workloads, but cap the runtime.
            # This suite is meant to be a throughput microbench, so even 1M scenarios
            # is enough to catch regressions and validate device parity.
            c["n_scenarios"] = min(int(c["n_scenarios"]), 1_000_000)
        # The 50M stress case isn't meaningful in smoke mode; drop it.
        suite_cases = [c for c in suite_cases if c["case_id"] != "ft_bernoulli_50m"]

    index_cases = []
    n_ok = 0
    n_failed = 0

    for c in suite_cases:
        case_id = c["case_id"]
        out_path = cases_dir / f"{case_id}.json"
        cmd = [
            sys.executable,
            str(run_py),
            "--case", case_id,
            "--workload", c["workload"],
            "--n-components", str(c["n_components"]),
            "--n-scenarios", str(c["n_scenarios"]),
            "--seed", str(args.seed),
            "--repeat", str(args.repeat),
            "--device", args.device,
            "--out", str(out_path),
        ]
        if args.deterministic:
            cmd.append("--deterministic")

        p = subprocess.run(cmd)
        if not out_path.exists():
            # Ensure the suite index always references existing case JSONs so recursive
            # validation can never fail with FileNotFoundError.
            stub = {
                "schema_version": "nextstat.montecarlo_safety_benchmark_result.v1",
                "suite": "montecarlo_safety",
                "case": case_id,
                "deterministic": bool(args.deterministic),
                "status": "failed",
                "reason": f"case runner failed (exit={p.returncode})",
                "meta": {
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                    "nextstat_version": nextstat.__version__,
                },
                "dataset": {"id": f"generated:montecarlo_safety:{case_id}", "sha256": "0" * 64, "spec": {}},
                "config": {"device": args.device, **c},
                "results": {},
                "timing": {},
                "baselines": {},
                "reproducibility": {},
            }
            out_path.write_text(json.dumps(stub, indent=2, sort_keys=True) + "\n")

        obj = json.loads(out_path.read_text())

        status = obj.get("status", "failed" if p.returncode != 0 else "warn")
        if status == "ok":
            n_ok += 1
        else:
            n_failed += 1

        sha = sha256_file(out_path) if out_path.exists() else "0" * 64
        timing = obj.get("timing", {})
        throughput = (timing.get("scenarios_per_sec", {}) or {}).get("median", 0)

        index_cases.append({
            "case": case_id,
            "path": os.path.relpath(out_path, out_dir),
            "sha256": sha,
            "status": status,
            "workload": c["workload"],
            "n_components": c["n_components"],
            "n_scenarios": c["n_scenarios"],
            "throughput_median": throughput,
        })

    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "nextstat_version": nextstat.__version__,
    }
    index = {
        "schema_version": "nextstat.montecarlo_safety_benchmark_suite_result.v1",
        "suite": "montecarlo_safety",
        "deterministic": args.deterministic,
        "device": args.device,
        "meta": meta,
        "cases": index_cases,
        "summary": {
            "n_cases": len(index_cases),
            "n_ok": n_ok,
            "n_failed": n_failed,
        },
    }

    # Write suite index — append device suffix for non-CPU so report.py can merge.
    if args.device == "cpu":
        suite_path = out_dir / "montecarlo_safety_suite.json"
    else:
        suite_path = out_dir / f"montecarlo_safety_suite_{args.device}.json"
    suite_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    print(f"\nSuite complete: {n_ok}/{len(index_cases)} OK, {n_failed} failed")
    print(f"Results: {suite_path}")

    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
