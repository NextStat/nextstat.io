#!/usr/bin/env python3
"""LAPS H100 benchmark suite orchestrator.

Runs the full benchmark matrix: weak scaling, strong scaling, dimension sweep,
fused kernel ablation, and LAPS vs CPU NUTS comparison.

Usage:
    python suite.py --out-dir /data/laps_h100_v1
    python suite.py --out-dir /data/laps_h100_v1 --seeds 42,123,456
    python suite.py --out-dir /data/laps_h100_v1 --groups weak_scaling,fused_ablation
    python suite.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark Matrix
# ──────────────────────────────────────────────────────────────────────────────

MATRIX: dict[str, list[dict[str, Any]]] = {
    "weak_scaling": [
        {"label": "1xH100_16K",  "n_chains": 16384,  "devices": "0",       "dim": 100, "model": "std_normal"},
        {"label": "2xH100_32K",  "n_chains": 32768,  "devices": "0,1",     "dim": 100, "model": "std_normal"},
        {"label": "4xH100_64K",  "n_chains": 65536,  "devices": "0,1,2,3", "dim": 100, "model": "std_normal"},
        {"label": "1xH100_33K",  "n_chains": 33792,  "devices": "0",       "dim": 100, "model": "std_normal"},
        {"label": "4xH100_132K", "n_chains": 135168, "devices": "0,1,2,3", "dim": 100, "model": "std_normal"},
    ],
    "strong_scaling": [
        {"label": "1xH100_132K", "n_chains": 132000, "devices": "0",       "dim": 100, "model": "std_normal"},
        {"label": "2xH100_132K", "n_chains": 132000, "devices": "0,1",     "dim": 100, "model": "std_normal"},
        {"label": "4xH100_132K", "n_chains": 132000, "devices": "0,1,2,3", "dim": 100, "model": "std_normal"},
    ],
    "dim_sweep": [
        {"label": "dim10",  "n_chains": 65536, "devices": "0,1,2,3", "dim": 10,  "model": "std_normal"},
        {"label": "dim50",  "n_chains": 65536, "devices": "0,1,2,3", "dim": 50,  "model": "std_normal"},
        {"label": "dim100", "n_chains": 65536, "devices": "0,1,2,3", "dim": 100, "model": "std_normal"},
        {"label": "dim200", "n_chains": 65536, "devices": "0,1,2,3", "dim": 200, "model": "std_normal"},
    ],
    "fused_ablation": [
        {"label": "batched",    "n_chains": 65536, "devices": "0,1,2,3", "dim": 100, "fused": 0,    "model": "std_normal"},
        {"label": "fused_500",  "n_chains": 65536, "devices": "0,1,2,3", "dim": 100, "fused": 500,  "model": "std_normal"},
        {"label": "fused_1000", "n_chains": 65536, "devices": "0,1,2,3", "dim": 100, "fused": 1000, "model": "std_normal"},
        {"label": "fused_2000", "n_chains": 65536, "devices": "0,1,2,3", "dim": 100, "fused": 2000, "model": "std_normal"},
    ],
    "model_comparison": [
        {"label": "std_normal",   "n_chains": 65536, "devices": "0,1,2,3", "dim": 100, "model": "std_normal"},
        {"label": "eight_schools","n_chains": 65536, "devices": "0,1,2,3", "dim": 10,  "model": "eight_schools"},
        {"label": "neal_funnel",  "n_chains": 65536, "devices": "0,1,2,3", "dim": 10,  "model": "neal_funnel"},
    ],
}


def run_case(
    case: dict[str, Any],
    group: str,
    seed: int,
    out_dir: Path,
    *,
    n_warmup: int,
    n_samples: int,
    sync_interval: int,
    welford_chains: int,
    batch_size: int,
) -> dict[str, Any] | None:
    """Run a single benchmark case via run.py subprocess."""
    label = f"{group}_{case['label']}"
    cmd = [
        sys.executable, str(Path(__file__).parent / "run.py"),
        "--label", label,
        "--model", case.get("model", "std_normal"),
        "--dim", str(case.get("dim", 100)),
        "--n-chains", str(case["n_chains"]),
        "--n-warmup", str(n_warmup),
        "--n-samples", str(n_samples),
        "--seed", str(seed),
        "--target-accept", "0.9",
        "--sync-interval", str(sync_interval),
        "--welford-chains", str(welford_chains),
        "--batch-size", str(batch_size),
        "--fused", str(case.get("fused", 0)),
        "--out", str(out_dir),
    ]
    if "devices" in case:
        cmd.extend(["--devices", case["devices"]])

    print(f"\n{'='*60}")
    print(f"[{group}] {case['label']} (seed={seed})")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.1f}s)")
        print(f"  stdout: {result.stdout[-500:]}")
        print(f"  stderr: {result.stderr[-500:]}")
        return None

    print(result.stdout.rstrip())

    # Load result JSON
    fname = f"{label}_seed{seed}.json"
    json_path = out_dir / fname
    if json_path.exists():
        return json.loads(json_path.read_text())
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="LAPS H100 benchmark suite")
    ap.add_argument("--out-dir", required=True, help="Output directory for results")
    ap.add_argument("--seeds", default="42", help="Comma-separated seeds (default: 42)")
    ap.add_argument("--groups", default=None,
                    help="Comma-separated groups to run (default: all). "
                         f"Available: {','.join(MATRIX.keys())}")
    ap.add_argument("--n-warmup", type=int, default=500)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--sync-interval", type=int, default=100)
    ap.add_argument("--welford-chains", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1000)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    groups = args.groups.split(",") if args.groups else list(MATRIX.keys())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    n_ok = 0
    n_fail = 0

    t_start = time.perf_counter()

    for group in groups:
        if group not in MATRIX:
            print(f"WARNING: unknown group '{group}', skipping")
            continue
        cases = MATRIX[group]
        for case in cases:
            for seed in seeds:
                doc = run_case(
                    case, group, seed, out_dir,
                    n_warmup=args.n_warmup,
                    n_samples=args.n_samples,
                    sync_interval=args.sync_interval,
                    welford_chains=args.welford_chains,
                    batch_size=args.batch_size,
                )
                if doc is not None:
                    results.append(doc)
                    n_ok += 1
                else:
                    n_fail += 1

    total_time = time.perf_counter() - t_start

    # Write suite index
    suite_index = {
        "schema": "nextstat.laps_h100_benchmark_suite.v1",
        "total_runs": n_ok + n_fail,
        "ok": n_ok,
        "failed": n_fail,
        "total_time_s": total_time,
        "seeds": seeds,
        "groups": groups,
    }
    index_path = out_dir / "suite_index.json"
    index_path.write_text(json.dumps(suite_index, indent=2) + "\n")

    print(f"\n{'='*60}")
    print(f"Suite complete: {n_ok} ok, {n_fail} failed, {total_time:.1f}s total")
    print(f"Results: {out_dir}")
    print(f"Index: {index_path}")

    # Generate report
    report_cmd = [sys.executable, str(Path(__file__).parent / "report.py"),
                  "--results-dir", str(out_dir)]
    subprocess.run(report_cmd, check=False)

    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
