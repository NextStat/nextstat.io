#!/usr/bin/env python3
"""
Library-level unbinned benchmark — measures pure fit time without CLI overhead.

Uses `nextstat.UnbinnedModel.from_config()` + `nextstat.fit()` directly,
bypassing process startup, JSON parse, and output serialization.

Usage:
    # From repo root, with nextstat wheel installed:
    python benchmarks/unbinned/bench_library.py --cases gauss_exp,cb_exp --n-events 10000 --repeats 5

    # Or via .venv that has the wheel:
    .venv/bin/python benchmarks/unbinned/bench_library.py --n-events 100000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "bindings" / "ns-py" / "python"))

# Reuse data generation from the cross-framework suite.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_suite import (
    CASE_BUILDERS,
    _write_parquet,
)


def _make_inline_spec(built: Dict[str, Any], case_name: str) -> Dict[str, Any]:
    """Convert a case spec to use inline data arrays instead of Parquet files."""
    spec = json.loads(json.dumps(built["ns_spec"]))  # deep copy
    columns = built["columns"]
    ch = spec["channels"][0]
    # Replace file reference with inline arrays
    if case_name == "product2d":
        ch["data"] = {
            "inline": {obs["name"]: columns[obs["name"]].tolist() for obs in ch["observables"]}
        }
    else:
        obs_name = ch["observables"][0]["name"]
        ch["data"] = {"inline": columns[obs_name].tolist()}
    return spec


def _bench_library_fit(spec_path: str, n_warmup: int, n_repeats: int) -> Dict[str, Any]:
    """Benchmark nextstat.fit() via Python bindings (no CLI overhead)."""
    import nextstat  # type: ignore

    model = nextstat.UnbinnedModel.from_config(spec_path)

    # Warmup
    for _ in range(n_warmup):
        nextstat.fit(model)

    times_ms: List[float] = []
    results: List[Any] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = nextstat.fit(model)
        dt = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt)
        results.append(result)

    last = results[-1]
    return {
        "times_ms": times_ms,
        "mean_ms": float(np.mean(times_ms)),
        "median_ms": float(np.median(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "nll": last.nll,
        "converged": last.converged,
        "n_iter": last.n_iter,
        "bestfit": list(last.parameters),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Library-level unbinned benchmark (no CLI overhead)")
    ap.add_argument("--cases", default="gauss_exp,cb_exp", help="Comma-separated case names")
    ap.add_argument("--n-events", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=5, help="Number of timed repeats per case")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup iterations (not timed)")
    ap.add_argument("--out", type=str, default=None, help="Output JSON path")
    args = ap.parse_args()

    keys = [k.strip() for k in args.cases.split(",")]
    rng = np.random.default_rng(args.seed)

    results: Dict[str, Any] = {
        "suite": "benchmarks/unbinned/bench_library.py",
        "mode": "library (no CLI overhead)",
        "seed": args.seed,
        "n_events": args.n_events,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "cases": {},
    }

    tmpdir = Path(tempfile.mkdtemp(prefix="ns_lib_bench_"))
    try:
        for k in keys:
            if k not in CASE_BUILDERS:
                print(f"WARNING: unknown case '{k}', skipping", file=sys.stderr)
                continue

            case_dir = tmpdir / k
            case_dir.mkdir(parents=True, exist_ok=True)

            built = CASE_BUILDERS[k](rng, args.n_events)

            parquet_path = case_dir / "observed.parquet"
            _write_parquet(columns=built["columns"], observables=built["observables"], path=parquet_path)

            spec = built["ns_spec"]
            spec["channels"][0]["data"]["file"] = str(parquet_path)
            spec_path = case_dir / "spec.json"
            spec_path.write_text(json.dumps(spec, indent=2))

            print(f"Benchmarking {k} (N={args.n_events}, {args.repeats} repeats)...", file=sys.stderr)
            bench = _bench_library_fit(str(spec_path), args.warmup, args.repeats)
            results["cases"][k] = bench
            print(
                f"  {k}: mean={bench['mean_ms']:.3f} ms, median={bench['median_ms']:.3f} ms, "
                f"min={bench['min_ms']:.3f} ms, nll={bench['nll']:.6f}",
                file=sys.stderr,
            )
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    out_json = json.dumps(results, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_json + "\n")
        print(f"Wrote {out_path}", file=sys.stderr)
    else:
        print(out_json)


if __name__ == "__main__":
    main()
