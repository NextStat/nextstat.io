#!/usr/bin/env python3
"""Compare a ROOT suite report against the latest recorded ROOT baseline manifest.

This is intended for cluster workflows where ROOT is not available on the submit host
at compare time. It compares already-produced JSON artifacts only.

Inputs:
- A baseline manifest JSON from `tests/record_baseline.py` (default: tmp/baselines/latest_root_manifest.json)
- A current aggregated suite JSON (e.g. output of `tests/aggregate_apex2_root_suite_reports.py`)

Exit codes:
- 0: OK (within slowdown thresholds)
- 2: FAIL (slowdown threshold exceeded / invalid case statuses)
- 3: baseline manifest missing/invalid
- 4: current report missing/invalid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_baseline_suite_path(manifest_path: Path) -> Path:
    if not manifest_path.exists():
        raise FileNotFoundError(str(manifest_path))
    d = _read_json(manifest_path)
    if not isinstance(d, dict):
        raise ValueError("manifest_not_object")
    baselines = d.get("baselines")
    if not isinstance(baselines, dict):
        raise ValueError("manifest_missing_baselines")
    root_suite = baselines.get("root_suite")
    if not (isinstance(root_suite, dict) and isinstance(root_suite.get("path"), str)):
        raise ValueError("manifest_missing_root_suite_path")
    p = Path(root_suite["path"])
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("tmp/baselines/latest_root_manifest.json"),
        help="Baseline ROOT manifest JSON (default: tmp/baselines/latest_root_manifest.json).",
    )
    ap.add_argument("--current", type=Path, required=True, help="Current ROOT suite report JSON.")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("tmp/root_suite_perf_compare.json"),
        help="Output JSON compare report path.",
    )
    ap.add_argument("--max-slowdown", type=float, default=1.30)
    ap.add_argument(
        "--min-baseline-s",
        type=float,
        default=0.0,
        help="Skip perf checks for cases where baseline scan time is below this threshold.",
    )
    ap.add_argument(
        "--allow-skipped",
        action="store_true",
        help="Do not treat skipped cases as failures.",
    )
    args = ap.parse_args()

    try:
        baseline_suite = _load_baseline_suite_path(args.manifest)
    except Exception as e:
        print(f"Failed to load baseline manifest: {e}", file=sys.stderr)
        return 3

    if not args.current.exists():
        print(f"Missing current report: {args.current}", file=sys.stderr)
        return 4

    # Import the baseline comparer from the same directory.
    # Avoid packaging assumptions; this script is intended to be run as a file.
    from compare_apex2_root_suite_to_baseline import main as _compare_main  # type: ignore

    argv = [
        "compare_apex2_root_suite_to_baseline.py",
        "--baseline",
        str(baseline_suite),
        "--current",
        str(args.current),
        "--out",
        str(args.out),
        "--max-slowdown",
        str(float(args.max_slowdown)),
        "--min-baseline-s",
        str(float(args.min_baseline_s)),
    ]
    if args.allow_skipped:
        argv.append("--allow-skipped")

    old_argv = sys.argv
    try:
        sys.argv = argv
        return int(_compare_main())
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
