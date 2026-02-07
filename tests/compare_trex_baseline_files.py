#!/usr/bin/env python3
"""Compare two `trex_baseline_v0` JSON files (numbers-first).

This is intended for:
- local sanity checks when refreshing baselines
- GitHub Actions manual workflow summary reports

Exit codes:
  0: OK
  2: FAIL (numeric diffs)
  3: invalid/missing input
  4: runner error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_import_paths() -> None:
    repo = _repo_root()
    tests_py = repo / "tests" / "python"
    for p in [tests_py]:
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)


_ensure_import_paths()

from _tolerances import (  # type: ignore  # noqa: E402
    EXPECTED_DATA_ATOL,
    PARAM_UNCERTAINTY_ATOL,
    PARAM_VALUE_ATOL,
    TWICE_NLL_ATOL,
    TWICE_NLL_RTOL,
)
from _trex_baseline_compare import Tol, compare_baseline_v0, format_report  # type: ignore  # noqa: E402


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True, help="Reference baseline JSON.")
    ap.add_argument("--candidate", type=Path, required=True, help="Candidate baseline JSON.")
    ap.add_argument("--out", type=Path, default=Path("tmp/trex_baseline_compare_report.json"))
    ap.add_argument("--max-diffs", type=int, default=50)
    args = ap.parse_args()

    if not args.baseline.exists():
        print(f"Missing --baseline: {args.baseline}", file=sys.stderr)
        return 3
    if not args.candidate.exists():
        print(f"Missing --candidate: {args.candidate}", file=sys.stderr)
        return 3

    try:
        ref = _read_json(args.baseline)
        cand = _read_json(args.candidate)
    except Exception as e:
        print(f"Failed to read JSON: {e}", file=sys.stderr)
        return 4

    if not isinstance(ref, Mapping) or not isinstance(cand, Mapping):
        print("Invalid JSON: expected objects at top-level", file=sys.stderr)
        return 4

    res = compare_baseline_v0(
        ref=ref,
        cand=cand,
        tol_twice_nll=Tol(atol=TWICE_NLL_ATOL, rtol=TWICE_NLL_RTOL),
        tol_expected_data=Tol(atol=EXPECTED_DATA_ATOL, rtol=0.0),
        tol_param_value=Tol(atol=PARAM_VALUE_ATOL, rtol=0.0),
        tol_param_unc=Tol(atol=PARAM_UNCERTAINTY_ATOL, rtol=0.0),
    )

    report = {
        "schema_version": "trex_baseline_compare_report_v0",
        "baseline_path": str(args.baseline),
        "candidate_path": str(args.candidate),
        "status": "ok" if res.ok else "fail",
        "n_diffs": int(len(res.diffs)),
        "worst": [
            {"path": d.path, "abs_diff": d.abs_diff, "rel_diff": d.rel_diff, "note": d.note}
            for d in res.worst(int(args.max_diffs))
        ],
        "report_text": format_report(res, max_lines=int(args.max_diffs)),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(report["report_text"])
    print(f"Wrote: {args.out}")
    return 0 if res.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

