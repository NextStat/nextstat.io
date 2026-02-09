#!/usr/bin/env python3
"""Compare a ROOT suite report against a recorded baseline report.

Intended for cluster workflows where:
- baseline is recorded once (e.g. `root_suite_baseline_*.json` via `tests/record_baseline.py --only root`)
- current suite is produced by HTCondor job arrays + aggregation
  (see `tests/aggregate_apex2_root_suite_reports.py`)

This script does not require ROOT to be installed; it compares already-produced JSON artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _case_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        name = r.get("name")
        if isinstance(name, str):
            out[name] = r
    return out


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not (v >= 0.0) or v != v:  # NaN
        return None
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True, help="Baseline ROOT suite report JSON.")
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

    if not args.baseline.exists():
        print(f"Missing baseline report: {args.baseline}", file=sys.stderr)
        return 3
    if not args.current.exists():
        print(f"Missing current report: {args.current}", file=sys.stderr)
        return 4

    try:
        base = _read_json(args.baseline)
        cur = _read_json(args.current)
    except Exception as e:
        print(f"Failed to read JSON: {e}", file=sys.stderr)
        return 4

    base_cases = base.get("cases") if isinstance(base, dict) else None
    cur_cases = cur.get("cases") if isinstance(cur, dict) else None
    if not isinstance(base_cases, list) or not isinstance(cur_cases, list):
        print("Invalid report schema: missing 'cases' list", file=sys.stderr)
        return 4

    base_idx = _case_index([c for c in base_cases if isinstance(c, dict)])
    cur_idx = _case_index([c for c in cur_cases if isinstance(c, dict)])

    cases_out: List[Dict[str, Any]] = []
    any_failed = False
    max_slow = 0.0

    for name, cur_row in sorted(cur_idx.items(), key=lambda kv: kv[0]):
        base_row = base_idx.get(name)
        row: Dict[str, Any] = {"name": name}
        if base_row is None:
            any_failed = True
            row.update({"ok": False, "reason": "missing_baseline_case"})
            cases_out.append(row)
            continue

        base_status = base_row.get("status")
        cur_status = cur_row.get("status")
        if base_status != "ok":
            # Baseline case was already failing (e.g. ROOT optimizer divergence).
            # Not a regression if current also fails; only flag if current regressed
            # in a *new* way (but that's hard to detect — treat as expected).
            row.update({"ok": True, "reason": f"expected_failure(baseline:{base_status},current:{cur_status})"})
            cases_out.append(row)
            continue

        if cur_status != "ok":
            if (cur_status == "skipped") and args.allow_skipped:
                row.update({"ok": True, "reason": "skipped_allowed"})
                cases_out.append(row)
                continue
            any_failed = True
            row.update({"ok": False, "reason": f"current_case_status:{cur_status}"})
            cases_out.append(row)
            continue

        b = _safe_float((base_row.get("timing_s") or {}).get("nextstat_profile_scan"))
        c = _safe_float((cur_row.get("timing_s") or {}).get("nextstat_profile_scan"))
        if b is None or c is None or not (b > 0.0):
            any_failed = True
            row.update({"ok": False, "reason": "missing_nextstat_profile_scan"})
            cases_out.append(row)
            continue

        slow = c / b
        max_slow = max(max_slow, slow)
        checked = b >= float(args.min_baseline_s)
        ok = (slow <= float(args.max_slowdown)) if checked else True
        if not ok:
            any_failed = True

        row.update(
            {
                "ok": bool(ok),
                "baseline": {"nextstat_profile_scan_s": float(b)},
                "current": {"nextstat_profile_scan_s": float(c)},
                "slowdown": {"nextstat_profile_scan": float(slow)},
                "checks": {"checked": bool(checked)},
                "thresholds": {"max_slowdown": float(args.max_slowdown)},
            }
        )
        cases_out.append(row)

    baseline_only = sorted(set(base_idx.keys()) - set(cur_idx.keys()))
    if baseline_only:
        # Treat as warning only; allows partial reruns.
        pass

    out: Dict[str, Any] = {
        "meta": {
            "timestamp": int(time.time()),
            "baseline_path": str(args.baseline),
            "current_path": str(args.current),
            "thresholds": {
                "max_slowdown": float(args.max_slowdown),
                "min_baseline_s": float(args.min_baseline_s),
                "allow_skipped": bool(args.allow_skipped),
            },
        },
        "status": "ok" if not any_failed else "fail",
        "summary": {
            "n_cases": int(len(cases_out)),
            "n_ok": int(sum(1 for c in cases_out if c.get("ok") is True)),
            "n_fail": int(sum(1 for c in cases_out if c.get("ok") is False)),
            "max_slowdown_nextstat_profile_scan": float(max_slow),
            "baseline_only": baseline_only,
        },
        "cases": cases_out,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote: {args.out}")
    return 0 if out["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())

