#!/usr/bin/env python3
"""Aggregate multiple Apex2 ROOT suite JSON reports into one.

Intended for HTCondor job-array workflows where each job runs:
  tests/apex2_root_suite_report.py --case-index <i> --out <json>

This script merges the per-job JSONs into one report with the same top-level schema:
  { "meta": {...}, "cases": [...], "summary": {...} }
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _summarize(cases: List[Dict[str, Any]]) -> Dict[str, int]:
    n_cases = len(cases)
    n_ok = sum(1 for c in cases if c.get("status") == "ok")
    n_fail = sum(1 for c in cases if c.get("status") == "fail")
    n_skip = sum(1 for c in cases if c.get("status") == "skipped")
    n_err = sum(1 for c in cases if c.get("status") == "error")
    return {
        "n_cases": int(n_cases),
        "n_ok": int(n_ok),
        "n_fail": int(n_fail),
        "n_skipped": int(n_skip),
        "n_error": int(n_err),
    }


def _thresholds_key(meta: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    thr = meta.get("thresholds")
    if not isinstance(thr, dict):
        return None
    try:
        dq = float(thr.get("dq_atol"))
        mu = float(thr.get("mu_hat_atol"))
    except Exception:
        return None
    return dq, mu


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-dir",
        type=Path,
        default=None,
        help="Directory to scan for per-case reports (ignored if --reports is provided).",
    )
    ap.add_argument(
        "--glob",
        type=str,
        default="apex2_root_case_*.json",
        help="Glob under --in-dir (default: apex2_root_case_*.json).",
    )
    ap.add_argument(
        "--reports",
        type=Path,
        nargs="*",
        default=None,
        help="Explicit list of report JSON paths to aggregate.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("tmp/apex2_root_suite_aggregate.json"),
        help="Output path for aggregated report.",
    )
    ap.add_argument(
        "--allow-skipped",
        action="store_true",
        help="Do not treat skipped cases as failure for exit code purposes.",
    )
    ap.add_argument(
        "--exit-nonzero-on-fail",
        action="store_true",
        help="Exit 2 if any case is fail/error (and skipped unless --allow-skipped).",
    )
    args = ap.parse_args()

    if args.reports:
        paths = [Path(p) for p in args.reports]
    else:
        if args.in_dir is None:
            raise SystemExit("Provide --in-dir or --reports")
        paths = sorted(Path(args.in_dir).glob(str(args.glob)))

    if not paths:
        raise SystemExit("No reports found")

    cases_out: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    thresholds_seen: Dict[str, int] = {}

    for p in paths:
        try:
            d = _read_json(p)
        except Exception as e:
            sources.append({"path": str(p), "status": "error", "reason": f"read_failed:{e}"})
            continue
        if not isinstance(d, dict):
            sources.append({"path": str(p), "status": "error", "reason": "not_object"})
            continue

        meta = d.get("meta")
        meta_d = meta if isinstance(meta, dict) else {}
        thr_key = _thresholds_key(meta_d)
        if thr_key is not None:
            thresholds_seen[str(thr_key)] = thresholds_seen.get(str(thr_key), 0) + 1

        cs = d.get("cases")
        if not isinstance(cs, list):
            sources.append({"path": str(p), "status": "error", "reason": "missing_cases_list"})
            continue
        n_added = 0
        for c in cs:
            if not isinstance(c, dict):
                continue
            c2 = dict(c)
            c2.setdefault("source_report", str(p))
            cases_out.append(c2)
            n_added += 1
        sources.append(
            {
                "path": str(p),
                "status": "ok",
                "n_cases": int(n_added),
                "case_index": meta_d.get("case_index"),
                "case_name": meta_d.get("case_name"),
            }
        )

    cases_out.sort(key=lambda c: str(c.get("name") or ""))
    summary = _summarize(cases_out)

    # Pick the most common thresholds key (best-effort, just for convenience).
    thresholds_out: Optional[Dict[str, float]] = None
    if thresholds_seen:
        best = max(thresholds_seen.items(), key=lambda kv: kv[1])[0]
        try:
            dq_s, mu_s = best.strip("()").split(",")
            thresholds_out = {"dq_atol": float(dq_s), "mu_hat_atol": float(mu_s)}
        except Exception:
            thresholds_out = None

    report: Dict[str, Any] = {
        "meta": {
            "timestamp": int(time.time()),
            "inputs": {"in_dir": str(args.in_dir) if args.in_dir else None, "glob": args.glob, "reports": [str(p) for p in paths]},
            "thresholds": thresholds_out,
            "sources": sources,
            "notes": {
                "thresholds_seen": thresholds_seen,
            },
        },
        "cases": cases_out,
        "summary": summary,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"Wrote: {args.out}")

    if args.exit_nonzero_on_fail:
        any_bad = False
        for c in cases_out:
            st = c.get("status")
            if st in ("fail", "error"):
                any_bad = True
                break
            if (st == "skipped") and (not args.allow_skipped):
                any_bad = True
                break
        return 2 if any_bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

