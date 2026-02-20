#!/usr/bin/env python3
"""NextStat Pharma IQ/OQ/PQ Validation Pack Auto-Runner.

Generates structured JSON results for all IQ/OQ/PQ test cases defined in
docs/validation/iq-oq-pq-protocol.md (NS-VAL-001 v2.0.0).

Usage:
    python tests/pharma_validation/runner.py --out tmp/pharma_validation.json [--deterministic]
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Pharma IQ/OQ/PQ Validation Runner")
    ap.add_argument("--out", type=Path, default=Path("tmp/pharma_validation.json"))
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--iq-only", action="store_true", help="Run only IQ tests")
    ap.add_argument("--oq-only", action="store_true", help="Run only OQ tests")
    ap.add_argument("--pq-only", action="store_true", help="Run only PQ tests")
    args = ap.parse_args()

    t0 = time.time()
    all_cases: list[dict] = []

    # Import test modules
    run_all = not (args.iq_only or args.oq_only or args.pq_only)

    if run_all or args.iq_only:
        from pharma_validation.iq import run_iq_tests

        all_cases.extend(run_iq_tests())

    if run_all or args.oq_only:
        from pharma_validation.oq_analytical import run_oq_analytical_tests
        from pharma_validation.oq_population import run_oq_population_tests

        all_cases.extend(run_oq_analytical_tests())
        all_cases.extend(run_oq_population_tests())

    if run_all or args.pq_only:
        from pharma_validation.pq import run_pq_tests

        all_cases.extend(run_pq_tests())

    n_ok = sum(1 for c in all_cases if c.get("ok") is True)
    n_fail = sum(1 for c in all_cases if c.get("ok") is False)
    n_skip = sum(1 for c in all_cases if c.get("ok") is None)
    status = "ok" if n_fail == 0 else "fail"

    report = {
        "schema_version": "nextstat.pharma_validation.v1",
        "meta": {
            "timestamp": None if args.deterministic else int(t0),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "wall_s": None if args.deterministic else round(time.time() - t0, 3),
        },
        "status": status,
        "summary": {
            "n_total": len(all_cases),
            "n_ok": n_ok,
            "n_fail": n_fail,
            "n_skip": n_skip,
            "categories": {},
        },
        "cases": all_cases,
    }

    # Compute per-category summaries
    for cat in ("IQ", "OQ", "PQ"):
        cat_cases = [c for c in all_cases if c.get("category") == cat]
        if cat_cases:
            report["summary"]["categories"][cat] = {
                "n_total": len(cat_cases),
                "n_ok": sum(1 for c in cat_cases if c.get("ok") is True),
                "n_fail": sum(1 for c in cat_cases if c.get("ok") is False),
                "n_skip": sum(1 for c in cat_cases if c.get("ok") is None),
            }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=False)

    print(f"Wrote: {args.out}")
    print(f"  Total: {len(all_cases)} | OK: {n_ok} | FAIL: {n_fail} | SKIP: {n_skip}")
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
