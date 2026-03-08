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
import math
import platform
import sys
import time
from pathlib import Path


def _ensure_repo_root_on_syspath() -> None:
    """Make direct script execution behave like repo-root package execution.

    The validation-pack workflow runs this file as:
      python tests/pharma_validation/runner.py

    In that mode Python puts ``tests/pharma_validation`` on ``sys.path[0]``,
    which is enough for ``pharma_validation.*`` imports but not for
    ``tests._tool_contract_helpers`` used by PQ-REF-011.  Release validation
    must not depend on an external PYTHONPATH tweak, so normalize the import
    root here.
    """

    tests_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[2]
    tests_root_str = str(tests_root)
    repo_root_str = str(repo_root)
    if tests_root_str not in sys.path:
        sys.path.insert(0, tests_root_str)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_syspath()


def _json_safe(value):  # type: ignore[no-untyped-def]
    """Convert Python objects into strict JSON-safe values.

    Validation-pack artifacts are consumed by Rust/serde_json in release CI.
    Python's default ``json.dump`` allows ``Infinity``/``NaN`` literals, but
    those are invalid JSON and break the release pipeline. Preserve the fact
    that the value was non-finite, but encode it as a string.
    """

    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


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

    # Determinism contract:
    # - This runner is used by the Validation Pack determinism gate (JSON/PDF bit-identical).
    # - Individual test wall times are inherently nondeterministic and would break the gate.
    if args.deterministic:
        def _scrub_timings(x):  # type: ignore[no-untyped-def]
            if isinstance(x, dict):
                for k, v in list(x.items()):
                    # Common timing keys used across IQ/OQ/PQ cases.
                    if k == "wall_s" or k.endswith("_wall_s") or k.endswith("_wall_ms"):
                        x[k] = None
                    else:
                        _scrub_timings(v)
            elif isinstance(x, list):
                for v in x:
                    _scrub_timings(v)

        for c in all_cases:
            if isinstance(c, dict) and "wall_s" in c:
                c["wall_s"] = None
            _scrub_timings(c)

    # Stable ordering (defensive): avoid accidental nondeterminism from list construction.
    all_cases = sorted(
        all_cases,
        key=lambda c: (
            str(c.get("category") or ""),
            str(c.get("test_id") or ""),
            str(c.get("section") or ""),
            str(c.get("title") or ""),
        ),
    )

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
    report = _json_safe(report)

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
        json.dump(report, f, indent=2, sort_keys=False, allow_nan=False)

    print(f"Wrote: {args.out}")
    print(f"  Total: {len(all_cases)} | OK: {n_ok} | FAIL: {n_fail} | SKIP: {n_skip}")
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
