#!/usr/bin/env python3
"""
Baseline drift comparator for unbinned benchmark smoke artifacts.

Compares one case/tool between two artifacts produced by:
  benchmarks/unbinned/run_suite.py

Gate semantics:
- if baseline artifact is missing: non-blocking skip (ok=true, skipped=true)
- regression when current tool result is failed/skipped
- regression when wall-time ratio exceeds configured threshold
- optional regression on absolute NLL drift

Stdlib-only by design.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


SUMMARY_SCHEMA_VERSION = "unbinned_benchmark_drift_summary_v1"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"failed to read JSON from {path}: {exc}") from exc


def _parse_meta(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--meta must be KEY=VALUE, got: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"--meta key must be non-empty, got: {item!r}")
        out[key] = value
    return out


def _find_case(payload: dict[str, Any], case_name: str) -> dict[str, Any] | None:
    cases = payload.get("cases")
    if not isinstance(cases, list):
        return None
    for item in cases:
        if not isinstance(item, dict):
            continue
        if item.get("case") == case_name:
            return item
    return None


def _as_finite_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    x = float(value)
    if not math.isfinite(x):
        return None
    return x


def _tool_regressed(baseline_tool: dict[str, Any], current_tool: dict[str, Any]) -> tuple[bool, str | None]:
    b_failed = bool(baseline_tool.get("failed"))
    b_skipped = bool(baseline_tool.get("skipped"))
    c_failed = bool(current_tool.get("failed"))
    c_skipped = bool(current_tool.get("skipped"))

    if c_failed:
        return True, "current_tool_failed"
    if c_skipped:
        return True, "current_tool_skipped"

    # If baseline was skipped/failed and current is healthy, this is improvement, not regression.
    if b_failed or b_skipped:
        return False, None
    return False, None


def _compare(
    *,
    baseline: dict[str, Any],
    current: dict[str, Any],
    case: str,
    tool: str,
    max_wall_regression_ratio: float,
    max_nll_abs_diff: float | None,
) -> dict[str, Any]:
    b_case = _find_case(baseline, case)
    c_case = _find_case(current, case)
    if b_case is None:
        return {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "ok": False,
            "regression": True,
            "reason": f"baseline_missing_case:{case}",
        }
    if c_case is None:
        return {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "ok": False,
            "regression": True,
            "reason": f"current_missing_case:{case}",
        }

    b_tool_raw = b_case.get(tool)
    c_tool_raw = c_case.get(tool)
    if not isinstance(b_tool_raw, dict):
        return {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "ok": False,
            "regression": True,
            "reason": f"baseline_missing_tool:{tool}",
        }
    if not isinstance(c_tool_raw, dict):
        return {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "ok": False,
            "regression": True,
            "reason": f"current_missing_tool:{tool}",
        }

    reasons: list[str] = []
    regressed, reason = _tool_regressed(b_tool_raw, c_tool_raw)
    if regressed and reason is not None:
        reasons.append(reason)

    b_wall_ms = _as_finite_number(b_tool_raw.get("_wall_ms"))
    c_wall_ms = _as_finite_number(c_tool_raw.get("_wall_ms"))
    wall_ratio = None
    wall_regression = False
    if b_wall_ms is not None and c_wall_ms is not None and b_wall_ms > 0.0:
        wall_ratio = c_wall_ms / b_wall_ms
        if wall_ratio > max_wall_regression_ratio:
            wall_regression = True
            reasons.append("wall_time_ratio_exceeded")

    b_nll = _as_finite_number(b_tool_raw.get("nll"))
    c_nll = _as_finite_number(c_tool_raw.get("nll"))
    nll_abs_diff = None
    nll_regression = False
    if b_nll is not None and c_nll is not None:
        nll_abs_diff = abs(c_nll - b_nll)
        if max_nll_abs_diff is not None and nll_abs_diff > max_nll_abs_diff:
            nll_regression = True
            reasons.append("nll_abs_diff_exceeded")

    regression = bool(reasons)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "ok": not regression,
        "regression": regression,
        "reasons": reasons,
        "case": case,
        "tool": tool,
        "thresholds": {
            "max_wall_regression_ratio": max_wall_regression_ratio,
            "max_nll_abs_diff": max_nll_abs_diff,
        },
        "baseline": {
            "schema_version": baseline.get("schema_version"),
            "seed": baseline.get("seed"),
            "n_events": baseline.get("n_events"),
            "_wall_ms": b_wall_ms,
            "nll": b_nll,
            "failed": bool(b_tool_raw.get("failed")),
            "skipped": bool(b_tool_raw.get("skipped")),
        },
        "current": {
            "schema_version": current.get("schema_version"),
            "seed": current.get("seed"),
            "n_events": current.get("n_events"),
            "_wall_ms": c_wall_ms,
            "nll": c_nll,
            "failed": bool(c_tool_raw.get("failed")),
            "skipped": bool(c_tool_raw.get("skipped")),
        },
        "metrics": {
            "wall_ratio": wall_ratio,
            "wall_regression": wall_regression,
            "nll_abs_diff": nll_abs_diff,
            "nll_regression": nll_regression,
        },
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Compare unbinned benchmark smoke artifact against baseline.")
    ap.add_argument("--baseline", type=Path, required=True, help="Baseline JSON artifact path")
    ap.add_argument("--current", type=Path, required=True, help="Current JSON artifact path")
    ap.add_argument("--out", type=Path, required=True, help="Output drift summary JSON path")
    ap.add_argument("--case", type=str, default="gauss_exp", help="Case name to compare (default: gauss_exp)")
    ap.add_argument("--tool", type=str, default="nextstat", help="Tool section to compare (default: nextstat)")
    ap.add_argument(
        "--max-wall-regression-ratio",
        type=float,
        default=2.5,
        help="Fail if current_wall_ms / baseline_wall_ms exceeds this ratio (default: 2.5)",
    )
    ap.add_argument(
        "--max-nll-abs-diff",
        type=float,
        default=None,
        help="Optional absolute NLL drift threshold for fail gate (default: disabled)",
    )
    ap.add_argument(
        "--meta",
        action="append",
        default=[],
        help="Optional metadata entries in KEY=VALUE form (repeatable)",
    )
    ap.add_argument(
        "--fail-on-regression",
        dest="fail_on_regression",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit non-zero when regression is detected (default: true)",
    )
    args = ap.parse_args(argv)

    metadata = _parse_meta(args.meta)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if not args.baseline.exists():
        payload: dict[str, Any] = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "ok": True,
            "skipped": True,
            "reason": f"baseline_missing:{args.baseline}",
        }
        if metadata:
            payload["metadata"] = metadata
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0

    baseline = _read_json(args.baseline)
    current = _read_json(args.current)
    summary = _compare(
        baseline=baseline,
        current=current,
        case=args.case,
        tool=args.tool,
        max_wall_regression_ratio=float(args.max_wall_regression_ratio),
        max_nll_abs_diff=args.max_nll_abs_diff,
    )
    if metadata:
        summary["metadata"] = metadata

    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if summary.get("ok") is True:
        return 0
    if args.fail_on_regression:
        reasons = summary.get("reasons", [])
        print(f"unbinned benchmark regression detected: reasons={reasons}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
