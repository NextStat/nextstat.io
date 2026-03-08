#!/usr/bin/env python3
"""Assess canonical MAMS suite results for core quality vs promotion readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _status_from_passed(passed: bool) -> str:
    return "passed" if passed else "failed"


def _metric_extreme(reviewed_cases: list[dict[str, Any]], *, metric_key: str, prefer: str) -> dict[str, Any]:
    best_case: str | None = None
    best_value: float | None = None
    for row in reviewed_cases:
        value = _safe_float(row.get(metric_key))
        if value is None:
            continue
        if best_value is None:
            best_case = str(row.get("case") or "unknown")
            best_value = value
            continue
        if prefer == "max" and value > best_value:
            best_case = str(row.get("case") or "unknown")
            best_value = value
        elif prefer == "min" and value < best_value:
            best_case = str(row.get("case") or "unknown")
            best_value = value
    return {"case": best_case, "value": best_value}


def _derive_review_summary(reviewed_cases: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    failing_cases = sorted({str(row.get("case")) for row in failures if row.get("case")})
    summary: dict[str, Any] = {
        "n_reviewed_cases": len(reviewed_cases),
        "n_failures": len(failures),
        "n_failing_cases": len(failing_cases),
        "failing_cases": failing_cases,
    }
    for metric_key, prefer, summary_key in [
        ("max_r_hat", "max", "worst_max_r_hat"),
        ("min_ess_bulk", "min", "worst_min_ess_bulk"),
        ("ess_per_sec", "min", "worst_ess_per_sec"),
    ]:
        extreme = _metric_extreme(reviewed_cases, metric_key=metric_key, prefer=prefer)
        if extreme["case"] is not None or extreme["value"] is not None:
            summary[summary_key] = extreme
    return summary


def _fmt(x: Any, *, digits: int = 3) -> str:
    v = _safe_float(x)
    if v is None:
        return "—"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.{digits}f}".rstrip("0").rstrip(".")


def main() -> int:
    ap = argparse.ArgumentParser(description="Assess MAMS suite core quality and promotion readiness.")
    ap.add_argument("suite_dir", help="Path to suite results directory (contains mams_suite.json)")
    ap.add_argument(
        "--promotion-backend",
        default="nextstat_mams",
        help="Backend to evaluate for promotion readiness (default: nextstat_mams)",
    )
    ap.add_argument(
        "--promotion-max-rhat",
        type=float,
        default=1.01,
        help="Fail promotion readiness if a reviewed case exceeds this R-hat threshold (default: 1.01)",
    )
    ap.add_argument(
        "--promotion-min-ess-bulk",
        type=float,
        default=0.0,
        help="Optional floor for min_ess_bulk; values <= 0 disable the check",
    )
    ap.add_argument(
        "--promotion-min-ess-per-sec",
        type=float,
        default=0.0,
        help="Optional floor for ess_per_sec; values <= 0 disable the check",
    )
    args = ap.parse_args()

    suite_dir = Path(args.suite_dir).resolve()
    suite_path = suite_dir / "mams_suite.json"
    if not suite_path.exists():
        raise SystemExit(f"missing suite artifact: {suite_path}")

    suite = _load_json(suite_path)
    summary = suite.get("summary") if isinstance(suite.get("summary"), dict) else {}
    parity = suite.get("parity") if isinstance(suite.get("parity"), dict) else {}
    cases = suite.get("cases") if isinstance(suite.get("cases"), list) else []

    core_failures: list[dict[str, Any]] = []
    core_warnings: list[dict[str, Any]] = []

    n_failed = int(summary.get("n_failed", 0) or 0)
    n_warn = int(summary.get("n_warn", 0) or 0)
    n_parity_fail = int(summary.get("n_parity_fail", 0) or 0)
    n_parity_warn = int(summary.get("n_parity_warn", 0) or 0)

    if n_failed > 0:
        core_failures.append(
            {
                "kind": "suite_execution",
                "reason": "suite_contains_failed_cases",
                "count": n_failed,
            }
        )
    if n_parity_fail > 0:
        core_failures.append(
            {
                "kind": "parity",
                "reason": "suite_contains_parity_failures",
                "count": n_parity_fail,
            }
        )
    if n_warn > 0:
        core_warnings.append(
            {
                "kind": "suite_execution",
                "reason": "suite_contains_warn_cases",
                "count": n_warn,
            }
        )
    if n_parity_warn > 0:
        core_warnings.append(
            {
                "kind": "parity",
                "reason": "suite_contains_parity_warnings",
                "count": n_parity_warn,
            }
        )

    core_passed = not core_failures

    reviewed_cases: list[dict[str, Any]] = []
    promotion_failures: list[dict[str, Any]] = []
    promotion_backend = str(args.promotion_backend)
    max_rhat = float(args.promotion_max_rhat)
    min_ess_bulk = float(args.promotion_min_ess_bulk)
    min_ess_per_sec = float(args.promotion_min_ess_per_sec)

    backend_cases = [c for c in cases if isinstance(c, dict) and str(c.get("backend")) == promotion_backend]
    if not backend_cases:
        promotion_failures.append(
            {
                "case": None,
                "reason": "no_matching_backend_cases",
                "backend": promotion_backend,
            }
        )

    if not core_passed:
        promotion_failures.append(
            {
                "case": None,
                "reason": "core_quality_not_valid",
            }
        )

    for case in backend_cases:
        case_id = str(case.get("case", "unknown"))
        case_status = str(case.get("status", "unknown"))
        case_rhat = _safe_float(case.get("max_r_hat"))
        case_ess_bulk = _safe_float(case.get("min_ess_bulk"))
        case_ess_per_sec = _safe_float(case.get("ess_per_sec"))

        reviewed_cases.append(
            {
                "case": case_id,
                "status": case_status,
                "max_r_hat": case_rhat,
                "min_ess_bulk": case_ess_bulk,
                "ess_per_sec": case_ess_per_sec,
            }
        )

        if case_status != "ok":
            promotion_failures.append(
                {
                    "case": case_id,
                    "reason": "case_status_not_ok",
                    "status": case_status,
                }
            )
            continue

        if case_rhat is None:
            promotion_failures.append(
                {
                    "case": case_id,
                    "reason": "missing_max_r_hat",
                }
            )
        elif case_rhat > max_rhat:
            promotion_failures.append(
                {
                    "case": case_id,
                    "reason": "max_r_hat_exceeds_threshold",
                    "observed": case_rhat,
                    "threshold": max_rhat,
                }
            )

        if min_ess_bulk > 0.0:
            if case_ess_bulk is None:
                promotion_failures.append(
                    {
                        "case": case_id,
                        "reason": "missing_min_ess_bulk",
                        "threshold": min_ess_bulk,
                    }
                )
            elif case_ess_bulk < min_ess_bulk:
                promotion_failures.append(
                    {
                        "case": case_id,
                        "reason": "min_ess_bulk_below_threshold",
                        "observed": case_ess_bulk,
                        "threshold": min_ess_bulk,
                    }
                )

        if min_ess_per_sec > 0.0:
            if case_ess_per_sec is None:
                promotion_failures.append(
                    {
                        "case": case_id,
                        "reason": "missing_ess_per_sec",
                        "threshold": min_ess_per_sec,
                    }
                )
            elif case_ess_per_sec < min_ess_per_sec:
                promotion_failures.append(
                    {
                        "case": case_id,
                        "reason": "ess_per_sec_below_threshold",
                        "observed": case_ess_per_sec,
                        "threshold": min_ess_per_sec,
                    }
                )

    promotion_passed = not promotion_failures
    review_summary = _derive_review_summary(reviewed_cases, promotion_failures)

    assessment = {
        "schema_version": "nextstat.mams_assessment.v1",
        "suite": "mams",
        "source_suite_path": "mams_suite.json",
        "source_suite_sha256": _sha256_file(suite_path),
        "source_suite_summary": summary,
        "core_quality": {
            "passed": core_passed,
            "status": _status_from_passed(core_passed),
            "failures": core_failures,
            "warnings": core_warnings,
        },
        "promotion_gate": {
            "passed": promotion_passed,
            "status": _status_from_passed(promotion_passed),
            "target_backend": promotion_backend,
            "policy": {
                "max_r_hat": max_rhat,
                "min_ess_bulk": min_ess_bulk if min_ess_bulk > 0.0 else None,
                "min_ess_per_sec": min_ess_per_sec if min_ess_per_sec > 0.0 else None,
            },
            "reviewed_cases": reviewed_cases,
            "failures": promotion_failures,
            "review_summary": review_summary,
        },
    }

    out_json = suite_dir / "mams_assessment.json"
    out_json.write_text(json.dumps(assessment, indent=2, sort_keys=True) + "\n")

    lines: list[str] = []
    lines.append("# MAMS Assessment")
    lines.append("")
    lines.append(f"Source suite: `{assessment['source_suite_path']}`")
    lines.append("")
    lines.append("## Core quality")
    lines.append("")
    lines.append(f"- status: `{assessment['core_quality']['status']}`")
    lines.append(f"- suite failed cases: `{n_failed}`")
    lines.append(f"- suite warn cases: `{n_warn}`")
    lines.append(f"- parity failed rows: `{n_parity_fail}`")
    lines.append(f"- parity warn rows: `{n_parity_warn}`")
    if core_failures:
        lines.append("- failures:")
        for row in core_failures:
            lines.append(f"  - `{row.get('reason')}` ({row.get('count', 'n/a')})")
    if core_warnings:
        lines.append("- warnings:")
        for row in core_warnings:
            lines.append(f"  - `{row.get('reason')}` ({row.get('count', 'n/a')})")
    lines.append("")
    lines.append("## Promotion gate")
    lines.append("")
    lines.append(f"- status: `{assessment['promotion_gate']['status']}`")
    lines.append(f"- backend: `{promotion_backend}`")
    lines.append(f"- max_r_hat threshold: `{_fmt(max_rhat, digits=4)}`")
    if min_ess_bulk > 0.0:
        lines.append(f"- min_ess_bulk threshold: `{_fmt(min_ess_bulk)}`")
    if min_ess_per_sec > 0.0:
        lines.append(f"- min_ess_per_sec threshold: `{_fmt(min_ess_per_sec)}`")
    lines.append(f"- reviewed cases: `{review_summary['n_reviewed_cases']}`")
    lines.append(f"- failing cases: `{', '.join(review_summary['failing_cases']) if review_summary['failing_cases'] else '—'}`")
    lines.append("")
    lines.append("| Metric | Worst case | Observed | Policy |")
    lines.append("|---|---|---:|---:|")
    for label, summary_key, policy_value in [
        ("max_r_hat", "worst_max_r_hat", max_rhat),
        ("min_ess_bulk", "worst_min_ess_bulk", min_ess_bulk if min_ess_bulk > 0.0 else None),
        ("ess_per_sec", "worst_ess_per_sec", min_ess_per_sec if min_ess_per_sec > 0.0 else None),
    ]:
        metric_summary = review_summary.get(summary_key) if isinstance(review_summary.get(summary_key), dict) else {}
        lines.append(
            f"| {label} | {metric_summary.get('case', '—')} | {_fmt(metric_summary.get('value'), digits=4)} | {_fmt(policy_value, digits=4)} |"
        )
    lines.append("")
    lines.append("| Case | Status | max R-hat | min ESS_bulk | ESS/s |")
    lines.append("|---|---|---:|---:|---:|")
    for row in reviewed_cases:
        lines.append(
            f"| {row['case']} | {row['status']} | {_fmt(row['max_r_hat'], digits=4)} | {_fmt(row['min_ess_bulk'])} | {_fmt(row['ess_per_sec'])} |"
        )
    if promotion_failures:
        lines.append("")
        lines.append("### Promotion failures")
        lines.append("")
        for row in promotion_failures:
            case_label = row.get("case") or "global"
            if "observed" in row and "threshold" in row:
                lines.append(
                    f"- `{case_label}`: `{row['reason']}` (observed `{_fmt(row['observed'], digits=4)}`, threshold `{_fmt(row['threshold'], digits=4)}`)"
                )
            else:
                lines.append(f"- `{case_label}`: `{row['reason']}`")

    out_md = suite_dir / "mams_assessment.md"
    out_md.write_text("\n".join(lines) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
