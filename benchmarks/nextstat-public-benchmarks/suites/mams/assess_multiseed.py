#!/usr/bin/env python3
"""Assess MAMS multi-seed repeatability evidence for stable-surface review."""

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


def _metric_extreme(reviewed_rows: list[dict[str, Any]], *, metric_key: str, prefer: str) -> dict[str, Any]:
    best_case: str | None = None
    best_value: float | None = None
    for row in reviewed_rows:
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


def _review_summary(
    reviewed_cases: list[dict[str, Any]],
    reviewed_parity_cases: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    failing_cases = sorted({str(row.get("case")) for row in failures if row.get("case")})
    summary: dict[str, Any] = {
        "n_reviewed_cases": len(reviewed_cases),
        "n_reviewed_parity_cases": len(reviewed_parity_cases),
        "n_failures": len(failures),
        "n_failing_cases": len(failing_cases),
        "failing_cases": failing_cases,
    }
    for metric_key, prefer, summary_key in [
        ("worst_max_r_hat", "max", "worst_max_r_hat"),
        ("worst_min_ess_bulk", "min", "worst_min_ess_bulk"),
        ("worst_min_ess_tail", "min", "worst_min_ess_tail"),
        ("worst_ess_per_sec", "min", "worst_ess_per_sec"),
        ("worst_accept_rate", "min", "worst_accept_rate"),
    ]:
        extreme = _metric_extreme(reviewed_cases, metric_key=metric_key, prefer=prefer)
        if extreme["case"] is not None or extreme["value"] is not None:
            summary[summary_key] = extreme
    parity_extreme = _metric_extreme(reviewed_parity_cases, metric_key="worst_max_z", prefer="max")
    if parity_extreme["case"] is not None or parity_extreme["value"] is not None:
        summary["worst_parity_max_z"] = parity_extreme
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
    ap = argparse.ArgumentParser(description="Assess MAMS multi-seed repeatability evidence.")
    ap.add_argument("summary_dir", help="Path to multiseed results directory (contains mams_multiseed_summary.json)")
    ap.add_argument(
        "--target-backend",
        default="nextstat_mams",
        help="Backend to review for repeatability gating (default: nextstat_mams)",
    )
    ap.add_argument(
        "--max-rhat",
        type=float,
        default=1.01,
        help="Fail repeatability gate if worst reviewed max_r_hat exceeds this threshold (default: 1.01)",
    )
    args = ap.parse_args()

    summary_dir = Path(args.summary_dir).resolve()
    summary_path = summary_dir / "mams_multiseed_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"missing multiseed summary artifact: {summary_path}")

    summary = _load_json(summary_path)
    cases = summary.get("cases") if isinstance(summary.get("cases"), list) else []
    parity = summary.get("parity") if isinstance(summary.get("parity"), dict) else {}
    parity_rows = parity.get("rows") if isinstance(parity.get("rows"), list) else []
    seeds = summary.get("seeds") if isinstance(summary.get("seeds"), list) else []

    target_backend = str(args.target_backend)
    max_rhat = float(args.max_rhat)
    parity_warn_z = _safe_float(parity.get("warn_z"))
    parity_fail_z = _safe_float(parity.get("fail_z"))
    expected_seed_count = len(seeds)

    reviewed_cases: list[dict[str, Any]] = []
    reviewed_parity_cases: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    backend_cases = [
        row for row in cases if isinstance(row, dict) and str(row.get("backend") or "") == target_backend
    ]
    if not backend_cases:
        failures.append(
            {
                "case": None,
                "reason": "no_matching_backend_cases",
                "backend": target_backend,
            }
        )

    parity_rows_by_case = {
        str(row.get("case") or "unknown"): row for row in parity_rows if isinstance(row, dict)
    }

    for row in backend_cases:
        case_id = str(row.get("case") or "unknown")
        statuses = row.get("statuses") if isinstance(row.get("statuses"), list) else []
        case_review = {
            "case": case_id,
            "backend": target_backend,
            "n_observed_seeds": len(statuses),
            "statuses": [str(status) for status in statuses],
            "worst_max_r_hat": max((_safe_float(v) for v in row.get("max_r_hat", [])), default=None),
            "worst_min_ess_bulk": min((_safe_float(v) for v in row.get("min_ess_bulk", [])), default=None),
            "worst_min_ess_tail": min((_safe_float(v) for v in row.get("min_ess_tail", [])), default=None),
            "worst_ess_per_sec": min((_safe_float(v) for v in row.get("ess_per_sec", [])), default=None),
            "worst_accept_rate": min((_safe_float(v) for v in row.get("accept_rate", [])), default=None),
        }
        reviewed_cases.append(case_review)

        if len(statuses) != expected_seed_count:
            failures.append(
                {
                    "case": case_id,
                    "reason": "missing_seed_rows",
                    "observed_count": len(statuses),
                    "expected_count": expected_seed_count,
                }
            )
        if any(str(status) != "ok" for status in statuses):
            failures.append(
                {
                    "case": case_id,
                    "reason": "backend_statuses_not_all_ok",
                    "statuses": [str(status) for status in statuses],
                }
            )
        case_rhat = _safe_float(case_review["worst_max_r_hat"])
        if case_rhat is None:
            failures.append(
                {
                    "case": case_id,
                    "reason": "missing_max_r_hat",
                }
            )
        elif case_rhat > max_rhat:
            failures.append(
                {
                    "case": case_id,
                    "reason": "max_r_hat_exceeds_threshold",
                    "observed": case_rhat,
                    "threshold": max_rhat,
                }
            )

        parity_row = parity_rows_by_case.get(case_id)
        if parity_row is None:
            failures.append(
                {
                    "case": case_id,
                    "reason": "missing_parity_row",
                }
            )
            continue

        parity_statuses = parity_row.get("statuses") if isinstance(parity_row.get("statuses"), list) else []
        parity_review = {
            "case": case_id,
            "n_observed_seeds": len(parity_statuses),
            "statuses": [str(status) for status in parity_statuses],
            "worst_max_z": max((_safe_float(v) for v in parity_row.get("max_z", [])), default=None),
        }
        reviewed_parity_cases.append(parity_review)

        if len(parity_statuses) != expected_seed_count:
            failures.append(
                {
                    "case": case_id,
                    "reason": "missing_parity_seed_rows",
                    "observed_count": len(parity_statuses),
                    "expected_count": expected_seed_count,
                }
            )
        if any(str(status) != "ok" for status in parity_statuses):
            failure: dict[str, Any] = {
                "case": case_id,
                "reason": "parity_statuses_not_all_ok",
                "statuses": [str(status) for status in parity_statuses],
                "observed": _safe_float(parity_review["worst_max_z"]),
            }
            if parity_warn_z is not None:
                failure["warn_threshold"] = parity_warn_z
            if parity_fail_z is not None:
                failure["fail_threshold"] = parity_fail_z
            failures.append(failure)

    passed = not failures
    review_summary = _review_summary(reviewed_cases, reviewed_parity_cases, failures)

    assessment = {
        "schema_version": "nextstat.mams_multiseed_assessment.v1",
        "suite": "mams",
        "source_summary_path": "mams_multiseed_summary.json",
        "source_summary_sha256": _sha256_file(summary_path),
        "repeatability_gate": {
            "passed": passed,
            "status": _status_from_passed(passed),
            "target_backend": target_backend,
            "policy": {
                "max_r_hat": max_rhat,
                "parity_warn_z": parity_warn_z,
                "parity_fail_z": parity_fail_z,
                "require_complete_seed_coverage": True,
                "require_all_backend_statuses_ok": True,
                "require_all_parity_statuses_ok": True,
            },
            "reviewed_cases": reviewed_cases,
            "reviewed_parity_cases": reviewed_parity_cases,
            "failures": failures,
            "review_summary": review_summary,
        },
    }

    out_json = summary_dir / "mams_multiseed_assessment.json"
    out_json.write_text(json.dumps(assessment, indent=2, sort_keys=True) + "\n")

    lines: list[str] = []
    lines.append("# MAMS Multi-Seed Assessment")
    lines.append("")
    lines.append(f"Source summary: `{assessment['source_summary_path']}`")
    lines.append("")
    lines.append("## Repeatability gate")
    lines.append("")
    lines.append(f"- status: `{assessment['repeatability_gate']['status']}`")
    lines.append(f"- backend: `{target_backend}`")
    lines.append(f"- max_r_hat threshold: `{_fmt(max_rhat, digits=4)}`")
    lines.append(f"- reviewed cases: `{review_summary['n_reviewed_cases']}`")
    lines.append(f"- reviewed parity cases: `{review_summary['n_reviewed_parity_cases']}`")
    lines.append(
        f"- failing cases: `{', '.join(review_summary['failing_cases']) if review_summary['failing_cases'] else '—'}`"
    )
    lines.append("")
    lines.append("| Metric | Worst case | Observed | Policy |")
    lines.append("|---|---|---:|---:|")
    for label, summary_key, policy_value in [
        ("max_r_hat", "worst_max_r_hat", max_rhat),
        ("min_ess_bulk", "worst_min_ess_bulk", None),
        ("min_ess_tail", "worst_min_ess_tail", None),
        ("ess_per_sec", "worst_ess_per_sec", None),
        ("accept_rate", "worst_accept_rate", None),
        ("parity_max_z", "worst_parity_max_z", parity_warn_z),
    ]:
        metric_summary = review_summary.get(summary_key) if isinstance(review_summary.get(summary_key), dict) else {}
        lines.append(
            f"| {label} | {metric_summary.get('case', '—')} | {_fmt(metric_summary.get('value'), digits=4)} | {_fmt(policy_value, digits=4)} |"
        )
    lines.append("")
    lines.append("| Case | Statuses | Worst max R-hat | Worst min ESS_bulk | Worst min ESS_tail | Worst ESS/s | Worst accept rate |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in reviewed_cases:
        lines.append(
            f"| {row['case']} | `{','.join(row['statuses']) or '—'}` | {_fmt(row['worst_max_r_hat'], digits=4)} | {_fmt(row['worst_min_ess_bulk'])} | {_fmt(row['worst_min_ess_tail'])} | {_fmt(row['worst_ess_per_sec'])} | {_fmt(row['worst_accept_rate'], digits=4)} |"
        )
    if reviewed_parity_cases:
        lines.append("")
        lines.append("| Case | Parity statuses | Worst max z |")
        lines.append("|---|---|---:|")
        for row in reviewed_parity_cases:
            lines.append(
                f"| {row['case']} | `{','.join(row['statuses']) or '—'}` | {_fmt(row['worst_max_z'], digits=4)} |"
            )
    if failures:
        lines.append("")
        lines.append("### Repeatability failures")
        lines.append("")
        for row in failures:
            case_label = row.get("case") or "global"
            if "observed" in row and "threshold" in row:
                lines.append(
                    f"- `{case_label}`: `{row['reason']}` (observed `{_fmt(row['observed'], digits=4)}`, threshold `{_fmt(row['threshold'], digits=4)}`)"
                )
            elif "observed_count" in row and "expected_count" in row:
                lines.append(
                    f"- `{case_label}`: `{row['reason']}` (observed `{row['observed_count']}`, expected `{row['expected_count']}`)"
                )
            elif "warn_threshold" in row or "fail_threshold" in row:
                lines.append(
                    f"- `{case_label}`: `{row['reason']}` (worst max z `{_fmt(row.get('observed'), digits=4)}`, warn `{_fmt(row.get('warn_threshold'), digits=4)}`, fail `{_fmt(row.get('fail_threshold'), digits=4)}`)"
                )
            else:
                lines.append(f"- `{case_label}`: `{row['reason']}`")

    out_md = summary_dir / "mams_multiseed_assessment.md"
    out_md.write_text("\n".join(lines) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
