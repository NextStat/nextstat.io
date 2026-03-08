#!/usr/bin/env python3
"""Assess MAMS stress multi-seed evidence for stable-surface review."""

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


def _review_summary(reviewed_cases: list[dict[str, Any]], failures: list[dict[str, Any]], *, reviewed_parity_cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    failing_cases = sorted({str(row.get("case")) for row in failures if row.get("case")})
    summary: dict[str, Any] = {
        "n_reviewed_cases": len(reviewed_cases),
        "n_failures": len(failures),
        "n_failing_cases": len(failing_cases),
        "failing_cases": failing_cases,
    }
    if reviewed_parity_cases is not None:
        summary["n_reviewed_parity_cases"] = len(reviewed_parity_cases)
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
    if reviewed_parity_cases is not None:
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


def _case_catalog(summary: dict[str, Any]) -> dict[str, dict[str, str]]:
    catalog: dict[str, dict[str, str]] = {}
    for row in summary.get("case_catalog") if isinstance(summary.get("case_catalog"), list) else []:
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("case") or "")
        if case_id:
            catalog[case_id] = {
                "case_tier": str(row.get("case_tier") or "unknown"),
                "parity_scope": str(row.get("parity_scope") or "informational"),
            }
    return catalog


def main() -> int:
    ap = argparse.ArgumentParser(description="Assess MAMS stress multi-seed evidence.")
    ap.add_argument("summary_dir", help="Path to stress multiseed results directory")
    ap.add_argument("--target-backend", default="nextstat_mams")
    ap.add_argument("--max-rhat", type=float, default=1.01)
    args = ap.parse_args()

    summary_dir = Path(args.summary_dir).resolve()
    summary_path = summary_dir / "mams_stress_multiseed_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"missing stress multiseed summary artifact: {summary_path}")

    summary = _load_json(summary_path)
    cases = summary.get("cases") if isinstance(summary.get("cases"), list) else []
    parity = summary.get("parity") if isinstance(summary.get("parity"), dict) else {}
    parity_rows = parity.get("rows") if isinstance(parity.get("rows"), list) else []
    seeds = summary.get("seeds") if isinstance(summary.get("seeds"), list) else []
    catalog = _case_catalog(summary)

    expected_seed_count = len(seeds)
    target_backend = str(args.target_backend)
    max_rhat = float(args.max_rhat)
    parity_warn_z = _safe_float(parity.get("warn_z"))
    parity_fail_z = _safe_float(parity.get("fail_z"))

    supported_cases = {
        case_id for case_id, meta in catalog.items()
        if meta.get("case_tier") == "supported"
    }
    control_cases = {
        case_id for case_id, meta in catalog.items()
        if meta.get("case_tier") == "pathological_control"
    }

    parity_rows_by_case = {
        str(row.get("case") or "unknown"): row for row in parity_rows if isinstance(row, dict)
    }
    backend_cases = [
        row for row in cases if isinstance(row, dict) and str(row.get("backend") or "") == target_backend
    ]

    supported_reviewed_cases: list[dict[str, Any]] = []
    supported_reviewed_parity: list[dict[str, Any]] = []
    supported_failures: list[dict[str, Any]] = []

    control_reviewed_cases: list[dict[str, Any]] = []
    control_failures: list[dict[str, Any]] = []

    for row in backend_cases:
        case_id = str(row.get("case") or "unknown")
        statuses = row.get("statuses") if isinstance(row.get("statuses"), list) else []
        review = {
            "case": case_id,
            "case_tier": catalog.get(case_id, {}).get("case_tier", "unknown"),
            "backend": target_backend,
            "n_observed_seeds": len(statuses),
            "statuses": [str(status) for status in statuses],
            "worst_max_r_hat": max((_safe_float(v) for v in row.get("max_r_hat", [])), default=None),
            "worst_min_ess_bulk": min((_safe_float(v) for v in row.get("min_ess_bulk", [])), default=None),
            "worst_min_ess_tail": min((_safe_float(v) for v in row.get("min_ess_tail", [])), default=None),
            "worst_ess_per_sec": min((_safe_float(v) for v in row.get("ess_per_sec", [])), default=None),
            "worst_accept_rate": min((_safe_float(v) for v in row.get("accept_rate", [])), default=None),
        }
        if case_id in supported_cases:
            supported_reviewed_cases.append(review)
            if len(statuses) != expected_seed_count:
                supported_failures.append(
                    {
                        "case": case_id,
                        "reason": "missing_seed_rows",
                        "observed_count": len(statuses),
                        "expected_count": expected_seed_count,
                    }
                )
            if any(str(status) != "ok" for status in statuses):
                supported_failures.append(
                    {
                        "case": case_id,
                        "reason": "backend_statuses_not_all_ok",
                        "statuses": [str(status) for status in statuses],
                    }
                )
            case_rhat = _safe_float(review["worst_max_r_hat"])
            if case_rhat is None:
                supported_failures.append({"case": case_id, "reason": "missing_max_r_hat"})
            elif case_rhat > max_rhat:
                supported_failures.append(
                    {
                        "case": case_id,
                        "reason": "max_r_hat_exceeds_threshold",
                        "observed": case_rhat,
                        "threshold": max_rhat,
                    }
                )

            parity_row = parity_rows_by_case.get(case_id)
            if parity_row is None:
                supported_failures.append({"case": case_id, "reason": "missing_parity_row"})
                continue
            parity_statuses = parity_row.get("statuses") if isinstance(parity_row.get("statuses"), list) else []
            parity_review = {
                "case": case_id,
                "case_tier": "supported",
                "n_observed_seeds": len(parity_statuses),
                "statuses": [str(status) for status in parity_statuses],
                "worst_max_z": max((_safe_float(v) for v in parity_row.get("max_z", [])), default=None),
            }
            supported_reviewed_parity.append(parity_review)
            if len(parity_statuses) != expected_seed_count:
                supported_failures.append(
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
                supported_failures.append(failure)
        elif case_id in control_cases:
            control_reviewed_cases.append(review)
            if len(statuses) != expected_seed_count:
                control_failures.append(
                    {
                        "case": case_id,
                        "reason": "missing_seed_rows",
                        "observed_count": len(statuses),
                        "expected_count": expected_seed_count,
                    }
                )
            if any(str(status) == "failed" for status in statuses):
                control_failures.append(
                    {
                        "case": case_id,
                        "reason": "backend_status_failed_on_control",
                        "statuses": [str(status) for status in statuses],
                    }
                )
            if _safe_float(review["worst_max_r_hat"]) is None:
                control_failures.append({"case": case_id, "reason": "missing_max_r_hat"})

    supported_passed = not supported_failures and bool(supported_reviewed_cases)
    control_passed = not control_failures and bool(control_reviewed_cases)
    stress_passed = supported_passed and control_passed

    supported_review_summary = _review_summary(
        supported_reviewed_cases,
        supported_failures,
        reviewed_parity_cases=supported_reviewed_parity,
    )
    control_review_summary = _review_summary(control_reviewed_cases, control_failures)

    assessment = {
        "schema_version": "nextstat.mams_stress_assessment.v1",
        "suite": "mams_stress",
        "source_summary_path": "mams_stress_multiseed_summary.json",
        "source_summary_sha256": _sha256_file(summary_path),
        "stress_readiness": {
            "passed": stress_passed,
            "status": _status_from_passed(stress_passed),
            "components": {
                "supported_repeatability_gate": _status_from_passed(supported_passed),
                "pathological_control_health": _status_from_passed(control_passed),
            },
        },
        "supported_repeatability_gate": {
            "passed": supported_passed,
            "status": _status_from_passed(supported_passed),
            "target_backend": target_backend,
            "policy": {
                "max_r_hat": max_rhat,
                "parity_warn_z": parity_warn_z,
                "parity_fail_z": parity_fail_z,
                "require_complete_seed_coverage": True,
                "require_all_backend_statuses_ok": True,
                "require_all_parity_statuses_ok": True,
            },
            "reviewed_cases": supported_reviewed_cases,
            "reviewed_parity_cases": supported_reviewed_parity,
            "failures": supported_failures,
            "review_summary": supported_review_summary,
        },
        "pathological_control_health": {
            "passed": control_passed,
            "status": _status_from_passed(control_passed),
            "target_backend": target_backend,
            "policy": {
                "require_complete_seed_coverage": True,
                "allow_warn_statuses": True,
                "fail_on_backend_status_failed": True,
            },
            "reviewed_cases": control_reviewed_cases,
            "failures": control_failures,
            "review_summary": control_review_summary,
        },
    }

    out_json = summary_dir / "mams_stress_assessment.json"
    out_json.write_text(json.dumps(assessment, indent=2, sort_keys=True) + "\n")

    lines: list[str] = []
    lines.append("# MAMS Stress Assessment")
    lines.append("")
    lines.append(f"Source summary: `{assessment['source_summary_path']}`")
    lines.append("")
    lines.append("## Stress readiness")
    lines.append("")
    lines.append(f"- status: `{assessment['stress_readiness']['status']}`")
    lines.append(f"- supported repeatability: `{assessment['stress_readiness']['components']['supported_repeatability_gate']}`")
    lines.append(f"- pathological controls: `{assessment['stress_readiness']['components']['pathological_control_health']}`")
    lines.append("")
    lines.append("## Supported repeatability gate")
    lines.append("")
    lines.append(f"- status: `{assessment['supported_repeatability_gate']['status']}`")
    lines.append(f"- backend: `{target_backend}`")
    lines.append(f"- max_r_hat threshold: `{_fmt(max_rhat, digits=4)}`")
    lines.append(
        f"- failing cases: `{', '.join(supported_review_summary.get('failing_cases', [])) if supported_review_summary.get('failing_cases') else '—'}`"
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
        metric_summary = supported_review_summary.get(summary_key) if isinstance(supported_review_summary.get(summary_key), dict) else {}
        lines.append(
            f"| {label} | {metric_summary.get('case', '—')} | {_fmt(metric_summary.get('value'), digits=4)} | {_fmt(policy_value, digits=4)} |"
        )
    lines.append("")
    lines.append("| Case | Statuses | Worst max R-hat | Worst min ESS_bulk | Worst ESS/s |")
    lines.append("|---|---|---:|---:|---:|")
    for row in supported_reviewed_cases:
        lines.append(
            f"| {row['case']} | `{','.join(row['statuses']) or '—'}` | {_fmt(row['worst_max_r_hat'], digits=4)} | {_fmt(row['worst_min_ess_bulk'])} | {_fmt(row['worst_ess_per_sec'])} |"
        )
    if supported_reviewed_parity:
        lines.append("")
        lines.append("| Case | Parity statuses | Worst max z |")
        lines.append("|---|---|---:|")
        for row in supported_reviewed_parity:
            lines.append(
                f"| {row['case']} | `{','.join(row['statuses']) or '—'}` | {_fmt(row['worst_max_z'], digits=4)} |"
            )
    if supported_failures:
        lines.append("")
        lines.append("### Supported repeatability failures")
        lines.append("")
        for row in supported_failures:
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

    lines.append("")
    lines.append("## Pathological controls")
    lines.append("")
    lines.append(f"- status: `{assessment['pathological_control_health']['status']}`")
    lines.append("")
    lines.append("| Case | Statuses | Worst max R-hat | Worst min ESS_bulk | Worst ESS/s |")
    lines.append("|---|---|---:|---:|---:|")
    for row in control_reviewed_cases:
        lines.append(
            f"| {row['case']} | `{','.join(row['statuses']) or '—'}` | {_fmt(row['worst_max_r_hat'], digits=4)} | {_fmt(row['worst_min_ess_bulk'])} | {_fmt(row['worst_ess_per_sec'])} |"
        )
    if control_failures:
        lines.append("")
        lines.append("### Pathological control failures")
        lines.append("")
        for row in control_failures:
            case_label = row.get("case") or "global"
            if "observed_count" in row and "expected_count" in row:
                lines.append(
                    f"- `{case_label}`: `{row['reason']}` (observed `{row['observed_count']}`, expected `{row['expected_count']}`)"
                )
            else:
                lines.append(f"- `{case_label}`: `{row['reason']}`")

    out_md = summary_dir / "mams_stress_assessment.md"
    out_md.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
