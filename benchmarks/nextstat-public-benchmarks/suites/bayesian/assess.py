#!/usr/bin/env python3
"""Assess canonical Bayesian suite results for core quality vs promotion readiness."""

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
    if abs(v) >= 1:
        return f"{v:.{digits}f}".rstrip("0").rstrip(".")
    if v == 0:
        return "0"
    return f"{v:.{digits}g}"


def _case_metrics(case_obj: dict[str, Any]) -> dict[str, Any]:
    diag = case_obj.get("diagnostics_summary") if isinstance(case_obj.get("diagnostics_summary"), dict) else {}
    timing = case_obj.get("timing") if isinstance(case_obj.get("timing"), dict) else {}
    ess_bulk_per_sec = timing.get("ess_bulk_per_sec") if isinstance(timing.get("ess_bulk_per_sec"), dict) else {}
    return {
        "status": str(case_obj.get("status", "unknown")),
        "reason": str(case_obj.get("reason")) if isinstance(case_obj.get("reason"), str) else None,
        "divergence_rate": _safe_float(diag.get("divergence_rate")),
        "max_treedepth_rate": _safe_float(diag.get("max_treedepth_rate")),
        "max_r_hat": _safe_float(diag.get("max_r_hat")),
        "min_ess_bulk": _safe_float(diag.get("min_ess_bulk")),
        "min_ess_tail": _safe_float(diag.get("min_ess_tail")),
        "min_ebfmi": _safe_float(diag.get("min_ebfmi")),
        "min_ess_bulk_per_sec": _safe_float(ess_bulk_per_sec.get("min")),
    }


def _metric_extreme(
    reviewed_cases: list[dict[str, Any]],
    *,
    metric_key: str,
    prefer: str,
) -> dict[str, Any]:
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


def _review_summary(reviewed_cases: list[dict[str, Any]], promotion_failures: list[dict[str, Any]]) -> dict[str, Any]:
    failing_cases = sorted({str(row.get("case")) for row in promotion_failures if row.get("case")})
    return {
        "n_reviewed_cases": len(reviewed_cases),
        "n_failures": len(promotion_failures),
        "n_failing_cases": len(failing_cases),
        "failing_cases": failing_cases,
        "worst_divergence_rate": _metric_extreme(reviewed_cases, metric_key="divergence_rate", prefer="max"),
        "worst_max_treedepth_rate": _metric_extreme(reviewed_cases, metric_key="max_treedepth_rate", prefer="max"),
        "worst_max_r_hat": _metric_extreme(reviewed_cases, metric_key="max_r_hat", prefer="max"),
        "worst_min_ess_bulk": _metric_extreme(reviewed_cases, metric_key="min_ess_bulk", prefer="min"),
        "worst_min_ess_tail": _metric_extreme(reviewed_cases, metric_key="min_ess_tail", prefer="min"),
        "worst_min_ess_bulk_per_sec": _metric_extreme(
            reviewed_cases,
            metric_key="min_ess_bulk_per_sec",
            prefer="min",
        ),
        "worst_min_ebfmi": _metric_extreme(reviewed_cases, metric_key="min_ebfmi", prefer="min"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Assess Bayesian suite core quality and promotion readiness.")
    ap.add_argument("suite_dir", help="Path to suite results directory (contains bayesian_suite.json)")
    ap.add_argument(
        "--promotion-backend",
        default="nextstat",
        help="Backend to evaluate for promotion readiness (default: nextstat)",
    )
    ap.add_argument(
        "--promotion-max-rhat",
        type=float,
        default=1.01,
        help="Fail promotion readiness if a reviewed case exceeds this R-hat threshold (default: 1.01)",
    )
    ap.add_argument(
        "--promotion-max-divergence-rate",
        type=float,
        default=0.0,
        help="Fail promotion readiness if divergence_rate exceeds this threshold (default: 0.0)",
    )
    ap.add_argument(
        "--promotion-max-treedepth-rate",
        type=float,
        default=0.0,
        help="Fail promotion readiness if max_treedepth_rate exceeds this threshold (default: 0.0)",
    )
    ap.add_argument(
        "--promotion-min-ebfmi",
        type=float,
        default=0.3,
        help="Fail promotion readiness if min_ebfmi falls below this threshold (default: 0.3)",
    )
    ap.add_argument(
        "--promotion-min-ess-bulk",
        type=float,
        default=0.0,
        help="Optional floor for min_ess_bulk; values <= 0 disable the check",
    )
    ap.add_argument(
        "--promotion-min-ess-tail",
        type=float,
        default=0.0,
        help="Optional floor for min_ess_tail; values <= 0 disable the check",
    )
    ap.add_argument(
        "--promotion-min-ess-bulk-per-sec",
        type=float,
        default=0.0,
        help="Optional floor for min_ess_bulk_per_sec; values <= 0 disable the check",
    )
    args = ap.parse_args()

    suite_dir = Path(args.suite_dir).resolve()
    suite_path = suite_dir / "bayesian_suite.json"
    if not suite_path.exists():
        raise SystemExit(f"missing suite artifact: {suite_path}")

    suite = _load_json(suite_path)
    summary = suite.get("summary") if isinstance(suite.get("summary"), dict) else {}
    parity = suite.get("parity") if isinstance(suite.get("parity"), dict) else {}
    cases = suite.get("cases") if isinstance(suite.get("cases"), list) else []

    n_failed = int(summary.get("n_failed", 0) or 0)
    n_warn = int(summary.get("n_warn", 0) or 0)
    n_parity_fail = int(summary.get("n_parity_fail", 0) or 0)
    n_parity_warn = int(summary.get("n_parity_warn", 0) or 0)

    core_failures: list[dict[str, Any]] = []
    core_warnings: list[dict[str, Any]] = []
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

    target_backend = str(args.promotion_backend)
    max_rhat = float(args.promotion_max_rhat)
    max_divergence_rate = float(args.promotion_max_divergence_rate)
    max_treedepth_rate = float(args.promotion_max_treedepth_rate)
    min_ebfmi = float(args.promotion_min_ebfmi)
    min_ess_bulk = float(args.promotion_min_ess_bulk)
    min_ess_tail = float(args.promotion_min_ess_tail)
    min_ess_bulk_per_sec = float(args.promotion_min_ess_bulk_per_sec)

    target_entries = [c for c in cases if isinstance(c, dict) and str(c.get("backend")) == target_backend]
    reviewed_cases: list[dict[str, Any]] = []
    promotion_failures: list[dict[str, Any]] = []

    if not target_entries:
        promotion_failures.append(
            {
                "case": None,
                "reason": "no_matching_backend_cases",
                "backend": target_backend,
            }
        )

    if not core_passed:
        promotion_failures.append(
            {
                "case": None,
                "reason": "core_quality_not_valid",
            }
        )

    for entry in target_entries:
        case_id = str(entry.get("case", "unknown"))
        rel_path = str(entry.get("path", ""))
        case_path = (suite_dir / rel_path).resolve()
        if not rel_path or not case_path.exists():
            reviewed_cases.append(
                {
                    "case": case_id,
                    "backend": target_backend,
                    "path": rel_path,
                    "status": "missing",
                    "reason": "missing_case_artifact",
                    "divergence_rate": None,
                    "max_treedepth_rate": None,
                    "max_r_hat": None,
                    "min_ess_bulk": None,
                    "min_ess_tail": None,
                    "min_ebfmi": None,
                    "min_ess_bulk_per_sec": None,
                }
            )
            promotion_failures.append(
                {
                    "case": case_id,
                    "reason": "missing_case_artifact",
                    "path": rel_path,
                }
            )
            continue

        case_obj = _load_json(case_path)
        metrics = _case_metrics(case_obj)
        reviewed = {
            "case": case_id,
            "backend": target_backend,
            "path": rel_path,
            **metrics,
        }
        reviewed_cases.append(reviewed)

        if reviewed["status"] != "ok":
            promotion_failures.append(
                {
                    "case": case_id,
                    "reason": "case_status_not_ok",
                    "status": reviewed["status"],
                    "detail": reviewed["reason"],
                }
            )
            continue

        if reviewed["max_r_hat"] is None:
            promotion_failures.append({"case": case_id, "reason": "missing_max_r_hat"})
        elif reviewed["max_r_hat"] > max_rhat:
            promotion_failures.append(
                {
                    "case": case_id,
                    "reason": "max_r_hat_exceeds_threshold",
                    "observed": reviewed["max_r_hat"],
                    "threshold": max_rhat,
                }
            )

        if reviewed["divergence_rate"] is None:
            promotion_failures.append({"case": case_id, "reason": "missing_divergence_rate"})
        elif reviewed["divergence_rate"] > max_divergence_rate:
            promotion_failures.append(
                {
                    "case": case_id,
                    "reason": "divergence_rate_exceeds_threshold",
                    "observed": reviewed["divergence_rate"],
                    "threshold": max_divergence_rate,
                }
            )

        if reviewed["max_treedepth_rate"] is None:
            promotion_failures.append({"case": case_id, "reason": "missing_max_treedepth_rate"})
        elif reviewed["max_treedepth_rate"] > max_treedepth_rate:
            promotion_failures.append(
                {
                    "case": case_id,
                    "reason": "max_treedepth_rate_exceeds_threshold",
                    "observed": reviewed["max_treedepth_rate"],
                    "threshold": max_treedepth_rate,
                }
            )

        if reviewed["min_ebfmi"] is None:
            promotion_failures.append({"case": case_id, "reason": "missing_min_ebfmi"})
        elif reviewed["min_ebfmi"] < min_ebfmi:
            promotion_failures.append(
                {
                    "case": case_id,
                    "reason": "min_ebfmi_below_threshold",
                    "observed": reviewed["min_ebfmi"],
                    "threshold": min_ebfmi,
                }
            )

        if min_ess_bulk > 0.0:
            if reviewed["min_ess_bulk"] is None:
                promotion_failures.append(
                    {
                        "case": case_id,
                        "reason": "missing_min_ess_bulk",
                        "threshold": min_ess_bulk,
                    }
                )
            elif reviewed["min_ess_bulk"] < min_ess_bulk:
                promotion_failures.append(
                    {
                        "case": case_id,
                        "reason": "min_ess_bulk_below_threshold",
                        "observed": reviewed["min_ess_bulk"],
                        "threshold": min_ess_bulk,
                    }
                )

        if min_ess_tail > 0.0:
            if reviewed["min_ess_tail"] is None:
                promotion_failures.append(
                    {
                        "case": case_id,
                        "reason": "missing_min_ess_tail",
                        "threshold": min_ess_tail,
                    }
                )
            elif reviewed["min_ess_tail"] < min_ess_tail:
                promotion_failures.append(
                    {
                        "case": case_id,
                        "reason": "min_ess_tail_below_threshold",
                        "observed": reviewed["min_ess_tail"],
                        "threshold": min_ess_tail,
                    }
                )

        if min_ess_bulk_per_sec > 0.0:
            if reviewed["min_ess_bulk_per_sec"] is None:
                promotion_failures.append(
                    {
                        "case": case_id,
                        "reason": "missing_min_ess_bulk_per_sec",
                        "threshold": min_ess_bulk_per_sec,
                    }
                )
            elif reviewed["min_ess_bulk_per_sec"] < min_ess_bulk_per_sec:
                promotion_failures.append(
                    {
                        "case": case_id,
                        "reason": "min_ess_bulk_per_sec_below_threshold",
                        "observed": reviewed["min_ess_bulk_per_sec"],
                        "threshold": min_ess_bulk_per_sec,
                    }
                )

    review_summary = _review_summary(reviewed_cases, promotion_failures)

    assessment = {
        "schema_version": "nextstat.bayesian_assessment.v1",
        "suite": "bayesian",
        "source_suite_path": "bayesian_suite.json",
        "source_suite_sha256": _sha256_file(suite_path),
        "source_suite_summary": summary,
        "parity_summary": {
            "compare": parity.get("compare"),
            "method": parity.get("method"),
            "warn_z": parity.get("warn_z"),
            "fail_z": parity.get("fail_z"),
            "n_rows": len(parity.get("rows", [])) if isinstance(parity.get("rows"), list) else 0,
        },
        "core_quality": {
            "passed": core_passed,
            "status": _status_from_passed(core_passed),
            "failures": core_failures,
            "warnings": core_warnings,
        },
        "promotion_gate": {
            "passed": not promotion_failures,
            "status": _status_from_passed(not promotion_failures),
            "target_backend": target_backend,
            "policy": {
                "max_r_hat": max_rhat,
                "max_divergence_rate": max_divergence_rate,
                "max_treedepth_rate": max_treedepth_rate,
                "min_ebfmi": min_ebfmi,
                "min_ess_bulk": min_ess_bulk if min_ess_bulk > 0.0 else None,
                "min_ess_tail": min_ess_tail if min_ess_tail > 0.0 else None,
                "min_ess_bulk_per_sec": min_ess_bulk_per_sec if min_ess_bulk_per_sec > 0.0 else None,
            },
            "reviewed_cases": reviewed_cases,
            "review_summary": review_summary,
            "failures": promotion_failures,
        },
    }

    out_json = suite_dir / "bayesian_assessment.json"
    out_json.write_text(json.dumps(assessment, indent=2, sort_keys=True) + "\n")

    lines: list[str] = []
    lines.append("# Bayesian Assessment")
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
    lines.append(f"- backend: `{target_backend}`")
    lines.append(f"- max_r_hat threshold: `{_fmt(max_rhat, digits=4)}`")
    lines.append(f"- divergence_rate threshold: `{_fmt(max_divergence_rate, digits=4)}`")
    lines.append(f"- max_treedepth_rate threshold: `{_fmt(max_treedepth_rate, digits=4)}`")
    lines.append(f"- min_ebfmi threshold: `{_fmt(min_ebfmi, digits=4)}`")
    if min_ess_bulk > 0.0:
        lines.append(f"- min_ess_bulk threshold: `{_fmt(min_ess_bulk)}`")
    if min_ess_tail > 0.0:
        lines.append(f"- min_ess_tail threshold: `{_fmt(min_ess_tail)}`")
    if min_ess_bulk_per_sec > 0.0:
        lines.append(f"- min_ess_bulk_per_sec threshold: `{_fmt(min_ess_bulk_per_sec)}`")
    lines.append("")
    lines.append("### Promotion health summary")
    lines.append("")
    lines.append(f"- reviewed cases: `{review_summary['n_reviewed_cases']}`")
    lines.append(f"- failure rows: `{review_summary['n_failures']}`")
    lines.append(f"- failing cases: `{', '.join(review_summary['failing_cases']) if review_summary['failing_cases'] else '—'}`")
    lines.append("")
    lines.append("| Metric | Worst case | Observed | Policy |")
    lines.append("|---|---|---:|---:|")
    for label, summary_key, policy_value in [
        ("divergence_rate", "worst_divergence_rate", max_divergence_rate),
        ("max_treedepth_rate", "worst_max_treedepth_rate", max_treedepth_rate),
        ("max_r_hat", "worst_max_r_hat", max_rhat),
        ("min_ebfmi", "worst_min_ebfmi", min_ebfmi),
        ("min_ess_bulk", "worst_min_ess_bulk", min_ess_bulk if min_ess_bulk > 0.0 else None),
        ("min_ess_tail", "worst_min_ess_tail", min_ess_tail if min_ess_tail > 0.0 else None),
        (
            "min_ess_bulk_per_sec",
            "worst_min_ess_bulk_per_sec",
            min_ess_bulk_per_sec if min_ess_bulk_per_sec > 0.0 else None,
        ),
    ]:
        metric_summary = review_summary[summary_key]
        lines.append(
            f"| {label} | {metric_summary['case'] or '—'} | {_fmt(metric_summary['value'], digits=4)} | {_fmt(policy_value, digits=4)} |"
        )
    lines.append("")
    lines.append("| Case | Status | div rate | td rate | max R-hat | min ESS_bulk | min ESS_tail | min ESS_bulk/s | min E-BFMI |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in reviewed_cases:
        lines.append(
            f"| {row['case']} | {row['status']} | {_fmt(row['divergence_rate'], digits=4)} | {_fmt(row['max_treedepth_rate'], digits=4)} | {_fmt(row['max_r_hat'], digits=4)} | {_fmt(row['min_ess_bulk'])} | {_fmt(row['min_ess_tail'])} | {_fmt(row['min_ess_bulk_per_sec'])} | {_fmt(row['min_ebfmi'], digits=4)} |"
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

    out_md = suite_dir / "bayesian_assessment.md"
    out_md.write_text("\n".join(lines) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
