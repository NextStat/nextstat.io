#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _simplified_likelihood_exporter_promotion_bundle import (
    BUNDLE_SCHEMA_VERSION,
    CHECK_SCHEMA_VERSION,
    DEFAULT_ACCEPTED_BUNDLE_DIR,
    PROMOTION_REPORT_SCHEMA_VERSION,
    load_json,
    now_utc,
    relative_or_absolute,
    sha256_path,
)
from _simplified_likelihood_exporter_stable_review import (
    ACCEPTANCE_DOC,
    ARTIFACT_REFERENCE_DOC,
    ASSESSMENT_SCHEMA_VERSION as SCHEMA_VERSION,
    CHECKLIST_DOC,
    EXPLICIT_BOUNDARIES,
    PROMOTION_RUNBOOK_DOC,
    REQUIRED_BENCHMARK_HOST,
    REQUIRED_MAX_ABS_Q_MU_DIFF,
    REQUIRED_MAX_UPPER_LIMIT_RATIO_DEVIATION,
    REQUIRED_MIN_NET_END_TO_END_UPPER_LIMIT_SPEEDUP,
    RUNTIME_GATE_DOC,
)


def _status_from_passed(passed: bool, *, ok: str, failed: str) -> str:
    return ok if passed else failed


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_summary(path: Path, payload: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "path": relative_or_absolute(path),
        "exists": path.exists(),
        "sha256": sha256_path(path) if path.exists() else None,
        "schema_version": payload.get("schema_version") if isinstance(payload, dict) else None,
        "status": payload.get("status") if isinstance(payload, dict) else None,
    }


def _safe_load(path: Path, *, label: str, failures: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not path.exists():
        failures.append({"reason": "missing_required_artifact", "artifact": label, "path": relative_or_absolute(path)})
        return None
    try:
        payload = load_json(path)
    except Exception as exc:  # pragma: no cover - defensive contract path
        failures.append(
            {
                "reason": "invalid_json_artifact",
                "artifact": label,
                "path": relative_or_absolute(path),
                "detail": str(exc),
            }
        )
        return None
    return payload


def build_assessment(*, bundle_dir: Path, deterministic: bool) -> dict[str, Any]:
    evidence_failures: list[dict[str, Any]] = []
    evidence_warnings: list[dict[str, Any]] = []

    bundle_path = bundle_dir / "promotion_evidence.json"
    check_path = bundle_dir / "promotion_evidence_check.json"
    report_path = bundle_dir / "promotion_bundle_promotion_report.json"

    bundle = _safe_load(bundle_path, label="promotion_evidence", failures=evidence_failures)
    check = _safe_load(check_path, label="promotion_evidence_check", failures=evidence_failures)
    report = _safe_load(report_path, label="promotion_report", failures=evidence_failures)

    if isinstance(bundle, dict) and bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        evidence_failures.append(
            {
                "reason": "bundle_schema_version_mismatch",
                "expected": BUNDLE_SCHEMA_VERSION,
                "actual": bundle.get("schema_version"),
            }
        )
    if isinstance(check, dict) and check.get("schema_version") != CHECK_SCHEMA_VERSION:
        evidence_failures.append(
            {
                "reason": "bundle_check_schema_version_mismatch",
                "expected": CHECK_SCHEMA_VERSION,
                "actual": check.get("schema_version"),
            }
        )
    if isinstance(report, dict) and report.get("schema_version") != PROMOTION_REPORT_SCHEMA_VERSION:
        evidence_failures.append(
            {
                "reason": "promotion_report_schema_version_mismatch",
                "expected": PROMOTION_REPORT_SCHEMA_VERSION,
                "actual": report.get("schema_version"),
            }
        )

    bundle_support_class = (
        str(bundle.get("support_class"))
        if isinstance(bundle, dict) and isinstance(bundle.get("support_class"), str)
        else None
    )
    if bundle_support_class is not None and bundle_support_class != "research-grade":
        evidence_failures.append(
            {
                "reason": "support_class_not_research_grade",
                "actual": bundle_support_class,
            }
        )

    check_ok = bool(check.get("ok", False)) if isinstance(check, dict) else False
    check_status = str(check.get("status")) if isinstance(check, dict) else None
    require_promotion_ready = bool(check.get("require_promotion_ready", False)) if isinstance(check, dict) else False
    if isinstance(check, dict) and (check_status != "passed" or not check_ok):
        evidence_failures.append(
            {
                "reason": "bundle_check_not_passed",
                "status": check_status,
                "ok": check_ok,
            }
        )
    if isinstance(check, dict) and not require_promotion_ready:
        evidence_failures.append({"reason": "bundle_check_not_require_promotion_ready"})

    promotion_report_status = str(report.get("status")) if isinstance(report, dict) else None
    promoted = bool(report.get("promoted", False)) if isinstance(report, dict) else False
    if isinstance(report, dict) and (promotion_report_status != "promoted" or not promoted):
        evidence_failures.append(
            {
                "reason": "promotion_report_not_promoted",
                "status": promotion_report_status,
                "promoted": promoted,
            }
        )

    evidence_passed = not evidence_failures

    bundle_summary = bundle.get("summary") if isinstance(bundle, dict) and isinstance(bundle.get("summary"), dict) else {}
    bundle_summary = bundle_summary if isinstance(bundle_summary, dict) else {}
    readiness = (
        check.get("checks", {}).get("promotion_readiness")
        if isinstance(check, dict) and isinstance(check.get("checks"), dict)
        else {}
    )
    readiness = readiness if isinstance(readiness, dict) else {}
    exporter_surface = (
        bundle.get("exporter_surface")
        if isinstance(bundle, dict) and isinstance(bundle.get("exporter_surface"), dict)
        else {}
    )
    exporter_surface = exporter_surface if isinstance(exporter_surface, dict) else {}
    explicit_boundaries = (
        list(exporter_surface.get("explicit_boundaries", []))
        if isinstance(exporter_surface.get("explicit_boundaries"), list)
        else []
    )

    reviewed_claims = {
        "bundle_check_passed": check_status == "passed" and check_ok,
        "promotion_report_status": promotion_report_status,
        "benchmark_host": readiness.get("actual_benchmark_host"),
        "future_stable_review_ready": bool(readiness.get("actual_future_stable_review_ready", False)),
        "research_grade_acceptance_supported": bool(
            readiness.get("actual_research_grade_acceptance_supported", False)
        ),
        "committed_snapshot_supported": bool(readiness.get("actual_committed_snapshot_supported", False)),
        "max_abs_q_mu_diff": readiness.get("actual_max_abs_q_mu_diff"),
        "max_upper_limit_ratio_deviation": readiness.get("actual_max_upper_limit_ratio_deviation"),
        "min_net_end_to_end_upper_limit_speedup": readiness.get(
            "actual_min_net_end_to_end_upper_limit_speedup"
        ),
        "explicit_boundaries": explicit_boundaries,
    }

    review_failures: list[dict[str, Any]] = []
    review_warnings: list[dict[str, Any]] = []

    if not evidence_passed:
        review_failures.append({"reason": "evidence_validity_not_passed"})
    if isinstance(check, dict) and (check_status != "passed" or not check_ok):
        review_failures.append(
            {
                "reason": "bundle_check_not_passed",
                "status": check_status,
                "ok": check_ok,
            }
        )
    if isinstance(report, dict) and (promotion_report_status != "promoted" or not promoted):
        review_failures.append(
            {
                "reason": "promotion_report_not_promoted",
                "status": promotion_report_status,
                "promoted": promoted,
            }
        )

    if reviewed_claims["benchmark_host"] != REQUIRED_BENCHMARK_HOST:
        review_failures.append(
            {
                "reason": "benchmark_host_mismatch",
                "expected": REQUIRED_BENCHMARK_HOST,
                "actual": reviewed_claims["benchmark_host"],
            }
        )
    if not reviewed_claims["future_stable_review_ready"]:
        review_failures.append({"reason": "future_stable_review_not_ready"})
    if not reviewed_claims["research_grade_acceptance_supported"]:
        review_failures.append({"reason": "research_grade_acceptance_not_supported"})
    if not reviewed_claims["committed_snapshot_supported"]:
        review_failures.append({"reason": "committed_snapshot_not_supported"})

    max_abs_q_mu_diff = reviewed_claims["max_abs_q_mu_diff"]
    if not isinstance(max_abs_q_mu_diff, (int, float)):
        review_failures.append({"reason": "missing_max_abs_q_mu_diff"})
    elif float(max_abs_q_mu_diff) > REQUIRED_MAX_ABS_Q_MU_DIFF:
        review_failures.append(
            {
                "reason": "max_abs_q_mu_diff_exceeds_threshold",
                "expected_max": REQUIRED_MAX_ABS_Q_MU_DIFF,
                "actual": float(max_abs_q_mu_diff),
            }
        )

    max_ul_ratio_dev = reviewed_claims["max_upper_limit_ratio_deviation"]
    if not isinstance(max_ul_ratio_dev, (int, float)):
        review_failures.append({"reason": "missing_max_upper_limit_ratio_deviation"})
    elif float(max_ul_ratio_dev) > REQUIRED_MAX_UPPER_LIMIT_RATIO_DEVIATION:
        review_failures.append(
            {
                "reason": "max_upper_limit_ratio_deviation_exceeds_threshold",
                "expected_max": REQUIRED_MAX_UPPER_LIMIT_RATIO_DEVIATION,
                "actual": float(max_ul_ratio_dev),
            }
        )

    min_net_speedup = reviewed_claims["min_net_end_to_end_upper_limit_speedup"]
    if not isinstance(min_net_speedup, (int, float)):
        review_failures.append({"reason": "missing_min_net_end_to_end_upper_limit_speedup"})
    elif float(min_net_speedup) < REQUIRED_MIN_NET_END_TO_END_UPPER_LIMIT_SPEEDUP:
        review_failures.append(
            {
                "reason": "min_net_end_to_end_upper_limit_speedup_below_threshold",
                "expected_min": REQUIRED_MIN_NET_END_TO_END_UPPER_LIMIT_SPEEDUP,
                "actual": float(min_net_speedup),
            }
        )

    missing_boundaries = [boundary for boundary in EXPLICIT_BOUNDARIES if boundary not in explicit_boundaries]
    for boundary in missing_boundaries:
        review_failures.append({"reason": "missing_explicit_boundary", "boundary": boundary})

    review_ready = not review_failures
    review_status = _status_from_passed(review_ready, ok="review_ready", failed="not_ready")

    benchmark_host = (
        reviewed_claims["benchmark_host"]
        if isinstance(reviewed_claims["benchmark_host"], str)
        else bundle_summary.get("benchmark_host")
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "surface": "simplified_likelihood_exporter",
        "support_class": "research-grade",
        "automatic_stable_promotion": False,
        "generated_at_utc": now_utc(deterministic),
        "bundle_dir": relative_or_absolute(bundle_dir),
        "source_artifacts": {
            "promotion_evidence": _artifact_summary(bundle_path, bundle),
            "promotion_evidence_check": _artifact_summary(check_path, check),
            "promotion_report": _artifact_summary(report_path, report),
        },
        "review_documents": {
            "checklist_doc": CHECKLIST_DOC,
            "acceptance_doc": ACCEPTANCE_DOC,
            "runtime_gate_doc": RUNTIME_GATE_DOC,
            "promotion_runbook_doc": PROMOTION_RUNBOOK_DOC,
            "artifact_reference_doc": ARTIFACT_REFERENCE_DOC,
        },
        "evidence_validity": {
            "passed": evidence_passed,
            "status": _status_from_passed(evidence_passed, ok="passed", failed="failed"),
            "failures": evidence_failures,
            "warnings": evidence_warnings,
        },
        "stable_review": {
            "ready": review_ready,
            "status": review_status,
            "policy": {
                "required_support_class": "research-grade",
                "automatic_stable_promotion": False,
                "required_benchmark_host": REQUIRED_BENCHMARK_HOST,
                "required_min_net_end_to_end_upper_limit_speedup": REQUIRED_MIN_NET_END_TO_END_UPPER_LIMIT_SPEEDUP,
                "required_max_abs_q_mu_diff": REQUIRED_MAX_ABS_Q_MU_DIFF,
                "required_max_upper_limit_ratio_deviation": REQUIRED_MAX_UPPER_LIMIT_RATIO_DEVIATION,
                "required_promotion_report_status": "promoted",
                "required_explicit_boundaries": EXPLICIT_BOUNDARIES,
            },
            "reviewed_claims": reviewed_claims,
            "failures": review_failures,
            "warnings": review_warnings,
        },
        "summary": {
            "status": review_status,
            "benchmark_host": benchmark_host,
            "future_stable_review_ready": bool(reviewed_claims["future_stable_review_ready"]),
            "automatic_stable_promotion": False,
            "next_action": (
                "maintainer stable review may proceed, but exporter support class remains research-grade until an explicit promotion decision"
                if review_ready
                else "fix evidence validity or threshold failures before requesting exporter stable review"
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assess whether the accepted simplified-likelihood exporter bundle is ready for a formal stable-review discussion."
    )
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_ACCEPTED_BUNDLE_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    bundle_dir = args.bundle_dir.resolve()
    out_path = (
        args.out.resolve()
        if args.out is not None
        else (bundle_dir / "stable_review_assessment.json")
    )

    assessment = build_assessment(bundle_dir=bundle_dir, deterministic=args.deterministic)
    _write_json(out_path, assessment)

    print(
        "Exporter stable-review assessment:",
        f"status={assessment['summary']['status']}",
        f"host={assessment['summary']['benchmark_host']}",
        f"out={relative_or_absolute(out_path)}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
