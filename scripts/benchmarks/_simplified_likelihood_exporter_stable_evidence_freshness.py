from __future__ import annotations

import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from _simplified_likelihood_export_benchmark import now_utc, relative_or_absolute
from _simplified_likelihood_exporter_stable_evidence_policy import (
    POLICY_DOC,
    POLICY_EXAMPLE,
    POLICY_SCHEMA,
)


SCHEMA_VERSION = "nextstat_simplified_likelihood_exporter_stable_evidence_freshness_report_v0"
FRESHNESS_DOC = (
    "docs/benchmarks/simplified-likelihood-exporter-stable-evidence-freshness-2026-03-09.md"
)
FRESHNESS_SCHEMA = (
    "docs/schemas/benchmarks/simplified_likelihood_exporter_stable_evidence_freshness_report_v0.schema.json"
)
FRESHNESS_EXAMPLE = (
    "docs/specs/benchmarks/simplified_likelihood_exporter_stable_evidence_freshness_report_v0.example.json"
)
DEFAULT_OUT_PATH = (
    Path("benchmarks")
    / "artifacts"
    / "simplified_likelihood_exporter_promotion_bundles"
    / "nextstat-bench"
    / "accepted"
    / "stable_evidence_freshness_report.json"
)

MAX_SNAPSHOT_AGE_DAYS = 45
REQUIRED_SNAPSHOT_STATUS = "persisted"
REQUIRED_PUBLIC_VALIDATION_STATUS = "ok"
REQUIRED_POLICY_STATUS = "accepted"
REQUIRED_STABLE_PROMOTION_STATUS = "accepted"
REQUIRED_STABLE_PROMOTION_SUPPORT_CLASS = "stable"


def _parse_snapshot_date(snapshot_id: str) -> date:
    match = re.search(r"(\d{8})T\d{6}Z", snapshot_id)
    if not match:
        raise ValueError(f"snapshot_id does not contain a UTC timestamp stamp: {snapshot_id!r}")
    return datetime.strptime(match.group(1), "%Y%m%d").date()


def _parse_reference_date(reference_date: str | None, *, snapshot_date: date, deterministic: bool) -> date:
    if reference_date:
        return date.fromisoformat(reference_date)
    if deterministic:
        return snapshot_date
    return datetime.now(UTC).date()


def _failure(reason: str, detail: str) -> dict[str, str]:
    return {"reason": reason, "detail": detail}


def build_freshness_report(
    *,
    snapshot_report_path: Path,
    snapshot_report: dict[str, Any],
    public_validation_report_path: Path,
    public_validation_report: dict[str, Any],
    stable_evidence_policy_path: Path,
    stable_evidence_policy: dict[str, Any],
    stable_promotion_decision_path: Path,
    stable_promotion_decision: dict[str, Any],
    reference_date: str | None,
    deterministic: bool,
) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    snapshot_id = snapshot_report.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("snapshot report missing snapshot_id")
    snapshot_date = _parse_snapshot_date(snapshot_id)
    evaluated_reference_date = _parse_reference_date(
        reference_date,
        snapshot_date=snapshot_date,
        deterministic=deterministic,
    )
    snapshot_age_days = (evaluated_reference_date - snapshot_date).days

    snapshot_status = snapshot_report.get("status")
    if snapshot_status != REQUIRED_SNAPSHOT_STATUS:
        failures.append(
            _failure(
                "snapshot_status_not_persisted",
                f"expected {REQUIRED_SNAPSHOT_STATUS}, got {snapshot_status!r}",
            )
        )

    public_validation_status = public_validation_report.get("status")
    if public_validation_status != REQUIRED_PUBLIC_VALIDATION_STATUS:
        failures.append(
            _failure(
                "public_validation_status_not_ok",
                f"expected {REQUIRED_PUBLIC_VALIDATION_STATUS}, got {public_validation_status!r}",
            )
        )

    policy_status = stable_evidence_policy.get("status")
    if policy_status != REQUIRED_POLICY_STATUS:
        failures.append(
            _failure(
                "stable_evidence_policy_not_accepted",
                f"expected {REQUIRED_POLICY_STATUS}, got {policy_status!r}",
            )
        )

    decision_status = stable_promotion_decision.get("status")
    if decision_status != REQUIRED_STABLE_PROMOTION_STATUS:
        failures.append(
            _failure(
                "stable_promotion_decision_not_accepted",
                f"expected {REQUIRED_STABLE_PROMOTION_STATUS}, got {decision_status!r}",
            )
        )
    decision_support_class = stable_promotion_decision.get("support_class")
    if decision_support_class != REQUIRED_STABLE_PROMOTION_SUPPORT_CLASS:
        failures.append(
            _failure(
                "stable_promotion_support_class_not_stable",
                f"expected {REQUIRED_STABLE_PROMOTION_SUPPORT_CLASS}, got {decision_support_class!r}",
            )
        )

    source_summary = snapshot_report.get("source_summary")
    if not isinstance(source_summary, dict):
        raise ValueError("snapshot report missing source_summary")
    public_summary = public_validation_report.get("summary")
    if not isinstance(public_summary, dict):
        raise ValueError("public validation report missing summary")
    policy_floor = stable_evidence_policy.get("stable_evidence_floor")
    if not isinstance(policy_floor, dict):
        raise ValueError("stable evidence policy missing stable_evidence_floor")

    benchmark_host = source_summary.get("benchmark_host")
    if benchmark_host != stable_evidence_policy.get("benchmark_host"):
        failures.append(
            _failure(
                "benchmark_host_mismatch",
                f"policy expects {stable_evidence_policy.get('benchmark_host')!r}, got {benchmark_host!r}",
            )
        )

    min_total_export_cases = int(policy_floor["min_total_export_matrix_case_count"])
    min_public_cases = int(policy_floor["min_public_reinterpretation_style_case_count"])
    min_synthetic_speedup = float(
        policy_floor["min_synthetic_control_net_end_to_end_upper_limit_speedup"]
    )
    min_public_speedup = float(policy_floor["min_public_validation_net_end_to_end_upper_limit_speedup"])

    export_case_count = int(source_summary["export_matrix_case_count"])
    public_case_count = int(public_summary["public_case_count"])
    synthetic_min_speedup = float(
        source_summary["export_matrix_synthetic_min_net_end_to_end_upper_limit_speedup"]
    )
    public_min_speedup = float(public_summary["min_net_end_to_end_upper_limit_speedup"])
    cases_outside_boundary = int(public_summary["cases_outside_promoted_stable_runtime_boundary"])

    if export_case_count < min_total_export_cases:
        failures.append(
            _failure(
                "export_case_count_below_floor",
                f"expected at least {min_total_export_cases}, got {export_case_count}",
            )
        )
    if public_case_count < min_public_cases:
        failures.append(
            _failure(
                "public_case_count_below_floor",
                f"expected at least {min_public_cases}, got {public_case_count}",
            )
        )
    if synthetic_min_speedup < min_synthetic_speedup:
        failures.append(
            _failure(
                "synthetic_speedup_below_floor",
                f"expected at least {min_synthetic_speedup}, got {synthetic_min_speedup}",
            )
        )
    if public_min_speedup < min_public_speedup:
        failures.append(
            _failure(
                "public_speedup_below_floor",
                f"expected at least {min_public_speedup}, got {public_min_speedup}",
            )
        )
    if cases_outside_boundary != 0:
        failures.append(
            _failure(
                "cases_outside_promoted_runtime_boundary",
                f"expected 0, got {cases_outside_boundary}",
            )
        )
    if snapshot_age_days < 0:
        failures.append(
            _failure(
                "reference_date_precedes_snapshot_date",
                f"reference_date {evaluated_reference_date.isoformat()} is earlier than snapshot_date {snapshot_date.isoformat()}",
            )
        )
    if snapshot_age_days > MAX_SNAPSHOT_AGE_DAYS:
        failures.append(
            _failure(
                "snapshot_age_exceeds_window",
                f"snapshot age {snapshot_age_days} exceeds max_snapshot_age_days {MAX_SNAPSHOT_AGE_DAYS}",
            )
        )
    elif snapshot_age_days >= 30:
        warnings.append(
            _failure(
                "snapshot_age_nearing_breach_window",
                f"snapshot age {snapshot_age_days} is within 15 days of max_snapshot_age_days {MAX_SNAPSHOT_AGE_DAYS}",
            )
        )

    passed = not failures
    status = "fresh" if passed else "breached"

    return {
        "schema_version": SCHEMA_VERSION,
        "surface": "simplified_likelihood_exporter",
        "report_kind": "stable_evidence_freshness",
        "status": status,
        "support_class": "stable",
        "generated_at_utc": now_utc(deterministic),
        "reference_date": evaluated_reference_date.isoformat(),
        "benchmark_host": benchmark_host,
        "freshness_policy": {
            "max_snapshot_age_days": MAX_SNAPSHOT_AGE_DAYS,
            "require_snapshot_status": REQUIRED_SNAPSHOT_STATUS,
            "require_public_validation_status": REQUIRED_PUBLIC_VALIDATION_STATUS,
            "require_policy_status": REQUIRED_POLICY_STATUS,
            "require_stable_promotion_status": REQUIRED_STABLE_PROMOTION_STATUS,
            "require_stable_promotion_support_class": REQUIRED_STABLE_PROMOTION_SUPPORT_CLASS,
            "require_policy_case_floors": True,
            "require_cases_inside_promoted_stable_runtime_boundary": True,
        },
        "source_artifacts": {
            "snapshot_report_path": relative_or_absolute(snapshot_report_path),
            "public_validation_report_path": relative_or_absolute(public_validation_report_path),
            "stable_evidence_policy_path": relative_or_absolute(stable_evidence_policy_path),
            "stable_promotion_decision_path": relative_or_absolute(stable_promotion_decision_path),
            "policy_doc_path": POLICY_DOC,
            "policy_schema_path": POLICY_SCHEMA,
            "policy_example_path": POLICY_EXAMPLE,
        },
        "freshness_observation": {
            "snapshot_id": snapshot_id,
            "snapshot_date": snapshot_date.isoformat(),
            "reference_date": evaluated_reference_date.isoformat(),
            "snapshot_age_days": snapshot_age_days,
            "snapshot_status": snapshot_status,
            "public_validation_status": public_validation_status,
            "policy_status": policy_status,
            "stable_promotion_status": decision_status,
            "stable_promotion_support_class": decision_support_class,
            "export_matrix_case_count": export_case_count,
            "public_case_count": public_case_count,
            "synthetic_min_net_end_to_end_upper_limit_speedup": synthetic_min_speedup,
            "public_min_net_end_to_end_upper_limit_speedup": public_min_speedup,
            "cases_outside_promoted_stable_runtime_boundary": cases_outside_boundary,
        },
        "validity": {
            "passed": passed,
            "status": "passed" if passed else "failed",
            "failures": failures,
            "warnings": warnings,
        },
        "summary": {
            "status": status,
            "benchmark_host": benchmark_host,
            "snapshot_id": snapshot_id,
            "snapshot_age_days": snapshot_age_days,
            "max_snapshot_age_days": MAX_SNAPSHOT_AGE_DAYS,
            "export_matrix_case_count": export_case_count,
            "public_case_count": public_case_count,
            "top_level_failures": [failure["reason"] for failure in failures],
        },
    }
