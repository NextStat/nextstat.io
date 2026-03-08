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
    REPO_ROOT,
    load_json,
    now_utc,
    relative_or_absolute,
    sha256_path,
)
from _simplified_likelihood_exporter_stable_candidate import (
    ACCEPTANCE_DOC,
    ARTIFACT_REFERENCE_DOC,
    BLOCKER_MATRIX_DOC,
    BLOCKER_MATRIX_SCHEMA_VERSION,
    PROMOTION_RUNBOOK_DOC,
    REQUIRED_BENCHMARK_HOST,
    REVIEW_PACKET_DOC,
    REVIEW_PACKET_SCHEMA_VERSION as SCHEMA_VERSION,
    RUNTIME_GATE_DOC,
    SOURCE_SEMANTICS_BOUNDARY_DOC,
    STABLE_REVIEW_CHECKLIST_DOC,
)
from _simplified_likelihood_exporter_stable_promotion import (
    DECISION_DOC as STABLE_PROMOTION_DECISION_DOC,
    DECISION_SCHEMA_VERSION as STABLE_PROMOTION_DECISION_SCHEMA_VERSION,
    RELEASE_PR_CHECKLIST_DOC as STABLE_PROMOTION_RELEASE_PR_CHECKLIST_DOC,
)
from _simplified_likelihood_exporter_stable_review import (
    ASSESSMENT_SCHEMA_VERSION as STABLE_REVIEW_SCHEMA_VERSION,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_summary(
    path: Path,
    payload: dict[str, Any] | None,
    *,
    include_sha256: bool = True,
) -> dict[str, Any]:
    return {
        "path": relative_or_absolute(path),
        "exists": path.exists(),
        "sha256": sha256_path(path) if include_sha256 and path.exists() else None,
        "schema_version": payload.get("schema_version") if isinstance(payload, dict) else None,
        "status": payload.get("status") if isinstance(payload, dict) else None,
    }


def _safe_load(path: Path, *, label: str, failures: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not path.exists():
        failures.append(
            {"reason": "missing_required_artifact", "artifact": label, "path": relative_or_absolute(path)}
        )
        return None
    try:
        return load_json(path)
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


def _benchmark_artifact_path(bundle: dict[str, Any] | None, *, bundle_dir: Path) -> Path:
    if isinstance(bundle, dict):
        source_snapshot = (
            bundle.get("source_snapshot") if isinstance(bundle.get("source_snapshot"), dict) else {}
        )
        source_snapshot = source_snapshot if isinstance(source_snapshot, dict) else {}
        benchmark_artifact = (
            source_snapshot.get("benchmark_artifact")
            if isinstance(source_snapshot.get("benchmark_artifact"), dict)
            else {}
        )
        benchmark_artifact = benchmark_artifact if isinstance(benchmark_artifact, dict) else {}
        source_path = benchmark_artifact.get("source_path")
        if isinstance(source_path, str) and source_path:
            path = Path(source_path)
            if path.is_absolute():
                return path
            return (REPO_ROOT / source_path).resolve()
    return (
        REPO_ROOT
        / "benchmarks"
        / "artifacts"
        / "simplified_likelihood_export_benchmarks"
        / REQUIRED_BENCHMARK_HOST
        / "current"
        / "apex2_simplified_likelihood_report.json"
    )


def _unexpected_schema_failure(
    *, artifact: str, expected: str, payload: dict[str, Any] | None
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    actual = payload.get("schema_version")
    if actual == expected:
        return None
    return {
        "reason": "unexpected_schema_version",
        "artifact": artifact,
        "expected": expected,
        "actual": actual,
    }


def build_review_packet(*, bundle_dir: Path, deterministic: bool) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []

    bundle_path = bundle_dir / "promotion_evidence.json"
    check_path = bundle_dir / "promotion_evidence_check.json"
    report_path = bundle_dir / "promotion_bundle_promotion_report.json"
    assessment_path = bundle_dir / "stable_review_assessment.json"
    blocker_matrix_path = bundle_dir / "stable_candidate_blocker_matrix.json"
    source_semantics_boundary_path = bundle_dir / "stable_source_semantics_boundary.json"
    stable_promotion_decision_path = bundle_dir / "stable_promotion_decision.json"

    bundle = _safe_load(bundle_path, label="promotion_evidence", failures=failures)
    check = _safe_load(check_path, label="promotion_evidence_check", failures=failures)
    report = _safe_load(report_path, label="promotion_report", failures=failures)
    assessment = _safe_load(
        assessment_path, label="stable_review_assessment", failures=failures
    )
    blocker_matrix = _safe_load(
        blocker_matrix_path, label="stable_candidate_blocker_matrix", failures=failures
    )
    stable_promotion_decision = None
    if stable_promotion_decision_path.exists():
        try:
            stable_promotion_decision = load_json(stable_promotion_decision_path)
        except Exception as exc:  # pragma: no cover - defensive contract path
            stable_promotion_decision = {
                "schema_version": None,
                "status": "invalid_json",
                "summary": {"detail": str(exc)},
            }
    source_semantics_boundary = None
    if source_semantics_boundary_path.exists():
        try:
            source_semantics_boundary = load_json(source_semantics_boundary_path)
        except Exception as exc:  # pragma: no cover - defensive contract path
            source_semantics_boundary = {
                "schema_version": None,
                "status": "invalid_json",
                "summary": {"detail": str(exc)},
            }

    benchmark_artifact_path = _benchmark_artifact_path(bundle, bundle_dir=bundle_dir)
    benchmark_artifact = _safe_load(
        benchmark_artifact_path, label="benchmark_artifact", failures=failures
    )

    for candidate in (
        _unexpected_schema_failure(
            artifact="promotion_evidence",
            expected=BUNDLE_SCHEMA_VERSION,
            payload=bundle,
        ),
        _unexpected_schema_failure(
            artifact="promotion_evidence_check",
            expected=CHECK_SCHEMA_VERSION,
            payload=check,
        ),
        _unexpected_schema_failure(
            artifact="promotion_report",
            expected=PROMOTION_REPORT_SCHEMA_VERSION,
            payload=report,
        ),
        _unexpected_schema_failure(
            artifact="stable_review_assessment",
            expected=STABLE_REVIEW_SCHEMA_VERSION,
            payload=assessment,
        ),
        _unexpected_schema_failure(
            artifact="stable_candidate_blocker_matrix",
            expected=BLOCKER_MATRIX_SCHEMA_VERSION,
            payload=blocker_matrix,
        ),
        _unexpected_schema_failure(
            artifact="stable_promotion_decision",
            expected=STABLE_PROMOTION_DECISION_SCHEMA_VERSION,
            payload=stable_promotion_decision,
        ),
        _unexpected_schema_failure(
            artifact="benchmark_artifact",
            expected="nextstat_apex2_simplified_likelihood_report_v0",
            payload=benchmark_artifact,
        ),
    ):
        if candidate is not None:
            failures.append(candidate)

    check_status = None
    check_ok = False
    if isinstance(check, dict):
        check_status = check.get("status")
        check_ok = bool(check.get("ok", False))
        if check_status != "passed" or not check_ok:
            failures.append(
                {
                    "reason": "bundle_check_not_passed",
                    "status": check_status,
                    "ok": check_ok,
                }
            )

    report_status = None
    if isinstance(report, dict):
        report_status = report.get("status")
        if report_status != "promoted" or not bool(report.get("promoted", False)):
            failures.append(
                {
                    "reason": "promotion_report_not_promoted",
                    "status": report_status,
                    "promoted": bool(report.get("promoted", False)),
                }
            )

    stable_review = (
        assessment.get("stable_review")
        if isinstance(assessment, dict) and isinstance(assessment.get("stable_review"), dict)
        else {}
    )
    stable_review = stable_review if isinstance(stable_review, dict) else {}
    stable_review_claims = (
        stable_review.get("reviewed_claims")
        if isinstance(stable_review.get("reviewed_claims"), dict)
        else {}
    )
    stable_review_claims = (
        stable_review_claims if isinstance(stable_review_claims, dict) else {}
    )
    if isinstance(assessment, dict):
        if stable_review.get("ready") is not True or stable_review.get("status") != "review_ready":
            failures.append(
                {
                    "reason": "stable_review_not_ready",
                    "status": stable_review.get("status"),
                    "ready": stable_review.get("ready"),
                }
            )

    foundation = (
        blocker_matrix.get("foundation")
        if isinstance(blocker_matrix, dict) and isinstance(blocker_matrix.get("foundation"), dict)
        else {}
    )
    foundation = foundation if isinstance(foundation, dict) else {}
    if isinstance(blocker_matrix, dict) and foundation.get("passed") is not True:
        failures.append(
            {
                "reason": "stable_candidate_foundation_not_passed",
                "status": foundation.get("status"),
            }
        )

    benchmark_summary = (
        benchmark_artifact.get("summary")
        if isinstance(benchmark_artifact, dict)
        and isinstance(benchmark_artifact.get("summary"), dict)
        else {}
    )
    benchmark_summary = benchmark_summary if isinstance(benchmark_summary, dict) else {}
    export_summary = (
        benchmark_artifact.get("export_matrix", {}).get("summary")
        if isinstance(benchmark_artifact, dict)
        and isinstance(benchmark_artifact.get("export_matrix"), dict)
        and isinstance(benchmark_artifact.get("export_matrix", {}).get("summary"), dict)
        else {}
    )
    export_summary = export_summary if isinstance(export_summary, dict) else {}
    matrix_summary = (
        blocker_matrix.get("summary")
        if isinstance(blocker_matrix, dict) and isinstance(blocker_matrix.get("summary"), dict)
        else {}
    )
    matrix_summary = matrix_summary if isinstance(matrix_summary, dict) else {}
    stable_promotion_decision_ok = bool(
        isinstance(stable_promotion_decision, dict)
        and stable_promotion_decision.get("schema_version")
        == STABLE_PROMOTION_DECISION_SCHEMA_VERSION
        and stable_promotion_decision.get("status") == "accepted"
        and stable_promotion_decision.get("support_class") == "stable"
        and stable_promotion_decision.get("automatic_stable_promotion") is False
    )

    matrix_blockers = (
        blocker_matrix.get("blockers")
        if isinstance(blocker_matrix, dict) and isinstance(blocker_matrix.get("blockers"), list)
        else []
    )
    matrix_blockers = matrix_blockers if isinstance(matrix_blockers, list) else []
    remaining_blockers = [
        blocker for blocker in matrix_blockers if isinstance(blocker, dict) and blocker.get("status") == "open"
    ]
    resolved_blockers = [
        blocker
        for blocker in matrix_blockers
        if isinstance(blocker, dict) and blocker.get("status") == "resolved"
    ]

    benchmark_host = None
    if isinstance(matrix_summary.get("benchmark_host"), str):
        benchmark_host = str(matrix_summary.get("benchmark_host"))
    elif isinstance(stable_review_claims.get("benchmark_host"), str):
        benchmark_host = str(stable_review_claims.get("benchmark_host"))
    elif isinstance(benchmark_summary.get("bench"), dict):
        benchmark_host = REQUIRED_BENCHMARK_HOST
    elif isinstance(benchmark_artifact, dict) and isinstance(benchmark_artifact.get("environment"), dict):
        environment = benchmark_artifact.get("environment")
        if isinstance(environment.get("hostname"), str):
            benchmark_host = str(environment.get("hostname"))

    packet_valid = len(failures) == 0
    review_packet_status = "ready" if packet_valid else "incomplete"
    open_blocker_count = len(remaining_blockers)

    source_artifacts = {
        "promotion_evidence": _artifact_summary(bundle_path, bundle),
        "promotion_evidence_check": _artifact_summary(check_path, check),
        "promotion_report": _artifact_summary(report_path, report),
        "stable_review_assessment": _artifact_summary(assessment_path, assessment),
        "stable_candidate_blocker_matrix": _artifact_summary(blocker_matrix_path, blocker_matrix),
        "stable_source_semantics_boundary": _artifact_summary(
            source_semantics_boundary_path, source_semantics_boundary
        ),
        "stable_promotion_decision": _artifact_summary(
            stable_promotion_decision_path,
            stable_promotion_decision,
            include_sha256=False,
        ),
        "benchmark_artifact": _artifact_summary(benchmark_artifact_path, benchmark_artifact),
    }

    support_class = "stable" if stable_promotion_decision_ok and open_blocker_count == 0 else "research-grade"
    if not packet_valid:
        recommendation_status = "insufficient_evidence"
    elif open_blocker_count > 0:
        recommendation_status = "hold_research_grade"
    elif stable_promotion_decision_ok:
        recommendation_status = "stable_promoted"
    else:
        recommendation_status = "stable_promotion_candidate"
    recommended_support_class = "stable" if recommendation_status in {"stable_promotion_candidate", "stable_promoted"} else "research-grade"
    next_action = (
        str(matrix_summary.get("next_action"))
        if isinstance(matrix_summary.get("next_action"), str)
        else "repair missing or invalid stable-candidate review inputs"
    )
    if packet_valid and open_blocker_count == 0 and stable_promotion_decision_ok:
        next_action = (
            "keep the accepted narrow stable exporter subset aligned and retain "
            "wider fallback modes as research-grade"
        )
    maintainer_recommendation = (
        "maintain research-grade support class and repair review inputs before any stable discussion"
        if not packet_valid
        else (
            "maintain research-grade support class and close the remaining stable blockers before any promotion decision"
            if open_blocker_count > 0
            else (
                "the narrow stable exporter subset is explicitly promoted; maintain the release-facing evidence and keep wider fallback modes research-grade"
                if stable_promotion_decision_ok
                else "explicit stable-promotion review may proceed"
            )
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "surface": "simplified_likelihood_exporter",
        "support_class": support_class,
        "automatic_stable_promotion": False,
        "generated_at_utc": now_utc(deterministic),
        "bundle_dir": relative_or_absolute(bundle_dir),
        "source_artifacts": source_artifacts,
        "review_documents": {
            "review_packet_doc": REVIEW_PACKET_DOC,
            "acceptance_doc": ACCEPTANCE_DOC,
            "runtime_gate_doc": RUNTIME_GATE_DOC,
            "promotion_runbook_doc": PROMOTION_RUNBOOK_DOC,
            "stable_review_checklist_doc": STABLE_REVIEW_CHECKLIST_DOC,
            "blocker_matrix_doc": BLOCKER_MATRIX_DOC,
            "stable_source_semantics_boundary_doc": SOURCE_SEMANTICS_BOUNDARY_DOC,
            "stable_promotion_decision_doc": STABLE_PROMOTION_DECISION_DOC,
            "stable_promotion_release_pr_checklist_doc": STABLE_PROMOTION_RELEASE_PR_CHECKLIST_DOC,
            "artifact_reference_doc": ARTIFACT_REFERENCE_DOC,
        },
        "packet_validity": {
            "passed": packet_valid,
            "status": "passed" if packet_valid else "failed",
            "failures": failures,
            "warnings": [],
        },
        "evidence_snapshot": {
            "benchmark_host": benchmark_host,
            "future_stable_review_ready": bool(
                stable_review_claims.get("future_stable_review_ready", False)
            ),
            "stable_review_status": stable_review.get("status"),
            "stable_candidate_status": matrix_summary.get("status"),
            "public_reinterpretation_style_case_count": int(
                export_summary.get(
                    "public_reinterpretation_style_case_count",
                    benchmark_summary.get(
                        "export_matrix_public_reinterpretation_style_case_count", 0
                    ),
                )
            ),
            "max_abs_q_mu_diff": stable_review_claims.get("max_abs_q_mu_diff"),
            "max_upper_limit_ratio_deviation": stable_review_claims.get(
                "max_upper_limit_ratio_deviation"
            ),
            "min_net_end_to_end_upper_limit_speedup": stable_review_claims.get(
                "min_net_end_to_end_upper_limit_speedup"
            ),
        },
        "review_packet": {
            "ready": packet_valid,
            "status": review_packet_status,
            "recommendation_status": recommendation_status,
            "recommended_support_class": recommended_support_class,
            "target_support_class": "stable",
            "open_blocker_count": open_blocker_count,
            "resolved_blocker_count": len(resolved_blockers),
            "next_action": next_action,
            "maintainer_recommendation": maintainer_recommendation,
        },
        "remaining_blockers": remaining_blockers,
        "summary": {
            "status": review_packet_status,
            "benchmark_host": benchmark_host,
            "review_packet_ready": packet_valid,
            "recommendation_status": recommendation_status,
            "open_blocker_count": open_blocker_count,
            "next_action": next_action,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_ACCEPTED_BUNDLE_DIR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    packet = build_review_packet(
        bundle_dir=args.bundle_dir.resolve(),
        deterministic=bool(args.deterministic),
    )
    _write_json(args.out.resolve(), packet)
    print(
        "Exporter stable-candidate review packet:",
        f"status={packet['summary']['status']}",
        f"recommendation={packet['summary']['recommendation_status']}",
        f"host={packet['summary']['benchmark_host']}",
        f"out={relative_or_absolute(args.out.resolve())}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
