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
    REQUIRED_BENCHMARK_HOST,
    load_json,
    now_utc,
    relative_or_absolute,
    sha256_path,
)
from _simplified_likelihood_exporter_stable_candidate import (
    BLOCKER_MATRIX_SCHEMA_VERSION,
    REVIEW_PACKET_SCHEMA_VERSION,
)
from _simplified_likelihood_exporter_stable_promotion import (
    ALIGNMENT_DOCUMENTS,
    DECISION_SCHEMA_VERSION as SCHEMA_VERSION,
    PROMOTED_COMMAND_NAMES,
    PROMOTED_SCHEMA_NAMES,
    RELEASE_ARTIFACT_FILENAMES,
    RELEASE_ARTIFACT_NAME,
    RELEASE_JOB_NAME,
    RELEASE_PR_CHECKLIST_DOC,
    RELEASE_WORKFLOW_PATH,
    RESEARCH_GRADE_EXCEPTIONS,
    STABLE_SCOPE,
    STANDALONE_WORKFLOW_PATH,
)
from _simplified_likelihood_exporter_stable_review import (
    ASSESSMENT_SCHEMA_VERSION as STABLE_REVIEW_SCHEMA_VERSION,
)
from _simplified_likelihood_exporter_stable_source_semantics import (
    SCHEMA_VERSION as SOURCE_SEMANTICS_BOUNDARY_SCHEMA_VERSION,
)


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


def _contains_all(text: str, required: list[str]) -> bool:
    return all(item in text for item in required)


def _workflow_consumption(
    *, release_workflow_path: Path, standalone_workflow_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    release_text = release_workflow_path.read_text(encoding="utf-8")
    # release.yml delegates to release-candidate.yml via workflow_call;
    # include its text so job/artifact references are visible.
    rc_path = release_workflow_path.parent / "release-candidate.yml"
    if rc_path.is_file():
        release_text += "\n" + rc_path.read_text(encoding="utf-8")
    stage_assets_path = release_workflow_path.parents[1] / "scripts" / "release_stage_assets.py"
    if stage_assets_path.is_file():
        release_text += "\n" + stage_assets_path.read_text(encoding="utf-8")
    standalone_text = standalone_workflow_path.read_text(encoding="utf-8")

    release_required = [
        "simplified-likelihood-exporter-surface:",
        f"name: {RELEASE_JOB_NAME}",
        RELEASE_ARTIFACT_NAME,
        "python3 -m scripts.release_stage_assets --dist-root dist --out-dir dist/release-assets",
        "files: dist/release-assets/*",
    ] + list(RELEASE_ARTIFACT_FILENAMES)
    standalone_required = [
        "name: Simplified Likelihood Exporter Surface",
        "Upload simplified-likelihood exporter artifacts",
        "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_policy.json",
        "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_promotion_decision.json",
    ]

    release_assets_wired = _contains_all(release_text, release_required)
    standalone_upload_present = _contains_all(standalone_text, standalone_required)

    if not release_assets_wired:
        failures.append(
            {
                "reason": "release_workflow_missing_exporter_stable_promotion_artifacts",
                "path": relative_or_absolute(release_workflow_path),
            }
        )
    if not standalone_upload_present:
        failures.append(
            {
                "reason": "standalone_workflow_missing_exporter_stable_promotion_artifacts",
                "path": relative_or_absolute(standalone_workflow_path),
            }
        )

    return (
        {
            "release_workflow_path": relative_or_absolute(release_workflow_path),
            "standalone_workflow_path": relative_or_absolute(standalone_workflow_path),
            "release_job_name": RELEASE_JOB_NAME,
            "release_artifact_name": RELEASE_ARTIFACT_NAME,
            "release_assets_wired": release_assets_wired,
            "standalone_upload_present": standalone_upload_present,
            "release_artifact_filenames": list(RELEASE_ARTIFACT_FILENAMES),
        },
        failures,
    )


def _checklist_consumption(checklist_path: Path) -> tuple[bool, list[dict[str, Any]]]:
    failures: list[dict[str, Any]] = []
    checklist_text = checklist_path.read_text(encoding="utf-8")
    required = [
        "stable_promotion_decision.json",
        "promotion_evidence.json",
        "promotion_evidence_check.json",
        "stable_candidate_blocker_matrix.json",
        "stable_candidate_review_packet.json",
        "stable_evidence_policy.json",
        RELEASE_WORKFLOW_PATH,
    ]
    ok = _contains_all(checklist_text, required)
    if not ok:
        failures.append(
            {
                "reason": "release_pr_checklist_missing_exporter_stable_promotion_refs",
                "path": relative_or_absolute(checklist_path),
            }
        )
    return ok, failures


def build_stable_promotion_decision(*, bundle_dir: Path, deterministic: bool) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []

    bundle_path = bundle_dir / "promotion_evidence.json"
    check_path = bundle_dir / "promotion_evidence_check.json"
    report_path = bundle_dir / "promotion_bundle_promotion_report.json"
    assessment_path = bundle_dir / "stable_review_assessment.json"
    boundary_path = bundle_dir / "stable_source_semantics_boundary.json"
    blocker_matrix_path = bundle_dir / "stable_candidate_blocker_matrix.json"
    review_packet_path = bundle_dir / "stable_candidate_review_packet.json"

    bundle = _safe_load(bundle_path, label="promotion_evidence", failures=failures)
    check = _safe_load(check_path, label="promotion_evidence_check", failures=failures)
    report = _safe_load(report_path, label="promotion_report", failures=failures)
    assessment = _safe_load(
        assessment_path, label="stable_review_assessment", failures=failures
    )
    boundary = _safe_load(boundary_path, label="stable_source_semantics_boundary", failures=failures)
    blocker_matrix = _safe_load(
        blocker_matrix_path, label="stable_candidate_blocker_matrix", failures=failures
    )
    review_packet = _safe_load(
        review_packet_path, label="stable_candidate_review_packet", failures=failures
    )
    benchmark_artifact_path = _benchmark_artifact_path(bundle, bundle_dir=bundle_dir)
    benchmark_artifact = _safe_load(
        benchmark_artifact_path, label="benchmark_artifact", failures=failures
    )

    for candidate in (
        _unexpected_schema_failure(
            artifact="promotion_evidence", expected=BUNDLE_SCHEMA_VERSION, payload=bundle
        ),
        _unexpected_schema_failure(
            artifact="promotion_evidence_check", expected=CHECK_SCHEMA_VERSION, payload=check
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
            artifact="stable_source_semantics_boundary",
            expected=SOURCE_SEMANTICS_BOUNDARY_SCHEMA_VERSION,
            payload=boundary,
        ),
        _unexpected_schema_failure(
            artifact="stable_candidate_blocker_matrix",
            expected=BLOCKER_MATRIX_SCHEMA_VERSION,
            payload=blocker_matrix,
        ),
        _unexpected_schema_failure(
            artifact="stable_candidate_review_packet",
            expected=REVIEW_PACKET_SCHEMA_VERSION,
            payload=review_packet,
        ),
        _unexpected_schema_failure(
            artifact="benchmark_artifact",
            expected="nextstat_apex2_simplified_likelihood_report_v0",
            payload=benchmark_artifact,
        ),
    ):
        if candidate is not None:
            failures.append(candidate)

    bundle_summary = bundle.get("summary") if isinstance(bundle, dict) and isinstance(bundle.get("summary"), dict) else {}
    bundle_summary = bundle_summary if isinstance(bundle_summary, dict) else {}
    check_readiness = (
        check.get("checks", {}).get("promotion_readiness")
        if isinstance(check, dict) and isinstance(check.get("checks"), dict)
        else {}
    )
    check_readiness = check_readiness if isinstance(check_readiness, dict) else {}
    stable_review = (
        assessment.get("stable_review")
        if isinstance(assessment, dict) and isinstance(assessment.get("stable_review"), dict)
        else {}
    )
    stable_review = stable_review if isinstance(stable_review, dict) else {}
    boundary_summary = (
        boundary.get("summary")
        if isinstance(boundary, dict) and isinstance(boundary.get("summary"), dict)
        else {}
    )
    boundary_summary = boundary_summary if isinstance(boundary_summary, dict) else {}
    matrix_summary = (
        blocker_matrix.get("summary")
        if isinstance(blocker_matrix, dict) and isinstance(blocker_matrix.get("summary"), dict)
        else {}
    )
    matrix_summary = matrix_summary if isinstance(matrix_summary, dict) else {}
    packet_summary = (
        review_packet.get("summary")
        if isinstance(review_packet, dict) and isinstance(review_packet.get("summary"), dict)
        else {}
    )
    packet_summary = packet_summary if isinstance(packet_summary, dict) else {}
    packet_body = (
        review_packet.get("review_packet")
        if isinstance(review_packet, dict) and isinstance(review_packet.get("review_packet"), dict)
        else {}
    )
    packet_body = packet_body if isinstance(packet_body, dict) else {}
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
    matrix_blockers = (
        blocker_matrix.get("blockers")
        if isinstance(blocker_matrix, dict) and isinstance(blocker_matrix.get("blockers"), list)
        else []
    )
    matrix_blockers = matrix_blockers if isinstance(matrix_blockers, list) else []
    open_blocker_ids = [
        str(blocker.get("blocker_id"))
        for blocker in matrix_blockers
        if isinstance(blocker, dict) and blocker.get("status") == "open"
    ]

    if bundle_summary.get("status") != "ok":
        failures.append({"reason": "promotion_bundle_not_ok", "status": bundle_summary.get("status")})
    if bundle.get("support_class") != "research-grade":  # type: ignore[union-attr]
        failures.append(
            {
                "reason": "unexpected_promotion_bundle_support_class",
                "support_class": bundle.get("support_class") if isinstance(bundle, dict) else None,
            }
        )
    if check.get("status") != "passed" or not bool(check.get("ok", False)):  # type: ignore[union-attr]
        failures.append(
            {
                "reason": "promotion_bundle_check_not_passed",
                "status": check.get("status") if isinstance(check, dict) else None,
                "ok": bool(check.get("ok", False)) if isinstance(check, dict) else False,
            }
        )
    if report.get("status") != "promoted" or not bool(report.get("promoted", False)):  # type: ignore[union-attr]
        failures.append(
            {
                "reason": "promotion_report_not_promoted",
                "status": report.get("status") if isinstance(report, dict) else None,
                "promoted": bool(report.get("promoted", False)) if isinstance(report, dict) else False,
            }
        )
    if (
        assessment.get("support_class") != "research-grade"  # type: ignore[union-attr]
        or assessment.get("automatic_stable_promotion") is not False  # type: ignore[union-attr]
        or stable_review.get("ready") is not True
        or stable_review.get("status") != "review_ready"
    ):
        failures.append(
            {
                "reason": "stable_review_assessment_not_review_ready",
                "status": stable_review.get("status"),
                "ready": stable_review.get("ready"),
            }
        )
    if (
        boundary.get("support_class") != "research-grade"  # type: ignore[union-attr]
        or boundary.get("target_support_class") != "stable"  # type: ignore[union-attr]
        or boundary.get("status") != "published"  # type: ignore[union-attr]
        or boundary_summary.get("status") != "published"
    ):
        failures.append(
            {
                "reason": "stable_source_semantics_boundary_not_published",
                "status": boundary.get("status") if isinstance(boundary, dict) else None,
            }
        )

    if matrix_summary.get("status") not in {"blocked", "ready"}:
        failures.append(
            {
                "reason": "stable_candidate_blocker_matrix_invalid_status",
                "status": matrix_summary.get("status"),
            }
        )
    allowed_predecision_blockers = {
        "stable_release_promotion_decision_not_yet_taken",
        # The first decision pass intentionally precedes the final review-packet
        # refresh, so this blocker may still be open until the decision artifact
        # exists and the packet is regenerated against it.
        "stable_candidate_review_packet_not_yet_published",
    }
    unexpected_open_blockers = [
        blocker_id
        for blocker_id in open_blocker_ids
        if blocker_id not in allowed_predecision_blockers
    ]
    if unexpected_open_blockers:
        failures.append(
            {
                "reason": "unexpected_open_stable_candidate_blockers",
                "open_blocker_ids": unexpected_open_blockers,
            }
        )
    if len(open_blocker_ids) > len(allowed_predecision_blockers):
        failures.append(
            {
                "reason": "too_many_open_stable_candidate_blockers",
                "open_blocker_ids": open_blocker_ids,
            }
        )

    if packet_summary.get("status") != "ready" or packet_body.get("ready") is not True:
        failures.append(
            {
                "reason": "stable_candidate_review_packet_not_ready",
                "status": packet_summary.get("status"),
            }
        )
    if packet_body.get("recommendation_status") not in {
        "hold_research_grade",
        "stable_promotion_candidate",
        "stable_promoted",
    }:
        failures.append(
            {
                "reason": "stable_candidate_review_packet_invalid_recommendation",
                "recommendation_status": packet_body.get("recommendation_status"),
            }
        )

    if benchmark_summary.get("status") != "ok" or export_summary.get("status") != "ok":
        failures.append(
            {
                "reason": "benchmark_artifact_not_ok",
                "status": benchmark_summary.get("status"),
                "export_status": export_summary.get("status"),
            }
        )
    if (
        bundle_summary.get("benchmark_host") != REQUIRED_BENCHMARK_HOST
        or check_readiness.get("actual_benchmark_host") != REQUIRED_BENCHMARK_HOST
    ):
        failures.append(
            {
                "reason": "unexpected_benchmark_host",
                "bundle_host": bundle_summary.get("benchmark_host"),
                "check_host": check_readiness.get("actual_benchmark_host"),
            }
        )

    release_consumption, release_failures = _workflow_consumption(
        release_workflow_path=REPO_ROOT / RELEASE_WORKFLOW_PATH,
        standalone_workflow_path=REPO_ROOT / STANDALONE_WORKFLOW_PATH,
    )
    failures.extend(release_failures)
    checklist_ok, checklist_failures = _checklist_consumption(REPO_ROOT / RELEASE_PR_CHECKLIST_DOC)
    failures.extend(checklist_failures)

    valid = len(failures) == 0
    status = "accepted" if valid else "incomplete"
    support_class = "stable" if valid else "research-grade"
    next_action = (
        "preserve the promoted narrow stable exporter subset through release gating and accepted evidence refreshes"
        if valid
        else "repair exporter stable-promotion prerequisites before treating the narrow exporter subset as stable"
    )

    source_artifacts = {
        "promotion_evidence": _artifact_summary(bundle_path, bundle),
        "promotion_evidence_check": _artifact_summary(check_path, check),
        "promotion_report": _artifact_summary(report_path, report),
        "stable_review_assessment": _artifact_summary(assessment_path, assessment),
        "stable_source_semantics_boundary": _artifact_summary(boundary_path, boundary),
        "stable_candidate_blocker_matrix": _artifact_summary(blocker_matrix_path, blocker_matrix),
        "stable_candidate_review_packet": _artifact_summary(review_packet_path, review_packet),
        "benchmark_artifact": _artifact_summary(benchmark_artifact_path, benchmark_artifact),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "surface": "simplified_likelihood_exporter",
        "support_class": support_class,
        "automatic_stable_promotion": False,
        "status": status,
        "generated_at_utc": now_utc(deterministic),
        "decision_date": "2026-03-09",
        "bundle_dir": relative_or_absolute(bundle_dir),
        "source_artifacts": source_artifacts,
        "alignment_documents": dict(ALIGNMENT_DOCUMENTS),
        "release_consumption": {
            **release_consumption,
            "release_pr_checklist_path": RELEASE_PR_CHECKLIST_DOC,
            "release_pr_checklist_verified": checklist_ok,
        },
        "decision": {
            "decision_status": status,
            "maintainer_decision": (
                "promote the narrow exporter subset to stable with explicit release-facing governance"
                if valid
                else "do not promote the exporter subset until the stable-promotion prerequisites are repaired"
            ),
            "benchmark_host": REQUIRED_BENCHMARK_HOST,
            "promoted_command_names": list(PROMOTED_COMMAND_NAMES),
            "promoted_schema_names": list(PROMOTED_SCHEMA_NAMES),
            "stable_scope": dict(STABLE_SCOPE),
            "research_grade_exceptions": list(RESEARCH_GRADE_EXCEPTIONS),
            "open_blocker_ids_observed": open_blocker_ids,
        },
        "validity": {
            "passed": valid,
            "status": "passed" if valid else "failed",
            "failures": failures,
            "warnings": [],
        },
        "summary": {
            "status": status,
            "benchmark_host": REQUIRED_BENCHMARK_HOST,
            "release_assets_wired": bool(release_consumption["release_assets_wired"]),
            "stable_scope_promoted": valid,
            "automatic_stable_promotion": False,
            "open_blocker_count_observed": len(open_blocker_ids),
            "next_action": next_action,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_ACCEPTED_BUNDLE_DIR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    decision = build_stable_promotion_decision(
        bundle_dir=args.bundle_dir.resolve(),
        deterministic=bool(args.deterministic),
    )
    out_path = args.out.resolve()
    _write_json(out_path, decision)
    print(
        "Exporter stable-promotion decision:",
        f"status={decision['status']}",
        f"support_class={decision['support_class']}",
        f"host={decision['summary']['benchmark_host']}",
        f"out={relative_or_absolute(out_path)}",
    )
    return 0 if decision["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
