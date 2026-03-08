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
    REQUIRED_MIN_PUBLIC_EXPORT_MATRIX_CASE_COUNT,
    load_json,
    now_utc,
    relative_or_absolute,
    sha256_path,
)
from _simplified_likelihood_exporter_stable_candidate import (
    ACCEPTANCE_DOC,
    ARTIFACT_REFERENCE_DOC,
    BLOCKER_MATRIX_DOC,
    BLOCKER_MATRIX_SCHEMA_VERSION as SCHEMA_VERSION,
    OPEN_BLOCKERS,
    PROMOTION_RUNBOOK_DOC,
    REQUIRED_MAX_ABS_Q_MU_DIFF,
    REQUIRED_MAX_UPPER_LIMIT_RATIO_DEVIATION,
    REQUIRED_MIN_NET_END_TO_END_UPPER_LIMIT_SPEEDUP,
    REVIEW_PACKET_DOC,
    REVIEW_PACKET_SCHEMA_VERSION,
    RUNTIME_GATE_DOC,
    SOURCE_SEMANTICS_BOUNDARY_DOC,
    SOURCE_SEMANTICS_BOUNDARY_SCHEMA_VERSION,
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


def _foundation_check(
    *,
    check_id: str,
    title: str,
    satisfied: bool,
    detail: str,
    evidence_refs: list[str],
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "title": title,
        "satisfied": satisfied,
        "status": "satisfied" if satisfied else "missing",
        "detail": detail,
        "evidence_refs": evidence_refs,
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


def build_blocker_matrix(*, bundle_dir: Path, deterministic: bool) -> dict[str, Any]:
    load_failures: list[dict[str, Any]] = []

    bundle_path = bundle_dir / "promotion_evidence.json"
    check_path = bundle_dir / "promotion_evidence_check.json"
    report_path = bundle_dir / "promotion_bundle_promotion_report.json"
    assessment_path = bundle_dir / "stable_review_assessment.json"
    review_packet_path = bundle_dir / "stable_candidate_review_packet.json"
    source_semantics_boundary_path = bundle_dir / "stable_source_semantics_boundary.json"
    stable_promotion_decision_path = bundle_dir / "stable_promotion_decision.json"

    bundle = _safe_load(bundle_path, label="promotion_evidence", failures=load_failures)
    check = _safe_load(check_path, label="promotion_evidence_check", failures=load_failures)
    report = _safe_load(report_path, label="promotion_report", failures=load_failures)
    assessment = _safe_load(
        assessment_path, label="stable_review_assessment", failures=load_failures
    )
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
    review_packet = None
    if review_packet_path.exists():
        try:
            review_packet = load_json(review_packet_path)
        except Exception as exc:  # pragma: no cover - defensive contract path
            review_packet = {
                "schema_version": None,
                "status": "invalid_json",
                "summary": {"detail": str(exc)},
            }
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

    benchmark_artifact_path = _benchmark_artifact_path(bundle, bundle_dir=bundle_dir)
    benchmark_artifact = _safe_load(
        benchmark_artifact_path, label="benchmark_artifact", failures=load_failures
    )

    source_artifacts = {
        "promotion_evidence": _artifact_summary(bundle_path, bundle),
        "promotion_evidence_check": _artifact_summary(check_path, check),
        "promotion_report": _artifact_summary(report_path, report),
        "stable_review_assessment": _artifact_summary(assessment_path, assessment),
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

    bundle_ok = isinstance(bundle, dict) and bundle.get("schema_version") == BUNDLE_SCHEMA_VERSION
    check_ok = isinstance(check, dict) and check.get("schema_version") == CHECK_SCHEMA_VERSION
    report_ok = isinstance(report, dict) and report.get("schema_version") == PROMOTION_REPORT_SCHEMA_VERSION
    assessment_ok = (
        isinstance(assessment, dict)
        and assessment.get("schema_version") == STABLE_REVIEW_SCHEMA_VERSION
    )
    source_semantics_boundary_ok = (
        isinstance(source_semantics_boundary, dict)
        and source_semantics_boundary.get("schema_version")
        == SOURCE_SEMANTICS_BOUNDARY_SCHEMA_VERSION
        and source_semantics_boundary.get("status") == "published"
        and source_semantics_boundary.get("support_class") == "research-grade"
        and source_semantics_boundary.get("target_support_class") == "stable"
    )
    stable_promotion_decision_ok = (
        isinstance(stable_promotion_decision, dict)
        and stable_promotion_decision.get("schema_version")
        == STABLE_PROMOTION_DECISION_SCHEMA_VERSION
        and stable_promotion_decision.get("status") == "accepted"
        and stable_promotion_decision.get("support_class") == "stable"
        and stable_promotion_decision.get("automatic_stable_promotion") is False
    )

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
    export_cases = (
        benchmark_artifact.get("export_matrix", {}).get("cases")
        if isinstance(benchmark_artifact, dict)
        and isinstance(benchmark_artifact.get("export_matrix"), dict)
        and isinstance(benchmark_artifact.get("export_matrix", {}).get("cases"), list)
        else []
    )
    export_cases = export_cases if isinstance(export_cases, list) else []
    export_case_names = []
    export_case_kinds = set()
    public_export_case_names = []
    for case in export_cases:
        if not isinstance(case, dict):
            continue
        name = case.get("name")
        if isinstance(name, str):
            export_case_names.append(name)
        case_kind = case.get("case_kind")
        if isinstance(case_kind, str):
            export_case_kinds.add(case_kind)
            if case_kind == "public_reinterpretation_style" and isinstance(name, str):
                public_export_case_names.append(name)

    benchmark_host = None
    if isinstance(benchmark_summary.get("bench"), dict):
        benchmark_host = REQUIRED_BENCHMARK_HOST
    benchmark_host = (
        str(bundle_summary.get("benchmark_host"))
        if isinstance(bundle_summary.get("benchmark_host"), str)
        else benchmark_host
    )
    if isinstance(check_readiness.get("actual_benchmark_host"), str):
        benchmark_host = str(check_readiness.get("actual_benchmark_host"))
    if isinstance(assessment, dict) and isinstance(assessment.get("summary"), dict):
        assessment_summary = assessment["summary"]
        if isinstance(assessment_summary.get("benchmark_host"), str):
            benchmark_host = str(assessment_summary.get("benchmark_host"))

    foundation_checks = [
        _foundation_check(
            check_id="accepted_bundle_integrity",
            title="Accepted exporter bundle remains internally consistent",
            satisfied=bool(
                bundle_ok
                and check_ok
                and report_ok
                and bundle_summary.get("status") == "ok"
                and check.get("status") == "passed"  # type: ignore[union-attr]
                and bool(check.get("ok", False))  # type: ignore[union-attr]
                and report.get("status") == "promoted"  # type: ignore[union-attr]
                and bool(report.get("promoted", False))  # type: ignore[union-attr]
            ),
            detail=(
                "promotion_evidence, promotion_evidence_check, and promotion report "
                "all exist with the expected schema versions and passing statuses"
            ),
            evidence_refs=[
                relative_or_absolute(bundle_path),
                relative_or_absolute(check_path),
                relative_or_absolute(report_path),
            ],
        ),
        _foundation_check(
            check_id="stable_review_ready",
            title="Accepted exporter bundle is already formal stable-review ready",
            satisfied=bool(
                assessment_ok
                and assessment.get("support_class") == "research-grade"  # type: ignore[union-attr]
                and assessment.get("automatic_stable_promotion") is False  # type: ignore[union-attr]
                and stable_review.get("ready") is True
                and stable_review.get("status") == "review_ready"
            ),
            detail=(
                "stable_review_assessment.json exists, validates to the expected "
                "contract, and remains review_ready without automatic promotion"
            ),
            evidence_refs=[relative_or_absolute(assessment_path)],
        ),
        _foundation_check(
            check_id="committed_nextstat_bench_snapshot",
            title="Committed nextstat-bench exporter snapshot remains valid",
            satisfied=bool(
                isinstance(benchmark_artifact, dict)
                and benchmark_artifact.get("schema_version")
                == "nextstat_apex2_simplified_likelihood_report_v0"
                and benchmark_summary.get("status") == "ok"
                and benchmark_summary.get("export_matrix_included") is True
                and export_summary.get("status") == "ok"
                and float(
                    check_readiness.get(
                        "actual_min_net_end_to_end_upper_limit_speedup",
                        export_summary.get("min_net_end_to_end_upper_limit_speedup", 0.0),
                    )
                )
                >= REQUIRED_MIN_NET_END_TO_END_UPPER_LIMIT_SPEEDUP
                and float(
                    check_readiness.get(
                        "actual_max_abs_q_mu_diff",
                        export_summary.get("max_abs_q_mu_diff", 999.0),
                    )
                )
                <= REQUIRED_MAX_ABS_Q_MU_DIFF
                and float(
                    check_readiness.get(
                        "actual_max_upper_limit_ratio_deviation",
                        export_summary.get("max_upper_limit_ratio_deviation", 999.0),
                    )
                )
                <= REQUIRED_MAX_UPPER_LIMIT_RATIO_DEVIATION
            ),
            detail=(
                "the committed nextstat-bench exporter matrix remains green for "
                "schema, fidelity, and net end-to-end speedup thresholds"
            ),
            evidence_refs=[relative_or_absolute(benchmark_artifact_path)],
        ),
        _foundation_check(
            check_id="governance_boundary_explicit",
            title="Exporter governance boundary is still explicit rather than implicit",
            satisfied=bool(
                isinstance(bundle, dict)
                and bundle.get("support_class") == "research-grade"
                and assessment_ok
                and assessment.get("support_class") == "research-grade"  # type: ignore[union-attr]
                and assessment.get("automatic_stable_promotion") is False  # type: ignore[union-attr]
                and (
                    stable_promotion_decision is None or stable_promotion_decision_ok
                )
            ),
            detail=(
                "the accepted evidence keeps automatic stable promotion disabled, "
                "and any explicit stable claim must remain versioned through "
                "stable_promotion_decision.json"
            ),
            evidence_refs=[
                relative_or_absolute(bundle_path),
                relative_or_absolute(assessment_path),
                relative_or_absolute(stable_promotion_decision_path),
            ],
        ),
    ]

    foundation_passed = not load_failures and all(check["satisfied"] for check in foundation_checks)

    blockers: list[dict[str, Any]] = []
    for template in OPEN_BLOCKERS:
        blocker_id = str(template["blocker_id"])
        status = "open"
        current_state = ""
        evidence_refs = [BLOCKER_MATRIX_DOC, ACCEPTANCE_DOC, RUNTIME_GATE_DOC]

        if blocker_id == "public_exporter_matrix_not_yet_part_of_stable_candidate_evidence":
            if len(public_export_case_names) >= REQUIRED_MIN_PUBLIC_EXPORT_MATRIX_CASE_COUNT:
                status = "resolved"
                current_state = (
                    "Committed exporter evidence already includes machine-classified public export cases: "
                    + ", ".join(public_export_case_names)
                )
            else:
                current_state = (
                    "Committed exporter matrix still lacks the required three-case public stable-evidence lane. "
                    f"Observed public cases={len(public_export_case_names)}; case kinds="
                    + ", ".join(sorted(export_case_kinds) or ["<none>"])
                    + "; cases="
                    + ", ".join(export_case_names or ["<none>"])
                )
            evidence_refs = [relative_or_absolute(benchmark_artifact_path), ACCEPTANCE_DOC, BLOCKER_MATRIX_DOC]
        elif blocker_id == "stable_source_semantics_boundary_not_yet_promoted":
            if source_semantics_boundary_ok:
                status = "resolved"
                current_state = (
                    "Accepted bundle now publishes a versioned stable-source-semantics "
                    "boundary artifact for the future stable exporter claim: pyhf-only, "
                    "single-POI, Gaussian-constrained source_model_constraints scope, "
                    "with reduced-coordinate rather than source-level nuisance semantics."
                )
            else:
                current_state = (
                    "Public docs still lack a committed versioned boundary artifact for the "
                    "future stable exporter claim, including Gaussian-only source "
                    "constraints and reduced/source identity caveats."
                )
            evidence_refs = [
                relative_or_absolute(source_semantics_boundary_path),
                SOURCE_SEMANTICS_BOUNDARY_DOC,
                ACCEPTANCE_DOC,
                RUNTIME_GATE_DOC,
                BLOCKER_MATRIX_DOC,
            ]
        elif blocker_id == "stable_candidate_review_packet_not_yet_published":
            review_packet_ready = bool(
                isinstance(review_packet, dict)
                and review_packet.get("schema_version") == REVIEW_PACKET_SCHEMA_VERSION
                and isinstance(review_packet.get("review_packet"), dict)
                and review_packet.get("review_packet", {}).get("ready") is True
            )
            if review_packet_ready:
                status = "resolved"
                current_state = (
                    "Accepted bundle now commits a dedicated stable-candidate review packet "
                    "artifact with maintainer recommendation and blocker summary."
                )
            else:
                current_state = (
                    "Accepted bundle contains promotion evidence, check, report, and stable review, "
                    "but no dedicated stable-candidate review packet artifact is committed."
                )
            evidence_refs = [
                relative_or_absolute(review_packet_path),
                REVIEW_PACKET_DOC,
                BLOCKER_MATRIX_DOC,
            ]
        elif blocker_id == "stable_release_promotion_decision_not_yet_taken":
            if stable_promotion_decision_ok:
                status = "resolved"
                current_state = (
                    "Accepted bundle now commits an explicit narrow stable-promotion "
                    "decision artifact and the release-facing governance consumes it."
                )
            else:
                current_state = (
                    "Exporter governance explicitly keeps automatic stable promotion disabled, "
                    "and release-facing stable promotion has not yet been taken through the "
                    "accepted decision artifact."
                )
            evidence_refs = [
                relative_or_absolute(stable_promotion_decision_path),
                STABLE_PROMOTION_DECISION_DOC,
                STABLE_PROMOTION_RELEASE_PR_CHECKLIST_DOC,
                STABLE_REVIEW_CHECKLIST_DOC,
                RUNTIME_GATE_DOC,
            ]

        blockers.append(
            {
                "blocker_id": blocker_id,
                "title": str(template["title"]),
                "category": str(template["category"]),
                "status": status,
                "blocking": status == "open",
                "current_state": current_state,
                "why_blocking_stable": str(template["why_blocking_stable"]),
                "exit_criteria": list(template["exit_criteria"]),
                "evidence_refs": evidence_refs,
            }
        )

    open_blocker_count = sum(1 for blocker in blockers if blocker["status"] == "open")
    stable_candidate_ready = foundation_passed and open_blocker_count == 0
    stable_candidate_status = "ready" if stable_candidate_ready else "blocked"
    support_class = "stable" if stable_promotion_decision_ok and stable_candidate_ready else "research-grade"

    next_action = (
        "close the remaining open blockers before promoting any exporter surface to stable"
        if open_blocker_count > 0
        else "keep the accepted narrow stable exporter subset aligned and retain wider fallback modes as research-grade"
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
            "blocker_matrix_doc": BLOCKER_MATRIX_DOC,
            "review_packet_doc": REVIEW_PACKET_DOC,
            "stable_source_semantics_boundary_doc": SOURCE_SEMANTICS_BOUNDARY_DOC,
            "stable_promotion_decision_doc": STABLE_PROMOTION_DECISION_DOC,
            "stable_promotion_release_pr_checklist_doc": STABLE_PROMOTION_RELEASE_PR_CHECKLIST_DOC,
            "acceptance_doc": ACCEPTANCE_DOC,
            "runtime_gate_doc": RUNTIME_GATE_DOC,
            "promotion_runbook_doc": PROMOTION_RUNBOOK_DOC,
            "stable_review_checklist_doc": STABLE_REVIEW_CHECKLIST_DOC,
            "artifact_reference_doc": ARTIFACT_REFERENCE_DOC,
        },
        "foundation": {
            "passed": foundation_passed,
            "status": "passed" if foundation_passed else "failed",
            "checks": foundation_checks,
            "failures": load_failures,
        },
        "stable_candidate": {
            "ready": stable_candidate_ready,
            "status": stable_candidate_status,
            "target_support_class": "stable",
            "automatic_stable_promotion": False,
            "open_blocker_count": open_blocker_count,
            "resolved_foundation_count": sum(
                1 for check in foundation_checks if bool(check["satisfied"])
            ),
            "next_action": next_action,
        },
        "blockers": blockers,
        "summary": {
            "status": stable_candidate_status,
            "benchmark_host": benchmark_host,
            "foundation_passed": foundation_passed,
            "open_blocker_count": open_blocker_count,
            "automatic_stable_promotion": False,
            "next_action": next_action,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_ACCEPTED_BUNDLE_DIR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    assessment = build_blocker_matrix(
        bundle_dir=args.bundle_dir.resolve(),
        deterministic=bool(args.deterministic),
    )
    _write_json(args.out.resolve(), assessment)
    print(
        "Exporter stable-candidate blocker matrix:",
        f"status={assessment['summary']['status']}",
        f"host={assessment['summary']['benchmark_host']}",
        f"out={relative_or_absolute(args.out.resolve())}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
