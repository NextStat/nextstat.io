#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from _simplified_likelihood_exporter_promotion_bundle import (
    BUNDLE_SCHEMA_VERSION,
    CHECK_SCHEMA_VERSION,
    DEFAULT_ACCEPTED_BUNDLE_DIR,
    DEFAULT_ACCEPTED_HISTORY_DIR,
    PROMOTION_ARTIFACT_SUITE,
    PROMOTION_REPORT_SCHEMA_VERSION,
    REPO_ROOT,
    REQUIRED_BENCHMARK_HOST,
    derive_stamp_from_path,
    load_json,
    now_utc,
)


VERIFY_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "benchmarks"
    / "verify_simplified_likelihood_exporter_promotion_evidence_bundle.py"
)
STABLE_REVIEW_ASSESS_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "benchmarks"
    / "assess_simplified_likelihood_exporter_stable_review.py"
)
STABLE_SOURCE_SEMANTICS_BOUNDARY_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "benchmarks"
    / "build_simplified_likelihood_exporter_stable_source_semantics_boundary.py"
)
STABLE_CANDIDATE_BLOCKER_MATRIX_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "benchmarks"
    / "assess_simplified_likelihood_exporter_stable_candidate_blockers.py"
)
STABLE_CANDIDATE_REVIEW_PACKET_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "benchmarks"
    / "build_simplified_likelihood_exporter_stable_candidate_review_packet.py"
)
STABLE_PROMOTION_DECISION_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "benchmarks"
    / "assess_simplified_likelihood_exporter_stable_promotion_decision.py"
)
STABLE_EVIDENCE_POLICY_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "benchmarks"
    / "build_simplified_likelihood_exporter_stable_evidence_policy.py"
)
STABLE_EVIDENCE_FRESHNESS_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "benchmarks"
    / "build_simplified_likelihood_exporter_stable_evidence_freshness_report.py"
)
SNAPSHOT_INDEX_SCRIPT = REPO_ROOT / "scripts" / "benchmarks" / "write_snapshot_index.py"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _same_path(recorded: str, expected: Path) -> bool:
    candidate = Path(recorded)
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate == expected.resolve()


def _bundle_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    summary = bundle.get("summary") if isinstance(bundle.get("summary"), dict) else {}
    return {
        "schema_version": bundle.get("schema_version"),
        "status": summary.get("status"),
        "support_class": bundle.get("support_class"),
        "benchmark_host": summary.get("benchmark_host"),
        "supports_future_stable_review": bool(
            summary.get("supports_future_stable_review", False)
        ),
        "supports_committed_snapshot": bool(summary.get("supports_committed_snapshot", False)),
        "artifact_count": int(summary.get("artifact_count", 0)),
        "required_artifact_count": int(summary.get("required_artifact_count", 0)),
    }


def _check_summary(check: dict[str, Any]) -> dict[str, Any]:
    readiness = (
        check.get("checks", {}).get("promotion_readiness")
        if isinstance(check.get("checks"), dict)
        else {}
    )
    readiness = readiness if isinstance(readiness, dict) else {}
    summary = check.get("summary") if isinstance(check.get("summary"), dict) else {}
    return {
        "schema_version": check.get("schema_version"),
        "status": check.get("status"),
        "ok": bool(check.get("ok", False)),
        "require_promotion_ready": bool(check.get("require_promotion_ready", False)),
        "actual_benchmark_host": readiness.get("actual_benchmark_host"),
        "actual_min_net_end_to_end_upper_limit_speedup": float(
            readiness.get("actual_min_net_end_to_end_upper_limit_speedup", 0.0)
        ),
        "top_level_errors": list(summary.get("top_level_errors", [])),
    }


def _report(
    *,
    status: str,
    promoted: bool,
    dry_run: bool,
    deterministic: bool,
    source_bundle_dir: Path,
    source_bundle_json_path: Path,
    source_bundle_check_path: Path,
    benchmark_artifact_source_path: str,
    accepted_bundle_dir: Path,
    history_dir: Path,
    accepted_snapshot_index_path: Path,
    source_bundle_summary: dict[str, Any],
    source_check_summary: dict[str, Any],
    actions: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": PROMOTION_REPORT_SCHEMA_VERSION,
        "surface": "simplified_likelihood_exporter",
        "status": status,
        "promoted": promoted,
        "dry_run": dry_run,
        "generated_at_utc": now_utc(deterministic),
        "source_bundle_dir": str(source_bundle_dir),
        "source_bundle_json_path": str(source_bundle_json_path),
        "source_bundle_check_path": str(source_bundle_check_path),
        "benchmark_artifact_source_path": benchmark_artifact_source_path,
        "accepted_bundle_dir": str(accepted_bundle_dir),
        "accepted_snapshot_index_path": str(accepted_snapshot_index_path),
        "history_dir": str(history_dir),
        "source_bundle_summary": source_bundle_summary,
        "source_check_summary": source_check_summary,
        "actions": actions,
        "summary": {
            "top_level_errors": errors,
        },
    }


def _cleanup_stage(stage_dir: Path) -> None:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)


def _run_bundle_verify(bundle_dir: Path, *, deterministic: bool) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(VERIFY_SCRIPT),
        "--bundle-dir",
        str(bundle_dir),
        "--out",
        str(bundle_dir / "promotion_evidence_check.json"),
        "--require-promotion-ready",
    ]
    if deterministic:
        cmd.append("--deterministic")
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_snapshot_index(
    *,
    accepted_bundle_dir: Path,
    snapshot_index_path: Path,
    snapshot_id: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SNAPSHOT_INDEX_SCRIPT),
            "--suite",
            PROMOTION_ARTIFACT_SUITE,
            "--artifacts-dir",
            str(accepted_bundle_dir),
            "--out",
            str(snapshot_index_path),
            "--snapshot-id",
            snapshot_id,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_stable_review_assessment(
    bundle_dir: Path, *, deterministic: bool
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(STABLE_REVIEW_ASSESS_SCRIPT),
        "--bundle-dir",
        str(bundle_dir),
        "--out",
        str(bundle_dir / "stable_review_assessment.json"),
    ]
    if deterministic:
        cmd.append("--deterministic")
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_stable_source_semantics_boundary(
    bundle_dir: Path, *, deterministic: bool
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(STABLE_SOURCE_SEMANTICS_BOUNDARY_SCRIPT),
        "--out",
        str(bundle_dir / "stable_source_semantics_boundary.json"),
    ]
    if deterministic:
        cmd.append("--deterministic")
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_stable_candidate_blocker_matrix(
    bundle_dir: Path, *, deterministic: bool
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(STABLE_CANDIDATE_BLOCKER_MATRIX_SCRIPT),
        "--bundle-dir",
        str(bundle_dir),
        "--out",
        str(bundle_dir / "stable_candidate_blocker_matrix.json"),
    ]
    if deterministic:
        cmd.append("--deterministic")
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_stable_candidate_review_packet(
    bundle_dir: Path, *, deterministic: bool
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(STABLE_CANDIDATE_REVIEW_PACKET_SCRIPT),
        "--bundle-dir",
        str(bundle_dir),
        "--out",
        str(bundle_dir / "stable_candidate_review_packet.json"),
    ]
    if deterministic:
        cmd.append("--deterministic")
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_stable_promotion_decision(
    bundle_dir: Path, *, deterministic: bool
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(STABLE_PROMOTION_DECISION_SCRIPT),
        "--bundle-dir",
        str(bundle_dir),
        "--out",
        str(bundle_dir / "stable_promotion_decision.json"),
    ]
    if deterministic:
        cmd.append("--deterministic")
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_stable_evidence_policy(
    *,
    bundle_dir: Path,
    benchmark_artifact_path: Path,
    public_validation_report_path: Path,
    deterministic: bool,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(STABLE_EVIDENCE_POLICY_SCRIPT),
        "--benchmark-artifact",
        str(benchmark_artifact_path),
        "--public-validation-report",
        str(public_validation_report_path),
        "--stable-promotion-decision",
        str(bundle_dir / "stable_promotion_decision.json"),
        "--out",
        str(bundle_dir / "stable_evidence_policy.json"),
    ]
    if deterministic:
        cmd.append("--deterministic")
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_stable_evidence_freshness(
    *,
    bundle_dir: Path,
    snapshot_report_path: Path,
    public_validation_report_path: Path,
    deterministic: bool,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(STABLE_EVIDENCE_FRESHNESS_SCRIPT),
        "--snapshot-report",
        str(snapshot_report_path),
        "--public-validation-report",
        str(public_validation_report_path),
        "--stable-evidence-policy",
        str(bundle_dir / "stable_evidence_policy.json"),
        "--stable-promotion-decision",
        str(bundle_dir / "stable_promotion_decision.json"),
        "--out",
        str(bundle_dir / "stable_evidence_freshness_report.json"),
    ]
    if deterministic:
        cmd.append("--deterministic")
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bundle-dir", type=Path, required=True)
    parser.add_argument("--accepted-dir", type=Path, default=DEFAULT_ACCEPTED_BUNDLE_DIR)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_ACCEPTED_HISTORY_DIR)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--snapshot-id", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    source_bundle_dir = args.source_bundle_dir.resolve()
    accepted_bundle_dir = args.accepted_dir.resolve()
    history_dir = args.history_dir.resolve()
    report_path = (
        args.report.resolve()
        if args.report is not None
        else (source_bundle_dir / "promotion_bundle_promotion_report.json")
    )
    accepted_snapshot_index_path = accepted_bundle_dir / "snapshot_index.json"
    source_bundle_json_path = source_bundle_dir / "promotion_evidence.json"
    source_bundle_check_path = source_bundle_dir / "promotion_evidence_check.json"

    errors: list[str] = []
    source_bundle_summary = {
        "schema_version": None,
        "status": None,
        "support_class": None,
        "benchmark_host": None,
        "supports_future_stable_review": False,
        "supports_committed_snapshot": False,
        "artifact_count": 0,
        "required_artifact_count": 0,
    }
    source_check_summary = {
        "schema_version": None,
        "status": None,
        "ok": False,
        "require_promotion_ready": False,
        "actual_benchmark_host": None,
        "actual_min_net_end_to_end_upper_limit_speedup": 0.0,
        "top_level_errors": [],
    }
    actions = {
        "archived_previous_accepted": False,
        "archived_previous_accepted_path": None,
        "archived_promoted_bundle": False,
        "archived_promoted_bundle_path": None,
        "accepted_updated": False,
        "accepted_snapshot_index_written": False,
        "accepted_snapshot_index_path": None,
    }
    benchmark_artifact_source_path = ""
    benchmark_artifact_path = Path()
    public_validation_report_path = Path()
    snapshot_report_path = Path()

    if not source_bundle_dir.exists():
        errors.append(f"source_bundle_dir_missing:{source_bundle_dir}")
    elif not source_bundle_dir.is_dir():
        errors.append(f"source_bundle_dir_not_directory:{source_bundle_dir}")

    if not source_bundle_json_path.exists():
        errors.append(f"source_bundle_json_missing:{source_bundle_json_path}")
    if not source_bundle_check_path.exists():
        errors.append(f"source_bundle_check_missing:{source_bundle_check_path}")

    if not errors:
        source_bundle = load_json(source_bundle_json_path)
        source_check = load_json(source_bundle_check_path)
        source_bundle_summary = _bundle_summary(source_bundle)
        source_check_summary = _check_summary(source_check)

        source_snapshot = (
            source_bundle.get("source_snapshot")
            if isinstance(source_bundle.get("source_snapshot"), dict)
            else {}
        )
        benchmark_artifact = (
            source_snapshot.get("benchmark_artifact")
            if isinstance(source_snapshot.get("benchmark_artifact"), dict)
            else {}
        )
        benchmark_artifact_source_path = str(benchmark_artifact.get("source_path", ""))
        benchmark_artifact_path = Path(benchmark_artifact_source_path).resolve()
        snapshot_report_path = benchmark_artifact_path.parent / "export_benchmark_snapshot_report.json"
        public_validation_report_path = (
            benchmark_artifact_path.parent / "export_public_validation_report.json"
        )

        if source_bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            errors.append(f"unexpected_source_bundle_schema:{source_bundle.get('schema_version')}")
        if source_check.get("schema_version") != CHECK_SCHEMA_VERSION:
            errors.append(f"unexpected_source_check_schema:{source_check.get('schema_version')}")
        if source_bundle_summary["status"] != "ok":
            errors.append(f"source_bundle_not_ok:{source_bundle_summary['status']}")
        if source_bundle_summary["support_class"] != "research-grade":
            errors.append(
                f"unexpected_source_bundle_support_class:{source_bundle_summary['support_class']}"
            )
        if source_bundle_summary["benchmark_host"] != REQUIRED_BENCHMARK_HOST:
            errors.append(
                f"unexpected_source_bundle_host:{source_bundle_summary['benchmark_host']}"
            )
        if not source_bundle_summary["supports_future_stable_review"]:
            errors.append("source_bundle_not_future_review_ready")
        if not source_bundle_summary["supports_committed_snapshot"]:
            errors.append("source_bundle_not_committed_snapshot_supported")
        if source_check_summary["status"] != "passed":
            errors.append(f"source_check_not_passed:{source_check_summary['status']}")
        if not source_check_summary["ok"]:
            errors.append("source_check_not_ok")
        if not source_check_summary["require_promotion_ready"]:
            errors.append("source_check_not_require_promotion_ready")
        if source_check_summary["actual_benchmark_host"] != REQUIRED_BENCHMARK_HOST:
            errors.append(
                f"unexpected_source_check_host:{source_check_summary['actual_benchmark_host']}"
            )

        snapshot_report = (
            source_snapshot.get("snapshot_report")
            if isinstance(source_snapshot.get("snapshot_report"), dict)
            else {}
        )
        expected_snapshot_report_path = accepted_bundle_dir / "promotion_evidence_check.json"
        if not args.dry_run and snapshot_report and not _same_path(
            str(snapshot_report.get("source_path", "")),
            REPO_ROOT
            / "benchmarks"
            / "artifacts"
            / "simplified_likelihood_export_benchmarks"
            / REQUIRED_BENCHMARK_HOST
            / "current"
            / "export_benchmark_snapshot_report.json",
        ):
            errors.append("unexpected_snapshot_report_source_path")

    if errors:
        report = _report(
            status="failed",
            promoted=False,
            dry_run=bool(args.dry_run),
            deterministic=bool(args.deterministic),
            source_bundle_dir=source_bundle_dir,
            source_bundle_json_path=source_bundle_json_path,
            source_bundle_check_path=source_bundle_check_path,
            benchmark_artifact_source_path=benchmark_artifact_source_path,
            accepted_bundle_dir=accepted_bundle_dir,
            history_dir=history_dir,
            accepted_snapshot_index_path=accepted_snapshot_index_path,
            source_bundle_summary=source_bundle_summary,
            source_check_summary=source_check_summary,
            actions=actions,
            errors=errors,
        )
        _write_json(report_path, report)
        print("Exporter promotion bundle promotion: status=failed")
        return 1

    snapshot_id = args.snapshot_id.strip() or (
        derive_stamp_from_path(source_bundle_dir) or "exporter-promotion-local"
    )

    if args.dry_run:
        report = _report(
            status="dry_run",
            promoted=False,
            dry_run=True,
            deterministic=bool(args.deterministic),
            source_bundle_dir=source_bundle_dir,
            source_bundle_json_path=source_bundle_json_path,
            source_bundle_check_path=source_bundle_check_path,
            benchmark_artifact_source_path=benchmark_artifact_source_path,
            accepted_bundle_dir=accepted_bundle_dir,
            history_dir=history_dir,
            accepted_snapshot_index_path=accepted_snapshot_index_path,
            source_bundle_summary=source_bundle_summary,
            source_check_summary=source_check_summary,
            actions=actions,
            errors=[],
        )
        _write_json(report_path, report)
        print("Exporter promotion bundle promotion: status=dry_run")
        return 0

    stage_dir = accepted_bundle_dir.parent / ".accepted_stage"
    _cleanup_stage(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_bundle_dir, stage_dir, dirs_exist_ok=True)

    verify_proc = _run_bundle_verify(stage_dir, deterministic=bool(args.deterministic))
    if verify_proc.returncode != 0:
        errors.append("accepted_stage_verify_failed")
        if verify_proc.stdout:
            errors.append(f"accepted_stage_verify_stdout:{verify_proc.stdout.strip()}")
        if verify_proc.stderr:
            errors.append(f"accepted_stage_verify_stderr:{verify_proc.stderr.strip()}")
        _cleanup_stage(stage_dir)
        report = _report(
            status="failed",
            promoted=False,
            dry_run=False,
            deterministic=bool(args.deterministic),
            source_bundle_dir=source_bundle_dir,
            source_bundle_json_path=source_bundle_json_path,
            source_bundle_check_path=source_bundle_check_path,
            benchmark_artifact_source_path=benchmark_artifact_source_path,
            accepted_bundle_dir=accepted_bundle_dir,
            history_dir=history_dir,
            accepted_snapshot_index_path=accepted_snapshot_index_path,
            source_bundle_summary=source_bundle_summary,
            source_check_summary=source_check_summary,
            actions=actions,
            errors=errors,
        )
        _write_json(report_path, report)
        print("Exporter promotion bundle promotion: status=failed")
        return 1

    history_dir.mkdir(parents=True, exist_ok=True)
    promoted_archive = history_dir / f"accepted_{snapshot_id}_promoted"
    actions["accepted_updated"] = True
    actions["accepted_snapshot_index_path"] = str(accepted_bundle_dir / "snapshot_index.json")
    actions["archived_promoted_bundle_path"] = str(promoted_archive)

    staged_report = _report(
        status="promoted",
        promoted=True,
        dry_run=False,
        deterministic=bool(args.deterministic),
        source_bundle_dir=source_bundle_dir,
        source_bundle_json_path=source_bundle_json_path,
        source_bundle_check_path=source_bundle_check_path,
        benchmark_artifact_source_path=benchmark_artifact_source_path,
        accepted_bundle_dir=accepted_bundle_dir,
        history_dir=history_dir,
        accepted_snapshot_index_path=accepted_bundle_dir / "snapshot_index.json",
        source_bundle_summary=source_bundle_summary,
        source_check_summary=source_check_summary,
        actions=actions,
        errors=[],
    )
    _write_json(stage_dir / "promotion_bundle_promotion_report.json", staged_report)

    if accepted_bundle_dir.exists():
        previous_stamp = derive_stamp_from_path(accepted_bundle_dir / "promotion_evidence.json") or "previous"
        archived_previous_path = history_dir / f"accepted_{previous_stamp}_previous"
        if archived_previous_path.exists():
            shutil.rmtree(archived_previous_path)
        shutil.move(str(accepted_bundle_dir), str(archived_previous_path))
        actions["archived_previous_accepted"] = True
        actions["archived_previous_accepted_path"] = str(archived_previous_path)

    if accepted_bundle_dir.exists():
        shutil.rmtree(accepted_bundle_dir)
    shutil.move(str(stage_dir), str(accepted_bundle_dir))

    actions["accepted_snapshot_index_written"] = True
    actions["archived_promoted_bundle"] = True

    report = _report(
        status="promoted",
        promoted=True,
        dry_run=False,
        deterministic=bool(args.deterministic),
        source_bundle_dir=source_bundle_dir,
        source_bundle_json_path=source_bundle_json_path,
        source_bundle_check_path=source_bundle_check_path,
        benchmark_artifact_source_path=benchmark_artifact_source_path,
        accepted_bundle_dir=accepted_bundle_dir,
        history_dir=history_dir,
        accepted_snapshot_index_path=accepted_bundle_dir / "snapshot_index.json",
        source_bundle_summary=source_bundle_summary,
        source_check_summary=source_check_summary,
        actions=actions,
        errors=[],
    )
    _write_json(accepted_bundle_dir / "promotion_bundle_promotion_report.json", report)

    source_semantics_proc = _run_stable_source_semantics_boundary(
        accepted_bundle_dir, deterministic=bool(args.deterministic)
    )
    if source_semantics_proc.returncode != 0:
        errors.append("accepted_stable_source_semantics_boundary_failed")
        if source_semantics_proc.stdout:
            errors.append(
                f"accepted_stable_source_semantics_boundary_stdout:{source_semantics_proc.stdout.strip()}"
            )
        if source_semantics_proc.stderr:
            errors.append(
                f"accepted_stable_source_semantics_boundary_stderr:{source_semantics_proc.stderr.strip()}"
            )
        failure_report = _report(
            status="failed",
            promoted=False,
            dry_run=False,
            deterministic=bool(args.deterministic),
            source_bundle_dir=source_bundle_dir,
            source_bundle_json_path=source_bundle_json_path,
            source_bundle_check_path=source_bundle_check_path,
            benchmark_artifact_source_path=benchmark_artifact_source_path,
            accepted_bundle_dir=accepted_bundle_dir,
            history_dir=history_dir,
            accepted_snapshot_index_path=accepted_snapshot_index_path,
            source_bundle_summary=source_bundle_summary,
            source_check_summary=source_check_summary,
            actions=actions,
            errors=errors,
        )
        _write_json(accepted_bundle_dir / "promotion_bundle_promotion_report.json", failure_report)
        _write_json(report_path, failure_report)
        print("Exporter promotion bundle promotion: status=failed")
        return 1

    assessment_proc = _run_stable_review_assessment(
        accepted_bundle_dir, deterministic=bool(args.deterministic)
    )
    if assessment_proc.returncode != 0:
        errors.append("accepted_stable_review_assessment_failed")
        if assessment_proc.stdout:
            errors.append(
                f"accepted_stable_review_assessment_stdout:{assessment_proc.stdout.strip()}"
            )
        if assessment_proc.stderr:
            errors.append(
                f"accepted_stable_review_assessment_stderr:{assessment_proc.stderr.strip()}"
            )
        failure_report = _report(
            status="failed",
            promoted=False,
            dry_run=False,
            deterministic=bool(args.deterministic),
            source_bundle_dir=source_bundle_dir,
            source_bundle_json_path=source_bundle_json_path,
            source_bundle_check_path=source_bundle_check_path,
            benchmark_artifact_source_path=benchmark_artifact_source_path,
            accepted_bundle_dir=accepted_bundle_dir,
            history_dir=history_dir,
            accepted_snapshot_index_path=accepted_snapshot_index_path,
            source_bundle_summary=source_bundle_summary,
            source_check_summary=source_check_summary,
            actions=actions,
            errors=errors,
        )
        _write_json(accepted_bundle_dir / "promotion_bundle_promotion_report.json", failure_report)
        _write_json(report_path, failure_report)
        print("Exporter promotion bundle promotion: status=failed")
        return 1

    blocker_matrix_proc = _run_stable_candidate_blocker_matrix(
        accepted_bundle_dir, deterministic=bool(args.deterministic)
    )
    if blocker_matrix_proc.returncode != 0:
        errors.append("accepted_stable_candidate_blocker_matrix_failed")
        if blocker_matrix_proc.stdout:
            errors.append(
                f"accepted_stable_candidate_blocker_matrix_stdout:{blocker_matrix_proc.stdout.strip()}"
            )
        if blocker_matrix_proc.stderr:
            errors.append(
                f"accepted_stable_candidate_blocker_matrix_stderr:{blocker_matrix_proc.stderr.strip()}"
            )
        failure_report = _report(
            status="failed",
            promoted=False,
            dry_run=False,
            deterministic=bool(args.deterministic),
            source_bundle_dir=source_bundle_dir,
            source_bundle_json_path=source_bundle_json_path,
            source_bundle_check_path=source_bundle_check_path,
            benchmark_artifact_source_path=benchmark_artifact_source_path,
            accepted_bundle_dir=accepted_bundle_dir,
            history_dir=history_dir,
            accepted_snapshot_index_path=accepted_snapshot_index_path,
            source_bundle_summary=source_bundle_summary,
            source_check_summary=source_check_summary,
            actions=actions,
            errors=errors,
        )
        _write_json(accepted_bundle_dir / "promotion_bundle_promotion_report.json", failure_report)
        _write_json(report_path, failure_report)
        print("Exporter promotion bundle promotion: status=failed")
        return 1

    review_packet_proc = _run_stable_candidate_review_packet(
        accepted_bundle_dir, deterministic=bool(args.deterministic)
    )
    if review_packet_proc.returncode != 0:
        errors.append("accepted_stable_candidate_review_packet_failed")
        if review_packet_proc.stdout:
            errors.append(
                f"accepted_stable_candidate_review_packet_stdout:{review_packet_proc.stdout.strip()}"
            )
        if review_packet_proc.stderr:
            errors.append(
                f"accepted_stable_candidate_review_packet_stderr:{review_packet_proc.stderr.strip()}"
            )
        failure_report = _report(
            status="failed",
            promoted=False,
            dry_run=False,
            deterministic=bool(args.deterministic),
            source_bundle_dir=source_bundle_dir,
            source_bundle_json_path=source_bundle_json_path,
            source_bundle_check_path=source_bundle_check_path,
            benchmark_artifact_source_path=benchmark_artifact_source_path,
            accepted_bundle_dir=accepted_bundle_dir,
            history_dir=history_dir,
            accepted_snapshot_index_path=accepted_snapshot_index_path,
            source_bundle_summary=source_bundle_summary,
            source_check_summary=source_check_summary,
            actions=actions,
            errors=errors,
        )
        _write_json(accepted_bundle_dir / "promotion_bundle_promotion_report.json", failure_report)
        _write_json(report_path, failure_report)
        print("Exporter promotion bundle promotion: status=failed")
        return 1

    stable_promotion_decision_proc = _run_stable_promotion_decision(
        accepted_bundle_dir, deterministic=bool(args.deterministic)
    )
    if stable_promotion_decision_proc.returncode != 0:
        errors.append("accepted_stable_promotion_decision_failed")
        if stable_promotion_decision_proc.stdout:
            errors.append(
                f"accepted_stable_promotion_decision_stdout:{stable_promotion_decision_proc.stdout.strip()}"
            )
        if stable_promotion_decision_proc.stderr:
            errors.append(
                f"accepted_stable_promotion_decision_stderr:{stable_promotion_decision_proc.stderr.strip()}"
            )
        failure_report = _report(
            status="failed",
            promoted=False,
            dry_run=False,
            deterministic=bool(args.deterministic),
            source_bundle_dir=source_bundle_dir,
            source_bundle_json_path=source_bundle_json_path,
            source_bundle_check_path=source_bundle_check_path,
            benchmark_artifact_source_path=benchmark_artifact_source_path,
            accepted_bundle_dir=accepted_bundle_dir,
            history_dir=history_dir,
            accepted_snapshot_index_path=accepted_snapshot_index_path,
            source_bundle_summary=source_bundle_summary,
            source_check_summary=source_check_summary,
            actions=actions,
            errors=errors,
        )
        _write_json(accepted_bundle_dir / "promotion_bundle_promotion_report.json", failure_report)
        _write_json(report_path, failure_report)
        print("Exporter promotion bundle promotion: status=failed")
        return 1

    blocker_matrix_proc = _run_stable_candidate_blocker_matrix(
        accepted_bundle_dir, deterministic=bool(args.deterministic)
    )
    if blocker_matrix_proc.returncode != 0:
        errors.append("accepted_stable_candidate_blocker_matrix_finalize_failed")
        if blocker_matrix_proc.stdout:
            errors.append(
                f"accepted_stable_candidate_blocker_matrix_finalize_stdout:{blocker_matrix_proc.stdout.strip()}"
            )
        if blocker_matrix_proc.stderr:
            errors.append(
                f"accepted_stable_candidate_blocker_matrix_finalize_stderr:{blocker_matrix_proc.stderr.strip()}"
            )
        failure_report = _report(
            status="failed",
            promoted=False,
            dry_run=False,
            deterministic=bool(args.deterministic),
            source_bundle_dir=source_bundle_dir,
            source_bundle_json_path=source_bundle_json_path,
            source_bundle_check_path=source_bundle_check_path,
            benchmark_artifact_source_path=benchmark_artifact_source_path,
            accepted_bundle_dir=accepted_bundle_dir,
            history_dir=history_dir,
            accepted_snapshot_index_path=accepted_snapshot_index_path,
            source_bundle_summary=source_bundle_summary,
            source_check_summary=source_check_summary,
            actions=actions,
            errors=errors,
        )
        _write_json(accepted_bundle_dir / "promotion_bundle_promotion_report.json", failure_report)
        _write_json(report_path, failure_report)
        print("Exporter promotion bundle promotion: status=failed")
        return 1

    review_packet_proc = _run_stable_candidate_review_packet(
        accepted_bundle_dir, deterministic=bool(args.deterministic)
    )
    if review_packet_proc.returncode != 0:
        errors.append("accepted_stable_candidate_review_packet_finalize_failed")
        if review_packet_proc.stdout:
            errors.append(
                f"accepted_stable_candidate_review_packet_finalize_stdout:{review_packet_proc.stdout.strip()}"
            )
        if review_packet_proc.stderr:
            errors.append(
                f"accepted_stable_candidate_review_packet_finalize_stderr:{review_packet_proc.stderr.strip()}"
            )
        failure_report = _report(
            status="failed",
            promoted=False,
            dry_run=False,
            deterministic=bool(args.deterministic),
            source_bundle_dir=source_bundle_dir,
            source_bundle_json_path=source_bundle_json_path,
            source_bundle_check_path=source_bundle_check_path,
            benchmark_artifact_source_path=benchmark_artifact_source_path,
            accepted_bundle_dir=accepted_bundle_dir,
            history_dir=history_dir,
            accepted_snapshot_index_path=accepted_snapshot_index_path,
            source_bundle_summary=source_bundle_summary,
            source_check_summary=source_check_summary,
            actions=actions,
            errors=errors,
        )
        _write_json(accepted_bundle_dir / "promotion_bundle_promotion_report.json", failure_report)
        _write_json(report_path, failure_report)
        print("Exporter promotion bundle promotion: status=failed")
        return 1

    stable_promotion_decision_proc = _run_stable_promotion_decision(
        accepted_bundle_dir, deterministic=bool(args.deterministic)
    )
    if stable_promotion_decision_proc.returncode != 0:
        errors.append("accepted_stable_promotion_decision_finalize_failed")
        if stable_promotion_decision_proc.stdout:
            errors.append(
                f"accepted_stable_promotion_decision_finalize_stdout:{stable_promotion_decision_proc.stdout.strip()}"
            )
        if stable_promotion_decision_proc.stderr:
            errors.append(
                f"accepted_stable_promotion_decision_finalize_stderr:{stable_promotion_decision_proc.stderr.strip()}"
            )
        failure_report = _report(
            status="failed",
            promoted=False,
            dry_run=False,
            deterministic=bool(args.deterministic),
            source_bundle_dir=source_bundle_dir,
            source_bundle_json_path=source_bundle_json_path,
            source_bundle_check_path=source_bundle_check_path,
            benchmark_artifact_source_path=benchmark_artifact_source_path,
            accepted_bundle_dir=accepted_bundle_dir,
            history_dir=history_dir,
            accepted_snapshot_index_path=accepted_snapshot_index_path,
            source_bundle_summary=source_bundle_summary,
            source_check_summary=source_check_summary,
            actions=actions,
            errors=errors,
        )
        _write_json(accepted_bundle_dir / "promotion_bundle_promotion_report.json", failure_report)
        _write_json(report_path, failure_report)
        print("Exporter promotion bundle promotion: status=failed")
        return 1

    stable_evidence_policy_proc = _run_stable_evidence_policy(
        bundle_dir=accepted_bundle_dir,
        benchmark_artifact_path=benchmark_artifact_path,
        public_validation_report_path=public_validation_report_path,
        deterministic=bool(args.deterministic),
    )
    if stable_evidence_policy_proc.returncode != 0:
        errors.append("accepted_stable_evidence_policy_finalize_failed")
        if stable_evidence_policy_proc.stdout:
            errors.append(
                f"accepted_stable_evidence_policy_finalize_stdout:{stable_evidence_policy_proc.stdout.strip()}"
            )
        if stable_evidence_policy_proc.stderr:
            errors.append(
                f"accepted_stable_evidence_policy_finalize_stderr:{stable_evidence_policy_proc.stderr.strip()}"
            )
        failure_report = _report(
            status="failed",
            promoted=False,
            dry_run=False,
            deterministic=bool(args.deterministic),
            source_bundle_dir=source_bundle_dir,
            source_bundle_json_path=source_bundle_json_path,
            source_bundle_check_path=source_bundle_check_path,
            benchmark_artifact_source_path=benchmark_artifact_source_path,
            accepted_bundle_dir=accepted_bundle_dir,
            history_dir=history_dir,
            accepted_snapshot_index_path=accepted_snapshot_index_path,
            source_bundle_summary=source_bundle_summary,
            source_check_summary=source_check_summary,
            actions=actions,
            errors=errors,
        )
        _write_json(accepted_bundle_dir / "promotion_bundle_promotion_report.json", failure_report)
        _write_json(report_path, failure_report)
        print("Exporter promotion bundle promotion: status=failed")
        return 1

    stable_evidence_freshness_proc = _run_stable_evidence_freshness(
        bundle_dir=accepted_bundle_dir,
        snapshot_report_path=snapshot_report_path,
        public_validation_report_path=public_validation_report_path,
        deterministic=bool(args.deterministic),
    )
    if stable_evidence_freshness_proc.returncode != 0:
        errors.append("accepted_stable_evidence_freshness_finalize_failed")
        if stable_evidence_freshness_proc.stdout:
            errors.append(
                f"accepted_stable_evidence_freshness_finalize_stdout:{stable_evidence_freshness_proc.stdout.strip()}"
            )
        if stable_evidence_freshness_proc.stderr:
            errors.append(
                f"accepted_stable_evidence_freshness_finalize_stderr:{stable_evidence_freshness_proc.stderr.strip()}"
            )
        failure_report = _report(
            status="failed",
            promoted=False,
            dry_run=False,
            deterministic=bool(args.deterministic),
            source_bundle_dir=source_bundle_dir,
            source_bundle_json_path=source_bundle_json_path,
            source_bundle_check_path=source_bundle_check_path,
            benchmark_artifact_source_path=benchmark_artifact_source_path,
            accepted_bundle_dir=accepted_bundle_dir,
            history_dir=history_dir,
            accepted_snapshot_index_path=accepted_snapshot_index_path,
            source_bundle_summary=source_bundle_summary,
            source_check_summary=source_check_summary,
            actions=actions,
            errors=errors,
        )
        _write_json(accepted_bundle_dir / "promotion_bundle_promotion_report.json", failure_report)
        _write_json(report_path, failure_report)
        print("Exporter promotion bundle promotion: status=failed")
        return 1

    snapshot_proc = _write_snapshot_index(
        accepted_bundle_dir=accepted_bundle_dir,
        snapshot_index_path=accepted_bundle_dir / "snapshot_index.json",
        snapshot_id=snapshot_id,
    )
    if snapshot_proc.returncode != 0:
        errors.append("accepted_snapshot_index_failed")
        if snapshot_proc.stdout:
            errors.append(f"accepted_snapshot_index_stdout:{snapshot_proc.stdout.strip()}")
        if snapshot_proc.stderr:
            errors.append(f"accepted_snapshot_index_stderr:{snapshot_proc.stderr.strip()}")
        failure_report = _report(
            status="failed",
            promoted=False,
            dry_run=False,
            deterministic=bool(args.deterministic),
            source_bundle_dir=source_bundle_dir,
            source_bundle_json_path=source_bundle_json_path,
            source_bundle_check_path=source_bundle_check_path,
            benchmark_artifact_source_path=benchmark_artifact_source_path,
            accepted_bundle_dir=accepted_bundle_dir,
            history_dir=history_dir,
            accepted_snapshot_index_path=accepted_snapshot_index_path,
            source_bundle_summary=source_bundle_summary,
            source_check_summary=source_check_summary,
            actions=actions,
            errors=errors,
        )
        _write_json(accepted_bundle_dir / "promotion_bundle_promotion_report.json", failure_report)
        _write_json(report_path, failure_report)
        print("Exporter promotion bundle promotion: status=failed")
        return 1

    if promoted_archive.exists():
        shutil.rmtree(promoted_archive)
    shutil.copytree(accepted_bundle_dir, promoted_archive)
    print("Exporter promotion bundle promotion: status=promoted")
    _write_json(report_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
