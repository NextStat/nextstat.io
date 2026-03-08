#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from _simplified_likelihood_promotion_bundle import (
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
    REPO_ROOT / "scripts" / "benchmarks" / "verify_simplified_likelihood_promotion_evidence_bundle.py"
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
        "benchmark_host": summary.get("benchmark_host"),
        "supports_speedup_claim": bool(summary.get("supports_speedup_claim", False)),
        "supports_public_fixture_matrix": bool(
            summary.get("supports_public_fixture_matrix", False)
        ),
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
        "actual_min_end_to_end_upper_limit_speedup": float(
            readiness.get("actual_min_end_to_end_upper_limit_speedup", 0.0)
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
        "surface": "simplified_likelihood",
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
        "benchmark_host": None,
        "supports_speedup_claim": False,
        "supports_public_fixture_matrix": False,
        "artifact_count": 0,
        "required_artifact_count": 0,
    }
    source_check_summary = {
        "schema_version": None,
        "status": None,
        "ok": False,
        "require_promotion_ready": False,
        "actual_benchmark_host": None,
        "actual_min_end_to_end_upper_limit_speedup": 0.0,
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

        benchmark_evidence = (
            source_bundle.get("benchmark_evidence")
            if isinstance(source_bundle.get("benchmark_evidence"), dict)
            else {}
        )
        benchmark_artifact_source_path = str(benchmark_evidence.get("source_path", ""))

        if source_bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            errors.append(f"unexpected_source_bundle_schema:{source_bundle.get('schema_version')}")
        if source_check.get("schema_version") != CHECK_SCHEMA_VERSION:
            errors.append(f"unexpected_source_check_schema:{source_check.get('schema_version')}")
        if source_bundle_summary["status"] != "ok":
            errors.append(f"source_bundle_not_ok:{source_bundle_summary['status']}")
        if source_bundle_summary["benchmark_host"] != REQUIRED_BENCHMARK_HOST:
            errors.append(
                f"unexpected_source_bundle_host:{source_bundle_summary['benchmark_host']}"
            )
        if not source_bundle_summary["supports_speedup_claim"]:
            errors.append("source_bundle_speedup_claim_not_supported")
        if not source_bundle_summary["supports_public_fixture_matrix"]:
            errors.append("source_bundle_public_fixture_matrix_not_supported")

        if source_check_summary["status"] != "passed":
            errors.append(f"source_check_not_passed:{source_check_summary['status']}")
        if not source_check_summary["ok"]:
            errors.append("source_check_ok_false")
        if not source_check_summary["require_promotion_ready"]:
            errors.append("source_check_missing_require_promotion_ready")
        if source_check_summary["actual_benchmark_host"] != REQUIRED_BENCHMARK_HOST:
            errors.append(
                f"unexpected_source_check_host:{source_check_summary['actual_benchmark_host']}"
            )
        if not _same_path(str(source_check.get("bundle_dir", "")), source_bundle_dir):
            errors.append("source_check_bundle_dir_mismatch")
        if not _same_path(str(source_check.get("bundle_json_path", "")), source_bundle_json_path):
            errors.append("source_check_bundle_json_path_mismatch")

        readiness = (
            source_check.get("checks", {}).get("promotion_readiness")
            if isinstance(source_check.get("checks"), dict)
            else {}
        )
        readiness = readiness if isinstance(readiness, dict) else {}
        if readiness.get("status") != "passed":
            errors.append(f"source_check_promotion_readiness_not_passed:{readiness.get('status')}")

    stamp = derive_stamp_from_path(source_bundle_dir) or now_utc(bool(args.deterministic)).replace(
        "-", ""
    ).replace(":", "")
    snapshot_id = args.snapshot_id.strip() or (
        f"{PROMOTION_ARTIFACT_SUITE}-{REQUIRED_BENCHMARK_HOST}-{stamp}"
    )
    stage_dir = accepted_bundle_dir.parent / f".accepted_tmp_{stamp}"
    previous_archive = history_dir / f"accepted_{stamp}_previous"
    promoted_archive = history_dir / f"accepted_{stamp}_promoted"

    if accepted_bundle_dir.resolve() == source_bundle_dir.resolve():
        errors.append("source_bundle_dir_equals_accepted_bundle_dir")
    if stage_dir.exists():
        errors.append(f"staging_dir_already_exists:{stage_dir}")
    if previous_archive.exists():
        errors.append(f"history_previous_exists:{previous_archive}")
    if promoted_archive.exists():
        errors.append(f"history_promoted_exists:{promoted_archive}")

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
        return 2

    try:
        shutil.copytree(source_bundle_dir, stage_dir)
        preflight = _run_bundle_verify(stage_dir, deterministic=bool(args.deterministic))
        if preflight.returncode != 0:
            errors.append(f"staging_verify_failed:{preflight.returncode}")
            if preflight.stdout.strip():
                errors.append(f"staging_verify_stdout:{preflight.stdout.strip()}")
            if preflight.stderr.strip():
                errors.append(f"staging_verify_stderr:{preflight.stderr.strip()}")
    finally:
        if errors:
            _cleanup_stage(stage_dir)

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
        return 2

    if args.dry_run:
        _cleanup_stage(stage_dir)
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
        print(
            "Simplified-likelihood promotion bundle promotion:",
            "status=dry_run",
            f"accepted_bundle_dir={accepted_bundle_dir}",
            sep=" ",
        )
        print(f"Promotion report written to {report_path}")
        return 0

    history_dir.mkdir(parents=True, exist_ok=True)
    accepted_bundle_dir.parent.mkdir(parents=True, exist_ok=True)

    if accepted_bundle_dir.exists():
        shutil.move(str(accepted_bundle_dir), str(previous_archive))
        actions["archived_previous_accepted"] = True
        actions["archived_previous_accepted_path"] = str(previous_archive)

    shutil.move(str(stage_dir), str(accepted_bundle_dir))
    actions["accepted_updated"] = True

    accepted_verify = _run_bundle_verify(
        accepted_bundle_dir, deterministic=bool(args.deterministic)
    )
    if accepted_verify.returncode != 0:
        errors.append(f"accepted_verify_failed:{accepted_verify.returncode}")
        if accepted_verify.stdout.strip():
            errors.append(f"accepted_verify_stdout:{accepted_verify.stdout.strip()}")
        if accepted_verify.stderr.strip():
            errors.append(f"accepted_verify_stderr:{accepted_verify.stderr.strip()}")

    provisional_status = "promoted" if not errors else "failed"
    provisional_report = _report(
        status=provisional_status,
        promoted=not errors,
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
    _write_json(report_path, provisional_report)
    _write_json(accepted_bundle_dir / "promotion_bundle_promotion_report.json", provisional_report)

    snapshot = _write_snapshot_index(
        accepted_bundle_dir=accepted_bundle_dir,
        snapshot_index_path=accepted_snapshot_index_path,
        snapshot_id=snapshot_id,
    )
    if snapshot.returncode != 0:
        errors.append(f"snapshot_index_failed:{snapshot.returncode}")
        if snapshot.stdout.strip():
            errors.append(f"snapshot_index_stdout:{snapshot.stdout.strip()}")
        if snapshot.stderr.strip():
            errors.append(f"snapshot_index_stderr:{snapshot.stderr.strip()}")
    else:
        actions["accepted_snapshot_index_written"] = True
        actions["accepted_snapshot_index_path"] = str(accepted_snapshot_index_path)

    status = "promoted" if not errors else "failed"
    report = _report(
        status=status,
        promoted=not errors,
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
    _write_json(accepted_bundle_dir / "promotion_bundle_promotion_report.json", report)

    if actions["accepted_snapshot_index_written"]:
        snapshot = _write_snapshot_index(
            accepted_bundle_dir=accepted_bundle_dir,
            snapshot_index_path=accepted_snapshot_index_path,
            snapshot_id=snapshot_id,
        )
        if snapshot.returncode != 0:
            errors.append(f"snapshot_index_refresh_failed:{snapshot.returncode}")
            if snapshot.stdout.strip():
                errors.append(f"snapshot_index_refresh_stdout:{snapshot.stdout.strip()}")
            if snapshot.stderr.strip():
                errors.append(f"snapshot_index_refresh_stderr:{snapshot.stderr.strip()}")
            status = "failed"
            report = _report(
                status=status,
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
            _write_json(accepted_bundle_dir / "promotion_bundle_promotion_report.json", report)

    if not errors:
        shutil.copytree(accepted_bundle_dir, promoted_archive)
        actions["archived_promoted_bundle"] = True
        actions["archived_promoted_bundle_path"] = str(promoted_archive)
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
            accepted_snapshot_index_path=accepted_snapshot_index_path,
            source_bundle_summary=source_bundle_summary,
            source_check_summary=source_check_summary,
            actions=actions,
            errors=errors,
        )
        _write_json(report_path, report)
        _write_json(accepted_bundle_dir / "promotion_bundle_promotion_report.json", report)
        _write_json(promoted_archive / "promotion_bundle_promotion_report.json", report)
        snapshot = _write_snapshot_index(
            accepted_bundle_dir=accepted_bundle_dir,
            snapshot_index_path=accepted_snapshot_index_path,
            snapshot_id=snapshot_id,
        )
        if snapshot.returncode != 0:
            errors.append(f"snapshot_index_post_archive_failed:{snapshot.returncode}")
            if snapshot.stdout.strip():
                errors.append(f"snapshot_index_post_archive_stdout:{snapshot.stdout.strip()}")
            if snapshot.stderr.strip():
                errors.append(f"snapshot_index_post_archive_stderr:{snapshot.stderr.strip()}")
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
            _write_json(accepted_bundle_dir / "promotion_bundle_promotion_report.json", report)
        promoted_snapshot = _write_snapshot_index(
            accepted_bundle_dir=promoted_archive,
            snapshot_index_path=promoted_archive / "snapshot_index.json",
            snapshot_id=f"{snapshot_id}-promoted",
        )
        if promoted_snapshot.returncode != 0:
            errors.append(f"promoted_snapshot_index_failed:{promoted_snapshot.returncode}")
            if promoted_snapshot.stdout.strip():
                errors.append(f"promoted_snapshot_index_stdout:{promoted_snapshot.stdout.strip()}")
            if promoted_snapshot.stderr.strip():
                errors.append(f"promoted_snapshot_index_stderr:{promoted_snapshot.stderr.strip()}")
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
            _write_json(accepted_bundle_dir / "promotion_bundle_promotion_report.json", report)
            _write_json(promoted_archive / "promotion_bundle_promotion_report.json", report)

    print(
        "Simplified-likelihood promotion bundle promotion:",
        f"status={'promoted' if not errors else 'failed'}",
        f"accepted_bundle_dir={accepted_bundle_dir}",
        sep=" ",
    )
    print(f"Promotion report written to {report_path}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
