#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from _simplified_likelihood_export_benchmark import (
    APEX2_REPORT_SCHEMA_VERSION,
    DEFAULT_CURRENT_DIR,
    DEFAULT_HISTORY_DIR,
    REPORT_SCHEMA_VERSION,
    REPO_ROOT,
    REQUIRED_BENCHMARK_HOST,
    SNAPSHOT_ARTIFACT_SUITE,
    derive_stamp_from_path,
    load_json,
    now_utc,
    relative_or_absolute,
)
from _simplified_likelihood_export_public_validation import (
    CURRENT_REPORT_FILENAME as PUBLIC_VALIDATION_REPORT_FILENAME,
    DEFAULT_CATALOG_PATH as DEFAULT_PUBLIC_VALIDATION_CATALOG_PATH,
    build_public_validation_report,
)


SNAPSHOT_INDEX_SCRIPT = REPO_ROOT / "scripts" / "benchmarks" / "write_snapshot_index.py"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float_or_zero(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _source_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    export_matrix = (
        report.get("export_matrix") if isinstance(report.get("export_matrix"), dict) else {}
    )
    export_matrix_summary = (
        export_matrix.get("summary") if isinstance(export_matrix.get("summary"), dict) else {}
    )
    export_matrix_cases = (
        export_matrix.get("cases") if isinstance(export_matrix.get("cases"), list) else []
    )
    environment = (
        report.get("environment") if isinstance(report.get("environment"), dict) else {}
    )
    synthetic_speedups = [
        _float_or_zero(
            case.get("bench", {}).get("speedup", {}).get("net_end_to_end_upper_limit")
        )
        for case in export_matrix_cases
        if isinstance(case, dict) and case.get("case_kind") == "synthetic"
    ]
    public_speedups = [
        _float_or_zero(
            case.get("bench", {}).get("speedup", {}).get("net_end_to_end_upper_limit")
        )
        for case in export_matrix_cases
        if isinstance(case, dict) and case.get("case_kind") == "public_reinterpretation_style"
    ]
    return {
        "schema_version": _string_or_none(report.get("schema_version")),
        "status": _string_or_none(summary.get("status")),
        "benchmark_host": _string_or_none(environment.get("hostname")),
        "all_schema_valid": bool(summary.get("all_schema_valid", False)),
        "all_fidelity_gates_pass": bool(summary.get("all_fidelity_gates_pass", False)),
        "all_performance_gates_pass": bool(summary.get("all_performance_gates_pass", False)),
        "case_count": int(summary.get("case_count", 0)),
        "export_matrix_included": bool(summary.get("export_matrix_included", False)),
        "export_matrix_status": _string_or_none(summary.get("export_matrix_status")),
        "export_matrix_case_count": int(summary.get("export_matrix_case_count", 0)),
        "export_matrix_case_kinds": list(summary.get("export_matrix_case_kinds", [])),
        "export_matrix_public_reinterpretation_style_case_count": int(
            summary.get("export_matrix_public_reinterpretation_style_case_count", 0)
        ),
        "export_matrix_min_net_end_to_end_upper_limit_speedup": _float_or_zero(
            summary.get("export_matrix_min_net_end_to_end_upper_limit_speedup")
        ),
        "export_matrix_synthetic_min_net_end_to_end_upper_limit_speedup": min(
            synthetic_speedups, default=0.0
        ),
        "export_matrix_public_reinterpretation_style_min_net_end_to_end_upper_limit_speedup": min(
            public_speedups, default=0.0
        ),
        "export_matrix_summary_status": _string_or_none(export_matrix_summary.get("status")),
        "export_matrix_summary_all_schema_valid": bool(
            export_matrix_summary.get("all_schema_valid", False)
        ),
        "export_matrix_summary_all_fidelity_gates_pass": bool(
            export_matrix_summary.get("all_fidelity_gates_pass", False)
        ),
        "export_matrix_summary_all_performance_gates_pass": bool(
            export_matrix_summary.get("all_performance_gates_pass", False)
        ),
        "public_fixture_matrix_included": bool(summary.get("public_fixture_matrix_included", False)),
        "public_fixture_matrix_status": _string_or_none(summary.get("public_fixture_matrix_status")),
    }


def _report(
    *,
    status: str,
    persisted: bool,
    dry_run: bool,
    generated_at_utc: str,
    snapshot_id: str,
    benchmark_artifact_source_path: str,
    current_dir: Path,
    current_benchmark_artifact_path: Path,
    current_report_path: Path,
    current_snapshot_index_path: Path,
    history_dir: Path,
    source_summary: dict[str, Any],
    actions: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "surface": "simplified_likelihood_export_benchmark",
        "status": status,
        "persisted": persisted,
        "dry_run": dry_run,
        "generated_at_utc": generated_at_utc,
        "snapshot_id": snapshot_id,
        "benchmark_artifact_source_path": benchmark_artifact_source_path,
        "current_dir": str(current_dir),
        "current_benchmark_artifact_path": str(current_benchmark_artifact_path),
        "current_report_path": str(current_report_path),
        "current_snapshot_index_path": str(current_snapshot_index_path),
        "history_dir": str(history_dir),
        "source_summary": source_summary,
        "actions": actions,
        "summary": {
            "top_level_errors": errors,
        },
    }


def _run_snapshot_index(
    *,
    current_dir: Path,
    snapshot_index_path: Path,
    snapshot_id: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SNAPSHOT_INDEX_SCRIPT),
            "--suite",
            SNAPSHOT_ARTIFACT_SUITE,
            "--artifacts-dir",
            str(current_dir),
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


def _default_snapshot_id(path: Path, deterministic: bool) -> str:
    derived = derive_stamp_from_path(path)
    if derived:
        return f"export-{derived}"
    return "export-19700101T000000Z" if deterministic else "export-local"


def _existing_current_stamp(current_dir: Path) -> str | None:
    current_report = current_dir / "export_benchmark_snapshot_report.json"
    if current_report.exists():
        try:
            payload = load_json(current_report)
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            recorded_source = payload.get("benchmark_artifact_source_path")
            if recorded_source:
                stamp = derive_stamp_from_path(Path(str(recorded_source)))
                if stamp:
                    return stamp
    current_artifact = current_dir / "apex2_simplified_likelihood_report.json"
    if current_artifact.exists():
        return derive_stamp_from_path(current_artifact)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-artifact", type=Path, required=True)
    parser.add_argument("--current-dir", type=Path, default=DEFAULT_CURRENT_DIR)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--snapshot-id", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    benchmark_artifact = args.benchmark_artifact.resolve()
    current_dir = args.current_dir.resolve()
    history_dir = args.history_dir.resolve()
    report_path = (
        args.report.resolve()
        if args.report is not None
        else benchmark_artifact.parent / "export_benchmark_snapshot_report.json"
    )
    public_validation_catalog_path = DEFAULT_PUBLIC_VALIDATION_CATALOG_PATH.resolve()
    generated_at_utc = now_utc(args.deterministic)
    current_benchmark_artifact_path = current_dir / "apex2_simplified_likelihood_report.json"
    current_report_path = current_dir / "export_benchmark_snapshot_report.json"
    current_public_validation_report_path = current_dir / PUBLIC_VALIDATION_REPORT_FILENAME
    current_snapshot_index_path = current_dir / "snapshot_index.json"
    snapshot_id = args.snapshot_id.strip() or _default_snapshot_id(
        benchmark_artifact, args.deterministic
    )

    errors: list[str] = []
    source_summary = {
        "schema_version": None,
        "status": None,
        "benchmark_host": None,
        "all_schema_valid": False,
        "all_fidelity_gates_pass": False,
        "all_performance_gates_pass": False,
        "case_count": 0,
        "export_matrix_included": False,
        "export_matrix_status": None,
        "export_matrix_case_count": 0,
        "export_matrix_case_kinds": [],
        "export_matrix_public_reinterpretation_style_case_count": 0,
        "export_matrix_min_net_end_to_end_upper_limit_speedup": 0.0,
        "export_matrix_synthetic_min_net_end_to_end_upper_limit_speedup": 0.0,
        "export_matrix_public_reinterpretation_style_min_net_end_to_end_upper_limit_speedup": 0.0,
        "export_matrix_summary_status": None,
        "export_matrix_summary_all_schema_valid": False,
        "export_matrix_summary_all_fidelity_gates_pass": False,
        "export_matrix_summary_all_performance_gates_pass": False,
        "public_fixture_matrix_included": False,
        "public_fixture_matrix_status": None,
    }
    actions = {
        "archived_previous_current": False,
        "archived_previous_current_path": None,
        "current_updated": False,
        "current_snapshot_index_written": False,
        "current_snapshot_index_path": None,
        "archived_persisted_snapshot": False,
        "archived_persisted_snapshot_path": None,
    }

    if not benchmark_artifact.exists():
        errors.append(f"benchmark_artifact_missing:{benchmark_artifact}")
    elif not benchmark_artifact.is_file():
        errors.append(f"benchmark_artifact_not_file:{benchmark_artifact}")

    if not errors:
        report = load_json(benchmark_artifact)
        source_summary = _source_summary(report)

        if report.get("schema_version") != APEX2_REPORT_SCHEMA_VERSION:
            errors.append(f"unexpected_schema_version:{report.get('schema_version')}")
        if source_summary["benchmark_host"] != REQUIRED_BENCHMARK_HOST:
            errors.append(f"unexpected_benchmark_host:{source_summary['benchmark_host']}")
        if source_summary["status"] != "ok":
            errors.append(f"benchmark_status_not_ok:{source_summary['status']}")
        if not source_summary["all_schema_valid"]:
            errors.append("benchmark_schema_validation_failed")
        if not source_summary["all_fidelity_gates_pass"]:
            errors.append("benchmark_fidelity_gates_failed")
        if not source_summary["export_matrix_included"]:
            errors.append("export_matrix_missing")
        if source_summary["export_matrix_status"] != "ok":
            errors.append(f"export_matrix_status_not_ok:{source_summary['export_matrix_status']}")
        if source_summary["export_matrix_case_count"] <= 0:
            errors.append("export_matrix_case_count_zero")
        if source_summary["export_matrix_summary_status"] != "ok":
            errors.append(
                f"export_matrix_summary_status_not_ok:{source_summary['export_matrix_summary_status']}"
            )
        if not source_summary["export_matrix_summary_all_schema_valid"]:
            errors.append("export_matrix_summary_schema_validation_failed")
        if not source_summary["export_matrix_summary_all_fidelity_gates_pass"]:
            errors.append("export_matrix_summary_fidelity_gates_failed")

    benchmark_artifact_source_path = relative_or_absolute(benchmark_artifact)
    if errors:
        status = "failed"
        exit_code = 1
    elif args.dry_run:
        status = "dry_run"
        exit_code = 0
    else:
        current_stamp = _existing_current_stamp(current_dir)
        source_stamp = derive_stamp_from_path(benchmark_artifact) or "unknown"
        current_archive_dir = history_dir / f"current_{current_stamp or source_stamp}_previous"
        persisted_archive_dir = history_dir / f"snapshot_{source_stamp}_persisted"

        if persisted_archive_dir.exists():
            errors.append(f"persisted_archive_already_exists:{persisted_archive_dir}")

        if not errors and current_dir.exists():
            if current_archive_dir.exists():
                errors.append(f"current_archive_already_exists:{current_archive_dir}")
            else:
                history_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(current_dir), str(current_archive_dir))
                actions["archived_previous_current"] = True
                actions["archived_previous_current_path"] = str(current_archive_dir)

        if errors:
            status = "failed"
            exit_code = 1
        else:
            current_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(benchmark_artifact, current_benchmark_artifact_path)
            actions["current_updated"] = True
            public_validation_report = build_public_validation_report(
                benchmark_artifact_path=current_benchmark_artifact_path,
                benchmark=load_json(current_benchmark_artifact_path),
                catalog_path=public_validation_catalog_path,
                catalog=load_json(public_validation_catalog_path),
                deterministic=bool(args.deterministic),
            )
            if public_validation_report.get("status") != "ok":
                raise SystemExit(
                    "public exporter validation report failed during benchmark persistence:\n"
                    f"{json.dumps(public_validation_report, indent=2, sort_keys=True)}"
                )
            _write_json(current_public_validation_report_path, public_validation_report)

            doc = _report(
                status="persisted",
                persisted=True,
                dry_run=False,
                generated_at_utc=generated_at_utc,
                snapshot_id=snapshot_id,
                benchmark_artifact_source_path=benchmark_artifact_source_path,
                current_dir=current_dir,
                current_benchmark_artifact_path=current_benchmark_artifact_path,
                current_report_path=current_report_path,
                current_snapshot_index_path=current_snapshot_index_path,
                history_dir=history_dir,
                source_summary=source_summary,
                actions={
                    **actions,
                    "current_snapshot_index_written": True,
                    "current_snapshot_index_path": str(current_snapshot_index_path),
                    "archived_persisted_snapshot": True,
                    "archived_persisted_snapshot_path": str(persisted_archive_dir),
                },
                errors=[],
            )
            _write_json(current_report_path, doc)
            if report_path != current_report_path:
                _write_json(report_path, doc)

            snapshot_proc = _run_snapshot_index(
                current_dir=current_dir,
                snapshot_index_path=current_snapshot_index_path,
                snapshot_id=snapshot_id,
            )
            if snapshot_proc.returncode != 0:
                raise SystemExit(
                    "failed to write snapshot index:\n"
                    f"stdout:\n{snapshot_proc.stdout}\n"
                    f"stderr:\n{snapshot_proc.stderr}"
                )

            shutil.copytree(current_dir, persisted_archive_dir)
            actions["archived_persisted_snapshot"] = True
            actions["archived_persisted_snapshot_path"] = str(persisted_archive_dir)

            status = "persisted"
            exit_code = 0

    if status != "persisted":
        doc = _report(
            status=status,
            persisted=False,
            dry_run=args.dry_run,
            generated_at_utc=generated_at_utc,
            snapshot_id=snapshot_id,
            benchmark_artifact_source_path=benchmark_artifact_source_path,
            current_dir=current_dir,
            current_benchmark_artifact_path=current_benchmark_artifact_path,
            current_report_path=current_report_path,
            current_snapshot_index_path=current_snapshot_index_path,
            history_dir=history_dir,
            source_summary=source_summary,
            actions=actions,
            errors=errors,
        )
        _write_json(report_path, doc)

    print(f"status={status}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
