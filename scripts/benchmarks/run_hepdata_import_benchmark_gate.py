#!/usr/bin/env python3
"""One-shot HEPData import benchmark gate: run, compare, and optionally promote."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "nextstat.hepdata_import_benchmark_gate_report.v1"
COMPARE_SCHEMA_VERSION = "nextstat.hepdata_import_benchmark_compare_report.v1"
PROMOTION_SCHEMA_VERSION = "nextstat.hepdata_import_benchmark_baseline_promotion_report.v1"
SUITE = "hepdata_import"
DEFAULT_ACCEPTED = REPO_ROOT / "benchmarks" / "artifacts" / "hepdata_import_baselines" / "nextstat-bench" / "accepted.json"
DEFAULT_REMOTE_RUNNER = "bash scripts/benchmarks/bench_hepdata_import_remote.sh"
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "benchmarks" / "compare_hepdata_import_benchmark.py"
PROMOTE_SCRIPT = REPO_ROOT / "scripts" / "benchmarks" / "promote_hepdata_import_benchmark_baseline.py"
REMOTE_DONE_PREFIX = "[hepdata-import-remote] done: "


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"failed to read JSON from {path}: {exc}") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_header(path: Path) -> dict[str, Any]:
    doc = _read_json(path)
    meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
    env = doc.get("environment") if isinstance(doc.get("environment"), dict) else {}
    cases = doc.get("cases") if isinstance(doc.get("cases"), list) else []
    return {
        "path": str(path),
        "schema_version": doc.get("schema_version"),
        "suite": doc.get("suite"),
        "deterministic": bool(doc.get("deterministic")),
        "host_policy": meta.get("host_policy"),
        "node": env.get("node"),
        "smoke": bool(meta.get("smoke")),
        "fit_enabled": bool(meta.get("fit_enabled")),
        "repeat": meta.get("repeat"),
        "fit_repeat": meta.get("fit_repeat"),
        "case_ids": [case.get("id") for case in cases if isinstance(case, dict)],
    }


def _empty_artifact(path: Path | None) -> dict[str, Any]:
    return {
        "path": None if path is None else str(path),
        "schema_version": None,
        "suite": None,
        "deterministic": False,
        "host_policy": None,
        "node": None,
        "smoke": False,
        "fit_enabled": False,
        "repeat": None,
        "fit_repeat": None,
        "case_ids": [],
    }


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_report_dir(current: Path) -> Path:
    return current.parent


def _parse_remote_artifact_dir(output: str) -> Path | None:
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if stripped.startswith(REMOTE_DONE_PREFIX):
            return Path(stripped[len(REMOTE_DONE_PREFIX) :].strip())
    return None


def _run_runner(runner_cmd: str) -> tuple[int, list[str], str]:
    argv = shlex.split(runner_cmd)
    proc = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, argv, proc.stdout


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _run_compare(accepted: Path, current: Path, compare_report: Path) -> tuple[int, dict[str, Any] | None, list[str]]:
    argv = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--baseline",
        str(accepted),
        "--current",
        str(current),
        "--out",
        str(compare_report),
    ]
    proc = subprocess.run(argv, cwd=REPO_ROOT, check=False)
    return proc.returncode, _read_json_if_exists(compare_report), argv


def _run_promote(
    *,
    accepted: Path,
    current: Path,
    compare_report: Path,
    promotion_report: Path,
    history_dir: Path,
    allow_review: bool,
    dry_run: bool,
) -> tuple[int, dict[str, Any] | None, list[str]]:
    argv = [
        sys.executable,
        str(PROMOTE_SCRIPT),
        "--accepted",
        str(accepted),
        "--current",
        str(current),
        "--compare-report",
        str(compare_report),
        "--report",
        str(promotion_report),
        "--history-dir",
        str(history_dir),
    ]
    if allow_review:
        argv.append("--allow-review")
    if dry_run:
        argv.append("--dry-run")
    proc = subprocess.run(argv, cwd=REPO_ROOT, check=False)
    return proc.returncode, _read_json_if_exists(promotion_report), argv


def _step_benchmark(
    *,
    current: Path | None,
    runner_cmd: str | None,
) -> tuple[dict[str, Any], Path | None, list[str]]:
    if current is not None:
        artifact_path = current
        return (
            {
                "mode": "provided_artifact",
                "status": "skipped",
                "ok": True,
                "exit_code": None,
                "command": [],
                "artifact_dir": str(artifact_path.parent),
                "artifact_path": str(artifact_path),
                "message": "using provided current artifact",
                "errors": [],
            },
            artifact_path,
            [],
        )

    if runner_cmd is None:
        runner_cmd = DEFAULT_REMOTE_RUNNER
    exit_code, argv, output = _run_runner(runner_cmd)
    artifact_dir = _parse_remote_artifact_dir(output)
    artifact_path = None if artifact_dir is None else artifact_dir / "summary.json"
    errors: list[str] = []
    status = "passed"
    ok = exit_code == 0
    if exit_code != 0:
        status = "failed"
        errors.append(f"runner_exit:{exit_code}")
    if artifact_dir is None:
        status = "failed"
        ok = False
        errors.append("runner_missing_done_path")
    if artifact_path is None or not artifact_path.exists():
        status = "failed"
        ok = False
        errors.append("runner_missing_summary_json")
    return (
        {
            "mode": "runner",
            "status": status,
            "ok": ok,
            "exit_code": exit_code,
            "command": argv,
            "artifact_dir": None if artifact_dir is None else str(artifact_dir),
            "artifact_path": None if artifact_path is None else str(artifact_path),
            "message": output.strip().splitlines()[-1] if output.strip() else "",
            "errors": errors,
        },
        artifact_path,
        argv,
    )


def _build_report(
    *,
    timestamp_utc: str,
    status: str,
    ok: bool,
    requires_review: bool,
    promotion_mode: str,
    allow_review: bool,
    accepted: Path,
    current: Path | None,
    compare_report: Path,
    promotion_report: Path | None,
    benchmark_step: dict[str, Any],
    compare_step: dict[str, Any],
    promotion_step: dict[str, Any],
    accepted_before: dict[str, Any],
    current_header: dict[str, Any],
    accepted_after: dict[str, Any],
    summary_errors: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "suite": SUITE,
        "timestamp_utc": timestamp_utc,
        "status": status,
        "ok": ok,
        "requires_review": requires_review,
        "promotion_mode": promotion_mode,
        "allow_review": allow_review,
        "accepted_baseline_path": str(accepted),
        "current_artifact_path": None if current is None else str(current),
        "compare_report_path": str(compare_report),
        "promotion_report_path": None if promotion_report is None else str(promotion_report),
        "steps": {
            "benchmark": benchmark_step,
            "compare": compare_step,
            "promotion": promotion_step,
        },
        "artifacts": {
            "accepted_before": accepted_before,
            "current": current_header,
            "accepted_after": accepted_after,
        },
        "summary": {
            "benchmark_status": benchmark_step.get("status"),
            "compare_status": compare_step.get("compare_status"),
            "promotion_status": promotion_step.get("promotion_status"),
            "top_level_errors": summary_errors,
        },
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run the HEPData import benchmark gate end-to-end.")
    parser.add_argument("--current", type=Path, default=None, help="Existing benchmark artifact to use instead of running the remote benchmark")
    parser.add_argument("--runner-cmd", type=str, default=None, help=f"Runner command when --current is not provided (default: {DEFAULT_REMOTE_RUNNER!r})")
    parser.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED, help=f"Accepted baseline path (default: {DEFAULT_ACCEPTED})")
    parser.add_argument("--compare-report", type=Path, default=None, help="Compare report path; defaults next to the current artifact")
    parser.add_argument("--promotion-report", type=Path, default=None, help="Promotion report path; defaults next to the current artifact")
    parser.add_argument("--report", type=Path, default=None, help="Gate report path; defaults next to the current artifact")
    parser.add_argument("--history-dir", type=Path, default=None, help="History directory for promotion; defaults to <accepted-dir>/history")
    parser.add_argument(
        "--promotion-mode",
        choices=["none", "dry_run", "apply"],
        default="none",
        help="Promotion mode after compare passes (default: none)",
    )
    parser.add_argument(
        "--allow-review",
        dest="allow_review",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow promotion when compare status=review (default: false)",
    )
    args = parser.parse_args(argv)

    timestamp_utc = _timestamp_utc()
    accepted_before = _artifact_header(args.accepted) if args.accepted.exists() else _empty_artifact(args.accepted)
    benchmark_step, current_artifact, _ = _step_benchmark(current=args.current, runner_cmd=args.runner_cmd)
    report_dir = _default_report_dir(current_artifact) if current_artifact is not None else REPO_ROOT / "bench_results" / "hepdata_import_benchmark"
    compare_report = args.compare_report or report_dir / "compare_report.json"
    promotion_report = None if args.promotion_mode == "none" else (args.promotion_report or report_dir / "promotion_report.json")
    gate_report = args.report or report_dir / "gate_report.json"
    history_dir = args.history_dir or args.accepted.parent / "history"

    compare_step: dict[str, Any]
    promotion_step: dict[str, Any]
    current_header = _empty_artifact(current_artifact)
    accepted_after = accepted_before
    summary_errors: list[str] = []

    if benchmark_step["status"] == "failed":
        summary_errors.append("benchmark_step_failed")

    if current_artifact is not None and current_artifact.exists():
        current_header = _artifact_header(current_artifact)

    if current_artifact is None:
        compare_step = {
            "status": "failed",
            "ok": False,
            "exit_code": None,
            "command": [],
            "report_path": str(compare_report),
            "compare_status": None,
            "message": "benchmark step did not produce a current artifact",
            "errors": ["missing_current_artifact"],
        }
        promotion_step = {
            "mode": args.promotion_mode,
            "status": "skipped",
            "ok": False,
            "exit_code": None,
            "command": [],
            "report_path": None if promotion_report is None else str(promotion_report),
            "promotion_status": None,
            "message": "promotion skipped because benchmark step failed",
            "errors": ["benchmark_step_failed"],
        }
        report = _build_report(
            timestamp_utc=timestamp_utc,
            status="failed",
            ok=False,
            requires_review=False,
            promotion_mode=args.promotion_mode,
            allow_review=bool(args.allow_review),
            accepted=args.accepted,
            current=current_artifact,
            compare_report=compare_report,
            promotion_report=promotion_report,
            benchmark_step=benchmark_step,
            compare_step=compare_step,
            promotion_step=promotion_step,
            accepted_before=accepted_before,
            current_header=current_header,
            accepted_after=accepted_after,
            summary_errors=summary_errors,
        )
        _write_json(gate_report, report)
        return 2

    compare_exit, compare_doc, compare_cmd = _run_compare(args.accepted, current_artifact, compare_report)
    compare_status = None if compare_doc is None else compare_doc.get("status")
    compare_errors: list[str] = []
    if compare_doc is None:
        compare_errors.append("compare_report_missing")
    else:
        if compare_doc.get("schema_version") != COMPARE_SCHEMA_VERSION:
            compare_errors.append(f"unexpected_compare_schema:{compare_doc.get('schema_version')}")
        if compare_doc.get("suite") != SUITE:
            compare_errors.append(f"unexpected_compare_suite:{compare_doc.get('suite')}")
        if compare_doc.get("baseline_path") != str(args.accepted):
            compare_errors.append("compare_baseline_path_mismatch")
        if compare_doc.get("current_path") != str(current_artifact):
            compare_errors.append("compare_current_path_mismatch")
    if compare_exit != 0:
        compare_errors.append(f"compare_runner_exit:{compare_exit}")
    compare_ok = compare_status in {"passed", "review"} and compare_exit == 0 and not compare_errors
    compare_step = {
        "status": "passed" if compare_ok else "failed",
        "ok": compare_ok,
        "exit_code": compare_exit,
        "command": compare_cmd,
        "report_path": str(compare_report),
        "compare_status": compare_status,
        "message": f"compare status={compare_status}",
        "errors": compare_errors,
    }
    if compare_step["status"] == "failed":
        summary_errors.append("compare_step_failed")

    if args.promotion_mode == "none":
        promotion_step = {
            "mode": "none",
            "status": "skipped",
            "ok": True,
            "exit_code": None,
            "command": [],
            "report_path": None,
            "promotion_status": None,
            "message": "promotion disabled",
            "errors": [],
        }
    elif compare_step["status"] == "failed":
        promotion_step = {
            "mode": args.promotion_mode,
            "status": "skipped",
            "ok": False,
            "exit_code": None,
            "command": [],
            "report_path": None if promotion_report is None else str(promotion_report),
            "promotion_status": None,
            "message": "promotion skipped because compare step failed",
            "errors": ["compare_step_failed"],
        }
    else:
        promote_exit, promote_doc, promote_cmd = _run_promote(
            accepted=args.accepted,
            current=current_artifact,
            compare_report=compare_report,
            promotion_report=promotion_report,
            history_dir=history_dir,
            allow_review=bool(args.allow_review),
            dry_run=args.promotion_mode == "dry_run",
        )
        promotion_status = None if promote_doc is None else promote_doc.get("status")
        promotion_errors: list[str] = []
        if promote_doc is None:
            promotion_errors.append("promotion_report_missing")
        else:
            if promote_doc.get("schema_version") != PROMOTION_SCHEMA_VERSION:
                promotion_errors.append(f"unexpected_promotion_schema:{promote_doc.get('schema_version')}")
            if promote_doc.get("suite") != SUITE:
                promotion_errors.append(f"unexpected_promotion_suite:{promote_doc.get('suite')}")
            if promote_doc.get("accepted_baseline_path") != str(args.accepted):
                promotion_errors.append("promotion_accepted_path_mismatch")
            if promote_doc.get("candidate_path") != str(current_artifact):
                promotion_errors.append("promotion_candidate_path_mismatch")
        if promote_exit != 0:
            promotion_errors.append(f"promotion_runner_exit:{promote_exit}")
        promotion_ok = promote_exit == 0 and not promotion_errors
        promotion_step = {
            "mode": args.promotion_mode,
            "status": "passed" if promotion_ok else "failed",
            "ok": promotion_ok,
            "exit_code": promote_exit,
            "command": promote_cmd,
            "report_path": str(promotion_report),
            "promotion_status": promotion_status,
            "message": f"promotion status={promotion_status}",
            "errors": promotion_errors,
        }
        if args.accepted.exists():
            accepted_after = _artifact_header(args.accepted)
        if promotion_step["status"] == "failed":
            summary_errors.append("promotion_step_failed")

    if benchmark_step["status"] == "failed" or compare_step["status"] == "failed" or (
        args.promotion_mode != "none" and promotion_step["status"] == "failed"
    ):
        status = "failed"
        ok = False
        requires_review = False
    elif compare_status == "review":
        status = "review"
        ok = True
        requires_review = True
    else:
        status = "passed"
        ok = True
        requires_review = False

    report = _build_report(
        timestamp_utc=timestamp_utc,
        status=status,
        ok=ok,
        requires_review=requires_review,
        promotion_mode=args.promotion_mode,
        allow_review=bool(args.allow_review),
        accepted=args.accepted,
        current=current_artifact,
        compare_report=compare_report,
        promotion_report=promotion_report,
        benchmark_step=benchmark_step,
        compare_step=compare_step,
        promotion_step=promotion_step,
        accepted_before=accepted_before,
        current_header=current_header,
        accepted_after=accepted_after,
        summary_errors=summary_errors,
    )
    _write_json(gate_report, report)

    if not ok:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
