#!/usr/bin/env python3
"""One-shot ads + weekly time-series benchmark gate: run, compare, and optionally promote."""

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
SCHEMA_VERSION = "nextstat.ads_timeseries_benchmark_gate_report.v1"
COMPARE_SCHEMA_VERSION = "nextstat.ads_timeseries_benchmark_compare_report.v1"
PROMOTION_SCHEMA_VERSION = "nextstat.ads_timeseries_benchmark_baseline_promotion_report.v1"
SUITE = "ads_timeseries_surface"
DEFAULT_ACCEPTED = (
    REPO_ROOT / "benchmarks" / "artifacts" / "ads_timeseries_baselines" / "nextstat-bench" / "accepted.json"
)
DEFAULT_REMOTE_RUNNER = "bash scripts/benchmarks/bench_ads_timeseries_surface_remote.sh"
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "benchmarks" / "compare_ads_timeseries_benchmark.py"
PROMOTE_SCRIPT = REPO_ROOT / "scripts" / "benchmarks" / "promote_ads_timeseries_benchmark_baseline.py"
REMOTE_DONE_PREFIX = "[ads-ts-remote] done: "


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
    protocol = doc.get("protocol") if isinstance(doc.get("protocol"), dict) else {}
    host = doc.get("host") if isinstance(doc.get("host"), dict) else {}
    binary = doc.get("binary") if isinstance(doc.get("binary"), dict) else {}
    results = doc.get("results") if isinstance(doc.get("results"), list) else []
    return {
        "path": str(path),
        "schema_version": doc.get("schema_version"),
        "suite": doc.get("suite"),
        "host_policy": meta.get("host_policy"),
        "hostname": host.get("hostname"),
        "build_profile": binary.get("build_profile"),
        "smoke": bool(meta.get("smoke")),
        "deterministic": bool(meta.get("deterministic")),
        "runs": protocol.get("runs"),
        "warmups": protocol.get("warmups"),
        "case_ids": [case.get("case_id") for case in results if isinstance(case, dict)],
    }


def _empty_artifact(path: Path | None) -> dict[str, Any]:
    return {
        "path": None if path is None else str(path),
        "schema_version": None,
        "suite": None,
        "host_policy": None,
        "hostname": None,
        "build_profile": None,
        "smoke": False,
        "deterministic": False,
        "runs": None,
        "warmups": None,
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
    artifact_path = None if artifact_dir is None else artifact_dir / "ads_timeseries_benchmark.json"
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
        errors.append("runner_missing_benchmark_json")
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
            "top_level_errors": summary_errors,
        },
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run the ads + weekly time-series benchmark gate.")
    parser.add_argument("--current", type=Path, default=None, help="Existing current benchmark artifact; skips runner when provided")
    parser.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED, help=f"Accepted baseline path (default: {DEFAULT_ACCEPTED})")
    parser.add_argument("--runner", type=str, default=None, help=f"Runner command (default: {DEFAULT_REMOTE_RUNNER!r})")
    parser.add_argument("--compare-report", type=Path, default=None, help="Compare report path; defaults to <artifact-dir>/compare_report.json")
    parser.add_argument("--promotion-report", type=Path, default=None, help="Promotion report path; defaults to <artifact-dir>/promotion_report.json")
    parser.add_argument("--report", type=Path, default=None, help="Gate report path; defaults to <artifact-dir>/gate_report.json")
    parser.add_argument("--history-dir", type=Path, default=None, help="Promotion history dir; defaults to <accepted-dir>/history")
    parser.add_argument(
        "--promotion-mode",
        choices=("none", "dry_run", "apply"),
        default="dry_run",
        help="Promotion mode for the accepted baseline (default: dry_run)",
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
    benchmark_step, current_artifact, benchmark_cmd = _step_benchmark(current=args.current, runner_cmd=args.runner)
    summary_errors: list[str] = list(benchmark_step.get("errors", []))

    if current_artifact is None:
        compare_report = args.compare_report or REPO_ROOT / "bench_results" / "ads_timeseries_surface" / "compare_report.json"
        gate_report = args.report or REPO_ROOT / "bench_results" / "ads_timeseries_surface" / "gate_report.json"
        promotion_report = args.promotion_report or REPO_ROOT / "bench_results" / "ads_timeseries_surface" / "promotion_report.json"
        accepted_before = _empty_artifact(args.accepted)
        current_header = _empty_artifact(None)
        accepted_after = _empty_artifact(args.accepted)
        compare_step = {
            "status": "skipped",
            "ok": False,
            "command": [],
            "report_path": str(compare_report),
            "compare_status": None,
            "errors": ["benchmark_artifact_missing"],
        }
        promotion_step = {
            "mode": args.promotion_mode,
            "status": "skipped",
            "ok": False,
            "command": [],
            "report_path": None if args.promotion_mode == "none" else str(promotion_report),
            "promotion_status": None,
            "errors": ["benchmark_artifact_missing"],
        }
        report = _build_report(
            timestamp_utc=timestamp_utc,
            status="failed",
            ok=False,
            requires_review=False,
            promotion_mode=args.promotion_mode,
            allow_review=bool(args.allow_review),
            accepted=args.accepted,
            current=None,
            compare_report=compare_report,
            promotion_report=promotion_report if args.promotion_mode != "none" else None,
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

    report_dir = _default_report_dir(current_artifact)
    compare_report = args.compare_report or report_dir / "compare_report.json"
    gate_report = args.report or report_dir / "gate_report.json"
    promotion_report = None if args.promotion_mode == "none" else (args.promotion_report or report_dir / "promotion_report.json")
    history_dir = args.history_dir or args.accepted.parent / "history"

    accepted_before = _empty_artifact(args.accepted)
    accepted_after = _empty_artifact(args.accepted)
    current_header = _artifact_header(current_artifact)
    if args.accepted.exists():
        accepted_before = _artifact_header(args.accepted)
        accepted_after = accepted_before
    else:
        summary_errors.append(f"accepted_missing:{args.accepted}")

    compare_exit, compare_doc, compare_cmd = _run_compare(args.accepted, current_artifact, compare_report)
    compare_errors: list[str] = []
    compare_status = None
    if compare_doc is None:
        compare_errors.append("compare_report_missing")
    else:
        compare_status = str(compare_doc.get("status"))
        if compare_doc.get("schema_version") != COMPARE_SCHEMA_VERSION:
            compare_errors.append(f"unexpected_compare_schema:{compare_doc.get('schema_version')}")
        if compare_doc.get("suite") != SUITE:
            compare_errors.append(f"unexpected_compare_suite:{compare_doc.get('suite')}")
        if compare_doc.get("baseline_path") != str(args.accepted):
            compare_errors.append("compare_baseline_path_mismatch")
        if compare_doc.get("current_path") != str(current_artifact):
            compare_errors.append("compare_current_path_mismatch")
    compare_step = {
        "status": "passed" if compare_exit == 0 and not compare_errors else "failed",
        "ok": compare_exit == 0 and not compare_errors,
        "command": compare_cmd,
        "exit_code": compare_exit,
        "report_path": str(compare_report),
        "compare_status": compare_status,
        "errors": compare_errors,
    }

    promotion_step: dict[str, Any]
    if args.promotion_mode == "none":
        promotion_step = {
            "mode": "none",
            "status": "skipped",
            "ok": True,
            "command": [],
            "report_path": None,
            "promotion_status": None,
            "errors": [],
        }
    else:
        promote_exit, promote_doc, promote_cmd = _run_promote(
            accepted=args.accepted,
            current=current_artifact,
            compare_report=compare_report,
            promotion_report=promotion_report if promotion_report is not None else report_dir / "promotion_report.json",
            history_dir=history_dir,
            allow_review=bool(args.allow_review),
            dry_run=args.promotion_mode == "dry_run",
        )
        promotion_errors: list[str] = []
        promotion_status = None
        if promote_doc is None:
            promotion_errors.append("promotion_report_missing")
        else:
            promotion_status = str(promote_doc.get("status"))
            if promote_doc.get("schema_version") != PROMOTION_SCHEMA_VERSION:
                promotion_errors.append(f"unexpected_promotion_schema:{promote_doc.get('schema_version')}")
            if promote_doc.get("suite") != SUITE:
                promotion_errors.append(f"unexpected_promotion_suite:{promote_doc.get('suite')}")
        promotion_step = {
            "mode": args.promotion_mode,
            "status": "passed" if promote_exit == 0 and not promotion_errors else "failed",
            "ok": promote_exit == 0 and not promotion_errors,
            "command": promote_cmd,
            "exit_code": promote_exit,
            "report_path": None if promotion_report is None else str(promotion_report),
            "promotion_status": promotion_status,
            "errors": promotion_errors,
        }
        if args.accepted.exists():
            accepted_after = _artifact_header(args.accepted)

    summary_errors.extend(compare_step["errors"])
    summary_errors.extend(promotion_step["errors"])

    status = "passed"
    requires_review = False
    if summary_errors or compare_step["status"] == "failed" or promotion_step["status"] == "failed":
        status = "failed"
    elif compare_status == "review":
        status = "review"
        requires_review = True

    report = _build_report(
        timestamp_utc=timestamp_utc,
        status=status,
        ok=status != "failed",
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

    if status == "failed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
