#!/usr/bin/env python3
"""One-shot ads variance-reduction benchmark gate: run, compare, and optionally promote."""

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
SCHEMA_VERSION = "nextstat.ads_variance_reduction_benchmark_gate_report.v1"
COMPARE_SCHEMA_VERSION = "nextstat.ads_variance_reduction_benchmark_compare_report.v1"
PROMOTION_SCHEMA_VERSION = "nextstat.ads_variance_reduction_benchmark_baseline_promotion_report.v1"
SUITE = "ads_variance_reduction_matrix"
DEFAULT_ACCEPTED = (
    REPO_ROOT
    / "benchmarks"
    / "artifacts"
    / "ads_variance_reduction_baselines"
    / "nextstat-bench"
    / "accepted.json"
)
DEFAULT_REMOTE_RUNNER = "bash scripts/benchmarks/bench_ads_variance_reduction_matrix_remote.sh"
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "benchmarks" / "compare_ads_variance_reduction_benchmark.py"
PROMOTE_SCRIPT = REPO_ROOT / "scripts" / "benchmarks" / "promote_ads_variance_reduction_benchmark_baseline.py"
REMOTE_DONE_PREFIX = "[ads-vr-remote] done: "


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
) -> tuple[dict[str, Any], Path | None]:
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
        )

    if runner_cmd is None:
        runner_cmd = DEFAULT_REMOTE_RUNNER
    exit_code, argv, output = _run_runner(runner_cmd)
    artifact_dir = _parse_remote_artifact_dir(output)
    artifact_path = None if artifact_dir is None else artifact_dir / "ads_variance_reduction_benchmark.json"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED, help="Accepted baseline artifact")
    parser.add_argument("--current", type=Path, default=None, help="Use an existing current artifact instead of running the remote runner")
    parser.add_argument("--compare-report", type=Path, default=None, help="Comparison report path")
    parser.add_argument("--promotion-report", type=Path, default=None, help="Promotion report path")
    parser.add_argument("--report", type=Path, default=None, help="Gate report path")
    parser.add_argument("--history-dir", type=Path, default=None, help="History directory for accepted baseline snapshots")
    parser.add_argument("--runner-cmd", type=str, default=None, help="Override the remote runner command")
    parser.add_argument(
        "--promotion-mode",
        choices=("none", "dry_run", "apply"),
        default="none",
        help="Promotion behavior after compare",
    )
    parser.add_argument(
        "--allow-review",
        dest="allow_review",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow promotion when compare status=review",
    )
    args = parser.parse_args(argv)

    compare_report = args.compare_report or (
        args.current.parent / "compare_report.json"
        if args.current is not None
        else REPO_ROOT / "bench_results" / "ads_variance_reduction_benchmark" / "compare_report.json"
    )
    gate_report = args.report or (
        args.current.parent / "gate_report.json"
        if args.current is not None
        else REPO_ROOT / "bench_results" / "ads_variance_reduction_benchmark" / "gate_report.json"
    )
    promotion_report = None
    if args.promotion_mode != "none":
        promotion_report = args.promotion_report or (
            args.current.parent / "promotion_report.json"
            if args.current is not None
            else REPO_ROOT / "bench_results" / "ads_variance_reduction_benchmark" / "promotion_report.json"
        )
    history_dir = args.history_dir or args.accepted.parent / "history"

    benchmark_step, current_artifact = _step_benchmark(current=args.current, runner_cmd=args.runner_cmd)
    timestamp_utc = _timestamp_utc()
    summary_errors: list[str] = []
    compare_step = {
        "status": "skipped",
        "ok": False,
        "exit_code": None,
        "command": [],
        "errors": [],
    }
    promotion_step = {
        "status": "skipped",
        "ok": False,
        "exit_code": None,
        "command": [],
        "errors": [],
    }

    accepted_before = _empty_artifact(args.accepted)
    accepted_after = _empty_artifact(args.accepted)
    current_header = _empty_artifact(current_artifact)

    if current_artifact is None or not current_artifact.exists():
        summary_errors.append("missing_current_artifact")
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

    accepted_before = _artifact_header(args.accepted) if args.accepted.exists() else _empty_artifact(args.accepted)
    accepted_after = accepted_before
    current_header = _artifact_header(current_artifact)

    compare_exit, compare_doc, compare_cmd = _run_compare(args.accepted, current_artifact, compare_report)
    compare_status = str(compare_doc.get("status")) if isinstance(compare_doc, dict) else "failed"
    compare_step = {
        "status": compare_status,
        "ok": compare_exit == 0,
        "exit_code": compare_exit,
        "command": compare_cmd,
        "errors": [] if compare_exit == 0 else [f"compare_exit:{compare_exit}"],
    }

    requires_review = compare_status == "review"
    status = "failed" if compare_status == "failed" else "review" if compare_status == "review" else "passed"
    ok = status == "passed"
    if isinstance(compare_doc, dict):
        if compare_doc.get("schema_version") != COMPARE_SCHEMA_VERSION:
            summary_errors.append(f"unexpected_compare_schema:{compare_doc.get('schema_version')}")
            status = "failed"
            ok = False
        if compare_doc.get("suite") != SUITE:
            summary_errors.append(f"unexpected_compare_suite:{compare_doc.get('suite')}")
            status = "failed"
            ok = False

    if args.promotion_mode != "none" and promotion_report is not None:
        dry_run = args.promotion_mode == "dry_run"
        promote_exit, promote_doc, promote_cmd = _run_promote(
            accepted=args.accepted,
            current=current_artifact,
            compare_report=compare_report,
            promotion_report=promotion_report,
            history_dir=history_dir,
            allow_review=bool(args.allow_review),
            dry_run=dry_run,
        )
        promotion_status = str(promote_doc.get("status")) if isinstance(promote_doc, dict) else "failed"
        promotion_step = {
            "status": promotion_status,
            "ok": promote_exit == 0,
            "exit_code": promote_exit,
            "command": promote_cmd,
            "errors": [] if promote_exit == 0 else [f"promotion_exit:{promote_exit}"],
        }
        if args.promotion_mode == "apply" and promote_exit == 0 and args.accepted.exists():
            accepted_after = _artifact_header(args.accepted)
        if promote_exit != 0:
            summary_errors.append(f"promotion_exit:{promote_exit}")
            status = "failed"
            ok = False
        if isinstance(promote_doc, dict):
            if promote_doc.get("schema_version") != PROMOTION_SCHEMA_VERSION:
                summary_errors.append(f"unexpected_promotion_schema:{promote_doc.get('schema_version')}")
                status = "failed"
                ok = False
            if promote_doc.get("suite") != SUITE:
                summary_errors.append(f"unexpected_promotion_suite:{promote_doc.get('suite')}")
                status = "failed"
                ok = False

    if compare_exit != 0 and compare_status != "review":
        summary_errors.append(f"compare_exit:{compare_exit}")
        status = "failed"
        ok = False

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
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
