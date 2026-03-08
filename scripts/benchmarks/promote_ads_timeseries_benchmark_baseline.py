#!/usr/bin/env python3
"""Promote a vetted ads + weekly time-series benchmark artifact to the accepted baseline."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "nextstat.ads_timeseries_benchmark_baseline_promotion_report.v1"
COMPARE_SCHEMA_VERSION = "nextstat.ads_timeseries_benchmark_compare_report.v1"
SUITE = "ads_timeseries_surface"
DEFAULT_ACCEPTED = (
    REPO_ROOT / "benchmarks" / "artifacts" / "ads_timeseries_baselines" / "nextstat-bench" / "accepted.json"
)
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "benchmarks" / "compare_ads_timeseries_benchmark.py"


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


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def _empty_artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
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


def _run_compare(*, accepted: Path, current: Path, compare_report: Path) -> tuple[int, dict[str, Any]]:
    cmd = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--baseline",
        str(accepted),
        "--current",
        str(current),
        "--out",
        str(compare_report),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    report = _read_json(compare_report)
    return proc.returncode, report


def _promotion_report(
    *,
    status: str,
    promoted: bool,
    allow_review: bool,
    dry_run: bool,
    accepted: Path,
    candidate: Path,
    compare_report: Path,
    compare_status: str | None,
    actions: dict[str, Any],
    errors: list[str],
    accepted_before: dict[str, Any],
    candidate_header: dict[str, Any],
    accepted_after: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "suite": SUITE,
        "status": status,
        "promoted": promoted,
        "allow_review": allow_review,
        "dry_run": dry_run,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "accepted_baseline_path": str(accepted),
        "candidate_path": str(candidate),
        "compare_report_path": str(compare_report),
        "compare_status": compare_status,
        "artifacts": {
            "accepted_before": accepted_before,
            "candidate": candidate_header,
            "accepted_after": accepted_after,
        },
        "actions": actions,
        "summary": {
            "top_level_errors": errors,
        },
    }


def _is_allowed_case_set_change(compare_doc: dict[str, Any]) -> bool:
    if str(compare_doc.get("status")) != "failed":
        return False

    summary = compare_doc.get("summary")
    top_level_errors = summary.get("top_level_errors") if isinstance(summary, dict) else None
    if top_level_errors != ["case_set_mismatch"]:
        return False

    cases = compare_doc.get("cases")
    if not isinstance(cases, list) or not cases:
        return False

    saw_new_case = False
    for case in cases:
        if not isinstance(case, dict):
            return False
        errors = case.get("errors")
        if not isinstance(errors, list):
            return False
        if "current_case_missing" in errors:
            return False
        if errors == ["baseline_case_missing"]:
            saw_new_case = True
            continue
        if errors:
            return False
    return saw_new_case


def _compare_has_review_cases(compare_doc: dict[str, Any]) -> bool:
    cases = compare_doc.get("cases")
    if not isinstance(cases, list):
        return False
    return any(isinstance(case, dict) and case.get("status") == "review" for case in cases)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Promote a vetted ads + weekly time-series benchmark artifact to the accepted baseline.")
    parser.add_argument("--current", type=Path, required=True, help="Current benchmark artifact to promote")
    parser.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED, help=f"Accepted baseline path (default: {DEFAULT_ACCEPTED})")
    parser.add_argument("--compare-report", type=Path, default=None, help="Comparison report path; defaults to <current-dir>/compare_report.json")
    parser.add_argument("--report", type=Path, default=None, help="Promotion report path; defaults to <current-dir>/promotion_report.json")
    parser.add_argument("--history-dir", type=Path, default=None, help="History directory; defaults to <accepted-dir>/history")
    parser.add_argument(
        "--allow-review",
        dest="allow_review",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow promotion when compare status=review (default: false)",
    )
    parser.add_argument(
        "--allow-case-set-change",
        dest="allow_case_set_change",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow promotion when compare fails only because the candidate adds new required benchmark cases",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate compare/promotion inputs without modifying the accepted baseline")
    args = parser.parse_args(argv)

    compare_report = args.compare_report or args.current.parent / "compare_report.json"
    promotion_report = args.report or args.current.parent / "promotion_report.json"
    history_dir = args.history_dir or args.accepted.parent / "history"

    errors: list[str] = []
    accepted_before = _empty_artifact(args.accepted)
    candidate_header = _empty_artifact(args.current)
    accepted_after = _empty_artifact(args.accepted)
    actions = {
        "archived_previous_baseline": False,
        "archived_previous_baseline_path": None,
        "archived_promoted_snapshot": False,
        "archived_promoted_snapshot_path": None,
        "accepted_updated": False,
        "history_dir": str(history_dir),
    }
    compare_status = None

    if not args.current.exists():
        errors.append(f"current_missing:{args.current}")
        report = _promotion_report(
            status="failed",
            promoted=False,
            allow_review=bool(args.allow_review),
            dry_run=bool(args.dry_run),
            accepted=args.accepted,
            candidate=args.current,
            compare_report=compare_report,
            compare_status=compare_status,
            actions=actions,
            errors=errors,
            accepted_before=accepted_before,
            candidate_header=candidate_header,
            accepted_after=accepted_after,
        )
        _write_json(promotion_report, report)
        return 2

    if not args.accepted.exists():
        errors.append(f"accepted_missing:{args.accepted}")
        candidate_header = _artifact_header(args.current)
        report = _promotion_report(
            status="failed",
            promoted=False,
            allow_review=bool(args.allow_review),
            dry_run=bool(args.dry_run),
            accepted=args.accepted,
            candidate=args.current,
            compare_report=compare_report,
            compare_status=compare_status,
            actions=actions,
            errors=errors,
            accepted_before=accepted_before,
            candidate_header=candidate_header,
            accepted_after=accepted_after,
        )
        _write_json(promotion_report, report)
        return 2

    accepted_before = _artifact_header(args.accepted)
    accepted_after = accepted_before
    candidate_header = _artifact_header(args.current)

    compare_exit, compare_doc = _run_compare(
        accepted=args.accepted,
        current=args.current,
        compare_report=compare_report,
    )
    compare_status = str(compare_doc.get("status")) if isinstance(compare_doc, dict) else None
    if compare_doc.get("schema_version") != COMPARE_SCHEMA_VERSION:
        errors.append(f"unexpected_compare_schema:{compare_doc.get('schema_version')}")
    if compare_doc.get("suite") != SUITE:
        errors.append(f"unexpected_compare_suite:{compare_doc.get('suite')}")
    if compare_doc.get("baseline_path") != str(args.accepted):
        errors.append("compare_baseline_path_mismatch")
    if compare_doc.get("current_path") != str(args.current):
        errors.append("compare_current_path_mismatch")
    if compare_exit not in {0, 2}:
        errors.append(f"compare_runner_failed:{compare_exit}")
    allowed_case_set_change = bool(args.allow_case_set_change) and _is_allowed_case_set_change(compare_doc)
    if compare_status == "failed" and not allowed_case_set_change:
        errors.append("compare_status_failed")
    if compare_status == "review" and not args.allow_review:
        errors.append("compare_status_review_requires_allow_review")
    if allowed_case_set_change and _compare_has_review_cases(compare_doc) and not args.allow_review:
        errors.append("case_set_change_contains_review_requires_allow_review")
    if args.current.resolve() == args.accepted.resolve() and not args.dry_run:
        errors.append("candidate_equals_accepted_baseline")

    if errors:
        report = _promotion_report(
            status="failed",
            promoted=False,
            allow_review=bool(args.allow_review),
            dry_run=bool(args.dry_run),
            accepted=args.accepted,
            candidate=args.current,
            compare_report=compare_report,
            compare_status=compare_status,
            actions=actions,
            errors=errors,
            accepted_before=accepted_before,
            candidate_header=candidate_header,
            accepted_after=accepted_after,
        )
        _write_json(promotion_report, report)
        return 2

    if args.dry_run:
        accepted_after = candidate_header
        report = _promotion_report(
            status="dry_run",
            promoted=False,
            allow_review=bool(args.allow_review),
            dry_run=True,
            accepted=args.accepted,
            candidate=args.current,
            compare_report=compare_report,
            compare_status=compare_status,
            actions=actions,
            errors=errors,
            accepted_before=accepted_before,
            candidate_header=candidate_header,
            accepted_after=accepted_after,
        )
        _write_json(promotion_report, report)
        return 0

    history_dir.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp_utc()
    previous_archive = history_dir / f"accepted_{stamp}_previous.json"
    promoted_archive = history_dir / f"accepted_{stamp}_promoted.json"

    shutil.copy2(args.accepted, previous_archive)
    actions["archived_previous_baseline"] = True
    actions["archived_previous_baseline_path"] = str(previous_archive)

    shutil.copy2(args.current, args.accepted)
    actions["accepted_updated"] = True

    shutil.copy2(args.accepted, promoted_archive)
    actions["archived_promoted_snapshot"] = True
    actions["archived_promoted_snapshot_path"] = str(promoted_archive)

    accepted_after = _artifact_header(args.accepted)
    report = _promotion_report(
        status="promoted",
        promoted=True,
        allow_review=bool(args.allow_review),
        dry_run=False,
        accepted=args.accepted,
        candidate=args.current,
        compare_report=compare_report,
        compare_status=compare_status,
        actions=actions,
        errors=errors,
        accepted_before=accepted_before,
        candidate_header=candidate_header,
        accepted_after=accepted_after,
    )
    _write_json(promotion_report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
