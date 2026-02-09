#!/usr/bin/env python3
"""
Semantic drift comparator for NextStat validation_report.json.

Why this exists:
- snapshot_index/replication_report are hash-level tools; across commits they often differ due to provenance
  (git commit, timestamps, environment fields) and "noisy" perf artifacts.
- For CI drift detection we want a correctness-focused gate: suite/overall status regressions.

This script compares two validation reports and:
- writes a machine-readable summary JSON
- exits non-zero only when a regression is detected (configurable)

Stdlib-only by design so it can run anywhere (CI, minimal envs).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STATUS_SEVERITY: Dict[Optional[str], int] = {
    # None means "missing suite" (e.g., older reports) -> treat like skipped for severity purposes.
    None: 1,
    "ok": 0,
    "skipped": 1,
    "fail": 2,
    "error": 2,
}


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to read JSON: {path}: {e}") from e


def _get_nested(obj: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    cur: Any = obj
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _coerce_status(v: Any) -> Optional[str]:
    if v is None:
        return None
    if not isinstance(v, str):
        return None
    s = v.strip()
    if s == "":
        return None
    return s


@dataclass(frozen=True)
class ExtractedReport:
    path: str
    schema_version: Optional[str]
    deterministic: Optional[bool]
    workspace_sha256: Optional[str]
    master_report_sha256: Optional[str]
    overall: Optional[str]
    suites: Dict[str, Optional[str]]

    nextstat_version: Optional[str]
    nextstat_git_commit: Optional[str]
    python_version: Optional[str]
    platform: Optional[str]


def _extract(path: Path, raw: Dict[str, Any]) -> ExtractedReport:
    suites_raw = _get_nested(raw, ("apex2_summary", "suites"))
    suites: Dict[str, Optional[str]] = {}
    if isinstance(suites_raw, dict):
        for suite_name, suite_obj in suites_raw.items():
            if not isinstance(suite_name, str) or suite_name.strip() == "":
                continue
            status = None
            if isinstance(suite_obj, dict):
                status = _coerce_status(suite_obj.get("status"))
            suites[suite_name] = status

    env = raw.get("environment") if isinstance(raw.get("environment"), dict) else {}
    assert isinstance(env, dict)

    return ExtractedReport(
        path=str(path),
        schema_version=_coerce_status(raw.get("schema_version")),
        deterministic=raw.get("deterministic") if isinstance(raw.get("deterministic"), bool) else None,
        workspace_sha256=_coerce_status(_get_nested(raw, ("dataset_fingerprint", "workspace_sha256"))),
        master_report_sha256=_coerce_status(_get_nested(raw, ("apex2_summary", "master_report_sha256"))),
        overall=_coerce_status(_get_nested(raw, ("apex2_summary", "overall"))),
        suites=suites,
        nextstat_version=_coerce_status(env.get("nextstat_version")),
        nextstat_git_commit=_coerce_status(env.get("nextstat_git_commit")),
        python_version=_coerce_status(env.get("python_version")),
        platform=_coerce_status(env.get("platform")),
    )


def _severity(status: Optional[str]) -> int:
    s = status
    if s not in STATUS_SEVERITY:
        # Unknown status isn't in schema, but older PDFs used "unknown" for missing.
        return 1
    return STATUS_SEVERITY[s]


def _is_regression(baseline: Optional[str], current: Optional[str]) -> bool:
    return _severity(baseline) < _severity(current)


def _compare_reports(b: ExtractedReport, c: ExtractedReport) -> Dict[str, Any]:
    all_suites = sorted(set(b.suites.keys()) | set(c.suites.keys()))
    suite_changes: List[Dict[str, Any]] = []
    suite_regressions = 0

    for name in all_suites:
        bs = b.suites.get(name)
        cs = c.suites.get(name)
        changed = bs != cs
        regression = _is_regression(bs, cs)
        if regression:
            suite_regressions += 1
        if changed or regression:
            suite_changes.append(
                {
                    "suite": name,
                    "baseline_status": bs if bs is not None else "missing",
                    "current_status": cs if cs is not None else "missing",
                    "regression": regression,
                }
            )

    overall_regression = (b.overall == "pass") and (c.overall == "fail")
    deterministic_regression = (b.deterministic is True) and (c.deterministic is False)

    ok = (suite_regressions == 0) and (not overall_regression) and (not deterministic_regression)

    return {
        "schema_version": "validation_drift_summary_v1",
        "ok": ok,
        "regressions": {
            "overall": overall_regression,
            "deterministic": deterministic_regression,
            "suite_statuses": suite_regressions,
        },
        "baseline": {
            "path": b.path,
            "schema_version": b.schema_version,
            "deterministic": b.deterministic,
            "workspace_sha256": b.workspace_sha256,
            "master_report_sha256": b.master_report_sha256,
            "overall": b.overall,
            "environment": {
                "nextstat_version": b.nextstat_version,
                "nextstat_git_commit": b.nextstat_git_commit,
                "python_version": b.python_version,
                "platform": b.platform,
            },
        },
        "current": {
            "path": c.path,
            "schema_version": c.schema_version,
            "deterministic": c.deterministic,
            "workspace_sha256": c.workspace_sha256,
            "master_report_sha256": c.master_report_sha256,
            "overall": c.overall,
            "environment": {
                "nextstat_version": c.nextstat_version,
                "nextstat_git_commit": c.nextstat_git_commit,
                "python_version": c.python_version,
                "platform": c.platform,
            },
        },
        "suite_changes": suite_changes,
    }


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(
        description="Compare two validation_report.json files semantically (suite status regressions)."
    )
    p.add_argument("--baseline", type=Path, required=True, help="Baseline validation_report.json")
    p.add_argument("--current", type=Path, required=True, help="Current validation_report.json")
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for validation_drift_summary.json",
    )
    p.add_argument(
        "--meta",
        action="append",
        default=[],
        help="Optional metadata entries (repeatable) in KEY=VALUE form.",
    )
    p.add_argument(
        "--fail-on-regression",
        dest="fail_on_regression",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit non-zero when a semantic regression is detected (default: true).",
    )
    args = p.parse_args(argv)

    meta: Dict[str, str] = {}
    for item in args.meta:
        if "=" not in item:
            raise SystemExit(f"--meta must be KEY=VALUE, got: {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise SystemExit(f"--meta key must be non-empty, got: {item!r}")
        meta[k] = v

    # Baseline can be missing (first run / older artifact); treat as non-blocking, but still emit a summary.
    if not args.baseline.exists():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "validation_drift_summary_v1",
            "ok": True,
            "skipped": True,
            "reason": f"baseline_missing: {args.baseline}",
        }
        if meta:
            payload["metadata"] = meta
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(payload["reason"], file=sys.stderr)
        return 0

    b_raw = _read_json(args.baseline)
    c_raw = _read_json(args.current)
    b = _extract(args.baseline, b_raw)
    c = _extract(args.current, c_raw)

    summary = _compare_reports(b, c)
    if meta:
        summary["metadata"] = meta
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if summary.get("ok") is True:
        return 0

    if args.fail_on_regression:
        # Provide a compact log line for CI.
        regs = summary.get("regressions", {})
        print(
            f"validation drift regression detected: overall={regs.get('overall')} "
            f"deterministic={regs.get('deterministic')} suite_statuses={regs.get('suite_statuses')}",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
