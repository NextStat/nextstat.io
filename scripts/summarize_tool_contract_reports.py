#!/usr/bin/env python3
"""Aggregate tool-contract runner reports into a compact dashboard.

Usage:
  python scripts/summarize_tool_contract_reports.py \
    --report tmp/reports/tool_contracts_fast_report.json \
    --report tmp/reports/tool_contracts_live_report.json \
    --out-json tmp/reports/tool_contract_dashboard.json \
    --out-md tmp/reports/tool_contract_dashboard.md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INPUT_SCHEMA_VERSION = "nextstat.tool_contract_runner_report.v1"
OUTPUT_SCHEMA_VERSION = "nextstat.tool_contract_dashboard.v1"
STATUS_ORDER = ("planned", "passed", "failed")
MODE_ORDER = ("fast", "live", "all")
FAILURE_CLASSIFICATION_ORDER = (
    "none",
    "schema_drift",
    "performance_budget_failure",
    "rust_contract_failure",
    "python_contract_failure",
    "live_server_failure",
    "unknown",
)
SEVERITY_ORDER = ("none", "high", "critical")
PERF_STATUS_ORDER = ("planned", "within_budget", "exceeded", "not_available")
PERF_STEP_LABEL = "Validate tool-contract performance budgets"

SCHEMA_DRIFT_LABELS = {
    "Check tool contract schemas",
    "Validate tool manifest",
    "Validate local tool discovery descriptor",
    "Check tool discovery descriptor examples",
    "Check tool reference docs",
    "Check tool goldens",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _validate_report_shape(path: Path, report: dict[str, Any]) -> None:
    if report.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise SystemExit(f"{path}: unsupported schema_version {report.get('schema_version')!r}")

    required_fields = {
        "mode",
        "dry_run",
        "status",
        "overall_pass",
        "step_count",
        "steps",
        "duration_s",
        "failed_step_index",
        "failed_step_label",
        "performance",
    }
    missing = sorted(field for field in required_fields if field not in report)
    if missing:
        raise SystemExit(f"{path}: missing required fields: {', '.join(missing)}")

    steps = report.get("steps")
    if not isinstance(steps, list):
        raise SystemExit(f"{path}: steps must be a list")
    if int(report.get("step_count")) != len(steps):
        raise SystemExit(
            f"{path}: step_count={report.get('step_count')} does not match len(steps)={len(steps)}"
        )
    if str(report.get("mode")) not in MODE_ORDER:
        raise SystemExit(f"{path}: unsupported mode {report.get('mode')!r}")
    if str(report.get("status")) not in {"planned", "passed", "failed"}:
        raise SystemExit(f"{path}: unsupported status {report.get('status')!r}")
    for index, step in enumerate(steps, start=1):
        step_status = str(step.get("status"))
        if step_status not in {"planned", "passed", "failed"}:
            raise SystemExit(f"{path}: step {index} has unsupported status {step_status!r}")


def _status_counts(items: list[str], *, known: tuple[str, ...]) -> dict[str, int]:
    counts = {name: 0 for name in known}
    for item in items:
        counts[str(item)] = counts.get(str(item), 0) + 1
    return counts


def _classify_failed_step(step: dict[str, Any]) -> dict[str, str]:
    label = str(step.get("label") or "")
    if label in SCHEMA_DRIFT_LABELS:
        return {
            "code": "schema_drift",
            "severity": "high",
            "reason": "Contract drift detected in manifest/schema/docs/golden validation.",
        }
    if label == PERF_STEP_LABEL:
        return {
            "code": "performance_budget_failure",
            "severity": "high",
            "reason": "Tool-contract performance budget validation failed.",
        }
    if label == "Run ns-server tool contract tests":
        return {
            "code": "rust_contract_failure",
            "severity": "critical",
            "reason": "Rust ns-server contract test lane failed.",
        }
    if label == "Run fast Python tool contract suite":
        return {
            "code": "python_contract_failure",
            "severity": "high",
            "reason": "Python tool-contract suite failed.",
        }
    if label == "Run live nextstat-server tool contract suite":
        return {
            "code": "live_server_failure",
            "severity": "critical",
            "reason": "Live auth-enabled nextstat-server contract suite failed.",
        }
    return {
        "code": "unknown",
        "severity": "high",
        "reason": "Failure did not match a known tool-contract incident class.",
    }


def _normalize_run(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    step_status_counts = _status_counts([str(step.get("status")) for step in report["steps"]], known=STATUS_ORDER)
    failed_steps = [
        {
            "index": int(step["index"]),
            "label": str(step["label"]),
            "command": str(step["command"]),
            "env_overrides": dict(step.get("env_overrides") or {}),
            "returncode": step.get("returncode"),
            "stdout_tail": step.get("stdout_tail"),
            "stderr_tail": step.get("stderr_tail"),
            "classification": _classify_failed_step(step),
        }
        for step in report["steps"]
        if str(step.get("status")) == "failed"
    ]
    failure_classification = (
        failed_steps[0]["classification"]
        if failed_steps
        else {
            "code": "none",
            "severity": "none",
            "reason": "No failed steps in this report.",
        }
    )
    performance = dict(report["performance"])
    return {
        "report_path": str(path),
        "mode": str(report["mode"]),
        "dry_run": bool(report["dry_run"]),
        "status": str(report["status"]),
        "overall_pass": bool(report["overall_pass"]),
        "step_count": int(report["step_count"]),
        "duration_s": float(report["duration_s"]),
        "failed_step_index": report.get("failed_step_index"),
        "failed_step_label": report.get("failed_step_label"),
        "step_status_counts": step_status_counts,
        "failed_steps": failed_steps,
        "failure_classification": failure_classification,
        "performance": performance,
    }


def _overall_status(runs: list[dict[str, Any]]) -> str:
    statuses = {str(run["status"]) for run in runs}
    if "failed" in statuses:
        return "failed"
    if statuses == {"planned"}:
        return "planned"
    if statuses == {"passed"}:
        return "passed"
    return "mixed"


def _render_markdown(dashboard: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Tool Contract Dashboard")
    lines.append("")
    lines.append(f"- overall status: `{dashboard['overall_status']}`")
    lines.append(f"- overall pass: `{str(dashboard['overall_pass']).lower()}`")
    lines.append(f"- report count: `{dashboard['report_count']}`")
    lines.append(f"- total steps: `{dashboard['totals']['step_count']}`")
    lines.append(f"- total duration_s: `{dashboard['totals']['duration_s']}`")
    mode_counts = dashboard["totals"]["mode_counts"]
    lines.append(
        "- mode counts: "
        + ", ".join(f"`{mode}={mode_counts.get(mode, 0)}`" for mode in MODE_ORDER if mode_counts.get(mode, 0))
    )
    failure_counts = dashboard["totals"]["failure_classification_counts"]
    non_zero_failure_counts = [
        f"`{code}={failure_counts.get(code, 0)}`"
        for code in FAILURE_CLASSIFICATION_ORDER
        if failure_counts.get(code, 0)
    ]
    if non_zero_failure_counts:
        lines.append("- failure classes: " + ", ".join(non_zero_failure_counts))
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    lines.append("| Report | Mode | Dry run | Status | Classification | Severity | Steps | Duration (s) | Failed step |")
    lines.append("|--------|------|---------|--------|----------------|----------|-------|--------------|-------------|")
    for run in dashboard["runs"]:
        failed_step = str(run["failed_step_label"] or "-")
        classification = run["failure_classification"]
        lines.append(
            f"| `{run['report_path']}` | `{run['mode']}` | `{str(run['dry_run']).lower()}` | "
            f"`{run['status']}` | `{classification['code']}` | `{classification['severity']}` | "
            f"`{run['step_count']}` | `{run['duration_s']}` | `{failed_step}` |"
        )

    lines.append("")
    lines.append("## Performance")
    lines.append("")
    lines.append(
        "- runner budget statuses: "
        + ", ".join(
            f"`{status}={dashboard['totals']['runner_budget_status_counts'][status]}`"
            for status in PERF_STATUS_ORDER
            if dashboard["totals"]["runner_budget_status_counts"][status]
        )
    )
    lines.append(
        "- live metrics statuses: "
        + ", ".join(
            f"`{status}={dashboard['totals']['live_metrics_budget_status_counts'][status]}`"
            for status in PERF_STATUS_ORDER
            if dashboard["totals"]["live_metrics_budget_status_counts"][status]
        )
    )
    lines.append("")
    lines.append("| Report | Runner Budget | Live Metrics | Runner Duration / Max |")
    lines.append("|--------|---------------|--------------|-----------------------|")
    for run in dashboard["runs"]:
        performance = run["performance"]
        runner_budget = performance["runner_budget"]
        live_metrics_budget = performance["live_metrics_budget"]
        lines.append(
            f"| `{run['report_path']}` | `{runner_budget['status']}` | `{live_metrics_budget['status']}` | "
            f"`{runner_budget['actual_total_duration_s']}/{runner_budget['max_total_duration_s']}` |"
        )

    lines.append("")
    lines.append("## Step Totals")
    lines.append("")
    for status in STATUS_ORDER:
        lines.append(f"- `{status}`: `{dashboard['totals']['step_status_counts'][status]}`")

    failed_runs = dashboard["totals"]["failed_report_paths"]
    if failed_runs:
        lines.append("")
        lines.append("## Failed Reports")
        lines.append("")
        for path in failed_runs:
            lines.append(f"- `{path}`")

    failed_step_runs = [run for run in dashboard["runs"] if run["failed_steps"]]
    if failed_step_runs:
        lines.append("")
        lines.append("## Failure Drilldown")
        lines.append("")
        for run in failed_step_runs:
            lines.append(f"### `{run['report_path']}`")
            lines.append("")
            for step in run["failed_steps"]:
                classification = step["classification"]
                lines.append(
                    f"- failed step: `#{step['index']} {step['label']}` (returncode=`{step['returncode']}`)"
                )
                lines.append(
                    f"- classification: `{classification['code']}` (severity=`{classification['severity']}`)"
                )
                lines.append(f"- reason: `{classification['reason']}`")
                lines.append(f"- command: `{step['command']}`")
                if step["env_overrides"]:
                    lines.append(f"- env_overrides: `{json.dumps(step['env_overrides'], sort_keys=True)}`")
                if step["stdout_tail"]:
                    lines.append("- stdout tail:")
                    lines.append("```text")
                    lines.append(str(step["stdout_tail"]))
                    lines.append("```")
                if step["stderr_tail"]:
                    lines.append("- stderr tail:")
                    lines.append("```text")
                    lines.append(str(step["stderr_tail"]))
                    lines.append("```")
                lines.append("")

    return "\n".join(lines) + "\n"


def build_dashboard(report_paths: list[Path]) -> dict[str, Any]:
    if not report_paths:
        raise SystemExit("at least one --report path is required")

    runs: list[dict[str, Any]] = []
    source_reports: list[str] = []
    for path in report_paths:
        report = _read_json(path)
        if not isinstance(report, dict):
            raise SystemExit(f"{path}: expected a JSON object")
        _validate_report_shape(path, report)
        runs.append(_normalize_run(path, report))
        source_reports.append(str(path))

    totals = {
        "duration_s": round(sum(float(run["duration_s"]) for run in runs), 6),
        "step_count": sum(int(run["step_count"]) for run in runs),
        "step_status_counts": _status_counts(
            [status for run in runs for status, count in run["step_status_counts"].items() for _ in range(int(count))],
            known=STATUS_ORDER,
        ),
        "mode_counts": _status_counts([str(run["mode"]) for run in runs], known=MODE_ORDER),
        "failed_report_paths": [str(run["report_path"]) for run in runs if run["status"] == "failed"],
        "failed_step_count": sum(len(run["failed_steps"]) for run in runs),
        "failure_classification_counts": _status_counts(
            [str(run["failure_classification"]["code"]) for run in runs],
            known=FAILURE_CLASSIFICATION_ORDER,
        ),
        "severity_counts": _status_counts(
            [str(run["failure_classification"]["severity"]) for run in runs],
            known=SEVERITY_ORDER,
        ),
        "runner_budget_status_counts": _status_counts(
            [str(run["performance"]["runner_budget"]["status"]) for run in runs],
            known=PERF_STATUS_ORDER,
        ),
        "live_metrics_budget_status_counts": _status_counts(
            [str(run["performance"]["live_metrics_budget"]["status"]) for run in runs],
            known=PERF_STATUS_ORDER,
        ),
    }

    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "generated_at": _utc_now_iso(),
        "report_count": len(runs),
        "source_reports": source_reports,
        "overall_status": _overall_status(runs),
        "overall_pass": all(bool(run["overall_pass"]) for run in runs),
        "runs": runs,
        "totals": totals,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", type=Path, required=True, help="input runner report (repeatable)")
    parser.add_argument("--out-json", type=Path, required=True, help="output dashboard JSON path")
    parser.add_argument("--out-md", type=Path, required=True, help="output dashboard markdown path")
    args = parser.parse_args(argv)

    dashboard = build_dashboard(args.report)
    _write_json(args.out_json, dashboard)
    _write_text(args.out_md, _render_markdown(dashboard))
    print(f"Wrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
