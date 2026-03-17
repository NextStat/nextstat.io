from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_GOVERNANCE = 20
EXIT_PERFORMANCE = 21
EXIT_INFRASTRUCTURE = 22

KIND_LABELS = {
    "success": "success",
    "governance": "governance",
    "performance": "performance",
    "infrastructure": "infrastructure",
}


def _parse_steps(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    steps: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        step_id, status, label = (line.split("\t", 2) + ["", ""])[:3]
        steps.append({"id": step_id, "status": status, "label": label or step_id})
    return steps


def _layer_status(steps: list[dict[str, str]]) -> str:
    if not steps:
        return "not_run"
    if any(step["status"] == "failed" for step in steps):
        return "failed"
    if any(step["status"] == "advisory" for step in steps):
        return "advisory"
    if all(step["status"] == "skipped" for step in steps):
        return "skipped"
    return "ok"


def _kind_from_exit_code(exit_code: int) -> str:
    if exit_code == EXIT_GOVERNANCE:
        return "governance"
    if exit_code == EXIT_PERFORMANCE:
        return "performance"
    if exit_code == EXIT_INFRASTRUCTURE:
        return "infrastructure"
    return "success"


def build_summary(
    *,
    governance_steps: list[dict[str, str]],
    performance_steps: list[dict[str, str]],
    exit_code: int,
    failure_step: str | None,
    message: str | None,
    version: str,
    release_tag: str,
) -> dict[str, Any]:
    failure_kind = _kind_from_exit_code(exit_code)
    return {
        "schema_version": "nextstat.apex2_pre_release_gate_summary.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "version": version,
        "release_tag": release_tag,
        "status": "ok" if exit_code == EXIT_OK else "failed",
        "has_advisories": any(
            step["status"] == "advisory" for step in governance_steps + performance_steps
        ),
        "exit_code": exit_code,
        "failure_kind": KIND_LABELS[failure_kind],
        "failure_step": failure_step,
        "message": message or "",
        "layers": {
            "governance": {
                "status": _layer_status(governance_steps),
                "steps": governance_steps,
            },
            "performance": {
                "status": _layer_status(performance_steps),
                "steps": performance_steps,
            },
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    governance = summary["layers"]["governance"]
    performance = summary["layers"]["performance"]
    lines = [
        "# Apex2 Pre-release Gate Summary",
        "",
        f"- Generated: `{summary['generated_at_utc']}`",
        f"- Version: `{summary['version']}`",
        f"- Release tag: `{summary['release_tag']}`",
        f"- Status: `{summary['status']}`",
        f"- Exit code: `{summary['exit_code']}`",
        f"- Failure kind: `{summary['failure_kind']}`",
    ]
    if summary.get("failure_step"):
        lines.append(f"- Failure step: `{summary['failure_step']}`")
    if summary.get("message"):
        lines.append(f"- Message: `{summary['message']}`")
    lines.extend(
        [
            "",
            "## Layer Status",
            "",
            "| Layer | Status |",
            "| --- | --- |",
            f"| Governance | `{governance['status']}` |",
            f"| Performance | `{performance['status']}` |",
            "",
            "## Governance Steps",
            "",
        ]
    )
    if governance["steps"]:
        for step in governance["steps"]:
            lines.append(f"- `{step['id']}`: `{step['status']}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Performance Steps", ""])
    if performance["steps"]:
        for step in performance["steps"]:
            lines.append(f"- `{step['id']}`: `{step['status']}`")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render an Apex2 pre-release gate summary.")
    parser.add_argument("--governance-steps", type=Path, required=True)
    parser.add_argument("--performance-steps", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--failure-step")
    parser.add_argument("--message")
    parser.add_argument("--version", required=True)
    parser.add_argument("--release-tag", required=True)
    args = parser.parse_args()

    summary = build_summary(
        governance_steps=_parse_steps(args.governance_steps),
        performance_steps=_parse_steps(args.performance_steps),
        exit_code=args.exit_code,
        failure_step=args.failure_step,
        message=args.message,
        version=args.version,
        release_tag=args.release_tag,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(render_markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
