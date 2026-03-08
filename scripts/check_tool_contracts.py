#!/usr/bin/env python3
"""Canonical runner for NextStat tool-contract checks.

Usage:
  python scripts/check_tool_contracts.py --mode fast
  python scripts/check_tool_contracts.py --mode live
  python scripts/check_tool_contracts.py --mode all
  python scripts/check_tool_contracts.py --mode fast --dry-run
  python scripts/check_tool_contracts.py --mode fast --report-json tmp/reports/tool_contracts_fast_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from tool_contract_performance_budget import (
    SCHEMA_VERSION as PERF_BUDGET_SCHEMA_VERSION,
    load_tool_contract_performance_budget,
    tool_contract_performance_budget_path,
    tool_contract_performance_budget_schema_path,
)


FAST_PYTEST_TARGETS = [
    "tests/python/test_tool_goldens.py",
    "tests/python/test_tool_result_schema_smoke.py",
    "tests/python/test_bindings_api.py",
    "tests/python/test_tools_contract_runtime.py",
    "tests/python/test_tools_server_transport.py",
    "tests/python/test_physics_assistant_server_only.py",
    "tests/python/test_tooling_scripts_server_mode.py",
    "tests/python/test_agent_bootstrap_packs.py",
    "tests/python/test_tool_contract_runner.py",
    "tests/python/test_tool_contract_dashboard.py",
    "tests/python/test_tool_contract_performance_budget.py",
    "tests/python/test_tool_contract_artifact_manifest.py",
    "tests/python/test_tool_contract_workflow.py",
]

LIVE_PYTEST_TARGETS = [
    "tests/python/test_tools_live_server_integration.py",
]

PERF_STEP_LABEL = "Validate tool-contract performance budgets"
PERF_STEP_COMMAND = "internal:validate-performance-budgets"
PERF_STATUS_ORDER = ("planned", "within_budget", "exceeded", "not_available")
LIVE_METRICS_SCHEMA_VERSION = "nextstat.tool_contract_live_metrics.v1"


@dataclass(frozen=True)
class Step:
    label: str
    argv: list[str]
    env_overrides: dict[str, str] | None = None


@dataclass(frozen=True)
class StepResult:
    index: int
    label: str
    argv: list[str]
    command: str
    env_overrides: dict[str, str]
    status: str
    returncode: int | None
    started_at: str | None
    finished_at: str | None
    duration_s: float | None
    stdout_tail: str | None
    stderr_tail: str | None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _python_argv() -> list[str]:
    return [sys.executable]


def _tool_contract_cargo_target_dir() -> str:
    override = os.environ.get("NEXTSTAT_TOOL_CONTRACT_CARGO_TARGET_DIR")
    if override:
        return override
    return str(_repo_root().parent / ".nextstat-cargo-target" / "tool-contracts")


def _tool_contract_bindings_cargo_target_dir() -> str:
    override = os.environ.get("NEXTSTAT_TOOL_CONTRACT_BINDINGS_CARGO_TARGET_DIR")
    if override:
        return override
    return str(_repo_root().parent / ".nextstat-cargo-target" / "tool-contracts-bindings")


def _fast_steps() -> list[Step]:
    py = _python_argv()
    return [
        Step("Check tool contract schemas", py + ["scripts/generate_tool_contract_schemas.py", "--check"]),
        Step("Validate tool manifest", py + ["scripts/validate_tool_manifest.py"]),
        Step(
            "Validate local tool discovery descriptor",
            py + ["scripts/validate_tool_schema_descriptor.py", "--transport", "local"],
        ),
        Step(
            "Check tool discovery descriptor examples",
            py + ["scripts/generate_tool_schema_examples.py", "--check"],
        ),
        Step("Check tool reference docs", py + ["scripts/generate_tool_reference_docs.py", "--check"]),
        Step(
            "Check agent bootstrap profile manifest schema",
            py + ["scripts/generate_agent_bootstrap_profile_manifest_schema.py", "--check"],
        ),
        Step("Check agent bootstrap packs", py + ["-m", "scripts.generate_agent_bootstrap_packs", "--check"]),
        Step("Check tool goldens", py + ["scripts/generate_tool_goldens.py", "--check"]),
        Step(
            "Run ns-server tool contract tests",
            ["cargo", "test", "-p", "ns-server", "--", "--test-threads=1"],
            env_overrides={"CARGO_TARGET_DIR": _tool_contract_cargo_target_dir()},
        ),
        Step(
            "Sync nextstat Python bindings into active environment",
            py + ["-m", "maturin", "develop", "-m", "bindings/ns-py/Cargo.toml"],
            env_overrides={"CARGO_TARGET_DIR": _tool_contract_bindings_cargo_target_dir()},
        ),
        Step("Run fast Python tool contract suite", py + ["-m", "pytest", "-q", *FAST_PYTEST_TARGETS]),
    ]


def _default_live_metrics_report_path(mode: str, report_json: Path | None) -> Path:
    if report_json is not None:
        return report_json.with_name(f"{report_json.stem}_live_metrics.json")
    return _repo_root() / "tmp" / "reports" / f"tool_contracts_{mode}_live_metrics_{os.getpid()}.json"


def _live_steps(live_metrics_path: Path | None) -> list[Step]:
    py = _python_argv()
    env_overrides = {"NS_RUN_LIVE_SERVER": "1"}
    if live_metrics_path is not None:
        env_overrides["NEXTSTAT_TOOL_CONTRACT_LIVE_METRICS_PATH"] = str(live_metrics_path)
    return [
        Step(
            "Run live nextstat-server tool contract suite",
            py + ["-m", "pytest", "-q", *LIVE_PYTEST_TARGETS],
            env_overrides=env_overrides,
        )
    ]


def _steps_for_mode(mode: str, *, live_metrics_path: Path | None) -> list[Step]:
    if mode == "fast":
        return _fast_steps()
    if mode == "live":
        return _live_steps(live_metrics_path)
    if mode == "all":
        return _fast_steps() + _live_steps(live_metrics_path)
    raise ValueError(f"unknown mode: {mode!r}")


def _format_command(step: Step) -> str:
    cmd = " ".join(shlex.quote(part) for part in step.argv)
    if not step.env_overrides:
        return cmd
    env_prefix = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in sorted(step.env_overrides.items())
    )
    return f"{env_prefix} {cmd}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tail_text(text: str | None, *, max_lines: int = 40, max_chars: int = 4000) -> str | None:
    if not text:
        return None
    lines = text.rstrip().splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    trimmed = "\n".join(lines).strip()
    if not trimmed:
        return None
    if len(trimmed) > max_chars:
        trimmed = trimmed[-max_chars:]
    return trimmed


def _run_step(step: Step, *, dry_run: bool, index: int) -> StepResult:
    print(f"[tool-contracts] {step.label}")
    command = _format_command(step)
    print(f"[tool-contracts] $ {command}")
    if dry_run:
        return StepResult(
            index=index,
            label=step.label,
            argv=list(step.argv),
            command=command,
            env_overrides=dict(step.env_overrides or {}),
            status="planned",
            returncode=0,
            started_at=None,
            finished_at=None,
            duration_s=None,
            stdout_tail=None,
            stderr_tail=None,
        )

    env = os.environ.copy()
    if step.env_overrides:
        env.update(step.env_overrides)

    started_at = _utc_now_iso()
    started_perf = time.perf_counter()
    completed = subprocess.run(
        step.argv,
        cwd=_repo_root(),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    finished_at = _utc_now_iso()
    duration_s = round(time.perf_counter() - started_perf, 6)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        print(
            f"[tool-contracts] step failed with exit code {completed.returncode}: {step.label}",
            file=sys.stderr,
        )
    return StepResult(
        index=index,
        label=step.label,
        argv=list(step.argv),
        command=command,
        env_overrides=dict(step.env_overrides or {}),
        status="passed" if completed.returncode == 0 else "failed",
        returncode=int(completed.returncode),
        started_at=started_at,
        finished_at=finished_at,
        duration_s=duration_s,
        stdout_tail=_tail_text(completed.stdout) if completed.returncode != 0 else None,
        stderr_tail=_tail_text(completed.stderr) if completed.returncode != 0 else None,
    )


def _read_live_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"live metrics at {path} must be a JSON object")
    if payload.get("schema_version") != LIVE_METRICS_SCHEMA_VERSION:
        raise RuntimeError(
            f"live metrics at {path} have unsupported schema_version {payload.get('schema_version')!r}"
        )
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"live metrics at {path} must contain a metrics object")
    for name, value in metrics.items():
        if not isinstance(name, str) or not name.strip():
            raise RuntimeError(f"live metrics at {path} contain an invalid metric name {name!r}")
        if not isinstance(value, (int, float)) or float(value) < 0:
            raise RuntimeError(f"live metric {name!r} at {path} must be a non-negative number")
    return payload


def _evaluate_performance(
    *,
    mode: str,
    dry_run: bool,
    step_results: list[StepResult],
    report_duration_s: float,
    live_metrics_path: Path | None,
) -> dict[str, Any]:
    budget_manifest = load_tool_contract_performance_budget()
    mode_budget = budget_manifest["runner_modes"][mode]
    runner_step_map = {step.label: step for step in step_results}

    runner_step_budgets: list[dict[str, Any]] = []
    runner_budget_status = "planned" if dry_run else "within_budget"
    if not dry_run and report_duration_s > float(mode_budget["max_total_duration_s"]):
        runner_budget_status = "exceeded"
    for label, budget in mode_budget["steps"].items():
        step = runner_step_map.get(label)
        actual_duration = step.duration_s if step is not None else None
        if dry_run:
            status = "planned"
        elif actual_duration is None:
            status = "not_available"
        elif float(actual_duration) <= float(budget["max_duration_s"]):
            status = "within_budget"
        else:
            status = "exceeded"
        if status in {"exceeded", "not_available"}:
            runner_budget_status = "exceeded" if status == "exceeded" else "not_available"
        runner_step_budgets.append(
            {
                "label": label,
                "description": budget["description"],
                "max_duration_s": float(budget["max_duration_s"]),
                "actual_duration_s": None if actual_duration is None else float(actual_duration),
                "status": status,
            }
        )

    live_metrics_payload = None if dry_run else _read_live_metrics(live_metrics_path)
    live_metric_budgets: list[dict[str, Any]] = []
    missing_live_metrics: list[str] = []
    if mode not in {"live", "all"}:
        live_metrics_status = "not_available"
    elif dry_run:
        live_metrics_status = "planned"
    elif live_metrics_payload is None:
        live_metrics_status = "not_available"
        missing_live_metrics = sorted(budget_manifest["live_metrics"].keys())
    else:
        live_metrics_status = "within_budget"
        metrics = live_metrics_payload["metrics"]
        for name, budget in budget_manifest["live_metrics"].items():
            actual_value = metrics.get(name)
            if actual_value is None:
                status = "not_available"
                missing_live_metrics.append(name)
            elif float(actual_value) <= float(budget["max_duration_s"]):
                status = "within_budget"
            else:
                status = "exceeded"
            if status in {"exceeded", "not_available"}:
                live_metrics_status = "exceeded" if status == "exceeded" else "not_available"
            live_metric_budgets.append(
                {
                    "name": name,
                    "description": budget["description"],
                    "max_duration_s": float(budget["max_duration_s"]),
                    "actual_duration_s": None if actual_value is None else float(actual_value),
                    "status": status,
                }
            )

    return {
        "budget_manifest_path": str(tool_contract_performance_budget_path()),
        "budget_schema_path": str(tool_contract_performance_budget_schema_path()),
        "budget_schema_version": PERF_BUDGET_SCHEMA_VERSION,
        "runner_budget": {
            "mode": mode,
            "status": runner_budget_status,
            "max_total_duration_s": float(mode_budget["max_total_duration_s"]),
            "actual_total_duration_s": float(report_duration_s),
            "step_budgets": runner_step_budgets,
        },
        "live_metrics_budget": {
            "status": live_metrics_status,
            "metrics_path": None if live_metrics_path is None else str(live_metrics_path),
            "missing_metrics": missing_live_metrics,
            "metrics": live_metric_budgets,
        },
    }


def _performance_validation_step(
    *,
    dry_run: bool,
    index: int,
    performance: dict[str, Any],
) -> StepResult:
    print(f"[tool-contracts] {PERF_STEP_LABEL}")
    print(f"[tool-contracts] $ {PERF_STEP_COMMAND}")
    runner_budget = performance["runner_budget"]
    live_metrics_budget = performance["live_metrics_budget"]
    started_at = None if dry_run else _utc_now_iso()
    finished_at = None if dry_run else _utc_now_iso()
    if dry_run:
        status = "planned"
        returncode = 0
        stderr_tail = None
    else:
        blocking_statuses = {runner_budget["status"]}
        if live_metrics_budget["status"] != "not_available":
            blocking_statuses.add(live_metrics_budget["status"])
        exceeded = "exceeded" in blocking_statuses or "not_available" in blocking_statuses
        status = "failed" if exceeded else "passed"
        returncode = 1 if exceeded else 0
        failure_lines: list[str] = []
        if runner_budget["status"] in {"exceeded", "not_available"}:
            failure_lines.append(
                f"runner_budget status={runner_budget['status']} actual_total_duration_s={runner_budget['actual_total_duration_s']} max_total_duration_s={runner_budget['max_total_duration_s']}"
            )
        if live_metrics_budget["status"] in {"exceeded", "not_available"}:
            failure_lines.append(
                f"live_metrics_budget status={live_metrics_budget['status']} missing_metrics={live_metrics_budget['missing_metrics']}"
            )
        stderr_tail = "\n".join(failure_lines) if failure_lines else None
    return StepResult(
        index=index,
        label=PERF_STEP_LABEL,
        argv=[PERF_STEP_COMMAND],
        command=PERF_STEP_COMMAND,
        env_overrides={},
        status=status,
        returncode=returncode,
        started_at=started_at,
        finished_at=finished_at,
        duration_s=None if dry_run else 0.0,
        stdout_tail=None,
        stderr_tail=stderr_tail,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast", "live", "all"], default="fast")
    parser.add_argument("--dry-run", action="store_true", help="print planned commands without executing them")
    parser.add_argument(
        "--report-json",
        type=Path,
        help="write a machine-readable JSON execution report to this path",
    )
    args = parser.parse_args(argv)

    live_metrics_path = (
        _default_live_metrics_report_path(args.mode, args.report_json)
        if args.mode in {"live", "all"}
        else None
    )
    steps = _steps_for_mode(args.mode, live_metrics_path=live_metrics_path)
    print(f"[tool-contracts] mode={args.mode} steps={len(steps)}")
    started_at = _utc_now_iso()
    started_perf = time.perf_counter()
    step_results: list[StepResult] = []
    exit_code = 0
    for index, step in enumerate(steps, start=1):
        result = _run_step(step, dry_run=args.dry_run, index=index)
        step_results.append(result)
        if result.returncode not in (None, 0):
            exit_code = int(result.returncode)
            break

    report_duration_s = round(time.perf_counter() - started_perf, 6)
    performance = _evaluate_performance(
        mode=args.mode,
        dry_run=bool(args.dry_run),
        step_results=step_results,
        report_duration_s=report_duration_s,
        live_metrics_path=live_metrics_path,
    )
    if args.dry_run or exit_code == 0:
        perf_result = _performance_validation_step(
            dry_run=bool(args.dry_run),
            index=len(step_results) + 1,
            performance=performance,
        )
        step_results.append(perf_result)
        if perf_result.returncode not in (None, 0):
            exit_code = int(perf_result.returncode)

    finished_at = _utc_now_iso()
    overall_status = "planned" if args.dry_run else ("passed" if exit_code == 0 else "failed")
    report = {
        "schema_version": "nextstat.tool_contract_runner_report.v1",
        "mode": args.mode,
        "dry_run": bool(args.dry_run),
        "status": overall_status,
        "overall_pass": exit_code == 0,
        "repo_root": str(_repo_root()),
        "python_executable": sys.executable,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": round(time.perf_counter() - started_perf, 6),
        "step_count": len(step_results),
        "failed_step_index": next((step.index for step in step_results if step.status == "failed"), None),
        "failed_step_label": next((step.label for step in step_results if step.status == "failed"), None),
        "performance": performance,
        "steps": [
            {
                "index": step.index,
                "label": step.label,
                "argv": step.argv,
                "command": step.command,
                "env_overrides": step.env_overrides,
                "status": step.status,
                "returncode": step.returncode,
                "started_at": step.started_at,
                "finished_at": step.finished_at,
                "duration_s": step.duration_s,
                "stdout_tail": step.stdout_tail,
                "stderr_tail": step.stderr_tail,
            }
            for step in step_results
        ],
    }
    if args.report_json is not None:
        _write_report(args.report_json, report)
        print(f"[tool-contracts] wrote report: {args.report_json}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
