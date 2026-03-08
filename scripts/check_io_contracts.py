#!/usr/bin/env python3
"""Canonical runner for NextStat IO-contract checks.

Usage:
  python scripts/check_io_contracts.py --family hepdata
  python scripts/check_io_contracts.py --family histograms_parquet
  python scripts/check_io_contracts.py --family all
  python scripts/check_io_contracts.py --family hepdata --dry-run
  python scripts/check_io_contracts.py --family hepdata --report-json tmp/reports/io_contracts_hepdata_report.json
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


HEPDATA_PYTEST_TARGETS = [
    "tests/python/test_hepdata_schema_smoke.py",
]

HISTOGRAMS_PARQUET_PYTEST_TARGETS = [
    "tests/python/test_histograms_parquet_manifest_schema_smoke.py",
]


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


def _io_contract_cargo_target_dir() -> str:
    override = os.environ.get("NEXTSTAT_IO_CONTRACT_CARGO_TARGET_DIR")
    if override:
        return override
    return str(_repo_root().parent / ".nextstat-cargo-target" / "io-contracts")


def _hepdata_cli_command() -> str:
    override = os.environ.get("NEXTSTAT_HEPDATA_CMD")
    if override:
        return override
    return f"{Path(_io_contract_cargo_target_dir()) / 'debug' / 'nextstat'} import hepdata"


def _hepdata_steps() -> list[Step]:
    py = _python_argv()
    cli_command = _hepdata_cli_command()
    steps: list[Step] = []

    if os.environ.get("NEXTSTAT_HEPDATA_CMD") is None:
        steps.append(
            Step(
                "Build isolated ns-cli binary for IO contracts",
                ["cargo", "build", "-p", "ns-cli"],
                env_overrides={"CARGO_TARGET_DIR": _io_contract_cargo_target_dir()},
            )
        )

    steps.extend(
        [
            Step(
                "Check HEPData schema examples",
                py + ["scripts/generate_hepdata_schema_examples.py", "--check"],
                env_overrides={"NEXTSTAT_HEPDATA_CMD": cli_command},
            ),
            Step(
                "Run HEPData schema smoke suite",
                py + ["-m", "pytest", "-q", *HEPDATA_PYTEST_TARGETS],
                env_overrides={
                    "NEXTSTAT_HEPDATA_CMD": cli_command,
                    "NEXTSTAT_HEPDATA_SKIP_GENERATOR_CHECK": "1",
                },
            ),
        ]
    )
    return steps


def _histograms_parquet_steps() -> list[Step]:
    py = _python_argv()
    return [
        Step(
            "Check histogram Parquet manifest examples",
            py + ["scripts/generate_histograms_parquet_schema_examples.py", "--check"],
        ),
        Step(
            "Run histogram Parquet manifest schema smoke suite",
            py + ["-m", "pytest", "-q", *HISTOGRAMS_PARQUET_PYTEST_TARGETS],
            env_overrides={"NEXTSTAT_HISTOGRAMS_PARQUET_SKIP_GENERATOR_CHECK": "1"},
        ),
    ]


def _steps_for_family(family: str) -> list[Step]:
    if family == "hepdata":
        return _hepdata_steps()
    if family == "histograms_parquet":
        return _histograms_parquet_steps()
    if family == "all":
        return _hepdata_steps() + _histograms_parquet_steps()
    raise ValueError(f"unknown family: {family!r}")


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
    print(f"[io-contracts] {step.label}")
    command = _format_command(step)
    print(f"[io-contracts] $ {command}")
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
            f"[io-contracts] step failed with exit code {completed.returncode}: {step.label}",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family",
        choices=["hepdata", "histograms_parquet", "all"],
        default="hepdata",
    )
    parser.add_argument("--dry-run", action="store_true", help="print planned commands without executing them")
    parser.add_argument(
        "--report-json",
        type=Path,
        help="write a machine-readable JSON execution report to this path",
    )
    args = parser.parse_args(argv)

    steps = _steps_for_family(args.family)
    print(f"[io-contracts] family={args.family} steps={len(steps)}")
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

    finished_at = _utc_now_iso()
    overall_status = "planned" if args.dry_run else ("passed" if exit_code == 0 else "failed")
    report = {
        "schema_version": "nextstat.io_contract_runner_report.v1",
        "family": args.family,
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
        print(f"[io-contracts] wrote report: {args.report_json}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
