#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "nextstat.tool_contract_performance_budget.v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _manifest_path() -> Path:
    return _repo_root() / "scripts" / "tool_contract_performance_budget_v1.json"


def _schema_path() -> Path:
    return (
        _repo_root()
        / "docs"
        / "schemas"
        / "tools"
        / "nextstat_tool_contract_performance_budget_v1.schema.json"
    )


def tool_contract_performance_budget_path() -> Path:
    return _manifest_path()


def tool_contract_performance_budget_schema_path() -> Path:
    return _schema_path()


def validate_tool_contract_performance_budget(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected schema_version={SCHEMA_VERSION!r}")

    runner_modes = manifest.get("runner_modes")
    if not isinstance(runner_modes, dict):
        raise ValueError("runner_modes must be an object")
    for mode in ("fast", "live", "all"):
        mode_budget = runner_modes.get(mode)
        if not isinstance(mode_budget, dict):
            raise ValueError(f"runner_modes.{mode} must be an object")
        max_total = mode_budget.get("max_total_duration_s")
        if not isinstance(max_total, (int, float)) or float(max_total) <= 0:
            raise ValueError(f"runner_modes.{mode}.max_total_duration_s must be > 0")
        steps = mode_budget.get("steps")
        if not isinstance(steps, dict):
            raise ValueError(f"runner_modes.{mode}.steps must be an object")
        for label, step_budget in steps.items():
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"runner_modes.{mode}.steps contains an invalid label: {label!r}")
            _validate_metric_budget(step_budget, path=f"runner_modes.{mode}.steps.{label}")

    live_metrics = manifest.get("live_metrics")
    if not isinstance(live_metrics, dict):
        raise ValueError("live_metrics must be an object")
    required_metrics = (
        "server_build_duration_s",
        "server_startup_duration_s",
        "tools_schema_get_duration_s",
        "workspace_audit_duration_s",
        "fit_duration_s",
        "e2e_discovery_duration_s",
    )
    missing_metrics = [name for name in required_metrics if name not in live_metrics]
    if missing_metrics:
        raise ValueError(f"live_metrics missing required entries: {', '.join(missing_metrics)}")
    for name, metric_budget in live_metrics.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"live_metrics contains an invalid metric name: {name!r}")
        _validate_metric_budget(metric_budget, path=f"live_metrics.{name}")


def _validate_metric_budget(metric_budget: Any, *, path: str) -> None:
    if not isinstance(metric_budget, dict):
        raise ValueError(f"{path} must be an object")
    max_duration = metric_budget.get("max_duration_s")
    if not isinstance(max_duration, (int, float)) or float(max_duration) <= 0:
        raise ValueError(f"{path}.max_duration_s must be > 0")
    description = metric_budget.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{path}.description must be a non-empty string")


def load_tool_contract_performance_budget() -> dict[str, Any]:
    manifest = json.loads(_manifest_path().read_text(encoding="utf-8"))
    validate_tool_contract_performance_budget(manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("json",), default="json")
    args = parser.parse_args(argv)

    manifest = load_tool_contract_performance_budget()
    if args.format == "json":
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
