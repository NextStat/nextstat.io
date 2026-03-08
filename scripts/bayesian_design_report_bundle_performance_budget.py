#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "nextstat.bayesian_design_report_bundle_performance_budget.v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _manifest_path() -> Path:
    return _repo_root() / "scripts" / "bayesian_design_report_bundle_performance_budget_v1.json"


def _schema_path() -> Path:
    return (
        _repo_root()
        / "docs"
        / "schemas"
        / "benchmarks"
        / "nextstat_bayesian_design_report_bundle_performance_budget_v1.schema.json"
    )


def bayesian_design_report_bundle_performance_budget_path() -> Path:
    return _manifest_path()


def bayesian_design_report_bundle_performance_budget_schema_path() -> Path:
    return _schema_path()


def _validate_case_budget(case_budget: Any, *, path: str) -> None:
    if not isinstance(case_budget, dict):
        raise ValueError(f"{path} must be an object")
    description = case_budget.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{path}.description must be a non-empty string")
    for field in (
        "max_bundle_duration_s",
        "max_manifest_regen_duration_s",
        "max_bundle_bytes",
        "max_manifest_bytes",
    ):
        value = case_budget.get(field)
        if not isinstance(value, (int, float)) or float(value) <= 0:
            raise ValueError(f"{path}.{field} must be > 0")


def validate_bayesian_design_report_bundle_performance_budget(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected schema_version={SCHEMA_VERSION!r}")

    runner_modes = manifest.get("runner_modes")
    if not isinstance(runner_modes, dict):
        raise ValueError("runner_modes must be an object")
    for mode in ("smoke", "release"):
        mode_doc = runner_modes.get(mode)
        if not isinstance(mode_doc, dict):
            raise ValueError(f"runner_modes.{mode} must be an object")
        repeat = mode_doc.get("repeat")
        if not isinstance(repeat, int) or repeat <= 0:
            raise ValueError(f"runner_modes.{mode}.repeat must be a positive integer")
        manifest_repeat = mode_doc.get("manifest_repeat")
        if not isinstance(manifest_repeat, int) or manifest_repeat <= 0:
            raise ValueError(f"runner_modes.{mode}.manifest_repeat must be a positive integer")

    cases = manifest.get("cases")
    if not isinstance(cases, dict):
        raise ValueError("cases must be an object")
    for case_id in ("beta_small", "beta_large", "normal_small", "normal_large"):
        if case_id not in cases:
            raise ValueError(f"cases missing required entry {case_id!r}")
        _validate_case_budget(cases[case_id], path=f"cases.{case_id}")


def load_bayesian_design_report_bundle_performance_budget() -> dict[str, Any]:
    manifest = json.loads(_manifest_path().read_text(encoding="utf-8"))
    validate_bayesian_design_report_bundle_performance_budget(manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("json",), default="json")
    args = parser.parse_args(argv)

    manifest = load_bayesian_design_report_bundle_performance_budget()
    if args.format == "json":
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
