#!/usr/bin/env python3
"""Helper for the canonical tool-contract artifact manifest."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


_MANIFEST_FILENAME = "tool_contract_artifact_manifest_v1.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _manifest_path() -> Path:
    return Path(__file__).with_name(_MANIFEST_FILENAME)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_tool_contract_artifact_manifest(manifest: dict[str, Any]) -> None:
    _require(
        manifest.get("schema_version") == "nextstat.tool_contract_artifact_manifest.v1",
        f"Invalid tool-contract artifact manifest schema_version in {_manifest_path()}",
    )

    reports = manifest.get("reports")
    _require(isinstance(reports, dict), "manifest.reports must be an object")
    _require(set(reports.keys()) == {"fast", "live"}, "manifest.reports must contain fast and live")

    artifact_names: set[str] = set()
    for report_name in ("fast", "live"):
        report = reports.get(report_name)
        _require(isinstance(report, dict), f"manifest.reports.{report_name} must be an object")
        _require(
            set(report.keys()) == {"runner_report_path", "artifact_name", "download_dir", "downloaded_report_path"},
            f"manifest.reports.{report_name} has unsupported keys",
        )
        runner_report_path = report.get("runner_report_path")
        artifact_name = report.get("artifact_name")
        download_dir = report.get("download_dir")
        downloaded_report_path = report.get("downloaded_report_path")
        _require(isinstance(runner_report_path, str) and runner_report_path, f"manifest.reports.{report_name}.runner_report_path must be a non-empty string")
        _require(isinstance(artifact_name, str) and artifact_name, f"manifest.reports.{report_name}.artifact_name must be a non-empty string")
        _require(isinstance(download_dir, str) and download_dir, f"manifest.reports.{report_name}.download_dir must be a non-empty string")
        _require(
            isinstance(downloaded_report_path, str) and downloaded_report_path,
            f"manifest.reports.{report_name}.downloaded_report_path must be a non-empty string",
        )
        _require(artifact_name not in artifact_names, f"Duplicate artifact_name in manifest.reports: {artifact_name}")
        artifact_names.add(artifact_name)

        runner_basename = Path(runner_report_path).name
        downloaded_path = Path(downloaded_report_path)
        _require(
            downloaded_path.name == runner_basename,
            f"manifest.reports.{report_name}.downloaded_report_path must end with {runner_basename}",
        )
        _require(
            str(downloaded_path.parent) == download_dir,
            f"manifest.reports.{report_name}.downloaded_report_path must live under {download_dir}",
        )

    dashboard = manifest.get("dashboard")
    _require(isinstance(dashboard, dict), "manifest.dashboard must be an object")
    _require(
        set(dashboard.keys()) == {"artifact_name", "json_path", "markdown_path"},
        "manifest.dashboard has unsupported keys",
    )
    dashboard_artifact_name = dashboard.get("artifact_name")
    _require(
        isinstance(dashboard_artifact_name, str) and dashboard_artifact_name,
        "manifest.dashboard.artifact_name must be a non-empty string",
    )
    _require(
        dashboard_artifact_name not in artifact_names,
        f"Duplicate dashboard artifact_name: {dashboard_artifact_name}",
    )
    for key in ("json_path", "markdown_path"):
        value = dashboard.get(key)
        _require(isinstance(value, str) and value, f"manifest.dashboard.{key} must be a non-empty string")


def load_tool_contract_artifact_manifest() -> dict[str, Any]:
    manifest = json.loads(_manifest_path().read_text(encoding="utf-8"))
    validate_tool_contract_artifact_manifest(manifest)
    return copy.deepcopy(manifest)


def to_github_outputs(manifest: dict[str, Any]) -> dict[str, str]:
    reports = manifest["reports"]
    dashboard = manifest["dashboard"]
    return {
        "fast_report_path": str(reports["fast"]["runner_report_path"]),
        "fast_artifact_name": str(reports["fast"]["artifact_name"]),
        "fast_download_dir": str(reports["fast"]["download_dir"]),
        "fast_downloaded_report_path": str(reports["fast"]["downloaded_report_path"]),
        "live_report_path": str(reports["live"]["runner_report_path"]),
        "live_artifact_name": str(reports["live"]["artifact_name"]),
        "live_download_dir": str(reports["live"]["download_dir"]),
        "live_downloaded_report_path": str(reports["live"]["downloaded_report_path"]),
        "dashboard_artifact_name": str(dashboard["artifact_name"]),
        "dashboard_json_path": str(dashboard["json_path"]),
        "dashboard_markdown_path": str(dashboard["markdown_path"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["json", "github-output"], default="json")
    args = parser.parse_args(argv)

    manifest = load_tool_contract_artifact_manifest()
    if args.format == "json":
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    for key, value in to_github_outputs(manifest).items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
