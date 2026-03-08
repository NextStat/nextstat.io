from __future__ import annotations

import json
from pathlib import Path

from scripts.release_candidate_bundle import build_bundle_manifest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_release_candidate_bundle_schema_has_expected_contract() -> None:
    schema = json.loads(
        (_repo_root() / "docs" / "schemas" / "releases" / "release_candidate_bundle_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert schema["properties"]["schema_version"]["const"] == "nextstat.release_candidate_bundle.v1"
    assert schema["properties"]["mode"]["enum"] == ["prepare", "publish"]


def test_release_candidate_bundle_manifest_marks_optional_entries() -> None:
    repo = _repo_root()
    manifest = build_bundle_manifest(
        "v0.10.0",
        "prepare",
        required_inputs={
            "release_surface_matrix_report_json": repo / "scripts" / "release_surface_matrix_v1.json",
            "release_surface_matrix_report_md": repo / "docs" / "releases" / "release-runbook.md",
            "release_manifest_json": repo / "docs" / "schemas" / "releases" / "release_manifest_v1.schema.json",
            "release_manifest_md": repo / "docs" / "releases" / "benchmark-artifact-policy.md",
        },
        optional_inputs={
            "baseline_compare_report_json": repo / "tmp" / "nonexistent_release_bundle_baseline.json",
        },
    )

    assert manifest["schema_version"] == "nextstat.release_candidate_bundle.v1"
    assert manifest["required_entries"][0]["required"] is True
    assert manifest["optional_entries"][0]["required"] is False
    assert manifest["optional_entries"][0]["present"] is False
