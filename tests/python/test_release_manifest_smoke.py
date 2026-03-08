from __future__ import annotations

import json
import re
from pathlib import Path

from scripts.release_manifest import build_manifest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _release_candidate_workflow() -> str:
    return (_repo_root() / ".github" / "workflows" / "release-candidate.yml").read_text(
        encoding="utf-8"
    )


def _release_publish_workflow() -> str:
    return (_repo_root() / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")


def _normalized_upload_artifact_names(workflow: str) -> set[str]:
    names = {
        match.replace("${{ matrix.target }}", "*")
        for match in re.findall(
            r"uses:\s+actions/upload-artifact@v4\s+with:\s+name:\s+([^\n]+)",
            workflow,
            flags=re.MULTILINE,
        )
    }
    return names


def _github_release_asset_globs(workflow: str) -> set[str]:
    lines = workflow.splitlines()
    start = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("files: "):
            value = stripped.split("files:", 1)[1].strip()
            return {value}
        if line.strip() == "files: |":
            start = idx + 1
            break
    assert start is not None

    globs: list[str] = []
    for line in lines[start:]:
        if not line.startswith("            "):
            break
        globs.append(line.strip())
    return set(globs)


def test_release_manifest_builds_prepare_contract() -> None:
    manifest = build_manifest("v0.10.0", "prepare")

    assert manifest["schema_version"] == "nextstat.release_manifest.v1"
    assert manifest["release_tag"] == "v0.10.0"
    assert manifest["version"] == "0.10.0"
    assert manifest["mode"] == "prepare"
    assert manifest["version_alignment"]["cargo_toml"] == "0.10.0"
    assert "release-candidate-bundle" in manifest["candidate_artifacts"]["workflow_artifacts"]
    assert manifest["candidate_artifacts"]["github_release_asset_globs"] == ["dist/release-assets/*"]
    assert manifest["pharma_release_policy"] == {
        "prerelease_python_install_mode": "local_artifact_only",
        "canonical_release_evidence_platform": "linux",
        "cross_platform_saem_mode": "acceptance_envelope",
        "canonical_release_evidence_artifact": "pharma_validation.json",
    }
    assert manifest["publish_targets"]["pypi"] == ["nextstat-cli", "nextstat"]


def test_release_manifest_schema_contains_release_candidate_fields() -> None:
    schema = json.loads(
        (_repo_root() / "docs" / "schemas" / "releases" / "release_manifest_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert schema["properties"]["schema_version"]["const"] == "nextstat.release_manifest.v1"
    assert schema["properties"]["mode"]["enum"] == ["prepare", "publish"]
    assert "candidate_artifacts" in schema["required"]
    assert "pharma_release_policy" in schema["required"]
    assert "publish_targets" in schema["required"]
    assert schema["properties"]["pharma_release_policy"]["properties"][
        "cross_platform_saem_mode"
    ]["enum"] == ["acceptance_envelope"]


def test_release_manifest_workflow_artifacts_match_release_candidate_uploads() -> None:
    manifest = build_manifest("v0.10.0", "prepare")
    workflow = _release_candidate_workflow()

    assert "crates-io-publish:" not in workflow
    expected = set(manifest["candidate_artifacts"]["workflow_artifacts"])
    actual = _normalized_upload_artifact_names(workflow)

    assert actual == expected


def test_release_manifest_release_asset_globs_match_publish_workflow() -> None:
    manifest = build_manifest("v0.10.0", "publish")
    workflow = _release_publish_workflow()

    expected = set(manifest["candidate_artifacts"]["github_release_asset_globs"])
    actual = _github_release_asset_globs(workflow)

    assert actual == expected
    assert "python3 -m scripts.release_stage_assets --dist-root dist --out-dir dist/release-assets" in workflow


def test_release_manifest_publish_targets_match_publish_workflow() -> None:
    manifest = build_manifest("v0.10.0", "publish")
    workflow = _release_publish_workflow()

    crates = re.findall(r"^\s+publish\s+([A-Za-z0-9_-]+)$", workflow, flags=re.MULTILINE)
    assert crates == manifest["publish_targets"]["crates_io"]

    assert "name: Publish nextstat-cli to PyPI" in workflow
    assert "name: Publish nextstat to PyPI" in workflow
    assert manifest["publish_targets"]["pypi"] == ["nextstat-cli", "nextstat"]


def test_release_manifest_pharma_policy_matches_workflow_and_docs() -> None:
    manifest = build_manifest("v0.10.0", "prepare")
    workflow = _release_candidate_workflow()
    runbook = (_repo_root() / "docs" / "releases" / "release-runbook.md").read_text(
        encoding="utf-8"
    )
    policy = (
        _repo_root() / "docs" / "releases" / "pharma-release-evidence-policy.md"
    ).read_text(encoding="utf-8")

    pharma = manifest["pharma_release_policy"]
    assert pharma["prerelease_python_install_mode"] == "local_artifact_only"
    assert "Candidate validation must stay independent of PyPI propagation." in workflow
    assert "local build artifacts" in runbook

    assert pharma["canonical_release_evidence_platform"] == "linux"
    assert "canonical Linux release evidence" in policy

    assert pharma["canonical_release_evidence_artifact"] == "pharma_validation.json"
    assert "pharma_validation.json" in workflow
    assert "pharma_validation.json" in policy

    assert pharma["cross_platform_saem_mode"] == "acceptance_envelope"
    assert "cross-platform SAEM compatibility" in policy
