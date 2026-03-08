from __future__ import annotations

from pathlib import Path

from _io_contract_doc_assertions import assert_doc_contains_strings


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_prepare_release_workflow_requires_explicit_release_tag_and_uses_candidate_workflow() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "prepare-release.yml").read_text(
        encoding="utf-8"
    )

    assert "workflow_dispatch:" in workflow
    assert "inputs:" in workflow
    assert "release_tag:" in workflow
    assert 'description: "Release tag in vX.Y.Z form for prepare-only validation"' in workflow
    assert "required: true" in workflow
    assert "type: string" in workflow
    assert "uses: ./.github/workflows/release-candidate.yml" in workflow
    assert "release_mode: prepare" in workflow


def test_release_workflow_is_tag_only_publish_wrapper() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch:" not in workflow
    assert 'tags:\n      - "v*"' in workflow
    assert "uses: ./.github/workflows/release-candidate.yml" in workflow
    assert "release_mode: publish" in workflow
    assert "crates-io-publish:" in workflow
    assert "github-release:" in workflow
    assert "needs: [release-candidate]" in workflow


def test_release_candidate_workflow_builds_release_candidate_bundle() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "release-candidate.yml").read_text(
        encoding="utf-8"
    )

    assert "python3 -m scripts.release_surface_matrix" in workflow
    assert "python3 -m scripts.release_candidate_bundle" in workflow
    assert "release-candidate-bundle" in workflow
    assert "docs/schemas/releases/release_candidate_bundle_v1.schema.json" in workflow


def test_release_runbook_describes_prepare_only_workflow_dispatch() -> None:
    repo = _repo_root()
    assert_doc_contains_strings(
        repo / "docs" / "releases" / "release-runbook.md",
        [
            "prepare-release.yml",
            "release-candidate.yml",
            "prepare-only",
            "release_tag",
            "tag push",
            "local build artifacts",
            "pharma_validation.json",
        ],
    )
