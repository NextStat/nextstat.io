from __future__ import annotations

from pathlib import Path

from _io_contract_doc_assertions import assert_doc_contains_strings


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_simplified_likelihood_stable_surface_workflow_tracks_promotion_artifacts() -> None:
    workflow = (
        _repo_root() / ".github" / "workflows" / "simplified-likelihood-stable-surface.yml"
    ).read_text(encoding="utf-8")

    assert "tests/python/test_simplified_likelihood_promotion_evidence_bundle_smoke.py" in workflow
    assert "tests/python/test_simplified_likelihood_promotion_evidence_check_smoke.py" in workflow
    assert "tests/python/test_simplified_likelihood_promotion_bundle_promotion_smoke.py" in workflow
    assert "benchmarks/artifacts/simplified_likelihood_promotion_bundles/**" in workflow
    assert '".gitignore"' in workflow or ".gitignore" in workflow
    assert "docs/schemas/benchmarks/simplified_likelihood_promotion_evidence_*.json" in workflow
    assert (
        "docs/schemas/benchmarks/simplified_likelihood_promotion_bundle_promotion_report_v0.schema.json"
        in workflow
    )
    assert "docs/schemas/benchmarks/snapshot_index_v1.schema.json" in workflow
    assert "docs/specs/benchmarks/simplified_likelihood_promotion_evidence_*.json" in workflow
    assert (
        "docs/specs/benchmarks/simplified_likelihood_promotion_bundle_promotion_report_v0.example.json"
        in workflow
    )
    assert "scripts/benchmarks/_simplified_likelihood_promotion_bundle.py" in workflow
    assert "scripts/benchmarks/build_simplified_likelihood_promotion_evidence_bundle.py" in workflow
    assert "scripts/benchmarks/verify_simplified_likelihood_promotion_evidence_bundle.py" in workflow
    assert "scripts/benchmarks/promote_simplified_likelihood_promotion_bundle.py" in workflow
    assert "scripts/benchmarks/write_snapshot_index.py" in workflow
    assert "Upload stable-surface promotion artifacts" in workflow
    assert "tmp/simplified-likelihood-stable-surface/apex2_simplified_likelihood_report.json" in workflow
    assert "tmp/simplified-likelihood-stable-surface/promotion_bundle/promotion_evidence.json" in workflow
    assert (
        "tmp/simplified-likelihood-stable-surface/promotion_bundle/promotion_evidence_check.json"
        in workflow
    )


def test_release_workflow_publishes_simplified_likelihood_promotion_artifacts() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "release-candidate.yml").read_text(
        encoding="utf-8"
    )

    assert "name: Simplified Likelihood Stable-Surface Gate" in workflow
    assert "Upload simplified-likelihood promotion artifacts" in workflow
    assert "simplified-likelihood-stable-surface-report" in workflow
    assert "tmp/simplified-likelihood-stable-surface/apex2_simplified_likelihood_report.json" in workflow
    assert "tmp/simplified-likelihood-stable-surface/promotion_bundle/promotion_evidence.json" in workflow
    assert (
        "tmp/simplified-likelihood-stable-surface/promotion_bundle/promotion_evidence_check.json"
        in workflow
    )
    assert "name: Release Candidate Manifest" in workflow

    publish_workflow = (_repo_root() / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    assert "dist/**/apex2_simplified_likelihood_report.json" in publish_workflow
    assert "dist/**/promotion_evidence.json" in publish_workflow
    assert "dist/**/promotion_evidence_check.json" in publish_workflow


def test_simplified_likelihood_release_docs_reference_promotion_check() -> None:
    repo = _repo_root()

    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-release-pr-checklist-2026-03-08.md",
        [
            "promotion_evidence.json",
            "promotion_evidence_check.json",
            "benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/",
            "snapshot_index.json",
            ".github/workflows/release.yml",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-release-notes-2026-03-08.md",
        [
            "promotion evidence bundle",
            "verification report",
            "committed `snapshot_index.json`",
            "release workflow",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-support-matrix-2026-03-08.md",
        [
            "promotion_evidence.json",
            "promotion_evidence_check.json",
            "three-artifact set",
            "benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/",
        ],
    )
