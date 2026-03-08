from __future__ import annotations

from pathlib import Path

from _io_contract_doc_assertions import assert_doc_contains_strings


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_release_workflow_publishes_simplified_likelihood_exporter_promotion_artifacts() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "release-candidate.yml").read_text(
        encoding="utf-8"
    )
    stage_script = (_repo_root() / "scripts" / "release_stage_assets.py").read_text(
        encoding="utf-8"
    )

    assert "name: Simplified Likelihood Exporter Surface Gate" in workflow
    assert "Upload simplified-likelihood exporter artifacts" in workflow
    assert "simplified-likelihood-exporter-surface-report" in workflow
    assert "stable_promotion_decision.json" in workflow
    assert "stable_candidate_blocker_matrix.json" in workflow
    assert "stable_candidate_review_packet.json" in workflow
    assert "stable_evidence_policy.json" in workflow
    assert "stable_evidence_freshness_report.json" in workflow
    assert "stable_source_semantics_boundary.json" in workflow
    assert "stable_review_assessment.json" in workflow
    assert "export_public_validation_report.json" in workflow
    assert "name: Release Candidate Manifest" in workflow
    publish_workflow = (_repo_root() / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    assert "python3 -m scripts.release_stage_assets --dist-root dist --out-dir dist/release-assets" in publish_workflow
    assert "files: dist/release-assets/*" in publish_workflow
    assert "stable_promotion_decision.json" in stage_script
    assert "stable_candidate_blocker_matrix.json" in stage_script
    assert "stable_candidate_review_packet.json" in stage_script
    assert "stable_evidence_policy.json" in stage_script
    assert "stable_evidence_freshness_report.json" in stage_script
    assert "stable_source_semantics_boundary.json" in stage_script
    assert "stable_review_assessment.json" in stage_script
    assert "export_public_validation_report.json" in stage_script


def test_simplified_likelihood_exporter_release_docs_reference_stable_promotion_decision() -> None:
    repo = _repo_root()

    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-release-notes-2026-03-08.md",
        [
            "stable exporter subset",
            "stable_evidence_policy.json",
            "stable_evidence_freshness_report.json",
            "stable_promotion_decision.json",
            "export_public_validation_report.json",
            "release workflow",
            "research-grade fallback",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-support-matrix-2026-03-08.md",
        [
            "`nextstat simplify workspace` | `stable`",
            "stable_evidence_policy.json",
            "stable_promotion_decision.json",
            "research-grade fallback",
        ],
    )
