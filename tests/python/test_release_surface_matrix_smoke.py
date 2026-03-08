from __future__ import annotations

from pathlib import Path

from scripts.release_surface_matrix import build_report, load_manifest, validate_manifest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_release_surface_matrix_manifest_is_valid() -> None:
    manifest = load_manifest()
    validate_manifest(manifest)

    required_ids = {
        surface["id"]
        for surface in manifest["surfaces"]
        if surface["required_for_release"]
    }
    assert required_ids == {
        "gvm_stable_first",
        "simplified_likelihood_stable_surface",
        "simplified_likelihood_exporter_surface",
        "m15_reporting_stable_surface",
    }


def test_release_surface_matrix_report_includes_required_and_optional_surfaces() -> None:
    manifest = load_manifest()
    report = build_report(
        manifest,
        [
            "crates/ns-inference/src/measurement_combine.rs",
            "scripts/benchmarks/bench_m15_reporting.py",
        ],
        "v0.10.0",
    )

    required_ids = {surface["id"] for surface in report["required_release_surfaces"]}
    touched_ids = {surface["id"] for surface in report["advisory_touched_surfaces"]}
    optional_ids = {surface["id"] for surface in report["optional_manual_surfaces"]}

    assert "gvm_stable_first" in required_ids
    assert "m15_reporting_stable_surface" in required_ids
    assert "gvm_stable_first" in touched_ids
    assert "m15_reporting_stable_surface" in touched_ids
    assert "root_trexfitter_parity" in optional_ids


def test_pre_release_gate_and_docs_reference_release_surface_outputs() -> None:
    repo = _repo_root()

    gate = (repo / "scripts" / "apex2" / "pre_release_gate.sh").read_text(encoding="utf-8")
    contributing = (repo / "CONTRIBUTING.md").read_text(encoding="utf-8")
    benchmark_index = (repo / "docs" / "benchmarks.md").read_text(encoding="utf-8")
    runbook = (repo / "docs" / "releases" / "release-runbook.md").read_text(encoding="utf-8")
    policy = (repo / "docs" / "releases" / "benchmark-artifact-policy.md").read_text(
        encoding="utf-8"
    )
    pharma_policy = (
        repo / "docs" / "releases" / "pharma-release-evidence-policy.md"
    ).read_text(encoding="utf-8")

    assert "tmp/release_surface_matrix_report.json" in gate
    assert "tmp/release_surface_matrix_report.md" in gate
    assert "scripts.release_surface_matrix" in gate
    assert "tmp/release_manifest.json" in gate
    assert "tmp/release_manifest.md" in gate
    assert "tmp/release_candidate_bundle" in gate
    assert "scripts.release_manifest" in gate
    assert "scripts.release_candidate_bundle" in gate

    assert "docs/releases/release-runbook.md" in contributing
    assert "docs/releases/benchmark-artifact-policy.md" in contributing
    assert "tmp/release_surface_matrix_report.json" in contributing
    assert "tmp/release_manifest.json" in contributing
    assert "tmp/release_candidate_bundle" in contributing

    assert "/docs/releases/release-runbook" in benchmark_index
    assert "/docs/releases/benchmark-artifact-policy" in benchmark_index
    assert "tmp/release_manifest.json" in benchmark_index

    assert "tmp/release_surface_matrix_report.json" in runbook
    assert "tmp/release_surface_matrix_report.md" in runbook
    assert "tmp/release_manifest.json" in runbook
    assert "tmp/release_manifest.md" in runbook
    assert "release_candidate_bundle" in runbook
    assert "GitHub Release assets" in policy
    assert "Never commit `tmp` outputs" in policy
    assert "canonical Linux release evidence" in pharma_policy
    assert "local build artifacts only" in pharma_policy
    assert "cross-platform SAEM compatibility" in pharma_policy
