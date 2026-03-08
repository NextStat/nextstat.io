from __future__ import annotations

import json
from pathlib import Path

from scripts.release_full_fidelity_simulation import (
    parse_upload_artifact_steps,
    run_simulation,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_release_candidate_workflow_uploads_are_parsed() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "release-candidate.yml").read_text(
        encoding="utf-8"
    )
    steps = parse_upload_artifact_steps(workflow)
    names = {step.name for step in steps}

    assert "simplified-likelihood-exporter-surface-report" in names
    assert "validation-pack" in names
    assert "release-candidate-bundle" in names
    assert "wheels-${{ matrix.target }}" in names


def test_release_full_fidelity_simulation_stages_canonical_release_assets(tmp_path: Path) -> None:
    report, staged_root = run_simulation("v0.10.0", "publish", tmp_path / "simulation")

    staged_names = set(report["staged_assets"])
    assert report["schema_version"] == "nextstat.release_full_fidelity_simulation.v1"
    assert "release_candidate_bundle_manifest.json" in staged_names
    assert "release_candidate_bundle_v1.schema.json" in staged_names
    assert "export_benchmark_snapshot_report.json" in staged_names
    assert "promotion_evidence.json" in staged_names
    assert "validation_report.json" in staged_names
    assert "m15_bundle_manifest.json" in staged_names
    assert any(name.endswith(".whl") for name in staged_names)
    assert any(name.endswith(".tar.gz") for name in staged_names)
    assert (staged_root / "export_benchmark_snapshot_report.json").exists()
    assert (staged_root / "promotion_evidence.json").exists()

    inventory = report["workflow_artifacts"]
    assert "simplified-likelihood-exporter-surface-report" in inventory
    assert any(
        item.endswith(
            "simplified-likelihood-exporter-surface-report/benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_benchmark_snapshot_report.json"
        )
        for item in inventory["simplified-likelihood-exporter-surface-report"]
    )
    assert any(
        item.endswith(
            "release-candidate-bundle/tmp/release_candidate_bundle/release_candidate_bundle_manifest.json"
        )
        for item in inventory["release-candidate-bundle"]
    )


def test_release_full_fidelity_simulation_report_is_json_serializable(tmp_path: Path) -> None:
    report, _ = run_simulation("v0.10.0", "prepare", tmp_path / "simulation")
    payload = json.dumps(report, indent=2, sort_keys=True)
    assert '"schema_version": "nextstat.release_full_fidelity_simulation.v1"' in payload
