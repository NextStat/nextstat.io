from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_release_workflow_renders_m15_validation_pack() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "release-candidate.yml").read_text(
        encoding="utf-8"
    )

    assert "mkdir -p artifacts artifacts2 artifacts_jsononly artifacts_m15 artifacts_m15_2" in workflow
    assert '"python-docx>=1.1"' in workflow
    assert "VALIDATION_PACK_OUT_DIR=artifacts_jsononly" in workflow
    assert 'VALIDATION_PACK_ARGS="--python python --nextstat-bin ${NEXTSTAT_BIN} --deterministic --json-only"' in workflow
    assert "cp artifacts_jsononly/pharma_validation.json artifacts_m15/pharma_validation.json" in workflow
    assert "canonical Linux pharma evidence" in workflow
    assert "not treated as cross-platform exact snapshot surfaces" in workflow
    assert "VALIDATION_PACK_M15_CONFIG=docs/specs/m15_config_v1.example.json" in workflow
    assert (
        'VALIDATION_PACK_ARGS="--apex2-master artifacts_jsononly/apex2_master_report.json --python python --nextstat-bin ${NEXTSTAT_BIN} --deterministic --skip-pharma-validation"'
        in workflow
    )
    assert "artifacts_m15/m15_bundle_manifest.json" in workflow
    assert "artifacts_m15/m15_profile_diff_report.json" in workflow
    assert "artifacts_m15/m15_report.md" in workflow
    assert "artifacts_m15/m15_report.pdf" in workflow
    assert "artifacts_m15/m15_report.docx" in workflow
    assert "artifacts_m15_2/m15_bundle_manifest.json" in workflow
    assert "artifacts_m15_2/m15_profile_diff_report.json" in workflow
    assert "artifacts_m15_2/m15_report.md" in workflow
    assert "artifacts_m15_2/m15_report.pdf" in workflow
    assert "artifacts_m15_2/m15_report.docx" in workflow


def test_release_workflow_publishes_m15_artifacts() -> None:
    candidate_workflow = (
        _repo_root() / ".github" / "workflows" / "release-candidate.yml"
    ).read_text(encoding="utf-8")
    workflow = (_repo_root() / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "artifacts_m15/*" in candidate_workflow
    assert "scripts/validation/write_manifest_digest.py \\" in candidate_workflow
    assert "--manifest artifacts_m15/m15_bundle_manifest.json" in candidate_workflow
    assert "docs/schemas/validation/m15_profile_diff_report_v1.schema.json" in candidate_workflow
    assert "--out artifacts_m15/m15_snapshot_index.json" in candidate_workflow
    assert "python3 -m scripts.release_stage_assets --dist-root dist --out-dir dist/release-assets" in workflow
    assert "files: dist/release-assets/*" in workflow


def test_release_workflow_uses_local_artifact_python_install_for_candidate_validation() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "release-candidate.yml").read_text(
        encoding="utf-8"
    )

    assert "Candidate validation must stay independent of PyPI propagation." in workflow
    assert "pip install --no-deps \"${WHEEL}\"" in workflow
    assert "standalone CLI binary instead of relying on nextstat-cli" in workflow


def test_release_workflow_runs_m15_stable_surface_gate() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "release-candidate.yml").read_text(
        encoding="utf-8"
    )

    assert "name: M15 Reporting Stable-Surface Gate" in workflow
    assert 'runs-on: ${{ fromJSON(\'["self-hosted","linux","x64","bench"]\') }}' in workflow
    assert "bash -n scripts/benchmarks/m15_reporting_stable_surface_gate.sh" in workflow
    assert "make m15-reporting-stable-surface-gate" in workflow
    assert "tmp/m15_reporting_stable_surface/m15_reporting_benchmark.json" in workflow
    assert "tmp/m15_reporting_stable_surface/m15_reporting_benchmark.md" in workflow
    assert "tmp/m15_reporting_stable_surface/m15_reporting_compare.json" in workflow
    assert "name: Release Candidate Manifest" in workflow
