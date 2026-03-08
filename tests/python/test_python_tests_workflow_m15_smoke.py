from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_python_tests_workflow_writes_m15_integrity_artifacts() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "python-tests.yml").read_text(encoding="utf-8")

    assert workflow.count("scripts/validation/write_manifest_digest.py \\") == 2
    assert '"python-docx>=1.1"' in workflow
    assert "docs/schemas/validation/m15_profile_diff_report_v1.schema.json" in workflow
    assert "artifacts_m15/m15_profile_diff_report.json" in workflow
    assert "m15_profile_diff_report_v1: schema ok" in workflow
    assert "--manifest artifacts/validation_pack_manifest.json" in workflow
    assert "--manifest artifacts_m15/m15_bundle_manifest.json" in workflow
    assert "--out artifacts/snapshot_index.json" in workflow
    assert "--out artifacts_m15/m15_snapshot_index.json" in workflow
    assert "artifacts_m15/m15_profile_diff_report_v1.schema.json" in workflow
    assert "artifacts_m15/m15_report.md" in workflow
    assert "artifacts_m15/m15_report.pdf" in workflow
    assert "artifacts_m15/m15_report.docx" in workflow
    assert "artifacts_m15/m15_bundle_manifest.sha256" in workflow
    assert "artifacts_m15/m15_bundle_manifest.sha256.bin" in workflow
    assert "artifacts_m15/m15_snapshot_index.json" in workflow


def test_python_tests_workflow_runs_m15_stable_surface_gate() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "python-tests.yml").read_text(encoding="utf-8")

    assert "name: M15 Reporting Stable-Surface Gate" in workflow
    assert 'runs-on: ${{ fromJSON(\'["self-hosted","linux","x64","bench"]\') }}' in workflow
    assert "needs: [validation-pack]" in workflow
    assert "bash -n scripts/benchmarks/m15_reporting_stable_surface_gate.sh" in workflow
    assert "make m15-reporting-stable-surface-gate" in workflow
    assert "tmp/m15_reporting_stable_surface/m15_reporting_benchmark.json" in workflow
    assert "tmp/m15_reporting_stable_surface/m15_reporting_benchmark.md" in workflow
    assert "tmp/m15_reporting_stable_surface/m15_reporting_compare.json" in workflow
