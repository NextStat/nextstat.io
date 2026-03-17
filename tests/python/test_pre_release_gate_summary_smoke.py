from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.apex2.pre_release_gate_summary import (
    EXIT_GOVERNANCE,
    EXIT_INFRASTRUCTURE,
    EXIT_PERFORMANCE,
    EXIT_OK,
    build_summary,
    render_markdown,
)


def test_build_summary_marks_governance_failures() -> None:
    summary = build_summary(
        governance_steps=[{"id": "release_surface_matrix", "status": "failed", "label": "release_surface_matrix"}],
        performance_steps=[{"id": "baseline_compare", "status": "ok", "label": "baseline_compare"}],
        exit_code=EXIT_GOVERNANCE,
        failure_step="release_surface_matrix",
        message="matrix drift",
        version="0.10.1",
        release_tag="v0.10.1",
    )
    assert summary["status"] == "failed"
    assert summary["failure_kind"] == "governance"
    assert summary["layers"]["governance"]["status"] == "failed"
    assert summary["layers"]["performance"]["status"] == "ok"


def test_build_summary_marks_performance_failures() -> None:
    summary = build_summary(
        governance_steps=[{"id": "release_manifest", "status": "ok", "label": "release_manifest"}],
        performance_steps=[{"id": "root_suite_compare", "status": "failed", "label": "root_suite_compare"}],
        exit_code=EXIT_PERFORMANCE,
        failure_step="root_suite_compare",
        message="perf drift",
        version="0.10.1",
        release_tag="v0.10.1",
    )
    assert summary["failure_kind"] == "performance"
    assert summary["layers"]["governance"]["status"] == "ok"
    assert summary["layers"]["performance"]["status"] == "failed"


def test_build_summary_marks_infrastructure_failures() -> None:
    summary = build_summary(
        governance_steps=[],
        performance_steps=[],
        exit_code=EXIT_INFRASTRUCTURE,
        failure_step="baseline_manifest",
        message="missing manifest",
        version="0.10.1",
        release_tag="v0.10.1",
    )
    assert summary["failure_kind"] == "infrastructure"
    assert summary["layers"]["governance"]["status"] == "not_run"
    assert summary["layers"]["performance"]["status"] == "not_run"


def test_render_markdown_mentions_layer_status_and_exit_code() -> None:
    summary = build_summary(
        governance_steps=[{"id": "hep_validation_bundle", "status": "ok", "label": "hep_validation_bundle"}],
        performance_steps=[{"id": "baseline_compare", "status": "skipped", "label": "baseline_compare"}],
        exit_code=EXIT_OK,
        failure_step=None,
        message="Apex2 pre-release gate passed.",
        version="0.10.1",
        release_tag="v0.10.1",
    )
    md = render_markdown(summary)
    assert "Apex2 Pre-release Gate Summary" in md
    assert "Governance" in md
    assert "Performance" in md
    assert "`0`" in md
    assert "`hep_validation_bundle`" in md


def test_build_summary_marks_performance_advisories() -> None:
    summary = build_summary(
        governance_steps=[{"id": "pytest", "status": "ok", "label": "pytest"}],
        performance_steps=[{"id": "baseline_compare", "status": "advisory", "label": "baseline_compare"}],
        exit_code=EXIT_OK,
        failure_step=None,
        message="local advisory",
        version="0.10.1",
        release_tag="v0.10.1",
    )
    assert summary["status"] == "ok"
    assert summary["has_advisories"] is True
    assert summary["layers"]["performance"]["status"] == "advisory"
    md = render_markdown(summary)
    assert "| Performance | `advisory` |" in md


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    governance_steps = tmp_path / "governance.tsv"
    performance_steps = tmp_path / "performance.tsv"
    governance_steps.write_text("release_surface_matrix\tok\trelease_surface_matrix\n", encoding="utf-8")
    performance_steps.write_text("baseline_compare\tfailed\tbaseline_compare\n", encoding="utf-8")
    out_json = tmp_path / "summary.json"
    out_md = tmp_path / "summary.md"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.apex2.pre_release_gate_summary",
            "--governance-steps",
            str(governance_steps),
            "--performance-steps",
            str(performance_steps),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
            "--exit-code",
            str(EXIT_PERFORMANCE),
            "--failure-step",
            "baseline_compare",
            "--message",
            "perf drift",
            "--version",
            "0.10.1",
            "--release-tag",
            "v0.10.1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["failure_kind"] == "performance"
    assert payload["layers"]["performance"]["status"] == "failed"
    assert "baseline_compare" in out_md.read_text(encoding="utf-8")
