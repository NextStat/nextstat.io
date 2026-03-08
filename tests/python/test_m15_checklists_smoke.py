from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_pull_request_template_includes_m15_gate() -> None:
    template = (_repo_root() / ".github" / "pull_request_template.md").read_text(encoding="utf-8")

    assert "### Regulated / M15 (if applicable)" in template
    assert "deterministic rerender remains intact" in template
    assert "hidden execution into the M15 render/bundle path" in template
    assert "docs/references/m15-reporting.md" in template
    assert ".internal/" not in template


def test_contributing_includes_m15_pr_and_release_gates() -> None:
    contributing = (_repo_root() / "CONTRIBUTING.md").read_text(encoding="utf-8")

    assert "### 1a. Additional Gate for M15 / Regulated Reporting Changes" in contributing
    assert "tests/python/test_m15_artifact_schema_smoke.py" in contributing
    assert "--test cli_m15_bundle" in contributing
    assert "No hidden model execution was introduced into the M15 render/bundle path." in contributing
    assert "m15_bundle_manifest.json" in contributing
    assert ".internal/" not in contributing


def test_m15_reference_doc_publishes_merge_and_release_gates() -> None:
    reference = (_repo_root() / "docs" / "references" / "m15-reporting.md").read_text(encoding="utf-8")

    assert "## PR and Release Gates" in reference
    assert "Keep schemas, examples, and `docs/references/m15-reporting.md` in sync." in reference
    assert "Preserve deterministic rerender for `assessment-table`, `map`, `mar`, `profile-diff`, and `bundle`." in reference
    assert "Do not introduce hidden model execution into the M15 render/bundle path." in reference
    assert "author-reviewer-approver flow" in reference
    assert "bundle completion only for `reviewed` or `approved` MAR artifacts" in reference
    assert "m15_profile_diff_report.json" in reference
    assert "m15_report.pdf" in reference
    assert "python -m nextstat.m15_report render" in reference
    assert "m15_bundle_manifest.sha256.bin" in reference
    assert ".internal/" not in reference


def test_internal_m15_signoff_runbook_exists() -> None:
    runbook = (
        _repo_root() / ".internal" / "docs" / "internal" / "2026-03-08-m15-author-reviewer-approver-runbook.md"
    ).read_text(encoding="utf-8")

    assert "M15 Author-Reviewer-Approver Runbook" in runbook
    assert "`review_plan.primary_author`" in runbook
    assert "`review_plan.qa_reviewer`" in runbook
    assert "`review_plan.approver`" in runbook
    assert "make m15-reporting-stable-surface-gate" in runbook
