from __future__ import annotations

from pathlib import Path

from _io_contract_doc_assertions import assert_doc_contains_strings


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_pull_request_template_includes_bayesian_design_gate() -> None:
    template = (_repo_root() / ".github" / "pull_request_template.md").read_text(
        encoding="utf-8"
    )

    assert "### Bayesian Design Stable Surface (if applicable)" in template
    assert "docs/references/bayesian-trial-design-artifacts.md" in template
    assert (
        "hidden execution into the frozen Bayesian report render/bundle path"
        in template
    )
    assert "frozen historical-control borrowing review path" in template
    assert "frozen robust-mixture prior review path" in template
    assert "nextstat-bench packaging gate" in template
    assert ".internal/" not in template


def test_contributing_includes_bayesian_design_pr_and_release_gates() -> None:
    contributing = (_repo_root() / "CONTRIBUTING.md").read_text(encoding="utf-8")

    assert (
        "### 1b. Additional Gate for Bayesian Design Stable Surface Changes"
        in contributing
    )
    assert "tests/python/test_bayes_design_stable_surface_regression.py" in contributing
    assert "tests/python/test_bayes_design_checklists_smoke.py" in contributing
    assert "docs/references/bayesian-trial-design-artifacts.md" in contributing
    assert (
        "bayesian_historical_control_borrowing_review_acceptance_v0.md" in contributing
    )
    assert (
        "bayesian_historical_control_borrowing_operating_characteristics_acceptance_v0.md"
        in contributing
    )
    assert "bayesian_robust_mixture_prior_review_acceptance_v0.md" in contributing
    assert (
        "bayesian_robust_mixture_prior_operating_characteristics_acceptance_v0.md"
        in contributing
    )
    assert (
        "No hidden execution was introduced into frozen Bayesian historical-control borrowing review paths."
        in contributing
    )
    assert (
        "No hidden execution was introduced into frozen Bayesian robust-mixture prior review paths."
        in contributing
    )
    assert "nextstat-bench" in contributing
    assert ".internal/" not in contributing


def test_bayesian_reference_doc_publishes_merge_and_release_gates() -> None:
    reference = (
        _repo_root() / "docs" / "references" / "bayesian-trial-design-artifacts.md"
    ).read_text(encoding="utf-8")

    assert "## PR and Release Gates" in reference
    assert (
        "Keep schemas, examples, and `docs/references/bayesian-trial-design-artifacts.md` in sync."
        in reference
    )
    assert (
        "Do not introduce hidden execution into frozen Bayesian report render/bundle paths."
        in reference
    )
    assert "nextstat-bench" in reference
    assert (
        "docs/specs/pharma/bayesian_design_regulatory_appendix_acceptance_v0.md"
        in reference
    )
    assert (
        "docs/specs/pharma/bayesian_design_appendix_render_acceptance_v0.md"
        in reference
    )
    assert (
        "docs/specs/pharma/bayesian_design_validation_pack_acceptance_v0.md"
        in reference
    )
    assert (
        "docs/specs/pharma/bayesian_prior_conflict_diagnostic_acceptance_v0.md"
        in reference
    )
    assert (
        "docs/specs/pharma/bayesian_historical_control_borrowing_review_acceptance_v0.md"
        in reference
    )
    assert (
        "docs/specs/pharma/bayesian_historical_control_borrowing_operating_characteristics_acceptance_v0.md"
        in reference
    )
    assert (
        "docs/specs/pharma/bayesian_robust_mixture_prior_review_acceptance_v0.md"
        in reference
    )
    assert (
        "docs/specs/pharma/bayesian_robust_mixture_prior_operating_characteristics_acceptance_v0.md"
        in reference
    )
    assert (
        "docs/benchmarks/bayesian-design-release-pr-checklist-2026-03-08.md"
        in reference
    )
    assert ".internal/" not in reference


def test_bayesian_release_pr_checklist_doc_is_published() -> None:
    assert_doc_contains_strings(
        _repo_root()
        / "docs"
        / "benchmarks"
        / "bayesian-design-release-pr-checklist-2026-03-08.md",
        [
            "# Bayesian Design Release PR Checklist",
            "docs/specs/pharma/bayesian_design_report_acceptance_v0.md",
            "docs/specs/pharma/bayesian_prior_conflict_diagnostic_acceptance_v0.md",
            "docs/specs/pharma/bayesian_historical_control_borrowing_review_acceptance_v0.md",
            "docs/specs/pharma/bayesian_historical_control_borrowing_operating_characteristics_acceptance_v0.md",
            "docs/specs/pharma/bayesian_robust_mixture_prior_review_acceptance_v0.md",
            "docs/specs/pharma/bayesian_robust_mixture_prior_operating_characteristics_acceptance_v0.md",
            "docs/specs/pharma/bayesian_design_report_bundle_acceptance_v0.md",
            "docs/specs/pharma/bayesian_design_regulatory_appendix_acceptance_v0.md",
            "docs/specs/pharma/bayesian_design_appendix_render_acceptance_v0.md",
            "docs/specs/pharma/bayesian_design_validation_pack_acceptance_v0.md",
            "docs/benchmarks/bayesian-design-report-packaging-runtime-gate.md",
            "docs/references/bayesian-trial-design-artifacts.md",
            "tests/python/test_validation_pack_execution_regression.py",
            "tests/python/test_bayes_design_stable_surface_regression.py",
            "tests/python/test_bayes_design_checklists_smoke.py",
            "build_*_prior_conflict_diagnostic(...)",
            "build_*_historical_control_borrowing_review(...)",
            "build_*_robust_mixture_prior_review(...)",
            "simulate_*_historical_control_borrowing_operating_characteristics(...)",
            "simulate_*_robust_mixture_prior_operating_characteristics(...)",
            "render_bayesian_regulatory_appendix_markdown(...)",
            "write_bayesian_regulatory_appendix_pdf(...)",
            "nextstat-bench",
        ],
    )
