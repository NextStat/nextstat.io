from __future__ import annotations

from pathlib import Path

from _io_contract_doc_assertions import assert_doc_contains_strings


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_simplified_likelihood_exporter_docs_publish_acceptance_and_gate() -> None:
    repo = _repo_root()

    assert_doc_contains_strings(
        repo / "docs" / "README.md",
        [
            "simplified-likelihood-exporter-acceptance-2026-03-09.md",
            "simplified-likelihood-exporter-runtime-gate.md",
            "simplified-likelihood-exporter-promotion-runbook-2026-03-09.md",
            "simplified-likelihood-exporter-stable-evidence-policy-2026-03-09.md",
            "simplified-likelihood-exporter-stable-evidence-freshness-2026-03-09.md",
            "simplified-likelihood-exporter-public-validation-surface-2026-03-09.md",
            "simplified-likelihood-exporter-stable-promotion-decision-2026-03-09.md",
            "simplified-likelihood-exporter-release-pr-checklist-2026-03-09.md",
            "simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md",
            "apex2_simplified_likelihood_export_public_case_catalog_v0.example.json",
            "simplified_likelihood_export_public_validation_report_v0.example.json",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks.md",
        [
            "simplified-likelihood-exporter-acceptance-2026-03-09",
            "simplified-likelihood-exporter-runtime-gate",
            "simplified-likelihood-exporter-promotion-runbook-2026-03-09",
            "simplified-likelihood-exporter-stable-evidence-policy-2026-03-09",
            "simplified-likelihood-exporter-stable-evidence-freshness-2026-03-09",
            "simplified-likelihood-exporter-public-validation-surface-2026-03-09",
            "simplified-likelihood-exporter-stable-promotion-decision-2026-03-09",
            "simplified-likelihood-exporter-release-pr-checklist-2026-03-09",
            "simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09",
            "apex2_simplified_likelihood_export_public_case_catalog_v0.example.json",
            "simplified_likelihood_export_public_validation_report_v0.example.json",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-benchmark-snapshot-2026-03-08.md",
        [
            "simplified-likelihood-exporter-acceptance-2026-03-09.md",
            "simplified-likelihood-exporter-runtime-gate.md",
            "simplified-likelihood-exporter-promotion-runbook-2026-03-09.md",
            "simplified-likelihood-exporter-stable-evidence-policy-2026-03-09.md",
            "simplified-likelihood-exporter-stable-evidence-freshness-2026-03-09.md",
            "simplified-likelihood-exporter-public-validation-surface-2026-03-09.md",
            "simplified-likelihood-exporter-stable-promotion-decision-2026-03-09.md",
            "simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md",
            "benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/",
            "simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/",
            "export_public_validation_report.json",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "references" / "simplified-likelihood-artifacts.md",
        [
            "simplified-likelihood-exporter-acceptance-2026-03-09.md",
            "simplified-likelihood-exporter-runtime-gate.md",
            "simplified-likelihood-exporter-promotion-runbook-2026-03-09.md",
            "simplified-likelihood-exporter-stable-evidence-policy-2026-03-09.md",
            "simplified-likelihood-exporter-stable-evidence-freshness-2026-03-09.md",
            "simplified-likelihood-exporter-public-validation-surface-2026-03-09.md",
            "simplified-likelihood-exporter-stable-promotion-decision-2026-03-09.md",
            "simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md",
            "benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/",
            "simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/",
            "simplified_likelihood_export_public_validation_report_v0",
            "simplified_likelihood_exporter_stable_evidence_policy_v0",
        ],
    )


def test_simplified_likelihood_exporter_acceptance_doc_covers_explicit_thresholds() -> None:
    repo = _repo_root()

    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-exporter-acceptance-2026-03-09.md",
        [
            "`stable`",
            "`nextstat simplify workspace`",
            "nextstat-bench",
            "full -> derived -> reinterpret",
            "max_abs_q_mu_diff <= 0.1",
            "upper_limit_ratio",
            "min_net_end_to_end_upper_limit_speedup >= 1.25x",
            "public_reinterpretation_style",
            "export_public_validation_report.json",
            "stable_evidence_policy.json",
            "stable_promotion_decision.json",
            "single-POI",
            "source_model_constraints",
        ],
    )


def test_simplified_likelihood_exporter_runtime_gate_doc_and_make_target_are_published() -> None:
    repo = _repo_root()

    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-exporter-runtime-gate.md",
        [
            "scripts/benchmarks/simplified_likelihood_exporter_surface_gate.sh",
            "make simplified-likelihood-exporter-surface-gate",
            ".github/workflows/simplified-likelihood-exporter-surface.yml",
            "benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/apex2_simplified_likelihood_report.json",
            "benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_benchmark_snapshot_report.json",
            "benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_public_validation_report.json",
            "benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/snapshot_index.json",
            "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_evidence.json",
            "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_evidence_check.json",
            "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_bundle_promotion_report.json",
            "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_policy.json",
            "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_freshness_report.json",
            "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_source_semantics_boundary.json",
            "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_promotion_decision.json",
            "min_net_end_to_end_upper_limit_speedup >= 1.25x",
            "public_reinterpretation_style",
            "stable evidence surface",
        ],
    )

    makefile = (repo / "Makefile").read_text(encoding="utf-8")
    assert ".PHONY:" in makefile
    assert "simplified-likelihood-exporter-surface-gate" in makefile
    assert "bash scripts/benchmarks/simplified_likelihood_exporter_surface_gate.sh" in makefile


def test_simplified_likelihood_exporter_surface_workflow_smoke() -> None:
    workflow = (
        _repo_root() / ".github" / "workflows" / "simplified-likelihood-exporter-surface.yml"
    ).read_text(encoding="utf-8")

    assert "name: Simplified Likelihood Exporter Surface" in workflow
    assert "bash -n scripts/benchmarks/simplified_likelihood_exporter_surface_gate.sh" in workflow
    assert "make simplified-likelihood-exporter-surface-gate" in workflow
    assert "tests/python/test_simplified_likelihood_exporter_gate_smoke.py" in workflow
    assert (
        "tests/python/test_simplified_likelihood_exporter_stable_promotion_decision_smoke.py"
        in workflow
    )
    assert (
        "tests/python/test_simplified_likelihood_exporter_stable_evidence_policy_smoke.py"
        in workflow
    )
    assert (
        "tests/python/test_simplified_likelihood_exporter_stable_evidence_freshness_smoke.py"
        in workflow
    )
    assert (
        "tests/python/test_release_workflow_simplified_likelihood_exporter_smoke.py"
        in workflow
    )
    assert (
        "tests/python/test_simplified_likelihood_export_public_validation_report_smoke.py"
        in workflow
    )
    assert "tests/python/test_simplified_likelihood_export_public_case_catalog_smoke.py" in workflow
    assert "tests/python/test_simplified_likelihood_exporter_promotion_bundle_smoke.py" in workflow
    assert (
        "tests/python/test_simplified_likelihood_exporter_stable_source_semantics_boundary_smoke.py"
        in workflow
    )
    assert "tests/python/test_simplified_likelihood_export_benchmark_snapshot_smoke.py" in workflow
    assert "tests/python/_simplified_likelihood_export_public_case_catalog.py" in workflow
    assert "scripts/benchmarks/_simplified_likelihood_exporter_promotion_bundle.py" in workflow
    assert (
        "scripts/benchmarks/build_simplified_likelihood_exporter_stable_source_semantics_boundary.py"
        in workflow
    )
    assert "scripts/benchmarks/apex2_simplified_likelihood_remote.sh" in workflow
    assert "benchmarks/artifacts/simplified_likelihood_export_benchmarks/**" in workflow
    assert "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/**" in workflow
    assert (
        "docs/benchmarks/simplified-likelihood-exporter-acceptance-2026-03-09.md" in workflow
    )
    assert "docs/benchmarks/simplified-likelihood-exporter-runtime-gate.md" in workflow
    assert "docs/benchmarks/simplified-likelihood-exporter-promotion-runbook-2026-03-09.md" in workflow
    assert (
        "docs/benchmarks/simplified-likelihood-exporter-stable-evidence-policy-2026-03-09.md"
        in workflow
    )
    assert (
        "docs/benchmarks/simplified-likelihood-exporter-stable-evidence-freshness-2026-03-09.md"
        in workflow
    )
    assert (
        "docs/benchmarks/simplified-likelihood-exporter-stable-promotion-decision-2026-03-09.md"
        in workflow
    )
    assert (
        "docs/benchmarks/simplified-likelihood-exporter-public-validation-surface-2026-03-09.md"
        in workflow
    )
    assert (
        "docs/benchmarks/simplified-likelihood-exporter-release-pr-checklist-2026-03-09.md"
        in workflow
    )
    assert (
        "docs/benchmarks/simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md"
        in workflow
    )
    assert (
        "docs/schemas/apex2/simplified_likelihood_report_v0.schema.json"
        in workflow
    )
    assert (
        "docs/schemas/benchmarks/simplified_likelihood_export_benchmark_snapshot_report_v0.schema.json"
        in workflow
    )
    assert (
        "docs/schemas/benchmarks/simplified_likelihood_export_public_validation_report_v0.schema.json"
        in workflow
    )
    assert (
        "docs/schemas/benchmarks/simplified_likelihood_exporter_promotion_evidence_bundle_v0.schema.json"
        in workflow
    )
    assert (
        "docs/schemas/benchmarks/simplified_likelihood_exporter_stable_source_semantics_boundary_v0.schema.json"
        in workflow
    )
    assert (
        "docs/schemas/benchmarks/simplified_likelihood_exporter_stable_evidence_policy_v0.schema.json"
        in workflow
    )
    assert (
        "docs/schemas/benchmarks/simplified_likelihood_exporter_stable_evidence_freshness_report_v0.schema.json"
        in workflow
    )
    assert (
        "docs/schemas/benchmarks/simplified_likelihood_exporter_stable_promotion_decision_v0.schema.json"
        in workflow
    )
    assert (
        "docs/specs/benchmarks/simplified_likelihood_exporter_promotion_evidence_bundle_v0.example.json"
        in workflow
    )
    assert (
        "docs/specs/apex2_simplified_likelihood_report_v0.example.json"
        in workflow
    )
    assert (
        "docs/specs/benchmarks/simplified_likelihood_export_benchmark_snapshot_report_v0.example.json"
        in workflow
    )
    assert (
        "docs/specs/benchmarks/simplified_likelihood_export_public_validation_report_v0.example.json"
        in workflow
    )
    assert (
        "docs/specs/benchmarks/simplified_likelihood_exporter_stable_source_semantics_boundary_v0.example.json"
        in workflow
    )


def test_apex2_remote_runner_uses_source_tree_python_package_instead_of_published_cli_dependency() -> None:
    workflow = (
        _repo_root() / ".github" / "workflows" / "simplified-likelihood-exporter-surface.yml"
    ).read_text(encoding="utf-8")
    script = (
        _repo_root() / "scripts" / "benchmarks" / "apex2_simplified_likelihood_remote.sh"
    ).read_text(encoding="utf-8")

    assert "maturin develop --release --skip-install" in script
    assert 'export PYTHONPATH="$REPO/bindings/ns-py/python${PYTHONPATH:+:$PYTHONPATH}"' in script
    assert (
        "docs/specs/benchmarks/simplified_likelihood_exporter_stable_evidence_policy_v0.example.json"
        in workflow
    )
    assert (
        "docs/specs/benchmarks/simplified_likelihood_exporter_stable_evidence_freshness_report_v0.example.json"
        in workflow
    )
    assert (
        "docs/specs/benchmarks/simplified_likelihood_exporter_stable_promotion_decision_v0.example.json"
        in workflow
    )
    assert "Upload simplified-likelihood exporter artifacts" in workflow
    assert (
        "benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/apex2_simplified_likelihood_report.json"
        in workflow
    )
    assert (
        "benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_benchmark_snapshot_report.json"
        in workflow
    )
    assert (
        "benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_public_validation_report.json"
        in workflow
    )
    assert (
        "benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/snapshot_index.json"
        in workflow
    )
    assert (
        "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_evidence.json"
        in workflow
    )
    assert (
        "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_source_semantics_boundary.json"
        in workflow
    )
    assert (
        "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_policy.json"
        in workflow
    )
    assert (
        "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_freshness_report.json"
        in workflow
    )
    assert (
        "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_promotion_decision.json"
        in workflow
    )
