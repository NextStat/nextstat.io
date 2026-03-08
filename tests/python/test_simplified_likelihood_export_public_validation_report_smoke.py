from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from _io_contract_doc_assertions import assert_doc_contains_strings


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_jsonschema(instance: dict, schema_path: Path) -> None:
    import jsonschema  # type: ignore

    schema = _load_json(schema_path)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(instance=instance, schema=schema)


def test_simplified_likelihood_export_public_validation_report_schema_example_and_builder_smoke(
    tmp_path: Path,
) -> None:
    repo = _repo_root()
    schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_export_public_validation_report_v0.schema.json"
    )
    example_path = (
        repo
        / "docs"
        / "specs"
        / "benchmarks"
        / "simplified_likelihood_export_public_validation_report_v0.example.json"
    )
    committed_report_path = (
        repo
        / "benchmarks"
        / "artifacts"
        / "simplified_likelihood_export_benchmarks"
        / "nextstat-bench"
        / "current"
        / "export_public_validation_report.json"
    )
    benchmark_artifact = (
        repo
        / "benchmarks"
        / "artifacts"
        / "simplified_likelihood_export_benchmarks"
        / "nextstat-bench"
        / "current"
        / "apex2_simplified_likelihood_report.json"
    )
    catalog_path = (
        repo / "docs" / "specs" / "apex2_simplified_likelihood_export_public_case_catalog_v0.example.json"
    )
    generated_path = tmp_path / "export_public_validation_report.json"

    example = _load_json(example_path)
    _validate_jsonschema(example, schema_path)
    assert (
        example["schema_version"]
        == "nextstat_simplified_likelihood_export_public_validation_report_v0"
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/build_simplified_likelihood_export_public_validation_report.py",
            "--benchmark-artifact",
            str(benchmark_artifact),
            "--catalog",
            str(catalog_path),
            "--out",
            str(generated_path),
            "--deterministic",
        ],
        cwd=repo,
        env={**os.environ},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "status=ok" in proc.stdout, proc.stdout

    committed_report = _load_json(committed_report_path)
    generated_report = _load_json(generated_path)
    _validate_jsonschema(committed_report, schema_path)
    _validate_jsonschema(generated_report, schema_path)
    assert committed_report == generated_report
    assert committed_report["status"] == "ok"
    assert committed_report["summary"]["benchmark_host"] == "nextstat-bench"
    assert committed_report["summary"]["public_case_count"] == 6
    assert committed_report["summary"]["catalog_case_count"] == 6
    assert committed_report["summary"]["public_case_names"] == [
        "atlas_public_dual_sr_dual_cr_gaussian_export_stable_example",
        "atlas_public_dual_sr_vr_dual_cr_gaussian_export_stable_example",
        "atlas_public_sr_cr_gaussian_export_stable_example",
        "cms_public_sr_cr_export_stable_example",
        "cms_public_sr_cr_asymmetric_gaussian_export_stable_example",
        "cms_public_dual_sr_cr_gaussian_export_stable_example",
    ]
    assert committed_report["summary"]["all_schema_valid"] is True
    assert committed_report["summary"]["all_fidelity_gates_pass"] is True
    assert committed_report["summary"]["all_performance_gates_pass"] is True
    assert committed_report["summary"]["all_cases_within_promoted_stable_runtime_boundary"] is True
    assert committed_report["summary"]["all_cases_gaussian_constrained_source_workspaces"] is True
    assert committed_report["summary"]["max_abs_q_mu_diff"] <= 0.1
    assert committed_report["summary"]["max_upper_limit_ratio_deviation"] <= 0.05
    assert committed_report["summary"]["min_net_end_to_end_upper_limit_speedup"] >= 0.75
    assert committed_report["boundary"]["surface_support_class"] == "stable-evidence"
    assert committed_report["boundary"]["does_not_expand_promoted_runtime_claim"] is True
    assert (
        committed_report["boundary"]["public_cases_outside_promoted_runtime_boundary_allowed"]
        is False
    )
    assert committed_report["summary"]["cases_outside_promoted_stable_runtime_boundary"] == 0
    assert committed_report["summary"]["observed_constraint_covariance_sources"] == [
        "source_model_constraints"
    ]
    assert "histosys" in committed_report["summary"]["observed_source_workspace_modifier_types"]
    assert "normsys" in committed_report["summary"]["observed_source_workspace_modifier_types"]


def test_simplified_likelihood_export_public_validation_surface_docs_are_published() -> None:
    repo = _repo_root()

    assert_doc_contains_strings(
        repo / "docs" / "README.md",
        [
            "simplified-likelihood-exporter-public-validation-surface-2026-03-09.md",
            "simplified_likelihood_export_public_validation_report_v0.example.json",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks.md",
        [
            "simplified-likelihood-exporter-public-validation-surface-2026-03-09",
            "simplified_likelihood_export_public_validation_report_v0.example.json",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-exporter-public-validation-surface-2026-03-09.md",
        [
            "stable evidence surface",
            "export_public_validation_report.json",
            "does not widen the promoted stable runtime claim",
            "public_reinterpretation_style",
            "source_model_constraints",
        ],
    )
