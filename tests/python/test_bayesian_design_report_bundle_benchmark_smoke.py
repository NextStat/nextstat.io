from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from _io_contract_doc_assertions import assert_doc_contains_strings


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_bayesian_design_report_bundle_benchmark_schema_and_example_smoke() -> None:
    jsonschema = pytest.importorskip("jsonschema")

    schema_path = (
        _repo_root()
        / "docs"
        / "schemas"
        / "benchmarks"
        / "bayesian_design_report_bundle_benchmark_result_v1.schema.json"
    )
    example_path = (
        _repo_root()
        / "docs"
        / "specs"
        / "pharma"
        / "bayesian_design_report_bundle_benchmark_result_v1.example.json"
    )

    schema_doc = _load_json(schema_path)
    example_doc = _load_json(example_path)

    assert schema_doc["$id"] == (
        "https://nextstat.io/schemas/benchmarks/bayesian_design_report_bundle_benchmark_result_v1.schema.json"
    )
    jsonschema.Draft202012Validator.check_schema(schema_doc)
    jsonschema.validate(example_doc, schema_doc)

    assert example_doc["schema_version"] == "nextstat.bayesian_design_report_bundle_benchmark_result.v1"
    assert example_doc["suite"] == "bayesian_design_report_bundle_packaging"
    assert len(example_doc["cases"]) >= 4


def test_bayesian_design_report_bundle_benchmark_runner_smoke(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    out_path = tmp_path / "summary.json"
    work_root = tmp_path / "work"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/benchmarks/bench_bayesian_design_report_bundle.py",
            "--smoke",
            "--deterministic",
            "--out",
            str(out_path),
            "--work-root",
            str(work_root),
        ],
        cwd=_repo_root(),
    )

    schema_doc = _load_json(
        _repo_root()
        / "docs"
        / "schemas"
        / "benchmarks"
        / "bayesian_design_report_bundle_benchmark_result_v1.schema.json"
    )
    report = _load_json(out_path)
    jsonschema.validate(report, schema_doc)

    assert report["schema_version"] == "nextstat.bayesian_design_report_bundle_benchmark_result.v1"
    assert report["suite"] == "bayesian_design_report_bundle_packaging"
    assert report["meta"]["smoke"] is True
    assert report["meta"]["host_policy"] == "nextstat-bench"
    assert report["deterministic"] is True
    assert report["budget"]["schema_version"] == (
        "nextstat.bayesian_design_report_bundle_performance_budget.v1"
    )

    cases = {case["id"]: case for case in report["cases"]}
    assert {"beta_small", "beta_large", "normal_small", "normal_large"} <= set(cases)
    assert all(case["status"] == "ok" for case in cases.values())
    assert all(case["validation"]["created_unix_ms_zero"] is True for case in cases.values())
    assert all(case["validation"]["summary_deterministic"] is True for case in cases.values())
    assert all(case["validation"]["required_artifacts_present"] is True for case in cases.values())
    assert all(case["budget_pass"]["bundle_duration"] is True for case in cases.values())
    assert all(case["budget_pass"]["manifest_regen_duration"] is True for case in cases.values())
    assert all(case["budget_pass"]["bundle_bytes"] is True for case in cases.values())
    assert all(case["budget_pass"]["manifest_bytes"] is True for case in cases.values())


def test_bayesian_design_report_bundle_benchmark_docs_publish_gate_workflow() -> None:
    assert_doc_contains_strings(
        _repo_root() / "docs" / "benchmarks" / "bayesian-design-report-packaging-runtime-gate.md",
        [
            "# Bayesian Design Report Packaging Runtime Gate",
            "nextstat-bench",
            "scripts/benchmarks/bench_bayesian_design_report_bundle.py",
            "scripts/bayesian_design_report_bundle_performance_budget_v1.json",
            "docs/schemas/benchmarks/bayesian_design_report_bundle_benchmark_result_v1.schema.json",
            "docs/specs/pharma/bayesian_design_report_bundle_benchmark_result_v1.example.json",
        ],
    )
    assert_doc_contains_strings(
        _repo_root() / "docs" / "specs" / "pharma" / "bayesian_design_report_bundle_acceptance_v0.md",
        [
            "docs/benchmarks/bayesian-design-report-packaging-runtime-gate.md",
            "nextstat-bench",
            "scripts/benchmarks/bench_bayesian_design_report_bundle.py",
            "scripts/bayesian_design_report_bundle_performance_budget_v1.json",
            "docs/schemas/benchmarks/bayesian_design_report_bundle_benchmark_result_v1.schema.json",
            "docs/specs/pharma/bayesian_design_report_bundle_benchmark_result_v1.example.json",
        ],
    )
    assert_doc_contains_strings(
        _repo_root() / "docs" / "references" / "bayesian-trial-design-artifacts.md",
        [
            "docs/benchmarks/bayesian-design-report-packaging-runtime-gate.md",
            "docs/specs/pharma/bayesian_design_report_bundle_benchmark_result_v1.example.json",
        ],
    )
