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


def test_simplified_likelihood_exporter_stable_evidence_policy_schema_example_and_builder_smoke(
    tmp_path: Path,
) -> None:
    repo = _repo_root()
    schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_evidence_policy_v0.schema.json"
    )
    example_path = (
        repo
        / "docs"
        / "specs"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_evidence_policy_v0.example.json"
    )
    current_dir = (
        repo
        / "benchmarks"
        / "artifacts"
        / "simplified_likelihood_export_benchmarks"
        / "nextstat-bench"
        / "current"
    )
    accepted_dir = (
        repo
        / "benchmarks"
        / "artifacts"
        / "simplified_likelihood_exporter_promotion_bundles"
        / "nextstat-bench"
        / "accepted"
    )

    example = _load_json(example_path)
    _validate_jsonschema(example, schema_path)
    assert (
        example["schema_version"]
        == "nextstat_simplified_likelihood_exporter_stable_evidence_policy_v0"
    )

    out_path = tmp_path / "stable_evidence_policy.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/build_simplified_likelihood_exporter_stable_evidence_policy.py",
            "--benchmark-artifact",
            str(current_dir / "apex2_simplified_likelihood_report.json"),
            "--public-validation-report",
            str(current_dir / "export_public_validation_report.json"),
            "--stable-promotion-decision",
            str(accepted_dir / "stable_promotion_decision.json"),
            "--out",
            str(out_path),
            "--deterministic",
        ],
        cwd=repo,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout

    generated = _load_json(out_path)
    _validate_jsonschema(generated, schema_path)
    assert generated["status"] == "accepted"
    assert generated["support_class"] == "stable"
    assert generated["benchmark_host"] == "nextstat-bench"
    assert generated["stable_evidence_floor"]["min_total_export_matrix_case_count"] == 10
    assert (
        generated["stable_evidence_floor"]["min_public_reinterpretation_style_case_count"]
        == 8
    )
    assert (
        generated["maintenance_cadence"]["refresh_cadence"]
        == "on_every_exporter_release_pr_or_public_case_admission"
    )
    assert generated["admission_policy"]["required_case_kind"] == "public_reinterpretation_style"


def test_simplified_likelihood_exporter_stable_evidence_policy_docs_and_committed_artifact_are_published() -> None:
    repo = _repo_root()

    assert_doc_contains_strings(
        repo / "docs" / "README.md",
        [
            "simplified-likelihood-exporter-stable-evidence-policy-2026-03-09.md",
            "simplified_likelihood_exporter_stable_evidence_policy_v0.example.json",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks.md",
        [
            "simplified-likelihood-exporter-stable-evidence-policy-2026-03-09",
            "simplified_likelihood_exporter_stable_evidence_policy_v0.example.json",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "references" / "simplified-likelihood-artifacts.md",
        [
            "simplified-likelihood-exporter-stable-evidence-policy-2026-03-09.md",
            "simplified_likelihood_exporter_stable_evidence_policy_v0",
            "stable_evidence_policy.json",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-exporter-release-pr-checklist-2026-03-09.md",
        [
            "stable_evidence_policy.json",
            "8 public / 10 total",
            "admission policy",
            "maintenance cadence",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-exporter-promotion-runbook-2026-03-09.md",
        [
            "stable_evidence_policy.json",
            "build_simplified_likelihood_exporter_stable_evidence_policy.py",
            "8 public reinterpretation-style cases",
        ],
    )

    accepted_policy_path = (
        repo
        / "benchmarks"
        / "artifacts"
        / "simplified_likelihood_exporter_promotion_bundles"
        / "nextstat-bench"
        / "accepted"
        / "stable_evidence_policy.json"
    )
    policy = _load_json(accepted_policy_path)
    _validate_jsonschema(
        policy,
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_evidence_policy_v0.schema.json",
    )
    assert policy["status"] == "accepted"
    assert policy["support_class"] == "stable"
