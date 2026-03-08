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


def _accepted_bundle_dir() -> Path:
    repo = _repo_root()
    return (
        repo
        / "benchmarks"
        / "artifacts"
        / "simplified_likelihood_exporter_promotion_bundles"
        / "nextstat-bench"
        / "accepted"
    )


def test_simplified_likelihood_exporter_stable_source_semantics_boundary_schema_example_and_generator_smoke(
    tmp_path: Path,
) -> None:
    repo = _repo_root()
    schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_source_semantics_boundary_v0.schema.json"
    )
    example_path = (
        repo
        / "docs"
        / "specs"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_source_semantics_boundary_v0.example.json"
    )

    example = _load_json(example_path)
    _validate_jsonschema(example, schema_path)
    assert (
        example["schema_version"]
        == "nextstat_simplified_likelihood_exporter_stable_source_semantics_boundary_v0"
    )

    boundary_path = tmp_path / "stable_source_semantics_boundary.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/build_simplified_likelihood_exporter_stable_source_semantics_boundary.py",
            "--out",
            str(boundary_path),
            "--deterministic",
        ],
        cwd=repo,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    generated = _load_json(boundary_path)
    _validate_jsonschema(generated, schema_path)

    assert generated["support_class"] == "research-grade"
    assert generated["target_support_class"] == "stable"
    assert generated["automatic_stable_promotion"] is False
    assert generated["status"] == "published"
    assert generated["future_stable_boundary"]["source_workspace_formats"] == ["pyhf"]
    assert generated["future_stable_boundary"]["poi_scope"] == "single_poi"
    assert (
        generated["future_stable_boundary"]["supported_constraint_covariance_source"]
        == "source_model_constraints"
    )
    assert generated["future_stable_boundary"]["supported_source_constraint_families"] == [
        "gaussian"
    ]
    assert (
        generated["future_stable_boundary"]["source_level_nuisance_identity_preserved"]
        is False
    )
    assert generated["summary"]["status"] == "published"
    assert generated["summary"]["blocker_resolution_supported"] is True


def test_simplified_likelihood_exporter_stable_source_semantics_boundary_docs_gate_and_workflow_are_published() -> None:
    repo = _repo_root()

    assert_doc_contains_strings(
        repo / "docs" / "README.md",
        [
            "simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md",
            "simplified_likelihood_exporter_stable_source_semantics_boundary_v0.example.json",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks.md",
        [
            "simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09",
            "simplified_likelihood_exporter_stable_source_semantics_boundary_v0.example.json",
        ],
    )
    assert_doc_contains_strings(
        repo
        / "docs"
        / "benchmarks"
        / "simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md",
        [
            "`nextstat simplify workspace`",
            "single-POI only",
            "source_model_constraints",
            "Gaussian-constrained",
            "stable_source_semantics_boundary.json",
            "published stable boundary for the promoted narrow exporter subset",
            "everything outside this boundary remains `research-grade`",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "references" / "simplified-likelihood-artifacts.md",
        [
            "simplified_likelihood_exporter_stable_source_semantics_boundary_v0",
            "build_simplified_likelihood_exporter_stable_source_semantics_boundary.py",
            "stable_source_semantics_boundary.json",
            "simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-exporter-runtime-gate.md",
        [
            "stable_source_semantics_boundary.json",
            "build_simplified_likelihood_exporter_stable_source_semantics_boundary.py",
            "simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md",
        ],
    )

    workflow = (
        repo / ".github" / "workflows" / "simplified-likelihood-exporter-surface.yml"
    ).read_text(encoding="utf-8")
    assert (
        "tests/python/test_simplified_likelihood_exporter_stable_source_semantics_boundary_smoke.py"
        in workflow
    )
    assert (
        "docs/schemas/benchmarks/simplified_likelihood_exporter_stable_source_semantics_boundary_v0.schema.json"
        in workflow
    )
    assert (
        "docs/specs/benchmarks/simplified_likelihood_exporter_stable_source_semantics_boundary_v0.example.json"
        in workflow
    )
    assert (
        "scripts/benchmarks/build_simplified_likelihood_exporter_stable_source_semantics_boundary.py"
        in workflow
    )
    assert (
        "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_source_semantics_boundary.json"
        in workflow
    )

    gate = (
        repo / "scripts" / "benchmarks" / "simplified_likelihood_exporter_surface_gate.sh"
    ).read_text(encoding="utf-8")
    assert "test_simplified_likelihood_exporter_stable_source_semantics_boundary_smoke.py" in gate
    assert "build_simplified_likelihood_exporter_stable_source_semantics_boundary.py" in gate
    assert "stable_source_semantics_boundary.json" in gate
    assert "simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md" in gate


def test_simplified_likelihood_exporter_committed_stable_source_semantics_boundary_is_published() -> None:
    repo = _repo_root()
    boundary_path = _accepted_bundle_dir() / "stable_source_semantics_boundary.json"
    schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_source_semantics_boundary_v0.schema.json"
    )

    boundary = _load_json(boundary_path)
    _validate_jsonschema(boundary, schema_path)
    assert boundary["status"] == "published"
    assert boundary["support_class"] == "research-grade"
    assert boundary["target_support_class"] == "stable"
    assert boundary["automatic_stable_promotion"] is False
    assert boundary["future_stable_boundary"]["poi_scope"] == "single_poi"
    assert (
        boundary["future_stable_boundary"]["supported_constraint_covariance_source"]
        == "source_model_constraints"
    )
