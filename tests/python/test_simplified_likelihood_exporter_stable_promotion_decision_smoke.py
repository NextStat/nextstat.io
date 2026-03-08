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


def test_simplified_likelihood_exporter_stable_promotion_decision_schema_example_and_generator_smoke(
    tmp_path: Path,
) -> None:
    repo = _repo_root()
    schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_promotion_decision_v0.schema.json"
    )
    example_path = (
        repo
        / "docs"
        / "specs"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_promotion_decision_v0.example.json"
    )

    example = _load_json(example_path)
    _validate_jsonschema(example, schema_path)
    assert (
        example["schema_version"]
        == "nextstat_simplified_likelihood_exporter_stable_promotion_decision_v0"
    )

    decision_path = tmp_path / "stable_promotion_decision.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/assess_simplified_likelihood_exporter_stable_promotion_decision.py",
            "--bundle-dir",
            str(_accepted_bundle_dir()),
            "--out",
            str(decision_path),
            "--deterministic",
        ],
        cwd=repo,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    generated = _load_json(decision_path)
    _validate_jsonschema(generated, schema_path)

    assert generated["support_class"] == "stable"
    assert generated["status"] == "accepted"
    assert generated["automatic_stable_promotion"] is False
    assert generated["release_consumption"]["release_assets_wired"] is True
    assert generated["release_consumption"]["standalone_upload_present"] is True
    assert generated["decision"]["decision_status"] == "accepted"
    assert generated["decision"]["benchmark_host"] == "nextstat-bench"
    assert generated["summary"]["status"] == "accepted"
    assert generated["summary"]["benchmark_host"] == "nextstat-bench"
    assert generated["summary"]["stable_scope_promoted"] is True
    assert generated["summary"]["open_blocker_count_observed"] == 0


def test_simplified_likelihood_exporter_stable_promotion_decision_docs_gate_and_workflow_are_published() -> None:
    repo = _repo_root()

    assert_doc_contains_strings(
        repo / "docs" / "README.md",
        [
            "simplified-likelihood-exporter-stable-promotion-decision-2026-03-09.md",
            "simplified-likelihood-exporter-release-pr-checklist-2026-03-09.md",
            "simplified_likelihood_exporter_stable_promotion_decision_v0.example.json",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks.md",
        [
            "simplified-likelihood-exporter-stable-promotion-decision-2026-03-09",
            "simplified-likelihood-exporter-release-pr-checklist-2026-03-09",
            "simplified_likelihood_exporter_stable_promotion_decision_v0.example.json",
        ],
    )
    assert_doc_contains_strings(
        repo
        / "docs"
        / "benchmarks"
        / "simplified-likelihood-exporter-stable-promotion-decision-2026-03-09.md",
        [
            "`nextstat simplify workspace`",
            "`stable`",
            "single-POI",
            "source_model_constraints",
            "Gaussian-constrained",
            "stable_promotion_decision.json",
            ".github/workflows/release.yml",
        ],
    )
    assert_doc_contains_strings(
        repo
        / "docs"
        / "benchmarks"
        / "simplified-likelihood-exporter-release-pr-checklist-2026-03-09.md",
        [
            "stable_promotion_decision.json",
            ".github/workflows/release.yml",
            "simplified-likelihood-exporter-surface.yml",
            "`stable`",
            "research-grade fallback",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "references" / "simplified-likelihood-artifacts.md",
        [
            "simplified_likelihood_exporter_stable_promotion_decision_v0",
            "assess_simplified_likelihood_exporter_stable_promotion_decision.py",
            "stable_promotion_decision.json",
            "simplified-likelihood-exporter-stable-promotion-decision-2026-03-09.md",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-exporter-runtime-gate.md",
        [
            "stable_promotion_decision.json",
            "assess_simplified_likelihood_exporter_stable_promotion_decision.py",
            "simplified-likelihood-exporter-stable-promotion-decision-2026-03-09.md",
        ],
    )

    workflow = (
        repo / ".github" / "workflows" / "simplified-likelihood-exporter-surface.yml"
    ).read_text(encoding="utf-8")
    assert (
        "tests/python/test_simplified_likelihood_exporter_stable_promotion_decision_smoke.py"
        in workflow
    )
    assert (
        "docs/schemas/benchmarks/simplified_likelihood_exporter_stable_promotion_decision_v0.schema.json"
        in workflow
    )
    assert (
        "docs/specs/benchmarks/simplified_likelihood_exporter_stable_promotion_decision_v0.example.json"
        in workflow
    )
    assert (
        "docs/benchmarks/simplified-likelihood-exporter-stable-promotion-decision-2026-03-09.md"
        in workflow
    )
    assert (
        "docs/benchmarks/simplified-likelihood-exporter-release-pr-checklist-2026-03-09.md"
        in workflow
    )
    assert (
        "scripts/benchmarks/assess_simplified_likelihood_exporter_stable_promotion_decision.py"
        in workflow
    )
    assert (
        "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_promotion_decision.json"
        in workflow
    )

    gate = (repo / "scripts" / "benchmarks" / "simplified_likelihood_exporter_surface_gate.sh").read_text(
        encoding="utf-8"
    )
    assert "test_simplified_likelihood_exporter_stable_promotion_decision_smoke.py" in gate
    assert "assess_simplified_likelihood_exporter_stable_promotion_decision.py" in gate
    assert "stable_promotion_decision.json" in gate
    assert "simplified-likelihood-exporter-stable-promotion-decision-2026-03-09.md" in gate


def test_simplified_likelihood_exporter_committed_stable_promotion_decision_is_published() -> None:
    repo = _repo_root()
    decision_path = _accepted_bundle_dir() / "stable_promotion_decision.json"
    schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_promotion_decision_v0.schema.json"
    )

    decision = _load_json(decision_path)
    _validate_jsonschema(decision, schema_path)
    assert decision["support_class"] == "stable"
    assert decision["status"] == "accepted"
    assert decision["automatic_stable_promotion"] is False
    assert decision["decision"]["decision_status"] == "accepted"
    assert decision["summary"]["release_assets_wired"] is True
    assert decision["summary"]["status"] == "accepted"
    assert decision["summary"]["benchmark_host"] == "nextstat-bench"
