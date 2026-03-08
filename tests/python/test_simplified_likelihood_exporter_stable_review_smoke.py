from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
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


def test_simplified_likelihood_exporter_stable_review_assessment_schema_example_and_generator_smoke(
    tmp_path: Path,
) -> None:
    repo = _repo_root()
    schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_review_assessment_v0.schema.json"
    )
    example_path = (
        repo
        / "docs"
        / "specs"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_review_assessment_v0.example.json"
    )

    example = _load_json(example_path)
    _validate_jsonschema(example, schema_path)
    assert (
        example["schema_version"]
        == "nextstat_simplified_likelihood_exporter_stable_review_assessment_v0"
    )

    assessment_path = tmp_path / "stable_review_assessment.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/assess_simplified_likelihood_exporter_stable_review.py",
            "--bundle-dir",
            str(_accepted_bundle_dir()),
            "--out",
            str(assessment_path),
            "--deterministic",
        ],
        cwd=repo,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    generated = _load_json(assessment_path)
    _validate_jsonschema(generated, schema_path)

    assert generated["support_class"] == "research-grade"
    assert generated["automatic_stable_promotion"] is False
    assert generated["evidence_validity"]["passed"] is True
    assert generated["stable_review"]["ready"] is True
    assert generated["stable_review"]["status"] == "review_ready"
    assert generated["stable_review"]["policy"]["required_benchmark_host"] == "nextstat-bench"
    assert generated["stable_review"]["policy"]["required_promotion_report_status"] == "promoted"
    assert generated["summary"]["status"] == "review_ready"
    assert generated["summary"]["benchmark_host"] == "nextstat-bench"


def test_simplified_likelihood_exporter_stable_review_assessment_marks_invalid_bundle_not_ready(
    tmp_path: Path,
) -> None:
    repo = _repo_root()
    bundle_dir = tmp_path / "bundle"
    shutil.copytree(_accepted_bundle_dir(), bundle_dir)

    check_path = bundle_dir / "promotion_evidence_check.json"
    check_doc = _load_json(check_path)
    check_doc["status"] = "failed"
    check_doc["ok"] = False
    check_doc["checks"]["promotion_readiness"]["status"] = "failed"
    check_doc["checks"]["promotion_readiness"]["ok"] = False
    check_doc["checks"]["promotion_readiness"]["actual_future_stable_review_ready"] = False
    check_doc["checks"]["promotion_readiness"]["errors"] = ["tampered for smoke"]
    check_path.write_text(json.dumps(check_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    assessment_path = tmp_path / "stable_review_assessment_invalid.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/assess_simplified_likelihood_exporter_stable_review.py",
            "--bundle-dir",
            str(bundle_dir),
            "--out",
            str(assessment_path),
            "--deterministic",
        ],
        cwd=repo,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    generated = _load_json(assessment_path)

    assert generated["evidence_validity"]["passed"] is False
    assert generated["stable_review"]["ready"] is False
    assert generated["stable_review"]["status"] == "not_ready"
    assert any(
        failure.get("reason") == "bundle_check_not_passed"
        for failure in generated["stable_review"]["failures"]
    )


def test_simplified_likelihood_exporter_stable_review_docs_gate_and_workflow_are_published() -> None:
    repo = _repo_root()

    assert_doc_contains_strings(
        repo / "docs" / "README.md",
        [
            "simplified-likelihood-exporter-stable-review-checklist-2026-03-09.md",
            "simplified_likelihood_exporter_stable_review_assessment_v0.example.json",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks.md",
        [
            "simplified-likelihood-exporter-stable-review-checklist-2026-03-09",
            "simplified_likelihood_exporter_stable_review_assessment_v0.example.json",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-exporter-stable-review-checklist-2026-03-09.md",
        [
            "`research-grade`",
            "automatic stable promotion",
            "stable_review_assessment.json",
            "review_ready",
            "does not by itself promote `nextstat simplify workspace` to `stable`",
            "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/",
            ".github/workflows/simplified-likelihood-exporter-surface.yml",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "references" / "simplified-likelihood-artifacts.md",
        [
            "simplified_likelihood_exporter_stable_review_assessment_v0",
            "assess_simplified_likelihood_exporter_stable_review.py",
            "stable_review_assessment.json",
            "simplified-likelihood-exporter-stable-review-checklist-2026-03-09.md",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-exporter-runtime-gate.md",
        [
            "stable_review_assessment.json",
            "assess_simplified_likelihood_exporter_stable_review.py",
            "simplified-likelihood-exporter-stable-review-checklist-2026-03-09.md",
        ],
    )

    workflow = (
        repo / ".github" / "workflows" / "simplified-likelihood-exporter-surface.yml"
    ).read_text(encoding="utf-8")
    assert "tests/python/test_simplified_likelihood_exporter_stable_review_smoke.py" in workflow
    assert (
        "docs/schemas/benchmarks/simplified_likelihood_exporter_stable_review_assessment_v0.schema.json"
        in workflow
    )
    assert (
        "docs/specs/benchmarks/simplified_likelihood_exporter_stable_review_assessment_v0.example.json"
        in workflow
    )
    assert "scripts/benchmarks/assess_simplified_likelihood_exporter_stable_review.py" in workflow
    assert (
        "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_review_assessment.json"
        in workflow
    )

    gate = (repo / "scripts" / "benchmarks" / "simplified_likelihood_exporter_surface_gate.sh").read_text(
        encoding="utf-8"
    )
    assert "test_simplified_likelihood_exporter_stable_review_smoke.py" in gate
    assert "assess_simplified_likelihood_exporter_stable_review.py" in gate
    assert "stable_review_assessment.json" in gate
    assert "simplified-likelihood-exporter-stable-review-checklist-2026-03-09.md" in gate


def test_simplified_likelihood_exporter_committed_stable_review_assessment_is_published() -> None:
    repo = _repo_root()
    assessment_path = _accepted_bundle_dir() / "stable_review_assessment.json"
    schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_review_assessment_v0.schema.json"
    )

    assessment = _load_json(assessment_path)
    _validate_jsonschema(assessment, schema_path)
    assert assessment["summary"]["status"] == "review_ready"
    assert assessment["automatic_stable_promotion"] is False
