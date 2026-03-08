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


def test_simplified_likelihood_exporter_stable_candidate_review_packet_schema_example_and_generator_smoke(
    tmp_path: Path,
) -> None:
    repo = _repo_root()
    schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_candidate_review_packet_v0.schema.json"
    )
    example_path = (
        repo
        / "docs"
        / "specs"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_candidate_review_packet_v0.example.json"
    )

    example = _load_json(example_path)
    _validate_jsonschema(example, schema_path)
    assert (
        example["schema_version"]
        == "nextstat_simplified_likelihood_exporter_stable_candidate_review_packet_v0"
    )

    review_packet_path = tmp_path / "stable_candidate_review_packet.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/build_simplified_likelihood_exporter_stable_candidate_review_packet.py",
            "--bundle-dir",
            str(_accepted_bundle_dir()),
            "--out",
            str(review_packet_path),
            "--deterministic",
        ],
        cwd=repo,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    generated = _load_json(review_packet_path)
    _validate_jsonschema(generated, schema_path)

    assert generated["support_class"] == "stable"
    assert generated["automatic_stable_promotion"] is False
    assert generated["packet_validity"]["passed"] is True
    assert generated["review_packet"]["ready"] is True
    assert generated["review_packet"]["status"] == "ready"
    assert generated["review_packet"]["recommendation_status"] == "stable_promoted"
    assert generated["review_packet"]["recommended_support_class"] == "stable"
    assert generated["review_packet"]["target_support_class"] == "stable"
    assert generated["review_packet"]["open_blocker_count"] == 0
    assert generated["summary"]["status"] == "ready"
    assert generated["summary"]["benchmark_host"] == "nextstat-bench"
    assert generated["summary"]["open_blocker_count"] == 0
    assert generated["remaining_blockers"] == []


def test_simplified_likelihood_exporter_stable_candidate_review_packet_marks_missing_blocker_matrix_invalid(
    tmp_path: Path,
) -> None:
    repo = _repo_root()
    bundle_dir = tmp_path / "bundle"
    shutil.copytree(_accepted_bundle_dir(), bundle_dir)
    (bundle_dir / "stable_candidate_blocker_matrix.json").unlink()

    review_packet_path = tmp_path / "stable_candidate_review_packet_invalid.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/build_simplified_likelihood_exporter_stable_candidate_review_packet.py",
            "--bundle-dir",
            str(bundle_dir),
            "--out",
            str(review_packet_path),
            "--deterministic",
        ],
        cwd=repo,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    generated = _load_json(review_packet_path)

    assert generated["packet_validity"]["passed"] is False
    assert generated["review_packet"]["ready"] is False
    assert generated["review_packet"]["status"] == "incomplete"
    assert any(
        failure.get("reason") == "missing_required_artifact"
        and failure.get("artifact") == "stable_candidate_blocker_matrix"
        for failure in generated["packet_validity"]["failures"]
    )


def test_simplified_likelihood_exporter_stable_candidate_review_packet_docs_gate_and_workflow_are_published() -> None:
    repo = _repo_root()

    assert_doc_contains_strings(
        repo / "docs" / "README.md",
        [
            "simplified-likelihood-exporter-stable-candidate-review-packet-2026-03-09.md",
            "simplified_likelihood_exporter_stable_candidate_review_packet_v0.example.json",
            "simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks.md",
        [
            "simplified-likelihood-exporter-stable-candidate-review-packet-2026-03-09",
            "simplified_likelihood_exporter_stable_candidate_review_packet_v0.example.json",
            "simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09",
        ],
    )
    assert_doc_contains_strings(
        repo
        / "docs"
        / "benchmarks"
        / "simplified-likelihood-exporter-stable-candidate-review-packet-2026-03-09.md",
        [
            "`stable`",
            "stable_candidate_review_packet.json",
            "stable source semantics boundary",
            "explicit stable promotion decision",
            "stable_promotion_decision.json",
            "nextstat-bench",
            "open_blocker_count = 0",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "references" / "simplified-likelihood-artifacts.md",
        [
            "simplified_likelihood_exporter_stable_candidate_review_packet_v0",
            "build_simplified_likelihood_exporter_stable_candidate_review_packet.py",
            "stable_candidate_review_packet.json",
            "stable_source_semantics_boundary.json",
            "stable_promotion_decision.json",
            "simplified-likelihood-exporter-stable-candidate-review-packet-2026-03-09.md",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-exporter-runtime-gate.md",
        [
            "stable_candidate_review_packet.json",
            "build_simplified_likelihood_exporter_stable_candidate_review_packet.py",
            "simplified-likelihood-exporter-stable-candidate-review-packet-2026-03-09.md",
            "stable_source_semantics_boundary.json",
            "stable_promotion_decision.json",
        ],
    )

    workflow = (
        repo / ".github" / "workflows" / "simplified-likelihood-exporter-surface.yml"
    ).read_text(encoding="utf-8")
    assert (
        "tests/python/test_simplified_likelihood_exporter_stable_candidate_review_packet_smoke.py"
        in workflow
    )
    assert (
        "docs/schemas/benchmarks/simplified_likelihood_exporter_stable_candidate_review_packet_v0.schema.json"
        in workflow
    )
    assert (
        "docs/specs/benchmarks/simplified_likelihood_exporter_stable_candidate_review_packet_v0.example.json"
        in workflow
    )
    assert (
        "docs/benchmarks/simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md"
        in workflow
    )
    assert (
        "docs/benchmarks/simplified-likelihood-exporter-stable-promotion-decision-2026-03-09.md"
        in workflow
    )
    assert (
        "scripts/benchmarks/build_simplified_likelihood_exporter_stable_candidate_review_packet.py"
        in workflow
    )
    assert (
        "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_review_packet.json"
        in workflow
    )

    gate = (repo / "scripts" / "benchmarks" / "simplified_likelihood_exporter_surface_gate.sh").read_text(
        encoding="utf-8"
    )
    assert "test_simplified_likelihood_exporter_stable_candidate_review_packet_smoke.py" in gate
    assert "build_simplified_likelihood_exporter_stable_candidate_review_packet.py" in gate
    assert "stable_candidate_review_packet.json" in gate
    assert "stable_source_semantics_boundary.json" in gate
    assert "stable_promotion_decision.json" in gate
    assert "simplified-likelihood-exporter-stable-candidate-review-packet-2026-03-09.md" in gate


def test_simplified_likelihood_exporter_committed_stable_candidate_review_packet_is_published() -> None:
    repo = _repo_root()
    packet_path = _accepted_bundle_dir() / "stable_candidate_review_packet.json"
    schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_candidate_review_packet_v0.schema.json"
    )

    packet = _load_json(packet_path)
    _validate_jsonschema(packet, schema_path)
    assert packet["summary"]["status"] == "ready"
    assert packet["summary"]["benchmark_host"] == "nextstat-bench"
    assert packet["support_class"] == "stable"
    assert packet["automatic_stable_promotion"] is False
    assert packet["review_packet"]["recommendation_status"] == "stable_promoted"
    assert packet["review_packet"]["open_blocker_count"] == 0
