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


def test_simplified_likelihood_exporter_stable_candidate_blocker_matrix_schema_example_and_generator_smoke(
    tmp_path: Path,
) -> None:
    repo = _repo_root()
    schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0.schema.json"
    )
    example_path = (
        repo
        / "docs"
        / "specs"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0.example.json"
    )

    example = _load_json(example_path)
    _validate_jsonschema(example, schema_path)
    assert (
        example["schema_version"]
        == "nextstat_simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0"
    )

    matrix_path = tmp_path / "stable_candidate_blocker_matrix.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/assess_simplified_likelihood_exporter_stable_candidate_blockers.py",
            "--bundle-dir",
            str(_accepted_bundle_dir()),
            "--out",
            str(matrix_path),
            "--deterministic",
        ],
        cwd=repo,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    generated = _load_json(matrix_path)
    _validate_jsonschema(generated, schema_path)

    assert generated["support_class"] == "stable"
    assert generated["automatic_stable_promotion"] is False
    assert generated["stable_candidate"]["ready"] is True
    assert generated["stable_candidate"]["status"] == "ready"
    assert generated["stable_candidate"]["open_blocker_count"] == 0
    assert generated["summary"]["benchmark_host"] == "nextstat-bench"
    assert generated["summary"]["status"] == "ready"
    public_blocker = next(
        blocker
        for blocker in generated["blockers"]
        if blocker["blocker_id"] == "public_exporter_matrix_not_yet_part_of_stable_candidate_evidence"
    )
    assert public_blocker["status"] == "resolved"
    assert public_blocker["blocking"] is False
    review_packet_blocker = next(
        blocker
        for blocker in generated["blockers"]
        if blocker["blocker_id"] == "stable_candidate_review_packet_not_yet_published"
    )
    assert review_packet_blocker["status"] == "resolved"
    assert review_packet_blocker["blocking"] is False
    source_semantics_blocker = next(
        blocker
        for blocker in generated["blockers"]
        if blocker["blocker_id"] == "stable_source_semantics_boundary_not_yet_promoted"
    )
    assert source_semantics_blocker["status"] == "resolved"
    assert source_semantics_blocker["blocking"] is False
    promotion_blocker = next(
        blocker
        for blocker in generated["blockers"]
        if blocker["blocker_id"] == "stable_release_promotion_decision_not_yet_taken"
    )
    assert promotion_blocker["status"] == "resolved"
    assert promotion_blocker["blocking"] is False


def test_simplified_likelihood_exporter_stable_candidate_blocker_matrix_marks_missing_review_prerequisite(
    tmp_path: Path,
) -> None:
    repo = _repo_root()
    bundle_dir = tmp_path / "bundle"
    shutil.copytree(_accepted_bundle_dir(), bundle_dir)

    assessment_path = bundle_dir / "stable_review_assessment.json"
    assessment = _load_json(assessment_path)
    assessment["stable_review"]["ready"] = False
    assessment["stable_review"]["status"] = "not_ready"
    assessment["summary"]["status"] = "not_ready"
    assessment_path.write_text(
        json.dumps(assessment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    matrix_path = tmp_path / "stable_candidate_blocker_matrix_invalid.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/assess_simplified_likelihood_exporter_stable_candidate_blockers.py",
            "--bundle-dir",
            str(bundle_dir),
            "--out",
            str(matrix_path),
            "--deterministic",
        ],
        cwd=repo,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    generated = _load_json(matrix_path)

    assert generated["foundation"]["passed"] is False
    assert generated["stable_candidate"]["ready"] is False
    assert generated["stable_candidate"]["status"] == "blocked"
    assert any(
        check.get("check_id") == "stable_review_ready"
        and check.get("satisfied") is False
        for check in generated["foundation"]["checks"]
    )


def test_simplified_likelihood_exporter_stable_candidate_blocker_matrix_resolves_public_exporter_blocker_from_case_kind(
    tmp_path: Path,
) -> None:
    repo = _repo_root()
    bundle_dir = tmp_path / "bundle"
    shutil.copytree(_accepted_bundle_dir(), bundle_dir)

    benchmark_artifact_path = tmp_path / "apex2_simplified_likelihood_report.json"
    benchmark_artifact = _load_json(
        repo
        / "benchmarks"
        / "artifacts"
        / "simplified_likelihood_export_benchmarks"
        / "nextstat-bench"
        / "current"
        / "apex2_simplified_likelihood_report.json"
    )
    export_matrix = benchmark_artifact["export_matrix"]
    export_matrix["cases"][0]["case_kind"] = "public_reinterpretation_style"
    export_matrix["cases"][1]["case_kind"] = "public_reinterpretation_style"
    export_matrix["cases"][2]["case_kind"] = "public_reinterpretation_style"
    export_matrix["summary"]["case_kinds"] = ["public_reinterpretation_style", "synthetic"]
    export_matrix["summary"]["synthetic_case_count"] = max(
        0, len(export_matrix["cases"]) - 3
    )
    export_matrix["summary"]["public_reinterpretation_style_case_count"] = 3
    benchmark_artifact["summary"]["export_matrix_case_kinds"] = [
        "public_reinterpretation_style",
        "synthetic",
    ]
    benchmark_artifact["summary"]["export_matrix_public_reinterpretation_style_case_count"] = 3
    benchmark_artifact_path.write_text(
        json.dumps(benchmark_artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    promotion_evidence_path = bundle_dir / "promotion_evidence.json"
    promotion_evidence = _load_json(promotion_evidence_path)
    promotion_evidence["source_snapshot"]["benchmark_artifact"]["source_path"] = str(
        benchmark_artifact_path
    )
    promotion_evidence_path.write_text(
        json.dumps(promotion_evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    matrix_path = tmp_path / "stable_candidate_blocker_matrix_public.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/assess_simplified_likelihood_exporter_stable_candidate_blockers.py",
            "--bundle-dir",
            str(bundle_dir),
            "--out",
            str(matrix_path),
            "--deterministic",
        ],
        cwd=repo,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    generated = _load_json(matrix_path)

    public_blocker = next(
        blocker
        for blocker in generated["blockers"]
        if blocker["blocker_id"] == "public_exporter_matrix_not_yet_part_of_stable_candidate_evidence"
    )
    assert public_blocker["status"] == "resolved"
    assert public_blocker["blocking"] is False
    assert generated["stable_candidate"]["open_blocker_count"] == 0


def test_simplified_likelihood_exporter_stable_candidate_blocker_matrix_requires_three_public_cases(
    tmp_path: Path,
) -> None:
    repo = _repo_root()
    bundle_dir = tmp_path / "bundle"
    shutil.copytree(_accepted_bundle_dir(), bundle_dir)

    benchmark_artifact_path = tmp_path / "apex2_simplified_likelihood_report_insufficient_public.json"
    benchmark_artifact = _load_json(
        repo
        / "benchmarks"
        / "artifacts"
        / "simplified_likelihood_export_benchmarks"
        / "nextstat-bench"
        / "current"
        / "apex2_simplified_likelihood_report.json"
    )
    export_matrix = benchmark_artifact["export_matrix"]
    export_matrix["cases"][2]["case_kind"] = "synthetic"
    export_matrix["summary"]["case_kinds"] = ["public_reinterpretation_style", "synthetic"]
    export_matrix["summary"]["synthetic_case_count"] = 3
    export_matrix["summary"]["public_reinterpretation_style_case_count"] = 2
    benchmark_artifact["summary"]["export_matrix_case_kinds"] = [
        "public_reinterpretation_style",
        "synthetic",
    ]
    benchmark_artifact["summary"]["export_matrix_public_reinterpretation_style_case_count"] = 2
    benchmark_artifact_path.write_text(
        json.dumps(benchmark_artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    promotion_evidence_path = bundle_dir / "promotion_evidence.json"
    promotion_evidence = _load_json(promotion_evidence_path)
    promotion_evidence["source_snapshot"]["benchmark_artifact"]["source_path"] = str(
        benchmark_artifact_path
    )
    promotion_evidence_path.write_text(
        json.dumps(promotion_evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    matrix_path = tmp_path / "stable_candidate_blocker_matrix_insufficient_public.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/assess_simplified_likelihood_exporter_stable_candidate_blockers.py",
            "--bundle-dir",
            str(bundle_dir),
            "--out",
            str(matrix_path),
            "--deterministic",
        ],
        cwd=repo,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    generated = _load_json(matrix_path)

    public_blocker = next(
        blocker
        for blocker in generated["blockers"]
        if blocker["blocker_id"] == "public_exporter_matrix_not_yet_part_of_stable_candidate_evidence"
    )
    assert public_blocker["status"] == "open"
    assert public_blocker["blocking"] is True


def test_simplified_likelihood_exporter_stable_candidate_docs_gate_and_workflow_are_published() -> None:
    repo = _repo_root()

    assert_doc_contains_strings(
        repo / "docs" / "README.md",
        [
            "simplified-likelihood-exporter-stable-candidate-blocker-matrix-2026-03-09.md",
            "simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0.example.json",
            "simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks.md",
        [
            "simplified-likelihood-exporter-stable-candidate-blocker-matrix-2026-03-09",
            "simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0.example.json",
            "simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09",
        ],
    )
    assert_doc_contains_strings(
        repo
        / "docs"
        / "benchmarks"
        / "simplified-likelihood-exporter-stable-candidate-blocker-matrix-2026-03-09.md",
        [
            "`stable`",
            "stable candidate",
            "stable_candidate_blocker_matrix.json",
            "nextstat-bench",
            "stable_promotion_decision.json",
            "public reinterpretation-style exporter matrix",
            "stable-candidate review packet",
            "stable_source_semantics_boundary.json",
            "open_blocker_count = 0",
            "stable source semantics boundary",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "references" / "simplified-likelihood-artifacts.md",
        [
            "simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0",
            "assess_simplified_likelihood_exporter_stable_candidate_blockers.py",
            "stable_candidate_blocker_matrix.json",
            "stable_source_semantics_boundary.json",
            "simplified-likelihood-exporter-stable-candidate-blocker-matrix-2026-03-09.md",
            "stable_promotion_decision.json",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-exporter-acceptance-2026-03-09.md",
        [
            "stable-candidate blocker matrix",
            "simplified-likelihood-exporter-stable-candidate-blocker-matrix-2026-03-09.md",
        ],
    )

    workflow = (
        repo / ".github" / "workflows" / "simplified-likelihood-exporter-surface.yml"
    ).read_text(encoding="utf-8")
    assert (
        "tests/python/test_simplified_likelihood_exporter_stable_candidate_blocker_matrix_smoke.py"
        in workflow
    )
    assert (
        "docs/schemas/benchmarks/simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0.schema.json"
        in workflow
    )
    assert (
        "docs/specs/benchmarks/simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0.example.json"
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
        "scripts/benchmarks/assess_simplified_likelihood_exporter_stable_candidate_blockers.py"
        in workflow
    )
    assert (
        "benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_blocker_matrix.json"
        in workflow
    )

    gate = (repo / "scripts" / "benchmarks" / "simplified_likelihood_exporter_surface_gate.sh").read_text(
        encoding="utf-8"
    )
    assert "test_simplified_likelihood_exporter_stable_candidate_blocker_matrix_smoke.py" in gate
    assert "assess_simplified_likelihood_exporter_stable_candidate_blockers.py" in gate
    assert "stable_candidate_blocker_matrix.json" in gate
    assert "stable_source_semantics_boundary.json" in gate
    assert "stable_promotion_decision.json" in gate
    assert "simplified-likelihood-exporter-stable-candidate-blocker-matrix-2026-03-09.md" in gate


def test_simplified_likelihood_exporter_committed_stable_candidate_blocker_matrix_is_published() -> None:
    repo = _repo_root()
    matrix_path = _accepted_bundle_dir() / "stable_candidate_blocker_matrix.json"
    schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_candidate_blocker_matrix_v0.schema.json"
    )

    matrix = _load_json(matrix_path)
    _validate_jsonschema(matrix, schema_path)
    assert matrix["summary"]["status"] == "ready"
    assert matrix["summary"]["benchmark_host"] == "nextstat-bench"
    assert matrix["support_class"] == "stable"
    assert matrix["automatic_stable_promotion"] is False
    assert matrix["stable_candidate"]["open_blocker_count"] == 0
    review_packet_blocker = next(
        blocker
        for blocker in matrix["blockers"]
        if blocker["blocker_id"] == "stable_candidate_review_packet_not_yet_published"
    )
    assert review_packet_blocker["status"] == "resolved"
    assert review_packet_blocker["blocking"] is False
    source_semantics_blocker = next(
        blocker
        for blocker in matrix["blockers"]
        if blocker["blocker_id"] == "stable_source_semantics_boundary_not_yet_promoted"
    )
    assert source_semantics_blocker["status"] == "resolved"
    assert source_semantics_blocker["blocking"] is False
    promotion_blocker = next(
        blocker
        for blocker in matrix["blockers"]
        if blocker["blocker_id"] == "stable_release_promotion_decision_not_yet_taken"
    )
    assert promotion_blocker["status"] == "resolved"
    assert promotion_blocker["blocking"] is False
