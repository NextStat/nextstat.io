from __future__ import annotations

import json
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


def _build_export_benchmark_artifact(tmp_path: Path) -> Path:
    repo = _repo_root()
    benchmark_artifact = tmp_path / "apex2_simplified_likelihood_export_benchmark_smoke.json"
    committed_artifact = (
        repo
        / "benchmarks"
        / "artifacts"
        / "simplified_likelihood_export_benchmarks"
        / "nextstat-bench"
        / "current"
        / "apex2_simplified_likelihood_report.json"
    )
    shutil.copy2(committed_artifact, benchmark_artifact)
    return benchmark_artifact


def test_simplified_likelihood_export_benchmark_snapshot_schema_example_and_runner_smoke(
    tmp_path: Path,
) -> None:
    repo = _repo_root()
    schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_export_benchmark_snapshot_report_v0.schema.json"
    )
    public_validation_schema_path = (
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
        / "simplified_likelihood_export_benchmark_snapshot_report_v0.example.json"
    )
    snapshot_schema_path = repo / "docs" / "schemas" / "benchmarks" / "snapshot_index_v1.schema.json"
    apex2_schema_path = (
        repo / "docs" / "schemas" / "apex2" / "simplified_likelihood_report_v0.schema.json"
    )

    example = _load_json(example_path)
    _validate_jsonschema(example, schema_path)
    assert (
        example["schema_version"]
        == "nextstat_simplified_likelihood_export_benchmark_snapshot_report_v0"
    )

    benchmark_artifact = _build_export_benchmark_artifact(tmp_path)
    current_dir = tmp_path / "current"
    history_dir = tmp_path / "history"
    dry_run_report = tmp_path / "export_benchmark_snapshot_report_dry_run.json"

    dry_run = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/persist_simplified_likelihood_export_benchmark.py",
            "--benchmark-artifact",
            str(benchmark_artifact),
            "--current-dir",
            str(current_dir),
            "--history-dir",
            str(history_dir),
            "--report",
            str(dry_run_report),
            "--snapshot-id",
            "smoke-export-snapshot",
            "--dry-run",
            "--deterministic",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "status=dry_run" in dry_run.stdout, dry_run.stdout

    dry_run_doc = _load_json(dry_run_report)
    _validate_jsonschema(dry_run_doc, schema_path)
    assert dry_run_doc["status"] == "dry_run"
    assert dry_run_doc["persisted"] is False
    assert not current_dir.exists()

    apply_report = current_dir / "export_benchmark_snapshot_report.json"
    apply = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/persist_simplified_likelihood_export_benchmark.py",
            "--benchmark-artifact",
            str(benchmark_artifact),
            "--current-dir",
            str(current_dir),
            "--history-dir",
            str(history_dir),
            "--report",
            str(apply_report),
            "--snapshot-id",
            "smoke-export-snapshot",
            "--deterministic",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "status=persisted" in apply.stdout, apply.stdout

    apply_doc = _load_json(apply_report)
    persisted_report = _load_json(current_dir / "apex2_simplified_likelihood_report.json")
    public_validation_report = _load_json(current_dir / "export_public_validation_report.json")
    snapshot_index = _load_json(current_dir / "snapshot_index.json")
    _validate_jsonschema(apply_doc, schema_path)
    _validate_jsonschema(persisted_report, apex2_schema_path)
    _validate_jsonschema(public_validation_report, public_validation_schema_path)
    _validate_jsonschema(snapshot_index, snapshot_schema_path)

    assert apply_doc["status"] == "persisted"
    assert apply_doc["persisted"] is True
    assert apply_doc["actions"]["current_updated"] is True
    assert apply_doc["actions"]["current_snapshot_index_written"] is True
    assert apply_doc["actions"]["archived_persisted_snapshot"] is True
    assert persisted_report["environment"]["hostname"] == "nextstat-bench"
    assert persisted_report["summary"]["export_matrix_included"] is True
    assert persisted_report["summary"]["export_matrix_status"] == "ok"
    assert persisted_report["summary"]["export_matrix_public_reinterpretation_style_case_count"] == 6
    assert (
        public_validation_report["schema_version"]
        == "nextstat_simplified_likelihood_export_public_validation_report_v0"
    )
    assert public_validation_report["status"] == "ok"
    assert public_validation_report["summary"]["benchmark_host"] == "nextstat-bench"
    assert public_validation_report["summary"]["public_case_count"] == 6
    assert public_validation_report["summary"]["public_case_names"] == [
        "atlas_public_dual_sr_dual_cr_gaussian_export_stable_example",
        "atlas_public_dual_sr_vr_dual_cr_gaussian_export_stable_example",
        "atlas_public_sr_cr_gaussian_export_stable_example",
        "cms_public_sr_cr_export_stable_example",
        "cms_public_sr_cr_asymmetric_gaussian_export_stable_example",
        "cms_public_dual_sr_cr_gaussian_export_stable_example",
    ]
    assert public_validation_report["summary"]["all_schema_valid"] is True
    assert public_validation_report["summary"]["all_fidelity_gates_pass"] is True
    assert public_validation_report["summary"]["all_performance_gates_pass"] is True
    assert public_validation_report["summary"]["all_cases_within_promoted_stable_runtime_boundary"] is True
    assert public_validation_report["summary"]["all_cases_gaussian_constrained_source_workspaces"] is True
    assert public_validation_report["summary"]["cases_outside_promoted_stable_runtime_boundary"] == 0
    assert public_validation_report["summary"]["observed_constraint_covariance_sources"] == [
        "source_model_constraints"
    ]
    assert set(persisted_report["summary"]["export_matrix_case_kinds"]) == {
        "public_reinterpretation_style",
        "synthetic",
    }
    assert snapshot_index["suite"] == "simplified_likelihood_export_benchmark_snapshot"
    assert snapshot_index["snapshot_id"] == "smoke-export-snapshot"
    artifact_paths = {artifact["path"] for artifact in snapshot_index["artifacts"]}
    assert "apex2_simplified_likelihood_report.json" in artifact_paths
    assert "export_benchmark_snapshot_report.json" in artifact_paths
    assert "export_public_validation_report.json" in artifact_paths

    persisted_archives = list(history_dir.glob("snapshot_*_persisted"))
    assert len(persisted_archives) == 1
    assert persisted_archives[0].joinpath("apex2_simplified_likelihood_report.json").exists()
    assert persisted_archives[0].joinpath("export_public_validation_report.json").exists()


def test_simplified_likelihood_export_benchmark_current_path_is_published() -> None:
    repo = _repo_root()
    current_dir = (
        repo
        / "benchmarks"
        / "artifacts"
        / "simplified_likelihood_export_benchmarks"
        / "nextstat-bench"
        / "current"
    )
    history_dir = (
        repo
        / "benchmarks"
        / "artifacts"
        / "simplified_likelihood_export_benchmarks"
        / "nextstat-bench"
        / "history"
    )
    report_schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_export_benchmark_snapshot_report_v0.schema.json"
    )
    public_validation_schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_export_public_validation_report_v0.schema.json"
    )
    snapshot_schema_path = repo / "docs" / "schemas" / "benchmarks" / "snapshot_index_v1.schema.json"
    apex2_schema_path = (
        repo / "docs" / "schemas" / "apex2" / "simplified_likelihood_report_v0.schema.json"
    )

    assert current_dir.joinpath("apex2_simplified_likelihood_report.json").exists()
    assert current_dir.joinpath("export_benchmark_snapshot_report.json").exists()
    assert current_dir.joinpath("export_public_validation_report.json").exists()
    assert current_dir.joinpath("snapshot_index.json").exists()
    persisted_archives = sorted(history_dir.glob("snapshot_*_persisted"))
    assert persisted_archives, "expected at least one persisted exporter snapshot archive"
    assert any(
        archive.joinpath("apex2_simplified_likelihood_report.json").exists()
        and archive.joinpath("export_public_validation_report.json").exists()
        for archive in persisted_archives
    )

    persisted_report = _load_json(current_dir / "apex2_simplified_likelihood_report.json")
    snapshot_report = _load_json(current_dir / "export_benchmark_snapshot_report.json")
    public_validation_report = _load_json(current_dir / "export_public_validation_report.json")
    snapshot_index = _load_json(current_dir / "snapshot_index.json")
    _validate_jsonschema(persisted_report, apex2_schema_path)
    _validate_jsonschema(snapshot_report, report_schema_path)
    _validate_jsonschema(public_validation_report, public_validation_schema_path)
    _validate_jsonschema(snapshot_index, snapshot_schema_path)

    assert persisted_report["environment"]["hostname"] == "nextstat-bench"
    assert persisted_report["summary"]["status"] == "ok"
    assert persisted_report["summary"]["export_matrix_included"] is True
    assert persisted_report["summary"]["export_matrix_status"] == "ok"
    assert (
        snapshot_report["schema_version"]
        == "nextstat_simplified_likelihood_export_benchmark_snapshot_report_v0"
    )
    assert snapshot_report["status"] == "persisted"
    assert snapshot_report["persisted"] is True
    assert snapshot_report["source_summary"]["benchmark_host"] == "nextstat-bench"
    assert snapshot_report["source_summary"]["export_matrix_status"] == "ok"
    assert snapshot_report["source_summary"]["export_matrix_case_count"] == 8
    assert (
        snapshot_report["source_summary"]["export_matrix_public_reinterpretation_style_case_count"]
        == 6
    )
    assert (
        public_validation_report["schema_version"]
        == "nextstat_simplified_likelihood_export_public_validation_report_v0"
    )
    assert public_validation_report["status"] == "ok"
    assert public_validation_report["summary"]["benchmark_host"] == "nextstat-bench"
    assert public_validation_report["summary"]["public_case_count"] == 6
    assert public_validation_report["summary"]["public_case_names"] == [
        "atlas_public_dual_sr_dual_cr_gaussian_export_stable_example",
        "atlas_public_dual_sr_vr_dual_cr_gaussian_export_stable_example",
        "atlas_public_sr_cr_gaussian_export_stable_example",
        "cms_public_sr_cr_export_stable_example",
        "cms_public_sr_cr_asymmetric_gaussian_export_stable_example",
        "cms_public_dual_sr_cr_gaussian_export_stable_example",
    ]
    assert public_validation_report["summary"]["all_schema_valid"] is True
    assert public_validation_report["summary"]["all_fidelity_gates_pass"] is True
    assert public_validation_report["summary"]["all_performance_gates_pass"] is True
    assert public_validation_report["summary"]["all_cases_within_promoted_stable_runtime_boundary"] is True
    assert public_validation_report["summary"]["all_cases_gaussian_constrained_source_workspaces"] is True
    assert public_validation_report["summary"]["cases_outside_promoted_stable_runtime_boundary"] == 0
    assert public_validation_report["summary"]["observed_constraint_covariance_sources"] == [
        "source_model_constraints"
    ]
    assert set(snapshot_report["source_summary"]["export_matrix_case_kinds"]) == {
        "public_reinterpretation_style",
        "synthetic",
    }
    assert snapshot_index["suite"] == "simplified_likelihood_export_benchmark_snapshot"
    artifact_paths = {artifact["path"] for artifact in snapshot_index["artifacts"]}
    assert "export_public_validation_report.json" in artifact_paths

    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-benchmark-snapshot-2026-03-08.md",
        [
            "benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/apex2_simplified_likelihood_report.json",
            "benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_benchmark_snapshot_report.json",
            "benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_public_validation_report.json",
            "benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/snapshot_index.json",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "references" / "simplified-likelihood-artifacts.md",
        [
            "simplified_likelihood_export_benchmark_snapshot_report_v0",
            "simplified_likelihood_export_public_validation_report_v0",
            "benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/",
        ],
    )
