from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from _io_contract_doc_assertions import assert_doc_contains_strings


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _nextstat_subprocess_env(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    source_bindings = repo / "bindings" / "ns-py" / "python"
    source_pkg = source_bindings / "nextstat"
    patterns = ("_core*.so", "_core*.pyd", "_core*.dylib", "_core*.dll")
    local_extension_present = any(any(source_pkg.glob(pattern)) for pattern in patterns)
    if local_extension_present or os.environ.get("NEXTSTAT_FORCE_PYTHONPATH") == "1":
        env["PYTHONPATH"] = str(source_bindings)
    else:
        env.pop("PYTHONPATH", None)
        env["NEXTSTAT_PREFER_INSTALLED"] = "1"
    return env


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_jsonschema(instance: dict, schema_path: Path) -> None:
    import jsonschema  # type: ignore

    schema = _load_json(schema_path)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(instance=instance, schema=schema)


def _build_promotion_ready_bundle(tmp_path: Path) -> Path:
    repo = _repo_root()
    benchmark_artifact = tmp_path / "apex2_simplified_likelihood_report_smoke.json"
    proc = subprocess.run(
        [
            sys.executable,
            "tests/apex2_simplified_likelihood_report.py",
            "--suite",
            "smoke",
            "--fit-repeat",
            "1",
            "--upper-limit-repeat",
            "1",
            "--include-public-fixtures",
            "--out",
            str(benchmark_artifact),
        ],
        cwd=repo,
        env=_nextstat_subprocess_env(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    assert benchmark_artifact.exists(), proc.stdout

    report = _load_json(benchmark_artifact)
    report.setdefault("environment", {})["hostname"] = "nextstat-bench"
    report["summary"]["status"] = "ok"
    report["summary"]["all_schema_valid"] = True
    report["summary"]["all_fidelity_gates_pass"] = True
    report["summary"]["all_performance_gates_pass"] = True
    report["summary"]["public_fixture_matrix_included"] = True
    report["summary"]["public_fixture_matrix_status"] = "ok"
    report["summary"]["public_fixture_matrix_fixture_count"] = 3
    report["summary"]["bench"]["min_speedup_end_to_end_upper_limit"] = 10.5
    report["public_fixture_matrix"]["summary"]["status"] = "ok"
    report["public_fixture_matrix"]["summary"]["all_schema_valid"] = True
    report["public_fixture_matrix"]["summary"]["all_runtime_gates_pass"] = True
    report["public_fixture_matrix"]["summary"]["all_derived_fidelity_gates_pass"] = True
    benchmark_artifact.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    bundle_dir = tmp_path / "bundle"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/benchmarks/build_simplified_likelihood_promotion_evidence_bundle.py",
            "--benchmark-artifact",
            str(benchmark_artifact),
            "--bundle-dir",
            str(bundle_dir),
            "--deterministic",
        ],
        cwd=repo,
    )
    subprocess.check_call(
        [
            sys.executable,
            "scripts/benchmarks/verify_simplified_likelihood_promotion_evidence_bundle.py",
            "--bundle-dir",
            str(bundle_dir),
            "--out",
            str(bundle_dir / "promotion_evidence_check.json"),
            "--require-promotion-ready",
            "--deterministic",
        ],
        cwd=repo,
    )
    return bundle_dir


def test_simplified_likelihood_promotion_bundle_promotion_schema_example_and_runner_smoke(
    tmp_path: Path,
) -> None:
    repo = _repo_root()
    schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_promotion_bundle_promotion_report_v0.schema.json"
    )
    example_path = (
        repo
        / "docs"
        / "specs"
        / "benchmarks"
        / "simplified_likelihood_promotion_bundle_promotion_report_v0.example.json"
    )
    snapshot_schema_path = repo / "docs" / "schemas" / "benchmarks" / "snapshot_index_v1.schema.json"
    bundle_schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_promotion_evidence_bundle_v0.schema.json"
    )
    check_schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_promotion_evidence_check_v0.schema.json"
    )

    example = _load_json(example_path)
    _validate_jsonschema(example, schema_path)
    assert (
        example["schema_version"]
        == "nextstat_simplified_likelihood_promotion_bundle_promotion_report_v0"
    )

    bundle_dir = _build_promotion_ready_bundle(tmp_path)
    accepted_dir = tmp_path / "accepted"
    history_dir = tmp_path / "history"

    dry_run_report = tmp_path / "promotion_report_dry_run.json"
    dry_run = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/promote_simplified_likelihood_promotion_bundle.py",
            "--source-bundle-dir",
            str(bundle_dir),
            "--accepted-dir",
            str(accepted_dir),
            "--history-dir",
            str(history_dir),
            "--report",
            str(dry_run_report),
            "--snapshot-id",
            "smoke-snapshot",
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
    assert dry_run_doc["promoted"] is False
    assert not accepted_dir.exists()

    apply_report = tmp_path / "promotion_report_apply.json"
    apply = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/promote_simplified_likelihood_promotion_bundle.py",
            "--source-bundle-dir",
            str(bundle_dir),
            "--accepted-dir",
            str(accepted_dir),
            "--history-dir",
            str(history_dir),
            "--report",
            str(apply_report),
            "--snapshot-id",
            "smoke-snapshot",
            "--deterministic",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "status=promoted" in apply.stdout, apply.stdout

    apply_doc = _load_json(apply_report)
    _validate_jsonschema(apply_doc, schema_path)
    assert apply_doc["status"] == "promoted"
    assert apply_doc["promoted"] is True
    assert apply_doc["actions"]["accepted_updated"] is True
    assert apply_doc["actions"]["accepted_snapshot_index_written"] is True
    assert apply_doc["actions"]["archived_previous_accepted"] is False
    assert apply_doc["actions"]["archived_promoted_bundle"] is True

    accepted_bundle = _load_json(accepted_dir / "promotion_evidence.json")
    accepted_check = _load_json(accepted_dir / "promotion_evidence_check.json")
    accepted_snapshot = _load_json(accepted_dir / "snapshot_index.json")
    _validate_jsonschema(accepted_bundle, bundle_schema_path)
    _validate_jsonschema(accepted_check, check_schema_path)
    _validate_jsonschema(accepted_snapshot, snapshot_schema_path)

    benchmark_bundle_path = Path(accepted_bundle["benchmark_evidence"]["bundle_path"])
    assert accepted_dir.joinpath(benchmark_bundle_path).exists()
    assert accepted_check["status"] == "passed"
    assert accepted_check["require_promotion_ready"] is True
    assert accepted_check["bundle_dir"] == str(accepted_dir)
    assert accepted_snapshot["suite"] == "simplified_likelihood_promotion_bundle"
    assert accepted_snapshot["snapshot_id"] == "smoke-snapshot"
    artifact_paths = {artifact["path"] for artifact in accepted_snapshot["artifacts"]}
    assert "promotion_evidence.json" in artifact_paths
    assert "promotion_evidence_check.json" in artifact_paths
    assert "snapshot_index.json" in artifact_paths

    promoted_archives = list(history_dir.glob("accepted_*_promoted"))
    assert len(promoted_archives) == 1
    assert promoted_archives[0].joinpath("promotion_evidence.json").exists()


def test_simplified_likelihood_promoted_bundle_path_and_snapshot_are_published() -> None:
    repo = _repo_root()
    accepted_dir = (
        repo
        / "benchmarks"
        / "artifacts"
        / "simplified_likelihood_promotion_bundles"
        / "nextstat-bench"
        / "accepted"
    )
    promoted_archive = (
        repo
        / "benchmarks"
        / "artifacts"
        / "simplified_likelihood_promotion_bundles"
        / "nextstat-bench"
        / "history"
        / "accepted_20260308T173340Z_promoted"
    )
    bundle_schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_promotion_evidence_bundle_v0.schema.json"
    )
    promotion_report_schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_promotion_bundle_promotion_report_v0.schema.json"
    )
    check_schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_promotion_evidence_check_v0.schema.json"
    )
    snapshot_schema_path = repo / "docs" / "schemas" / "benchmarks" / "snapshot_index_v1.schema.json"

    assert accepted_dir.exists()
    assert promoted_archive.exists()
    assert accepted_dir.joinpath("promotion_bundle_promotion_report.json").exists()

    accepted_bundle = _load_json(accepted_dir / "promotion_evidence.json")
    accepted_promotion_report = _load_json(accepted_dir / "promotion_bundle_promotion_report.json")
    accepted_check = _load_json(accepted_dir / "promotion_evidence_check.json")
    accepted_snapshot = _load_json(accepted_dir / "snapshot_index.json")
    _validate_jsonschema(accepted_bundle, bundle_schema_path)
    _validate_jsonschema(accepted_promotion_report, promotion_report_schema_path)
    _validate_jsonschema(accepted_check, check_schema_path)
    _validate_jsonschema(accepted_snapshot, snapshot_schema_path)

    assert accepted_bundle["summary"]["benchmark_host"] == "nextstat-bench"
    assert accepted_promotion_report["status"] == "promoted"
    assert accepted_bundle["summary"]["supports_speedup_claim"] is True
    assert accepted_check["status"] == "passed"
    assert accepted_check["require_promotion_ready"] is True
    assert accepted_check["checks"]["promotion_readiness"]["actual_benchmark_host"] == "nextstat-bench"
    assert (
        accepted_check["checks"]["promotion_readiness"]["actual_min_end_to_end_upper_limit_speedup"]
        >= 10.0
    )
    artifact_paths = {artifact["path"] for artifact in accepted_snapshot["artifacts"]}
    assert "promotion_evidence.json" in artifact_paths
    assert "promotion_evidence_check.json" in artifact_paths
    assert "files/benchmark/apex2_simplified_likelihood_report.json" in artifact_paths


def test_simplified_likelihood_promoted_bundle_paths_are_not_gitignored() -> None:
    repo = _repo_root()
    accepted_bundle = (
        repo
        / "benchmarks"
        / "artifacts"
        / "simplified_likelihood_promotion_bundles"
        / "nextstat-bench"
        / "accepted"
        / "promotion_evidence.json"
    )
    accepted_snapshot = (
        repo
        / "benchmarks"
        / "artifacts"
        / "simplified_likelihood_promotion_bundles"
        / "nextstat-bench"
        / "accepted"
        / "snapshot_index.json"
    )

    for candidate in (accepted_bundle, accepted_snapshot):
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", str(candidate)],
            cwd=repo,
            check=False,
        )
        assert ignored.returncode == 1


def test_simplified_likelihood_promotion_bundle_docs_publish_contract() -> None:
    repo = _repo_root()
    assert_doc_contains_strings(
        repo / "docs" / "references" / "simplified-likelihood-artifacts.md",
        [
            "simplified_likelihood_promotion_bundle_promotion_report_v0",
            "scripts/benchmarks/promote_simplified_likelihood_promotion_bundle.py",
            "benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted",
            "snapshot_index.json",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-promotion-runbook-2026-03-08.md",
        [
            "promote_simplified_likelihood_promotion_bundle.py",
            "benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted",
            "snapshot_index.json",
            "accepted_20260308T173340Z_promoted",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-benchmark-snapshot-2026-03-08.md",
        [
            "benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/promotion_evidence.json",
            "benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/promotion_evidence_check.json",
            "benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/promotion_bundle_promotion_report.json",
            "benchmarks/artifacts/simplified_likelihood_promotion_bundles/nextstat-bench/accepted/snapshot_index.json",
            "accepted_20260308T173340Z_promoted",
        ],
    )
