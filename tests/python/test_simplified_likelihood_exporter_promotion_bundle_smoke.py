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


def _current_exporter_paths(repo: Path) -> tuple[Path, Path, Path]:
    current_dir = (
        repo
        / "benchmarks"
        / "artifacts"
        / "simplified_likelihood_export_benchmarks"
        / "nextstat-bench"
        / "current"
    )
    return (
        current_dir / "apex2_simplified_likelihood_report.json",
        current_dir / "export_benchmark_snapshot_report.json",
        current_dir / "snapshot_index.json",
    )


def _build_bundle(tmp_path: Path) -> Path:
    repo = _repo_root()
    benchmark_artifact, snapshot_report, snapshot_index = _current_exporter_paths(repo)
    bundle_dir = tmp_path / "bundle"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/build_simplified_likelihood_exporter_promotion_evidence_bundle.py",
            "--benchmark-artifact",
            str(benchmark_artifact),
            "--snapshot-report",
            str(snapshot_report),
            "--snapshot-index",
            str(snapshot_index),
            "--bundle-dir",
            str(bundle_dir),
            "--deterministic",
        ],
        cwd=repo,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    bundle_json = bundle_dir / "promotion_evidence.json"
    assert bundle_json.exists(), proc.stdout
    assert proc.returncode == 0, proc.stdout
    return bundle_dir


def test_simplified_likelihood_exporter_promotion_bundle_schema_example_and_generator_smoke(
    tmp_path: Path,
) -> None:
    repo = _repo_root()
    schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_promotion_evidence_bundle_v0.schema.json"
    )
    example_path = (
        repo
        / "docs"
        / "specs"
        / "benchmarks"
        / "simplified_likelihood_exporter_promotion_evidence_bundle_v0.example.json"
    )

    example = _load_json(example_path)
    _validate_jsonschema(example, schema_path)
    assert (
        example["schema_version"]
        == "nextstat_simplified_likelihood_exporter_promotion_evidence_bundle_v0"
    )

    bundle_dir = _build_bundle(tmp_path)
    generated = _load_json(bundle_dir / "promotion_evidence.json")
    _validate_jsonschema(generated, schema_path)

    assert generated["surface"] == "simplified_likelihood_exporter"
    assert generated["bundle_kind"] == "promotion_readiness_evidence"
    assert generated["support_class"] == "research-grade"
    assert generated["summary"]["status"] == "ok"
    assert generated["summary"]["benchmark_host"] == "nextstat-bench"
    assert generated["summary"]["supports_future_stable_review"] is True
    assert generated["summary"]["supports_committed_snapshot"] is True
    assert generated["promotion_claims"]["future_stable_review_ready"] is True
    assert generated["promotion_claims"]["research_grade_acceptance_supported"] is True

    roles = {artifact["role"] for artifact in generated["artifacts"]}
    assert {
        "benchmark_artifact",
        "current_snapshot_report",
        "current_snapshot_index",
        "exporter_acceptance_doc",
        "exporter_runtime_gate_doc",
        "exporter_promotion_runbook_doc",
        "exporter_stable_evidence_policy_doc",
        "exporter_stable_evidence_freshness_doc",
        "exporter_stable_source_semantics_doc",
        "derive_schema",
        "export_report_schema",
        "export_snapshot_report_schema",
        "export_public_case_catalog_schema",
        "exporter_stable_evidence_policy_schema",
        "exporter_stable_evidence_freshness_schema",
        "exporter_stable_source_semantics_schema",
        "derive_example",
        "derived_example",
        "export_public_case_catalog_example",
        "exporter_stable_evidence_policy_example",
        "exporter_stable_evidence_freshness_example",
        "exporter_stable_source_semantics_example",
    }.issubset(roles)


def test_simplified_likelihood_exporter_promotion_check_schema_example_and_generator_smoke(
    tmp_path: Path,
) -> None:
    repo = _repo_root()
    schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_promotion_evidence_check_v0.schema.json"
    )
    example_path = (
        repo
        / "docs"
        / "specs"
        / "benchmarks"
        / "simplified_likelihood_exporter_promotion_evidence_check_v0.example.json"
    )

    example = _load_json(example_path)
    _validate_jsonschema(example, schema_path)
    assert (
        example["schema_version"]
        == "nextstat_simplified_likelihood_exporter_promotion_evidence_check_v0"
    )

    bundle_dir = _build_bundle(tmp_path)
    report_path = bundle_dir / "promotion_evidence_check.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/verify_simplified_likelihood_exporter_promotion_evidence_bundle.py",
            "--bundle-dir",
            str(bundle_dir),
            "--out",
            str(report_path),
            "--require-promotion-ready",
            "--deterministic",
        ],
        cwd=repo,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout
    generated = _load_json(report_path)
    _validate_jsonschema(generated, schema_path)
    assert generated["status"] == "passed"
    assert generated["require_promotion_ready"] is True
    assert generated["checks"]["promotion_readiness"]["status"] == "passed"
    assert generated["checks"]["promotion_readiness"]["actual_benchmark_host"] == "nextstat-bench"
    assert (
        generated["checks"]["promotion_readiness"][
            "actual_min_net_end_to_end_upper_limit_speedup"
        ]
        >= 1.25
    )

    tampered = bundle_dir / "files" / "docs" / "benchmarks" / "simplified-likelihood-support-matrix-2026-03-08.md"
    tampered.write_text(tampered.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")

    tampered_report = bundle_dir / "promotion_evidence_check_tampered.json"
    tampered_proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/verify_simplified_likelihood_exporter_promotion_evidence_bundle.py",
            "--bundle-dir",
            str(bundle_dir),
            "--out",
            str(tampered_report),
            "--deterministic",
        ],
        cwd=repo,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert tampered_proc.returncode == 1, tampered_proc.stdout
    tampered_doc = _load_json(tampered_report)
    _validate_jsonschema(tampered_doc, schema_path)
    assert "support_matrix_doc" in tampered_doc["checks"]["inventory"]["sha256_mismatches"]


def test_simplified_likelihood_exporter_promotion_report_schema_and_promoter_smoke(
    tmp_path: Path,
) -> None:
    repo = _repo_root()
    schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_promotion_bundle_promotion_report_v0.schema.json"
    )
    example_path = (
        repo
        / "docs"
        / "specs"
        / "benchmarks"
        / "simplified_likelihood_exporter_promotion_bundle_promotion_report_v0.example.json"
    )
    bundle_schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_promotion_evidence_bundle_v0.schema.json"
    )
    check_schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_promotion_evidence_check_v0.schema.json"
    )
    snapshot_schema_path = repo / "docs" / "schemas" / "benchmarks" / "snapshot_index_v1.schema.json"

    example = _load_json(example_path)
    _validate_jsonschema(example, schema_path)
    assert (
        example["schema_version"]
        == "nextstat_simplified_likelihood_exporter_promotion_bundle_promotion_report_v0"
    )

    bundle_dir = _build_bundle(tmp_path)
    subprocess.check_call(
        [
            sys.executable,
            "scripts/benchmarks/verify_simplified_likelihood_exporter_promotion_evidence_bundle.py",
            "--bundle-dir",
            str(bundle_dir),
            "--out",
            str(bundle_dir / "promotion_evidence_check.json"),
            "--require-promotion-ready",
            "--deterministic",
        ],
        cwd=repo,
    )

    accepted_dir = tmp_path / "accepted"
    history_dir = tmp_path / "history"
    report_path = tmp_path / "promotion_report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/promote_simplified_likelihood_exporter_promotion_bundle.py",
            "--source-bundle-dir",
            str(bundle_dir),
            "--accepted-dir",
            str(accepted_dir),
            "--history-dir",
            str(history_dir),
            "--report",
            str(report_path),
            "--snapshot-id",
            "exporter-smoke",
            "--deterministic",
        ],
        cwd=repo,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout

    report = _load_json(report_path)
    _validate_jsonschema(report, schema_path)
    assert report["status"] == "promoted"
    assert report["promoted"] is True
    assert report["actions"]["accepted_updated"] is True
    assert report["actions"]["accepted_snapshot_index_written"] is True

    accepted_bundle = _load_json(accepted_dir / "promotion_evidence.json")
    accepted_check = _load_json(accepted_dir / "promotion_evidence_check.json")
    accepted_review = _load_json(accepted_dir / "stable_review_assessment.json")
    accepted_blockers = _load_json(accepted_dir / "stable_candidate_blocker_matrix.json")
    accepted_review_packet = _load_json(accepted_dir / "stable_candidate_review_packet.json")
    accepted_policy = _load_json(accepted_dir / "stable_evidence_policy.json")
    accepted_freshness = _load_json(accepted_dir / "stable_evidence_freshness_report.json")
    accepted_decision = _load_json(accepted_dir / "stable_promotion_decision.json")
    accepted_snapshot = _load_json(accepted_dir / "snapshot_index.json")
    _validate_jsonschema(accepted_bundle, bundle_schema_path)
    _validate_jsonschema(accepted_check, check_schema_path)
    _validate_jsonschema(accepted_snapshot, snapshot_schema_path)
    assert accepted_check["status"] == "passed"
    assert accepted_snapshot["suite"] == "simplified_likelihood_exporter_promotion_bundle"
    assert accepted_review["bundle_dir"] == str(accepted_dir)
    assert accepted_review["source_artifacts"]["promotion_evidence"]["path"] == str(
        accepted_dir / "promotion_evidence.json"
    )
    assert accepted_blockers["bundle_dir"] == str(accepted_dir)
    assert accepted_blockers["source_artifacts"]["promotion_evidence"]["path"] == str(
        accepted_dir / "promotion_evidence.json"
    )
    assert accepted_review_packet["bundle_dir"] == str(accepted_dir)
    assert accepted_review_packet["review_packet"]["open_blocker_count"] == 0
    assert accepted_policy["status"] == "accepted"
    assert accepted_policy["support_class"] == "stable"
    assert accepted_freshness["status"] == "fresh"
    assert accepted_decision["bundle_dir"] == str(accepted_dir)
    accepted_snapshot_paths = {entry["path"] for entry in accepted_snapshot["artifacts"]}
    assert "promotion_bundle_promotion_report.json" in accepted_snapshot_paths
    assert "stable_review_assessment.json" in accepted_snapshot_paths
    assert "stable_candidate_blocker_matrix.json" in accepted_snapshot_paths
    assert "stable_candidate_review_packet.json" in accepted_snapshot_paths
    assert "stable_evidence_policy.json" in accepted_snapshot_paths
    assert "stable_evidence_freshness_report.json" in accepted_snapshot_paths
    assert "stable_promotion_decision.json" in accepted_snapshot_paths
    archived_bundle_dir = Path(report["actions"]["archived_promoted_bundle_path"])
    assert archived_bundle_dir.exists()
    assert (archived_bundle_dir / "promotion_bundle_promotion_report.json").exists()
    assert (archived_bundle_dir / "snapshot_index.json").exists()
    assert (archived_bundle_dir / "stable_review_assessment.json").exists()
    assert (archived_bundle_dir / "stable_candidate_blocker_matrix.json").exists()
    assert (archived_bundle_dir / "stable_candidate_review_packet.json").exists()
    assert (archived_bundle_dir / "stable_evidence_policy.json").exists()
    assert (archived_bundle_dir / "stable_evidence_freshness_report.json").exists()
    assert (archived_bundle_dir / "stable_promotion_decision.json").exists()


def test_simplified_likelihood_exporter_promotion_docs_and_committed_bundle_are_published() -> None:
    repo = _repo_root()

    assert_doc_contains_strings(
        repo / "docs" / "README.md",
        [
            "simplified-likelihood-exporter-promotion-runbook-2026-03-09.md",
            "simplified_likelihood_exporter_promotion_evidence_bundle_v0.example.json",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "references" / "simplified-likelihood-artifacts.md",
        [
            "simplified_likelihood_exporter_promotion_evidence_bundle_v0",
            "simplified_likelihood_exporter_promotion_evidence_check_v0",
            "simplified_likelihood_exporter_promotion_bundle_promotion_report_v0",
            "simplified_likelihood_exporter_stable_evidence_policy_v0",
            "simplified_likelihood_exporter_stable_evidence_freshness_report_v0",
            "scripts/benchmarks/build_simplified_likelihood_exporter_promotion_evidence_bundle.py",
            "scripts/benchmarks/build_simplified_likelihood_exporter_stable_evidence_policy.py",
            "scripts/benchmarks/build_simplified_likelihood_exporter_stable_evidence_freshness_report.py",
            "scripts/benchmarks/verify_simplified_likelihood_exporter_promotion_evidence_bundle.py",
            "scripts/benchmarks/promote_simplified_likelihood_exporter_promotion_bundle.py",
            "simplified-likelihood-exporter-promotion-runbook-2026-03-09.md",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-exporter-acceptance-2026-03-09.md",
        [
            "simplified-likelihood-exporter-promotion-runbook-2026-03-09.md",
            "simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/",
            "stable_evidence_policy.json",
            "stable_evidence_freshness_report.json",
        ],
    )

    accepted_dir = (
        repo
        / "benchmarks"
        / "artifacts"
        / "simplified_likelihood_exporter_promotion_bundles"
        / "nextstat-bench"
        / "accepted"
    )
    bundle_path = accepted_dir / "promotion_evidence.json"
    check_path = accepted_dir / "promotion_evidence_check.json"
    report_path = accepted_dir / "promotion_bundle_promotion_report.json"
    snapshot_path = accepted_dir / "snapshot_index.json"
    review_packet_path = accepted_dir / "stable_candidate_review_packet.json"
    policy_path = accepted_dir / "stable_evidence_policy.json"
    freshness_path = accepted_dir / "stable_evidence_freshness_report.json"
    decision_path = accepted_dir / "stable_promotion_decision.json"

    bundle = _load_json(bundle_path)
    check = _load_json(check_path)
    report = _load_json(report_path)
    snapshot = _load_json(snapshot_path)
    review_packet = _load_json(review_packet_path)
    policy = _load_json(policy_path)
    freshness = _load_json(freshness_path)
    decision = _load_json(decision_path)

    _validate_jsonschema(
        bundle,
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_promotion_evidence_bundle_v0.schema.json",
    )
    _validate_jsonschema(
        check,
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_promotion_evidence_check_v0.schema.json",
    )
    _validate_jsonschema(
        report,
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_promotion_bundle_promotion_report_v0.schema.json",
    )
    _validate_jsonschema(
        snapshot,
        repo / "docs" / "schemas" / "benchmarks" / "snapshot_index_v1.schema.json",
    )
    _validate_jsonschema(
        review_packet,
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_candidate_review_packet_v0.schema.json",
    )
    _validate_jsonschema(
        policy,
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_evidence_policy_v0.schema.json",
    )
    _validate_jsonschema(
        freshness,
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_evidence_freshness_report_v0.schema.json",
    )
    _validate_jsonschema(
        decision,
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_promotion_decision_v0.schema.json",
    )

    assert bundle["summary"]["benchmark_host"] == "nextstat-bench"
    assert bundle["summary"]["supports_future_stable_review"] is True
    assert check["status"] == "passed"
    assert check["require_promotion_ready"] is True
    assert report["status"] == "promoted"
    assert snapshot["suite"] == "simplified_likelihood_exporter_promotion_bundle"
    assert review_packet["review_packet"]["open_blocker_count"] == 0
    assert policy["status"] == "accepted"
    assert freshness["status"] == "fresh"
    assert decision["status"] == "accepted"
