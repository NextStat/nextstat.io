from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path
import re
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


def _snapshot_date_from_id(snapshot_id: str) -> date:
    match = re.search(r"(\d{8})T\d{6}Z", snapshot_id)
    assert match is not None, f"snapshot_id missing UTC stamp: {snapshot_id}"
    stamp = match.group(1)
    return date.fromisoformat(f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}")


def test_simplified_likelihood_exporter_stable_evidence_freshness_schema_example_and_builder_smoke(
    tmp_path: Path,
) -> None:
    repo = _repo_root()
    schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_evidence_freshness_report_v0.schema.json"
    )
    example_path = (
        repo
        / "docs"
        / "specs"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_evidence_freshness_report_v0.example.json"
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
    snapshot_report = _load_json(current_dir / "export_benchmark_snapshot_report.json")
    snapshot_date = _snapshot_date_from_id(snapshot_report["snapshot_id"])
    breached_reference_date = snapshot_date + timedelta(days=60)

    example = _load_json(example_path)
    _validate_jsonschema(example, schema_path)
    assert (
        example["schema_version"]
        == "nextstat_simplified_likelihood_exporter_stable_evidence_freshness_report_v0"
    )

    out_path = tmp_path / "stable_evidence_freshness_report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/build_simplified_likelihood_exporter_stable_evidence_freshness_report.py",
            "--snapshot-report",
            str(current_dir / "export_benchmark_snapshot_report.json"),
            "--public-validation-report",
            str(current_dir / "export_public_validation_report.json"),
            "--stable-evidence-policy",
            str(accepted_dir / "stable_evidence_policy.json"),
            "--stable-promotion-decision",
            str(accepted_dir / "stable_promotion_decision.json"),
            "--reference-date",
            snapshot_date.isoformat(),
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
    assert generated["status"] == "fresh"
    assert generated["support_class"] == "stable"
    assert generated["benchmark_host"] == "nextstat-bench"
    assert generated["freshness_policy"]["max_snapshot_age_days"] == 45
    assert generated["freshness_observation"]["reference_date"] == snapshot_date.isoformat()
    assert generated["freshness_observation"]["snapshot_age_days"] == 0
    assert generated["validity"]["passed"] is True

    breached_path = tmp_path / "stable_evidence_freshness_report_breached.json"
    breached_proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/build_simplified_likelihood_exporter_stable_evidence_freshness_report.py",
            "--snapshot-report",
            str(current_dir / "export_benchmark_snapshot_report.json"),
            "--public-validation-report",
            str(current_dir / "export_public_validation_report.json"),
            "--stable-evidence-policy",
            str(accepted_dir / "stable_evidence_policy.json"),
            "--stable-promotion-decision",
            str(accepted_dir / "stable_promotion_decision.json"),
            "--reference-date",
            breached_reference_date.isoformat(),
            "--out",
            str(breached_path),
            "--deterministic",
        ],
        cwd=repo,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert breached_proc.returncode == 0, breached_proc.stdout
    breached = _load_json(breached_path)
    _validate_jsonschema(breached, schema_path)
    assert breached["status"] == "breached"
    assert breached["validity"]["passed"] is False
    assert breached["freshness_observation"]["snapshot_age_days"] > 45


def test_simplified_likelihood_exporter_stable_evidence_freshness_docs_and_committed_artifact_are_published() -> None:
    repo = _repo_root()

    assert_doc_contains_strings(
        repo / "docs" / "README.md",
        [
            "simplified-likelihood-exporter-stable-evidence-freshness-2026-03-09.md",
            "simplified_likelihood_exporter_stable_evidence_freshness_report_v0.example.json",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks.md",
        [
            "simplified-likelihood-exporter-stable-evidence-freshness-2026-03-09",
            "simplified_likelihood_exporter_stable_evidence_freshness_report_v0.example.json",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "references" / "simplified-likelihood-artifacts.md",
        [
            "simplified-likelihood-exporter-stable-evidence-freshness-2026-03-09.md",
            "simplified_likelihood_exporter_stable_evidence_freshness_report_v0",
            "stable_evidence_freshness_report.json",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-exporter-runtime-gate.md",
        [
            "stable_evidence_freshness_report.json",
            "max_snapshot_age_days = 45",
            "freshness breach",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-exporter-release-pr-checklist-2026-03-09.md",
        [
            "stable_evidence_freshness_report.json",
            "freshness breach",
            "45-day",
        ],
    )

    accepted_report_path = (
        repo
        / "benchmarks"
        / "artifacts"
        / "simplified_likelihood_exporter_promotion_bundles"
        / "nextstat-bench"
        / "accepted"
        / "stable_evidence_freshness_report.json"
    )
    report = _load_json(accepted_report_path)
    _validate_jsonschema(
        report,
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_exporter_stable_evidence_freshness_report_v0.schema.json",
    )
    assert report["status"] == "fresh"
    assert report["support_class"] == "stable"
