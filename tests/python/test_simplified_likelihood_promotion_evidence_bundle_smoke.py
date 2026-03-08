from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
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
    try:
        import jsonschema  # type: ignore
    except Exception:
        return

    schema = _load_json(schema_path)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(instance=instance, schema=schema)


def test_simplified_likelihood_promotion_evidence_bundle_schema_example_and_generator_smoke(tmp_path: Path):
    repo = _repo_root()
    schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_promotion_evidence_bundle_v0.schema.json"
    )
    example_path = (
        repo
        / "docs"
        / "specs"
        / "benchmarks"
        / "simplified_likelihood_promotion_evidence_bundle_v0.example.json"
    )
    assert schema_path.exists(), f"missing schema: {schema_path}"
    assert example_path.exists(), f"missing example: {example_path}"

    example = _load_json(example_path)
    assert example["schema_version"] == "nextstat_simplified_likelihood_promotion_evidence_bundle_v0"
    _validate_jsonschema(example, schema_path)

    benchmark_artifact = tmp_path / "apex2_simplified_likelihood_report_smoke.json"
    report_proc = subprocess.run(
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
        check=True,
    )
    assert "status=ok" in report_proc.stdout, report_proc.stdout

    bundle_dir = tmp_path / "bundle"
    proc = subprocess.run(
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
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "status=ok" in proc.stdout, proc.stdout

    generated_path = bundle_dir / "promotion_evidence.json"
    assert generated_path.exists(), f"missing generated bundle summary: {generated_path}"
    generated = _load_json(generated_path)
    _validate_jsonschema(generated, schema_path)

    assert generated["schema_version"] == "nextstat_simplified_likelihood_promotion_evidence_bundle_v0"
    assert generated["surface"] == "simplified_likelihood"
    assert generated["summary"]["status"] == "ok"
    assert generated["summary"]["supports_public_fixture_matrix"] is True
    assert generated["summary"]["supports_speedup_claim"] is False
    assert generated["benchmark_evidence"]["summary"]["status"] == "ok"
    assert generated["benchmark_evidence"]["summary"]["public_fixture_matrix_included"] is True

    roles = {artifact["role"] for artifact in generated["artifacts"]}
    assert {
        "benchmark_artifact",
        "acceptance_doc",
        "support_matrix_doc",
        "benchmark_snapshot_doc",
        "input_schema",
        "audit_schema",
        "apex2_report_schema",
        "public_fixture_catalog_schema",
    }.issubset(roles)

    for artifact in generated["artifacts"]:
        bundle_path = bundle_dir / artifact["bundle_path"]
        assert bundle_path.exists(), f"missing copied artifact for role {artifact['role']}: {bundle_path}"


def test_simplified_likelihood_promotion_evidence_bundle_docs_publish_contract():
    repo = _repo_root()
    assert_doc_contains_strings(
        repo / "docs" / "references" / "simplified-likelihood-artifacts.md",
        [
            "simplified_likelihood_promotion_evidence_bundle_v0",
            "scripts/benchmarks/build_simplified_likelihood_promotion_evidence_bundle.py",
            "docs/specs/benchmarks/simplified_likelihood_promotion_evidence_bundle_v0.example.json",
            "validator-facing evidence bundle",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-promotion-runbook-2026-03-08.md",
        [
            "promotion evidence bundle",
            "build_simplified_likelihood_promotion_evidence_bundle.py",
            "promotion_evidence.json",
        ],
    )
