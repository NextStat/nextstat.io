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


def _build_smoke_bundle(tmp_path: Path, *, promotion_ready: bool) -> Path:
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

    if promotion_ready:
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
    build_proc = subprocess.run(
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
        check=False,
    )
    bundle_json = bundle_dir / "promotion_evidence.json"
    assert bundle_json.exists(), build_proc.stdout
    if promotion_ready:
        assert build_proc.returncode == 0, build_proc.stdout
    return bundle_dir


def test_simplified_likelihood_promotion_evidence_check_schema_example_and_generator_smoke(
    tmp_path: Path,
):
    repo = _repo_root()
    schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_promotion_evidence_check_v0.schema.json"
    )
    example_path = (
        repo
        / "docs"
        / "specs"
        / "benchmarks"
        / "simplified_likelihood_promotion_evidence_check_v0.example.json"
    )
    assert schema_path.exists(), f"missing schema: {schema_path}"
    assert example_path.exists(), f"missing example: {example_path}"

    example = _load_json(example_path)
    assert example["schema_version"] == "nextstat_simplified_likelihood_promotion_evidence_check_v0"
    _validate_jsonschema(example, schema_path)

    bundle_dir = _build_smoke_bundle(tmp_path, promotion_ready=False)
    report_path = bundle_dir / "promotion_evidence_check.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/verify_simplified_likelihood_promotion_evidence_bundle.py",
            "--bundle-dir",
            str(bundle_dir),
            "--out",
            str(report_path),
            "--deterministic",
        ],
        cwd=repo,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "status=passed" in proc.stdout, proc.stdout

    generated = _load_json(report_path)
    _validate_jsonschema(generated, schema_path)
    assert generated["status"] == "passed"
    assert generated["ok"] is True
    assert generated["checks"]["promotion_readiness"]["status"] == "not_requested"
    assert generated["checks"]["inventory"]["hash_verified_count"] == generated["bundle_summary"]["artifact_count"]


def test_simplified_likelihood_promotion_evidence_check_detects_hash_mismatch(tmp_path: Path):
    repo = _repo_root()
    schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_promotion_evidence_check_v0.schema.json"
    )
    bundle_dir = _build_smoke_bundle(tmp_path, promotion_ready=False)
    tampered = bundle_dir / "files" / "docs" / "benchmarks" / "simplified-likelihood-support-matrix-2026-03-08.md"
    tampered.write_text(tampered.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")

    report_path = bundle_dir / "promotion_evidence_check.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/verify_simplified_likelihood_promotion_evidence_bundle.py",
            "--bundle-dir",
            str(bundle_dir),
            "--out",
            str(report_path),
            "--deterministic",
        ],
        cwd=repo,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1, proc.stdout

    generated = _load_json(report_path)
    _validate_jsonschema(generated, schema_path)
    assert generated["status"] == "failed"
    assert "support_matrix_doc" in generated["checks"]["inventory"]["sha256_mismatches"]
    assert "sha256_mismatch:support_matrix_doc" in generated["summary"]["top_level_errors"]


def test_simplified_likelihood_promotion_evidence_check_can_require_promotion_ready(tmp_path: Path):
    repo = _repo_root()
    schema_path = (
        repo
        / "docs"
        / "schemas"
        / "benchmarks"
        / "simplified_likelihood_promotion_evidence_check_v0.schema.json"
    )
    bundle_dir = _build_smoke_bundle(tmp_path, promotion_ready=True)
    report_path = bundle_dir / "promotion_evidence_check.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/verify_simplified_likelihood_promotion_evidence_bundle.py",
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
        check=True,
    )
    assert "status=passed" in proc.stdout, proc.stdout

    generated = _load_json(report_path)
    _validate_jsonschema(generated, schema_path)
    assert generated["status"] == "passed"
    assert generated["require_promotion_ready"] is True
    assert generated["checks"]["promotion_readiness"]["status"] == "passed"
    assert generated["checks"]["promotion_readiness"]["actual_benchmark_host"] == "nextstat-bench"
    assert (
        generated["checks"]["promotion_readiness"]["actual_min_end_to_end_upper_limit_speedup"]
        >= 10.0
    )


def test_simplified_likelihood_promotion_evidence_check_docs_publish_contract():
    repo = _repo_root()
    assert_doc_contains_strings(
        repo / "docs" / "references" / "simplified-likelihood-artifacts.md",
        [
            "simplified_likelihood_promotion_evidence_check_v0",
            "verify_simplified_likelihood_promotion_evidence_bundle.py",
            "docs/specs/benchmarks/simplified_likelihood_promotion_evidence_check_v0.example.json",
        ],
    )
    assert_doc_contains_strings(
        repo / "docs" / "benchmarks" / "simplified-likelihood-promotion-runbook-2026-03-08.md",
        [
            "promotion_evidence_check.json",
            "verify_simplified_likelihood_promotion_evidence_bundle.py",
            "require-promotion-ready",
        ],
    )
