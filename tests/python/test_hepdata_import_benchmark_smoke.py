from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from _io_contract_doc_assertions import assert_doc_contains_strings


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _accepted_baseline_path() -> Path:
    return _repo_root() / "benchmarks" / "artifacts" / "hepdata_import_baselines" / "nextstat-bench" / "accepted.json"


def _comparison_schema_path() -> Path:
    return (
        _repo_root()
        / "docs"
        / "schemas"
        / "benchmarks"
        / "hepdata_import_benchmark_compare_report_v1.schema.json"
    )


def _promotion_schema_path() -> Path:
    return (
        _repo_root()
        / "docs"
        / "schemas"
        / "benchmarks"
        / "hepdata_import_benchmark_baseline_promotion_report_v1.schema.json"
    )


def _gate_schema_path() -> Path:
    return (
        _repo_root()
        / "docs"
        / "schemas"
        / "benchmarks"
        / "hepdata_import_benchmark_gate_report_v1.schema.json"
    )


def test_hepdata_import_benchmark_schema_and_example_smoke() -> None:
    jsonschema = pytest.importorskip("jsonschema")

    schema_path = (
        _repo_root() / "docs" / "schemas" / "benchmarks" / "hepdata_import_benchmark_result_v1.schema.json"
    )
    example_path = _repo_root() / "docs" / "specs" / "hepdata_import_benchmark_result_v1.example.json"

    schema_doc = _load_json(schema_path)
    example_doc = _load_json(example_path)

    assert schema_doc["$id"] == (
        "https://nextstat.io/schemas/benchmarks/hepdata_import_benchmark_result_v1.schema.json"
    )
    jsonschema.Draft202012Validator.check_schema(schema_doc)
    jsonschema.validate(example_doc, schema_doc)

    assert example_doc["schema_version"] == "nextstat.hepdata_import_benchmark_result.v1"
    assert example_doc["suite"] == "hepdata_import"
    assert len(example_doc["cases"]) >= 4
    cases = {case["id"]: case for case in example_doc["cases"]}
    assert cases["curated_catalog"]["stages"]["import_total_s"] == cases["curated_catalog"]["timing"]["best_s"]
    assert cases["direct_patch_catalog_cached"]["stages"]["extract_s"] > 0
    assert cases["curated_materialize_offline"]["stages"]["download_s"] == 0
    assert cases["direct_materialize_network"]["stages"]["download_s"] > 0
    assert cases["direct_materialize_network"]["stages"]["fit_s"] is not None


def test_hepdata_import_benchmark_accepted_baseline_and_compare_example_smoke() -> None:
    jsonschema = pytest.importorskip("jsonschema")

    result_schema = _load_json(
        _repo_root() / "docs" / "schemas" / "benchmarks" / "hepdata_import_benchmark_result_v1.schema.json"
    )
    compare_schema = _load_json(_comparison_schema_path())
    compare_example = _load_json(
        _repo_root() / "docs" / "specs" / "hepdata_import_benchmark_compare_report_v1.example.json"
    )
    accepted_baseline = _load_json(_accepted_baseline_path())

    jsonschema.Draft202012Validator.check_schema(compare_schema)
    jsonschema.validate(accepted_baseline, result_schema)
    jsonschema.validate(compare_example, compare_schema)

    assert accepted_baseline["schema_version"] == "nextstat.hepdata_import_benchmark_result.v1"
    assert accepted_baseline["meta"]["host_policy"] == "nextstat-bench"
    assert compare_example["schema_version"] == "nextstat.hepdata_import_benchmark_compare_report.v1"
    assert compare_example["status"] == "passed"
    assert compare_example["summary"]["failed_cases"] == 0


def test_hepdata_import_benchmark_promotion_schema_and_example_smoke() -> None:
    jsonschema = pytest.importorskip("jsonschema")

    promotion_schema = _load_json(_promotion_schema_path())
    promotion_example = _load_json(
        _repo_root() / "docs" / "specs" / "hepdata_import_benchmark_baseline_promotion_report_v1.example.json"
    )

    jsonschema.Draft202012Validator.check_schema(promotion_schema)
    jsonschema.validate(promotion_example, promotion_schema)

    assert promotion_example["schema_version"] == (
        "nextstat.hepdata_import_benchmark_baseline_promotion_report.v1"
    )
    assert promotion_example["status"] == "dry_run"
    assert promotion_example["promoted"] is False
    assert promotion_example["dry_run"] is True
    assert promotion_example["compare_status"] == "passed"


def test_hepdata_import_benchmark_gate_schema_and_example_smoke() -> None:
    jsonschema = pytest.importorskip("jsonschema")

    gate_schema = _load_json(_gate_schema_path())
    gate_example = _load_json(
        _repo_root() / "docs" / "specs" / "hepdata_import_benchmark_gate_report_v1.example.json"
    )

    jsonschema.Draft202012Validator.check_schema(gate_schema)
    jsonschema.validate(gate_example, gate_schema)

    assert gate_example["schema_version"] == "nextstat.hepdata_import_benchmark_gate_report.v1"
    assert gate_example["status"] == "passed"
    assert gate_example["promotion_mode"] == "dry_run"
    assert gate_example["steps"]["benchmark"]["mode"] == "provided_artifact"
    assert gate_example["steps"]["compare"]["compare_status"] == "passed"
    assert gate_example["steps"]["promotion"]["promotion_status"] == "dry_run"


def test_hepdata_import_benchmark_runner_smoke(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    out_path = tmp_path / "summary.json"
    work_root = tmp_path / "work"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/benchmarks/bench_hepdata_import.py",
            "--smoke",
            "--deterministic",
            "--out",
            str(out_path),
            "--work-root",
            str(work_root),
        ],
        cwd=_repo_root(),
    )

    schema_doc = _load_json(
        _repo_root() / "docs" / "schemas" / "benchmarks" / "hepdata_import_benchmark_result_v1.schema.json"
    )
    report = _load_json(out_path)
    jsonschema.validate(report, schema_doc)

    assert report["schema_version"] == "nextstat.hepdata_import_benchmark_result.v1"
    assert report["suite"] == "hepdata_import"
    assert report["meta"]["smoke"] is True

    cases = {case["id"]: case for case in report["cases"]}
    assert {
        "curated_catalog",
        "direct_patch_catalog_cached",
        "curated_materialize_offline",
        "direct_materialize_network",
    } <= set(cases)
    assert all(case["status"] == "ok" for case in cases.values())
    assert cases["curated_materialize_offline"]["validation"]["lockfile_written"] is True
    assert cases["direct_materialize_network"]["validation"]["download_mode"] == "network"
    assert cases["direct_materialize_network"]["dataset"]["doi"] == "https://doi.org/10.17182/hepdata.90607.v3/r3"
    assert cases["curated_catalog"]["stages"]["import_total_s"] == cases["curated_catalog"]["timing"]["best_s"]
    assert cases["direct_patch_catalog_cached"]["stages"]["extract_s"] >= 0
    assert cases["curated_materialize_offline"]["stages"]["download_s"] == 0
    assert cases["curated_materialize_offline"]["stages"]["materialize_s"] >= 0
    assert cases["direct_materialize_network"]["stages"]["discovery_s"] >= 0
    assert cases["direct_materialize_network"]["stages"]["download_s"] >= 0
    assert cases["direct_materialize_network"]["stages"]["extract_s"] >= 0
    assert cases["direct_materialize_network"]["stages"]["materialize_s"] >= 0
    assert cases["direct_materialize_network"]["stages"]["fit_s"] is not None


def test_hepdata_import_benchmark_compare_runner_passes_on_accepted_baseline(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    out_path = tmp_path / "compare.json"
    accepted_baseline = _accepted_baseline_path()
    subprocess.check_call(
        [
            sys.executable,
            "scripts/benchmarks/compare_hepdata_import_benchmark.py",
            "--baseline",
            str(accepted_baseline),
            "--current",
            str(accepted_baseline),
            "--out",
            str(out_path),
        ],
        cwd=_repo_root(),
    )

    compare_schema = _load_json(_comparison_schema_path())
    report = _load_json(out_path)
    jsonschema.validate(report, compare_schema)

    assert report["status"] == "passed"
    assert report["ok"] is True
    assert report["requires_review"] is False
    assert report["summary"]["failed_cases"] == 0
    assert report["summary"]["review_cases"] == 0
    cases = {case["id"]: case for case in report["cases"]}
    assert all(case["status"] == "passed" for case in cases.values())
    download_metric = next(
        metric for metric in cases["direct_materialize_network"]["metrics"] if metric["name"] == "download_s"
    )
    assert download_metric["status"] == "skipped_floor"


def test_hepdata_import_benchmark_compare_runner_reports_review_and_fail(tmp_path: Path) -> None:
    review_current = tmp_path / "review_current.json"
    fail_current = tmp_path / "fail_current.json"
    review_report = tmp_path / "review_report.json"
    fail_report = tmp_path / "fail_report.json"
    accepted_baseline = _accepted_baseline_path()
    baseline_doc = _load_json(accepted_baseline)

    review_doc = json.loads(json.dumps(baseline_doc))
    for case in review_doc["cases"]:
        if case["id"] == "curated_materialize_offline":
            case["timing"]["best_s"] = round(case["timing"]["best_s"] * 1.2, 6)
            case["timing"]["per_run_s"][0] = case["timing"]["best_s"]
            case["stages"]["import_total_s"] = case["timing"]["best_s"]
            break
    review_current.write_text(json.dumps(review_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    review = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/compare_hepdata_import_benchmark.py",
            "--baseline",
            str(accepted_baseline),
            "--current",
            str(review_current),
            "--out",
            str(review_report),
        ],
        cwd=_repo_root(),
        check=False,
    )
    assert review.returncode == 0
    review_doc = _load_json(review_report)
    assert review_doc["status"] == "review"
    assert review_doc["requires_review"] is True
    review_case = next(case for case in review_doc["cases"] if case["id"] == "curated_materialize_offline")
    assert review_case["status"] == "review"

    fail_doc = json.loads(json.dumps(baseline_doc))
    for case in fail_doc["cases"]:
        if case["id"] == "direct_patch_catalog_cached":
            case["timing"]["best_s"] = round(case["timing"]["best_s"] * 1.5, 6)
            case["timing"]["per_run_s"][0] = case["timing"]["best_s"]
            case["stages"]["import_total_s"] = case["timing"]["best_s"]
            break
    fail_current.write_text(json.dumps(fail_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    failed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/compare_hepdata_import_benchmark.py",
            "--baseline",
            str(accepted_baseline),
            "--current",
            str(fail_current),
            "--out",
            str(fail_report),
        ],
        cwd=_repo_root(),
        check=False,
    )
    assert failed.returncode == 2
    fail_report_doc = _load_json(fail_report)
    assert fail_report_doc["status"] == "failed"
    fail_case = next(case for case in fail_report_doc["cases"] if case["id"] == "direct_patch_catalog_cached")
    assert fail_case["status"] == "failed"


def test_hepdata_import_benchmark_compare_normalizes_fixture_doi_identity(tmp_path: Path) -> None:
    accepted_baseline = _accepted_baseline_path()
    baseline_doc = _load_json(accepted_baseline)
    current_doc = json.loads(json.dumps(baseline_doc))

    for case in baseline_doc["cases"]:
        if case["id"] == "direct_materialize_network":
            case["dataset"]["doi"] = "http://127.0.0.1:30123/download"
            break
    for case in current_doc["cases"]:
        if case["id"] == "direct_materialize_network":
            case["dataset"]["doi"] = "https://doi.org/10.17182/hepdata.90607.v3/r3"
            break

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    report_path = tmp_path / "compare.json"
    baseline_path.write_text(json.dumps(baseline_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    current_path.write_text(json.dumps(current_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    subprocess.check_call(
        [
            sys.executable,
            "scripts/benchmarks/compare_hepdata_import_benchmark.py",
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
            "--out",
            str(report_path),
        ],
        cwd=_repo_root(),
    )

    report_doc = _load_json(report_path)
    assert report_doc["status"] == "passed"
    case_doc = next(case for case in report_doc["cases"] if case["id"] == "direct_materialize_network")
    assert case_doc["identity"]["dataset"]["matches"] is True


def test_hepdata_import_benchmark_promote_runner_dry_run_and_promote(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    accepted_dir = tmp_path / "accepted"
    accepted_path = accepted_dir / "accepted.json"
    current_path = tmp_path / "current.json"
    compare_report = tmp_path / "compare_report.json"
    dry_run_report = tmp_path / "dry_run_report.json"
    promote_report = tmp_path / "promote_report.json"
    history_dir = tmp_path / "history"
    accepted_dir.mkdir(parents=True, exist_ok=True)

    baseline_doc = _load_json(_accepted_baseline_path())
    accepted_path.write_text(json.dumps(baseline_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    current_doc = json.loads(json.dumps(baseline_doc))
    current_doc["meta"]["note"] = "candidate"
    current_path.write_text(json.dumps(current_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    subprocess.check_call(
        [
            sys.executable,
            "scripts/benchmarks/promote_hepdata_import_benchmark_baseline.py",
            "--accepted",
            str(accepted_path),
            "--current",
            str(current_path),
            "--compare-report",
            str(compare_report),
            "--report",
            str(dry_run_report),
            "--history-dir",
            str(history_dir),
            "--dry-run",
        ],
        cwd=_repo_root(),
    )

    promotion_schema = _load_json(_promotion_schema_path())
    dry_run_doc = _load_json(dry_run_report)
    jsonschema.validate(dry_run_doc, promotion_schema)
    assert dry_run_doc["status"] == "dry_run"
    assert dry_run_doc["promoted"] is False
    assert dry_run_doc["actions"]["accepted_updated"] is False
    assert _load_json(accepted_path).get("meta", {}).get("note") is None

    subprocess.check_call(
        [
            sys.executable,
            "scripts/benchmarks/promote_hepdata_import_benchmark_baseline.py",
            "--accepted",
            str(accepted_path),
            "--current",
            str(current_path),
            "--compare-report",
            str(compare_report),
            "--report",
            str(promote_report),
            "--history-dir",
            str(history_dir),
        ],
        cwd=_repo_root(),
    )

    promoted_doc = _load_json(promote_report)
    jsonschema.validate(promoted_doc, promotion_schema)
    assert promoted_doc["status"] == "promoted"
    assert promoted_doc["promoted"] is True
    assert promoted_doc["actions"]["accepted_updated"] is True
    assert promoted_doc["actions"]["archived_previous_baseline"] is True
    assert promoted_doc["actions"]["archived_promoted_snapshot"] is True
    assert _load_json(accepted_path)["meta"]["note"] == "candidate"
    previous_path = Path(promoted_doc["actions"]["archived_previous_baseline_path"])
    promoted_path = Path(promoted_doc["actions"]["archived_promoted_snapshot_path"])
    assert previous_path.exists()
    assert promoted_path.exists()


def test_hepdata_import_benchmark_promote_runner_blocks_review_without_override(tmp_path: Path) -> None:
    accepted_path = tmp_path / "accepted.json"
    current_path = tmp_path / "review_current.json"
    compare_report = tmp_path / "compare_report.json"
    promote_report = tmp_path / "promote_report.json"

    baseline_doc = _load_json(_accepted_baseline_path())
    accepted_path.write_text(json.dumps(baseline_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    review_doc = json.loads(json.dumps(baseline_doc))
    for case in review_doc["cases"]:
        if case["id"] == "curated_materialize_offline":
            case["timing"]["best_s"] = round(case["timing"]["best_s"] * 1.2, 6)
            case["timing"]["per_run_s"][0] = case["timing"]["best_s"]
            case["stages"]["import_total_s"] = case["timing"]["best_s"]
            break
    current_path.write_text(json.dumps(review_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    blocked = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/promote_hepdata_import_benchmark_baseline.py",
            "--accepted",
            str(accepted_path),
            "--current",
            str(current_path),
            "--compare-report",
            str(compare_report),
            "--report",
            str(promote_report),
        ],
        cwd=_repo_root(),
        check=False,
    )
    assert blocked.returncode == 2
    blocked_doc = _load_json(promote_report)
    assert blocked_doc["status"] == "failed"
    assert "compare_status_review_requires_allow_review" in blocked_doc["summary"]["top_level_errors"]

    allowed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/promote_hepdata_import_benchmark_baseline.py",
            "--accepted",
            str(accepted_path),
            "--current",
            str(current_path),
            "--compare-report",
            str(compare_report),
            "--report",
            str(promote_report),
            "--allow-review",
            "--dry-run",
        ],
        cwd=_repo_root(),
        check=False,
    )
    assert allowed.returncode == 0
    allowed_doc = _load_json(promote_report)
    assert allowed_doc["status"] == "dry_run"
    assert allowed_doc["compare_status"] == "review"
    assert allowed_doc["allow_review"] is True


def test_hepdata_import_benchmark_gate_runner_with_provided_artifact(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    compare_report = tmp_path / "compare_report.json"
    gate_report = tmp_path / "gate_report.json"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/benchmarks/run_hepdata_import_benchmark_gate.py",
            "--current",
            str(_accepted_baseline_path()),
            "--compare-report",
            str(compare_report),
            "--report",
            str(gate_report),
            "--promotion-mode",
            "none",
        ],
        cwd=_repo_root(),
    )

    gate_schema = _load_json(_gate_schema_path())
    compare_schema = _load_json(_comparison_schema_path())
    gate_doc = _load_json(gate_report)
    compare_doc = _load_json(compare_report)
    jsonschema.validate(gate_doc, gate_schema)
    jsonschema.validate(compare_doc, compare_schema)

    assert gate_doc["status"] == "passed"
    assert gate_doc["promotion_mode"] == "none"
    assert gate_doc["steps"]["benchmark"]["status"] == "skipped"
    assert gate_doc["steps"]["benchmark"]["mode"] == "provided_artifact"
    assert gate_doc["steps"]["compare"]["status"] == "passed"
    assert gate_doc["steps"]["compare"]["compare_status"] == "passed"
    assert gate_doc["steps"]["promotion"]["status"] == "skipped"
    assert gate_doc["steps"]["promotion"]["mode"] == "none"
    assert gate_doc["summary"]["top_level_errors"] == []


def test_hepdata_import_benchmark_gate_runner_dry_run_promotion(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    compare_report = tmp_path / "compare_report.json"
    promotion_report = tmp_path / "promotion_report.json"
    gate_report = tmp_path / "gate_report.json"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/benchmarks/run_hepdata_import_benchmark_gate.py",
            "--current",
            str(_accepted_baseline_path()),
            "--compare-report",
            str(compare_report),
            "--promotion-report",
            str(promotion_report),
            "--report",
            str(gate_report),
            "--promotion-mode",
            "dry_run",
        ],
        cwd=_repo_root(),
    )

    gate_schema = _load_json(_gate_schema_path())
    promotion_schema = _load_json(_promotion_schema_path())
    gate_doc = _load_json(gate_report)
    promotion_doc = _load_json(promotion_report)
    jsonschema.validate(gate_doc, gate_schema)
    jsonschema.validate(promotion_doc, promotion_schema)

    assert gate_doc["status"] == "passed"
    assert gate_doc["promotion_mode"] == "dry_run"
    assert gate_doc["steps"]["promotion"]["status"] == "passed"
    assert gate_doc["steps"]["promotion"]["promotion_status"] == "dry_run"
    assert gate_doc["summary"]["promotion_status"] == "dry_run"
    assert promotion_doc["status"] == "dry_run"


def test_hepdata_import_benchmark_gate_runner_executes_fake_remote_flow(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    artifact_dir = tmp_path / "remote_artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_dir / "summary.json"
    summary_path.write_text(
        _accepted_baseline_path().read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    runner_script = tmp_path / "fake_remote_runner.sh"
    runner_script.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' '[hepdata-import-remote] done: {artifact_dir}'\n",
        encoding="utf-8",
    )
    runner_script.chmod(0o755)

    compare_report = tmp_path / "compare_report.json"
    gate_report = tmp_path / "gate_report.json"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/benchmarks/run_hepdata_import_benchmark_gate.py",
            "--runner-cmd",
            f"sh {runner_script}",
            "--compare-report",
            str(compare_report),
            "--report",
            str(gate_report),
            "--promotion-mode",
            "none",
        ],
        cwd=_repo_root(),
    )

    gate_schema = _load_json(_gate_schema_path())
    gate_doc = _load_json(gate_report)
    jsonschema.validate(gate_doc, gate_schema)

    assert gate_doc["status"] == "passed"
    assert gate_doc["steps"]["benchmark"]["mode"] == "runner"
    assert gate_doc["steps"]["benchmark"]["status"] == "passed"
    assert gate_doc["steps"]["benchmark"]["artifact_path"] == str(summary_path)
    assert gate_doc["steps"]["compare"]["compare_status"] == "passed"


def test_hepdata_import_benchmark_gate_runner_fails_when_runner_exits_non_zero(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    artifact_dir = tmp_path / "remote_artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_dir / "summary.json"
    summary_path.write_text(
        _accepted_baseline_path().read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    runner_script = tmp_path / "fake_remote_runner_fail.sh"
    runner_script.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' '[hepdata-import-remote] done: {artifact_dir}'\n"
        "exit 7\n",
        encoding="utf-8",
    )
    runner_script.chmod(0o755)

    compare_report = tmp_path / "compare_report.json"
    gate_report = tmp_path / "gate_report.json"
    failed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/run_hepdata_import_benchmark_gate.py",
            "--runner-cmd",
            f"sh {runner_script}",
            "--compare-report",
            str(compare_report),
            "--report",
            str(gate_report),
            "--promotion-mode",
            "none",
        ],
        cwd=_repo_root(),
        check=False,
    )
    assert failed.returncode == 2

    gate_schema = _load_json(_gate_schema_path())
    gate_doc = _load_json(gate_report)
    jsonschema.validate(gate_doc, gate_schema)

    assert gate_doc["status"] == "failed"
    assert gate_doc["steps"]["benchmark"]["status"] == "failed"
    assert "runner_exit:7" in gate_doc["steps"]["benchmark"]["errors"]
    assert "benchmark_step_failed" in gate_doc["summary"]["top_level_errors"]
    assert gate_doc["steps"]["compare"]["compare_status"] == "passed"


def test_hepdata_import_benchmark_docs_publish_gate_workflow() -> None:
    assert_doc_contains_strings(
        _repo_root() / "docs" / "benchmarks.md",
        [
            "HEPData Import Acceptance Criteria (Stable Surface v1)",
            "HEPData Import Runtime Gate",
            "HEPData Import Benchmark Snapshot: 2026-03-08",
            "HEPData Import Stable-Surface Support Matrix",
            "HEPData Import Stable-Surface Release Notes",
            "HEPData Import Release PR Checklist",
            "HEPData Import Promotion Runbook",
        ],
    )
    assert_doc_contains_strings(
        _repo_root() / "docs" / "benchmarks" / "hepdata-import-runtime-gate.md",
        [
            "# HEPData Import Runtime Gate",
            "docs/benchmarks/hepdata-import-benchmark-snapshot-2026-03-08.md",
            "docs/benchmarks/hepdata-import-support-matrix-2026-03-08.md",
            "docs/benchmarks/hepdata-import-release-notes-2026-03-08.md",
            "docs/benchmarks/hepdata-import-promotion-runbook-2026-03-08.md",
            "docs/benchmarks/hepdata-import-release-pr-checklist-2026-03-08.md",
            "nextstat-bench",
            "scripts/benchmarks/bench_hepdata_import.py",
            "scripts/benchmarks/compare_hepdata_import_benchmark.py",
            "scripts/benchmarks/promote_hepdata_import_benchmark_baseline.py",
            "scripts/benchmarks/run_hepdata_import_benchmark_gate.py",
            "scripts/benchmarks/bench_hepdata_import_remote.sh",
            "benchmarks/artifacts/hepdata_import_baselines/nextstat-bench/accepted.json",
            "docs/schemas/benchmarks/hepdata_import_benchmark_result_v1.schema.json",
            "docs/specs/hepdata_import_benchmark_result_v1.example.json",
            "docs/schemas/benchmarks/hepdata_import_benchmark_compare_report_v1.schema.json",
            "docs/specs/hepdata_import_benchmark_compare_report_v1.example.json",
            "docs/schemas/benchmarks/hepdata_import_benchmark_baseline_promotion_report_v1.schema.json",
            "docs/specs/hepdata_import_benchmark_baseline_promotion_report_v1.example.json",
            "docs/schemas/benchmarks/hepdata_import_benchmark_gate_report_v1.schema.json",
            "docs/specs/hepdata_import_benchmark_gate_report_v1.example.json",
            "stage-level timing breakdown",
            "benchmarks/nextstat-public-benchmarks",
            "python3 suites/hep/run.py --deterministic --out out/hep_simple_nll.json",
            "BENCH_SSH_USER=<user> bash scripts/benchmarks/bench_hepdata_import_remote.sh",
            "python3 scripts/benchmarks/run_hepdata_import_benchmark_gate.py --promotion-mode dry_run",
        ],
    )
    assert_doc_contains_strings(
        _repo_root() / "docs" / "specs" / "hep" / "hepdata_import_acceptance_v1.md",
        [
            "docs/benchmarks/hepdata-import-benchmark-snapshot-2026-03-08.md",
            "docs/benchmarks/hepdata-import-support-matrix-2026-03-08.md",
            "docs/benchmarks/hepdata-import-release-notes-2026-03-08.md",
            "docs/benchmarks/hepdata-import-promotion-runbook-2026-03-08.md",
            "docs/benchmarks/hepdata-import-release-pr-checklist-2026-03-08.md",
            "docs/benchmarks/hepdata-import-runtime-gate.md",
            "scripts/benchmarks/bench_hepdata_import.py --deterministic --out bench_results/hepdata_import_benchmark/summary.json",
            "scripts/benchmarks/compare_hepdata_import_benchmark.py --current bench_results/hepdata_import_benchmark/summary.json",
            "scripts/benchmarks/promote_hepdata_import_benchmark_baseline.py --current bench_results/hepdata_import_benchmark/summary.json",
            "bash scripts/benchmarks/bench_hepdata_import_remote.sh",
            "benchmarks/artifacts/hepdata_import_baselines/nextstat-bench/accepted.json",
            "docs/schemas/benchmarks/hepdata_import_benchmark_result_v1.schema.json",
            "docs/specs/hepdata_import_benchmark_result_v1.example.json",
            "docs/schemas/benchmarks/hepdata_import_benchmark_compare_report_v1.schema.json",
            "docs/specs/hepdata_import_benchmark_compare_report_v1.example.json",
            "docs/schemas/benchmarks/hepdata_import_benchmark_baseline_promotion_report_v1.schema.json",
            "docs/specs/hepdata_import_benchmark_baseline_promotion_report_v1.example.json",
            "docs/schemas/benchmarks/hepdata_import_benchmark_gate_report_v1.schema.json",
            "docs/specs/hepdata_import_benchmark_gate_report_v1.example.json",
            "scripts/benchmarks/run_hepdata_import_benchmark_gate.py --promotion-mode dry_run",
        ],
    )
    assert_doc_contains_strings(
        _repo_root() / "docs" / "references" / "cli.md",
        [
            "docs/benchmarks/hepdata-import-runtime-gate.md",
        ],
    )


def test_hepdata_import_benchmark_snapshot_publishes_frozen_evidence() -> None:
    assert_doc_contains_strings(
        _repo_root() / "docs" / "benchmarks" / "hepdata-import-benchmark-snapshot-2026-03-08.md",
        [
            "# HEPData Import Benchmark Snapshot: 2026-03-08",
            "benchmarks/artifacts/hepdata_import_baselines/nextstat-bench/accepted.json",
            "accepted_20260308T171936Z_previous.json",
            "accepted_20260308T171936Z_promoted.json",
            "intentionally not committed",
            "docs/benchmarks/hepdata-import-promotion-runbook-2026-03-08.md",
            "docs/benchmarks/hepdata-import-release-pr-checklist-2026-03-08.md",
            "python3 scripts/benchmarks/run_hepdata_import_benchmark_gate.py --promotion-mode dry_run",
            "--current tmp/hepdata_import_benchmark_20260308T164825Z/nextstat-bench/summary.json",
            "--promotion-mode apply",
            "BENCH_SSH_USER=<user> bash scripts/benchmarks/bench_hepdata_import_remote.sh",
            "curated_catalog",
            "direct_patch_catalog_cached",
            "curated_materialize_offline",
            "direct_materialize_network",
            "logical HEPData DOI",
        ],
    )

    assert_doc_contains_strings(
        _repo_root() / "docs" / "benchmarks" / "hepdata-import-support-matrix-2026-03-08.md",
        [
            "# HEPData Import Stable-Surface Support Matrix",
            "nextstat import hepdata --list",
            "nextstat import hepdata --list-patches --doi <url> --dataset-id <id>",
            "nextstat.hepdata_import.v1",
            "nextstat.hepdata_lock.v1",
            "nextstat.hepdata_import_benchmark_gate_report.v1",
            "python3 scripts/check_io_contracts.py --family hepdata",
            "python3 scripts/benchmarks/run_hepdata_import_benchmark_gate.py",
            "research-grade",
            "hidden DOI-to-dataset inference",
            "docs/benchmarks/hepdata-import-benchmark-snapshot-2026-03-08.md",
            "docs/benchmarks/hepdata-import-promotion-runbook-2026-03-08.md",
            "docs/benchmarks/hepdata-import-release-pr-checklist-2026-03-08.md",
            "docs/references/cli.md",
        ],
    )

    assert_doc_contains_strings(
        _repo_root() / "docs" / "benchmarks" / "hepdata-import-release-notes-2026-03-08.md",
        [
            "# HEPData Import Stable-Surface Release Notes",
            "nextstat import hepdata --list",
            "nextstat import hepdata --doi <url> --dataset-id <id> ...",
            "nextstat.hepdata_import.v1",
            "nextstat.hepdata_import_benchmark_gate_report.v1",
            "logical HEPData DOI",
            "docs/benchmarks/hepdata-import-support-matrix-2026-03-08.md",
            "docs/benchmarks/hepdata-import-benchmark-snapshot-2026-03-08.md",
            "docs/benchmarks/hepdata-import-promotion-runbook-2026-03-08.md",
            "docs/benchmarks/hepdata-import-release-pr-checklist-2026-03-08.md",
            "accepted.json",
            "hidden DOI-to-dataset inference",
        ],
    )

    assert_doc_contains_strings(
        _repo_root() / "docs" / "benchmarks" / "hepdata-import-promotion-runbook-2026-03-08.md",
        [
            "# HEPData Import Promotion Runbook",
            "nextstat-bench",
            "python3 scripts/benchmarks/run_hepdata_import_benchmark_gate.py --promotion-mode dry_run",
            "python3 scripts/benchmarks/run_hepdata_import_benchmark_gate.py \\",
            "python3 scripts/check_io_contracts.py --family hepdata",
            "cargo test -p ns-cli --test cli_import_hepdata",
            "pytest -q tests/python/test_hepdata_import_benchmark_smoke.py",
            "logical HEPData DOI",
            "docs/benchmarks/hepdata-import-benchmark-snapshot-2026-03-08.md",
            "docs/benchmarks/hepdata-import-support-matrix-2026-03-08.md",
            "docs/benchmarks/hepdata-import-release-notes-2026-03-08.md",
        ],
    )

    assert_doc_contains_strings(
        _repo_root() / "docs" / "benchmarks" / "hepdata-import-release-pr-checklist-2026-03-08.md",
        [
            "# HEPData Import Release PR Checklist",
            "docs/benchmarks/hepdata-import-promotion-runbook-2026-03-08.md",
            "docs/benchmarks/hepdata-import-release-pr-checklist-2026-03-08.md",
            "docs/benchmarks/hepdata-import-benchmark-snapshot-2026-03-08.md",
            "docs/benchmarks/hepdata-import-support-matrix-2026-03-08.md",
            "docs/benchmarks/hepdata-import-release-notes-2026-03-08.md",
            "python3 scripts/check_io_contracts.py --family hepdata",
            "python3 scripts/benchmarks/run_hepdata_import_benchmark_gate.py --promotion-mode dry_run",
            "git diff --check -- <touched files>",
            "hidden DOI-to-dataset inference",
            "what is stable now",
        ],
    )


def test_hepdata_import_remote_runner_script_syntax() -> None:
    subprocess.check_call(
        ["bash", "-n", "scripts/benchmarks/bench_hepdata_import_remote.sh"],
        cwd=_repo_root(),
    )


def test_hepdata_import_benchmark_compare_script_syntax() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "py_compile", "scripts/benchmarks/compare_hepdata_import_benchmark.py"],
        cwd=_repo_root(),
    )


def test_hepdata_import_benchmark_promote_script_syntax() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "py_compile", "scripts/benchmarks/promote_hepdata_import_benchmark_baseline.py"],
        cwd=_repo_root(),
    )


def test_hepdata_import_benchmark_gate_script_syntax() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "py_compile", "scripts/benchmarks/run_hepdata_import_benchmark_gate.py"],
        cwd=_repo_root(),
    )
