from __future__ import annotations

import json
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from _io_contract_doc_assertions import assert_doc_contains_strings


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_executable(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _make_nextstat_stub(tmp_path: Path) -> Path:
    stub_path = tmp_path / "nextstat_stub.py"
    _write_executable(
        stub_path,
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

args = sys.argv[1:]

def arg_value(flag: str) -> str:
    idx = args.index(flag)
    return args[idx + 1]

def write_json(path: str, payload: dict[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")

if args == ["--version"]:
    print("nextstat 0.0.0-stub")
elif args[:1] == ["validation-report"]:
    write_json(arg_value("--out"), {"schema_version": "validation_report_v1", "status": "ok"})
elif args[:2] == ["m15", "assessment-table"]:
    write_json(arg_value("--output"), {"schema_version": "m15_assessment_table_v1", "status": "ok"})
elif args[:2] == ["m15", "profile-diff"]:
    write_json(arg_value("--output"), {"schema_version": "m15_profile_diff_report_v1", "status": "ok"})
elif args[:2] == ["m15", "map"]:
    write_json(arg_value("--output"), {"schema_version": "m15_map_v1", "status": "ok"})
elif args[:2] == ["m15", "mar"]:
    write_json(arg_value("--output"), {"schema_version": "m15_mar_v1", "status": "ok"})
elif args[:2] == ["m15", "bundle"]:
    write_json(arg_value("--output"), {"schema_version": "m15_bundle_manifest_v1", "bundle_status": "complete"})
else:
    raise SystemExit(f"unexpected nextstat args: {args}")
""",
    )
    return stub_path


def test_m15_reporting_benchmark_schema_and_example_smoke() -> None:
    jsonschema = pytest.importorskip("jsonschema")

    schema_path = (
        _repo_root() / "docs" / "schemas" / "benchmarks" / "m15_reporting_benchmark_result_v1.schema.json"
    )
    example_path = _repo_root() / "docs" / "specs" / "pharma" / "m15_reporting_benchmark_result_v1.example.json"

    schema_doc = _load_json(schema_path)
    example_doc = _load_json(example_path)

    assert schema_doc["$id"] == (
        "https://nextstat.io/schemas/benchmarks/m15_reporting_benchmark_result_v1.schema.json"
    )
    jsonschema.Draft202012Validator.check_schema(schema_doc)
    jsonschema.validate(example_doc, schema_doc)

    assert example_doc["schema_version"] == "nextstat.m15_reporting_benchmark_result.v1"
    assert example_doc["suite"] == "m15_reporting"
    assert example_doc["meta"]["host_policy"] == "nextstat-bench"
    assert len(example_doc["results"]) >= 6


def test_m15_reporting_benchmark_accepted_baseline_and_compare_example_smoke() -> None:
    jsonschema = pytest.importorskip("jsonschema")

    result_schema = _load_json(
        _repo_root() / "docs" / "schemas" / "benchmarks" / "m15_reporting_benchmark_result_v1.schema.json"
    )
    compare_schema = _load_json(
        _repo_root()
        / "docs"
        / "schemas"
        / "benchmarks"
        / "m15_reporting_benchmark_compare_report_v1.schema.json"
    )
    compare_example = _load_json(
        _repo_root() / "docs" / "specs" / "pharma" / "m15_reporting_benchmark_compare_report_v1.example.json"
    )
    accepted_baseline = _load_json(
        _repo_root() / "benchmarks" / "artifacts" / "m15_reporting_baselines" / "nextstat-bench" / "accepted.json"
    )

    jsonschema.Draft202012Validator.check_schema(compare_schema)
    jsonschema.validate(accepted_baseline, result_schema)
    jsonschema.validate(compare_example, compare_schema)

    assert accepted_baseline["schema_version"] == "nextstat.m15_reporting_benchmark_result.v1"
    assert accepted_baseline["meta"]["host_policy"] == "nextstat-bench"
    assert accepted_baseline["host"]["hostname"] == "nextstat-bench"
    assert accepted_baseline["binary"]["build_profile"] == "release"
    assert compare_example["schema_version"] == "nextstat.m15_reporting_benchmark_compare_report.v1"
    assert compare_example["status"] == "passed"
    assert compare_example["policy"]["required_hostname"] == "nextstat-bench"
    assert compare_example["environment_checks"]["hostname"]["matches"] is True
    assert compare_example["summary"]["failed_cases"] == 0


def test_m15_reporting_benchmark_runner_smoke(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    repo_root = _repo_root()
    nextstat_stub = _make_nextstat_stub(tmp_path)
    out_path = tmp_path / "summary.json"
    markdown_path = tmp_path / "summary.md"
    work_root = tmp_path / "work"

    subprocess.check_call(
        [
            sys.executable,
            "scripts/benchmarks/bench_m15_reporting.py",
            "--nextstat-bin",
            str(nextstat_stub),
            "--smoke",
            "--deterministic",
            "--out",
            str(out_path),
            "--markdown-out",
            str(markdown_path),
            "--work-root",
            str(work_root),
        ],
        cwd=repo_root,
    )

    schema_doc = _load_json(
        repo_root / "docs" / "schemas" / "benchmarks" / "m15_reporting_benchmark_result_v1.schema.json"
    )
    report = _load_json(out_path)
    jsonschema.validate(report, schema_doc)

    assert report["schema_version"] == "nextstat.m15_reporting_benchmark_result.v1"
    assert report["suite"] == "m15_reporting"
    assert report["meta"]["smoke"] is True
    assert report["meta"]["host_policy"] == "nextstat-bench"
    assert report["deterministic"] is True
    assert report["binary"]["version"] == "nextstat 0.0.0-stub"
    assert report["binary"]["sha256"]
    assert report["binary"]["build_profile"] == "unknown"
    cases = {case["case_id"]: case for case in report["results"]}
    assert {
        "m15_assessment_table",
        "m15_map",
        "m15_mar",
        "m15_bundle",
        "validation_pack_base_json_only",
        "validation_pack_m15_json_only",
    } <= set(cases)
    assert all(case["status"] == "ok" for case in cases.values())
    assert all(
        item["status"] == "ok"
        for case in cases.values()
        for item in case["validation"]["validated_artifacts"]
    )
    assert report["derived"]["validation_pack_m15_over_base_median_ratio"] > 0

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# M15 Reporting Benchmark Baseline" in markdown
    assert "validation_pack_m15_json_only" in markdown


def test_m15_reporting_benchmark_compare_runner_passes_on_accepted_baseline(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    repo_root = _repo_root()
    out_path = tmp_path / "compare.json"
    accepted_baseline = (
        repo_root / "benchmarks" / "artifacts" / "m15_reporting_baselines" / "nextstat-bench" / "accepted.json"
    )

    subprocess.check_call(
        [
            sys.executable,
            "scripts/benchmarks/compare_m15_reporting_benchmark.py",
            "--baseline",
            str(accepted_baseline),
            "--current",
            str(accepted_baseline),
            "--out",
            str(out_path),
        ],
        cwd=repo_root,
    )

    compare_schema = _load_json(
        repo_root
        / "docs"
        / "schemas"
        / "benchmarks"
        / "m15_reporting_benchmark_compare_report_v1.schema.json"
    )
    report = _load_json(out_path)
    jsonschema.validate(report, compare_schema)

    assert report["status"] == "passed"
    assert report["ok"] is True
    assert report["requires_review"] is False
    assert report["summary"]["failed_cases"] == 0
    assert report["summary"]["review_cases"] == 0
    assert report["derived"]["validation_pack_m15_over_base_median_ratio"]["status"] == "passed"
    cases = {case["id"]: case for case in report["cases"]}
    assert all(case["status"] == "passed" for case in cases.values())


def test_m15_reporting_benchmark_compare_runner_reports_review_and_fail(tmp_path: Path) -> None:
    repo_root = _repo_root()
    review_current = tmp_path / "review_current.json"
    fail_current = tmp_path / "fail_current.json"
    review_report = tmp_path / "review_report.json"
    fail_report = tmp_path / "fail_report.json"
    accepted_baseline = (
        repo_root / "benchmarks" / "artifacts" / "m15_reporting_baselines" / "nextstat-bench" / "accepted.json"
    )
    baseline_doc = _load_json(accepted_baseline)

    review_doc = json.loads(json.dumps(baseline_doc))
    for case in review_doc["results"]:
        if case["case_id"] == "validation_pack_m15_json_only":
            case["median_s"] = round(case["median_s"] * 1.25, 6)
            case["max_s"] = round(case["max_s"] * 1.25, 6)
            break
    review_current.write_text(json.dumps(review_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    review = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/compare_m15_reporting_benchmark.py",
            "--baseline",
            str(accepted_baseline),
            "--current",
            str(review_current),
            "--out",
            str(review_report),
        ],
        cwd=repo_root,
        check=False,
    )
    assert review.returncode == 0
    review_doc = _load_json(review_report)
    assert review_doc["status"] == "review"
    assert review_doc["requires_review"] is True
    review_case = next(case for case in review_doc["cases"] if case["id"] == "validation_pack_m15_json_only")
    assert review_case["status"] == "review"

    fail_doc = json.loads(json.dumps(baseline_doc))
    for case in fail_doc["results"]:
        if case["case_id"] == "m15_bundle":
            case["median_s"] = round(case["median_s"] * 2.10, 6)
            case["max_s"] = round(case["max_s"] * 2.10, 6)
            break
    fail_current.write_text(json.dumps(fail_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    failed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/compare_m15_reporting_benchmark.py",
            "--baseline",
            str(accepted_baseline),
            "--current",
            str(fail_current),
            "--out",
            str(fail_report),
        ],
        cwd=repo_root,
        check=False,
    )
    assert failed.returncode == 2
    fail_report_doc = _load_json(fail_report)
    assert fail_report_doc["status"] == "failed"
    fail_case = next(case for case in fail_report_doc["cases"] if case["id"] == "m15_bundle")
    assert fail_case["status"] == "failed"


def test_m15_reporting_benchmark_compare_runner_rejects_wrong_hostname(tmp_path: Path) -> None:
    repo_root = _repo_root()
    wrong_host_current = tmp_path / "wrong_host_current.json"
    wrong_host_report = tmp_path / "wrong_host_report.json"
    accepted_baseline = (
        repo_root / "benchmarks" / "artifacts" / "m15_reporting_baselines" / "nextstat-bench" / "accepted.json"
    )
    current_doc = _load_json(accepted_baseline)
    current_doc["host"]["hostname"] = "not-nextstat-bench"
    wrong_host_current.write_text(json.dumps(current_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    failed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/compare_m15_reporting_benchmark.py",
            "--baseline",
            str(accepted_baseline),
            "--current",
            str(wrong_host_current),
            "--out",
            str(wrong_host_report),
        ],
        cwd=repo_root,
        check=False,
    )

    assert failed.returncode == 2
    report = _load_json(wrong_host_report)
    assert report["status"] == "failed"
    assert report["environment_checks"]["hostname"]["matches"] is False
    assert "unexpected_hostname:not-nextstat-bench" in report["summary"]["top_level_errors"]


def test_m15_reporting_benchmark_docs_publish_gate_workflow() -> None:
    assert_doc_contains_strings(
        _repo_root() / "docs" / "benchmarks" / "m15-reporting-runtime-gate.md",
        [
            "# M15 Reporting Runtime Gate",
            "nextstat-bench",
            "scripts/benchmarks/bench_m15_reporting.py",
            "scripts/benchmarks/bench_m15_reporting_remote.sh",
            "scripts/benchmarks/m15_reporting_stable_surface_gate.sh",
            "scripts/benchmarks/compare_m15_reporting_benchmark.py",
            ".github/workflows/m15-reporting-stable-surface.yml",
            "benchmarks/artifacts/m15_reporting_baselines/nextstat-bench/accepted.json",
            "docs/schemas/benchmarks/m15_reporting_benchmark_result_v1.schema.json",
            "docs/specs/pharma/m15_reporting_benchmark_result_v1.example.json",
            "docs/schemas/benchmarks/m15_reporting_benchmark_compare_report_v1.schema.json",
            "docs/specs/pharma/m15_reporting_benchmark_compare_report_v1.example.json",
        ],
    )
    assert_doc_contains_strings(
        _repo_root() / "docs" / "references" / "m15-reporting.md",
        [
            "docs/benchmarks/m15-reporting-runtime-gate.md",
            "scripts/benchmarks/bench_m15_reporting.py",
            "scripts/benchmarks/bench_m15_reporting_remote.sh",
            "scripts/benchmarks/m15_reporting_stable_surface_gate.sh",
            "docs/specs/pharma/m15_reporting_benchmark_result_v1.example.json",
            "scripts/benchmarks/compare_m15_reporting_benchmark.py",
            "benchmarks/artifacts/m15_reporting_baselines/nextstat-bench/accepted.json",
            "docs/specs/pharma/m15_reporting_benchmark_compare_report_v1.example.json",
            ".github/workflows/m15-reporting-stable-surface.yml",
            "make m15-reporting-stable-surface-gate",
        ],
    )
    assert_doc_contains_strings(
        _repo_root() / ".internal" / "docs" / "benchmarks" / "benchmark-inventory.md",
        [
            "scripts/benchmarks/bench_m15_reporting.py",
            "scripts/benchmarks/bench_m15_reporting_remote.sh",
            "scripts/benchmarks/compare_m15_reporting_benchmark.py",
            "make m15-reporting-stable-surface-gate",
            "tmp/m15_reporting_stable_surface/",
            "tmp/m15_reporting_benchmark_<STAMP>/nextstat-bench/",
            "benchmarks/artifacts/m15_reporting_baselines/nextstat-bench/accepted.json",
        ],
    )


def test_m15_reporting_remote_runner_script_syntax() -> None:
    subprocess.check_call(
        ["bash", "-n", "scripts/benchmarks/bench_m15_reporting_remote.sh"],
        cwd=_repo_root(),
    )


def test_m15_reporting_benchmark_script_syntax() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "py_compile", "scripts/benchmarks/bench_m15_reporting.py"],
        cwd=_repo_root(),
    )


def test_m15_reporting_compare_script_syntax() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "py_compile", "scripts/benchmarks/compare_m15_reporting_benchmark.py"],
        cwd=_repo_root(),
    )


def test_m15_reporting_stable_surface_gate_script_syntax() -> None:
    subprocess.check_call(
        ["bash", "-n", "scripts/benchmarks/m15_reporting_stable_surface_gate.sh"],
        cwd=_repo_root(),
    )


def test_m15_reporting_stable_surface_workflow_smoke() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "m15-reporting-stable-surface.yml").read_text(
        encoding="utf-8"
    )

    assert "name: M15 Reporting Stable Surface" in workflow
    assert "name: Contract Gate" in workflow
    assert "name: Benchmark Compare Gate" in workflow
    assert 'runs-on: ${{ fromJSON(\'["self-hosted","linux","x64","bench"]\') }}' in workflow
    assert "bash -n scripts/benchmarks/m15_reporting_stable_surface_gate.sh" in workflow
    assert "make m15-reporting-stable-surface-gate" in workflow
    assert "tmp/m15_reporting_stable_surface/m15_reporting_benchmark.json" in workflow
    assert "tmp/m15_reporting_stable_surface/m15_reporting_benchmark.md" in workflow
    assert "tmp/m15_reporting_stable_surface/m15_reporting_compare.json" in workflow
