from __future__ import annotations

import json
import os
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


def _result_schema_path() -> Path:
    return (
        _repo_root()
        / "docs"
        / "schemas"
        / "benchmarks"
        / "ads_variance_reduction_benchmark_result_v1.schema.json"
    )


def _result_example_path() -> Path:
    return (
        _repo_root()
        / "docs"
        / "specs"
        / "benchmarks"
        / "ads_variance_reduction_benchmark_result_v1.example.json"
    )


def _compare_schema_path() -> Path:
    return (
        _repo_root()
        / "docs"
        / "schemas"
        / "benchmarks"
        / "ads_variance_reduction_benchmark_compare_report_v1.schema.json"
    )


def _compare_example_path() -> Path:
    return (
        _repo_root()
        / "docs"
        / "specs"
        / "benchmarks"
        / "ads_variance_reduction_benchmark_compare_report_v1.example.json"
    )


def _gate_schema_path() -> Path:
    return (
        _repo_root()
        / "docs"
        / "schemas"
        / "benchmarks"
        / "ads_variance_reduction_benchmark_gate_report_v1.schema.json"
    )


def _gate_example_path() -> Path:
    return (
        _repo_root()
        / "docs"
        / "specs"
        / "benchmarks"
        / "ads_variance_reduction_benchmark_gate_report_v1.example.json"
    )


def _promotion_schema_path() -> Path:
    return (
        _repo_root()
        / "docs"
        / "schemas"
        / "benchmarks"
        / "ads_variance_reduction_benchmark_baseline_promotion_report_v1.schema.json"
    )


def _promotion_example_path() -> Path:
    return (
        _repo_root()
        / "docs"
        / "specs"
        / "benchmarks"
        / "ads_variance_reduction_benchmark_baseline_promotion_report_v1.example.json"
    )


def _scenario_manifest_path() -> Path:
    return (
        _repo_root()
        / "tests"
        / "fixtures"
        / "variance_reduction_benchmark"
        / "scenario_matrix.json"
    )


def _make_nextstat_python_stub(tmp_path: Path) -> Path:
    pkg_root = tmp_path / "py_stub"
    module_root = pkg_root / "nextstat"
    module_root.mkdir(parents=True, exist_ok=True)

    (module_root / "__init__.py").write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            __version__ = "0.0.0-stub"

            from . import ads


            def set_threads(_: int) -> None:
                return None


            def set_eval_mode(_: str) -> None:
                return None
            """
        ),
        encoding="utf-8",
    )

    (module_root / "ads.py").write_text(
        textwrap.dedent(
            """
            from __future__ import annotations


            def _mean(values):
                return sum(values) / len(values)


            def _sample_variance(values, mean_value):
                if len(values) <= 1:
                    return 0.0
                return sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)


            def _variance_of_difference(control_outcomes, variant_outcomes):
                mean_control = _mean(control_outcomes)
                mean_variant = _mean(variant_outcomes)
                return (
                    _sample_variance(control_outcomes, mean_control) / max(1, len(control_outcomes))
                    + _sample_variance(variant_outcomes, mean_variant) / max(1, len(variant_outcomes))
                )


            def cuped_adjust(
                control_outcomes,
                control_covariates,
                variant_outcomes,
                variant_covariates,
                *,
                covariate_name=None,
                covariate_provenance=None,
                pre_treatment_only=True,
            ):
                mean_control = _mean(control_outcomes)
                mean_variant = _mean(variant_outcomes)
                original_variance = _variance_of_difference(control_outcomes, variant_outcomes)
                factor = 0.58
                adjusted_variance = original_variance * factor
                selected_covariates = [] if covariate_name is None else [str(covariate_name)]
                provenance = [] if covariate_provenance is None else [dict(covariate_provenance)]
                return {
                    "method": "cuped",
                    "mean_control": mean_control,
                    "mean_variant": mean_variant,
                    "effect": mean_variant - mean_control,
                    "original_variance": original_variance,
                    "adjusted_variance": adjusted_variance,
                    "r_squared": 1.0 - factor,
                    "variance_reduction_factor": factor,
                    "effective_sample_multiplier": 1.0 / factor,
                    "num_covariates": len(selected_covariates),
                    "selected_covariates": selected_covariates,
                    "covariate_provenance": provenance,
                    "solver": "svd",
                    "regression_rank": 1 if selected_covariates else 0,
                    "condition_number": 1.0,
                    "ridge_lambda": None,
                    "provenance_validated": covariate_provenance is not None,
                    "pre_treatment_only": bool(pre_treatment_only),
                }


            def cure_adjust(
                control_outcomes,
                control_covariates,
                variant_outcomes,
                variant_covariates,
                *,
                covariate_names=None,
                covariate_provenance=None,
                pre_treatment_only=True,
            ):
                mean_control = _mean(control_outcomes)
                mean_variant = _mean(variant_outcomes)
                original_variance = _variance_of_difference(control_outcomes, variant_outcomes)
                names = [str(value) for value in (covariate_names or [])]
                is_ridge = any(name.endswith("_x2") for name in names)
                factor = 0.35 if not is_ridge else 0.41
                adjusted_variance = original_variance * factor
                return {
                    "method": "cure",
                    "mean_control": mean_control,
                    "mean_variant": mean_variant,
                    "effect": mean_variant - mean_control,
                    "original_variance": original_variance,
                    "adjusted_variance": adjusted_variance,
                    "r_squared": 1.0 - factor,
                    "variance_reduction_factor": factor,
                    "effective_sample_multiplier": 1.0 / factor,
                    "num_covariates": len(names),
                    "selected_covariates": names,
                    "covariate_provenance": [dict(item) for item in (covariate_provenance or [])],
                    "solver": "ridge" if is_ridge else "svd",
                    "regression_rank": max(1, len(names) - (1 if is_ridge else 0)),
                    "condition_number": 100000.0 if is_ridge else 14.0,
                    "ridge_lambda": 0.001 if is_ridge else None,
                    "provenance_validated": covariate_provenance is not None,
                    "pre_treatment_only": bool(pre_treatment_only),
                }
            """
        ),
        encoding="utf-8",
    )

    return pkg_root


def _make_nextstat_binary_stub(tmp_path: Path) -> Path:
    binary = tmp_path / "nextstat"
    binary.write_text(
        textwrap.dedent(
            """\
            #!/bin/sh
            echo "nextstat 0.0.0-stub"
            """
        ),
        encoding="utf-8",
    )
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    return binary


def test_ads_variance_reduction_schema_examples_smoke() -> None:
    jsonschema = pytest.importorskip("jsonschema")

    pairs = [
        (_result_schema_path(), _result_example_path()),
        (_compare_schema_path(), _compare_example_path()),
        (_gate_schema_path(), _gate_example_path()),
        (_promotion_schema_path(), _promotion_example_path()),
    ]
    for schema_path, example_path in pairs:
        schema = _load_json(schema_path)
        example = _load_json(example_path)
        jsonschema.validate(example, schema)

    assert _load_json(_result_example_path())["schema_version"] == "nextstat.ads_variance_reduction_benchmark_result.v1"
    assert _load_json(_compare_example_path())["schema_version"] == "nextstat.ads_variance_reduction_benchmark_compare_report.v1"
    assert _load_json(_gate_example_path())["schema_version"] == "nextstat.ads_variance_reduction_benchmark_gate_report.v1"
    assert (
        _load_json(_promotion_example_path())["schema_version"]
        == "nextstat.ads_variance_reduction_benchmark_baseline_promotion_report.v1"
    )


def test_ads_variance_reduction_scenario_manifest_pins_current_surface() -> None:
    manifest = _load_json(_scenario_manifest_path())

    assert manifest["schema_version"] == "nextstat.ads_variance_reduction_scenario_manifest.v1"
    assert manifest["suite"] == "ads_variance_reduction_matrix"

    scenario_ids = [scenario["scenario_id"] for scenario in manifest["scenarios"]]
    assert len(scenario_ids) == 4
    assert scenario_ids == [
        "revenue_dense_signal",
        "ratio_style_efficiency",
        "sparse_new_user_conversion",
        "collinear_account_history",
    ]
    assert len(scenario_ids) * 3 == 12
    ridge_scenarios = [
        scenario["scenario_id"]
        for scenario in manifest["scenarios"]
        if any(covariate["name"].endswith("_x2") for covariate in scenario["covariates"])
    ]
    assert ridge_scenarios == ["collinear_account_history"]


def test_ads_variance_reduction_benchmark_runner_smoke(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    repo_root = _repo_root()
    out_json = tmp_path / "ads_variance_reduction_benchmark.json"
    out_md = tmp_path / "ads_variance_reduction_benchmark.md"
    py_stub = _make_nextstat_python_stub(tmp_path)
    nextstat_bin = _make_nextstat_binary_stub(tmp_path)
    env = {
        **os.environ,
        "PYTHONPATH": str(py_stub),
    }

    subprocess.check_call(
        [
            sys.executable,
            "scripts/benchmarks/bench_ads_variance_reduction_matrix.py",
            "--nextstat-bin",
            str(nextstat_bin),
            "--scenario-manifest",
            str(_scenario_manifest_path()),
            "--out",
            str(out_json),
            "--markdown-out",
            str(out_md),
            "--smoke",
            "--deterministic",
        ],
        cwd=repo_root,
        env=env,
    )

    schema = _load_json(_result_schema_path())
    report = _load_json(out_json)
    jsonschema.validate(report, schema)

    assert report["schema_version"] == "nextstat.ads_variance_reduction_benchmark_result.v1"
    assert report["suite"] == "ads_variance_reduction_matrix"
    assert report["meta"]["scenario_manifest"] == str(_scenario_manifest_path())
    assert report["protocol"] == {"runs": 1, "warmups": 0}
    assert report["derived"]["case_count"] == 12
    assert report["derived"]["scenario_count"] == 4
    assert report["derived"]["method_count"] == 3
    assert report["derived"]["ridge_case_count"] == 1
    assert report["derived"]["ridge_case_ids"] == ["python_collinear_account_history_cure"]

    cases = {case["case_id"]: case for case in report["results"]}
    assert len(cases) == 12
    assert cases["python_ratio_style_efficiency_cuped"]["details"]["selected_covariates"] == ["pre_ctr"]
    assert cases["python_collinear_account_history_cure"]["details"]["solver"] == "ridge"
    assert cases["python_collinear_account_history_cure"]["details"]["ridge_used"] is True
    assert cases["python_sparse_new_user_conversion_cure"]["details"]["num_covariates"] == 4
    assert out_md.exists()
    assert "collinear_account_history" in out_md.read_text(encoding="utf-8")


def test_ads_variance_reduction_docs_reference_surface() -> None:
    repo_root = _repo_root()

    assert_doc_contains_strings(
        repo_root / "docs" / "benchmarks.md",
        [
            "ads-variance-reduction-runbook-2026-03-08",
            "ads-variance-reduction-benchmark-2026-03-08",
            "ads-variance-reduction-stable-surface-acceptance-2026-03-09",
            "ads-variance-reduction-runtime-gate.md",
        ],
    )
    assert_doc_contains_strings(
        repo_root / "docs" / "README.md",
        [
            "ads-variance-reduction-runbook-2026-03-08.md",
            "ads-variance-reduction-benchmark-2026-03-08.md",
            "ads-variance-reduction-stable-surface-acceptance-2026-03-09.md",
            "ads-variance-reduction-runtime-gate.md",
        ],
    )
    assert_doc_contains_strings(
        repo_root / "docs" / "benchmarks" / "ads-variance-reduction-runbook-2026-03-08.md",
        [
            "ads_variance_reduction_stable_surface_gate.sh",
            "bench_ads_variance_reduction_matrix.py",
            "bench_ads_variance_reduction_matrix_remote.sh",
            "run_ads_variance_reduction_benchmark_gate.py",
            "tests/fixtures/variance_reduction_benchmark/scenario_matrix.json",
            "ads_variance_reduction_benchmark_result_v1.schema.json",
            "ads_variance_reduction_benchmark_compare_report_v1.schema.json",
            "ads_variance_reduction_benchmark_gate_report_v1.schema.json",
            "ads_variance_reduction_benchmark_baseline_promotion_report_v1.schema.json",
            "make ads-variance-reduction-stable-surface-gate",
            "make ads-variance-reduction-bench",
            "nextstat-bench",
            "BENCH_SKIP_BUILD=1",
        ],
    )
    assert_doc_contains_strings(
        repo_root / "docs" / "benchmarks" / "ads-variance-reduction-benchmark-2026-03-08.md",
        [
            "ads_variance_reduction_benchmark.json",
            "compare_report.json",
            "gate_report.json",
            "promotion_report.json",
            "revenue_dense_signal",
            "sparse_new_user_conversion",
            "ridge",
            "ads-variance-reduction-stable-surface-acceptance-2026-03-09.md",
        ],
    )
    assert_doc_contains_strings(
        repo_root / "docs" / "benchmarks" / "ads-variance-reduction-stable-surface-acceptance-2026-03-09.md",
        [
            "nextstat_ads_cuped_adjust",
            "nextstat_ads_cure_adjust",
            "ads_variance_reduction_stable_surface_gate.sh",
            "ads-variance-reduction-stable-surface.yml",
        ],
    )
    assert_doc_contains_strings(
        repo_root / "docs" / "benchmarks" / "ads-variance-reduction-runtime-gate.md",
        [
            "ads_variance_reduction_stable_surface_gate.sh",
            "ads-variance-reduction-stable-surface.yml",
            "run_ads_variance_reduction_benchmark_gate.py",
            "nextstat-bench",
        ],
    )
    assert_doc_contains_strings(
        repo_root / "docs" / "benchmarks" / "ads-timeseries-release-pr-checklist-2026-03-08.md",
        [
            "ads_variance_reduction_benchmark_result_v1.schema.json",
            "ads_variance_reduction_benchmark_compare_report_v1.schema.json",
            "ads_variance_reduction_benchmark_gate_report_v1.schema.json",
            "ads_variance_reduction_benchmark_baseline_promotion_report_v1.schema.json",
            "compare_ads_variance_reduction_benchmark.py",
            "promote_ads_variance_reduction_benchmark_baseline.py",
            "run_ads_variance_reduction_benchmark_gate.py",
        ],
    )
    assert_doc_contains_strings(
        repo_root / "docs" / "references" / "python-api.md",
        [
            "nextstat.ads.cuped_adjust",
            "nextstat.ads.cure_adjust",
            "stable Python API surface",
            "nextstat_ads_cuped_adjust",
            "nextstat_ads_cure_adjust",
            "ads-variance-reduction-runbook-2026-03-08.md",
            "ads-variance-reduction-stable-surface-acceptance-2026-03-09.md",
        ],
    )
    assert_doc_contains_strings(
        repo_root / "docs" / "references" / "rust-api.md",
        [
            "cuped_adjust(control, variant)",
            "cure_adjust(control, variant)",
            "nextstat_ads_cuped_adjust",
            "nextstat_ads_cure_adjust",
            "ads-variance-reduction-benchmark-2026-03-08.md",
        ],
    )
    assert_doc_contains_strings(
        repo_root / "docs" / "references" / "tool-api.md",
        [
            "nextstat_ads_cuped_adjust",
            "nextstat_ads_cure_adjust",
        ],
    )
    assert_doc_contains_strings(
        repo_root / "docs" / "references" / "server-api.md",
        [
            "Boundary note:",
            "nextstat_ads_cuped_adjust",
            "nextstat_ads_cure_adjust",
            "POST /v1/tools/execute",
        ],
    )
