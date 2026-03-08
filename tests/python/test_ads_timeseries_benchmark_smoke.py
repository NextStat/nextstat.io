from __future__ import annotations

import json
import math
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


def _accepted_baseline_path() -> Path:
    return _repo_root() / "benchmarks" / "artifacts" / "ads_timeseries_baselines" / "nextstat-bench" / "accepted.json"


def _comparison_schema_path() -> Path:
    return (
        _repo_root()
        / "docs"
        / "schemas"
        / "benchmarks"
        / "ads_timeseries_benchmark_compare_report_v1.schema.json"
    )


def _result_schema_path() -> Path:
    return (
        _repo_root()
        / "docs"
        / "schemas"
        / "benchmarks"
        / "ads_timeseries_benchmark_result_v1.schema.json"
    )


def _promotion_schema_path() -> Path:
    return (
        _repo_root()
        / "docs"
        / "schemas"
        / "benchmarks"
        / "ads_timeseries_benchmark_baseline_promotion_report_v1.schema.json"
    )


def _gate_schema_path() -> Path:
    return (
        _repo_root()
        / "docs"
        / "schemas"
        / "benchmarks"
        / "ads_timeseries_benchmark_gate_report_v1.schema.json"
    )


def _write_executable(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _make_nextstat_cli_stub(tmp_path: Path) -> Path:
    stub_path = tmp_path / "nextstat_stub.py"
    _write_executable(
        stub_path,
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

args = sys.argv[1:]

if args == ["--version"]:
    print("nextstat 0.0.0-stub")
    raise SystemExit(0)

if args[:3] != ["timeseries", "kalman-filter", "--input"]:
    raise SystemExit(f"unexpected nextstat args: {args}")

input_path = Path(args[3])
output_path = Path(args[args.index("--output") + 1])
is_linear = "linear" in input_path.name
state_dim = 8 if is_linear else 7
log_likelihood = -26.9 if is_linear else -27.3

payload = {
    "log_likelihood": log_likelihood,
    "filtered_means": [[0.0] * state_dim for _ in range(8)],
    "filtered_covs": [[[1.0 if i == j else 0.0 for j in range(state_dim)] for i in range(state_dim)] for _ in range(8)],
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
""",
    )
    return stub_path


def _make_nextstat_python_stub(tmp_path: Path) -> Path:
    pkg_root = tmp_path / "py_stub"
    module_root = pkg_root / "nextstat"
    module_root.mkdir(parents=True, exist_ok=True)

    (module_root / "__init__.py").write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            __version__ = "0.0.0-stub"

            from . import ads, timeseries


            def set_threads(_: int) -> None:
                return None


            def set_eval_mode(_: str) -> None:
                return None


            def kalman_filter(model, ys):
                state_dim = int(model["state_dim"])
                return {
                    "log_likelihood": -27.0 + 0.1 * (state_dim - 7),
                    "filtered_means": [[0.0] * state_dim for _ in ys],
                    "filtered_covs": [
                        [[1.0 if i == j else 0.0 for j in range(state_dim)] for i in range(state_dim)]
                        for _ in ys
                    ],
                }
            """
        ),
        encoding="utf-8",
    )

    (module_root / "ads.py").write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            import math


            class BetaBinomialModel:
                def __init__(self, alpha: float, beta: float) -> None:
                    self.alpha = float(alpha)
                    self.beta = float(beta)

                @staticmethod
                def fit_from_counts(successes, trials):
                    total_success = float(sum(successes))
                    total_trials = float(sum(trials))
                    concentration = max(total_trials / 5.0, 2.0)
                    mean = total_success / total_trials
                    return BetaBinomialModel(mean * concentration, (1.0 - mean) * concentration)

                def mean(self) -> float:
                    return self.alpha / (self.alpha + self.beta)

                def overdispersion(self) -> float:
                    return 1.0 / (self.alpha + self.beta + 1.0)

                def posterior(self, successes: int, trials: int):
                    return BetaBinomialModel(self.alpha + successes, self.beta + trials - successes)


            class DelayCorrectionModel:
                def __init__(self, lambda_: float, lambda_se: float | None = None) -> None:
                    self.lambda_ = float(lambda_)
                    self.lambda_se = None if lambda_se is None else float(lambda_se)

                @staticmethod
                def fit_from_lag_buckets(_):
                    return DelayCorrectionModel(0.47, 0.02)

                def observed_fraction(self, window_days: float) -> float:
                    if window_days <= 0.0:
                        return 0.0
                    return 1.0 - math.exp(-self.lambda_ * window_days)

                def correct(self, observed_count: float, window_days: float):
                    fraction = self.observed_fraction(window_days)
                    return observed_count / fraction, observed_count * 0.1


            def hill(x: float, ec: float, slope: float) -> float:
                if x == 0.0:
                    return 0.0
                return 1.0 / (1.0 + (x / ec) ** (-slope))


            def adstock_geometric(spend, decay: float):
                if not spend:
                    return []
                out = [float(spend[0])]
                for value in spend[1:]:
                    out.append(float(value) + float(decay) * out[-1])
                return out


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
                if not pre_treatment_only:
                    raise ValueError("CUPED/CURE covariates must be pre-treatment only")
                return {
                    "method": "cuped",
                    "solver": "svd",
                    "num_covariates": 1,
                    "r_squared": 0.36,
                    "variance_reduction_factor": 0.64,
                    "effective_sample_multiplier": 1.5625,
                    "effect": 1.0,
                    "selected_covariates": [] if covariate_name is None else [str(covariate_name)],
                    "covariate_provenance": [] if covariate_provenance is None else [dict(covariate_provenance)],
                    "provenance_validated": covariate_provenance is not None,
                    "condition_number": 1.0,
                    "ridge_lambda": None,
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
                if not pre_treatment_only:
                    raise ValueError("CUPED/CURE covariates must be pre-treatment only")
                return {
                    "method": "cure",
                    "solver": "ridge",
                    "num_covariates": 2,
                    "r_squared": 0.44,
                    "variance_reduction_factor": 0.56,
                    "effective_sample_multiplier": 1.78571429,
                    "effect": 1.0,
                    "selected_covariates": [] if covariate_names is None else [str(v) for v in covariate_names],
                    "covariate_provenance": [] if covariate_provenance is None else [dict(v) for v in covariate_provenance],
                    "provenance_validated": covariate_provenance is not None,
                    "condition_number": 1.0,
                    "ridge_lambda": 0.0001756,
                    "pre_treatment_only": bool(pre_treatment_only),
                }
            """
        ),
        encoding="utf-8",
    )

    (module_root / "timeseries.py").write_text(
        textwrap.dedent(
            """
            from __future__ import annotations


            def local_level_weekly_model(*, q_level, q_weekly, r, level0=0.0, p0_level=1.0, p0_weekly=1.0):
                return {"kind": "local_level_weekly", "state_dim": 7}


            def local_linear_trend_weekly_model(
                *,
                q_level,
                q_slope,
                q_weekly,
                r,
                level0=0.0,
                slope0=0.0,
                p0_level=1.0,
                p0_slope=1.0,
                p0_weekly=1.0,
            ):
                return {"kind": "local_linear_trend_weekly", "state_dim": 8}
            """
        ),
        encoding="utf-8",
    )

    return pkg_root


def test_ads_timeseries_benchmark_schema_and_example_smoke() -> None:
    jsonschema = pytest.importorskip("jsonschema")

    schema_path = (
        _repo_root() / "docs" / "schemas" / "benchmarks" / "ads_timeseries_benchmark_result_v1.schema.json"
    )
    example_path = _repo_root() / "docs" / "specs" / "benchmarks" / "ads_timeseries_benchmark_result_v1.example.json"

    schema_doc = _load_json(schema_path)
    example_doc = _load_json(example_path)

    assert schema_doc["$id"] == (
        "https://nextstat.io/schemas/benchmarks/ads_timeseries_benchmark_result_v1.schema.json"
    )
    jsonschema.Draft202012Validator.check_schema(schema_doc)
    jsonschema.validate(example_doc, schema_doc)

    assert example_doc["schema_version"] == "nextstat.ads_timeseries_benchmark_result.v1"
    assert example_doc["suite"] == "ads_timeseries_surface"
    assert example_doc["meta"]["host_policy"] == "nextstat-bench"
    assert example_doc["protocol"]["runs"] == 5
    assert example_doc["protocol"]["warmups"] == 1
    assert len(example_doc["results"]) == 9


def test_ads_timeseries_benchmark_accepted_baseline_and_compare_example_smoke() -> None:
    jsonschema = pytest.importorskip("jsonschema")

    repo_root = _repo_root()
    result_schema = _load_json(
        repo_root / "docs" / "schemas" / "benchmarks" / "ads_timeseries_benchmark_result_v1.schema.json"
    )
    compare_schema = _load_json(_comparison_schema_path())
    compare_example = _load_json(
        repo_root / "docs" / "specs" / "benchmarks" / "ads_timeseries_benchmark_compare_report_v1.example.json"
    )
    accepted_baseline = _load_json(_accepted_baseline_path())

    jsonschema.Draft202012Validator.check_schema(compare_schema)
    jsonschema.validate(accepted_baseline, result_schema)
    jsonschema.validate(compare_example, compare_schema)

    assert accepted_baseline["schema_version"] == "nextstat.ads_timeseries_benchmark_result.v1"
    assert accepted_baseline["meta"]["host_policy"] == "nextstat-bench"
    assert accepted_baseline["host"]["hostname"] == "nextstat-bench"
    assert accepted_baseline["binary"]["build_profile"] == "release"
    assert accepted_baseline["protocol"]["runs"] == 5
    assert accepted_baseline["protocol"]["warmups"] == 1
    assert compare_example["schema_version"] == "nextstat.ads_timeseries_benchmark_compare_report.v1"
    assert compare_example["status"] == "passed"
    assert compare_example["policy"]["required_hostname"] == "nextstat-bench"
    assert compare_example["summary"]["failed_cases"] == 0


def test_ads_timeseries_benchmark_promotion_schema_and_example_smoke() -> None:
    jsonschema = pytest.importorskip("jsonschema")

    promotion_schema = _load_json(_promotion_schema_path())
    promotion_example = _load_json(
        _repo_root() / "docs" / "specs" / "benchmarks" / "ads_timeseries_benchmark_baseline_promotion_report_v1.example.json"
    )

    jsonschema.Draft202012Validator.check_schema(promotion_schema)
    jsonschema.validate(promotion_example, promotion_schema)

    assert promotion_example["schema_version"] == (
        "nextstat.ads_timeseries_benchmark_baseline_promotion_report.v1"
    )
    assert promotion_example["status"] == "dry_run"
    assert promotion_example["promoted"] is False
    assert promotion_example["dry_run"] is True
    assert promotion_example["compare_status"] == "passed"


def test_ads_timeseries_benchmark_gate_schema_and_example_smoke() -> None:
    jsonschema = pytest.importorskip("jsonschema")

    gate_schema = _load_json(_gate_schema_path())
    gate_example = _load_json(
        _repo_root() / "docs" / "specs" / "benchmarks" / "ads_timeseries_benchmark_gate_report_v1.example.json"
    )

    jsonschema.Draft202012Validator.check_schema(gate_schema)
    jsonschema.validate(gate_example, gate_schema)

    assert gate_example["schema_version"] == "nextstat.ads_timeseries_benchmark_gate_report.v1"
    assert gate_example["status"] == "passed"
    assert gate_example["promotion_mode"] == "dry_run"
    assert gate_example["steps"]["benchmark"]["mode"] == "provided_artifact"
    assert gate_example["steps"]["compare"]["compare_status"] == "passed"
    assert gate_example["steps"]["promotion"]["promotion_status"] == "dry_run"


def test_ads_timeseries_benchmark_runner_smoke(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    repo_root = _repo_root()
    nextstat_cli_stub = _make_nextstat_cli_stub(tmp_path)
    nextstat_py_stub = _make_nextstat_python_stub(tmp_path)
    out_path = tmp_path / "summary.json"
    markdown_path = tmp_path / "summary.md"
    work_root = tmp_path / "work"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(nextstat_py_stub)

    subprocess.check_call(
        [
            sys.executable,
            "scripts/benchmarks/bench_ads_timeseries_surface.py",
            "--nextstat-bin",
            str(nextstat_cli_stub),
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
        env=env,
    )

    schema_doc = _load_json(_result_schema_path())
    report = _load_json(out_path)
    jsonschema.validate(report, schema_doc)

    assert report["schema_version"] == "nextstat.ads_timeseries_benchmark_result.v1"
    assert report["suite"] == "ads_timeseries_surface"
    assert report["meta"]["smoke"] is True
    assert report["meta"]["deterministic"] is True
    assert report["meta"]["host_policy"] == "nextstat-bench"
    assert report["protocol"]["runs"] == 1
    assert report["protocol"]["warmups"] == 0
    assert report["binary"]["version"] == "nextstat 0.0.0-stub"
    assert report["binary"]["sha256"]
    assert report["python"]["nextstat_version"] == "0.0.0-stub"
    assert report["derived"]["all_cases_ok"] is True
    assert report["derived"]["case_count"] == 9
    cases = {case["case_id"]: case for case in report["results"]}
    assert set(cases) == {
        "python_beta_binomial_fit_from_counts",
        "python_delay_correction_fit_from_lag_buckets",
        "python_cuped_adjust",
        "python_cure_adjust",
        "python_response_curve_helpers",
        "python_kalman_local_level_weekly_filter",
        "python_kalman_local_linear_trend_weekly_filter",
        "cli_kalman_local_level_weekly_filter",
        "cli_kalman_local_linear_trend_weekly_filter",
    }
    assert all(case["status"] == "ok" for case in cases.values())
    assert cases["python_cuped_adjust"]["details"]["method"] == "cuped"
    assert cases["python_cuped_adjust"]["details"]["num_covariates"] == 1
    assert cases["python_cuped_adjust"]["details"]["selected_covariates"] == ["pre_clicks"]
    assert cases["python_cuped_adjust"]["details"]["provenance_validated"] is True
    assert "variance_reduction_factor" in cases["python_cuped_adjust"]["details"]
    assert cases["python_cure_adjust"]["details"]["solver"] == "ridge"
    assert cases["python_cure_adjust"]["details"]["num_covariates"] == 2
    assert cases["python_cure_adjust"]["details"]["selected_covariates"] == [
        "pre_clicks",
        "pre_impressions",
    ]
    assert cases["python_cure_adjust"]["details"]["provenance_validated"] is True
    assert "effective_sample_multiplier" in cases["python_cure_adjust"]["details"]
    assert cases["python_kalman_local_level_weekly_filter"]["details"]["state_dim"] == 7
    assert cases["cli_kalman_local_linear_trend_weekly_filter"]["details"]["state_dim"] == 8

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# Ads + Time Series Stable Surface Benchmark" in markdown
    assert "- runs: `1`" in markdown
    assert "- warmups: `0`" in markdown
    assert "cli_kalman_local_linear_trend_weekly_filter" in markdown


def test_ads_timeseries_benchmark_schemas_pin_current_nine_case_surface() -> None:
    result_schema = _load_json(_result_schema_path())
    result_properties = result_schema["properties"]
    result_items = result_properties["results"]["items"]

    assert result_properties["results"]["minItems"] == 9
    assert result_properties["results"]["maxItems"] == 9
    assert result_properties["derived"]["properties"]["case_count"]["const"] == 9
    assert result_properties["derived"]["properties"]["python_case_count"]["const"] == 7
    assert result_properties["derived"]["properties"]["cli_case_count"]["const"] == 2
    assert "python_cuped_adjust" in result_items["properties"]["case_id"]["enum"]
    assert "python_cure_adjust" in result_items["properties"]["case_id"]["enum"]

    cuped_rule = next(
        rule
        for rule in result_items["allOf"]
        if rule["if"]["properties"]["case_id"]["const"] == "python_cuped_adjust"
    )
    cure_rule = next(
        rule
        for rule in result_items["allOf"]
        if rule["if"]["properties"]["case_id"]["const"] == "python_cure_adjust"
    )
    assert cuped_rule["then"]["properties"]["details"]["properties"]["method"]["const"] == "cuped"
    assert cuped_rule["then"]["properties"]["details"]["properties"]["solver"]["const"] == "svd"
    assert cure_rule["then"]["properties"]["details"]["properties"]["method"]["const"] == "cure"
    assert cure_rule["then"]["properties"]["details"]["properties"]["solver"]["const"] == "ridge"

    compare_schema = _load_json(_comparison_schema_path())
    compare_policy = compare_schema["properties"]["policy"]["properties"]["case_ids"]
    compare_cases = compare_schema["properties"]["cases"]
    assert compare_policy["minItems"] == 9
    assert compare_policy["maxItems"] == 9
    assert compare_cases["minItems"] == 9
    assert compare_cases["maxItems"] == 9


def test_ads_timeseries_benchmark_compare_runner_passes_on_accepted_baseline(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    repo_root = _repo_root()
    out_path = tmp_path / "compare.json"
    accepted_baseline = _accepted_baseline_path()

    subprocess.check_call(
        [
            sys.executable,
            "scripts/benchmarks/compare_ads_timeseries_benchmark.py",
            "--baseline",
            str(accepted_baseline),
            "--current",
            str(accepted_baseline),
            "--out",
            str(out_path),
        ],
        cwd=repo_root,
    )

    compare_schema = _load_json(_comparison_schema_path())
    report = _load_json(out_path)
    jsonschema.validate(report, compare_schema)

    assert report["status"] == "passed"
    assert report["ok"] is True
    assert report["requires_review"] is False
    assert report["summary"]["failed_cases"] == 0
    assert report["summary"]["review_cases"] == 0
    assert report["environment_checks"]["hostname"]["matches"] is True
    assert report["environment_checks"]["release_build"]["matches"] is True
    cases = {case["id"]: case for case in report["cases"]}
    assert all(case["status"] == "passed" for case in cases.values())


def test_ads_timeseries_benchmark_compare_runner_reports_review_and_fail(tmp_path: Path) -> None:
    repo_root = _repo_root()
    review_current = tmp_path / "review_current.json"
    fail_current = tmp_path / "fail_current.json"
    review_report = tmp_path / "review_report.json"
    fail_report = tmp_path / "fail_report.json"
    accepted_baseline = _accepted_baseline_path()
    baseline_doc = _load_json(accepted_baseline)

    review_doc = json.loads(json.dumps(baseline_doc))
    for case in review_doc["results"]:
        if case["case_id"] == "cli_kalman_local_level_weekly_filter":
            case["median_s"] = round(case["median_s"] * 1.4, 6)
            case["min_s"] = round(case["min_s"] * 1.4, 6)
            case["max_s"] = round(case["max_s"] * 1.4, 6)
            case["samples_s"] = [round(value * 1.4, 6) for value in case["samples_s"]]
            break
    review_current.write_text(json.dumps(review_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    review = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/compare_ads_timeseries_benchmark.py",
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
    review_case = next(case for case in review_doc["cases"] if case["id"] == "cli_kalman_local_level_weekly_filter")
    assert review_case["status"] == "review"

    fail_doc = json.loads(json.dumps(baseline_doc))
    for case in fail_doc["results"]:
        if case["case_id"] == "cli_kalman_local_linear_trend_weekly_filter":
            case["median_s"] = round(case["median_s"] * 1.8, 6)
            case["min_s"] = round(case["min_s"] * 1.8, 6)
            case["max_s"] = round(case["max_s"] * 1.8, 6)
            case["samples_s"] = [round(value * 1.8, 6) for value in case["samples_s"]]
            break
    fail_current.write_text(json.dumps(fail_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    failed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/compare_ads_timeseries_benchmark.py",
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
    fail_case = next(case for case in fail_report_doc["cases"] if case["id"] == "cli_kalman_local_linear_trend_weekly_filter")
    assert fail_case["status"] == "failed"


def test_ads_timeseries_benchmark_compare_runner_reports_review_on_details_change(tmp_path: Path) -> None:
    repo_root = _repo_root()
    current_path = tmp_path / "details_current.json"
    out_path = tmp_path / "details_review.json"
    accepted_baseline = _accepted_baseline_path()
    baseline_doc = _load_json(accepted_baseline)

    current_doc = json.loads(json.dumps(baseline_doc))
    for case in current_doc["results"]:
        if case["case_id"] == "python_cure_adjust":
            case["details"]["selected_covariates"] = [
                "pre_clicks",
                "pre_impressions",
                "pre_spend",
            ]
            break
    current_path.write_text(json.dumps(current_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    review = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/compare_ads_timeseries_benchmark.py",
            "--baseline",
            str(accepted_baseline),
            "--current",
            str(current_path),
            "--out",
            str(out_path),
        ],
        cwd=repo_root,
        check=False,
    )

    assert review.returncode == 0
    review_doc = _load_json(out_path)
    assert review_doc["status"] == "review"
    assert review_doc["requires_review"] is True
    review_case = next(case for case in review_doc["cases"] if case["id"] == "python_cure_adjust")
    assert review_case["status"] == "review"
    assert "details_changed" in review_case["warnings"]


def test_ads_timeseries_benchmark_promote_runner_dry_run_and_promote(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    repo_root = _repo_root()
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
    current_doc["meta"]["git_commit"] = "1111111111111111111111111111111111111111"
    current_path.write_text(json.dumps(current_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    subprocess.check_call(
        [
            sys.executable,
            "scripts/benchmarks/promote_ads_timeseries_benchmark_baseline.py",
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
        cwd=repo_root,
    )

    promotion_schema = _load_json(_promotion_schema_path())
    dry_run_doc = _load_json(dry_run_report)
    jsonschema.validate(dry_run_doc, promotion_schema)
    assert dry_run_doc["status"] == "dry_run"
    assert dry_run_doc["promoted"] is False
    assert dry_run_doc["actions"]["accepted_updated"] is False
    assert _load_json(accepted_path)["meta"]["git_commit"] == baseline_doc["meta"]["git_commit"]

    subprocess.check_call(
        [
            sys.executable,
            "scripts/benchmarks/promote_ads_timeseries_benchmark_baseline.py",
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
        cwd=repo_root,
    )

    promoted_doc = _load_json(promote_report)
    jsonschema.validate(promoted_doc, promotion_schema)
    assert promoted_doc["status"] == "promoted"
    assert promoted_doc["promoted"] is True
    assert promoted_doc["actions"]["accepted_updated"] is True
    assert promoted_doc["actions"]["archived_previous_baseline"] is True
    assert promoted_doc["actions"]["archived_promoted_snapshot"] is True
    assert _load_json(accepted_path)["meta"]["git_commit"] == "1111111111111111111111111111111111111111"
    previous_path = Path(promoted_doc["actions"]["archived_previous_baseline_path"])
    promoted_path = Path(promoted_doc["actions"]["archived_promoted_snapshot_path"])
    assert previous_path.exists()
    assert promoted_path.exists()


def test_ads_timeseries_benchmark_promote_runner_blocks_review_without_override(tmp_path: Path) -> None:
    repo_root = _repo_root()
    accepted_path = tmp_path / "accepted.json"
    current_path = tmp_path / "review_current.json"
    compare_report = tmp_path / "compare_report.json"
    promote_report = tmp_path / "promote_report.json"

    baseline_doc = _load_json(_accepted_baseline_path())
    accepted_path.write_text(json.dumps(baseline_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    review_doc = json.loads(json.dumps(baseline_doc))
    for case in review_doc["results"]:
        if case["case_id"] == "cli_kalman_local_level_weekly_filter":
            case["median_s"] = round(case["median_s"] * 1.4, 6)
            case["min_s"] = round(case["min_s"] * 1.4, 6)
            case["max_s"] = round(case["max_s"] * 1.4, 6)
            case["samples_s"] = [round(value * 1.4, 6) for value in case["samples_s"]]
            break
    current_path.write_text(json.dumps(review_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    blocked = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/promote_ads_timeseries_benchmark_baseline.py",
            "--accepted",
            str(accepted_path),
            "--current",
            str(current_path),
            "--compare-report",
            str(compare_report),
            "--report",
            str(promote_report),
        ],
        cwd=repo_root,
        check=False,
    )
    assert blocked.returncode == 2
    blocked_doc = _load_json(promote_report)
    assert blocked_doc["status"] == "failed"
    assert "compare_status_review_requires_allow_review" in blocked_doc["summary"]["top_level_errors"]

    allowed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/promote_ads_timeseries_benchmark_baseline.py",
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
        cwd=repo_root,
        check=False,
    )
    assert allowed.returncode == 0
    allowed_doc = _load_json(promote_report)
    assert allowed_doc["status"] == "dry_run"
    assert allowed_doc["compare_status"] == "review"
    assert allowed_doc["allow_review"] is True


def test_ads_timeseries_benchmark_promote_runner_allows_explicit_case_set_widening(tmp_path: Path) -> None:
    repo_root = _repo_root()
    accepted_path = tmp_path / "accepted.json"
    current_path = tmp_path / "current.json"
    compare_report = tmp_path / "compare_report.json"
    promote_report = tmp_path / "promote_report.json"

    baseline_doc = _load_json(_accepted_baseline_path())
    removed_case_ids = {"cli_kalman_local_level_weekly_filter", "cli_kalman_local_linear_trend_weekly_filter"}
    accepted_doc = json.loads(json.dumps(baseline_doc))
    accepted_doc["results"] = [
        case for case in accepted_doc["results"] if case["case_id"] not in removed_case_ids
    ]
    accepted_doc["derived"]["case_count"] = len(accepted_doc["results"])
    accepted_doc["derived"]["python_case_count"] = sum(
        case["surface"] == "python" for case in accepted_doc["results"]
    )
    accepted_doc["derived"]["cli_case_count"] = sum(
        case["surface"] == "cli" for case in accepted_doc["results"]
    )
    accepted_path.write_text(json.dumps(accepted_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    current_path.write_text(json.dumps(baseline_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    blocked = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/promote_ads_timeseries_benchmark_baseline.py",
            "--accepted",
            str(accepted_path),
            "--current",
            str(current_path),
            "--compare-report",
            str(compare_report),
            "--report",
            str(promote_report),
            "--dry-run",
        ],
        cwd=repo_root,
        check=False,
    )
    assert blocked.returncode == 2
    blocked_doc = _load_json(promote_report)
    assert blocked_doc["status"] == "failed"
    assert "compare_status_failed" in blocked_doc["summary"]["top_level_errors"]

    allowed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmarks/promote_ads_timeseries_benchmark_baseline.py",
            "--accepted",
            str(accepted_path),
            "--current",
            str(current_path),
            "--compare-report",
            str(compare_report),
            "--report",
            str(promote_report),
            "--allow-case-set-change",
            "--dry-run",
        ],
        cwd=repo_root,
        check=False,
    )
    assert allowed.returncode == 0
    allowed_doc = _load_json(promote_report)
    assert allowed_doc["status"] == "dry_run"
    assert allowed_doc["compare_status"] == "failed"


def test_ads_timeseries_benchmark_gate_runner_with_provided_artifact(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    repo_root = _repo_root()
    compare_report = tmp_path / "compare_report.json"
    gate_report = tmp_path / "gate_report.json"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/benchmarks/run_ads_timeseries_benchmark_gate.py",
            "--current",
            str(_accepted_baseline_path()),
            "--compare-report",
            str(compare_report),
            "--report",
            str(gate_report),
            "--promotion-mode",
            "none",
        ],
        cwd=repo_root,
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


def test_ads_timeseries_benchmark_docs_reference_surface() -> None:
    repo_root = _repo_root()

    assert_doc_contains_strings(
        repo_root / "docs" / "benchmarks.md",
        [
            "ads-timeseries-stable-surface-acceptance-2026-03-08",
            "ads-timeseries-support-matrix-2026-03-08",
            "ads-timeseries-runtime-gate.md",
            "ads-timeseries-promotion-runbook-2026-03-08",
            "ads-timeseries-benchmark-snapshot-2026-03-08",
        ],
    )
    assert_doc_contains_strings(
        repo_root / "docs" / "README.md",
        [
            "ads-timeseries-stable-surface-acceptance-2026-03-08.md",
            "ads-timeseries-support-matrix-2026-03-08.md",
            "ads-timeseries-runtime-gate.md",
            "ads-timeseries-benchmark-snapshot-2026-03-08.md",
            "ads-timeseries-release-pr-checklist-2026-03-08.md",
        ],
    )
    assert_doc_contains_strings(
        repo_root / "docs" / "benchmarks" / "ads-timeseries-runtime-gate.md",
        [
            "compare_ads_timeseries_benchmark.py",
            "promote_ads_timeseries_benchmark_baseline.py",
            "run_ads_timeseries_benchmark_gate.py",
            "--allow-case-set-change",
            "python_cuped_adjust",
            "python_cure_adjust",
            "benchmarks/artifacts/ads_timeseries_baselines/nextstat-bench/accepted.json",
            "docs/schemas/benchmarks/ads_timeseries_benchmark_compare_report_v1.schema.json",
            "docs/specs/benchmarks/ads_timeseries_benchmark_compare_report_v1.example.json",
            "docs/schemas/benchmarks/ads_timeseries_benchmark_baseline_promotion_report_v1.schema.json",
            "docs/specs/benchmarks/ads_timeseries_benchmark_baseline_promotion_report_v1.example.json",
            "docs/schemas/benchmarks/ads_timeseries_benchmark_gate_report_v1.schema.json",
            "docs/specs/benchmarks/ads_timeseries_benchmark_gate_report_v1.example.json",
        ],
    )
    assert_doc_contains_strings(
        repo_root / "docs" / "benchmarks" / "ads-timeseries-promotion-runbook-2026-03-08.md",
        [
            "--allow-case-set-change",
            "python_cuped_adjust",
            "python_cure_adjust",
            "nextstat-bench",
        ],
    )
