from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import nextstat
import nextstat.hep as ns_hep

nextstat.set_threads(1)


def _numeric_tolerance_for_path(path: str) -> float:
    # Cross-platform BLAS/LAPACK order can move tiny solver-parity drift metrics
    # in GVM calibration reports by ~1e-6 on Linux vs macOS. Keep the default
    # parity contract strict everywhere else.
    gvm_drift_suffixes = (
        ".max_sigma_rel_diff",
        ".max_fit_sigma_rel_diff",
        ".fit_sigma_rel_diff",
        ".sigma_rel_diff",
    )
    if path.endswith(gvm_drift_suffixes):
        return 1e-4
    return 1e-10


def _assert_solver_parity_markdown_close(
    actual: str,
    expected: str,
    *,
    relaxed_column_indexes: tuple[int, ...],
) -> None:
    actual_lines = actual.splitlines()
    expected_lines = expected.splitlines()
    assert len(actual_lines) == len(expected_lines)
    for line_no, (actual_line, expected_line) in enumerate(zip(actual_lines, expected_lines), start=1):
        if actual_line == expected_line:
            continue
        if not actual_line.startswith("| ") or not expected_line.startswith("| "):
            assert actual_line == expected_line, f"markdown mismatch at line {line_no}"
            continue
        actual_cells = [cell.strip() for cell in actual_line.strip("|").split("|")]
        expected_cells = [cell.strip() for cell in expected_line.strip("|").split("|")]
        assert len(actual_cells) == len(expected_cells), f"column count mismatch at line {line_no}"
        for idx, (actual_cell, expected_cell) in enumerate(zip(actual_cells, expected_cells)):
            if idx in relaxed_column_indexes:
                try:
                    assert math.isclose(
                        float(actual_cell),
                        float(expected_cell),
                        rel_tol=0.0,
                        abs_tol=1e-4,
                    ), (
                        f"numeric markdown mismatch at line {line_no}, column {idx}: "
                        f"actual={actual_cell} expected={expected_cell}"
                    )
                    continue
                except ValueError:
                    pass
            assert actual_cell == expected_cell, (
                f"markdown mismatch at line {line_no}, column {idx}: "
                f"actual={actual_cell!r} expected={expected_cell!r}"
            )


def assert_json_close(actual: object, expected: object, path: str = "$") -> None:
    if isinstance(actual, dict) and isinstance(expected, dict):
        assert actual.keys() == expected.keys(), f"object keys differ at {path}"
        for key in expected:
            assert_json_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(actual, list) and isinstance(expected, list):
        assert len(actual) == len(expected), f"array length differs at {path}"
        for idx, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            assert_json_close(actual_item, expected_item, f"{path}[{idx}]")
        return
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        atol = _numeric_tolerance_for_path(path)
        tol = max(atol, atol * max(abs(float(actual)), abs(float(expected))))
        assert math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tol), (
            f"numeric mismatch at {path}: actual={actual} expected={expected} tol={tol}"
        )
        return
    assert actual == expected, f"value mismatch at {path}: actual={actual!r} expected={expected!r}"


def test_combine_measurements_accepts_dict(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake(spec_json: str, *, ci_level: float = 0.68, solver: str = "auto") -> str:
        captured["spec"] = json.loads(spec_json)
        captured["ci_level"] = ci_level
        captured["solver"] = solver
        return json.dumps({"schema_version": "nextstat_measurement_combination_result_v0", "mu_hat": 1.23})

    monkeypatch.setattr(ns_hep, "_measurement_combine_json", _fake)

    out = ns_hep.combine_measurements(
        {
            "schema_version": "nextstat_measurement_combination_v0",
            "poi": "mu",
            "measurements": [{"name": "m1", "value": 1.0}],
            "stat_covariance": [[1.0]],
            "systematics": [],
        },
        ci_level=0.95,
    )

    assert out["mu_hat"] == 1.23
    assert captured["ci_level"] == 0.95
    assert captured["solver"] == "auto"
    assert captured["spec"] and captured["spec"]["poi"] == "mu"


def test_gvm_json_close_relaxes_only_solver_parity_sigma_drift_metrics() -> None:
    assert _numeric_tolerance_for_path("$.aggregate.max_sigma_rel_diff") == 1e-4
    assert _numeric_tolerance_for_path("$.aggregate.max_fit_sigma_rel_diff") == 1e-4
    assert _numeric_tolerance_for_path("$.scenarios[0].fit_sigma_rel_diff") == 1e-4
    assert _numeric_tolerance_for_path("$.scenarios[0].sigma_rel_diff") == 1e-4
    assert _numeric_tolerance_for_path("$.reference.optimizer.nll") == 1e-10


def test_combine_measurements_accepts_path(monkeypatch, tmp_path: Path) -> None:
    payload = {
        "schema_version": "nextstat_measurement_combination_v0",
        "poi": "mu",
        "measurements": [{"name": "m1", "value": 1.0}],
        "stat_covariance": [[1.0]],
        "systematics": [],
    }
    spec_path = tmp_path / "measurement_combination.json"
    spec_path.write_text(json.dumps(payload), encoding="utf-8")

    def _fake(spec_json: str, *, ci_level: float = 0.68, solver: str = "auto") -> str:
        assert json.loads(spec_json) == payload
        return json.dumps({"ok": True})

    monkeypatch.setattr(ns_hep, "_measurement_combine_json", _fake)
    assert ns_hep.combine_measurements(spec_path) == {"ok": True}


def test_build_measurement_combination_spec_matches_cli(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    measurements = tmp_path / "measurements.csv"
    stat_covariance = tmp_path / "stat_covariance.csv"
    systematics = tmp_path / "systematics.csv"
    correlations = tmp_path / "correlations.csv"

    measurements.write_text("name,value\nm1,1.0\nm2,3.0\n", encoding="utf-8")
    stat_covariance.write_text(
        "measurement,m2,m1\nm2,4.0,0.0\nm1,0.0,1.0\n", encoding="utf-8"
    )
    systematics.write_text(
        "systematic,measurement,magnitude,error_on_error,aux_mean\n"
        "s1,m1,0.2,0.1,0.0\n"
        "s1,m2,0.3,0.1,0.0\n",
        encoding="utf-8",
    )
    correlations.write_text(
        "systematic,row_measurement,col_measurement,corr\ns1,m1,m2,0.5\n",
        encoding="utf-8",
    )

    py_spec = ns_hep.build_measurement_combination_spec(
        measurements,
        stat_covariance,
        poi="mu",
        systematics_table=systematics,
        correlations_table=correlations,
    )

    cli = subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "ns-cli",
            "--",
            "combine-measurements-build-spec",
            "--poi",
            "mu",
            "--measurements",
            str(measurements),
            "--stat-covariance",
            str(stat_covariance),
            "--systematics",
            str(systematics),
            "--correlations",
            str(correlations),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    cli_spec = json.loads(cli.stdout)

    assert_json_close(py_spec, cli_spec)
    assert py_spec["schema_version"] == "nextstat_measurement_combination_v0"


def test_stable_first_example_bundle_roundtrips_through_python() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    example_dir = repo_root / "docs" / "examples" / "gvm-stable-first"
    manifest = example_dir / "manifest.yaml"
    committed_spec = json.loads((example_dir / "spec.json").read_text(encoding="utf-8"))

    built_spec = ns_hep.build_measurement_combination_spec(
        example_dir / "measurements.csv",
        example_dir / "stat_covariance.csv",
        poi="mu",
        systematics_table=example_dir / "systematics.csv",
        correlations_table=example_dir / "correlations.csv",
    )
    assert_json_close(built_spec, committed_spec)
    manifest_spec = ns_hep.build_measurement_combination_spec_from_manifest(manifest)
    assert_json_close(manifest_spec, committed_spec)

    fit = ns_hep.combine_measurements(committed_spec, solver="auto")
    assert fit["stability"] == "stable"
    assert fit["diagnostics"]["requested_solver"] == "auto"

    calibration = ns_hep.calibrate_measurements(committed_spec, solver="auto", n_toys=8, seed=42)
    assert calibration["stability"] == "stable"

    study = ns_hep.calibrate_measurements_study(
        committed_spec, solver="auto", n_toys=8, seeds=[42, 43]
    )
    assert study["stability"] == "stable"


def test_combine_measurements_matches_cli_on_literature_fixture() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fixture = repo_root / "tests" / "fixtures" / "measurement_combine_gvm_topmass_full.json"

    py_out = ns_hep.combine_measurements(fixture, ci_level=0.683)

    cli = subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "ns-cli",
            "--",
            "combine-measurements",
            "--input",
            str(fixture),
            "--threads",
            "1",
            "--ci-level",
            "0.683",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    cli_out = json.loads(cli.stdout)

    assert py_out == cli_out
    assert py_out["stability"] == "stable"
    assert abs(py_out["mu_hat"] - 172.51) <= 5e-3


def test_combine_measurements_surfaces_bartlett_for_full_correlated_gvm_case(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fixture = repo_root / "tests" / "fixtures" / "measurement_combine_gvm_topmass_full.json"
    spec = json.loads(fixture.read_text(encoding="utf-8"))
    for syst in spec["systematics"]:
        if syst["name"] == "b-JES":
            syst["error_on_error"] = 0.5
            break
    else:  # pragma: no cover
        raise AssertionError("b-JES systematic missing from literature fixture")

    spec_path = tmp_path / "full_gvm.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    out = ns_hep.combine_measurements(spec_path, ci_level=0.68)
    bartlett = out["diagnostics"]["bartlett"]
    validity = out["diagnostics"]["perturbative_validity"]

    assert bartlett["supported"] is True
    assert bartlett["method"] == "lawley_order_eps2_general"
    assert bartlett["b_mu"] == bartlett["b_mu"]
    assert bartlett["q_star"] == bartlett["q_star"]
    assert validity["threshold"] == 1.0
    assert len(validity["condition_values"]) == 1


def test_combine_measurements_accepts_solver_modes(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fixture = repo_root / "tests" / "fixtures" / "measurement_combine_gvm_topmass_full.json"
    spec = json.loads(fixture.read_text(encoding="utf-8"))
    for syst in spec["systematics"]:
        if syst["name"] == "b-JES":
            syst["error_on_error"] = 0.05
            break
    else:  # pragma: no cover
        raise AssertionError("b-JES systematic missing from literature fixture")

    spec_path = tmp_path / "paper_modes.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    paper = ns_hep.combine_measurements(spec_path, ci_level=0.68, solver="numerical-paper")
    analytic = ns_hep.combine_measurements(
        spec_path, ci_level=0.68, solver="analytic-perturbative"
    )

    assert paper["optimizer"]["method"] == "numerical_profile_gvm_original_theta"
    assert analytic["optimizer"]["method"] == "analytic_perturbative_order_eps2"
    assert abs(analytic["mu_hat"] - paper["mu_hat"]) < 1e-2


def test_combine_measurements_defaults_to_auto_solver_contract(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fixture = repo_root / "tests" / "fixtures" / "measurement_combine_gvm_topmass_full.json"
    spec = json.loads(fixture.read_text(encoding="utf-8"))
    for syst in spec["systematics"]:
        if syst["name"] == "b-JES":
            syst["error_on_error"] = 0.05
            break
    else:  # pragma: no cover
        raise AssertionError("b-JES systematic missing from literature fixture")

    spec_path = tmp_path / "auto_default.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    out = ns_hep.combine_measurements(spec_path, ci_level=0.68)
    assert out["optimizer"]["method"] == "analytic_perturbative_order_eps2"
    assert out["diagnostics"]["requested_solver"] == "auto"
    assert out["diagnostics"]["effective_solver"] == "analytic-perturbative"


def test_combine_measurements_default_auto_falls_back_to_numerical_paper(tmp_path: Path) -> None:
    spec = {
        "schema_version": "nextstat_measurement_combination_v0",
        "poi": "mu",
        "measurements": [
            {"name": "m1", "value": 0.0},
            {"name": "m2", "value": 0.1},
            {"name": "m3", "value": 3.0},
        ],
        "stat_covariance": [
            [0.04, 0.0, 0.0],
            [0.0, 0.04, 0.0],
            [0.0, 0.0, 0.04],
        ],
        "systematics": [
            {
                "name": "new",
                "magnitudes": [0.0, 0.0, 0.2],
                "corr": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "error_on_error": 1.5,
            }
        ],
    }
    spec_path = tmp_path / "auto_fallback.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    out = ns_hep.combine_measurements(spec_path, ci_level=0.68)
    assert out["optimizer"]["method"] == "numerical_profile_gvm_original_theta"
    assert out["diagnostics"]["requested_solver"] == "auto"
    assert out["diagnostics"]["effective_solver"] == "numerical-paper"


def test_calibrate_measurements_returns_research_grade_report(tmp_path: Path) -> None:
    spec = {
        "schema_version": "nextstat_measurement_combination_v0",
        "poi": "mu",
        "measurements": [
            {"name": "m1", "value": 0.0},
            {"name": "m2", "value": 0.1},
            {"name": "m3", "value": 3.0},
        ],
        "stat_covariance": [
            [0.04, 0.0, 0.0],
            [0.0, 0.04, 0.0],
            [0.0, 0.0, 0.04],
        ],
        "systematics": [
            {
                "name": "new",
                "magnitudes": [0.0, 0.0, 0.2],
                "corr": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "error_on_error": 0.3,
            }
        ],
    }
    spec_path = tmp_path / "calibration.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    out = ns_hep.calibrate_measurements(spec_path, ci_level=0.68, n_toys=16, seed=123)

    assert out["schema_version"] == "nextstat_measurement_combination_calibration_v0"
    assert out["stability"] == "stable"
    assert out["n_toys"] == 16
    assert out["seed"] == 123
    assert out["summary"]["mean_q"] == out["summary"]["mean_q"]
    assert out["summary"]["mean_q_star"] == out["summary"]["mean_q_star"]
    assert out["summary"]["mean_sigma"] == out["summary"]["mean_sigma"]
    assert out["summary"]["mean_sigma_star_to_sigma_ratio"] == out["summary"]["mean_sigma_star_to_sigma_ratio"]


def test_calibrate_measurements_default_uses_auto_fit_and_paper_toy_generation(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fixture = repo_root / "tests" / "fixtures" / "measurement_combine_gvm_topmass_full.json"
    spec = json.loads(fixture.read_text(encoding="utf-8"))
    for syst in spec["systematics"]:
        if syst["name"] == "b-JES":
            syst["error_on_error"] = 0.05
            break
    else:  # pragma: no cover
        raise AssertionError("b-JES systematic missing from literature fixture")

    spec_path = tmp_path / "auto_calibration_default.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    out = ns_hep.calibrate_measurements(spec_path, ci_level=0.68, n_toys=8, seed=2026)
    assert out["reference"]["optimizer"]["method"] == "analytic_perturbative_order_eps2"
    assert (
        out["summary"]["toy_generation_method"]
        == "measurement_side_gvm_unbiased_sigma2_star_normalized_spec_original_theta"
    )


def test_calibrate_measurements_matches_committed_artifact() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = repo_root / "tests" / "fixtures" / "measurement_combine_calibration_outlier_input.json"
    output_fixture = repo_root / "tests" / "fixtures" / "measurement_combine_calibration_outlier_report.json"

    out = ns_hep.calibrate_measurements(input_fixture, ci_level=0.68, n_toys=16, seed=123)
    expected = json.loads(output_fixture.read_text(encoding="utf-8"))

    assert_json_close(out, expected)


def test_calibrate_measurements_study_returns_research_grade_report() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = repo_root / "tests" / "fixtures" / "measurement_combine_calibration_outlier_input.json"

    out = ns_hep.calibrate_measurements_study(
        input_fixture,
        ci_level=0.68,
        n_toys=16,
        seeds=[123, 124, 125],
    )

    assert out["schema_version"] == "nextstat_measurement_combination_calibration_study_v0"
    assert out["stability"] == "stable"
    assert out["seeds"] == [123, 124, 125]
    assert len(out["per_seed"]) == 3
    assert out["aggregate"]["n_runs"] == 3
    assert out["aggregate"]["mean_of_mean_q"] == out["aggregate"]["mean_of_mean_q"]
    assert (
        out["aggregate"]["max_abs_mean_sigma_star_to_sigma_ratio_delta_from_reference"]
        == out["aggregate"]["max_abs_mean_sigma_star_to_sigma_ratio_delta_from_reference"]
    )


def test_calibrate_measurements_study_matches_committed_artifact() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = repo_root / "tests" / "fixtures" / "measurement_combine_calibration_outlier_input.json"
    output_fixture = (
        repo_root / "tests" / "fixtures" / "measurement_combine_calibration_outlier_study_report.json"
    )

    out = ns_hep.calibrate_measurements_study(
        input_fixture,
        ci_level=0.68,
        n_toys=16,
        seeds=[123, 124, 125],
    )
    expected = json.loads(output_fixture.read_text(encoding="utf-8"))

    assert_json_close(out, expected)


def test_study_measurement_combination_scenarios_returns_research_grade_report() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = repo_root / "tests" / "fixtures" / "measurement_combine_calibration_outlier_input.json"
    scenarios_fixture = repo_root / "tests" / "fixtures" / "measurement_combine_outlier_scenarios.json"

    out = ns_hep.study_measurement_combination_scenarios(input_fixture, scenarios_fixture, ci_level=0.68)

    assert out["schema_version"] == "nextstat_measurement_combination_scenario_study_v0"
    assert out["stability"] == "research-grade"
    assert len(out["scenarios"]) == 3
    assert out["aggregate"]["n_scenarios"] == 3
    assert out["aggregate"]["max_sigma_ratio_to_baseline"] == out["aggregate"]["max_sigma_ratio_to_baseline"]


def test_study_measurement_combination_scenarios_matches_committed_artifact() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = repo_root / "tests" / "fixtures" / "measurement_combine_calibration_outlier_input.json"
    scenarios_fixture = repo_root / "tests" / "fixtures" / "measurement_combine_outlier_scenarios.json"
    output_fixture = repo_root / "tests" / "fixtures" / "measurement_combine_outlier_scenario_study_report.json"

    out = ns_hep.study_measurement_combination_scenarios(input_fixture, scenarios_fixture, ci_level=0.68)
    expected = json.loads(output_fixture.read_text(encoding="utf-8"))

    assert_json_close(out, expected)


def test_study_measurement_combination_scenarios_accepts_solver_modes(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = repo_root / "tests" / "fixtures" / "measurement_combine_gvm_topmass_full.json"
    scenarios = {
        "schema_version": "nextstat_measurement_combination_scenarios_v0",
        "scenarios": [
            {
                "name": "bjes_0p05",
                "error_on_error": [{"systematic": "b-JES", "value": 0.05}],
            }
        ],
    }
    scenarios_path = tmp_path / "paper_scenarios.json"
    scenarios_path.write_text(json.dumps(scenarios), encoding="utf-8")

    out = ns_hep.study_measurement_combination_scenarios(
        input_fixture,
        scenarios_path,
        ci_level=0.68,
        solver="analytic-perturbative",
    )

    assert out["baseline"]["optimizer"]["method"] == "closed_form_blue"
    assert out["scenarios"][0]["result"]["optimizer"]["method"] == "analytic_perturbative_order_eps2"


def test_calibrate_measurement_combination_scenarios_returns_research_grade_report() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = repo_root / "tests" / "fixtures" / "measurement_combine_calibration_outlier_input.json"
    scenarios_fixture = repo_root / "tests" / "fixtures" / "measurement_combine_outlier_scenarios.json"

    out = ns_hep.calibrate_measurement_combination_scenarios(
        input_fixture,
        scenarios_fixture,
        ci_level=0.68,
        n_toys=16,
        seeds=[123, 124, 125],
    )

    assert out["schema_version"] == "nextstat_measurement_combination_calibration_campaign_v0"
    assert out["stability"] == "research-grade"
    assert out["seeds"] == [123, 124, 125]
    assert len(out["scenarios"]) == 3
    assert out["aggregate"]["n_scenarios"] == 3
    assert out["aggregate"]["max_fit_sigma_ratio_to_baseline"] == out["aggregate"]["max_fit_sigma_ratio_to_baseline"]


def test_calibrate_measurement_combination_scenarios_matches_committed_artifact() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = repo_root / "tests" / "fixtures" / "measurement_combine_calibration_outlier_input.json"
    scenarios_fixture = repo_root / "tests" / "fixtures" / "measurement_combine_outlier_scenarios.json"
    output_fixture = (
        repo_root
        / "tests"
        / "fixtures"
        / "measurement_combine_outlier_calibration_campaign_report.json"
    )

    out = ns_hep.calibrate_measurement_combination_scenarios(
        input_fixture,
        scenarios_fixture,
        ci_level=0.68,
        n_toys=16,
        seeds=[123, 124, 125],
    )
    expected = json.loads(output_fixture.read_text(encoding="utf-8"))

    assert_json_close(out, expected)


def test_calibrate_measurement_combination_scenarios_accepts_solver_modes(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = repo_root / "tests" / "fixtures" / "measurement_combine_gvm_topmass_full.json"
    scenarios = {
        "schema_version": "nextstat_measurement_combination_scenarios_v0",
        "scenarios": [
            {
                "name": "bjes_0p05",
                "error_on_error": [{"systematic": "b-JES", "value": 0.05}],
            }
        ],
    }
    scenarios_path = tmp_path / "paper_campaign_scenarios.json"
    scenarios_path.write_text(json.dumps(scenarios), encoding="utf-8")

    out = ns_hep.calibrate_measurement_combination_scenarios(
        input_fixture,
        scenarios_path,
        ci_level=0.68,
        solver="analytic-perturbative",
        n_toys=8,
        seeds=[2026, 2027],
    )

    assert out["scenarios"][0]["fit"]["optimizer"]["method"] == "analytic_perturbative_order_eps2"
    assert (
        out["scenarios"][0]["calibration"]["toy_generation_method"]
        == "measurement_side_gvm_unbiased_sigma2_star_normalized_spec_original_theta"
    )


def test_compare_measurement_combination_scenario_study_solvers_returns_research_grade_report(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = repo_root / "tests" / "fixtures" / "measurement_combine_gvm_topmass_full.json"
    scenarios = {
        "schema_version": "nextstat_measurement_combination_scenarios_v0",
        "scenarios": [
            {
                "name": "bjes_0p05",
                "error_on_error": [{"systematic": "b-JES", "value": 0.05}],
            }
        ],
    }
    scenarios_path = tmp_path / "paper_parity_scenarios.json"
    scenarios_path.write_text(json.dumps(scenarios), encoding="utf-8")

    out = ns_hep.compare_measurement_combination_scenario_study_solvers(
        input_fixture,
        scenarios_path,
        ci_level=0.68,
        lhs_solver="numerical-paper",
        rhs_solver="analytic-perturbative",
    )

    assert out["schema_version"] == "nextstat_measurement_combination_scenario_study_solver_parity_v0"
    assert out["lhs_solver"] == "numerical-paper"
    assert out["rhs_solver"] == "analytic-perturbative"
    assert out["aggregate"]["n_scenarios"] == 1


def test_render_measurement_combination_scenario_study_solver_parity_matches_fixture(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = repo_root / "tests" / "fixtures" / "measurement_combine_gvm_topmass_full.json"
    output_fixture = (
        repo_root / "tests" / "fixtures" / "measurement_combine_solver_parity_scenario_study_report.md"
    )
    scenarios = {
        "schema_version": "nextstat_measurement_combination_scenarios_v0",
        "scenarios": [
            {
                "name": "bjes_0p05",
                "error_on_error": [{"systematic": "b-JES", "value": 0.05}],
            }
        ],
    }
    scenarios_path = tmp_path / "paper_parity_scenarios_render.json"
    scenarios_path.write_text(json.dumps(scenarios), encoding="utf-8")

    report = ns_hep.compare_measurement_combination_scenario_study_solvers(
        input_fixture,
        scenarios_path,
        ci_level=0.68,
        lhs_solver="numerical-paper",
        rhs_solver="analytic-perturbative",
    )
    markdown = ns_hep.render_measurement_combination_scenario_study_solver_parity(report)

    _assert_solver_parity_markdown_close(
        markdown,
        output_fixture.read_text(encoding="utf-8"),
        relaxed_column_indexes=(5,),
    )


def test_compare_measurement_combination_scenario_study_solver_reports_matches_direct_path(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = repo_root / "tests" / "fixtures" / "measurement_combine_gvm_topmass_full.json"
    scenarios = {
        "schema_version": "nextstat_measurement_combination_scenarios_v0",
        "scenarios": [
            {
                "name": "bjes_0p05",
                "error_on_error": [{"systematic": "b-JES", "value": 0.05}],
            }
        ],
    }
    lhs = ns_hep.study_measurement_combination_scenarios(
        input_fixture,
        scenarios,
        ci_level=0.68,
        solver="numerical-paper",
    )
    rhs = ns_hep.study_measurement_combination_scenarios(
        input_fixture,
        scenarios,
        ci_level=0.68,
        solver="analytic-perturbative",
    )
    from_reports = ns_hep.compare_measurement_combination_scenario_study_solver_reports(
        lhs,
        rhs,
        lhs_solver="numerical-paper",
        rhs_solver="analytic-perturbative",
    )
    direct = ns_hep.compare_measurement_combination_scenario_study_solvers(
        input_fixture,
        scenarios,
        ci_level=0.68,
        lhs_solver="numerical-paper",
        rhs_solver="analytic-perturbative",
    )
    assert_json_close(from_reports, direct)


def test_summarize_measurement_combination_scenario_study_solver_parity_matches_fixture() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = repo_root / "tests" / "fixtures" / "measurement_combine_gvm_topmass_full.json"
    output_fixture = (
        repo_root
        / "tests"
        / "fixtures"
        / "measurement_combine_solver_parity_scenario_study_digest.json"
    )
    scenarios = {
        "schema_version": "nextstat_measurement_combination_scenarios_v0",
        "scenarios": [
            {
                "name": "bjes_0p05",
                "error_on_error": [{"systematic": "b-JES", "value": 0.05}],
            }
        ],
    }

    report = ns_hep.compare_measurement_combination_scenario_study_solvers(
        input_fixture,
        scenarios,
        ci_level=0.68,
        lhs_solver="numerical-paper",
        rhs_solver="analytic-perturbative",
    )
    summary = ns_hep.summarize_measurement_combination_scenario_study_solver_parity(report)

    assert_json_close(summary, json.loads(output_fixture.read_text(encoding="utf-8")))


def test_render_measurement_combination_scenario_study_solver_parity_summary_matches_fixture() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = (
        repo_root
        / "tests"
        / "fixtures"
        / "measurement_combine_solver_parity_scenario_study_digest.json"
    )
    output_fixture = (
        repo_root
        / "tests"
        / "fixtures"
        / "measurement_combine_solver_parity_scenario_study_digest.md"
    )

    markdown = ns_hep.render_measurement_combination_scenario_study_solver_parity_summary(
        input_fixture
    )
    assert markdown == output_fixture.read_text(encoding="utf-8")


def test_compare_measurement_combination_calibration_campaign_solvers_returns_research_grade_report(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = repo_root / "tests" / "fixtures" / "measurement_combine_gvm_topmass_full.json"
    scenarios = {
        "schema_version": "nextstat_measurement_combination_scenarios_v0",
        "scenarios": [
            {
                "name": "bjes_0p05",
                "error_on_error": [{"systematic": "b-JES", "value": 0.05}],
            }
        ],
    }
    scenarios_path = tmp_path / "paper_parity_campaign_scenarios.json"
    scenarios_path.write_text(json.dumps(scenarios), encoding="utf-8")

    out = ns_hep.compare_measurement_combination_calibration_campaign_solvers(
        input_fixture,
        scenarios_path,
        ci_level=0.68,
        lhs_solver="numerical-paper",
        rhs_solver="analytic-perturbative",
        n_toys=8,
        seeds=[2026, 2027],
    )

    assert out["schema_version"] == "nextstat_measurement_combination_calibration_campaign_solver_parity_v0"
    assert out["lhs_solver"] == "numerical-paper"
    assert out["rhs_solver"] == "analytic-perturbative"
    assert out["aggregate"]["n_scenarios"] == 1


def test_render_measurement_combination_calibration_campaign_solver_parity_matches_fixture(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = repo_root / "tests" / "fixtures" / "measurement_combine_gvm_topmass_full.json"
    output_fixture = (
        repo_root
        / "tests"
        / "fixtures"
        / "measurement_combine_solver_parity_calibration_campaign_report.md"
    )
    scenarios = {
        "schema_version": "nextstat_measurement_combination_scenarios_v0",
        "scenarios": [
            {
                "name": "bjes_0p05",
                "error_on_error": [{"systematic": "b-JES", "value": 0.05}],
            }
        ],
    }
    scenarios_path = tmp_path / "paper_parity_campaign_render.json"
    scenarios_path.write_text(json.dumps(scenarios), encoding="utf-8")

    report = ns_hep.compare_measurement_combination_calibration_campaign_solvers(
        input_fixture,
        scenarios_path,
        ci_level=0.68,
        lhs_solver="numerical-paper",
        rhs_solver="analytic-perturbative",
        n_toys=8,
        seeds=[2026, 2027],
    )
    markdown = ns_hep.render_measurement_combination_calibration_campaign_solver_parity(report)

    _assert_solver_parity_markdown_close(
        markdown,
        output_fixture.read_text(encoding="utf-8"),
        relaxed_column_indexes=(4,),
    )


def test_compare_measurement_combination_calibration_campaign_solver_reports_matches_direct_path(
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = repo_root / "tests" / "fixtures" / "measurement_combine_gvm_topmass_full.json"
    scenarios = {
        "schema_version": "nextstat_measurement_combination_scenarios_v0",
        "scenarios": [
            {
                "name": "bjes_0p05",
                "error_on_error": [{"systematic": "b-JES", "value": 0.05}],
            }
        ],
    }
    lhs = ns_hep.calibrate_measurement_combination_scenarios(
        input_fixture,
        scenarios,
        ci_level=0.68,
        solver="numerical-paper",
        n_toys=8,
        seeds=[2026, 2027],
    )
    rhs = ns_hep.calibrate_measurement_combination_scenarios(
        input_fixture,
        scenarios,
        ci_level=0.68,
        solver="analytic-perturbative",
        n_toys=8,
        seeds=[2026, 2027],
    )
    from_reports = ns_hep.compare_measurement_combination_calibration_campaign_solver_reports(
        lhs,
        rhs,
        lhs_solver="numerical-paper",
        rhs_solver="analytic-perturbative",
    )
    direct = ns_hep.compare_measurement_combination_calibration_campaign_solvers(
        input_fixture,
        scenarios,
        ci_level=0.68,
        lhs_solver="numerical-paper",
        rhs_solver="analytic-perturbative",
        n_toys=8,
        seeds=[2026, 2027],
    )
    assert_json_close(from_reports, direct)


def test_summarize_measurement_combination_calibration_campaign_solver_parity_matches_fixture() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = repo_root / "tests" / "fixtures" / "measurement_combine_gvm_topmass_full.json"
    output_fixture = (
        repo_root
        / "tests"
        / "fixtures"
        / "measurement_combine_solver_parity_calibration_campaign_digest.json"
    )
    scenarios = {
        "schema_version": "nextstat_measurement_combination_scenarios_v0",
        "scenarios": [
            {
                "name": "bjes_0p05",
                "error_on_error": [{"systematic": "b-JES", "value": 0.05}],
            }
        ],
    }

    report = ns_hep.compare_measurement_combination_calibration_campaign_solvers(
        input_fixture,
        scenarios,
        ci_level=0.68,
        lhs_solver="numerical-paper",
        rhs_solver="analytic-perturbative",
        n_toys=8,
        seeds=[2026, 2027],
    )
    summary = ns_hep.summarize_measurement_combination_calibration_campaign_solver_parity(report)

    assert_json_close(summary, json.loads(output_fixture.read_text(encoding="utf-8")))


def test_render_measurement_combination_calibration_campaign_solver_parity_summary_matches_fixture() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = (
        repo_root
        / "tests"
        / "fixtures"
        / "measurement_combine_solver_parity_calibration_campaign_digest.json"
    )
    output_fixture = (
        repo_root
        / "tests"
        / "fixtures"
        / "measurement_combine_solver_parity_calibration_campaign_digest.md"
    )

    markdown = ns_hep.render_measurement_combination_calibration_campaign_solver_parity_summary(
        input_fixture
    )
    assert markdown == output_fixture.read_text(encoding="utf-8")


def test_summarize_measurement_combination_calibration_campaign_returns_research_grade_report() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = (
        repo_root / "tests" / "fixtures" / "measurement_combine_outlier_calibration_campaign_report.json"
    )

    out = ns_hep.summarize_measurement_combination_calibration_campaign(input_fixture)

    assert out["schema_version"] == "nextstat_measurement_combination_calibration_campaign_summary_v0"
    assert out["stability"] == "research-grade"
    assert out["aggregate"]["n_scenarios"] == 3
    assert out["dominant_calibration_scenario"] == "new_0p5"


def test_summarize_measurement_combination_calibration_campaign_matches_committed_artifact() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = (
        repo_root / "tests" / "fixtures" / "measurement_combine_outlier_calibration_campaign_report.json"
    )
    output_fixture = (
        repo_root
        / "tests"
        / "fixtures"
        / "measurement_combine_outlier_calibration_campaign_summary.json"
    )

    out = ns_hep.summarize_measurement_combination_calibration_campaign(input_fixture)
    expected = json.loads(output_fixture.read_text(encoding="utf-8"))

    assert_json_close(out, expected)


def test_render_measurement_combination_calibration_campaign_summary_matches_committed_artifact() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = (
        repo_root / "tests" / "fixtures" / "measurement_combine_outlier_calibration_campaign_summary.json"
    )
    output_fixture = (
        repo_root / "tests" / "fixtures" / "measurement_combine_outlier_calibration_campaign_summary.md"
    )

    out = ns_hep.render_measurement_combination_calibration_campaign_summary(input_fixture)
    expected = output_fixture.read_text(encoding="utf-8")

    assert out == expected


def test_build_measurement_combination_calibration_campaign_brief_matches_committed_artifact() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    inputs = [
        repo_root / "tests" / "fixtures" / "measurement_combine_outlier_calibration_campaign_summary.json",
        repo_root
        / "tests"
        / "fixtures"
        / "measurement_combine_calibration_topmass_full_campaign_summary.json",
    ]
    output_fixture = (
        repo_root / "tests" / "fixtures" / "measurement_combine_calibration_campaign_brief.json"
    )

    out = ns_hep.build_measurement_combination_calibration_campaign_brief(
        inputs,
        labels=["outlier", "topmass_full"],
    )
    expected = json.loads(output_fixture.read_text(encoding="utf-8"))

    assert_json_close(out, expected)


def test_render_measurement_combination_calibration_campaign_brief_matches_committed_artifact() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = (
        repo_root / "tests" / "fixtures" / "measurement_combine_calibration_campaign_brief.json"
    )
    output_fixture = (
        repo_root / "tests" / "fixtures" / "measurement_combine_calibration_campaign_brief.md"
    )

    out = ns_hep.render_measurement_combination_calibration_campaign_brief(input_fixture)
    expected = output_fixture.read_text(encoding="utf-8")

    assert out == expected


def test_build_measurement_combination_calibration_campaign_family_report_matches_committed_artifact() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    inputs = [
        repo_root / "tests" / "fixtures" / "measurement_combine_calibration_campaign_brief.json",
        repo_root / "tests" / "fixtures" / "measurement_combine_calibration_topmass_only_brief.json",
    ]
    output_fixture = (
        repo_root / "tests" / "fixtures" / "measurement_combine_calibration_campaign_family_report.json"
    )

    out = ns_hep.build_measurement_combination_calibration_campaign_family_report(
        inputs,
        labels=["cross_fixture", "topmass_only"],
    )
    expected = json.loads(output_fixture.read_text(encoding="utf-8"))

    assert_json_close(out, expected)


def test_render_measurement_combination_calibration_campaign_family_report_matches_committed_artifact() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = (
        repo_root / "tests" / "fixtures" / "measurement_combine_calibration_campaign_family_report.json"
    )
    output_fixture = (
        repo_root / "tests" / "fixtures" / "measurement_combine_calibration_campaign_family_report.md"
    )

    out = ns_hep.render_measurement_combination_calibration_campaign_family_report(input_fixture)
    expected = output_fixture.read_text(encoding="utf-8")

    assert out == expected


def test_build_measurement_combination_calibration_campaign_family_matrix_matches_committed_artifact() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = (
        repo_root / "tests" / "fixtures" / "measurement_combine_calibration_campaign_family_report.json"
    )
    output_fixture = (
        repo_root / "tests" / "fixtures" / "measurement_combine_calibration_campaign_family_matrix.json"
    )

    out = ns_hep.build_measurement_combination_calibration_campaign_family_matrix(input_fixture)
    expected = json.loads(output_fixture.read_text(encoding="utf-8"))

    assert_json_close(out, expected)


def test_render_measurement_combination_calibration_campaign_family_matrix_matches_committed_artifact() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = (
        repo_root / "tests" / "fixtures" / "measurement_combine_calibration_campaign_family_matrix.json"
    )
    output_fixture = (
        repo_root / "tests" / "fixtures" / "measurement_combine_calibration_campaign_family_matrix.md"
    )

    out = ns_hep.render_measurement_combination_calibration_campaign_family_matrix(input_fixture)
    expected = output_fixture.read_text(encoding="utf-8")

    assert out == expected


def test_build_measurement_combination_calibration_campaign_portfolio_matches_committed_artifact() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    inputs = [
        repo_root / "tests" / "fixtures" / "measurement_combine_calibration_campaign_family_matrix.json",
        repo_root / "tests" / "fixtures" / "measurement_combine_calibration_topmass_only_family_matrix.json",
    ]
    output_fixture = (
        repo_root / "tests" / "fixtures" / "measurement_combine_calibration_campaign_portfolio.json"
    )

    out = ns_hep.build_measurement_combination_calibration_campaign_portfolio(
        inputs,
        labels=["cross_portfolio", "topmass_only_portfolio"],
    )
    expected = json.loads(output_fixture.read_text(encoding="utf-8"))

    assert_json_close(out, expected)


def test_render_measurement_combination_calibration_campaign_portfolio_matches_committed_artifact() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = (
        repo_root / "tests" / "fixtures" / "measurement_combine_calibration_campaign_portfolio.json"
    )
    output_fixture = (
        repo_root / "tests" / "fixtures" / "measurement_combine_calibration_campaign_portfolio.md"
    )

    out = ns_hep.render_measurement_combination_calibration_campaign_portfolio(input_fixture)
    expected = output_fixture.read_text(encoding="utf-8")

    assert out == expected


def test_build_measurement_combination_calibration_campaign_portfolio_stability_matches_committed_artifact() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    inputs = [
        repo_root / "tests" / "fixtures" / "measurement_combine_calibration_campaign_portfolio.json",
        repo_root / "tests" / "fixtures" / "measurement_combine_calibration_campaign_portfolio.json",
    ]
    output_fixture = (
        repo_root / "tests" / "fixtures" / "measurement_combine_calibration_campaign_portfolio_stability.json"
    )

    out = ns_hep.build_measurement_combination_calibration_campaign_portfolio_stability(
        inputs,
        labels=["seedgrid_a", "seedgrid_b"],
    )
    expected = json.loads(output_fixture.read_text(encoding="utf-8"))

    assert_json_close(out, expected)


def test_render_measurement_combination_calibration_campaign_portfolio_stability_matches_committed_artifact() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_fixture = (
        repo_root / "tests" / "fixtures" / "measurement_combine_calibration_campaign_portfolio_stability.json"
    )
    output_fixture = (
        repo_root / "tests" / "fixtures" / "measurement_combine_calibration_campaign_portfolio_stability.md"
    )

    out = ns_hep.render_measurement_combination_calibration_campaign_portfolio_stability(input_fixture)
    expected = output_fixture.read_text(encoding="utf-8")

    assert out == expected
