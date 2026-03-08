import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _nextstat_subprocess_env(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    source_bindings = repo / "bindings" / "ns-py" / "python"
    source_pkg = source_bindings / "nextstat"
    patterns = ("_core*.so", "_core*.pyd", "_core*.dylib", "_core*.dll")
    local_extension_present = any(any(source_pkg.glob(pattern)) for pattern in patterns)
    existing_pythonpath = env.get("PYTHONPATH", "")
    existing_entries = [entry for entry in existing_pythonpath.split(os.pathsep) if entry]
    source_bindings_str = str(source_bindings)
    source_pythonpath_requested = source_bindings_str in existing_entries
    if (
        source_pythonpath_requested
        or local_extension_present
        or os.environ.get("NEXTSTAT_FORCE_PYTHONPATH") == "1"
    ):
        env["PYTHONPATH"] = str(source_bindings)
        env.pop("NEXTSTAT_PREFER_INSTALLED", None)
    else:
        env.pop("PYTHONPATH", None)
        env["NEXTSTAT_PREFER_INSTALLED"] = "1"
    return env


def test_nextstat_subprocess_env_prefers_installed_when_repo_has_no_local_extension(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("NEXTSTAT_FORCE_PYTHONPATH", raising=False)
    repo = _repo_root()
    env = _nextstat_subprocess_env(tmp_path)
    assert env.get("NEXTSTAT_PREFER_INSTALLED") == "1"
    assert env.get("PYTHONPATH") is None

    source_bindings = repo / "bindings" / "ns-py" / "python"
    source_pkg = source_bindings / "nextstat"
    patterns = ("_core*.so", "_core*.pyd", "_core*.dylib", "_core*.dll")
    local_extension_present = any(any(source_pkg.glob(pattern)) for pattern in patterns)
    if local_extension_present:
        env_with_local = _nextstat_subprocess_env(repo)
        assert env_with_local.get("PYTHONPATH") == str(source_bindings)


def test_nextstat_subprocess_env_preserves_source_tree_pythonpath(monkeypatch):
    repo = _repo_root()
    source_bindings = repo / "bindings" / "ns-py" / "python"
    monkeypatch.setenv("PYTHONPATH", str(source_bindings))
    monkeypatch.delenv("NEXTSTAT_FORCE_PYTHONPATH", raising=False)

    env = _nextstat_subprocess_env(repo)
    assert env.get("PYTHONPATH") == str(source_bindings)
    assert env.get("NEXTSTAT_PREFER_INSTALLED") is None


def _run_apex2_report(*args: str) -> tuple[subprocess.CompletedProcess[str], dict]:
    repo = _repo_root()
    out_path = repo / "tmp" / "apex2_simplified_likelihood_report_smoke.json"
    out_path.unlink(missing_ok=True)
    proc = subprocess.run(
        [
            sys.executable,
            "tests/apex2_simplified_likelihood_report.py",
            "--out",
            str(out_path),
            *args,
        ],
        cwd=repo,
        env=_nextstat_subprocess_env(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    assert out_path.exists(), f"expected Apex2 report at {out_path}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return proc, json.loads(out_path.read_text(encoding="utf-8"))


def _build_nextstat_cli_binary() -> Path:
    repo = _repo_root()
    target_dir = Path("/tmp/nextstat_apex2_export_matrix_smoke_target")
    subprocess.run(
        ["cargo", "build", "-q", "-p", "ns-cli"],
        cwd=repo,
        env={**os.environ, "CARGO_TARGET_DIR": str(target_dir)},
        capture_output=True,
        text=True,
        check=True,
    )
    binary = target_dir / "debug" / "nextstat"
    assert binary.exists(), f"expected nextstat binary at {binary}"
    return binary


def test_apex2_simplified_likelihood_report_schema_example_and_runner_smoke():
    repo = _repo_root()
    schema_path = repo / "docs" / "schemas" / "apex2" / "simplified_likelihood_report_v0.schema.json"
    example_path = repo / "docs" / "specs" / "apex2_simplified_likelihood_report_v0.example.json"

    assert schema_path.exists(), f"missing schema: {schema_path}"
    assert example_path.exists(), f"missing example: {example_path}"

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    example = json.loads(example_path.read_text(encoding="utf-8"))
    assert schema.get("$schema"), "schema must declare $schema"
    assert schema.get("$id"), "schema must declare $id"
    assert example.get("schema_version") == "nextstat_apex2_simplified_likelihood_report_v0"

    try:
        import jsonschema  # type: ignore
    except Exception:
        jsonschema = None

    if jsonschema is not None:
        jsonschema.validate(instance=example, schema=schema)

    proc, generated = _run_apex2_report(
        "--suite",
        "smoke",
        "--fit-repeat",
        "1",
        "--upper-limit-repeat",
        "1",
        "--deterministic",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert generated["schema_version"] == "nextstat_apex2_simplified_likelihood_report_v0"
    assert generated["summary"]["case_count"] == 1
    assert generated["summary"]["all_schema_valid"] is True
    assert generated["summary"]["all_fidelity_gates_pass"] is True
    assert generated["cases"][0]["name"] == "synthetic_covariance_smoke"
    assert generated["cases"][0]["validation"]["runtime_audit_ok"] is True
    assert generated["cases"][0]["factorization"]["method"] == "symmetric_eigendecomposition"

    if jsonschema is not None:
        jsonschema.validate(instance=generated, schema=schema)


def test_apex2_simplified_likelihood_report_public_fixture_matrix_smoke():
    repo = _repo_root()
    schema_path = repo / "docs" / "schemas" / "apex2" / "simplified_likelihood_report_v0.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    try:
        import jsonschema  # type: ignore
    except Exception:
        jsonschema = None

    proc, generated = _run_apex2_report(
        "--suite",
        "smoke",
        "--fit-repeat",
        "1",
        "--upper-limit-repeat",
        "1",
        "--deterministic",
        "--include-public-fixtures",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    matrix = generated["public_fixture_matrix"]
    assert matrix["catalog_schema_version"] == "nextstat_simplified_likelihood_public_fixture_catalog_v0"
    assert matrix["summary"]["status"] == "ok"
    assert matrix["summary"]["all_schema_valid"] is True
    assert matrix["summary"]["all_runtime_gates_pass"] is True
    assert matrix["summary"]["all_derived_fidelity_gates_pass"] is True
    assert matrix["summary"]["fixture_count"] == 3
    assert matrix["summary"]["derived_fixture_count"] == 1
    assert matrix["summary"]["fixtures_with_embedded_fidelity_evidence"] == 1
    assert set(matrix["summary"]["source_formats"]) == {
        "basis",
        "covariance",
        "derived_from_workspace",
    }

    by_id = {case["fixture_id"]: case for case in matrix["cases"]}
    assert by_id["atlas_basis_two_bin_public_example"]["status"] == "ok"
    assert by_id["cms_covariance_three_bin_public_example"]["diagnostics"]["factorization"]["method"] == (
        "symmetric_eigendecomposition"
    )
    assert by_id["atlas_derived_workspace_public_example"]["evidence"] == {
        "full_vs_simplified_fidelity_supported": True,
        "embedded_fidelity_diagnostics_present": True,
    }
    assert by_id["atlas_derived_workspace_public_example"]["gates"]["derived_fidelity"] is True

    if jsonschema is not None:
        jsonschema.validate(instance=generated, schema=schema)


def test_apex2_simplified_likelihood_report_export_matrix_smoke():
    repo = _repo_root()
    schema_path = repo / "docs" / "schemas" / "apex2" / "simplified_likelihood_report_v0.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    nextstat_cli = _build_nextstat_cli_binary()

    try:
        import jsonschema  # type: ignore
    except Exception:
        jsonschema = None

    proc, generated = _run_apex2_report(
        "--suite",
        "smoke",
        "--fit-repeat",
        "1",
        "--upper-limit-repeat",
        "1",
        "--export-repeat",
        "1",
        "--deterministic",
        "--include-export-matrix",
        "--nextstat-cli",
        str(nextstat_cli),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    export_matrix = generated["export_matrix"]
    assert export_matrix["summary"]["status"] == "ok"
    assert export_matrix["summary"]["case_count"] == 1
    assert export_matrix["summary"]["all_schema_valid"] is True
    assert export_matrix["summary"]["all_fidelity_gates_pass"] is True
    assert export_matrix["summary"]["all_performance_gates_pass"] is True
    assert export_matrix["summary"]["min_net_end_to_end_upper_limit_speedup"] >= 0.0

    case = export_matrix["cases"][0]
    assert case["name"] == "synthetic_covariance_smoke"
    assert case["export_report_schema_version"] == "nextstat_simplified_likelihood_export_report_v0"
    assert case["validation"]["runtime_export_ok"] is True
    assert case["validation"]["export_report_schema_valid"] is True
    assert case["output"]["schema_version"] == "nextstat_simplified_likelihood_v0"
    assert case["output"]["uncertainty_model_kind"] == "basis"

    if jsonschema is not None:
        jsonschema.validate(instance=generated, schema=schema)


def test_apex2_simplified_likelihood_report_export_matrix_ci_suite_preserves_fidelity():
    nextstat_cli = _build_nextstat_cli_binary()

    proc, generated = _run_apex2_report(
        "--suite",
        "ci",
        "--fit-repeat",
        "1",
        "--upper-limit-repeat",
        "1",
        "--export-repeat",
        "1",
        "--deterministic",
        "--include-export-matrix",
        "--nextstat-cli",
        str(nextstat_cli),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    export_matrix = generated["export_matrix"]
    assert export_matrix["summary"]["case_count"] == 2
    assert export_matrix["summary"]["all_schema_valid"] is True
    assert export_matrix["summary"]["all_fidelity_gates_pass"] is True

    by_name = {case["name"]: case for case in export_matrix["cases"]}
    assert by_name["synthetic_covariance_smoke"]["status"] == "ok"
    assert by_name["synthetic_covariance_medium"]["status"] == "ok"
    assert by_name["synthetic_covariance_medium"]["fidelity"]["max_abs_q_mu_diff"] <= 0.1


def test_apex2_simplified_likelihood_report_export_matrix_with_public_cases_smoke():
    repo = _repo_root()
    schema_path = repo / "docs" / "schemas" / "apex2" / "simplified_likelihood_report_v0.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    nextstat_cli = _build_nextstat_cli_binary()

    try:
        import jsonschema  # type: ignore
    except Exception:
        jsonschema = None

    proc, generated = _run_apex2_report(
        "--suite",
        "smoke",
        "--fit-repeat",
        "1",
        "--upper-limit-repeat",
        "1",
        "--export-repeat",
        "1",
        "--deterministic",
        "--include-export-matrix",
        "--include-export-public-cases",
        "--nextstat-cli",
        str(nextstat_cli),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr

    export_matrix = generated["export_matrix"]
    summary = export_matrix["summary"]
    assert summary["status"] == "ok"
    assert summary["case_count"] == 8
    assert summary["synthetic_case_count"] >= 1
    assert summary["public_reinterpretation_style_case_count"] == 7
    assert set(summary["case_kinds"]) == {"synthetic", "public_reinterpretation_style"}

    public_cases = [
        case for case in export_matrix["cases"] if case["case_kind"] == "public_reinterpretation_style"
    ]
    assert public_cases
    assert {case["name"] for case in public_cases} == {
        "atlas_public_dual_sr_vr_dual_cr_gaussian_export_stable_example",
        "atlas_public_sr_cr_gaussian_export_stable_example",
        "atlas_public_dual_sr_dual_cr_gaussian_export_stable_example",
        "cms_public_sr_cr_asymmetric_gaussian_export_stable_example",
        "cms_public_dual_sr_cr_gaussian_export_stable_example",
        "cms_public_dual_sr_vr_cr_gaussian_export_stable_example",
        "cms_public_sr_cr_export_stable_example",
    }
    assert all(case["source_workspace_path"].endswith(".json") for case in public_cases)
    assert {case["experiment"] for case in public_cases} == {"ATLAS", "CMS"}
    assert all(case["validation"]["runtime_export_ok"] is True for case in public_cases)
    assert all(case["output"]["schema_version"] == "nextstat_simplified_likelihood_v0" for case in public_cases)

    if jsonschema is not None:
        jsonschema.validate(instance=generated, schema=schema)
