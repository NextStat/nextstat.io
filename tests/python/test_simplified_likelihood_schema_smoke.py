import json
import os
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_simplified_likelihood_schemas_examples_and_cli_audit_smoke(tmp_path: Path):
    repo = _repo_root()
    cases = [
        (
            repo / "docs" / "schemas" / "hep" / "simplified_likelihood_v0.schema.json",
            [
                repo / "docs" / "specs" / "hep" / "simplified_likelihood_v0.example.json",
                repo / "docs" / "specs" / "hep" / "simplified_likelihood_derived_from_workspace_v0.example.json",
                repo / "tests" / "fixtures" / "sl_basis_two_bin.json",
                repo / "tests" / "fixtures" / "sl_covariance_three_bin.json",
            ],
            "nextstat_simplified_likelihood_v0",
        ),
        (
            repo / "docs" / "schemas" / "hep" / "simplified_likelihood_audit_v0.schema.json",
            [repo / "docs" / "specs" / "hep" / "simplified_likelihood_audit_v0.example.json"],
            "nextstat_simplified_likelihood_audit_v0",
        ),
        (
            repo / "docs" / "schemas" / "hep" / "simplified_likelihood_derive_v0.schema.json",
            [repo / "docs" / "specs" / "hep" / "simplified_likelihood_derive_v0.example.json"],
            "nextstat_simplified_likelihood_derive_v0",
        ),
        (
            repo / "docs" / "schemas" / "hep" / "simplified_likelihood_export_report_v0.schema.json",
            [
                repo / "docs" / "specs" / "hep" / "simplified_likelihood_export_report_v0.example.json"
            ],
            "nextstat_simplified_likelihood_export_report_v0",
        ),
    ]

    try:
        import jsonschema  # type: ignore
    except Exception:
        jsonschema = None

    for schema_path, example_paths, expected_version in cases:
        assert schema_path.exists(), f"missing schema: {schema_path}"
        schema = json.loads(schema_path.read_text())
        assert schema.get("$schema"), "schema must declare $schema"
        assert schema.get("$id"), "schema must declare $id"
        assert schema.get("type") == "object"

        for example_path in example_paths:
            assert example_path.exists(), f"missing example: {example_path}"
            example = json.loads(example_path.read_text())
            assert example.get("schema_version") == expected_version
            if jsonschema is not None:
                jsonschema.validate(instance=example, schema=schema)

    if jsonschema is not None:
        audit_schema = json.loads(
            (repo / "docs" / "schemas" / "hep" / "simplified_likelihood_audit_v0.schema.json")
            .read_text()
        )
        proc = subprocess.run(
            [
                "cargo",
                "run",
                "-q",
                "-p",
                "ns-cli",
                "--",
                "audit",
                "--input",
                str(repo / "tests" / "fixtures" / "sl_covariance_three_bin.json"),
                "--format",
                "json",
            ],
            cwd=repo,
            env={
                **os.environ,
                "CARGO_TARGET_DIR": "/tmp/nextstat_sl_surface_target",
            },
            capture_output=True,
            text=True,
            check=True,
        )
        audit = json.loads(proc.stdout)
        assert audit["schema_version"] == "nextstat_simplified_likelihood_audit_v0"
        assert audit["input_schema_version"] == "nextstat_simplified_likelihood_v0"
        jsonschema.validate(instance=audit, schema=audit_schema)

        export_report_schema = json.loads(
            (
                repo
                / "docs"
                / "schemas"
                / "hep"
                / "simplified_likelihood_export_report_v0.schema.json"
            ).read_text()
        )
        fit_path = tmp_path / "fit.json"
        derive_config_path = tmp_path / "derive.json"
        simplified_path = tmp_path / "simplified.json"
        report_path = tmp_path / "simplified_export_report.json"
        fit_proc = subprocess.run(
            [
                "cargo",
                "run",
                "-q",
                "-p",
                "ns-cli",
                "--",
                "fit",
                "--input",
                str(repo / "tests" / "fixtures" / "simple_workspace.json"),
                "--output",
                str(fit_path),
                "--threads",
                "1",
            ],
            cwd=repo,
            env={
                **os.environ,
                "CARGO_TARGET_DIR": "/tmp/nextstat_sl_surface_target",
            },
            capture_output=True,
            text=True,
            check=True,
        )
        assert fit_proc.returncode == 0
        derive_config_path.write_text(
            json.dumps(
                {
                    "schema_version": "nextstat_simplified_likelihood_derive_v0",
                    "source_workspace": {
                        "format": "pyhf",
                        "schema_version": "pyhf_workspace_v1",
                        "poi_name": "mu",
                    },
                    "fit_result": {
                        "schema_version": "nextstat_fit_result_v0",
                        "background_state": "postfit_background",
                    },
                    "selection": {
                        "channels": ["singlechannel"],
                        "bins": ["singlechannel/bin0", "singlechannel/bin1"],
                    },
                    "reduction": {
                        "output_uncertainty_model": "basis",
                        "basis_method": "eigen",
                        "explained_variance_target": 0.9,
                        "constraint_covariance_source": "aligned_fit_covariance",
                        "max_components": 1,
                        "split_stat_covariance": True,
                    },
                    "jacobian": {
                        "method": "finite_difference",
                        "relative_step": 0.01,
                        "absolute_step_floor": 0.000001,
                    },
                    "fidelity_smoke": {
                        "random_draws": 8,
                        "qmu_test_mu": 1.0,
                        "upper_limit_cl": 0.95,
                    },
                    "output_contract": {
                        "schema_version": "nextstat_simplified_likelihood_v0",
                        "require_factorization_diagnostics": True,
                        "require_fidelity_diagnostics": True,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        export_proc = subprocess.run(
            [
                "cargo",
                "run",
                "-q",
                "-p",
                "ns-cli",
                "--",
                "simplify",
                "workspace",
                "--input",
                str(repo / "tests" / "fixtures" / "simple_workspace.json"),
                "--fit",
                str(fit_path),
                "--derive-config",
                str(derive_config_path),
                "--experiment",
                "ATLAS",
                "--analysis-id",
                "pytest-export-report",
                "--reference",
                "internal-test",
                "--output",
                str(simplified_path),
                "--report",
                str(report_path),
                "--threads",
                "1",
            ],
            cwd=repo,
            env={
                **os.environ,
                "CARGO_TARGET_DIR": "/tmp/nextstat_sl_surface_target",
            },
            capture_output=True,
            text=True,
            check=True,
        )
        assert export_proc.returncode == 0
        export_report = json.loads(report_path.read_text())
        assert export_report["schema_version"] == "nextstat_simplified_likelihood_export_report_v0"
        assert export_report["output"]["schema_version"] == "nextstat_simplified_likelihood_v0"
        jsonschema.validate(instance=export_report, schema=export_report_schema)
