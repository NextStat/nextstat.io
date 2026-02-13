import json
import subprocess
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_schema(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_unbinned_toy_parity_schemas_smoke(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)

    report_cpu = {
        "schema_version": "nextstat.unbinned_toy_parity_report.v1",
        "kind": "api_cli",
        "backend": "cpu",
        "ok": True,
        "n_toys": 8,
        "seed": 123,
        "metrics": {
            "compared_nll": 8,
            "max_abs_delta_nll": 1e-9,
            "max_abs_delta_poi": 1e-9,
        },
        "thresholds": {
            "nll_abs_tol": 1e-6,
            "poi_abs_tol": 1e-6,
        },
    }
    report_gpu = {
        "schema_version": "nextstat.unbinned_toy_parity_report.v1",
        "kind": "cli_gpu",
        "backend": "cuda",
        "ok": True,
        "n_toys": 8,
        "seed": 333,
        "metrics": {
            "compared_pairs": 8,
            "convergence_mismatch_count": 0,
            "max_abs_delta_nll": 1e-6,
            "max_abs_delta_poi": 1e-4,
        },
        "thresholds": {
            "nll_abs_tol": 1e-6,
            "poi_abs_tol": 2e-4,
            "max_convergence_mismatch": 1,
        },
    }

    p_cpu = reports_dir / "api_cli_cpu.json"
    p_cuda = reports_dir / "cli_gpu_cuda.json"
    p_cpu.write_text(json.dumps(report_cpu, indent=2), encoding="utf-8")
    p_cuda.write_text(json.dumps(report_gpu, indent=2), encoding="utf-8")

    input_schema = _load_schema(
        _repo_root() / "docs" / "schemas" / "benchmarks" / "unbinned_toy_parity_report_v1.schema.json"
    )
    jsonschema.validate(report_cpu, input_schema)
    jsonschema.validate(report_gpu, input_schema)

    out = tmp_path / "matrix.json"
    subprocess.check_call(
        [
            "python3",
            str(_repo_root() / "scripts" / "benchmarks" / "aggregate_unbinned_toy_parity_reports.py"),
            "--reports-dir",
            str(reports_dir),
            "--out",
            str(out),
        ]
    )

    matrix = json.loads(out.read_text(encoding="utf-8"))
    matrix_schema = _load_schema(
        _repo_root()
        / "docs"
        / "schemas"
        / "benchmarks"
        / "unbinned_toy_parity_matrix_report_v1.schema.json"
    )
    jsonschema.validate(matrix, matrix_schema)

    assert matrix["schema_version"] == "nextstat.unbinned_toy_parity_matrix_report.v1"
    assert matrix["n_reports"] == 2
    assert matrix["backends"]["cpu"]["kinds"]["api_cli"]["backend"] == "cpu"
    assert matrix["backends"]["cuda"]["kinds"]["cli_gpu"]["backend"] == "cuda"
