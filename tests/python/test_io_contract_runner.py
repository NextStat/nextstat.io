from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _report_schema_path() -> Path:
    return (
        _repo_root()
        / "docs"
        / "schemas"
        / "io"
        / "nextstat_io_contract_runner_report_v1.schema.json"
    )


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_runner_module():
    module_path = _repo_root() / "scripts" / "check_io_contracts.py"
    spec = importlib.util.spec_from_file_location("check_io_contracts", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_report(report: dict) -> None:
    schema = json.loads(_report_schema_path().read_text(encoding="utf-8"))
    try:
        import jsonschema  # type: ignore
    except Exception:
        return
    jsonschema.validate(instance=report, schema=schema)


def _runner_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("NEXTSTAT_HEPDATA_CMD", None)
    env.pop("NEXTSTAT_IO_CONTRACT_CARGO_TARGET_DIR", None)
    return env


def test_io_contract_runner_dry_run_lists_expected_hepdata_steps():
    proc = subprocess.run(
        [sys.executable, "scripts/check_io_contracts.py", "--family", "hepdata", "--dry-run"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        env=_runner_env(),
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    stdout = proc.stdout
    assert "[io-contracts] family=hepdata" in stdout
    assert "cargo build -p ns-cli" in stdout
    assert "scripts/generate_hepdata_schema_examples.py --check" in stdout
    assert "tests/python/test_hepdata_schema_smoke.py" in stdout
    assert "NEXTSTAT_HEPDATA_SKIP_GENERATOR_CHECK=1" in stdout


def test_io_contract_runner_dry_run_lists_expected_histograms_parquet_steps():
    proc = subprocess.run(
        [sys.executable, "scripts/check_io_contracts.py", "--family", "histograms_parquet", "--dry-run"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        env=_runner_env(),
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    stdout = proc.stdout
    assert "[io-contracts] family=histograms_parquet" in stdout
    assert "scripts/generate_histograms_parquet_schema_examples.py --check" in stdout
    assert "tests/python/test_histograms_parquet_manifest_schema_smoke.py" in stdout
    assert "NEXTSTAT_HISTOGRAMS_PARQUET_SKIP_GENERATOR_CHECK=1" in stdout
    assert "cargo build -p ns-cli" not in stdout


def test_io_contract_runner_dry_run_all_lists_both_families():
    proc = subprocess.run(
        [sys.executable, "scripts/check_io_contracts.py", "--family", "all", "--dry-run"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        env=_runner_env(),
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    stdout = proc.stdout
    assert "[io-contracts] family=all" in stdout
    assert "scripts/generate_hepdata_schema_examples.py --check" in stdout
    assert "scripts/generate_histograms_parquet_schema_examples.py --check" in stdout


def test_io_contract_runner_dry_run_writes_report(tmp_path: Path):
    report_path = tmp_path / "io_contracts_hepdata_report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/check_io_contracts.py",
            "--family",
            "hepdata",
            "--dry-run",
            "--report-json",
            str(report_path),
        ],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        env=_runner_env(),
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert report_path.exists(), "runner must write the requested JSON report"

    report = _load_report(report_path)
    _validate_report(report)

    assert report["schema_version"] == "nextstat.io_contract_runner_report.v1"
    assert report["family"] == "hepdata"
    assert report["dry_run"] is True
    assert report["status"] == "planned"
    assert report["overall_pass"] is True
    assert report["failed_step_index"] is None
    assert report["failed_step_label"] is None
    assert report["step_count"] == len(report["steps"])
    assert report["started_at"].endswith("Z")
    assert report["finished_at"].endswith("Z")
    assert report["duration_s"] >= 0

    steps = report["steps"]
    assert steps, "runner report must contain planned steps"
    assert all(step["status"] == "planned" for step in steps)
    assert all(step["returncode"] == 0 for step in steps)
    assert all(step["started_at"] is None for step in steps)
    assert all(step["finished_at"] is None for step in steps)
    assert all(step["duration_s"] is None for step in steps)
    assert all(step.get("stdout_tail") is None for step in steps)
    assert all(step.get("stderr_tail") is None for step in steps)

    assert steps[0]["label"] == "Build isolated ns-cli binary for IO contracts"
    assert steps[0]["env_overrides"]["CARGO_TARGET_DIR"].endswith(".nextstat-cargo-target/io-contracts")
    assert steps[1]["label"] == "Check HEPData schema examples"
    assert steps[2]["label"] == "Run HEPData schema smoke suite"
    assert steps[2]["env_overrides"]["NEXTSTAT_HEPDATA_SKIP_GENERATOR_CHECK"] == "1"
    assert "scripts/generate_hepdata_schema_examples.py --check" in steps[1]["command"]


def test_io_contract_runner_captures_failure_output_tails(capsys):
    module = _load_runner_module()
    step = module.Step(
        label="Failing io smoke step",
        argv=[
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('hello from stdout'); "
                "print('boom from stderr', file=sys.stderr); "
                "raise SystemExit(4)"
            ),
        ],
    )

    result = module._run_step(step, dry_run=False, index=1)
    captured = capsys.readouterr()

    assert result.status == "failed"
    assert result.returncode == 4
    assert result.stdout_tail == "hello from stdout"
    assert result.stderr_tail == "boom from stderr"
    assert "hello from stdout" in captured.out
    assert "boom from stderr" in captured.err
