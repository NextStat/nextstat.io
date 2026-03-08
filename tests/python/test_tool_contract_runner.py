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
        / "tools"
        / "nextstat_tool_contract_runner_report_v1.schema.json"
    )


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_runner_module():
    module_path = _repo_root() / "scripts" / "check_tool_contracts.py"
    spec = importlib.util.spec_from_file_location("check_tool_contracts", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expected_tool_contract_cargo_target_dir() -> str:
    override = os.environ.get("NEXTSTAT_TOOL_CONTRACT_CARGO_TARGET_DIR")
    if override:
        return override
    return str(_repo_root().parent / ".nextstat-cargo-target" / "tool-contracts")


def _expected_tool_contract_bindings_cargo_target_dir() -> str:
    override = os.environ.get("NEXTSTAT_TOOL_CONTRACT_BINDINGS_CARGO_TARGET_DIR")
    if override:
        return override
    return str(_repo_root().parent / ".nextstat-cargo-target" / "tool-contracts-bindings")


def _validate_report(report: dict) -> None:
    schema = json.loads(_report_schema_path().read_text(encoding="utf-8"))
    try:
        import jsonschema  # type: ignore
    except Exception:
        return
    jsonschema.validate(instance=report, schema=schema)


def test_tool_contract_runner_dry_run_fast_lists_expected_steps():
    proc = subprocess.run(
        [sys.executable, "scripts/check_tool_contracts.py", "--mode", "fast", "--dry-run"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    stdout = proc.stdout
    assert "[tool-contracts] mode=fast" in stdout
    assert "scripts/generate_tool_contract_schemas.py --check" in stdout
    assert "scripts/validate_tool_manifest.py" in stdout
    assert "scripts/validate_tool_schema_descriptor.py --transport local" in stdout
    assert "scripts/generate_tool_schema_examples.py --check" in stdout
    assert "scripts/generate_tool_reference_docs.py --check" in stdout
    assert "scripts/generate_agent_bootstrap_profile_manifest_schema.py --check" in stdout
    assert "-m scripts.generate_agent_bootstrap_packs --check" in stdout
    assert "scripts/generate_tool_goldens.py --check" in stdout
    assert "cargo test -p ns-server -- --test-threads=1" in stdout
    assert "-m maturin develop -m bindings/ns-py/Cargo.toml" in stdout
    assert "internal:validate-performance-budgets" in stdout
    assert "tests/python/test_agent_bootstrap_packs.py" in stdout
    assert "tests/python/test_tool_contract_runner.py" in stdout


def test_tool_contract_runner_dry_run_live_sets_live_env():
    proc = subprocess.run(
        [sys.executable, "scripts/check_tool_contracts.py", "--mode", "live", "--dry-run"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    stdout = proc.stdout
    assert "[tool-contracts] mode=live" in stdout
    assert "NS_RUN_LIVE_SERVER=1" in stdout
    assert "tests/python/test_tools_live_server_integration.py" in stdout


def test_tool_contract_runner_dry_run_fast_writes_report(tmp_path: Path):
    report_path = tmp_path / "tool_contracts_fast_report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/check_tool_contracts.py",
            "--mode",
            "fast",
            "--dry-run",
            "--report-json",
            str(report_path),
        ],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert report_path.exists(), "runner must write the requested JSON report"

    report = _load_report(report_path)
    _validate_report(report)

    assert report["schema_version"] == "nextstat.tool_contract_runner_report.v1"
    assert report["mode"] == "fast"
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
    cargo_steps = [step for step in steps if step["label"] == "Run ns-server tool contract tests"]
    assert len(cargo_steps) == 1
    assert cargo_steps[0]["env_overrides"]["CARGO_TARGET_DIR"] == _expected_tool_contract_cargo_target_dir()
    bindings_steps = [
        step for step in steps if step["label"] == "Sync nextstat Python bindings into active environment"
    ]
    assert len(bindings_steps) == 1
    assert (
        bindings_steps[0]["env_overrides"]["CARGO_TARGET_DIR"]
        == _expected_tool_contract_bindings_cargo_target_dir()
    )
    bootstrap_steps = [step for step in steps if step["label"] == "Check agent bootstrap packs"]
    assert len(bootstrap_steps) == 1
    assert "-m scripts.generate_agent_bootstrap_packs --check" in bootstrap_steps[0]["command"]
    manifest_schema_steps = [
        step for step in steps if step["label"] == "Check agent bootstrap profile manifest schema"
    ]
    assert len(manifest_schema_steps) == 1
    assert (
        "scripts/generate_agent_bootstrap_profile_manifest_schema.py --check"
        in manifest_schema_steps[0]["command"]
    )
    assert steps[0]["label"] == "Check tool contract schemas"
    assert steps[-1]["label"] == "Validate tool-contract performance budgets"
    assert "scripts/generate_tool_contract_schemas.py" in steps[0]["command"]
    assert "-m maturin develop -m bindings/ns-py/Cargo.toml" in bindings_steps[0]["command"]
    assert steps[-1]["command"] == "internal:validate-performance-budgets"
    assert report["performance"]["budget_schema_version"] == "nextstat.tool_contract_performance_budget.v1"
    assert report["performance"]["runner_budget"]["status"] == "planned"
    assert report["performance"]["live_metrics_budget"]["status"] == "not_available"


def test_tool_contract_runner_dry_run_live_writes_report(tmp_path: Path):
    report_path = tmp_path / "tool_contracts_live_report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/check_tool_contracts.py",
            "--mode",
            "live",
            "--dry-run",
            "--report-json",
            str(report_path),
        ],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert report_path.exists(), "runner must write the requested JSON report"

    report = _load_report(report_path)
    _validate_report(report)

    assert report["mode"] == "live"
    assert report["dry_run"] is True
    assert report["status"] == "planned"
    assert report["step_count"] == 2
    step = report["steps"][0]
    assert step["status"] == "planned"
    assert step["env_overrides"]["NS_RUN_LIVE_SERVER"] == "1"
    assert "NEXTSTAT_TOOL_CONTRACT_LIVE_METRICS_PATH" in step["env_overrides"]
    assert step["argv"][-1] == "tests/python/test_tools_live_server_integration.py"
    perf_step = report["steps"][1]
    assert perf_step["label"] == "Validate tool-contract performance budgets"
    assert perf_step["status"] == "planned"
    assert report["performance"]["runner_budget"]["status"] == "planned"
    assert report["performance"]["live_metrics_budget"]["status"] == "planned"
    assert step.get("stdout_tail") is None
    assert step.get("stderr_tail") is None


def test_tool_contract_runner_captures_failure_output_tails(capsys):
    module = _load_runner_module()
    step = module.Step(
        label="Failing smoke step",
        argv=[
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('hello from stdout'); "
                "print('boom from stderr', file=sys.stderr); "
                "raise SystemExit(3)"
            ),
        ],
    )

    result = module._run_step(step, dry_run=False, index=1)
    captured = capsys.readouterr()

    assert result.status == "failed"
    assert result.returncode == 3
    assert result.stdout_tail == "hello from stdout"
    assert result.stderr_tail == "boom from stderr"
    assert "hello from stdout" in captured.out
    assert "boom from stderr" in captured.err
