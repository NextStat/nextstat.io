from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dashboard_schema_path() -> Path:
    return (
        _repo_root()
        / "docs"
        / "schemas"
        / "tools"
        / "nextstat_tool_contract_dashboard_v1.schema.json"
    )


def _load_dashboard_module():
    module_path = _repo_root() / "scripts" / "summarize_tool_contract_reports.py"
    spec = importlib.util.spec_from_file_location("summarize_tool_contract_reports", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_dashboard(payload: dict) -> None:
    schema = json.loads(_dashboard_schema_path().read_text(encoding="utf-8"))
    try:
        import jsonschema  # type: ignore
    except Exception:
        return
    jsonschema.validate(instance=payload, schema=schema)


def test_tool_contract_dashboard_aggregates_dry_run_reports(tmp_path: Path):
    fast_report = tmp_path / "tool_contracts_fast_report.json"
    live_report = tmp_path / "tool_contracts_live_report.json"
    dashboard_json = tmp_path / "tool_contract_dashboard.json"
    dashboard_md = tmp_path / "tool_contract_dashboard.md"

    for mode, path in (("fast", fast_report), ("live", live_report)):
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/check_tool_contracts.py",
                "--mode",
                mode,
                "--dry-run",
                "--report-json",
                str(path),
            ],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr or proc.stdout
        assert path.exists(), f"runner must write {mode} report"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_tool_contract_reports.py",
            "--report",
            str(fast_report),
            "--report",
            str(live_report),
            "--out-json",
            str(dashboard_json),
            "--out-md",
            str(dashboard_md),
        ],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert dashboard_json.exists(), "dashboard JSON must be written"
    assert dashboard_md.exists(), "dashboard markdown must be written"

    dashboard = json.loads(dashboard_json.read_text(encoding="utf-8"))
    _validate_dashboard(dashboard)

    assert dashboard["schema_version"] == "nextstat.tool_contract_dashboard.v1"
    assert dashboard["report_count"] == 2
    assert dashboard["overall_status"] == "planned"
    assert dashboard["overall_pass"] is True
    assert dashboard["source_reports"] == [str(fast_report), str(live_report)]
    assert len(dashboard["runs"]) == 2
    assert dashboard["totals"]["mode_counts"] == {"fast": 1, "live": 1, "all": 0}
    assert dashboard["totals"]["step_status_counts"]["failed"] == 0
    assert dashboard["totals"]["step_status_counts"]["passed"] == 0
    assert dashboard["totals"]["step_status_counts"]["planned"] == 14
    assert dashboard["totals"]["step_count"] == 14
    assert dashboard["totals"]["failed_report_paths"] == []
    assert dashboard["totals"]["failed_step_count"] == 0
    assert dashboard["totals"]["failure_classification_counts"] == {
        "none": 2,
        "schema_drift": 0,
        "performance_budget_failure": 0,
        "rust_contract_failure": 0,
        "python_contract_failure": 0,
        "live_server_failure": 0,
        "unknown": 0,
    }
    assert dashboard["totals"]["severity_counts"] == {"none": 2, "high": 0, "critical": 0}
    assert dashboard["totals"]["runner_budget_status_counts"] == {
        "planned": 2,
        "within_budget": 0,
        "exceeded": 0,
        "not_available": 0,
    }
    assert dashboard["totals"]["live_metrics_budget_status_counts"] == {
        "planned": 1,
        "within_budget": 0,
        "exceeded": 0,
        "not_available": 1,
    }
    assert all(run["failed_steps"] == [] for run in dashboard["runs"])
    assert all(run["failure_classification"]["code"] == "none" for run in dashboard["runs"])
    assert all(run["failure_classification"]["severity"] == "none" for run in dashboard["runs"])
    assert all("performance" in run for run in dashboard["runs"])

    md = dashboard_md.read_text(encoding="utf-8")
    assert "# Tool Contract Dashboard" in md
    assert "overall status: `planned`" in md
    assert f"`{fast_report}`" in md
    assert f"`{live_report}`" in md
    assert "`fast=1`" in md
    assert "`live=1`" in md
    assert "`none=2`" in md
    assert "## Performance" in md


def test_tool_contract_dashboard_classifies_failed_step_labels():
    module = _load_dashboard_module()

    schema_drift = module._classify_failed_step({"label": "Check tool contract schemas"})
    rust_failure = module._classify_failed_step({"label": "Run ns-server tool contract tests"})
    budget_failure = module._classify_failed_step({"label": "Validate tool-contract performance budgets"})
    python_failure = module._classify_failed_step({"label": "Run fast Python tool contract suite"})
    live_failure = module._classify_failed_step({"label": "Run live nextstat-server tool contract suite"})
    unknown = module._classify_failed_step({"label": "Some future step"})

    assert schema_drift["code"] == "schema_drift"
    assert schema_drift["severity"] == "high"
    assert budget_failure["code"] == "performance_budget_failure"
    assert budget_failure["severity"] == "high"
    assert rust_failure["code"] == "rust_contract_failure"
    assert rust_failure["severity"] == "critical"
    assert python_failure["code"] == "python_contract_failure"
    assert python_failure["severity"] == "high"
    assert live_failure["code"] == "live_server_failure"
    assert live_failure["severity"] == "critical"
    assert unknown["code"] == "unknown"
    assert unknown["severity"] == "high"


def test_tool_contract_dashboard_includes_failure_drilldown(tmp_path: Path):
    failed_report = tmp_path / "tool_contracts_failed_report.json"
    dashboard_json = tmp_path / "tool_contract_dashboard_failed.json"
    dashboard_md = tmp_path / "tool_contract_dashboard_failed.md"

    failed_report.write_text(
        json.dumps(
            {
                "schema_version": "nextstat.tool_contract_runner_report.v1",
                "mode": "fast",
                "dry_run": False,
                "status": "failed",
                "overall_pass": False,
                "repo_root": str(_repo_root()),
                "python_executable": sys.executable,
                "started_at": "2026-03-08T12:00:00Z",
                "finished_at": "2026-03-08T12:00:05Z",
                "duration_s": 5.0,
                "step_count": 2,
                "failed_step_index": 2,
                "failed_step_label": "Run fast Python tool contract suite",
                "performance": {
                    "budget_manifest_path": str(_repo_root() / "scripts" / "tool_contract_performance_budget_v1.json"),
                    "budget_schema_path": str(
                        _repo_root()
                        / "docs"
                        / "schemas"
                        / "tools"
                        / "nextstat_tool_contract_performance_budget_v1.schema.json"
                    ),
                    "budget_schema_version": "nextstat.tool_contract_performance_budget.v1",
                    "runner_budget": {
                        "mode": "fast",
                        "status": "within_budget",
                        "max_total_duration_s": 1200.0,
                        "actual_total_duration_s": 5.0,
                        "step_budgets": [],
                    },
                    "live_metrics_budget": {
                        "status": "not_available",
                        "metrics_path": None,
                        "missing_metrics": [],
                        "metrics": [],
                    },
                },
                "steps": [
                    {
                        "index": 1,
                        "label": "Check tool contract schemas",
                        "argv": [sys.executable, "scripts/generate_tool_contract_schemas.py", "--check"],
                        "command": f"{sys.executable} scripts/generate_tool_contract_schemas.py --check",
                        "env_overrides": {},
                        "status": "passed",
                        "returncode": 0,
                        "started_at": "2026-03-08T12:00:00Z",
                        "finished_at": "2026-03-08T12:00:01Z",
                        "duration_s": 1.0,
                        "stdout_tail": None,
                        "stderr_tail": None,
                    },
                    {
                        "index": 2,
                        "label": "Run fast Python tool contract suite",
                        "argv": [sys.executable, "-m", "pytest", "-q", "tests/python/test_tools_contract_runtime.py"],
                        "command": f"{sys.executable} -m pytest -q tests/python/test_tools_contract_runtime.py",
                        "env_overrides": {"NS_RUN_LIVE_SERVER": "0"},
                        "status": "failed",
                        "returncode": 2,
                        "started_at": "2026-03-08T12:00:01Z",
                        "finished_at": "2026-03-08T12:00:05Z",
                        "duration_s": 4.0,
                        "stdout_tail": "FAILED tests/python/test_tools_contract_runtime.py::test_example",
                        "stderr_tail": "AssertionError: expected 42, got 0",
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_tool_contract_reports.py",
            "--report",
            str(failed_report),
            "--out-json",
            str(dashboard_json),
            "--out-md",
            str(dashboard_md),
        ],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout

    dashboard = json.loads(dashboard_json.read_text(encoding="utf-8"))
    _validate_dashboard(dashboard)

    assert dashboard["overall_status"] == "failed"
    assert dashboard["overall_pass"] is False
    assert dashboard["totals"]["failed_report_paths"] == [str(failed_report)]
    assert dashboard["totals"]["failed_step_count"] == 1
    assert dashboard["totals"]["failure_classification_counts"]["python_contract_failure"] == 1
    assert dashboard["totals"]["severity_counts"]["high"] == 1
    run = dashboard["runs"][0]
    assert run["failure_classification"]["code"] == "python_contract_failure"
    assert run["failure_classification"]["severity"] == "high"
    assert run["failed_steps"][0]["label"] == "Run fast Python tool contract suite"
    assert run["failed_steps"][0]["classification"]["code"] == "python_contract_failure"
    assert run["failed_steps"][0]["stdout_tail"] == "FAILED tests/python/test_tools_contract_runtime.py::test_example"
    assert run["failed_steps"][0]["stderr_tail"] == "AssertionError: expected 42, got 0"

    md = dashboard_md.read_text(encoding="utf-8")
    assert "## Failure Drilldown" in md
    assert "`python_contract_failure`" in md
    assert "severity=`high`" in md
    assert "Run fast Python tool contract suite" in md
    assert "FAILED tests/python/test_tools_contract_runtime.py::test_example" in md
    assert "AssertionError: expected 42, got 0" in md
