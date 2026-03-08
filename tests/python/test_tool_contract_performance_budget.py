from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _schema_path() -> Path:
    return (
        _repo_root()
        / "docs"
        / "schemas"
        / "tools"
        / "nextstat_tool_contract_performance_budget_v1.schema.json"
    )


def _load_helper():
    module_path = _repo_root() / "scripts" / "tool_contract_performance_budget.py"
    spec = importlib.util.spec_from_file_location("tool_contract_performance_budget", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tool_contract_performance_budget_manifest_smoke():
    helper = _load_helper()
    manifest = helper.load_tool_contract_performance_budget()

    assert manifest["schema_version"] == "nextstat.tool_contract_performance_budget.v1"
    assert manifest["runner_modes"]["fast"]["max_total_duration_s"] > 0
    assert manifest["runner_modes"]["live"]["max_total_duration_s"] > 0
    assert manifest["runner_modes"]["all"]["max_total_duration_s"] > 0
    assert "Check agent bootstrap packs" in manifest["runner_modes"]["fast"]["steps"]
    assert "server_build_duration_s" in manifest["live_metrics"]
    assert "fit_duration_s" in manifest["live_metrics"]

    try:
        import jsonschema  # type: ignore
    except Exception:
        jsonschema = None

    if jsonschema is not None:
        schema = json.loads(_schema_path().read_text(encoding="utf-8"))
        jsonschema.validate(instance=manifest, schema=schema)

    proc = subprocess.run(
        [sys.executable, "scripts/tool_contract_performance_budget.py", "--format", "json"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    printed = json.loads(proc.stdout)
    assert printed == manifest
