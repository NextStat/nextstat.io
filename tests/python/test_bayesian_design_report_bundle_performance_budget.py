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
        / "benchmarks"
        / "nextstat_bayesian_design_report_bundle_performance_budget_v1.schema.json"
    )


def _load_helper():
    module_path = _repo_root() / "scripts" / "bayesian_design_report_bundle_performance_budget.py"
    spec = importlib.util.spec_from_file_location(
        "bayesian_design_report_bundle_performance_budget",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bayesian_design_report_bundle_performance_budget_manifest_smoke() -> None:
    helper = _load_helper()
    manifest = helper.load_bayesian_design_report_bundle_performance_budget()

    assert manifest["schema_version"] == "nextstat.bayesian_design_report_bundle_performance_budget.v1"
    assert manifest["runner_modes"]["smoke"]["repeat"] >= 1
    assert manifest["runner_modes"]["release"]["repeat"] >= 1
    assert manifest["runner_modes"]["smoke"]["manifest_repeat"] >= 1
    assert manifest["runner_modes"]["release"]["manifest_repeat"] >= 1
    assert {"beta_small", "beta_large", "normal_small", "normal_large"} <= set(manifest["cases"])

    try:
        import jsonschema  # type: ignore
    except Exception:
        jsonschema = None

    if jsonschema is not None:
        schema = json.loads(_schema_path().read_text(encoding="utf-8"))
        jsonschema.validate(instance=manifest, schema=schema)

    proc = subprocess.run(
        [sys.executable, "scripts/bayesian_design_report_bundle_performance_budget.py", "--format", "json"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    printed = json.loads(proc.stdout)
    assert printed == manifest
