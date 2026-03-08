from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_manifest_module():
    module_path = _repo_root() / "scripts" / "tool_contract_artifact_manifest.py"
    spec = importlib.util.spec_from_file_location("tool_contract_artifact_manifest", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tool_contract_artifact_manifest_schema_and_exports():
    manifest_path = _repo_root() / "scripts" / "tool_contract_artifact_manifest_v1.json"
    schema_path = (
        _repo_root()
        / "docs"
        / "schemas"
        / "tools"
        / "nextstat_tool_contract_artifact_manifest_v1.schema.json"
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    try:
        import jsonschema  # type: ignore
    except Exception:
        jsonschema = None

    if jsonschema is not None:
        jsonschema.validate(instance=manifest, schema=schema)

    module = _load_manifest_module()
    loaded = module.load_tool_contract_artifact_manifest()
    outputs = module.to_github_outputs(loaded)

    assert loaded["schema_version"] == "nextstat.tool_contract_artifact_manifest.v1"
    assert outputs["fast_report_path"] == loaded["reports"]["fast"]["runner_report_path"]
    assert outputs["fast_artifact_name"] == loaded["reports"]["fast"]["artifact_name"]
    assert outputs["fast_download_dir"] == loaded["reports"]["fast"]["download_dir"]
    assert outputs["fast_downloaded_report_path"] == loaded["reports"]["fast"]["downloaded_report_path"]
    assert outputs["live_report_path"] == loaded["reports"]["live"]["runner_report_path"]
    assert outputs["live_artifact_name"] == loaded["reports"]["live"]["artifact_name"]
    assert outputs["live_download_dir"] == loaded["reports"]["live"]["download_dir"]
    assert outputs["live_downloaded_report_path"] == loaded["reports"]["live"]["downloaded_report_path"]
    assert outputs["dashboard_artifact_name"] == loaded["dashboard"]["artifact_name"]
    assert outputs["dashboard_json_path"] == loaded["dashboard"]["json_path"]
    assert outputs["dashboard_markdown_path"] == loaded["dashboard"]["markdown_path"]

    proc = subprocess.run(
        [sys.executable, "scripts/tool_contract_artifact_manifest.py", "--format", "github-output"],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    emitted = dict(
        line.split("=", 1)
        for line in proc.stdout.splitlines()
        if line.strip()
    )
    assert emitted == outputs
