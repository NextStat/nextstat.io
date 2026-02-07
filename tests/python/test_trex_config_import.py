"""TRExFitter `.config` -> analysis spec importer (dependency-light)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from nextstat.trex_config.importer import trex_config_file_to_analysis_spec_v0


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "trex_config" / "minimal_tutorial.config"


def test_importer_returns_valid_shape():
    spec, report = trex_config_file_to_analysis_spec_v0(FIXTURE, out_path="analysis.yaml")

    assert spec["schema_version"] == "trex_analysis_spec_v0"
    assert spec["inputs"]["mode"] == "trex_config_yaml"
    cfg = spec["inputs"]["trex_config_yaml"]
    assert cfg["read_from"] == "NTUP"
    assert cfg["tree_name"] == "events"
    assert cfg["measurement"] == "meas"
    assert cfg["poi"] == "mu"
    assert cfg["regions"][0]["name"] == "SR"
    assert cfg["samples"][0]["name"] == "signal"
    assert report["version"] == 1
    assert isinstance(report["mapped"], list)
    assert isinstance(report["unmapped"], list)

    # This fixture contains keys we intentionally do not support yet.
    unmapped_keys = {u["key"] for u in report["unmapped"]}
    assert "Title" in unmapped_keys
    assert "Type" in unmapped_keys


def test_cli_import_config_writes_files(tmp_path: Path):
    out = tmp_path / "analysis.yaml"
    rep = tmp_path / "mapping.json"

    cmd = [
        sys.executable,
        "-m",
        "nextstat.trex_config.cli",
        "import-config",
        "--config",
        str(FIXTURE),
        "--out",
        str(out),
        "--report",
        str(rep),
        "--overwrite",
    ]
    env = dict(os.environ)
    # Ensure repo python sources are importable in dev runs.
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "bindings" / "ns-py" / "python")
    p = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)

    assert out.exists()
    assert rep.exists()
    payload = json.loads(p.stdout)
    assert payload["analysis_spec"].endswith("analysis.yaml")
    assert payload["mapping_report"].endswith("mapping.json")
