from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_hep_dataset_bootstrap_ci_smoke(tmp_path: Path) -> None:
    has_root_cli = shutil.which("root") is not None
    has_uproot = importlib.util.find_spec("uproot") is not None
    if not has_root_cli and not has_uproot:
        pytest.skip("Neither ROOT CLI nor uproot is available")

    nextstat = Path("target/debug/nextstat")
    if not nextstat.exists():
        pytest.skip("target/debug/nextstat is missing")

    out_dir = tmp_path / "hep_dataset_bca_smoke"
    root_writer = "uproot" if has_uproot else "root-cli"
    cmd = [
        sys.executable,
        "scripts/benchmarks/bench_hep_dataset_bootstrap_ci.py",
        "--runs",
        "1",
        "--n-events",
        "80",
        "--n-bootstrap",
        "20",
        "--threads",
        "1",
        "--root-writer",
        root_writer,
        "--nextstat-bin",
        str(nextstat),
        "--out-dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    summary_path = out_dir / "summary.json"
    assert summary_path.exists(), "summary.json must be created"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["config"]["runs"] == 1
    scenarios = summary["scenarios"]
    assert set(scenarios.keys()) == {"gauss_mu_mid", "gauss_mu_boundary_low"}

    for scenario in scenarios.values():
        for method in ("percentile", "bca"):
            row = scenario[method]
            assert row["runs"] == 1
            assert 0.0 <= row["coverage_vs_true_poi"] <= 1.0
            assert row["median_width"] >= 0.0
