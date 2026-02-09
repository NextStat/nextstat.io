from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_record_trex_baseline_prereq_only_missing_tools_is_actionable() -> None:
    repo = Path(__file__).resolve().parents[2]
    script = repo / "tests" / "record_trex_baseline.py"
    assert script.exists()

    env = os.environ.copy()
    env["PATH"] = ""  # force missing root/hist2workspace even if present on the machine

    p = subprocess.run(
        [sys.executable, str(script), "--export-dir", "tests/fixtures/histfactory", "--prereq-only"],
        cwd=str(repo),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    assert p.returncode == 3
    # Must be JSON (actionable) and mention missing prereqs keys.
    assert '"ok": false' in p.stdout.lower()
    assert "hist2workspace" in p.stdout
    assert "root" in p.stdout

