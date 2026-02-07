from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_record_trex_baseline_help_and_missing_args_are_actionable() -> None:
    repo = Path(__file__).resolve().parents[2]
    script = repo / "tests" / "record_trex_baseline.py"
    assert script.exists()

    p_help = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert p_help.returncode == 0
    assert "--export-dir" in p_help.stdout

    # Missing args: must not crash with traceback, should exit with code 2.
    env = os.environ.copy()
    env["PATH"] = ""
    p = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(repo),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert p.returncode == 2
    assert "provide --export-dir or --trex-config" in p.stdout.lower()

