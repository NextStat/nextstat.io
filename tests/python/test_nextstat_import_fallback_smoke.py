from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _installed_core_exists_outside_repo(repo: Path) -> bool:
    repo_bindings = (repo / "bindings" / "ns-py" / "python" / "nextstat").resolve()
    patterns = ("nextstat/_core*.so", "nextstat/_core*.pyd", "nextstat/_core*.dylib", "nextstat/_core*.dll")
    for entry in sys.path:
        if not entry:
            continue
        try:
            base = Path(entry).resolve()
        except OSError:
            continue
        candidate_dir = base / "nextstat"
        if candidate_dir == repo_bindings.parent or not candidate_dir.is_dir():
            continue
        for pattern in patterns:
            if any(base.glob(pattern)):
                return True
    return False


def test_nextstat_source_wrapper_can_fall_back_to_installed_core(tmp_path: Path):
    repo = _repo_root()
    if not _installed_core_exists_outside_repo(repo):
        pytest.skip("installed nextstat._core outside repo source tree is unavailable in this environment")

    source_pkg = repo / "bindings" / "ns-py" / "python" / "nextstat"
    wrapper_root = tmp_path / "shadow"
    wrapper_pkg = wrapper_root / "nextstat"
    shutil.copytree(source_pkg, wrapper_pkg)
    for pattern in ("_core*.so", "_core*.pyd", "_core*.dylib", "_core*.dll"):
        for candidate in wrapper_pkg.glob(pattern):
            candidate.unlink()

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import nextstat; "
                "assert callable(nextstat.set_threads), nextstat.set_threads; "
                "import nextstat._core as core; "
                "print(core.__file__)"
            ),
        ],
        cwd=repo,
        env={
            **os.environ,
            "PYTHONPATH": str(wrapper_root),
            "NEXTSTAT_PREFER_INSTALLED": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert str(wrapper_pkg) not in proc.stdout, proc.stdout
