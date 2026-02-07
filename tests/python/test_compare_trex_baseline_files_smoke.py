import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def test_compare_trex_baseline_files_smoke(tmp_path: Path):
    repo = _repo_root()
    script = repo / "tests" / "compare_trex_baseline_files.py"
    assert script.exists()

    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    out = tmp_path / "out.json"

    # Minimal trex_baseline_v0-compatible payload (meta fields are not compared here).
    base = {
        "schema_version": "trex_baseline_v0",
        "meta": {"created_at": "2026-01-01T00:00:00Z", "deterministic": True, "threads": 1},
        "fit": {"twice_nll": 1.0, "parameters": [{"name": "mu", "value": 1.0, "uncertainty": 0.1}], "covariance": None},
        "expected_data": {"pyhf_main": [1.0, 2.0], "pyhf_with_aux": [1.0, 2.0, 3.0]},
    }
    cand = json.loads(json.dumps(base))
    cand["fit"]["twice_nll"] = 1.0

    _write(a, base)
    _write(b, cand)

    p = subprocess.run(
        [sys.executable, str(script), "--baseline", str(a), "--candidate", str(b), "--out", str(out)],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert p.returncode == 0, p.stdout
    assert out.exists()

