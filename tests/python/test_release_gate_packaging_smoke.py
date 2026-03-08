from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_release_grade_python_gates_use_local_wheelhouse_and_prefer_installed_bindings() -> None:
    repo = _repo_root()
    scripts = [
        repo / "scripts" / "gvm" / "stable_first_gate.sh",
        repo / "scripts" / "benchmarks" / "simplified_likelihood_stable_surface_gate.sh",
        repo / "scripts" / "benchmarks" / "simplified_likelihood_exporter_surface_gate.sh",
    ]

    for path in scripts:
        text = path.read_text(encoding="utf-8")
        assert "cd bindings/ns-cli-py" in text, path
        assert 'nextstat_cli-*.whl' in text, path
        assert 'nextstat-*.whl' in text, path
        assert 'NEXTSTAT_PREFER_INSTALLED=1' in text, path
        assert 'assert callable(nextstat.set_threads)' in text, path
