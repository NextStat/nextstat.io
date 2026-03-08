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
        assert 'run_maturin()' in text, path
        assert '"-m" "maturin"' in text, path
        assert '${repo_root}/.venv/bin/python' in text, path
        assert 'elif [[ -x "${repo_root}/.venv/bin/python" ]]; then' in text, path


def test_release_build_config_does_not_force_native_cpu_for_apple_arm() -> None:
    lines = (_repo_root() / ".cargo" / "config.toml").read_text(encoding="utf-8").splitlines()
    active = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    assert "[target.aarch64-apple-darwin]" not in active
    assert 'rustflags = ["-C", "target-cpu=native"]' not in active


def test_release_candidate_cli_aarch64_linux_build_pins_ring_arm_asm_env() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "release-candidate.yml").read_text(
        encoding="utf-8"
    )
    assert "CFLAGS_aarch64_unknown_linux_gnu" in workflow
    assert "-D__ARM_ARCH=8 -march=armv8-a" in workflow
    assert "matrix.target == 'aarch64-unknown-linux-gnu'" in workflow
