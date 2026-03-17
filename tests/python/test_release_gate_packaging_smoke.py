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
        repo / "scripts" / "benchmarks" / "histfactory_stable_surface_gate.sh",
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
        assert '--interpreter "${py}"' in text, path


def test_apex2_pre_release_gate_builds_nextstat_wheel_for_the_install_python() -> None:
    gate = (_repo_root() / "scripts" / "apex2" / "pre_release_gate.sh").read_text(encoding="utf-8")
    assert "os.path.abspath(sys.executable)" in gate
    assert "realpath(sys.executable)" not in gate
    assert '--interpreter "${py}"' in gate
    assert 'maturin build --release --interpreter "${py}" -o "${wheelhouse}"' in gate
    assert '"nextstat==${version}"' in gate
    assert 'APEX2_PERF_POLICY' in gate
    assert 'APEX2_CANONICAL_PERF_RUNNER' in gate
    assert "passed with performance advisories on non-canonical hardware" in gate


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


def test_histfactory_release_gate_installs_maturin_for_local_wheelhouse_build() -> None:
    workflow = (_repo_root() / ".github" / "workflows" / "release-candidate.yml").read_text(
        encoding="utf-8"
    )
    histfactory_job = workflow.split("histfactory-stable-surface:")[1].split("hepdata-import-stable-surface:")[0]
    assert '"maturin>=1.11,<2.0"' in histfactory_job
    assert 'make histfactory-stable-surface-gate' in histfactory_job


def test_trex_analysis_spec_runners_prefer_wheelhouse_cli_binary() -> None:
    repo = _repo_root()
    files = [
        repo / "scripts" / "trex" / "run_analysis_spec.py",
        repo / "tests" / "record_trex_analysis_spec_baseline.py",
        repo / "tests" / "compare_trex_analysis_spec_with_latest_baseline.py",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert '".venv" / "bin" / "nextstat"' in text, path
        assert 'shutil.which("nextstat")' in text, path


def test_trex_analysis_spec_materializers_resolve_baseline_dir_to_absolute_paths() -> None:
    repo = _repo_root()
    files = [
        repo / "tests" / "record_trex_analysis_spec_baseline.py",
        repo / "tests" / "compare_trex_analysis_spec_with_latest_baseline.py",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert 'baseline_compare["enabled"] = False' in text, path
        assert 'baseline_compare["baseline_dir"]' in text, path
        assert 'resolve(spec_base, baseline_compare.get("baseline_dir"))' in text or 'resolve(baseline_compare.get("baseline_dir"))' in text, path
    compare_text = files[1].read_text(encoding="utf-8")
    assert "baseline_manifest_dir=args.manifest.parent" in compare_text
