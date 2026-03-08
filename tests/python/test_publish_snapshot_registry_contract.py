import importlib.util
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLISH_SCRIPT = REPO_ROOT / "benchmarks" / "nextstat-public-benchmarks" / "scripts" / "publish_snapshot.py"
BAYESIAN_REMOTE_RUNNER = REPO_ROOT / "scripts" / "benchmarks" / "publish_bayesian_snapshot_remote.sh"
MAMS_REMOTE_RUNNER = REPO_ROOT / "scripts" / "benchmarks" / "publish_mams_snapshot_remote.sh"


def _load_publish_snapshot_module():
    spec = importlib.util.spec_from_file_location("publish_snapshot_module", PUBLISH_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_registry_path_for_canonical_snapshots_root() -> None:
    mod = _load_publish_snapshot_module()
    out_root = REPO_ROOT / "benchmarks" / "nextstat-public-benchmarks" / "manifests" / "snapshots"
    assert mod.default_registry_path(out_root) == out_root.parent / "snapshot_registry.json"


def test_default_registry_path_for_custom_out_root(tmp_path: Path) -> None:
    mod = _load_publish_snapshot_module()
    out_root = tmp_path / "nextstat_bayesian_publisher_snapshots_20260309T000000Z"
    assert mod.default_registry_path(out_root) == out_root / "snapshot_registry.json"


def test_write_snapshot_registry_invokes_writer_and_validation(monkeypatch, tmp_path: Path) -> None:
    mod = _load_publish_snapshot_module()
    out_root = tmp_path / "snapshots"
    out_root.mkdir()

    calls: list[tuple[list[str], str | None]] = []
    validated: list[Path] = []

    def _fake_check_call(cmd, cwd=None, env=None):
        calls.append((list(cmd), cwd))

    monkeypatch.setattr(mod.subprocess, "check_call", _fake_check_call)
    monkeypatch.setattr(mod, "validate_artifact", lambda path, repo_root, env: validated.append(path))

    out = mod.write_snapshot_registry(out_root=out_root, repo_root=tmp_path, env={})

    assert out == (tmp_path / "snapshot_registry.json")
    assert len(calls) == 1
    assert calls[0][0][1] == "scripts/write_snapshot_registry.py"
    assert "--snapshots-root" in calls[0][0]
    assert str(out_root) in calls[0][0]
    assert "--out" in calls[0][0]
    assert str(out) in calls[0][0]
    assert validated == [out]


def test_publish_bayesian_snapshot_remote_runner_syncs_registry_contract() -> None:
    script = BAYESIAN_REMOTE_RUNNER.read_text(encoding="utf-8")
    assert "snapshot_registry.json" in script
    assert 'validate_artifacts.py" --strict "$SNAPSHOT_ROOT"' in script
    assert 'rsync -az --rsh="${RSYNC_RSH_CMD}" "${REMOTE_SPEC}:${BENCH_REMOTE_SNAPSHOT_ROOT}/" "${BENCH_LOCAL_ROOT}/"' in script


def test_publish_snapshot_mams_path_renders_report_and_snapshot_snippet(monkeypatch, tmp_path: Path) -> None:
    mod = _load_publish_snapshot_module()
    out_root = tmp_path / "snapshots"
    calls: list[list[str]] = []
    registry_calls: list[Path] = []

    def _fake_check_call(cmd, cwd=None, env=None):
        cmd_list = [str(part) for part in cmd]
        calls.append(cmd_list)
        if "suites/mams/suite.py" in cmd_list:
            out_dir = Path(cmd_list[cmd_list.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "mams_suite.json").write_text('{"schema_version":"nextstat.mams_benchmark_suite_result.v1"}\n')
            return
        if "suites/mams/report.py" in cmd_list:
            if "--snippet" in cmd_list:
                out_path = Path(cmd_list[cmd_list.index("--out") + 1])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text("# MAMS snippet\n")
            else:
                suite_dir = Path(cmd_list[-1])
                suite_dir.mkdir(parents=True, exist_ok=True)
                (suite_dir / "mams_benchmark_report.md").write_text("# MAMS report\n")
            return
        if "suites/mams/assess.py" in cmd_list:
            suite_dir = Path(cmd_list[-1])
            (suite_dir / "mams_assessment.json").write_text('{"schema_version":"nextstat.mams_assessment.v1"}\n')
            return
        if "scripts/write_baseline_manifest.py" in cmd_list:
            out_path = Path(cmd_list[cmd_list.index("--out") + 1])
            out_path.write_text('{"schema_version":"nextstat.baseline_manifest.v1"}\n')
            return
        if "scripts/write_snapshot_index.py" in cmd_list:
            out_path = Path(cmd_list[cmd_list.index("--out") + 1])
            out_path.write_text('{"schema_version":"nextstat.snapshot_index.v1"}\n')
            return

    def _fake_write_snapshot_registry(*, out_root, repo_root, env, registry_out=None):
        out_path = (registry_out or (Path(out_root) / "snapshot_registry.json")).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text('{"schema_version":"nextstat.snapshot_registry.v1","generated_at":"2026-03-09T00:00:00Z","entry_count":0,"entries":[]}\n')
        registry_calls.append(out_path)
        return out_path

    monkeypatch.setattr(mod.subprocess, "check_call", _fake_check_call)
    monkeypatch.setattr(mod, "validate_artifact", lambda path, repo_root, env: None)
    monkeypatch.setattr(mod, "write_snapshot_registry", _fake_write_snapshot_registry)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(PUBLISH_SCRIPT),
            "--snapshot-id",
            "mams-publisher-contract",
            "--out-root",
            str(out_root),
            "--mams",
            "--deterministic",
        ],
    )

    rc = mod.main()
    assert rc == 0

    snapshot_dir = out_root / "mams-publisher-contract"
    assert (snapshot_dir / "mams" / "mams_benchmark_report.md").read_text(encoding="utf-8") == "# MAMS report\n"
    assert (snapshot_dir / "README_snippet_mams.md").read_text(encoding="utf-8") == "# MAMS snippet\n"
    assert any("suites/mams/report.py" in call for call in calls)
    assert any("suites/mams/assess.py" in call for call in calls)
    assert any("--snippet" in call for call in calls if "suites/mams/report.py" in call)
    assert registry_calls == [(out_root / "snapshot_registry.json").resolve()]


def test_publish_mams_snapshot_remote_runner_syncs_registry_contract() -> None:
    script = MAMS_REMOTE_RUNNER.read_text(encoding="utf-8")
    assert "snapshot_registry.json" in script
    assert '"scripts/publish_snapshot.py"' in script
    assert '"--mams"' in script
    assert 'validate_artifacts.py" --strict "$SNAPSHOT_ROOT"' in script
    assert 'rsync -az --rsh="${RSYNC_RSH_CMD}" "${REMOTE_SPEC}:${BENCH_REMOTE_SNAPSHOT_ROOT}/" "${BENCH_LOCAL_ROOT}/"' in script


def test_publish_bayesian_snapshot_remote_runner_syntax() -> None:
    subprocess.check_call(
        ["bash", "-n", str(BAYESIAN_REMOTE_RUNNER)],
        cwd=REPO_ROOT,
    )


def test_publish_mams_snapshot_remote_runner_syntax() -> None:
    subprocess.check_call(
        ["bash", "-n", str(MAMS_REMOTE_RUNNER)],
        cwd=REPO_ROOT,
    )


def test_publish_snapshot_script_syntax() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "py_compile", str(PUBLISH_SCRIPT)],
        cwd=REPO_ROOT,
    )
