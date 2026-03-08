from __future__ import annotations

import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _top_level_benchmark_artifact_dirs() -> list[Path]:
    root = _repo_root() / "benchmarks" / "artifacts"
    return sorted(path for path in root.iterdir() if path.is_dir())


def _scan_public_files_for(needle: str, *, include_root_files: bool = False) -> bool:
    """Search public-facing source files for *needle* (pure Python, no rg dependency)."""
    repo = _repo_root()
    ref_exts = {".md", ".py", ".rs", ".sh", ".json", ".yml", ".yaml"}
    ref_scopes = ["docs", "tests", "scripts", ".github"]

    if include_root_files:
        for fname in ("README.md", "CONTRIBUTING.md"):
            fpath = repo / fname
            if fpath.is_file():
                try:
                    if needle in fpath.read_text(encoding="utf-8", errors="replace"):
                        return True
                except OSError:
                    pass

    for scope in ref_scopes:
        scope_path = repo / scope
        if not scope_path.is_dir():
            continue
        for file_path in scope_path.rglob("*"):
            if not file_path.is_file() or file_path.suffix not in ref_exts:
                continue
            try:
                if needle in file_path.read_text(encoding="utf-8", errors="replace"):
                    return True
            except OSError:
                continue
    return False


def _has_public_reference(name: str) -> bool:
    return _scan_public_files_for(name)


def _has_public_path_reference(rel_path: str) -> bool:
    return _scan_public_files_for(rel_path, include_root_files=True)


def _tracked_benchmark_public_files() -> list[Path]:
    repo = _repo_root()
    proc = subprocess.run(
        [
            "git",
            "ls-files",
            "docs/benchmarks",
            "docs/references",
            "docs/specs",
            "benchmarks/artifacts",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    rel_paths = [
        line
        for line in proc.stdout.splitlines()
        if (line.endswith(".md") or line.endswith(".json")) and (repo / line).exists()
    ]
    return [repo / rel for rel in rel_paths]


def _tracked_files_under(*paths: str) -> list[str]:
    repo = _repo_root()
    proc = subprocess.run(
        ["git", "ls-files", *paths],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        line
        for line in proc.stdout.splitlines()
        if line.strip() and (repo / line).exists()
    ]


def test_tracked_public_benchmark_files_do_not_embed_repo_absolute_paths() -> None:
    repo_root = _repo_root().as_posix()
    offenders: list[str] = []
    for path in _tracked_benchmark_public_files():
        text = path.read_text(encoding="utf-8")
        if repo_root in text:
            offenders.append(str(path.relative_to(_repo_root())))
    assert offenders == []


def test_benchmark_artifact_tree_has_no_tracked_macos_junk() -> None:
    repo = _repo_root()
    offenders = [
        str(path.relative_to(repo))
        for path in (repo / "benchmarks" / "artifacts").rglob(".DS_Store")
    ]
    assert offenders == []


def test_primary_public_benchmark_snapshots_do_not_link_to_ephemeral_tmp_artifacts() -> None:
    repo = _repo_root()
    primary_snapshot_docs = [
        repo / "docs" / "benchmarks" / "ads-timeseries-benchmark-snapshot-2026-03-08.md",
        repo / "docs" / "benchmarks" / "ads-variance-reduction-benchmark-2026-03-08.md",
        repo / "docs" / "benchmarks" / "hepdata-import-benchmark-snapshot-2026-03-08.md",
        repo / "docs" / "benchmarks" / "simplified-likelihood-benchmark-snapshot-2026-03-08.md",
    ]
    offenders: list[str] = []
    for path in primary_snapshot_docs:
        text = path.read_text(encoding="utf-8")
        if "](/tmp/" in text or "](tmp/" in text:
            offenders.append(str(path.relative_to(repo)))
    assert offenders == []


def test_large_top_level_benchmark_artifact_dirs_must_be_publicly_referenced() -> None:
    offenders: list[str] = []
    size_threshold_kib = 200
    for path in _top_level_benchmark_artifact_dirs():
        size_kib = int(
            subprocess.check_output(["du", "-sk", str(path)], cwd=_repo_root(), text=True).split()[0]
        )
        if size_kib < size_threshold_kib:
            continue
        if not _has_public_reference(path.name):
            offenders.append(f"{path.relative_to(_repo_root())} ({size_kib} KiB)")
    assert offenders == []


def test_simplified_likelihood_history_dirs_must_have_exact_public_references() -> None:
    repo = _repo_root()
    history_roots = [
        repo / "benchmarks" / "artifacts" / "simplified_likelihood_promotion_bundles" / "nextstat-bench" / "history",
        repo / "benchmarks" / "artifacts" / "simplified_likelihood_export_benchmarks" / "nextstat-bench" / "history",
        repo / "benchmarks" / "artifacts" / "simplified_likelihood_exporter_promotion_bundles" / "nextstat-bench" / "history",
    ]
    offenders: list[str] = []
    for history_root in history_roots:
        for path in sorted(child for child in history_root.iterdir() if child.is_dir()):
            if not _has_public_reference(path.name):
                offenders.append(str(path.relative_to(repo)))
    assert offenders == []


def test_exporter_history_dirs_do_not_embed_deep_file_copies_that_break_windows_checkout() -> None:
    repo = _repo_root()
    history_root = (
        repo
        / "benchmarks"
        / "artifacts"
        / "simplified_likelihood_exporter_promotion_bundles"
        / "nextstat-bench"
        / "history"
    )
    offenders: list[str] = []
    for path in sorted(child for child in history_root.iterdir() if child.is_dir()):
        files_dir = path / "files"
        if files_dir.exists():
            offenders.append(str(files_dir.relative_to(repo)))
    assert offenders == []


def test_seed_benchmark_repo_does_not_track_local_runtime_dirs_or_bytecode() -> None:
    tracked = _tracked_files_under("benchmarks/nextstat-public-benchmarks")
    offenders = [
        path
        for path in tracked
        if (
            "/.venv/" in path
            or "/tmp/" in path
            or "/out/" in path
            or "/__pycache__/" in path
            or path.endswith(".pyc")
            or path.endswith(".pyo")
        )
    ]
    assert offenders == []


def test_tracked_bench_results_summaries_must_have_exact_public_references() -> None:
    tracked = _tracked_files_under("bench_results")
    offenders = [path for path in tracked if not _has_public_path_reference(path)]
    assert offenders == []
