import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_bench_results_tracks_only_summary_json() -> None:
    tracked_out = subprocess.check_output(
        ["git", "ls-files", "--", "bench_results"],
        text=True,
        cwd=str(_repo_root()),
    )
    deleted_out = subprocess.check_output(
        ["git", "ls-files", "--deleted", "--", "bench_results"],
        text=True,
        cwd=str(_repo_root()),
    )
    tracked = {line.strip() for line in tracked_out.splitlines() if line.strip()}
    deleted = {line.strip() for line in deleted_out.splitlines() if line.strip()}
    live_tracked = sorted(tracked - deleted)

    offenders = [p for p in live_tracked if not p.endswith("/summary.json")]
    assert not offenders, (
        "Only summary.json files are allowed under bench_results. "
        f"Unexpected tracked files: {offenders}"
    )
