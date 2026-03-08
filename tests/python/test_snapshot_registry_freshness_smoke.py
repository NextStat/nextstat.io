import subprocess
import sys
import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_committed_snapshot_registry_is_fresh() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "benchmarks/nextstat-public-benchmarks/scripts/write_snapshot_registry.py",
            "--snapshots-root",
            "benchmarks/nextstat-public-benchmarks/manifests/snapshots",
            "--out",
            "benchmarks/nextstat-public-benchmarks/manifests/snapshot_registry.json",
            "--check",
        ],
        cwd=_repo_root(),
    )


def test_public_benchmarks_makefile_checks_registry_freshness() -> None:
    makefile = (_repo_root() / "benchmarks" / "nextstat-public-benchmarks" / "Makefile").read_text(
        encoding="utf-8"
    )
    assert "scripts/write_snapshot_registry.py --snapshots-root manifests/snapshots --out manifests/snapshot_registry.json --check" in makefile
    assert "check-registry:" in makefile


def test_public_benchmarks_publish_templates_check_and_upload_registry() -> None:
    publish = (_repo_root() / "benchmarks" / "nextstat-public-benchmarks" / "ci" / "publish.yml").read_text(
        encoding="utf-8"
    )
    publish_gpu = (
        _repo_root() / "benchmarks" / "nextstat-public-benchmarks" / "ci" / "publish_gpu.yml"
    ).read_text(encoding="utf-8")
    for workflow in (publish, publish_gpu):
        assert "Check snapshot registry freshness" in workflow
        assert "python scripts/write_snapshot_registry.py \\" in workflow
        assert "--snapshots-root manifests/snapshots \\" in workflow
        assert "--out manifests/snapshot_registry.json \\" in workflow
        assert "--check" in workflow
        assert "manifests/snapshot_registry.json" in workflow


def test_committed_snapshot_registry_surfaces_host_backed_bayesian_health() -> None:
    registry = json.loads(
        (_repo_root() / "benchmarks" / "nextstat-public-benchmarks" / "manifests" / "snapshot_registry.json").read_text(
            encoding="utf-8"
        )
    )
    health_entries = [entry for entry in registry["entries"] if entry.get("suite_health")]
    assert health_entries, "committed snapshot registry must contain at least one health-complete snapshot"

    bayesian_rows = [
        row
        for entry in health_entries
        for row in entry.get("suite_health", [])
        if row.get("suite") == "bayesian"
    ]
    assert bayesian_rows, "committed snapshot registry must surface Bayesian suite health"
    assert any(row["promotion_gate"]["reviewed_case_count"] > 0 for row in bayesian_rows)
    assert any(row["promotion_gate"]["passed"] is True for row in bayesian_rows)


def test_committed_snapshot_registry_surfaces_host_backed_mams_health() -> None:
    registry = json.loads(
        (_repo_root() / "benchmarks" / "nextstat-public-benchmarks" / "manifests" / "snapshot_registry.json").read_text(
            encoding="utf-8"
        )
    )
    mams_rows = [
        row
        for entry in registry["entries"]
        for row in entry.get("suite_health", [])
        if row.get("suite") == "mams"
    ]
    assert mams_rows, "committed snapshot registry must surface MAMS suite health"
    assert any(row["promotion_gate"]["reviewed_case_count"] > 0 for row in mams_rows)
    assert any(row["core_quality"]["passed"] is True for row in mams_rows)
    assert any(row["promotion_gate"]["passed"] is False for row in mams_rows)
    assert any(
        isinstance(row["promotion_gate"].get("review_summary", {}).get("worst_max_r_hat"), dict)
        for row in mams_rows
    )
