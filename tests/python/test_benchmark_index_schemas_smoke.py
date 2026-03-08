import json
import subprocess
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_snapshot_index_schema_smoke(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "a.txt").write_text("hello\n", encoding="utf-8")
    (artifacts_dir / "b.bin").write_bytes(b"\x00\x01\x02")
    (artifacts_dir / ".replication").mkdir()
    (artifacts_dir / ".replication" / "ignore.txt").write_text("ignore\n", encoding="utf-8")

    out = artifacts_dir / "snapshot_index.json"
    subprocess.check_call(
        [
            "python3",
            str(_repo_root() / "scripts" / "benchmarks" / "write_snapshot_index.py"),
            "--suite",
            "smoke-suite",
            "--artifacts-dir",
            str(artifacts_dir),
            "--out",
            str(out),
            "--snapshot-id",
            "smoke-snapshot",
        ]
    )

    schema = json.loads(
        (_repo_root() / "docs" / "schemas" / "benchmarks" / "snapshot_index_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    inst = json.loads(out.read_text(encoding="utf-8"))
    jsonschema.validate(inst, schema)
    paths = [a["path"] for a in inst.get("artifacts", [])]
    assert ".replication/ignore.txt" not in paths


def test_snapshot_index_surfaces_suite_health_from_assessments(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    artifacts_dir = tmp_path / "artifacts"
    (artifacts_dir / "bayesian").mkdir(parents=True)
    (artifacts_dir / "mams").mkdir(parents=True)

    bayesian_assessment = {
        "schema_version": "nextstat.bayesian_assessment.v1",
        "suite": "bayesian",
        "source_suite_path": "bayesian_suite.json",
        "source_suite_sha256": "a" * 64,
        "source_suite_summary": {},
        "parity_summary": {},
        "core_quality": {"passed": True, "status": "passed", "failures": [], "warnings": []},
        "promotion_gate": {
            "passed": False,
            "status": "failed",
            "target_backend": "nextstat",
            "policy": {},
            "reviewed_cases": [
                {
                    "case": "hier_random_intercept_non_centered",
                    "backend": "nextstat",
                    "path": "cases/hier_random_intercept_non_centered__nextstat.json",
                    "status": "ok",
                    "reason": None,
                    "divergence_rate": 0.000125,
                    "max_treedepth_rate": 0.0,
                    "max_r_hat": 1.0142,
                    "min_ess_bulk": 1200.0,
                    "min_ess_tail": 900.0,
                    "min_ebfmi": 0.57,
                    "min_ess_bulk_per_sec": 500.0,
                }
            ],
            "review_summary": {
                "n_reviewed_cases": 1,
                "n_failures": 1,
                "n_failing_cases": 1,
                "failing_cases": ["hier_random_intercept_non_centered"],
                "worst_max_r_hat": {"case": "hier_random_intercept_non_centered", "value": 1.0142},
            },
            "failures": [
                {
                    "case": "hier_random_intercept_non_centered",
                    "reason": "max_r_hat_exceeds_threshold",
                    "observed": 1.0142,
                    "threshold": 1.01,
                }
            ],
        },
    }
    (artifacts_dir / "bayesian" / "bayesian_assessment.json").write_text(
        json.dumps(bayesian_assessment) + "\n", encoding="utf-8"
    )

    mams_assessment = {
        "schema_version": "nextstat.mams_assessment.v1",
        "suite": "mams",
        "source_suite_path": "mams_suite.json",
        "source_suite_sha256": "b" * 64,
        "source_suite_summary": {},
        "core_quality": {"passed": True, "status": "passed", "failures": [], "warnings": []},
        "promotion_gate": {
            "passed": False,
            "status": "failed",
            "target_backend": "nextstat_mams",
            "policy": {},
            "reviewed_cases": [
                {
                    "case": "glm_logistic",
                    "status": "ok",
                    "max_r_hat": 1.0112,
                    "min_ess_bulk": 900.0,
                    "ess_per_sec": 250.0,
                }
            ],
            "failures": [
                {
                    "case": "glm_logistic",
                    "reason": "max_r_hat_exceeds_threshold",
                    "observed": 1.0112,
                    "threshold": 1.01,
                }
            ],
        },
    }
    (artifacts_dir / "mams" / "mams_assessment.json").write_text(
        json.dumps(mams_assessment) + "\n", encoding="utf-8"
    )

    out = artifacts_dir / "snapshot_index.json"
    subprocess.check_call(
        [
            "python3",
            str(_repo_root() / "scripts" / "benchmarks" / "write_snapshot_index.py"),
            "--suite",
            "snapshot-health-smoke",
            "--artifacts-dir",
            str(artifacts_dir),
            "--out",
            str(out),
            "--snapshot-id",
            "snapshot-health-smoke",
        ]
    )

    schema = json.loads(
        (_repo_root() / "docs" / "schemas" / "benchmarks" / "snapshot_index_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    inst = json.loads(out.read_text(encoding="utf-8"))
    jsonschema.validate(inst, schema)

    suite_health = {row["suite"]: row for row in inst["suite_health"]}
    assert suite_health["bayesian"]["promotion_gate"]["failing_cases"] == [
        "hier_random_intercept_non_centered"
    ]
    assert suite_health["bayesian"]["promotion_gate"]["review_summary"]["worst_max_r_hat"] == {
        "case": "hier_random_intercept_non_centered",
        "value": 1.0142,
    }
    assert suite_health["mams"]["promotion_gate"]["review_summary"]["worst_ess_per_sec"] == {
        "case": "glm_logistic",
        "value": 250.0,
    }


def test_replication_report_schema_smoke(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    root = tmp_path / "runs"
    orig = root / "orig"
    rep = root / "rep"
    orig.mkdir(parents=True)
    rep.mkdir(parents=True)

    (orig / "x.txt").write_text("one\n", encoding="utf-8")
    (rep / "x.txt").write_text("two\n", encoding="utf-8")

    orig_index = orig / "snapshot_index.json"
    rep_index = rep / "snapshot_index.json"
    subprocess.check_call(
        [
            "python3",
            str(_repo_root() / "scripts" / "benchmarks" / "write_snapshot_index.py"),
            "--suite",
            "orig-suite",
            "--artifacts-dir",
            str(orig),
            "--out",
            str(orig_index),
            "--snapshot-id",
            "orig",
        ]
    )
    subprocess.check_call(
        [
            "python3",
            str(_repo_root() / "scripts" / "benchmarks" / "write_snapshot_index.py"),
            "--suite",
            "rep-suite",
            "--artifacts-dir",
            str(rep),
            "--out",
            str(rep_index),
            "--snapshot-id",
            "rep",
        ]
    )

    out = rep / "replication_report.json"
    subprocess.check_call(
        [
            "python3",
            str(_repo_root() / "scripts" / "benchmarks" / "write_replication_report.py"),
            "--original-index",
            str(orig_index),
            "--replica-index",
            str(rep_index),
            "--out",
            str(out),
            "--notes",
            "smoke",
        ]
    )

    schema = json.loads(
        (_repo_root() / "docs" / "schemas" / "benchmarks" / "replication_report_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    inst = json.loads(out.read_text(encoding="utf-8"))
    jsonschema.validate(inst, schema)


def test_snapshot_registry_schema_smoke(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")

    artifacts_a = tmp_path / "snap_a"
    artifacts_b = tmp_path / "snap_b"
    artifacts_a.mkdir()
    artifacts_b.mkdir()

    (artifacts_a / "a.txt").write_text("hello\n", encoding="utf-8")
    (artifacts_b / "b.txt").write_text("world\n", encoding="utf-8")

    bayesian_assessment = {
        "schema_version": "nextstat.bayesian_assessment.v1",
        "suite": "bayesian",
        "source_suite_path": "bayesian_suite.json",
        "source_suite_sha256": "a" * 64,
        "source_suite_summary": {},
        "parity_summary": {},
        "core_quality": {"passed": True, "status": "passed", "failures": [], "warnings": []},
        "promotion_gate": {
            "passed": True,
            "status": "passed",
            "target_backend": "nextstat",
            "policy": {},
            "reviewed_cases": [],
            "review_summary": {
                "n_reviewed_cases": 0,
                "n_failures": 0,
                "n_failing_cases": 0,
                "failing_cases": [],
            },
            "failures": [],
        },
    }
    (artifacts_a / "bayesian_assessment.json").write_text(json.dumps(bayesian_assessment) + "\n", encoding="utf-8")

    index_a = artifacts_a / "snapshot_index.json"
    index_b = artifacts_b / "snapshot_index.json"
    subprocess.check_call(
        [
            "python3",
            str(_repo_root() / "scripts" / "benchmarks" / "write_snapshot_index.py"),
            "--suite",
            "snap-a",
            "--artifacts-dir",
            str(artifacts_a),
            "--out",
            str(index_a),
            "--snapshot-id",
            "snap-a",
        ]
    )
    subprocess.check_call(
        [
            "python3",
            str(_repo_root() / "scripts" / "benchmarks" / "write_snapshot_index.py"),
            "--suite",
            "snap-b",
            "--artifacts-dir",
            str(artifacts_b),
            "--out",
            str(index_b),
            "--snapshot-id",
            "snap-b",
        ]
    )

    out = tmp_path / "snapshot_registry.json"
    subprocess.check_call(
        [
            "python3",
            str(_repo_root() / "scripts" / "benchmarks" / "write_snapshot_registry.py"),
            "--snapshot-index",
            str(index_a),
            "--snapshot-index",
            str(index_b),
            "--out",
            str(out),
        ]
    )

    schema = json.loads(
        (_repo_root() / "docs" / "schemas" / "benchmarks" / "snapshot_registry_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    inst = json.loads(out.read_text(encoding="utf-8"))
    jsonschema.validate(inst, schema)
    assert inst["entry_count"] == 2
    assert inst["generated_at"] == inst["entries"][0]["generated_at"]
    by_snapshot = {row["snapshot_id"]: row for row in inst["entries"]}
    assert by_snapshot["snap-a"]["suite_health"][0]["suite"] == "bayesian"
    assert by_snapshot["snap-b"]["suite_health"] == []
    assert by_snapshot["snap-a"]["snapshot_index_path"] == "snap_a/snapshot_index.json"
    assert by_snapshot["snap-b"]["snapshot_index_path"] == "snap_b/snapshot_index.json"

    subprocess.check_call(
        [
            "python3",
            str(_repo_root() / "benchmarks" / "nextstat-public-benchmarks" / "scripts" / "validate_artifacts.py"),
            "--strict",
            str(out),
        ]
    )


def test_snapshot_registry_check_mode_detects_drift(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "snapshots" / "snap"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "x.txt").write_text("hello\n", encoding="utf-8")

    index_path = artifacts_dir / "snapshot_index.json"
    subprocess.check_call(
        [
            "python3",
            str(_repo_root() / "scripts" / "benchmarks" / "write_snapshot_index.py"),
            "--suite",
            "snap",
            "--artifacts-dir",
            str(artifacts_dir),
            "--out",
            str(index_path),
            "--snapshot-id",
            "snap",
        ]
    )

    out = tmp_path / "snapshot_registry.json"
    script = _repo_root() / "scripts" / "benchmarks" / "write_snapshot_registry.py"
    subprocess.check_call(
        [
            "python3",
            str(script),
            "--snapshots-root",
            str(tmp_path / "snapshots"),
            "--out",
            str(out),
        ]
    )
    subprocess.check_call(
        [
            "python3",
            str(script),
            "--snapshots-root",
            str(tmp_path / "snapshots"),
            "--out",
            str(out),
            "--check",
        ]
    )

    out.write_text(out.read_text(encoding="utf-8").replace("\"entry_count\": 1", "\"entry_count\": 99"), encoding="utf-8")
    failed = subprocess.run(
        [
            "python3",
            str(script),
            "--snapshots-root",
            str(tmp_path / "snapshots"),
            "--out",
            str(out),
            "--check",
        ],
        capture_output=True,
        text=True,
    )
    assert failed.returncode == 1
    assert "out of date:" in (failed.stderr or failed.stdout)
