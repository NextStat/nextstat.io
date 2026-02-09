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
