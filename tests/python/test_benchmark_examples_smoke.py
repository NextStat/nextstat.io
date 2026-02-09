import json
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_snapshot_index_example_validates() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (_repo_root() / "docs" / "schemas" / "benchmarks" / "snapshot_index_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    example = json.loads((_repo_root() / "docs" / "specs" / "snapshot_index_v1.example.json").read_text(encoding="utf-8"))
    jsonschema.validate(example, schema)


def test_replication_report_example_validates() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (_repo_root() / "docs" / "schemas" / "benchmarks" / "replication_report_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    example = json.loads(
        (_repo_root() / "docs" / "specs" / "replication_report_v1.example.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(example, schema)

