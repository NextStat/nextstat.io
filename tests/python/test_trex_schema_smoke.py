import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_trex_baseline_schema_is_valid_json_and_has_expected_shape():
    schema_path = _repo_root() / "docs" / "schemas" / "trex" / "baseline_v0.schema.json"
    assert schema_path.exists(), f"missing schema: {schema_path}"

    schema = json.loads(schema_path.read_text())
    assert schema.get("$schema"), "schema must declare $schema"
    assert schema.get("$id"), "schema must declare $id"
    assert schema.get("type") == "object"
    props = schema.get("properties") or {}
    assert "schema_version" in props
    assert "meta" in props
    assert "fit" in props
    assert "expected_data" in props

