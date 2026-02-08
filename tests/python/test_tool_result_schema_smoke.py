import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_tool_result_schema_and_example_smoke():
    schema_path = _repo_root() / "docs" / "schemas" / "tools" / "nextstat_tool_result_v1.schema.json"
    assert schema_path.exists(), f"missing schema: {schema_path}"

    schema = json.loads(schema_path.read_text())
    assert schema.get("$schema"), "schema must declare $schema"
    assert schema.get("$id"), "schema must declare $id"
    assert schema.get("type") == "object"

    example_path = _repo_root() / "docs" / "specs" / "nextstat_tool_result_v1.example.json"
    assert example_path.exists(), f"missing example: {example_path}"
    example = json.loads(example_path.read_text())
    assert example.get("schema_version") == "nextstat.tool_result.v1"

    try:
        import jsonschema  # type: ignore
    except Exception:
        jsonschema = None

    if jsonschema is not None:
        jsonschema.validate(instance=example, schema=schema)

