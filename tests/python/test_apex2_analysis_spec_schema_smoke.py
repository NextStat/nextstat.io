import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_apex2_analysis_spec_schema_and_example_smoke():
    schema_path = _repo_root() / "docs" / "schemas" / "apex2" / "analysis_spec_v0.schema.json"
    assert schema_path.exists(), f"missing schema: {schema_path}"

    schema = json.loads(schema_path.read_text())
    assert schema.get("$schema"), "schema must declare $schema"
    assert schema.get("$id"), "schema must declare $id"
    assert schema.get("type") == "object"

    # The example spec is written in JSON-compatible YAML so we can parse it without extra deps.
    spec_path = _repo_root() / "docs" / "specs" / "apex2_analysis_spec_v0.yaml"
    assert spec_path.exists(), f"missing spec: {spec_path}"
    spec = json.loads(spec_path.read_text())
    assert spec.get("schema_version") == "apex2_analysis_spec_v0"

    # Optional strict validation (jsonschema comes via pyhf dependency in the validation env).
    try:
        import jsonschema  # type: ignore
    except Exception:
        jsonschema = None

    if jsonschema is not None:
        jsonschema.validate(instance=spec, schema=schema)

