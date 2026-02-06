import json
from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_trex_analysis_spec_schema_and_example_validate():
    schema_path = _repo_root() / "docs" / "schemas" / "trex" / "analysis_spec_v0.schema.json"
    assert schema_path.exists(), f"missing schema: {schema_path}"
    schema = json.loads(schema_path.read_text())
    assert schema.get("$schema")
    assert schema.get("$id")

    spec_path = _repo_root() / "docs" / "specs" / "trex" / "analysis_spec_v0.yaml"
    assert spec_path.exists(), f"missing spec: {spec_path}"
    spec = yaml.safe_load(spec_path.read_text())
    assert spec.get("schema_version") == "trex_analysis_spec_v0"

    import jsonschema  # type: ignore

    jsonschema.validate(instance=spec, schema=schema)

