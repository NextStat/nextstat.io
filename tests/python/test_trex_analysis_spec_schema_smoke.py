import json
from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_schema() -> dict:
    schema_path = _repo_root() / "docs" / "schemas" / "trex" / "analysis_spec_v0.schema.json"
    assert schema_path.exists(), f"missing schema: {schema_path}"
    schema = json.loads(schema_path.read_text())
    assert schema.get("$schema")
    assert schema.get("$id")
    return schema


def _spec_paths() -> list[Path]:
    repo = _repo_root()
    out: list[Path] = []
    out.append(repo / "docs" / "specs" / "trex" / "analysis_spec_v0.yaml")
    examples_dir = repo / "docs" / "specs" / "trex" / "examples"
    if examples_dir.exists():
        out.extend(sorted(examples_dir.glob("*.yaml")))
    return out


def test_trex_analysis_spec_schema_and_examples_validate():
    schema = _load_schema()

    import jsonschema  # type: ignore

    for spec_path in _spec_paths():
        assert spec_path.exists(), f"missing spec: {spec_path}"
        spec = yaml.safe_load(spec_path.read_text())
        assert spec.get("schema_version") == "trex_analysis_spec_v0"
        jsonschema.validate(instance=spec, schema=schema)

