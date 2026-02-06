import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_schema(rel: str) -> dict:
    path = _repo_root() / rel
    assert path.exists(), f"missing schema: {path}"
    schema = json.loads(path.read_text())
    assert schema.get("$schema"), "schema must declare $schema"
    assert schema.get("$id"), "schema must declare $id"
    assert schema.get("type") == "object"
    return schema


def test_trex_report_distributions_schema_smoke():
    schema = _load_schema("docs/schemas/trex/report_distributions_v0.schema.json")
    props = schema.get("properties") or {}
    assert "schema_version" in props
    assert "meta" in props
    assert "channels" in props


def test_trex_report_pulls_schema_smoke():
    schema = _load_schema("docs/schemas/trex/report_pulls_v0.schema.json")
    props = schema.get("properties") or {}
    assert "schema_version" in props
    assert "meta" in props
    assert "entries" in props


def test_trex_report_corr_schema_smoke():
    schema = _load_schema("docs/schemas/trex/report_corr_v0.schema.json")
    props = schema.get("properties") or {}
    assert "schema_version" in props
    assert "meta" in props
    assert "parameter_names" in props
    assert "corr" in props


def test_trex_report_yields_schema_smoke():
    schema = _load_schema("docs/schemas/trex/report_yields_v0.schema.json")
    props = schema.get("properties") or {}
    assert "schema_version" in props
    assert "meta" in props
    assert "channels" in props

