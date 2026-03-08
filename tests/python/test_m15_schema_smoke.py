import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _assert_schema_and_example(schema_name: str) -> None:
    schema_path = _repo_root() / "docs" / "schemas" / "validation" / f"{schema_name}.schema.json"
    assert schema_path.exists(), f"missing schema: {schema_path}"

    schema = json.loads(schema_path.read_text())
    assert schema.get("$schema"), "schema must declare $schema"
    assert schema.get("$id"), "schema must declare $id"
    assert schema.get("type") == "object"

    example_path = _repo_root() / "docs" / "specs" / f"{schema_name}.example.json"
    assert example_path.exists(), f"missing example: {example_path}"
    example = json.loads(example_path.read_text())
    assert example.get("schema_version") == schema_name

    try:
        import jsonschema  # type: ignore
    except Exception:
        jsonschema = None

    if jsonschema is not None:
        jsonschema.validate(instance=example, schema=schema)


def test_m15_assessment_table_schema_and_example_smoke():
    _assert_schema_and_example("m15_config_v1")
    _assert_schema_and_example("m15_bundle_manifest_v1")
    _assert_schema_and_example("m15_assessment_table_v1")


def test_m15_map_schema_and_example_smoke():
    _assert_schema_and_example("m15_map_v1")


def test_m15_mar_schema_and_example_smoke():
    _assert_schema_and_example("m15_mar_v1")


def test_m15_profile_diff_schema_and_example_smoke():
    _assert_schema_and_example("m15_profile_diff_report_v1")
