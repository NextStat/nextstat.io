import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_m15_schemas_and_examples_smoke():
    pairs = [
        (
            "docs/schemas/validation/m15_config_v1.schema.json",
            "docs/specs/m15_config_v1.example.json",
            "m15_config_v1",
        ),
        (
            "docs/schemas/validation/m15_assessment_table_v1.schema.json",
            "docs/specs/m15_assessment_table_v1.example.json",
            "m15_assessment_table_v1",
        ),
        (
            "docs/schemas/validation/m15_map_v1.schema.json",
            "docs/specs/m15_map_v1.example.json",
            "m15_map_v1",
        ),
        (
            "docs/schemas/validation/m15_mar_v1.schema.json",
            "docs/specs/m15_mar_v1.example.json",
            "m15_mar_v1",
        ),
        (
            "docs/schemas/validation/m15_profile_diff_report_v1.schema.json",
            "docs/specs/m15_profile_diff_report_v1.example.json",
            "m15_profile_diff_report_v1",
        ),
        (
            "docs/schemas/validation/m15_bundle_manifest_v1.schema.json",
            "docs/specs/m15_bundle_manifest_v1.example.json",
            "m15_bundle_manifest_v1",
        ),
    ]

    try:
        import jsonschema  # type: ignore
    except Exception:
        jsonschema = None

    expected_profile_enum = ["ich_core", "ema_step5_2026", "fda_draft_2024"]

    for schema_rel, example_rel, version in pairs:
        schema_path = _repo_root() / schema_rel
        example_path = _repo_root() / example_rel

        assert schema_path.exists(), f"missing schema: {schema_path}"
        assert example_path.exists(), f"missing example: {example_path}"

        schema = json.loads(schema_path.read_text())
        example = json.loads(example_path.read_text())

        assert schema.get("$schema"), f"{schema_rel} must declare $schema"
        assert schema.get("$id"), f"{schema_rel} must declare $id"
        assert schema.get("type") == "object", f"{schema_rel} must be an object schema"
        assert example.get("schema_version") == version, f"{example_rel} must declare {version}"
        if "jurisdiction_profile" in schema["properties"]:
            assert (
                schema["properties"]["jurisdiction_profile"].get("enum") == expected_profile_enum
            ), f"{schema_rel} must expose the canonical jurisdiction profile matrix"
        if "selected_profile" in schema["properties"]:
            assert (
                schema["properties"]["selected_profile"]["$ref"] == "#/$defs/jurisdiction_profile"
            ), f"{schema_rel} must expose the canonical selected_profile matrix"
            assert (
                schema["properties"]["compared_profiles"]["items"]["$ref"]
                == "#/$defs/jurisdiction_profile"
            ), f"{schema_rel} must expose the canonical compared_profiles matrix"
        if version in {"m15_assessment_table_v1", "m15_map_v1", "m15_mar_v1"}:
            assert (
                "profile_requirements" in schema["required"]
            ), f"{schema_rel} must require profile_requirements"
            assert (
                schema["properties"]["profile_requirements"]["$ref"]
                == "#/$defs/profile_requirements"
            ), f"{schema_rel} must define profile_requirements via a reusable schema"

        if jsonschema is not None:
            jsonschema.validate(instance=example, schema=schema)
