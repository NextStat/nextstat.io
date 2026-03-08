from __future__ import annotations

import json
from pathlib import Path

from _simplified_likelihood_export_public_case_catalog import (
    catalog_example_path,
    catalog_schema_path,
    load_catalog,
    resolve_workspace_path,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _modifier_types(workspace: dict) -> set[str]:
    return {
        str(modifier["type"])
        for channel in workspace.get("channels", [])
        for sample in channel.get("samples", [])
        for modifier in sample.get("modifiers", [])
    }


def _validate_jsonschema(instance: dict, schema_path: Path) -> None:
    try:
        import jsonschema  # type: ignore
    except Exception:
        return

    schema = _load_json(schema_path)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(instance=instance, schema=schema)


def test_simplified_likelihood_export_public_case_catalog_matches_schema_and_examples() -> None:
    catalog_path = catalog_example_path()
    schema_path = catalog_schema_path()
    assert catalog_path.exists(), f"missing catalog example: {catalog_path}"
    assert schema_path.exists(), f"missing catalog schema: {schema_path}"

    catalog = load_catalog()
    _validate_jsonschema(catalog, schema_path)

    assert catalog["schema_version"] == "nextstat_simplified_likelihood_export_public_case_catalog_v0"
    assert catalog["stability"] == "apex2"
    assert len(catalog["cases"]) == 8

    case_ids = {case["case_id"] for case in catalog["cases"]}
    assert "atlas_public_sr_cr_gaussian_export_stable_example" in case_ids
    assert "atlas_public_dual_sr_dual_cr_gaussian_export_stable_example" in case_ids
    assert "atlas_public_dual_sr_vr_dual_cr_gaussian_export_stable_example" in case_ids
    assert "cms_public_sr_cr_asymmetric_gaussian_export_stable_example" in case_ids
    assert "cms_public_sr_cr_export_stable_example" in case_ids
    assert "cms_public_dual_sr_cr_gaussian_export_stable_example" in case_ids
    assert "cms_public_sr_dual_cr_gaussian_export_stable_example" in case_ids
    assert "cms_public_sr_vr_dual_cr_gaussian_export_stable_example" in case_ids

    for case in catalog["cases"]:
        workspace_path = resolve_workspace_path(case)
        assert workspace_path.exists(), f"missing workspace for case {case['case_id']}: {workspace_path}"
        workspace = _load_json(workspace_path)
        assert case["case_kind"] == "public_reinterpretation_style"
        assert case["source_workspace_format"] == "pyhf"
        assert case["source_workspace_schema_version"] == "pyhf_workspace_v1"
        assert case["constraint_covariance_source"] == "source_model_constraints"
        assert case["output_uncertainty_model"] == "basis"
        assert case["reference"]
        assert case["selection"]["channels"]
        assert case["selection"]["bins"]
        assert _modifier_types(workspace).issubset({"histosys", "lumi", "normfactor", "normsys"})
