from __future__ import annotations

import json
import math
from pathlib import Path

import nextstat

from _simplified_likelihood_public_fixture_catalog import (
    catalog_example_path,
    catalog_schema_path,
    load_catalog,
    resolve_workspace_path,
    simplified_workspace_schema_path,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_jsonschema(instance: dict, schema_path: Path) -> None:
    try:
        import jsonschema  # type: ignore
    except Exception:
        return

    schema = _load_json(schema_path)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(instance=instance, schema=schema)


def _observed_limit(value) -> float:
    if isinstance(value, tuple):
        return float(value[0])
    return float(value)


def test_simplified_likelihood_public_fixture_catalog_matches_schema_and_examples():
    catalog_path = catalog_example_path()
    schema_path = catalog_schema_path()
    assert catalog_path.exists(), f"missing catalog example: {catalog_path}"
    assert schema_path.exists(), f"missing catalog schema: {schema_path}"

    catalog = load_catalog()
    _validate_jsonschema(catalog, schema_path)

    assert catalog["schema_version"] == "nextstat_simplified_likelihood_public_fixture_catalog_v0"
    assert catalog["stability"] == "apex2"
    assert len(catalog["fixtures"]) == 3

    stable_surface = {"audit", "fit", "hypotest", "upper-limit", "scan"}
    research_surface = {"significance", "hypotest-toys", "viz-ranking"}

    for fixture in catalog["fixtures"]:
        workspace_path = resolve_workspace_path(fixture)
        assert workspace_path.exists(), f"missing workspace for fixture {fixture['fixture_id']}: {workspace_path}"

        workspace = _load_json(workspace_path)
        _validate_jsonschema(workspace, simplified_workspace_schema_path())

        metadata = workspace["metadata"]
        uncertainty_model = workspace["uncertainty_model"]

        assert metadata["experiment"] == fixture["experiment"]
        assert metadata["analysis_id"] == fixture["analysis_id"]
        assert metadata["source_format"] == fixture["source_format"]
        assert metadata["reference"] == fixture["reference"]
        assert uncertainty_model["kind"] == fixture["uncertainty_model_kind"]
        assert set(fixture["stable_commands"]) == stable_surface
        assert set(fixture["research_grade_commands"]) == research_surface
        assert fixture["ranking_semantics"] == "reduced_basis_only"

        if fixture["source_format"] == "derived_from_workspace":
            diagnostics = workspace.get("diagnostics", {})
            assert diagnostics.get("factorization"), "derived fixture must carry factorization diagnostics"
            assert diagnostics.get("fidelity"), "derived fixture must carry fidelity diagnostics"


def test_simplified_likelihood_public_fixture_catalog_runtime_smoke():
    catalog = load_catalog()

    for fixture in catalog["fixtures"]:
        workspace_path = resolve_workspace_path(fixture)
        workspace_json = workspace_path.read_text(encoding="utf-8")

        audit = nextstat.workspace_audit(workspace_json)
        assert audit["schema_version"] == "nextstat_simplified_likelihood_audit_v0"
        assert audit["input_schema_version"] == "nextstat_simplified_likelihood_v0"
        assert audit["source_format"] == fixture["source_format"]

        model = nextstat.HistFactoryModel.from_workspace(workspace_json)
        fit = nextstat.fit(model)
        assert fit.success

        cls = nextstat.hypotest(1.0, model)
        assert 0.0 <= float(cls) <= 1.0

        observed_limit = _observed_limit(
            nextstat.upper_limit(
                model,
                method="root",
                alpha=0.05,
                lo=0.0,
                hi=10.0,
                rtol=1e-3,
                max_iter=60,
            )
        )
        assert math.isfinite(observed_limit)

        scan = nextstat.profile_scan(model, [0.0, 1.0, 2.0])
        assert len(scan["points"]) == 3
        assert math.isfinite(float(scan["mu_hat"]))
