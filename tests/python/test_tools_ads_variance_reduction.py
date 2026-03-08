from __future__ import annotations

import pytest

from tests._tool_contract_helpers import (
    build_canonical_ads_cuped_adjust_tool_case,
    build_canonical_ads_cure_adjust_tool_case,
)


def test_ads_variance_reduction_tools_are_advertised_in_local_descriptor():
    pytest.importorskip("nextstat")
    from nextstat.tools import get_toolkit_descriptor

    descriptor = get_toolkit_descriptor()
    tool_names = {tool["function"]["name"] for tool in descriptor["tools"]}
    capabilities = {entry["name"]: entry for entry in descriptor["capabilities"]}

    assert "nextstat_ads_cuped_adjust" in tool_names
    assert "nextstat_ads_cure_adjust" in tool_names

    cuped = capabilities["nextstat_ads_cuped_adjust"]
    assert cuped["local_available"] is True
    assert cuped["server_available"] is True
    assert cuped["server_policy"]["availability"] == "exposed"
    assert cuped["server_policy"]["reason_code"] == "server_safe_subset"

    cure = capabilities["nextstat_ads_cure_adjust"]
    assert cure["local_available"] is True
    assert cure["server_available"] is True
    assert cure["server_policy"]["availability"] == "exposed"
    assert cure["server_policy"]["reason_code"] == "server_safe_subset"


def test_execute_tool_ads_cuped_adjust_contract():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    name, arguments = build_canonical_ads_cuped_adjust_tool_case()
    out = execute_tool(name, {**arguments, "execution": {"deterministic": True}})

    assert out["ok"] is True, out
    assert out["meta"]["tool_name"] == name
    assert out["result"]["method"] == "cuped"
    assert out["result"]["num_covariates"] == 1
    assert out["result"]["selected_covariates"] == ["pre_clicks"]
    assert out["result"]["provenance_validated"] is True
    assert out["result"]["pre_treatment_only"] is True


def test_execute_tool_ads_cure_adjust_contract():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    name, arguments = build_canonical_ads_cure_adjust_tool_case()
    out = execute_tool(name, {**arguments, "execution": {"deterministic": True}})

    assert out["ok"] is True, out
    assert out["meta"]["tool_name"] == name
    assert out["result"]["method"] == "cure"
    assert out["result"]["num_covariates"] == 2
    assert out["result"]["selected_covariates"] == ["pre_clicks", "pre_impressions"]
    assert out["result"]["provenance_validated"] is True
    assert out["result"]["pre_treatment_only"] is True
