import json
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_fixture_text(rel: str) -> str:
    p = _repo_root() / rel
    return p.read_text()


def test_tools_execute_tool_envelope_smoke():
    nextstat = pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    ws = _load_fixture_text("tests/fixtures/simple_workspace.json")

    r = execute_tool("nextstat_fit", {"workspace_json": ws, "execution": {"deterministic": True}})
    assert r.get("schema_version") == "nextstat.tool_result.v1"
    assert r.get("ok") is True, r
    assert isinstance(r.get("result"), dict)
    assert r.get("error") is None
    assert r.get("meta", {}).get("tool_name") == "nextstat_fit"
    assert r.get("meta", {}).get("nextstat_version") == getattr(nextstat, "__version__", None)


def test_tools_hypotest_and_upper_limit_shapes_smoke():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    ws = _load_fixture_text("tests/fixtures/simple_workspace.json")

    ht = execute_tool(
        "nextstat_hypotest",
        {"workspace_json": ws, "mu": 1.0, "execution": {"deterministic": True}},
    )
    assert ht.get("ok") is True, ht
    payload = ht["result"]
    assert set(payload.keys()) >= {"mu", "cls", "clsb", "clb"}
    assert 0.0 <= float(payload["cls"]) <= 1.0

    ul_obs = execute_tool(
        "nextstat_upper_limit",
        {"workspace_json": ws, "expected": False, "execution": {"deterministic": True}},
    )
    assert ul_obs.get("ok") is True, ul_obs
    assert "obs_limit" in ul_obs["result"]

    ul_exp = execute_tool(
        "nextstat_upper_limit",
        {"workspace_json": ws, "expected": True, "execution": {"deterministic": True}},
    )
    assert ul_exp.get("ok") is True, ul_exp
    assert "exp_limits" in ul_exp["result"]
    assert isinstance(ul_exp["result"]["exp_limits"], list)


def test_tools_discovery_asymptotic_smoke():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    ws = _load_fixture_text("tests/fixtures/simple_workspace.json")

    r = execute_tool(
        "nextstat_discovery_asymptotic",
        {"workspace_json": ws, "execution": {"deterministic": True}},
    )
    assert r.get("ok") is True, r
    payload = r["result"]
    assert set(payload.keys()) >= {"q0", "z0", "p0", "nll_hat", "nll_mu0"}
    assert 0.0 <= float(payload["p0"]) <= 1.0
