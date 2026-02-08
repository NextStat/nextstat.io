import json
from pathlib import Path
from typing import Any

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _assert_json_close(a: Any, b: Any, *, rtol: float, atol: float, path: str = "$") -> None:
    if _is_number(a) and _is_number(b):
        af = float(a)
        bf = float(b)
        diff = abs(af - bf)
        if diff <= atol:
            return
        denom = max(abs(af), abs(bf), 1.0)
        if diff / denom <= rtol:
            return
        raise AssertionError(f"{path}: {af} != {bf} (diff={diff}, rtol={rtol}, atol={atol})")

    if type(a) is not type(b):
        raise AssertionError(f"{path}: type mismatch {type(a)} != {type(b)}")

    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            raise AssertionError(f"{path}: keys mismatch {set(a.keys())} != {set(b.keys())}")
        for k in a.keys():
            _assert_json_close(a[k], b[k], rtol=rtol, atol=atol, path=f"{path}.{k}")
        return

    if isinstance(a, list):
        if len(a) != len(b):
            raise AssertionError(f"{path}: length mismatch {len(a)} != {len(b)}")
        for i, (ai, bi) in enumerate(zip(a, b)):
            _assert_json_close(ai, bi, rtol=rtol, atol=atol, path=f"{path}[{i}]")
        return

    if a != b:
        raise AssertionError(f"{path}: {a!r} != {b!r}")


def _normalize_envelope(x: dict[str, Any]) -> dict[str, Any]:
    # Golden comparisons should focus on semantics, not build metadata.
    x = json.loads(json.dumps(x))
    meta = x.get("meta", {})
    if isinstance(meta, dict):
        meta.pop("nextstat_version", None)
    return x


def test_tool_goldens_simple_workspace_deterministic():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    golden_path = (
        _repo_root()
        / "tests"
        / "fixtures"
        / "tool_goldens"
        / "simple_workspace_deterministic.v1.json"
    )
    assert golden_path.exists(), f"missing golden file: {golden_path} (run scripts/generate_tool_goldens.py)"

    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    assert golden.get("schema_version") == "nextstat.tool_goldens.v1"

    tools: dict[str, Any] = golden.get("tools", {})
    assert isinstance(tools, dict) and tools, "golden tools map must be non-empty"

    ws = (_repo_root() / "tests" / "fixtures" / "simple_workspace.json").read_text(encoding="utf-8")

    # Keep tolerances reasonably permissive across platforms; parity mode should be stable.
    rtol = 1e-6
    atol = 1e-8

    for name, golden_env in tools.items():
        # Reconstruct args used by generator.
        args: dict[str, Any] = {"workspace_json": ws, "execution": {"deterministic": True}}
        if name == "nextstat_hypotest":
            args["mu"] = 1.0
        elif name == "nextstat_hypotest_toys":
            args.update({"mu": 1.0, "n_toys": 200, "seed": 42})
        elif name == "nextstat_upper_limit":
            args.update({"expected": True})
        elif name == "nextstat_ranking":
            args.update({"top_n": 5})
        elif name == "nextstat_scan":
            args.update({"start": 0.0, "stop": 2.0, "points": 5})
        elif name in ("nextstat_workspace_audit", "nextstat_fit", "nextstat_discovery_asymptotic"):
            pass
        else:
            raise AssertionError(f"unknown golden tool name: {name}")

        got = execute_tool(name, args)
        assert got.get("ok") is True, got

        _assert_json_close(
            _normalize_envelope(got),
            _normalize_envelope(golden_env),
            rtol=rtol,
            atol=atol,
            path=f"tool:{name}",
        )

