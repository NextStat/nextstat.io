"""Generate deterministic tool golden outputs.

This script is intended to be run in an environment where the `nextstat` wheel
is installed (e.g. CI or bindings/ns-py/.venv).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_text(rel: str) -> str:
    return (_repo_root() / rel).read_text(encoding="utf-8")


def main() -> None:
    import nextstat  # noqa: F401
    from nextstat.tools import execute_tool

    ws = _load_text("tests/fixtures/simple_workspace.json")

    calls: list[tuple[str, dict[str, Any]]] = [
        ("nextstat_workspace_audit", {"workspace_json": ws}),
        ("nextstat_fit", {"workspace_json": ws}),
        ("nextstat_hypotest", {"workspace_json": ws, "mu": 1.0}),
        ("nextstat_hypotest_toys", {"workspace_json": ws, "mu": 1.0, "n_toys": 200, "seed": 42}),
        ("nextstat_upper_limit", {"workspace_json": ws, "expected": True}),
        ("nextstat_ranking", {"workspace_json": ws, "top_n": 5}),
        ("nextstat_scan", {"workspace_json": ws, "start": 0.0, "stop": 2.0, "points": 5}),
        ("nextstat_discovery_asymptotic", {"workspace_json": ws}),
    ]

    out: dict[str, Any] = {
        "schema_version": "nextstat.tool_goldens.v1",
        "fixture": "tests/fixtures/simple_workspace.json",
        "tools": {},
    }

    for name, args in calls:
        args = dict(args)
        args["execution"] = {"deterministic": True}
        out["tools"][name] = execute_tool(name, args)

    out_path = _repo_root() / "tests" / "fixtures" / "tool_goldens" / "simple_workspace_deterministic.v1.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

