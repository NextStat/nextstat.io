"""Smoke runner for nextstat.tools.

Runs a small deterministic tool sequence against a workspace fixture and prints results.
Intended for quick local verification and CI debugging.
"""

from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    from nextstat.tools import execute_tool

    ws = (_repo_root() / "tests" / "fixtures" / "simple_workspace.json").read_text(encoding="utf-8")

    calls = [
        ("nextstat_workspace_audit", {"workspace_json": ws}),
        ("nextstat_fit", {"workspace_json": ws}),
        ("nextstat_hypotest", {"workspace_json": ws, "mu": 1.0}),
        ("nextstat_upper_limit", {"workspace_json": ws, "expected": True}),
        ("nextstat_discovery_asymptotic", {"workspace_json": ws}),
    ]

    for name, args in calls:
        args = dict(args)
        args["execution"] = {"deterministic": True}
        out = execute_tool(name, args)
        print(f"\n== {name} ==")
        print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

