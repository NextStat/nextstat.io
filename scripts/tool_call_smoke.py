"""Smoke runner for nextstat.tools.

Runs a small deterministic tool sequence against a workspace fixture and prints results.
Intended for quick local verification and CI debugging.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--transport",
        default="local",
        choices=["local", "server"],
        help="Tool execution transport: local (Python bindings) or server (nextstat-server HTTP).",
    )
    ap.add_argument(
        "--server-url",
        default="",
        help="Base URL for --transport=server (e.g. http://127.0.0.1:3742). "
        "If omitted, NEXTSTAT_SERVER_URL / NEXTSTAT_TOOLS_SERVER_URL may be used.",
    )
    ap.add_argument("--timeout-s", type=float, default=30.0, help="HTTP timeout in seconds (server mode).")
    ap.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable fallback to local execution when server calls fail.",
    )
    cli = ap.parse_args()

    from nextstat.tools import execute_tool

    ws = (_repo_root() / "tests" / "fixtures" / "simple_workspace.json").read_text(encoding="utf-8")

    calls = [
        ("nextstat_workspace_audit", {"workspace_json": ws}),
        ("nextstat_fit", {"workspace_json": ws}),
        ("nextstat_hypotest", {"workspace_json": ws, "mu": 1.0}),
        ("nextstat_upper_limit", {"workspace_json": ws, "expected": True}),
        ("nextstat_discovery_asymptotic", {"workspace_json": ws}),
    ]

    for name, tool_args in calls:
        tool_args = dict(tool_args)
        tool_args["execution"] = {"deterministic": True}
        out = execute_tool(
            name,
            tool_args,
            transport=cli.transport,
            server_url=(cli.server_url or None),
            timeout_s=float(cli.timeout_s),
            fallback_to_local=not bool(cli.no_fallback),
        )
        print(f"\n== {name} ==")
        print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
