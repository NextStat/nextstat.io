"""Sync tool reference doc sections from the canonical tool manifest.

Usage:
  python scripts/generate_tool_reference_docs.py
  python scripts/generate_tool_reference_docs.py --check
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _manifest_path() -> Path:
    return _repo_root() / "bindings" / "ns-py" / "python" / "nextstat" / "_tool_manifest_v1.json"


def _load_manifest_helper():
    module_path = _repo_root() / "bindings" / "ns-py" / "python" / "nextstat" / "_tool_manifest.py"
    spec = importlib.util.spec_from_file_location("nextstat._tool_manifest", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load manifest helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_manifest() -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    helper = _load_manifest_helper()
    manifest = json.loads(_manifest_path().read_text(encoding="utf-8"))
    helper.validate_tool_manifest(manifest)
    tools = manifest.get("tools")
    if not isinstance(tools, list):
        raise RuntimeError(f"Invalid manifest: missing tools list in {_manifest_path()}")
    return helper, manifest, tools


def _transport_tool(record: dict[str, Any], transport: str) -> dict[str, Any] | None:
    section = record.get(transport)
    if not isinstance(section, dict):
        return None
    tool = section.get("tool")
    return tool if isinstance(tool, dict) else None


def _description(record: dict[str, Any], transport: str) -> str:
    tool = _transport_tool(record, transport)
    if tool is None:
        fallback = "local" if transport == "server" else "server"
        tool = _transport_tool(record, fallback)
    if tool is None:
        return ""
    function = tool.get("function")
    if not isinstance(function, dict):
        return ""
    description = function.get("description")
    if not isinstance(description, str):
        return ""
    text = " ".join(description.split())
    parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    return parts[0].strip()


def _escape_md(text: str) -> str:
    return text.replace("|", "\\|")


def _policy_legend(records: list[dict[str, Any]], helper: Any) -> list[str]:
    seen: dict[str, str] = {}
    for record in records:
        policy = helper.get_server_policy(record)
        seen.setdefault(policy["reason_code"], policy["reason"])
    lines = ["Server policy codes:"]
    for reason_code, reason in seen.items():
        lines.append(f"- `{reason_code}`: {reason}")
    return lines


def _local_capability_matrix(records: list[dict[str, Any]], helper: Any) -> str:
    local_records = [record for record in records if _transport_tool(record, "local") is not None]
    server_names = {record["name"] for record in records if helper.get_server_policy(record)["availability"] == "exposed"}

    lines = [
        "Generated from `bindings/ns-py/python/nextstat/_tool_manifest_v1.json`.",
        "",
        f"Local tools: {len(local_records)}. Server-safe subset: {len(server_names)}.",
        "",
        *_policy_legend(records, helper),
        "",
        "| Tool | Server | Policy | Summary |",
        "|------|:------:|--------|---------|",
    ]
    for record in local_records:
        name = record["name"]
        summary = _escape_md(_description(record, "local"))
        policy = helper.get_server_policy(record)
        server = "Yes" if name in server_names else "No"
        lines.append(f"| `{name}` | {server} | `{policy['reason_code']}` | {summary} |")
    return "\n".join(lines)


def _server_subset_table(records: list[dict[str, Any]], helper: Any) -> str:
    server_records = [record for record in records if helper.get_server_policy(record)["availability"] == "exposed"]
    policy = helper.get_server_policy(server_records[0]) if server_records else None
    lines = [
        "Generated from `bindings/ns-py/python/nextstat/_tool_manifest_v1.json`.",
        "",
        f"Server-safe tools: {len(server_records)}.",
        f"Policy code: `{policy['reason_code']}`." if policy is not None else "",
        "",
        "| Tool | Summary |",
        "|------|---------|",
    ]
    for record in server_records:
        name = record["name"]
        summary = _escape_md(_description(record, "server"))
        lines.append(f"| `{name}` | {summary} |")
    return "\n".join(lines)


def _guidance_section(helper: Any, transport: str) -> str:
    guidance = helper.build_tool_guidance(transport)
    hints = guidance.get("hints", [])
    recipes = guidance.get("recipes", [])
    lines = [
        "Generated from `bindings/ns-py/python/nextstat/_tool_manifest_v1.json`.",
        "",
        "Transport hints:",
    ]
    for hint in hints:
        lines.append(f"- {hint}")
    for recipe in recipes:
        tools = ", ".join(f"`{tool}`" for tool in recipe["tools"])
        docs = ", ".join(f"`{doc}`" for doc in recipe["docs"])
        lines.extend(
            [
                "",
                f"### `{recipe['id']}` — {recipe['title']}",
                f"Summary: {recipe['summary']}",
                f"Tools: {tools}",
                f'Starter prompt: "{recipe["prompt"]}"',
                f"Docs: {docs}",
            ]
        )
    return "\n".join(lines)


def _replace_block(text: str, begin: str, end: str, body: str) -> str:
    pattern = re.compile(rf"{re.escape(begin)}.*?{re.escape(end)}", re.DOTALL)
    replacement = f"{begin}\n{body.rstrip()}\n{end}"
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"Failed to replace block {begin} .. {end}")
    return updated


def generate_docs() -> dict[Path, str]:
    helper, _, records = _load_manifest()

    tool_api_path = _repo_root() / "docs" / "references" / "tool-api.md"
    tool_api_text = tool_api_path.read_text(encoding="utf-8")
    tool_api_text = _replace_block(
        tool_api_text,
        "<!-- BEGIN GENERATED TOOL CAPABILITY MATRIX -->",
        "<!-- END GENERATED TOOL CAPABILITY MATRIX -->",
        _local_capability_matrix(records, helper),
    )
    tool_api_text = _replace_block(
        tool_api_text,
        "<!-- BEGIN GENERATED SERVER TOOL SUBSET -->",
        "<!-- END GENERATED SERVER TOOL SUBSET -->",
        _server_subset_table(records, helper),
    )
    tool_api_text = _replace_block(
        tool_api_text,
        "<!-- BEGIN GENERATED LOCAL GUIDANCE RECIPES -->",
        "<!-- END GENERATED LOCAL GUIDANCE RECIPES -->",
        _guidance_section(helper, "local"),
    )
    tool_api_text = _replace_block(
        tool_api_text,
        "<!-- BEGIN GENERATED SERVER GUIDANCE RECIPES -->",
        "<!-- END GENERATED SERVER GUIDANCE RECIPES -->",
        _guidance_section(helper, "server"),
    )

    server_api_path = _repo_root() / "docs" / "references" / "server-api.md"
    server_api_text = server_api_path.read_text(encoding="utf-8")
    server_api_text = _replace_block(
        server_api_text,
        "<!-- BEGIN GENERATED SERVER TOOL SUBSET -->",
        "<!-- END GENERATED SERVER TOOL SUBSET -->",
        _server_subset_table(records, helper),
    )
    server_api_text = _replace_block(
        server_api_text,
        "<!-- BEGIN GENERATED SERVER GUIDANCE RECIPES -->",
        "<!-- END GENERATED SERVER GUIDANCE RECIPES -->",
        _guidance_section(helper, "server"),
    )

    return {
        tool_api_path: tool_api_text,
        server_api_path: server_api_text,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated output differs")
    args = parser.parse_args(argv)

    generated = generate_docs()
    dirty: list[Path] = []

    for path, content in generated.items():
        current = path.read_text(encoding="utf-8")
        if current != content:
            dirty.append(path)
            if not args.check:
                path.write_text(content, encoding="utf-8")

    if args.check and dirty:
        for path in dirty:
            print(f"out of date: {path}", file=sys.stderr)
        return 1

    if not args.check:
        for path in generated:
            print(f"Wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
