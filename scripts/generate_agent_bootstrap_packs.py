"""Generate model-specific agent bootstrap packs from canonical manifests.

Usage:
  python -m scripts.generate_agent_bootstrap_packs
  python -m scripts.generate_agent_bootstrap_packs --check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent, indent
from typing import Any

from scripts.bootstrap_artifact_paths import (
    bootstrap_pack_output_path as _pack_output_path,
    bootstrap_pack_relative_path as _pack_relative_path,
    bootstrap_provider_example_output_path as _provider_example_output_path,
    bootstrap_provider_example_relative_path as _provider_example_relative_path,
    bootstrap_reference_doc_output_path as _doc_path,
    bootstrap_reference_doc_relative_path as _reference_doc_relative_path,
)
from scripts import agent_bootstrap_profile_manifest as _profile_manifest
from scripts.repo_module_loader import load_repo_module


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_tool_manifest_helper():
    return load_repo_module(
        "nextstat._tool_manifest",
        "bindings/ns-py/python/nextstat/_tool_manifest.py",
    )


def _load_profile_manifest_helper():
    return _profile_manifest


def _pack_schema_path() -> Path:
    return (
        _repo_root()
        / "docs"
        / "schemas"
        / "tools"
        / "nextstat_agent_bootstrap_pack_v1.schema.json"
    )


def _workspace_output_path(workspace_output: dict[str, Any]) -> Path:
    return _repo_root() / workspace_output["path"]


def _replace_block(text: str, begin: str, end: str, body: str) -> str:
    start = text.find(begin)
    stop = text.find(end)
    if start == -1 or stop == -1 or stop < start:
        raise RuntimeError(f"Failed to replace block {begin} .. {end}")
    return text[: start + len(begin)] + "\n" + body.rstrip() + "\n" + text[stop:]


def _dedup(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _build_pack(profile: dict[str, Any], transport: str, tool_helper: Any) -> dict[str, Any]:
    guidance = tool_helper.build_tool_guidance(transport)
    transport_cfg = profile["transports"][transport]
    recipe_docs = [doc for recipe in guidance["recipes"] for doc in recipe["docs"]]
    return {
        "schema_version": "nextstat.agent_bootstrap_pack.v1",
        "client": profile["id"],
        "transport": transport,
        "title": f"{profile['title']} ({transport})",
        "summary": profile["summary"],
        "discovery_contract": {
            "descriptor_schema_version": "nextstat.tool_schema.v1",
            "callable_surface": "tools",
            "policy_surface": "capabilities",
            "guidance_surface": "guidance",
        },
        "instructions": _dedup(
            list(profile["instructions"]) + list(transport_cfg["instructions"]) + list(guidance["hints"])
        ),
        "recipes": guidance["recipes"],
        "snippets": transport_cfg["snippets"],
        "references": _dedup(
            list(profile["references"]) + list(transport_cfg["references"]) + recipe_docs
        ),
    }


def _profile_matrix(profiles: list[dict[str, Any]]) -> str:
    lines = [
        "Generated from `scripts/agent_bootstrap_profile_manifest_v1.json` and the canonical tool manifest guidance.",
        "",
        "| Client | Summary | Local Pack | Server Pack |",
        "|--------|---------|------------|-------------|",
    ]
    for profile in profiles:
        local_pack_path = _pack_relative_path(profile["id"], "local")
        server_pack_path = _pack_relative_path(profile["id"], "server")
        lines.append(
            f"| `{profile['id']}` | {profile['summary']} | "
            f"`{local_pack_path}` | "
            f"`{server_pack_path}` |"
        )
    return "\n".join(lines)


def _profile_details(profiles: list[dict[str, Any]], tool_helper: Any) -> str:
    lines: list[str] = [
        "Generated from `scripts/agent_bootstrap_profile_manifest_v1.json` plus transport-aware guidance from `bindings/ns-py/python/nextstat/_tool_manifest_v1.json`.",
    ]
    for profile in profiles:
        lines.extend(
            [
                "",
                f"## `{profile['id']}` — {profile['title']}",
                profile["summary"],
                "",
                "Global instructions:",
            ]
        )
        lines.extend(f"- {instruction}" for instruction in profile["instructions"])
        for transport in ("local", "server"):
            pack_path = _pack_relative_path(profile["id"], transport)
            guidance = tool_helper.build_tool_guidance(transport)
            transport_cfg = profile["transports"][transport]
            lines.extend(
                [
                    "",
                    f"### `{transport}`",
                    f"Pack: `{pack_path}`",
                    f"Recipes: {len(guidance['recipes'])}",
                    "Transport instructions:",
                ]
            )
            lines.extend(f"- {instruction}" for instruction in transport_cfg["instructions"])
            lines.extend(
                [
                    "Bootstrap snippet:",
                    f"```text\n{transport_cfg['snippets']['bootstrap']}\n```",
                    "Execution-loop snippet:",
                    f"```text\n{transport_cfg['snippets']['execution_loop']}\n```",
                ]
            )
    return "\n".join(lines)


def _workspace_file_matrix(workspace_output_profiles: list[dict[str, Any]]) -> str:
    lines = [
        "Generated from the canonical workspace-output bootstrap profiles and transport-aware tool guidance.",
        "",
        "| Profile | File | Purpose |",
        "|---------|------|---------|",
    ]
    for profile in workspace_output_profiles:
        for workspace_output in profile.get("workspace_outputs", []):
            lines.append(
                f"| `{profile['id']}` | `{workspace_output['path']}` | {workspace_output['purpose']} |"
            )
    return "\n".join(lines)


def _generate_pack_schema(profiles: list[dict[str, Any]]) -> str:
    client_ids = [profile["id"] for profile in profiles]
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://nextstat.io/schemas/tools/nextstat_agent_bootstrap_pack_v1.schema.json",
        "title": "NextStat Agent Bootstrap Pack v1",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "client",
            "transport",
            "title",
            "summary",
            "discovery_contract",
            "instructions",
            "recipes",
            "snippets",
            "references",
        ],
        "properties": {
            "schema_version": {"const": "nextstat.agent_bootstrap_pack.v1"},
            "client": {
                "type": "string",
                "enum": client_ids,
            },
            "transport": {
                "type": "string",
                "enum": ["local", "server"],
            },
            "title": {"type": "string", "minLength": 1},
            "summary": {"type": "string", "minLength": 1},
            "discovery_contract": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "descriptor_schema_version",
                    "callable_surface",
                    "policy_surface",
                    "guidance_surface",
                ],
                "properties": {
                    "descriptor_schema_version": {"const": "nextstat.tool_schema.v1"},
                    "callable_surface": {"const": "tools"},
                    "policy_surface": {"const": "capabilities"},
                    "guidance_surface": {"const": "guidance"},
                },
            },
            "instructions": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
            "recipes": {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/$defs/recipe"},
            },
            "snippets": {
                "type": "object",
                "additionalProperties": False,
                "required": ["bootstrap", "execution_loop"],
                "properties": {
                    "bootstrap": {"type": "string", "minLength": 1},
                    "execution_loop": {"type": "string", "minLength": 1},
                },
            },
            "references": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
        },
        "$defs": {
            "recipe": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "transport",
                    "title",
                    "summary",
                    "prompt",
                    "tools",
                    "docs",
                ],
                "properties": {
                    "id": {"type": "string", "pattern": "^[a-z0-9_]+$"},
                    "transport": {"type": "string", "enum": ["local", "server"]},
                    "title": {"type": "string", "minLength": 1},
                    "summary": {"type": "string", "minLength": 1},
                    "prompt": {"type": "string", "minLength": 1},
                    "tools": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "pattern": "^nextstat_[a-z0-9_]+$"},
                    },
                    "docs": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            }
        },
    }
    return json.dumps(schema, indent=2, sort_keys=False) + "\n"


def _runnable_example_profiles(profile_helper: Any, profile_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return profile_helper.get_runnable_example_profiles(profile_manifest)


def _provider_example_matrix(profiles: list[dict[str, Any]]) -> str:
    profile_labels = ", ".join(f"`{profile['id']}`" for profile in profiles)
    lines = [
        f"Generated from the canonical {profile_labels} bootstrap profiles plus transport-aware guidance.",
        "",
        "| Script | Purpose |",
        "|--------|---------|",
    ]
    for profile in profiles:
        runnable = profile["runnable_example"]
        client = profile["id"]
        lines.append(
            f"| `{_provider_example_relative_path(client, 'local')}` | {runnable['local_purpose']} |"
        )
        lines.append(
            f"| `{_provider_example_relative_path(client, 'server')}` | {runnable['server_purpose']} |"
        )
    return "\n".join(lines)


def _recipe_bullets(recipes: list[dict[str, Any]]) -> list[str]:
    return [f"- `{recipe['id']}` — {recipe['summary']}" for recipe in recipes]


def _generate_copilot_instructions(profile: dict[str, Any], tool_helper: Any) -> str:
    local_guidance = tool_helper.build_tool_guidance("local")
    server_guidance = tool_helper.build_tool_guidance("server")
    local_cfg = profile["transports"]["local"]
    server_cfg = profile["transports"]["server"]
    lines: list[str] = [
        "<!-- Generated by scripts/generate_agent_bootstrap_packs.py. Do not edit directly. -->",
        "# NextStat Copilot Instructions",
        "",
        "These instructions are generated from the canonical NextStat workspace-output bootstrap profile and tool manifest.",
        "",
        "## Core Rules",
    ]
    lines.extend(f"- {instruction}" for instruction in profile["instructions"])
    lines.extend(
        [
            "",
            "## Local Transport",
        ]
    )
    lines.extend(f"- {instruction}" for instruction in local_cfg["instructions"])
    lines.extend(
        [
            "",
            "Bootstrap snippet:",
            "```python",
            local_cfg["snippets"]["bootstrap"],
            "```",
            "",
            "Execution loop:",
            "```text",
            local_cfg["snippets"]["execution_loop"],
            "```",
            "",
            "Canonical local recipe seeds:",
        ]
    )
    lines.extend(_recipe_bullets(local_guidance["recipes"]))
    lines.extend(
        [
            "",
            "## Server Transport",
        ]
    )
    lines.extend(f"- {instruction}" for instruction in server_cfg["instructions"])
    lines.extend(
        [
            "",
            "Bootstrap snippet:",
            "```python",
            server_cfg["snippets"]["bootstrap"],
            "```",
            "",
            "Execution loop:",
            "```text",
            server_cfg["snippets"]["execution_loop"],
            "```",
            "",
            "Canonical server recipe seeds:",
        ]
    )
    lines.extend(_recipe_bullets(server_guidance["recipes"]))
    lines.extend(
        [
            "",
            "## References",
        ]
    )
    lines.extend(f"- `{ref}`" for ref in _dedup(list(profile["references"]) + list(local_cfg["references"]) + list(server_cfg["references"])))
    return "\n".join(lines) + "\n"


def _generate_cursor_rule(profile: dict[str, Any], tool_helper: Any) -> str:
    local_guidance = tool_helper.build_tool_guidance("local")
    server_guidance = tool_helper.build_tool_guidance("server")
    local_cfg = profile["transports"]["local"]
    server_cfg = profile["transports"]["server"]
    lines: list[str] = [
        "---",
        "description: NextStat tool-surface rules for Cursor and other editor assistants.",
        "globs: **/*",
        "alwaysApply: false",
        "---",
        "",
        "<!-- Generated by scripts/generate_agent_bootstrap_packs.py. Do not edit directly. -->",
        "# NextStat Tool Surface",
        "",
        "Use the canonical NextStat descriptor instead of inventing tools or transport behavior.",
        "",
        "## Core Rules",
    ]
    lines.extend(f"- {instruction}" for instruction in profile["instructions"])
    lines.extend(
        [
            "",
            "## Local Transport",
        ]
    )
    lines.extend(f"- {instruction}" for instruction in local_cfg["instructions"])
    lines.extend(
        [
            "",
            "Local bootstrap:",
            "```text",
            local_cfg["snippets"]["bootstrap"],
            "```",
            "",
            "Local execution loop:",
            "```text",
            local_cfg["snippets"]["execution_loop"],
            "```",
            "",
            "Local recipe seeds:",
        ]
    )
    lines.extend(_recipe_bullets(local_guidance["recipes"]))
    lines.extend(
        [
            "",
            "## Server Transport",
        ]
    )
    lines.extend(f"- {instruction}" for instruction in server_cfg["instructions"])
    lines.extend(
        [
            "",
            "Server bootstrap:",
            "```text",
            server_cfg["snippets"]["bootstrap"],
            "```",
            "",
            "Server execution loop:",
            "```text",
            server_cfg["snippets"]["execution_loop"],
            "```",
            "",
            "Server recipe seeds:",
        ]
    )
    lines.extend(_recipe_bullets(server_guidance["recipes"]))
    return "\n".join(lines) + "\n"


def _generate_provider_example(profile: dict[str, Any], transport: str) -> str:
    client = profile["id"]
    runnable = profile["runnable_example"]
    assert transport in {"local", "server"}
    title = profile["title"]
    pack_rel = _pack_relative_path(client, transport)
    payload_key = runnable["payload_key"]
    prompt_key = runnable["prompt_key"]
    instruction_key = runnable["instruction_key"]
    default_model = runnable["default_model"]
    descriptor_loader = (
        'descriptor = get_toolkit_descriptor(transport="local")'
        if transport == "local"
        else dedent(
            """
            if not args.server_url:
                raise SystemExit("--server-url is required for server transport unless --print-pack is used")
            if not args.api_key:
                raise SystemExit("--api-key is required for server transport unless --print-pack is used")
            descriptor = get_toolkit_descriptor(
                transport="server",
                server_url=args.server_url,
                api_key=args.api_key,
            )
            """
        ).strip()
    )
    template = dedent(
        f"""\
        #!/usr/bin/env python3
        # Generated by scripts/generate_agent_bootstrap_packs.py. Do not edit directly.
        from __future__ import annotations

        import argparse
        import json
        import os
        from pathlib import Path

        CLIENT = "{client}"
        TRANSPORT = "{transport}"
        TITLE = "{title}"
        PACK_RELATIVE_PATH = "{pack_rel}"
        DEFAULT_MODEL = "{default_model}"
        PROVIDER_PAYLOAD_KEY = "{payload_key}"
        PROVIDER_PROMPT_KEY = "{prompt_key}"
        PROVIDER_INSTRUCTION_KEY = "{instruction_key}"


        def _repo_root() -> Path:
            return Path(__file__).resolve().parents[4]


        def _load_pack() -> dict:
            return json.loads((_repo_root() / PACK_RELATIVE_PATH).read_text(encoding="utf-8"))


        def _build_parser() -> argparse.ArgumentParser:
            parser = argparse.ArgumentParser(
                description=f"Runnable NextStat {{CLIENT}} {{TRANSPORT}} provider example generated from canonical bootstrap packs."
            )
            parser.add_argument("--model", default=DEFAULT_MODEL, help="Provider model name.")
            parser.add_argument(
                "--prompt",
                default="Audit the available NextStat tools and choose the best matching recipe.",
                help="Provider prompt or task seed.",
            )
            parser.add_argument("--recipe-id", default=None, help="Optional recipe id filter.")
            parser.add_argument(
                "--print-pack",
                action="store_true",
                help="Print the generated bootstrap pack without loading a live NextStat descriptor.",
            )
            parser.add_argument(
                "--server-url",
                default=os.environ.get("NEXTSTAT_TOOLS_SERVER_URL"),
                help="nextstat-server URL for server transport.",
            )
            parser.add_argument(
                "--api-key",
                default=os.environ.get("NEXTSTAT_TOOLS_API_KEY") or os.environ.get("NEXTSTAT_SERVER_API_KEY"),
                help="Bearer API key for server transport.",
            )
            return parser


        def _select_recipes(pack: dict, recipe_id: str | None) -> list[dict]:
            recipes = pack["recipes"]
            if recipe_id is None:
                return recipes
            selected = [recipe for recipe in recipes if recipe["id"] == recipe_id]
            if not selected:
                raise SystemExit(f"Unknown recipe id: {{recipe_id}}")
            return selected


        def _build_request(pack: dict, descriptor: dict, model: str, prompt: str, recipes: list[dict]) -> dict:
            return {{
                "provider": CLIENT,
                "transport": TRANSPORT,
                "title": TITLE,
                "model": model,
                PROVIDER_PROMPT_KEY: prompt,
                PROVIDER_INSTRUCTION_KEY: "\\n".join(pack["instructions"]),
                PROVIDER_PAYLOAD_KEY: descriptor["tools"],
                "guidance_hints": descriptor["guidance"]["hints"],
                "recipe_ids": [recipe["id"] for recipe in recipes],
                "recipes": recipes,
                "bootstrap_snippet": pack["snippets"]["bootstrap"],
                "execution_loop_snippet": pack["snippets"]["execution_loop"],
                "references": pack["references"],
            }}


        def main() -> int:
            parser = _build_parser()
            args = parser.parse_args()
            pack = _load_pack()
            if args.print_pack:
                print(json.dumps(pack, indent=2, sort_keys=True))
                return 0

            from nextstat.tools import get_toolkit_descriptor

        __DESCRIPTOR_LOADER__

            recipes = _select_recipes(pack, args.recipe_id)
            request = _build_request(pack, descriptor, args.model, args.prompt, recipes)
            print(json.dumps(request, indent=2, sort_keys=True))
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    )
    return template.replace("__DESCRIPTOR_LOADER__", indent(descriptor_loader, " " * 4)) + "\n"


def _generate_anthropic_example(profile: dict[str, Any], transport: str) -> str:
    client = profile["id"]
    runnable = profile["runnable_example"]
    assert transport in {"local", "server"}
    title = profile["title"]
    pack_rel = _pack_relative_path(client, transport)
    payload_key = runnable["payload_key"]
    prompt_key = runnable["prompt_key"]
    instruction_key = runnable["instruction_key"]
    default_model = runnable["default_model"]
    descriptor_loader = (
        'descriptor = get_toolkit_descriptor(transport="local")'
        if transport == "local"
        else dedent(
            """
            if not args.server_url:
                raise SystemExit("--server-url is required for server transport unless --print-pack is used")
            if not args.api_key:
                raise SystemExit("--api-key is required for server transport unless --print-pack is used")
            descriptor = get_toolkit_descriptor(
                transport="server",
                server_url=args.server_url,
                api_key=args.api_key,
            )
            """
        ).strip()
    )
    template = dedent(
        f"""\
        #!/usr/bin/env python3
        # Generated by scripts/generate_agent_bootstrap_packs.py. Do not edit directly.
        from __future__ import annotations

        import argparse
        import json
        import os
        from pathlib import Path

        CLIENT = "{client}"
        TRANSPORT = "{transport}"
        TITLE = "{title}"
        PACK_RELATIVE_PATH = "{pack_rel}"
        DEFAULT_MODEL = "{default_model}"
        PROVIDER_PAYLOAD_KEY = "{payload_key}"
        PROVIDER_PROMPT_KEY = "{prompt_key}"
        PROVIDER_INSTRUCTION_KEY = "{instruction_key}"


        def _repo_root() -> Path:
            return Path(__file__).resolve().parents[4]


        def _load_pack() -> dict:
            return json.loads((_repo_root() / PACK_RELATIVE_PATH).read_text(encoding="utf-8"))


        def _build_parser() -> argparse.ArgumentParser:
            parser = argparse.ArgumentParser(
                description=f"Runnable NextStat {{CLIENT}} {{TRANSPORT}} Anthropic tool-use example generated from canonical bootstrap packs."
            )
            parser.add_argument("--model", default=DEFAULT_MODEL, help="Provider model name.")
            parser.add_argument(
                "--prompt",
                default="Audit the available NextStat tools and choose the best matching recipe.",
                help="User prompt or task seed.",
            )
            parser.add_argument("--recipe-id", default=None, help="Optional recipe id filter.")
            parser.add_argument(
                "--print-pack",
                action="store_true",
                help="Print the generated bootstrap pack without loading a live NextStat descriptor.",
            )
            parser.add_argument(
                "--server-url",
                default=os.environ.get("NEXTSTAT_TOOLS_SERVER_URL"),
                help="nextstat-server URL for server transport.",
            )
            parser.add_argument(
                "--api-key",
                default=os.environ.get("NEXTSTAT_TOOLS_API_KEY") or os.environ.get("NEXTSTAT_SERVER_API_KEY"),
                help="Bearer API key for server transport.",
            )
            return parser


        def _select_recipes(pack: dict, recipe_id: str | None) -> list[dict]:
            recipes = pack["recipes"]
            if recipe_id is None:
                return recipes
            selected = [recipe for recipe in recipes if recipe["id"] == recipe_id]
            if not selected:
                raise SystemExit(f"Unknown recipe id: {{recipe_id}}")
            return selected


        def _build_request(pack: dict, descriptor: dict, model: str, prompt: str, recipes: list[dict]) -> dict:
            return {{
                "provider": CLIENT,
                "transport": TRANSPORT,
                "title": TITLE,
                "model": model,
                PROVIDER_INSTRUCTION_KEY: "\\n".join(pack["instructions"]),
                PROVIDER_PROMPT_KEY: [{{"role": "user", "content": prompt}}],
                PROVIDER_PAYLOAD_KEY: descriptor["tools"],
                "guidance_hints": descriptor["guidance"]["hints"],
                "recipe_ids": [recipe["id"] for recipe in recipes],
                "recipes": recipes,
                "bootstrap_snippet": pack["snippets"]["bootstrap"],
                "execution_loop_snippet": pack["snippets"]["execution_loop"],
                "references": pack["references"],
            }}


        def main() -> int:
            parser = _build_parser()
            args = parser.parse_args()
            pack = _load_pack()
            if args.print_pack:
                print(json.dumps(pack, indent=2, sort_keys=True))
                return 0

            from nextstat.tools import get_toolkit_descriptor

        __DESCRIPTOR_LOADER__

            recipes = _select_recipes(pack, args.recipe_id)
            request = _build_request(pack, descriptor, args.model, args.prompt, recipes)
            print(json.dumps(request, indent=2, sort_keys=True))
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    )
    return template.replace("__DESCRIPTOR_LOADER__", indent(descriptor_loader, " " * 4)) + "\n"


def _generate_mcp_example(profile: dict[str, Any], transport: str) -> str:
    runnable = profile["runnable_example"]
    assert transport in {"local", "server"}
    title = profile["title"]
    client = profile["id"]
    pack_rel = _pack_relative_path(client, transport)
    payload_key = runnable["payload_key"]
    descriptor_loader = (
        dedent(
            """
            from nextstat.tools import get_mcp_tools, get_toolkit_descriptor

            descriptor = get_toolkit_descriptor(transport="local")
            mcp_tools = get_mcp_tools()
            dispatch = {
                "mode": "handle_mcp_call",
                "entrypoint": "nextstat.tools.handle_mcp_call(name, arguments)",
            }
            """
        ).strip()
        if transport == "local"
        else dedent(
            """
            from nextstat.tools import get_toolkit_descriptor

            if not args.server_url:
                raise SystemExit("--server-url is required for server transport unless --print-pack is used")
            if not args.api_key:
                raise SystemExit("--api-key is required for server transport unless --print-pack is used")
            descriptor = get_toolkit_descriptor(
                transport="server",
                server_url=args.server_url,
                api_key=args.api_key,
            )
            mcp_tools = [
                {
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "inputSchema": tool["function"]["parameters"],
                }
                for tool in descriptor["tools"]
            ]
            dispatch = {
                "mode": "execute_tool_server",
                "entrypoint": "nextstat.tools.execute_tool(name, arguments, transport=\\"server\\", server_url=..., api_key=..., fallback_to_local=False)",
            }
            """
        ).strip()
    )
    template = dedent(
        f"""\
        #!/usr/bin/env python3
        # Generated by scripts/generate_agent_bootstrap_packs.py. Do not edit directly.
        from __future__ import annotations

        import argparse
        import json
        import os
        from pathlib import Path

        CLIENT = "{client}"
        TRANSPORT = "{transport}"
        TITLE = "{title}"
        PACK_RELATIVE_PATH = "{pack_rel}"


        def _repo_root() -> Path:
            return Path(__file__).resolve().parents[4]


        def _load_pack() -> dict:
            return json.loads((_repo_root() / PACK_RELATIVE_PATH).read_text(encoding="utf-8"))


        def _build_parser() -> argparse.ArgumentParser:
            parser = argparse.ArgumentParser(
                description=f"Runnable NextStat MCP {{TRANSPORT}} bootstrap generated from canonical packs."
            )
            parser.add_argument("--recipe-id", default=None, help="Optional recipe id filter.")
            parser.add_argument(
                "--print-pack",
                action="store_true",
                help="Print the generated bootstrap pack without loading a live NextStat descriptor.",
            )
            parser.add_argument(
                "--server-url",
                default=os.environ.get("NEXTSTAT_TOOLS_SERVER_URL"),
                help="nextstat-server URL for server transport.",
            )
            parser.add_argument(
                "--api-key",
                default=os.environ.get("NEXTSTAT_TOOLS_API_KEY") or os.environ.get("NEXTSTAT_SERVER_API_KEY"),
                help="Bearer API key for server transport.",
            )
            return parser


        def _select_recipes(pack: dict, recipe_id: str | None) -> list[dict]:
            recipes = pack["recipes"]
            if recipe_id is None:
                return recipes
            selected = [recipe for recipe in recipes if recipe["id"] == recipe_id]
            if not selected:
                raise SystemExit(f"Unknown recipe id: {{recipe_id}}")
            return selected


        def _build_config(pack: dict, descriptor: dict, mcp_tools: list[dict], recipes: list[dict], dispatch: dict) -> dict:
            return {{
                "client": CLIENT,
                "transport": TRANSPORT,
                "title": TITLE,
                "tool_server_name": "nextstat" if TRANSPORT == "local" else "nextstat-server",
                "{payload_key}": mcp_tools,
                "guidance_hints": descriptor["guidance"]["hints"],
                "recipe_ids": [recipe["id"] for recipe in recipes],
                "recipes": recipes,
                "bootstrap_snippet": pack["snippets"]["bootstrap"],
                "execution_loop_snippet": pack["snippets"]["execution_loop"],
                "references": pack["references"],
                "dispatch": dispatch,
            }}


        def main() -> int:
            parser = _build_parser()
            args = parser.parse_args()
            pack = _load_pack()
            if args.print_pack:
                print(json.dumps(pack, indent=2, sort_keys=True))
                return 0

        __DESCRIPTOR_LOADER__

            recipes = _select_recipes(pack, args.recipe_id)
            config = _build_config(pack, descriptor, mcp_tools, recipes, dispatch)
            print(json.dumps(config, indent=2, sort_keys=True))
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    )
    return template.replace("__DESCRIPTOR_LOADER__", indent(descriptor_loader, " " * 4)) + "\n"


def _workspace_output_renderer_generators():
    return {
        "copilot_instructions": _generate_copilot_instructions,
        "cursor_rule": _generate_cursor_rule,
    }


def _runnable_renderer_generators():
    return {
        "provider_example": _generate_provider_example,
        "anthropic_example": _generate_anthropic_example,
        "mcp_example": _generate_mcp_example,
    }


def _validate_renderer_generator_coverage(
    workspace_output_registry: dict[str, Any],
    runnable_template_registry: dict[str, Any],
) -> None:
    workspace_renderers = {entry["renderer"] for entry in workspace_output_registry.values()}
    runnable_renderers = {entry["renderer"] for entry in runnable_template_registry.values()}
    workspace_generator_renderers = set(_workspace_output_renderer_generators())
    runnable_generator_renderers = set(_runnable_renderer_generators())
    if workspace_renderers != workspace_generator_renderers:
        raise RuntimeError(
            "workspace-output renderer generators must exactly cover manifest registries: "
            f"expected={sorted(workspace_renderers)} actual={sorted(workspace_generator_renderers)}"
        )
    if runnable_renderers != runnable_generator_renderers:
        raise RuntimeError(
            "runnable renderer generators must exactly cover manifest registries: "
            f"expected={sorted(runnable_renderers)} actual={sorted(runnable_generator_renderers)}"
        )


def _resolve_workspace_output_renderer_generator(renderer: str):
    generator = _workspace_output_renderer_generators().get(renderer)
    if not callable(generator):
        raise RuntimeError(f"Unsupported workspace-output bootstrap renderer: {renderer}")
    return generator


def _resolve_runnable_renderer_generator(renderer: str):
    generator = _runnable_renderer_generators().get(renderer)
    if not callable(generator):
        raise RuntimeError(f"Unsupported runnable bootstrap renderer: {renderer}")
    return generator


def _generate_reference_doc(
    profiles: list[dict[str, Any]],
    runnable_profiles: list[dict[str, Any]],
    workspace_output_profiles: list[dict[str, Any]],
    tool_helper: Any,
) -> str:
    path = _doc_path()
    text = path.read_text(encoding="utf-8")
    text = _replace_block(
        text,
        "<!-- BEGIN GENERATED IDE WORKSPACE FILES -->",
        "<!-- END GENERATED IDE WORKSPACE FILES -->",
        _workspace_file_matrix(workspace_output_profiles),
    )
    text = _replace_block(
        text,
        "<!-- BEGIN GENERATED PROVIDER EXAMPLES -->",
        "<!-- END GENERATED PROVIDER EXAMPLES -->",
        _provider_example_matrix(runnable_profiles),
    )
    text = _replace_block(
        text,
        "<!-- BEGIN GENERATED AGENT PROFILE MATRIX -->",
        "<!-- END GENERATED AGENT PROFILE MATRIX -->",
        _profile_matrix(profiles),
    )
    text = _replace_block(
        text,
        "<!-- BEGIN GENERATED AGENT PROFILE DETAILS -->",
        "<!-- END GENERATED AGENT PROFILE DETAILS -->",
        _profile_details(profiles, tool_helper),
    )
    return text


def generate_outputs() -> dict[Path, str]:
    tool_helper = _load_tool_manifest_helper()
    profile_helper = _load_profile_manifest_helper()
    profile_manifest = profile_helper.load_agent_bootstrap_profile_manifest()
    profiles = profile_manifest["profiles"]
    runnable_profiles = _runnable_example_profiles(profile_helper, profile_manifest)
    workspace_output_profiles = profile_helper.get_workspace_output_profiles(profile_manifest)
    workspace_output_registry = profile_helper.get_workspace_output_template_registry(profile_manifest)
    runnable_template_registry = profile_helper.get_runnable_template_registry(profile_manifest)
    _validate_renderer_generator_coverage(workspace_output_registry, runnable_template_registry)

    outputs: dict[Path, str] = {}
    for profile in profiles:
        for transport in ("local", "server"):
            pack = _build_pack(profile, transport, tool_helper)
            outputs[_pack_output_path(profile["id"], transport)] = json.dumps(
                pack, indent=2, sort_keys=True
            ) + "\n"

    outputs[_pack_schema_path()] = _generate_pack_schema(profiles)
    outputs[_doc_path()] = _generate_reference_doc(
        profiles, runnable_profiles, workspace_output_profiles, tool_helper
    )
    for profile in workspace_output_profiles:
        for workspace_output in profile.get("workspace_outputs", []):
            workspace_template = workspace_output_registry[workspace_output["template_family"]]
            generator = _resolve_workspace_output_renderer_generator(workspace_template["renderer"])
            outputs[_workspace_output_path(workspace_output)] = generator(profile, tool_helper)
    for profile in runnable_profiles:
        for transport in ("local", "server"):
            template_family = profile["runnable_example"]["template_family"]
            generator = _resolve_runnable_renderer_generator(
                runnable_template_registry[template_family]["renderer"]
            )
            outputs[_provider_example_output_path(profile["id"], transport)] = generator(
                profile, transport
            )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated output differs")
    args = parser.parse_args(argv)

    generated = generate_outputs()
    dirty: list[Path] = []

    for path, content in generated.items():
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current != content:
            dirty.append(path)
            if not args.check:
                path.parent.mkdir(parents=True, exist_ok=True)
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
