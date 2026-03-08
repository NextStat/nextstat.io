"""Sync strict tool-result schemas from the canonical tool manifest.

This script does not try to derive result-shape definitions from runtime code.
Those strict payload shapes remain explicit JSON Schema contracts. What this
script owns is the manifest-driven linkage between:

- tool names present in strict schemas
- per-tool `$defs` references used in result validation conditions

Usage:
  python scripts/generate_tool_contract_schemas.py
  python scripts/generate_tool_contract_schemas.py --check
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _manifest_path() -> Path:
    return _repo_root() / "bindings" / "ns-py" / "python" / "nextstat" / "_tool_manifest_v1.json"


def _strict_paths() -> tuple[Path, Path]:
    root = _repo_root() / "docs" / "schemas" / "tools"
    return (
        root / "nextstat_tool_result_strict_v1.schema.json",
        root / "nextstat_tool_result_server_strict_v1.schema.json",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest_helper():
    module_path = _repo_root() / "bindings" / "ns-py" / "python" / "nextstat" / "_tool_manifest.py"
    spec = importlib.util.spec_from_file_location("nextstat._tool_manifest", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load manifest helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dump_json(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=False) + "\n"


def _tool_mapping(record: dict[str, Any]) -> tuple[str, str] | None:
    name = record.get("name")
    strict_ref = record.get("strict_result_ref")
    if isinstance(name, str) and isinstance(strict_ref, str):
        return name, strict_ref
    return None


def _is_server_exposed(record: dict[str, Any]) -> bool:
    server = record.get("server")
    return isinstance(server, dict) and isinstance(server.get("tool"), dict)


def _base_all_of(schema: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for clause in schema.get("allOf", []):
        tool_const = (
            clause.get("if", {})
            .get("properties", {})
            .get("meta", {})
            .get("properties", {})
            .get("tool_name", {})
            .get("const")
        )
        if tool_const is None:
            out.append(clause)
    return out


def _tool_clause(tool_name: str, strict_ref: str) -> dict[str, Any]:
    return {
        "if": {
            "properties": {
                "ok": {"const": True},
                "meta": {
                    "properties": {"tool_name": {"const": tool_name}},
                    "required": ["tool_name"],
                },
            },
            "required": ["ok", "meta"],
        },
        "then": {"properties": {"result": {"$ref": f"#/$defs/{strict_ref}"}}},
    }


def _sync_schema(
    schema: dict[str, Any],
    *,
    tool_pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    synced = json.loads(json.dumps(schema))
    names = [name for name, _ in tool_pairs]

    meta = synced["properties"]["meta"]["properties"]["tool_name"]
    meta["enum"] = names

    base = _base_all_of(synced)
    synced["allOf"] = base + [_tool_clause(name, strict_ref) for name, strict_ref in tool_pairs]
    return synced


def generate_synced_schemas() -> dict[Path, str]:
    manifest = _load_json(_manifest_path())
    _load_manifest_helper().validate_tool_manifest(manifest)
    tools = manifest.get("tools")
    if not isinstance(tools, list):
        raise RuntimeError(f"Invalid manifest: missing tools list in {_manifest_path()}")

    local_pairs = [pair for record in tools if (pair := _tool_mapping(record)) is not None]
    server_pairs = [
        pair
        for record in tools
        if _is_server_exposed(record) and (pair := _tool_mapping(record)) is not None
    ]

    local_path, server_path = _strict_paths()
    local_schema = _sync_schema(_load_json(local_path), tool_pairs=local_pairs)
    server_schema = _sync_schema(_load_json(server_path), tool_pairs=server_pairs)

    return {
        local_path: _dump_json(local_schema),
        server_path: _dump_json(server_schema),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated output differs")
    args = parser.parse_args(argv)

    generated = generate_synced_schemas()
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
