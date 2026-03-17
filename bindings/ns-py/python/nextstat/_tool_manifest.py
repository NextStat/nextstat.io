from __future__ import annotations

import base64
import copy
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


_MANIFEST_FILENAME = "_tool_manifest_v1.json"
_TOOL_NAME_RE = re.compile(r"^nextstat_[a-z0-9_]+$")
_POLICY_CODE_RE = re.compile(r"^[a-z0-9_]+$")
_GUIDANCE_ID_RE = re.compile(r"^[a-z0-9_]+$")


def _manifest_path() -> Path:
    return Path(__file__).with_name(_MANIFEST_FILENAME)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _validate_tool_definition(tool: Any, *, record_name: str, path: str) -> None:
    _require(isinstance(tool, dict), f"{path} must be an object")
    _require(tool.get("type") == "function", f"{path}.type must be 'function'")

    function = tool.get("function")
    _require(isinstance(function, dict), f"{path}.function must be an object")
    _require(function.get("name") == record_name, f"{path}.function.name must equal {record_name!r}")
    _require(
        isinstance(function.get("description"), str) and function["description"].strip(),
        f"{path}.function.description must be a non-empty string",
    )

    parameters = function.get("parameters")
    _require(isinstance(parameters, dict), f"{path}.function.parameters must be an object")
    _require(parameters.get("type") == "object", f"{path}.function.parameters.type must be 'object'")
    _require(
        isinstance(parameters.get("properties"), dict),
        f"{path}.function.parameters.properties must be an object",
    )
    required = parameters.get("required")
    _require(isinstance(required, list), f"{path}.function.parameters.required must be a list")
    _require(
        all(isinstance(item, str) and item for item in required),
        f"{path}.function.parameters.required must contain only non-empty strings",
    )


def _validate_manifest_value(value: Any, *, path: str) -> None:
    if isinstance(value, dict):
        keys = set(value.keys())
        if keys == {"$fixture_text"}:
            ref = value["$fixture_text"]
            _require(isinstance(ref, str) and ref, f"{path}.$fixture_text must be a non-empty string")
            return
        if keys == {"$repo_path"}:
            ref = value["$repo_path"]
            _require(isinstance(ref, str) and ref, f"{path}.$repo_path must be a non-empty string")
            return
        if keys == {"$repo_bytes_base64"}:
            ref = value["$repo_bytes_base64"]
            _require(
                isinstance(ref, str) and ref,
                f"{path}.$repo_bytes_base64 must be a non-empty string",
            )
            return
        for key, nested in value.items():
            _require(isinstance(key, str) and key, f"{path} contains an invalid key")
            _validate_manifest_value(nested, path=f"{path}.{key}")
        return

    if isinstance(value, list):
        for idx, nested in enumerate(value):
            _validate_manifest_value(nested, path=f"{path}[{idx}]")
        return


def _validate_policy_rule(policy: Any, *, path: str) -> None:
    _require(isinstance(policy, dict), f"{path} must be an object")
    _require(set(policy.keys()) == {"reason_code", "reason"}, f"{path} must contain reason_code and reason")
    reason_code = policy.get("reason_code")
    _require(
        isinstance(reason_code, str) and _POLICY_CODE_RE.fullmatch(reason_code) is not None,
        f"{path}.reason_code must match {_POLICY_CODE_RE.pattern}",
    )
    reason = policy.get("reason")
    _require(isinstance(reason, str) and reason.strip(), f"{path}.reason must be a non-empty string")


def _is_server_exposed_record(record: dict[str, Any]) -> bool:
    server = record.get("server")
    return isinstance(server, dict) and isinstance(server.get("tool"), dict)


def _validate_policy_manifest(manifest: dict[str, Any], *, tool_names: set[str]) -> None:
    policies = manifest.get("policies")
    _require(isinstance(policies, dict), "manifest.policies must be an object")
    _require(set(policies.keys()) == {"server"}, "manifest.policies must contain only 'server'")

    server = policies.get("server")
    _require(isinstance(server, dict), "manifest.policies.server must be an object")
    _require(set(server.keys()).issubset({"defaults", "overrides"}), "manifest.policies.server has unsupported keys")

    defaults = server.get("defaults")
    _require(isinstance(defaults, dict), "manifest.policies.server.defaults must be an object")
    _require(
        set(defaults.keys()) == {"exposed", "local_only"},
        "manifest.policies.server.defaults must contain exposed and local_only",
    )
    _validate_policy_rule(defaults.get("exposed"), path="manifest.policies.server.defaults.exposed")
    _validate_policy_rule(defaults.get("local_only"), path="manifest.policies.server.defaults.local_only")

    overrides = server.get("overrides", {})
    _require(isinstance(overrides, dict), "manifest.policies.server.overrides must be an object")
    for tool_name, policy in overrides.items():
        _require(
            isinstance(tool_name, str) and tool_name in tool_names,
            f"manifest.policies.server.overrides contains unknown tool {tool_name!r}",
        )
        _validate_policy_rule(policy, path=f"manifest.policies.server.overrides[{tool_name!r}]")


def _is_repo_root(path: Path) -> bool:
    return (
        path / "docs" / "references" / "tool-api.md"
    ).exists() and (path / "scripts").is_dir()


def _repo_root() -> Path:
    env_root = os.environ.get("NEXTSTAT_REPO_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if _is_repo_root(candidate):
            return candidate

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if _is_repo_root(candidate):
            return candidate

    manifest_path = _manifest_path().resolve()
    for candidate in manifest_path.parents:
        if _is_repo_root(candidate):
            return candidate

    return _manifest_path().parents[4]


def _validate_guidance_manifest(
    manifest: dict[str, Any],
    *,
    local_tool_names: set[str],
    server_tool_names: set[str],
) -> None:
    guidance = manifest.get("guidance")
    _require(isinstance(guidance, dict), "manifest.guidance must be an object")
    _require(
        set(guidance.keys()) == {"transport_hints", "recipes"},
        "manifest.guidance must contain transport_hints and recipes",
    )

    transport_hints = guidance.get("transport_hints")
    _require(isinstance(transport_hints, dict), "manifest.guidance.transport_hints must be an object")
    _require(
        set(transport_hints.keys()) == {"local", "server"},
        "manifest.guidance.transport_hints must contain local and server",
    )
    for transport in ("local", "server"):
        hints = transport_hints.get(transport)
        _require(
            isinstance(hints, list) and hints,
            f"manifest.guidance.transport_hints.{transport} must be a non-empty list",
        )
        _require(
            all(isinstance(hint, str) and hint.strip() for hint in hints),
            f"manifest.guidance.transport_hints.{transport} must contain only non-empty strings",
        )

    recipes = guidance.get("recipes")
    _require(isinstance(recipes, list) and recipes, "manifest.guidance.recipes must be a non-empty list")
    seen_recipe_ids: set[str] = set()
    referenced_local: set[str] = set()
    referenced_server: set[str] = set()
    repo_root = _repo_root()

    for idx, recipe in enumerate(recipes):
        path = f"manifest.guidance.recipes[{idx}]"
        _require(isinstance(recipe, dict), f"{path} must be an object")
        _require(
            set(recipe.keys()) == {"id", "transport", "title", "summary", "prompt", "tools", "docs"},
            f"{path} must contain id, transport, title, summary, prompt, tools, and docs",
        )
        recipe_id = recipe.get("id")
        _require(
            isinstance(recipe_id, str) and _GUIDANCE_ID_RE.fullmatch(recipe_id) is not None,
            f"{path}.id must match {_GUIDANCE_ID_RE.pattern}",
        )
        _require(recipe_id not in seen_recipe_ids, f"Duplicate guidance recipe id: {recipe_id}")
        seen_recipe_ids.add(recipe_id)

        transport = recipe.get("transport")
        _require(transport in ("local", "server"), f"{path}.transport must be 'local' or 'server'")
        for field in ("title", "summary", "prompt"):
            value = recipe.get(field)
            _require(
                isinstance(value, str) and value.strip(),
                f"{path}.{field} must be a non-empty string",
            )

        tools = recipe.get("tools")
        _require(isinstance(tools, list) and tools, f"{path}.tools must be a non-empty list")
        _require(
            all(isinstance(tool_name, str) and tool_name.strip() for tool_name in tools),
            f"{path}.tools must contain only non-empty strings",
        )
        _require(len(set(tools)) == len(tools), f"{path}.tools must not contain duplicates")

        docs = recipe.get("docs")
        _require(isinstance(docs, list) and docs, f"{path}.docs must be a non-empty list")
        for doc_path in docs:
            _require(isinstance(doc_path, str) and doc_path.strip(), f"{path}.docs must contain only non-empty strings")
            _require((repo_root / doc_path).exists(), f"{path}.docs contains missing repo path: {doc_path}")

        allowed = local_tool_names if transport == "local" else server_tool_names
        for tool_name in tools:
            _require(
                tool_name in allowed,
                f"{path}.tools contains {tool_name!r}, which is not available on transport {transport!r}",
            )
        if transport == "local":
            referenced_local.update(tools)
        else:
            referenced_server.update(tools)

    _require(
        local_tool_names.issubset(referenced_local),
        "manifest.guidance.recipes must cover every local tool at least once",
    )
    _require(
        server_tool_names.issubset(referenced_server),
        "manifest.guidance.recipes must cover every server tool at least once",
    )


def validate_tool_manifest(manifest: dict[str, Any]) -> None:
    _require(
        manifest.get("schema_version") == "nextstat.tool_manifest.v1",
        f"Invalid tool manifest schema_version in {_manifest_path()}",
    )
    tools = manifest.get("tools")
    _require(isinstance(tools, list) and tools, f"Invalid tool manifest: missing tools list in {_manifest_path()}")

    seen_names: set[str] = set()
    local_tool_names: set[str] = set()
    server_tool_names: set[str] = set()
    for idx, record in enumerate(tools):
        path = f"tools[{idx}]"
        _require(isinstance(record, dict), f"{path} must be an object")

        name = record.get("name")
        _require(
            isinstance(name, str) and _TOOL_NAME_RE.fullmatch(name) is not None,
            f"{path}.name must be a non-empty nextstat_* string",
        )
        _require(name not in seen_names, f"Duplicate tool manifest name: {name}")
        seen_names.add(name)

        local = record.get("local")
        _require(isinstance(local, dict), f"{path}.local must be an object")
        _require(set(local.keys()) == {"tool"}, f"{path}.local must contain only 'tool'")
        _validate_tool_definition(local.get("tool"), record_name=name, path=f"{path}.local.tool")
        local_tool_names.add(name)

        server = record.get("server")
        if server is not None:
            _require(isinstance(server, dict), f"{path}.server must be an object when present")
            _require(set(server.keys()) == {"tool"}, f"{path}.server must contain only 'tool'")
            _validate_tool_definition(server.get("tool"), record_name=name, path=f"{path}.server.tool")
            server_tool_names.add(name)

        strict_result_ref = record.get("strict_result_ref")
        if strict_result_ref is not None:
            _require(
                isinstance(strict_result_ref, str) and strict_result_ref.strip(),
                f"{path}.strict_result_ref must be a non-empty string",
            )

        golden_cases = record.get("golden_cases")
        if golden_cases is not None:
            _require(isinstance(golden_cases, dict) and golden_cases, f"{path}.golden_cases must be a non-empty object")
            for case_name, case in golden_cases.items():
                case_path = f"{path}.golden_cases[{case_name!r}]"
                _require(isinstance(case_name, str) and case_name, f"{path}.golden_cases contains an invalid case name")
                _require(isinstance(case, dict), f"{case_path} must be an object")
                _require(set(case.keys()) == {"arguments"}, f"{case_path} must contain only 'arguments'")
                arguments = case.get("arguments")
                _require(isinstance(arguments, dict), f"{case_path}.arguments must be an object")
                _validate_manifest_value(arguments, path=f"{case_path}.arguments")

    _validate_policy_manifest(manifest, tool_names=seen_names)
    _validate_guidance_manifest(
        manifest,
        local_tool_names=local_tool_names,
        server_tool_names=server_tool_names,
    )


@lru_cache(maxsize=1)
def _load_tool_manifest_cached() -> dict[str, Any]:
    path = _manifest_path()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_tool_manifest(manifest)
    return manifest


def load_tool_manifest() -> dict[str, Any]:
    return copy.deepcopy(_load_tool_manifest_cached())


def get_tool_records() -> list[dict[str, Any]]:
    manifest = _load_tool_manifest_cached()
    tools = manifest.get("tools")
    if not isinstance(tools, list):
        raise RuntimeError(f"Invalid tool manifest: missing 'tools' list in { _manifest_path() }")
    return copy.deepcopy(tools)


def get_transport_tools(transport: str) -> list[dict[str, Any]]:
    if transport not in ("local", "server"):
        raise ValueError(f"Unknown transport: {transport!r}. Use 'local' or 'server'.")

    out: list[dict[str, Any]] = []
    for record in _load_tool_manifest_cached().get("tools", []):
        section = record.get(transport)
        if not isinstance(section, dict):
            continue
        tool = section.get("tool")
        if isinstance(tool, dict):
            out.append(copy.deepcopy(tool))
    return out


def get_tool_names_for_transport(transport: str) -> list[str]:
    return [
        tool["function"]["name"]
        for tool in get_transport_tools(transport)
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    ]


def get_tool_record(name: str) -> dict[str, Any] | None:
    for record in _load_tool_manifest_cached().get("tools", []):
        if record.get("name") == name:
            return copy.deepcopy(record)
    return None


def get_server_policy(name_or_record: str | dict[str, Any]) -> dict[str, Any]:
    manifest = _load_tool_manifest_cached()
    if isinstance(name_or_record, str):
        record = get_tool_record(name_or_record)
        if record is None:
            raise KeyError(f"Unknown tool in manifest: {name_or_record}")
    else:
        record = copy.deepcopy(name_or_record)

    server_policies = manifest["policies"]["server"]
    defaults = server_policies["defaults"]
    availability = "exposed" if _is_server_exposed_record(record) else "local_only"
    policy = copy.deepcopy(defaults[availability])
    overrides = server_policies.get("overrides", {})
    name = record.get("name")
    if isinstance(name, str) and isinstance(overrides.get(name), dict):
        policy.update(copy.deepcopy(overrides[name]))
    policy["availability"] = availability
    return policy


def build_tool_capabilities() -> list[dict[str, Any]]:
    capabilities: list[dict[str, Any]] = []
    for record in get_tool_records():
        name = record.get("name")
        if not isinstance(name, str):
            continue
        local_available = isinstance(record.get("local"), dict) and isinstance(
            record["local"].get("tool"), dict
        )
        server_available = isinstance(record.get("server"), dict) and isinstance(
            record["server"].get("tool"), dict
        )
        capabilities.append(
            {
                "name": name,
                "local_available": local_available,
                "server_available": server_available,
                "server_policy": get_server_policy(record),
            }
        )
    return capabilities


def build_tool_guidance(transport: str) -> dict[str, Any]:
    if transport not in ("local", "server"):
        raise ValueError(f"Unknown transport: {transport!r}. Use 'local' or 'server'.")
    guidance = copy.deepcopy(_load_tool_manifest_cached()["guidance"])
    transport_hints = guidance["transport_hints"][transport]
    recipes = [recipe for recipe in guidance["recipes"] if recipe.get("transport") == transport]
    return {
        "hints": transport_hints,
        "recipes": recipes,
    }


def build_toolkit_descriptor(transport: str) -> dict[str, Any]:
    if transport not in ("local", "server"):
        raise ValueError(f"Unknown transport: {transport!r}. Use 'local' or 'server'.")
    return {
        "schema_version": "nextstat.tool_schema.v1",
        "transport": transport,
        "tools": get_transport_tools(transport),
        "capabilities": build_tool_capabilities(),
        "guidance": build_tool_guidance(transport),
    }


def _resolve_manifest_value(value: Any, repo_root: Path) -> Any:
    if isinstance(value, dict):
        if set(value.keys()) == {"$fixture_text"}:
            rel = value["$fixture_text"]
            if not isinstance(rel, str):
                raise TypeError(f"$fixture_text must be a string, got: {type(rel)!r}")
            return (repo_root / rel).read_text(encoding="utf-8")
        if set(value.keys()) == {"$repo_path"}:
            rel = value["$repo_path"]
            if not isinstance(rel, str):
                raise TypeError(f"$repo_path must be a string, got: {type(rel)!r}")
            return str(repo_root / rel)
        if set(value.keys()) == {"$repo_bytes_base64"}:
            rel = value["$repo_bytes_base64"]
            if not isinstance(rel, str):
                raise TypeError(f"$repo_bytes_base64 must be a string, got: {type(rel)!r}")
            return base64.b64encode((repo_root / rel).read_bytes()).decode("ascii")
        return {k: _resolve_manifest_value(v, repo_root) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_manifest_value(item, repo_root) for item in value]
    return value


def resolve_golden_case_arguments(tool_name: str, case_name: str, repo_root: str | Path) -> dict[str, Any]:
    record = get_tool_record(tool_name)
    if record is None:
        raise KeyError(f"Unknown tool in manifest: {tool_name}")
    cases = record.get("golden_cases")
    if not isinstance(cases, dict) or case_name not in cases:
        raise KeyError(f"Tool {tool_name} has no golden case {case_name!r}")
    case = cases[case_name]
    if not isinstance(case, dict):
        raise TypeError(f"Invalid golden case for {tool_name}:{case_name}")
    arguments = case.get("arguments")
    if not isinstance(arguments, dict):
        raise TypeError(f"Golden case {tool_name}:{case_name} must contain 'arguments'")
    return _resolve_manifest_value(arguments, Path(repo_root))
