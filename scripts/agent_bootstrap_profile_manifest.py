#!/usr/bin/env python3
"""Helper for canonical NextStat agent bootstrap profiles."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any


_MANIFEST_FILENAME = "agent_bootstrap_profile_manifest_v1.json"
_PROFILE_ID_RE = re.compile(r"^[a-z0-9_]+$")
_WORKSPACE_OUTPUT_ID_RE = re.compile(r"^[a-z0-9_]+$")
_REGISTRY_ID_RE = re.compile(r"^[a-z0-9_]+$")
_RUNNABLE_FIELD_RE = re.compile(r"^[a-z_]+$")
_KNOWN_RUNNABLE_FIELDS = {
    "kind",
    "template_family",
    "local_purpose",
    "server_purpose",
    "payload_key",
    "prompt_key",
    "instruction_key",
    "default_model",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _manifest_path() -> Path:
    return Path(__file__).with_name(_MANIFEST_FILENAME)


def _schema_path() -> Path:
    return _repo_root() / "docs" / "schemas" / "tools" / "nextstat_agent_bootstrap_profile_manifest_v1.schema.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _validate_registries(manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    registries = manifest.get("registries")
    _require(isinstance(registries, dict), "manifest.registries must be an object")
    _require(
        set(registries.keys()) == {"workspace_output_template_families", "runnable_template_families"},
        "manifest.registries has unsupported keys",
    )

    workspace_registry = registries.get("workspace_output_template_families")
    _require(
        isinstance(workspace_registry, dict) and workspace_registry,
        "manifest.registries.workspace_output_template_families must be a non-empty object",
    )
    for template_family, entry in workspace_registry.items():
        _require(
            isinstance(template_family, str) and _REGISTRY_ID_RE.fullmatch(template_family) is not None,
            "workspace output template family ids must match ^[a-z0-9_]+$",
        )
        _require(
            isinstance(entry, dict) and set(entry.keys()) == {"renderer"},
            f"workspace output template family {template_family} must only define renderer",
        )
        _require(
            isinstance(entry.get("renderer"), str)
            and _REGISTRY_ID_RE.fullmatch(entry["renderer"]) is not None,
            f"workspace output template family {template_family}.renderer must match {_REGISTRY_ID_RE.pattern}",
        )

    runnable_registry = registries.get("runnable_template_families")
    _require(
        isinstance(runnable_registry, dict) and runnable_registry,
        "manifest.registries.runnable_template_families must be a non-empty object",
    )
    for template_family, entry in runnable_registry.items():
        _require(
            isinstance(template_family, str) and _REGISTRY_ID_RE.fullmatch(template_family) is not None,
            "runnable template family ids must match ^[a-z0-9_]+$",
        )
        _require(
            isinstance(entry, dict) and set(entry.keys()) == {"renderer", "allowed_kinds", "required_fields"},
            f"runnable template family {template_family} must define renderer, allowed_kinds, and required_fields",
        )
        _require(
            isinstance(entry.get("renderer"), str)
            and _REGISTRY_ID_RE.fullmatch(entry["renderer"]) is not None,
            f"runnable template family {template_family}.renderer must match {_REGISTRY_ID_RE.pattern}",
        )
        allowed_kinds = entry.get("allowed_kinds")
        _require(
            isinstance(allowed_kinds, list) and allowed_kinds,
            f"runnable template family {template_family}.allowed_kinds must be a non-empty list",
        )
        _require(
            len(set(allowed_kinds)) == len(allowed_kinds),
            f"runnable template family {template_family}.allowed_kinds must not contain duplicates",
        )
        _require(
            all(isinstance(kind, str) and _REGISTRY_ID_RE.fullmatch(kind) is not None for kind in allowed_kinds),
            f"runnable template family {template_family}.allowed_kinds must contain only valid ids",
        )
        required_fields = entry.get("required_fields")
        _require(
            isinstance(required_fields, list) and required_fields,
            f"runnable template family {template_family}.required_fields must be a non-empty list",
        )
        _require(
            len(set(required_fields)) == len(required_fields),
            f"runnable template family {template_family}.required_fields must not contain duplicates",
        )
        _require(
            all(isinstance(field, str) and _RUNNABLE_FIELD_RE.fullmatch(field) is not None for field in required_fields),
            f"runnable template family {template_family}.required_fields must contain only valid field names",
        )
        _require(
            set(required_fields).issubset(_KNOWN_RUNNABLE_FIELDS),
            f"runnable template family {template_family}.required_fields contains unsupported fields",
        )
        _require(
            {"kind", "template_family", "local_purpose", "server_purpose", "payload_key"}.issubset(required_fields),
            f"runnable template family {template_family}.required_fields must include core runnable fields",
        )

    return workspace_registry, runnable_registry


def validate_agent_bootstrap_profile_manifest(manifest: dict[str, Any]) -> None:
    _require(
        manifest.get("schema_version") == "nextstat.agent_bootstrap_profile_manifest.v1",
        f"Invalid agent bootstrap profile manifest schema_version in {_manifest_path()}",
    )
    workspace_output_registry, runnable_template_registry = _validate_registries(manifest)
    coverage_policy = manifest.get("coverage_policy")
    _require(isinstance(coverage_policy, dict), "manifest.coverage_policy must be an object")
    _require(
        set(coverage_policy.keys())
        == {
            "required_profile_ids",
            "required_runnable_profile_ids",
            "workspace_output_owner_profile_ids",
            "required_workspace_output_profile_ids",
        },
        "manifest.coverage_policy has unsupported keys",
    )
    required_profile_ids = coverage_policy.get("required_profile_ids")
    required_runnable_profile_ids = coverage_policy.get("required_runnable_profile_ids")
    workspace_output_owner_profile_ids = coverage_policy.get("workspace_output_owner_profile_ids")
    required_workspace_output_profile_ids = coverage_policy.get("required_workspace_output_profile_ids")
    _require(
        isinstance(required_profile_ids, list) and required_profile_ids,
        "manifest.coverage_policy.required_profile_ids must be a non-empty list",
    )
    _require(
        isinstance(required_runnable_profile_ids, list),
        "manifest.coverage_policy.required_runnable_profile_ids must be a list",
    )
    _require(
        isinstance(workspace_output_owner_profile_ids, list) and workspace_output_owner_profile_ids,
        "manifest.coverage_policy.workspace_output_owner_profile_ids must be a non-empty list",
    )
    _require(
        isinstance(required_workspace_output_profile_ids, list),
        "manifest.coverage_policy.required_workspace_output_profile_ids must be a list",
    )
    _require(
        all(isinstance(profile_id, str) and _PROFILE_ID_RE.fullmatch(profile_id) is not None for profile_id in required_profile_ids),
        "manifest.coverage_policy.required_profile_ids must contain only valid profile ids",
    )
    _require(
        all(isinstance(profile_id, str) and _PROFILE_ID_RE.fullmatch(profile_id) is not None for profile_id in required_runnable_profile_ids),
        "manifest.coverage_policy.required_runnable_profile_ids must contain only valid profile ids",
    )
    _require(
        all(
            isinstance(profile_id, str) and _PROFILE_ID_RE.fullmatch(profile_id) is not None
            for profile_id in workspace_output_owner_profile_ids
        ),
        "manifest.coverage_policy.workspace_output_owner_profile_ids must contain only valid profile ids",
    )
    _require(
        all(
            isinstance(profile_id, str) and _PROFILE_ID_RE.fullmatch(profile_id) is not None
            for profile_id in required_workspace_output_profile_ids
        ),
        "manifest.coverage_policy.required_workspace_output_profile_ids must contain only valid profile ids",
    )
    _require(
        len(set(required_profile_ids)) == len(required_profile_ids),
        "manifest.coverage_policy.required_profile_ids must not contain duplicates",
    )
    _require(
        len(set(required_runnable_profile_ids)) == len(required_runnable_profile_ids),
        "manifest.coverage_policy.required_runnable_profile_ids must not contain duplicates",
    )
    _require(
        len(set(workspace_output_owner_profile_ids)) == len(workspace_output_owner_profile_ids),
        "manifest.coverage_policy.workspace_output_owner_profile_ids must not contain duplicates",
    )
    _require(
        len(set(required_workspace_output_profile_ids)) == len(required_workspace_output_profile_ids),
        "manifest.coverage_policy.required_workspace_output_profile_ids must not contain duplicates",
    )
    required_profile_id_set = set(required_profile_ids)
    required_runnable_profile_id_set = set(required_runnable_profile_ids)
    workspace_output_owner_profile_id_set = set(workspace_output_owner_profile_ids)
    required_workspace_output_profile_id_set = set(required_workspace_output_profile_ids)
    _require(
        required_runnable_profile_id_set.issubset(required_profile_id_set),
        "manifest.coverage_policy.required_runnable_profile_ids must be a subset of required_profile_ids",
    )
    _require(
        workspace_output_owner_profile_id_set.issubset(required_profile_id_set),
        "manifest.coverage_policy.workspace_output_owner_profile_ids must be a subset of required_profile_ids",
    )
    _require(
        required_workspace_output_profile_id_set.issubset(workspace_output_owner_profile_id_set),
        "manifest.coverage_policy.required_workspace_output_profile_ids must be a subset of workspace_output_owner_profile_ids",
    )
    profiles = manifest.get("profiles")
    _require(isinstance(profiles, list) and profiles, "manifest.profiles must be a non-empty list")

    repo_root = _repo_root()
    seen_ids: set[str] = set()
    runnable_ids: set[str] = set()
    for idx, profile in enumerate(profiles):
        path = f"profiles[{idx}]"
        _require(isinstance(profile, dict), f"{path} must be an object")
        allowed_keys = {
            "id",
            "title",
            "summary",
            "references",
            "instructions",
            "transports",
            "runnable_example",
            "workspace_outputs",
        }
        _require(set(profile.keys()).issubset(allowed_keys), f"{path} has unsupported keys")
        profile_id = profile.get("id")
        _require(
            isinstance(profile_id, str) and _PROFILE_ID_RE.fullmatch(profile_id) is not None,
            f"{path}.id must match {_PROFILE_ID_RE.pattern}",
        )
        _require(profile_id not in seen_ids, f"Duplicate profile id: {profile_id}")
        seen_ids.add(profile_id)
        for field in ("title", "summary"):
            value = profile.get(field)
            _require(isinstance(value, str) and value.strip(), f"{path}.{field} must be a non-empty string")
        references = profile.get("references")
        _require(isinstance(references, list) and references, f"{path}.references must be a non-empty list")
        for ref in references:
            _require(isinstance(ref, str) and ref.strip(), f"{path}.references contains an invalid value")
            _require((repo_root / ref).exists(), f"{path}.references contains missing repo path: {ref}")
        instructions = profile.get("instructions")
        _require(
            isinstance(instructions, list) and instructions,
            f"{path}.instructions must be a non-empty list",
        )
        _require(
            all(isinstance(item, str) and item.strip() for item in instructions),
            f"{path}.instructions must contain only non-empty strings",
        )
        workspace_outputs = profile.get("workspace_outputs")
        if workspace_outputs is not None:
            _require(
                profile_id in workspace_output_owner_profile_id_set,
                f"{path}.workspace_outputs is only supported for profiles listed in manifest.coverage_policy.workspace_output_owner_profile_ids",
            )
            _require(
                isinstance(workspace_outputs, list) and workspace_outputs,
                f"{path}.workspace_outputs must be a non-empty list",
            )
            seen_workspace_output_ids: set[str] = set()
            seen_workspace_output_paths: set[str] = set()
            for output_idx, workspace_output in enumerate(workspace_outputs):
                output_path = f"{path}.workspace_outputs[{output_idx}]"
                _require(isinstance(workspace_output, dict), f"{output_path} must be an object")
                _require(
                    set(workspace_output.keys()) == {"id", "path", "template_family", "purpose"},
                    f"{output_path} has unsupported keys",
                )
                workspace_output_id = workspace_output.get("id")
                _require(
                    isinstance(workspace_output_id, str)
                    and _WORKSPACE_OUTPUT_ID_RE.fullmatch(workspace_output_id) is not None,
                    f"{output_path}.id must match {_WORKSPACE_OUTPUT_ID_RE.pattern}",
                )
                _require(
                    workspace_output_id not in seen_workspace_output_ids,
                    f"Duplicate workspace output id in {path}: {workspace_output_id}",
                )
                seen_workspace_output_ids.add(workspace_output_id)
                workspace_output_rel_path = workspace_output.get("path")
                _require(
                    isinstance(workspace_output_rel_path, str) and workspace_output_rel_path.strip(),
                    f"{output_path}.path must be a non-empty string",
                )
                _require(
                    workspace_output_rel_path not in seen_workspace_output_paths,
                    f"Duplicate workspace output path in {path}: {workspace_output_rel_path}",
                )
                seen_workspace_output_paths.add(workspace_output_rel_path)
                template_family = workspace_output.get("template_family")
                _require(
                    template_family in workspace_output_registry,
                    f"{output_path}.template_family must be defined in manifest.registries.workspace_output_template_families",
                )
                purpose = workspace_output.get("purpose")
                _require(
                    isinstance(purpose, str) and purpose.strip(),
                    f"{output_path}.purpose must be a non-empty string",
                )
        elif profile_id in required_workspace_output_profile_id_set:
            _require(
                False,
                f"{path}.workspace_outputs must be present for profiles listed in manifest.coverage_policy.required_workspace_output_profile_ids",
            )
        runnable_example = profile.get("runnable_example")
        if runnable_example is not None:
            runnable_ids.add(profile_id)
            _require(isinstance(runnable_example, dict), f"{path}.runnable_example must be an object")
            keys = set(runnable_example.keys())
            kind = runnable_example.get("kind")
            template_family = runnable_example.get("template_family")
            _require(
                template_family in runnable_template_registry,
                f"{path}.runnable_example.template_family must be defined in manifest.registries.runnable_template_families",
            )
            template_entry = runnable_template_registry[template_family]
            _require(
                keys == set(template_entry["required_fields"]),
                f"{path}.runnable_example keys must match the canonical required_fields for template_family={template_family}",
            )
            _require(
                kind in template_entry["allowed_kinds"],
                f"{path}.runnable_example.kind must be allowed for template_family={template_family}",
            )
            payload_key = runnable_example.get("payload_key")
            _require(
                isinstance(payload_key, str) and payload_key.strip(),
                f"{path}.runnable_example.payload_key must be a non-empty string",
            )
            for field in ("local_purpose", "server_purpose"):
                value = runnable_example.get(field)
                _require(
                    isinstance(value, str) and value.strip(),
                    f"{path}.runnable_example.{field} must be a non-empty string",
                )
            if {"prompt_key", "instruction_key", "default_model"}.issubset(keys):
                for field in ("prompt_key", "instruction_key", "default_model"):
                    value = runnable_example.get(field)
                    _require(
                        isinstance(value, str) and value.strip(),
                        f"{path}.runnable_example.{field} must be a non-empty string",
                    )
        transports = profile.get("transports")
        _require(isinstance(transports, dict), f"{path}.transports must be an object")
        _require(set(transports.keys()) == {"local", "server"}, f"{path}.transports must contain local and server")
        for transport in ("local", "server"):
            transport_cfg = transports.get(transport)
            transport_path = f"{path}.transports.{transport}"
            _require(isinstance(transport_cfg, dict), f"{transport_path} must be an object")
            _require(
                set(transport_cfg.keys()) == {"instructions", "snippets", "references"},
                f"{transport_path} has unsupported keys",
            )
            transport_instructions = transport_cfg.get("instructions")
            _require(
                isinstance(transport_instructions, list) and transport_instructions,
                f"{transport_path}.instructions must be a non-empty list",
            )
            _require(
                all(isinstance(item, str) and item.strip() for item in transport_instructions),
                f"{transport_path}.instructions must contain only non-empty strings",
            )
            snippets = transport_cfg.get("snippets")
            _require(isinstance(snippets, dict), f"{transport_path}.snippets must be an object")
            _require(
                set(snippets.keys()) == {"bootstrap", "execution_loop"},
                f"{transport_path}.snippets must contain bootstrap and execution_loop",
            )
            for key in ("bootstrap", "execution_loop"):
                snippet = snippets.get(key)
                _require(
                    isinstance(snippet, str) and snippet.strip(),
                    f"{transport_path}.snippets.{key} must be a non-empty string",
                )
            transport_refs = transport_cfg.get("references")
            _require(
                isinstance(transport_refs, list) and transport_refs,
                f"{transport_path}.references must be a non-empty list",
            )
            for ref in transport_refs:
                _require(
                    isinstance(ref, str) and ref.strip(),
                    f"{transport_path}.references contains an invalid value",
                )
                _require(
                    (repo_root / ref).exists(),
                    f"{transport_path}.references contains missing repo path: {ref}",
                )

    _require(
        seen_ids == required_profile_id_set,
        f"agent bootstrap profiles must exactly cover manifest.coverage_policy.required_profile_ids: {sorted(required_profile_id_set)}",
    )
    _require(
        runnable_ids == required_runnable_profile_id_set,
        "profiles with runnable_example must exactly cover manifest.coverage_policy.required_runnable_profile_ids",
    )


def load_agent_bootstrap_profile_manifest() -> dict[str, Any]:
    manifest = json.loads(_manifest_path().read_text(encoding="utf-8"))
    validate_agent_bootstrap_profile_manifest(manifest)
    return copy.deepcopy(manifest)


def get_runnable_example_profiles(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    validate_agent_bootstrap_profile_manifest(manifest)
    return [
        copy.deepcopy(profile)
        for profile in manifest["profiles"]
        if isinstance(profile.get("runnable_example"), dict)
    ]


def get_workspace_output_profiles(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    validate_agent_bootstrap_profile_manifest(manifest)
    return [
        copy.deepcopy(profile)
        for profile in manifest["profiles"]
        if isinstance(profile.get("workspace_outputs"), list)
    ]


def get_workspace_output_template_registry(manifest: dict[str, Any]) -> dict[str, Any]:
    validate_agent_bootstrap_profile_manifest(manifest)
    return copy.deepcopy(manifest["registries"]["workspace_output_template_families"])


def get_runnable_template_registry(manifest: dict[str, Any]) -> dict[str, Any]:
    validate_agent_bootstrap_profile_manifest(manifest)
    return copy.deepcopy(manifest["registries"]["runnable_template_families"])


def build_agent_bootstrap_profile_manifest_schema(manifest: dict[str, Any]) -> dict[str, Any]:
    validate_agent_bootstrap_profile_manifest(manifest)
    workspace_renderers = sorted(
        {
            entry["renderer"]
            for entry in manifest["registries"]["workspace_output_template_families"].values()
        }
    )
    runnable_renderers = sorted(
        {
            entry["renderer"]
            for entry in manifest["registries"]["runnable_template_families"].values()
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://nextstat.io/schemas/tools/nextstat_agent_bootstrap_profile_manifest_v1.schema.json",
        "title": "NextStat Agent Bootstrap Profile Manifest v1",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "registries", "coverage_policy", "profiles"],
        "properties": {
            "schema_version": {"const": "nextstat.agent_bootstrap_profile_manifest.v1"},
            "registries": {"$ref": "#/$defs/registries"},
            "coverage_policy": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "required_profile_ids",
                    "required_runnable_profile_ids",
                    "workspace_output_owner_profile_ids",
                    "required_workspace_output_profile_ids",
                ],
                "properties": {
                    "required_profile_ids": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string", "pattern": "^[a-z0-9_]+$"},
                    },
                    "required_runnable_profile_ids": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {"type": "string", "pattern": "^[a-z0-9_]+$"},
                    },
                    "workspace_output_owner_profile_ids": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string", "pattern": "^[a-z0-9_]+$"},
                    },
                    "required_workspace_output_profile_ids": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {"type": "string", "pattern": "^[a-z0-9_]+$"},
                    },
                },
            },
            "profiles": {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/$defs/profile"},
            },
        },
        "$defs": {
            "workspace_output_template_family_registry_entry": {
                "type": "object",
                "additionalProperties": False,
                "required": ["renderer"],
                "properties": {
                    "renderer": {
                        "type": "string",
                        "enum": workspace_renderers,
                    }
                },
            },
            "runnable_template_family_registry_entry": {
                "type": "object",
                "additionalProperties": False,
                "required": ["renderer", "allowed_kinds", "required_fields"],
                "properties": {
                    "renderer": {
                        "type": "string",
                        "enum": runnable_renderers,
                    },
                    "allowed_kinds": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string", "pattern": "^[a-z0-9_]+$"},
                    },
                    "required_fields": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string", "pattern": "^[a-z_]+$"},
                    },
                },
            },
            "registries": {
                "type": "object",
                "additionalProperties": False,
                "required": ["workspace_output_template_families", "runnable_template_families"],
                "properties": {
                    "workspace_output_template_families": {
                        "type": "object",
                        "minProperties": 1,
                        "propertyNames": {"pattern": "^[a-z0-9_]+$"},
                        "additionalProperties": {
                            "$ref": "#/$defs/workspace_output_template_family_registry_entry"
                        },
                    },
                    "runnable_template_families": {
                        "type": "object",
                        "minProperties": 1,
                        "propertyNames": {"pattern": "^[a-z0-9_]+$"},
                        "additionalProperties": {
                            "$ref": "#/$defs/runnable_template_family_registry_entry"
                        },
                    },
                },
            },
            "workspace_output": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "path", "template_family", "purpose"],
                "properties": {
                    "id": {"type": "string", "pattern": "^[a-z0-9_]+$"},
                    "path": {"type": "string", "minLength": 1},
                    "template_family": {"type": "string", "pattern": "^[a-z0-9_]+$"},
                    "purpose": {"type": "string", "minLength": 1},
                },
            },
            "transport_config": {
                "type": "object",
                "additionalProperties": False,
                "required": ["instructions", "snippets", "references"],
                "properties": {
                    "instructions": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
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
            },
            "runnable_example": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "kind",
                    "template_family",
                    "local_purpose",
                    "server_purpose",
                    "payload_key",
                ],
                "properties": {
                    "kind": {"type": "string", "pattern": "^[a-z0-9_]+$"},
                    "template_family": {"type": "string", "pattern": "^[a-z0-9_]+$"},
                    "local_purpose": {"type": "string", "minLength": 1},
                    "server_purpose": {"type": "string", "minLength": 1},
                    "payload_key": {"type": "string", "minLength": 1},
                    "prompt_key": {"type": "string", "minLength": 1},
                    "instruction_key": {"type": "string", "minLength": 1},
                    "default_model": {"type": "string", "minLength": 1},
                },
            },
            "profile": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "title",
                    "summary",
                    "references",
                    "instructions",
                    "transports",
                ],
                "properties": {
                    "id": {"type": "string", "pattern": "^[a-z0-9_]+$"},
                    "title": {"type": "string", "minLength": 1},
                    "summary": {"type": "string", "minLength": 1},
                    "references": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "instructions": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "transports": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["local", "server"],
                        "properties": {
                            "local": {"$ref": "#/$defs/transport_config"},
                            "server": {"$ref": "#/$defs/transport_config"},
                        },
                    },
                    "runnable_example": {"$ref": "#/$defs/runnable_example"},
                    "workspace_outputs": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"$ref": "#/$defs/workspace_output"},
                    },
                },
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["json", "paths"], default="json")
    args = parser.parse_args(argv)

    manifest = load_agent_bootstrap_profile_manifest()
    if args.format == "json":
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    print(f"manifest={_manifest_path()}")
    print(f"schema={_schema_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
