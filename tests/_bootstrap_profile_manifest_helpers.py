from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts import agent_bootstrap_profile_manifest as _profile_manifest
from scripts.bootstrap_artifact_paths import (
    bootstrap_pack_output_path,
    bootstrap_provider_example_output_path,
    bootstrap_reference_doc_output_path,
)
from scripts.repo_module_loader import load_repo_module


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_agent_bootstrap_profile_manifest_helper():
    return _profile_manifest


def load_agent_bootstrap_profile_manifest() -> dict[str, Any]:
    helper = load_agent_bootstrap_profile_manifest_helper()
    return helper.load_agent_bootstrap_profile_manifest()


def get_runnable_example_profiles() -> list[dict[str, Any]]:
    helper = load_agent_bootstrap_profile_manifest_helper()
    manifest = helper.load_agent_bootstrap_profile_manifest()
    return helper.get_runnable_example_profiles(manifest)


def get_workspace_output_profiles() -> list[dict[str, Any]]:
    helper = load_agent_bootstrap_profile_manifest_helper()
    manifest = helper.load_agent_bootstrap_profile_manifest()
    return helper.get_workspace_output_profiles(manifest)


def get_workspace_output_template_registry() -> dict[str, Any]:
    helper = load_agent_bootstrap_profile_manifest_helper()
    manifest = helper.load_agent_bootstrap_profile_manifest()
    return helper.get_workspace_output_template_registry(manifest)


def get_runnable_template_registry() -> dict[str, Any]:
    helper = load_agent_bootstrap_profile_manifest_helper()
    manifest = helper.load_agent_bootstrap_profile_manifest()
    return helper.get_runnable_template_registry(manifest)


def bootstrap_reference_doc_path() -> Path:
    return bootstrap_reference_doc_output_path()


def bootstrap_pack_path(client: str, transport: str) -> Path:
    return bootstrap_pack_output_path(client, transport)


def bootstrap_provider_example_path(client: str, transport: str) -> Path:
    return bootstrap_provider_example_output_path(client, transport)
