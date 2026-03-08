"""Shared bootstrap artifact path builders.

Generated/bootstrap tooling and tests should use this module as the single
source of truth for canonical bootstrap artifact paths.
"""

from __future__ import annotations

from pathlib import Path


def bootstrap_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def bootstrap_pack_relative_path(client: str, transport: str) -> str:
    return f"docs/specs/agent_bootstrap/nextstat_{client}_{transport}_bootstrap_v1.json"


def bootstrap_provider_example_relative_path(client: str, transport: str) -> str:
    return f"docs/specs/agent_bootstrap/examples/nextstat_{client}_{transport}_example.py"


def bootstrap_reference_doc_relative_path() -> str:
    return "docs/references/agent-bootstrap.md"


def bootstrap_pack_output_path(client: str, transport: str) -> Path:
    return bootstrap_repo_root() / Path(bootstrap_pack_relative_path(client, transport))


def bootstrap_provider_example_output_path(client: str, transport: str) -> Path:
    return bootstrap_repo_root() / Path(bootstrap_provider_example_relative_path(client, transport))


def bootstrap_reference_doc_output_path() -> Path:
    return bootstrap_repo_root() / Path(bootstrap_reference_doc_relative_path())
