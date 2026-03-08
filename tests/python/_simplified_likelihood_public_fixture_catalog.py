from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def catalog_schema_path() -> Path:
    return (
        repo_root()
        / "docs"
        / "schemas"
        / "apex2"
        / "simplified_likelihood_public_fixture_catalog_v0.schema.json"
    )


def catalog_example_path() -> Path:
    return repo_root() / "docs" / "specs" / "apex2_simplified_likelihood_public_fixture_catalog_v0.example.json"


def simplified_workspace_schema_path() -> Path:
    return repo_root() / "docs" / "schemas" / "hep" / "simplified_likelihood_v0.schema.json"


def load_catalog() -> dict[str, Any]:
    return json.loads(catalog_example_path().read_text(encoding="utf-8"))


def resolve_workspace_path(fixture: dict[str, Any]) -> Path:
    return repo_root() / fixture["workspace_json_path"]
