"""Validate the canonical tool manifest against schema and semantic invariants."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _manifest_path() -> Path:
    return _repo_root() / "bindings" / "ns-py" / "python" / "nextstat" / "_tool_manifest_v1.json"


def _schema_path() -> Path:
    return _repo_root() / "docs" / "schemas" / "tools" / "nextstat_tool_manifest_v1.schema.json"


def _load_manifest_module():
    module_path = _repo_root() / "bindings" / "ns-py" / "python" / "nextstat" / "_tool_manifest.py"
    spec = importlib.util.spec_from_file_location("nextstat._tool_manifest", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load manifest helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    manifest = json.loads(_manifest_path().read_text(encoding="utf-8"))
    schema = json.loads(_schema_path().read_text(encoding="utf-8"))

    try:
        import jsonschema  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("jsonschema is required to validate the tool manifest") from exc

    jsonschema.validate(instance=manifest, schema=schema)

    manifest_module = _load_manifest_module()
    manifest_module.validate_tool_manifest(manifest)

    print(f"Validated {_manifest_path()} against {_schema_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
