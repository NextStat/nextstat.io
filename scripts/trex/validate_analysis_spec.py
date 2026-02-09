#!/usr/bin/env python3
"""Validate a TREx analysis spec YAML against its JSON Schema.

Goals:
- IDE-friendly schema-driven config (autocomplete + validation).
- Clear CLI errors with JSON-pointer-like paths.

Usage:
  ./.venv/bin/python scripts/trex/validate_analysis_spec.py \
    --spec docs/specs/trex/analysis_spec_v0.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text())


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _fmt_path(parts: Iterable[Any]) -> str:
    out = "$"
    for p in parts:
        if isinstance(p, int):
            out += f"[{p}]"
        else:
            out += f".{p}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True, help="Path to analysis spec YAML.")
    ap.add_argument(
        "--schema",
        type=Path,
        default=_repo_root() / "docs" / "schemas" / "trex" / "analysis_spec_v0.schema.json",
        help="Path to JSON Schema (default: repo schema).",
    )
    args = ap.parse_args()

    if not args.spec.exists():
        raise SystemExit(f"Missing spec: {args.spec}")
    if not args.schema.exists():
        raise SystemExit(f"Missing schema: {args.schema}")

    spec = _load_yaml(args.spec)
    schema = _load_json(args.schema)

    try:
        import jsonschema  # type: ignore
    except Exception as e:
        raise SystemExit(f"Missing dependency jsonschema: {e}")

    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(spec), key=lambda e: list(e.path))
    if not errors:
        print("OK")
        return 0

    print(f"FAIL ({len(errors)} error(s))")
    for e in errors[:50]:
        loc = _fmt_path(e.path)
        msg = e.message
        print(f"- {loc}: {msg}")
    if len(errors) > 50:
        print(f"... ({len(errors) - 50} more)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

