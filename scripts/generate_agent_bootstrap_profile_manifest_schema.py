#!/usr/bin/env python3
"""Generate the canonical NextStat agent bootstrap profile manifest schema.

Usage:
  python scripts/generate_agent_bootstrap_profile_manifest_schema.py
  python scripts/generate_agent_bootstrap_profile_manifest_schema.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_bootstrap_profile_manifest import (  # noqa: E402
    build_agent_bootstrap_profile_manifest_schema,
    load_agent_bootstrap_profile_manifest,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _schema_path() -> Path:
    return (
        _repo_root()
        / "docs"
        / "schemas"
        / "tools"
        / "nextstat_agent_bootstrap_profile_manifest_v1.schema.json"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated output differs")
    args = parser.parse_args(argv)

    manifest = load_agent_bootstrap_profile_manifest()
    schema = build_agent_bootstrap_profile_manifest_schema(manifest)
    content = json.dumps(schema, indent=2, sort_keys=False) + "\n"
    path = _schema_path()
    current = path.read_text(encoding="utf-8") if path.exists() else None

    if args.check:
        if current != content:
            print(f"out of date: {path}", file=sys.stderr)
            return 1
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
