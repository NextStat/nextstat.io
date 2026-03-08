"""Generate discovery descriptor example fixtures from the canonical manifest.

Usage:
  python scripts/generate_tool_schema_examples.py
  python scripts/generate_tool_schema_examples.py --check
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_manifest_helper():
    module_path = _repo_root() / "bindings" / "ns-py" / "python" / "nextstat" / "_tool_manifest.py"
    spec = importlib.util.spec_from_file_location("nextstat._tool_manifest", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load manifest helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _example_paths() -> dict[str, Path]:
    specs = _repo_root() / "docs" / "specs"
    return {
        "local": specs / "nextstat_tool_schema_local_v1.example.json",
        "server": specs / "nextstat_tool_schema_server_v1.example.json",
    }


def generate_examples() -> dict[Path, str]:
    helper = _load_manifest_helper()
    return {
        path: json.dumps(helper.build_toolkit_descriptor(transport), indent=2, sort_keys=True) + "\n"
        for transport, path in _example_paths().items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated output differs")
    args = parser.parse_args(argv)

    generated = generate_examples()
    dirty: list[Path] = []

    for path, content in generated.items():
        current = path.read_text(encoding="utf-8") if path.exists() else None
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
