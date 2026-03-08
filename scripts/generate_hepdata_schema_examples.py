"""Generate canonical HEPData schema example fixtures.

Usage:
  python scripts/generate_hepdata_schema_examples.py
  python scripts/generate_hepdata_schema_examples.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hepdata_example_helpers import generate_examples


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
