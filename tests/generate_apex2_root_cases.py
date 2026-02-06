#!/usr/bin/env python3
"""Generate an Apex2 ROOT parity cases file by scanning a directory.

This is a convenience helper for TRExFitter / HistFactory exports.
It finds `combination.xml` files (or a user-provided glob), and writes a JSON
cases file consumable by `tests/apex2_root_suite_report.py`.

Run:
  ./.venv/bin/python tests/generate_apex2_root_cases.py \\
    --search-dir /abs/path/to/trex/output \\
    --out tmp/root_cases.json

Then:
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_root_suite_report.py \\
    --cases tmp/root_cases.json --keep-going
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _rel_or_abs(path: Path, *, base: Path, absolute: bool) -> str:
    if absolute:
        return str(path.resolve())
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path.resolve())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-dir", type=Path, required=True)
    ap.add_argument(
        "--glob",
        type=str,
        default="**/combination.xml",
        help="Glob relative to --search-dir.",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute paths in cases JSON (default: prefer relative where possible).",
    )
    ap.add_argument(
        "--include-fixtures",
        action="store_true",
        help="Also include built-in fixture case(s) for smoke.",
    )
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--stop", type=float, default=5.0)
    ap.add_argument("--points", type=int, default=51)
    args = ap.parse_args()

    search_dir = args.search_dir.resolve()
    if not search_dir.is_dir():
        raise SystemExit(f"--search-dir is not a directory: {search_dir}")

    combo_paths = sorted(search_dir.glob(args.glob))
    cases: List[Dict[str, Any]] = []

    if args.include_fixtures:
        cases.append(
            {
                "name": "simple_fixture",
                "mode": "pyhf-json",
                "pyhf_json": "tests/fixtures/simple_workspace.json",
                "measurement": "GaussExample",
                "mu_grid": {"start": args.start, "stop": args.stop, "points": args.points},
            }
        )

    for p in combo_paths:
        if not p.is_file():
            continue
        name = f"{p.parent.name}"
        cases.append(
            {
                "name": name,
                "mode": "histfactory-xml",
                "histfactory_xml": _rel_or_abs(p, base=Path.cwd(), absolute=args.absolute_paths),
                "rootdir": _rel_or_abs(p.parent, base=Path.cwd(), absolute=args.absolute_paths),
                "mu_grid": {"start": args.start, "stop": args.stop, "points": args.points},
            }
        )

    out_obj = {"cases": cases}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_obj, indent=2))

    print(f"Found {len(combo_paths)} combination.xml file(s)")
    print(f"Wrote {len(cases)} case(s) to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

