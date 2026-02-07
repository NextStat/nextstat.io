"""Standalone CLI for workspace preprocessing.

Usage:
  python -m nextstat.analysis.preprocess.cli \\
    --input workspace.json \\
    --output workspace_preprocessed.json \\
    [--config preprocessing_config.json] \\
    [--provenance provenance.json]

When invoked without --config, the default TREx-standard pipeline is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import preprocess_workspace


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="nextstat-preprocess",
        description="Apply systematics preprocessing to a pyhf workspace JSON.",
    )
    ap.add_argument("--input", type=Path, required=True, help="Input workspace JSON path.")
    ap.add_argument("--output", type=Path, required=True, help="Output workspace JSON path.")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Preprocessing config JSON (the execution.preprocessing section). "
        "If omitted, default TREx pipeline is used.",
    )
    ap.add_argument(
        "--provenance",
        type=Path,
        default=None,
        help="Optional path to write preprocessing provenance JSON.",
    )
    args = ap.parse_args(argv)

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1

    ws: dict[str, Any] = json.loads(args.input.read_text())

    config: dict[str, Any] | None = None
    if args.config is not None:
        if not args.config.exists():
            print(f"Error: config file not found: {args.config}", file=sys.stderr)
            return 1
        config = json.loads(args.config.read_text())

    result = preprocess_workspace(ws, config=config, in_place=True)

    # Write output
    out_parent = args.output.parent
    if str(out_parent) and not out_parent.exists():
        out_parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.workspace, separators=(",", ":"), sort_keys=True))

    # Write provenance
    if args.provenance is not None:
        prov_parent = args.provenance.parent
        if str(prov_parent) and not prov_parent.exists():
            prov_parent.mkdir(parents=True, exist_ok=True)
        args.provenance.write_text(json.dumps(result.provenance_dict(), indent=2, sort_keys=True))

    n_changed = sum(1 for s in result.provenance.steps for r in s.records if r.changed)
    n_total = sum(len(s.records) for s in result.provenance.steps)
    print(json.dumps({
        "steps": len(result.provenance.steps),
        "records_total": n_total,
        "records_changed": n_changed,
        "input_sha256": result.provenance.input_sha256[:16],
        "output_sha256": result.provenance.output_sha256[:16],
    }))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
