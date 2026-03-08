#!/usr/bin/env python3
"""Compatibility wrapper for `nextstat import hepdata`.

This preserves the historical script entrypoint used by older docs and opt-in
HEPData parity tooling, while delegating the real implementation to the product
CLI command:

  nextstat import hepdata

Supported legacy flags map as follows:
- `--manifest` -> `--manifest`
- `--out` -> `--out-dir`
- `--cache` -> `--cache-dir`
- `--lock` -> `--lock`
- `--dataset` -> `--dataset` (repeatable)
- `--clean` -> `--clean`
- `--offline` -> `--offline`
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = ROOT / "hepdata" / "manifest.json"
DEFAULT_CACHE_DIR = ROOT / "hepdata" / "_cache"
DEFAULT_OUT_DIR = ROOT / "hepdata" / "workspaces"
DEFAULT_LOCK_PATH = ROOT / "hepdata" / "workspaces.lock.json"


def _nextstat_cli_prefix() -> List[str]:
    override = os.environ.get("NEXTSTAT_HEPDATA_CMD")
    if override:
        return override.split()
    return ["cargo", "run", "-p", "ns-cli", "--", "import", "hepdata"]


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--lock", default=str(DEFAULT_LOCK_PATH))
    ap.add_argument("--dataset", action="append", default=[], help="Dataset id to fetch (repeatable)")
    ap.add_argument("--clean", action="store_true", help="Delete existing cache/output before fetching")
    ap.add_argument("--offline", action="store_true", help="Require cached archives and skip network download")
    args = ap.parse_args(argv)

    cmd = _nextstat_cli_prefix()
    cmd += ["--manifest", str(Path(args.manifest))]
    cmd += ["--out-dir", str(Path(args.out))]
    cmd += ["--cache-dir", str(Path(args.cache))]
    cmd += ["--lock", str(Path(args.lock))]
    if args.clean:
        cmd.append("--clean")
    if args.offline:
        cmd.append("--offline")
    for dataset in args.dataset:
        cmd += ["--dataset", dataset]

    if shutil.which(cmd[0]) is None:
        print(f"ERROR: required command not found: {cmd[0]}", file=sys.stderr)
        return 127

    completed = subprocess.run(cmd, cwd=str(ROOT.parent))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
