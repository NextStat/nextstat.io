#!/usr/bin/env python3
"""Migrate PF3.1 legacy artifact JSONs to schema'd v1 forms.

PF3.1 remote runners originally emitted:
- `*.meta.json` objects without `schema_version`
- `summary.json` as a bare list (no `schema_version`)

This tool upgrades those files in-place so strict JSON schema validation can be
applied retroactively to older runs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def migrate_meta(path: Path) -> bool:
    obj = load_json(path)
    if not isinstance(obj, dict):
        return False
    if isinstance(obj.get("schema_version"), str):
        return False

    # Heuristic: legacy meta always had these fields.
    required = {"name", "spec", "n_toys", "rc", "elapsed_s", "args", "ts_unix"}
    if not required.issubset(set(obj.keys())):
        return False

    obj = dict(obj)
    obj["schema_version"] = "nextstat.pf31_run_meta.v1"
    obj.setdefault("generated_at", utc_now_iso())
    write_json(path, obj)
    return True


def migrate_case_summary(path: Path) -> bool:
    obj = load_json(path)
    if isinstance(obj, dict) and isinstance(obj.get("schema_version"), str):
        return False

    # Legacy case summary was a list of meta-like dicts.
    if not isinstance(obj, list):
        return False

    out = {
        "schema_version": "nextstat.pf31_case_summary.v1",
        "generated_at": utc_now_iso(),
        "run_root": str(path.parent),
        "rows": obj,
    }
    write_json(path, out)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Migrate PF3.1 legacy artifacts to schema'd v1")
    ap.add_argument("--root", required=True, help="PF3.1 publication run root (contains case dirs)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"root not found or not a directory: {root}")

    n_meta = 0
    n_summary = 0
    for p in sorted(root.rglob("*.meta.json")):
        if migrate_meta(p):
            n_meta += 1

    for p in sorted(root.rglob("summary.json")):
        if migrate_case_summary(p):
            n_summary += 1

    print(f"[pf31-migrate] root={root}")
    print(f"[pf31-migrate] meta_upgraded={n_meta}")
    print(f"[pf31-migrate] case_summaries_upgraded={n_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

