"""Write a machine-readable registry from one or more snapshot indices.

The registry is meant for discovery-facing surfaces that need a compact view of
published snapshots without reparsing each snapshot directory on every read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_suite_health(doc: dict[str, Any]) -> list[dict[str, Any]]:
    rows = doc.get("suite_health")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _iter_snapshot_indices(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for raw in args.snapshot_index or []:
        p = Path(raw).resolve()
        if not p.exists():
            raise SystemExit(f"snapshot_index not found: {p}")
        paths.append(p)
    for raw in args.snapshots_root or []:
        root = Path(raw).resolve()
        if not root.exists():
            raise SystemExit(f"snapshots root not found: {root}")
        if root.is_file():
            if root.name != "snapshot_index.json":
                raise SystemExit(f"expected snapshot_index.json file, got: {root}")
            paths.append(root)
            continue
        paths.extend(sorted(root.rglob("snapshot_index.json")))
    # Stable unique order.
    dedup: dict[str, Path] = {}
    for path in paths:
        dedup[str(path)] = path
    return [dedup[key] for key in sorted(dedup.keys())]


def _registry_generated_at(entries: list[dict[str, Any]]) -> str:
    for row in entries:
        value = str(row.get("generated_at") or "").strip()
        if value:
            return value
    return "1970-01-01T00:00:00Z"


def _snapshot_index_path(path: Path, *, out_parent: Path) -> str:
    try:
        return os.path.relpath(path, out_parent)
    except Exception:
        return str(path)


def _build_registry(indices: list[Path], *, out_parent: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for path in indices:
        doc = _load_json(path)
        if doc.get("schema_version") != "nextstat.snapshot_index.v1":
            raise SystemExit(f"unexpected snapshot_index schema_version in {path}")
        artifact_rows = doc.get("artifacts") if isinstance(doc.get("artifacts"), list) else []
        entries.append(
            {
                "snapshot_id": str(doc.get("snapshot_id") or ""),
                "suite": str(doc.get("suite") or ""),
                "generated_at": str(doc.get("generated_at") or ""),
                "snapshot_index_path": _snapshot_index_path(path, out_parent=out_parent),
                "snapshot_index_sha256": _sha256_bytes(path.read_bytes()),
                "artifact_count": len([row for row in artifact_rows if isinstance(row, dict)]),
                "git": doc.get("git") if isinstance(doc.get("git"), dict) else {},
                "workflow": doc.get("workflow") if isinstance(doc.get("workflow"), dict) else {},
                "suite_health": _normalize_suite_health(doc),
            }
        )

    entries.sort(
        key=lambda row: (
            str(row.get("generated_at") or ""),
            str(row.get("snapshot_id") or ""),
            str(row.get("snapshot_index_path") or ""),
        ),
        reverse=True,
    )

    return {
        "schema_version": "nextstat.snapshot_registry.v1",
        "generated_at": _registry_generated_at(entries),
        "entry_count": len(entries),
        "entries": entries,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--snapshot-index",
        action="append",
        default=[],
        help="Path to a snapshot_index.json (repeatable).",
    )
    ap.add_argument(
        "--snapshots-root",
        action="append",
        default=[],
        help="Directory to scan recursively for snapshot_index.json (repeatable).",
    )
    ap.add_argument("--out", required=True, help="Output registry JSON path.")
    ap.add_argument("--check", action="store_true", help="Fail if generated registry differs from --out.")
    args = ap.parse_args()

    indices = _iter_snapshot_indices(args)
    if not indices:
        raise SystemExit("no snapshot indices provided")

    out_path = Path(args.out).resolve()
    out_doc = _build_registry(indices, out_parent=out_path.parent)
    content = json.dumps(out_doc, indent=2, sort_keys=True) + "\n"
    if args.check:
        current = out_path.read_text(encoding="utf-8") if out_path.exists() else None
        if current != content:
            print(f"out of date: {out_path}", file=sys.stderr)
            return 1
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
