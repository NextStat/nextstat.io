#!/usr/bin/env python3
"""Validate unbinned benchmark artifacts against local JSON schemas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCHEMA_MAP = {
    "nextstat.unbinned_run_suite_result.v1": "schemas/unbinned_run_suite_result_v1.schema.json",
    "nextstat.unbinned_cpu_symmetry_bench.v1": "schemas/unbinned_cpu_symmetry_bench_v1.schema.json",
    "nextstat.pf31_publication_matrix.v1": "schemas/pf31_publication_matrix_v1.schema.json",
    "nextstat.pf31_run_meta.v1": "schemas/pf31_run_meta_v1.schema.json",
    "nextstat.pf31_case_summary.v1": "schemas/pf31_case_summary_v1.schema.json",
    "nextstat.pf31_publication_summary.v1": "schemas/pf31_publication_summary_v1.schema.json",
    "nextstat.pf34_metal_matrix.v1": "schemas/pf34_metal_matrix_v1.schema.json",
    "nextstat.pf34_metal_preflight.v1": "schemas/pf34_metal_preflight_v1.schema.json",
    "nextstat.pf34_metal_run_manifest.v1": "schemas/pf34_metal_run_manifest_v1.schema.json",
    "nextstat.pf34_metal_run_meta.v1": "schemas/pf34_metal_run_meta_v1.schema.json",
    "nextstat.pf34_metal_summary.v1": "schemas/pf34_metal_summary_v1.schema.json",
    "nextstat_unbinned_spec_v0": "schemas/unbinned_spec_v0.schema.json",
    "nextstat_metrics_v0": "schemas/nextstat_metrics_v0.schema.json",
    "nextstat.snapshot_index.v1": "schemas/snapshot_index_v1.schema.json",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _schema_path(root: Path, schema_version: str) -> Path | None:
    rel = SCHEMA_MAP.get(schema_version)
    return (root / rel).resolve() if rel else None


def _validate_instance(instance_path: Path, schema_path: Path) -> None:
    import jsonschema  # type: ignore

    schema = _load_json(schema_path)
    inst = _load_json(instance_path)
    jsonschema.validate(inst, schema)


def validate_path(path: Path, strict: bool) -> int:
    root = Path(__file__).resolve().parent

    try:
        obj = _load_json(path)
    except Exception as e:
        print(f"invalid json: {path}: {type(e).__name__}:{e}", file=sys.stderr)
        return 1

    if not isinstance(obj, dict):
        if strict:
            print(f"unsupported json root (expected object): {path}", file=sys.stderr)
            return 1
        return 0

    sv = obj.get("schema_version")
    if not isinstance(sv, str) or not sv.strip():
        if strict:
            print(f"missing schema_version: {path}", file=sys.stderr)
            return 1
        return 0

    schema_path = _schema_path(root, sv)
    if schema_path is None:
        if strict:
            print(f"unknown schema_version={sv}: {path}", file=sys.stderr)
            return 1
        return 0

    try:
        _validate_instance(path, schema_path)
    except Exception as e:
        print(f"schema fail: {path}: {type(e).__name__}:{e}", file=sys.stderr)
        return 1

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate unbinned benchmark artifacts (JSON schemas).")
    ap.add_argument("paths", nargs="+", help="JSON files and/or directories.")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    targets: list[Path] = []
    for p in args.paths:
        pp = Path(p).resolve()
        if pp.is_dir():
            # `*.out.json` files are stdout captures (tool output) and are not currently
            # part of the schema-validated artifact contract. Keep them for debugging,
            # but do not fail strict validation on them.
            targets.extend(sorted(x for x in pp.rglob("*.json") if not x.name.endswith(".out.json")))
        else:
            targets.append(pp)

    rc = 0
    for t in targets:
        rc |= validate_path(t, strict=bool(args.strict))
    return 0 if rc == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
