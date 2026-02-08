#!/usr/bin/env python3
"""Publish a benchmark snapshot locally (seed script).

Creates an immutable snapshot directory containing:
- suite outputs (e.g. `hep/hep_suite.json` + per-case results)
- a baseline manifest (`baseline_manifest.json`)
- schema validation (fails fast on invalid artifacts)

This is meant to be runnable by outsiders in the standalone benchmarks repo.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def git_short_sha(repo: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo))
        return out.decode().strip()
    except Exception:
        return "unknown"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def validate_json(instance_path: Path, schema_path: Path) -> None:
    import jsonschema  # type: ignore

    schema = load_json(schema_path)
    inst = load_json(instance_path)
    jsonschema.validate(inst, schema)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-id", default="", help="Snapshot id (default: auto-generated).")
    ap.add_argument("--out-root", default="manifests/snapshots", help="Root directory for snapshots.")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--hep", action="store_true", help="Run HEP suite.")
    ap.add_argument("--pharma", action="store_true", help="Run pharma suite.")
    ap.add_argument("--fit", action="store_true", help="Also benchmark MLE fits where supported.")
    ap.add_argument("--fit-repeat", type=int, default=3)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_root = (repo_root / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    snapshot_id = str(args.snapshot_id).strip()
    if not snapshot_id:
        ts = time.strftime("%Y%m%d-%H%M%S")
        snapshot_id = f"snapshot-{ts}-{git_short_sha(repo_root)}"

    snap_dir = out_root / snapshot_id
    if snap_dir.exists():
        raise SystemExit(f"snapshot dir already exists: {snap_dir}")
    snap_dir.mkdir(parents=True, exist_ok=False)

    results: list[Path] = []

    run_any = bool(args.hep or args.pharma)
    run_hep = bool(args.hep or (not run_any))
    run_pharma = bool(args.pharma or (not run_any))

    if run_hep:
        hep_out = snap_dir / "hep"
        cmd = [
            sys.executable,
            "suites/hep/suite.py",
            "--out-dir",
            str(hep_out),
        ]
        if args.deterministic:
            cmd.append("--deterministic")
        if args.fit:
            cmd.extend(["--fit", "--fit-repeat", str(int(args.fit_repeat))])
        subprocess.check_call(cmd, cwd=str(repo_root))
        results.append(hep_out / "hep_suite.json")
        # Generate a small README snippet for humans (does not replace machine JSONs).
        subprocess.check_call(
            [
                sys.executable,
                "suites/hep/report.py",
                "--suite",
                str(hep_out / "hep_suite.json"),
                "--format",
                "markdown",
                "--out",
                str(snap_dir / "README_snippet.md"),
            ],
            cwd=str(repo_root),
        )

    if run_pharma:
        pharma_out = snap_dir / "pharma"
        cmd = [
            sys.executable,
            "suites/pharma/suite.py",
            "--out-dir",
            str(pharma_out),
        ]
        if args.deterministic:
            cmd.append("--deterministic")
        if args.fit:
            cmd.extend(["--fit", "--fit-repeat", str(int(args.fit_repeat))])
        subprocess.check_call(cmd, cwd=str(repo_root))
        results.append(pharma_out / "pharma_suite.json")
        subprocess.check_call(
            [
                sys.executable,
                "suites/pharma/report.py",
                "--suite",
                str(pharma_out / "pharma_suite.json"),
                "--out",
                str(snap_dir / "README_snippet_pharma.md"),
            ],
            cwd=str(repo_root),
        )

    # Validate suite + case schemas.
    schema_case = repo_root / "manifests/schema/benchmark_result_v1.schema.json"
    schema_suite = repo_root / "manifests/schema/benchmark_suite_result_v1.schema.json"
    schema_pharma_case = repo_root / "manifests/schema/pharma_benchmark_result_v1.schema.json"
    schema_pharma_suite = repo_root / "manifests/schema/pharma_benchmark_suite_result_v1.schema.json"
    for r in results:
        obj = load_json(r)
        sv = str(obj.get("schema_version", ""))
        if sv == "nextstat.benchmark_suite_result.v1":
            validate_json(r, schema_suite)
            for e in obj.get("cases", []):
                validate_json((r.parent / e["path"]).resolve(), schema_case)
        elif sv == "nextstat.pharma_benchmark_suite_result.v1":
            validate_json(r, schema_pharma_suite)
            for e in obj.get("cases", []):
                validate_json((r.parent / e["path"]).resolve(), schema_pharma_case)
        elif sv == "nextstat.benchmark_result.v1":
            validate_json(r, schema_case)
        elif sv == "nextstat.pharma_benchmark_result.v1":
            validate_json(r, schema_pharma_case)

    # Write baseline manifest referencing produced results; dataset list will be derived from results.
    baseline_path = snap_dir / "baseline_manifest.json"
    cmd = [
        sys.executable,
        "scripts/write_baseline_manifest.py",
        "--snapshot-id",
        snapshot_id,
        "--out",
        str(baseline_path),
    ]
    if args.deterministic:
        cmd.append("--deterministic")
    for r in results:
        cmd.extend(["--result", str(r)])
    subprocess.check_call(cmd, cwd=str(repo_root))

    validate_json(baseline_path, repo_root / "manifests/schema/baseline_manifest_v1.schema.json")
    print(f"snapshot ready: {snap_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
