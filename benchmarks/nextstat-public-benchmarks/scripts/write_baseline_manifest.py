#!/usr/bin/env python3
"""Write a minimal baseline manifest for a benchmark snapshot.

This is a seed script; in the standalone benchmarks repo, this is the canonical
machine-readable snapshot descriptor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo: Path) -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo))
        return out.decode().strip()
    except Exception:
        return "unknown"

def try_nextstat_version() -> str:
    try:
        import nextstat  # type: ignore

        v = getattr(nextstat, "__version__", None)
        return str(v) if v else "unknown"
    except Exception:
        return "unknown"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def collect_datasets_from_result(result_path: Path) -> list[dict[str, str]]:
    """Collect dataset ids + hashes from benchmark result JSONs.

    Supports:
    - benchmark_result_v1 (single-case): uses `dataset`
    - benchmark_suite_result_v1 (suite index): opens each case JSON and uses its `dataset`
    - pharma_benchmark_result_v1 (single-case): uses `dataset`
    - pharma_benchmark_suite_result_v1 (suite index): opens each case JSON and uses its `dataset`
    """
    obj = load_json(result_path)
    sv = str(obj.get("schema_version", ""))

    out: list[dict[str, str]] = []
    if sv in ("nextstat.benchmark_result.v1", "nextstat.pharma_benchmark_result.v1"):
        ds = obj.get("dataset") or {}
        ds_id = ds.get("id")
        ds_sha = ds.get("sha256")
        if isinstance(ds_id, str) and isinstance(ds_sha, str):
            out.append({"id": ds_id, "sha256": ds_sha})
        return out

    if sv in ("nextstat.benchmark_suite_result.v1", "nextstat.pharma_benchmark_suite_result.v1"):
        base = result_path.parent
        for e in obj.get("cases", []):
            try:
                rel = e["path"]
                case_obj = load_json((base / rel).resolve())
                ds = case_obj.get("dataset") or {}
                ds_id = ds.get("id")
                ds_sha = ds.get("sha256")
                if isinstance(ds_id, str) and isinstance(ds_sha, str):
                    out.append({"id": ds_id, "sha256": ds_sha})
            except Exception:
                # Ignore malformed entries; schema validation should catch this elsewhere.
                pass
        return out

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-id", required=True)
    ap.add_argument("--out", required=True, help="Output manifest JSON path.")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--harness-repo", default="nextstat-public-benchmarks")
    ap.add_argument("--result", action="append", default=[], help="Suite result JSON path (repeatable).")
    ap.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Optional dataset file path to hash (repeatable).",
    )
    ap.add_argument("--nextstat-wheel", default="", help="Optional: path to measured NextStat wheel for hashing.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for p in args.result:
        pp = Path(p).resolve()
        suite = pp.stem
        try:
            suite = str(load_json(pp).get("suite") or suite)
        except Exception:
            pass
        results.append({"suite": suite, "path": os.path.relpath(pp, out_path.parent), "sha256": sha256_file(pp)})

    datasets: list[dict[str, Any]] = []
    for r in args.result:
        datasets.extend(collect_datasets_from_result(Path(r).resolve()))
    for p in args.dataset:
        pp = Path(p).resolve()
        if pp.exists():
            datasets.append({"id": os.path.relpath(pp, repo_root), "sha256": sha256_file(pp)})

    # Deduplicate by (id, sha256) while preserving order.
    seen = set()
    datasets2: list[dict[str, Any]] = []
    for d in datasets:
        key = (d.get("id"), d.get("sha256"))
        if key in seen:
            continue
        if isinstance(key[0], str) and isinstance(key[1], str):
            seen.add(key)
            datasets2.append(d)

    wheel_sha = ""
    if args.nextstat_wheel:
        wp = Path(args.nextstat_wheel).resolve()
        if wp.exists():
            wheel_sha = sha256_file(wp)

    doc: dict[str, Any] = {
        "schema_version": "nextstat.baseline_manifest.v1",
        "snapshot_id": args.snapshot_id,
        "deterministic": bool(args.deterministic),
        "harness": {"repo": args.harness_repo, "git_commit": git_commit(repo_root)},
        "nextstat": {"version": try_nextstat_version(), **({"wheel_sha256": wheel_sha} if wheel_sha else {})},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "datasets": datasets2,
        "results": results,
    }

    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
