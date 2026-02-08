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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-id", required=True)
    ap.add_argument("--out", required=True, help="Output manifest JSON path.")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--harness-repo", default="nextstat-public-benchmarks")
    ap.add_argument("--result", action="append", default=[], help="Suite result JSON path (repeatable).")
    ap.add_argument("--dataset", action="append", default=[], help="Dataset file path (repeatable).")
    ap.add_argument("--nextstat-wheel", default="", help="Optional: path to measured NextStat wheel for hashing.")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for p in args.result:
        pp = Path(p).resolve()
        results.append({"suite": pp.stem, "path": os.path.relpath(pp, out_path.parent), "sha256": sha256_file(pp)})

    datasets: list[dict[str, Any]] = []
    for p in args.dataset:
        pp = Path(p).resolve()
        datasets.append({"id": os.path.relpath(pp, repo_root), "sha256": sha256_file(pp)})

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
        "nextstat": {"version": "unknown", **({"wheel_sha256": wheel_sha} if wheel_sha else {})},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "datasets": datasets,
        "results": results,
    }

    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

