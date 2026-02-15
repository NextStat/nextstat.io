#!/usr/bin/env python3
"""Collect scheduler shard outputs into the standard manifest contract.

Reads `manifest.json` produced by `render_unbinned_fit_toys_scheduler.py`,
scans for shard outputs in the run directory, and writes back an updated
manifest with `results[]` entries compatible with merge tooling.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect scheduler shard outputs into manifest.json results[].")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--in-place", action="store_true", help="overwrite the input manifest")
    ap.add_argument("--out", type=Path, default=None, help="optional output manifest path")
    args = ap.parse_args()

    manifest = load_json(args.manifest)
    run_dir = Path(manifest.get("run_dir") or args.manifest.parent)
    shards = int(manifest.get("shards") or 0)
    if shards <= 0:
        raise SystemExit("manifest missing shards")

    results: list[dict[str, Any]] = []
    for i in range(shards):
        out_json = run_dir / f"shard_{i}.out.json"
        metrics_json = run_dir / f"shard_{i}.metrics.json"
        stdout = run_dir / f"shard_{i}.stdout"
        stderr = run_dir / f"shard_{i}.stderr"
        host_file = run_dir / f"shard_{i}.host"
        rc_file = run_dir / f"shard_{i}.rc"
        elapsed_file = run_dir / f"shard_{i}.elapsed_s"

        host = host_file.read_text(encoding="utf-8").strip() if host_file.exists() else "unknown"
        rc = None
        if rc_file.exists():
            try:
                rc = int(rc_file.read_text(encoding="utf-8").strip())
            except Exception:
                rc = 255
        elapsed_s = 0.0
        if elapsed_file.exists():
            try:
                elapsed_s = float(elapsed_file.read_text(encoding="utf-8").strip())
            except Exception:
                elapsed_s = 0.0

        # Scheduler wrappers are expected to write `.rc`, but in case only the
        # output JSON is present (manual runs, partial collections), accept it.
        ok = out_json.exists() and (rc is None or rc == 0)
        status = "ok" if ok else "failed"

        # This matches the fields produced by the SSH launcher (a subset is used by the merge tool).
        results.append(
            {
                "schema_version": "nextstat.cpu_farm_shard_result.v1",
                "shard_index": i,
                "host": host,
                "target": host,
                "status": status,
                "n_toys": None,
                "toy_start": None,
                "seed": None,
                "threads": None,
                "weight": 1.0,
                "rc": rc,
                "elapsed_s": elapsed_s,
                "local_dir": str(run_dir),
                "local_output": str(out_json) if out_json.exists() else None,
                "local_metrics": str(metrics_json) if metrics_json.exists() else None,
                "local_stdout": str(stdout) if stdout.exists() else None,
                "local_stderr": str(stderr) if stderr.exists() else None,
                "remote_dir": None,
                "error": None if ok else "missing output or non-zero rc",
            }
        )

    # Backfill from plans where possible.
    plan_by_idx: dict[int, dict[str, Any]] = {}
    for p in manifest.get("plans") or []:
        if isinstance(p, dict) and isinstance(p.get("shard_index"), int):
            plan_by_idx[int(p["shard_index"])] = p
    for r in results:
        idx = int(r["shard_index"])
        p = plan_by_idx.get(idx)
        if p:
            for k in ["n_toys", "toy_start", "seed", "threads", "weight"]:
                if r.get(k) is None and k in p:
                    r[k] = p[k]

    manifest["collected_at_utc"] = dt.datetime.now(tz=dt.timezone.utc).isoformat()
    manifest["results"] = results

    out_path = args.manifest if args.in_place else (args.out or (args.manifest.parent / "manifest.collected.json"))
    write_json(out_path, manifest)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
