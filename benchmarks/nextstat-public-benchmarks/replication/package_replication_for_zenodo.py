#!/usr/bin/env python3
"""Package a replication bundle for Zenodo (seed tool).

Creates:
- `<bundle_name>.tar.gz` archive
- `<bundle_name>.tar.gz.sha256`
- `zenodo_deposition.json` derived from `replication/zenodo_replication.json` + inputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--published-doi", required=True)
    ap.add_argument("--published-url", required=True)
    ap.add_argument("--template", default="benchmarks/nextstat-public-benchmarks/replication/zenodo_replication.json")
    args = ap.parse_args()

    bundle_dir = Path(args.bundle_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    name = bundle_dir.name
    archive_path = out_dir / f"{name}.tar.gz"
    if archive_path.exists():
        raise SystemExit(f"archive already exists: {archive_path}")

    with tarfile.open(archive_path, "w:gz") as tf:
        tf.add(bundle_dir, arcname=name)

    digest = sha256_file(archive_path)
    (out_dir / f"{name}.tar.gz.sha256").write_text(f"{digest}  {name}.tar.gz\n")

    tmpl_path = Path(args.template).resolve()
    meta = load_json(tmpl_path) if tmpl_path.exists() else {}
    meta["title"] = f"NextStat Replication Bundle: {name}"
    desc = meta.get("description", "")
    meta["description"] = (
        f"{desc}\n\n"
        f"Published snapshot DOI: {args.published_doi}\n"
        f"Published snapshot URL: {args.published_url}\n"
        f"Replication bundle sha256: {digest}\n"
    )

    (out_dir / "zenodo_deposition.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(str(archive_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

