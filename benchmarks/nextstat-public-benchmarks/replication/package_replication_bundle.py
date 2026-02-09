#!/usr/bin/env python3
"""Package a replication bundle directory as tar.gz + sha256 (seed tool)."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--out-dir", default="", help="Output directory (default: bundle dir parent).")
    args = ap.parse_args()

    bundle_dir = Path(args.bundle_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else bundle_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    name = bundle_dir.name
    out_path = out_dir / f"{name}.tar.gz"
    if out_path.exists():
        raise SystemExit(f"already exists: {out_path}")

    with tarfile.open(out_path, "w:gz") as tf:
        tf.add(bundle_dir, arcname=name)

    digest = sha256_file(out_path)
    (out_dir / f"{name}.tar.gz.sha256").write_text(f"{digest}  {name}.tar.gz\n")
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

