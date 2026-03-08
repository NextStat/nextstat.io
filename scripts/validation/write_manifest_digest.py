"""Write SHA-256 digest sidecars for a manifest JSON.

This is intentionally stdlib-only so workflows can reuse it without
bootstrapping extra dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def _default_out_path(manifest: Path, suffix: str) -> Path:
    stem = manifest.name[:-5] if manifest.name.endswith(".json") else manifest.name
    return manifest.with_name(f"{stem}{suffix}")


def _sha256_bytes(payload: bytes) -> bytes:
    return hashlib.sha256(payload).digest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="manifest file to hash")
    ap.add_argument("--hex-out", default="", help="optional path for hex digest output")
    ap.add_argument("--bin-out", default="", help="optional path for raw digest output")
    args = ap.parse_args()

    manifest = Path(args.manifest).resolve()
    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest}")
    if not manifest.is_file():
        raise SystemExit(f"manifest is not a file: {manifest}")

    payload = manifest.read_bytes()
    digest = _sha256_bytes(payload)

    hex_out = Path(args.hex_out).resolve() if args.hex_out else _default_out_path(manifest, ".sha256")
    bin_out = Path(args.bin_out).resolve() if args.bin_out else _default_out_path(manifest, ".sha256.bin")

    hex_out.parent.mkdir(parents=True, exist_ok=True)
    bin_out.parent.mkdir(parents=True, exist_ok=True)

    hex_out.write_text(digest.hex() + "\n", encoding="utf-8")
    bin_out.write_bytes(digest)


if __name__ == "__main__":
    main()
