#!/usr/bin/env python3
"""Fetch and extract a Zenodo record snapshot (seed tool).

Downloads the first `.tar.gz` file attached to a Zenodo record and verifies:
- file sha256 matches the `Archive sha256:` embedded in the record description (if present)

Then extracts to `<out-dir>/extracted/<top_folder>/` and prints the extracted snapshot directory path.

Note: In some locked-down environments, Python processes may not have working DNS/network access.
In those cases, fetch using shell `curl` and `tar` directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path
from typing import Any


def zenodo_record_api(server: str, record_id: str) -> str:
    if server == "production":
        return f"https://zenodo.org/api/records/{record_id}"
    if server == "sandbox":
        return f"https://sandbox.zenodo.org/api/records/{record_id}"
    raise ValueError(f"unknown server: {server}")


def load_json_bytes(b: bytes) -> dict[str, Any]:
    return json.loads(b.decode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def curl_get_bytes(url: str) -> bytes:
    return subprocess.check_output(["curl", "-sS", url])


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["curl", "-sSLo", str(out_path), url])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", choices=["production", "sandbox"], default="production")
    ap.add_argument("--record-id", required=True)
    ap.add_argument("--out-dir", default="tmp/zenodo_records", help="Output directory root (downloads + extracted).")
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve()
    rec_dir = out_root / str(args.record_id)
    rec_dir.mkdir(parents=True, exist_ok=True)

    rec = load_json_bytes(curl_get_bytes(zenodo_record_api(str(args.server), str(args.record_id))))
    (rec_dir / "record.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")

    desc = ((rec.get("metadata") or {}).get("description") or "").strip()
    m = re.search(r"Archive sha256: ([a-f0-9]{64})", desc)
    expected_sha = m.group(1) if m else ""

    files = rec.get("files") or []
    tar_key = ""
    tar_url = ""
    for f in files:
        k = str(f.get("key") or "")
        if k.endswith(".tar.gz"):
            tar_key = k
            links = f.get("links") or {}
            tar_url = str(links.get("self") or links.get("download") or "")
            break
    if not tar_key or not tar_url:
        raise SystemExit("no .tar.gz file found on record")

    archive_path = rec_dir / tar_key
    download(tar_url, archive_path)

    got_sha = sha256_file(archive_path)
    (rec_dir / "archive.sha256").write_text(got_sha + "\n")
    if expected_sha and expected_sha != got_sha:
        raise SystemExit(f"sha256 mismatch: expected {expected_sha} got {got_sha}")

    extracted_root = rec_dir / "extracted"
    extracted_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(extracted_root)

    # Heuristic: top-level folder name equals tar stem before .tar.gz
    # but we also scan for baseline_manifest.json.
    candidates = list(extracted_root.glob("*/baseline_manifest.json"))
    if candidates:
        snap_dir = candidates[0].parent
    else:
        snap_dir = extracted_root

    print(str(snap_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
