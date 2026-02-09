#!/usr/bin/env python3
"""Publish a packaged snapshot archive to Zenodo (seed tool).

This script intentionally reads the token from an environment variable to avoid
leaking secrets into shell history or files.

Flow:
1) Ensure we have an archive (.tar.gz) + sha256 + zenodo_deposition.json by running package_snapshot.py
2) Create a new deposition (draft)
3) Upload the archive to the deposition bucket
4) Set metadata
5) Optionally publish
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def zenodo_base(server: str) -> str:
    if server == "production":
        return "https://zenodo.org/api"
    if server == "sandbox":
        return "https://sandbox.zenodo.org/api"
    raise ValueError(f"unknown server: {server}")


def req_json(method: str, url: str, *, payload: dict[str, Any] | None, token: str) -> dict[str, Any]:
    qs = urllib.parse.urlencode({"access_token": token})
    u = url + ("&" if "?" in url else "?") + qs
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(u, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Zenodo HTTP {e.code}: {body}") from None


def put_file(url: str, *, file_path: Path) -> None:
    # Zenodo bucket URLs do not require access_token (they are scoped to the deposition).
    # However, some deployments still require it; if we need that, we can switch to
    # the /deposit/depositions/{id}/files endpoint. For now, try bucket PUT first.
    data = file_path.read_bytes()
    req = urllib.request.Request(url, data=data, method="PUT", headers={"Content-Type": "application/octet-stream"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Zenodo upload HTTP {e.code}: {body}") from None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", choices=["sandbox", "production"], default="sandbox")
    ap.add_argument("--token-env", default="ZENODO_TOKEN")
    ap.add_argument("--snapshot-dir", required=True, help="Snapshot dir containing baseline_manifest.json")
    ap.add_argument("--out-dir", default="zenodo/out", help="Where package_snapshot.py writes artifacts")
    ap.add_argument("--publish", action="store_true", help="Publish the deposition (makes it public).")
    args = ap.parse_args()

    token = os.environ.get(str(args.token_env), "").strip()
    if not token:
        raise SystemExit(f"missing token env var {args.token_env}")

    repo_root = Path(__file__).resolve().parents[1]
    snapshot_dir = Path(args.snapshot_dir).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir = out_dir.resolve()

    # 1) Package snapshot (archive + sha256 + zenodo_deposition.json)
    pkg_py = repo_root / "zenodo/package_snapshot.py"
    rc = subprocess.call(
        [sys.executable, str(pkg_py), "--snapshot-dir", str(snapshot_dir), "--out-dir", str(out_dir)],
        cwd=str(repo_root),
    )
    if rc != 0:
        print(f"package_snapshot.py exited {rc} (archive may already exist, continuing)")

    manifest = load_json(snapshot_dir / "baseline_manifest.json")
    snapshot_id = str(manifest.get("snapshot_id") or snapshot_dir.name)
    archive_path = out_dir / f"{snapshot_id}.tar.gz"
    meta_path = out_dir / "zenodo_deposition.json"
    if not archive_path.exists():
        raise SystemExit(f"missing archive: {archive_path}")
    if not meta_path.exists():
        raise SystemExit(f"missing metadata: {meta_path}")

    meta = load_json(meta_path)
    payload = {"metadata": meta}

    base = zenodo_base(str(args.server))

    # 2) Create deposition draft.
    dep = req_json("POST", f"{base}/deposit/depositions", payload={}, token=token)
    dep_id = dep.get("id")
    links = dep.get("links") or {}
    bucket = links.get("bucket")
    if not dep_id or not bucket:
        raise SystemExit(f"unexpected deposition response: {dep}")

    # 3) Upload file to bucket.
    upload_url = str(bucket).rstrip("/") + "/" + urllib.parse.quote(archive_path.name)
    try:
        put_file(upload_url, file_path=archive_path)
    except RuntimeError as e:
        # Some installations require the access token on bucket URLs.
        upload_url2 = upload_url + ("&" if "?" in upload_url else "?") + urllib.parse.urlencode({"access_token": token})
        put_file(upload_url2, file_path=archive_path)

    # 4) Set metadata (may require a short delay after upload).
    for i in range(5):
        try:
            req_json("PUT", f"{base}/deposit/depositions/{dep_id}", payload=payload, token=token)
            break
        except RuntimeError:
            if i == 4:
                raise
            time.sleep(1.0)

    # 5) Optionally publish.
    if args.publish:
        req_json("POST", f"{base}/deposit/depositions/{dep_id}/actions/publish", payload=None, token=token)

    # Print only non-sensitive identifiers.
    print(json.dumps({"server": args.server, "deposition_id": dep_id, "snapshot_id": snapshot_id}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
