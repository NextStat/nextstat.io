#!/usr/bin/env python3
"""Fetch and materialize pyhf JSON workspaces from HEPData (via DOI resources).

This script is intentionally:
- opt-in (downloads external artifacts)
- cache-aware (writes under tests/hepdata/_cache/)
- repo-clean (materialized JSONs live under tests/hepdata/workspaces/ and should
  not be committed)

Outputs:
- tests/hepdata/workspaces/<dataset-id>/*.json (materialized workspaces)
- tests/hepdata/workspaces.lock.json (sha256 + provenance for reproducibility)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = ROOT / "hepdata" / "manifest.json"
DEFAULT_CACHE_DIR = ROOT / "hepdata" / "_cache"
DEFAULT_OUT_DIR = ROOT / "hepdata" / "workspaces"
DEFAULT_LOCK_PATH = ROOT / "hepdata" / "workspaces.lock.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_redirects(url: str, dest: Path) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "nextstat-hepdata-fetch/1.0",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(req) as resp:
        final_url = getattr(resp, "geturl", lambda: url)()
        headers = dict(resp.headers.items())
        with dest.open("wb") as f:
            shutil.copyfileobj(resp, f)
    return {"url": url, "final_url": final_url, "headers": headers}


def _safe_extract_tar(tar: tarfile.TarFile, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for member in tar.getmembers():
        # Guard against path traversal.
        p = dest / member.name
        if not str(p.resolve()).startswith(str(dest.resolve())):
            raise RuntimeError(f"Refusing to extract path outside destination: {member.name}")
    tar.extractall(dest)  # noqa: S202 (controlled above)


def _extract_archive(archive_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            _safe_extract_tar(tf, dest)
        return
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest)  # noqa: S202 (trusted remote, into dedicated cache)
        return
    raise RuntimeError(f"Unknown archive format: {archive_path}")


def _extract_nested_json_archives(root: Path) -> int:
    """Extract nested *.json.tgz / *.json.tar.gz archives in-place.

    HEPData likelihood bundles sometimes ship JSON workspaces as nested archives
    inside the top-level download.
    """
    n = 0
    patterns = ["**/*.json.tgz", "**/*.json.tar.gz", "**/*.json.tar", "**/*.tgz"]
    for pat in patterns:
        for p in root.glob(pat):
            # Avoid re-extracting if we already expanded.
            # Heuristic: if a sibling .json exists, assume extracted.
            if p.suffixes[-2:] == [".json", ".tgz"]:
                sibling_json = p.with_suffix("")  # drops .tgz -> .json
                if sibling_json.exists():
                    continue
            if p.name.endswith(".json.tgz"):
                sibling_json = p.with_suffix("")
                if sibling_json.exists():
                    continue
            try:
                if tarfile.is_tarfile(p):
                    with tarfile.open(p, "r:*") as tf:
                        _safe_extract_tar(tf, p.parent)
                    n += 1
            except tarfile.TarError:
                continue
    return n


def _find_one(root: Path, filename: str) -> Path:
    matches = list(root.glob(f"**/{filename}"))
    if not matches:
        raise FileNotFoundError(f"Did not find '{filename}' under {root}")
    if len(matches) > 1:
        # Prefer the shallowest match.
        matches.sort(key=lambda p: len(p.parts))
    return matches[0]


def _find_optional(root: Path, filename: str | None, *, default: str) -> Path:
    """Find a file by explicit filename (preferred) or fall back to a default name."""
    if filename is not None:
        return _find_one(root, filename)
    return _find_one(root, default)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _materialize_bkgonly(extracted_dir: Path, out_dir: Path, *, bkgonly_filename: str | None) -> Path:
    bkg = _find_optional(extracted_dir, bkgonly_filename, default="BkgOnly.json")
    out = out_dir / "BkgOnly.json"
    shutil.copy2(bkg, out)
    return out


def _materialize_patch(
    extracted_dir: Path,
    out_dir: Path,
    *,
    patch_id: str,
    patch_name: str | None,
    bkgonly_filename: str | None,
    patchset_filename: str | None,
) -> Path:
    bkg = _load_json(_find_optional(extracted_dir, bkgonly_filename, default="BkgOnly.json"))
    patchset = _load_json(_find_optional(extracted_dir, patchset_filename, default="patchset.json"))

    patched: dict[str, Any] | None = None

    # Prefer NextStat's native PatchSet support (no pyhf dependency).
    try:
        import nextstat  # type: ignore
    except Exception:
        nextstat = None  # type: ignore

    if nextstat is not None:
        if patch_name is None:
            patches = patchset.get("patches") or []
            if not patches:
                raise RuntimeError("patchset.json contains no patches")
            meta = patches[0].get("metadata") or {}
            patch_name = meta.get("name")
            if not isinstance(patch_name, str) or not patch_name:
                raise RuntimeError("Failed to infer patch name from patchset.json (missing metadata.name)")
        patched_json = nextstat.apply_patchset(
            json.dumps(bkg),
            json.dumps(patchset),
            patch_name=patch_name,
        )
        patched = json.loads(patched_json)

    if patched is None:
        # Fallback: use pyhf if available.
        try:
            import pyhf  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Materializing patched workspaces requires either NextStat (with native PatchSet support) "
                "or pyhf. Install one of:\n"
                "- pip install nextstat\n"
                "- pip install 'nextstat[validation]'  # includes pyhf\n"
            ) from e

        ps = pyhf.PatchSet(patchset)
        if patch_name is None:
            names = list(ps.patch_names)
            if not names:
                raise RuntimeError("patchset.json contains no patches")
            patch_name = names[0]
        patched = ps.apply(bkg, patch_name)

    out = out_dir / f"patched__{patch_id}.json"
    _write_json(out, patched)
    return out


@dataclass(frozen=True)
class Dataset:
    id: str
    name: str
    doi: str
    materialize_bkgonly: bool
    patches: list[dict[str, Any]]
    bkgonly_filename: str | None
    patchset_filename: str | None


def _load_manifest(manifest_path: Path) -> list[Dataset]:
    raw = json.loads(manifest_path.read_text())
    datasets: list[Dataset] = []
    for d in raw.get("datasets", []):
        mat = d.get("materialize", {})
        datasets.append(
            Dataset(
                id=str(d["id"]),
                name=str(d.get("name", d["id"])),
                doi=str(d["doi"]),
                materialize_bkgonly=bool(mat.get("bkgonly", True)),
                patches=list(mat.get("patches", [])),
                bkgonly_filename=mat.get("bkgonly_filename"),
                patchset_filename=mat.get("patchset_filename"),
            )
        )
    return datasets


def _iter_json_files(root: Path) -> Iterable[Path]:
    return sorted([p for p in root.glob("**/*.json") if p.is_file()])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--cache", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--lock", default=str(DEFAULT_LOCK_PATH))
    ap.add_argument("--dataset", action="append", default=[], help="Dataset id to fetch (repeatable)")
    ap.add_argument("--clean", action="store_true", help="Delete existing cache/output before fetching")
    args = ap.parse_args(argv)

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out)
    cache_dir = Path(args.cache)
    lock_path = Path(args.lock)

    if args.clean:
        shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)
        if lock_path.exists():
            lock_path.unlink()

    datasets = _load_manifest(manifest_path)
    selected = set(args.dataset)
    if selected:
        datasets = [d for d in datasets if d.id in selected]
        missing = selected - {d.id for d in datasets}
        if missing:
            raise SystemExit(f"Unknown dataset id(s): {sorted(missing)}")

    lock: dict[str, Any] = {"generated_by": str(Path(__file__).name), "datasets": []}

    for d in datasets:
        slug = d.id.replace("/", "_")
        ds_cache = cache_dir / slug
        ds_cache.mkdir(parents=True, exist_ok=True)

        archive_path = ds_cache / "download"
        extracted_dir = ds_cache / "extracted"
        ds_out_dir = out_dir / slug

        if archive_path.exists() and archive_path.stat().st_size > 0:
            print(f"[hepdata] {d.id}: using cached download: {archive_path}")
            meta = {"url": d.doi, "cached": True, "path": str(archive_path)}
        else:
            print(f"[hepdata] {d.id}: downloading {d.doi}")
            meta = _download_with_redirects(d.doi, archive_path)

        print(f"[hepdata] {d.id}: extracting archive")
        if extracted_dir.exists():
            shutil.rmtree(extracted_dir)
        _extract_archive(archive_path, extracted_dir)

        # Some bundles contain nested JSON tarballs under likelihoods/.
        n_nested = _extract_nested_json_archives(extracted_dir)
        if n_nested:
            print(f"[hepdata] {d.id}: extracted {n_nested} nested archive(s)")

        ds_out_dir.mkdir(parents=True, exist_ok=True)
        materialized: list[dict[str, Any]] = []

        if d.materialize_bkgonly:
            p = _materialize_bkgonly(extracted_dir, ds_out_dir, bkgonly_filename=d.bkgonly_filename)
            materialized.append({"kind": "bkgonly", "path": str(p), "sha256": _sha256_file(p)})

        for patch in d.patches:
            patch_id = str(patch.get("id") or "patch")
            patch_name = patch.get("patch_name")
            p = _materialize_patch(
                extracted_dir,
                ds_out_dir,
                patch_id=patch_id,
                patch_name=patch_name,
                bkgonly_filename=patch.get("bkgonly_filename", d.bkgonly_filename),
                patchset_filename=patch.get("patchset_filename", d.patchset_filename),
            )
            materialized.append(
                {
                    "kind": "patched",
                    "patch_id": patch_id,
                    "patch_name": patch_name,
                    "path": str(p),
                    "sha256": _sha256_file(p),
                }
            )

        lock["datasets"].append(
            {
                "id": d.id,
                "name": d.name,
                "doi": d.doi,
                "download": meta,
                "materialized": materialized,
            }
        )

    _write_json(lock_path, lock)
    print(f"[hepdata] wrote lockfile: {lock_path}")

    # Sanity: keep output directory tidy (only JSONs).
    for p in _iter_json_files(out_dir):
        _ = p  # placeholder for future checks

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
