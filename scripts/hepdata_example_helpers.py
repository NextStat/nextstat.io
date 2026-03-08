"""Shared helpers for HEPData example artifacts and schema smoke tests."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Iterator

CANONICAL_DATASET_ID = "custom.hepdata.90607.v3.r3.example"
CANONICAL_DISPLAY_NAME = "ATLAS 1Lbb Custom Example"
CANONICAL_DOI = "https://doi.org/10.17182/hepdata.90607.v3/r3"
CANONICAL_BKGONLY_FILENAME = "BkgOnly.json"
CANONICAL_PATCHSET_FILENAME = "patchset.json"
PREFERRED_BENCHMARK_PATCH = "C1N2_Wh_hbb_175_25"

_CANONICAL_ROOT = PurePosixPath("/tmp/nextstat-hepdata")
_CANONICAL_CACHE_DIR = _CANONICAL_ROOT / "cache"
_CANONICAL_OUT_DIR = _CANONICAL_ROOT / "out"
_CANONICAL_LOCK_PATH = _CANONICAL_ROOT / "workspaces.lock.json"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def import_schema_path() -> Path:
    return repo_root() / "docs" / "schemas" / "io" / "hepdata_import_v1.schema.json"


def lock_schema_path() -> Path:
    return repo_root() / "docs" / "schemas" / "io" / "hepdata_lock_v1.schema.json"


def example_path(name: str) -> Path:
    return repo_root() / "docs" / "specs" / name


def fixture_archive() -> Path:
    return repo_root() / "tests" / "hepdata" / "_cache" / "hepdata.90607.v3.r3" / "download"


def nextstat_prefix() -> list[str]:
    override = os.environ.get("NEXTSTAT_HEPDATA_CMD")
    if override:
        return shlex.split(override)
    return ["cargo", "run", "-q", "-p", "ns-cli", "--", "import", "hepdata"]


def run_json(args: list[str]) -> dict:
    proc = subprocess.run(
        nextstat_prefix() + args,
        cwd=repo_root(),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    return json.loads(proc.stdout)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_cached_download(cache_dir: Path, dataset_id: str) -> Path:
    ds_cache = cache_dir / dataset_id.replace("/", "_")
    ds_cache.mkdir(parents=True, exist_ok=True)
    archive_path = ds_cache / "download"
    shutil.copy(fixture_archive(), archive_path)
    return archive_path


@contextmanager
def static_archive_server(archive_bytes: bytes) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/download":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(archive_bytes)))
            self.end_headers()
            self.wfile.write(archive_bytes)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/download"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def _canonical_download_path(dataset_id: str) -> str:
    return str(_CANONICAL_CACHE_DIR / dataset_id / "download")


def _canonical_dataset_dir(dataset_id: str) -> PurePosixPath:
    return _CANONICAL_OUT_DIR / dataset_id


def _selected_patch_names(available_patch_names: list[str]) -> list[str]:
    if not available_patch_names:
        raise RuntimeError("expected at least one HEPData patch name")
    selected = [available_patch_names[0]]
    benchmark = (
        PREFERRED_BENCHMARK_PATCH
        if PREFERRED_BENCHMARK_PATCH in available_patch_names
        else available_patch_names[min(1, len(available_patch_names) - 1)]
    )
    if benchmark not in selected:
        selected.append(benchmark)
    return selected


def preferred_benchmark_patch_name(available_patch_names: list[str]) -> str:
    if not available_patch_names:
        raise RuntimeError("expected at least one HEPData patch name")
    if PREFERRED_BENCHMARK_PATCH in available_patch_names:
        return PREFERRED_BENCHMARK_PATCH
    return available_patch_names[0]


def _canonicalize_download(download: dict, dataset_id: str) -> dict:
    return {
        "url": CANONICAL_DOI,
        "mode": download["mode"],
        "cached": download["cached"],
        "path": _canonical_download_path(dataset_id),
        "sha256": download["sha256"],
    }


def _canonicalize_inputs(inputs: dict, selected_patch_names: list[str]) -> dict:
    return {
        "bkgonly_filename": inputs["bkgonly_filename"],
        "patchset_filename": inputs["patchset_filename"],
        "available_patch_names": selected_patch_names,
    }


def _canonicalize_timings(timings: dict) -> dict:
    download_s = 0.17 if float(timings.get("download_s", 0.0)) > 0.0 else 0.0
    extract_archive_s = 0.08 if float(timings.get("extract_archive_s", 0.0)) > 0.0 else 0.0
    extract_nested_archives_s = (
        0.03 if float(timings.get("extract_nested_archives_s", 0.0)) > 0.0 else 0.0
    )
    archive_prepare_s = round(download_s + extract_archive_s + extract_nested_archives_s, 6)
    inspect_inputs_s = 0.02 if float(timings.get("inspect_inputs_s", 0.0)) > 0.0 else 0.0
    materialize_bkgonly_s = (
        0.01 if float(timings.get("materialize_bkgonly_s", 0.0)) > 0.0 else 0.0
    )
    materialize_patches_s = (
        0.04 if float(timings.get("materialize_patches_s", 0.0)) > 0.0 else 0.0
    )
    materialize_total_s = round(materialize_bkgonly_s + materialize_patches_s, 6)
    total_s = round(archive_prepare_s + inspect_inputs_s + materialize_total_s, 6)
    return {
        "total_s": total_s,
        "archive_prepare_s": archive_prepare_s,
        "download_s": download_s,
        "extract_archive_s": extract_archive_s,
        "extract_nested_archives_s": extract_nested_archives_s,
        "inspect_inputs_s": inspect_inputs_s,
        "materialize_bkgonly_s": materialize_bkgonly_s,
        "materialize_patches_s": materialize_patches_s,
        "materialize_total_s": materialize_total_s,
    }


def _find_patch(materialize_patches: list[dict], patch_name: str) -> dict:
    for patch in materialize_patches:
        if patch.get("patch_name") == patch_name or patch.get("id") == patch_name:
            return patch
    raise RuntimeError(f"patch not found in materialize catalog: {patch_name}")


def build_curated_catalog_example() -> dict:
    return run_json(["--list"])


def build_direct_list_patches_example() -> dict:
    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td) / "cache"
        seed_cached_download(cache_dir, CANONICAL_DATASET_ID)
        raw = run_json(
            [
                "--list-patches",
                "--doi",
                CANONICAL_DOI,
                "--dataset-id",
                CANONICAL_DATASET_ID,
                "--bkgonly-filename",
                CANONICAL_BKGONLY_FILENAME,
                "--patchset-filename",
                CANONICAL_PATCHSET_FILENAME,
                "--cache-dir",
                str(cache_dir),
                "--offline",
            ]
        )

    dataset = raw["datasets"][0]
    selected_patch_names = _selected_patch_names(dataset["inputs"]["available_patch_names"])
    selected_patches = [
        _find_patch(dataset["materialize"]["patches"], patch_name)
        for patch_name in selected_patch_names
    ]

    return {
        "schema_version": raw["schema_version"],
        "mode": raw["mode"],
        "source": raw["source"],
        "source_mode": raw["source_mode"],
        "manifest": raw["manifest"],
        "datasets": [
            {
                "id": CANONICAL_DATASET_ID,
                "name": dataset["name"],
                "doi": CANONICAL_DOI,
                "download": _canonicalize_download(dataset["download"], CANONICAL_DATASET_ID),
                "inputs": _canonicalize_inputs(dataset["inputs"], selected_patch_names),
                "timings": _canonicalize_timings(dataset["timings"]),
                "materialize": {
                    "bkgonly": dataset["materialize"]["bkgonly"],
                    "bkgonly_filename": dataset["materialize"]["bkgonly_filename"],
                    "patchset_filename": dataset["materialize"]["patchset_filename"],
                    "patches": selected_patches,
                },
            }
        ],
    }


def build_direct_materialize_example() -> tuple[dict, dict]:
    archive_bytes = fixture_archive().read_bytes()

    with tempfile.TemporaryDirectory() as td:
        cache_dir = Path(td) / "cache"
        out_dir = Path(td) / "out"
        lock_path = Path(td) / "workspaces.lock.json"

        seed_cached_download(cache_dir, CANONICAL_DATASET_ID)
        list_patches = run_json(
            [
                "--list-patches",
                "--doi",
                CANONICAL_DOI,
                "--dataset-id",
                CANONICAL_DATASET_ID,
                "--bkgonly-filename",
                CANONICAL_BKGONLY_FILENAME,
                "--patchset-filename",
                CANONICAL_PATCHSET_FILENAME,
                "--cache-dir",
                str(cache_dir),
                "--offline",
            ]
        )
        patch_names = _selected_patch_names(
            list_patches["datasets"][0]["inputs"]["available_patch_names"]
        )
        benchmark_patch_name = patch_names[-1]

        with static_archive_server(archive_bytes) as doi_url:
            summary = run_json(
                [
                    "--doi",
                    doi_url,
                    "--dataset-id",
                    CANONICAL_DATASET_ID,
                    "--display-name",
                    CANONICAL_DISPLAY_NAME,
                    "--bkgonly-filename",
                    CANONICAL_BKGONLY_FILENAME,
                    "--patchset-filename",
                    CANONICAL_PATCHSET_FILENAME,
                    "--patch",
                    f"benchmark={benchmark_patch_name}",
                    "--cache-dir",
                    str(cache_dir),
                    "--out-dir",
                    str(out_dir),
                    "--lock",
                    str(lock_path),
                ]
            )

        lock_doc = load_json(lock_path)

    materialized_dataset = summary["datasets"][0]
    lock_dataset = lock_doc["datasets"][0]
    canonical_inputs = _canonicalize_inputs(materialized_dataset["inputs"], patch_names)
    canonical_summary = {
        "schema_version": summary["schema_version"],
        "mode": summary["mode"],
        "source": summary["source"],
        "source_mode": summary["source_mode"],
        "manifest": summary["manifest"],
        "out_dir": str(_CANONICAL_OUT_DIR),
        "cache_dir": str(_CANONICAL_CACHE_DIR),
        "lock": str(_CANONICAL_LOCK_PATH),
        "datasets": [
            {
                "id": CANONICAL_DATASET_ID,
                "name": CANONICAL_DISPLAY_NAME,
                "doi": CANONICAL_DOI,
                "download": _canonicalize_download(materialized_dataset["download"], CANONICAL_DATASET_ID),
                "inputs": canonical_inputs,
                "timings": _canonicalize_timings(materialized_dataset["timings"]),
                "materialized": [
                    {
                        "kind": "bkgonly",
                        "path": str(_canonical_dataset_dir(CANONICAL_DATASET_ID) / CANONICAL_BKGONLY_FILENAME),
                    },
                    {
                        "kind": "patched",
                        "patch_id": "benchmark",
                        "patch_name": benchmark_patch_name,
                        "path": str(_canonical_dataset_dir(CANONICAL_DATASET_ID) / "patched__benchmark.json"),
                    },
                ],
            }
        ],
    }
    canonical_lock = {
        "schema_version": lock_doc["schema_version"],
        "source_mode": lock_doc["source_mode"],
        "generated_by": lock_doc["generated_by"],
        "datasets": [
            {
                "id": CANONICAL_DATASET_ID,
                "name": CANONICAL_DISPLAY_NAME,
                "doi": CANONICAL_DOI,
                "download": _canonicalize_download(lock_dataset["download"], CANONICAL_DATASET_ID),
                "inputs": canonical_inputs,
                "materialized": [
                    {
                        "kind": "bkgonly",
                        "path": str(_canonical_dataset_dir(CANONICAL_DATASET_ID) / CANONICAL_BKGONLY_FILENAME),
                        "sha256": lock_dataset["materialized"][0]["sha256"],
                    },
                    {
                        "kind": "patched",
                        "patch_id": "benchmark",
                        "patch_name": benchmark_patch_name,
                        "path": str(_canonical_dataset_dir(CANONICAL_DATASET_ID) / "patched__benchmark.json"),
                        "sha256": lock_dataset["materialized"][1]["sha256"],
                    },
                ],
            }
        ],
    }
    return canonical_summary, canonical_lock


def generate_examples() -> dict[Path, str]:
    direct_materialize, lock_example = build_direct_materialize_example()
    documents = {
        example_path("hepdata_import_v1.catalog.example.json"): build_curated_catalog_example(),
        example_path("hepdata_import_v1.list_patches.example.json"): build_direct_list_patches_example(),
        example_path("hepdata_import_v1.materialize.example.json"): direct_materialize,
        example_path("hepdata_lock_v1.example.json"): lock_example,
    }
    return {path: json.dumps(document, indent=2) + "\n" for path, document in documents.items()}
