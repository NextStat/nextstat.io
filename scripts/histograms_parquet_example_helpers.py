"""Shared helpers for histogram Parquet manifest example artifacts and schema smoke tests."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator


FIXED_CREATED_AT = "2026-01-01T00:00:00Z"
CANONICAL_ROOT = PurePosixPath("/tmp/nextstat-histograms-parquet")
CANONICAL_PARQUET_PATH = CANONICAL_ROOT / "histograms.parquet"
CANONICAL_OBSERVATIONS_PATH = CANONICAL_ROOT / "histograms.parquet.observations.json"
CANONICAL_MANIFEST_PATH = CANONICAL_ROOT / "histograms.parquet.manifest.json"


@dataclass(frozen=True)
class RuntimeBundle:
    parquet_path: Path
    manifest_path: Path
    observations_path: Path
    manifest: dict
    observations: dict[str, list[float]]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def schema_path() -> Path:
    return repo_root() / "docs" / "schemas" / "io" / "histograms_parquet_manifest_v1.schema.json"


def example_path() -> Path:
    return repo_root() / "docs" / "specs" / "histograms_parquet_manifest_v1.example.json"


def _ensure_bindings_path() -> None:
    bindings_path = repo_root() / "bindings" / "ns-py" / "python"
    bindings_str = str(bindings_path)
    if bindings_str not in sys.path:
        sys.path.insert(0, bindings_str)


def load_arrow_io():
    _ensure_bindings_path()
    return importlib.import_module("nextstat.arrow_io")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_histogram_table():
    import pyarrow as pa  # type: ignore

    return pa.table(
        {
            "channel": ["CR", "CR", "SR", "SR"],
            "sample": ["background", "control", "signal", "background"],
            "yields": [
                [42.0, 38.0],
                [7.5, 6.5],
                [1.8, 2.1, 1.1],
                [11.0, 8.0, 4.5],
            ],
            "stat_error": [
                [4.2, 3.8],
                [0.8, 0.7],
                [0.2, 0.2, 0.1],
                [1.1, 0.8, 0.5],
            ],
        }
    )


def build_observations() -> dict[str, list[float]]:
    return {
        "CR": [50.0, 45.0],
        "SR": [14.0, 11.0, 6.0],
    }


@contextmanager
def runtime_bundle() -> Iterator[RuntimeBundle]:
    arrow_io = load_arrow_io()
    table = build_histogram_table()
    observations = build_observations()

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        parquet_path = tmpdir / "histograms.parquet"
        manifest_path = tmpdir / "histograms.parquet.manifest.json"
        observations_path = tmpdir / "histograms.parquet.observations.json"
        manifest = arrow_io.write_histograms_parquet(
            table,
            parquet_path,
            compression="zstd",
            write_manifest=True,
            manifest_path=manifest_path,
            poi="mu",
            observations=observations,
            observations_path=observations_path,
        )
        arrow_io.validate_histograms_parquet_manifest(manifest)
        yield RuntimeBundle(
            parquet_path=parquet_path,
            manifest_path=manifest_path,
            observations_path=observations_path,
            manifest=manifest,
            observations=observations,
        )


def canonicalize_manifest(manifest: dict) -> dict:
    doc = json.loads(json.dumps(manifest))
    doc["created_at_utc"] = FIXED_CREATED_AT
    doc["parquet_path"] = str(CANONICAL_PARQUET_PATH)
    doc["observations_path"] = str(CANONICAL_OBSERVATIONS_PATH)
    return doc


def generate_example() -> dict:
    with runtime_bundle() as bundle:
        return canonicalize_manifest(bundle.manifest)


def generate_examples() -> dict[Path, str]:
    return {
        example_path(): json.dumps(generate_example(), indent=2) + "\n",
    }
