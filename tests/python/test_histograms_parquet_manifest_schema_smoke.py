from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import jsonschema  # type: ignore
from _io_contract_doc_assertions import assert_doc_contains_strings


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_histograms_helpers():
    module_path = _repo_root() / "scripts" / "histograms_parquet_example_helpers.py"
    spec = importlib.util.spec_from_file_location("histograms_parquet_example_helpers", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_HELPERS = _load_histograms_helpers()
_schema_path = _HELPERS.schema_path
_example_path = _HELPERS.example_path
_load_json = _HELPERS.load_json


def _validate_schema(instance: dict, schema_path: Path) -> None:
    schema = _load_json(schema_path)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(instance=instance, schema=schema)


def test_histograms_parquet_manifest_schema_file_smoke() -> None:
    schema_path = _schema_path()
    assert schema_path.exists(), f"missing schema: {schema_path}"

    schema_doc = _load_json(schema_path)
    assert schema_doc["$id"] == "https://nextstat.io/schemas/io/histograms_parquet_manifest_v1.schema.json"
    assert schema_doc["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema_doc["type"] == "object"


def test_histograms_parquet_manifest_example_matches_schema() -> None:
    example_path = _example_path()
    assert example_path.exists(), f"missing example: {example_path}"
    example = _load_json(example_path)
    _validate_schema(example, _schema_path())
    assert example["schema_version"] == "nextstat.histograms_parquet_manifest.v1"
    assert example["observations_path"] is not None

    if os.environ.get("NEXTSTAT_HISTOGRAMS_PARQUET_SKIP_GENERATOR_CHECK") != "1":
        check = subprocess.run(
            [sys.executable, "scripts/generate_histograms_parquet_schema_examples.py", "--check"],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        assert check.returncode == 0, check.stderr or check.stdout


def test_histograms_parquet_manifest_runtime_matches_schema() -> None:
    with _HELPERS.runtime_bundle() as bundle:
        _validate_schema(bundle.manifest, _schema_path())
        assert bundle.manifest["schema_version"] == "nextstat.histograms_parquet_manifest.v1"
        assert bundle.manifest["observations_path"] == str(bundle.observations_path.resolve())
        assert bundle.manifest["stats"]["has_stat_error"] is True
        assert bundle.manifest["stats"]["n_rows"] == 4
        assert bundle.manifest["stats"]["channels"] == [
            {"name": "CR", "n_bins": 2},
            {"name": "SR", "n_bins": 3},
        ]

        observations_doc = json.loads(bundle.observations_path.read_text(encoding="utf-8"))
        assert observations_doc == bundle.observations


def test_histograms_parquet_reference_docs_publish_contract_workflow() -> None:
    assert_doc_contains_strings(
        _repo_root() / "docs" / "references" / "arrow-parquet-io.md",
        [
            "## Published Contract",
            "docs/specs/histograms_parquet_manifest_v1.example.json",
            "scripts/generate_histograms_parquet_schema_examples.py",
            "scripts/check_io_contracts.py --family histograms_parquet",
            "docs/schemas/io/nextstat_io_contract_runner_report_v1.schema.json",
        ],
    )
    assert_doc_contains_strings(
        _repo_root() / "docs" / "references" / "python-api.md",
        [
            "Contract references:",
            "docs/specs/histograms_parquet_manifest_v1.example.json",
            "scripts/generate_histograms_parquet_schema_examples.py",
            "scripts/check_io_contracts.py --family histograms_parquet",
            "docs/schemas/io/nextstat_io_contract_runner_report_v1.schema.json",
        ],
    )
