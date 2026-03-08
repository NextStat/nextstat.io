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


def _load_hepdata_helpers():
    module_path = _repo_root() / "scripts" / "hepdata_example_helpers.py"
    spec = importlib.util.spec_from_file_location("hepdata_example_helpers", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_HELPERS = _load_hepdata_helpers()
_import_schema_path = _HELPERS.import_schema_path
_lock_schema_path = _HELPERS.lock_schema_path
_example_path = _HELPERS.example_path
_fixture_archive = _HELPERS.fixture_archive
_run_json = _HELPERS.run_json
_load_json = _HELPERS.load_json
_seed_cached_download = _HELPERS.seed_cached_download
_static_archive_server = _HELPERS.static_archive_server


def _validate_schema(instance: dict, schema_path: Path) -> None:
    schema = _load_json(schema_path)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(instance=instance, schema=schema)


def test_hepdata_schema_files_smoke() -> None:
    import_schema = _import_schema_path()
    lock_schema = _lock_schema_path()

    assert import_schema.exists(), f"missing schema: {import_schema}"
    assert lock_schema.exists(), f"missing schema: {lock_schema}"

    import_doc = _load_json(import_schema)
    lock_doc = _load_json(lock_schema)

    assert import_doc["$id"] == "https://nextstat.io/schemas/io/hepdata_import_v1.schema.json"
    assert lock_doc["$id"] == "https://nextstat.io/schemas/io/hepdata_lock_v1.schema.json"
    assert import_doc["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert lock_doc["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_hepdata_example_artifacts_match_schemas() -> None:
    import_examples = [
        (
            "hepdata_import_v1.catalog.example.json",
            "catalog",
            "curated",
        ),
        (
            "hepdata_import_v1.list_patches.example.json",
            "catalog",
            "direct_doi",
        ),
        (
            "hepdata_import_v1.materialize.example.json",
            "materialize",
            "direct_doi",
        ),
    ]

    for example_name, expected_mode, expected_source_mode in import_examples:
        example_path = _example_path(example_name)
        assert example_path.exists(), f"missing example: {example_path}"
        example = _load_json(example_path)
        _validate_schema(example, _import_schema_path())
        assert example["schema_version"] == "nextstat.hepdata_import.v1"
        assert example["mode"] == expected_mode
        assert example["source_mode"] == expected_source_mode

    lock_example_path = _example_path("hepdata_lock_v1.example.json")
    assert lock_example_path.exists(), f"missing example: {lock_example_path}"
    lock_example = _load_json(lock_example_path)
    _validate_schema(lock_example, _lock_schema_path())
    assert lock_example["schema_version"] == "nextstat.hepdata_lock.v1"
    assert lock_example["source_mode"] == "direct_doi"

    if os.environ.get("NEXTSTAT_HEPDATA_SKIP_GENERATOR_CHECK") != "1":
        check = subprocess.run(
            [sys.executable, "scripts/generate_hepdata_schema_examples.py", "--check"],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        assert check.returncode == 0, check.stderr or check.stdout


def test_hepdata_curated_catalog_matches_schema() -> None:
    summary = _run_json(["--list"])
    _validate_schema(summary, _import_schema_path())

    assert summary["schema_version"] == "nextstat.hepdata_import.v1"
    assert summary["mode"] == "catalog"
    assert summary["source_mode"] == "curated"


def test_hepdata_direct_patch_catalog_matches_schema(tmp_path: Path) -> None:
    dataset_id = "custom.hepdata.90607.v3.r3.catalog.schema"
    cache_dir = tmp_path / "cache"
    _seed_cached_download(cache_dir, dataset_id)

    summary = _run_json(
        [
            "--list-patches",
            "--doi",
            "https://doi.org/10.17182/hepdata.90607.v3/r3",
            "--dataset-id",
            dataset_id,
            "--bkgonly-filename",
            "BkgOnly.json",
            "--patchset-filename",
            "patchset.json",
            "--cache-dir",
            str(cache_dir),
            "--offline",
        ]
    )
    _validate_schema(summary, _import_schema_path())

    assert summary["mode"] == "catalog"
    assert summary["source_mode"] == "direct_doi"
    assert summary["datasets"][0]["download"]["mode"] == "cached"
    assert summary["datasets"][0]["timings"]["inspect_inputs_s"] > 0
    assert summary["datasets"][0]["timings"]["download_s"] == 0
    assert summary["datasets"][0]["timings"]["extract_archive_s"] > 0


def test_hepdata_curated_materialize_summary_and_lock_match_schemas(tmp_path: Path) -> None:
    dataset_id = "hepdata.90607.v3.r3"
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "out"
    lock_path = tmp_path / "workspaces.lock.json"
    _seed_cached_download(cache_dir, dataset_id)

    summary = _run_json(
        [
            "--dataset",
            dataset_id,
            "--cache-dir",
            str(cache_dir),
            "--out-dir",
            str(out_dir),
            "--lock",
            str(lock_path),
            "--offline",
        ]
    )
    _validate_schema(summary, _import_schema_path())

    lock_doc = _load_json(lock_path)
    _validate_schema(lock_doc, _lock_schema_path())

    assert summary["mode"] == "materialize"
    assert summary["source_mode"] == "curated"
    assert summary["datasets"][0]["timings"]["download_s"] == 0
    assert summary["datasets"][0]["timings"]["materialize_total_s"] > 0
    assert lock_doc["source_mode"] == "curated"


def test_hepdata_direct_network_materialize_summary_and_lock_match_schemas(tmp_path: Path) -> None:
    dataset_id = "custom.hepdata.90607.v3.r3.network.schema"
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "out"
    lock_path = tmp_path / "workspaces.lock.json"
    archive_bytes = _fixture_archive().read_bytes()

    with _static_archive_server(archive_bytes) as doi_url:
        summary = _run_json(
            [
                "--doi",
                doi_url,
                "--dataset-id",
                dataset_id,
                "--display-name",
                "Network 90607 Schema",
                "--bkgonly-filename",
                "BkgOnly.json",
                "--patchset-filename",
                "patchset.json",
                "--patch",
                "first_patch",
                "--cache-dir",
                str(cache_dir),
                "--out-dir",
                str(out_dir),
                "--lock",
                str(lock_path),
            ]
        )

    _validate_schema(summary, _import_schema_path())

    lock_doc = _load_json(lock_path)
    _validate_schema(lock_doc, _lock_schema_path())

    assert summary["source_mode"] == "direct_doi"
    assert summary["datasets"][0]["download"]["mode"] == "network"
    assert summary["datasets"][0]["download"]["cached"] is False
    assert summary["datasets"][0]["timings"]["download_s"] > 0
    assert summary["datasets"][0]["timings"]["materialize_total_s"] > 0
    assert lock_doc["source_mode"] == "direct_doi"
    assert lock_doc["datasets"][0]["download"]["mode"] == "network"


def test_hepdata_reference_docs_publish_contract_workflow() -> None:
    assert_doc_contains_strings(
        _repo_root() / "docs" / "references" / "cli.md",
        [
            "### HEPData import catalog / materialization",
            "nextstat config schema --name hepdata_import_v1",
            "docs/specs/hepdata_import_v1.catalog.example.json",
            "docs/specs/hepdata_import_v1.list_patches.example.json",
            "docs/specs/hepdata_import_v1.materialize.example.json",
            "scripts/generate_hepdata_schema_examples.py [--check]",
            "scripts/check_io_contracts.py --family hepdata",
            "docs/schemas/io/nextstat_io_contract_runner_report_v1.schema.json",
            "docs/specs/hep/hepdata_import_acceptance_v1.md",
            "docs/benchmarks/hepdata-import-benchmark-snapshot-2026-03-08.md",
            "docs/benchmarks/hepdata-import-support-matrix-2026-03-08.md",
            "docs/benchmarks/hepdata-import-release-notes-2026-03-08.md",
            "docs/benchmarks/hepdata-import-promotion-runbook-2026-03-08.md",
            "docs/benchmarks/hepdata-import-release-pr-checklist-2026-03-08.md",
            "nextstat config schema --name hepdata_lock_v1",
            "docs/specs/hepdata_lock_v1.example.json",
        ],
    )


def test_hepdata_acceptance_spec_covers_release_gates() -> None:
    assert_doc_contains_strings(
        _repo_root() / "docs" / "specs" / "hep" / "hepdata_import_acceptance_v1.md",
        [
            "# HEPData Import Acceptance Criteria (Stable Surface v1)",
            "nextstat import hepdata --list",
            "nextstat import hepdata --list-patches --doi <url> --dataset-id <id>",
            'schema_version = "nextstat.hepdata_import.v1"',
            'schema_version = "nextstat.hepdata_lock.v1"',
            "python scripts/check_io_contracts.py --family hepdata",
            "pytest -q tests/python/test_hepdata_schema_smoke.py",
            "cargo test -p ns-cli --test cli_import_hepdata",
            "cargo test -p ns-cli --test cli_bundle_more_commands",
            "docs-only and test-only changes do **not** require a bench run",
            "nextstat-bench",
            "python suites/hep/run.py --deterministic --out out/hep_simple_nll.json",
        ],
    )
