"""ROOT/TREx boundary decision tests (RWS-08).

Verifies that ROOT/TREx parity surfaces have clear governance:
  1. TREx import CLIs are linked to root_trexfitter_parity release slice
  2. HistFactory import/export CLIs are linked to histfactory_stable_surface
  3. No import_export surfaces remain in migration
  4. import_export surfaces with release refs have real governance
  5. Migration ceiling decreases after RWS-08 (import_export closure)

RWS-08 ADR:
  "Remove ambiguity between public stable workflow narrative and optional
   release parity treatment."

Decision: ROOT/TREx parity is optional (required_for_release=false).
TREx import utilities support this optional slice.

NOTE: RWS-08 does NOT claim zero global migration. 36 HEP surfaces
(infrastructure, unbinned, viz, preprocess) remain honestly in
not_release_governed — they have no release slice. Zero migration
is the RWS-10 goal after validation bundle coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def repo_matrix() -> dict:
    return json.loads(
        (REPO / "repo_surface_matrix_v1.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def all_surfaces(repo_matrix: dict) -> list[dict]:
    return repo_matrix["surfaces"]


@pytest.fixture(scope="module")
def import_export_surfaces(all_surfaces: list[dict]) -> list[dict]:
    return [
        s for s in all_surfaces
        if s["owner_slice"] == "import_export"
        and s["surface_kind"] == "runtime"
    ]


# ── TREx import CLIs linked to release slice ──────────────────────────


TREX_CLI_IDS = [
    "hep.import_export.import trex-config.cli",
    "hep.import_export.trex import-config.cli",
]


def test_trex_import_clis_exist(all_surfaces: list[dict]) -> None:
    by_id = {s["surface_id"]: s for s in all_surfaces}
    missing = [sid for sid in TREX_CLI_IDS if sid not in by_id]
    assert not missing, f"TREx import CLI surfaces missing: {missing}"


def test_trex_import_clis_linked_to_parity(all_surfaces: list[dict]) -> None:
    """TREx import CLIs should link to root_trexfitter_parity."""
    by_id = {s["surface_id"]: s for s in all_surfaces}
    for sid in TREX_CLI_IDS:
        s = by_id[sid]
        assert s.get("release_surface_ref") == "root_trexfitter_parity", (
            f"{sid}: should link to root_trexfitter_parity, "
            f"got '{s.get('release_surface_ref')}'"
        )


def test_trex_import_clis_not_in_migration(all_surfaces: list[dict]) -> None:
    by_id = {s["surface_id"]: s for s in all_surfaces}
    migration = [
        sid for sid in TREX_CLI_IDS
        if by_id[sid]["release_status"] == "not_release_governed"
    ]
    assert not migration, (
        f"TREx import CLIs still in migration: {migration}"
    )


# ── HistFactory import/export linked to release slice ─────────────────


HISTFACTORY_IE_IDS = [
    "hep.import_export.import histfactory.cli",
    "hep.import_export.export histfactory.cli",
    "hep.import_export.import cabinetry.cli",
]


def test_histfactory_ie_linked_to_release(all_surfaces: list[dict]) -> None:
    """HistFactory/cabinetry import CLIs should link to histfactory_stable_surface."""
    by_id = {s["surface_id"]: s for s in all_surfaces}
    for sid in HISTFACTORY_IE_IDS:
        s = by_id[sid]
        assert s.get("release_surface_ref") == "histfactory_stable_surface", (
            f"{sid}: should link to histfactory_stable_surface, "
            f"got '{s.get('release_surface_ref')}'"
        )


# ── No import_export migration rows ──────────────────────────────────


def test_no_import_export_migration_rows(import_export_surfaces: list[dict]) -> None:
    migration = [
        s["surface_id"]
        for s in import_export_surfaces
        if s["public_status"] == "stable"
        and s["release_status"] == "not_release_governed"
    ]
    assert not migration, (
        f"import_export surfaces still in migration: {migration}"
    )


# ── Migration ceiling after RWS-08 ───────────────────────────────────


def test_global_migration_ceiling_after_rws08(repo_matrix: dict) -> None:
    """RWS-08 should decrease migration count (was <=42 after RWS-07).

    RWS-08 governs 6 import_export surfaces. Remaining 36 HEP surfaces
    (infrastructure, unbinned, viz, preprocess) stay honestly in migration.
    """
    migration_rows = [
        s
        for s in repo_matrix["surfaces"]
        if s["public_status"] == "stable"
        and s["release_status"] == "not_release_governed"
    ]
    # Was 42 after RWS-07, RWS-08 governs 6 import_export surfaces → <=36
    assert len(migration_rows) <= 36, (
        f"After RWS-08, migration ceiling should be <=36. "
        f"Got {len(migration_rows)} rows."
    )
