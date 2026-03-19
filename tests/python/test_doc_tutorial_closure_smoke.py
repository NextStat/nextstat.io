"""API/tutorial/reference surface normalization tests (RWS-07).

Verifies that documentation and tutorial surfaces are not free-floating:
  1. Platform documentation surfaces promoted from migration
  2. HEP documentation surfaces promoted from migration
  3. HEP tutorial surfaces promoted from migration
  4. No documentation/tutorial surface remains stable+not_release_governed
  5. All doc/tutorial surfaces have correct surface_kind (not runtime)
  6. Global migration ceiling decreases after RWS-07

RWS-07 ADR:
  "Stop API/reference docs from drifting independently of product slices
   while keeping documentation surfaces distinct from runtime surfaces."

Acceptance:
  "No stable API/reference doc is free-floating."
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
def platform_docs(all_surfaces: list[dict]) -> list[dict]:
    return [
        s for s in all_surfaces
        if s["domain"] == "platform"
        and s["surface_kind"] == "documentation"
    ]


@pytest.fixture(scope="module")
def hep_docs(all_surfaces: list[dict]) -> list[dict]:
    return [
        s for s in all_surfaces
        if s["domain"] == "hep"
        and s["surface_kind"] == "documentation"
    ]


@pytest.fixture(scope="module")
def hep_tutorials(all_surfaces: list[dict]) -> list[dict]:
    return [
        s for s in all_surfaces
        if s["domain"] == "hep"
        and s["surface_kind"] == "tutorial"
    ]


# ── Platform documentation promoted ──────────────────────────────────


def test_platform_docs_exist(platform_docs: list[dict]) -> None:
    """Platform domain should have documentation surfaces."""
    assert len(platform_docs) >= 9, (
        f"Expected >=9 platform doc surfaces, got {len(platform_docs)}"
    )


def test_platform_docs_promoted(platform_docs: list[dict]) -> None:
    """Platform docs should be promoted from not_release_governed."""
    migration = [
        s["surface_id"]
        for s in platform_docs
        if s["public_status"] == "stable"
        and s["release_status"] == "not_release_governed"
    ]
    assert not migration, (
        f"Platform docs still in migration: {migration}"
    )


def test_platform_docs_are_documentation_kind(platform_docs: list[dict]) -> None:
    """Platform docs should have surface_kind=documentation (not runtime)."""
    wrong_kind = [
        f"{s['surface_id']} (kind={s['surface_kind']})"
        for s in platform_docs
        if s["surface_kind"] != "documentation"
    ]
    assert not wrong_kind, (
        f"Platform docs with wrong surface_kind: {wrong_kind}"
    )


# ── HEP documentation promoted ──────────────────────────────────────


def test_hep_docs_promoted(hep_docs: list[dict]) -> None:
    """HEP documentation surfaces should be promoted from not_release_governed."""
    migration = [
        s["surface_id"]
        for s in hep_docs
        if s["public_status"] == "stable"
        and s["release_status"] == "not_release_governed"
    ]
    assert not migration, (
        f"HEP docs still in migration: {migration}"
    )


# ── HEP tutorials promoted ──────────────────────────────────────────


def test_hep_tutorials_promoted(hep_tutorials: list[dict]) -> None:
    """HEP tutorials should be promoted from not_release_governed."""
    migration = [
        s["surface_id"]
        for s in hep_tutorials
        if s["public_status"] == "stable"
        and s["release_status"] == "not_release_governed"
    ]
    assert not migration, (
        f"HEP tutorials still in migration: {migration}"
    )


def test_hep_tutorials_are_tutorial_kind(hep_tutorials: list[dict]) -> None:
    """HEP tutorials should have surface_kind=tutorial."""
    wrong_kind = [
        f"{s['surface_id']} (kind={s['surface_kind']})"
        for s in hep_tutorials
        if s["surface_kind"] != "tutorial"
    ]
    assert not wrong_kind, (
        f"HEP tutorials with wrong surface_kind: {wrong_kind}"
    )


# ── No doc/tutorial migration rows after closure ─────────────────────


def test_no_doc_tutorial_migration_rows(all_surfaces: list[dict]) -> None:
    """After RWS-07, no doc/tutorial surface should be stable+not_release_governed."""
    migration = [
        s["surface_id"]
        for s in all_surfaces
        if s["surface_kind"] in ("documentation", "tutorial")
        and s["public_status"] == "stable"
        and s["release_status"] == "not_release_governed"
    ]
    assert not migration, (
        f"After RWS-07, doc/tutorial migration rows must be zero. "
        f"Still in migration ({len(migration)}): {migration}"
    )


# ── ADR-specific: named docs must be governed ─────────────────────────


ADR_REQUIRED_DOCS = [
    "platform.cli.cli_reference.docs",
    "platform.python_api.python_api_reference.docs",
    "platform.rust_api.rust_api_reference.docs",
    "platform.viz.plot_artifacts.docs",
    "platform.config.analysis_config.docs",
    "platform.io.arrow_parquet_io.docs",
]


def test_adr_named_docs_exist_and_governed(all_surfaces: list[dict]) -> None:
    """ADR explicitly names these docs as requiring governance."""
    by_id = {s["surface_id"]: s for s in all_surfaces}
    missing = [sid for sid in ADR_REQUIRED_DOCS if sid not in by_id]
    assert not missing, f"ADR-required doc surfaces missing: {missing}"

    ungoverned = [
        sid for sid in ADR_REQUIRED_DOCS
        if by_id[sid]["release_status"] == "not_release_governed"
    ]
    assert not ungoverned, (
        f"ADR-required docs still not governed: {ungoverned}"
    )


# ── Global migration ceiling decreases ────────────────────────────────


def test_global_migration_ceiling_after_rws07(repo_matrix: dict) -> None:
    """RWS-07 should decrease migration count (was <=63 after RWS-06)."""
    migration_rows = [
        s
        for s in repo_matrix["surfaces"]
        if s["public_status"] == "stable"
        and s["release_status"] == "not_release_governed"
    ]
    # Was 63, RWS-07 promotes 21 doc/tutorial surfaces from migration
    assert len(migration_rows) <= 42, (
        f"After RWS-07, migration ceiling should be <=42. "
        f"Got {len(migration_rows)} rows."
    )
