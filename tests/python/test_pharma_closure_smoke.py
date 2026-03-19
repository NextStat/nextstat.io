"""Pharma public stable surface normalization tests (RWS-06).

Verifies that the pharma domain has machine-readable maturity governance:
  1. M15 doc surfaces promoted from migration → optional, linked to release slice
  2. PK/NLME runtime has gate_ref and support_contract_ref
  3. PK/NLME tutorial surfaces promoted from migration
  4. Survival runtime has gate_ref and support_contract_ref
  5. Survival tutorial promoted from migration
  6. No pharma surfaces remain in migration-only state after closure
  7. Global migration ceiling decreases after RWS-06

RWS-06 ADR:
  "Give pharma the same machine-readable maturity model HEP already has."

Pharma slices:
  - m15_reporting: release-required, fully governed (CI, gate, validation-pack)
  - pk_nlme: stable-optional (FOCE/SAEM core + tutorials)
  - survival: stable-optional (survival core + tutorial)
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
def pharma_surfaces(repo_matrix: dict) -> list[dict]:
    return [s for s in repo_matrix["surfaces"] if s["domain"] == "pharma"]


@pytest.fixture(scope="module")
def m15_surfaces(pharma_surfaces: list[dict]) -> list[dict]:
    return [s for s in pharma_surfaces if s["owner_slice"] == "m15_reporting"]


@pytest.fixture(scope="module")
def pk_nlme_surfaces(pharma_surfaces: list[dict]) -> list[dict]:
    return [s for s in pharma_surfaces if s["owner_slice"] == "pk_nlme"]


@pytest.fixture(scope="module")
def survival_surfaces(pharma_surfaces: list[dict]) -> list[dict]:
    return [s for s in pharma_surfaces if s["owner_slice"] == "survival"]


# ── M15 doc surfaces promoted ─────────────────────────────────────────


def test_m15_doc_surfaces_promoted(m15_surfaces: list[dict]) -> None:
    """M15 doc surfaces should be promoted to required (same as runtime)."""
    docs = [
        s for s in m15_surfaces
        if s["surface_kind"] == "documentation"
        and s["release_status"] == "not_release_governed"
        and s["public_status"] == "stable"
    ]
    assert not docs, (
        f"M15 doc surfaces still in migration: "
        f"{[s['surface_id'] for s in docs]}"
    )


def test_m15_doc_surfaces_link_release(m15_surfaces: list[dict]) -> None:
    """M15 doc surfaces should cross-link to m15_reporting_stable_surface."""
    docs = [s for s in m15_surfaces if s["surface_kind"] == "documentation"]
    for s in docs:
        assert s.get("release_surface_ref") == "m15_reporting_stable_surface", (
            f"{s['surface_id']}: should link to m15_reporting_stable_surface"
        )


# ── PK/NLME runtime governance ────────────────────────────────────────


def test_pk_nlme_runtime_surface_exists(pk_nlme_surfaces: list[dict]) -> None:
    """PK/NLME has a Python API (FOCE/SAEM) — must have a runtime row."""
    runtime = [s for s in pk_nlme_surfaces if s["surface_kind"] == "runtime"]
    assert len(runtime) >= 1, (
        "pharma.pk_nlme runtime surface missing from repo_surface_matrix_v1.json"
    )


def test_pk_nlme_runtime_has_gate_ref(pk_nlme_surfaces: list[dict]) -> None:
    runtime = [s for s in pk_nlme_surfaces if s["surface_kind"] == "runtime"]
    for s in runtime:
        assert s.get("gate_ref"), (
            f"{s['surface_id']}: runtime surface must have gate_ref"
        )


def test_pk_nlme_runtime_has_support_contract_ref(pk_nlme_surfaces: list[dict]) -> None:
    runtime = [s for s in pk_nlme_surfaces if s["surface_kind"] == "runtime"]
    for s in runtime:
        assert s.get("support_contract_ref"), (
            f"{s['surface_id']}: runtime surface must have support_contract_ref"
        )


def test_pk_nlme_runtime_promoted(pk_nlme_surfaces: list[dict]) -> None:
    """PK/NLME runtime should be promoted from not_release_governed."""
    runtime = [
        s for s in pk_nlme_surfaces
        if s["surface_kind"] == "runtime"
        and s["release_status"] == "not_release_governed"
    ]
    assert not runtime, (
        f"PK/NLME runtime still in migration: "
        f"{[s['surface_id'] for s in runtime]}"
    )


# ── PK/NLME tutorial surfaces promoted ────────────────────────────────


def test_pk_nlme_tutorials_promoted(pk_nlme_surfaces: list[dict]) -> None:
    """PK/NLME tutorials should be promoted from not_release_governed."""
    tutorials = [
        s for s in pk_nlme_surfaces
        if s["surface_kind"] == "tutorial"
        and s["release_status"] == "not_release_governed"
        and s["public_status"] == "stable"
    ]
    assert not tutorials, (
        f"PK/NLME tutorials still in migration: "
        f"{[s['surface_id'] for s in tutorials]}"
    )


def test_pk_nlme_has_minimum_surfaces(pk_nlme_surfaces: list[dict]) -> None:
    """PK/NLME: >=6 surfaces (1 runtime + 5 tutorials)."""
    assert len(pk_nlme_surfaces) >= 6, (
        f"Expected >=6 pk_nlme surfaces, got {len(pk_nlme_surfaces)}: "
        f"{[s['surface_id'] for s in pk_nlme_surfaces]}"
    )


# ── Survival runtime governance ───────────────────────────────────────


def test_survival_runtime_surface_exists(survival_surfaces: list[dict]) -> None:
    """Survival has a Python API — must have a runtime row."""
    runtime = [s for s in survival_surfaces if s["surface_kind"] == "runtime"]
    assert len(runtime) >= 1, (
        "pharma.survival runtime surface missing from repo_surface_matrix_v1.json"
    )


def test_survival_runtime_has_gate_ref(survival_surfaces: list[dict]) -> None:
    runtime = [s for s in survival_surfaces if s["surface_kind"] == "runtime"]
    for s in runtime:
        assert s.get("gate_ref"), (
            f"{s['surface_id']}: runtime surface must have gate_ref"
        )


def test_survival_runtime_has_support_contract_ref(survival_surfaces: list[dict]) -> None:
    runtime = [s for s in survival_surfaces if s["surface_kind"] == "runtime"]
    for s in runtime:
        assert s.get("support_contract_ref"), (
            f"{s['surface_id']}: runtime surface must have support_contract_ref"
        )


def test_survival_runtime_promoted(survival_surfaces: list[dict]) -> None:
    """Survival runtime should be promoted from not_release_governed."""
    runtime = [
        s for s in survival_surfaces
        if s["surface_kind"] == "runtime"
        and s["release_status"] == "not_release_governed"
    ]
    assert not runtime, (
        f"Survival runtime still in migration: "
        f"{[s['surface_id'] for s in runtime]}"
    )


# ── Survival tutorial promoted ────────────────────────────────────────


def test_survival_tutorial_promoted(survival_surfaces: list[dict]) -> None:
    """Survival tutorial should be promoted from not_release_governed."""
    tutorials = [
        s for s in survival_surfaces
        if s["surface_kind"] == "tutorial"
        and s["release_status"] == "not_release_governed"
        and s["public_status"] == "stable"
    ]
    assert not tutorials, (
        f"Survival tutorials still in migration: "
        f"{[s['surface_id'] for s in tutorials]}"
    )


def test_survival_has_minimum_surfaces(survival_surfaces: list[dict]) -> None:
    """Survival: >=2 surfaces (1 runtime + 1 tutorial)."""
    assert len(survival_surfaces) >= 2, (
        f"Expected >=2 survival surfaces, got {len(survival_surfaces)}: "
        f"{[s['surface_id'] for s in survival_surfaces]}"
    )


# ── No pharma migration rows after closure ────────────────────────────


def test_no_pharma_migration_rows(pharma_surfaces: list[dict]) -> None:
    """After RWS-06, no pharma surface should be stable+not_release_governed."""
    migration = [
        s["surface_id"]
        for s in pharma_surfaces
        if s["public_status"] == "stable"
        and s["release_status"] == "not_release_governed"
    ]
    assert not migration, (
        f"After RWS-06, pharma migration rows must be zero. "
        f"Still in migration: {migration}"
    )


# ── Global migration ceiling decreases ────────────────────────────────


def test_global_migration_ceiling_after_rws06(repo_matrix: dict) -> None:
    """RWS-06 should decrease migration count (was <=73 after RWS-05)."""
    migration_rows = [
        s
        for s in repo_matrix["surfaces"]
        if s["public_status"] == "stable"
        and s["release_status"] == "not_release_governed"
    ]
    # Was 73, RWS-06 promotes 10 pharma surfaces from migration
    assert len(migration_rows) <= 63, (
        f"After RWS-06, migration ceiling should be <=63. "
        f"Got {len(migration_rows)} rows."
    )
