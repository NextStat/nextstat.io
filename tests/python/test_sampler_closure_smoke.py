"""NUTS/WALNUTS/MAMS sampler stable closure tests (RWS-05).

Verifies that sampler surfaces are properly governed:
  1. Runtime surface rows exist for NUTS, WALNUTS, MAMS (all have Python API)
  2. Runtime surfaces have gate_ref and support_contract_ref
  3. No sampler surfaces remain in migration-only state after closure
  4. Sampler documentation surfaces are promoted from migration

RWS-05 ADR:
  "Convert sampler stable claims into governed product slices."

Governance model: optional repo-bundle-only (no dedicated CI workflow yet).
All three samplers are shipped public stable API via nextstat.sample().
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
def sampler_surfaces(repo_matrix: dict) -> list[dict]:
    return [
        s for s in repo_matrix["surfaces"]
        if s["domain"] == "bayesian"
        and s["owner_slice"] in ("nuts", "walnuts", "mams")
    ]


# ── Runtime surfaces must exist ──────────────────────────────────────────


def test_nuts_runtime_surface_exists(sampler_surfaces: list[dict]) -> None:
    """NUTS is a shipped public API — must have a runtime row."""
    nuts_rt = [
        s for s in sampler_surfaces
        if s["owner_slice"] == "nuts" and s["surface_kind"] == "runtime"
    ]
    assert len(nuts_rt) >= 1, (
        "NUTS runtime surface missing from repo_surface_matrix_v1.json. "
        "sample(method='nuts') is a shipped public API."
    )


def test_walnuts_runtime_surface_exists(sampler_surfaces: list[dict]) -> None:
    """WALNUTS is a shipped public API — must have a runtime row."""
    walnuts_rt = [
        s for s in sampler_surfaces
        if s["owner_slice"] == "walnuts" and s["surface_kind"] == "runtime"
    ]
    assert len(walnuts_rt) >= 1, (
        "WALNUTS runtime surface missing from repo_surface_matrix_v1.json. "
        "sample(method='walnuts') is a shipped public API."
    )


def test_mams_runtime_surface_exists(sampler_surfaces: list[dict]) -> None:
    """MAMS is a shipped public API — must have a runtime row."""
    mams_rt = [
        s for s in sampler_surfaces
        if s["owner_slice"] == "mams" and s["surface_kind"] == "runtime"
    ]
    assert len(mams_rt) >= 1, (
        "MAMS runtime surface missing from repo_surface_matrix_v1.json. "
        "sample(method='mams') is a shipped public API."
    )


# ── Runtime surfaces have governance refs ────────────────────────────────


def test_sampler_runtime_surfaces_have_gate_ref(sampler_surfaces: list[dict]) -> None:
    runtime = [s for s in sampler_surfaces if s["surface_kind"] == "runtime"]
    missing = [s["surface_id"] for s in runtime if not s.get("gate_ref")]
    assert not missing, (
        f"Sampler runtime surfaces missing gate_ref: {missing}"
    )


def test_sampler_runtime_surfaces_have_support_contract_ref(sampler_surfaces: list[dict]) -> None:
    runtime = [s for s in sampler_surfaces if s["surface_kind"] == "runtime"]
    missing = [s["surface_id"] for s in runtime if not s.get("support_contract_ref")]
    assert not missing, (
        f"Sampler runtime surfaces missing support_contract_ref: {missing}"
    )


# ── No migration-only sampler rows after closure ────────────────────────


def test_no_sampler_migration_rows(sampler_surfaces: list[dict]) -> None:
    """After RWS-05, no sampler surface should be stable+not_release_governed."""
    migration = [
        s["surface_id"]
        for s in sampler_surfaces
        if s["public_status"] == "stable"
        and s["release_status"] == "not_release_governed"
    ]
    assert not migration, (
        f"After RWS-05, sampler migration rows must be zero. "
        f"Still in migration: {migration}"
    )


# ── Documentation surfaces promoted ─────────────────────────────────────


def test_sampler_doc_surfaces_promoted(sampler_surfaces: list[dict]) -> None:
    """NUTS/WALNUTS doc surfaces should be promoted from not_release_governed."""
    docs = [
        s for s in sampler_surfaces
        if s["surface_kind"] == "documentation"
        and s["release_status"] == "not_release_governed"
        and s["public_status"] == "stable"
    ]
    assert not docs, (
        f"Sampler doc surfaces still in migration: "
        f"{[s['surface_id'] for s in docs]}"
    )


# ── Minimum surface count ───────────────────────────────────────────────


def test_sampler_domain_has_minimum_surfaces(sampler_surfaces: list[dict]) -> None:
    """After closure: >=5 sampler surfaces (3 runtime + 2 docs)."""
    assert len(sampler_surfaces) >= 5, (
        f"Expected >=5 sampler surfaces, got {len(sampler_surfaces)}: "
        f"{[s['surface_id'] for s in sampler_surfaces]}"
    )


# ── Global migration ceiling decreases ──────────────────────────────────


def test_global_migration_ceiling_after_rws05(repo_matrix: dict) -> None:
    """RWS-05 should decrease migration count (was <=75 after RWS-04)."""
    migration_rows = [
        s
        for s in repo_matrix["surfaces"]
        if s["public_status"] == "stable"
        and s["release_status"] == "not_release_governed"
    ]
    # Was 75, RWS-05 promotes 2 sampler doc surfaces
    assert len(migration_rows) <= 73, (
        f"After RWS-05, migration ceiling should be <=73. "
        f"Got {len(migration_rows)} rows."
    )
