"""Bayesian design stable closure tests (RWS-04).

Verifies that the Bayesian design product surface is properly governed:
  1. A runtime surface row exists in the repo matrix (Python API is real)
  2. Documentation surfaces have acceptance_ref pointing to their spec docs
  3. The runtime surface has gate_ref and support_contract_ref
  4. A release slice exists in release_surface_matrix_v1.json (optional/manual)
  5. No bayes_design surfaces remain in migration-only state after closure

RWS-04 ADR:
  "Convert existing FDA Bayesian stable closure from narrative/internal truth
   into repo-machine truth."

Slice decision: ONE release slice (bayesian_design_stable_surface) covering
the runtime API + all documentation facets. Optional for release.
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
def release_matrix() -> dict:
    return json.loads(
        (REPO / "scripts" / "release_surface_matrix_v1.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="module")
def bayes_surfaces(repo_matrix: dict) -> list[dict]:
    return [
        s for s in repo_matrix["surfaces"]
        if s["domain"] == "bayesian" and s["owner_slice"] == "bayes_design"
    ]


@pytest.fixture(scope="module")
def release_ids(release_matrix: dict) -> set[str]:
    return {s["id"] for s in release_matrix["surfaces"]}


# ── Runtime surface must exist ───────────────────────────────────────────


def test_bayesian_design_runtime_surface_exists(bayes_surfaces: list[dict]) -> None:
    """bayes_design has a Python API (bayes_design.py) — must have a runtime row."""
    runtime = [s for s in bayes_surfaces if s["surface_kind"] == "runtime"]
    assert len(runtime) >= 1, (
        "bayesian.bayes_design runtime surface missing from repo_surface_matrix_v1.json. "
        "bayes_design.py exists as a Python API."
    )


def test_bayesian_design_runtime_has_gate_ref(bayes_surfaces: list[dict]) -> None:
    runtime = [s for s in bayes_surfaces if s["surface_kind"] == "runtime"]
    for s in runtime:
        assert s.get("gate_ref"), (
            f"{s['surface_id']}: runtime surface must have a gate_ref"
        )


def test_bayesian_design_runtime_has_support_contract_ref(bayes_surfaces: list[dict]) -> None:
    runtime = [s for s in bayes_surfaces if s["surface_kind"] == "runtime"]
    for s in runtime:
        assert s.get("support_contract_ref"), (
            f"{s['surface_id']}: runtime surface must have support_contract_ref"
        )


# ── Documentation surfaces have acceptance refs ─────────────────────────


def test_bayesian_design_doc_surfaces_have_acceptance_ref(bayes_surfaces: list[dict]) -> None:
    """Each doc surface with an acceptance spec should reference it."""
    docs_needing_ref = [
        s for s in bayes_surfaces
        if s["surface_kind"] == "documentation"
        and "acceptance" in s["surface_id"]
    ]
    missing = [s["surface_id"] for s in docs_needing_ref if not s.get("acceptance_ref")]
    assert not missing, (
        f"Documentation surfaces missing acceptance_ref: {missing}"
    )


# ── Release matrix integration ──────────────────────────────────────────


def test_bayesian_design_in_release_matrix(release_ids: set[str]) -> None:
    assert "bayesian_design_stable_surface" in release_ids, (
        "bayesian_design should have a release slice in release_surface_matrix_v1.json"
    )


def test_bayesian_design_runtime_links_release(bayes_surfaces: list[dict]) -> None:
    runtime = [s for s in bayes_surfaces if s["surface_kind"] == "runtime"]
    for s in runtime:
        assert s.get("release_surface_ref") == "bayesian_design_stable_surface", (
            f"{s['surface_id']}: should cross-link to bayesian_design_stable_surface"
        )


# ── No migration-only rows after closure ─────────────────────────────────


def test_no_bayesian_design_migration_rows(bayes_surfaces: list[dict]) -> None:
    """After RWS-04, no bayes_design surface should be stable+not_release_governed."""
    migration = [
        s["surface_id"]
        for s in bayes_surfaces
        if s["public_status"] == "stable"
        and s["release_status"] == "not_release_governed"
    ]
    assert not migration, (
        f"After RWS-04, bayes_design migration rows must be zero. "
        f"Still in migration: {migration}"
    )


# ── Global migration ceiling decreases ──────────────────────────────────


def test_global_migration_ceiling_after_rws04(repo_matrix: dict) -> None:
    """RWS-04 should decrease migration count further (was ≤89 after RWS-03)."""
    migration_rows = [
        s
        for s in repo_matrix["surfaces"]
        if s["public_status"] == "stable"
        and s["release_status"] == "not_release_governed"
    ]
    # Was 83 migration rows, RWS-04 promotes 8 bayes_design docs from migration
    assert len(migration_rows) <= 75, (
        f"After RWS-04, migration ceiling should be ≤75. "
        f"Got {len(migration_rows)} rows."
    )
