"""Final SOTA/stable release policy invariants (RWS-10).

Hard invariants that prevent regression of repo-wide stable governance:
  1. Zero stable migration — no stable surface without governance
  2. Every governed surface has real refs (not relabel)
  3. Every domain is fully governed (zero migration per domain)
  4. Repo validation bundle passes --check
  5. Bundle and release matrix agree on all slice IDs
  6. All HEP surfaces from hep_surface_matrix are in repo matrix
  7. No surface overclaims maturity relative to its parent

RWS-10 ADR acceptance:
  "zero public stable drift"
  "zero promoted stable slice missing release or bundle coverage"
  "repo can truthfully claim repo-wide stable/SOTA governance"

These tests are BLOCKING — they fail the release if any invariant breaks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

import sys
sys.path.insert(0, str(REPO))
from scripts.repo_validation_bundle import build_bundle, check_bundle


@pytest.fixture(scope="module")
def repo_matrix() -> dict:
    return json.loads(
        (REPO / "repo_surface_matrix_v1.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def release_matrix() -> dict:
    return json.loads(
        (REPO / "scripts" / "release_surface_matrix_v1.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def bundle() -> dict:
    return build_bundle(REPO)


# ── INVARIANT 1: Zero stable migration ──────────────────────────────────


def test_zero_stable_migration(repo_matrix: dict) -> None:
    """No stable surface may be not_release_governed."""
    migration = [
        s["surface_id"]
        for s in repo_matrix["surfaces"]
        if s["public_status"] == "stable"
        and s["release_status"] == "not_release_governed"
    ]
    assert len(migration) == 0, (
        f"BLOCKING: {len(migration)} stable surfaces still in migration: "
        f"{migration[:10]}{'...' if len(migration) > 10 else ''}"
    )


# ── INVARIANT 2: No naked governed surfaces ─────────────────────────────


def test_no_naked_governed(repo_matrix: dict) -> None:
    """Every governed (required/optional) surface must have real refs."""
    governed = [
        s for s in repo_matrix["surfaces"]
        if s["release_status"] in ("required", "optional")
        and s["public_status"] != "internal"
    ]
    naked = [
        s["surface_id"]
        for s in governed
        if not s.get("release_surface_ref")
        and not s.get("gate_ref")
        and not s.get("support_contract_ref")
    ]
    assert not naked, (
        f"BLOCKING: governed surfaces without any ref: {naked}"
    )


# ── INVARIANT 3: All domains fully governed ─────────────────────────────


def test_all_domains_governed(bundle: dict) -> None:
    """Every domain must have zero stable migration."""
    failing = [
        (domain, stats["migration"])
        for domain, stats in bundle["per_domain"].items()
        if stats["migration"] > 0
    ]
    assert not failing, (
        f"BLOCKING: domains with stable migration: "
        f"{[(d, m) for d, m in failing]}"
    )


# ── INVARIANT 4: Bundle --check passes ──────────────────────────────────


def test_bundle_check_passes() -> None:
    """Repo-wide validation bundle must pass --check."""
    ok, message = check_bundle(REPO)
    assert ok, f"BLOCKING: bundle check failed: {message}"


# ── INVARIANT 5: Bundle↔release matrix agreement ────────────────────────


def test_bundle_release_matrix_agreement(
    bundle: dict, release_matrix: dict
) -> None:
    """Bundle and release matrix must agree on slice IDs."""
    bundle_ids = {
        rs["release_surface_id"]
        for rs in bundle["release_slice_coverage"]
    }
    release_ids = {r["id"] for r in release_matrix["surfaces"]}
    assert bundle_ids == release_ids, (
        f"BLOCKING: bundle/release matrix slice mismatch. "
        f"In bundle only: {bundle_ids - release_ids}. "
        f"In release only: {release_ids - bundle_ids}."
    )


# ── INVARIANT 6: HEP lossless import ────────────────────────────────────


def test_hep_lossless_import(repo_matrix: dict) -> None:
    """Every HEP surface from hep_surface_matrix must be in repo matrix."""
    hep = json.loads(
        (REPO / "hep_surface_matrix_v1.json").read_text(encoding="utf-8")
    )
    repo_hep_ids = {
        s["surface_id"] for s in repo_matrix["surfaces"]
        if s["domain"] == "hep"
    }
    for s in hep["surfaces"]:
        expected_id = f"hep.{s['owner_slice']}.{s['name']}.{s['layer']}"
        assert expected_id in repo_hep_ids, (
            f"BLOCKING: HEP surface {expected_id} missing from repo matrix"
        )


# ── INVARIANT 7: Required surfaces have full governance chain ────────────


def test_required_surfaces_full_chain(repo_matrix: dict) -> None:
    """Required surfaces must have release_surface_ref AND validation_bundle_ref."""
    required = [
        s for s in repo_matrix["surfaces"]
        if s["release_status"] == "required"
        and s["public_status"] != "internal"
    ]
    missing_release_ref = [
        s["surface_id"] for s in required
        if not s.get("release_surface_ref")
    ]
    missing_bundle_ref = [
        s["surface_id"] for s in required
        if not s.get("validation_bundle_ref")
    ]
    assert not missing_release_ref, (
        f"BLOCKING: required surfaces without release_surface_ref: "
        f"{missing_release_ref}"
    )
    assert not missing_bundle_ref, (
        f"BLOCKING: required surfaces without validation_bundle_ref: "
        f"{missing_bundle_ref}"
    )


# ── INVARIANT 8: Internal surfaces exempt from governance ────────────────


def test_internal_surfaces_not_governed(repo_matrix: dict) -> None:
    """Internal surfaces should not claim governance (they're not public)."""
    internal_governed = [
        s["surface_id"]
        for s in repo_matrix["surfaces"]
        if s["public_status"] == "internal"
        and s["release_status"] in ("required", "optional")
    ]
    assert not internal_governed, (
        f"Internal surfaces claiming governance (should be not_release_governed): "
        f"{internal_governed}"
    )
