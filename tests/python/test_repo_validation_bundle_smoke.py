"""Repo-wide validation bundle tests (RWS-09).

Verifies that the canonical repo-wide validation bundle:
  1. Builds without error and contains all expected sections
  2. Every promoted stable slice appears in the bundle
  3. Release slices have linked surfaces (no orphan release slices)
  4. Governance quality checks pass (no naked optional, no bare required)
  5. Bundle and release matrix agree on slice IDs
  6. --check mode passes
  7. Migration tracking is honest (matches repo_surface_matrix)
  8. Bundle uses the bundle-slot contract from RWS-01

RWS-09 ADR:
  "Produce one canonical repo-wide release truth."

Acceptance:
  "Every promoted stable slice appears in the canonical repo bundle."
  "Every release candidate emits the bundle."
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# Import the bundle builder directly
import sys
sys.path.insert(0, str(REPO))
from scripts.repo_validation_bundle import build_bundle, check_bundle


@pytest.fixture(scope="module")
def bundle() -> dict:
    return build_bundle(REPO)


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


# ── Bundle structure ─────────────────────────────────────────────────────


def test_bundle_schema_version(bundle: dict) -> None:
    assert bundle["schema_version"] == "nextstat.repo_validation_bundle.v1"


def test_bundle_has_required_sections(bundle: dict) -> None:
    required_keys = [
        "schema_version",
        "generated_at_utc",
        "summary",
        "per_domain",
        "release_slice_coverage",
        "governance_quality",
        "migration",
        "bundle_slot_contract",
    ]
    missing = [k for k in required_keys if k not in bundle]
    assert not missing, f"Bundle missing sections: {missing}"


def test_bundle_summary_counts(bundle: dict, repo_matrix: dict) -> None:
    """Bundle summary must match repo_surface_matrix_v1.json."""
    expected_total = len(repo_matrix["surfaces"])
    assert bundle["summary"]["total_surfaces"] == expected_total


# ── Release slice coverage ───────────────────────────────────────────────


def test_all_release_slices_in_bundle(bundle: dict, release_matrix: dict) -> None:
    """Every release slice must appear in the bundle."""
    bundle_slice_ids = {
        rs["release_surface_id"]
        for rs in bundle["release_slice_coverage"]
    }
    release_ids = {r["id"] for r in release_matrix["surfaces"]}
    missing = release_ids - bundle_slice_ids
    assert not missing, f"Release slices missing from bundle: {missing}"


def test_no_orphan_release_slices(bundle: dict) -> None:
    """Every release slice must have at least one linked surface."""
    orphans = [
        rs["release_surface_id"]
        for rs in bundle["release_slice_coverage"]
        if rs["linked_surface_count"] == 0
    ]
    assert not orphans, f"Release slices with zero linked surfaces: {orphans}"


def test_required_release_slices_have_gates(bundle: dict) -> None:
    """Required release slices must have make_target or workflow_job."""
    missing_gate = [
        rs["release_surface_id"]
        for rs in bundle["release_slice_coverage"]
        if rs["required_for_release"]
        and not rs["make_target"]
        and not rs["workflow_job"]
    ]
    assert not missing_gate, (
        f"Required release slices without make_target or workflow_job: "
        f"{missing_gate}"
    )


# ── Governance quality ───────────────────────────────────────────────────


def test_governance_quality_ok(bundle: dict) -> None:
    """Governance quality must pass (no bare governed, no naked optional)."""
    gq = bundle["governance_quality"]
    assert gq["quality_ok"], (
        f"Governance quality FAIL: "
        f"bare={gq['bare_governed_surfaces']}, "
        f"required_no_ref={gq['required_without_release_ref']}, "
        f"optional_no_support={gq['optional_without_support_or_gate']}"
    )


def test_governance_multi_ref_ratio(bundle: dict) -> None:
    """At least 50% of governed surfaces should have 2+ refs."""
    gq = bundle["governance_quality"]
    assert gq["multi_ref_ratio"] >= 0.50, (
        f"Multi-ref ratio {gq['multi_ref_ratio']:.0%} < 50%. "
        f"{gq['multi_ref_count']}/{gq['governed_total']}"
    )


# ── Migration tracking ──────────────────────────────────────────────────


def test_migration_matches_repo_matrix(bundle: dict, repo_matrix: dict) -> None:
    """Migration count in bundle must match repo_surface_matrix."""
    expected = sum(
        1 for s in repo_matrix["surfaces"]
        if s["public_status"] == "stable"
        and s["release_status"] == "not_release_governed"
    )
    assert bundle["migration"]["total"] == expected, (
        f"Migration count mismatch: bundle={bundle['migration']['total']}, "
        f"matrix={expected}"
    )


def test_migration_surfaces_listed(bundle: dict) -> None:
    """Migration section must enumerate all migrating surface IDs."""
    assert len(bundle["migration"]["surfaces"]) == bundle["migration"]["total"]


def test_migration_ceiling_after_rws08(bundle: dict) -> None:
    """After RWS-08, migration should be <=36."""
    assert bundle["migration"]["total"] <= 36, (
        f"Migration ceiling exceeded: {bundle['migration']['total']} > 36"
    )


# ── Bundle-slot contract from RWS-01 ────────────────────────────────────


def test_bundle_slot_contract_present(bundle: dict) -> None:
    """Bundle must include the bundle-slot contract from RWS-01."""
    bsc = bundle.get("bundle_slot_contract", {})
    assert bsc.get("version") == "v1", (
        f"Expected bundle_slot_contract.version='v1', got {bsc.get('version')!r}"
    )


def test_bundle_slot_contract_has_validation_slots(bundle: dict) -> None:
    """Bundle-slot contract must list at least hep_validation_bundle."""
    bsc = bundle.get("bundle_slot_contract", {})
    slots = bsc.get("validation_bundle_slots", [])
    assert "hep_validation_bundle" in slots, (
        f"hep_validation_bundle missing from bundle_slot_contract.validation_bundle_slots: "
        f"{slots}"
    )


# ── --check mode ─────────────────────────────────────────────────────────


def test_check_mode_passes() -> None:
    """--check mode must return OK."""
    ok, message = check_bundle(REPO)
    assert ok, f"check_bundle failed: {message}"


# ── Per-domain fully governed ────────────────────────────────────────────


def test_pharma_fully_governed(bundle: dict) -> None:
    """Pharma domain must be fully governed (zero migration)."""
    pharma = bundle["per_domain"].get("pharma", {})
    assert pharma.get("fully_governed"), (
        f"Pharma not fully governed: migration={pharma.get('migration')}"
    )


def test_bayesian_fully_governed(bundle: dict) -> None:
    """Bayesian domain must be fully governed (zero migration)."""
    bayesian = bundle["per_domain"].get("bayesian", {})
    assert bayesian.get("fully_governed"), (
        f"Bayesian not fully governed: migration={bayesian.get('migration')}"
    )


def test_platform_fully_governed(bundle: dict) -> None:
    """Platform domain must be fully governed (zero migration)."""
    platform = bundle["per_domain"].get("platform", {})
    assert platform.get("fully_governed"), (
        f"Platform not fully governed: migration={platform.get('migration')}"
    )
