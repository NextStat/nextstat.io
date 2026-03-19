"""Ads domain internal governance classification tests (RWS-03).

Ads is an internal-only domain — NOT public-facing. RWS-03 ensures:
  1. All ads surfaces are classified as ``public_status: internal``
  2. Ads surfaces with engineering governance (CI, gates, baselines)
     are properly registered in the repo matrix with correct refs
  3. No ads surface appears in the public release contract
  4. No ads surface claims public stable status

Surfaces with internal engineering governance:
  - ads-timeseries (CI workflow, gate script, support matrix, baselines)
  - ads-variance-reduction (CI workflow, gate script, acceptance doc, baselines)

Surface without governance infrastructure:
  - ads-churn (tutorial only, no gate/workflow/baseline)
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
def ads_surfaces(repo_matrix: dict) -> list[dict]:
    return [s for s in repo_matrix["surfaces"] if s["domain"] == "ads"]


@pytest.fixture(scope="module")
def release_ids(release_matrix: dict) -> set[str]:
    return {s["id"] for s in release_matrix["surfaces"]}


# ── All ads surfaces must be internal ────────────────────────────────────


def test_all_ads_surfaces_are_internal(ads_surfaces: list[dict]) -> None:
    """Ads is internal-only — no surface should be stable or research."""
    non_internal = [
        f"{s['surface_id']} (public_status={s['public_status']!r})"
        for s in ads_surfaces
        if s["public_status"] != "internal"
    ]
    assert not non_internal, (
        f"Ads is internal-only but {len(non_internal)} surface(s) have "
        f"non-internal public_status:\n" + "\n".join(f"  - {n}" for n in non_internal)
    )


def test_no_ads_surface_in_release_matrix(
    ads_surfaces: list[dict], release_ids: set[str]
) -> None:
    """Internal ads surfaces should not appear in the public release contract."""
    in_release = [
        s["surface_id"]
        for s in ads_surfaces
        if s.get("release_surface_ref") and s["release_surface_ref"] in release_ids
    ]
    assert not in_release, (
        f"Internal ads surfaces should not be in release matrix: {in_release}"
    )


# ── Ads-timeseries: must exist with internal governance refs ─────────────


def test_ads_timeseries_runtime_surface_exists(ads_surfaces: list[dict]) -> None:
    ts_ids = [s["surface_id"] for s in ads_surfaces if "timeseries" in s["owner_slice"]]
    assert len(ts_ids) >= 1, (
        "ads-timeseries runtime surface missing from repo_surface_matrix_v1.json"
    )


def test_ads_timeseries_has_gate_ref(ads_surfaces: list[dict]) -> None:
    ts = [s for s in ads_surfaces if "timeseries" in s["owner_slice"] and s["surface_kind"] == "runtime"]
    for s in ts:
        assert s.get("gate_ref"), (
            f"{s['surface_id']}: has a gate script but no gate_ref"
        )


def test_ads_timeseries_has_workflow_ref(ads_surfaces: list[dict]) -> None:
    ts = [s for s in ads_surfaces if "timeseries" in s["owner_slice"] and s["surface_kind"] == "runtime"]
    for s in ts:
        assert s.get("workflow_ref"), (
            f"{s['surface_id']}: has a CI workflow but no workflow_ref"
        )


def test_ads_timeseries_has_support_contract_ref(ads_surfaces: list[dict]) -> None:
    ts = [s for s in ads_surfaces if "timeseries" in s["owner_slice"] and s["surface_kind"] == "runtime"]
    for s in ts:
        assert s.get("support_contract_ref"), (
            f"{s['surface_id']}: has a support matrix doc but no support_contract_ref"
        )


# ── Ads-variance-reduction: must exist with internal governance refs ─────


def test_ads_variance_reduction_runtime_surface_exists(ads_surfaces: list[dict]) -> None:
    vr_ids = [s["surface_id"] for s in ads_surfaces if "variance_reduction" in s["owner_slice"]]
    assert len(vr_ids) >= 1, (
        "ads-variance-reduction runtime surface missing from repo_surface_matrix_v1.json"
    )


def test_ads_variance_reduction_has_gate_ref(ads_surfaces: list[dict]) -> None:
    vr = [s for s in ads_surfaces if "variance_reduction" in s["owner_slice"] and s["surface_kind"] == "runtime"]
    for s in vr:
        assert s.get("gate_ref"), (
            f"{s['surface_id']}: has a gate script but no gate_ref"
        )


def test_ads_variance_reduction_has_workflow_ref(ads_surfaces: list[dict]) -> None:
    vr = [s for s in ads_surfaces if "variance_reduction" in s["owner_slice"] and s["surface_kind"] == "runtime"]
    for s in vr:
        assert s.get("workflow_ref"), (
            f"{s['surface_id']}: has a CI workflow but no workflow_ref"
        )


def test_ads_variance_reduction_has_support_contract_ref(ads_surfaces: list[dict]) -> None:
    vr = [s for s in ads_surfaces if "variance_reduction" in s["owner_slice"] and s["surface_kind"] == "runtime"]
    for s in vr:
        assert s.get("support_contract_ref"), (
            f"{s['surface_id']}: has acceptance docs but no support_contract_ref"
        )


# ── Ads domain completeness ─────────────────────────────────────────────


def test_ads_domain_has_minimum_surfaces(ads_surfaces: list[dict]) -> None:
    """After closure, ads should have at least 4 surfaces (2 runtime + churn runtime + tutorial)."""
    assert len(ads_surfaces) >= 4, (
        f"Expected >=4 ads surfaces after closure, got {len(ads_surfaces)}: "
        f"{[s['surface_id'] for s in ads_surfaces]}"
    )


# ── No ads migration rows remain ────────────────────────────────────────


def test_no_ads_stable_migration_rows(ads_surfaces: list[dict]) -> None:
    """No ads surface should be in stable+not_release_governed migration state."""
    migration = [
        s["surface_id"]
        for s in ads_surfaces
        if s["public_status"] == "stable"
        and s["release_status"] == "not_release_governed"
    ]
    assert not migration, (
        f"Ads surfaces still in migration (stable+not_release_governed): {migration}"
    )
