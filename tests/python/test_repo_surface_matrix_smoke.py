"""Repo-wide surface matrix smoke tests.

Validates the canonical repo-wide surface inventory:
- Lossless HEP import (every HEP surface has a repo row)
- Release-surface cross-link (every release surface has a repo row)
- Schema correctness (required fields, valid enums, uniqueness)
- Bundle-slot contract (every promoted slice has a validation_bundle_ref)
- Documentation surface coverage (status: stable docs are tracked)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

VALID_DOMAINS = {"hep", "pharma", "bayesian", "ads", "general", "platform"}
VALID_SURFACE_KINDS = {"runtime", "documentation", "tutorial", "parity", "artifact"}
VALID_PUBLIC_STATUSES = {"stable", "research", "internal"}
VALID_RELEASE_STATUSES = {"required", "optional", "not_release_governed"}
VALID_LAYERS = {"cli", "python", "tool", "server", "docs"}

REQUIRED_FIELDS = {
    "surface_id",
    "domain",
    "owner_slice",
    "interface_layer",
    "surface_kind",
    "public_status",
    "release_status",
}


@pytest.fixture(scope="module")
def repo_matrix() -> dict:
    return json.loads(
        (REPO / "repo_surface_matrix_v1.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def hep_matrix() -> dict:
    return json.loads(
        (REPO / "hep_surface_matrix_v1.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def release_matrix() -> dict:
    return json.loads(
        (REPO / "scripts" / "release_surface_matrix_v1.json").read_text(
            encoding="utf-8"
        )
    )


# ── Schema tests ──────────────────────────────────────────────────────────


def test_schema_version(repo_matrix: dict) -> None:
    assert repo_matrix["schema_version"] == "nextstat.repo_surface_matrix.v1"


def test_has_summary(repo_matrix: dict) -> None:
    s = repo_matrix["summary"]
    assert s["total"] > 0
    assert "by_domain" in s
    assert "by_surface_kind" in s
    assert "by_public_status" in s
    assert "by_release_status" in s


def test_surfaces_is_nonempty(repo_matrix: dict) -> None:
    assert len(repo_matrix["surfaces"]) > 0


def test_every_surface_has_required_fields(repo_matrix: dict) -> None:
    for surf in repo_matrix["surfaces"]:
        missing = REQUIRED_FIELDS - set(surf.keys())
        assert not missing, f"Surface {surf.get('surface_id', '???')} missing: {missing}"


def test_every_surface_has_valid_domain(repo_matrix: dict) -> None:
    for surf in repo_matrix["surfaces"]:
        assert surf["domain"] in VALID_DOMAINS, (
            f"{surf['surface_id']}: invalid domain {surf['domain']!r}"
        )


def test_every_surface_has_valid_surface_kind(repo_matrix: dict) -> None:
    for surf in repo_matrix["surfaces"]:
        assert surf["surface_kind"] in VALID_SURFACE_KINDS, (
            f"{surf['surface_id']}: invalid surface_kind {surf['surface_kind']!r}"
        )


def test_every_surface_has_valid_public_status(repo_matrix: dict) -> None:
    for surf in repo_matrix["surfaces"]:
        assert surf["public_status"] in VALID_PUBLIC_STATUSES, (
            f"{surf['surface_id']}: invalid public_status {surf['public_status']!r}"
        )


def test_every_surface_has_valid_release_status(repo_matrix: dict) -> None:
    for surf in repo_matrix["surfaces"]:
        assert surf["release_status"] in VALID_RELEASE_STATUSES, (
            f"{surf['surface_id']}: invalid release_status {surf['release_status']!r}"
        )


def test_every_surface_has_valid_layer(repo_matrix: dict) -> None:
    for surf in repo_matrix["surfaces"]:
        assert surf["interface_layer"] in VALID_LAYERS, (
            f"{surf['surface_id']}: invalid layer {surf['interface_layer']!r}"
        )


def test_no_duplicate_surface_ids(repo_matrix: dict) -> None:
    ids = [s["surface_id"] for s in repo_matrix["surfaces"]]
    dupes = [sid for sid in ids if ids.count(sid) > 1]
    assert not dupes, f"Duplicate surface_id(s): {set(dupes)}"


# ── Lossless HEP import ──────────────────────────────────────────────────


def test_every_hep_surface_has_repo_row(
    repo_matrix: dict, hep_matrix: dict
) -> None:
    repo_ids = {s["surface_id"] for s in repo_matrix["surfaces"]}
    for hep_surf in hep_matrix["surfaces"]:
        expected_id = f"hep.{hep_surf['owner_slice']}.{hep_surf['name']}.{hep_surf['layer']}"
        assert expected_id in repo_ids, (
            f"HEP surface {hep_surf['name']} ({hep_surf['layer']}) missing from repo matrix"
        )


def test_hep_import_preserves_maturity(
    repo_matrix: dict, hep_matrix: dict
) -> None:
    repo_by_id = {s["surface_id"]: s for s in repo_matrix["surfaces"]}
    for hep_surf in hep_matrix["surfaces"]:
        sid = f"hep.{hep_surf['owner_slice']}.{hep_surf['name']}.{hep_surf['layer']}"
        repo_surf = repo_by_id.get(sid)
        assert repo_surf is not None, f"Missing {sid}"
        assert repo_surf["public_status"] == hep_surf["maturity_class"], (
            f"{sid}: public_status {repo_surf['public_status']!r} != "
            f"maturity_class {hep_surf['maturity_class']!r}"
        )


def test_hep_import_preserves_support_matrix_ref(
    repo_matrix: dict, hep_matrix: dict
) -> None:
    repo_by_id = {s["surface_id"]: s for s in repo_matrix["surfaces"]}
    for hep_surf in hep_matrix["surfaces"]:
        if not hep_surf.get("support_matrix_ref"):
            continue
        sid = f"hep.{hep_surf['owner_slice']}.{hep_surf['name']}.{hep_surf['layer']}"
        repo_surf = repo_by_id[sid]
        assert repo_surf["support_contract_ref"] == hep_surf["support_matrix_ref"]


# ── Release-surface cross-link ────────────────────────────────────────────


def test_every_release_surface_has_repo_row(
    repo_matrix: dict, release_matrix: dict
) -> None:
    release_refs = {
        s["release_surface_ref"]
        for s in repo_matrix["surfaces"]
        if s.get("release_surface_ref")
    }
    for rel_surf in release_matrix["surfaces"]:
        assert rel_surf["id"] in release_refs, (
            f"Release surface {rel_surf['id']!r} not referenced by any repo surface"
        )


def test_required_release_surfaces_are_required_in_repo(
    repo_matrix: dict, release_matrix: dict
) -> None:
    required_ids = {
        s["id"] for s in release_matrix["surfaces"] if s["required_for_release"]
    }
    for surf in repo_matrix["surfaces"]:
        ref = surf.get("release_surface_ref", "")
        if ref in required_ids:
            assert surf["release_status"] == "required", (
                f"{surf['surface_id']}: release_surface_ref={ref!r} is "
                f"required_for_release but release_status={surf['release_status']!r}"
            )


def test_optional_release_surfaces_are_optional_in_repo(
    repo_matrix: dict, release_matrix: dict
) -> None:
    optional_ids = {
        s["id"] for s in release_matrix["surfaces"] if not s["required_for_release"]
    }
    for surf in repo_matrix["surfaces"]:
        ref = surf.get("release_surface_ref", "")
        if ref in optional_ids:
            assert surf["release_status"] == "optional", (
                f"{surf['surface_id']}: release_surface_ref={ref!r} is "
                f"optional but release_status={surf['release_status']!r}"
            )


# ── Summary consistency ───────────────────────────────────────────────────


def test_summary_total_matches_surfaces(repo_matrix: dict) -> None:
    assert repo_matrix["summary"]["total"] == len(repo_matrix["surfaces"])


def test_summary_by_domain_sums_to_total(repo_matrix: dict) -> None:
    assert sum(repo_matrix["summary"]["by_domain"].values()) == repo_matrix["summary"]["total"]


def test_summary_by_public_status_sums_to_total(repo_matrix: dict) -> None:
    assert sum(repo_matrix["summary"]["by_public_status"].values()) == repo_matrix["summary"]["total"]


# ── Bundle-slot contract ──────────────────────────────────────────────────


def test_bundle_slot_contract_exists(repo_matrix: dict) -> None:
    bsc = repo_matrix["bundle_slot_contract"]
    assert bsc["version"] == "v1"
    assert "required_fields" in bsc
    assert "validation_bundle_slots" in bsc


def test_hep_bundle_slot_present(repo_matrix: dict) -> None:
    slots = repo_matrix["bundle_slot_contract"]["validation_bundle_slots"]
    assert "hep_validation_bundle" in slots


def test_every_required_surface_has_bundle_ref(repo_matrix: dict) -> None:
    for surf in repo_matrix["surfaces"]:
        if surf["release_status"] == "required":
            assert surf.get("validation_bundle_ref"), (
                f"{surf['surface_id']}: release_status=required but no validation_bundle_ref"
            )


# ── Documentation surface coverage ───────────────────────────────────────


def test_documentation_surfaces_exist(repo_matrix: dict) -> None:
    doc_surfaces = [
        s for s in repo_matrix["surfaces"] if s["surface_kind"] == "documentation"
    ]
    assert len(doc_surfaces) >= 5, (
        f"Expected ≥5 documentation surfaces, got {len(doc_surfaces)}"
    )


def test_tutorial_surfaces_exist(repo_matrix: dict) -> None:
    tut_surfaces = [
        s for s in repo_matrix["surfaces"] if s["surface_kind"] == "tutorial"
    ]
    assert len(tut_surfaces) >= 3, (
        f"Expected ≥3 tutorial surfaces, got {len(tut_surfaces)}"
    )


# ── Ref path validity ────────────────────────────────────────────────────


def test_support_contract_refs_are_valid_paths(repo_matrix: dict) -> None:
    for surf in repo_matrix["surfaces"]:
        ref = surf.get("support_contract_ref", "")
        if ref:
            assert (REPO / ref).exists(), (
                f"{surf['surface_id']}: support_contract_ref does not exist: {ref}"
            )


def test_acceptance_refs_are_valid_paths(repo_matrix: dict) -> None:
    for surf in repo_matrix["surfaces"]:
        ref = surf.get("acceptance_ref", "")
        if ref:
            assert (REPO / ref).exists(), (
                f"{surf['surface_id']}: acceptance_ref does not exist: {ref}"
            )


def test_gate_refs_are_valid_paths(repo_matrix: dict) -> None:
    for surf in repo_matrix["surfaces"]:
        ref = surf.get("gate_ref", "")
        if ref:
            assert (REPO / ref).exists(), (
                f"{surf['surface_id']}: gate_ref does not exist: {ref}"
            )


def test_workflow_refs_are_valid_paths(repo_matrix: dict) -> None:
    for surf in repo_matrix["surfaces"]:
        ref = surf.get("workflow_ref", "")
        if ref:
            assert (REPO / ref).exists(), (
                f"{surf['surface_id']}: workflow_ref does not exist: {ref}"
            )
