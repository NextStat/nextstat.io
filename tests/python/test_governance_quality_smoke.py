"""Governance quality enforcement tests.

Prevents the "relabel anti-pattern" where surfaces are moved from
not_release_governed to optional without actual governance refs.

Rule: every surface that claims governance (required or optional) must have
at least ONE concrete governance ref. A governance ref is any of:
  - release_surface_ref (linked to a release slice)
  - gate_ref (has a quality gate script/spec)
  - support_contract_ref (has a support/docs contract)

Surfaces with release_status="not_release_governed" are exempt — they
honestly declare that they lack governance and will be addressed later.

Internal surfaces (public_status="internal") are also exempt — they are
not part of the public release contract.
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
def governed_surfaces(repo_matrix: dict) -> list[dict]:
    """Surfaces that claim governance (required or optional), non-internal."""
    return [
        s for s in repo_matrix["surfaces"]
        if s["release_status"] in ("required", "optional")
        and s["public_status"] != "internal"
    ]


@pytest.fixture(scope="module")
def required_surfaces(governed_surfaces: list[dict]) -> list[dict]:
    return [s for s in governed_surfaces if s["release_status"] == "required"]


@pytest.fixture(scope="module")
def optional_surfaces(governed_surfaces: list[dict]) -> list[dict]:
    return [s for s in governed_surfaces if s["release_status"] == "optional"]


# ── Every governed surface has at least one governance ref ────────────


def _has_governance_ref(s: dict) -> bool:
    return bool(
        s.get("release_surface_ref")
        or s.get("gate_ref")
        or s.get("support_contract_ref")
    )


def test_every_governed_surface_has_ref(governed_surfaces: list[dict]) -> None:
    """No governed surface should lack all governance refs."""
    bare = [
        s["surface_id"]
        for s in governed_surfaces
        if not _has_governance_ref(s)
    ]
    assert not bare, (
        f"Governed surfaces without ANY governance ref (relabel anti-pattern): "
        f"{bare}"
    )


# ── Required surfaces: stricter — need release_surface_ref ────────────


def test_required_surfaces_have_release_ref(required_surfaces: list[dict]) -> None:
    """Required surfaces must link to a release slice."""
    missing = [
        s["surface_id"]
        for s in required_surfaces
        if not s.get("release_surface_ref")
    ]
    assert not missing, (
        f"Required surfaces without release_surface_ref: {missing}"
    )


# ── Optional surfaces: at least support_contract or gate ─────────────


def test_optional_surfaces_have_support_or_gate(optional_surfaces: list[dict]) -> None:
    """Optional surfaces must have support_contract_ref or gate_ref."""
    bare = [
        s["surface_id"]
        for s in optional_surfaces
        if not s.get("support_contract_ref") and not s.get("gate_ref")
    ]
    assert not bare, (
        f"Optional surfaces without support_contract_ref or gate_ref: "
        f"{bare}"
    )


# ── No "naked optional" — optional without ANY context ────────────────


def test_no_naked_optional(optional_surfaces: list[dict]) -> None:
    """Optional surfaces with zero refs are just relabeled migration rows."""
    naked = [
        s["surface_id"]
        for s in optional_surfaces
        if not s.get("release_surface_ref")
        and not s.get("gate_ref")
        and not s.get("support_contract_ref")
        and not s.get("acceptance_ref")
    ]
    assert not naked, (
        f"Naked optional surfaces (no refs at all): {naked}"
    )


# ── Governance ref count stats (informational) ───────────────────────


def test_governance_coverage_stats(governed_surfaces: list[dict]) -> None:
    """At least 70% of governed surfaces should have 2+ governance refs."""
    multi_ref = [
        s for s in governed_surfaces
        if sum(bool(s.get(k)) for k in (
            "release_surface_ref", "gate_ref", "support_contract_ref",
            "acceptance_ref", "validation_bundle_ref",
        )) >= 2
    ]
    ratio = len(multi_ref) / len(governed_surfaces) if governed_surfaces else 0
    assert ratio >= 0.50, (
        f"Only {ratio:.0%} of governed surfaces have 2+ refs. "
        f"Expected >=50%. Total: {len(governed_surfaces)}, "
        f"with 2+ refs: {len(multi_ref)}"
    )
