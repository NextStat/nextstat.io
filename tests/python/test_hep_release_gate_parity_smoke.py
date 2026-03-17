"""HEP-ADR-07: Release gate parity between HEP and release surface matrices.

Ensures that every promoted stable HEP slice (one with an existing support
matrix) is protected by a required-for-release gate in the release surface
matrix.  Without this check, a promoted HEP slice can silently lack CI-level
release protection.

Cross-references:
  - hep_surface_matrix_v1.json  (ADR-01: what is stable / research)
  - scripts/release_surface_matrix_v1.json  (what gates block a release)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HEP_MATRIX_PATH = REPO / "hep_surface_matrix_v1.json"
RELEASE_MATRIX_PATH = REPO / "scripts" / "release_surface_matrix_v1.json"

# Promoted HEP slices: those with existing support matrices whose stable
# surfaces are checked by test_promoted_stable_surfaces_have_support_matrix_ref.
PROMOTED_HEP_SLICES = {"simplified_likelihood", "gvm", "hepdata", "histfactory"}

# Known gaps: promoted HEP slices that have acceptance docs and pytest
# coverage but lack a dedicated CI-level release surface gate.
# Each entry documents the tracking ADR for promotion.
KNOWN_UNGATED_PROMOTED_SLICES: dict[str, str] = {}


@pytest.fixture(scope="module")
def hep_matrix():
    assert HEP_MATRIX_PATH.exists(), (
        f"hep_surface_matrix_v1.json not found at {HEP_MATRIX_PATH}. "
        "Run scripts/hep_surface_matrix.py to generate it."
    )
    with open(HEP_MATRIX_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def release_matrix():
    assert RELEASE_MATRIX_PATH.exists(), (
        f"release_surface_matrix_v1.json not found at {RELEASE_MATRIX_PATH}."
    )
    with open(RELEASE_MATRIX_PATH) as f:
        return json.load(f)


def _release_gate_ids(release_matrix: dict) -> set[str]:
    """Return IDs of required-for-release surfaces."""
    return {
        s["id"]
        for s in release_matrix["surfaces"]
        if s.get("required_for_release")
    }


def _hep_promoted_slices(hep_matrix: dict) -> set[str]:
    """Return owner_slices that have at least one stable surface with a
    support_matrix_ref (i.e. promoted slices)."""
    return {
        s["owner_slice"]
        for s in hep_matrix["surfaces"]
        if s.get("maturity_class") == "stable" and s.get("support_matrix_ref")
    }


def test_hep_surface_matrix_check_passes():
    """The HEP surface matrix generator --check must pass."""
    py = str(REPO / ".venv" / "bin" / "python")
    if not Path(py).exists():
        py = sys.executable
    result = subprocess.run(
        [py, str(REPO / "scripts" / "hep_surface_matrix.py"), "--check"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0, (
        f"hep_surface_matrix.py --check failed:\n{result.stdout}\n{result.stderr}"
    )


def test_every_promoted_hep_slice_has_release_gate(hep_matrix, release_matrix):
    """Every promoted HEP slice must have a corresponding required-for-release
    gate, unless it is in the known-ungated allowlist."""
    gate_ids = _release_gate_ids(release_matrix)
    promoted = _hep_promoted_slices(hep_matrix)

    # A promoted slice is "gated" if any release gate ID contains the slice name
    # (e.g. "simplified_likelihood" matches "simplified_likelihood_stable_surface").
    ungated = []
    for slice_name in sorted(promoted):
        if slice_name in KNOWN_UNGATED_PROMOTED_SLICES:
            continue
        has_gate = any(slice_name in gid for gid in gate_ids)
        if not has_gate:
            ungated.append(slice_name)

    assert not ungated, (
        f"{len(ungated)} promoted HEP slice(s) without release gate:\n"
        + "\n".join(f"  - {s}" for s in ungated)
        + "\n\nEither add a release surface gate or add to "
        "KNOWN_UNGATED_PROMOTED_SLICES with a tracking ADR."
    )


def test_every_release_gate_maps_to_hep_surface(hep_matrix, release_matrix):
    """Every required-for-release gate should correspond to at least one
    surface in the HEP matrix (or be a non-HEP gate like M15)."""
    non_hep_gates = {"m15_reporting_stable_surface"}
    gate_ids = _release_gate_ids(release_matrix)
    hep_owners = {s["owner_slice"] for s in hep_matrix["surfaces"]}

    orphaned = []
    for gid in sorted(gate_ids):
        if gid in non_hep_gates:
            continue
        has_hep = any(owner in gid for owner in hep_owners)
        if not has_hep:
            orphaned.append(gid)

    assert not orphaned, (
        f"{len(orphaned)} release gate(s) without HEP surface coverage:\n"
        + "\n".join(f"  - {g}" for g in orphaned)
    )


def test_known_ungated_slices_are_still_promoted(hep_matrix):
    """Known-ungated exceptions must still be promoted slices.
    If a slice is removed from promoted, remove it from the allowlist."""
    promoted = _hep_promoted_slices(hep_matrix)
    stale = set(KNOWN_UNGATED_PROMOTED_SLICES) - promoted - PROMOTED_HEP_SLICES
    assert not stale, (
        f"KNOWN_UNGATED_PROMOTED_SLICES contains non-promoted slices: {stale}\n"
        "Remove them from the allowlist."
    )


def test_promoted_hep_slices_constant_matches_matrix(hep_matrix):
    """PROMOTED_HEP_SLICES constant must match the actually promoted slices
    derived from the matrix."""
    actual = _hep_promoted_slices(hep_matrix)
    assert PROMOTED_HEP_SLICES == actual, (
        f"PROMOTED_HEP_SLICES mismatch:\n"
        f"  constant: {sorted(PROMOTED_HEP_SLICES)}\n"
        f"  actual:   {sorted(actual)}\n"
        "Update PROMOTED_HEP_SLICES to match the matrix."
    )
