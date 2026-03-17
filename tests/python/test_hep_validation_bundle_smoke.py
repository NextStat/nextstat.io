"""HEP validation bundle smoke tests.

Ensures the canonical HEP validation bundle:
- Generates without error
- Reports 141/141 stable
- Covers all promoted slices with release gates
- References valid support matrix paths
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load_hep_matrix() -> dict:
    return json.loads((REPO / "hep_surface_matrix_v1.json").read_text(encoding="utf-8"))


def _load_release_matrix() -> dict:
    return json.loads(
        (REPO / "scripts" / "release_surface_matrix_v1.json").read_text(encoding="utf-8")
    )


def _build_bundle() -> dict:
    from scripts.hep_validation_bundle import build_bundle

    return build_bundle(_load_hep_matrix(), _load_release_matrix())


def test_bundle_reports_all_stable():
    bundle = _build_bundle()
    assert bundle["summary"]["all_stable"] is True
    assert bundle["summary"]["research"] == 0
    assert bundle["summary"]["stable"] == bundle["summary"]["total_surfaces"]


def test_bundle_total_is_141():
    bundle = _build_bundle()
    assert bundle["summary"]["total_surfaces"] == 141


def test_every_slice_fully_stable():
    bundle = _build_bundle()
    for owner, counts in bundle["per_slice"].items():
        assert counts["fully_stable"] is True, (
            f"Slice {owner} has {counts['research']} research surface(s)"
        )


def test_release_gates_cover_promoted_slices():
    bundle = _build_bundle()
    gate_ids = {g["id"] for g in bundle["release_gates"]}
    promoted_slices = {"gvm", "simplified_likelihood", "hepdata", "histfactory"}
    for ps in promoted_slices:
        has_gate = any(ps in gid for gid in gate_ids)
        assert has_gate, f"Promoted slice {ps} has no release gate in bundle"


def test_support_matrix_refs_exist():
    bundle = _build_bundle()
    for ref in bundle["support_matrix_refs"]:
        path = REPO / ref
        assert path.exists(), f"Support matrix ref does not exist: {ref}"


def test_parity_contracts_present():
    bundle = _build_bundle()
    assert "histfactory" in bundle["parity_contracts"]
    assert "simplified_likelihood" in bundle["parity_contracts"]
    assert "gvm" in bundle["parity_contracts"]
    assert "hepdata" in bundle["parity_contracts"]
    assert bundle["parity_contracts"]["histfactory"]["reference"] == "pyhf"
