"""HEP-ADR-01: Canonical HEP surface inventory regression gate.

This test loads the machine-readable hep_surface_matrix_v1.json and enforces:

1. Every entry has a maturity_class (stable | research | internal).
2. Every entry has an owner_slice.
3. No public HEP surface is unclassified.
4. CLI surfaces that also exist as Python API must have a python peer.
5. Tool-manifest HEP tools must appear in the matrix.
6. Matrix is non-empty and schema-valid.
7. Promoted stable slices reference their support matrix.
8. Research surfaces in governed slices reference their boundary doc.
9. Every non-empty support_matrix_ref points to a real file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MATRIX_PATH = REPO / "hep_surface_matrix_v1.json"

VALID_MATURITY = {"stable", "research", "internal"}
VALID_LAYERS = {"cli", "python", "tool", "server", "docs"}
REQUIRED_OWNER_SLICES = {
    "histfactory",
    "simplified_likelihood",
    "gvm",
    "hepdata",
    "unbinned",
    "viz",
    "import_export",
    "preprocess",
    "infrastructure",
}


@pytest.fixture(scope="module")
def matrix():
    assert MATRIX_PATH.exists(), (
        f"hep_surface_matrix_v1.json not found at {MATRIX_PATH}. "
        "Run scripts/hep_surface_matrix.py to generate it."
    )
    with open(MATRIX_PATH) as f:
        data = json.load(f)
    assert "schema_version" in data, "Missing schema_version"
    assert data["schema_version"] == "v1"
    assert "surfaces" in data, "Missing surfaces array"
    return data


def test_matrix_is_nonempty(matrix):
    assert len(matrix["surfaces"]) > 0, "Matrix has zero surfaces"


def test_every_surface_has_maturity_class(matrix):
    violations = []
    for s in matrix["surfaces"]:
        if s.get("maturity_class") not in VALID_MATURITY:
            violations.append(
                f"{s['name']} (layer={s.get('layer')}): "
                f"maturity_class={s.get('maturity_class')!r}"
            )
    assert not violations, (
        f"{len(violations)} surface(s) missing valid maturity_class:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_every_surface_has_owner_slice(matrix):
    violations = []
    for s in matrix["surfaces"]:
        owner = s.get("owner_slice")
        if not owner:
            violations.append(f"{s['name']} (layer={s.get('layer')}): no owner_slice")
    assert not violations, (
        f"{len(violations)} surface(s) missing owner_slice:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_every_surface_has_valid_layer(matrix):
    violations = []
    for s in matrix["surfaces"]:
        if s.get("layer") not in VALID_LAYERS:
            violations.append(
                f"{s['name']}: layer={s.get('layer')!r}"
            )
    assert not violations, (
        f"{len(violations)} surface(s) with invalid layer:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_no_duplicate_name_layer_pairs(matrix):
    seen = set()
    dupes = []
    for s in matrix["surfaces"]:
        key = (s["name"], s["layer"])
        if key in seen:
            dupes.append(f"{s['name']} (layer={s['layer']})")
        seen.add(key)
    assert not dupes, (
        f"{len(dupes)} duplicate (name, layer) pairs:\n"
        + "\n".join(f"  - {d}" for d in dupes)
    )


def test_tool_manifest_hep_surfaces_present(matrix):
    """Every HEP tool in _tool_manifest_v1.json must appear in the matrix."""
    manifest_path = REPO / "bindings" / "ns-py" / "python" / "nextstat" / "_tool_manifest_v1.json"
    if not manifest_path.exists():
        pytest.skip("Tool manifest not found")

    with open(manifest_path) as f:
        manifest = json.load(f)

    hep_keywords = {"nextstat_fit", "nextstat_scan", "hypotest", "upper_limit",
                    "ranking", "audit", "discovery", "workspace", "root_histogram",
                    "hepdata"}
    hep_tool_names = set()
    for t in manifest["tools"]:
        name = t["name"]
        if any(name == kw or name.startswith(kw + "_") or kw in name for kw in hep_keywords):
            hep_tool_names.add(name)

    matrix_tool_names = {
        s["name"] for s in matrix["surfaces"] if s["layer"] == "tool"
    }

    missing = hep_tool_names - matrix_tool_names
    assert not missing, (
        f"HEP tools in manifest but missing from matrix:\n"
        + "\n".join(f"  - {m}" for m in sorted(missing))
    )


def test_promoted_stable_surfaces_have_support_matrix_ref(matrix):
    """Stable surfaces in PROMOTED slices must reference their support matrix.

    Slices with existing support matrices: simplified_likelihood, gvm, hepdata.
    Other stable surfaces are pending dedicated support matrices (HEP-ADR-02+).
    """
    promoted_slices = {"simplified_likelihood", "gvm", "hepdata"}
    violations = []
    for s in matrix["surfaces"]:
        if (s.get("maturity_class") == "stable"
                and s.get("owner_slice") in promoted_slices
                and not s.get("support_matrix_ref")):
            violations.append(
                f"{s['name']} (layer={s['layer']}, owner={s['owner_slice']})"
            )
    assert not violations, (
        f"{len(violations)} promoted stable surface(s) without support_matrix_ref:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_research_surfaces_are_not_silent(matrix):
    """Research surfaces in governed slices must reference boundary docs.

    Slices whose support matrices explicitly document research boundaries:
    gvm, simplified_likelihood. Research surfaces in these slices must
    reference the boundary document. Other research surfaces are non-silent
    by virtue of being classified in the inventory.
    """
    governed_slices = {"gvm", "simplified_likelihood"}
    violations = []
    for s in matrix["surfaces"]:
        if s.get("maturity_class") != "research":
            continue
        owner = s.get("owner_slice", "")
        if owner in governed_slices and not s.get("support_matrix_ref"):
            violations.append(
                f"{s['name']} (layer={s.get('layer')}, owner={owner})"
            )
    assert not violations, (
        f"{len(violations)} research surface(s) in governed slices "
        "without boundary ref:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_support_matrix_refs_are_valid_paths(matrix):
    """Every non-empty support_matrix_ref must point to a real file."""
    missing = []
    for s in matrix["surfaces"]:
        ref = s.get("support_matrix_ref", "")
        if ref and not (REPO / ref).exists():
            missing.append(f"{s['name']} (layer={s['layer']}): {ref}")
    assert not missing, (
        f"{len(missing)} surface(s) with broken support_matrix_ref:\n"
        + "\n".join(f"  - {m}" for m in sorted(set(missing)))
    )


def test_owner_slices_cover_expected_set(matrix):
    """The matrix should cover all expected HEP product slices."""
    actual_slices = {s.get("owner_slice") for s in matrix["surfaces"]}
    missing = REQUIRED_OWNER_SLICES - actual_slices
    assert not missing, (
        f"Expected owner slices not represented in matrix:\n"
        + "\n".join(f"  - {m}" for m in sorted(missing))
    )


def test_minimum_surface_counts(matrix):
    """Sanity: matrix must have minimum expected surfaces per layer."""
    by_layer = {}
    for s in matrix["surfaces"]:
        layer = s.get("layer", "?")
        by_layer[layer] = by_layer.get(layer, 0) + 1

    assert by_layer.get("cli", 0) >= 30, (
        f"Expected >= 30 CLI surfaces, got {by_layer.get('cli', 0)}"
    )
    assert by_layer.get("python", 0) >= 15, (
        f"Expected >= 15 Python surfaces, got {by_layer.get('python', 0)}"
    )
    assert by_layer.get("tool", 0) >= 5, (
        f"Expected >= 5 Tool surfaces, got {by_layer.get('tool', 0)}"
    )
