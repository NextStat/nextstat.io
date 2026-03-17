"""HEP-ADR-02: HistFactory core stable-surface promotion gate.

Verifies that the HistFactory core stable subset has the required promotion
artifacts: support matrix, acceptance doc, gate script, and release surface
matrix entry.

Narrow stable subset (deterministic CPU parity on pyhf JSON inputs):
  - workspace_audit
  - fit
  - hypotest (asymptotic)
  - upper-limit
  - scan
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# ── Required promotion artifacts ────────────────────────────────────────────

SUPPORT_MATRIX = REPO / "docs" / "benchmarks" / "histfactory-support-matrix-2026-03-17.md"
ACCEPTANCE_DOC = REPO / "docs" / "benchmarks" / "histfactory-stable-surface-acceptance-2026-03-17.md"
GATE_SCRIPT = REPO / "scripts" / "benchmarks" / "histfactory_stable_surface_gate.sh"

RELEASE_MATRIX_PATH = REPO / "scripts" / "release_surface_matrix_v1.json"
HEP_MATRIX_PATH = REPO / "hep_surface_matrix_v1.json"

# ── Narrow stable subset ───────────────────────────────────────────────────

HISTFACTORY_STABLE_CLI = {
    "run", "validate", "fit", "audit", "hypotest", "upper-limit", "scan",
}

HISTFACTORY_STABLE_PY = {
    "HistFactoryModel", "from_pyhf", "fit", "hypotest", "upper_limit",
    "profile_scan", "workspace_audit",
}


# ── Tests ──────────────────────────────────────────────────────────────────

def test_support_matrix_exists():
    """HistFactory must have a dedicated support matrix document."""
    assert SUPPORT_MATRIX.exists(), (
        f"Missing: {SUPPORT_MATRIX.relative_to(REPO)}\n"
        "Create a HistFactory support matrix following the GVM/SL template."
    )


def test_support_matrix_covers_stable_subset():
    """Support matrix must mention every CLI surface in the stable subset."""
    if not SUPPORT_MATRIX.exists():
        pytest.skip("Support matrix not yet created")
    text = SUPPORT_MATRIX.read_text()
    missing = [cmd for cmd in HISTFACTORY_STABLE_CLI if cmd not in text]
    assert not missing, (
        f"Support matrix missing CLI surfaces: {missing}"
    )


def test_acceptance_doc_exists():
    """HistFactory must have a stable-surface acceptance document."""
    assert ACCEPTANCE_DOC.exists(), (
        f"Missing: {ACCEPTANCE_DOC.relative_to(REPO)}\n"
        "Create a HistFactory acceptance doc following the SL template."
    )


def test_acceptance_doc_defines_fidelity_gates():
    """Acceptance doc must define fidelity tolerances for pyhf parity."""
    if not ACCEPTANCE_DOC.exists():
        pytest.skip("Acceptance doc not yet created")
    text = ACCEPTANCE_DOC.read_text()
    required_terms = ["1e-8", "1e-6", "pyhf", "parity"]
    missing = [t for t in required_terms if t not in text]
    assert not missing, (
        f"Acceptance doc missing required terms: {missing}"
    )


def test_gate_script_exists():
    """HistFactory must have a dedicated gate script."""
    assert GATE_SCRIPT.exists(), (
        f"Missing: {GATE_SCRIPT.relative_to(REPO)}\n"
        "Create a HistFactory gate script following the SL gate template."
    )


def test_gate_script_is_executable():
    """Gate script must be executable."""
    if not GATE_SCRIPT.exists():
        pytest.skip("Gate script not yet created")
    import os
    assert os.access(GATE_SCRIPT, os.X_OK), (
        f"{GATE_SCRIPT.relative_to(REPO)} is not executable. "
        "Run: chmod +x scripts/benchmarks/histfactory_stable_surface_gate.sh"
    )


def test_gate_script_builds_local_wheelhouse_with_the_gate_python():
    """HistFactory gate must validate the installed wheel, not source-shadowed bindings."""
    if not GATE_SCRIPT.exists():
        pytest.skip("Gate script not yet created")
    text = GATE_SCRIPT.read_text(encoding="utf-8")
    assert 'run_maturin()' in text
    assert '"-m" "maturin"' in text
    assert 'maturin build --release --interpreter "${py}" -o "${wheelhouse}"' in text
    assert 'NEXTSTAT_PREFER_INSTALLED=1 PYTHONPATH="" "${py}" - <<\'PY\'' in text
    assert 'assert callable(nextstat.set_threads)' in text
    assert 'PYTHONPATH="${py_path}" "${py}" -m pytest' in text


def test_release_surface_matrix_has_histfactory(release_matrix):
    """release_surface_matrix_v1.json must contain a histfactory entry."""
    ids = {s["id"] for s in release_matrix["surfaces"]}
    assert "histfactory_stable_surface" in ids, (
        "release_surface_matrix_v1.json missing 'histfactory_stable_surface' entry.\n"
        f"Current surface IDs: {sorted(ids)}"
    )


def test_histfactory_release_surface_is_required(release_matrix):
    """The histfactory release surface must be required_for_release."""
    for s in release_matrix["surfaces"]:
        if s["id"] == "histfactory_stable_surface":
            assert s.get("required_for_release") is True, (
                "histfactory_stable_surface must have required_for_release: true"
            )
            return
    pytest.fail("histfactory_stable_surface not found in release surface matrix")


def test_hep_matrix_histfactory_surfaces_have_support_ref(hep_matrix):
    """Stable HistFactory surfaces must reference the new support matrix."""
    violations = []
    for s in hep_matrix["surfaces"]:
        if (s.get("owner_slice") == "histfactory"
                and s.get("maturity_class") == "stable"
                and not s.get("support_matrix_ref")):
            violations.append(f"{s['name']} (layer={s['layer']})")

    assert not violations, (
        f"{len(violations)} stable histfactory surface(s) without support_matrix_ref:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def release_matrix():
    assert RELEASE_MATRIX_PATH.exists()
    with open(RELEASE_MATRIX_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def hep_matrix():
    assert HEP_MATRIX_PATH.exists()
    with open(HEP_MATRIX_PATH) as f:
        return json.load(f)
