"""HEP-ADR-06: HEPData import stable-surface promotion gate.

Verifies that the HEPData import stable subset has the required promotion
artifacts: support matrix, acceptance spec, runtime gate doc, gate runner,
release surface matrix entry, and benchmark baseline.

Narrow stable subset (CLI-only, offline-first deterministic import):
  - import hepdata
  - import patchset
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# ── Required promotion artifacts ────────────────────────────────────────────

SUPPORT_MATRIX = REPO / "docs" / "benchmarks" / "hepdata-import-support-matrix-2026-03-08.md"
ACCEPTANCE_SPEC = REPO / "docs" / "specs" / "hep" / "hepdata_import_acceptance_v1.md"
RUNTIME_GATE_DOC = REPO / "docs" / "benchmarks" / "hepdata-import-runtime-gate.md"
GATE_RUNNER = REPO / "scripts" / "benchmarks" / "run_hepdata_import_benchmark_gate.py"
BENCH_RUNNER = REPO / "scripts" / "benchmarks" / "bench_hepdata_import.py"
COMPARE_RUNNER = REPO / "scripts" / "benchmarks" / "compare_hepdata_import_benchmark.py"
PROMOTE_RUNNER = REPO / "scripts" / "benchmarks" / "promote_hepdata_import_benchmark_baseline.py"
ACCEPTED_BASELINE = (
    REPO / "benchmarks" / "artifacts" / "hepdata_import_baselines"
    / "nextstat-bench" / "accepted.json"
)

RELEASE_MATRIX_PATH = REPO / "scripts" / "release_surface_matrix_v1.json"
HEP_MATRIX_PATH = REPO / "hep_surface_matrix_v1.json"

# ── Stable subset ──────────────────────────────────────────────────────────

HEPDATA_STABLE_CLI = {"import hepdata"}


# ── Tests ──────────────────────────────────────────────────────────────────

def test_support_matrix_exists():
    """HEPData must have a dedicated support matrix document."""
    assert SUPPORT_MATRIX.exists(), (
        f"Missing: {SUPPORT_MATRIX.relative_to(REPO)}\n"
        "The HEPData import support matrix should already exist."
    )


def test_support_matrix_covers_stable_subset():
    """Support matrix must mention every CLI surface in the stable subset."""
    text = SUPPORT_MATRIX.read_text()
    missing = [cmd for cmd in HEPDATA_STABLE_CLI if cmd not in text]
    assert not missing, (
        f"Support matrix missing CLI surfaces: {missing}"
    )


def test_acceptance_spec_exists():
    """HEPData must have a dedicated acceptance spec."""
    assert ACCEPTANCE_SPEC.exists(), (
        f"Missing: {ACCEPTANCE_SPEC.relative_to(REPO)}"
    )


def test_runtime_gate_doc_exists():
    """HEPData must have a runtime gate document."""
    assert RUNTIME_GATE_DOC.exists(), (
        f"Missing: {RUNTIME_GATE_DOC.relative_to(REPO)}"
    )


def test_gate_runner_exists():
    """HEPData must have a one-shot gate runner script."""
    assert GATE_RUNNER.exists(), (
        f"Missing: {GATE_RUNNER.relative_to(REPO)}"
    )


def test_bench_runner_exists():
    """HEPData must have a benchmark runner script."""
    assert BENCH_RUNNER.exists(), (
        f"Missing: {BENCH_RUNNER.relative_to(REPO)}"
    )


def test_compare_runner_exists():
    """HEPData must have a compare runner script."""
    assert COMPARE_RUNNER.exists(), (
        f"Missing: {COMPARE_RUNNER.relative_to(REPO)}"
    )


def test_promote_runner_exists():
    """HEPData must have a promote runner script."""
    assert PROMOTE_RUNNER.exists(), (
        f"Missing: {PROMOTE_RUNNER.relative_to(REPO)}"
    )


def test_accepted_baseline_exists():
    """HEPData must have an accepted benchmark baseline."""
    assert ACCEPTED_BASELINE.exists(), (
        f"Missing: {ACCEPTED_BASELINE.relative_to(REPO)}\n"
        "Record a baseline on nextstat-bench first."
    )


def test_release_surface_matrix_has_hepdata(release_matrix):
    """release_surface_matrix_v1.json must contain a hepdata entry."""
    ids = {s["id"] for s in release_matrix["surfaces"]}
    assert "hepdata_import_stable_surface" in ids, (
        "release_surface_matrix_v1.json missing 'hepdata_import_stable_surface' entry.\n"
        f"Current surface IDs: {sorted(ids)}"
    )


def test_hepdata_release_surface_is_required(release_matrix):
    """The hepdata release surface must be required_for_release."""
    for s in release_matrix["surfaces"]:
        if s["id"] == "hepdata_import_stable_surface":
            assert s.get("required_for_release") is True, (
                "hepdata_import_stable_surface must have required_for_release: true"
            )
            return
    pytest.fail("hepdata_import_stable_surface not found in release surface matrix")


def test_hep_matrix_hepdata_surfaces_have_support_ref(hep_matrix):
    """Stable HEPData surfaces must reference the support matrix."""
    violations = []
    for s in hep_matrix["surfaces"]:
        if (s.get("owner_slice") == "hepdata"
                and s.get("maturity_class") == "stable"
                and not s.get("support_matrix_ref")):
            violations.append(f"{s['name']} (layer={s['layer']})")

    assert not violations, (
        f"{len(violations)} stable hepdata surface(s) without support_matrix_ref:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_hepdata_no_longer_in_known_ungated():
    """After ADR-06, hepdata must be removed from KNOWN_UNGATED_PROMOTED_SLICES."""
    gate_parity_test = (
        REPO / "tests" / "python" / "test_hep_release_gate_parity_smoke.py"
    )
    text = gate_parity_test.read_text()
    import ast
    tree = ast.parse(text)
    for node in ast.walk(tree):
        target = None
        value = None
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "KNOWN_UNGATED_PROMOTED_SLICES":
                    target = t
                    value = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "KNOWN_UNGATED_PROMOTED_SLICES":
                target = node.target
                value = node.value
        if target and value and isinstance(value, ast.Dict):
            keys = [k.value for k in value.keys if isinstance(k, ast.Constant)]
            assert "hepdata" not in keys, (
                "hepdata must be removed from KNOWN_UNGATED_PROMOTED_SLICES "
                "now that ADR-06 provides a dedicated release gate."
            )
            return
    pytest.fail("Could not find KNOWN_UNGATED_PROMOTED_SLICES in test_hep_release_gate_parity_smoke.py")


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
