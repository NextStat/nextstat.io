"""R public surface invariants for literal whole-product SOTA."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

CLAIM_MATRIX = REPO / ".internal" / "claims" / "nextstat_sota_claim_matrix_v1.json"
R_NAMESPACE = REPO / "bindings" / "ns-r" / "NAMESPACE"
R_DOCS = REPO / "docs" / "references" / "r-bindings.md"
R_README = REPO / "bindings" / "ns-r" / "README.md"
RUST_CI = REPO / ".github" / "workflows" / "rust-tests.yml"


def _r_entry() -> dict:
    data = json.loads(CLAIM_MATRIX.read_text(encoding="utf-8"))
    return {e["domain"]: e for e in data["domains"]}["r"]


def test_r_namespace_exists() -> None:
    assert R_NAMESPACE.exists(), f"Missing R NAMESPACE: {R_NAMESPACE}"


def test_r_namespace_has_exports() -> None:
    text = R_NAMESPACE.read_text(encoding="utf-8")
    exports = [l for l in text.splitlines() if l.startswith("export(")]
    assert len(exports) >= 10, f"R NAMESPACE has only {len(exports)} exports"


def test_r_is_sota() -> None:
    assert _r_entry()["status"] == "sota"


def test_r_proof_refs_valid() -> None:
    entry = _r_entry()
    for ref in entry["proof_refs"]:
        assert (REPO / ref).exists(), f"R proof_ref {ref!r} missing"


def test_r_guard_snippets_in_readme() -> None:
    entry = _r_entry()
    readme = (REPO / entry["guard_doc"]).read_text(encoding="utf-8")
    for snippet in entry["guard_snippets"]:
        assert snippet in readme, f"R guard snippet missing: {snippet!r}"


def test_r_docs_exist() -> None:
    assert R_DOCS.exists(), f"Missing R docs: {R_DOCS}"


def test_r_docs_mark_stable_source_build_boundary() -> None:
    text = R_DOCS.read_text(encoding="utf-8")
    assert "status: stable" in text
    assert "Stable source-build boundary" in text
    assert "R CMD INSTALL --library=tmp/r-lib bindings/ns-r" in text


def test_r_ci_smoke_job_exists() -> None:
    text = RUST_CI.read_text(encoding="utf-8")
    assert "r-bindings-smoke:" in text
    assert "R CMD INSTALL --library=tmp/r-lib bindings/ns-r" in text
    assert "library(testthat); library(nextstat)" in text


def test_r_binding_readme_is_stable() -> None:
    text = R_README.read_text(encoding="utf-8")
    assert "stable repo-source R package" in text
