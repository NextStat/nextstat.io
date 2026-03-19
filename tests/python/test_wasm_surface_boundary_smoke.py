"""WASM public surface invariants for literal whole-product SOTA."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

CLAIM_MATRIX = REPO / ".internal" / "claims" / "nextstat_sota_claim_matrix_v1.json"
WASM_SRC = REPO / "bindings" / "ns-wasm" / "src" / "lib.rs"
WASM_DOCS = REPO / "docs" / "references" / "wasm-playground.md"
WASM_BUILD_SCRIPT = REPO / "scripts" / "playground_build_wasm.sh"
WASM_JS = REPO / "playground" / "pkg" / "ns_wasm.js"
WASM_WASM = REPO / "playground" / "pkg" / "ns_wasm_bg.wasm"
RUST_CI = REPO / ".github" / "workflows" / "rust-tests.yml"


def _wasm_entry() -> dict:
    data = json.loads(CLAIM_MATRIX.read_text(encoding="utf-8"))
    return {e["domain"]: e for e in data["domains"]}["wasm"]


def test_wasm_source_exists() -> None:
    assert WASM_SRC.exists(), f"Missing WASM source: {WASM_SRC}"


def test_wasm_ci_exists() -> None:
    text = RUST_CI.read_text(encoding="utf-8").lower()
    assert "wasm" in text, "CI workflow must reference WASM"


def test_wasm_is_sota() -> None:
    assert _wasm_entry()["status"] == "sota"


def test_wasm_proof_refs_valid() -> None:
    entry = _wasm_entry()
    for ref in entry["proof_refs"]:
        assert (REPO / ref).exists(), f"WASM proof_ref {ref!r} missing"


def test_wasm_guard_snippets_in_readme() -> None:
    entry = _wasm_entry()
    readme = (REPO / entry["guard_doc"]).read_text(encoding="utf-8")
    for snippet in entry["guard_snippets"]:
        assert snippet in readme, f"WASM guard snippet missing: {snippet!r}"


def test_wasm_docs_and_build_script_exist() -> None:
    assert WASM_DOCS.exists(), f"Missing WASM docs: {WASM_DOCS}"
    assert WASM_BUILD_SCRIPT.exists(), f"Missing WASM build script: {WASM_BUILD_SCRIPT}"


def test_wasm_docs_mark_stable_source_build_boundary() -> None:
    text = WASM_DOCS.read_text(encoding="utf-8")
    assert "status: stable" in text
    assert "Stable source-build boundary" in text
    assert "make playground-build-wasm" in text


def test_wasm_pkg_artifacts_present() -> None:
    assert WASM_JS.exists(), f"Missing built WASM JS glue: {WASM_JS}"
    assert WASM_WASM.exists(), f"Missing built WASM binary: {WASM_WASM}"


def test_wasm_ci_jobs_exist() -> None:
    text = RUST_CI.read_text(encoding="utf-8")
    assert "wasm-smoke:" in text
    assert "wasm-playground-build:" in text
    assert "bash scripts/playground_build_wasm.sh" in text
