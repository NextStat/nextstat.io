"""Docs discovery invariants (NSS-08).

Validates that every promoted proof/access surface is reachable from
top-level docs, and that no benchmark page is orphaned.

ADR: audit/2026-03-19_nextstat-to-sota-adr-internal.md
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

README = REPO / "README.md"
BENCHMARKS_HUB = REPO / "docs" / "benchmarks.md"
CLAIM_MATRIX = REPO / ".internal" / "claims" / "nextstat_sota_claim_matrix_v1.json"


# -- Key documentation files that must exist -------------------------


REQUIRED_DOCS = [
    "docs/README.md",
    "docs/benchmarks.md",
    "docs/references/validation-and-release-discipline.md",
    "docs/references/cli.md",
    "docs/references/python-api.md",
    "docs/references/r-bindings.md",
    "docs/references/rust-api.md",
    "docs/references/wasm-playground.md",
    "docs/references/hep-stable-surface.md",
    "docs/releases/release-runbook.md",
    "docs/releases/benchmark-artifact-policy.md",
]


@pytest.mark.parametrize("doc_path", REQUIRED_DOCS)
def test_required_doc_exists(doc_path: str) -> None:
    assert (REPO / doc_path).exists(), f"Missing required doc: {doc_path}"


# -- README links to key entry points --------------------------------


def test_readme_links_docs_index() -> None:
    text = README.read_text(encoding="utf-8")
    assert "docs/README.md" in text or "docs index" in text.lower()


def test_readme_links_benchmarks_hub() -> None:
    text = README.read_text(encoding="utf-8")
    assert "docs/benchmarks.md" in text or "benchmarks hub" in text.lower()


def test_readme_links_cli_reference() -> None:
    text = README.read_text(encoding="utf-8")
    assert "docs/references/cli.md" in text


def test_readme_links_python_api() -> None:
    text = README.read_text(encoding="utf-8")
    assert "docs/references/python-api.md" in text


def test_readme_links_r_bindings() -> None:
    text = README.read_text(encoding="utf-8")
    assert "docs/references/r-bindings.md" in text


def test_readme_links_wasm_reference() -> None:
    text = README.read_text(encoding="utf-8")
    assert "docs/references/wasm-playground.md" in text


# -- Benchmark suite pages exist and are linked ----------------------


BENCHMARK_SUITE_PAGES = [
    "docs/benchmarks/suites/econometrics.md",
    "docs/benchmarks/suites/timeseries.md",
    "docs/benchmarks/suites/bayesian.md",
]


@pytest.mark.parametrize("suite_page", BENCHMARK_SUITE_PAGES)
def test_benchmark_suite_page_exists(suite_page: str) -> None:
    assert (REPO / suite_page).exists(), f"Missing benchmark page: {suite_page}"


def test_benchmark_hub_links_all_suites() -> None:
    text = BENCHMARKS_HUB.read_text(encoding="utf-8").lower()
    for keyword in ["econometrics", "timeseries", "bayesian"]:
        assert keyword in text, (
            f"Benchmark hub does not reference suite: {keyword}"
        )


# -- Claim matrix proof_refs all point to existing files -------------


def test_all_claim_matrix_refs_exist() -> None:
    data = json.loads(CLAIM_MATRIX.read_text(encoding="utf-8"))
    missing = []
    for entry in data["domains"]:
        for ref_group in ("proof_refs", "access_refs"):
            for ref in entry.get(ref_group, []):
                if not (REPO / ref).exists():
                    missing.append(f"{entry['domain']}/{ref_group}: {ref}")
    assert not missing, f"Missing refs: {missing}"


# -- Guard docs all exist -------------------------------------------


def test_all_guard_docs_exist() -> None:
    data = json.loads(CLAIM_MATRIX.read_text(encoding="utf-8"))
    for entry in data["domains"]:
        guard = REPO / entry["guard_doc"]
        assert guard.exists(), (
            f"{entry['domain']}: guard_doc {entry['guard_doc']} missing"
        )


# -- Key tutorials exist --------------------------------------------


REQUIRED_TUTORIALS = [
    "docs/tutorials/phase-8-timeseries.md",
]


@pytest.mark.parametrize("tutorial", REQUIRED_TUTORIALS)
def test_tutorial_exists(tutorial: str) -> None:
    assert (REPO / tutorial).exists(), f"Missing tutorial: {tutorial}"
