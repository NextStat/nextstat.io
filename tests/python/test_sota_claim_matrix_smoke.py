from __future__ import annotations

import json
from pathlib import Path

from scripts.sota_claim_matrix import build_report, check_matrix, load_matrix, render_markdown, validate_matrix


REPO_ROOT = Path(__file__).resolve().parents[2]
CLAIM_MATRIX = REPO_ROOT / ".internal" / "claims" / "nextstat_sota_claim_matrix_v1.json"
ALLOWED_STATUSES = {"sota", "proof_pending", "access_pending", "scoped_out"}
EXPECTED_DOMAINS = {"hep", "pharma", "bayesian", "econometrics", "timeseries", "gpu", "r", "wasm"}


def _load_matrix() -> dict:
    return json.loads(CLAIM_MATRIX.read_text(encoding="utf-8"))


def _domains_by_name() -> dict[str, dict]:
    data = _load_matrix()
    domains = data["domains"]
    return {entry["domain"]: entry for entry in domains}


def test_claim_matrix_exists_and_has_expected_schema_version() -> None:
    data = _load_matrix()
    assert data["schema_version"] == "nextstat.sota_claim_matrix.v1"


def test_claim_matrix_script_loader_matches_raw_json() -> None:
    assert load_matrix(REPO_ROOT) == _load_matrix()


def test_claim_matrix_domains_are_complete_and_unique() -> None:
    by_name = _domains_by_name()
    assert set(by_name) == EXPECTED_DOMAINS
    assert len(by_name) == len(EXPECTED_DOMAINS)


def test_claim_matrix_statuses_are_allowed() -> None:
    for domain, entry in _domains_by_name().items():
        assert entry["status"] in ALLOWED_STATUSES, domain


def test_claim_matrix_validates_via_script() -> None:
    validate_matrix(_load_matrix(), REPO_ROOT)


def test_claim_matrix_references_exist() -> None:
    for domain, entry in _domains_by_name().items():
        refs = [*entry.get("proof_refs", []), *entry.get("access_refs", []), entry["guard_doc"]]
        for ref in refs:
            path = REPO_ROOT / ref
            assert path.exists(), f"{domain}: missing ref {ref}"


def test_sota_domains_have_proof_and_access_refs() -> None:
    for domain, entry in _domains_by_name().items():
        if entry["status"] != "sota":
            continue
        assert entry["proof_refs"], domain
        assert entry["access_refs"], domain
        assert entry["guard_snippets"], domain


def test_non_sota_domains_have_explicit_public_guards() -> None:
    for domain, entry in _domains_by_name().items():
        if entry["status"] == "sota":
            continue
        assert entry["guard_snippets"], domain
        assert entry["scope"], domain


def test_guard_snippets_exist_in_guard_docs() -> None:
    for domain, entry in _domains_by_name().items():
        guard_doc = REPO_ROOT / entry["guard_doc"]
        text = guard_doc.read_text(encoding="utf-8")
        for snippet in entry["guard_snippets"]:
            assert snippet in text, f"{domain}: missing guard snippet {snippet!r}"


def test_current_claim_boundary_is_narrow() -> None:
    by_name = _domains_by_name()
    assert {d for d, e in by_name.items() if e["status"] == "sota"} == {
        "bayesian",
        "econometrics",
        "gpu",
        "hep",
        "pharma",
        "r",
        "timeseries",
        "wasm",
    }
    assert {d for d, e in by_name.items() if e["status"] == "scoped_out"} == set()
    assert {d for d, e in by_name.items() if e["status"] == "proof_pending"} == set()
    assert {d for d, e in by_name.items() if e["status"] == "access_pending"} == set()


def test_ads_is_not_part_of_public_sota_claim_matrix() -> None:
    assert "ads" not in _domains_by_name()


def test_claim_matrix_report_summarizes_boundary() -> None:
    report = build_report(REPO_ROOT)
    assert report["schema_version"] == "nextstat.sota_claim_matrix_report.v1"
    assert report["summary"]["total_domains"] == len(EXPECTED_DOMAINS)
    assert report["summary"]["sota_domains"] == ["bayesian", "econometrics", "gpu", "hep", "pharma", "r", "timeseries", "wasm"]
    assert report["summary"]["proof_pending_domains"] == []
    assert report["summary"]["access_pending_domains"] == []


def test_claim_matrix_report_markdown_mentions_current_boundary() -> None:
    md = render_markdown(build_report(REPO_ROOT))
    assert "# SOTA Claim Matrix Report" in md
    assert "`sota`: `bayesian`, `econometrics`, `gpu`, `hep`, `pharma`, `r`, `timeseries`, `wasm`" in md
    assert "`proof_pending`: none" in md
    assert "`access_pending`: none" in md
    assert "`scoped_out`: none" in md


def test_check_mode_passes() -> None:
    ok, message = check_matrix(REPO_ROOT)
    assert ok, message
    assert "8/8 domains currently claim SOTA" in message
