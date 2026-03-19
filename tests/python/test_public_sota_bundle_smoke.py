"""Public SOTA bundle smoke tests (NSS-09).

Validates that the SOTA bundle generator produces a consistent,
complete artifact covering all claim matrix domains.

ADR: audit/2026-03-19_nextstat-to-sota-adr-internal.md
"""

from __future__ import annotations

from pathlib import Path

from scripts.public_sota_bundle import build_bundle, check_bundle, render_bundle_markdown

REPO = Path(__file__).resolve().parents[2]

EXPECTED_DOMAINS = {"hep", "pharma", "bayesian", "econometrics", "timeseries", "gpu", "r", "wasm"}


def test_bundle_check_passes() -> None:
    ok, message = check_bundle(REPO)
    assert ok, message


def test_bundle_schema_version() -> None:
    bundle = build_bundle(REPO)
    assert bundle["schema_version"] == "nextstat.public_sota_bundle.v1"


def test_bundle_covers_all_domains() -> None:
    bundle = build_bundle(REPO)
    domains = {e["domain"] for e in bundle["domains"]}
    assert domains == EXPECTED_DOMAINS


def test_bundle_summary_counts_consistent() -> None:
    bundle = build_bundle(REPO)
    s = bundle["summary"]
    total_from_counts = sum(s["status_counts"].values())
    assert total_from_counts == s["total_domains"]

    all_listed = (
        s["sota_domains"]
        + s["proof_pending_domains"]
        + s["access_pending_domains"]
        + s["scoped_out_domains"]
    )
    assert len(all_listed) == s["total_domains"]


def test_bundle_sota_domains_match_matrix() -> None:
    bundle = build_bundle(REPO)
    assert set(bundle["summary"]["sota_domains"]) == {
        "hep",
        "pharma",
        "econometrics",
        "bayesian",
        "timeseries",
        "gpu",
        "r",
        "wasm",
    }


def test_bundle_proof_pending_match() -> None:
    bundle = build_bundle(REPO)
    assert set(bundle["summary"]["proof_pending_domains"]) == set()


def test_bundle_access_pending_match() -> None:
    bundle = build_bundle(REPO)
    assert set(bundle["summary"]["access_pending_domains"]) == set()


def test_bundle_scoped_out_match() -> None:
    bundle = build_bundle(REPO)
    assert set(bundle["summary"]["scoped_out_domains"]) == set()


def test_bundle_markdown_renders() -> None:
    bundle = build_bundle(REPO)
    md = render_bundle_markdown(bundle)
    assert "# NextStat Public SOTA Bundle v1" in md
    assert "SOTA Boundary" in md
    assert "Domain Inventory" in md


def test_bundle_markdown_mentions_all_statuses() -> None:
    bundle = build_bundle(REPO)
    md = render_bundle_markdown(bundle)
    assert "SOTA" in md
    assert "Proof Pending" in md
    assert "Access Pending" in md
    assert "Scoped Out" in md


def test_bundle_every_domain_has_scope() -> None:
    bundle = build_bundle(REPO)
    for entry in bundle["domains"]:
        assert entry["scope"], f"{entry['domain']} has empty scope"


# ── Evidence richness tests ───────────────────────────────────────

DOMAINS_WITH_SNAPSHOTS = {"hep", "pharma", "econometrics", "bayesian", "timeseries"}


def test_bundle_every_domain_has_evidence() -> None:
    """Every domain must have an evidence section."""
    bundle = build_bundle(REPO)
    for entry in bundle["domains"]:
        assert "evidence" in entry, f"{entry['domain']} missing evidence section"


def test_bundle_snapshot_domains_have_benchmark_evidence() -> None:
    """Domains with benchmark suites must have snapshot evidence."""
    bundle = build_bundle(REPO)
    for entry in bundle["domains"]:
        if entry["domain"] in DOMAINS_WITH_SNAPSHOTS:
            snap = entry["evidence"].get("benchmark_snapshot")
            assert snap is not None, (
                f"{entry['domain']}: expected benchmark_snapshot in evidence"
            )
            assert snap["n_cases"] > 0, (
                f"{entry['domain']}: benchmark_snapshot has 0 cases"
            )


def test_bundle_sota_domains_have_clean_snapshots() -> None:
    """SOTA domains with snapshots must have all_ok=True."""
    bundle = build_bundle(REPO)
    for entry in bundle["domains"]:
        if entry["status"] != "sota":
            continue
        snap = entry["evidence"].get("benchmark_snapshot")
        if snap is None:
            continue
        assert snap["all_ok"], (
            f"{entry['domain']}: SOTA but snapshot has issues: "
            f"{snap['cases_with_issues']}"
        )


def test_bundle_hep_has_governance_evidence() -> None:
    """HEP domain must include governance surface count."""
    bundle = build_bundle(REPO)
    hep = next(e for e in bundle["domains"] if e["domain"] == "hep")
    gov = hep["evidence"].get("governance")
    assert gov is not None, "HEP missing governance evidence"
    assert gov["total_surfaces"] > 100, (
        f"HEP governance: only {gov['total_surfaces']} surfaces"
    )
    assert gov["stable"] == gov["total_surfaces"], (
        f"HEP: {gov['stable']}/{gov['total_surfaces']} stable "
        f"(expected all stable for SOTA)"
    )


def test_bundle_pharma_has_schema_evidence() -> None:
    """Pharma domain must include validation schema count."""
    bundle = build_bundle(REPO)
    pharma = next(e for e in bundle["domains"] if e["domain"] == "pharma")
    schemas = pharma["evidence"].get("validation_schemas")
    assert schemas is not None, "Pharma missing validation_schemas evidence"
    assert schemas >= 10, f"Pharma: only {schemas} validation schemas"


def test_bundle_proof_refs_all_valid() -> None:
    """All proof refs across all domains must point to existing files."""
    bundle = build_bundle(REPO)
    for entry in bundle["domains"]:
        ev = entry["evidence"]
        assert ev["proof_refs_valid"], (
            f"{entry['domain']}: missing proof refs: {ev['missing_proof_refs']}"
        )


def test_bundle_econometrics_documents_allowed_aipw_exception() -> None:
    bundle = build_bundle(REPO)
    econometrics = next(e for e in bundle["domains"] if e["domain"] == "econometrics")
    snap = econometrics["evidence"]["benchmark_snapshot"]
    aipw_issue = next(
        issue for issue in snap["cases_with_issues"] if issue["case"] == "aipw_ate"
    )
    assert aipw_issue["parity_status"] == "skipped"
    assert aipw_issue["allowed"] is True
    assert "documented competitor-baseline exception" in aipw_issue["reason"]
    assert snap["all_ok"] is True


def test_bundle_bayesian_uses_results_v10_evidence() -> None:
    bundle = build_bundle(REPO)
    bayesian = next(e for e in bundle["domains"] if e["domain"] == "bayesian")
    snap = bayesian["evidence"]["benchmark_snapshot"]
    assert snap["source"].endswith("results_v10/bayesian_multiseed_summary.json")
    assert snap["backends"] == "cmdstanpy,nextstat"
    assert snap["all_ok"] is True


def test_bundle_markdown_has_evidence_section() -> None:
    """Bundle markdown must include Domain Evidence section."""
    bundle = build_bundle(REPO)
    md = render_bundle_markdown(bundle)
    assert "## Domain Evidence" in md
    for domain in EXPECTED_DOMAINS:
        assert f"### {domain}" in md, (
            f"Domain Evidence section missing entry for {domain}"
        )


def test_bundle_r_has_binding_evidence() -> None:
    bundle = build_bundle(REPO)
    r_entry = next(e for e in bundle["domains"] if e["domain"] == "r")
    evidence = r_entry["evidence"]["r_surface"]
    assert evidence["export_count"] >= 20
    assert evidence["test_file_count"] >= 5


def test_bundle_wasm_has_build_evidence() -> None:
    bundle = build_bundle(REPO)
    wasm_entry = next(e for e in bundle["domains"] if e["domain"] == "wasm")
    evidence = wasm_entry["evidence"]["wasm_surface"]
    assert evidence["pkg_artifacts_present"] is True
    assert evidence["ci_jobs"]["wasm_smoke"] is True
    assert evidence["ci_jobs"]["wasm_playground_build"] is True
