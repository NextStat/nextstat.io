"""Econometrics public-proof closure invariants (NSS-02).

Validates that the econometrics benchmark surface has the public proof
infrastructure required for a SOTA claim:

  1. Public benchmark page exists and is not draft.
  2. Suite harness exists (run.py, suite.py, README.md).
  3. JSON schema exists for per-case results.
  4. Named competitor baselines are documented in the benchmark page.
  5. Estimator boundary for did_staggered is explicitly documented.
  6. At least one committed snapshot contains econometrics cases.
  7. Snapshot cases have expected schema version.
  8. Claim matrix entry exists for econometrics.

ADR: audit/2026-03-19_nextstat-to-sota-adr-internal.md
Checklist: audit/2026-03-19_nextstat-to-sota-apex2-checklist-internal.md
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

BENCHMARK_PAGE = REPO / "docs" / "benchmarks" / "suites" / "econometrics.md"
SUITE_DIR = REPO / "benchmarks" / "nextstat-public-benchmarks" / "suites" / "econometrics"
SNAPSHOTS_DIR = REPO / "benchmarks" / "nextstat-public-benchmarks" / "manifests" / "snapshots"
CLAIM_MATRIX = REPO / ".internal" / "claims" / "nextstat_sota_claim_matrix_v1.json"

EXPECTED_COMPETITORS = ["statsmodels", "linearmodels", "pyfixest"]

# Cases that must appear in the suite README for completeness
EXPECTED_CASES_IN_README = [
    "panel fixed effects",
    "DiD TWFE",
    "wild cluster bootstrap",
    "staggered",
    "event study",
    "IV",
    "2SLS",
    "AIPW",
]


# ── INVARIANT 1: Public benchmark page exists and is stable ──────────


def test_benchmark_page_exists() -> None:
    """Public econometrics benchmark page must exist."""
    assert BENCHMARK_PAGE.exists(), (
        f"Missing public benchmark page: {BENCHMARK_PAGE}"
    )


def test_benchmark_page_not_draft() -> None:
    """Benchmark page must not be in draft status."""
    text = BENCHMARK_PAGE.read_text(encoding="utf-8")
    # Check YAML frontmatter status field
    assert "status: draft" not in text, (
        "Benchmark page is still marked as draft. "
        "Promote to stable before claiming SOTA."
    )


# ── INVARIANT 2: Suite harness exists ────────────────────────────────


@pytest.mark.parametrize("filename", ["run.py", "suite.py", "README.md"])
def test_suite_harness_files_exist(filename: str) -> None:
    """Suite harness must have all required files."""
    path = SUITE_DIR / filename
    assert path.exists(), f"Missing suite file: {path}"


# ── INVARIANT 3: Named competitor baselines documented ───────────────


def test_competitors_named_in_benchmark_page() -> None:
    """All expected competitors must be named in the benchmark page."""
    text = BENCHMARK_PAGE.read_text(encoding="utf-8").lower()
    for comp in EXPECTED_COMPETITORS:
        assert comp.lower() in text, (
            f"Competitor {comp!r} not named in benchmark page. "
            f"All competitor baselines must be explicit."
        )


# ── INVARIANT 4: Estimator boundary documented ──────────────────────


def test_estimator_boundary_documented() -> None:
    """The did_staggered estimator boundary must be explicitly documented."""
    text = BENCHMARK_PAGE.read_text(encoding="utf-8").lower()
    # Must contain language about estimator difference
    assert "estimator boundary" in text or "algorithm choice" in text, (
        "did_staggered estimator boundary not documented in benchmark page. "
        "Must explain algorithm difference vs pyfixest."
    )
    # Must mention the specific algorithm
    assert "group-time att" in text, (
        "NextStat's group-time ATT estimator not named in benchmark page."
    )
    assert "callaway" in text or "doubly-robust" in text, (
        "pyfixest's Callaway-Sant'Anna estimator not referenced in benchmark page."
    )


# ── INVARIANT 5: Benchmark page covers all cases ────────────────────


def test_benchmark_page_covers_all_cases() -> None:
    """Benchmark page must mention all expected case categories."""
    text = BENCHMARK_PAGE.read_text(encoding="utf-8").lower()
    missing = [c for c in EXPECTED_CASES_IN_README if c.lower() not in text]
    assert not missing, (
        f"Benchmark page missing case categories: {missing}"
    )


# ── INVARIANT 6: At least one committed snapshot has econometrics ────


def test_committed_snapshot_exists() -> None:
    """At least one committed snapshot must contain econometrics cases."""
    if not SNAPSHOTS_DIR.exists():
        pytest.fail("Snapshots directory does not exist")

    econometrics_snapshots = []
    for snap_dir in sorted(SNAPSHOTS_DIR.iterdir()):
        econ_dir = snap_dir / "econometrics"
        if econ_dir.is_dir():
            suite_json = econ_dir / "econometrics_suite.json"
            if suite_json.exists():
                econometrics_snapshots.append(snap_dir.name)

    assert len(econometrics_snapshots) > 0, (
        "No committed snapshot contains econometrics cases. "
        "Run the suite on canonical host and commit results."
    )


def _load_latest_snapshot() -> tuple[Path | None, dict | None]:
    """Find and load the latest econometrics snapshot."""
    if not SNAPSHOTS_DIR.exists():
        return None, None
    latest = None
    for snap_dir in sorted(SNAPSHOTS_DIR.iterdir()):
        suite_json = snap_dir / "econometrics" / "econometrics_suite.json"
        if suite_json.exists():
            latest = suite_json
    if latest is None:
        return None, None
    return latest, json.loads(latest.read_text(encoding="utf-8"))


def test_snapshot_has_valid_schema() -> None:
    """Latest econometrics snapshot must have valid schema version."""
    path, data = _load_latest_snapshot()
    if data is None:
        pytest.skip("No econometrics snapshot found")

    assert data["schema_version"] == "nextstat.econometrics_benchmark_suite_result.v1", (
        f"Unexpected schema version: {data['schema_version']}"
    )
    assert data["summary"]["n_cases"] >= 8, (
        f"Snapshot has only {data['summary']['n_cases']} cases (expected >= 8)"
    )
    assert data["summary"]["n_failed"] == 0, (
        f"Snapshot has {data['summary']['n_failed']} failed cases"
    )


# ── INVARIANT 6b: Proof-grade snapshot quality (SOTA gate) ────────


def test_snapshot_no_warn_cases() -> None:
    """SOTA gate: no case may have status=warn or status=failed."""
    _, data = _load_latest_snapshot()
    if data is None:
        pytest.skip("No econometrics snapshot found")

    warn_or_fail = [
        f"{c['case']} ({c['status']})"
        for c in data["cases"]
        if c["status"] in ("warn", "failed")
    ]
    assert not warn_or_fail, (
        f"Snapshot has non-ok cases — cannot claim SOTA: {warn_or_fail}"
    )


def test_snapshot_no_warn_parity() -> None:
    """SOTA gate: no case may have parity_status=warn or parity_status=failed."""
    _, data = _load_latest_snapshot()
    if data is None:
        pytest.skip("No econometrics snapshot found")

    bad_parity = [
        f"{c['case']} (parity={c['parity_status']})"
        for c in data["cases"]
        if c["parity_status"] in ("warn", "failed")
    ]
    assert not bad_parity, (
        f"Snapshot has parity warnings — competitor comparison incomplete: {bad_parity}"
    )


def test_snapshot_skipped_parity_documented() -> None:
    """Any case with parity_status=skipped must be in the documented exceptions list."""
    _, data = _load_latest_snapshot()
    if data is None:
        pytest.skip("No econometrics snapshot found")

    # Cases where parity skip is acceptable (no external baseline exists)
    PARITY_SKIP_ALLOWED = {"aipw_ate"}

    skipped = [
        c["case"]
        for c in data["cases"]
        if c["parity_status"] == "skipped"
    ]
    undocumented = [c for c in skipped if c not in PARITY_SKIP_ALLOWED]
    assert not undocumented, (
        f"Cases with parity_status=skipped not in allowed list: {undocumented}. "
        f"Either add competitor baseline or add to PARITY_SKIP_ALLOWED with justification."
    )


def test_aipw_exception_is_explicit_in_public_benchmark_page() -> None:
    """The allowed AIPW parity skip must be documented publicly."""
    text = BENCHMARK_PAGE.read_text(encoding="utf-8").lower()
    assert "aipw" in text
    assert "no external parity yet" in text, (
        "Econometrics benchmark page must state that AIPW has no external parity yet."
    )


# ── INVARIANT 7: Suite README documents scope ────────────────────────


def test_suite_readme_documents_cases() -> None:
    """Suite README must list all benchmark cases."""
    readme = SUITE_DIR / "README.md"
    text = readme.read_text(encoding="utf-8").lower()
    for keyword in ["panel fixed effects", "staggered", "iv", "2sls", "aipw"]:
        assert keyword.lower() in text, (
            f"Suite README missing case keyword: {keyword!r}"
        )


def test_suite_readme_documents_competitors() -> None:
    """Suite README must name competitor libraries."""
    readme = SUITE_DIR / "README.md"
    text = readme.read_text(encoding="utf-8").lower()
    for comp in EXPECTED_COMPETITORS:
        assert comp.lower() in text, (
            f"Suite README missing competitor: {comp!r}"
        )


# ── INVARIANT 8: Claim matrix entry exists ──────────────────────────


def test_claim_matrix_has_econometrics() -> None:
    """Claim matrix must have an econometrics entry."""
    if not CLAIM_MATRIX.exists():
        pytest.skip("Claim matrix not found")

    data = json.loads(CLAIM_MATRIX.read_text(encoding="utf-8"))
    domains = {e["domain"]: e for e in data["domains"]}
    assert "econometrics" in domains, (
        "Econometrics not in claim matrix"
    )


def test_claim_matrix_econometrics_has_proof_refs() -> None:
    """Claim matrix econometrics entry must have proof refs."""
    if not CLAIM_MATRIX.exists():
        pytest.skip("Claim matrix not found")

    data = json.loads(CLAIM_MATRIX.read_text(encoding="utf-8"))
    domains = {e["domain"]: e for e in data["domains"]}
    entry = domains["econometrics"]
    assert len(entry.get("proof_refs", [])) > 0, (
        "Econometrics claim matrix entry has no proof_refs"
    )
    # Verify proof refs point to existing paths
    for ref in entry["proof_refs"]:
        assert (REPO / ref).exists(), (
            f"Econometrics proof_ref {ref!r} does not exist"
        )


# ── INVARIANT 9: Parity methodology is explicit ─────────────────────


def test_parity_methodology_documented() -> None:
    """Benchmark page must explain parity methodology."""
    text = BENCHMARK_PAGE.read_text(encoding="utf-8").lower()
    assert "machine-precision" in text or "machine precision" in text, (
        "Benchmark page must document where parity is machine-precision"
    )
    assert "approximate" in text, (
        "Benchmark page must document where parity is approximate"
    )
