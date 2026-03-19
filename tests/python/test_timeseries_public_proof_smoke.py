"""Time series public-proof closure invariants (NSS-03).

Validates that the time series benchmark surface now has the public proof
infrastructure required for a SOTA claim:

  1. Public benchmark page exists and is not draft.
  2. Suite harness exists.
  3. Named competitor baselines are documented in the benchmark page.
  4. At least one committed snapshot contains time series cases.
  5. Snapshot cases have expected schema version and clean parity.
  6. Parity test goldens exist for AR(1) Kalman.
  7. Claim matrix entry exists for timeseries with `status=sota`.
  8. Parity methodology is documented.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

BENCHMARK_PAGE = REPO / "docs" / "benchmarks" / "suites" / "timeseries.md"
SUITE_DIR = REPO / "benchmarks" / "nextstat-public-benchmarks" / "suites" / "timeseries"
SNAPSHOTS_DIR = REPO / "benchmarks" / "nextstat-public-benchmarks" / "manifests" / "snapshots"
CLAIM_MATRIX = REPO / ".internal" / "claims" / "nextstat_sota_claim_matrix_v1.json"
GOLDENS_FIXTURE = REPO / "tests" / "fixtures" / "statsmodels_ar1_kalman_goldens.json"

EXPECTED_COMPETITORS = ["pykalman", "statsmodels", "arch"]
EXPECTED_CASES_IN_README = ["kalman", "smoother", "garch"]


def _load_latest_snapshot() -> tuple[Path | None, dict | None]:
    if not SNAPSHOTS_DIR.exists():
        return None, None
    latest = None
    for snap_dir in sorted(SNAPSHOTS_DIR.iterdir()):
        suite_json = snap_dir / "timeseries" / "timeseries_suite.json"
        if suite_json.exists():
            latest = suite_json
    if latest is None:
        return None, None
    return latest, json.loads(latest.read_text(encoding="utf-8"))


def test_benchmark_page_exists() -> None:
    assert BENCHMARK_PAGE.exists(), f"Missing public benchmark page: {BENCHMARK_PAGE}"


def test_benchmark_page_not_draft() -> None:
    text = BENCHMARK_PAGE.read_text(encoding="utf-8")
    assert "status: draft" not in text


@pytest.mark.parametrize("filename", ["run.py", "suite.py", "README.md"])
def test_suite_harness_files_exist(filename: str) -> None:
    path = SUITE_DIR / filename
    assert path.exists(), f"Missing suite file: {path}"


def test_competitors_named_in_benchmark_page() -> None:
    text = BENCHMARK_PAGE.read_text(encoding="utf-8").lower()
    for comp in EXPECTED_COMPETITORS:
        assert comp in text, f"Competitor {comp!r} not named in benchmark page."


def test_competitors_named_in_suite_readme() -> None:
    readme = SUITE_DIR / "README.md"
    text = readme.read_text(encoding="utf-8").lower()
    for comp in EXPECTED_COMPETITORS:
        assert comp in text, f"Suite README missing competitor: {comp!r}"


def test_committed_snapshot_exists() -> None:
    path, data = _load_latest_snapshot()
    assert path is not None and data is not None, "No committed timeseries snapshot found."


def test_snapshot_has_valid_schema() -> None:
    _, data = _load_latest_snapshot()
    if data is None:
        pytest.skip("No timeseries snapshot found")
    assert data["schema_version"] == "nextstat.timeseries_benchmark_suite_result.v1"
    assert data["summary"]["n_cases"] >= 4
    assert data["summary"]["n_failed"] == 0


def test_snapshot_has_no_warn_cases() -> None:
    _, data = _load_latest_snapshot()
    if data is None:
        pytest.skip("No timeseries snapshot found")
    bad = [c["case"] for c in data["cases"] if c["status"] != "ok"]
    assert not bad, f"Snapshot has non-ok cases: {bad}"


def test_snapshot_has_no_warn_or_skipped_parity() -> None:
    _, data = _load_latest_snapshot()
    if data is None:
        pytest.skip("No timeseries snapshot found")
    bad = [f"{c['case']} ({c['parity_status']})" for c in data["cases"] if c["parity_status"] != "ok"]
    assert not bad, f"Snapshot has non-ok parity rows: {bad}"


def test_snapshot_covers_both_kinds() -> None:
    _, data = _load_latest_snapshot()
    if data is None:
        pytest.skip("No timeseries snapshot found")
    kinds = {c["kind"] for c in data["cases"]}
    assert "kalman_local_level" in kinds
    assert "garch11" in kinds


def test_kalman_ar1_golden_fixture_exists() -> None:
    assert GOLDENS_FIXTURE.exists(), f"Missing AR(1) Kalman golden fixture: {GOLDENS_FIXTURE}"


def test_kalman_ar1_golden_fixture_has_expected_keys() -> None:
    data = json.loads(GOLDENS_FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert len(data) > 0


def test_claim_matrix_has_timeseries() -> None:
    data = json.loads(CLAIM_MATRIX.read_text(encoding="utf-8"))
    domains = {e["domain"]: e for e in data["domains"]}
    assert "timeseries" in domains


def test_claim_matrix_timeseries_is_sota() -> None:
    data = json.loads(CLAIM_MATRIX.read_text(encoding="utf-8"))
    domains = {e["domain"]: e for e in data["domains"]}
    assert domains["timeseries"]["status"] == "sota"


def test_claim_matrix_timeseries_has_proof_refs() -> None:
    data = json.loads(CLAIM_MATRIX.read_text(encoding="utf-8"))
    domains = {e["domain"]: e for e in data["domains"]}
    entry = domains["timeseries"]
    assert len(entry.get("proof_refs", [])) >= 2
    for ref in entry["proof_refs"]:
        assert (REPO / ref).exists(), f"timeseries proof_ref {ref!r} does not exist"


def test_parity_methodology_documented() -> None:
    text = BENCHMARK_PAGE.read_text(encoding="utf-8").lower()
    assert "machine-precision" in text or "machine precision" in text
    assert "approximate" in text


def test_benchmark_page_documents_current_promoted_subset() -> None:
    text = BENCHMARK_PAGE.read_text(encoding="utf-8").lower()
    assert "current promoted public proof subset" in text
    assert "4/4 ok" in text
    assert "pykalman" in text
    assert "arch" in text


def test_suite_readme_documents_cases() -> None:
    readme = SUITE_DIR / "README.md"
    text = readme.read_text(encoding="utf-8").lower()
    for keyword in EXPECTED_CASES_IN_README:
        assert keyword in text, f"Suite README missing case keyword: {keyword!r}"
