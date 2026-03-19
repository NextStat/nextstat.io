"""Bayesian public-proof closure invariants (NSS-04).

Validates that the Bayesian benchmark surface now has the public proof
infrastructure required for a SOTA claim:

  1. Public benchmark page exists and is not draft.
  2. Suite harness exists.
  3. Current promoted proof subset is explicit and CmdStan-backed.
  4. Committed `results_v10` multiseed + derived metrics artifacts exist.
  5. Committed `results_v10` rows are clean (no warn/failed statuses).
  6. NUTS sampler reference doc exists with benchmark numbers.
  7. NUTS paper exists.
  8. Claim matrix entry exists with `status=sota`.
  9. SBC calibration infrastructure exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

BENCHMARK_PAGE = REPO / "docs" / "benchmarks" / "suites" / "bayesian.md"
SUITE_DIR = REPO / "benchmarks" / "nextstat-public-benchmarks" / "suites" / "bayesian"
CLAIM_MATRIX = REPO / ".internal" / "claims" / "nextstat_sota_claim_matrix_v1.json"
NUTS_REF = REPO / "docs" / "references" / "nuts-sampler.md"
NUTS_PAPER = REPO / "docs" / "papers" / "nuts-progressive-sampling.md"
RESULTS_V10 = SUITE_DIR / "results_v10"

EXPECTED_COMPETITORS = ["cmdstan", "pymc"]
EXPECTED_CASES = {
    "histfactory_simple_8p",
    "glm_logistic_regression",
    "hier_random_intercept_non_centered",
    "eight_schools_non_centered",
}


def test_benchmark_page_exists() -> None:
    assert BENCHMARK_PAGE.exists(), f"Missing public benchmark page: {BENCHMARK_PAGE}"


def test_benchmark_page_not_draft() -> None:
    text = BENCHMARK_PAGE.read_text(encoding="utf-8")
    assert "status: draft" not in text


@pytest.mark.parametrize("filename", ["run.py", "suite.py", "assess.py", "README.md"])
def test_suite_harness_files_exist(filename: str) -> None:
    path = SUITE_DIR / filename
    assert path.exists(), f"Missing suite file: {path}"


def test_competitors_named_in_benchmark_page() -> None:
    text = BENCHMARK_PAGE.read_text(encoding="utf-8").lower()
    for comp in EXPECTED_COMPETITORS:
        assert comp in text, f"Competitor {comp!r} not named in benchmark page."


def test_results_v10_artifacts_exist() -> None:
    assert (RESULTS_V10 / "bayesian_multiseed_summary.json").exists()
    assert (RESULTS_V10 / "derived_metrics.json").exists()


def test_benchmark_page_documents_promoted_cmdstan_subset() -> None:
    text = BENCHMARK_PAGE.read_text(encoding="utf-8").lower()
    assert "current promoted public proof subset" in text
    assert "cmdstan-backed" in text
    assert "pymc remains a supported optional harness path" in text


def test_results_v10_multiseed_summary_is_clean() -> None:
    data = json.loads((RESULTS_V10 / "bayesian_multiseed_summary.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == "nextstat.bayesian_multiseed_summary.v1"
    assert data["backends"] == "cmdstanpy,nextstat"
    for row in data["cases"]:
        assert set(row["statuses"]) == {"ok"}, f"Dirty row: {row['case']}::{row['backend']} -> {row['statuses']}"


def test_results_v10_covers_expected_cases() -> None:
    data = json.loads((RESULTS_V10 / "bayesian_multiseed_summary.json").read_text(encoding="utf-8"))
    cases = {row["case"] for row in data["cases"]}
    assert cases == EXPECTED_CASES


def test_results_v10_derived_metrics_are_cmdstan_backed() -> None:
    data = json.loads((RESULTS_V10 / "derived_metrics.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == "nextstat.bayesian_derived_metrics.v2"
    cases = data["ess_per_leapfrog"]["cases"]
    assert set(cases) == EXPECTED_CASES
    for row in cases.values():
        assert "cmdstan" in row
        assert "nextstat" in row


def test_nuts_sampler_reference_exists() -> None:
    assert NUTS_REF.exists(), f"Missing NUTS reference: {NUTS_REF}"


def test_nuts_reference_has_benchmark_numbers() -> None:
    text = NUTS_REF.read_text(encoding="utf-8").lower()
    assert "ess" in text
    assert "cmdstan" in text
    assert "results_v10" in text


def test_nuts_paper_exists() -> None:
    assert NUTS_PAPER.exists(), f"Missing NUTS paper: {NUTS_PAPER}"


def test_claim_matrix_has_bayesian() -> None:
    data = json.loads(CLAIM_MATRIX.read_text(encoding="utf-8"))
    domains = {e["domain"]: e for e in data["domains"]}
    assert "bayesian" in domains


def test_claim_matrix_bayesian_is_sota() -> None:
    data = json.loads(CLAIM_MATRIX.read_text(encoding="utf-8"))
    domains = {e["domain"]: e for e in data["domains"]}
    assert domains["bayesian"]["status"] == "sota"


def test_claim_matrix_bayesian_has_proof_refs() -> None:
    data = json.loads(CLAIM_MATRIX.read_text(encoding="utf-8"))
    domains = {e["domain"]: e for e in data["domains"]}
    entry = domains["bayesian"]
    assert len(entry.get("proof_refs", [])) >= 2
    for ref in entry["proof_refs"]:
        assert (REPO / ref).exists(), f"bayesian proof_ref {ref!r} does not exist"


def test_claim_matrix_bayesian_points_to_results_v10() -> None:
    data = json.loads(CLAIM_MATRIX.read_text(encoding="utf-8"))
    domains = {e["domain"]: e for e in data["domains"]}
    entry = domains["bayesian"]
    assert any("results_v10/bayesian_multiseed_summary.json" in ref for ref in entry["proof_refs"])
    assert any("results_v10/derived_metrics.json" in ref for ref in entry["proof_refs"])


def test_sbc_test_exists() -> None:
    sbc_test = REPO / "tests" / "python" / "test_sbc_nuts.py"
    assert sbc_test.exists(), "SBC NUTS test file must exist"


def test_sbc_harness_exists() -> None:
    sbc_harness = REPO / "tests" / "python" / "_sbc_sampler_suite.py"
    assert sbc_harness.exists(), "SBC sampler harness must exist"
