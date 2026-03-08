import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_SCRIPT = REPO_ROOT / "benchmarks" / "nextstat-public-benchmarks" / "suites" / "bayesian" / "report.py"


def test_bayesian_report_surfaces_health_verdict_from_assessment(tmp_path: Path) -> None:
    suite_path = tmp_path / "bayesian_suite.json"
    assessment_path = tmp_path / "bayesian_assessment.json"
    out_path = tmp_path / "README_snippet_bayesian.md"

    suite_doc = {
        "schema_version": "nextstat.bayesian_benchmark_suite_result.v1",
        "suite": "bayesian",
        "meta": {"python": "3.13.0", "platform": "test-platform", "nextstat_version": "0.9.9"},
        "cases": [
            {
                "case": "hier_random_intercept_non_centered",
                "backend": "nextstat",
                "status": "ok",
                "wall_time_s": 1.0,
                "min_ess_bulk": 1200.0,
                "min_ess_tail": 900.0,
                "max_r_hat": 1.002,
                "ess_per_grad": 0.5,
                "grad_per_sec": 1000.0,
                "min_ess_bulk_per_sec": 500.0,
            }
        ],
        "parity": {"rows": []},
    }
    suite_path.write_text(json.dumps(suite_doc) + "\n", encoding="utf-8")

    assessment_doc = {
        "schema_version": "nextstat.bayesian_assessment.v1",
        "suite": "bayesian",
        "source_suite_path": "bayesian_suite.json",
        "source_suite_sha256": "a" * 64,
        "source_suite_summary": {"n_failed": 0, "n_warn": 0, "n_parity_fail": 0, "n_parity_warn": 0},
        "parity_summary": {"compare": "nextstat_dense vs nextstat", "method": "mean_zscore", "warn_z": 8.0, "fail_z": 12.0, "n_rows": 0},
        "core_quality": {"passed": True, "status": "passed", "failures": [], "warnings": []},
        "promotion_gate": {
            "passed": False,
            "status": "failed",
            "target_backend": "nextstat",
            "policy": {
                "max_r_hat": 1.01,
                "max_divergence_rate": 0.0,
                "max_treedepth_rate": 0.0,
                "min_ebfmi": 0.3,
                "min_ess_bulk": None,
                "min_ess_tail": None,
                "min_ess_bulk_per_sec": None,
            },
            "reviewed_cases": [],
            "review_summary": {
                "n_reviewed_cases": 1,
                "n_failures": 1,
                "n_failing_cases": 1,
                "failing_cases": ["hier_random_intercept_non_centered"],
                "worst_divergence_rate": {"case": "hier_random_intercept_non_centered", "value": 0.000125},
                "worst_max_treedepth_rate": {"case": "hier_random_intercept_non_centered", "value": 0.0},
                "worst_max_r_hat": {"case": "hier_random_intercept_non_centered", "value": 1.0142},
                "worst_min_ess_bulk": {"case": "hier_random_intercept_non_centered", "value": 1200.0},
                "worst_min_ess_tail": {"case": "hier_random_intercept_non_centered", "value": 900.0},
                "worst_min_ess_bulk_per_sec": {"case": "hier_random_intercept_non_centered", "value": 500.0},
                "worst_min_ebfmi": {"case": "hier_random_intercept_non_centered", "value": 0.57},
            },
            "failures": [
                {
                    "case": "hier_random_intercept_non_centered",
                    "reason": "max_r_hat_exceeds_threshold",
                    "observed": 1.0142,
                    "threshold": 1.01,
                }
            ],
        },
    }
    assessment_path.write_text(json.dumps(assessment_doc) + "\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(REPORT_SCRIPT),
            "--suite",
            str(suite_path),
            "--assessment",
            str(assessment_path),
            "--out",
            str(out_path),
        ],
        check=True,
    )

    md = out_path.read_text(encoding="utf-8")
    assert "## Health verdict" in md
    assert "promotion gate (nextstat): `failed`" in md
    assert "0.000125" in md
    assert "hier_random_intercept_non_centered" in md
