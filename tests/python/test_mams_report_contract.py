import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_SCRIPT = REPO_ROOT / "benchmarks" / "nextstat-public-benchmarks" / "suites" / "mams" / "report.py"


def test_mams_snapshot_snippet_surfaces_health_verdict_from_assessment(tmp_path: Path) -> None:
    suite_path = tmp_path / "mams_suite.json"
    assessment_path = tmp_path / "mams_assessment.json"
    out_path = tmp_path / "README_snippet_mams.md"

    suite_doc = {
        "schema_version": "nextstat.mams_benchmark_suite_result.v1",
        "suite": "mams",
        "meta": {"python": "3.13.0", "platform": "test-platform", "nextstat_version": "0.9.9"},
        "config": {"n_chains": 4, "n_warmup": 1000, "n_samples": 2000, "target_accept": 0.9},
        "cases": [
            {
                "case": "std_normal_10d",
                "backend": "nextstat_mams",
                "status": "ok",
                "wall_time_s": 0.1,
                "n_grad_evals": 1000,
                "min_ess_bulk": 3000.0,
                "ess_per_grad": 0.25,
                "ess_per_sec": 25000.0,
                "max_r_hat": 1.0024,
            },
            {
                "case": "std_normal_10d",
                "backend": "nextstat_nuts",
                "status": "ok",
                "wall_time_s": 0.2,
                "n_grad_evals": 1500,
                "min_ess_bulk": 3500.0,
                "ess_per_grad": 0.15,
                "ess_per_sec": 20000.0,
                "max_r_hat": 1.001,
            },
        ],
        "parity": {"warn_z": 8.0, "fail_z": 12.0, "rows": []},
    }
    suite_path.write_text(json.dumps(suite_doc) + "\n", encoding="utf-8")

    assessment_doc = {
        "schema_version": "nextstat.mams_assessment.v1",
        "suite": "mams",
        "source_suite_path": "mams_suite.json",
        "source_suite_sha256": "a" * 64,
        "source_suite_summary": {"n_failed": 0, "n_warn": 0, "n_parity_fail": 0, "n_parity_warn": 0},
        "core_quality": {"passed": True, "status": "passed", "failures": [], "warnings": []},
        "promotion_gate": {
            "passed": False,
            "status": "failed",
            "target_backend": "nextstat_mams",
            "policy": {"max_r_hat": 1.01, "min_ess_bulk": None, "min_ess_per_sec": None},
            "reviewed_cases": [
                {
                    "case": "std_normal_10d",
                    "status": "ok",
                    "max_r_hat": 1.0142,
                    "min_ess_bulk": 1200.0,
                    "ess_per_sec": 500.0,
                }
            ],
            "review_summary": {
                "n_reviewed_cases": 1,
                "n_failures": 1,
                "n_failing_cases": 1,
                "failing_cases": ["std_normal_10d"],
                "worst_max_r_hat": {"case": "std_normal_10d", "value": 1.0142},
                "worst_min_ess_bulk": {"case": "std_normal_10d", "value": 1200.0},
                "worst_ess_per_sec": {"case": "std_normal_10d", "value": 500.0},
            },
            "failures": [
                {
                    "case": "std_normal_10d",
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
            "--snippet",
        ],
        check=True,
    )

    md = out_path.read_text(encoding="utf-8")
    assert "## Health verdict" in md
    assert "promotion gate (nextstat_mams): `failed`" in md
    assert "failing cases: `std_normal_10d`" in md
    assert "1.0142" in md
    assert "1,200" in md
    assert "500" in md


def test_mams_detailed_report_default_path_is_preserved(tmp_path: Path) -> None:
    suite_dir = tmp_path / "mams"
    suite_dir.mkdir()
    suite_doc = {
        "schema_version": "nextstat.mams_benchmark_suite_result.v1",
        "suite": "mams",
        "meta": {"python": "3.13.0", "platform": "test-platform", "nextstat_version": "0.9.9"},
        "config": {"n_chains": 4, "n_warmup": 1000, "n_samples": 2000, "target_accept": 0.9},
        "cases": [],
        "parity": {"rows": []},
    }
    (suite_dir / "mams_suite.json").write_text(json.dumps(suite_doc) + "\n", encoding="utf-8")

    subprocess.run([sys.executable, str(REPORT_SCRIPT), str(suite_dir)], check=True)

    md = (suite_dir / "mams_benchmark_report.md").read_text(encoding="utf-8")
    assert md.startswith("# MAMS Benchmark Suite Results")
