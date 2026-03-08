import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSESS_SCRIPT = REPO_ROOT / "benchmarks" / "nextstat-public-benchmarks" / "suites" / "mams" / "assess.py"


def _write_suite(tmp_path: Path, *, n_failed: int = 0, case_status: str = "ok") -> Path:
    suite_dir = tmp_path / "mams"
    suite_dir.mkdir()
    suite = {
        "schema_version": "nextstat.mams_benchmark_suite_result.v1",
        "suite": "mams",
        "deterministic": True,
        "meta": {
            "python": "3.13.0",
            "platform": "test-platform",
            "nextstat_version": "0.9.9",
        },
        "config": {
            "n_chains": 4,
            "n_warmup": 1000,
            "n_samples": 2000,
            "target_accept": 0.9,
            "seeds": [42],
            "backends": ["nextstat_mams", "nextstat_nuts"],
        },
        "cases": [
            {
                "case": "std_normal_10d",
                "backend": "nextstat_mams",
                "seed": 42,
                "path": "cases/std_normal_10d__nextstat_mams__s42.json",
                "sha256": "0" * 64,
                "status": case_status,
                "wall_time_s": 0.1,
                "n_grad_evals": 1000,
                "min_ess_bulk": 3000.0,
                "ess_per_grad": 0.25,
                "ess_per_sec": 25000.0,
                "max_r_hat": 1.0024,
            },
            {
                "case": "neal_funnel_2d",
                "backend": "nextstat_mams",
                "seed": 42,
                "path": "cases/neal_funnel_2d__nextstat_mams__s42.json",
                "sha256": "1" * 64,
                "status": "ok",
                "wall_time_s": 0.12,
                "n_grad_evals": 1000,
                "min_ess_bulk": 348.0,
                "ess_per_grad": 0.0018,
                "ess_per_sec": 2959.0,
                "max_r_hat": 1.0267,
            },
        ],
        "summary": {
            "n_total": 2,
            "n_ok": 2 - n_failed,
            "n_warn": 0,
            "n_failed": n_failed,
            "n_parity_warn": 0,
            "n_parity_fail": 0,
        },
        "parity": {
            "method": "mean_zscore",
            "note": "test fixture",
            "warn_z": 8.0,
            "fail_z": 12.0,
            "rows": [],
        },
    }
    (suite_dir / "mams_suite.json").write_text(json.dumps(suite) + "\n", encoding="utf-8")
    return suite_dir


def test_mams_assessment_separates_core_quality_from_promotion_gate(tmp_path: Path) -> None:
    suite_dir = _write_suite(tmp_path)

    subprocess.run([sys.executable, str(ASSESS_SCRIPT), str(suite_dir)], check=True)

    assessment = json.loads((suite_dir / "mams_assessment.json").read_text(encoding="utf-8"))
    assert assessment["core_quality"]["passed"] is True
    assert assessment["promotion_gate"]["passed"] is False
    failures = assessment["promotion_gate"]["failures"]
    assert any(
        failure.get("case") == "neal_funnel_2d"
        and failure.get("reason") == "max_r_hat_exceeds_threshold"
        for failure in failures
    )
    review_summary = assessment["promotion_gate"]["review_summary"]
    assert review_summary["n_reviewed_cases"] == 2
    assert review_summary["failing_cases"] == ["neal_funnel_2d"]
    assert review_summary["worst_max_r_hat"]["case"] == "neal_funnel_2d"
    assert review_summary["worst_min_ess_bulk"]["case"] == "neal_funnel_2d"
    assert review_summary["worst_ess_per_sec"]["case"] == "neal_funnel_2d"

    subprocess.run(
        [
            sys.executable,
            str(ASSESS_SCRIPT),
            str(suite_dir),
            "--promotion-max-rhat",
            "1.05",
        ],
        check=True,
    )

    relaxed = json.loads((suite_dir / "mams_assessment.json").read_text(encoding="utf-8"))
    assert relaxed["core_quality"]["passed"] is True
    assert relaxed["promotion_gate"]["passed"] is True


def test_mams_assessment_fails_promotion_when_core_quality_is_invalid(tmp_path: Path) -> None:
    suite_dir = _write_suite(tmp_path, n_failed=1, case_status="failed")

    subprocess.run([sys.executable, str(ASSESS_SCRIPT), str(suite_dir)], check=True)

    assessment = json.loads((suite_dir / "mams_assessment.json").read_text(encoding="utf-8"))
    assert assessment["core_quality"]["passed"] is False
    assert assessment["promotion_gate"]["passed"] is False
    assert any(
        failure.get("reason") == "core_quality_not_valid"
        for failure in assessment["promotion_gate"]["failures"]
    )
    assert assessment["promotion_gate"]["review_summary"]["n_failures"] >= 1
