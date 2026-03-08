import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSESS_SCRIPT = REPO_ROOT / "benchmarks" / "nextstat-public-benchmarks" / "suites" / "bayesian" / "assess.py"
VALIDATE_SCRIPT = REPO_ROOT / "benchmarks" / "nextstat-public-benchmarks" / "scripts" / "validate_artifacts.py"


def _write_case(
    suite_dir: Path,
    *,
    case: str,
    backend: str,
    status: str,
    divergence_rate: float,
    max_treedepth_rate: float,
    max_r_hat: float,
    min_ess_bulk: float,
    min_ess_tail: float,
    min_ebfmi: float,
    min_ess_bulk_per_sec: float,
    reason: str | None = None,
) -> str:
    rel_path = f"cases/{case}__{backend}.json"
    case_path = suite_dir / rel_path
    case_path.parent.mkdir(parents=True, exist_ok=True)
    case_doc = {
        "schema_version": "nextstat.bayesian_benchmark_result.v1",
        "suite": "bayesian",
        "case": case,
        "backend": backend,
        "deterministic": True,
        "status": status,
        "reason": reason,
        "environment": {"node": "nextstat-bench"},
        "meta": {"python": "3.13.0", "platform": "test-platform", "nextstat_version": "0.9.9"},
        "dataset": {"id": f"generated:{case}", "sha256": "a" * 64},
        "config": {
            "n_chains": 4,
            "n_warmup": 500,
            "n_samples": 1000,
            "seed": 42,
            "dataset_seed": 12345,
            "max_treedepth": 10,
            "target_accept": 0.8,
            "init_jitter_rel": 0.1,
        },
        "timing": {
            "wall_time_s": 1.0,
            "ess_bulk_per_sec": {"min": min_ess_bulk_per_sec, "median": min_ess_bulk_per_sec},
        },
        "diagnostics_summary": {
            "divergence_rate": divergence_rate,
            "max_treedepth_rate": max_treedepth_rate,
            "max_r_hat": max_r_hat,
            "min_ess_bulk": min_ess_bulk,
            "min_ess_tail": min_ess_tail,
            "min_ebfmi": min_ebfmi,
        },
    }
    case_path.write_text(json.dumps(case_doc) + "\n", encoding="utf-8")
    return rel_path


def _write_suite(tmp_path: Path, *, n_failed: int = 0, target_case_status: str = "ok") -> Path:
    suite_dir = tmp_path / "bayesian"
    suite_dir.mkdir()
    path_ok = _write_case(
        suite_dir,
        case="glm_logistic_regression",
        backend="nextstat",
        status=target_case_status,
        reason="sample_failed" if target_case_status != "ok" else None,
        divergence_rate=0.0,
        max_treedepth_rate=0.0,
        max_r_hat=1.002,
        min_ess_bulk=2000.0,
        min_ess_tail=1500.0,
        min_ebfmi=0.7,
        min_ess_bulk_per_sec=5000.0,
    )
    path_fail_gate = _write_case(
        suite_dir,
        case="eight_schools_non_centered",
        backend="nextstat",
        status="ok",
        divergence_rate=0.0,
        max_treedepth_rate=0.0,
        max_r_hat=1.015,
        min_ess_bulk=1200.0,
        min_ess_tail=900.0,
        min_ebfmi=0.8,
        min_ess_bulk_per_sec=1500.0,
    )
    suite = {
        "schema_version": "nextstat.bayesian_benchmark_suite_result.v1",
        "suite": "bayesian",
        "deterministic": True,
        "meta": {
            "python": "3.13.0",
            "platform": "test-platform",
            "nextstat_version": "0.9.9",
        },
        "cases": [
            {
                "case": "glm_logistic_regression",
                "backend": "nextstat",
                "path": path_ok,
                "sha256": "0" * 64,
                "status": target_case_status,
                "wall_time_s": 1.0,
                "min_ess_bulk": 2000.0,
                "min_ess_tail": 1500.0,
                "max_r_hat": 1.002,
                "min_ess_bulk_per_sec": 5000.0,
            },
            {
                "case": "eight_schools_non_centered",
                "backend": "nextstat",
                "path": path_fail_gate,
                "sha256": "1" * 64,
                "status": "ok",
                "wall_time_s": 1.0,
                "min_ess_bulk": 1200.0,
                "min_ess_tail": 900.0,
                "max_r_hat": 1.015,
                "min_ess_bulk_per_sec": 1500.0,
            },
        ],
        "summary": {
            "n_cases": 2,
            "n_ok": 2 - n_failed,
            "n_warn": 0,
            "n_failed": n_failed,
            "worst_case": "eight_schools_non_centered::nextstat",
            "n_parity_warn": 0,
            "n_parity_fail": 0,
        },
        "parity": {
            "method": "mean_zscore",
            "compare": "nextstat_dense vs nextstat",
            "warn_z": 8.0,
            "fail_z": 12.0,
            "rows": [],
        },
    }
    (suite_dir / "bayesian_suite.json").write_text(json.dumps(suite) + "\n", encoding="utf-8")
    return suite_dir


def test_bayesian_assessment_separates_core_quality_from_promotion_gate(tmp_path: Path) -> None:
    suite_dir = _write_suite(tmp_path)

    subprocess.run([sys.executable, str(ASSESS_SCRIPT), str(suite_dir)], check=True)
    subprocess.run([sys.executable, str(VALIDATE_SCRIPT), "--strict", str(suite_dir / "bayesian_assessment.json")], check=True)

    assessment = json.loads((suite_dir / "bayesian_assessment.json").read_text(encoding="utf-8"))
    assert assessment["core_quality"]["passed"] is True
    assert assessment["promotion_gate"]["passed"] is False
    assert assessment["promotion_gate"]["review_summary"]["n_reviewed_cases"] == 2
    assert assessment["promotion_gate"]["review_summary"]["n_failing_cases"] == 1
    assert assessment["promotion_gate"]["review_summary"]["failing_cases"] == ["eight_schools_non_centered"]
    assert assessment["promotion_gate"]["review_summary"]["worst_max_r_hat"] == {
        "case": "eight_schools_non_centered",
        "value": 1.015,
    }
    assert any(
        failure.get("case") == "eight_schools_non_centered"
        and failure.get("reason") == "max_r_hat_exceeds_threshold"
        for failure in assessment["promotion_gate"]["failures"]
    )

    subprocess.run(
        [
            sys.executable,
            str(ASSESS_SCRIPT),
            str(suite_dir),
            "--promotion-max-rhat",
            "1.02",
        ],
        check=True,
    )
    relaxed = json.loads((suite_dir / "bayesian_assessment.json").read_text(encoding="utf-8"))
    assert relaxed["core_quality"]["passed"] is True
    assert relaxed["promotion_gate"]["passed"] is True


def test_bayesian_assessment_fails_promotion_when_core_quality_is_invalid(tmp_path: Path) -> None:
    suite_dir = _write_suite(tmp_path, n_failed=1, target_case_status="failed")

    subprocess.run([sys.executable, str(ASSESS_SCRIPT), str(suite_dir)], check=True)

    assessment = json.loads((suite_dir / "bayesian_assessment.json").read_text(encoding="utf-8"))
    assert assessment["core_quality"]["passed"] is False
    assert assessment["promotion_gate"]["passed"] is False
    assert any(
        failure.get("reason") == "core_quality_not_valid"
        for failure in assessment["promotion_gate"]["failures"]
    )
