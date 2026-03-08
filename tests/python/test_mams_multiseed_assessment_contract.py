import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSESS_SCRIPT = (
    REPO_ROOT / "benchmarks" / "nextstat-public-benchmarks" / "suites" / "mams" / "assess_multiseed.py"
)


def test_mams_multiseed_assessment_passes_clean_repeatability_bundle(tmp_path: Path) -> None:
    out_dir = tmp_path / "multiseed"
    out_dir.mkdir(parents=True)

    summary = {
        "schema_version": "nextstat.mams_multiseed_summary.v1",
        "generated_at": "2026-03-09T00:00:00Z",
        "suite": "mams",
        "seeds": [42, 0, 123],
        "backends": "nextstat_mams,nextstat_nuts",
        "config": {
            "n_chains": 4,
            "n_warmup": 2000,
            "n_samples": 2000,
            "dataset_seed": 12345,
            "target_accept": 0.985,
            "run_timeout_s": 300.0,
            "parity_warn_z": 8.0,
            "parity_fail_z": 12.0,
            "deterministic": True,
        },
        "cases": [
            {
                "case": "glm_logistic",
                "backend": "nextstat_mams",
                "statuses": ["ok", "ok", "ok"],
                "wall_time_s": [1.0, 1.1, 0.9],
                "n_grad_evals": [2000.0, 2200.0, 2100.0],
                "n_integration_steps": [1998.0, 2198.0, 2098.0],
                "ess_per_grad": [0.08, 0.09, 0.085],
                "ess_per_sec": [1800.0, 1700.0, 1900.0],
                "ess_per_sec_warm": [1900.0, 1800.0, 2000.0],
                "min_ess_bulk": [1700.0, 1650.0, 1750.0],
                "min_ess_tail": [2200.0, 2150.0, 2250.0],
                "max_r_hat": [1.003, 1.005, 1.004],
                "accept_rate": [0.98, 0.979, 0.981],
                "configs": [{}, {}, {}],
            }
        ],
        "parity": {
            "method": "mean_zscore",
            "warn_z": 8.0,
            "fail_z": 12.0,
            "rows": [
                {
                    "case": "glm_logistic",
                    "statuses": ["ok", "ok", "ok"],
                    "max_z": [1.1, 1.4, 0.9],
                    "seed_rows": [],
                }
            ],
        },
    }
    (out_dir / "mams_multiseed_summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")

    completed = subprocess.run([sys.executable, str(ASSESS_SCRIPT), str(out_dir)], check=False)
    assert completed.returncode == 0

    assessment = json.loads((out_dir / "mams_multiseed_assessment.json").read_text(encoding="utf-8"))
    gate = assessment["repeatability_gate"]
    assert gate["status"] == "passed"
    assert gate["review_summary"]["n_reviewed_cases"] == 1
    assert gate["review_summary"]["n_reviewed_parity_cases"] == 1
    assert gate["review_summary"]["failing_cases"] == []
    assert gate["review_summary"]["worst_max_r_hat"] == {"case": "glm_logistic", "value": 1.005}
    assert gate["review_summary"]["worst_parity_max_z"] == {"case": "glm_logistic", "value": 1.4}

    md = (out_dir / "mams_multiseed_assessment.md").read_text(encoding="utf-8")
    assert "## Repeatability gate" in md
    assert "status: `passed`" in md


def test_mams_multiseed_assessment_surfaces_rhat_and_parity_failures(tmp_path: Path) -> None:
    out_dir = tmp_path / "multiseed"
    out_dir.mkdir(parents=True)

    summary = {
        "schema_version": "nextstat.mams_multiseed_summary.v1",
        "generated_at": "2026-03-09T00:00:00Z",
        "suite": "mams",
        "seeds": [42, 0, 123],
        "backends": "nextstat_mams,nextstat_nuts",
        "config": {
            "n_chains": 4,
            "n_warmup": 2000,
            "n_samples": 2000,
            "dataset_seed": 12345,
            "target_accept": 0.985,
            "run_timeout_s": 300.0,
            "parity_warn_z": 8.0,
            "parity_fail_z": 12.0,
            "deterministic": True,
        },
        "cases": [
            {
                "case": "neal_funnel_2d",
                "backend": "nextstat_mams",
                "statuses": ["ok", "ok", "ok"],
                "wall_time_s": [1.0, 1.0, 1.0],
                "n_grad_evals": [2000.0, 2000.0, 2000.0],
                "n_integration_steps": [1998.0, 1998.0, 1998.0],
                "ess_per_grad": [0.08, 0.08, 0.08],
                "ess_per_sec": [1500.0, 1400.0, 1450.0],
                "ess_per_sec_warm": [1600.0, 1500.0, 1550.0],
                "min_ess_bulk": [800.0, 780.0, 790.0],
                "min_ess_tail": [900.0, 880.0, 890.0],
                "max_r_hat": [1.009, 1.01094, 1.008],
                "accept_rate": [0.98, 0.977, 0.979],
                "configs": [{}, {}, {}],
            }
        ],
        "parity": {
            "method": "mean_zscore",
            "warn_z": 8.0,
            "fail_z": 12.0,
            "rows": [
                {
                    "case": "neal_funnel_2d",
                    "statuses": ["ok", "warn", "ok"],
                    "max_z": [3.5, 8.5, 2.7],
                    "seed_rows": [],
                }
            ],
        },
    }
    (out_dir / "mams_multiseed_summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")

    completed = subprocess.run([sys.executable, str(ASSESS_SCRIPT), str(out_dir)], check=False)
    assert completed.returncode == 0

    assessment = json.loads((out_dir / "mams_multiseed_assessment.json").read_text(encoding="utf-8"))
    gate = assessment["repeatability_gate"]
    assert gate["status"] == "failed"
    assert gate["review_summary"]["failing_cases"] == ["neal_funnel_2d"]
    reasons = {row["reason"] for row in gate["failures"]}
    assert "max_r_hat_exceeds_threshold" in reasons
    assert "parity_statuses_not_all_ok" in reasons
    assert gate["review_summary"]["worst_max_r_hat"] == {"case": "neal_funnel_2d", "value": 1.01094}
    assert gate["review_summary"]["worst_parity_max_z"] == {"case": "neal_funnel_2d", "value": 8.5}

    md = (out_dir / "mams_multiseed_assessment.md").read_text(encoding="utf-8")
    assert "### Repeatability failures" in md
    assert "parity_statuses_not_all_ok" in md
