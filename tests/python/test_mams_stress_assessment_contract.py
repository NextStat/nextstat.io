import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSESS_SCRIPT = REPO_ROOT / "benchmarks" / "nextstat-public-benchmarks" / "suites" / "mams" / "assess_stress_multiseed.py"


def test_mams_stress_assessment_passes_supported_and_controls(tmp_path: Path) -> None:
    out_dir = tmp_path / "stress"
    out_dir.mkdir(parents=True)

    summary = {
        "schema_version": "nextstat.mams_stress_multiseed_summary.v1",
        "generated_at": "2026-03-09T00:00:00Z",
        "suite": "mams_stress",
        "seeds": [42, 0, 123],
        "backends": "nextstat_mams,nextstat_nuts",
        "config": {
            "n_chains": 4,
            "n_warmup": 3500,
            "n_samples": 2000,
            "dataset_seed": 12345,
            "target_accept": 0.985,
            "run_timeout_s": 300.0,
            "parity_warn_z": 8.0,
            "parity_fail_z": 12.0,
            "n_groups": 20,
            "n_per_group": 20,
            "deterministic": True,
        },
        "case_catalog": [
            {"case": "neal_funnel_ncp_10d", "case_tier": "supported", "parity_scope": "required", "description": "supported"},
            {"case": "hier_random_intercept_non_centered", "case_tier": "supported", "parity_scope": "required", "description": "supported"},
            {"case": "neal_funnel_10d_centered", "case_tier": "pathological_control", "parity_scope": "informational", "description": "control"},
        ],
        "cases": [
            {
                "case": "neal_funnel_ncp_10d",
                "case_tier": "supported",
                "backend": "nextstat_mams",
                "statuses": ["ok", "ok", "ok"],
                "wall_time_s": [1.0, 1.0, 1.0],
                "n_grad_evals": [2000.0, 2000.0, 2000.0],
                "n_integration_steps": [1998.0, 1998.0, 1998.0],
                "ess_per_grad": [0.1, 0.11, 0.12],
                "ess_per_sec": [2000.0, 1900.0, 2100.0],
                "ess_per_sec_warm": [2200.0, 2100.0, 2300.0],
                "min_ess_bulk": [1800.0, 1750.0, 1700.0],
                "min_ess_tail": [1500.0, 1450.0, 1400.0],
                "max_r_hat": [1.003, 1.004, 1.005],
                "accept_rate": [0.98, 0.981, 0.979],
                "configs": [{}, {}, {}],
            },
            {
                "case": "hier_random_intercept_non_centered",
                "case_tier": "supported",
                "backend": "nextstat_mams",
                "statuses": ["ok", "ok", "ok"],
                "wall_time_s": [1.5, 1.4, 1.6],
                "n_grad_evals": [2600.0, 2550.0, 2500.0],
                "n_integration_steps": [2598.0, 2548.0, 2498.0],
                "ess_per_grad": [0.06, 0.061, 0.063],
                "ess_per_sec": [1000.0, 950.0, 980.0],
                "ess_per_sec_warm": [1100.0, 1050.0, 1080.0],
                "min_ess_bulk": [1200.0, 1180.0, 1150.0],
                "min_ess_tail": [900.0, 880.0, 870.0],
                "max_r_hat": [1.008, 1.009, 1.007],
                "accept_rate": [0.977, 0.978, 0.979],
                "configs": [{}, {}, {}],
            },
            {
                "case": "neal_funnel_10d_centered",
                "case_tier": "pathological_control",
                "backend": "nextstat_mams",
                "statuses": ["warn", "warn", "ok"],
                "wall_time_s": [2.0, 2.1, 2.2],
                "n_grad_evals": [3000.0, 3050.0, 3100.0],
                "n_integration_steps": [2998.0, 3048.0, 3098.0],
                "ess_per_grad": [0.03, 0.031, 0.032],
                "ess_per_sec": [500.0, 510.0, 520.0],
                "ess_per_sec_warm": [550.0, 560.0, 570.0],
                "min_ess_bulk": [650.0, 640.0, 630.0],
                "min_ess_tail": [400.0, 390.0, 380.0],
                "max_r_hat": [1.12, 1.10, 1.08],
                "accept_rate": [0.96, 0.95, 0.97],
                "configs": [{}, {}, {}],
            },
        ],
        "parity": {
            "method": "mean_zscore",
            "note": "test",
            "warn_z": 8.0,
            "fail_z": 12.0,
            "rows": [
                {
                    "case": "neal_funnel_ncp_10d",
                    "case_tier": "supported",
                    "parity_scope": "required",
                    "statuses": ["ok", "ok", "ok"],
                    "max_z": [1.0, 1.1, 1.2],
                    "seed_rows": [],
                },
                {
                    "case": "hier_random_intercept_non_centered",
                    "case_tier": "supported",
                    "parity_scope": "required",
                    "statuses": ["ok", "ok", "ok"],
                    "max_z": [1.3, 1.4, 1.2],
                    "seed_rows": [],
                },
                {
                    "case": "neal_funnel_10d_centered",
                    "case_tier": "pathological_control",
                    "parity_scope": "informational",
                    "statuses": ["warn", "failed", "warn"],
                    "max_z": [9.0, 13.0, 11.0],
                    "seed_rows": [],
                },
            ],
        },
    }
    (out_dir / "mams_stress_multiseed_summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")

    completed = subprocess.run([sys.executable, str(ASSESS_SCRIPT), str(out_dir)], check=False)
    assert completed.returncode == 0

    assessment = json.loads((out_dir / "mams_stress_assessment.json").read_text(encoding="utf-8"))
    assert assessment["stress_readiness"]["status"] == "passed"
    assert assessment["supported_repeatability_gate"]["status"] == "passed"
    assert assessment["pathological_control_health"]["status"] == "passed"

    md = (out_dir / "mams_stress_assessment.md").read_text(encoding="utf-8")
    assert "## Supported repeatability gate" in md
    assert "## Pathological controls" in md


def test_mams_stress_assessment_fails_supported_and_control_health(tmp_path: Path) -> None:
    out_dir = tmp_path / "stress"
    out_dir.mkdir(parents=True)

    summary = {
        "schema_version": "nextstat.mams_stress_multiseed_summary.v1",
        "generated_at": "2026-03-09T00:00:00Z",
        "suite": "mams_stress",
        "seeds": [42, 0, 123],
        "backends": "nextstat_mams,nextstat_nuts",
        "config": {
            "n_chains": 4,
            "n_warmup": 3500,
            "n_samples": 2000,
            "dataset_seed": 12345,
            "target_accept": 0.985,
            "run_timeout_s": 300.0,
            "parity_warn_z": 8.0,
            "parity_fail_z": 12.0,
            "n_groups": 20,
            "n_per_group": 20,
            "deterministic": True,
        },
        "case_catalog": [
            {"case": "neal_funnel_ncp_10d", "case_tier": "supported", "parity_scope": "required", "description": "supported"},
            {"case": "neal_funnel_10d_centered", "case_tier": "pathological_control", "parity_scope": "informational", "description": "control"},
        ],
        "cases": [
            {
                "case": "neal_funnel_ncp_10d",
                "case_tier": "supported",
                "backend": "nextstat_mams",
                "statuses": ["ok", "warn", "ok"],
                "wall_time_s": [1.0, 1.0, 1.0],
                "n_grad_evals": [2000.0, 2000.0, 2000.0],
                "n_integration_steps": [1998.0, 1998.0, 1998.0],
                "ess_per_grad": [0.1, 0.1, 0.1],
                "ess_per_sec": [1800.0, 1750.0, 1700.0],
                "ess_per_sec_warm": [1900.0, 1850.0, 1800.0],
                "min_ess_bulk": [1200.0, 1180.0, 1190.0],
                "min_ess_tail": [900.0, 890.0, 880.0],
                "max_r_hat": [1.005, 1.013, 1.004],
                "accept_rate": [0.98, 0.97, 0.98],
                "configs": [{}, {}, {}],
            },
            {
                "case": "neal_funnel_10d_centered",
                "case_tier": "pathological_control",
                "backend": "nextstat_mams",
                "statuses": ["warn", "failed", "warn"],
                "wall_time_s": [2.0, 2.1, 2.2],
                "n_grad_evals": [3000.0, 3050.0, 3100.0],
                "n_integration_steps": [2998.0, 3048.0, 3098.0],
                "ess_per_grad": [0.03, 0.031, 0.032],
                "ess_per_sec": [500.0, 510.0, 520.0],
                "ess_per_sec_warm": [550.0, 560.0, 570.0],
                "min_ess_bulk": [650.0, 640.0, 630.0],
                "min_ess_tail": [400.0, 390.0, 380.0],
                "max_r_hat": [1.12, 1.10, 1.08],
                "accept_rate": [0.96, 0.95, 0.97],
                "configs": [{}, {}, {}],
            },
        ],
        "parity": {
            "method": "mean_zscore",
            "note": "test",
            "warn_z": 8.0,
            "fail_z": 12.0,
            "rows": [
                {
                    "case": "neal_funnel_ncp_10d",
                    "case_tier": "supported",
                    "parity_scope": "required",
                    "statuses": ["ok", "warn", "ok"],
                    "max_z": [1.0, 8.5, 1.2],
                    "seed_rows": [],
                }
            ],
        },
    }
    (out_dir / "mams_stress_multiseed_summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")

    completed = subprocess.run([sys.executable, str(ASSESS_SCRIPT), str(out_dir)], check=False)
    assert completed.returncode == 0

    assessment = json.loads((out_dir / "mams_stress_assessment.json").read_text(encoding="utf-8"))
    assert assessment["stress_readiness"]["status"] == "failed"
    assert assessment["supported_repeatability_gate"]["status"] == "failed"
    assert assessment["pathological_control_health"]["status"] == "failed"
    reasons = {row["reason"] for row in assessment["supported_repeatability_gate"]["failures"]}
    assert "backend_statuses_not_all_ok" in reasons
    assert "max_r_hat_exceeds_threshold" in reasons
    assert "parity_statuses_not_all_ok" in reasons
    control_reasons = {row["reason"] for row in assessment["pathological_control_health"]["failures"]}
    assert "backend_status_failed_on_control" in control_reasons
