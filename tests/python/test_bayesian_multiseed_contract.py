import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT / "benchmarks" / "nextstat-public-benchmarks" / "suites" / "bayesian" / "multiseed.py"
)


def test_multiseed_reuse_existing_surfaces_health_metrics(tmp_path: Path) -> None:
    out_dir = tmp_path / "multiseed"
    seed_dir = out_dir / "seed_42"
    cases_dir = seed_dir / "cases"
    cases_dir.mkdir(parents=True)

    suite_obj = {
        "schema_version": "nextstat.bayesian_benchmark_suite_result.v1",
        "suite": "bayesian",
        "cases": [
            {
                "case": "histfactory_simple_8p",
                "backend": "nextstat",
                "status": "ok",
                "wall_time_s": 1.0,
                "min_ess_bulk": 500.0,
                "min_ess_tail": 450.0,
                "max_r_hat": 1.001,
                "min_ess_bulk_per_sec": 500.0,
                "path": "cases/histfactory_simple_8p__nextstat.json",
            }
        ],
    }
    (seed_dir / "bayesian_suite.json").write_text(json.dumps(suite_obj) + "\n")

    case_obj = {
        "schema_version": "nextstat.bayesian_benchmark_result.v1",
        "suite": "bayesian",
        "case": "histfactory_simple_8p",
        "backend": "nextstat",
        "config": {
            "dataset_seed": 12345,
            "init_jitter_rel": 0.1,
            "max_treedepth": 10,
            "metric": "diagonal",
            "n_chains": 4,
            "n_samples": 2000,
            "n_warmup": 1000,
            "seed": 42,
            "target_accept": 0.8,
        },
        "timing": {
            "wall_time_s": 1.0,
            "ess_bulk_per_sec": {"min": 500.0},
            "ess_tail_per_sec": {"min": 450.0},
        },
        "diagnostics_summary": {
            "divergence_rate": 0.000125,
            "max_treedepth_rate": 0.0,
            "max_r_hat": 1.001,
            "min_ess_bulk": 500.0,
            "min_ess_tail": 450.0,
            "min_ebfmi": 0.7,
        },
    }
    (cases_dir / "histfactory_simple_8p__nextstat.json").write_text(json.dumps(case_obj) + "\n")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--out-dir",
            str(out_dir),
            "--seeds",
            "42",
            "--reuse-existing",
        ],
        check=False,
    )
    assert completed.returncode == 0

    summary = json.loads((out_dir / "bayesian_multiseed_summary.json").read_text())
    assert summary["backends"] == "nextstat"
    row = summary["cases"][0]
    assert row["divergence_rate"] == [0.000125]
    assert row["max_treedepth_rate"] == [0.0]
    assert row["min_ebfmi"] == [0.7]
    assert row["min_ess_tail"] == [450.0]

    md = (out_dir / "bayesian_multiseed_summary.md").read_text()
    assert "## Health Summary" in md
    assert "0.000125" in md
    assert "--reuse-existing" in md
