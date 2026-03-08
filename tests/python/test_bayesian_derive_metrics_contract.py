import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT / "benchmarks" / "nextstat-public-benchmarks" / "suites" / "bayesian" / "derive_metrics.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("nextstat_bayesian_derive_metrics", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metric_backend_name_maps_supported_backends() -> None:
    module = _load_module()
    assert module._metric_backend_name("nextstat") == "nextstat"
    assert module._metric_backend_name("cmdstanpy") == "cmdstan"
    assert module._metric_backend_name("pymc") == "pymc"


def test_derive_metrics_builds_case_summary_from_multiseed_dir(tmp_path: Path) -> None:
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
                "path": "cases/histfactory_simple_8p__nextstat.json",
            },
            {
                "case": "histfactory_simple_8p",
                "backend": "cmdstanpy",
                "status": "ok",
                "path": "cases/histfactory_simple_8p__cmdstanpy.json",
            },
            {
                "case": "histfactory_simple_8p",
                "backend": "pymc",
                "status": "ok",
                "path": "cases/histfactory_simple_8p__pymc.json",
            },
        ],
    }
    (seed_dir / "bayesian_suite.json").write_text(json.dumps(suite_obj) + "\n")

    base_case = {
        "schema_version": "nextstat.bayesian_benchmark_result.v1",
        "suite": "bayesian",
        "case": "histfactory_simple_8p",
        "config": {
            "dataset_seed": 12345,
            "init_jitter_rel": 0.1,
            "max_treedepth": 10,
            "metric": "diagonal",
            "n_chains": 4,
            "n_samples": 2000,
            "n_warmup": 1000,
            "target_accept": 0.8,
        },
        "timing": {"n_grad_evals": 2000, "wall_time_s": 1.0},
        "diagnostics_summary": {"min_ess_bulk": 500.0},
    }
    nextstat_case = dict(base_case, backend="nextstat")
    cmdstan_case = dict(base_case, backend="cmdstanpy", timing={"n_grad_evals": 2500, "wall_time_s": 1.0}, diagnostics_summary={"min_ess_bulk": 400.0})
    pymc_case = dict(base_case, backend="pymc", timing={"n_grad_evals": 4000, "wall_time_s": 1.0}, diagnostics_summary={"min_ess_bulk": 200.0})

    (cases_dir / "histfactory_simple_8p__nextstat.json").write_text(json.dumps(nextstat_case) + "\n")
    (cases_dir / "histfactory_simple_8p__cmdstanpy.json").write_text(json.dumps(cmdstan_case) + "\n")
    (cases_dir / "histfactory_simple_8p__pymc.json").write_text(json.dumps(pymc_case) + "\n")

    completed = subprocess.run([sys.executable, str(SCRIPT_PATH), str(out_dir)], check=False)
    assert completed.returncode == 0

    derived = json.loads((out_dir / "derived_metrics.json").read_text())
    assert derived["schema_version"] == "nextstat.bayesian_derived_metrics.v2"
    case = derived["ess_per_leapfrog"]["cases"]["histfactory_simple_8p"]
    assert case["by_seed"]["42"]["nextstat"]["ess_per_leapfrog"] == 0.25
    assert case["by_seed"]["42"]["cmdstan"]["ess_per_leapfrog"] == 0.16
    assert case["by_seed"]["42"]["pymc"]["ess_per_leapfrog"] == 0.05
    assert case["nextstat"]["mean"] == 0.25
    assert case["cmdstan"]["mean"] == 0.16
    assert case["pymc"]["mean"] == 0.05
    assert abs(case["ratio"] - (0.25 / 0.16)) < 1e-12
