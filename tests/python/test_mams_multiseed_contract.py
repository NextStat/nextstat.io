import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
MULTISEED_SCRIPT = REPO_ROOT / "benchmarks" / "nextstat-public-benchmarks" / "suites" / "mams" / "multiseed.py"
SUITE_SCRIPT = REPO_ROOT / "benchmarks" / "nextstat-public-benchmarks" / "suites" / "mams" / "suite.py"


def test_mams_multiseed_reuse_existing_surfaces_dataset_seed_and_parity(tmp_path: Path) -> None:
    out_dir = tmp_path / "multiseed"
    seed_dir = out_dir / "seed_42"
    cases_dir = seed_dir / "cases"
    cases_dir.mkdir(parents=True)

    suite_obj = {
        "schema_version": "nextstat.mams_benchmark_suite_result.v1",
        "suite": "mams",
        "cases": [
            {
                "case": "glm_logistic",
                "backend": "nextstat_mams",
                "seed": 42,
                "status": "ok",
                "wall_time_s": 1.0,
                "n_grad_evals": 1200,
                "min_ess_bulk": 450.0,
                "ess_per_grad": 0.375,
                "ess_per_sec": 180.0,
                "max_r_hat": 1.004,
                "path": "cases/glm_logistic__nextstat_mams__s42.json",
                "sha256": "0" * 64,
            },
            {
                "case": "glm_logistic",
                "backend": "nextstat_nuts",
                "seed": 42,
                "status": "ok",
                "wall_time_s": 1.2,
                "n_grad_evals": 1300,
                "min_ess_bulk": 420.0,
                "ess_per_grad": 0.323,
                "ess_per_sec": 160.0,
                "max_r_hat": 1.006,
                "path": "cases/glm_logistic__nextstat_nuts__s42.json",
                "sha256": "1" * 64,
            },
        ],
        "parity": {
            "method": "mean_zscore",
            "warn_z": 8.0,
            "fail_z": 12.0,
            "rows": [
                {
                    "case": "glm_logistic",
                    "seed": 42,
                    "status": "ok",
                    "max_z": 2.25,
                    "worst": [{"param": "beta[1]", "z": 2.25}],
                }
            ],
        },
    }
    (seed_dir / "mams_suite.json").write_text(json.dumps(suite_obj) + "\n", encoding="utf-8")

    case_template = {
        "schema_version": "nextstat.mams_benchmark_result.v1",
        "suite": "mams",
        "case": "glm_logistic",
        "deterministic": True,
        "status": "ok",
        "environment": {},
        "meta": {
            "python": "3.12.0",
            "platform": "test",
            "nextstat_version": "0.0.0",
        },
        "dataset": {"id": "glm_logistic", "sha256": "2" * 64},
        "config": {
            "dataset_seed": 12345,
            "n_chains": 4,
            "n_samples": 2000,
            "n_warmup": 1000,
            "seed": 42,
            "target_accept": 0.9,
        },
        "timing": {"wall_time_s": 1.0},
        "metrics": {
            "wall_time_s": 1.0,
            "n_grad_evals": 1200,
            "n_integration_steps": 1198,
            "ess_per_grad": 0.375,
            "ess_per_sec": 180.0,
            "ess_per_sec_warm": 220.0,
            "min_ess_bulk": 450.0,
            "min_ess_tail": 410.0,
            "max_r_hat": 1.004,
            "accept_rate": 0.92,
        },
    }
    (cases_dir / "glm_logistic__nextstat_mams__s42.json").write_text(json.dumps(case_template) + "\n", encoding="utf-8")
    nuts_case = dict(case_template)
    nuts_case["backend"] = "nextstat_nuts"
    nuts_case["timing"] = {"wall_time_s": 1.2}
    nuts_case["metrics"] = dict(case_template["metrics"])
    nuts_case["metrics"]["wall_time_s"] = 1.2
    nuts_case["metrics"]["ess_per_grad"] = 0.323
    nuts_case["metrics"]["ess_per_sec"] = 160.0
    nuts_case["metrics"]["ess_per_sec_warm"] = 190.0
    nuts_case["metrics"]["min_ess_bulk"] = 420.0
    nuts_case["metrics"]["min_ess_tail"] = 390.0
    nuts_case["metrics"]["max_r_hat"] = 1.006
    nuts_case["metrics"]["accept_rate"] = 0.89
    (cases_dir / "glm_logistic__nextstat_nuts__s42.json").write_text(json.dumps(nuts_case) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(MULTISEED_SCRIPT),
            "--out-dir",
            str(out_dir),
            "--seeds",
            "42",
            "--reuse-existing",
        ],
        check=False,
    )
    assert completed.returncode == 0

    summary = json.loads((out_dir / "mams_multiseed_summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["dataset_seed"] == 12345
    assert summary["backends"] == "nextstat_mams,nextstat_nuts"
    row = next(item for item in summary["cases"] if item["backend"] == "nextstat_mams")
    assert row["ess_per_sec_warm"] == [220.0]
    assert row["min_ess_tail"] == [410.0]
    assert row["accept_rate"] == [0.92]
    assert row["n_integration_steps"] == [1198.0]
    parity_row = summary["parity"]["rows"][0]
    assert parity_row["max_z"] == [2.25]
    assert parity_row["seed_rows"][0]["worst"] == [{"param": "beta[1]", "z": 2.25}]

    md = (out_dir / "mams_multiseed_summary.md").read_text(encoding="utf-8")
    assert "## Parity Summary" in md
    assert "dataset_seed=12345" in md
    assert "sampler variation" in md


def test_mams_suite_forwards_dataset_seed_to_case_runner(monkeypatch, tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("mams_suite_contract", SUITE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls: list[list[str]] = []
    real_run = subprocess.run

    def fake_run(*popenargs, **kwargs):  # noqa: ANN001
        cmd = popenargs[0]
        cmd_list = [str(part) for part in cmd]
        if "--out" not in cmd_list:
            return real_run(*popenargs, **kwargs)
        calls.append(cmd_list)
        out_path = Path(cmd_list[cmd_list.index("--out") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        backend = cmd_list[cmd_list.index("--backend") + 1]
        case = cmd_list[cmd_list.index("--case") + 1]
        seed = int(cmd_list[cmd_list.index("--seed") + 1])
        dataset_seed = int(cmd_list[cmd_list.index("--dataset-seed") + 1])
        out_path.write_text(
            json.dumps(
                {
                    "schema_version": "nextstat.mams_benchmark_result.v1",
                    "suite": "mams",
                    "case": case,
                    "backend": backend,
                    "deterministic": True,
                    "status": "ok",
                    "environment": {},
                    "meta": {
                        "python": "3.12.0",
                        "platform": "test",
                        "nextstat_version": "0.0.0",
                    },
                    "dataset": {"id": case, "sha256": "0" * 64},
                    "config": {
                        "dataset_seed": dataset_seed,
                        "n_chains": 1,
                        "n_samples": 2,
                        "n_warmup": 1,
                        "seed": seed,
                        "target_accept": 0.9,
                    },
                    "timing": {"wall_time_s": 0.1},
                    "metrics": {
                        "wall_time_s": 0.1,
                        "n_grad_evals": 10,
                        "min_ess_bulk": 5.0,
                        "ess_per_grad": 0.5,
                        "ess_per_sec": 50.0,
                        "max_r_hat": 1.0,
                        "posterior_summary": {"status": "missing"},
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd_list, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setitem(sys.modules, "nextstat", SimpleNamespace(__version__="0.0.0"))

    out_dir = tmp_path / "suite"
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(SUITE_SCRIPT),
            "--out-dir",
            str(out_dir),
            "--backends",
            "nextstat_mams,nextstat_nuts",
            "--seeds",
            "42",
            "--n-chains",
            "1",
            "--warmup",
            "1",
            "--samples",
            "2",
            "--dataset-seed",
            "12345",
            "--deterministic",
        ]
        rc = module.main()
    finally:
        sys.argv = old_argv

    assert rc == 0
    assert calls
    assert all("--dataset-seed" in call for call in calls)
    assert all(call[call.index("--dataset-seed") + 1] == "12345" for call in calls)

    suite_doc = json.loads((out_dir / "mams_suite.json").read_text(encoding="utf-8"))
    assert suite_doc["config"]["dataset_seed"] == 12345
