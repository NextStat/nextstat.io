import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
MULTISEED_SCRIPT = REPO_ROOT / "benchmarks" / "nextstat-public-benchmarks" / "suites" / "mams" / "stress_multiseed.py"
SUITE_SCRIPT = REPO_ROOT / "benchmarks" / "nextstat-public-benchmarks" / "suites" / "mams" / "stress_suite.py"


def test_mams_stress_multiseed_reuse_existing_surfaces_case_catalog_and_parity_scope(tmp_path: Path) -> None:
    out_dir = tmp_path / "stress"
    seed_dir = out_dir / "seed_42"
    cases_dir = seed_dir / "cases"
    cases_dir.mkdir(parents=True)

    suite_obj = {
        "schema_version": "nextstat.mams_stress_benchmark_suite_result.v1",
        "suite": "mams_stress",
        "deterministic": True,
        "meta": {"python": "3.12.0", "platform": "test", "nextstat_version": "0.0.0"},
        "config": {
            "n_chains": 4,
            "n_warmup": 3500,
            "n_samples": 2000,
            "dataset_seed": 12345,
            "target_accept": 0.985,
            "n_groups": 20,
            "n_per_group": 20,
            "seeds": [42],
            "backends": ["nextstat_mams", "nextstat_nuts"],
            "parity_warn_z": 8.0,
            "parity_fail_z": 12.0,
        },
        "case_catalog": [
            {
                "case": "neal_funnel_ncp_10d",
                "case_tier": "supported",
                "parity_scope": "required",
                "description": "supported",
            },
            {
                "case": "neal_funnel_10d_centered",
                "case_tier": "pathological_control",
                "parity_scope": "informational",
                "description": "control",
            },
        ],
        "cases": [
            {
                "case": "neal_funnel_ncp_10d",
                "case_tier": "supported",
                "backend": "nextstat_mams",
                "seed": 42,
                "status": "ok",
                "wall_time_s": 1.0,
                "n_grad_evals": 1200,
                "min_ess_bulk": 450.0,
                "ess_per_grad": 0.375,
                "ess_per_sec": 180.0,
                "max_r_hat": 1.004,
                "path": "cases/neal_funnel_ncp_10d__nextstat_mams__s42.json",
                "sha256": "0" * 64,
            },
            {
                "case": "neal_funnel_ncp_10d",
                "case_tier": "supported",
                "backend": "nextstat_nuts",
                "seed": 42,
                "status": "ok",
                "wall_time_s": 1.2,
                "n_grad_evals": 1400,
                "min_ess_bulk": 420.0,
                "ess_per_grad": 0.300,
                "ess_per_sec": 150.0,
                "max_r_hat": 1.006,
                "path": "cases/neal_funnel_ncp_10d__nextstat_nuts__s42.json",
                "sha256": "1" * 64,
            },
            {
                "case": "neal_funnel_10d_centered",
                "case_tier": "pathological_control",
                "backend": "nextstat_mams",
                "seed": 42,
                "status": "warn",
                "wall_time_s": 1.5,
                "n_grad_evals": 1600,
                "min_ess_bulk": 300.0,
                "ess_per_grad": 0.1875,
                "ess_per_sec": 120.0,
                "max_r_hat": 1.120,
                "path": "cases/neal_funnel_10d_centered__nextstat_mams__s42.json",
                "sha256": "2" * 64,
            },
        ],
        "summary": {
            "n_total": 3,
            "n_ok": 2,
            "n_warn": 1,
            "n_failed": 0,
            "n_parity_warn": 0,
            "n_parity_fail": 0,
        },
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
                    "seed": 42,
                    "status": "ok",
                    "max_z": 1.5,
                    "worst": [{"param": "y", "z": 1.5}],
                }
            ],
        },
    }
    (seed_dir / "mams_stress_suite.json").write_text(json.dumps(suite_obj) + "\n", encoding="utf-8")

    case_template = {
        "schema_version": "nextstat.mams_benchmark_result.v1",
        "suite": "mams",
        "deterministic": True,
        "environment": {},
        "meta": {"python": "3.12.0", "platform": "test", "nextstat_version": "0.0.0"},
        "dataset": {"id": "case", "sha256": "3" * 64},
        "config": {
            "benchmark_seed": 42,
            "cold_start_seed": 42,
            "dataset_seed": 12345,
            "n_chains": 4,
            "n_samples": 2000,
            "n_warmup": 3500,
            "reported_draws_seed": 43,
            "reported_draws_source": "warm_start",
            "seed": 42,
            "target_accept": 0.985,
            "warm_start_seed": 43,
            "n_groups": 20,
            "n_per_group": 20,
            "init_l": None,
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
        "status": "ok",
    }
    for file_name, backend, case, status, rhat in [
        ("neal_funnel_ncp_10d__nextstat_mams__s42.json", "nextstat_mams", "neal_funnel_ncp_10d", "ok", 1.004),
        ("neal_funnel_ncp_10d__nextstat_nuts__s42.json", "nextstat_nuts", "neal_funnel_ncp_10d", "ok", 1.006),
        ("neal_funnel_10d_centered__nextstat_mams__s42.json", "nextstat_mams", "neal_funnel_10d_centered", "warn", 1.120),
    ]:
        obj = dict(case_template)
        obj["backend"] = backend
        obj["case"] = case
        obj["status"] = status
        obj["metrics"] = dict(case_template["metrics"])
        obj["metrics"]["max_r_hat"] = rhat
        obj["config"] = dict(case_template["config"])
        obj["config"]["init_l"] = 2.0 if backend == "nextstat_mams" else None
        (cases_dir / file_name).write_text(json.dumps(obj) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(MULTISEED_SCRIPT), "--out-dir", str(out_dir), "--seeds", "42", "--reuse-existing"],
        check=False,
    )
    assert completed.returncode == 0

    summary = json.loads((out_dir / "mams_stress_multiseed_summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["n_groups"] == 20
    assert summary["config"]["n_per_group"] == 20
    assert summary["seed_semantics"]["warm_start_seed_offset"] == 1
    assert summary["case_catalog"][0]["case"] == "neal_funnel_ncp_10d"
    supported_row = next(item for item in summary["cases"] if item["case"] == "neal_funnel_ncp_10d" and item["backend"] == "nextstat_mams")
    assert supported_row["case_tier"] == "supported"
    assert supported_row["configs"][0]["reported_draws_seed"] == 43
    assert supported_row["config_overrides"] == {"init_l": 2.0}
    parity_row = summary["parity"]["rows"][0]
    assert parity_row["parity_scope"] == "required"

    md = (out_dir / "mams_stress_multiseed_summary.md").read_text(encoding="utf-8")
    assert "## Case Catalog" in md
    assert "init_l=2.0" in md
    assert "pathological_control" in md
    assert "reported posterior/diagnostic metrics come from `config.reported_draws_seed`" in md


def test_mams_stress_suite_forwards_hier_case_parameters(monkeypatch, tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("mams_stress_suite_contract", SUITE_SCRIPT)
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
                    "meta": {"python": "3.12.0", "platform": "test", "nextstat_version": "0.0.0"},
                    "dataset": {"id": case, "sha256": "0" * 64},
                    "config": {
                        "benchmark_seed": seed,
                        "cold_start_seed": seed,
                        "dataset_seed": 12345,
                        "n_chains": 1,
                        "n_samples": 2,
                        "n_warmup": 1,
                        "reported_draws_seed": seed + 1,
                        "reported_draws_source": "warm_start",
                        "seed": seed,
                        "target_accept": 0.985,
                        "warm_start_seed": seed + 1,
                        "n_groups": 11 if case == "hier_random_intercept_non_centered" else None,
                        "n_per_group": 13 if case == "hier_random_intercept_non_centered" else None,
                        "init_l": 2.0 if case == "hier_random_intercept_non_centered" and backend == "nextstat_mams" else None,
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
            "--out-dir", str(out_dir),
            "--backends", "nextstat_mams",
            "--seeds", "42",
            "--n-chains", "1",
            "--warmup", "1",
            "--samples", "2",
            "--n-groups", "11",
            "--n-per-group", "13",
            "--deterministic",
        ]
        rc = module.main()
    finally:
        sys.argv = old_argv

    assert rc == 0
    hier_calls = [call for call in calls if call[call.index("--case") + 1] == "hier_random_intercept_non_centered"]
    assert hier_calls
    assert all(call[call.index("--n-groups") + 1] == "11" for call in hier_calls)
    assert all(call[call.index("--n-per-group") + 1] == "13" for call in hier_calls)
    hier_mams_calls = [call for call in hier_calls if call[call.index("--backend") + 1] == "nextstat_mams"]
    assert hier_mams_calls
    assert all("--init-l" in call for call in hier_mams_calls)
    assert all(call[call.index("--init-l") + 1] == "2.0" for call in hier_mams_calls)

    suite_doc = json.loads((out_dir / "mams_stress_suite.json").read_text(encoding="utf-8"))
    assert suite_doc["suite"] == "mams_stress"
    assert any(row["case_tier"] == "pathological_control" for row in suite_doc["case_catalog"])
    hier_suite_row = next(
        row
        for row in suite_doc["cases"]
        if row["case"] == "hier_random_intercept_non_centered" and row["backend"] == "nextstat_mams"
    )
    assert hier_suite_row["config_overrides"] == {"init_l": 2.0}
