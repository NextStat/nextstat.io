from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

from _bayesian_sampler_matrix import (  # noqa: E402
    BenchResult,
    CANONICAL_ADMISSION_POLICY,
    SamplerMatrixConfig,
    config_from_args,
    method_sample_kwargs,
    pairwise_ratios,
    parse_methods,
    quality_summary,
)
from _bayesian_nextstat_bench import collect_environment  # noqa: E402


def test_parse_methods_deduplicates_and_preserves_order() -> None:
    assert parse_methods("nuts,walnuts,nuts,mams") == ["nuts", "walnuts", "mams"]


def test_method_sample_kwargs_respects_method_surface() -> None:
    config = SamplerMatrixConfig(
        methods=("nuts", "walnuts", "mams"),
        models=("std_normal_10d",),
        seeds=(42,),
        out_dir=Path("/tmp/sampler_matrix_test"),
        n_chains=4,
        n_warmup=100,
        n_samples=100,
        metric="diagonal",
        glm_n=1000,
        glm_p=10,
        target_accept=0.8,
        max_treedepth=10,
        mams_max_leapfrog=512,
        mams_diagonal_precond=True,
    )

    assert method_sample_kwargs(config, "nuts") == {"max_treedepth": 10}
    assert method_sample_kwargs(config, "walnuts") == {"max_treedepth": 10}
    assert method_sample_kwargs(config, "mams") == {
        "max_leapfrog": 512,
        "diagonal_precond": True,
    }


def test_pairwise_ratios_use_first_method_as_baseline() -> None:
    comparisons = pairwise_ratios(
        {
            "nuts": {"ess_bulk_per_sec": 10.0, "ess_tail_per_sec": 5.0, "ess_bulk_per_leapfrog": 2.0, "ess_tail_per_leapfrog": 1.0, "leapfrogs_per_sec": 20.0, "median_wall_secs": 4.0},
            "walnuts": {"ess_bulk_per_sec": 12.0, "ess_tail_per_sec": 6.0, "ess_bulk_per_leapfrog": 2.2, "ess_tail_per_leapfrog": 1.1, "leapfrogs_per_sec": 21.0, "median_wall_secs": 3.0},
            "mams": {"ess_bulk_per_sec": 8.0, "ess_tail_per_sec": 4.0, "ess_bulk_per_leapfrog": 3.0, "ess_tail_per_leapfrog": 1.4, "leapfrogs_per_sec": 9.0, "median_wall_secs": 5.0},
        },
        methods=["nuts", "walnuts", "mams"],
        baseline_method="nuts",
    )

    assert comparisons["walnuts_over_nuts"]["ess_bulk_per_sec"] == pytest.approx(1.2)
    assert comparisons["mams_over_nuts"]["ess_bulk_per_leapfrog"] == pytest.approx(1.5)


def test_config_from_args_accepts_dense_metric() -> None:
    class Args:
        methods = "nuts,walnuts"
        models = "std_normal_10d"
        seeds = "42"
        out_dir = "/tmp/sampler_matrix_test"
        n_chains = 4
        n_warmup = 100
        n_samples = 100
        metric = "dense"
        glm_n = 1000
        glm_p = 10
        target_accept = 0.8
        max_treedepth = 10
        mams_max_leapfrog = 1024
        mams_diagonal_precond = True

    config = config_from_args(
        Args(),
        allowed_models=("std_normal_10d",),
        allowed_methods=("nuts", "walnuts"),
    )
    assert config.metric == "dense"


def test_quality_summary_uses_discovery_gate_language() -> None:
    summary = quality_summary(
        [
            BenchResult(
                model="std_normal_10d",
                engine="nuts",
                seed=42,
                wall_secs=1.0,
                ess_bulk={"theta": 100.0},
                ess_tail={"theta": 90.0},
                r_hat={"theta": 1.001},
                divergence_rate=0.0,
                ebfmi=[0.8],
                n_leapfrog_sampling_total=100,
                n_leapfrog_warmup_total=50,
                n_leapfrog_total=150,
                mean_tree_depth=3.0,
                mean_accept_prob=0.9,
            )
        ]
    )

    assert "discovery_gate_passed" in summary
    assert "quality_ok" not in summary
    assert summary["discovery_gate_passed"] is True


def test_canonical_admission_policy_keeps_exact_review_contract() -> None:
    required = CANONICAL_ADMISSION_POLICY["required_contract"]

    assert required["host"] == "nextstat-bench"
    assert required["runner"] == "scripts/benchmarks/bench_walnuts_vs_nuts.py"
    assert required["methods"] == ["nuts", "walnuts"]
    assert required["seeds"] == [42, 123, 777]
    assert required["n_chains"] == 4
    assert required["n_warmup"] == 1000
    assert required["n_samples"] == 1000
    assert required["metric"] == "diagonal"
    assert required["target_accept"] == pytest.approx(0.8)
    assert required["max_treedepth"] == 10
    assert required["uses_shipped_product_defaults"] is True


def test_collect_environment_records_accelerator_runtime_metadata() -> None:
    meta = collect_environment(
        "diagonal",
        1000,
        10,
        [42],
        models=["std_normal_10d"],
        methods=["nuts", "walnuts"],
    )

    accel = meta["accelerator_runtime"]
    assert set(accel) >= {
        "cuda_runtime_available",
        "metal_runtime_available",
        "nvidia_smi_present",
        "nvidia_gpus",
    }
    assert isinstance(accel["nvidia_gpus"], list)


def test_collect_environment_records_execution_lane_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEXTSTAT_BENCH_HOST_POLICY", "nextstat-bench-htcondor-gpu")
    monkeypatch.setenv("NEXTSTAT_BENCH_SUBMIT_HOST", "nextstat-bench")
    monkeypatch.setenv("NEXTSTAT_BENCH_EXECUTE_HOST", "nextstat-gex44")
    monkeypatch.setenv("NEXTSTAT_BENCH_SCHEDULER", "htcondor")

    meta = collect_environment(
        "diagonal",
        1000,
        10,
        [42],
        models=["std_normal_10d"],
        methods=["nuts", "walnuts"],
    )

    lane = meta["execution_lane"]
    assert lane["host_policy"] == "nextstat-bench-htcondor-gpu"
    assert lane["submit_host"] == "nextstat-bench"
    assert lane["execute_host"] == "nextstat-gex44"
    assert lane["scheduler"] == "htcondor"
