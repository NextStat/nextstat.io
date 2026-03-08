from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))

from _bayesian_nextstat_bench import collect_environment  # noqa: E402


def test_collect_environment_records_direct_v100_execution_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEXTSTAT_BENCH_HOST_POLICY", "v100-direct-gpu")
    monkeypatch.setenv("NEXTSTAT_BENCH_SUBMIT_HOST", "dev-macbook")
    monkeypatch.setenv("NEXTSTAT_BENCH_BUILD_HOST", "nextstat-bench")
    monkeypatch.setenv("NEXTSTAT_BENCH_EXECUTE_HOST", "v100")
    monkeypatch.setenv("NEXTSTAT_BENCH_SCHEDULER", "split-build-memfd")

    meta = collect_environment(
        "diagonal",
        4096,
        32,
        [42],
        models=["glm_logistic"],
        methods=["walnuts"],
    )

    lane = meta["execution_lane"]
    assert lane["host_policy"] == "v100-direct-gpu"
    assert lane["submit_host"] == "dev-macbook"
    assert lane["build_host"] == "nextstat-bench"
    assert lane["execute_host"] == "v100"
    assert lane["scheduler"] == "split-build-memfd"
