#!/usr/bin/env python3
"""
Benchmark NextStat vs pyhf NLL computation performance.

Measures:
1. Single NLL evaluation time
2. Gradient computation time (for optimization)
3. Memory usage
"""

import json
import time
from pathlib import Path
import pyhf
import numpy as np
import nextstat


def load_fixture(name: str) -> dict:
    """Load a test fixture JSON file."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    with open(fixtures_dir / name) as f:
        return json.load(f)


def benchmark_pyhf_nll(n_iterations=1000):
    """Benchmark pyhf NLL computation."""
    print("=" * 70)
    print("pyhf NLL Benchmark")
    print("=" * 70)

    workspace = load_fixture("simple_workspace.json")
    ws = pyhf.Workspace(workspace)
    model = ws.model("GaussExample")
    observations = ws.data(model)

    params = np.array([1.0, 1.0, 1.0])

    # Warmup
    for _ in range(10):
        pyhf.infer.mle.twice_nll(params, observations, model)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        nll = pyhf.infer.mle.twice_nll(params, observations, model)
    end = time.perf_counter()

    elapsed = end - start
    per_call = elapsed / n_iterations * 1e6  # microseconds

    print(f"\nIterations: {n_iterations}")
    print(f"Total time: {elapsed:.4f} s")
    print(f"Per call:   {per_call:.2f} µs")
    print(f"Throughput: {n_iterations / elapsed:.0f} calls/sec")

    return per_call


def benchmark_nextstat_nll(n_iterations=100000):
    """Benchmark NextStat NLL computation (Python → Rust extension)."""
    print("\n" + "=" * 70)
    print("NextStat NLL Benchmark")
    print("=" * 70)

    workspace = load_fixture("simple_workspace.json")
    model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
    params = model.suggested_init()

    # Warmup
    for _ in range(50):
        model.nll(params)

    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = model.nll(params)
    end = time.perf_counter()

    elapsed = end - start
    per_call = elapsed / n_iterations * 1e6  # microseconds

    print(f"\nIterations: {n_iterations}")
    print(f"Total time: {elapsed:.4f} s")
    print(f"Per call:   {per_call:.2f} µs")
    print(f"Throughput: {n_iterations / elapsed:.0f} calls/sec")
    print("\nNote: This includes Python→Rust FFI + argument conversion overhead.")

    return per_call


def benchmark_pyhf_gradient(n_iterations=100):
    """Benchmark pyhf gradient computation."""
    print("\n" + "=" * 70)
    print("pyhf Gradient Benchmark")
    print("=" * 70)

    workspace = load_fixture("simple_workspace.json")
    ws = pyhf.Workspace(workspace)
    model = ws.model("GaussExample")
    observations = ws.data(model)

    params = np.array([1.0, 1.0, 1.0])

    # Warmup
    for _ in range(5):
        pyhf.infer.mle.gradient(params, observations, model)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        grad = pyhf.infer.mle.gradient(params, observations, model)
    end = time.perf_counter()

    elapsed = end - start
    per_call = elapsed / n_iterations * 1e3  # milliseconds

    print(f"\nIterations: {n_iterations}")
    print(f"Total time: {elapsed:.4f} s")
    print(f"Per call:   {per_call:.2f} ms")
    print(f"Throughput: {n_iterations / elapsed:.0f} calls/sec")

    return per_call


def benchmark_pyhf_fit(n_iterations=10):
    """Benchmark full pyhf fit."""
    print("\n" + "=" * 70)
    print("pyhf Full Fit Benchmark")
    print("=" * 70)

    workspace = load_fixture("simple_workspace.json")
    ws = pyhf.Workspace(workspace)
    model = ws.model("GaussExample")
    observations = ws.data(model)

    # Warmup
    for _ in range(2):
        pyhf.infer.mle.fit(observations, model)

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        result = pyhf.infer.mle.fit(observations, model)
    end = time.perf_counter()

    elapsed = end - start
    per_call = elapsed / n_iterations * 1e3  # milliseconds

    print(f"\nIterations: {n_iterations}")
    print(f"Total time: {elapsed:.4f} s")
    print(f"Per call:   {per_call:.2f} ms")
    print(f"Best-fit:   {result}")

    return per_call


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print("Performance Benchmark: pyhf (Python)")
    print("=" * 70)
    print("\nHardware: Apple Silicon")
    import sys
    print(f"Python: {sys.version.split()[0]}")
    print(f"pyhf: {pyhf.__version__}")
    print(f"nextstat: {nextstat.__version__}")
    print()

    nll_time = benchmark_pyhf_nll(n_iterations=1000)
    ns_nll_time = benchmark_nextstat_nll(n_iterations=100000)
    # grad_time = benchmark_pyhf_gradient(n_iterations=100)  # pyhf doesn't expose gradient API
    fit_time = benchmark_pyhf_fit(n_iterations=10)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"NLL computation:  {nll_time:.2f} µs/call")
    print(f"NextStat NLL:     {ns_nll_time:.2f} µs/call")
    print(f"Speedup:          {nll_time / ns_nll_time:.1f}x (including Python overhead)")
    # print(f"Gradient:         {grad_time:.2f} ms/call")
    print(f"Full fit:         {fit_time:.2f} ms/call")
    print()
    print("For core (no-Python) benchmarks, use the Rust Criterion bench: crates/ns-translate/benches/nll_benchmark.rs")


if __name__ == "__main__":
    main()
