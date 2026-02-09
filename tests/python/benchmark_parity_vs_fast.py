"""Benchmark: Parity mode vs Fast mode throughput.

Measures the overhead of Kahan summation and single-thread constraints.
Run with: python tests/python/benchmark_parity_vs_fast.py

Not a pytest test — this is a standalone benchmark script.
"""

import json
import time
from pathlib import Path

import nextstat

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


def bench_nll_throughput(model, params, n_evals=1000):
    """Measure NLL evals/sec."""
    # Warmup
    for _ in range(10):
        model.nll(params)

    start = time.perf_counter()
    for _ in range(n_evals):
        model.nll(params)
    elapsed = time.perf_counter() - start
    return n_evals / elapsed


def bench_grad_throughput(model, params, n_evals=500):
    """Measure gradient evals/sec."""
    for _ in range(5):
        model.grad_nll(params)

    start = time.perf_counter()
    for _ in range(n_evals):
        model.grad_nll(params)
    elapsed = time.perf_counter() - start
    return n_evals / elapsed


def bench_fit(model, n_runs=5):
    """Measure fit wall-clock time."""
    mle = nextstat.MaximumLikelihoodEstimator()
    # Warmup
    mle.fit(model)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        mle.fit(model)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return sum(times) / len(times)


def bench_batch_toys(model, params, n_toys=100):
    """Measure batch toy throughput (toys/sec)."""
    # Warmup
    nextstat.fit_toys_batch(model, params, n_toys=5, seed=0)

    start = time.perf_counter()
    nextstat.fit_toys_batch(model, params, n_toys=n_toys, seed=42)
    elapsed = time.perf_counter() - start
    return n_toys / elapsed


WORKSPACES = [
    ("simple_workspace.json", "GaussExample", "simple (2 bins)"),
    ("complex_workspace.json", "measurement", "complex (8 params)"),
]

# Large workspaces (optional, may not be in fixtures)
LARGE_WORKSPACES = [
    ("workspace_tHu.json", "tHu", "tHu (184 params)"),
    ("tttt-prod_workspace.json", "tttt_tttt", "tttt (249 params)"),
]


def main():
    results = []

    all_workspaces = list(WORKSPACES)
    for fixture, meas, label in LARGE_WORKSPACES:
        if (FIXTURES_DIR / fixture).exists():
            all_workspaces.append((fixture, meas, label))

    for fixture, measurement, label in all_workspaces:
        ws = load_fixture(fixture)
        model = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))
        params = model.suggested_init()
        n_params = len(params)

        row = {"workspace": label, "n_params": n_params}

        # Adjust iteration counts for large models
        nll_iters = 1000 if n_params < 50 else 200
        grad_iters = 500 if n_params < 50 else 100
        toy_count = 100 if n_params < 50 else 20

        for mode in ("fast", "parity"):
            nextstat.set_eval_mode(mode)
            row[f"nll_{mode}"] = bench_nll_throughput(model, params, nll_iters)
            row[f"grad_{mode}"] = bench_grad_throughput(model, params, grad_iters)
            row[f"fit_{mode}"] = bench_fit(model)
            row[f"toys_{mode}"] = bench_batch_toys(model, params, toy_count)

        # Compute ratios (fast / parity)
        row["nll_ratio"] = row["nll_fast"] / max(row["nll_parity"], 1)
        row["grad_ratio"] = row["grad_fast"] / max(row["grad_parity"], 1)
        row["fit_ratio"] = row["fit_parity"] / max(row["fit_fast"], 1e-9)
        row["toys_ratio"] = row["toys_fast"] / max(row["toys_parity"], 1)

        results.append(row)

    # Restore fast mode
    nextstat.set_eval_mode("fast")

    # Print markdown table
    print("\n## Parity vs Fast Mode Benchmark\n")
    print("| Workspace | Params | NLL/s (Fast) | NLL/s (Parity) | Ratio | Grad/s (Fast) | Grad/s (Parity) | Ratio | Fit ms (Fast) | Fit ms (Parity) | Ratio | Toys/s (Fast) | Toys/s (Parity) | Ratio |")
    print("|-----------|--------|-------------|---------------|-------|--------------|----------------|-------|--------------|----------------|-------|--------------|----------------|-------|")

    for r in results:
        print(
            f"| {r['workspace']} | {r['n_params']} "
            f"| {r['nll_fast']:.0f} | {r['nll_parity']:.0f} | {r['nll_ratio']:.2f}x "
            f"| {r['grad_fast']:.0f} | {r['grad_parity']:.0f} | {r['grad_ratio']:.2f}x "
            f"| {r['fit_fast']*1000:.1f} | {r['fit_parity']*1000:.1f} | {r['fit_ratio']:.2f}x "
            f"| {r['toys_fast']:.0f} | {r['toys_parity']:.0f} | {r['toys_ratio']:.2f}x |"
        )

    print("\n> Ratio = Fast / Parity (higher = Fast is faster)")
    print("> NLL/Grad: evals/sec, Fit: milliseconds, Toys: toys/sec")
    print("> Parity mode uses Kahan summation + single thread (threads=1)")


if __name__ == "__main__":
    main()
