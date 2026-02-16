#!/usr/bin/env python3
"""
Micro-benchmark for Metal unbinned gradient kernel (unbinned_batch_nll_grad).
Measures GPU batch toy fit timing to isolate kernel performance.

Usage:
    .venv/bin/python benchmarks/unbinned/bench_metal_grad_kernel.py
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "bindings" / "ns-py" / "python"))

import nextstat  # type: ignore


def generate_gauss_exp_spec(n_events: int, seed: int = 42) -> tuple[str, str]:
    """Generate Gauss+Exp model with inline data, return (spec_path, data_path)."""
    rng = np.random.default_rng(seed)
    a, b = 60.0, 120.0
    mu_sig, sigma_sig, lam_bkg = 91.0, 2.5, -0.03

    n_sig = int(n_events * 0.25)
    n_bkg = n_events - n_sig

    sig_data = rng.normal(mu_sig, sigma_sig, n_sig)
    sig_data = sig_data[(sig_data >= a) & (sig_data <= b)]
    while len(sig_data) < n_sig:
        extra = rng.normal(mu_sig, sigma_sig, n_sig)
        extra = extra[(extra >= a) & (extra <= b)]
        sig_data = np.concatenate([sig_data, extra])[:n_sig]

    bkg_data = a + rng.exponential(-1.0 / lam_bkg, n_bkg * 3)
    bkg_data = bkg_data[(bkg_data >= a) & (bkg_data <= b)][:n_bkg]
    while len(bkg_data) < n_bkg:
        extra = a + rng.exponential(-1.0 / lam_bkg, n_bkg * 3)
        extra = extra[(extra >= a) & (extra <= b)]
        bkg_data = np.concatenate([bkg_data, extra])[:n_bkg]

    data = np.concatenate([sig_data, bkg_data])
    rng.shuffle(data)

    spec = {
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                {"name": "mu", "init": 1.0, "bounds": [0.0, 5.0]},
                {"name": "mu_sig", "init": 91.0, "bounds": [85.0, 95.0]},
                {"name": "sigma_sig", "init": 2.5, "bounds": [0.5, 10.0]},
                {"name": "lambda_bkg", "init": -0.03, "bounds": [-0.1, -0.001]},
            ],
        },
        "channels": [
            {
                "name": "main",
                "data": {"inline": data.tolist()},
                "observables": [{"name": "mass", "bounds": [a, b]}],
                "processes": [
                    {
                        "name": "signal",
                        "pdf": {
                            "type": "gaussian",
                            "observable": "mass",
                            "params": ["mu_sig", "sigma_sig"],
                        },
                        "yield": {"type": "scaled", "base_yield": float(n_sig), "scale": "mu"},
                    },
                    {
                        "name": "background",
                        "pdf": {
                            "type": "exponential",
                            "observable": "mass",
                            "params": ["lambda_bkg"],
                        },
                        "yield": {"type": "fixed", "value": float(n_bkg)},
                    },
                ],
            }
        ],
    }

    td = tempfile.mkdtemp(prefix="ns_bench_metal_")
    spec_path = os.path.join(td, "spec.json")
    with open(spec_path, "w") as f:
        json.dump(spec, f)
    return spec_path, td


def bench_fit_cpu(spec_path: str, n_warmup: int = 2, n_repeats: int = 5) -> dict:
    """Benchmark CPU fit (single, no toys) for baseline."""
    model = nextstat.UnbinnedModel.from_config(spec_path)
    for _ in range(n_warmup):
        nextstat.fit(model)
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        r = nextstat.fit(model)
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "type": "cpu_single_fit",
        "times_ms": times,
        "median_ms": float(np.median(times)),
        "nll": r["nll"],
        "n_iter": r["n_iter"],
    }


def main():
    cases = [
        ("gauss_exp_1k", 1000),
        ("gauss_exp_10k", 10000),
        ("gauss_exp_100k", 100000),
    ]
    n_toys = 100
    repeats = 3

    print(f"Metal Gradient Kernel Micro-Benchmark")
    print(f"Chip: Apple M5 (assumed), n_toys={n_toys}, repeats={repeats}")
    print("=" * 70)

    for label, n_events in cases:
        spec_path, td = generate_gauss_exp_spec(n_events)

        # CPU single fit baseline
        cpu = bench_fit_cpu(spec_path)
        print(f"\n{label} (N={n_events}):")
        print(f"  CPU single fit: {cpu['median_ms']:.2f} ms ({cpu['n_iter']} iters)")

        # GPU batch toy fits via CLI
        import subprocess

        binary = str(REPO / ".nextstat-cargo-target" / "release" / "nextstat")
        if not os.path.exists(binary):
            print(f"  [skip GPU] release binary not found at {binary}")
            continue

        for trial in range(repeats):
            cmd = [
                binary,
                "unbinned-fit-toys",
                spec_path,
                "--n-toys", str(n_toys),
                "--seed", str(42 + trial),
                "--device", "metal",
                "--batch",
            ]
            t0 = time.perf_counter()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            wall = (time.perf_counter() - t0) * 1000
            if result.returncode != 0:
                print(f"  GPU trial {trial}: FAILED ({result.stderr[:200]})")
                continue
            # Parse output for timing
            lines = result.stdout.strip().split("\n")
            print(f"  GPU trial {trial}: {wall:.0f} ms wall ({n_toys} toys)")

        # Cleanup
        import shutil
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    main()
