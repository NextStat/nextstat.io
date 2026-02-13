#!/usr/bin/env python3
"""PF3.1-OPT5: Reproducible 2-GPU regression benchmark harness.

Self-contained test that verifies multi-GPU scaling hasn't regressed.
Generates its own spec + data, runs 1-GPU vs 2-GPU, checks:
  1. Wall-time scaling ratio ≥ SCALING_THRESHOLD (default 1.5×)
  2. Numerical parity: POI means within PARITY_TOL
  3. Convergence rate parity

Usage:
  python3 scripts/benchmarks/pf31_regression_bench.py [--binary PATH] [--n-toys N]

Requirements: 2+ NVIDIA GPUs with CUDA, nextstat binary built with --features cuda.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time

SCALING_THRESHOLD = 1.5  # 2 GPU must be at least 1.5× faster than 1 GPU
PARITY_TOL = 0.05       # max relative difference in mean POI between 1 and 2 GPU
N_EVENTS = 50_000       # events per toy (moderate — enough to saturate GPU)
N_TOYS_DEFAULT = 200    # fast but statistically meaningful


def find_binary():
    """Find nextstat binary."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "../../target/release/nextstat"),
        "/root/nextstat.io/target/release/nextstat",
        "nextstat",
    ]
    for c in candidates:
        p = os.path.abspath(c)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def detect_gpus():
    """Return number of CUDA GPUs via nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            return len(r.stdout.strip().splitlines())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def generate_spec(tmpdir, n_events):
    """Generate a Gauss+Exp spec with synthetic data."""
    import random
    random.seed(42)

    mu_true, sigma_true = 0.0, 1.0
    tau_true = 2.0
    n_sig_true, n_bkg_true = 3000.0, 2000.0

    data = []
    lo, hi = -5.0, 10.0
    for _ in range(n_events):
        if random.random() < n_sig_true / (n_sig_true + n_bkg_true):
            x = random.gauss(mu_true, sigma_true)
        else:
            x = -tau_true * math.log(1 - random.random() * (1 - math.exp(-hi / tau_true)))
        x = max(lo, min(hi, x))
        data.append(x)

    data_path = os.path.join(tmpdir, "data.json")
    with open(data_path, "w") as f:
        json.dump({"x": data}, f)

    spec = {
        "channels": [{
            "name": "sr",
            "observables": [{"name": "x", "bounds": [lo, hi]}],
            "data": {"source": "data.json"},
            "processes": [
                {
                    "name": "signal",
                    "pdf": {"type": "gaussian", "params": ["mu", "sigma"]},
                    "yield": {"expr": "n_sig"},
                },
                {
                    "name": "background",
                    "pdf": {"type": "exponential", "params": ["tau"]},
                    "yield": {"expr": "n_bkg"},
                },
            ],
        }],
        "parameters": [
            {"name": "mu", "init": 0.0, "bounds": [-3.0, 3.0]},
            {"name": "sigma", "init": 1.0, "bounds": [0.3, 5.0]},
            {"name": "tau", "init": 2.0, "bounds": [0.1, 20.0]},
            {"name": "n_sig", "init": 3000.0, "bounds": [0.0, 10000.0]},
            {"name": "n_bkg", "init": 2000.0, "bounds": [0.0, 10000.0]},
        ],
        "poi": "n_sig",
    }

    spec_path = os.path.join(tmpdir, "spec.json")
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)

    return spec_path


def run_fit_toys(binary, spec, n_toys, n_gpu, seed=42, timeout=600):
    """Run unbinned-fit-toys and return (wall_s, result_dict) or (wall_s, None)."""
    devs = ",".join(str(i) for i in range(n_gpu))
    cmd = [
        binary, "unbinned-fit-toys",
        "--config", spec,
        "--n-toys", str(n_toys),
        "--seed", str(seed),
        "--gpu", "cuda",
        "--gpu-sample-toys",
        "--gpu-native",
        "--gpu-devices", devs,
        "--gpu-shards", str(n_gpu),
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return time.time() - t0, None
    wall = time.time() - t0

    if proc.returncode != 0:
        sys.stderr.write(f"FAIL ({n_gpu} GPU): {proc.stderr.strip()[:300]}\n")
        return wall, None

    try:
        d = json.loads(proc.stdout)
        return wall, d
    except json.JSONDecodeError:
        return wall, None


def extract_poi_stats(result):
    """Extract POI mean, std, convergence from result JSON."""
    r = result["results"]
    poi_vals = r.get("poi_values", [])
    conv = r["n_converged"]
    total = r["n_toys"]
    errs = r["n_error"]

    if poi_vals:
        finite = [v for v in poi_vals if v is not None and math.isfinite(v)]
        if finite:
            mean_poi = sum(finite) / len(finite)
            var_poi = sum((v - mean_poi) ** 2 for v in finite) / len(finite)
            return mean_poi, math.sqrt(var_poi), conv, total, errs

    return None, None, conv, total, errs


def main():
    parser = argparse.ArgumentParser(description="PF3.1 Multi-GPU Regression Benchmark")
    parser.add_argument("--binary", help="Path to nextstat binary")
    parser.add_argument("--n-toys", type=int, default=N_TOYS_DEFAULT)
    parser.add_argument("--n-events", type=int, default=N_EVENTS)
    parser.add_argument("--scaling-threshold", type=float, default=SCALING_THRESHOLD)
    parser.add_argument("--json-output", help="Write results to JSON file")
    args = parser.parse_args()

    binary = args.binary or find_binary()
    if not binary:
        sys.exit("ERROR: nextstat binary not found. Use --binary PATH.")

    n_gpus = detect_gpus()
    if n_gpus < 2:
        sys.exit(f"ERROR: need ≥2 GPUs, found {n_gpus}")

    print("=" * 60)
    print("PF3.1-OPT5 Multi-GPU Regression Benchmark")
    print(f"  Binary:    {binary}")
    print(f"  GPUs:      {n_gpus}")
    print(f"  Toys:      {args.n_toys}")
    print(f"  Events:    ~{args.n_events}")
    print(f"  Threshold: {args.scaling_threshold:.2f}×")
    print("=" * 60)

    with tempfile.TemporaryDirectory(prefix="pf31_regbench_") as tmpdir:
        print("\n[1/3] Generating spec + data ...", end="", flush=True)
        spec = generate_spec(tmpdir, args.n_events)
        print(" done")

        print(f"\n[2/3] Running 1 GPU ({args.n_toys} toys) ...", end="", flush=True)
        wall_1, res_1 = run_fit_toys(binary, spec, args.n_toys, 1)
        if res_1 is None:
            sys.exit("ERROR: 1-GPU run failed")
        poi_mean_1, poi_std_1, conv_1, total_1, err_1 = extract_poi_stats(res_1)
        print(f" {wall_1:.1f}s ({conv_1}/{total_1} conv)")

        print(f"\n[3/3] Running 2 GPU ({args.n_toys} toys) ...", end="", flush=True)
        wall_2, res_2 = run_fit_toys(binary, spec, args.n_toys, 2)
        if res_2 is None:
            sys.exit("ERROR: 2-GPU run failed")
        poi_mean_2, poi_std_2, conv_2, total_2, err_2 = extract_poi_stats(res_2)
        print(f" {wall_2:.1f}s ({conv_2}/{total_2} conv)")

    # --- Analysis ---
    scaling = wall_1 / wall_2 if wall_2 > 0 else 0.0

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  1 GPU:  {wall_1:7.2f}s  (conv {conv_1}/{total_1}, err {err_1})")
    print(f"  2 GPU:  {wall_2:7.2f}s  (conv {conv_2}/{total_2}, err {err_2})")
    print(f"  Scaling: {scaling:.2f}×")

    failures = []

    # Check 1: scaling
    if scaling < args.scaling_threshold:
        failures.append(
            f"SCALING: {scaling:.2f}× < {args.scaling_threshold:.2f}× threshold"
        )
        print(f"  [FAIL] Scaling {scaling:.2f}× below threshold {args.scaling_threshold:.2f}×")
    else:
        print(f"  [PASS] Scaling {scaling:.2f}× ≥ {args.scaling_threshold:.2f}×")

    # Check 2: numerical parity
    if poi_mean_1 is not None and poi_mean_2 is not None:
        if abs(poi_mean_1) > 1e-10:
            rel_diff = abs(poi_mean_2 - poi_mean_1) / abs(poi_mean_1)
        else:
            rel_diff = abs(poi_mean_2 - poi_mean_1)
        if rel_diff > PARITY_TOL:
            failures.append(
                f"PARITY: POI mean diff {rel_diff:.4f} > {PARITY_TOL}"
            )
            print(f"  [FAIL] POI parity: 1GPU={poi_mean_1:.4f}, 2GPU={poi_mean_2:.4f}, diff={rel_diff:.4f}")
        else:
            print(f"  [PASS] POI parity: 1GPU={poi_mean_1:.4f}, 2GPU={poi_mean_2:.4f}, diff={rel_diff:.4f}")
    else:
        print("  [SKIP] POI parity: no POI values in output")

    # Check 3: convergence rate
    conv_rate_1 = conv_1 / total_1 if total_1 > 0 else 0
    conv_rate_2 = conv_2 / total_2 if total_2 > 0 else 0
    if conv_rate_1 > 0 and conv_rate_2 / conv_rate_1 < 0.8:
        failures.append(
            f"CONVERGENCE: 2GPU rate {conv_rate_2:.2%} << 1GPU rate {conv_rate_1:.2%}"
        )
        print(f"  [FAIL] Convergence: 1GPU={conv_rate_1:.1%}, 2GPU={conv_rate_2:.1%}")
    else:
        print(f"  [PASS] Convergence: 1GPU={conv_rate_1:.1%}, 2GPU={conv_rate_2:.1%}")

    # JSON output
    if args.json_output:
        out = {
            "benchmark": "pf31_regression",
            "n_toys": args.n_toys,
            "n_events": args.n_events,
            "wall_1gpu_s": wall_1,
            "wall_2gpu_s": wall_2,
            "scaling": scaling,
            "poi_mean_1gpu": poi_mean_1,
            "poi_mean_2gpu": poi_mean_2,
            "conv_1gpu": conv_1,
            "conv_2gpu": conv_2,
            "pass": len(failures) == 0,
            "failures": failures,
        }
        with open(args.json_output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to {args.json_output}")

    print()
    if failures:
        print("VERDICT: FAIL")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("VERDICT: PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
