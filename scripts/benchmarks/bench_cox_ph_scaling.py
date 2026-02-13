#!/usr/bin/env python3
"""Cox PH scaling benchmark: NextStat vs lifelines vs R survival::coxph().

N from 1K to 1M, timing median ± IQR over 50 runs.
Outputs JSON results + matplotlib scaling plot.

Usage:
    python scripts/benchmarks/bench_cox_ph_scaling.py [--runs 50] [--out-dir bench_results]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Data generation (shared across all frameworks)
# ---------------------------------------------------------------------------

def generate_cox_data(n: int, seed: int = 42) -> dict:
    """Generate synthetic survival data with 4 covariates."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.integers(0, 2, size=n).astype(float)
    x4 = rng.standard_normal(n)

    # True coefficients: beta = [0.5, -0.3, 0.8, -0.1]
    lp = 0.5 * x1 - 0.3 * x2 + 0.8 * x3 - 0.1 * x4
    baseline_hazard = 0.1
    scale = 1.0 / (baseline_hazard * np.exp(lp))
    times = rng.exponential(scale)

    # ~30% censoring
    censor_times = rng.exponential(np.median(times) * 2.0, size=n)
    events = (times <= censor_times).astype(bool)
    obs_times = np.minimum(times, censor_times)

    return {
        "times": obs_times.tolist(),
        "events": events.tolist(),
        "x": np.column_stack([x1, x2, x3, x4]),
    }


# ---------------------------------------------------------------------------
# NextStat benchmark
# ---------------------------------------------------------------------------

def _try_import_nextstat():
    try:
        import nextstat
        if callable(getattr(nextstat, "churn_risk_model", None)):
            return nextstat
        return None
    except ImportError:
        return None


def _write_cox_json(data: dict, path: Path):
    """Write data to JSON for CLI consumption."""
    payload = {
        "times": data["times"] if isinstance(data["times"], list) else data["times"].tolist(),
        "events": data["events"] if isinstance(data["events"], list) else data["events"].tolist(),
        "covariates": [row.tolist() for row in data["x"]],
        "covariate_names": ["x1", "x2", "x3", "x4"],
    }
    path.write_text(json.dumps(payload))


def bench_nextstat(data: dict, n_runs: int, tmp_dir: Path = None) -> list[float]:
    ns = _try_import_nextstat()

    if ns is not None:
        # Fast path: Python bindings available.
        times = data["times"] if isinstance(data["times"], list) else data["times"].tolist()
        events = data["events"] if isinstance(data["events"], list) else data["events"].tolist()
        covariates = [row.tolist() for row in data["x"]]
        names = ["x1", "x2", "x3", "x4"]

        timings = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            ns.churn_risk_model(times, events, covariates, names, conf_level=0.95)
            elapsed = time.perf_counter() - t0
            timings.append(elapsed)
        return timings

    # Fallback: CLI path.
    if tmp_dir is None:
        tmp_dir = Path("/tmp/nextstat_bench")
        tmp_dir.mkdir(exist_ok=True)

    nextstat_bin = _find_nextstat_binary()
    if nextstat_bin is None:
        print("  [SKIP] nextstat binary not found", file=sys.stderr)
        return []

    json_path = tmp_dir / "cox_input.json"
    _write_cox_json(data, json_path)

    timings = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        subprocess.run(
            [nextstat_bin, "churn", "risk-model", "-i", str(json_path)],
            capture_output=True, check=True,
        )
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
    return timings


def _find_nextstat_binary() -> str | None:
    """Find the nextstat CLI binary."""
    # Check release build first.
    candidates = [
        Path(__file__).resolve().parents[2] / "target" / "release" / "nextstat",
        Path(__file__).resolve().parents[2] / "target" / "debug" / "nextstat",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # Check PATH.
    try:
        subprocess.run(["nextstat", "--version"], capture_output=True, check=True)
        return "nextstat"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


# ---------------------------------------------------------------------------
# lifelines benchmark
# ---------------------------------------------------------------------------

def bench_lifelines(data: dict, n_runs: int) -> list[float]:
    try:
        import pandas as pd
        from lifelines import CoxPHFitter
    except ImportError:
        print("  [SKIP] lifelines not installed", file=sys.stderr)
        return []

    df = pd.DataFrame({
        "T": data["times"],
        "E": data["events"],
        "x1": data["x"][:, 0],
        "x2": data["x"][:, 1],
        "x3": data["x"][:, 2],
        "x4": data["x"][:, 3],
    })

    timings = []
    for _ in range(n_runs):
        cph = CoxPHFitter()
        t0 = time.perf_counter()
        cph.fit(df, duration_col="T", event_col="E", show_progress=False)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
    return timings


# ---------------------------------------------------------------------------
# R survival::coxph() benchmark
# ---------------------------------------------------------------------------

R_SCRIPT_TEMPLATE = r"""
library(survival)
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
n_runs <- as.integer(args[2])

d <- read.csv(data_path)
timings <- numeric(n_runs)
for (i in seq_len(n_runs)) {{
    t0 <- proc.time()["elapsed"]
    fit <- coxph(Surv(T, E) ~ x1 + x2 + x3 + x4, data = d)
    t1 <- proc.time()["elapsed"]
    timings[i] <- t1 - t0
}}
cat(paste(timings, collapse = ","))
"""


def bench_r(data: dict, n_runs: int, tmp_dir: Path) -> list[float]:
    try:
        subprocess.run(["Rscript", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("  [SKIP] Rscript not found", file=sys.stderr)
        return []

    # Write CSV.
    csv_path = tmp_dir / "cox_bench_data.csv"
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["T", "E", "x1", "x2", "x3", "x4"])
        for i in range(len(data["times"])):
            writer.writerow([
                data["times"][i],
                int(data["events"][i]),
                data["x"][i, 0],
                data["x"][i, 1],
                data["x"][i, 2],
                data["x"][i, 3],
            ])

    # Write R script.
    r_path = tmp_dir / "cox_bench.R"
    r_path.write_text(R_SCRIPT_TEMPLATE)

    try:
        result = subprocess.run(
            ["Rscript", str(r_path), str(csv_path), str(n_runs)],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            print(f"  [SKIP] R failed: {result.stderr[:200]}", file=sys.stderr)
            return []
        timings = [float(x) for x in result.stdout.strip().split(",")]
        return timings
    except subprocess.TimeoutExpired:
        print("  [SKIP] R timed out", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(results: list[dict], out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not installed, no plot generated", file=sys.stderr)
        return

    frameworks = {}
    for r in results:
        for fw_name, fw_data in r["frameworks"].items():
            if not fw_data["timings"]:
                continue
            if fw_name not in frameworks:
                frameworks[fw_name] = {"n": [], "median": [], "q25": [], "q75": []}
            frameworks[fw_name]["n"].append(r["n"])
            frameworks[fw_name]["median"].append(fw_data["median_s"])
            frameworks[fw_name]["q25"].append(fw_data["q25_s"])
            frameworks[fw_name]["q75"].append(fw_data["q75_s"])

    colors = {"nextstat": "#2563eb", "lifelines": "#dc2626", "r_survival": "#16a34a"}
    markers = {"nextstat": "o", "lifelines": "s", "r_survival": "^"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for fw_name, d in frameworks.items():
        n = np.array(d["n"])
        med = np.array(d["median"])
        q25 = np.array(d["q25"])
        q75 = np.array(d["q75"])
        color = colors.get(fw_name, "#666666")
        marker = markers.get(fw_name, "D")
        ax.plot(n, med, f"-{marker}", color=color, label=fw_name, linewidth=2, markersize=7)
        ax.fill_between(n, q25, q75, alpha=0.15, color=color)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (observations)", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title("Cox PH Scaling: NextStat vs lifelines vs R survival", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cox PH scaling benchmark")
    parser.add_argument("--runs", type=int, default=50, help="Number of timing runs per (N, framework)")
    parser.add_argument("--out-dir", type=str, default="bench_results", help="Output directory")
    parser.add_argument("--skip-r", action="store_true", help="Skip R benchmark")
    parser.add_argument("--skip-lifelines", action="store_true", help="Skip lifelines benchmark")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    ns = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
    results = []

    for n in ns:
        print(f"\n=== N = {n:,} ===")
        data = generate_cox_data(n, seed=42)
        entry = {"n": n, "frameworks": {}}

        # NextStat
        print(f"  NextStat ({args.runs} runs)...", end="", flush=True)
        t_ns = bench_nextstat(data, args.runs, tmp_dir)
        med = float(np.median(t_ns))
        q25 = float(np.percentile(t_ns, 25))
        q75 = float(np.percentile(t_ns, 75))
        entry["frameworks"]["nextstat"] = {
            "median_s": med, "q25_s": q25, "q75_s": q75, "timings": t_ns,
        }
        print(f" {med*1000:.1f} ms [{q25*1000:.1f}, {q75*1000:.1f}]")

        # lifelines
        if not args.skip_lifelines:
            # For N >= 500K, reduce runs to avoid excessive time.
            ll_runs = min(args.runs, 10) if n >= 500_000 else args.runs
            print(f"  lifelines ({ll_runs} runs)...", end="", flush=True)
            t_ll = bench_lifelines(data, ll_runs)
            if t_ll:
                med_ll = float(np.median(t_ll))
                q25_ll = float(np.percentile(t_ll, 25))
                q75_ll = float(np.percentile(t_ll, 75))
                entry["frameworks"]["lifelines"] = {
                    "median_s": med_ll, "q25_s": q25_ll, "q75_s": q75_ll, "timings": t_ll,
                }
                print(f" {med_ll*1000:.1f} ms [{q25_ll*1000:.1f}, {q75_ll*1000:.1f}]")
            else:
                entry["frameworks"]["lifelines"] = {"median_s": None, "q25_s": None, "q75_s": None, "timings": []}
        else:
            entry["frameworks"]["lifelines"] = {"median_s": None, "q25_s": None, "q75_s": None, "timings": []}

        # R
        if not args.skip_r:
            r_runs = min(args.runs, 5) if n >= 500_000 else min(args.runs, 20)
            print(f"  R survival ({r_runs} runs)...", end="", flush=True)
            t_r = bench_r(data, r_runs, tmp_dir)
            if t_r:
                med_r = float(np.median(t_r))
                q25_r = float(np.percentile(t_r, 25))
                q75_r = float(np.percentile(t_r, 75))
                entry["frameworks"]["r_survival"] = {
                    "median_s": med_r, "q25_s": q25_r, "q75_s": q75_r, "timings": t_r,
                }
                print(f" {med_r*1000:.1f} ms [{q25_r*1000:.1f}, {q75_r*1000:.1f}]")
            else:
                entry["frameworks"]["r_survival"] = {"median_s": None, "q25_s": None, "q75_s": None, "timings": []}
        else:
            entry["frameworks"]["r_survival"] = {"median_s": None, "q25_s": None, "q75_s": None, "timings": []}

        results.append(entry)

    # Save JSON.
    json_path = out_dir / "cox_ph_scaling.json"
    # Strip raw timings for compact JSON.
    compact = []
    for r in results:
        cr = {"n": r["n"], "frameworks": {}}
        for fw, fd in r["frameworks"].items():
            cr["frameworks"][fw] = {
                "median_s": fd["median_s"],
                "q25_s": fd["q25_s"],
                "q75_s": fd["q75_s"],
                "n_runs": len(fd["timings"]),
            }
        compact.append(cr)
    json_path.write_text(json.dumps(compact, indent=2))
    print(f"\nResults saved to {json_path}")

    # Plot.
    make_plot(results, out_dir / "cox_ph_scaling.png")


if __name__ == "__main__":
    main()
