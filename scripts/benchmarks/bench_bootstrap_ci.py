#!/usr/bin/env python3
"""Bootstrap CI benchmark: Rayon-parallel NextStat vs single-thread lifelines.

B from 100 to 10,000 resamples on 100K subscribers.
Uses `nextstat churn bootstrap-hr` for Rayon-parallel bootstrap (single CLI call per B).
Compares with lifelines sequential bootstrap.

Usage:
    python scripts/benchmarks/bench_bootstrap_ci.py [--n-obs 100000] [--runs 3] [--out-dir bench_results]
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from _parse_utils import parse_json_stdout


def _find_nextstat_binary() -> str | None:
    candidates = [
        Path(__file__).resolve().parents[2] / "target" / "release" / "nextstat",
        Path(__file__).resolve().parents[2] / "target" / "debug" / "nextstat",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def generate_and_write_data(n: int, seed: int, out_path: Path, nextstat_bin: str):
    """Generate data via CLI and write to JSON."""
    subprocess.run(
        [nextstat_bin, "churn", "generate-data",
         "--n-customers", str(n), "--seed", str(seed), "-o", str(out_path)],
        capture_output=True, check=True,
    )


# ---------------------------------------------------------------------------
# NextStat benchmark (Rayon-parallel bootstrap via single CLI call)
# ---------------------------------------------------------------------------

def bench_nextstat_bootstrap(
    data_path: Path, nextstat_bin: str, n_bootstrap: int, seed: int
) -> tuple[float, int, int]:
    """Returns (wall_time_s, n_converged, n_bootstrap)."""
    t0 = time.perf_counter()
    r = subprocess.run(
        [nextstat_bin, "churn", "bootstrap-hr",
         "-i", str(data_path),
         "--n-bootstrap", str(n_bootstrap),
         "--seed", str(seed)],
        capture_output=True, text=True, check=True,
    )
    wall = time.perf_counter() - t0
    out = parse_json_stdout(r.stdout)
    return wall, out["n_converged"], out.get("elapsed_s", wall)


# ---------------------------------------------------------------------------
# lifelines benchmark (single-thread sequential bootstrap)
# ---------------------------------------------------------------------------

def bench_lifelines_bootstrap(
    data_path: Path, n_bootstrap: int, seed: int
) -> tuple[float, int]:
    """Returns (wall_time_s, n_converged)."""
    try:
        import pandas as pd
        from lifelines import CoxPHFitter
    except ImportError:
        return -1.0, 0

    with open(data_path) as f:
        raw = json.load(f)

    times = np.array(raw["times"])
    events = np.array(raw["events"])
    covariates = np.array(raw["covariates"])
    n = len(times)
    rng = np.random.default_rng(seed)

    n_converged = 0
    t0 = time.perf_counter()
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        df = pd.DataFrame({
            "T": times[idx],
            "E": events[idx],
            "x1": covariates[idx, 0] if covariates.ndim == 2 else [covariates[i][0] for i in idx],
            "x2": covariates[idx, 1] if covariates.ndim == 2 else [covariates[i][1] for i in idx],
            "x3": covariates[idx, 2] if covariates.ndim == 2 else [covariates[i][2] for i in idx],
            "x4": covariates[idx, 3] if covariates.ndim == 2 else [covariates[i][3] for i in idx],
        })
        cph = CoxPHFitter()
        try:
            cph.fit(df, duration_col="T", event_col="E", show_progress=False)
            n_converged += 1
        except Exception:
            pass
    wall = time.perf_counter() - t0
    return wall, n_converged


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(results: list[dict], out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not installed", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bs_ns = [r["B"] for r in results if r.get("nextstat_rayon_s")]
    t_ns = [r["nextstat_rayon_s"] for r in results if r.get("nextstat_rayon_s")]
    bs_ll = [r["B"] for r in results if r.get("lifelines_s") and r["lifelines_s"] > 0]
    t_ll = [r["lifelines_s"] for r in results if r.get("lifelines_s") and r["lifelines_s"] > 0]

    ax.plot(bs_ns, t_ns, "-o", color="#2563eb", label="NextStat (Rayon)", linewidth=2, markersize=7)
    if t_ll:
        ax.plot(bs_ll, t_ll, "-s", color="#dc2626", label="lifelines (single-thread)", linewidth=2, markersize=7)
    ax.set_xlabel("Bootstrap resamples (B)", fontsize=12)
    ax.set_ylabel("Wall time (seconds)", fontsize=12)
    ax.set_title(f"Bootstrap Cox PH CI — {results[0]['n_obs']:,} observations", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    paired = [(r["B"], r["nextstat_rayon_s"], r["lifelines_s"])
              for r in results if r.get("nextstat_rayon_s") and r.get("lifelines_s") and r["lifelines_s"] > 0]
    if paired:
        speedup = [ll / ns for _, ns, ll in paired]
        labels = [str(b) for b, _, _ in paired]
        ax2.bar(range(len(paired)), speedup, color="#2563eb", alpha=0.7)
        ax2.set_xticks(range(len(paired)))
        ax2.set_xticklabels(labels)
        ax2.set_xlabel("Bootstrap resamples (B)", fontsize=12)
        ax2.set_ylabel("Speedup (lifelines / NextStat)", fontsize=12)
        ax2.set_title("Speedup vs lifelines", fontsize=13)
        ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(0.5, 0.5, "lifelines not available\nfor speedup comparison",
                 ha="center", va="center", fontsize=12, transform=ax2.transAxes)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bootstrap CI benchmark")
    parser.add_argument("--n-obs", type=int, default=100_000, help="Number of observations")
    parser.add_argument("--runs", type=int, default=3, help="Repeat each B-count this many times (take median)")
    parser.add_argument("--out-dir", type=str, default="bench_results", help="Output directory")
    parser.add_argument("--skip-lifelines", action="store_true")
    args = parser.parse_args()

    nextstat_bin = _find_nextstat_binary()
    if nextstat_bin is None:
        print("ERROR: nextstat binary not found. Run `cargo build --release -p ns-cli` first.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate data once.
    data_path = out_dir / "bootstrap_data.json"
    print(f"Generating {args.n_obs:,} observations...", end="", flush=True)
    generate_and_write_data(args.n_obs, seed=42, out_path=data_path, nextstat_bin=nextstat_bin)
    print(" done")

    b_values = [100, 250, 500, 1_000, 2_500, 5_000, 10_000]
    results = []

    for B in b_values:
        print(f"\n=== B = {B:,} ===")
        entry = {"B": B, "n_obs": args.n_obs}

        # NextStat (Rayon-parallel, single CLI call).
        ns_times = []
        ns_rayon_times = []
        for r in range(args.runs):
            wall, n_conv, rayon_s = bench_nextstat_bootstrap(data_path, nextstat_bin, B, seed=42 + r)
            ns_times.append(wall)
            ns_rayon_times.append(rayon_s)
            print(f"  NextStat run {r+1}/{args.runs}: {rayon_s:.2f}s Rayon ({wall:.2f}s wall), {n_conv}/{B} converged")

        entry["nextstat_wall_s"] = float(np.median(ns_times))
        entry["nextstat_rayon_s"] = float(np.median(ns_rayon_times))

        # lifelines (sequential, Python-side loop).
        if not args.skip_lifelines:
            ll_runs_count = min(args.runs, 2) if B <= 1_000 else 1
            if B > 2_500:
                print(f"  lifelines: skipping B={B} (too slow single-thread)")
                entry["lifelines_s"] = None
            else:
                ll_times = []
                for r in range(ll_runs_count):
                    t, n_conv = bench_lifelines_bootstrap(data_path, B, seed=42 + r)
                    if t < 0:
                        print("  [SKIP] lifelines not installed")
                        break
                    ll_times.append(t)
                    print(f"  lifelines run {r+1}: {t:.2f}s, {n_conv}/{B} converged")
                entry["lifelines_s"] = float(np.median(ll_times)) if ll_times else None
        else:
            entry["lifelines_s"] = None

        if entry.get("lifelines_s") and entry["lifelines_s"] > 0 and entry["nextstat_rayon_s"] > 0:
            speedup = entry["lifelines_s"] / entry["nextstat_rayon_s"]
            print(f"  Speedup: {speedup:.1f}× (Rayon vs lifelines)")

        results.append(entry)

    # Save JSON.
    json_path = out_dir / "bootstrap_ci.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {json_path}")

    make_plot(results, out_dir / "bootstrap_ci.png")


if __name__ == "__main__":
    main()
