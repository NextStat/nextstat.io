#!/usr/bin/env python3
"""NUTS v11 benchmark: NextStat NUTS vs CmdStan.

Models:
  1. StdNormal 10D — baseline sanity
  2. Eight Schools (non-centered) — classic hierarchical
  3. GLM Logistic (n=1000, p=10) — moderate correlated posterior

Methodology:
  - Fixed dataset_seed for reproducible data generation
  - 3 sampling seeds per model × {4 chains, 1000 warmup, 1000 samples}
  - NextStat metric default: diagonal (matches CmdStan default Euclidean metric)
  - Matched target densities:
      * Eight Schools: non-centered in BOTH engines
      * GLM logistic: Normal priors in BOTH engines
  - Metrics: ESS_bulk/s, ESS_tail/s, R-hat, divergence rate, E-BFMI
  - CmdStan via cmdstanpy
  - NS via nextstat.sample()
  - Output: JSON artifact (+ env metadata) + markdown table

Usage:
    python bench_nuts_vs_cmdstan.py [--seeds 42,43,44] [--out-dir results/]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Always benchmark the local extension/package from this repo, not any globally
# installed nextstat wheel in site-packages.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "bindings" / "ns-py" / "python"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from _bayesian_nextstat_bench import (
    BenchResult,
    DATASET_SEED,
    DEFAULT_SEEDS,
    _assert_nextstat_harness_contract,
    _ns_available,
    collect_environment,
    gen_eight_schools_data,
    gen_glm_logistic_data,
    max_rhat,
    median_wall,
    min_ess_per_sec,
    run_nextstat_eight_schools,
    run_nextstat_glm_logistic,
    run_nextstat_std_normal,
)


# ── Configuration ──────────────────────────────────────────────────────

N_CHAINS = 4
N_WARMUP = 1000
N_SAMPLES = 1000
# ── Stan model code ───────────────────────────────────────────────────

STAN_STD_NORMAL = """
data {
  int<lower=1> D;
}
parameters {
  vector[D] x;
}
model {
  x ~ std_normal();
}
"""

STAN_EIGHT_SCHOOLS_NCP = """
data {
  int<lower=0> J;
  array[J] real y;
  array[J] real<lower=0> sigma;
}
parameters {
  real mu;
  real<lower=0> tau;
  array[J] real theta_raw;
}
transformed parameters {
  array[J] real theta;
  for (j in 1:J)
    theta[j] = mu + tau * theta_raw[j];
}
model {
  mu ~ normal(0, 5);
  tau ~ cauchy(0, 5);
  theta_raw ~ std_normal();
  y ~ normal(theta, sigma);
}
"""

STAN_GLM_LOGISTIC = """
data {
  int<lower=0> N;
  int<lower=0> P;
  matrix[N, P] X;
  array[N] int<lower=0, upper=1> y;
}
parameters {
  real alpha;
  vector[P] beta;
}
model {
  alpha ~ normal(0, 5);
  beta ~ normal(0, 2.5);
  y ~ bernoulli_logit(alpha + X * beta);
}
"""


# ── CmdStan runner ────────────────────────────────────────────────────

def run_cmdstan(
    stan_code: str,
    data: dict,
    model_name: str,
    seed: int,
) -> BenchResult:
    """Compile + sample with CmdStanPy, return BenchResult."""
    try:
        from cmdstanpy import CmdStanModel, cmdstan_path, set_cmdstan_path
    except ImportError:
        print("  [skip] cmdstanpy not installed")
        return BenchResult(model=model_name, engine="cmdstan", seed=seed, wall_secs=0.0)

    # Auto-discover workspace-local CmdStan install if global path is not configured.
    try:
        _ = cmdstan_path()
    except Exception:
        local_root = Path("vendor/cmdstan")
        candidates = sorted(local_root.glob("cmdstan-*"))
        if candidates:
            set_cmdstan_path(str(candidates[-1].resolve()))

    with tempfile.TemporaryDirectory() as tmpdir:
        stan_file = Path(tmpdir) / f"{model_name}.stan"
        stan_file.write_text(stan_code)
        data_file = Path(tmpdir) / f"{model_name}.json"
        data_file.write_text(json.dumps(data))

        sm = CmdStanModel(stan_file=str(stan_file))

        t0 = time.perf_counter()
        fit = sm.sample(
            data=str(data_file),
            chains=N_CHAINS,
            iter_warmup=N_WARMUP,
            iter_sampling=N_SAMPLES,
            seed=seed,
            show_console=False,
        )
        wall_secs = time.perf_counter() - t0

    # Extract diagnostics
    summary = fit.summary()
    diag = fit.diagnose()

    # Compute per-parameter ESS and R-hat from summary.
    # Keep only base model parameters for apples-to-apples parity with NextStat.
    def keep_param(name: str) -> bool:
        if model_name == "std_normal_10d":
            return name.startswith("x[")
        if model_name == "eight_schools":
            return name == "mu" or name == "tau" or name.startswith("theta_raw[")
        if model_name == "glm_logistic":
            return name == "alpha" or name.startswith("beta[")
        return True

    ess_bulk = {}
    ess_tail = {}
    r_hat = {}
    for param in summary.index:
        if param in ("lp__", "accept_stat__", "stepsize__", "treedepth__",
                      "n_leapfrog__", "divergent__", "energy__"):
            continue
        if not keep_param(param):
            continue
        row = summary.loc[param]
        if "ESS_bulk" in row and np.isfinite(row["ESS_bulk"]):
            ess_bulk[param] = float(row["ESS_bulk"])
        if "ESS_tail" in row and np.isfinite(row["ESS_tail"]):
            ess_tail[param] = float(row["ESS_tail"])
        if "R_hat" in row and np.isfinite(row["R_hat"]):
            r_hat[param] = float(row["R_hat"])

    # Divergences
    n_div = int(np.sum(fit.method_variables()["divergent__"]))
    n_total = N_CHAINS * N_SAMPLES
    div_rate = n_div / n_total if n_total > 0 else 0.0

    # E-BFMI per chain
    energy = fit.method_variables()["energy__"]  # shape (n_samples, n_chains)
    ebfmi = []
    for c in range(N_CHAINS):
        e = energy[:, c] if energy.ndim > 1 else energy
        de = np.diff(e)
        bfmi = float(np.mean(de**2) / np.var(e)) if np.var(e) > 0 else 0.0
        ebfmi.append(bfmi)

    return BenchResult(
        model=model_name,
        engine="cmdstan",
        seed=seed,
        wall_secs=wall_secs,
        ess_bulk=ess_bulk,
        ess_tail=ess_tail,
        r_hat=r_hat,
        divergence_rate=div_rate,
        ebfmi=ebfmi,
    )


# ── Aggregate & report ────────────────────────────────────────────────

def format_table(ns_results: dict, stan_results: dict, models: list[str]) -> str:
    """Format markdown comparison table."""
    lines = [
        "| Model | NS ESS_bulk/s | Stan ESS_bulk/s | Ratio | NS R-hat | Stan R-hat | NS div% | Stan div% |",
        "|-------|--------------|-----------------|-------|----------|------------|---------|-----------|",
    ]
    for m in models:
        ns = ns_results.get(m, [])
        stan = stan_results.get(m, [])
        ns_ess = min_ess_per_sec(ns, "ess_bulk")
        stan_ess = min_ess_per_sec(stan, "ess_bulk")
        ratio_str = f"{(ns_ess / stan_ess):.2f}x" if stan_ess > 0 else "n/a"
        ns_rhat = max_rhat(ns)
        stan_rhat = max_rhat(stan)
        ns_div = statistics.median([r.divergence_rate * 100 for r in ns]) if ns else 0
        stan_div = statistics.median([r.divergence_rate * 100 for r in stan]) if stan else 0
        lines.append(
            f"| {m} | {ns_ess:.0f} | {stan_ess:.0f} | {ratio_str} "
            f"| {ns_rhat:.4f} | {stan_rhat:.4f} | {ns_div:.1f}% | {stan_div:.1f}% |"
        )
    return "\n".join(lines)
# ── Main ──────────────────────────────────────────────────────────────

def main():
    global N_CHAINS, N_WARMUP, N_SAMPLES

    parser = argparse.ArgumentParser(description="NUTS v11: NS vs CmdStan benchmark")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                        help="Comma-separated sampling seeds")
    parser.add_argument("--out-dir", default="results/nuts_v11", help="Output directory")
    parser.add_argument("--skip-cmdstan", action="store_true", help="Skip CmdStan runs")
    parser.add_argument("--skip-nextstat", action="store_true", help="Skip NextStat runs")
    parser.add_argument("--n-chains", type=int, default=N_CHAINS, help="Number of chains")
    parser.add_argument("--n-warmup", type=int, default=N_WARMUP, help="Warmup iterations")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES, help="Sampling iterations")
    parser.add_argument(
        "--metric",
        default="diagonal",
        help="Metric type for NextStat: diagonal (default, CmdStan-like), dense, auto",
    )
    parser.add_argument(
        "--glm-n",
        type=int,
        default=1000,
        help="Number of observations for GLM logistic benchmark",
    )
    parser.add_argument(
        "--glm-p",
        type=int,
        default=10,
        help="Number of predictors for GLM logistic benchmark",
    )
    args = parser.parse_args()
    if args.n_chains <= 0 or args.n_warmup < 0 or args.n_samples <= 0:
        raise SystemExit("Invalid run config: n_chains>0, n_warmup>=0, n_samples>0 required")

    N_CHAINS = args.n_chains
    N_WARMUP = args.n_warmup
    N_SAMPLES = args.n_samples

    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = ["std_normal_10d", "eight_schools", "glm_logistic"]

    all_results: list[dict[str, Any]] = []
    ns_by_model: dict[str, list[BenchResult]] = {m: [] for m in models}
    stan_by_model: dict[str, list[BenchResult]] = {m: [] for m in models}

    # ── NextStat runs ──
    if not args.skip_nextstat:
        if not _ns_available():
            print("[warn] nextstat not importable — skipping NS runs")
        else:
            _assert_nextstat_harness_contract()
            import nextstat  # type: ignore
            print(f"[ns] module={getattr(nextstat, '__file__', 'unknown')}")
            print(f"[ns] version={getattr(nextstat, '__version__', 'unknown')}")
            for seed in seeds:
                print(f"[ns] std_normal_10d seed={seed} ...", end=" ", flush=True)
                r = run_nextstat_std_normal(
                    seed,
                    metric=args.metric,
                    n_chains=args.n_chains,
                    n_warmup=args.n_warmup,
                    n_samples=args.n_samples,
                )
                ns_by_model["std_normal_10d"].append(r)
                all_results.append(r.__dict__)
                print(f"{r.wall_secs:.2f}s")

                print(f"[ns] eight_schools seed={seed} ...", end=" ", flush=True)
                r = run_nextstat_eight_schools(
                    seed,
                    metric=args.metric,
                    n_chains=args.n_chains,
                    n_warmup=args.n_warmup,
                    n_samples=args.n_samples,
                )
                ns_by_model["eight_schools"].append(r)
                all_results.append(r.__dict__)
                print(f"{r.wall_secs:.2f}s")

                print(f"[ns] glm_logistic seed={seed} ...", end=" ", flush=True)
                r = run_nextstat_glm_logistic(
                    seed,
                    n=args.glm_n,
                    p=args.glm_p,
                    metric=args.metric,
                    n_chains=args.n_chains,
                    n_warmup=args.n_warmup,
                    n_samples=args.n_samples,
                )
                ns_by_model["glm_logistic"].append(r)
                all_results.append(r.__dict__)
                print(f"{r.wall_secs:.2f}s")

    # ── CmdStan runs ──
    if not args.skip_cmdstan:
        for seed in seeds:
            print(f"[cmdstan] std_normal_10d seed={seed} ...", end=" ", flush=True)
            r = run_cmdstan(STAN_STD_NORMAL, {"D": 10}, "std_normal_10d", seed)
            stan_by_model["std_normal_10d"].append(r)
            all_results.append(r.__dict__)
            print(f"{r.wall_secs:.2f}s")

            print(f"[cmdstan] eight_schools seed={seed} ...", end=" ", flush=True)
            r = run_cmdstan(
                STAN_EIGHT_SCHOOLS_NCP,
                gen_eight_schools_data(),
                "eight_schools",
                seed,
            )
            stan_by_model["eight_schools"].append(r)
            all_results.append(r.__dict__)
            print(f"{r.wall_secs:.2f}s")

            print(f"[cmdstan] glm_logistic seed={seed} ...", end=" ", flush=True)
            data = gen_glm_logistic_data(n=args.glm_n, p=args.glm_p)
            r = run_cmdstan(STAN_GLM_LOGISTIC, data, "glm_logistic", seed)
            stan_by_model["glm_logistic"].append(r)
            all_results.append(r.__dict__)
            print(f"{r.wall_secs:.2f}s")

    # ── Save JSON artifact ──
    metadata = collect_environment(
        args.metric,
        args.glm_n,
        args.glm_p,
        seeds,
        models=models,
        methods=["nuts"],
        n_chains=args.n_chains,
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
    )
    try:
        from cmdstanpy import cmdstan_path  # type: ignore

        metadata["cmdstan"] = {"path": cmdstan_path()}
    except Exception:
        metadata["cmdstan"] = None
    summary = {}
    for m in models:
        ns = ns_by_model.get(m, [])
        stan = stan_by_model.get(m, [])
        ns_ess = min_ess_per_sec(ns, "ess_bulk")
        stan_ess = min_ess_per_sec(stan, "ess_bulk")
        summary[m] = {
            "ns_ess_bulk_per_s": ns_ess,
            "stan_ess_bulk_per_s": stan_ess,
            "ns_over_stan": (ns_ess / stan_ess) if stan_ess > 0 else None,
            "ns_rhat_max_median": max_rhat(ns) if ns else None,
            "stan_rhat_max_median": max_rhat(stan) if stan else None,
            "ns_divergence_rate_median": (
                statistics.median([r.divergence_rate for r in ns]) if ns else None
            ),
            "stan_divergence_rate_median": (
                statistics.median([r.divergence_rate for r in stan]) if stan else None
            ),
        }

    artifact = {
        "metadata": metadata,
        "summary": summary,
        "runs": all_results,
    }

    json_path = out_dir / "bench_nuts_vs_cmdstan.json"
    with open(json_path, "w") as f:
        json.dump(artifact, f, indent=2, default=str)
    print(f"\n[saved] {json_path}")

    # ── Summary table ──
    table = format_table(ns_by_model, stan_by_model, models)
    md_path = out_dir / "bench_nuts_vs_cmdstan.md"
    with open(md_path, "w") as f:
        f.write(f"# NUTS v11: NextStat vs CmdStan\n\n")
        f.write(f"Seeds: {seeds} | Chains: {N_CHAINS} | Warmup: {N_WARMUP} | Samples: {N_SAMPLES}\n\n")
        f.write(f"GLM config: N={args.glm_n}, P={args.glm_p}\n\n")
        f.write("Environment:\n")
        f.write(f"- Python: {metadata.get('python')}\n")
        f.write(
            f"- Platform: {metadata.get('platform', {}).get('system')} "
            f"{metadata.get('platform', {}).get('release')} "
            f"({metadata.get('platform', {}).get('machine')})\n"
        )
        f.write(f"- Git commit: {metadata.get('git_commit')}\n")
        f.write(f"- NextStat: {metadata.get('nextstat_version')}\n")
        cmdstan_meta = metadata.get("cmdstan")
        f.write(
            f"- CmdStan path: {cmdstan_meta.get('path') if isinstance(cmdstan_meta, dict) else None}\n"
        )
        f.write("\n")
        f.write("Target parity:\n")
        f.write("- Eight Schools: non-centered in both engines\n")
        f.write("- GLM logistic: alpha~N(0,5), beta_j~N(0,2.5) in both engines\n\n")
        f.write(table)
        f.write("\n\n")

        # Per-model detail
        for m in models:
            f.write(f"\n## {m}\n\n")
            ns = ns_by_model.get(m, [])
            stan = stan_by_model.get(m, [])
            if ns:
                f.write(f"- NS wall time (median): {median_wall(ns):.2f}s\n")
                f.write(f"- NS min ESS_bulk/s: {min_ess_per_sec(ns, 'ess_bulk'):.0f}\n")
                f.write(f"- NS min ESS_tail/s: {min_ess_per_sec(ns, 'ess_tail'):.0f}\n")
                f.write(f"- NS metric: {ns[0].metric_type}\n")
            if stan:
                f.write(f"- Stan wall time (median): {median_wall(stan):.2f}s\n")
                f.write(f"- Stan min ESS_bulk/s: {min_ess_per_sec(stan, 'ess_bulk'):.0f}\n")
                f.write(f"- Stan min ESS_tail/s: {min_ess_per_sec(stan, 'ess_tail'):.0f}\n")

    print(f"[saved] {md_path}")
    print(f"\n{table}")


if __name__ == "__main__":
    main()
