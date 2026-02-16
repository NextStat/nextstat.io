#!/usr/bin/env python3
"""A100 Triple Benchmark: NS LAPS GPU vs NS CPU MAMS vs BlackJAX GPU.

Runs models with cold (first run) and warm (JIT-cached) timings.
Reports: wall time, ESS, ESS/s for each configuration.

Three comparisons:
  1) NS LAPS GPU (4096 chains, 1 A100)
  2) NS CPU MAMS (4 chains, 256 vCPU)
  3) BlackJAX adjusted_mclmc GPU (4096 chains, 1 A100, cold + warm)

Usage:
    python a100_triple_bench.py [--n-chains-gpu 4096] [--n-samples 2000] [--seed 42]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# Environment snapshot for reproducibility
sys.path.insert(0, str(Path(__file__).resolve().parent / "nextstat-public-benchmarks"))
from scripts.bench_env import collect_environment, print_environment


# ---------------------------------------------------------------------------
# ESS / R-hat via arviz
# ---------------------------------------------------------------------------

def _ess_rhat(draws_3d):
    """Compute (min_ESS, max_Rhat) from draws array (n_chains, n_samples, n_params)."""
    try:
        import arviz as az
        n_chains, n_samples, n_params = draws_3d.shape
        data_dict = {f"x{i}": draws_3d[:, :, i] for i in range(n_params)}
        dataset = az.convert_to_dataset(data_dict)
        ess = az.ess(dataset)
        rh = az.rhat(dataset)
        ess_vals = [float(ess[k].values) for k in ess.data_vars]
        rh_vals = [float(rh[k].values) for k in rh.data_vars]
        return min(ess_vals) if ess_vals else None, max(rh_vals) if rh_vals else None
    except Exception as e:
        print(f"    arviz error: {e}")
        return None, None


def _make_glm_data(n=5000, p=20, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta_true = rng.standard_normal(p) * 0.5
    logits = X @ beta_true
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logits))).astype(float)
    return X, y, n, p


def _extract_stats(sample_stats):
    """Extract gradient eval count from NextStat sample_stats."""
    if not isinstance(sample_stats, dict):
        return None
    n_leapfrog = sample_stats.get("n_leapfrog")
    if not isinstance(n_leapfrog, list) or not n_leapfrog:
        return None
    if isinstance(n_leapfrog[0], list):
        return sum(int(v) for chain in n_leapfrog for v in chain)
    return sum(int(v) for v in n_leapfrog)


# ---------------------------------------------------------------------------
# NS LAPS GPU
# ---------------------------------------------------------------------------

def bench_ns_laps(
    model_name,
    n_chains,
    n_warmup,
    n_samples,
    seed,
    report_chains,
    glm_warmup_floor=200,
    glm_target_accept=0.95,
    funnel_warmup_floor=500,
    funnel_target_accept=0.9,
):
    import nextstat

    if model_name == "std_normal_10d":
        laps_model, laps_data = "std_normal", {"dim": 10}
    elif model_name == "eight_schools":
        laps_model = "eight_schools"
        laps_data = {
            "y": [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
            "sigma": [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
        }
    elif model_name == "neal_funnel_10d":
        laps_model, laps_data = "neal_funnel", {"dim": 10}
    elif model_name == "glm_logistic":
        X, y, n, p = _make_glm_data()
        laps_model = "glm_logistic"
        laps_data = {"x": X.ravel().tolist(), "y": y.tolist(), "n": n, "p": p}
    else:
        raise ValueError(model_name)

    # Geometry-specific controls.
    #
    # Quality-speed policy:
    # - GLM: enforce a warmup floor and slightly stricter target_accept to
    #   reduce R-hat drift in short benchmark runs.
    # - Funnel: R-hat-first tuning for short benchmark budgets.
    if model_name == "neal_funnel_10d":
        sync_interval = 50
        max_leapfrog = 16384
        target_accept = funnel_target_accept
        n_warmup_eff = max(n_warmup, funnel_warmup_floor)
    elif model_name == "glm_logistic":
        sync_interval = 50
        max_leapfrog = 8192
        target_accept = glm_target_accept
        n_warmup_eff = max(n_warmup, glm_warmup_floor)
    else:
        sync_interval = 50
        max_leapfrog = 8192
        target_accept = 0.9
        n_warmup_eff = n_warmup
    report_chains = min(report_chains, n_chains)

    t0 = time.perf_counter()
    result = nextstat.sample(
        laps_model,
        method="laps",
        model_data=laps_data,
        n_chains=n_chains,
        n_warmup=n_warmup_eff,
        n_samples=n_samples,
        seed=seed,
        target_accept=target_accept,
        max_leapfrog=max_leapfrog,
        device_ids=[0],
        sync_interval=sync_interval,
        welford_chains=256,
        batch_size=1000,
        fused_transitions=1000,
        report_chains=report_chains,
        diagonal_precond=True,  # Per-chain ε + Welford works for all models
    )
    wall_s = time.perf_counter() - t0

    # Extract diagnostics
    diag = result.get("diagnostics", {})
    ess_bulk = diag.get("ess_bulk", {})
    r_hat = diag.get("r_hat", {})
    min_ess = None
    max_rhat = None
    if ess_bulk:
        vals = [float(v) for v in ess_bulk.values() if v is not None and math.isfinite(float(v))]
        min_ess = min(vals) if vals else None
    if r_hat:
        vals = [float(v) for v in r_hat.values() if v is not None and math.isfinite(float(v))]
        max_rhat = max(vals) if vals else None
    quality = diag.get("quality", {})
    quality_status = quality.get("status") if isinstance(quality, dict) else None
    quality_failures = quality.get("failures", []) if isinstance(quality, dict) else []
    n_report_chains = result.get("n_report_chains", result.get("n_chains"))
    n_grad_evals = _extract_stats(result.get("sample_stats", {}))
    ess_per_grad = (min_ess / n_grad_evals) if (min_ess is not None and n_grad_evals) else None

    return {"engine": "NS_LAPS_GPU", "model": model_name, "wall_s": wall_s,
            "wall_warm": wall_s, "min_ess": min_ess, "max_rhat": max_rhat,
            "quality_status": quality_status, "quality_failures": quality_failures,
            "n_report_chains": n_report_chains,
            "effective_n_warmup": n_warmup_eff,
            "effective_target_accept": target_accept,
            "effective_sync_interval": sync_interval,
            "effective_max_leapfrog": max_leapfrog,
            "n_chains": n_chains, "n_samples": n_samples,
            "n_grad_evals": n_grad_evals, "ess_per_grad": ess_per_grad}


# ---------------------------------------------------------------------------
# NS CPU MAMS
# ---------------------------------------------------------------------------

def bench_ns_cpu_mams(model_name, n_chains, n_warmup, n_samples, seed):
    import nextstat

    if model_name == "std_normal_10d":
        model_obj = nextstat.StdNormalModel(dim=10)
    elif model_name == "eight_schools":
        model_obj = nextstat.EightSchoolsModel(
            y=[28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
            sigma=[15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
        )
    elif model_name == "neal_funnel_10d":
        model_obj = nextstat.FunnelModel(dim=10)
    elif model_name == "glm_logistic":
        X, y, n, p = _make_glm_data()
        model_obj = nextstat.LogisticRegressionModel(
            X.tolist(), [int(yi) for yi in y],
        )
    else:
        raise ValueError(model_name)

    t0 = time.perf_counter()
    result = nextstat.sample(
        model_obj,
        method="mams",
        n_chains=n_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        seed=seed,
        target_accept=0.9,
    )
    wall_s = time.perf_counter() - t0

    diag = result.get("diagnostics", {})
    ess_bulk = diag.get("ess_bulk", {})
    r_hat = diag.get("r_hat", {})
    min_ess = None
    max_rhat = None
    if ess_bulk:
        vals = [float(v) for v in ess_bulk.values() if v is not None and math.isfinite(float(v))]
        min_ess = min(vals) if vals else None
    if r_hat:
        vals = [float(v) for v in r_hat.values() if v is not None and math.isfinite(float(v))]
        max_rhat = max(vals) if vals else None

    n_grad_evals = _extract_stats(result.get("sample_stats", {}))
    ess_per_grad = (min_ess / n_grad_evals) if (min_ess is not None and n_grad_evals) else None

    return {"engine": "NS_CPU_MAMS", "model": model_name, "wall_s": wall_s,
            "wall_warm": wall_s, "min_ess": min_ess, "max_rhat": max_rhat,
            "n_chains": n_chains, "n_samples": n_samples,
            "n_grad_evals": n_grad_evals, "ess_per_grad": ess_per_grad}


# ---------------------------------------------------------------------------
# BlackJAX GPU (adjusted_mclmc)
# ---------------------------------------------------------------------------

def _bj_logdensity(model_name):
    """Return (logdensity_fn, dim) for BlackJAX."""
    import jax.numpy as jnp

    if model_name == "std_normal_10d":
        dim = 10
        def logdensity(x):
            return -0.5 * jnp.sum(x ** 2)
        return logdensity, dim

    elif model_name == "eight_schools":
        y = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
        sigma = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
        dim = 10  # mu, log_tau, theta_raw[0..7]
        def logdensity(x):
            mu = x[0]
            log_tau = x[1]
            tau = jnp.exp(log_tau)
            theta_raw = x[2:]
            theta = mu + tau * theta_raw
            ll = -0.5 * jnp.sum(((y - theta) / sigma) ** 2)
            lp_mu = -0.5 * (mu / 5.0) ** 2
            lp_tau = -jnp.log(1 + (tau / 5.0) ** 2) + log_tau
            lp_theta = -0.5 * jnp.sum(theta_raw ** 2)
            return ll + lp_mu + lp_tau + lp_theta
        return logdensity, dim

    elif model_name == "neal_funnel_10d":
        dim = 10
        def logdensity(x):
            v = x[0]
            log_p_v = -0.5 * v ** 2 / 9.0
            log_p_x = -0.5 * jnp.sum(x[1:] ** 2) * jnp.exp(-v) - 0.5 * (dim - 1) * v
            return log_p_v + log_p_x
        return logdensity, dim

    elif model_name == "glm_logistic":
        X_np, y_np, n, p = _make_glm_data()
        X = jnp.array(X_np)
        y_jax = jnp.array(y_np)
        dim = p
        def logdensity(beta):
            logits = X @ beta
            ll = jnp.sum(y_jax * logits - jnp.logaddexp(0.0, logits))
            lp = -0.5 * jnp.sum(beta ** 2)
            return ll + lp
        return logdensity, dim

    raise ValueError(model_name)


def _blackjax_manual_warmup(logdensity, dim, n_warmup, key, target_accept=0.9):
    """Manual warmup for adjusted_mclmc (BlackJAX 1.3 built-in warmup is broken).

    Simple Dual Averaging on a single chain to find step_size and n_steps.
    """
    import jax
    import jax.numpy as jnp
    import blackjax

    kernel = blackjax.adjusted_mclmc.build_kernel(
        logdensity_fn=logdensity,
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
    )
    state = blackjax.mcmc.adjusted_mclmc.init(
        position=jnp.zeros(dim), logdensity_fn=logdensity,
    )

    # Dual averaging for step_size
    L = float(np.pi * np.sqrt(dim))
    step_size = L / 10.0
    mu = np.log(10.0 * step_size)
    log_step_size_bar = 0.0
    h_bar = 0.0
    gamma, t0, kappa = 0.05, 10.0, 0.75

    for i in range(n_warmup):
        key, step_key = jax.random.split(key)
        n_steps = max(1, int(round(L / step_size)))
        state, info = kernel(
            rng_key=step_key, state=state,
            step_size=step_size, num_integration_steps=n_steps,
        )
        accept = float(info.acceptance_rate)

        # DA update
        m = i + 1
        w = 1.0 / (m + t0)
        h_bar = (1 - w) * h_bar + w * (target_accept - accept)
        log_step_size = mu - (np.sqrt(m) / gamma) * h_bar
        step_size = float(np.exp(log_step_size))
        step_size = max(1e-5, min(step_size, L * 0.5))
        m_pow = m ** (-kappa)
        log_step_size_bar = m_pow * log_step_size + (1 - m_pow) * log_step_size_bar

    step_size = float(np.exp(log_step_size_bar))
    step_size = max(1e-5, min(step_size, L * 0.5))
    n_steps = max(1, int(round(L / step_size)))
    return state, step_size, n_steps


def bench_blackjax(model_name, n_chains, n_warmup, n_samples, seed, report_chains):
    import jax
    import jax.numpy as jnp
    import blackjax

    logdensity, dim = _bj_logdensity(model_name)
    key = jax.random.PRNGKey(seed)

    key, warmup_key, sample_key, init_key = jax.random.split(key, 4)

    t0_cold = time.perf_counter()

    # Manual warmup (BlackJAX 1.3 adjusted_mclmc_find_L_and_step_size is broken)
    warmed_state, step_size, n_steps = _blackjax_manual_warmup(
        logdensity, dim, n_warmup, warmup_key,
    )

    kernel = blackjax.adjusted_mclmc.build_kernel(
        logdensity_fn=logdensity,
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
    )

    # Multi-chain sampling
    chain_keys = jax.random.split(sample_key, n_chains)
    init_keys = jax.random.split(init_key, n_chains)
    init_positions = jax.vmap(lambda k: jax.random.normal(k, (dim,)) * 0.5)(init_keys)

    init_states = jax.vmap(
        lambda pos, k: blackjax.mcmc.adjusted_mclmc.init(
            position=pos, logdensity_fn=logdensity,
        )
    )(init_positions, init_keys)

    def one_step(state, rng_key):
        state, info = kernel(
            rng_key=rng_key, state=state,
            step_size=step_size, num_integration_steps=n_steps,
        )
        return state, state.position

    def run_chain(init_state, chain_key):
        keys = jax.random.split(chain_key, n_samples)
        final_state, positions = jax.lax.scan(one_step, init_state, keys)
        return positions

    # Cold run (JIT compile + execute)
    # block_until_ready + device_get = host-ready (symmetric with NS LAPS
    # which returns host-side numpy arrays via PyO3).
    all_positions = jax.vmap(run_chain)(init_states, chain_keys)
    all_positions.block_until_ready()
    _ = jax.device_get(all_positions)  # host transfer for fair timing
    wall_cold = time.perf_counter() - t0_cold

    # Warm run (JIT cached)
    key2 = jax.random.PRNGKey(seed + 1000)
    chain_keys2 = jax.random.split(key2, n_chains)

    t0_warm = time.perf_counter()
    all_positions2 = jax.vmap(run_chain)(init_states, chain_keys2)
    all_positions2.block_until_ready()
    draws_host = jax.device_get(all_positions2)  # host transfer for fair timing
    wall_warm = time.perf_counter() - t0_warm

    # ESS/R-hat should use the same report-chains budget as NS LAPS.
    n_report = min(n_chains, report_chains)
    draws_np = np.array(draws_host[:n_report])  # (n_report, n_samples, dim)
    min_ess, max_rhat = _ess_rhat(draws_np)
    n_grad_evals = int(n_steps * n_chains * n_samples)
    ess_per_grad = (min_ess / n_grad_evals) if (min_ess is not None and n_grad_evals) else None

    return {"engine": "BlackJAX_GPU", "model": model_name,
            "wall_s": wall_cold, "wall_warm": wall_warm,
            "min_ess": min_ess, "max_rhat": max_rhat,
            "n_chains": n_chains, "n_samples": n_samples,
            "n_report_chains": n_report,
            "tuned_step_size": step_size, "tuned_n_steps": n_steps,
            "n_grad_evals": n_grad_evals, "ess_per_grad": ess_per_grad}


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def fmt(val, digits=1):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        if abs(val) > 1e6:
            return f"{val/1e6:.1f}M"
        if abs(val) > 1e3:
            return f"{val/1e3:.{digits}f}K"
        return f"{val:.{digits}f}"
    return str(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chains-gpu", type=int, default=4096)
    ap.add_argument("--n-chains-cpu", type=int, default=4)
    ap.add_argument("--n-warmup", type=int, default=500)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", default="std_normal_10d,eight_schools,neal_funnel_10d,glm_logistic")
    ap.add_argument("--out", default="/tmp/a100_bench")
    ap.add_argument("--skip-cpu", action="store_true", help="Skip NS CPU MAMS")
    ap.add_argument("--skip-blackjax", action="store_true", help="Skip BlackJAX GPU")
    ap.add_argument("--skip-laps", action="store_true", help="Skip NS LAPS GPU")
    ap.add_argument("--report-chains", type=int, default=64,
                    help="Number of chains used for ESS/R-hat for both GPU engines.")
    ap.add_argument("--laps-glm-warmup-floor", type=int, default=200,
                    help="Minimum warmup used for LAPS glm_logistic (quality-speed guardrail).")
    ap.add_argument("--laps-glm-target-accept", type=float, default=0.95,
                    help="target_accept used for LAPS glm_logistic.")
    ap.add_argument("--laps-funnel-warmup-floor", type=int, default=500,
                    help="Minimum warmup used for LAPS neal_funnel_10d.")
    ap.add_argument("--laps-funnel-target-accept", type=float, default=0.9,
                    help="target_accept used for LAPS neal_funnel_10d.")
    args = ap.parse_args()

    models = args.models.split(",")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect and print environment snapshot BEFORE any benchmarks
    env = print_environment()

    results = []

    for model in models:
        print(f"\n{'='*70}")
        print(f"  Model: {model}")
        print(f"{'='*70}")

        # 1) NS LAPS GPU
        if not args.skip_laps:
            print(f"\n  [NS LAPS GPU] {args.n_chains_gpu} chains, {args.n_samples} samples...")
            try:
                r = bench_ns_laps(
                    model, args.n_chains_gpu, args.n_warmup, args.n_samples, args.seed,
                    args.report_chains,
                    args.laps_glm_warmup_floor,
                    args.laps_glm_target_accept,
                    args.laps_funnel_warmup_floor,
                    args.laps_funnel_target_accept,
                )
                ess_s = r['min_ess'] / r['wall_warm'] if r['min_ess'] and r['wall_warm'] else None
                print(
                    f"    wall={r['wall_s']:.2f}s  min_ESS={fmt(r['min_ess'])}  ESS/s={fmt(ess_s)}  "
                    f"ESS/grad={fmt(r.get('ess_per_grad'),4)}  "
                    f"R-hat={fmt(r['max_rhat'],4)}  quality={r.get('quality_status','n/a')}  "
                    f"report={r.get('n_report_chains','?')}"
                )
                results.append(r)
            except Exception as e:
                print(f"    FAILED: {e}")
                results.append({"engine": "NS_LAPS_GPU", "model": model, "error": str(e)})

        # 2) NS CPU MAMS
        if not args.skip_cpu:
            print(f"\n  [NS CPU MAMS] {args.n_chains_cpu} chains, {args.n_samples} samples...")
            try:
                r = bench_ns_cpu_mams(model, args.n_chains_cpu, args.n_warmup, args.n_samples, args.seed)
                ess_s = r['min_ess'] / r['wall_warm'] if r['min_ess'] and r['wall_warm'] else None
                print(
                    f"    wall={r['wall_s']:.2f}s  min_ESS={fmt(r['min_ess'])}  ESS/s={fmt(ess_s)}  "
                    f"ESS/grad={fmt(r.get('ess_per_grad'),4)}  R-hat={fmt(r['max_rhat'],4)}"
                )
                results.append(r)
            except Exception as e:
                print(f"    FAILED: {e}")
                results.append({"engine": "NS_CPU_MAMS", "model": model, "error": str(e)})

        # 3) BlackJAX GPU
        if not args.skip_blackjax:
            print(f"\n  [BlackJAX GPU] {args.n_chains_gpu} chains, {args.n_samples} samples...")
            try:
                r = bench_blackjax(
                    model, args.n_chains_gpu, args.n_warmup, args.n_samples, args.seed,
                    args.report_chains,
                )
                ess_s_cold = r['min_ess'] / r['wall_s'] if r['min_ess'] and r['wall_s'] else None
                ess_s_warm = r['min_ess'] / r['wall_warm'] if r['min_ess'] and r['wall_warm'] else None
                extra = ""
                if "tuned_step_size" in r:
                    extra = f"  eps={r['tuned_step_size']:.4f}  n_steps={r['tuned_n_steps']}"
                print(
                    f"    cold={r['wall_s']:.2f}s  warm={r['wall_warm']:.2f}s  min_ESS={fmt(r['min_ess'])}  "
                    f"ESS/s(warm)={fmt(ess_s_warm)}  ESS/grad={fmt(r.get('ess_per_grad'),4)}  "
                    f"R-hat={fmt(r['max_rhat'],4)}  report={r.get('n_report_chains','?')}{extra}"
                )
                results.append(r)
            except Exception as e:
                print(f"    FAILED: {e}")
                results.append({"engine": "BlackJAX_GPU", "model": model, "error": str(e)})

    # --- Summary Table ---
    print(f"\n\n{'='*128}")
    print(f"  A100-SXM4-80GB BENCHMARK — GPU: {args.n_chains_gpu} chains | CPU: {args.n_chains_cpu} chains | {args.n_samples} samples | warmup: {args.n_warmup}")
    print(f"{'='*128}")
    header = (
        f"{'Model':<18} | {'Engine':<16} | {'Cold(s)':<9} | {'Warm(s)':<9} | "
        f"{'min_ESS':<11} | {'ESS/s(warm)':<12} | {'ESS/grad':<10} | {'R-hat':<8} | "
        f"{'Q':<5} | {'Rpt':>4} | {'Chains':>6}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        if "error" in r:
            print(f"{r['model']:<18} | {r['engine']:<16} | {'FAIL':<9} | {'':<9} | {'':<11} | {'':<12} | {'':<8} | {'':<5} | {'':>4} | {'':<6}")
            continue

        wall_cold = r.get("wall_s", 0)
        wall_warm = r.get("wall_warm", wall_cold)
        min_ess = r.get("min_ess")
        max_rhat = r.get("max_rhat")
        quality = r.get("quality_status", "")
        n_report = r.get("n_report_chains", "")
        ess_per_s = min_ess / wall_warm if min_ess and wall_warm and wall_warm > 0 else None
        ess_per_grad = r.get("ess_per_grad")
        n_ch = r.get("n_chains", "?")

        print(
            f"{r['model']:<18} | {r['engine']:<16} | {wall_cold:<9.2f} | {wall_warm:<9.2f} | "
            f"{fmt(min_ess):<11} | {fmt(ess_per_s):<12} | {fmt(ess_per_grad,4):<10} | {fmt(max_rhat,4):<8} | "
            f"{str(quality):<5} | {str(n_report):>4} | {n_ch:>6}"
        )

    # Save JSON with environment snapshot
    out_path = out_dir / "a100_triple_bench.json"
    artifact = {
        "environment": env,
        "config": {
            "n_chains_gpu": args.n_chains_gpu,
            "n_chains_cpu": args.n_chains_cpu,
            "n_warmup": args.n_warmup,
            "n_samples": args.n_samples,
            "seed": args.seed,
            "report_chains": args.report_chains,
            "models": models,
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2, default=str)
    print(f"\nJSON: {out_path}")


if __name__ == "__main__":
    main()
