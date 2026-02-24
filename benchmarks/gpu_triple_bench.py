#!/usr/bin/env python3
"""GPU Triple Benchmark: NS LAPS GPU vs NS CPU MAMS vs BlackJAX GPU.

Runs models with cold (first run) and warm (JIT-cached) timings.
Reports: wall time, ESS, ESS/s for each configuration.

Three comparisons:
  1) NS LAPS GPU (4096 chains, 1 GPU)
  2) NS CPU MAMS (4 chains, 256 vCPU)
  3) BlackJAX adjusted_mclmc GPU (4096 chains, 1 GPU, cold + warm)

Usage:
    python gpu_triple_bench.py [--n-chains-gpu 4096] [--n-samples 2000] [--seed 42]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# Always benchmark the local extension/package from this repo, not any
# globally installed nextstat wheel in site-packages.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "bindings" / "ns-py" / "python"))

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


def _extract_stats(sample_stats, max_chains=None):
    """Extract gradient eval count from NextStat sample_stats."""
    if not isinstance(sample_stats, dict):
        return None
    n_leapfrog = sample_stats.get("n_leapfrog")
    if not isinstance(n_leapfrog, list) or not n_leapfrog:
        return None
    if isinstance(n_leapfrog[0], list):
        chains = n_leapfrog[:max_chains] if max_chains is not None else n_leapfrog
        return sum(int(v) for chain in chains for v in chain)
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
    phase_times = result.get("phase_times", {}) if isinstance(result, dict) else {}
    wall_sampling = phase_times.get("sampling_s") if isinstance(phase_times, dict) else None
    if not isinstance(wall_sampling, (int, float)) or not math.isfinite(float(wall_sampling)) or wall_sampling <= 0:
        wall_sampling = wall_s

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
    n_grad_evals = _extract_stats(result.get("sample_stats", {}), max_chains=n_report_chains)
    ess_per_grad = (min_ess / n_grad_evals) if (min_ess is not None and n_grad_evals) else None
    # n_grad_evals comes from sampling sample_stats, so use sampling wall time.
    grad_per_sec = (n_grad_evals / wall_sampling) if (n_grad_evals and wall_sampling > 0) else None

    return {"engine": "NS_LAPS_GPU", "model": model_name, "wall_s": wall_s,
            "wall_sampling": float(wall_sampling), "min_ess": min_ess, "max_rhat": max_rhat,
            "quality_status": quality_status, "quality_failures": quality_failures,
            "n_report_chains": n_report_chains,
            "phase_times": phase_times if isinstance(phase_times, dict) else {},
            "effective_n_warmup": n_warmup_eff,
            "effective_target_accept": target_accept,
            "effective_sync_interval": sync_interval,
            "effective_max_leapfrog": max_leapfrog,
            "n_chains": n_chains, "n_samples": n_samples,
            "n_grad_evals": n_grad_evals, "ess_per_grad": ess_per_grad,
            "grad_per_sec": grad_per_sec}


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

    n_grad_evals = _extract_stats(result.get("sample_stats", {}), max_chains=n_chains)
    ess_per_grad = (min_ess / n_grad_evals) if (min_ess is not None and n_grad_evals) else None
    # NOTE: CPU MAMS sample() does not return phase_times, so wall_sampling
    # falls back to wall_s (total incl. warmup).  This underestimates grad/s
    # but preserves the decomposition identity ESS/s = ESS/grad × grad/s
    # because ESS/s also uses the same denominator.
    phase_times = result.get("phase_times", {}) if isinstance(result, dict) else {}
    wall_sampling = phase_times.get("sampling_s") if isinstance(phase_times, dict) else None
    if not isinstance(wall_sampling, (int, float)) or not math.isfinite(float(wall_sampling)) or wall_sampling <= 0:
        wall_sampling = wall_s
    grad_per_sec = (
        n_grad_evals / float(wall_sampling)
    ) if (n_grad_evals and float(wall_sampling) > 0) else None

    return {"engine": "NS_CPU_MAMS", "model": model_name, "wall_s": wall_s,
            "wall_sampling": float(wall_sampling), "min_ess": min_ess, "max_rhat": max_rhat,
            "n_chains": n_chains, "n_samples": n_samples,
            "n_grad_evals": n_grad_evals, "ess_per_grad": ess_per_grad,
            "grad_per_sec": grad_per_sec}


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


def _blackjax_builtin_warmup(logdensity, dim, n_warmup, key, target_accept=0.9):
    """Use BlackJAX built-in adjusted_mclmc warmup (L + eps + diagonal mass)."""
    import jax
    import jax.numpy as jnp
    import blackjax

    init_key, warmup_key = jax.random.split(key)
    initial_position = 0.1 * jax.random.normal(init_key, (dim,))
    init_state = blackjax.adjusted_mclmc.init(
        position=initial_position,
        logdensity_fn=logdensity,
    )

    raw_kernel = blackjax.adjusted_mclmc.build_kernel(logdensity_fn=logdensity)

    def warmup_kernel(
        rng_key,
        state,
        step_size,
        num_integration_steps=10,
        L_proposal_factor=jnp.inf,
        **_kwargs,
    ):
        return raw_kernel(
            rng_key, state, step_size, num_integration_steps, L_proposal_factor
        )

    warmed_state, tuned_params, _ = blackjax.adjusted_mclmc_find_L_and_step_size(
        mclmc_kernel=warmup_kernel,
        num_steps=n_warmup,
        state=init_state,
        rng_key=warmup_key,
        target=target_accept,
        diagonal_preconditioning=True,
    )
    step_size = float(tuned_params.step_size)
    l_val = float(tuned_params.L)
    n_steps = max(1, int(round(l_val / step_size)))
    return warmed_state, tuned_params, step_size, n_steps


def bench_blackjax(model_name, n_chains, n_warmup, n_samples, seed, report_chains):
    import jax
    import blackjax

    # Fairness: NextStat CUDA path runs f64, so force BlackJAX to x64 too.
    # Without this, JAX defaults may silently run f32 and inflate throughput.
    jax.config.update("jax_enable_x64", True)

    logdensity, dim = _bj_logdensity(model_name)
    key = jax.random.PRNGKey(seed)

    key, warmup_key, sample_key, init_key = jax.random.split(key, 4)

    # Use BlackJAX native warmup path for fair, source-of-truth configuration.
    t0_warmup = time.perf_counter()
    warmed_state, tuned_params, step_size, n_steps = _blackjax_builtin_warmup(
        logdensity, dim, n_warmup, warmup_key, target_accept=0.9
    )
    wall_warmup = time.perf_counter() - t0_warmup

    sampling_alg = blackjax.adjusted_mclmc(
        logdensity_fn=logdensity,
        step_size=step_size,
        num_integration_steps=n_steps,
        inverse_mass_matrix=tuned_params.inverse_mass_matrix,
    )

    # Multi-chain sampling
    chain_keys = jax.random.split(sample_key, n_chains)
    init_keys = jax.random.split(init_key, n_chains)
    # Use warmup output as the center for multi-chain initialization so the
    # benchmarked sampling phase starts from adapted geometry, not cold random
    # points unrelated to DA-tuned state.
    init_positions = jax.vmap(
        lambda k: warmed_state.position + jax.random.normal(k, (dim,)) * 0.5
    )(init_keys)

    init_states = jax.vmap(
        lambda pos, k: blackjax.mcmc.adjusted_mclmc.init(
            position=pos, logdensity_fn=logdensity,
        )
    )(init_positions, init_keys)

    def one_step(state, rng_key):
        state, _ = sampling_alg.step(rng_key, state)
        return state, state.position

    def run_chain(init_state, chain_key):
        keys = jax.random.split(chain_key, n_samples)
        final_state, positions = jax.lax.scan(one_step, init_state, keys)
        return final_state, positions

    # Cold run (JIT compile + execute)
    # block_until_ready + device_get = host-ready (symmetric with NS LAPS
    # which returns host-side numpy arrays via PyO3).
    t0_cold_sampling = time.perf_counter()
    final_states_cold, all_positions = jax.vmap(run_chain)(init_states, chain_keys)
    all_positions.block_until_ready()
    _ = jax.device_get(all_positions)  # host transfer for fair timing
    wall_cold_sampling = time.perf_counter() - t0_cold_sampling
    wall_cold = wall_warmup + wall_cold_sampling

    # Warm run (JIT cached)
    key2 = jax.random.PRNGKey(seed + 1000)
    key2, _ = jax.random.split(key2)
    chain_keys2 = jax.random.split(key2, n_chains)

    t0_warm = time.perf_counter()
    _, all_positions2 = jax.vmap(run_chain)(final_states_cold, chain_keys2)
    all_positions2.block_until_ready()
    draws_host = jax.device_get(all_positions2)  # host transfer for fair timing
    wall_warm = time.perf_counter() - t0_warm

    # ESS/R-hat should use the same report-chains budget as NS LAPS.
    n_report = min(n_chains, report_chains)
    draws_np = np.array(draws_host[:n_report])  # (n_report, n_samples, dim)
    min_ess, max_rhat = _ess_rhat(draws_np)
    # ESS/R-hat are computed on `n_report` chains, so ESS/grad must use the
    # same chain budget to avoid denominator asymmetry vs NS LAPS reporting.
    n_grad_evals = int(n_steps * n_report * n_samples)
    ess_per_grad = (min_ess / n_grad_evals) if (min_ess is not None and n_grad_evals) else None
    # Use cold sampling time for grad/s — matches NS LAPS which reports
    # first-and-only sampling phase.  wall_warm (JIT-cached 2nd run) is kept
    # in the artifact for reference but NOT used for the primary metric so
    # the decomposition identity ESS/s = (ESS/grad) × (grad/s) holds with
    # a single, consistent denominator across all engines.
    grad_per_sec = (n_grad_evals / wall_cold_sampling) if (n_grad_evals and wall_cold_sampling > 0) else None

    return {"engine": "BlackJAX_GPU", "model": model_name,
            "wall_s": wall_cold, "wall_sampling": wall_cold_sampling,
            "min_ess": min_ess, "max_rhat": max_rhat,
            "n_chains": n_chains, "n_samples": n_samples,
            "n_report_chains": n_report,
            "tuned_step_size": step_size, "tuned_n_steps": n_steps,
            "tuned_L": float(tuned_params.L),
            "warmup_method": "blackjax.adjusted_mclmc_find_L_and_step_size",
            "warmup_s": wall_warmup,
            "cold_sampling_s": wall_cold_sampling,
            "warm_sampling_s": wall_warm,
            "n_grad_evals": n_grad_evals, "ess_per_grad": ess_per_grad,
            "grad_per_sec": grad_per_sec}


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
    ap.add_argument("--out", default="/tmp/gpu_bench")
    ap.add_argument("--skip-cpu", action="store_true", help="Skip NS CPU MAMS")
    ap.add_argument("--skip-blackjax", action="store_true", help="Skip BlackJAX GPU")
    ap.add_argument("--skip-laps", action="store_true", help="Skip NS LAPS GPU")
    ap.add_argument("--report-chains", type=int, default=256,
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
                ess_s = r['min_ess'] / r['wall_sampling'] if r['min_ess'] and r['wall_sampling'] else None
                print(
                    f"    wall_total={r['wall_s']:.2f}s  wall_sampling={r['wall_sampling']:.2f}s  "
                    f"min_ESS={fmt(r['min_ess'])}  ESS/s={fmt(ess_s)}  "
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
                ess_s_samp = r['min_ess'] / r['wall_sampling'] if r['min_ess'] and r['wall_sampling'] else None
                print(
                    f"    wall_total={r['wall_s']:.2f}s  wall_sampling={r['wall_sampling']:.2f}s  "
                    f"min_ESS={fmt(r['min_ess'])}  "
                    f"ESS/s(samp)={fmt(ess_s_samp)}  "
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
                ess_s_samp = r['min_ess'] / r['wall_sampling'] if r['min_ess'] and r['wall_sampling'] else None
                extra = ""
                if "tuned_step_size" in r:
                    cap_txt = f"/{r['max_n_steps_cap']}" if r.get("max_n_steps_cap") else ""
                    extra = (
                        f"  warmup={r.get('warmup_s', 0.0):.2f}s"
                        f"  cold_sampling={r.get('cold_sampling_s', 0.0):.2f}s"
                        f"  eps={r['tuned_step_size']:.4f}  "
                        f"n_steps={r['tuned_n_steps']}{cap_txt}"
                    )
                print(
                    f"    cold={r['wall_s']:.2f}s  samp={r['wall_sampling']:.2f}s  min_ESS={fmt(r['min_ess'])}  "
                    f"ESS/s(samp)={fmt(ess_s_samp)}  ESS/grad={fmt(r.get('ess_per_grad'),4)}  "
                    f"R-hat={fmt(r['max_rhat'],4)}  report={r.get('n_report_chains','?')}{extra}"
                )
                results.append(r)
            except Exception as e:
                print(f"    FAILED: {e}")
                results.append({"engine": "BlackJAX_GPU", "model": model, "error": str(e)})

    # --- Summary Table ---
    print(f"\n\n{'='*128}")
    print(f"  GPU BENCHMARK — GPU: {args.n_chains_gpu} chains | CPU: {args.n_chains_cpu} chains | {args.n_samples} samples | warmup: {args.n_warmup}")
    print(f"{'='*128}")
    header = (
        f"{'Model':<18} | {'Engine':<16} | {'Total(s)':<9} | {'Samp(s)':<9} | "
        f"{'min_ESS':<11} | {'ESS/s(samp)':<12} | {'ESS/grad':<10} | {'grad/s':<10} | {'R-hat':<8} | "
        f"{'Q':<5} | {'Rpt':>4} | {'Chains':>6}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        if "error" in r:
            print(f"{r['model']:<18} | {r['engine']:<16} | {'FAIL':<9} | {'':<9} | {'':<11} | {'':<12} | {'':<10} | {'':<10} | {'':<8} | {'':<5} | {'':>4} | {'':<6}")
            continue

        wall_total = r.get("wall_s", 0)
        wall_samp_raw = r.get("wall_sampling")
        wall_samp = (
            float(wall_samp_raw)
            if isinstance(wall_samp_raw, (int, float)) and math.isfinite(float(wall_samp_raw))
            else None
        )
        min_ess = r.get("min_ess")
        max_rhat = r.get("max_rhat")
        quality = r.get("quality_status", "")
        n_report = r.get("n_report_chains", "")
        ess_per_s = min_ess / wall_samp if min_ess and wall_samp and wall_samp > 0 else None
        ess_per_grad = r.get("ess_per_grad")
        grad_per_s = r.get("grad_per_sec")
        n_ch = r.get("n_chains", "?")
        wall_samp_txt = f"{wall_samp:<9.2f}" if wall_samp is not None else f"{'N/A':<9}"

        print(
            f"{r['model']:<18} | {r['engine']:<16} | {wall_total:<9.2f} | {wall_samp_txt} | "
            f"{fmt(min_ess):<11} | {fmt(ess_per_s):<12} | {fmt(ess_per_grad,4):<10} | {fmt(grad_per_s):<10} | {fmt(max_rhat,4):<8} | "
            f"{str(quality):<5} | {str(n_report):>4} | {n_ch:>6}"
        )

    # Save JSON with environment snapshot
    out_path = out_dir / "gpu_triple_bench.json"
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
