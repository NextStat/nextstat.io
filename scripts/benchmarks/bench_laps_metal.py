#!/usr/bin/env python3
"""LAPS Metal benchmark: StdNormal, EightSchools, NealFunnel, GlmLogistic.

Measures wall time, phase breakdown, R-hat, ESS/s, divergence rate.
Mirrors CUDA benchmark table from laps-gpu-sampler.md.
"""
import json
import time
import sys
import platform
import subprocess
import numpy as np

import nextstat

SEED = 42


def collect_environment():
    chip = platform.processor() or "unknown"
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        pass
    gpu = "unknown"
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"], text=True
        )
        for line in out.splitlines():
            if "Chipset Model" in line:
                gpu = line.split(":")[-1].strip()
                break
    except Exception:
        pass
    return {
        "chip": chip,
        "gpu": gpu,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def run_case(label, model, model_data, n_chains, n_warmup, n_samples, dim,
             report_chains=32, batch_size=1000, fused_transitions=1000):
    """Run a single LAPS benchmark case and return metrics dict."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  chains={n_chains}, warmup={n_warmup}, samples={n_samples}, dim={dim}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    result = nextstat.sample(
        model,
        method="laps",
        model_data=model_data,
        n_chains=n_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        seed=SEED,
        report_chains=report_chains,
        batch_size=batch_size,
        fused_transitions=fused_transitions,
    )
    t1 = time.perf_counter()

    wall = result.get("wall_time_s", t1 - t0)
    phases = result.get("phase_times", {})
    diag = result.get("diagnostics", {})
    quality = diag.get("quality", {})

    r_hat_dict = diag.get("r_hat", {})
    r_hat_max = max(r_hat_dict.values()) if r_hat_dict else float("nan")

    div_rate = diag.get("divergence_rate", 0.0)

    # ESS: use min ESS bulk across params
    ess_bulk_dict = diag.get("ess_bulk", {})
    ess_values = list(ess_bulk_dict.values()) if ess_bulk_dict else []
    ess_min = min(ess_values) if ess_values else float("nan")
    ess_per_s = ess_min / wall if wall > 0 and ess_min == ess_min else float("nan")

    # Posterior: from posterior dict {param_name: [chain0_samples, chain1_samples, ...]}
    posterior = result.get("posterior", {})
    param_names = list(posterior.keys())
    if param_names:
        # Collect all samples across chains for each param
        all_samples = []
        for pname in param_names:
            chain_draws = posterior[pname]  # list of chains, each a list of draws
            flat = []
            for c in chain_draws:
                if isinstance(c, list):
                    flat.extend(c)
                else:
                    flat.append(c)
            all_samples.append(flat)
        all_samples = np.array(all_samples)  # [n_params, n_draws]
        means = np.mean(all_samples, axis=1)
        stds = np.std(all_samples, axis=1)
    else:
        means = stds = np.array([])

    print(f"\n  Wall time:       {wall:.3f} s")
    print(f"  Phase init:      {phases.get('init_s', '?'):.3f} s")
    print(f"  Phase warmup:    {phases.get('warmup_s', '?'):.3f} s")
    print(f"  Phase sampling:  {phases.get('sampling_s', '?'):.3f} s")
    print(f"  R-hat max:       {r_hat_max:.4f}")
    print(f"  ESS min:         {ess_min:.1f}")
    print(f"  ESS/s:           {ess_per_s:.1f}")
    print(f"  Divergence rate: {div_rate:.4f}")
    status = quality.get("status", "?") if quality else "?"
    print(f"  Quality status:  {status}")

    if len(means) > 0:
        show = min(5, dim)
        print(f"  Posterior mean[0..{show}]: {np.round(means[:show], 4)}")
        print(f"  Posterior std[0..{show}]:  {np.round(stds[:show], 4)}")

    return {
        "label": label,
        "n_chains": n_chains,
        "n_warmup": n_warmup,
        "n_samples": n_samples,
        "dim": dim,
        "wall_s": round(wall, 3),
        "init_s": round(phases.get("init_s", 0), 3),
        "warmup_s": round(phases.get("warmup_s", 0), 3),
        "sampling_s": round(phases.get("sampling_s", 0), 3),
        "r_hat_max": round(r_hat_max, 4),
        "ess_min": round(ess_min, 1) if ess_min == ess_min else None,
        "ess_per_s": round(ess_per_s, 1) if ess_per_s == ess_per_s else None,
        "divergence_rate": round(div_rate, 4),
        "quality_status": quality.get("status", "?") if quality else "?",
        "posterior_mean_0": round(float(means[0]), 4) if len(means) > 0 else None,
        "posterior_std_0": round(float(stds[0]), 4) if len(stds) > 0 else None,
    }


def main():
    env = collect_environment()
    print(f"LAPS Metal Benchmark")
    print(f"Chip: {env['chip']}, GPU: {env['gpu']}")
    print(f"Python: {env['python']}")

    results = []

    # --- Case 1: StdNormal 10d ---
    results.append(run_case(
        label="std_normal_10d",
        model="std_normal",
        model_data={"dim": 10},
        n_chains=256,
        n_warmup=100,
        n_samples=100,
        dim=10,
    ))

    # --- Case 2: StdNormal 10d, more chains ---
    results.append(run_case(
        label="std_normal_10d_4096ch",
        model="std_normal",
        model_data={"dim": 10},
        n_chains=4096,
        n_warmup=200,
        n_samples=500,
        dim=10,
    ))

    # --- Case 3: EightSchools ---
    results.append(run_case(
        label="eight_schools",
        model="eight_schools",
        model_data={
            "y": [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
            "sigma": [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
        },
        n_chains=4096,
        n_warmup=500,
        n_samples=2000,
        dim=10,
    ))

    # --- Case 4: NealFunnel 10d ---
    results.append(run_case(
        label="neal_funnel_10d",
        model="neal_funnel",
        model_data={"dim": 10},
        n_chains=4096,
        n_warmup=500,
        n_samples=2000,
        dim=10,
    ))

    # --- Case 5: NealFunnel 10d Riemannian ---
    results.append(run_case(
        label="neal_funnel_riemannian_10d",
        model="neal_funnel_riemannian",
        model_data={"dim": 10},
        n_chains=4096,
        n_warmup=500,
        n_samples=2000,
        dim=10,
    ))

    # --- Case 6: GLM logistic n=200 p=6 ---
    rng = np.random.default_rng(123)
    n_obs, p = 200, 6
    X = rng.standard_normal((n_obs, p))
    beta_true = rng.standard_normal(p) * 0.5
    prob = 1.0 / (1.0 + np.exp(-X @ beta_true))
    y = rng.binomial(1, prob).astype(float)
    results.append(run_case(
        label="glm_logistic_n200_p6",
        model="glm_logistic",
        model_data={"x": X.ravel().tolist(), "y": y.tolist(), "n": n_obs, "p": p},
        n_chains=4096,
        n_warmup=500,
        n_samples=2000,
        dim=p,
    ))

    # --- Case 6: GLM logistic n=1000 p=20 ---
    n_obs, p = 1000, 20
    X = rng.standard_normal((n_obs, p))
    beta_true = rng.standard_normal(p) * 0.3
    prob = 1.0 / (1.0 + np.exp(-X @ beta_true))
    y = rng.binomial(1, prob).astype(float)
    results.append(run_case(
        label="glm_logistic_n1000_p20",
        model="glm_logistic",
        model_data={"x": X.ravel().tolist(), "y": y.tolist(), "n": n_obs, "p": p},
        n_chains=4096,
        n_warmup=500,
        n_samples=2000,
        dim=p,
    ))

    # --- Case 7: GLM logistic n=5000 p=20 (heavy) ---
    n_obs, p = 5000, 20
    X = rng.standard_normal((n_obs, p))
    beta_true = rng.standard_normal(p) * 0.3
    prob = 1.0 / (1.0 + np.exp(-X @ beta_true))
    y = rng.binomial(1, prob).astype(float)
    results.append(run_case(
        label="glm_logistic_n5000_p20",
        model="glm_logistic",
        model_data={"x": X.ravel().tolist(), "y": y.tolist(), "n": n_obs, "p": p},
        n_chains=4096,
        n_warmup=500,
        n_samples=2000,
        dim=p,
    ))

    # --- Summary table ---
    print("\n\n" + "="*90)
    print("LAPS Metal Benchmark Summary")
    print(f"Chip: {env['chip']}, GPU: {env['gpu']}")
    print("="*90)
    hdr = f"{'Model':<30} {'chains':>6} {'w+s':>8} {'Wall(s)':>8} {'R-hat':>7} {'ESS/s':>8} {'Div%':>6} {'Status':>8}"
    print(hdr)
    print("-"*90)
    for r in results:
        ws = f"{r['n_warmup']}+{r['n_samples']}"
        ess = f"{r['ess_per_s']:.0f}" if r['ess_per_s'] is not None else "N/A"
        rh = f"{r['r_hat_max']:.3f}" if r['r_hat_max'] == r['r_hat_max'] else "N/A"
        div = f"{r['divergence_rate']*100:.1f}" if r['divergence_rate'] is not None else "N/A"
        print(f"{r['label']:<30} {r['n_chains']:>6} {ws:>8} {r['wall_s']:>8.2f} {rh:>7} {ess:>8} {div:>6} {r['quality_status']:>8}")
    print("="*90)

    # Save JSON artifact
    artifact = {"environment": env, "results": results}
    out_path = "bench_results/laps_metal_benchmark.json"
    import os
    os.makedirs("bench_results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
