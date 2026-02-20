#!/usr/bin/env python3
"""SCM benchmark: Stepwise Covariate Modeling on synthetic 1-cpt oral PK data.

Generates synthetic data for 20 subjects with 10 timepoints each,
adds 3 covariates (WT significant on CL, AGE significant on V, SEX not
significant), and runs nextstat.scm() with timing.

Usage:
    python bench_scm.py [--n-subjects 20] [--seed 42] [--out-dir results/]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time


def generate_synthetic_data(
    n_subjects: int,
    seed: int,
) -> dict:
    """Generate synthetic 1-cpt oral PK data with known covariate effects.

    True effects:
      - WT on CL: power relationship, exponent = 0.75 (allometric)
      - AGE on V: exponential relationship, coefficient = -0.01 (V decreases with age)
      - SEX on Ka: NO true effect (noise covariate)
    """
    rng = random.Random(seed)

    # True population PK parameters
    cl_pop = 1.2    # L/h
    v_pop = 15.0    # L
    ka_pop = 2.0    # 1/h
    dose = 100.0    # mg
    bioav = 1.0
    sigma = 0.05    # additive residual error SD

    # IIV
    omega_cl = 0.15
    omega_v = 0.15
    omega_ka = 0.15

    # True covariate effects
    wt_exponent = 0.75   # power on CL
    age_coeff = -0.01    # exponential on V

    sample_times = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
    n_obs_per = len(sample_times)
    wt_center = 70.0
    age_center = 45.0

    # Generate per-subject covariates
    weights = [40.0 + 60.0 * rng.random() for _ in range(n_subjects)]
    ages = [20.0 + 50.0 * rng.random() for _ in range(n_subjects)]
    sexes = [float(rng.randint(0, 1)) for _ in range(n_subjects)]

    times = []
    y = []
    subject_idx = []
    wt_obs = []
    age_obs = []
    sex_obs = []

    for sid in range(n_subjects):
        wt = weights[sid]
        age = ages[sid]
        sex = sexes[sid]

        eta_cl = rng.gauss(0, omega_cl)
        eta_v = rng.gauss(0, omega_v)
        eta_ka = rng.gauss(0, omega_ka)

        # Individual PK with true covariate effects
        cl_i = cl_pop * (wt / wt_center) ** wt_exponent * math.exp(eta_cl)
        v_i = v_pop * math.exp(age_coeff * (age - age_center)) * math.exp(eta_v)
        ka_i = ka_pop * math.exp(eta_ka)  # no sex effect

        ke = cl_i / v_i
        if abs(ka_i - ke) < 1e-10:
            ka_i += 0.01

        for t in sample_times:
            c = (bioav * dose * ka_i) / (v_i * (ka_i - ke)) * (
                math.exp(-ke * t) - math.exp(-ka_i * t)
            )
            obs = max(0.0, c + rng.gauss(0, sigma))

            times.append(t)
            y.append(obs)
            subject_idx.append(sid)
            wt_obs.append(wt)
            age_obs.append(age)
            sex_obs.append(sex)

    return {
        "times": times,
        "y": y,
        "subject_idx": subject_idx,
        "n_subjects": n_subjects,
        "n_obs_per": n_obs_per,
        "dose": dose,
        "bioav": bioav,
        "sigma": sigma,
        "covariates": [wt_obs, age_obs, sex_obs],
        "covariate_names": ["WT", "AGE", "SEX"],
        "weights": weights,
        "ages": ages,
        "sexes": sexes,
        "true_effects": {
            "WT_on_CL": {"relationship": "power", "coefficient": wt_exponent},
            "AGE_on_V": {"relationship": "exponential", "coefficient": age_coeff},
            "SEX_on_Ka": {"relationship": "proportional", "coefficient": 0.0},
        },
    }


def run_benchmark(n_subjects: int, seed: int) -> dict:
    """Run SCM benchmark and return results."""
    import nextstat

    data = generate_synthetic_data(n_subjects, seed)

    print(f"Data: {data['n_subjects']} subjects, "
          f"{data['n_obs_per']} timepoints each, "
          f"{len(data['times'])} total observations")
    print(f"Covariates: {data['covariate_names']}")
    print(f"True effects: {data['true_effects']}")
    print()

    t0 = time.perf_counter()
    result = nextstat.scm(
        data["times"],
        data["y"],
        data["subject_idx"],
        data["n_subjects"],
        data["covariates"],
        data["covariate_names"],
        dose=data["dose"],
        bioavailability=data["bioav"],
        error_model="additive",
        sigma=data["sigma"],
        theta_init=[1.0, 10.0, 1.5],
        omega_init=[0.30, 0.30, 0.30],
        param_names=["CL", "V", "Ka"],
        relationships=["power", "exponential", "proportional"],
        forward_alpha=0.05,
        backward_alpha=0.01,
    )
    elapsed = time.perf_counter() - t0

    return {
        "result": result,
        "elapsed_s": elapsed,
        "data": data,
    }


def print_results_table(bench: dict) -> None:
    """Print benchmark results in tabular format."""
    result = bench["result"]
    elapsed = bench["elapsed_s"]
    data = bench["data"]

    print("=" * 72)
    print("SCM Benchmark Results")
    print("=" * 72)
    print(f"Subjects: {data['n_subjects']}")
    print(f"Observations: {len(data['times'])}")
    print(f"Covariates tested: {len(data['covariate_names'])}")
    print(f"Candidates (cov x param): {len(data['covariate_names']) * 3}")
    print(f"Wall time: {elapsed:.3f} s")
    print()

    print(f"Base OFV:  {result['base_ofv']:.4f}")
    print(f"Final OFV: {result['final_ofv']:.4f}")
    print(f"OFV drop:  {result['base_ofv'] - result['final_ofv']:.4f}")
    print(f"Forward steps:  {result['n_forward_steps']}")
    print(f"Backward steps: {result['n_backward_steps']}")
    print()

    # Selected covariates table
    print(f"{'Covariate':<20s} {'Relationship':<16s} {'delta_OFV':>10s} "
          f"{'p-value':>10s} {'Coefficient':>12s} {'True Coeff':>12s}")
    print("-" * 80)

    true_effects = data["true_effects"]
    for step in result["selected"]:
        true_coeff = ""
        for key, val in true_effects.items():
            if key == step["name"]:
                true_coeff = f"{val['coefficient']:.4f}"
                break
        print(
            f"{step['name']:<20s} {step['relationship']:<16s} "
            f"{step['delta_ofv']:>10.4f} "
            f"{step['p_value']:>10.6f} "
            f"{step['coefficient']:>12.4f} "
            f"{true_coeff:>12s}"
        )

    if not result["selected"]:
        print("  (no covariates selected)")

    print()

    # Forward trace
    print("Forward selection trace:")
    print(f"  {'Step':<5s} {'Candidate':<20s} {'delta_OFV':>10s} {'p-value':>10s} {'Status':>10s}")
    print("  " + "-" * 55)
    for i, step in enumerate(result["forward_trace"], 1):
        status = "ADDED" if step["included"] else "rejected"
        print(
            f"  {i:<5d} {step['name']:<20s} "
            f"{step['delta_ofv']:>10.4f} "
            f"{step['p_value']:>10.6f} "
            f"{status:>10s}"
        )

    if result["backward_trace"]:
        print()
        print("Backward elimination trace:")
        print(f"  {'Step':<5s} {'Candidate':<20s} {'delta_OFV':>10s} {'p-value':>10s} {'Status':>10s}")
        print("  " + "-" * 55)
        for i, step in enumerate(result["backward_trace"], 1):
            status = "KEPT" if step["included"] else "REMOVED"
            print(
                f"  {i:<5d} {step['name']:<20s} "
                f"{step['delta_ofv']:>10.4f} "
                f"{step['p_value']:>10.6f} "
                f"{status:>10s}"
            )

    print()
    print(f"Final theta: {[f'{v:.4f}' for v in result['theta']]}")
    print(f"Timing: {elapsed:.3f} s")


def collect_environment() -> dict:
    """Collect environment metadata."""
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def main():
    parser = argparse.ArgumentParser(description="SCM benchmark")
    parser.add_argument("--n-subjects", type=int, default=20,
                        help="Number of subjects (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory to save JSON artifact")
    args = parser.parse_args()

    bench = run_benchmark(args.n_subjects, args.seed)
    print_results_table(bench)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        artifact = {
            "benchmark": "scm",
            "n_subjects": args.n_subjects,
            "seed": args.seed,
            "elapsed_s": bench["elapsed_s"],
            "base_ofv": bench["result"]["base_ofv"],
            "final_ofv": bench["result"]["final_ofv"],
            "n_forward_steps": bench["result"]["n_forward_steps"],
            "n_backward_steps": bench["result"]["n_backward_steps"],
            "selected": bench["result"]["selected"],
            "forward_trace": bench["result"]["forward_trace"],
            "backward_trace": bench["result"]["backward_trace"],
            "theta": bench["result"]["theta"],
            "environment": collect_environment(),
        }
        path = os.path.join(args.out_dir, "bench_scm.json")
        with open(path, "w") as f:
            json.dump(artifact, f, indent=2)
        print(f"\nArtifact saved: {path}")


if __name__ == "__main__":
    main()
