#!/usr/bin/env python3
"""
NextStat vs NONMEM Parity Benchmark Suite

Runs Theophylline, Warfarin, and Phase I IV bolus fits with both SAEM and FOCE,
times each fit, prints comparison tables, and saves results as JSON artifact.

Usage:
    python scripts/benchmarks/bench_nonmem_parity.py [--seed 42] [--json out.json]

Requires: nextstat Python package (pip install nextstat or maturin develop)
"""

import argparse
import inspect
import json
import sys
import time
from typing import Any

import numpy as np

try:
    import nextstat
except ImportError:
    print("ERROR: nextstat package not installed. Run: maturin develop -p ns-py")
    sys.exit(1)


def _nlme_dose_kwarg_name() -> str:
    """Return supported NLME dose kwarg name (`doses` preferred, `dose` legacy)."""
    foce_params = inspect.signature(nextstat.nlme_foce).parameters
    saem_params = inspect.signature(nextstat.nlme_saem).parameters
    if "doses" in foce_params and "doses" in saem_params:
        return "doses"
    if "dose" in foce_params and "dose" in saem_params:
        return "dose"
    raise RuntimeError(
        "Unsupported nextstat NLME API: expected both nlme_foce/nlme_saem to expose "
        "either `doses` or `dose` keyword"
    )


def _assert_nonmem_harness_contract() -> None:
    """Ensure runtime nextstat supports APIs required by this benchmark."""
    saem_params = inspect.signature(nextstat.nlme_saem).parameters
    if "model" not in saem_params:
        raise RuntimeError(
            "nextstat.nlme_saem() missing `model` kwarg; this harness needs multi-model SAEM "
            "(1cpt_oral + 2cpt_iv). Rebuild/install current ns-py wheel."
        )


def _nlme_dose_kwargs(dose: float, dose_kw: str) -> dict[str, Any]:
    if dose_kw == "doses":
        return {"doses": [float(dose)]}
    if dose_kw == "dose":
        return {"dose": float(dose)}
    raise ValueError(f"Unsupported NLME dose kwarg: {dose_kw}")


# ============================================================================
# Theophylline dataset (Boeckmann, Sheiner, Beal 1994)
# ============================================================================

THEOPHYLLINE_DATA = {
    # (ID, TIME, DV, AMT, EVID) — 12 subjects, single oral dose
    "subjects": [
        {"id": 1, "dose": 4.02, "obs": [(0.25, 2.84), (0.57, 6.57), (1.12, 10.50), (2.02, 9.66), (3.82, 8.58), (5.10, 8.36), (7.03, 7.47), (9.05, 6.89), (12.12, 5.94), (24.37, 3.28)]},
        {"id": 2, "dose": 4.40, "obs": [(0.27, 1.72), (0.52, 7.91), (1.00, 8.31), (1.92, 8.33), (3.50, 6.85), (5.02, 6.08), (7.03, 5.40), (9.00, 4.55), (12.00, 3.01), (24.30, 0.90)]},
        {"id": 3, "dose": 4.53, "obs": [(0.27, 4.40), (0.58, 6.90), (1.02, 8.20), (2.02, 7.80), (3.62, 7.50), (5.08, 6.20), (7.07, 5.30), (9.00, 4.90), (12.15, 3.70), (24.17, 1.05)]},
        {"id": 4, "dose": 4.40, "obs": [(0.35, 1.89), (0.60, 4.60), (1.07, 8.60), (2.13, 8.38), (3.50, 7.54), (5.02, 6.88), (7.02, 5.78), (9.02, 5.33), (11.98, 4.19), (24.65, 1.15)]},
        {"id": 5, "dose": 5.86, "obs": [(0.30, 2.02), (0.52, 5.63), (1.00, 11.40), (2.02, 9.33), (3.50, 8.74), (5.02, 7.56), (7.02, 7.09), (9.00, 5.90), (12.00, 4.37), (24.35, 1.57)]},
        {"id": 6, "dose": 4.00, "obs": [(0.27, 1.29), (0.58, 3.08), (1.15, 6.44), (2.03, 6.32), (3.57, 5.53), (5.00, 4.94), (7.00, 4.02), (9.22, 3.46), (12.10, 2.78), (23.85, 0.92)]},
        {"id": 7, "dose": 4.95, "obs": [(0.25, 3.05), (0.50, 3.05), (1.02, 7.31), (2.02, 7.56), (3.53, 6.59), (5.05, 5.88), (7.15, 4.73), (9.22, 4.57), (12.10, 3.00), (24.12, 1.25)]},
        {"id": 8, "dose": 4.53, "obs": [(0.25, 7.37), (0.52, 9.03), (0.98, 7.14), (2.02, 6.33), (3.53, 5.66), (5.05, 5.67), (7.15, 4.24), (9.22, 4.11), (12.10, 3.16), (24.12, 1.12)]},
        {"id": 9, "dose": 3.10, "obs": [(0.25, 0.00), (0.50, 2.89), (1.00, 4.25), (2.00, 4.00), (3.52, 4.17), (5.07, 2.80), (7.07, 2.60), (9.03, 2.44), (12.05, 1.36), (24.15, 0.00)]},
        {"id": 10, "dose": 5.50, "obs": [(0.37, 3.52), (0.77, 7.48), (1.02, 9.40), (2.05, 8.80), (3.55, 7.63), (5.05, 6.90), (7.08, 6.38), (9.38, 5.21), (12.10, 4.42), (24.22, 1.63)]},
        {"id": 11, "dose": 4.92, "obs": [(0.25, 1.49), (0.50, 4.73), (0.98, 7.56), (1.98, 6.60), (3.60, 5.11), (5.02, 4.57), (7.17, 3.18), (8.80, 2.83), (11.60, 2.26), (24.43, 0.86)]},
        {"id": 12, "dose": 5.30, "obs": [(0.25, 1.25), (0.50, 3.96), (1.00, 7.82), (2.00, 9.72), (3.52, 9.75), (5.07, 8.57), (7.08, 6.59), (9.38, 6.11), (12.10, 4.57), (24.22, 1.17)]},
    ]
}


def prepare_theophylline():
    """Flatten Theophylline data into arrays for NextStat API."""
    times = []
    y = []
    subject_idx = []
    doses = []

    for sid, subj in enumerate(THEOPHYLLINE_DATA["subjects"]):
        doses.append(subj["dose"])
        for t, dv in subj["obs"]:
            times.append(t)
            y.append(dv)
            subject_idx.append(sid)

    mean_dose = sum(doses) / len(doses)
    n_subjects = len(THEOPHYLLINE_DATA["subjects"])
    return times, y, subject_idx, n_subjects, mean_dose


def generate_warfarin_synthetic(n_subjects: int = 32, seed: int = 42):
    """Generate synthetic Warfarin-like dataset."""
    rng = np.random.default_rng(seed)

    cl_pop, v_pop, ka_pop = 0.134, 8.0, 1.0
    omega_cl, omega_v, omega_ka = 0.20, 0.15, 0.25
    sigma = 0.3
    dose = 100.0
    sampling_times = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0, 36.0, 48.0, 72.0]

    times, y, subject_idx = [], [], []

    for sid in range(n_subjects):
        eta_cl = rng.normal(0, omega_cl)
        eta_v = rng.normal(0, omega_v)
        eta_ka = rng.normal(0, omega_ka)
        cl_i = cl_pop * np.exp(eta_cl)
        v_i = v_pop * np.exp(eta_v)
        ka_i = ka_pop * np.exp(eta_ka)

        for t in sampling_times:
            ke = cl_i / v_i
            c = (dose * ka_i / (v_i * (ka_i - ke))) * (np.exp(-ke * t) - np.exp(-ka_i * t))
            obs = max(0.0, c + rng.normal(0, sigma))
            times.append(t)
            y.append(obs)
            subject_idx.append(sid)

    return times, y, subject_idx, n_subjects, dose, sigma


def generate_phase1_iv_2cpt(n_subjects: int = 24, seed: int = 42):
    """Generate synthetic Phase I IV bolus dataset (2-compartment)."""
    rng = np.random.default_rng(seed)

    cl_pop, v1_pop, q_pop, v2_pop = 5.0, 10.0, 15.0, 20.0
    omega_sds = [0.20, 0.15, 0.20, 0.15]
    sigma = 0.1
    dose = 100.0
    sampling_times = [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0, 48.0]

    times, y, subject_idx = [], [], []

    for sid in range(n_subjects):
        etas = [rng.normal(0, sd) for sd in omega_sds]
        cl_i = cl_pop * np.exp(etas[0])
        v1_i = v1_pop * np.exp(etas[1])
        q_i = q_pop * np.exp(etas[2])
        v2_i = v2_pop * np.exp(etas[3])

        # 2-compartment IV bolus analytical solution
        k10 = cl_i / v1_i
        k12 = q_i / v1_i
        k21 = q_i / v2_i
        beta_sum = k10 + k12 + k21
        beta_prod = k10 * k21
        disc = np.sqrt(max(0, beta_sum ** 2 - 4 * beta_prod))
        alpha = (beta_sum + disc) / 2
        beta = (beta_sum - disc) / 2

        for t in sampling_times:
            a_coeff = dose * (alpha - k21) / (v1_i * (alpha - beta))
            b_coeff = dose * (k21 - beta) / (v1_i * (alpha - beta))
            c = a_coeff * np.exp(-alpha * t) + b_coeff * np.exp(-beta * t)
            obs = max(0.01, c + rng.normal(0, sigma))
            times.append(t)
            y.append(obs)
            subject_idx.append(sid)

    return times, y, subject_idx, n_subjects, dose, sigma


def run_saem_1cpt(times, y, subject_idx, n_subjects, dose, sigma,
                  theta_init, omega_init, error_model="proportional",
                  n_burn=400, n_iter=300, seed=42, dose_kw="doses"):
    """Run SAEM 1-cpt oral fit and return result dict + elapsed time."""
    t0 = time.perf_counter()
    result = nextstat.nlme_saem(
        times=times,
        y=y,
        subject_idx=subject_idx,
        n_subjects=n_subjects,
        model="1cpt_oral",
        bioavailability=1.0,
        error_model=error_model,
        sigma=sigma,
        theta_init=theta_init,
        omega_init=omega_init,
        n_burn=n_burn,
        n_iter=n_iter,
        seed=seed,
        **_nlme_dose_kwargs(dose, dose_kw),
    )
    elapsed = time.perf_counter() - t0
    return result, elapsed


def run_foce_1cpt(times, y, subject_idx, n_subjects, dose, sigma,
                  theta_init, omega_init, error_model="proportional", dose_kw="doses"):
    """Run FOCE 1-cpt oral fit and return result dict + elapsed time."""
    t0 = time.perf_counter()
    result = nextstat.nlme_foce(
        times=times,
        y=y,
        subject_idx=subject_idx,
        n_subjects=n_subjects,
        bioavailability=1.0,
        error_model=error_model,
        sigma=sigma,
        theta_init=theta_init,
        omega_init=omega_init,
        max_outer_iter=200,
        max_inner_iter=30,
        tol=1e-5,
        interaction=True,
        **_nlme_dose_kwargs(dose, dose_kw),
    )
    elapsed = time.perf_counter() - t0
    return result, elapsed


def run_saem_2cpt_iv(times, y, subject_idx, n_subjects, dose, sigma,
                     theta_init, omega_init, n_burn=400, n_iter=300, seed=42, dose_kw="doses"):
    """Run SAEM 2-cpt IV fit and return result dict + elapsed time."""
    t0 = time.perf_counter()
    result = nextstat.nlme_saem(
        times=times,
        y=y,
        subject_idx=subject_idx,
        n_subjects=n_subjects,
        model="2cpt_iv",
        bioavailability=1.0,
        error_model="additive",
        sigma=sigma,
        theta_init=theta_init,
        omega_init=omega_init,
        n_burn=n_burn,
        n_iter=n_iter,
        seed=seed,
        **_nlme_dose_kwargs(dose, dose_kw),
    )
    elapsed = time.perf_counter() - t0
    return result, elapsed


def print_table(title: str, rows: list[dict[str, Any]]):
    """Print a formatted comparison table."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  {'Parameter':<12} | {'Reference':>10} | {'Estimate':>10} | {'Rel Diff':>10}")
    print(f"  {'-' * 12}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}")
    for row in rows:
        ref = row["reference"]
        est = row["estimate"]
        rel = abs(est - ref) / abs(ref) * 100 if ref != 0 else 0.0
        print(f"  {row['name']:<12} | {ref:>10.4f} | {est:>10.4f} | {rel:>9.1f}%")


def main():
    dose_kw = _nlme_dose_kwarg_name()
    _assert_nonmem_harness_contract()
    parser = argparse.ArgumentParser(description="NextStat vs NONMEM Parity Benchmark")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
    parser.add_argument("--json", type=str, default=None, help="Output JSON artifact path")
    args = parser.parse_args()

    results: dict[str, Any] = {
        "benchmark": "nonmem_parity",
        "seed": args.seed,
        "nlme_dose_kwarg": dose_kw,
        "fits": [],
    }

    print()
    print("=" * 70)
    print("  NextStat vs NONMEM: Parameter Estimation Parity Benchmark")
    print("=" * 70)

    # ---- Theophylline SAEM ----
    times, y, subject_idx, n_subjects, mean_dose = prepare_theophylline()
    theo_saem, t_theo_saem = run_saem_1cpt(
        times, y, subject_idx, n_subjects, mean_dose, 0.7,
        theta_init=[0.04, 0.45, 1.5],
        omega_init=[0.30, 0.25, 0.50],
        error_model="proportional",
        seed=args.seed,
        dose_kw=dose_kw,
    )
    print_table("Theophylline SAEM", [
        {"name": "CL/F", "reference": 0.040, "estimate": theo_saem["theta"][0]},
        {"name": "V/F", "reference": 0.50, "estimate": theo_saem["theta"][1]},
        {"name": "Ka", "reference": 1.50, "estimate": theo_saem["theta"][2]},
    ])
    print(f"  OFV: {theo_saem['ofv']:.2f}  |  Converged: {theo_saem['converged']}  |  Time: {t_theo_saem * 1000:.0f} ms")
    results["fits"].append({
        "dataset": "Theophylline", "method": "SAEM", "model": "1cpt_oral",
        "theta": list(theo_saem["theta"]), "omega": list(theo_saem["omega"]),
        "ofv": theo_saem["ofv"], "converged": theo_saem["converged"],
        "time_ms": t_theo_saem * 1000,
    })

    # ---- Theophylline FOCE ----
    theo_foce, t_theo_foce = run_foce_1cpt(
        times, y, subject_idx, n_subjects, mean_dose, 0.7,
        theta_init=[0.04, 0.45, 1.5],
        omega_init=[0.30, 0.25, 0.50],
        error_model="proportional",
        dose_kw=dose_kw,
    )
    print_table("Theophylline FOCEI", [
        {"name": "CL/F", "reference": 0.040, "estimate": theo_foce["theta"][0]},
        {"name": "V/F", "reference": 0.50, "estimate": theo_foce["theta"][1]},
        {"name": "Ka", "reference": 1.50, "estimate": theo_foce["theta"][2]},
    ])
    print(f"  OFV: {theo_foce['ofv']:.2f}  |  Converged: {theo_foce['converged']}  |  Time: {t_theo_foce * 1000:.0f} ms")
    results["fits"].append({
        "dataset": "Theophylline", "method": "FOCEI", "model": "1cpt_oral",
        "theta": list(theo_foce["theta"]), "omega": list(theo_foce["omega"]),
        "ofv": theo_foce["ofv"], "converged": theo_foce["converged"],
        "time_ms": t_theo_foce * 1000,
    })

    # ---- Warfarin SAEM ----
    wt, wy, ws, wn, wdose, wsigma = generate_warfarin_synthetic(32, args.seed)
    warf_saem, t_warf_saem = run_saem_1cpt(
        wt, wy, ws, wn, wdose, wsigma,
        theta_init=[0.10, 5.0, 0.5],
        omega_init=[0.30, 0.30, 0.30],
        error_model="additive",
        seed=args.seed,
        dose_kw=dose_kw,
    )
    print_table("Warfarin SAEM", [
        {"name": "CL (L/h)", "reference": 0.134, "estimate": warf_saem["theta"][0]},
        {"name": "V (L)", "reference": 8.0, "estimate": warf_saem["theta"][1]},
        {"name": "Ka (1/h)", "reference": 1.0, "estimate": warf_saem["theta"][2]},
    ])
    print(f"  OFV: {warf_saem['ofv']:.2f}  |  Converged: {warf_saem['converged']}  |  Time: {t_warf_saem * 1000:.0f} ms")
    results["fits"].append({
        "dataset": "Warfarin", "method": "SAEM", "model": "1cpt_oral",
        "theta": list(warf_saem["theta"]), "omega": list(warf_saem["omega"]),
        "ofv": warf_saem["ofv"], "converged": warf_saem["converged"],
        "time_ms": t_warf_saem * 1000,
    })

    # ---- Phase I IV 2-cpt SAEM ----
    it, iy, si, ni, idose, isigma = generate_phase1_iv_2cpt(24, args.seed)
    iv_saem, t_iv_saem = run_saem_2cpt_iv(
        it, iy, si, ni, idose, isigma,
        theta_init=[4.0, 8.0, 12.0, 15.0],
        omega_init=[0.30, 0.30, 0.30, 0.30],
        seed=args.seed,
        dose_kw=dose_kw,
    )
    print_table("Phase I IV 2-cpt SAEM", [
        {"name": "CL (L/h)", "reference": 5.0, "estimate": iv_saem["theta"][0]},
        {"name": "V1 (L)", "reference": 10.0, "estimate": iv_saem["theta"][1]},
        {"name": "Q (L/h)", "reference": 15.0, "estimate": iv_saem["theta"][2]},
        {"name": "V2 (L)", "reference": 20.0, "estimate": iv_saem["theta"][3]},
    ])
    print(f"  OFV: {iv_saem['ofv']:.2f}  |  Converged: {iv_saem['converged']}  |  Time: {t_iv_saem * 1000:.0f} ms")
    results["fits"].append({
        "dataset": "Phase I IV", "method": "SAEM", "model": "2cpt_iv",
        "theta": list(iv_saem["theta"]), "omega": list(iv_saem["omega"]),
        "ofv": iv_saem["ofv"], "converged": iv_saem["converged"],
        "time_ms": t_iv_saem * 1000,
    })

    # ---- Summary ----
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Dataset':<18} | {'Model':<12} | {'Method':<8} | {'Time (ms)':>10} | {'Status'}")
    print(f"  {'-' * 18}-+-{'-' * 12}-+-{'-' * 8}-+-{'-' * 10}-+-{'-' * 8}")
    for fit in results["fits"]:
        status = "PASS" if fit["converged"] else "WARN"
        print(f"  {fit['dataset']:<18} | {fit['model']:<12} | {fit['method']:<8} | {fit['time_ms']:>9.0f} | {status}")
    print()

    total_ms = sum(f["time_ms"] for f in results["fits"])
    print(f"  Total runtime: {total_ms:.0f} ms ({total_ms / 1000:.1f} s)")
    print()

    # Save JSON artifact
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to: {args.json}")
    else:
        default_path = "artifacts/nonmem_parity_benchmark.json"
        import os
        os.makedirs("artifacts", exist_ok=True)
        with open(default_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to: {default_path}")


if __name__ == "__main__":
    main()
