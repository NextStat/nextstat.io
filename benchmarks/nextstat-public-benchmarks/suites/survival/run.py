#!/usr/bin/env python3
"""Survival analysis benchmark: Cox PH, Kaplan-Meier, Weibull AFT.

This is intentionally small and self-contained so outsiders can rerun it.

Cases:
- cox_ph_1k_5p: Cox PH, 1000 subjects, 5 covariates
- cox_ph_10k_10p: Cox PH, 10000 subjects, 10 covariates
- kaplan_meier_1k: Kaplan-Meier curve comparison (1000 subjects, 2 groups)
- weibull_aft_1k: Parametric Weibull AFT (1000 subjects)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

import nextstat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.bench_env import collect_environment


def sha256_json_obj(obj: dict) -> str:
    b = (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _pctl(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * float(p)
    i = int(k)
    j = min(i + 1, len(xs) - 1)
    a = xs[i]
    b = xs[j]
    t = k - i
    return a * (1.0 - t) + b * t


def _summary(values: list[float]) -> dict[str, float]:
    return {"min": min(values) if values else 0.0, "median": _pctl(values, 0.5), "p95": _pctl(values, 0.95)}


def _maybe_import(name: str) -> tuple[bool, Optional[str]]:
    try:
        __import__(name)
        mod = sys.modules.get(name)
        return True, str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return False, None


def _max_abs_rel_diff(a: list[float], b: list[float]) -> tuple[Optional[float], Optional[float]]:
    if not a or not b or len(a) != len(b):
        return None, None
    abs_d = 0.0
    rel_d = 0.0
    for x, y in zip(a, b):
        d = abs(float(x) - float(y))
        abs_d = max(abs_d, d)
        denom = max(abs(float(x)), abs(float(y)), 1.0)
        rel_d = max(rel_d, d / denom)
    return float(abs_d), float(rel_d)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def gen_cox_data(n: int, p: int, seed: int) -> tuple[list[float], list[bool], list[list[float]], list[float]]:
    rng = np.random.default_rng(int(seed))
    X = rng.normal(size=(int(n), int(p)))
    beta_true = np.linspace(0.5, -0.3, int(p))
    hazard = np.exp(X @ beta_true)
    # Exponential baseline: T ~ Exp(hazard)
    times = rng.exponential(1.0 / hazard)
    # Random censoring
    censor_times = rng.exponential(3.0, int(n))
    events = (times <= censor_times).tolist()
    times = np.minimum(times, censor_times).tolist()
    return times, events, X.tolist(), beta_true.tolist()


def gen_km_data(n: int, seed: int) -> tuple[list[float], list[bool], list[int]]:
    rng = np.random.default_rng(int(seed))
    groups = rng.choice([0, 1], size=int(n)).tolist()
    # Group 0: baseline, Group 1: treatment (better survival)
    times_raw: list[float] = []
    for g in groups:
        rate = 1.0 if g == 0 else 0.5
        times_raw.append(float(rng.exponential(1.0 / rate)))
    censor = rng.exponential(3.0, int(n))
    events = [t <= c for t, c in zip(times_raw, censor)]
    times = [min(t, c) for t, c in zip(times_raw, censor)]
    return times, events, groups


def gen_weibull_data(n: int, seed: int, shape: float = 1.5, scale: float = 2.0) -> tuple[list[float], list[bool]]:
    rng = np.random.default_rng(int(seed))
    times_raw = rng.weibull(float(shape), int(n)) * float(scale)
    censor = rng.exponential(5.0, int(n))
    events = (times_raw <= censor).tolist()
    times = np.minimum(times_raw, censor).tolist()
    return times, events


# ---------------------------------------------------------------------------
# Truth-recovery DGP generators
# ---------------------------------------------------------------------------

def gen_truth_recovery_weibull(n: int, seed: int, shape_true: float = 1.5, scale_true: float = 2.0,
                                censoring_rate: float = 0.3) -> tuple[list[float], list[bool], dict]:
    """Weibull DGP with controlled censoring fraction via bisection on censoring scale."""
    rng = np.random.default_rng(seed)
    times_raw = rng.weibull(shape_true, n) * scale_true

    # Bisection to find censoring scale that gives ~censoring_rate fraction censored
    lo, hi = 0.01, 1000.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        censor_times = rng.exponential(mid, n)
        rng2 = np.random.default_rng(seed)  # reset to get same times_raw
        _ = rng2.weibull(shape_true, n) * scale_true
        frac_censored = np.mean(times_raw > censor_times)
        if frac_censored < censoring_rate:
            hi = mid
        else:
            lo = mid

    # Final draw with found scale
    rng_final = np.random.default_rng(seed + 999999)
    censor_times = rng_final.exponential((lo + hi) / 2.0, n)
    events = (times_raw <= censor_times).tolist()
    obs_times = np.minimum(times_raw, censor_times).tolist()
    actual_cens = 1.0 - np.mean(events)
    dgp = {"distribution": "weibull", "shape": shape_true, "scale": scale_true,
           "censoring_rate_target": censoring_rate, "censoring_rate_actual": float(actual_cens)}
    return obs_times, events, dgp


def gen_truth_recovery_exponential(n: int, seed: int, rate_true: float = 0.5,
                                    censoring_rate: float = 0.3) -> tuple[list[float], list[bool], dict]:
    """Exponential DGP with controlled censoring."""
    rng = np.random.default_rng(seed)
    times_raw = rng.exponential(1.0 / rate_true, n)

    # Bisection for censoring scale
    lo, hi = 0.01, 1000.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        censor_times = rng.exponential(mid, n)
        rng2 = np.random.default_rng(seed)
        _ = rng2.exponential(1.0 / rate_true, n)
        frac = np.mean(times_raw > censor_times)
        if frac < censoring_rate:
            hi = mid
        else:
            lo = mid

    rng_final = np.random.default_rng(seed + 999999)
    censor_times = rng_final.exponential((lo + hi) / 2.0, n)
    events = (times_raw <= censor_times).tolist()
    obs_times = np.minimum(times_raw, censor_times).tolist()
    actual_cens = 1.0 - np.mean(events)
    dgp = {"distribution": "exponential", "rate": rate_true,
           "censoring_rate_target": censoring_rate, "censoring_rate_actual": float(actual_cens)}
    return obs_times, events, dgp


def gen_truth_recovery_cox(n: int, p: int, seed: int,
                            censoring_rate: float = 0.3) -> tuple[list[float], list[bool], list[list[float]], list[float], dict]:
    """Cox PH DGP with known beta_true and controlled censoring."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta_true = np.linspace(0.5, -0.3, p)
    hazard = np.exp(X @ beta_true)
    times_raw = rng.exponential(1.0 / hazard)

    # Bisection for censoring
    lo, hi = 0.01, 1000.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        rng_c = np.random.default_rng(seed + 888888)
        censor_times = rng_c.exponential(mid, n)
        frac = np.mean(times_raw > censor_times)
        if frac < censoring_rate:
            hi = mid
        else:
            lo = mid

    rng_final = np.random.default_rng(seed + 888888)
    censor_times = rng_final.exponential((lo + hi) / 2.0, n)
    events = (times_raw <= censor_times).tolist()
    obs_times = np.minimum(times_raw, censor_times).tolist()
    actual_cens = 1.0 - np.mean(events)
    dgp = {"distribution": "cox_exponential", "beta_true": beta_true.tolist(), "p": p,
           "censoring_rate_target": censoring_rate, "censoring_rate_actual": float(actual_cens)}
    return obs_times, events, X.tolist(), beta_true.tolist(), dgp


def gen_interval_censored_lognormal(n: int, seed: int, mu: float = 1.0, sigma: float = 0.5,
                                     inspection_interval: float = 1.0, exact_frac: float = 0.2,
                                     censoring_rate: float = 0.3) -> tuple[list[float], list[float], list[str], dict]:
    """Simulate interval-censored LogNormal data.

    True failure ~ LogNormal(mu, sigma). Inspections at t=k*interval.
    """
    rng = np.random.default_rng(seed)
    times_raw = rng.lognormal(mu, sigma, n)
    max_time = np.percentile(times_raw, (1.0 - censoring_rate) * 100)

    time_lower = []
    time_upper = []
    censor_type = []

    for t_true in times_raw:
        if t_true > max_time:
            time_lower.append(float(max_time))
            time_upper.append(float(max_time))
            censor_type.append("right")
        elif rng.random() < exact_frac:
            time_lower.append(float(t_true))
            time_upper.append(float(t_true))
            censor_type.append("exact")
        else:
            k = int(np.ceil(t_true / inspection_interval))
            t_lo = max(0.0, (k - 1) * inspection_interval)
            t_hi = k * inspection_interval
            time_lower.append(float(t_lo))
            time_upper.append(float(t_hi))
            censor_type.append("interval")

    dgp = {"distribution": "lognormal", "mu": mu, "sigma": sigma,
           "inspection_interval": inspection_interval, "exact_frac": exact_frac,
           "frac_exact": float(sum(1 for ct in censor_type if ct == "exact") / n),
           "frac_right": float(sum(1 for ct in censor_type if ct == "right") / n),
           "frac_interval": float(sum(1 for ct in censor_type if ct == "interval") / n)}
    return time_lower, time_upper, censor_type, dgp


def gen_interval_censored_exponential(n: int, seed: int, rate: float = 0.5,
                                       inspection_interval: float = 1.0, exact_frac: float = 0.2,
                                       censoring_rate: float = 0.3) -> tuple[list[float], list[float], list[str], dict]:
    """Simulate interval-censored Exponential data.

    True failure ~ Exp(rate). Inspections at t=k*interval.
    """
    rng = np.random.default_rng(seed)
    times_raw = rng.exponential(1.0 / rate, n)
    max_time = np.percentile(times_raw, (1.0 - censoring_rate) * 100)

    time_lower = []
    time_upper = []
    censor_type = []

    for t_true in times_raw:
        if t_true > max_time:
            time_lower.append(float(max_time))
            time_upper.append(float(max_time))
            censor_type.append("right")
        elif rng.random() < exact_frac:
            time_lower.append(float(t_true))
            time_upper.append(float(t_true))
            censor_type.append("exact")
        else:
            k = int(np.ceil(t_true / inspection_interval))
            t_lo = max(0.0, (k - 1) * inspection_interval)
            t_hi = k * inspection_interval
            time_lower.append(float(t_lo))
            time_upper.append(float(t_hi))
            censor_type.append("interval")

    dgp = {"distribution": "exponential", "rate": rate,
           "inspection_interval": inspection_interval, "exact_frac": exact_frac,
           "frac_exact": float(sum(1 for ct in censor_type if ct == "exact") / n),
           "frac_right": float(sum(1 for ct in censor_type if ct == "right") / n),
           "frac_interval": float(sum(1 for ct in censor_type if ct == "interval") / n)}
    return time_lower, time_upper, censor_type, dgp


def gen_interval_censored_weibull(n: int, seed: int, shape: float = 1.5, scale: float = 2.0,
                                   inspection_interval: float = 1.0, exact_frac: float = 0.2,
                                   censoring_rate: float = 0.3) -> tuple[list[float], list[float], list[str], dict]:
    """Simulate aviation inspection data: interval-censored Weibull.

    True failure ~ Weibull(shape, scale). Inspections at t=k*interval.
    exact_frac of failures observed exactly (in-flight detection).
    Right-censored at max_time.
    """
    rng = np.random.default_rng(seed)
    times_raw = rng.weibull(shape, n) * scale
    max_time = np.percentile(times_raw, (1.0 - censoring_rate) * 100)

    time_lower = []
    time_upper = []
    censor_type = []

    for t_true in times_raw:
        if t_true > max_time:
            # Right-censored
            time_lower.append(float(max_time))
            time_upper.append(float(max_time))
            censor_type.append("right")
        elif rng.random() < exact_frac:
            # Exact observation (in-flight detection)
            time_lower.append(float(t_true))
            time_upper.append(float(t_true))
            censor_type.append("exact")
        else:
            # Interval-censored: detected between inspections
            k = int(np.ceil(t_true / inspection_interval))
            t_lo = max(0.0, (k - 1) * inspection_interval)
            t_hi = k * inspection_interval
            time_lower.append(float(t_lo))
            time_upper.append(float(t_hi))
            censor_type.append("interval")

    actual_exact = sum(1 for ct in censor_type if ct == "exact") / n
    actual_right = sum(1 for ct in censor_type if ct == "right") / n
    actual_interval = sum(1 for ct in censor_type if ct == "interval") / n
    dgp = {"distribution": "weibull", "shape": shape, "scale": scale,
           "inspection_interval": inspection_interval, "exact_frac": exact_frac,
           "frac_exact": float(actual_exact), "frac_right": float(actual_right),
           "frac_interval": float(actual_interval)}
    return time_lower, time_upper, censor_type, dgp


# ---------------------------------------------------------------------------
# Truth-recovery MC loop
# ---------------------------------------------------------------------------

def run_truth_recovery(kind: str, n: int, p: int, n_replicates: int,
                        censoring_rates: list[float], seed_base: int) -> dict:
    """Run MC truth-recovery: n_replicates x len(censoring_rates) fits.

    Returns aggregated results per censoring rate.
    """
    import math
    results_by_censoring = []

    for cr in censoring_rates:
        param_records: dict[str, list[dict]] = {}
        n_converged = 0
        n_total = n_replicates

        for rep in range(n_replicates):
            seed_rep = seed_base + rep * 1000 + int(cr * 10000)

            try:
                if kind == "truth_weibull":
                    times, events, dgp = gen_truth_recovery_weibull(n, seed_rep, censoring_rate=cr)
                    model = nextstat.WeibullSurvivalModel(times=times, events=events)
                    true_params = {"log_k": math.log(dgp["shape"]), "log_lambda": math.log(dgp["scale"])}
                elif kind == "truth_exponential":
                    times, events, dgp = gen_truth_recovery_exponential(n, seed_rep, censoring_rate=cr)
                    model = nextstat.ExponentialSurvivalModel(times=times, events=events)
                    true_params = {"log_rate": math.log(dgp["rate"])}
                elif kind == "truth_cox":
                    times, events, X, beta_true, dgp = gen_truth_recovery_cox(n, p, seed_rep, censoring_rate=cr)
                    model = nextstat.CoxPhModel(times=times, events=events, x=X, ties="efron")
                    true_params = {f"beta{i+1}": float(b) for i, b in enumerate(beta_true)}
                elif kind == "truth_ic_weibull":
                    tl, tu, ct, dgp = gen_interval_censored_weibull(n, seed_rep, censoring_rate=cr)
                    model = nextstat.IntervalCensoredWeibullModel(time_lower=tl, time_upper=tu, censor_type=ct)
                    true_params = {"log_k": math.log(dgp["shape"]), "log_lambda": math.log(dgp["scale"])}
                elif kind == "truth_ic_lognormal":
                    tl, tu, ct, dgp = gen_interval_censored_lognormal(n, seed_rep, censoring_rate=cr)
                    model = nextstat.IntervalCensoredLogNormalModel(time_lower=tl, time_upper=tu, censor_type=ct)
                    true_params = {"mu": dgp["mu"], "log_sigma": math.log(dgp["sigma"])}
                elif kind == "truth_ic_exponential":
                    tl, tu, ct, dgp = gen_interval_censored_exponential(n, seed_rep, censoring_rate=cr)
                    model = nextstat.IntervalCensoredExponentialModel(time_lower=tl, time_upper=tu, censor_type=ct)
                    true_params = {"log_rate": math.log(dgp["rate"])}
                else:
                    continue

                result = nextstat.fit(model)
                if not result.converged:
                    continue
                n_converged += 1

                names = model.parameter_names()
                for j, name in enumerate(names):
                    if name not in param_records:
                        param_records[name] = []
                    hat = float(result.parameters[j])
                    se = float(result.uncertainties[j]) if j < len(result.uncertainties) else 0.0
                    true_val = true_params.get(name, 0.0)
                    # 95% Wald CI
                    ci_lo = hat - 1.96 * se
                    ci_hi = hat + 1.96 * se
                    covered = ci_lo <= true_val <= ci_hi
                    param_records[name].append({
                        "hat": hat, "true": true_val, "se": se,
                        "bias": hat - true_val, "covered": covered,
                    })
            except Exception:
                continue

        # Aggregate per parameter
        param_summaries = []
        for name, recs in param_records.items():
            if not recs:
                continue
            biases = [r["bias"] for r in recs]
            mean_bias = float(np.mean(biases))
            rmse = float(np.sqrt(np.mean([b**2 for b in biases])))
            coverage = float(np.mean([r["covered"] for r in recs]))
            mean_hat = float(np.mean([r["hat"] for r in recs]))
            true_val = recs[0]["true"]
            param_summaries.append({
                "name": name,
                "true": true_val,
                "mean_hat": mean_hat,
                "bias": mean_bias,
                "rmse": rmse,
                "coverage_95": coverage,
                "n_estimates": len(recs),
            })

        results_by_censoring.append({
            "censoring_rate": cr,
            "n_converged": n_converged,
            "n_total": n_total,
            "convergence_rate": n_converged / max(n_total, 1),
            "params": param_summaries,
        })

    return {"results_by_censoring": results_by_censoring}


# ---------------------------------------------------------------------------
# NextStat fitting
# ---------------------------------------------------------------------------

def _fit_cox_ph(times: list[float], events: list[bool], X: list[list[float]]) -> dict[str, Any]:
    model = nextstat.CoxPhModel(times=times, events=[bool(e) for e in events], x=X, ties="efron")
    result = nextstat.fit(model)
    return {
        "parameters": [float(v) for v in result.parameters],
        "nll": float(result.nll),
        "converged": bool(result.converged),
    }


def _fit_kaplan_meier(times: list[float], events: list[bool], conf_level: float = 0.95) -> dict[str, Any]:
    km = nextstat.kaplan_meier(times=times, events=[bool(e) for e in events], conf_level=float(conf_level))
    return {
        "times": [float(v) for v in km["time"]],
        "survival": [float(v) for v in km["survival"]],
        "ci_lower": [float(v) for v in km["ci_lower"]],
        "ci_upper": [float(v) for v in km["ci_upper"]],
    }


def _fit_log_rank(times: list[float], events: list[bool], groups: list[int]) -> dict[str, Any]:
    lr = nextstat.log_rank_test(times=times, events=[bool(e) for e in events], groups=groups)
    return {
        "statistic": float(lr["chi_squared"]),
        "p_value": float(lr["p_value"]),
        "df": int(lr["df"]),
    }


def _fit_weibull_aft(times: list[float], events: list[bool]) -> dict[str, Any]:
    model = nextstat.WeibullSurvivalModel(times=times, events=[bool(e) for e in events])
    result = nextstat.fit(model)
    return {
        "parameters": [float(v) for v in result.parameters],
        "nll": float(result.nll),
        "converged": bool(result.converged),
    }


# ---------------------------------------------------------------------------
# Baseline: lifelines
# ---------------------------------------------------------------------------

def _baseline_lifelines_cox(times: list[float], events: list[bool], X: list[list[float]]) -> dict[str, Any]:
    from lifelines import CoxPHFitter  # type: ignore[import-not-found]
    import pandas as pd  # type: ignore[import-not-found]

    p = len(X[0])
    df = pd.DataFrame(np.asarray(X, dtype=float), columns=[f"x{i}" for i in range(p)])
    df["T"] = times
    df["E"] = [int(e) for e in events]
    cph = CoxPHFitter()
    cph.fit(df, duration_col="T", event_col="E")
    coef = [float(v) for v in cph.params_.values]
    loglik = float(cph.log_likelihood_)
    return {"parameters": coef, "partial_loglik": loglik}


def _baseline_lifelines_km(times: list[float], events: list[bool]) -> dict[str, Any]:
    from lifelines import KaplanMeierFitter  # type: ignore[import-not-found]

    kmf = KaplanMeierFitter()
    kmf.fit(durations=times, event_observed=[int(e) for e in events])
    sf = kmf.survival_function_
    return {
        "times": [float(v) for v in sf.index.tolist()],
        "survival": [float(v) for v in sf.iloc[:, 0].tolist()],
    }


def _baseline_lifelines_weibull(times: list[float], events: list[bool]) -> dict[str, Any]:
    from lifelines import WeibullAFTFitter  # type: ignore[import-not-found]
    import pandas as pd  # type: ignore[import-not-found]

    df = pd.DataFrame({"T": times, "E": [int(e) for e in events]})
    wf = WeibullAFTFitter()
    wf.fit(df, duration_col="T", event_col="E")
    params = {}
    for name, val in wf.params_.items():
        key = "_".join(name) if isinstance(name, tuple) else str(name)
        if hasattr(val, "items"):
            for sub_name, sub_val in val.items():
                sub_key = "_".join(sub_name) if isinstance(sub_name, tuple) else str(sub_name)
                params[f"{key}_{sub_key}"] = float(sub_val)
        else:
            params[key] = float(val)
    return {"parameters": params}


def _baseline_sksurv_cox(times: list[float], events: list[bool], X: list[list[float]]) -> dict[str, Any]:
    from sksurv.linear_model import CoxPHSurvivalAnalysis  # type: ignore[import-not-found]

    X_arr = np.asarray(X, dtype=float)
    y_struct = np.array(
        [(bool(e), float(t)) for e, t in zip(events, times)],
        dtype=[("event", bool), ("time", float)],
    )
    cph = CoxPHSurvivalAnalysis()
    cph.fit(X_arr, y_struct)
    coef = [float(v) for v in cph.coef_]
    return {"parameters": coef}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--kind", required=True, choices=["cox_ph", "kaplan_meier", "weibull_aft", "truth_weibull", "truth_cox", "truth_exponential", "truth_ic_weibull", "truth_ic_lognormal", "truth_ic_exponential"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--n-replicates", type=int, default=200)
    ap.add_argument("--censoring-rates", type=str, default="0.1,0.3,0.5,0.7")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kind = str(args.kind)
    seed = int(args.seed)
    repeat = int(args.repeat)
    n = int(args.n)
    p = int(args.p)
    n_replicates = int(args.n_replicates)
    censoring_rates = [float(x) for x in args.censoring_rates.split(",")]

    # Dependency availability
    has_lifelines, lifelines_v = _maybe_import("lifelines")
    has_sksurv, sksurv_v = _maybe_import("sksurv")
    has_pandas, pandas_v = _maybe_import("pandas")

    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "nextstat_version": nextstat.__version__,
        "numpy_version": np.__version__,
        "lifelines_version": lifelines_v,
        "scikit_survival_version": sksurv_v,
        "pandas_version": pandas_v,
    }

    # --- Truth-recovery mode: MC loop (early return) ---
    if kind.startswith("truth_"):
        spec = {"kind": kind, "n": n, "seed": seed, "n_replicates": n_replicates,
                "censoring_rates": censoring_rates}
        if kind == "truth_cox":
            spec["p"] = p

        t0 = time.perf_counter()
        truth_results = run_truth_recovery(kind, n, p, n_replicates, censoring_rates, seed)
        wall = time.perf_counter() - t0

        # Build DGP info
        if kind == "truth_weibull":
            dgp_info = {"distribution": "weibull", "shape": 1.5, "scale": 2.0}
        elif kind == "truth_exponential":
            dgp_info = {"distribution": "exponential", "rate": 0.5}
        elif kind == "truth_cox":
            dgp_info = {"distribution": "cox_exponential", "p": p,
                        "beta_true": np.linspace(0.5, -0.3, p).tolist()}
        elif kind == "truth_ic_weibull":
            dgp_info = {"distribution": "weibull_ic", "shape": 1.5, "scale": 2.0}
        elif kind == "truth_ic_lognormal":
            dgp_info = {"distribution": "lognormal_ic", "mu": 1.0, "sigma": 0.5}
        elif kind == "truth_ic_exponential":
            dgp_info = {"distribution": "exponential_ic", "rate": 0.5}
        else:
            dgp_info = {}

        cfg = {"kind": kind, "n": n, "n_replicates": n_replicates,
               "censoring_rates": censoring_rates}
        if kind == "truth_cox":
            cfg["p"] = p

        obj = {
            "schema_version": "nextstat.survival_benchmark_result.v1",
            "suite": "survival",
            "case": str(args.case),
            "deterministic": bool(args.deterministic),
            "environment": collect_environment(),
            "status": "ok",
            "reason": None,
            "meta": meta,
            "dataset": {"id": f"generated:truth_recovery:{args.case}", "sha256": sha256_json_obj(spec), "spec": spec},
            "config": cfg,
            "truth_recovery": {
                "dgp": dgp_info,
                "n_replicates": n_replicates,
                **truth_results,
            },
            "timing": {"wall_time_s": {"min": wall, "median": wall, "p95": wall}},
            "parity": {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}},
            "results": {"nextstat": truth_results, "baseline": None},
        }
        out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

        # Print summary
        for cr_result in truth_results.get("results_by_censoring", []):
            cr = cr_result["censoring_rate"]
            conv = cr_result["convergence_rate"]
            params = cr_result.get("params", [])
            coverages = [p["coverage_95"] for p in params]
            avg_cov = np.mean(coverages) if coverages else 0.0
            print(f"  cens={cr:.1f}: conv={conv:.1%}, avg_coverage={avg_cov:.3f}")

        return 0

    # Generate data and build spec
    if kind == "cox_ph":
        spec = {"kind": kind, "n": n, "p": p, "seed": seed}
        times, events, X, beta_true = gen_cox_data(n, p, seed)
    elif kind == "kaplan_meier":
        spec = {"kind": kind, "n": n, "seed": seed}
        times, events, groups = gen_km_data(n, seed)
        X = []
        beta_true = []
    else:  # weibull_aft
        spec = {"kind": kind, "n": n, "seed": seed}
        times, events = gen_weibull_data(n, seed)
        X = []
        beta_true = []
        groups = []

    dataset = {"id": f"generated:survival:{args.case}", "sha256": sha256_json_obj(spec), "spec": spec}

    status = "ok"
    reason: Optional[str] = None

    # --- NextStat fit ---
    try:
        if kind == "cox_ph":
            ns_result = _fit_cox_ph(times, events, X)
        elif kind == "kaplan_meier":
            ns_km = _fit_kaplan_meier(times, events)
            ns_lr = _fit_log_rank(times, events, groups)
            ns_result = {"kaplan_meier": ns_km, "log_rank": ns_lr}
        else:
            ns_result = _fit_weibull_aft(times, events)
    except Exception as e:
        obj = {
            "schema_version": "nextstat.survival_benchmark_result.v1",
            "suite": "survival",
            "case": str(args.case),
            "deterministic": bool(args.deterministic),
            "environment": collect_environment(),
            "status": "failed",
            "reason": f"nextstat_error:{type(e).__name__}:{e}",
            "meta": meta,
            "dataset": dataset,
            "config": {"kind": kind, "n": n, "p": p},
            "parity": {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}},
            "timing": {"wall_time_s": {"min": 0.0, "median": 0.0, "p95": 0.0}, "raw": {"repeat": repeat, "policy": "median", "runs_s": []}},
            "results": {"nextstat": {}, "baseline": None},
        }
        out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        return 2

    # --- Timing ---
    runs_s: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        if kind == "cox_ph":
            _fit_cox_ph(times, events, X)
        elif kind == "kaplan_meier":
            _fit_kaplan_meier(times, events)
            _fit_log_rank(times, events, groups)
        else:
            _fit_weibull_aft(times, events)
        runs_s.append(float(time.perf_counter() - t0))

    timing = {"wall_time_s": _summary(runs_s), "raw": {"repeat": repeat, "policy": "median", "runs_s": [float(x) for x in runs_s]}}

    # --- Parity ---
    baseline_obj: Optional[dict[str, Any]] = None
    parity: dict[str, Any] = {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}}
    timing_baseline: dict[str, Any] = {}

    try:
        if kind == "cox_ph":
            # Primary baseline: lifelines
            if has_lifelines and has_pandas:
                bl = _baseline_lifelines_cox(times, events, X)
                baseline_obj = {"backend": "lifelines", "parameters": bl["parameters"], "partial_loglik": bl["partial_loglik"]}
                runs_bl: list[float] = []
                for _ in range(repeat):
                    t0 = time.perf_counter()
                    _baseline_lifelines_cox(times, events, X)
                    runs_bl.append(float(time.perf_counter() - t0))
                timing_baseline = {"name": "lifelines", "wall_time_s": _summary(runs_bl), "raw": {"repeat": repeat, "runs_s": runs_bl}}
                coef_abs, coef_rel = _max_abs_rel_diff(ns_result["parameters"], bl["parameters"])
                # Partial log-likelihood comparison (lifelines reports log-lik, NS reports NLL = -log-lik)
                partial_loglik_diff: Optional[float] = None
                if bl["partial_loglik"] is not None and ns_result["nll"] is not None:
                    # lifelines log_likelihood_ is the partial log-likelihood (positive or negative)
                    # NS nll is the negative partial log-likelihood
                    partial_loglik_diff = abs(float(ns_result["nll"]) + float(bl["partial_loglik"]))
                parity = {
                    "status": "ok" if coef_abs is not None else "warn",
                    "reference": {"name": "lifelines", "version": str(lifelines_v or "")},
                    "metrics": {
                        "coef_max_abs_diff": coef_abs,
                        "coef_max_rel_diff": coef_rel,
                        "partial_loglik_diff": partial_loglik_diff,
                    },
                }

                # Secondary baseline: scikit-survival (append to metrics)
                if has_sksurv:
                    try:
                        bl_sk = _baseline_sksurv_cox(times, events, X)
                        sk_abs, sk_rel = _max_abs_rel_diff(ns_result["parameters"], bl_sk["parameters"])
                        parity["metrics"]["sksurv_coef_max_abs_diff"] = sk_abs
                        parity["metrics"]["sksurv_coef_max_rel_diff"] = sk_rel
                    except Exception:
                        pass
            elif has_sksurv:
                bl_sk = _baseline_sksurv_cox(times, events, X)
                baseline_obj = {"backend": "scikit-survival", "parameters": bl_sk["parameters"]}
                sk_abs, sk_rel = _max_abs_rel_diff(ns_result["parameters"], bl_sk["parameters"])
                parity = {
                    "status": "ok" if sk_abs is not None else "warn",
                    "reference": {"name": "scikit-survival", "version": str(sksurv_v or "")},
                    "metrics": {
                        "coef_max_abs_diff": sk_abs,
                        "coef_max_rel_diff": sk_rel,
                    },
                }
            else:
                raise RuntimeError("missing_dependency:lifelines,scikit-survival")

        elif kind == "kaplan_meier":
            if has_lifelines:
                bl_km = _baseline_lifelines_km(times, events)
                runs_bl: list[float] = []
                for _ in range(repeat):
                    t0 = time.perf_counter()
                    _baseline_lifelines_km(times, events)
                    runs_bl.append(float(time.perf_counter() - t0))
                timing_baseline = {"name": "lifelines", "wall_time_s": _summary(runs_bl), "raw": {"repeat": repeat, "runs_s": runs_bl}}
                # Compare survival curves at common time points
                ns_times = ns_result["kaplan_meier"]["times"]
                ns_surv = ns_result["kaplan_meier"]["survival"]
                bl_times = bl_km["times"]
                bl_surv = bl_km["survival"]

                # Build survival lookup from baseline (step function)
                common_times = sorted(set(ns_times) & set(bl_times))
                if common_times:
                    ns_lookup = dict(zip(ns_times, ns_surv))
                    bl_lookup = dict(zip(bl_times, bl_surv))
                    ns_common = [float(ns_lookup[t]) for t in common_times]
                    bl_common = [float(bl_lookup[t]) for t in common_times]
                    surv_abs, surv_rel = _max_abs_rel_diff(ns_common, bl_common)
                else:
                    surv_abs, surv_rel = None, None

                baseline_obj = {"backend": "lifelines", "n_times": len(bl_times)}
                parity = {
                    "status": "ok" if surv_abs is not None else "warn",
                    "reference": {"name": "lifelines", "version": str(lifelines_v or "")},
                    "metrics": {
                        "survival_max_abs_diff": surv_abs,
                        "survival_max_rel_diff": surv_rel,
                        "n_common_times": len(common_times) if common_times else 0,
                    },
                }
            else:
                raise RuntimeError("missing_dependency:lifelines")

        elif kind == "weibull_aft":
            if has_lifelines and has_pandas:
                bl_w = _baseline_lifelines_weibull(times, events)
                baseline_obj = {"backend": "lifelines", "parameters": bl_w["parameters"]}
                runs_bl: list[float] = []
                for _ in range(repeat):
                    t0 = time.perf_counter()
                    _baseline_lifelines_weibull(times, events)
                    runs_bl.append(float(time.perf_counter() - t0))
                timing_baseline = {"name": "lifelines", "wall_time_s": _summary(runs_bl), "raw": {"repeat": repeat, "runs_s": runs_bl}}
                # Lifelines Weibull AFT parameterisation differs from NextStat;
                # record raw parameters for manual inspection.
                parity = {
                    "status": "ok",
                    "reference": {"name": "lifelines", "version": str(lifelines_v or "")},
                    "metrics": {
                        "note": "parameter_comparison_requires_reparameterisation",
                        "ns_parameters": ns_result["parameters"],
                        "lifelines_parameters": bl_w["parameters"],
                    },
                }
            else:
                raise RuntimeError("missing_dependency:lifelines")

    except Exception as e:
        status = "warn"
        reason = f"baseline_unavailable:{type(e).__name__}:{e}"
        parity = {"status": "warn", "reference": {"name": "", "version": ""}, "metrics": {}}

    # --- Config ---
    cfg: dict[str, Any] = {"kind": kind, "n": n}
    if kind == "cox_ph":
        cfg["p"] = p
        cfg["ties"] = "efron"
        cfg["n_events"] = sum(1 for e in events if e)
    elif kind == "kaplan_meier":
        cfg["n_groups"] = 2
        cfg["n_events"] = sum(1 for e in events if e)
    else:
        cfg["n_events"] = sum(1 for e in events if e)

    obj = {
        "schema_version": "nextstat.survival_benchmark_result.v1",
        "suite": "survival",
        "case": str(args.case),
        "deterministic": bool(args.deterministic),
        "environment": collect_environment(),
        "status": status,
        "reason": reason,
        "meta": meta,
        "dataset": dataset,
        "config": cfg,
        "parity": parity,
        "timing": timing,
        "timing_baseline": timing_baseline,
        "results": {"nextstat": ns_result, "baseline": baseline_obj},
    }
    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return 0 if status != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
