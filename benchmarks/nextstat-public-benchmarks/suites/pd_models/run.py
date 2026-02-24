#!/usr/bin/env python3
"""PD (pharmacodynamic) benchmark runner: Emax, Sigmoid Emax, IDR models.

Cases:
- emax:          Emax dose-response fit via scipy.optimize + NS NLL
- sigmoid_emax:  Sigmoid Emax (Hill) fit via scipy.optimize + NS NLL
- idr_type1:     IDR Type I (InhibitProduction) simulation timing
- idr_type3:     IDR Type III (StimulateProduction) simulation timing

Baselines:
- Emax / Sigmoid-Emax: pure-Python manual NLL + scipy.optimize.minimize (timing baseline)
- IDR: no Python competitor; parity status = "skipped"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.bench_env import collect_environment

# ---------------------------------------------------------------------------
# Graceful import of nextstat._core
# ---------------------------------------------------------------------------

_NS_AVAILABLE = False
_NS_IMPORT_ERROR: Optional[str] = None
try:
    import nextstat._core as _ns  # type: ignore
    _NS_AVAILABLE = True
except ImportError as e:
    _NS_IMPORT_ERROR = f"ImportError: {e}"
except Exception as e:
    _NS_IMPORT_ERROR = f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

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
    return xs[i] * (1.0 - (k - i)) + xs[j] * (k - i)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values) if values else 0.0,
        "median": _pctl(values, 0.5),
        "p95": _pctl(values, 0.95),
    }


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def gen_emax_data(seed: int, n_conc: int = 20) -> dict[str, Any]:
    """Generate synthetic Emax dose-response data with proportional noise."""
    rng = np.random.default_rng(int(seed))
    true_e0, true_emax, true_ec50 = 5.0, 100.0, 10.0
    conc = np.concatenate([[0.0], np.logspace(-1, 2, n_conc - 1)]).tolist()
    model_pred = [true_e0 + true_emax * c / (true_ec50 + c) for c in conc]
    obs = [max(0.01, p * (1.0 + rng.normal(0, 0.1))) for p in model_pred]
    return {
        "conc": conc,
        "obs": obs,
        "true_params": [true_e0, true_emax, true_ec50],
        "param_names": ["E0", "Emax", "EC50"],
    }


def gen_sigmoid_emax_data(seed: int, n_conc: int = 20) -> dict[str, Any]:
    """Generate synthetic Sigmoid Emax (Hill) dose-response data."""
    rng = np.random.default_rng(int(seed))
    true_e0, true_emax, true_ec50, true_gamma = 5.0, 100.0, 10.0, 2.0
    conc = np.concatenate([[0.0], np.logspace(-1, 2, n_conc - 1)]).tolist()
    model_pred = [
        true_e0 + true_emax * (c ** true_gamma) / (true_ec50 ** true_gamma + c ** true_gamma)
        for c in conc
    ]
    obs = [max(0.01, p * (1.0 + rng.normal(0, 0.1))) for p in model_pred]
    return {
        "conc": conc,
        "obs": obs,
        "true_params": [true_e0, true_emax, true_ec50, true_gamma],
        "param_names": ["E0", "Emax", "EC50", "gamma"],
    }


def gen_idr_data(seed: int, idr_type: str, n_times: int = 50) -> dict[str, Any]:
    """Generate synthetic IDR data with single IV bolus concentration profile.

    Concentration: C(t) = (dose/V) * exp(-CL/V * t)  with CL=3, V=10, dose=100.
    IDR params: kin=1, kout=0.1, max_effect depends on type, c50=5.
    """
    rng = np.random.default_rng(int(seed))
    cl, v, dose = 3.0, 10.0, 100.0
    times = np.linspace(0, 48, n_times).tolist()
    conc = [(dose / v) * math.exp(-cl / v * t) for t in times]

    if idr_type == "type1":
        max_effect = 0.8
        label = "idr_type1"
    elif idr_type == "type3":
        max_effect = 2.0
        label = "idr_type3"
    else:
        raise ValueError(f"Unknown idr_type: {idr_type}")

    kin, kout, c50 = 1.0, 0.1, 5.0

    return {
        "conc_times": times,
        "conc_values": conc,
        "output_times": times,
        "true_params": [kin, kout, max_effect, c50],
        "param_names": ["kin", "kout", "max_effect", "c50"],
        "idr_type": idr_type,
        "label": label,
        "pk": {"CL": cl, "V": v, "dose": dose},
    }


# ---------------------------------------------------------------------------
# NextStat fits: Emax / Sigmoid-Emax
# ---------------------------------------------------------------------------

def _fit_emax_ns(data: dict[str, Any]) -> dict[str, Any]:
    """Fit Emax model via scipy minimize over NS NLL."""
    from scipy.optimize import minimize

    conc = data["conc"]
    obs = data["obs"]

    def neg_ll(params):
        try:
            return _ns.emax_nll(
                params[0], params[1], max(params[2], 0.01),
                conc, obs, error_model="proportional", sigma=0.1,
            )
        except Exception:
            return 1e10

    x0 = [1.0, 50.0, 5.0]
    res = minimize(neg_ll, x0, method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8})
    return {
        "params": res.x.tolist(),
        "nll": float(res.fun),
        "converged": bool(res.success),
        "n_iter": int(res.nit),
    }


def _fit_sigmoid_emax_ns(data: dict[str, Any]) -> dict[str, Any]:
    """Fit Sigmoid Emax (Hill) model via scipy minimize over NS NLL."""
    from scipy.optimize import minimize

    conc = data["conc"]
    obs = data["obs"]

    def neg_ll(params):
        try:
            return _ns.sigmoid_emax_nll(
                params[0], params[1], max(params[2], 0.01), max(params[3], 0.1),
                conc, obs, error_model="proportional", sigma=0.1,
            )
        except Exception:
            return 1e10

    x0 = [1.0, 50.0, 5.0, 1.5]
    res = minimize(neg_ll, x0, method="Nelder-Mead", options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8})
    return {
        "params": res.x.tolist(),
        "nll": float(res.fun),
        "converged": bool(res.success),
        "n_iter": int(res.nit),
    }


# ---------------------------------------------------------------------------
# NextStat IDR simulation
# ---------------------------------------------------------------------------

def _simulate_idr_ns(data: dict[str, Any]) -> dict[str, Any]:
    """Time IDR simulation via nextstat._core.idr_simulate."""
    kin, kout, max_effect, c50 = data["true_params"]
    result = _ns.idr_simulate(
        data["idr_type"],
        kin, kout, max_effect, c50,
        data["conc_times"],
        data["conc_values"],
        data["output_times"],
    )
    return {"response": [float(r) for r in result["response"]]}


# ---------------------------------------------------------------------------
# Pure-Python baseline fits: Emax / Sigmoid-Emax (manual NLL + scipy)
# ---------------------------------------------------------------------------

def _emax_predict_py(e0: float, emax: float, ec50: float, conc: list[float]) -> list[float]:
    return [e0 + emax * c / (ec50 + c) for c in conc]


def _sigmoid_emax_predict_py(
    e0: float, emax: float, ec50: float, gamma: float, conc: list[float]
) -> list[float]:
    return [e0 + emax * (c ** gamma) / (ec50 ** gamma + c ** gamma) for c in conc]


def _proportional_nll_py(pred: list[float], obs: list[float], sigma: float) -> float:
    """Proportional error NLL: obs ~ N(pred, (sigma*pred)^2)."""
    nll = 0.0
    for p, o in zip(pred, obs):
        sd = max(sigma * abs(p), 1e-10)
        nll += 0.5 * ((o - p) / sd) ** 2 + math.log(sd) + 0.5 * math.log(2.0 * math.pi)
    return nll


def _fit_emax_baseline(data: dict[str, Any]) -> dict[str, Any]:
    """Fit Emax via pure-Python NLL + scipy Nelder-Mead (timing baseline)."""
    from scipy.optimize import minimize

    conc = data["conc"]
    obs = data["obs"]

    def neg_ll(params):
        try:
            pred = _emax_predict_py(params[0], params[1], max(params[2], 0.01), conc)
            return _proportional_nll_py(pred, obs, 0.1)
        except Exception:
            return 1e10

    x0 = [1.0, 50.0, 5.0]
    res = minimize(neg_ll, x0, method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8})
    return {
        "params": res.x.tolist(),
        "nll": float(res.fun),
        "converged": bool(res.success),
        "n_iter": int(res.nit),
    }


def _fit_sigmoid_emax_baseline(data: dict[str, Any]) -> dict[str, Any]:
    """Fit Sigmoid Emax via pure-Python NLL + scipy Nelder-Mead (timing baseline)."""
    from scipy.optimize import minimize

    conc = data["conc"]
    obs = data["obs"]

    def neg_ll(params):
        try:
            pred = _sigmoid_emax_predict_py(params[0], params[1], max(params[2], 0.01), max(params[3], 0.1), conc)
            return _proportional_nll_py(pred, obs, 0.1)
        except Exception:
            return 1e10

    x0 = [1.0, 50.0, 5.0, 1.5]
    res = minimize(neg_ll, x0, method="Nelder-Mead", options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8})
    return {
        "params": res.x.tolist(),
        "nll": float(res.fun),
        "converged": bool(res.success),
        "n_iter": int(res.nit),
    }


# ---------------------------------------------------------------------------
# Parity helpers
# ---------------------------------------------------------------------------

def _param_parity(
    fitted: list[float], true: list[float], names: list[str], *, threshold: float = 0.3
) -> dict[str, Any]:
    """Compute per-parameter recovery metrics. Status 'ok' if all rel_diff < threshold."""
    if len(fitted) != len(true):
        return {"status": "warn", "reason": "param_count_mismatch"}
    recovery: dict[str, Any] = {}
    max_rel = 0.0
    for name, hat, tru in zip(names, fitted, true):
        rel_err = abs(hat - tru) / max(abs(tru), 1e-10)
        recovery[name] = {"hat": float(hat), "true": float(tru), "rel_err": float(rel_err)}
        max_rel = max(max_rel, rel_err)
    status = "ok" if max_rel < threshold else "warn"
    return {"status": status, "max_rel_diff": float(max_rel), "recovery": recovery}


def _idr_steady_state_check(data: dict[str, Any], response: list[float]) -> dict[str, Any]:
    """Check IDR endpoint against analytical steady-state (kout*R_ss = kin at C=0)."""
    kin, kout, _max_effect, _c50 = data["true_params"]
    r_ss_analytic = kin / kout  # Baseline steady state (no drug)
    r_end = response[-1] if response else float("nan")
    # At t=48h with CL/V=0.3 half-life ~2.3h, drug is washed out; response should
    # be returning toward baseline.  We check that the endpoint is within 50% of
    # the analytic baseline (loose — transient hasn't fully decayed).
    rel_diff = abs(r_end - r_ss_analytic) / max(abs(r_ss_analytic), 1e-10)
    status = "ok" if rel_diff < 0.5 else "warn"
    return {
        "status": status,
        "r_ss_analytic": float(r_ss_analytic),
        "r_endpoint": float(r_end),
        "rel_diff": float(rel_diff),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="PD benchmark runner")
    ap.add_argument("--case", required=True, help="Case id for reporting.")
    ap.add_argument("--kind", required=True,
                    choices=["emax", "sigmoid_emax", "idr_type1", "idr_type3"])
    ap.add_argument("--out", required=True, help="Output JSON path.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--n-conc", type=int, default=20, help="Number of concentration points (Emax/SigmoidEmax).")
    ap.add_argument("--n-times", type=int, default=50, help="Number of time points (IDR).")
    ap.add_argument("--skip-baselines", action="store_true", help="Skip pure-Python baseline fits.")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kind = str(args.kind)
    seed = int(args.seed)
    repeat = int(args.repeat)

    # -- Check nextstat availability ----------------------------------------
    if not _NS_AVAILABLE:
        obj = {
            "schema_version": "nextstat.pd_benchmark_result.v1",
            "suite": "pd_models",
            "case": str(args.case),
            "environment": collect_environment(),
            "status": "failed",
            "reason": f"nextstat_unavailable:{_NS_IMPORT_ERROR}",
        }
        out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        return 2

    # -- Generate data ------------------------------------------------------
    spec: dict[str, Any] = {"kind": kind, "seed": seed}
    if kind == "emax":
        data = gen_emax_data(seed, n_conc=int(args.n_conc))
        spec["n_conc"] = int(args.n_conc)
    elif kind == "sigmoid_emax":
        data = gen_sigmoid_emax_data(seed, n_conc=int(args.n_conc))
        spec["n_conc"] = int(args.n_conc)
    elif kind == "idr_type1":
        data = gen_idr_data(seed, idr_type="type1", n_times=int(args.n_times))
        spec["n_times"] = int(args.n_times)
    elif kind == "idr_type3":
        data = gen_idr_data(seed, idr_type="type3", n_times=int(args.n_times))
        spec["n_times"] = int(args.n_times)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    dataset = {
        "id": f"generated:pd:{args.case}",
        "sha256": sha256_json_obj(spec),
        "spec": spec,
    }

    status = "ok"
    reason: Optional[str] = None

    # -- Emax / Sigmoid-Emax: fit + timing ----------------------------------
    if kind in ("emax", "sigmoid_emax"):
        fit_fn = _fit_emax_ns if kind == "emax" else _fit_sigmoid_emax_ns
        baseline_fn = _fit_emax_baseline if kind == "emax" else _fit_sigmoid_emax_baseline

        # NS warmup + single fit
        try:
            ns_result = fit_fn(data)
        except Exception as e:
            obj = {
                "schema_version": "nextstat.pd_benchmark_result.v1",
                "suite": "pd_models",
                "case": str(args.case),
                "environment": collect_environment(),
                "status": "failed",
                "reason": f"nextstat_error:{type(e).__name__}:{e}",
                "dataset": dataset,
            }
            out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
            return 2

        # NS timing
        ns_runs: list[float] = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            fit_fn(data)
            ns_runs.append(float(time.perf_counter() - t0))

        # Baseline timing (pure-Python NLL + scipy)
        baseline_result: Optional[dict[str, Any]] = None
        baseline_runs: list[float] = []
        if not args.skip_baselines:
            try:
                baseline_result = baseline_fn(data)
                for _ in range(repeat):
                    t0 = time.perf_counter()
                    baseline_fn(data)
                    baseline_runs.append(float(time.perf_counter() - t0))
            except Exception as e:
                baseline_result = None
                reason = f"baseline_error:{type(e).__name__}:{e}"

        # Speedup
        ns_median = _pctl(ns_runs, 0.5)
        baseline_median = _pctl(baseline_runs, 0.5) if baseline_runs else 0.0
        speedup = baseline_median / ns_median if ns_median > 0 and baseline_median > 0 else None

        # Parity: parameter recovery vs true params
        parity = _param_parity(
            ns_result["params"], data["true_params"], data["param_names"],
        )

        timing: dict[str, Any] = {
            "wall_time_s": {
                "nextstat": _summary(ns_runs),
            },
            "raw": {
                "repeat": repeat,
                "policy": "median",
                "ns_runs_s": ns_runs,
            },
        }
        if baseline_runs:
            timing["wall_time_s"]["baseline_py"] = _summary(baseline_runs)
            timing["raw"]["baseline_runs_s"] = baseline_runs
        if speedup is not None:
            timing["speedup_vs_baseline"] = float(speedup)

        results: dict[str, Any] = {"nextstat": ns_result}
        if baseline_result is not None:
            results["baseline_py"] = baseline_result

        obj = {
            "schema_version": "nextstat.pd_benchmark_result.v1",
            "suite": "pd_models",
            "case": str(args.case),
            "environment": collect_environment(),
            "status": status,
            "reason": reason,
            "dataset": dataset,
            "config": {"kind": kind, "n_conc": int(args.n_conc), "seed": seed},
            "parity": parity,
            "timing": timing,
            "results": results,
        }

    # -- IDR: simulation timing ---------------------------------------------
    else:
        # Warmup
        try:
            idr_result = _simulate_idr_ns(data)
        except Exception as e:
            obj = {
                "schema_version": "nextstat.pd_benchmark_result.v1",
                "suite": "pd_models",
                "case": str(args.case),
                "environment": collect_environment(),
                "status": "failed",
                "reason": f"nextstat_error:{type(e).__name__}:{e}",
                "dataset": dataset,
            }
            out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
            return 2

        # Timing
        ns_runs: list[float] = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            _simulate_idr_ns(data)
            ns_runs.append(float(time.perf_counter() - t0))

        # Parity: steady-state check
        parity_check = _idr_steady_state_check(data, idr_result["response"])

        timing = {
            "wall_time_s": {"nextstat": _summary(ns_runs)},
            "raw": {"repeat": repeat, "policy": "median", "ns_runs_s": ns_runs},
        }

        obj = {
            "schema_version": "nextstat.pd_benchmark_result.v1",
            "suite": "pd_models",
            "case": str(args.case),
            "environment": collect_environment(),
            "status": status,
            "reason": reason,
            "dataset": dataset,
            "config": {"kind": kind, "n_times": int(args.n_times), "idr_type": data["idr_type"], "seed": seed},
            "parity": {"status": parity_check["status"], "steady_state": parity_check},
            "timing": timing,
            "results": {"nextstat": idr_result},
        }

    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return 0 if status != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
