#!/usr/bin/env python3
"""EVT (Extreme Value Theory) benchmark seed: GEV/GPD fitting + parity vs scipy.

Cases:
- gev_block_maxima_500:  500 samples from GEV(mu, sigma, xi)
- gev_block_maxima_5000: 5000 samples from GEV(mu, sigma, xi)
- gpd_threshold_500:     500 exceedances from GPD(sigma, xi)
- gpd_threshold_5000:    5000 exceedances from GPD(sigma, xi)

Baseline: scipy.stats.genextreme / scipy.stats.genpareto (used internally by pyextremes).
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


def sha256_json_obj(obj: dict[str, Any]) -> str:
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


# ---------------------------------------------------------------------------
# Data generators (inverse CDF sampling)
# ---------------------------------------------------------------------------


def gen_gev_data(n: int, seed: int, mu: float = 100.0, sigma: float = 10.0, xi: float = 0.1) -> list[float]:
    """Generate n samples from GEV(mu, sigma, xi) via inverse CDF."""
    rng = np.random.default_rng(int(seed))
    u = rng.uniform(size=int(n))
    if abs(xi) > 1e-8:
        data = mu + sigma * ((-np.log(u)) ** (-xi) - 1.0) / xi
    else:
        data = mu - sigma * np.log(-np.log(u))
    return [float(x) for x in data]


def gen_gpd_data(n: int, seed: int, sigma: float = 2.0, xi: float = 0.2) -> list[float]:
    """Generate n exceedances from GPD(sigma, xi) via inverse CDF."""
    rng = np.random.default_rng(int(seed))
    u = rng.uniform(size=int(n))
    if abs(xi) > 1e-8:
        data = sigma * ((1.0 - u) ** (-xi) - 1.0) / xi
    else:
        data = -sigma * np.log(1.0 - u)
    return [float(x) for x in data]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="Case id for reporting.")
    ap.add_argument("--model", required=True, choices=["gev", "gpd"], help="Model type.")
    ap.add_argument("--n", type=int, required=True, help="Number of samples / exceedances.")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=50, help="Number of timing repeats.")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--mu", type=float, default=100.0, help="GEV location parameter (mu).")
    ap.add_argument("--sigma", type=float, default=10.0, help="Scale parameter (sigma).")
    ap.add_argument("--xi", type=float, default=0.1, help="Shape parameter (xi).")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_type = str(args.model)
    n = int(args.n)
    seed = int(args.seed)
    repeat = int(args.repeat)
    mu = float(args.mu)
    sigma = float(args.sigma)
    xi = float(args.xi)

    # ---- Generate data ----
    if model_type == "gev":
        data = gen_gev_data(n, seed, mu=mu, sigma=sigma, xi=xi)
        spec = {"kind": "gev", "n": n, "seed": seed, "mu": mu, "sigma": sigma, "xi": xi}
        true_params = {"mu": mu, "sigma": sigma, "xi": xi}
    else:
        data = gen_gpd_data(n, seed, sigma=sigma, xi=xi)
        spec = {"kind": "gpd", "n": n, "seed": seed, "sigma": sigma, "xi": xi}
        true_params = {"sigma": sigma, "xi": xi}

    dataset_id = f"generated:evt:{model_type}:n{n}:seed{seed}"
    dataset_sha = sha256_json_obj(spec)

    has_scipy, scipy_v = _maybe_import("scipy")

    status = "ok"
    reason: Optional[str] = None

    # ---- NextStat fit ----
    try:
        if model_type == "gev":
            ns_model = nextstat.GevModel(data=data)
        else:
            ns_model = nextstat.GpdModel(exceedances=data)

        ns_result = nextstat.fit(ns_model)
        ns_params = list(ns_result.parameters)
        ns_nll = float(ns_result.nll)
        ns_converged = bool(ns_result.converged)
        ns_n_evaluations = int(ns_result.n_evaluations)
        ns_termination = str(ns_result.termination_reason)
    except Exception as e:
        obj: dict[str, Any] = {
            "schema_version": "nextstat.evt_benchmark_result.v1",
            "suite": "evt",
            "case": str(args.case),
            "deterministic": bool(args.deterministic),
            "status": "failed",
            "reason": f"nextstat_error:{type(e).__name__}:{e}",
            "meta": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "nextstat_version": nextstat.__version__,
                "numpy_version": np.__version__,
                "scipy_version": scipy_v,
            },
            "dataset": {"id": dataset_id, "sha256": dataset_sha, "spec": spec},
            "config": {"model": model_type, "n": n, "true_params": true_params},
            "parity": {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}},
            "timing": {"fit_time_s": {"min": 0.0, "median": 0.0, "p95": 0.0}, "raw": {"repeat": repeat, "policy": "median", "runs_s": {"nextstat": [], "scipy": []}}},
            "results": {"nextstat": None, "baseline": None},
        }
        out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        return 2

    # ---- NextStat timing ----
    ns_runs_s: list[float] = []
    for _ in range(repeat):
        if model_type == "gev":
            m = nextstat.GevModel(data=data)
        else:
            m = nextstat.GpdModel(exceedances=data)
        t0 = time.perf_counter()
        nextstat.fit(m)
        ns_runs_s.append(float(time.perf_counter() - t0))

    # ---- Unpack NextStat params ----
    if model_type == "gev":
        ns_named = {"mu": float(ns_params[0]), "sigma": float(ns_params[1]), "xi": float(ns_params[2])}
    else:
        ns_named = {"sigma": float(ns_params[0]), "xi": float(ns_params[1])}

    ns_result_obj: dict[str, Any] = {
        "params": ns_named,
        "nll": float(ns_nll),
        "converged": bool(ns_converged),
        "n_evaluations": int(ns_n_evaluations),
        "termination_reason": str(ns_termination),
    }

    # ---- Scipy baseline ----
    baseline_obj: Optional[dict[str, Any]] = None
    parity: dict[str, Any] = {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}}
    scipy_runs_s: list[float] = []

    try:
        if not has_scipy:
            raise RuntimeError("missing_dependency:scipy")

        from scipy import stats  # type: ignore[import-not-found]

        data_arr = np.asarray(data, dtype=float)

        if model_type == "gev":
            # scipy genextreme uses c = -xi convention
            t0 = time.perf_counter()
            shape_sc, loc_sc, scale_sc = stats.genextreme.fit(data_arr)
            _ = time.perf_counter() - t0

            sc_xi = float(-shape_sc)
            sc_mu = float(loc_sc)
            sc_sigma = float(scale_sc)
            sc_nll = float(-stats.genextreme.logpdf(data_arr, shape_sc, loc=loc_sc, scale=scale_sc).sum())
            sc_named = {"mu": sc_mu, "sigma": sc_sigma, "xi": sc_xi}

            # Timing
            for _ in range(repeat):
                t0 = time.perf_counter()
                stats.genextreme.fit(data_arr)
                scipy_runs_s.append(float(time.perf_counter() - t0))
        else:
            # GPD: fix loc=0
            t0 = time.perf_counter()
            shape_sc, loc_sc, scale_sc = stats.genpareto.fit(data_arr, floc=0)
            _ = time.perf_counter() - t0

            sc_xi = float(shape_sc)
            sc_sigma = float(scale_sc)
            sc_nll = float(-stats.genpareto.logpdf(data_arr, shape_sc, loc=0, scale=scale_sc).sum())
            sc_named = {"sigma": sc_sigma, "xi": sc_xi}

            # Timing
            for _ in range(repeat):
                t0 = time.perf_counter()
                stats.genpareto.fit(data_arr, floc=0)
                scipy_runs_s.append(float(time.perf_counter() - t0))

        baseline_obj = {"params": sc_named, "nll": float(sc_nll)}

        # Parity metrics
        param_abs_diffs: list[float] = []
        param_rel_diffs: list[float] = []
        for key in ns_named:
            ns_v = float(ns_named[key])
            sc_v = float(sc_named[key])
            d = abs(ns_v - sc_v)
            param_abs_diffs.append(d)
            denom = max(abs(ns_v), abs(sc_v), 1e-12)
            param_rel_diffs.append(d / denom)

        nll_abs_diff = abs(float(ns_nll) - float(sc_nll))
        nll_rel_diff = nll_abs_diff / max(abs(float(ns_nll)), abs(float(sc_nll)), 1e-12)

        parity = {
            "status": "ok",
            "reference": {"name": "scipy", "version": str(scipy_v or "")},
            "metrics": {
                "param_max_abs_diff": float(max(param_abs_diffs)),
                "param_max_rel_diff": float(max(param_rel_diffs)),
                "nll_abs_diff": float(nll_abs_diff),
                "nll_rel_diff": float(nll_rel_diff),
            },
        }
    except Exception as e:
        status = "warn"
        reason = f"baseline_unavailable:{type(e).__name__}:{e}"
        parity = {"status": "warn", "reference": {"name": "", "version": ""}, "metrics": {}}

    # ---- Speedup ----
    ns_median = _pctl(ns_runs_s, 0.5)
    sc_median = _pctl(scipy_runs_s, 0.5) if scipy_runs_s else 0.0
    speedup = sc_median / ns_median if ns_median > 0 and sc_median > 0 else None

    # ---- Build artifact ----
    obj = {
        "schema_version": "nextstat.evt_benchmark_result.v1",
        "suite": "evt",
        "case": str(args.case),
        "deterministic": bool(args.deterministic),
        "status": status,
        "reason": reason,
        "meta": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "nextstat_version": nextstat.__version__,
            "numpy_version": np.__version__,
            "scipy_version": scipy_v,
        },
        "dataset": {"id": dataset_id, "sha256": dataset_sha, "spec": spec},
        "config": {"model": model_type, "n": n, "true_params": true_params},
        "parity": parity,
        "timing": {
            "fit_time_s": {
                "nextstat": _summary(ns_runs_s),
                **({"scipy": _summary(scipy_runs_s)} if scipy_runs_s else {}),
            },
            **({"speedup_scipy_over_nextstat": float(speedup)} if speedup is not None else {}),
            "raw": {
                "repeat": int(repeat),
                "policy": "median",
                "runs_s": {
                    "nextstat": [float(x) for x in ns_runs_s],
                    **({"scipy": [float(x) for x in scipy_runs_s]} if scipy_runs_s else {}),
                },
            },
        },
        "results": {
            "nextstat": ns_result_obj,
            "baseline": baseline_obj,
        },
    }
    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return 0 if status != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
