#!/usr/bin/env python3
"""Time Series benchmark runner: Kalman filter/smoother/EM + GARCH(1,1).

Backends:
- `nextstat` (always available)
- `pykalman` (optional Kalman parity)
- `statsmodels` (optional Kalman parity via UnobservedComponents)
- `arch` (optional GARCH parity)

Cases:
- kalman_local_level_500: 500-point local level model (filter + smooth + EM)
- kalman_local_level_5000: 5000-point local level model
- garch11_1000: GARCH(1,1) on 1000 simulated returns
- garch11_5000: GARCH(1,1) on 5000 simulated returns
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


# ---------------------------------------------------------------------------
# Data generation (deterministic, seeded)
# ---------------------------------------------------------------------------


def gen_local_level(n: int, seed: int, q: float = 0.1, r: float = 1.0) -> tuple[list[float], list[float]]:
    """Generate a local level (random walk + noise) time series.

    Returns (observations, latent_states).
    """
    rng = np.random.default_rng(int(seed))
    x = np.zeros(int(n))
    y = np.zeros(int(n))
    x[0] = rng.normal(0, 1)
    y[0] = x[0] + rng.normal(0, np.sqrt(float(r)))
    for t in range(1, int(n)):
        x[t] = x[t - 1] + rng.normal(0, np.sqrt(float(q)))
        y[t] = x[t] + rng.normal(0, np.sqrt(float(r)))
    return y.tolist(), x.tolist()


def gen_garch11(n: int, seed: int, omega: float = 0.01, alpha: float = 0.1, beta: float = 0.85) -> list[float]:
    """Generate GARCH(1,1) returns."""
    rng = np.random.default_rng(int(seed))
    returns = np.zeros(int(n))
    sigma2 = np.zeros(int(n))
    sigma2[0] = float(omega) / (1.0 - float(alpha) - float(beta))
    returns[0] = rng.normal(0, np.sqrt(sigma2[0]))
    for t in range(1, int(n)):
        sigma2[t] = float(omega) + float(alpha) * returns[t - 1] ** 2 + float(beta) * sigma2[t - 1]
        returns[t] = rng.normal(0, np.sqrt(sigma2[t]))
    return returns.tolist()


# ---------------------------------------------------------------------------
# NextStat fits
# ---------------------------------------------------------------------------


def _fit_kalman_ns(y: list[float], em_max_iter: int = 100, em_tol: float = 1e-6) -> dict[str, Any]:
    """Run NextStat Kalman filter + smooth + EM on a local level model."""
    model = nextstat.KalmanModel(
        f=[[1.0]],
        q=[[0.1]],
        h=[[1.0]],
        r=[[1.0]],
        m0=[0.0],
        p0=[[1.0]],
    )
    ys = [[yi] for yi in y]

    filtered = nextstat.kalman_filter(model, ys)
    smoothed = nextstat.kalman_smooth(model, ys)
    em_result = nextstat.kalman_em(model, ys, max_iter=int(em_max_iter), tol=float(em_tol), estimate_q=True, estimate_r=True)

    return {
        "filter_log_likelihood": float(filtered["log_likelihood"]),
        "smooth_log_likelihood": float(smoothed.get("log_likelihood", filtered["log_likelihood"])),
        "em_log_likelihood": float(em_result["loglik_trace"][-1]) if em_result.get("loglik_trace") else None,
        "em_converged": bool(em_result["converged"]),
        "em_n_iter": int(em_result["n_iter"]),
        "em_q": float(em_result["q"][0][0]) if em_result.get("q") else None,
        "em_r": float(em_result["r"][0][0]) if em_result.get("r") else None,
    }


def _fit_garch11_ns(returns: list[float]) -> dict[str, Any]:
    """Run NextStat GARCH(1,1) fit."""
    try:
        from nextstat._core import garch11_fit
        result = garch11_fit(returns, max_iter=1000, tol=1e-8)
    except (ImportError, AttributeError):
        import nextstat.timeseries
        result = nextstat.timeseries.garch11_fit(returns)

    params = result["params"]
    return {
        "omega": float(params["omega"]),
        "alpha": float(params["alpha"]),
        "beta": float(params["beta"]),
        "log_likelihood": float(result["log_likelihood"]),
        "converged": bool(result["converged"]),
    }


# ---------------------------------------------------------------------------
# Baseline: pykalman
# ---------------------------------------------------------------------------


def _baseline_pykalman(y: list[float], em_n_iter: int = 100) -> dict[str, Any]:
    from pykalman import KalmanFilter  # type: ignore[import-not-found]

    Y = np.asarray(y, dtype=float).reshape(-1, 1)
    kf = KalmanFilter(
        transition_matrices=[[1.0]],
        observation_matrices=[[1.0]],
        initial_state_mean=[0.0],
        initial_state_covariance=[[1.0]],
        n_dim_obs=1,
        n_dim_state=1,
    )
    kf_em = kf.em(Y, n_iter=int(em_n_iter))
    filtered_means, filtered_covs = kf_em.filter(Y)
    ll = kf_em.loglikelihood(Y)

    q_est = float(kf_em.transition_covariance[0, 0])
    r_est = float(kf_em.observation_covariance[0, 0])

    return {
        "log_likelihood": float(ll),
        "em_q": float(q_est),
        "em_r": float(r_est),
    }


# ---------------------------------------------------------------------------
# Baseline: statsmodels UnobservedComponents
# ---------------------------------------------------------------------------


def _baseline_statsmodels_kalman(y: list[float]) -> dict[str, Any]:
    from statsmodels.tsa.statespace.structural import UnobservedComponents  # type: ignore[import-not-found]

    Y = np.asarray(y, dtype=float)
    mod = UnobservedComponents(Y, level="local level")
    res = mod.fit(disp=False)

    return {
        "log_likelihood": float(res.llf),
        "sigma2_level": float(res.params[0]) if len(res.params) > 0 else None,
        "sigma2_irregular": float(res.params[1]) if len(res.params) > 1 else None,
    }


# ---------------------------------------------------------------------------
# Baseline: arch (GARCH)
# ---------------------------------------------------------------------------


def _baseline_arch_garch(returns: list[float]) -> dict[str, Any]:
    from arch import arch_model  # type: ignore[import-not-found]

    r = np.asarray(returns, dtype=float)
    am = arch_model(r, vol="GARCH", p=1, q=1, mean="Zero", rescale=False)
    res = am.fit(disp="off")

    return {
        "omega": float(res.params["omega"]),
        "alpha": float(res.params["alpha[1]"]),
        "beta": float(res.params["beta[1]"]),
        "log_likelihood": float(res.loglikelihood),
    }


# ---------------------------------------------------------------------------
# Parity helpers
# ---------------------------------------------------------------------------


def _scalar_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return abs(float(a) - float(b))


def _scalar_rel_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    denom = max(abs(float(a)), abs(float(b)), 1e-15)
    return abs(float(a) - float(b)) / denom


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--kind", required=True, choices=["kalman_local_level", "garch11"])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument(
        "--baseline-repeat",
        type=int,
        default=1,
        help="Repeat count for baseline timing loops (parity itself runs once).",
    )
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kind = str(args.kind)
    n = int(args.n)
    seed = int(args.seed)
    repeat = int(args.repeat)
    baseline_repeat = max(1, int(args.baseline_repeat))

    has_pykalman, pykalman_v = _maybe_import("pykalman")
    has_sm, sm_v = _maybe_import("statsmodels")
    has_arch, arch_v = _maybe_import("arch")

    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "nextstat_version": nextstat.__version__,
        "numpy_version": np.__version__,
        "pykalman_version": pykalman_v,
        "statsmodels_version": sm_v,
        "arch_version": arch_v,
    }

    # Generate dataset
    if kind == "kalman_local_level":
        spec = {"kind": kind, "n": n, "seed": seed, "q_true": 0.1, "r_true": 1.0}
        y, x_true = gen_local_level(n, seed, q=0.1, r=1.0)
    else:
        spec = {"kind": kind, "n": n, "seed": seed, "omega_true": 0.01, "alpha_true": 0.1, "beta_true": 0.85}
        returns = gen_garch11(n, seed, omega=0.01, alpha=0.1, beta=0.85)

    dataset = {"id": f"generated:timeseries:{args.case}", "sha256": sha256_json_obj(spec), "spec": spec}

    status = "ok"
    reason: Optional[str] = None

    # --- NextStat fit ---
    try:
        if kind == "kalman_local_level":
            ns_result = _fit_kalman_ns(y)
        else:
            ns_result = _fit_garch11_ns(returns)
    except Exception as e:
        obj = {
            "schema_version": "nextstat.timeseries_benchmark_result.v1",
            "suite": "timeseries",
            "case": str(args.case),
            "deterministic": bool(args.deterministic),
            "environment": collect_environment(),
            "status": "failed",
            "reason": f"nextstat_error:{type(e).__name__}:{e}",
            "meta": meta,
            "dataset": dataset,
            "config": {"kind": kind, "n": n},
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
        if kind == "kalman_local_level":
            _ = _fit_kalman_ns(y)
        else:
            _ = _fit_garch11_ns(returns)
        runs_s.append(float(time.perf_counter() - t0))

    timing = {"wall_time_s": _summary(runs_s), "raw": {"repeat": repeat, "policy": "median", "runs_s": [float(x) for x in runs_s]}}

    # --- Parity ---
    baseline_obj: Optional[dict[str, Any]] = None
    parity: dict[str, Any] = {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}}
    timing_baseline: dict[str, Any] = {}

    try:
        if args.skip_baselines:
            raise RuntimeError("baseline_skipped")

        if kind == "kalman_local_level":
            # Primary parity: pykalman (best EM-to-EM comparison)
            if has_pykalman:
                bl = _baseline_pykalman(y, em_n_iter=100)
                baseline_obj = {"backend": "pykalman", **bl}
                runs_bl: list[float] = []
                for _ in range(baseline_repeat):
                    t0 = time.perf_counter()
                    _baseline_pykalman(y, em_n_iter=100)
                    runs_bl.append(float(time.perf_counter() - t0))
                timing_baseline = {"name": "pykalman", "wall_time_s": _summary(runs_bl), "raw": {"repeat": baseline_repeat, "runs_s": runs_bl}}
                metrics: dict[str, Any] = {
                    "log_likelihood_abs_diff": _scalar_diff(ns_result.get("em_log_likelihood"), bl.get("log_likelihood")),
                    "log_likelihood_rel_diff": _scalar_rel_diff(ns_result.get("em_log_likelihood"), bl.get("log_likelihood")),
                    "q_abs_diff": _scalar_diff(ns_result.get("em_q"), bl.get("em_q")),
                    "q_rel_diff": _scalar_rel_diff(ns_result.get("em_q"), bl.get("em_q")),
                    "r_abs_diff": _scalar_diff(ns_result.get("em_r"), bl.get("em_r")),
                    "r_rel_diff": _scalar_rel_diff(ns_result.get("em_r"), bl.get("em_r")),
                }
                parity = {
                    "status": "ok",
                    "reference": {"name": "pykalman", "version": str(pykalman_v or "")},
                    "metrics": metrics,
                }
            elif has_sm:
                # Fallback: statsmodels UnobservedComponents (MLE, not EM — weaker parity)
                bl = _baseline_statsmodels_kalman(y)
                baseline_obj = {"backend": "statsmodels", **bl}
                runs_bl: list[float] = []
                for _ in range(baseline_repeat):
                    t0 = time.perf_counter()
                    _baseline_statsmodels_kalman(y)
                    runs_bl.append(float(time.perf_counter() - t0))
                timing_baseline = {"name": "statsmodels", "wall_time_s": _summary(runs_bl), "raw": {"repeat": baseline_repeat, "runs_s": runs_bl}}
                metrics = {
                    "log_likelihood_abs_diff": _scalar_diff(ns_result.get("filter_log_likelihood"), bl.get("log_likelihood")),
                    "log_likelihood_rel_diff": _scalar_rel_diff(ns_result.get("filter_log_likelihood"), bl.get("log_likelihood")),
                }
                parity = {
                    "status": "ok",
                    "reference": {"name": "statsmodels", "version": str(sm_v or "")},
                    "metrics": metrics,
                }
            else:
                parity = {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}}
        else:
            # GARCH parity: arch package
            if has_arch:
                bl = _baseline_arch_garch(returns)
                baseline_obj = {"backend": "arch", **bl}
                runs_bl: list[float] = []
                for _ in range(baseline_repeat):
                    t0 = time.perf_counter()
                    _baseline_arch_garch(returns)
                    runs_bl.append(float(time.perf_counter() - t0))
                timing_baseline = {"name": "arch", "wall_time_s": _summary(runs_bl), "raw": {"repeat": baseline_repeat, "runs_s": runs_bl}}
                metrics = {
                    "omega_abs_diff": _scalar_diff(ns_result.get("omega"), bl.get("omega")),
                    "omega_rel_diff": _scalar_rel_diff(ns_result.get("omega"), bl.get("omega")),
                    "alpha_abs_diff": _scalar_diff(ns_result.get("alpha"), bl.get("alpha")),
                    "alpha_rel_diff": _scalar_rel_diff(ns_result.get("alpha"), bl.get("alpha")),
                    "beta_abs_diff": _scalar_diff(ns_result.get("beta"), bl.get("beta")),
                    "beta_rel_diff": _scalar_rel_diff(ns_result.get("beta"), bl.get("beta")),
                    "log_likelihood_abs_diff": _scalar_diff(ns_result.get("log_likelihood"), bl.get("log_likelihood")),
                    "log_likelihood_rel_diff": _scalar_rel_diff(ns_result.get("log_likelihood"), bl.get("log_likelihood")),
                }
                parity = {
                    "status": "ok",
                    "reference": {"name": "arch", "version": str(arch_v or "")},
                    "metrics": metrics,
                }
            else:
                parity = {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}}

    except Exception as e:
        if str(e) == "baseline_skipped":
            parity = {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}}
        else:
            status = "warn"
            reason = f"baseline_unavailable:{type(e).__name__}:{e}"
            parity = {"status": "warn", "reference": {"name": "", "version": ""}, "metrics": {}}

    cfg: dict[str, Any] = {"kind": kind, "n": n}
    cfg["repeat"] = repeat
    cfg["baseline_repeat"] = baseline_repeat
    if kind == "kalman_local_level":
        cfg["em_max_iter"] = 100
        cfg["em_tol"] = 1e-6
    else:
        cfg["max_iter"] = 1000
        cfg["tol"] = 1e-8

    obj = {
        "schema_version": "nextstat.timeseries_benchmark_result.v1",
        "suite": "timeseries",
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
