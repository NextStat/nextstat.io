#!/usr/bin/env python3
"""GARCH / Stochastic Volatility benchmark runner.

Backends:
- `nextstat` (always available)
- `arch` (optional parity for GARCH, EGARCH, GJR-GARCH)

Cases:
- garch11: Standard GARCH(1,1) — nextstat._core.garch11_fit
- sv_logchi2: Stochastic Volatility log-chi2 — nextstat._core.sv_logchi2_fit
- egarch11: EGARCH(1,1) — nextstat._core.egarch11_fit (may not exist yet)
- gjr_garch11: GJR-GARCH(1,1) — nextstat._core.gjr_garch11_fit (may not exist yet)
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


def gen_garch11(n: int, seed: int, omega: float = 0.01, alpha: float = 0.1, beta: float = 0.85) -> list[float]:
    """Generate GARCH(1,1) returns.

    sigma2[0] = omega/(1-alpha-beta), sigma2[t] = omega + alpha*r[t-1]^2 + beta*sigma2[t-1]
    """
    rng = np.random.default_rng(int(seed))
    returns = np.zeros(int(n))
    sigma2 = np.zeros(int(n))
    sigma2[0] = float(omega) / (1.0 - float(alpha) - float(beta))
    returns[0] = rng.normal(0, np.sqrt(sigma2[0]))
    for t in range(1, int(n)):
        sigma2[t] = float(omega) + float(alpha) * returns[t - 1] ** 2 + float(beta) * sigma2[t - 1]
        returns[t] = rng.normal(0, np.sqrt(sigma2[t]))
    return returns.tolist()


def gen_sv_data(n: int, seed: int, mu: float = -0.5, phi: float = 0.97, sigma: float = 0.15) -> list[float]:
    """Generate Stochastic Volatility log-chi2 data.

    h[t] = mu + phi*(h[t-1]-mu) + sigma*eta[t]
    y[t] = exp(h[t]/2)*eps[t]
    """
    rng = np.random.default_rng(int(seed))
    h = np.zeros(int(n))
    y = np.zeros(int(n))
    h[0] = float(mu)
    y[0] = np.exp(h[0] / 2.0) * rng.normal(0, 1)
    for t in range(1, int(n)):
        h[t] = float(mu) + float(phi) * (h[t - 1] - float(mu)) + float(sigma) * rng.normal(0, 1)
        y[t] = np.exp(h[t] / 2.0) * rng.normal(0, 1)
    return y.tolist()


def gen_egarch_data(n: int, seed: int, omega: float = -0.1, alpha: float = 0.15, gamma: float = -0.05, beta: float = 0.98) -> list[float]:
    """Generate EGARCH(1,1) returns.

    log(sigma2[t]) = omega + alpha*(|z[t-1]| - E|z|) + gamma*z[t-1] + beta*log(sigma2[t-1])
    """
    rng = np.random.default_rng(int(seed))
    e_abs_z = np.sqrt(2.0 / np.pi)  # E[|z|] for z ~ N(0,1)
    returns = np.zeros(int(n))
    log_sigma2 = np.zeros(int(n))
    # Stationary initial: log(sigma2) = omega / (1 - beta)
    log_sigma2[0] = float(omega) / (1.0 - float(beta))
    returns[0] = rng.normal(0, np.sqrt(np.exp(log_sigma2[0])))
    for t in range(1, int(n)):
        sig_prev = np.sqrt(np.exp(log_sigma2[t - 1]))
        z_prev = returns[t - 1] / sig_prev if sig_prev > 1e-15 else 0.0
        log_sigma2[t] = (
            float(omega)
            + float(alpha) * (abs(z_prev) - e_abs_z)
            + float(gamma) * z_prev
            + float(beta) * log_sigma2[t - 1]
        )
        returns[t] = rng.normal(0, np.sqrt(np.exp(log_sigma2[t])))
    return returns.tolist()


def gen_gjr_data(n: int, seed: int, omega: float = 0.01, alpha: float = 0.05, gamma: float = 0.10, beta: float = 0.85) -> list[float]:
    """Generate GJR-GARCH(1,1) returns.

    sigma2[t] = omega + alpha*r[t-1]^2 + gamma*r[t-1]^2*I(r<0) + beta*sigma2[t-1]
    """
    rng = np.random.default_rng(int(seed))
    returns = np.zeros(int(n))
    sigma2 = np.zeros(int(n))
    # Stationary variance: omega / (1 - alpha - gamma/2 - beta)
    denom = 1.0 - float(alpha) - float(gamma) / 2.0 - float(beta)
    sigma2[0] = float(omega) / denom if denom > 1e-15 else float(omega)
    returns[0] = rng.normal(0, np.sqrt(sigma2[0]))
    for t in range(1, int(n)):
        indicator = 1.0 if returns[t - 1] < 0 else 0.0
        sigma2[t] = (
            float(omega)
            + float(alpha) * returns[t - 1] ** 2
            + float(gamma) * returns[t - 1] ** 2 * indicator
            + float(beta) * sigma2[t - 1]
        )
        returns[t] = rng.normal(0, np.sqrt(sigma2[t]))
    return returns.tolist()


# ---------------------------------------------------------------------------
# NextStat fits
# ---------------------------------------------------------------------------

# Feature availability flags — set at import time, graceful fallback.
_HAS_GARCH11 = False
_HAS_SV_LOGCHI2 = False
_HAS_EGARCH11 = False
_HAS_GJR_GARCH11 = False

try:
    from nextstat._core import garch11_fit as _ns_garch11_fit
    _HAS_GARCH11 = True
except (ImportError, AttributeError):
    _ns_garch11_fit = None  # type: ignore[assignment]

try:
    from nextstat._core import sv_logchi2_fit as _ns_sv_logchi2_fit
    _HAS_SV_LOGCHI2 = True
except (ImportError, AttributeError):
    _ns_sv_logchi2_fit = None  # type: ignore[assignment]

try:
    from nextstat._core import egarch11_fit as _ns_egarch11_fit
    _HAS_EGARCH11 = True
except (ImportError, AttributeError):
    _ns_egarch11_fit = None  # type: ignore[assignment]

try:
    from nextstat._core import gjr_garch11_fit as _ns_gjr_garch11_fit
    _HAS_GJR_GARCH11 = True
except (ImportError, AttributeError):
    _ns_gjr_garch11_fit = None  # type: ignore[assignment]


def _fit_garch11_ns(returns: list[float]) -> dict[str, Any]:
    """Run NextStat GARCH(1,1) fit."""
    if not _HAS_GARCH11:
        raise ImportError("nextstat._core.garch11_fit not available")
    result = _ns_garch11_fit(returns, max_iter=1000, tol=1e-8)  # type: ignore[misc]
    params = result["params"]
    return {
        "omega": float(params["omega"]),
        "alpha": float(params["alpha"]),
        "beta": float(params["beta"]),
        "log_likelihood": float(result["log_likelihood"]),
        "converged": bool(result["converged"]),
    }


def _fit_sv_logchi2_ns(returns: list[float]) -> dict[str, Any]:
    """Run NextStat SV log-chi2 fit."""
    if not _HAS_SV_LOGCHI2:
        raise ImportError("nextstat._core.sv_logchi2_fit not available")
    result = _ns_sv_logchi2_fit(returns, max_iter=1000, tol=1e-8)  # type: ignore[misc]
    params = result["params"]
    return {
        "mu": float(params["mu"]),
        "phi": float(params["phi"]),
        "sigma": float(params["sigma"]),
        "log_likelihood": float(result["log_likelihood"]),
        "converged": bool(result["converged"]),
    }


def _fit_egarch11_ns(returns: list[float]) -> dict[str, Any]:
    """Run NextStat EGARCH(1,1) fit."""
    if not _HAS_EGARCH11:
        raise ImportError("nextstat._core.egarch11_fit not available")
    result = _ns_egarch11_fit(returns, max_iter=1000, tol=1e-6)  # type: ignore[misc]
    params = result["params"]
    return {
        "omega": float(params["omega"]),
        "alpha": float(params["alpha"]),
        "gamma": float(params["gamma"]),
        "beta": float(params["beta"]),
        "log_likelihood": float(result["log_likelihood"]),
        "converged": bool(result["converged"]),
    }


def _fit_gjr_garch11_ns(returns: list[float]) -> dict[str, Any]:
    """Run NextStat GJR-GARCH(1,1) fit."""
    if not _HAS_GJR_GARCH11:
        raise ImportError("nextstat._core.gjr_garch11_fit not available")
    result = _ns_gjr_garch11_fit(returns, max_iter=1000, tol=1e-6)  # type: ignore[misc]
    params = result["params"]
    return {
        "omega": float(params["omega"]),
        "alpha": float(params["alpha"]),
        "gamma": float(params["gamma"]),
        "beta": float(params["beta"]),
        "log_likelihood": float(result["log_likelihood"]),
        "converged": bool(result["converged"]),
    }


_NS_FIT_DISPATCH = {
    "garch11": _fit_garch11_ns,
    "sv_logchi2": _fit_sv_logchi2_ns,
    "egarch11": _fit_egarch11_ns,
    "gjr_garch11": _fit_gjr_garch11_ns,
}

_NS_AVAILABLE = {
    "garch11": _HAS_GARCH11,
    "sv_logchi2": _HAS_SV_LOGCHI2,
    "egarch11": _HAS_EGARCH11,
    "gjr_garch11": _HAS_GJR_GARCH11,
}


# ---------------------------------------------------------------------------
# Baselines: arch package
# ---------------------------------------------------------------------------


def _baseline_arch_garch(returns: list[float]) -> dict[str, Any]:
    from arch import arch_model  # type: ignore[import-not-found]

    am = arch_model(np.asarray(returns), vol="GARCH", p=1, q=1, mean="Zero", rescale=False)
    res = am.fit(disp="off")
    return {
        "omega": float(res.params["omega"]),
        "alpha": float(res.params["alpha[1]"]),
        "beta": float(res.params["beta[1]"]),
        "log_likelihood": float(res.loglikelihood),
    }


def _baseline_arch_egarch(returns: list[float]) -> dict[str, Any]:
    from arch import arch_model  # type: ignore[import-not-found]

    am = arch_model(np.asarray(returns), vol="EGARCH", p=1, o=1, q=1, mean="Zero", rescale=False)
    res = am.fit(disp="off")
    return {
        "omega": float(res.params["omega"]),
        "alpha": float(res.params["alpha[1]"]),
        "gamma": float(res.params["gamma[1]"]),
        "beta": float(res.params["beta[1]"]),
        "log_likelihood": float(res.loglikelihood),
    }


def _baseline_arch_gjr(returns: list[float]) -> dict[str, Any]:
    from arch import arch_model  # type: ignore[import-not-found]

    am = arch_model(np.asarray(returns), vol="GARCH", p=1, o=1, q=1, mean="Zero", rescale=False)
    res = am.fit(disp="off")
    return {
        "omega": float(res.params["omega"]),
        "alpha": float(res.params["alpha[1]"]),
        "gamma": float(res.params["gamma[1]"]),
        "beta": float(res.params["beta[1]"]),
        "log_likelihood": float(res.loglikelihood),
    }


_BASELINE_DISPATCH = {
    "garch11": _baseline_arch_garch,
    "egarch11": _baseline_arch_egarch,
    "gjr_garch11": _baseline_arch_gjr,
    # SV: no good Python competitor. Parity skipped.
}


# ---------------------------------------------------------------------------
# Data generation dispatch
# ---------------------------------------------------------------------------

_DATA_GEN_DISPATCH = {
    "garch11": lambda n, seed: gen_garch11(n, seed, omega=0.01, alpha=0.1, beta=0.85),
    "sv_logchi2": lambda n, seed: gen_sv_data(n, seed, mu=-0.5, phi=0.97, sigma=0.15),
    "egarch11": lambda n, seed: gen_egarch_data(n, seed, omega=-0.1, alpha=0.15, gamma=-0.05, beta=0.98),
    "gjr_garch11": lambda n, seed: gen_gjr_data(n, seed, omega=0.01, alpha=0.05, gamma=0.10, beta=0.85),
}

_DATA_SPEC = {
    "garch11": {"omega_true": 0.01, "alpha_true": 0.1, "beta_true": 0.85},
    "sv_logchi2": {"mu_true": -0.5, "phi_true": 0.97, "sigma_true": 0.15},
    "egarch11": {"omega_true": -0.1, "alpha_true": 0.15, "gamma_true": -0.05, "beta_true": 0.98},
    "gjr_garch11": {"omega_true": 0.01, "alpha_true": 0.05, "gamma_true": 0.10, "beta_true": 0.85},
}

_TOL_MAP = {
    "garch11": 1e-8,
    "sv_logchi2": 1e-8,
    "egarch11": 1e-6,
    "gjr_garch11": 1e-6,
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


def _compute_parity_garch(ns_result: dict, bl: dict) -> dict[str, Any]:
    """Parity metrics for GARCH(1,1): omega, alpha, beta, log_likelihood."""
    return {
        "omega_abs_diff": _scalar_diff(ns_result.get("omega"), bl.get("omega")),
        "omega_rel_diff": _scalar_rel_diff(ns_result.get("omega"), bl.get("omega")),
        "alpha_abs_diff": _scalar_diff(ns_result.get("alpha"), bl.get("alpha")),
        "alpha_rel_diff": _scalar_rel_diff(ns_result.get("alpha"), bl.get("alpha")),
        "beta_abs_diff": _scalar_diff(ns_result.get("beta"), bl.get("beta")),
        "beta_rel_diff": _scalar_rel_diff(ns_result.get("beta"), bl.get("beta")),
        "log_likelihood_abs_diff": _scalar_diff(ns_result.get("log_likelihood"), bl.get("log_likelihood")),
        "log_likelihood_rel_diff": _scalar_rel_diff(ns_result.get("log_likelihood"), bl.get("log_likelihood")),
    }


def _compute_parity_egarch(ns_result: dict, bl: dict) -> dict[str, Any]:
    """Parity metrics for EGARCH/GJR: omega, alpha, gamma, beta, log_likelihood."""
    return {
        "omega_abs_diff": _scalar_diff(ns_result.get("omega"), bl.get("omega")),
        "omega_rel_diff": _scalar_rel_diff(ns_result.get("omega"), bl.get("omega")),
        "alpha_abs_diff": _scalar_diff(ns_result.get("alpha"), bl.get("alpha")),
        "alpha_rel_diff": _scalar_rel_diff(ns_result.get("alpha"), bl.get("alpha")),
        "gamma_abs_diff": _scalar_diff(ns_result.get("gamma"), bl.get("gamma")),
        "gamma_rel_diff": _scalar_rel_diff(ns_result.get("gamma"), bl.get("gamma")),
        "beta_abs_diff": _scalar_diff(ns_result.get("beta"), bl.get("beta")),
        "beta_rel_diff": _scalar_rel_diff(ns_result.get("beta"), bl.get("beta")),
        "log_likelihood_abs_diff": _scalar_diff(ns_result.get("log_likelihood"), bl.get("log_likelihood")),
        "log_likelihood_rel_diff": _scalar_rel_diff(ns_result.get("log_likelihood"), bl.get("log_likelihood")),
    }


_PARITY_DISPATCH = {
    "garch11": _compute_parity_garch,
    "egarch11": _compute_parity_egarch,
    "gjr_garch11": _compute_parity_egarch,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--kind", required=True, choices=["garch11", "sv_logchi2", "egarch11", "gjr_garch11"])
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

    has_arch, arch_v = _maybe_import("arch")

    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "nextstat_version": nextstat.__version__,
        "numpy_version": np.__version__,
        "arch_version": arch_v,
        "ns_garch11_available": _HAS_GARCH11,
        "ns_sv_logchi2_available": _HAS_SV_LOGCHI2,
        "ns_egarch11_available": _HAS_EGARCH11,
        "ns_gjr_garch11_available": _HAS_GJR_GARCH11,
    }

    # Generate dataset
    spec: dict[str, Any] = {"kind": kind, "n": n, "seed": seed}
    spec.update(_DATA_SPEC[kind])
    returns = _DATA_GEN_DISPATCH[kind](n, seed)
    dataset = {"id": f"generated:garch_family:{args.case}", "sha256": sha256_json_obj(spec), "spec": spec}

    status = "ok"
    reason: Optional[str] = None

    # --- Check NS availability ---
    if not _NS_AVAILABLE[kind]:
        obj = {
            "schema_version": "nextstat.garch_family_benchmark_result.v1",
            "suite": "garch_family",
            "case": str(args.case),
            "deterministic": bool(args.deterministic),
            "environment": collect_environment(),
            "status": "skipped",
            "reason": f"nextstat._core.{kind}_fit not available (ImportError)",
            "meta": meta,
            "dataset": dataset,
            "config": {"kind": kind, "n": n, "max_iter": 1000, "tol": _TOL_MAP[kind]},
            "parity": {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}},
            "timing": {"wall_time_s": {"min": 0.0, "median": 0.0, "p95": 0.0}, "raw": {"repeat": repeat, "policy": "median", "runs_s": []}},
            "timing_baseline": {},
            "results": {"nextstat": {}, "baseline": None},
        }
        out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        print(f"  [{args.case}] SKIPPED — {kind}_fit not in nextstat._core")
        return 0

    # --- NextStat fit ---
    ns_fit_fn = _NS_FIT_DISPATCH[kind]
    try:
        ns_result = ns_fit_fn(returns)
    except Exception as e:
        obj = {
            "schema_version": "nextstat.garch_family_benchmark_result.v1",
            "suite": "garch_family",
            "case": str(args.case),
            "deterministic": bool(args.deterministic),
            "environment": collect_environment(),
            "status": "failed",
            "reason": f"nextstat_error:{type(e).__name__}:{e}",
            "meta": meta,
            "dataset": dataset,
            "config": {"kind": kind, "n": n, "max_iter": 1000, "tol": _TOL_MAP[kind]},
            "parity": {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}},
            "timing": {"wall_time_s": {"min": 0.0, "median": 0.0, "p95": 0.0}, "raw": {"repeat": repeat, "policy": "median", "runs_s": []}},
            "timing_baseline": {},
            "results": {"nextstat": {}, "baseline": None},
        }
        out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        return 2

    # --- Timing (NextStat) ---
    runs_s: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = ns_fit_fn(returns)
        runs_s.append(float(time.perf_counter() - t0))

    timing = {"wall_time_s": _summary(runs_s), "raw": {"repeat": repeat, "policy": "median", "runs_s": [float(x) for x in runs_s]}}

    # --- Parity (baselines) ---
    baseline_obj: Optional[dict[str, Any]] = None
    parity: dict[str, Any] = {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}}
    timing_baseline: dict[str, Any] = {}

    try:
        if args.skip_baselines:
            raise RuntimeError("baseline_skipped")

        if kind == "sv_logchi2":
            # No good Python SV competitor.  Parity skipped.
            parity = {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}}
        elif has_arch and kind in _BASELINE_DISPATCH:
            baseline_fn = _BASELINE_DISPATCH[kind]
            bl = baseline_fn(returns)
            baseline_obj = {"backend": "arch", **bl}

            # Baseline timing
            runs_bl: list[float] = []
            for _ in range(baseline_repeat):
                t0 = time.perf_counter()
                baseline_fn(returns)
                runs_bl.append(float(time.perf_counter() - t0))
            timing_baseline = {"name": "arch", "wall_time_s": _summary(runs_bl), "raw": {"repeat": baseline_repeat, "runs_s": runs_bl}}

            # Parity metrics
            parity_fn = _PARITY_DISPATCH[kind]
            metrics = parity_fn(ns_result, bl)
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

    cfg: dict[str, Any] = {
        "kind": kind,
        "n": n,
        "repeat": repeat,
        "baseline_repeat": baseline_repeat,
        "max_iter": 1000,
        "tol": _TOL_MAP[kind],
    }

    obj = {
        "schema_version": "nextstat.garch_family_benchmark_result.v1",
        "suite": "garch_family",
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
