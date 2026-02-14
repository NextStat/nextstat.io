#!/usr/bin/env python3
"""GLM benchmark: single-case runner with parity against statsmodels, scikit-learn, glum.

Cases (4 families x 3 sizes = 12):
- linear_1k, linear_10k, linear_100k
- logistic_1k, logistic_10k, logistic_100k
- poisson_1k, poisson_10k, poisson_100k
- negbin_1k, negbin_10k, negbin_100k

Each case generates synthetic data with p=10 covariates, fits with NextStat,
then compares coefficients and log-likelihood against available competitors.
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


# ---------------------------------------------------------------------------
# Utility helpers (same pattern as econometrics suite)
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
    a = xs[i]
    b = xs[j]
    t = k - i
    return a * (1.0 - t) + b * t


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values) if values else 0.0,
        "median": _pctl(values, 0.5),
        "p95": _pctl(values, 0.95),
    }


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

def gen_glm_data(family: str, n: int, p: int, seed: int) -> dict[str, Any]:
    """Generate synthetic GLM data for the given family."""
    rng = np.random.default_rng(int(seed))
    X = rng.normal(size=(n, p)).astype(float)
    beta_true = np.linspace(0.5, -0.5, p).astype(float)
    intercept = 0.5
    eta = intercept + X @ beta_true

    if family == "linear":
        y = (eta + rng.normal(0, 1, n)).astype(float)
    elif family == "logistic":
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob).astype(int)
    elif family == "poisson":
        y = rng.poisson(np.exp(np.clip(eta, -10, 10))).astype(int)
    elif family == "negbin":
        mu = np.exp(np.clip(eta, -10, 10))
        r = 5.0  # dispersion parameter
        p_nb = r / (r + mu)
        y = rng.negative_binomial(r, p_nb).astype(int)
    else:
        raise ValueError(f"Unknown family: {family}")

    return {
        "X": X.tolist(),
        "y": y.tolist(),
        "beta_true": [float(intercept)] + beta_true.tolist(),
        "family": family,
        "n": n,
        "p": p,
    }


# ---------------------------------------------------------------------------
# NextStat fit
# ---------------------------------------------------------------------------

def _fit_ns(data: dict[str, Any]) -> tuple[list[float], float, bool, dict[str, Any]]:
    """Fit with NextStat, return (coefficients, nll, converged, extra)."""
    family = str(data["family"])
    X = data["X"]
    y = data["y"]

    if family == "linear":
        model = nextstat.LinearRegressionModel(x=X, y=y, include_intercept=True)
    elif family == "logistic":
        model = nextstat.LogisticRegressionModel(x=X, y=y, include_intercept=True)
    elif family == "poisson":
        model = nextstat.PoissonRegressionModel(x=X, y=y, include_intercept=True)
    elif family == "negbin":
        model = nextstat.NegativeBinomialRegressionModel(x=X, y=y, include_intercept=True)
    else:
        raise ValueError(f"Unknown family: {family}")

    result = nextstat.fit(model)
    coef = [float(c) for c in result.parameters]
    nll = float(result.nll)
    converged = bool(result.converged)
    extra = {
        "n_evaluations": int(result.n_evaluations),
        "termination_reason": str(result.termination_reason),
    }
    return coef, nll, converged, extra


# ---------------------------------------------------------------------------
# Baseline competitors
# ---------------------------------------------------------------------------

_STATSMODELS_FAMILY_MAP = {
    "linear": "Gaussian",
    "logistic": "Binomial",
    "poisson": "Poisson",
    "negbin": "NegativeBinomial",
}


def _baseline_statsmodels(data: dict[str, Any]) -> tuple[list[float], float, dict[str, Any]]:
    """Fit with statsmodels GLM, return (coefficients, log_likelihood, extra)."""
    import statsmodels.api as sm  # type: ignore[import-not-found]

    family_name = str(data["family"])
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    X_c = sm.add_constant(X)

    if family_name == "linear":
        fam = sm.families.Gaussian()
    elif family_name == "logistic":
        fam = sm.families.Binomial()
    elif family_name == "poisson":
        fam = sm.families.Poisson()
    elif family_name == "negbin":
        # statsmodels NegativeBinomial with alpha=1/r (dispersion)
        fam = sm.families.NegativeBinomial(alpha=1.0 / 5.0)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    model = sm.GLM(y, X_c, family=fam)
    res = model.fit(maxiter=200)
    coef = [float(v) for v in np.asarray(res.params).ravel()]
    llf = float(res.llf)
    return coef, llf, {"backend": "statsmodels", "converged": bool(res.converged)}


def _baseline_sklearn(data: dict[str, Any]) -> tuple[list[float], Optional[float], dict[str, Any]]:
    """Fit with scikit-learn (linear/logistic only), return (coefficients, None, extra)."""
    family_name = str(data["family"])
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"], dtype=float)

    if family_name == "linear":
        from sklearn.linear_model import LinearRegression  # type: ignore[import-not-found]
        lr = LinearRegression(fit_intercept=True)
        lr.fit(X, y)
        coef = [float(lr.intercept_)] + [float(c) for c in lr.coef_]
        return coef, None, {"backend": "sklearn"}
    elif family_name == "logistic":
        from sklearn.linear_model import LogisticRegression  # type: ignore[import-not-found]
        lr = LogisticRegression(solver="lbfgs", max_iter=1000, C=1e10, fit_intercept=True)
        lr.fit(X, y)
        coef = [float(lr.intercept_[0])] + [float(c) for c in lr.coef_[0]]
        return coef, None, {"backend": "sklearn"}
    else:
        raise ValueError(f"sklearn does not support family: {family_name}")


def _baseline_glum(data: dict[str, Any]) -> tuple[list[float], Optional[float], dict[str, Any]]:
    """Fit with glum, return (coefficients, None, extra)."""
    from glum import GeneralizedLinearRegressor  # type: ignore[import-not-found]

    family_name = str(data["family"])
    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"], dtype=float)

    glum_family_map = {
        "linear": "normal",
        "logistic": "binomial",
        "poisson": "poisson",
        "negbin": "negative.binomial",
    }
    fam = glum_family_map.get(family_name)
    if fam is None:
        raise ValueError(f"glum does not support family: {family_name}")

    m = GeneralizedLinearRegressor(family=fam, fit_intercept=True, alpha=0)
    m.fit(X, y)
    coef = [float(m.intercept_)] + [float(c) for c in m.coef_]
    return coef, None, {"backend": "glum"}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="GLM benchmark single-case runner")
    ap.add_argument("--case", required=True, help="Case ID (e.g. linear_10k)")
    ap.add_argument("--family", required=True, choices=["linear", "logistic", "poisson", "negbin"])
    ap.add_argument("--n", required=True, type=int, help="Number of observations")
    ap.add_argument("--p", type=int, default=10, help="Number of covariates")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", type=int, default=50, help="Number of timing repetitions")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    family = str(args.family)
    n = int(args.n)
    p = int(args.p)
    seed = int(args.seed)
    repeat = int(args.repeat)

    # Dataset spec and generation
    spec = {"family": family, "n": n, "p": p, "seed": seed}
    data = gen_glm_data(family, n, p, seed)
    dataset = {
        "id": f"generated:glm:{args.case}",
        "sha256": sha256_json_obj(spec),
        "spec": spec,
    }

    # Detect available competitors
    has_sm, sm_v = _maybe_import("statsmodels")
    has_sklearn, sklearn_v = _maybe_import("sklearn")
    has_glum, glum_v = _maybe_import("glum")

    status = "ok"
    reason: Optional[str] = None

    # --------------- NextStat fit ---------------
    try:
        coef_ns, nll_ns, converged_ns, extra_ns = _fit_ns(data)
    except Exception as e:
        obj = {
            "schema_version": "nextstat.glm_benchmark_result.v1",
            "suite": "glm",
            "case": str(args.case),
            "deterministic": bool(args.deterministic),
            "status": "failed",
            "reason": f"nextstat_error:{type(e).__name__}:{e}",
            "meta": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "nextstat_version": nextstat.__version__,
                "numpy_version": np.__version__,
                "statsmodels_version": sm_v,
                "sklearn_version": sklearn_v,
                "glum_version": glum_v,
            },
            "dataset": dataset,
            "config": {"family": family, "n": n, "p": p},
            "parity": {},
            "timing": {
                "wall_time_s": {"min": 0.0, "median": 0.0, "p95": 0.0},
                "raw": {"repeat": repeat, "policy": "median", "runs_s": []},
            },
            "results": {"nextstat": {"coef": [], "nll": None, "converged": False, "extra": {}}, "baselines": {}},
        }
        out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        return 2

    # --------------- Timing ---------------
    runs_s: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _fit_ns(data)
        runs_s.append(float(time.perf_counter() - t0))

    timing = {
        "wall_time_s": _summary(runs_s),
        "raw": {"repeat": repeat, "policy": "median", "runs_s": [float(x) for x in runs_s]},
    }

    # --------------- Baselines & Parity ---------------
    baselines: dict[str, Any] = {}
    parity: dict[str, Any] = {}
    timing_baselines: dict[str, Any] = {}

    # statsmodels
    try:
        if has_sm:
            coef_sm, llf_sm, extra_sm = _baseline_statsmodels(data)
            baselines["statsmodels"] = {"coef": coef_sm, "llf": llf_sm, "extra": extra_sm}
            runs_sm: list[float] = []
            for _ in range(repeat):
                t0 = time.perf_counter()
                _baseline_statsmodels(data)
                runs_sm.append(float(time.perf_counter() - t0))
            timing_baselines["statsmodels"] = {"wall_time_s": _summary(runs_sm), "raw": {"repeat": repeat, "runs_s": runs_sm}}
            coef_abs, coef_rel = _max_abs_rel_diff(coef_ns, coef_sm)
            # Compare log-likelihood: NS returns NLL, statsmodels returns LLF.
            # NLL = -LLF, so compare -nll_ns vs llf_sm.
            llf_diff = abs(-nll_ns - llf_sm) if llf_sm is not None else None
            parity["statsmodels"] = {
                "status": "ok" if coef_abs is not None else "warn",
                "reference": {"name": "statsmodels", "version": str(sm_v or "")},
                "metrics": {
                    "coef_max_abs_diff": coef_abs,
                    "coef_max_rel_diff": coef_rel,
                    "llf_abs_diff": float(llf_diff) if llf_diff is not None else None,
                },
            }
    except Exception as e:
        parity["statsmodels"] = {
            "status": "warn",
            "reference": {"name": "statsmodels", "version": str(sm_v or "")},
            "metrics": {},
            "error": f"{type(e).__name__}:{e}",
        }

    # scikit-learn (linear + logistic only)
    if family in ("linear", "logistic"):
        try:
            if has_sklearn:
                coef_sk, _, extra_sk = _baseline_sklearn(data)
                baselines["sklearn"] = {"coef": coef_sk, "extra": extra_sk}
                runs_sk: list[float] = []
                for _ in range(repeat):
                    t0 = time.perf_counter()
                    _baseline_sklearn(data)
                    runs_sk.append(float(time.perf_counter() - t0))
                timing_baselines["sklearn"] = {"wall_time_s": _summary(runs_sk), "raw": {"repeat": repeat, "runs_s": runs_sk}}
                coef_abs, coef_rel = _max_abs_rel_diff(coef_ns, coef_sk)
                parity["sklearn"] = {
                    "status": "ok" if coef_abs is not None else "warn",
                    "reference": {"name": "sklearn", "version": str(sklearn_v or "")},
                    "metrics": {
                        "coef_max_abs_diff": coef_abs,
                        "coef_max_rel_diff": coef_rel,
                    },
                }
        except Exception as e:
            parity["sklearn"] = {
                "status": "warn",
                "reference": {"name": "sklearn", "version": str(sklearn_v or "")},
                "metrics": {},
                "error": f"{type(e).__name__}:{e}",
            }

    # glum
    try:
        if has_glum:
            coef_gl, _, extra_gl = _baseline_glum(data)
            baselines["glum"] = {"coef": coef_gl, "extra": extra_gl}
            runs_gl: list[float] = []
            for _ in range(repeat):
                t0 = time.perf_counter()
                _baseline_glum(data)
                runs_gl.append(float(time.perf_counter() - t0))
            timing_baselines["glum"] = {"wall_time_s": _summary(runs_gl), "raw": {"repeat": repeat, "runs_s": runs_gl}}
            coef_abs, coef_rel = _max_abs_rel_diff(coef_ns, coef_gl)
            parity["glum"] = {
                "status": "ok" if coef_abs is not None else "warn",
                "reference": {"name": "glum", "version": str(glum_v or "")},
                "metrics": {
                    "coef_max_abs_diff": coef_abs,
                    "coef_max_rel_diff": coef_rel,
                },
            }
    except Exception as e:
        parity["glum"] = {
            "status": "warn",
            "reference": {"name": "glum", "version": str(glum_v or "")},
            "metrics": {},
            "error": f"{type(e).__name__}:{e}",
        }

    # Aggregate parity status
    parity_statuses = [str(v.get("status", "skipped")) for v in parity.values()]
    if "warn" in parity_statuses:
        status = "warn"
        reason = "one_or_more_baseline_warnings"

    cfg = {
        "family": family,
        "n": n,
        "p": p,
        "n_coefficients": len(coef_ns),
    }

    obj = {
        "schema_version": "nextstat.glm_benchmark_result.v1",
        "suite": "glm",
        "case": str(args.case),
        "deterministic": bool(args.deterministic),
        "status": status,
        "reason": reason,
        "meta": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "nextstat_version": nextstat.__version__,
            "numpy_version": np.__version__,
            "statsmodels_version": sm_v,
            "sklearn_version": sklearn_v,
            "glum_version": glum_v,
        },
        "dataset": dataset,
        "config": cfg,
        "parity": parity,
        "timing": timing,
        "timing_baselines": timing_baselines,
        "results": {
            "nextstat": {
                "coef": coef_ns,
                "nll": nll_ns,
                "converged": converged_ns,
                "extra": extra_ns,
            },
            "baselines": baselines,
        },
    }
    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return 0 if status != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
