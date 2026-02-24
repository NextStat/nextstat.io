#!/usr/bin/env python3
"""Competing Risks (survival analysis) benchmark runner.

Backends:
- `nextstat` (always available) -- cumulative_incidence, gray_test, fine_gray_fit
- `lifelines` (optional) -- AalenJohansenFitter for CIF parity
- `R cmprsk` (optional, via subprocess) -- CIF + Fine-Gray parity

Cases:
- cif_1k / cif_10k:               Cumulative Incidence Function estimation
- gray_1k / gray_10k:             Gray's test for group comparison
- fine_gray_1k_p5 / fine_gray_10k_p10: Fine-Gray subdistribution hazard regression
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.bench_env import collect_environment


# ---------------------------------------------------------------------------
# Utility helpers (same pattern as garch_family/run.py)
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


def _as_f64_list(x: Any) -> list[float]:
    """Normalize JSON-ish number/list into a list[float]."""
    if x is None:
        return []
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, list):
        out: list[float] = []
        for v in x:
            try:
                out.append(float(v))
            except Exception:
                pass
        return out
    return []


def _maybe_import(name: str) -> tuple[bool, Optional[str]]:
    try:
        __import__(name)
        mod = sys.modules.get(name)
        return True, str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return False, None


# ---------------------------------------------------------------------------
# Graceful import of nextstat._core competing-risks functions
# ---------------------------------------------------------------------------

_HAS_CIF = False
_HAS_GRAY = False
_HAS_FINE_GRAY = False

try:
    from nextstat._core import cumulative_incidence as _ns_cumulative_incidence
    _HAS_CIF = True
except (ImportError, AttributeError):
    _ns_cumulative_incidence = None  # type: ignore[assignment]

try:
    from nextstat._core import gray_test as _ns_gray_test
    _HAS_GRAY = True
except (ImportError, AttributeError):
    _ns_gray_test = None  # type: ignore[assignment]

try:
    from nextstat._core import fine_gray_fit as _ns_fine_gray_fit
    _HAS_FINE_GRAY = True
except (ImportError, AttributeError):
    _ns_fine_gray_fit = None  # type: ignore[assignment]

# Lifelines (optional competitor for CIF)
_HAS_LIFELINES = False
_LIFELINES_VERSION: Optional[str] = None
try:
    from lifelines import AalenJohansenFitter  # type: ignore[import-not-found]
    import lifelines  # type: ignore[import-not-found]
    _HAS_LIFELINES = True
    _LIFELINES_VERSION = str(lifelines.__version__)
except (ImportError, AttributeError):
    AalenJohansenFitter = None  # type: ignore[assignment,misc]

# R cmprsk (optional competitor via subprocess)
_HAS_R_CMPRSK = False
_R_VERSION: Optional[str] = None


def _check_r_cmprsk() -> tuple[bool, Optional[str]]:
    """Check if R and cmprsk package are available."""
    try:
        r = subprocess.run(
            [
                "Rscript",
                "-e",
                "cat(paste(R.version$major, R.version$minor, sep='.')); cat('|'); cat(as.character(packageVersion('cmprsk')))",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and "|" in r.stdout:
            parts = r.stdout.strip().split("|")
            return True, parts[0].strip()
        return False, None
    except Exception:
        return False, None


_HAS_R_CMPRSK, _R_VERSION = _check_r_cmprsk()

# NextStat version
try:
    import nextstat
    _NS_VERSION = str(nextstat.__version__)
except ImportError:
    _NS_VERSION = "unavailable"


# ---------------------------------------------------------------------------
# Data generation (deterministic, seeded)
# ---------------------------------------------------------------------------

def gen_competing_risks_data(
    n: int, seed: int, n_groups: int = 2
) -> dict[str, Any]:
    """Generate competing risks survival data.

    Event types: 0=censored (~30%), 1=event of interest (~40%), 2=competing event (~30%).
    Groups: binary (0/1) with different hazard rates.
    """
    rng = np.random.default_rng(int(seed))

    # Group assignments
    group = rng.integers(0, n_groups, size=n)

    # Generate times from exponential distributions with group-dependent hazards
    # Group 0: baseline hazard
    # Group 1: higher hazard for event 1
    times = np.zeros(n)
    events = np.zeros(n, dtype=int)

    for i in range(n):
        g = int(group[i])
        # Hazard rates per event type, varying by group
        lambda_censor = 0.3
        lambda_event1 = 0.5 if g == 0 else 0.7
        lambda_event2 = 0.3 if g == 0 else 0.25

        # Generate competing event times
        t_censor = rng.exponential(1.0 / lambda_censor)
        t_event1 = rng.exponential(1.0 / lambda_event1)
        t_event2 = rng.exponential(1.0 / lambda_event2)

        # Observed time is minimum; event is the corresponding type
        t_min = min(t_censor, t_event1, t_event2)
        times[i] = t_min
        if t_min == t_censor:
            events[i] = 0
        elif t_min == t_event1:
            events[i] = 1
        else:
            events[i] = 2

    return {
        "time": times.tolist(),
        "event": [int(e) for e in events],
        "group": [int(g) for g in group],
        "n": int(n),
        "n_groups": int(n_groups),
    }


def gen_fine_gray_data(
    n: int, p: int, seed: int
) -> dict[str, Any]:
    """Generate competing risks data with covariates for Fine-Gray regression.

    Event types: 0=censored (~30%), 1=event of interest (~40%), 2=competing event (~30%).
    Covariates: p columns of standard normal.
    True coefficients: beta_j = 0.5 * (-1)^j for j=0..p-1.
    """
    rng = np.random.default_rng(int(seed))

    x = rng.standard_normal((n, p))
    # True coefficients
    true_beta = np.array([0.5 * ((-1) ** j) for j in range(p)])

    # Linear predictor
    lp = x @ true_beta

    times = np.zeros(n)
    events = np.zeros(n, dtype=int)

    for i in range(n):
        # Hazard rates modulated by linear predictor
        lambda_censor = 0.3
        lambda_event1 = 0.5 * np.exp(lp[i])
        lambda_event2 = 0.3

        t_censor = rng.exponential(1.0 / lambda_censor)
        t_event1 = rng.exponential(1.0 / max(lambda_event1, 1e-10))
        t_event2 = rng.exponential(1.0 / lambda_event2)

        t_min = min(t_censor, t_event1, t_event2)
        times[i] = t_min
        if t_min == t_censor:
            events[i] = 0
        elif t_min == t_event1:
            events[i] = 1
        else:
            events[i] = 2

    return {
        "time": times.tolist(),
        "event": [int(e) for e in events],
        "x": x.tolist(),
        "n": int(n),
        "p": int(p),
        "true_beta": true_beta.tolist(),
    }


# ---------------------------------------------------------------------------
# NextStat functions
# ---------------------------------------------------------------------------

def _run_cif_ns(data: dict[str, Any]) -> dict[str, Any]:
    """Run NextStat cumulative_incidence.

    API: cumulative_incidence(times, events, target_cause, *, conf_level=0.95)
    No ``groups`` parameter -- call once per group to compare.
    """
    if not _HAS_CIF:
        raise ImportError("nextstat._core.cumulative_incidence not available")

    # Compute CIF for each group separately (target_cause=1)
    groups = data["group"]
    unique_groups = sorted(set(groups))
    per_group: dict[str, dict[str, Any]] = {}
    for g in unique_groups:
        idx = [i for i, gg in enumerate(groups) if gg == g]
        g_times = [data["time"][i] for i in idx]
        g_events = [data["event"][i] for i in idx]
        result = _ns_cumulative_incidence(  # type: ignore[misc]
            g_times, g_events, 1, conf_level=0.95,
        )
        per_group[f"group_{g}"] = {
            "n_times": len(result["times"]) if "times" in result else 0,
            "n_events": int(result.get("n_events", 0)),
            "n_at_risk": int(result.get("n_at_risk", 0)) if "n_at_risk" in result else len(g_times),
        }

    # Also compute the overall CIF (all subjects)
    result_all = _ns_cumulative_incidence(  # type: ignore[misc]
        data["time"], data["event"], 1, conf_level=0.95,
    )
    return {
        "n_times": len(result_all["times"]) if "times" in result_all else 0,
        "n_events": int(result_all.get("n_events", 0)),
        "n_at_risk": int(result_all.get("n_at_risk", 0)) if "n_at_risk" in result_all else len(data["time"]),
        "per_group": per_group,
    }


def _run_gray_ns(data: dict[str, Any]) -> dict[str, Any]:
    """Run NextStat gray_test.

    API: gray_test(times, events, groups, target_cause)
    """
    if not _HAS_GRAY:
        raise ImportError("nextstat._core.gray_test not available")
    result = _ns_gray_test(  # type: ignore[misc]
        data["time"], data["event"], data["group"], 1,
    )
    return {
        "test_statistic": float(result["statistic"]),
        "p_value": float(result["p_value"]),
        "df": int(result.get("df", 1)),
    }


def _run_fine_gray_ns(data: dict[str, Any]) -> dict[str, Any]:
    """Run NextStat fine_gray_fit.

    API: fine_gray_fit(times, events, x, p, target_cause)
    ``x`` is a flattened list (n*p floats), ``p`` is the number of covariates.
    """
    if not _HAS_FINE_GRAY:
        raise ImportError("nextstat._core.fine_gray_fit not available")

    # Flatten the n x p covariate matrix into a single list
    x_nested: list[list[float]] = data["x"]
    x_flat: list[float] = [float(v) for row in x_nested for v in row]
    n_covariates: int = data["p"]

    result = _ns_fine_gray_fit(  # type: ignore[misc]
        data["time"], data["event"], x_flat, n_covariates, 1,
    )
    return {
        "coefficients": [float(c) for c in result["coefficients"]],
        "std_errors": [float(s) for s in result["se"]],
        "z_values": [float(z) for z in result["z"]],
        "p_values": [float(p_) for p_ in result["p_values"]],
        "n_events": int(result.get("n_events", 0)),
        "n_subjects": int(result.get("n", 0)),
        "log_likelihood": float(result.get("log_likelihood", 0.0)),
    }


_NS_RUN_DISPATCH = {
    "cif": _run_cif_ns,
    "gray": _run_gray_ns,
    "fine_gray": _run_fine_gray_ns,
}

_NS_AVAILABLE = {
    "cif": _HAS_CIF,
    "gray": _HAS_GRAY,
    "fine_gray": _HAS_FINE_GRAY,
}


# ---------------------------------------------------------------------------
# Baselines: lifelines (CIF only)
# ---------------------------------------------------------------------------

def _baseline_lifelines_cif(data: dict[str, Any]) -> dict[str, Any]:
    """Run lifelines AalenJohansenFitter for CIF."""
    import pandas as pd  # type: ignore[import-not-found]

    aj = AalenJohansenFitter()
    df = pd.DataFrame({
        "time": data["time"],
        "event": data["event"],
    })
    aj.fit(df["time"], df["event"], event_of_interest=1)
    cif = aj.cumulative_density_
    return {
        "backend": "lifelines",
        "n_times": len(cif),
    }


# ---------------------------------------------------------------------------
# Baselines: R cmprsk (CIF + Fine-Gray)
# ---------------------------------------------------------------------------

def _baseline_r_cif(data: dict[str, Any], repeat: int = 1) -> dict[str, Any]:
    """Run R cmprsk::cuminc for CIF/Gray baseline.

    Important: we do timing *inside* the single R process to avoid attributing
    repeated R startup/package-load overhead to the baseline.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "time": data["time"],
            "event": data["event"],
            "group": data["group"],
        }, f)
        data_path = f.name

    r_script = f"""
library(cmprsk)
library(jsonlite)
d <- fromJSON("{data_path}")
n_rep <- {int(max(1, int(repeat)))}
runs <- numeric(n_rep)
result <- NULL
for (i in seq_len(n_rep)) {{
  tt <- system.time({{ result <- cuminc(d$time, d$event, d$group) }})[["elapsed"]]
  runs[i] <- tt
}}
cat(toJSON(list(
    n_times = length(result[[1]]$time),
    test_statistic = if (!is.null(result$Tests)) result$Tests[1, 1] else NA,
    runs_s = as.list(as.numeric(runs))
), auto_unbox=TRUE))
"""
    try:
        r = subprocess.run(
            ["Rscript", "-e", r_script],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            raise RuntimeError(f"R cmprsk cuminc failed: {r.stderr[:200]}")
        result = json.loads(r.stdout.strip())
        return {"backend": "R_cmprsk", **result}
    finally:
        Path(data_path).unlink(missing_ok=True)


def _baseline_r_fine_gray(data: dict[str, Any], repeat: int = 1) -> dict[str, Any]:
    """Run R cmprsk::crr for Fine-Gray regression.

    Important: we do timing *inside* the single R process to avoid attributing
    repeated R startup/package-load overhead to the baseline.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "time": data["time"],
            "event": data["event"],
            "x": data["x"],
        }, f)
        data_path = f.name

    r_script = f"""
library(cmprsk)
library(jsonlite)
d <- fromJSON("{data_path}")
# jsonlite::fromJSON() will typically simplify a nested array into a matrix/data.frame
# already, so `d$x` is usually NOT a list. Handle all three cases robustly.
if (is.data.frame(d$x) || is.matrix(d$x)) {{
  x_mat <- as.matrix(d$x)
}} else {{
  x_mat <- as.matrix(do.call(rbind, d$x))
}}
storage.mode(x_mat) <- "double"
n_rep <- {int(max(1, int(repeat)))}
runs <- numeric(n_rep)
result <- NULL
for (i in seq_len(n_rep)) {{
  tt <- system.time({{ result <- crr(d$time, d$event, x_mat, failcode=1) }})[["elapsed"]]
  runs[i] <- tt
}}
cat(toJSON(list(
    coefficients = as.numeric(result$coef),
    std_errors = as.numeric(sqrt(diag(result$var))),
    converged = result$converged,
    runs_s = as.list(as.numeric(runs))
), auto_unbox=TRUE))
"""
    try:
        # Fine-Gray baseline in R can be very slow at n=10k+. Use a higher timeout
        # so the baseline is available for speedup comparisons.
        n = int(data.get("n", 0) or 0)
        p = int(data.get("p", 0) or 0)
        timeout_s = 120
        if n >= 10_000 or p >= 10:
            timeout_s = 900
        timeout_s = int(timeout_s * max(1, int(repeat)))

        r = subprocess.run(
            ["Rscript", "-e", r_script],
            capture_output=True, text=True, timeout=timeout_s,
        )
        if r.returncode != 0:
            raise RuntimeError(f"R cmprsk crr failed: {r.stderr[:200]}")
        result = json.loads(r.stdout.strip())
        return {"backend": "R_cmprsk", **result}
    finally:
        Path(data_path).unlink(missing_ok=True)


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


def _compute_parity_fine_gray(ns_result: dict, bl: dict) -> dict[str, Any]:
    """Parity metrics for Fine-Gray: coefficient-wise comparison."""
    ns_coefs = ns_result.get("coefficients", [])
    bl_coefs = bl.get("coefficients", [])
    if not ns_coefs or not bl_coefs or len(ns_coefs) != len(bl_coefs):
        return {"status": "warn", "reason": "coefficient_count_mismatch"}

    max_rel = 0.0
    coef_parity = []
    for i, (nc, bc) in enumerate(zip(ns_coefs, bl_coefs)):
        abs_d = _scalar_diff(nc, bc)
        rel_d = _scalar_rel_diff(nc, bc)
        coef_parity.append({
            f"coef_{i}_abs_diff": abs_d,
            f"coef_{i}_rel_diff": rel_d,
        })
        if rel_d is not None and rel_d > max_rel:
            max_rel = rel_d

    status = "ok" if max_rel < 0.15 else "warn"
    return {"status": status, "max_rel_diff": float(max_rel), "coefficients": coef_parity}


def _compute_parity_gray_vs_r(ns_result: dict, bl: dict) -> dict[str, Any]:
    """Parity metrics for Gray's test vs R cmprsk."""
    ns_stat = ns_result.get("test_statistic")
    bl_stat = bl.get("test_statistic")
    return {
        "test_statistic_abs_diff": _scalar_diff(ns_stat, bl_stat),
        "test_statistic_rel_diff": _scalar_rel_diff(ns_stat, bl_stat),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Competing Risks benchmark runner")
    ap.add_argument("--case", required=True, help="Case id for reporting.")
    ap.add_argument("--kind", required=True, choices=["cif", "gray", "fine_gray"])
    ap.add_argument("--n", type=int, required=True, help="Number of subjects.")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--p", type=int, default=5, help="Number of covariates (Fine-Gray only).")
    ap.add_argument(
        "--baseline-repeat", type=int, default=1,
        help="Repeat count for baseline timing loops.",
    )
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kind = str(args.kind)
    n = int(args.n)
    p = int(args.p)
    seed = int(args.seed)
    repeat = int(args.repeat)
    baseline_repeat = max(1, int(args.baseline_repeat))

    meta: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "nextstat_version": _NS_VERSION,
        "numpy_version": np.__version__,
        "lifelines_version": _LIFELINES_VERSION,
        "r_version": _R_VERSION,
        "has_r_cmprsk": _HAS_R_CMPRSK,
        "ns_cif_available": _HAS_CIF,
        "ns_gray_available": _HAS_GRAY,
        "ns_fine_gray_available": _HAS_FINE_GRAY,
    }

    # -- Generate data ---------------------------------------------------------
    spec: dict[str, Any] = {"kind": kind, "n": n, "seed": seed}
    if kind == "fine_gray":
        data = gen_fine_gray_data(n, p, seed)
        spec["p"] = p
    else:
        data = gen_competing_risks_data(n, seed, n_groups=2)
        spec["n_groups"] = 2

    dataset = {
        "id": f"generated:competing_risks:{args.case}",
        "sha256": sha256_json_obj(spec),
        "spec": spec,
    }

    cfg: dict[str, Any] = {"kind": kind, "n": n, "seed": seed}
    if kind == "fine_gray":
        cfg["p"] = p
        cfg["max_iter"] = 100
        cfg["tol"] = 1e-8

    status = "ok"
    reason: Optional[str] = None

    # -- Check NS availability -------------------------------------------------
    if not _NS_AVAILABLE[kind]:
        func_name = {
            "cif": "cumulative_incidence",
            "gray": "gray_test",
            "fine_gray": "fine_gray_fit",
        }[kind]
        obj = {
            "schema_version": "nextstat.competing_risks_benchmark_result.v1",
            "suite": "competing_risks",
            "case": str(args.case),
            "deterministic": bool(args.deterministic),
            "environment": collect_environment(),
            "status": "skipped",
            "reason": f"nextstat._core.{func_name} not available (ImportError)",
            "meta": meta,
            "dataset": dataset,
            "config": cfg,
            "parity": {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}},
            "timing": {
                "wall_time_s": {"min": 0.0, "median": 0.0, "p95": 0.0},
                "raw": {"repeat": repeat, "policy": "median", "runs_s": []},
            },
            "timing_baseline": {},
            "results": {"nextstat": {}, "baseline": None},
        }
        out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        print(f"  [{args.case}] SKIPPED -- {func_name} not in nextstat._core")
        return 0

    # -- NextStat run ----------------------------------------------------------
    ns_run_fn = _NS_RUN_DISPATCH[kind]
    try:
        ns_result = ns_run_fn(data)
    except Exception as e:
        obj = {
            "schema_version": "nextstat.competing_risks_benchmark_result.v1",
            "suite": "competing_risks",
            "case": str(args.case),
            "deterministic": bool(args.deterministic),
            "environment": collect_environment(),
            "status": "failed",
            "reason": f"nextstat_error:{type(e).__name__}:{e}",
            "meta": meta,
            "dataset": dataset,
            "config": cfg,
            "parity": {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}},
            "timing": {
                "wall_time_s": {"min": 0.0, "median": 0.0, "p95": 0.0},
                "raw": {"repeat": repeat, "policy": "median", "runs_s": []},
            },
            "timing_baseline": {},
            "results": {"nextstat": {}, "baseline": None},
        }
        out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        return 2

    # -- Timing (NextStat) -----------------------------------------------------
    runs_s: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        ns_run_fn(data)
        runs_s.append(float(time.perf_counter() - t0))

    timing = {
        "wall_time_s": _summary(runs_s),
        "raw": {"repeat": repeat, "policy": "median", "runs_s": [float(x) for x in runs_s]},
    }

    # -- Baselines -------------------------------------------------------------
    baseline_obj: Optional[dict[str, Any]] = None
    parity: dict[str, Any] = {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}}
    timing_baseline: dict[str, Any] = {}

    try:
        if args.skip_baselines:
            raise RuntimeError("baseline_skipped")

        if kind == "cif":
            # Try lifelines first, then R cmprsk
            if _HAS_LIFELINES:
                bl = _baseline_lifelines_cif(data)
                baseline_obj = bl

                # Baseline timing
                runs_bl: list[float] = []
                for _ in range(baseline_repeat):
                    t0 = time.perf_counter()
                    _baseline_lifelines_cif(data)
                    runs_bl.append(float(time.perf_counter() - t0))
                timing_baseline = {
                    "name": "lifelines",
                    "wall_time_s": _summary(runs_bl),
                    "raw": {"repeat": baseline_repeat, "runs_s": runs_bl},
                }
                parity = {
                    "status": "ok",
                    "reference": {"name": "lifelines", "version": str(_LIFELINES_VERSION or "")},
                    "metrics": {},
                }
            elif _HAS_R_CMPRSK:
                bl = _baseline_r_cif(data, repeat=baseline_repeat)
                baseline_obj = bl

                runs_bl = _as_f64_list(bl.get("runs_s"))
                timing_baseline = {
                    "name": "R_cmprsk",
                    "wall_time_s": _summary(runs_bl),
                    "raw": {"repeat": baseline_repeat, "runs_s": runs_bl},
                }
                parity = {
                    "status": "ok",
                    "reference": {"name": "R_cmprsk", "version": str(_R_VERSION or "")},
                    "metrics": {},
                }

        elif kind == "gray":
            # Gray's test -- R cmprsk as baseline
            if _HAS_R_CMPRSK:
                bl = _baseline_r_cif(data, repeat=baseline_repeat)  # cuminc returns Gray's test statistic
                baseline_obj = bl

                runs_bl = _as_f64_list(bl.get("runs_s"))
                timing_baseline = {
                    "name": "R_cmprsk",
                    "wall_time_s": _summary(runs_bl),
                    "raw": {"repeat": baseline_repeat, "runs_s": runs_bl},
                }

                metrics = _compute_parity_gray_vs_r(ns_result, bl)
                parity = {
                    "status": "ok",
                    "reference": {"name": "R_cmprsk", "version": str(_R_VERSION or "")},
                    "metrics": metrics,
                }

        elif kind == "fine_gray":
            # Fine-Gray -- R cmprsk::crr as baseline
            if _HAS_R_CMPRSK:
                bl = _baseline_r_fine_gray(data, repeat=baseline_repeat)
                baseline_obj = bl

                runs_bl = _as_f64_list(bl.get("runs_s"))
                timing_baseline = {
                    "name": "R_cmprsk",
                    "wall_time_s": _summary(runs_bl),
                    "raw": {"repeat": baseline_repeat, "runs_s": runs_bl},
                }

                metrics = _compute_parity_fine_gray(ns_result, bl)
                parity = {
                    "status": metrics.get("status", "warn"),
                    "reference": {"name": "R_cmprsk", "version": str(_R_VERSION or "")},
                    "metrics": metrics,
                }

    except Exception as e:
        if str(e) == "baseline_skipped":
            parity = {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}}
        else:
            status = "warn"
            reason = f"baseline_unavailable:{type(e).__name__}:{e}"
            parity = {"status": "warn", "reference": {"name": "", "version": ""}, "metrics": {}}

    obj = {
        "schema_version": "nextstat.competing_risks_benchmark_result.v1",
        "suite": "competing_risks",
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
