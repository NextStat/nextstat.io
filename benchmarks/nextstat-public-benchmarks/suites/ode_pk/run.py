#!/usr/bin/env python3
"""ODE PK (pharmacokinetic) benchmark runner.

Backends:
- `nextstat` (always available) — nextstat._core.ode_pk_solve / ode_pk_nll_py
- `scipy` (baseline) — scipy.integrate.solve_ivp with equivalent ODE systems

Cases:
- transit_1cpt_sparse:  Transit absorption, 20 time points, single dose
- transit_1cpt_dense:   Transit absorption, 200 time points, multiple doses
- mm_1cpt_sparse:       Michaelis-Menten 1-cpt, 20 time points
- mm_1cpt_dense:        Michaelis-Menten 1-cpt, 200 time points
- tmdd_sparse:          TMDD, 30 time points
- tmdd_dense:           TMDD, 300 time points, multiple doses
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
# Graceful import of nextstat._core
# ---------------------------------------------------------------------------

_NS_AVAILABLE = False
_NS_IMPORT_ERROR: Optional[str] = None
_HAS_ODE_PK_SOLVE = False
_HAS_ODE_PK_NLL = False

try:
    import nextstat._core as _ns  # type: ignore
    _NS_AVAILABLE = True
except ImportError as e:
    _NS_IMPORT_ERROR = f"ImportError: {e}"
except Exception as e:
    _NS_IMPORT_ERROR = f"{type(e).__name__}: {e}"

if _NS_AVAILABLE:
    try:
        from nextstat._core import ode_pk_solve as _ns_ode_pk_solve  # type: ignore
        _HAS_ODE_PK_SOLVE = True
    except (ImportError, AttributeError):
        _ns_ode_pk_solve = None  # type: ignore[assignment]

    try:
        from nextstat._core import ode_pk_nll_py as _ns_ode_pk_nll  # type: ignore
        _HAS_ODE_PK_NLL = True
    except (ImportError, AttributeError):
        _ns_ode_pk_nll = None  # type: ignore[assignment]

try:
    import nextstat
    _NS_VERSION = str(nextstat.__version__)
except ImportError:
    _NS_VERSION = "unavailable"


# ---------------------------------------------------------------------------
# Data generation (deterministic, seeded)
# ---------------------------------------------------------------------------

# Actual Rust API parameter specs:
#   transit_1cpt: [n_transit, ktr, cl, v1]
#   mm_1cpt: [vmax, km, v1]
#   tmdd: [kon, koff, kel, ksyn, kdeg, kint, v]
# sigma is NOT a model param — passed separately to NLL

_TRUE_PARAMS = {
    "transit_1cpt": {
        "n_transit": 3.0, "ktr": 0.8, "cl": 2.0, "v1": 10.0, "sigma": 0.1,
    },
    "mm_1cpt": {
        "vmax": 5.0, "km": 2.0, "v1": 10.0, "sigma": 0.1,
    },
    "tmdd": {
        "kon": 0.1, "koff": 0.01, "kel": 0.05, "ksyn": 1.0, "kdeg": 0.1, "kint": 0.03, "v": 10.0, "sigma": 0.1,
    },
}

# Param ordering for ode_pk_solve (sigma excluded)
_PARAM_NAMES = {
    "transit_1cpt": ["n_transit", "ktr", "cl", "v1"],
    "mm_1cpt": ["vmax", "km", "v1"],
    "tmdd": ["kon", "koff", "kel", "ksyn", "kdeg", "kint", "v"],
}


def _params_list(model_type: str) -> list[float]:
    """Return true params as ordered list for the model type."""
    tp = _TRUE_PARAMS[model_type]
    return [float(tp[k]) for k in _PARAM_NAMES[model_type]]


def gen_transit_1cpt_data(
    seed: int, n_times: int, dose_amount: float, multi_dose: bool = False,
) -> dict[str, Any]:
    """Generate transit absorption 1-compartment PK data."""
    rng = np.random.default_rng(int(seed))
    tp = _TRUE_PARAMS["transit_1cpt"]

    if multi_dose:
        dose_times = [0.0, 12.0, 24.0]
        t_max = 48.0
    else:
        dose_times = [0.0]
        t_max = 24.0

    times = np.linspace(0.5, t_max, n_times).tolist()

    # Simulate with scipy to generate "true" concentrations
    true_conc = _scipy_transit_1cpt_solve(
        tp["cl"], tp["v1"], tp["ktr"], int(tp["n_transit"]),
        times, dose_amount, dose_times,
    )

    # Add proportional noise: obs = true * (1 + sigma * randn)
    sigma = tp["sigma"]
    obs = [max(0.001, c * (1.0 + sigma * rng.normal())) for c in true_conc]

    dose_amounts = [dose_amount] * len(dose_times)
    dose_routes = ["oral"] * len(dose_times)

    return {
        "model_type": "transit_1cpt",
        "obs_times": times,
        "obs_values": obs,
        "true_conc": true_conc,
        "dose_amounts": dose_amounts,
        "dose_times": dose_times,
        "dose_routes": dose_routes,
        "params": _params_list("transit_1cpt"),
        "param_names": _PARAM_NAMES["transit_1cpt"],
        "true_params_dict": dict(tp),
    }


def gen_mm_1cpt_data(
    seed: int, n_times: int, dose_amount: float,
) -> dict[str, Any]:
    """Generate Michaelis-Menten 1-compartment PK data (IV bolus)."""
    rng = np.random.default_rng(int(seed))
    tp = _TRUE_PARAMS["mm_1cpt"]

    dose_times = [0.0]
    times = np.linspace(0.5, 24.0, n_times).tolist()

    true_conc = _scipy_mm_1cpt_solve(
        tp["v1"], tp["vmax"], tp["km"],
        times, dose_amount, dose_times,
    )

    sigma = tp["sigma"]
    obs = [max(0.001, c * (1.0 + sigma * rng.normal())) for c in true_conc]

    dose_amounts = [dose_amount] * len(dose_times)
    dose_routes = ["iv_bolus"] * len(dose_times)

    return {
        "model_type": "mm_1cpt",
        "obs_times": times,
        "obs_values": obs,
        "true_conc": true_conc,
        "dose_amounts": dose_amounts,
        "dose_times": dose_times,
        "dose_routes": dose_routes,
        "params": _params_list("mm_1cpt"),
        "param_names": _PARAM_NAMES["mm_1cpt"],
        "true_params_dict": dict(tp),
    }


def gen_tmdd_data(
    seed: int, n_times: int, dose_amount: float, multi_dose: bool = False,
) -> dict[str, Any]:
    """Generate TMDD PK data (IV bolus)."""
    rng = np.random.default_rng(int(seed))
    tp = _TRUE_PARAMS["tmdd"]

    if multi_dose:
        dose_times = [0.0, 24.0, 48.0]
        t_max = 96.0
    else:
        dose_times = [0.0]
        t_max = 72.0

    times = np.linspace(0.5, t_max, n_times).tolist()

    true_conc = _scipy_tmdd_solve(
        tp["kon"], tp["koff"], tp["kel"], tp["ksyn"], tp["kdeg"], tp["kint"], tp["v"],
        times, dose_amount, dose_times,
    )

    sigma = tp["sigma"]
    obs = [max(0.001, c * (1.0 + sigma * rng.normal())) for c in true_conc]

    dose_amounts = [dose_amount] * len(dose_times)
    dose_routes = ["iv_bolus"] * len(dose_times)

    return {
        "model_type": "tmdd",
        "obs_times": times,
        "obs_values": obs,
        "true_conc": true_conc,
        "dose_amounts": dose_amounts,
        "dose_times": dose_times,
        "dose_routes": dose_routes,
        "params": _params_list("tmdd"),
        "param_names": _PARAM_NAMES["tmdd"],
        "true_params_dict": dict(tp),
    }


# ---------------------------------------------------------------------------
# Scipy ODE solvers (used for data generation AND as competitor baseline)
# ---------------------------------------------------------------------------

def _scipy_transit_1cpt_solve(
    cl: float, v1: float, ktr: float, n_transit: int,
    times: list[float], dose_amount: float, dose_times: list[float],
) -> list[float]:
    """Solve transit absorption 1-cpt model via scipy solve_ivp.

    State vector matches NS Rust: [Transit_0, ..., Transit_{n-1}, Central]
    (n_transit transit compartments + 1 central = n_transit + 1 states).

    Returns AMOUNTS in central compartment (not concentrations), matching NS
    ode_pk_solve which returns raw state[obs_compartment].
    """
    from scipy.integrate import solve_ivp

    ke = cl / v1  # elimination rate
    n_comp = n_transit + 1  # transit compartments + central

    def ode_rhs(t: float, y: np.ndarray) -> np.ndarray:
        dydt = np.zeros(n_comp)
        if n_transit > 0:
            dydt[0] = -ktr * y[0]  # first transit
            for i in range(1, n_transit):
                dydt[i] = ktr * y[i - 1] - ktr * y[i]
        central_idx = n_transit
        inp = ktr * y[n_transit - 1] if n_transit > 0 else 0.0
        dydt[central_idx] = inp - ke * y[central_idx]  # central (amount)
        return dydt

    # Sort dose times, simulate segment by segment
    dose_times_sorted = sorted(dose_times)
    all_times = sorted(set(times))
    y0 = np.zeros(n_comp)

    concentrations = {}
    central_idx = n_transit

    for d_idx, d_time in enumerate(dose_times_sorted):
        # Apply dose at d_time: add to first transit compartment
        y0[0] += dose_amount

        # Find next dose time or end
        if d_idx + 1 < len(dose_times_sorted):
            t_end_seg = dose_times_sorted[d_idx + 1]
        else:
            t_end_seg = max(all_times) + 1.0

        # Observation times in this segment
        seg_times = [t for t in all_times if d_time < t <= t_end_seg]
        if not seg_times:
            # Just advance state to next dose time
            if d_idx + 1 < len(dose_times_sorted):
                sol = solve_ivp(ode_rhs, [d_time, t_end_seg], y0, method="RK45",
                                rtol=1e-8, atol=1e-10, dense_output=True)
                y0 = sol.sol(t_end_seg)
            continue

        t_eval = np.array(seg_times)
        sol = solve_ivp(ode_rhs, [d_time, max(seg_times) + 0.001], y0, method="RK45",
                        t_eval=t_eval, rtol=1e-8, atol=1e-10, dense_output=True)

        for i, t in enumerate(seg_times):
            concentrations[t] = max(0.0, float(sol.y[central_idx, i]))  # amount, not /v1

        # Advance state
        if d_idx + 1 < len(dose_times_sorted):
            y0 = sol.sol(t_end_seg)

    return [concentrations.get(t, 0.0) for t in times]


def _scipy_mm_1cpt_solve(
    v1: float, vmax: float, km: float,
    times: list[float], dose_amount: float, dose_times: list[float],
) -> list[float]:
    """Solve Michaelis-Menten 1-cpt model via scipy solve_ivp (IV bolus).

    Returns AMOUNTS in central compartment (not concentrations), matching NS
    ode_pk_solve which returns raw state[obs_compartment].
    """
    from scipy.integrate import solve_ivp

    def ode_rhs(t: float, y: np.ndarray) -> np.ndarray:
        A = y[0]  # amount in central
        C = A / v1  # concentration
        dAdt = -vmax * C / (km + C)
        return np.array([dAdt])

    dose_times_sorted = sorted(dose_times)
    all_times = sorted(set(times))
    y0 = np.array([0.0])
    concentrations = {}

    for d_idx, d_time in enumerate(dose_times_sorted):
        y0[0] += dose_amount

        if d_idx + 1 < len(dose_times_sorted):
            t_end_seg = dose_times_sorted[d_idx + 1]
        else:
            t_end_seg = max(all_times) + 1.0

        seg_times = [t for t in all_times if d_time < t <= t_end_seg]
        if not seg_times:
            if d_idx + 1 < len(dose_times_sorted):
                sol = solve_ivp(ode_rhs, [d_time, t_end_seg], y0, method="RK45",
                                rtol=1e-8, atol=1e-10, dense_output=True)
                y0 = sol.sol(t_end_seg)
            continue

        t_eval = np.array(seg_times)
        sol = solve_ivp(ode_rhs, [d_time, max(seg_times) + 0.001], y0, method="RK45",
                        t_eval=t_eval, rtol=1e-8, atol=1e-10, dense_output=True)

        for i, t in enumerate(seg_times):
            concentrations[t] = max(0.0, float(sol.y[0, i]))  # amount, not /v1

        if d_idx + 1 < len(dose_times_sorted):
            y0 = sol.sol(t_end_seg)

    return [concentrations.get(t, 0.0) for t in times]


def _scipy_tmdd_solve(
    kon: float, koff: float, kel: float, ksyn: float, kdeg: float, kint: float, v: float,
    times: list[float], dose_amount: float, dose_times: list[float],
) -> list[float]:
    """Solve TMDD model via scipy solve_ivp (IV bolus).

    State vector matches NS Rust: [L, R, LR] — all CONCENTRATIONS.
    Dose is added directly to L (concentration), matching NS behavior.
    R0 = ksyn/kdeg (steady-state receptor concentration).
    """
    from scipy.integrate import solve_ivp

    R0 = ksyn / kdeg  # steady-state receptor concentration

    def ode_rhs(t: float, y: np.ndarray) -> np.ndarray:
        l = max(y[0], 0.0)   # free drug concentration
        r = max(y[1], 0.0)   # free receptor concentration
        lr = max(y[2], 0.0)  # complex concentration

        binding = kon * l * r
        dissociation = koff * lr

        dl_dt = -binding + dissociation - kel * l
        dr_dt = ksyn - kdeg * r - binding + dissociation
        dlr_dt = binding - dissociation - kint * lr

        return np.array([dl_dt, dr_dt, dlr_dt])

    dose_times_sorted = sorted(dose_times)
    all_times = sorted(set(times))
    # Initial state: all zeros (matching NS Rust OdePkSolver which initializes y=vec![0.0; dim])
    # Note: NS does NOT pre-initialize R to R0=ksyn/kdeg. The receptor reaches
    # baseline organically through the ksyn synthesis term.
    y0 = np.array([0.0, 0.0, 0.0])
    concentrations = {}

    for d_idx, d_time in enumerate(dose_times_sorted):
        y0[0] += dose_amount  # IV bolus: add to concentration (matching NS)

        if d_idx + 1 < len(dose_times_sorted):
            t_end_seg = dose_times_sorted[d_idx + 1]
        else:
            t_end_seg = max(all_times) + 1.0

        seg_times = [t for t in all_times if d_time < t <= t_end_seg]
        if not seg_times:
            if d_idx + 1 < len(dose_times_sorted):
                sol = solve_ivp(ode_rhs, [d_time, t_end_seg], y0, method="RK45",
                                rtol=1e-8, atol=1e-10, dense_output=True)
                y0 = sol.sol(t_end_seg)
            continue

        t_eval = np.array(seg_times)
        sol = solve_ivp(ode_rhs, [d_time, max(seg_times) + 0.001], y0, method="RK45",
                        t_eval=t_eval, rtol=1e-8, atol=1e-10, dense_output=True)

        for i, t in enumerate(seg_times):
            concentrations[t] = max(0.0, float(sol.y[0, i]))  # already concentration

        if d_idx + 1 < len(dose_times_sorted):
            y0 = sol.sol(t_end_seg)

    return [concentrations.get(t, 0.0) for t in times]


# ---------------------------------------------------------------------------
# Data generation dispatch
# ---------------------------------------------------------------------------

_DATA_GEN_DISPATCH = {
    "transit_1cpt_sparse": lambda seed: gen_transit_1cpt_data(seed, n_times=20, dose_amount=100.0, multi_dose=False),
    "transit_1cpt_dense": lambda seed: gen_transit_1cpt_data(seed, n_times=200, dose_amount=100.0, multi_dose=True),
    "mm_1cpt_sparse": lambda seed: gen_mm_1cpt_data(seed, n_times=20, dose_amount=500.0),
    "mm_1cpt_dense": lambda seed: gen_mm_1cpt_data(seed, n_times=200, dose_amount=500.0),
    "tmdd_sparse": lambda seed: gen_tmdd_data(seed, n_times=30, dose_amount=100.0, multi_dose=False),
    "tmdd_dense": lambda seed: gen_tmdd_data(seed, n_times=300, dose_amount=100.0, multi_dose=True),
}


# ---------------------------------------------------------------------------
# NextStat ODE PK: solve + NLL
# ---------------------------------------------------------------------------

def _ns_solve(data: dict[str, Any]) -> dict[str, Any]:
    """Solve ODE PK model via nextstat._core.ode_pk_solve."""
    if not _HAS_ODE_PK_SOLVE:
        raise ImportError("nextstat._core.ode_pk_solve not available")

    result = _ns_ode_pk_solve(  # type: ignore[misc]
        data["model_type"],
        data["params"],
        data["dose_times"],
        data["dose_amounts"],
        data["dose_routes"],
        data["obs_times"],
        solver="rk45",
        rtol=1e-8,
        atol=1e-10,
    )
    return {
        "concentrations": [float(c) for c in result["concentrations"]],
    }


def _ns_nll(data: dict[str, Any]) -> dict[str, Any]:
    """Compute NLL via nextstat._core.ode_pk_nll_py."""
    if not _HAS_ODE_PK_NLL:
        raise ImportError("nextstat._core.ode_pk_nll_py not available")

    sigma = data["true_params_dict"]["sigma"]

    result = _ns_ode_pk_nll(  # type: ignore[misc]
        data["model_type"],
        data["params"],
        data["dose_times"],
        data["dose_amounts"],
        data["dose_routes"],
        data["obs_times"],
        data["obs_values"],
        sigma=sigma,
        solver="rk45",
        rtol=1e-8,
        atol=1e-10,
    )
    # ode_pk_nll_py returns a scalar float (the NLL value)
    return {
        "nll": float(result),
    }


# ---------------------------------------------------------------------------
# Scipy baseline: solve + NLL
# ---------------------------------------------------------------------------

def _scipy_solve(data: dict[str, Any]) -> dict[str, Any]:
    """Solve ODE PK model via scipy (baseline)."""
    model_type = data["model_type"]
    tp = data["true_params_dict"]
    times = data["obs_times"]
    dose_amount = data["dose_amounts"][0]  # all doses same amount in these benchmarks
    dose_times = data["dose_times"]

    if model_type == "transit_1cpt":
        conc = _scipy_transit_1cpt_solve(
            tp["cl"], tp["v1"], tp["ktr"], int(tp["n_transit"]),
            times, dose_amount, dose_times,
        )
    elif model_type == "mm_1cpt":
        conc = _scipy_mm_1cpt_solve(
            tp["v1"], tp["vmax"], tp["km"],
            times, dose_amount, dose_times,
        )
    elif model_type == "tmdd":
        conc = _scipy_tmdd_solve(
            tp["kon"], tp["koff"], tp["kel"], tp["ksyn"], tp["kdeg"], tp["kint"], tp["v"],
            times, dose_amount, dose_times,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return {"concentrations": conc}


def _scipy_nll(data: dict[str, Any]) -> dict[str, Any]:
    """Compute additive-error Gaussian NLL from scipy-solved concentrations.

    Matches NS Rust: NLL = (n/2)*ln(2*pi*sigma^2) + (1/(2*sigma^2))*sum((obs-pred)^2)
    """
    sol = _scipy_solve(data)
    conc = sol["concentrations"]
    obs = data["obs_values"]
    sigma = data["true_params_dict"]["sigma"]

    n = len(obs)
    sse = sum((o - pred) ** 2 for pred, o in zip(conc, obs))
    nll = 0.5 * n * math.log(2.0 * math.pi * sigma * sigma) + 0.5 * sse / (sigma * sigma)

    return {"nll": nll}


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


def _conc_parity(ns_conc: list[float], scipy_conc: list[float]) -> dict[str, Any]:
    """Compute concentration-vector parity metrics."""
    if len(ns_conc) != len(scipy_conc):
        return {"status": "warn", "reason": "length_mismatch"}

    abs_diffs = [abs(a - b) for a, b in zip(ns_conc, scipy_conc)]
    rel_diffs = [
        abs(a - b) / max(abs(a), abs(b), 1e-15) for a, b in zip(ns_conc, scipy_conc)
    ]
    max_abs = max(abs_diffs) if abs_diffs else 0.0
    max_rel = max(rel_diffs) if rel_diffs else 0.0
    mean_rel = sum(rel_diffs) / len(rel_diffs) if rel_diffs else 0.0

    status = "ok" if max_rel < 0.05 else "warn"  # 5% tolerance for ODE solver differences
    return {
        "status": status,
        "max_abs_diff": float(max_abs),
        "max_rel_diff": float(max_rel),
        "mean_rel_diff": float(mean_rel),
        "n_points": len(ns_conc),
    }


def _nll_parity(ns_nll: float, scipy_nll: float) -> dict[str, Any]:
    """Compute NLL parity metrics."""
    abs_diff = abs(ns_nll - scipy_nll)
    rel_diff = abs_diff / max(abs(ns_nll), abs(scipy_nll), 1e-15)
    status = "ok" if rel_diff < 0.01 else "warn"  # 1% tolerance
    return {
        "status": status,
        "ns_nll": float(ns_nll),
        "scipy_nll": float(scipy_nll),
        "abs_diff": float(abs_diff),
        "rel_diff": float(rel_diff),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="ODE PK benchmark runner")
    ap.add_argument("--case", required=True, help="Case id for reporting.")
    ap.add_argument("--case-key", required=True,
                    choices=list(_DATA_GEN_DISPATCH.keys()),
                    help="Data generation key.")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", type=int, default=50)
    ap.add_argument("--baseline-repeat", type=int, default=5,
                    help="Repeat count for scipy baseline timing loops.")
    ap.add_argument("--skip-baselines", action="store_true",
                    help="Skip scipy baseline solve/NLL.")
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    case_key = str(args.case_key)
    seed = int(args.seed)
    repeat = int(args.repeat)
    baseline_repeat = max(1, int(args.baseline_repeat))

    meta = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "nextstat_version": _NS_VERSION,
        "numpy_version": np.__version__,
        "ns_ode_pk_solve_available": _HAS_ODE_PK_SOLVE,
        "ns_ode_pk_nll_available": _HAS_ODE_PK_NLL,
    }
    try:
        import scipy
        meta["scipy_version"] = str(scipy.__version__)
    except ImportError:
        meta["scipy_version"] = None

    # -- Generate data --------------------------------------------------------
    data = _DATA_GEN_DISPATCH[case_key](seed)
    model_type = data["model_type"]

    spec: dict[str, Any] = {
        "case_key": case_key,
        "model_type": model_type,
        "seed": seed,
        "n_times": len(data["obs_times"]),
        "dose_amounts": data["dose_amounts"],
        "n_doses": len(data["dose_times"]),
        "dose_routes": data["dose_routes"],
    }
    spec.update(data["true_params_dict"])
    dataset = {
        "id": f"generated:ode_pk:{args.case}",
        "sha256": sha256_json_obj(spec),
        "spec": spec,
    }

    status = "ok"
    reason: Optional[str] = None

    # -- Check NS availability ------------------------------------------------
    if not _HAS_ODE_PK_SOLVE or not _HAS_ODE_PK_NLL:
        missing = []
        if not _HAS_ODE_PK_SOLVE:
            missing.append("ode_pk_solve")
        if not _HAS_ODE_PK_NLL:
            missing.append("ode_pk_nll_py")
        obj = {
            "schema_version": "nextstat.ode_pk_benchmark_result.v1",
            "suite": "ode_pk",
            "case": str(args.case),
            "environment": collect_environment(),
            "status": "skipped",
            "reason": f"nextstat._core missing: {', '.join(missing)}",
            "meta": meta,
            "dataset": dataset,
            "config": {"case_key": case_key, "model_type": model_type, "seed": seed},
            "parity": {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}},
            "timing": {"wall_time_s": {"nextstat_solve": _summary([]), "nextstat_nll": _summary([])},
                       "raw": {"repeat": repeat, "policy": "median", "ns_solve_runs_s": [], "ns_nll_runs_s": []}},
            "timing_baseline": {},
            "results": {"nextstat": {}, "baseline": None},
        }
        out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        print(f"  [{args.case}] SKIPPED -- {', '.join(missing)} not in nextstat._core")
        return 0

    # -- NextStat: warmup + single solve/NLL ----------------------------------
    try:
        ns_solve_result = _ns_solve(data)
        ns_nll_result = _ns_nll(data)
    except Exception as e:
        obj = {
            "schema_version": "nextstat.ode_pk_benchmark_result.v1",
            "suite": "ode_pk",
            "case": str(args.case),
            "environment": collect_environment(),
            "status": "failed",
            "reason": f"nextstat_error:{type(e).__name__}:{e}",
            "meta": meta,
            "dataset": dataset,
            "config": {"case_key": case_key, "model_type": model_type, "seed": seed},
            "parity": {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}},
            "timing": {"wall_time_s": {"nextstat_solve": _summary([]), "nextstat_nll": _summary([])},
                       "raw": {"repeat": repeat, "policy": "median", "ns_solve_runs_s": [], "ns_nll_runs_s": []}},
            "timing_baseline": {},
            "results": {"nextstat": {}, "baseline": None},
        }
        out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
        print(f"  [{args.case}] FAILED -- {type(e).__name__}: {e}")
        return 2

    # -- NextStat timing: solve -----------------------------------------------
    ns_solve_runs: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ns_solve(data)
        ns_solve_runs.append(float(time.perf_counter() - t0))

    # -- NextStat timing: NLL -------------------------------------------------
    ns_nll_runs: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ns_nll(data)
        ns_nll_runs.append(float(time.perf_counter() - t0))

    # -- Scipy baseline -------------------------------------------------------
    baseline_solve_result: Optional[dict[str, Any]] = None
    baseline_nll_result: Optional[dict[str, Any]] = None
    parity: dict[str, Any] = {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}}
    timing_baseline: dict[str, Any] = {}

    try:
        if args.skip_baselines:
            raise RuntimeError("baseline_skipped")

        # Scipy solve
        baseline_solve_result = _scipy_solve(data)
        baseline_nll_result = _scipy_nll(data)

        # Baseline timing: solve
        bl_solve_runs: list[float] = []
        for _ in range(baseline_repeat):
            t0 = time.perf_counter()
            _scipy_solve(data)
            bl_solve_runs.append(float(time.perf_counter() - t0))

        # Baseline timing: NLL
        bl_nll_runs: list[float] = []
        for _ in range(baseline_repeat):
            t0 = time.perf_counter()
            _scipy_nll(data)
            bl_nll_runs.append(float(time.perf_counter() - t0))

        scipy_v = "unknown"
        try:
            import scipy
            scipy_v = str(scipy.__version__)
        except ImportError:
            pass

        timing_baseline = {
            "name": "scipy",
            "wall_time_s": {
                "solve": _summary(bl_solve_runs),
                "nll": _summary(bl_nll_runs),
            },
            "raw": {
                "repeat": baseline_repeat,
                "solve_runs_s": bl_solve_runs,
                "nll_runs_s": bl_nll_runs,
            },
        }

        # Parity: concentration + NLL
        conc_par = _conc_parity(ns_solve_result["concentrations"],
                                baseline_solve_result["concentrations"])
        nll_par = _nll_parity(ns_nll_result["nll"], baseline_nll_result["nll"])

        parity_status = "ok" if conc_par["status"] == "ok" and nll_par["status"] == "ok" else "warn"
        parity = {
            "status": parity_status,
            "reference": {"name": "scipy", "version": scipy_v},
            "metrics": {
                "concentration": conc_par,
                "nll": nll_par,
            },
        }

        # Speedups
        ns_solve_median = _pctl(ns_solve_runs, 0.5)
        bl_solve_median = _pctl(bl_solve_runs, 0.5)
        solve_speedup = bl_solve_median / ns_solve_median if ns_solve_median > 0 and bl_solve_median > 0 else None

        ns_nll_median = _pctl(ns_nll_runs, 0.5)
        bl_nll_median = _pctl(bl_nll_runs, 0.5)
        nll_speedup = bl_nll_median / ns_nll_median if ns_nll_median > 0 and bl_nll_median > 0 else None

    except Exception as e:
        solve_speedup = None
        nll_speedup = None
        if str(e) == "baseline_skipped":
            parity = {"status": "skipped", "reference": {"name": "", "version": ""}, "metrics": {}}
        else:
            status = "warn"
            reason = f"baseline_unavailable:{type(e).__name__}:{e}"
            parity = {"status": "warn", "reference": {"name": "", "version": ""}, "metrics": {}}

    # -- Build output ---------------------------------------------------------
    timing: dict[str, Any] = {
        "wall_time_s": {
            "nextstat_solve": _summary(ns_solve_runs),
            "nextstat_nll": _summary(ns_nll_runs),
        },
        "raw": {
            "repeat": repeat,
            "policy": "median",
            "ns_solve_runs_s": [float(x) for x in ns_solve_runs],
            "ns_nll_runs_s": [float(x) for x in ns_nll_runs],
        },
    }
    if solve_speedup is not None:
        timing["speedup_solve_vs_scipy"] = float(solve_speedup)
    if nll_speedup is not None:
        timing["speedup_nll_vs_scipy"] = float(nll_speedup)

    cfg: dict[str, Any] = {
        "case_key": case_key,
        "model_type": model_type,
        "n_times": len(data["obs_times"]),
        "dose_amounts": data["dose_amounts"],
        "n_doses": len(data["dose_times"]),
        "dose_routes": data["dose_routes"],
        "repeat": repeat,
        "baseline_repeat": baseline_repeat,
        "seed": seed,
        "solver": "rk45",
        "rtol": 1e-8,
        "atol": 1e-10,
    }

    results: dict[str, Any] = {
        "nextstat": {
            "solve": ns_solve_result,
            "nll": ns_nll_result,
        },
    }
    if baseline_solve_result is not None:
        results["baseline"] = {
            "solve": baseline_solve_result,
            "nll": baseline_nll_result,
        }
    else:
        results["baseline"] = None

    obj = {
        "schema_version": "nextstat.ode_pk_benchmark_result.v1",
        "suite": "ode_pk",
        "case": str(args.case),
        "environment": collect_environment(),
        "status": status,
        "reason": reason,
        "meta": meta,
        "dataset": dataset,
        "config": cfg,
        "parity": parity,
        "timing": timing,
        "timing_baseline": timing_baseline,
        "results": results,
    }
    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")

    # Print summary line
    ns_solve_med = _pctl(ns_solve_runs, 0.5)
    ns_nll_med = _pctl(ns_nll_runs, 0.5)
    extra = ""
    if solve_speedup is not None:
        extra += f" solve_speedup={solve_speedup:.1f}x"
    if nll_speedup is not None:
        extra += f" nll_speedup={nll_speedup:.1f}x"
    par_st = parity.get("status", "skipped")
    print(f"  [{args.case}] {status.upper()} solve={ns_solve_med:.6f}s nll={ns_nll_med:.6f}s parity={par_st}{extra}")

    return 0 if status != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
