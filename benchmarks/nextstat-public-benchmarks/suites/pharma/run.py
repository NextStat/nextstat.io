#!/usr/bin/env python3
"""Pharma benchmark runner: PK/NLME timing + parameter recovery.

Supports:
- pk:           1-cpt oral NLL timing + optional MLE fit
- nlme:         1-cpt oral NLME (LogDensityModel) NLL timing
- pk_2cpt_iv:   2-cpt IV NLL timing + optional MLE fit
- pk_2cpt_oral: 2-cpt oral NLL timing + optional MLE fit
- foce:         FOCE/FOCEI population PK — wall-clock fit + parameter recovery
- saem:         SAEM population PK — wall-clock fit + parameter recovery
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import random
import sys
import time
import timeit
from pathlib import Path
from typing import Any, Callable

import nextstat
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.bench_env import collect_environment


def sha256_json_obj(obj: dict[str, Any]) -> str:
    b = (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def bench_time_per_call_raw(
    fn: Callable[[], Any], *, target_s: float = 0.25, repeat: int = 5
) -> tuple[int, list[float]]:
    number = 1
    while True:
        t = timeit.timeit(fn, number=number)
        if t >= target_s or number >= 1_000_000:
            break
        number *= 2
    times = [timeit.timeit(fn, number=number) / number for _ in range(repeat)]
    return number, times


def bench_wall_time_raw(fn: Callable[[], Any], *, repeat: int = 3) -> list[float]:
    times: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


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


def _nlme_dose_kwargs(dose: float, dose_kw: str) -> dict[str, Any]:
    if dose_kw == "doses":
        return {"doses": [float(dose)]}
    if dose_kw == "dose":
        return {"dose": float(dose)}
    raise ValueError(f"Unsupported NLME dose kwarg: {dose_kw}")


def _conc_oral_1cpt(dose: float, bioav: float, cl: float, v: float, ka: float, t: float) -> float:
    """1-compartment oral concentration (inline for data generation)."""
    ke = cl / v
    if abs(ka - ke) < 1e-12:
        ka_adj = ka * 1.0001
    else:
        ka_adj = ka
    return (dose * bioav * ka_adj / (v * (ka_adj - ke))) * (math.exp(-ke * t) - math.exp(-ka_adj * t))


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

def generate_pk_1c_oral(*, n_obs: int, seed: int, dose: float, sigma: float) -> dict[str, Any]:
    rng = random.Random(int(seed))
    n_obs = int(n_obs)
    t0, t1 = 0.25, 24.0
    if n_obs <= 1:
        times = [t0]
    else:
        ratio = (t1 / t0) ** (1.0 / (n_obs - 1))
        times = [t0 * (ratio**i) for i in range(n_obs)]
    y0 = [0.0] * n_obs
    m0 = nextstat.OneCompartmentOralPkModel(times, y0, dose=float(dose), sigma=float(sigma))
    true_params = [1.2, 12.0, 0.9]
    mu = m0.predict(true_params)
    y = [float(max(0.0, mu_i + rng.gauss(0.0, float(sigma)))) for mu_i in mu]
    return {
        "kind": "pk_1c_oral", "times": times, "y": y,
        "dose": float(dose), "sigma": float(sigma), "seed": int(seed),
        "true_params": true_params,
    }


def generate_nlme_1c_oral(
    *, n_subjects: int, n_obs_per_subject: int, seed: int, dose: float, sigma: float
) -> dict[str, Any]:
    rng = random.Random(int(seed))
    n_obs_per_subject = int(n_obs_per_subject)
    t0, t1 = 0.25, 24.0
    if n_obs_per_subject <= 1:
        times_one = [t0]
    else:
        ratio = (t1 / t0) ** (1.0 / (n_obs_per_subject - 1))
        times_one = [t0 * (ratio**i) for i in range(n_obs_per_subject)]
    true_pop = [1.2, 12.0, 0.9]
    times: list[float] = []
    y: list[float] = []
    subject_idx: list[int] = []
    for s in range(int(n_subjects)):
        subj = list(true_pop)
        subj[1] *= float(1.0 + rng.gauss(0.0, 0.15))
        subj[2] *= float(1.0 + rng.gauss(0.0, 0.10))
        m0 = nextstat.OneCompartmentOralPkModel(times_one, [0.0] * len(times_one), dose=float(dose), sigma=float(sigma))
        mu = m0.predict(subj)
        for t, mu_i in zip(times_one, mu):
            times.append(float(t))
            y.append(float(max(0.0, mu_i + rng.gauss(0.0, float(sigma)))))
            subject_idx.append(int(s))
    return {
        "kind": "nlme_pk_1c_oral",
        "times": times, "y": y, "subject_idx": subject_idx,
        "n_subjects": int(n_subjects), "n_obs_per_subject": int(n_obs_per_subject),
        "dose": float(dose), "sigma": float(sigma), "seed": int(seed),
        "true_params": true_pop,
    }


def generate_pop_1c_oral(
    *, n_subjects: int, n_obs_per_subject: int, seed: int, dose: float,
    error_model: str, sigma: float, sigma_add: float | None,
    omega_cl: float, omega_v: float, omega_ka: float,
) -> dict[str, Any]:
    """Generate population PK data with proper lognormal random effects."""
    rng = random.Random(int(seed))
    cl_pop, v_pop, ka_pop = 0.134, 8.0, 1.0
    bioav = 1.0

    # Sampling schedule: common PK design
    if n_obs_per_subject <= 8:
        times_one = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0][:n_obs_per_subject]
    else:
        times_one = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0, 36.0, 48.0, 72.0][:n_obs_per_subject]

    times: list[float] = []
    y: list[float] = []
    subject_idx: list[int] = []

    for s in range(int(n_subjects)):
        eta_cl = rng.gauss(0.0, omega_cl)
        eta_v = rng.gauss(0.0, omega_v)
        eta_ka = rng.gauss(0.0, omega_ka)
        cl_i = cl_pop * math.exp(eta_cl)
        v_i = v_pop * math.exp(eta_v)
        ka_i = ka_pop * math.exp(eta_ka)

        for t in times_one:
            c = _conc_oral_1cpt(dose, bioav, cl_i, v_i, ka_i, t)
            # Apply error model
            if error_model == "proportional":
                noise_sd = sigma * abs(c) if c > 0 else sigma * 0.01
            elif error_model == "combined":
                sa = sigma_add if sigma_add is not None else 0.1
                noise_sd = math.sqrt(sa**2 + (sigma * c)**2) if c > 0 else sa
            else:  # additive
                noise_sd = sigma
            obs = max(0.0, c + rng.gauss(0.0, noise_sd))
            times.append(float(t))
            y.append(float(obs))
            subject_idx.append(int(s))

    return {
        "kind": "pop_pk_1c_oral",
        "times": times, "y": y, "subject_idx": subject_idx,
        "n_subjects": int(n_subjects), "n_obs_per_subject": len(times_one),
        "dose": float(dose), "error_model": error_model,
        "sigma": float(sigma), "sigma_add": sigma_add,
        "seed": int(seed),
        "true_theta": [cl_pop, v_pop, ka_pop],
        "true_omega": [omega_cl, omega_v, omega_ka],
    }


def generate_pk_2cpt_iv(*, n_obs: int, seed: int, dose: float, sigma: float) -> dict[str, Any]:
    """Generate 2-compartment IV data from known parameters."""
    rng = random.Random(int(seed))
    true_params = [1.0, 10.0, 20.0, 0.5]  # CL, V1, V2, Q
    cl, v1, v2, q = true_params

    n_obs = int(n_obs)
    times = [0.2 + i * 0.4 for i in range(n_obs)]

    # Analytical 2-cpt IV concentration
    k10 = cl / v1
    k12 = q / v1
    k21 = q / v2
    s = k10 + k12 + k21
    p = k10 * k21
    disc = max(0.0, s * s - 4.0 * p)
    alpha = 0.5 * (s + math.sqrt(disc))
    beta = 0.5 * (s - math.sqrt(disc))
    ab = alpha - beta
    if abs(ab) < 1e-12:
        y = [float(max(0.0, (dose / v1) * math.exp(-0.5 * (alpha + beta) * t) + rng.gauss(0, sigma))) for t in times]
    else:
        ca = (alpha - k21) / ab
        cb = (k21 - beta) / ab
        y = [float(max(0.0, (dose / v1) * (ca * math.exp(-alpha * t) + cb * math.exp(-beta * t)) + rng.gauss(0, sigma)))
             for t in times]

    return {
        "kind": "pk_2cpt_iv", "times": times, "y": y,
        "dose": float(dose), "sigma": float(sigma), "seed": int(seed),
        "true_params": true_params,
    }


def generate_pk_2cpt_oral(*, n_obs: int, seed: int, dose: float, sigma: float) -> dict[str, Any]:
    """Generate 2-compartment oral data from known parameters."""
    rng = random.Random(int(seed))
    true_params = [1.0, 10.0, 20.0, 0.5, 1.5]  # CL, V1, V2, Q, Ka
    cl, v1, v2, q, ka = true_params
    bioav = 1.0

    n_obs = int(n_obs)
    times = sorted(np.geomspace(0.25, 24.0, n_obs).tolist())

    k10 = cl / v1
    k12 = q / v1
    k21 = q / v2
    s = k10 + k12 + k21
    p = k10 * k21
    disc = max(0.0, s * s - 4.0 * p)
    alpha = 0.5 * (s + math.sqrt(disc))
    beta = 0.5 * (s - math.sqrt(disc))

    pref = ka * bioav * dose / v1
    y: list[float] = []
    for t in times:
        da = (ka - alpha) * (beta - alpha)
        db = (ka - beta) * (alpha - beta)
        dc = (alpha - ka) * (beta - ka)
        if abs(da) < 1e-12 or abs(db) < 1e-12 or abs(dc) < 1e-12:
            ka_p = ka * 1.0001
            da = (ka_p - alpha) * (beta - alpha)
            db = (ka_p - beta) * (alpha - beta)
            dc = (alpha - ka_p) * (beta - ka_p)
            ta = (k21 - alpha) / da * math.exp(-alpha * t)
            tb = (k21 - beta) / db * math.exp(-beta * t)
            tc = (k21 - ka_p) / dc * math.exp(-ka_p * t)
        else:
            ta = (k21 - alpha) / da * math.exp(-alpha * t)
            tb = (k21 - beta) / db * math.exp(-beta * t)
            tc = (k21 - ka) / dc * math.exp(-ka * t)
        c = pref * (ta + tb + tc)
        y.append(float(max(0.0, c + rng.gauss(0, sigma))))

    return {
        "kind": "pk_2cpt_oral", "times": times, "y": y,
        "dose": float(dose), "sigma": float(sigma), "seed": int(seed),
        "true_params": true_params, "bioavailability": bioav,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    dose_kw = _nlme_dose_kwarg_name()
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="pk_1c_oral", help="Case id for reporting.")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--target-s", type=float, default=0.25)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--fit", action="store_true", help="Also time an MLE fit (wall-clock).")
    ap.add_argument("--fit-repeat", type=int, default=3)

    ap.add_argument("--model", choices=["pk", "nlme", "pk_2cpt_iv", "pk_2cpt_oral", "foce", "saem"], default="pk")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dose", type=float, default=100.0)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--sigma-add", type=float, default=None)
    ap.add_argument("--error-model", choices=["additive", "proportional", "combined"], default="additive")
    ap.add_argument("--n-obs", type=int, default=8, help="For pk: number of observations.")
    ap.add_argument("--n-subjects", type=int, default=8, help="For nlme/foce/saem: number of subjects.")
    ap.add_argument("--n-obs-per-subject", type=int, default=6, help="For nlme/foce/saem: observations per subject.")
    ap.add_argument("--omega-cl", type=float, default=0.20)
    ap.add_argument("--omega-v", type=float, default=0.15)
    ap.add_argument("--omega-ka", type=float, default=0.25)
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_kind = str(args.model)

    # -----------------------------------------------------------------------
    # Build model + dataset
    # -----------------------------------------------------------------------
    if model_kind == "nlme":
        spec = generate_nlme_1c_oral(
            n_subjects=int(args.n_subjects), n_obs_per_subject=int(args.n_obs_per_subject),
            seed=int(args.seed), dose=float(args.dose), sigma=float(args.sigma),
        )
        model = nextstat.OneCompartmentOralPkNlmeModel(
            spec["times"], spec["y"], spec["subject_idx"], int(spec["n_subjects"]),
            dose=float(spec["dose"]), sigma=float(spec["sigma"]),
        )
    elif model_kind == "pk_2cpt_iv":
        spec = generate_pk_2cpt_iv(
            n_obs=int(args.n_obs), seed=int(args.seed),
            dose=float(args.dose), sigma=float(args.sigma),
        )
        model = nextstat.TwoCompartmentIvPkModel(
            spec["times"], spec["y"], dose=float(spec["dose"]),
            error_model="additive", sigma=float(spec["sigma"]),
        )
    elif model_kind == "pk_2cpt_oral":
        spec = generate_pk_2cpt_oral(
            n_obs=int(args.n_obs), seed=int(args.seed),
            dose=float(args.dose), sigma=float(args.sigma),
        )
        model = nextstat.TwoCompartmentOralPkModel(
            spec["times"], spec["y"], dose=float(spec["dose"]),
            error_model="additive", sigma=float(spec["sigma"]),
        )
    elif model_kind in ("foce", "saem"):
        spec = generate_pop_1c_oral(
            n_subjects=int(args.n_subjects), n_obs_per_subject=int(args.n_obs_per_subject),
            seed=int(args.seed), dose=float(args.dose),
            error_model=str(args.error_model), sigma=float(args.sigma),
            sigma_add=args.sigma_add,
            omega_cl=float(args.omega_cl), omega_v=float(args.omega_v), omega_ka=float(args.omega_ka),
        )
        model = None  # FOCE/SAEM use their own API
    else:
        spec = generate_pk_1c_oral(
            n_obs=int(args.n_obs), seed=int(args.seed),
            dose=float(args.dose), sigma=float(args.sigma),
        )
        model = nextstat.OneCompartmentOralPkModel(
            spec["times"], spec["y"], dose=float(spec["dose"]), sigma=float(spec["sigma"]),
        )

    # -----------------------------------------------------------------------
    # NLL timing (for individual PK models)
    # -----------------------------------------------------------------------
    nll_timing: dict[str, Any] = {}
    if model is not None:
        params = list(model.suggested_init())
        def f_nll():
            return model.nll(params)
        for _ in range(10):
            f_nll()
        number, times = bench_time_per_call_raw(f_nll, target_s=float(args.target_s), repeat=int(args.repeat))
        t = min(times)
        nll_timing = {
            "nll_time_s_per_call": {"nextstat": float(t)},
            "raw": {
                "number": int(number), "repeat": int(args.repeat),
                "target_s": float(args.target_s), "policy": "min",
                "per_call_s": {"nextstat": [float(x) for x in times]},
            },
        }

    # -----------------------------------------------------------------------
    # Dataset ID
    # -----------------------------------------------------------------------
    kind = str(spec["kind"])
    if "n_subjects" in spec:
        dataset_id = (
            f"generated:{kind}:n_sub{int(spec['n_subjects'])}:"
            f"n_obs{int(spec.get('n_obs_per_subject', 0))}:seed{int(args.seed)}"
        )
    else:
        dataset_id = f"generated:{kind}:n_obs{int(args.n_obs)}:seed{int(args.seed)}"
    dataset_sha = sha256_json_obj(spec)

    # -----------------------------------------------------------------------
    # Doc skeleton
    # -----------------------------------------------------------------------
    doc: dict[str, Any] = {
        "schema_version": "nextstat.pharma_benchmark_result.v2",
        "suite": "pharma",
        "case": str(args.case),
        "deterministic": bool(args.deterministic),
        "environment": collect_environment(),
        "meta": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "nextstat_version": nextstat.__version__,
            "nlme_dose_kwarg": dose_kw,
        },
        "dataset": {"id": dataset_id, "sha256": dataset_sha, "spec": spec},
        "model": {
            "kind": kind,
            "n_subjects": int(spec.get("n_subjects", 1)),
            "n_obs": int(len(spec["times"])),
            "n_params": int(model.n_params()) if model is not None else 3,
        },
    }

    if nll_timing:
        doc["timing"] = nll_timing

    # -----------------------------------------------------------------------
    # MLE fit (individual PK models)
    # -----------------------------------------------------------------------
    if args.fit and model is not None:
        fit_repeat = int(args.fit_repeat)
        try:
            def fit_once():
                mle = nextstat.MaximumLikelihoodEstimator()
                return mle.fit(model)

            fit_once()  # warmup
            fit_times = bench_wall_time_raw(fit_once, repeat=fit_repeat)
            fit_t = min(fit_times) if fit_times else 0.0
            res = fit_once()

            recovery: dict[str, Any] = {}
            true_params = spec.get("true_params")
            if true_params:
                for i, (hat, tru) in enumerate(zip(res.parameters, true_params)):
                    rel_err = abs(hat - tru) / abs(tru) if abs(tru) > 1e-12 else abs(hat)
                    recovery[f"param_{i}"] = {"hat": float(hat), "true": float(tru), "rel_err": float(rel_err)}

            doc["fit"] = {
                "status": "ok",
                "time_s": {"nextstat": float(fit_t)},
                "raw": {"repeat": int(fit_repeat), "policy": "min", "per_fit_s": {"nextstat": [float(x) for x in fit_times]}},
                "meta": {
                    "nextstat_nll": float(getattr(res, "nll", float("nan"))),
                    "nextstat_converged": bool(getattr(res, "converged", False)),
                    "nextstat_n_iter": int(getattr(res, "n_iter", 0)),
                    "nextstat_termination_reason": str(getattr(res, "termination_reason", "")),
                },
                "recovery": recovery,
            }
        except Exception as e:
            doc["fit"] = {"status": "failed", "reason": f"{type(e).__name__}: {e}"}

    # -----------------------------------------------------------------------
    # FOCE population fit
    # -----------------------------------------------------------------------
    if model_kind == "foce":
        true_theta = spec["true_theta"]
        true_omega = spec["true_omega"]
        em = str(args.error_model)
        sigma_val = float(args.sigma)
        sigma_add_val = args.sigma_add

        # Perturbed init: 10% off truth
        theta_init = [x * 1.1 for x in true_theta]
        omega_init = [x * 1.1 for x in true_omega]

        try:
            def foce_fit():
                return nextstat.nlme_foce(
                    spec["times"], spec["y"], spec["subject_idx"], int(spec["n_subjects"]),
                    bioavailability=1.0,
                    error_model=em, sigma=sigma_val, sigma_add=sigma_add_val,
                    theta_init=theta_init, omega_init=omega_init,
                    max_outer_iter=300, max_inner_iter=30, tol=1e-4, interaction=True,
                    **_nlme_dose_kwargs(float(spec["dose"]), dose_kw),
                )

            foce_fit()  # warmup
            fit_times = bench_wall_time_raw(foce_fit, repeat=int(args.fit_repeat))
            fit_t = min(fit_times) if fit_times else 0.0
            result = foce_fit()

            recovery = {}
            labels = ["CL", "V", "Ka"]
            for i, (hat, tru, label) in enumerate(zip(result["theta"], true_theta, labels)):
                rel_err = abs(hat - tru) / abs(tru) if abs(tru) > 1e-12 else abs(hat)
                recovery[label] = {"hat": float(hat), "true": float(tru), "rel_err": float(rel_err)}
            for i, (hat, tru, label) in enumerate(zip(result["omega"], true_omega, [f"w_{l}" for l in labels])):
                rel_err = abs(hat - tru) / abs(tru) if abs(tru) > 1e-12 else abs(hat)
                recovery[label] = {"hat": float(hat), "true": float(tru), "rel_err": float(rel_err)}

            doc["foce"] = {
                "status": "ok",
                "time_s": {"nextstat": float(fit_t)},
                "raw": {"repeat": int(args.fit_repeat), "policy": "min", "per_fit_s": {"nextstat": [float(x) for x in fit_times]}},
                "meta": {
                    "ofv": float(result["ofv"]),
                    "converged": bool(result["converged"]),
                    "n_iter": int(result["n_iter"]),
                    "error_model": em,
                },
                "recovery": recovery,
            }
        except Exception as e:
            doc["foce"] = {"status": "failed", "reason": f"{type(e).__name__}: {e}"}

    # -----------------------------------------------------------------------
    # SAEM population fit
    # -----------------------------------------------------------------------
    if model_kind == "saem":
        true_theta = spec["true_theta"]
        true_omega = spec["true_omega"]
        em = str(args.error_model)
        sigma_val = float(args.sigma)
        sigma_add_val = args.sigma_add

        theta_init = [x * 1.1 for x in true_theta]
        omega_init = [x * 1.1 for x in true_omega]

        try:
            def saem_fit():
                return nextstat.nlme_saem(
                    spec["times"], spec["y"], spec["subject_idx"], int(spec["n_subjects"]),
                    bioavailability=1.0,
                    error_model=em, sigma=sigma_val, sigma_add=sigma_add_val,
                    theta_init=theta_init, omega_init=omega_init,
                    n_burn=200, n_iter=100, n_chains=1, seed=12345, tol=1e-4,
                    **_nlme_dose_kwargs(float(spec["dose"]), dose_kw),
                )

            saem_fit()  # warmup
            fit_times = bench_wall_time_raw(saem_fit, repeat=int(args.fit_repeat))
            fit_t = min(fit_times) if fit_times else 0.0
            result = saem_fit()

            recovery = {}
            labels = ["CL", "V", "Ka"]
            for i, (hat, tru, label) in enumerate(zip(result["theta"], true_theta, labels)):
                rel_err = abs(hat - tru) / abs(tru) if abs(tru) > 1e-12 else abs(hat)
                recovery[label] = {"hat": float(hat), "true": float(tru), "rel_err": float(rel_err)}
            for i, (hat, tru, label) in enumerate(zip(result["omega"], true_omega, [f"w_{l}" for l in labels])):
                rel_err = abs(hat - tru) / abs(tru) if abs(tru) > 1e-12 else abs(hat)
                recovery[label] = {"hat": float(hat), "true": float(tru), "rel_err": float(rel_err)}

            doc["saem"] = {
                "status": "ok",
                "time_s": {"nextstat": float(fit_t)},
                "raw": {"repeat": int(args.fit_repeat), "policy": "min", "per_fit_s": {"nextstat": [float(x) for x in fit_times]}},
                "meta": {
                    "ofv": float(result["ofv"]),
                    "converged": bool(result["converged"]),
                    "n_iter": int(result["n_iter"]),
                    "error_model": em,
                },
                "recovery": recovery,
            }
        except Exception as e:
            doc["saem"] = {"status": "failed", "reason": f"{type(e).__name__}: {e}"}

    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
