#!/usr/bin/env python3
"""Apex2 runner: Pharma reference suite (simulated PK/NLME).

Goal: provide deterministic, machine-readable evidence for:
- PK analytic correctness (1-compartment oral dosing closed form vs predict()).
- PK fit recovery on synthetic (noiseless) data.
- NLME surface sanity (finite NLL/grad) + fit smoke on synthetic multi-subject data.

Run:
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python \
    tests/apex2_pharma_reference_report.py --out tmp/apex2_pharma_reference_report.json
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from _apex2_json import write_report_json


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _oral_1c_analytic(
    times: List[float], *, cl: float, v: float, ka: float, dose: float, f: float
) -> List[float]:
    # One-compartment oral dosing with first-order absorption and elimination.
    # C(t) = (F*D*ka/(V*(ka-ke))) * (exp(-ke t) - exp(-ka t))
    # Limit for ka->ke: C(t) = (F*D/V) * ka * t * exp(-ka t)
    ke = float(cl) / float(v)
    out: List[float] = []
    for t in times:
        t = float(t)
        if t < 0:
            raise ValueError("time must be >= 0")
        if abs(float(ka) - ke) < 1e-12:
            c = (float(f) * float(dose) / float(v)) * float(ka) * t * math.exp(-float(ka) * t)
        else:
            pref = (float(f) * float(dose) * float(ka)) / (float(v) * (float(ka) - ke))
            c = pref * (math.exp(-ke * t) - math.exp(-float(ka) * t))
        out.append(float(c))
    return out


def _case_pk_predict_analytic(nextstat_mod) -> Dict[str, Any]:
    times = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    cl, v, ka = 1.2, 15.0, 2.0
    dose, f = 100.0, 1.0

    analytic = _oral_1c_analytic(times, cl=cl, v=v, ka=ka, dose=dose, f=f)
    model = nextstat_mod.OneCompartmentOralPkModel(
        times,
        [0.0] * len(times),
        dose=dose,
        bioavailability=f,
        sigma=0.05,
        lloq=None,
        lloq_policy="censored",
    )
    pred = [float(x) for x in model.predict([cl, v, ka])]
    max_abs = max((abs(a - b) for a, b in zip(analytic, pred)), default=0.0)
    ok = max_abs <= 1e-10 and all(_is_finite(x) and x >= 0.0 for x in pred)
    return {"name": "pk_predict_analytic", "ok": bool(ok), "max_abs_err": float(max_abs)}


def _case_pk_fit_recovery(nextstat_mod) -> Dict[str, Any]:
    times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    true = [1.2, 15.0, 2.0]  # (cl, v, ka)
    dose, f = 100.0, 1.0

    y = _oral_1c_analytic(times, cl=true[0], v=true[1], ka=true[2], dose=dose, f=f)
    model = nextstat_mod.OneCompartmentOralPkModel(
        times,
        y,
        dose=dose,
        bioavailability=f,
        sigma=0.05,
        lloq=None,
        lloq_policy="censored",
    )
    init = [float(x) for x in model.suggested_init()]
    nll_init = float(model.nll(init))

    mle = nextstat_mod.MaximumLikelihoodEstimator(max_iter=200, tol=1e-7, m=10)
    fit = mle.fit(model)
    best = [float(x) for x in fit.bestfit]
    nll_best = float(fit.nll)

    max_abs_param_err = max(abs(float(a) - float(b)) for a, b in zip(true, best))
    ok = (
        len(best) == 3
        and all(_is_finite(x) and x > 0.0 for x in best)
        and _is_finite(nll_best)
        and _is_finite(nll_init)
        and nll_best <= nll_init + 1e-9
        and max_abs_param_err <= 5e-2
    )
    return {
        "name": "pk_fit_recovery",
        "ok": bool(ok),
        "nll_init": float(nll_init),
        "nll_best": float(nll_best),
        "max_abs_param_err": float(max_abs_param_err),
    }


def _case_nlme_fit_smoke(nextstat_mod) -> Dict[str, Any]:
    n_subjects = 3
    times_per = [0.25, 0.5, 1.0, 2.0, 4.0]

    # Generate deterministic synthetic observations from a base PK model (no noise).
    base = nextstat_mod.OneCompartmentOralPkModel(
        times_per,
        [0.0] * len(times_per),
        dose=100.0,
        bioavailability=1.0,
        sigma=0.05,
        lloq=None,
        lloq_policy="censored",
    )
    cl_pop, v_pop, ka_pop = 1.2, 15.0, 2.0
    y_per = [float(x) for x in base.predict([cl_pop, v_pop, ka_pop])]

    times: List[float] = []
    y: List[float] = []
    subject_idx: List[int] = []
    for sid in range(int(n_subjects)):
        for j, t in enumerate(times_per):
            times.append(float(t))
            y.append(float(y_per[j]))
            subject_idx.append(int(sid))

    model = nextstat_mod.OneCompartmentOralPkNlmeModel(
        times,
        y,
        subject_idx,
        int(n_subjects),
        dose=100.0,
        bioavailability=1.0,
        sigma=0.05,
        lloq=None,
        lloq_policy="censored",
    )
    dim = int(model.n_params())
    init = [float(x) for x in model.suggested_init()]
    nll_init = float(model.nll(init))
    g = [float(v) for v in model.grad_nll(init)]

    mle = nextstat_mod.MaximumLikelihoodEstimator(max_iter=120, tol=1e-6, m=10)
    fit = mle.fit(model)
    best = [float(x) for x in fit.bestfit]
    nll_best = float(fit.nll)

    ok = (
        dim == len(init)
        and len(g) == dim
        and len(best) == dim
        and all(_is_finite(v) for v in g)
        and _is_finite(nll_init)
        and _is_finite(nll_best)
        and nll_best <= nll_init + 1e-9
    )
    return {
        "name": "nlme_fit_smoke",
        "ok": bool(ok),
        "n_subjects": int(n_subjects),
        "dim": int(dim),
        "nll_init": float(nll_init),
        "nll_best": float(nll_best),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("tmp/apex2_pharma_reference_report.json"))
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Make JSON output deterministic (omit timestamps/timings).",
    )
    args = ap.parse_args()

    repo = _repo_root()
    t0 = time.time()

    report: Dict[str, Any] = {
        "meta": {
            "timestamp": int(t0),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "prereqs": {
                "nextstat": True,
            },
        },
        "status": None,
        "cases": [],
        "summary": None,
    }

    # Ensure local editable package can be imported if present.
    # This mirrors other Apex2 runners; master runner also injects PYTHONPATH when needed.
    env_py = os.environ.get("PYTHONPATH", "")
    editable = repo / "bindings" / "ns-py" / "python"
    if editable.exists() and str(editable) not in env_py.split(os.pathsep):
        os.environ["PYTHONPATH"] = (env_py + os.pathsep + str(editable)) if env_py else str(editable)

    try:
        import nextstat  # type: ignore
    except ModuleNotFoundError as e:
        report["meta"]["prereqs"]["nextstat"] = False
        report["status"] = "skipped"
        report["reason"] = f"import_nextstat_failed:{e}"
        write_report_json(args.out, report, deterministic=bool(args.deterministic))
        print(f"Wrote: {args.out}")
        return 0

    cases = [
        _case_pk_predict_analytic(nextstat),
        _case_pk_fit_recovery(nextstat),
        _case_nlme_fit_smoke(nextstat),
    ]

    n_ok = sum(1 for c in cases if c.get("ok") is True)
    n_failed = len(cases) - n_ok
    status = "ok" if n_failed == 0 else "fail"

    report["cases"] = cases
    report["status"] = status
    report["summary"] = {
        "status": status,
        "n_cases": int(len(cases)),
        "n_ok": int(n_ok),
        "n_failed": int(n_failed),
    }
    report["meta"]["wall_s"] = float(time.time() - t0)

    write_report_json(args.out, report, deterministic=bool(args.deterministic))
    print(f"Wrote: {args.out}")
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())

