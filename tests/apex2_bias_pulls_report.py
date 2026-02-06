#!/usr/bin/env python3
"""Apex2 runner: bias/pulls regression (NextStat vs pyhf).

This is intentionally *not* part of default CI; it is a slow validation harness
that can be run manually or in a nightly job.

It mirrors `tests/python/test_bias_pulls.py`, but produces a JSON report artifact.

Run:
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_bias_pulls_report.py --out tmp/bias_pulls.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_workspace(fixture: str) -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / fixture).read_text())


def _pyhf_model_and_data(workspace: dict[str, Any], measurement_name: str):
    import numpy as np
    import pyhf

    ws = pyhf.Workspace(workspace)
    model = ws.model(
        measurement_name=measurement_name,
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        },
    )
    data = np.asarray(ws.data(model), dtype=float)
    return model, data


def _pyhf_nll(model, data, params) -> float:
    import pyhf

    # pyhf returns twice_nll (tensor); NextStat uses NLL.
    return float(pyhf.infer.mle.twice_nll(params, data, model).item()) / 2.0


def _numerical_uncertainties(model, data, bestfit) -> Any:
    """Diagonal uncertainties via numerical Hessian of NLL (pyhf reference)."""
    import numpy as np

    n = len(bestfit)
    h_step = 1e-4
    damping = 1e-9

    f0 = _pyhf_nll(model, data, bestfit)
    hess = np.zeros((n, n), dtype=float)

    for i in range(n):
        hi = h_step * max(abs(bestfit[i]), 1.0)
        xp = bestfit.copy()
        xm = bestfit.copy()
        xp[i] += hi
        xm[i] -= hi
        fp = _pyhf_nll(model, data, xp)
        fm = _pyhf_nll(model, data, xm)
        hess[i, i] = (fp - 2.0 * f0 + fm) / (hi * hi)

        for j in range(i + 1, n):
            hj = h_step * max(abs(bestfit[j]), 1.0)
            xpp = bestfit.copy()
            xpm = bestfit.copy()
            xmp = bestfit.copy()
            xmm = bestfit.copy()
            xpp[i] += hi
            xpp[j] += hj
            xpm[i] += hi
            xpm[j] -= hj
            xmp[i] -= hi
            xmp[j] += hj
            xmm[i] -= hi
            xmm[j] -= hj
            fij = (
                _pyhf_nll(model, data, xpp)
                - _pyhf_nll(model, data, xpm)
                - _pyhf_nll(model, data, xmp)
                + _pyhf_nll(model, data, xmm)
            ) / (4.0 * hi * hj)
            hess[i, j] = fij
            hess[j, i] = fij

    hess = hess + np.eye(n) * damping
    cov = np.linalg.inv(hess)
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _std(xs: List[float]) -> float:
    import math

    if len(xs) < 2:
        return float("nan")
    mu = _mean(xs)
    v = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return float(math.sqrt(v))


def _run_case(
    *,
    key: str,
    fixture: str,
    measurement: str,
    n_toys: int,
    seed: int,
    mu_truth: float,
    pull_mean_delta_max: float,
    pull_std_delta_max: float,
    coverage_1sigma_delta_max: float,
) -> Dict[str, Any]:
    import numpy as np

    import nextstat
    import pyhf

    workspace = _load_workspace(fixture)
    model, data_nominal = _pyhf_model_and_data(workspace, measurement_name=measurement)

    pars_true = np.asarray(model.config.suggested_init(), dtype=float)
    poi_idx = int(model.config.poi_index)
    pars_true[poi_idx] = float(mu_truth)

    expected = np.asarray(model.expected_data(pars_true), dtype=float)
    n_main = int(model.config.nmaindata)

    pulls_pyhf: List[float] = []
    pulls_ns: List[float] = []
    cover_pyhf: List[bool] = []
    cover_ns: List[bool] = []

    ns_model = nextstat.from_pyhf(json.dumps(workspace))
    ns_poi_idx = ns_model.poi_index()
    if ns_poi_idx is None:
        raise RuntimeError("NextStat model has no POI index")
    ns_poi_idx = int(ns_poi_idx)

    rng = np.random.default_rng(seed)
    for _ in range(n_toys):
        toy = data_nominal.copy()
        toy[:n_main] = rng.poisson(expected[:n_main])

        bestfit_pyhf = np.asarray(pyhf.infer.mle.fit(toy, model), dtype=float)
        unc_pyhf = _numerical_uncertainties(model, toy, bestfit_pyhf)
        mu_hat_pyhf = float(bestfit_pyhf[poi_idx])
        mu_sig_pyhf = float(unc_pyhf[poi_idx])
        if not (np.isfinite(mu_sig_pyhf) and mu_sig_pyhf > 0.0):
            continue
        pulls_pyhf.append((mu_hat_pyhf - mu_truth) / mu_sig_pyhf)
        cover_pyhf.append(abs(mu_hat_pyhf - mu_truth) <= mu_sig_pyhf)

        res_ns = nextstat.fit(ns_model, data=toy[:n_main].tolist())
        mu_hat_ns = float(res_ns.bestfit[ns_poi_idx])
        mu_sig_ns = float(res_ns.uncertainties[ns_poi_idx])
        if not (np.isfinite(mu_sig_ns) and mu_sig_ns > 0.0):
            continue
        pulls_ns.append((mu_hat_ns - mu_truth) / mu_sig_ns)
        cover_ns.append(abs(mu_hat_ns - mu_truth) <= mu_sig_ns)

    n = min(len(pulls_pyhf), len(pulls_ns))
    if n == 0:
        return {
            "name": key,
            "status": "error",
            "reason": "no_valid_toys",
            "n_toys": int(n_toys),
            "seed": int(seed),
        }

    # Compare summary statistics (NextStat vs pyhf deltas).
    d_mean = float(_mean(pulls_ns) - _mean(pulls_pyhf))
    d_std = float(_std(pulls_ns) - _std(pulls_pyhf))
    d_cov = float(_mean([1.0 if b else 0.0 for b in cover_ns]) - _mean([1.0 if b else 0.0 for b in cover_pyhf]))

    ok = (
        abs(d_mean) <= float(pull_mean_delta_max)
        and abs(d_std) <= float(pull_std_delta_max)
        and abs(d_cov) <= float(coverage_1sigma_delta_max)
    )

    return {
        "name": key,
        "status": "ok" if ok else "fail",
        "fixture": fixture,
        "measurement": measurement,
        "mu_truth": float(mu_truth),
        "n_toys_requested": int(n_toys),
        "n_toys_used": int(n),
        "pyhf": {
            "pull_mean": float(_mean(pulls_pyhf)),
            "pull_std": float(_std(pulls_pyhf)),
            "coverage_1sigma": float(_mean([1.0 if b else 0.0 for b in cover_pyhf])),
        },
        "nextstat": {
            "pull_mean": float(_mean(pulls_ns)),
            "pull_std": float(_std(pulls_ns)),
            "coverage_1sigma": float(_mean([1.0 if b else 0.0 for b in cover_ns])),
        },
        "delta": {"mean": d_mean, "std": d_std, "coverage_1sigma": d_cov},
        "thresholds": {
            "pull_mean_delta_max": float(pull_mean_delta_max),
            "pull_std_delta_max": float(pull_std_delta_max),
            "coverage_1sigma_delta_max": float(coverage_1sigma_delta_max),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("tmp/apex2_bias_pulls.json"))
    ap.add_argument("--n-toys", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mu-truth", type=float, default=1.0)
    ap.add_argument("--fixtures", type=str, default="simple", help="simple,complex,all or comma-separated keys")
    ap.add_argument("--pull-mean-delta-max", type=float, default=0.05)
    ap.add_argument("--pull-std-delta-max", type=float, default=0.05)
    ap.add_argument("--coverage-1sigma-delta-max", type=float, default=0.03)
    args = ap.parse_args()

    cases = {
        "simple": ("simple_workspace.json", "GaussExample"),
        "complex": ("complex_workspace.json", "measurement"),
    }
    if args.fixtures.strip().lower() == "all":
        keys = ["simple", "complex"]
    else:
        keys = [k.strip().lower() for k in args.fixtures.split(",") if k.strip()]

    # Prereqs: allow "skipped" report if dependencies are missing.
    prereqs: Dict[str, Any] = {}
    try:
        import numpy as np  # noqa: F401
        import pyhf  # noqa: F401
        import nextstat  # noqa: F401
    except ModuleNotFoundError as e:
        prereqs["ok"] = False
        prereqs["reason"] = f"missing_dependency:{e}"
        report = {"meta": {"prereqs": prereqs}, "cases": [], "summary": {"status": "skipped"}}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"Wrote: {args.out}")
        return 0

    try:
        import scipy  # noqa: F401
        prereqs["scipy"] = True
    except ModuleNotFoundError:
        prereqs["scipy"] = False

    t0 = time.time()
    report: Dict[str, Any] = {
        "meta": {
            "timestamp": int(t0),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "prereqs": prereqs,
            "params": {
                "n_toys": int(args.n_toys),
                "seed": int(args.seed),
                "mu_truth": float(args.mu_truth),
                "fixtures": args.fixtures,
            },
        },
        "cases": [],
        "summary": {},
    }

    any_failed = False
    any_ran = False
    for case_idx, key in enumerate(keys):
        if key not in cases:
            report["cases"].append({"name": key, "status": "error", "reason": "unknown_fixture_key"})
            any_failed = True
            continue
        fixture, measurement = cases[key]
        try:
            row = _run_case(
                key=key,
                fixture=fixture,
                measurement=measurement,
                n_toys=int(args.n_toys),
                seed=int(args.seed + case_idx),
                mu_truth=float(args.mu_truth),
                pull_mean_delta_max=float(args.pull_mean_delta_max),
                pull_std_delta_max=float(args.pull_std_delta_max),
                coverage_1sigma_delta_max=float(args.coverage_1sigma_delta_max),
            )
        except Exception as e:
            row = {"name": key, "status": "skipped", "reason": f"exception:{type(e).__name__}:{e}"}
        if row.get("status") in ("ok", "fail"):
            any_ran = True
        if row.get("status") not in ("ok", "skipped"):
            any_failed = True
        report["cases"].append(row)

    n_ok = sum(1 for c in report["cases"] if c.get("status") == "ok")
    n_fail = sum(1 for c in report["cases"] if c.get("status") == "fail")
    n_skip = sum(1 for c in report["cases"] if c.get("status") == "skipped")
    n_err = sum(1 for c in report["cases"] if c.get("status") == "error")

    status = "skipped" if (not any_ran and n_skip > 0 and not any_failed) else ("fail" if any_failed else "ok")
    report["summary"] = {
        "status": status,
        "n_cases": int(len(report["cases"])),
        "n_ok": int(n_ok),
        "n_fail": int(n_fail),
        "n_skipped": int(n_skip),
        "n_error": int(n_err),
        "wall_s": float(time.time() - t0),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"Wrote: {args.out}")

    # Exit policy: only fail if we actually ran and got a failure/error.
    if status == "fail":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

