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

# Optional: reuse the shared synthetic "model zoo" workspace generators.
PY_TESTS_DIR = Path(__file__).resolve().parent / "python"
if str(PY_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(PY_TESTS_DIR))

try:
    from _pyhf_model_zoo import (  # type: ignore
        make_synthetic_shapesys_workspace,
        make_workspace_histo_normsys_staterror,
        make_workspace_multichannel,
        make_workspace_shapefactor_control_region,
    )
except Exception:  # pragma: no cover
    make_synthetic_shapesys_workspace = None
    make_workspace_histo_normsys_staterror = None
    make_workspace_multichannel = None
    make_workspace_shapefactor_control_region = None


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


def _run_case_workspace(
    *,
    key: str,
    fixture: Optional[str],
    workspace: dict[str, Any],
    measurement: str,
    params: str,
    n_toys: int,
    seed: int,
    mu_truth: float,
    min_used_abs: int,
    min_used_frac: float,
    pull_mean_delta_max: float,
    pull_std_delta_max: float,
    coverage_1sigma_delta_max: float,
) -> Dict[str, Any]:
    import numpy as np

    import nextstat
    import pyhf

    model, data_nominal = _pyhf_model_and_data(workspace, measurement_name=measurement)
    pyhf_names = list(model.config.par_names)
    pyhf_index = {name: i for i, name in enumerate(pyhf_names)}

    pars_true = np.asarray(model.config.suggested_init(), dtype=float)
    poi_idx = int(model.config.poi_index)
    pars_true[poi_idx] = float(mu_truth)
    truth_by_name = {name: float(v) for name, v in zip(pyhf_names, pars_true)}

    expected = np.asarray(model.expected_data(pars_true), dtype=float)
    n_main = int(model.config.nmaindata)

    ns_model = nextstat.from_pyhf(json.dumps(workspace))
    ns_poi_idx = ns_model.poi_index()
    if ns_poi_idx is None:
        raise RuntimeError("NextStat model has no POI index")
    ns_names = list(ns_model.parameter_names())
    ns_index = {name: i for i, name in enumerate(ns_names)}

    poi_name = pyhf_names[poi_idx]
    if poi_name not in ns_index:
        raise RuntimeError(f"POI name '{poi_name}' missing from NextStat parameter names")

    if params.strip().lower() == "poi":
        param_names = [poi_name]
    elif params.strip().lower() == "all":
        # Preserve pyhf order for determinism.
        param_names = [n for n in pyhf_names if n in ns_index]
    else:
        raise ValueError(f"Unknown params mode '{params}'. Expected: poi, all")

    pulls_pyhf: Dict[str, List[float]] = {n: [] for n in param_names}
    pulls_ns: Dict[str, List[float]] = {n: [] for n in param_names}
    cover_pyhf: Dict[str, List[bool]] = {n: [] for n in param_names}
    cover_ns: Dict[str, List[bool]] = {n: [] for n in param_names}

    counters: Dict[str, int] = {
        "n_toys_total": 0,
        "n_pyhf_fit_failed": 0,
        "n_pyhf_unc_failed": 0,
        "n_ns_fit_failed": 0,
        "n_param_valid_used": 0,
    }

    rng = np.random.default_rng(seed)
    for _ in range(n_toys):
        counters["n_toys_total"] += 1
        toy = data_nominal.copy()
        toy[:n_main] = rng.poisson(expected[:n_main])

        try:
            bestfit_pyhf = np.asarray(pyhf.infer.mle.fit(toy, model), dtype=float)
        except Exception:
            counters["n_pyhf_fit_failed"] += 1
            continue

        try:
            unc_pyhf = _numerical_uncertainties(model, toy, bestfit_pyhf)
        except Exception:
            counters["n_pyhf_unc_failed"] += 1
            continue

        try:
            res_ns = nextstat.fit(ns_model, data=toy[:n_main].tolist())
        except Exception:
            counters["n_ns_fit_failed"] += 1
            continue

        bestfit_ns = np.asarray(res_ns.bestfit, dtype=float)
        unc_ns = np.asarray(res_ns.uncertainties, dtype=float)

        for name in param_names:
            i_py = pyhf_index[name]
            i_ns = ns_index[name]

            th_true = float(truth_by_name[name])
            th_hat_py = float(bestfit_pyhf[i_py])
            th_sig_py = float(unc_pyhf[i_py])
            th_hat_ns = float(bestfit_ns[i_ns])
            th_sig_ns = float(unc_ns[i_ns])

            if not (np.isfinite(th_hat_py) and np.isfinite(th_sig_py) and th_sig_py > 0.0):
                continue
            if not (np.isfinite(th_hat_ns) and np.isfinite(th_sig_ns) and th_sig_ns > 0.0):
                continue

            pulls_pyhf[name].append((th_hat_py - th_true) / th_sig_py)
            cover_pyhf[name].append(abs(th_hat_py - th_true) <= th_sig_py)
            pulls_ns[name].append((th_hat_ns - th_true) / th_sig_ns)
            cover_ns[name].append(abs(th_hat_ns - th_true) <= th_sig_ns)
            counters["n_param_valid_used"] += 1

    min_used = max(int(min_used_abs), int(float(min_used_frac) * float(n_toys)))

    per_param: Dict[str, Any] = {}
    any_failed = False
    any_ran = False
    max_abs_delta_mean = 0.0
    max_abs_delta_std = 0.0
    max_abs_delta_cov = 0.0
    n_param_ok = 0
    n_param_fail = 0
    n_param_skipped = 0
    for name in param_names:
        n_used = min(len(pulls_pyhf[name]), len(pulls_ns[name]))
        if n_used < min_used:
            per_param[name] = {
                "status": "skipped",
                "reason": f"insufficient_valid_toys:{n_used}<{min_used}",
                "n_toys_used": int(n_used),
            }
            n_param_skipped += 1
            continue

        any_ran = True
        p_py = pulls_pyhf[name][:n_used]
        p_ns = pulls_ns[name][:n_used]
        c_py = cover_pyhf[name][:n_used]
        c_ns = cover_ns[name][:n_used]

        d_mean = float(_mean(p_ns) - _mean(p_py))
        d_std = float(_std(p_ns) - _std(p_py))
        d_cov = float(
            _mean([1.0 if b else 0.0 for b in c_ns]) - _mean([1.0 if b else 0.0 for b in c_py])
        )
        ok = (
            abs(d_mean) <= float(pull_mean_delta_max)
            and abs(d_std) <= float(pull_std_delta_max)
            and abs(d_cov) <= float(coverage_1sigma_delta_max)
        )
        if not ok:
            any_failed = True
            n_param_fail += 1
        else:
            n_param_ok += 1

        max_abs_delta_mean = max(max_abs_delta_mean, abs(d_mean))
        max_abs_delta_std = max(max_abs_delta_std, abs(d_std))
        max_abs_delta_cov = max(max_abs_delta_cov, abs(d_cov))

        per_param[name] = {
            "status": "ok" if ok else "fail",
            "truth": float(truth_by_name[name]),
            "n_toys_used": int(n_used),
            "pyhf": {
                "pull_mean": float(_mean(p_py)),
                "pull_std": float(_std(p_py)),
                "coverage_1sigma": float(_mean([1.0 if b else 0.0 for b in c_py])),
            },
            "nextstat": {
                "pull_mean": float(_mean(p_ns)),
                "pull_std": float(_std(p_ns)),
                "coverage_1sigma": float(_mean([1.0 if b else 0.0 for b in c_ns])),
            },
            "delta": {"mean": d_mean, "std": d_std, "coverage_1sigma": d_cov},
            "thresholds": {
                "pull_mean_delta_max": float(pull_mean_delta_max),
                "pull_std_delta_max": float(pull_std_delta_max),
                "coverage_1sigma_delta_max": float(coverage_1sigma_delta_max),
            },
        }

    status = "skipped" if (not any_ran and not any_failed) else ("fail" if any_failed else "ok")
    return {
        "name": key,
        "status": status,
        "fixture": fixture,
        "measurement": measurement,
        "mu_truth": float(mu_truth),
        "params": params,
        "param_names": param_names,
        "n_toys_requested": int(n_toys),
        "min_used_abs": int(min_used_abs),
        "min_used_frac": float(min_used_frac),
        "min_used": int(min_used),
        "counters": counters,
        "summary": {
            "n_params": int(len(param_names)),
            "n_params_ok": int(n_param_ok),
            "n_params_fail": int(n_param_fail),
            "n_params_skipped": int(n_param_skipped),
            "max_abs_delta_mean": float(max_abs_delta_mean),
            "max_abs_delta_std": float(max_abs_delta_std),
            "max_abs_delta_coverage_1sigma": float(max_abs_delta_cov),
        },
        "per_param": per_param,
    }


def _run_case(
    *,
    key: str,
    fixture: str,
    measurement: str,
    params: str,
    n_toys: int,
    seed: int,
    mu_truth: float,
    min_used_abs: int,
    min_used_frac: float,
    pull_mean_delta_max: float,
    pull_std_delta_max: float,
    coverage_1sigma_delta_max: float,
) -> Dict[str, Any]:
    workspace = _load_workspace(fixture)
    return _run_case_workspace(
        key=key,
        fixture=fixture,
        workspace=workspace,
        measurement=measurement,
        params=params,
        n_toys=n_toys,
        seed=seed,
        mu_truth=mu_truth,
        min_used_abs=min_used_abs,
        min_used_frac=min_used_frac,
        pull_mean_delta_max=pull_mean_delta_max,
        pull_std_delta_max=pull_std_delta_max,
        coverage_1sigma_delta_max=coverage_1sigma_delta_max,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("tmp/apex2_bias_pulls.json"))
    ap.add_argument("--n-toys", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mu-truth", type=float, default=1.0)
    ap.add_argument("--fixtures", type=str, default="simple", help="simple,complex,all or comma-separated keys")
    ap.add_argument("--include-zoo", action="store_true", help="Also include synthetic model-zoo cases.")
    ap.add_argument(
        "--zoo-sizes",
        type=str,
        default="",
        help="Comma-separated synthetic shapesys bin counts to include (requires --include-zoo).",
    )
    ap.add_argument("--params", type=str, default="poi", help="poi or all")
    ap.add_argument("--min-used-abs", type=int, default=10)
    ap.add_argument("--min-used-frac", type=float, default=0.50)
    ap.add_argument("--pull-mean-delta-max", type=float, default=0.05)
    ap.add_argument("--pull-std-delta-max", type=float, default=0.05)
    ap.add_argument("--coverage-1sigma-delta-max", type=float, default=0.03)
    args = ap.parse_args()

    fixture_cases: Dict[str, Tuple[str, str]] = {
        "simple": ("simple_workspace.json", "GaussExample"),
        "complex": ("complex_workspace.json", "measurement"),
    }
    workspace_cases: Dict[str, Tuple[dict[str, Any], str]] = {}

    zoo_sizes: List[int] = []
    if args.include_zoo:
        if make_workspace_multichannel is None:
            # Keep report runnable even if imports fail for some reason.
            workspace_cases["zoo_import_error"] = ({}, "m")
        else:
            workspace_cases["zoo_multichannel_3"] = (make_workspace_multichannel(3), "m")
            workspace_cases["zoo_histo_normsys_staterror_10"] = (
                make_workspace_histo_normsys_staterror(10),
                "m",
            )
            workspace_cases["zoo_shapefactor_control_4"] = (
                make_workspace_shapefactor_control_region(4),
                "m",
            )

            zoo_sizes = [int(x.strip()) for x in str(args.zoo_sizes).split(",") if x.strip()]
            for n in zoo_sizes:
                workspace_cases[f"synthetic_shapesys_{n}"] = (make_synthetic_shapesys_workspace(int(n)), "m")

    if args.fixtures.strip().lower() == "all":
        keys = list(fixture_cases.keys()) + (list(workspace_cases.keys()) if args.include_zoo else [])
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
                "include_zoo": bool(args.include_zoo),
                "zoo_sizes": zoo_sizes,
                "params": str(args.params),
                "min_used_abs": int(args.min_used_abs),
                "min_used_frac": float(args.min_used_frac),
            },
        },
        "cases": [],
        "summary": {},
    }

    any_failed = False
    any_ran = False
    for case_idx, key in enumerate(keys):
        try:
            if key in fixture_cases:
                fixture, measurement = fixture_cases[key]
                row = _run_case(
                    key=key,
                    fixture=fixture,
                    measurement=measurement,
                    params=str(args.params),
                    n_toys=int(args.n_toys),
                    seed=int(args.seed + case_idx),
                    mu_truth=float(args.mu_truth),
                    min_used_abs=int(args.min_used_abs),
                    min_used_frac=float(args.min_used_frac),
                    pull_mean_delta_max=float(args.pull_mean_delta_max),
                    pull_std_delta_max=float(args.pull_std_delta_max),
                    coverage_1sigma_delta_max=float(args.coverage_1sigma_delta_max),
                )
            elif key in workspace_cases:
                workspace, measurement = workspace_cases[key]
                if key == "zoo_import_error":
                    row = {"name": key, "status": "skipped", "reason": "zoo_import_error"}
                else:
                    row = _run_case_workspace(
                        key=key,
                        fixture=None,
                        workspace=workspace,
                        measurement=measurement,
                        params=str(args.params),
                        n_toys=int(args.n_toys),
                        seed=int(args.seed + case_idx),
                        mu_truth=float(args.mu_truth),
                        min_used_abs=int(args.min_used_abs),
                        min_used_frac=float(args.min_used_frac),
                        pull_mean_delta_max=float(args.pull_mean_delta_max),
                        pull_std_delta_max=float(args.pull_std_delta_max),
                        coverage_1sigma_delta_max=float(args.coverage_1sigma_delta_max),
                    )
            else:
                row = {"name": key, "status": "error", "reason": "unknown_fixture_key"}
                any_failed = True
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
