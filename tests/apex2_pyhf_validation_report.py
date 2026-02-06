#!/usr/bin/env python3
"""Apex2 validation runner: pyhf reference vs NextStat.

Methodology: Planning -> Exploration -> Execution -> Verification

This script is the "Execution + Verification" artifact for the pyhf phase.
It runs a deterministic suite of HistFactory workspaces through:
  1) pyhf reference
  2) nextstat released implementation

and produces a JSON report with:
  - NLL parity (init + random points)
  - expected_data parity (full + main-only)
  - profiling: time per NLL call (Python overhead included)
  - optional: fit profiling

Run:
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_pyhf_validation_report.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
import timeit
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pyhf

import nextstat


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@dataclass(frozen=True)
class Case:
    name: str
    workspace: dict[str, Any]
    measurement: str


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / name).read_text())


def pyhf_model_and_data(workspace: dict[str, Any], measurement_name: str):
    ws = pyhf.Workspace(workspace)
    model = ws.model(
        measurement_name=measurement_name,
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        },
    )
    data = ws.data(model)
    return model, data


def pyhf_nll(model, data, params) -> float:
    return float(pyhf.infer.mle.twice_nll(params, data, model).item()) / 2.0


def map_params_by_name(src_names, src_params, dst_names, dst_init):
    dst_index = {name: i for i, name in enumerate(dst_names)}
    out = list(dst_init)
    for name, value in zip(src_names, src_params):
        out[dst_index[name]] = float(value)
    return out


def sample_params(
    rng: random.Random, init: List[float], bounds: List[Tuple[float, float]]
) -> List[float]:
    out: List[float] = []
    for x0, (lo, hi) in zip(init, bounds):
        lo_f = float(lo)
        hi_f = float(hi)
        if not (lo_f < hi_f):
            out.append(float(x0))
            continue
        span = hi_f - lo_f
        center = min(max(float(x0), lo_f), hi_f)
        half = 0.25 * span
        a = max(lo_f, center - half)
        b = min(hi_f, center + half)
        if not (a < b):
            a, b = lo_f, hi_f
        out.append(rng.uniform(a, b))
    return out


def make_synthetic_shapesys_workspace(n_bins: int) -> dict[str, Any]:
    signal = {
        "name": "signal",
        "data": [5.0] * n_bins,
        "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
    }
    bkg = {
        "name": "background",
        "data": [50.0] * n_bins,
        "modifiers": [{"name": "uncorr_bkguncrt", "type": "shapesys", "data": [5.0] * n_bins}],
    }
    return {
        "channels": [{"name": "c", "samples": [signal, bkg]}],
        "observations": [{"name": "c", "data": [53.0] * n_bins}],
        "measurements": [{"name": "m", "config": {"poi": "mu", "parameters": []}}],
        "version": "1.0.0",
    }


def make_suite(*, sizes: List[int]) -> List[Case]:
    suite: List[Case] = [
        Case("fixture_simple", load_fixture("simple_workspace.json"), "GaussExample"),
        Case("fixture_complex", load_fixture("complex_workspace.json"), "measurement"),
    ]
    for n in sizes:
        suite.append(Case(f"synthetic_shapesys_{n}", make_synthetic_shapesys_workspace(n), "m"))
    return suite


def bench_time_per_call(fn: Callable[[], Any], *, target_s: float = 0.25, repeat: int = 5) -> float:
    number = 1
    while True:
        t = timeit.timeit(fn, number=number)
        if t >= target_s or number >= 1_000_000:
            break
        number *= 2
    times = [timeit.timeit(fn, number=number) / number for _ in range(repeat)]
    return min(times)


def bench_fit_time(fn: Callable[[], Any], *, repeat: int = 3) -> float:
    times: List[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def max_abs_diff(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return float("inf")
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="2,16,64,256", help="Comma-separated synthetic bin counts.")
    ap.add_argument("--n-random", type=int, default=8, help="Random points per model.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed.")
    ap.add_argument("--fit", action="store_true", help="Also profile full MLE fits.")
    ap.add_argument("--out", type=Path, default=Path("tmp/apex2_pyhf_report.json"))
    ap.add_argument("--nll-atol", type=float, default=1e-9)
    ap.add_argument("--nll-rtol", type=float, default=0.0)
    ap.add_argument("--expected-data-atol", type=float, default=1e-10)
    args = ap.parse_args()

    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    suite = make_suite(sizes=sizes)
    rng = random.Random(args.seed)

    report: Dict[str, Any] = {
        "meta": {
            "timestamp": int(time.time()),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pyhf_version": pyhf.__version__,
            "nextstat_version": nextstat.__version__,
            "cwd": os.getcwd(),
            "params": {
                "sizes": sizes,
                "n_random": args.n_random,
                "seed": args.seed,
                "fit": bool(args.fit),
                "nll_atol": args.nll_atol,
                "nll_rtol": args.nll_rtol,
                "expected_data_atol": args.expected_data_atol,
            },
        },
        "cases": [],
        "summary": {},
    }

    failures: List[Dict[str, Any]] = []

    for case in suite:
        py_model, py_data = pyhf_model_and_data(case.workspace, case.measurement)
        py_init = list(map(float, py_model.config.suggested_init()))
        py_bounds = [(float(a), float(b)) for a, b in py_model.config.suggested_bounds()]
        py_names = list(py_model.config.par_names)

        ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(case.workspace))
        ns_names = ns_model.parameter_names()
        ns_init = ns_model.suggested_init()

        if set(ns_names) != set(py_names):
            failures.append(
                {
                    "case": case.name,
                    "reason": "parameter_name_set_mismatch",
                    "missing": sorted(set(py_names) - set(ns_names)),
                    "extra": sorted(set(ns_names) - set(py_names)),
                }
            )
            continue

        def ns_params(py_params: List[float]) -> List[float]:
            return map_params_by_name(py_names, py_params, ns_names, ns_init)

        # Parity checks: init + random points (+ POI variations)
        points: List[Tuple[str, List[float]]] = [("suggested_init", py_init)]
        for i in range(args.n_random):
            points.append((f"random_{i}", sample_params(rng, py_init, py_bounds)))

        poi_idx = py_model.config.poi_index
        if poi_idx is not None:
            for mu in [0.0, 0.5, 2.0]:
                p = list(py_init)
                p[int(poi_idx)] = float(mu)
                points.append((f"poi_{mu}", p))

        nll_diffs: List[float] = []
        exp_full_diffs: List[float] = []
        exp_main_diffs: List[float] = []

        max_abs_nll = 0.0
        max_abs_nll_point: Optional[Dict[str, Any]] = None
        max_abs_nll_py = 0.0

        max_abs_exp_full = 0.0
        max_abs_exp_full_point: Optional[Dict[str, Any]] = None
        max_abs_exp_main = 0.0
        max_abs_exp_main_point: Optional[Dict[str, Any]] = None

        for _tag, p in points:
            nll_py = pyhf_nll(py_model, py_data, p)
            nll_ns = float(ns_model.nll(ns_params(p)))
            delta_nll = nll_ns - nll_py
            nll_diffs.append(delta_nll)
            max_abs_nll_py = max(max_abs_nll_py, abs(nll_py))
            if abs(delta_nll) >= max_abs_nll:
                max_abs_nll = abs(delta_nll)
                max_abs_nll_point = {
                    "tag": _tag,
                    "pyhf": float(nll_py),
                    "nextstat": float(nll_ns),
                    "delta": float(delta_nll),
                }

            exp_py_full = [float(x) for x in py_model.expected_data(p)]
            exp_ns_full = [float(x) for x in ns_model.expected_data(ns_params(p), include_auxdata=True)]
            d_full = max_abs_diff(exp_ns_full, exp_py_full)
            exp_full_diffs.append(d_full)
            if d_full >= max_abs_exp_full:
                max_abs_exp_full = float(d_full)
                max_abs_exp_full_point = {"tag": _tag, "max_abs_delta": float(d_full)}

            exp_py_main = [float(x) for x in py_model.expected_data(p, include_auxdata=False)]
            exp_ns_main = [float(x) for x in ns_model.expected_data(ns_params(p), include_auxdata=False)]
            d_main = max_abs_diff(exp_ns_main, exp_py_main)
            exp_main_diffs.append(d_main)
            if d_main >= max_abs_exp_main:
                max_abs_exp_main = float(d_main)
                max_abs_exp_main_point = {"tag": _tag, "max_abs_delta": float(d_main)}

        # Thresholds
        nll_allowed = max(float(args.nll_atol), float(args.nll_rtol) * float(max_abs_nll_py))

        parity_ok = (
            float(max_abs_nll) <= nll_allowed
            and float(max_abs_exp_full) <= float(args.expected_data_atol)
            and float(max_abs_exp_main) <= float(args.expected_data_atol)
        )

        # Profiling: NLL time-per-call (Python overhead included)
        ns_params_init = ns_params(py_init)

        def f_pyhf():
            return pyhf_nll(py_model, py_data, py_init)

        def f_ns():
            return ns_model.nll(ns_params_init)

        for _ in range(10):
            f_pyhf()
            f_ns()

        pyhf_t = bench_time_per_call(f_pyhf)
        ns_t = bench_time_per_call(f_ns)
        speedup = pyhf_t / ns_t if ns_t > 0 else float("inf")

        fit_info: Optional[Dict[str, Any]] = None
        if args.fit and poi_idx is not None:
            def fit_pyhf():
                pyhf.infer.mle.fit(py_data, py_model)

            def fit_ns():
                mle = nextstat.MaximumLikelihoodEstimator()
                mle.fit(ns_model)

            fit_pyhf()
            fit_ns()
            pyhf_fit = bench_fit_time(fit_pyhf)
            ns_fit = bench_fit_time(fit_ns)
            fit_info = {
                "pyhf_wall_s": pyhf_fit,
                "nextstat_wall_s": ns_fit,
                "speedup": (pyhf_fit / ns_fit) if ns_fit > 0 else float("inf"),
            }

        n_main_bins = sum(len(obs["data"]) for obs in case.workspace.get("observations", []))

        case_row: Dict[str, Any] = {
            "name": case.name,
            "measurement": case.measurement,
            "n_main_bins": int(n_main_bins),
            "n_params_pyhf": int(py_model.config.npars),
            "parity": {
                "ok": bool(parity_ok),
                "max_abs_delta_nll": float(max_abs_nll),
                "max_abs_delta_nll_point": max_abs_nll_point,
                "nll_allowed": float(nll_allowed),
                "max_abs_delta_expected_full": float(max_abs_exp_full),
                "max_abs_delta_expected_full_point": max_abs_exp_full_point,
                "max_abs_delta_expected_main": float(max_abs_exp_main),
                "max_abs_delta_expected_main_point": max_abs_exp_main_point,
            },
            "perf": {
                "pyhf_nll_wall_s": float(pyhf_t),
                "nextstat_nll_wall_s": float(ns_t),
                "speedup": float(speedup),
            },
        }
        if fit_info is not None:
            case_row["perf"]["fit"] = fit_info

        report["cases"].append(case_row)

        if not parity_ok:
            failures.append(
                {
                    "case": case.name,
                    "reason": "parity_threshold_exceeded",
                    "max_abs_delta_nll": float(max_abs_nll),
                    "nll_allowed": float(nll_allowed),
                    "max_abs_delta_expected_full": float(max_abs_exp_full),
                    "max_abs_delta_expected_main": float(max_abs_exp_main),
                }
            )

    n_cases = len(report["cases"])
    n_ok = sum(1 for c in report["cases"] if c["parity"]["ok"])
    report["summary"] = {
        "n_cases": int(n_cases),
        "n_ok": int(n_ok),
        "n_failed": int(n_cases - n_ok),
        "failures": failures,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    # Concise stdout summary
    print("=" * 88)
    print("Apex2 report: pyhf vs nextstat")
    print("=" * 88)
    print(f"cases: {n_ok}/{n_cases} parity OK")
    header = f"{'case':<28} {'bins':>6} | {'max|ΔNLL|':>12} {'max|Δexp(full)|':>14} {'speedup':>9}"
    print(header)
    print("-" * len(header))
    for c in report["cases"]:
        print(
            f"{c['name']:<28} {c['n_main_bins']:>6} | {c['parity']['max_abs_delta_nll']:>12.3e} {c['parity']['max_abs_delta_expected_full']:>14.3e} {c['perf']['speedup']:>8.2f}x"
        )
    print(f"Wrote: {args.out}")

    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
