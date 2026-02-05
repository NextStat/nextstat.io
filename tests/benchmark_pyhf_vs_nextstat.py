#!/usr/bin/env python3
"""Benchmark NextStat vs pyhf on a small suite of pyhf workspaces.

This is meant to answer:
  1) Are results still matching the reference (pyhf)?
  2) How much faster/slower are we, end-to-end from Python?

Run:
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/benchmark_pyhf_vs_nextstat.py

Optional:
  --sizes 2,16,64,256,1024    benchmark synthetic scaling models
  --fit                       also benchmark full MLE fits
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import timeit
from pathlib import Path
from typing import Any, Callable

import pyhf

import nextstat


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


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


def make_synthetic_shapesys_workspace(n_bins: int) -> dict[str, Any]:
    # Matches the synthetic used in `crates/ns-translate/benches/model_benchmark.rs`.
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


def bench_time_per_call(fn: Callable[[], Any], *, target_s: float = 0.25, repeat: int = 5) -> float:
    # Calibrate to a stable `number` so the timing window is long enough.
    number = 1
    while True:
        t = timeit.timeit(fn, number=number)
        if t >= target_s or number >= 1_000_000:
            break
        number *= 2
    times = [timeit.timeit(fn, number=number) / number for _ in range(repeat)]
    return min(times)


def bench_fit_time(fn: Callable[[], Any], *, repeat: int = 3) -> float:
    # Full fits are expensive; keep this simple and stable.
    times: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        default="2,16,64,256,1024",
        help="Comma-separated synthetic bin counts to benchmark.",
    )
    parser.add_argument("--fit", action="store_true", help="Also benchmark full MLE fits.")
    args = parser.parse_args()

    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]

    suite: list[tuple[str, dict[str, Any], str]] = [
        ("fixture_simple", load_fixture("simple_workspace.json"), "GaussExample"),
        ("fixture_complex", load_fixture("complex_workspace.json"), "measurement"),
    ]
    for n in sizes:
        suite.append((f"synthetic_shapesys_{n}", make_synthetic_shapesys_workspace(n), "m"))

    print("=" * 88)
    print("NextStat vs pyhf benchmark (Python API)")
    print("=" * 88)
    print(f"Python:   {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"pyhf:     {pyhf.__version__}")
    print(f"nextstat: {nextstat.__version__}")
    print()

    header = f"{'model':<28} {'bins':>6} | {'pyhf NLL (µs)':>14} {'ns NLL (µs)':>14} {'speedup':>9}"
    print(header)
    print("-" * len(header))

    for name, workspace, measurement_name in suite:
        pyhf_model, pyhf_data = pyhf_model_and_data(workspace, measurement_name)
        pyhf_params = list(pyhf_model.config.suggested_init())

        ns_model = nextstat.HistFactoryModel.from_workspace(json.dumps(workspace))
        ns_params = map_params_by_name(
            pyhf_model.config.par_names,
            pyhf_params,
            ns_model.parameter_names(),
            ns_model.suggested_init(),
        )

        # Sanity: NLL must match (tight). If it doesn't, benchmarking is meaningless.
        pyhf_val = pyhf_nll(pyhf_model, pyhf_data, pyhf_params)
        ns_val = float(ns_model.nll(ns_params))
        if abs(ns_val - pyhf_val) > 1e-10:
            raise SystemExit(
                f"NLL mismatch for {name}: nextstat={ns_val:.15f} pyhf={pyhf_val:.15f}"
            )

        def f_pyhf():
            return pyhf_nll(pyhf_model, pyhf_data, pyhf_params)

        def f_ns():
            return ns_model.nll(ns_params)

        # Warmup
        for _ in range(10):
            f_pyhf()
            f_ns()

        pyhf_t = bench_time_per_call(f_pyhf)
        ns_t = bench_time_per_call(f_ns)

        # bins for reporting
        n_main_bins = sum(len(obs["data"]) for obs in workspace.get("observations", []))

        speedup = pyhf_t / ns_t if ns_t > 0 else float("inf")
        print(
            f"{name:<28} {n_main_bins:>6} | {pyhf_t*1e6:>14.2f} {ns_t*1e6:>14.2f} {speedup:>8.2f}x"
        )

        if args.fit and pyhf_model.config.poi_index is not None:
            def fit_pyhf():
                pyhf.infer.mle.fit(pyhf_data, pyhf_model)

            def fit_ns():
                mle = nextstat.MaximumLikelihoodEstimator()
                mle.fit(ns_model)

            # Fit warmup (once each)
            fit_pyhf()
            fit_ns()

            pyhf_fit = bench_fit_time(fit_pyhf)
            ns_fit = bench_fit_time(fit_ns)
            fit_speedup = pyhf_fit / ns_fit if ns_fit > 0 else float("inf")
            print(
                f"{'':<28} {'':>6} | {'pyhf fit (ms)':>14} {'ns fit (ms)':>14} {'':>9}"
            )
            print(
                f"{'':<28} {'':>6} | {pyhf_fit*1e3:>14.2f} {ns_fit*1e3:>14.2f} {fit_speedup:>8.2f}x"
            )

    print()
    print("Notes:")
    print("- NLL timings include Python overhead (pyhf + nextstat extension calls).")
    print("- For core Rust-only numbers, use `cargo bench` (Criterion benches).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
