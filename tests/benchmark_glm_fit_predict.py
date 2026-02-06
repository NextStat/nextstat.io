#!/usr/bin/env python3
"""Benchmark NextStat GLM fit/predict (public Python API).

Run:
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/benchmark_glm_fit_predict.py

Notes:
- Timings include Python overhead + list conversions inside the high-level surface.
- For Rust-only numbers, use `cargo bench -p ns-inference --bench glm_fit_predict_benchmark`.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import timeit
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

import nextstat
from nextstat.glm import linear, logistic, negbin, poisson


def bench_time_per_call(fn: Callable[[], Any], *, target_s: float = 0.25, repeat: int = 7) -> float:
    number = 1
    while True:
        t = timeit.timeit(fn, number=number)
        if t >= target_s or number >= 1_000_000:
            break
        number *= 2
    times = [timeit.timeit(fn, number=number) / number for _ in range(repeat)]
    # Median is more robust than min for regression gating on shared/dev machines.
    return float(statistics.median(times))


def bench_fit_time(fn: Callable[[], Any], *, warmup: int = 1, repeat: int = 5) -> float:
    # Warm up once to reduce one-time effects (allocs, code paths, caches).
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    # Median is more stable than min when CPU frequency/background load fluctuates.
    return float(statistics.median(times))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Stable enough for this synthetic.
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class Dataset:
    x: list[list[float]]
    y_lin: list[float]
    y_log: list[int]
    y_pois: list[int]
    y_nb: list[int]


def make_dataset(n: int, p: int, seed: int, *, nb_alpha: float) -> Dataset:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, p)).astype(np.float64)

    beta = np.zeros((p + 1,), dtype=np.float64)
    beta[0] = 0.1
    beta[1:] = np.arange(p, dtype=np.float64) * 0.01 - 0.05

    eta = beta[0] + x @ beta[1:]

    y_lin = (eta + rng.normal(0.0, 0.2, size=n)).astype(np.float64)

    pr = _sigmoid(eta).clip(1e-6, 1.0 - 1e-6)
    y_log = (rng.random(n) < pr).astype(np.int64)

    mu = np.exp(eta).clip(1e-6, 50.0)
    y_pois = rng.poisson(lam=mu).astype(np.int64)

    # NB2(mean/dispersion) via Gamma-Poisson mixture:
    # Var(Y) = mu + alpha * mu^2 <=> lambda ~ Gamma(k=1/alpha, scale=alpha*mu), Y ~ Poisson(lambda)
    alpha = float(nb_alpha)
    if not (alpha > 0.0) or not math.isfinite(alpha):
        raise ValueError("nb_alpha must be finite and > 0")
    k = 1.0 / alpha
    lam_nb = rng.gamma(shape=k, scale=alpha * mu)
    y_nb = rng.poisson(lam=lam_nb).astype(np.int64)

    # Convert once: the public surface consumes sequences and converts to lists internally.
    x_list = x.tolist()
    return Dataset(
        x=x_list,
        y_lin=y_lin.tolist(),
        y_log=y_log.tolist(),
        y_pois=y_pois.tolist(),
        y_nb=y_nb.tolist(),
    )

def _maybe_write_report(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="200,2000,20000", help="Comma-separated n values.")
    parser.add_argument("--p", type=int, default=20, help="Feature count (without intercept).")
    parser.add_argument("--l2", type=float, default=0.0, help="Optional ridge (0 disables).")
    parser.add_argument(
        "--nb-alpha",
        type=float,
        default=0.5,
        help="Negative binomial dispersion alpha (>0) for synthetic data generation.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON report output path (e.g. tmp/p6_glm_fit_predict.json).",
    )
    args = parser.parse_args()

    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    p = int(args.p)
    l2 = float(args.l2)
    l2_arg = None if l2 <= 0.0 else l2
    nb_alpha = float(args.nb_alpha)

    report: dict[str, Any] = {
        "meta": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "nextstat_version": nextstat.__version__,
            "settings": {"p": p, "l2": l2_arg, "nb_alpha": nb_alpha, "sizes": sizes},
        },
        "results": [],
    }

    print("=" * 88)
    print("NextStat GLM fit/predict benchmark (Python API)")
    print("=" * 88)
    print(f"Python:    {sys.version.split()[0]}")
    print(f"Platform:  {platform.platform()}")
    print(f"nextstat:  {nextstat.__version__}")
    print(f"settings:  p={p}, l2={l2_arg}, nb_alpha={nb_alpha}")
    print()

    header = (
        f"{'model':<12} {'n':>7} {'p':>5} | "
        f"{'fit (ms)':>10} {'predict (us)':>12}"
    )
    print(header)
    print("-" * len(header))

    for n in sizes:
        ds = make_dataset(n, p, seed=123 + n, nb_alpha=nb_alpha)

        # Linear
        def fit_lin():
            return linear.fit(ds.x, ds.y_lin, include_intercept=True, l2=l2_arg)

        t_fit = bench_fit_time(fit_lin)
        fit_res = fit_lin()
        t_pred = bench_time_per_call(lambda: fit_res.predict(ds.x))
        print(f"{'linear':<12} {n:>7} {p:>5} | {t_fit*1e3:>10.2f} {t_pred*1e6:>12.2f}")
        report["results"].append(
            {
                "model": "linear",
                "n": int(n),
                "p": int(p),
                "fit_s": float(t_fit),
                "predict_s": float(t_pred),
            }
        )

        # Logistic
        def fit_log():
            return logistic.fit(ds.x, ds.y_log, include_intercept=True, l2=l2_arg)

        t_fit = bench_fit_time(fit_log)
        fit_res = fit_log()
        t_pred = bench_time_per_call(lambda: fit_res.predict_proba(ds.x))
        print(f"{'logistic':<12} {n:>7} {p:>5} | {t_fit*1e3:>10.2f} {t_pred*1e6:>12.2f}")
        report["results"].append(
            {
                "model": "logistic",
                "n": int(n),
                "p": int(p),
                "fit_s": float(t_fit),
                "predict_s": float(t_pred),
            }
        )

        # Poisson
        def fit_pois():
            return poisson.fit(ds.x, ds.y_pois, include_intercept=True, l2=l2_arg)

        t_fit = bench_fit_time(fit_pois)
        fit_res = fit_pois()
        t_pred = bench_time_per_call(lambda: fit_res.predict_mean(ds.x))
        print(f"{'poisson':<12} {n:>7} {p:>5} | {t_fit*1e3:>10.2f} {t_pred*1e6:>12.2f}")
        report["results"].append(
            {
                "model": "poisson",
                "n": int(n),
                "p": int(p),
                "fit_s": float(t_fit),
                "predict_s": float(t_pred),
            }
        )

        # Negative binomial (no regularization surface yet; benchmark unregularized baseline)
        def fit_nb():
            return negbin.fit(ds.x, ds.y_nb, include_intercept=True)

        t_fit = bench_fit_time(fit_nb)
        fit_res = fit_nb()
        t_pred = bench_time_per_call(lambda: fit_res.predict_mean(ds.x))
        print(f"{'negbin':<12} {n:>7} {p:>5} | {t_fit*1e3:>10.2f} {t_pred*1e6:>12.2f}")
        report["results"].append(
            {
                "model": "negbin",
                "n": int(n),
                "p": int(p),
                "fit_s": float(t_fit),
                "predict_s": float(t_pred),
            }
        )

    print()
    print("Notes:")
    print("- fit includes covariance estimation where applicable (Rust MLE + finite-diff Hessian).")
    print("- predict timings measure list-producing helpers (not numpy).")
    print("- For Rust-only numbers: `cargo bench -p ns-inference --bench glm_fit_predict_benchmark`.")

    _maybe_write_report(args.out, report)
    if args.out is not None:
        print(f"Wrote: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
