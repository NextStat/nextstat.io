#!/usr/bin/env python3
"""Apex2 SBC report runner (Phase 5.4.2).

This is a standalone, JSON-producing version of the SBC tests in
`tests/python/test_sbc_nuts.py`, intended for manual/nightly runs.

Run:
  NS_RUN_SLOW=1 NS_SBC_RUNS=20 NS_SBC_WARMUP=200 NS_SBC_SAMPLES=200 \
    PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_sbc_report.py \
      --out tmp/apex2_sbc_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from _apex2_json import write_report_json


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _var(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    mu = _mean(xs)
    return float(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _flatten_chains(posterior: Mapping[str, List[List[float]]], name: str) -> List[float]:
    out: List[float] = []
    for chain in posterior[name]:
        out.extend([float(v) for v in chain])
    return out


def _ranks_u01(draws: Sequence[float], truth: float) -> float:
    n = len(draws)
    if n == 0:
        return float("nan")
    r = sum(1 for v in draws if float(v) < float(truth))
    return float(r / n)


def _assert_sbc_u01(samples_u: Sequence[float], *, max_mean_delta: float, max_var_delta: float) -> Dict[str, Any]:
    # For U(0,1): E[u]=0.5, Var[u]=1/12.
    mu = _mean(samples_u)
    v = _var(samples_u)
    ok_mu = abs(mu - 0.5) <= max_mean_delta
    ok_v = abs(v - (1.0 / 12.0)) <= max_var_delta
    return {
        "ok": bool(ok_mu and ok_v),
        "mean_u": float(mu),
        "var_u": float(v),
        "abs_mean_delta": float(abs(mu - 0.5)),
        "abs_var_delta": float(abs(v - (1.0 / 12.0))),
        "max_mean_delta": float(max_mean_delta),
        "max_var_delta": float(max_var_delta),
    }


def _require_slow_from_env() -> tuple[int, int, int, int] | None:
    if os.environ.get("NS_RUN_SLOW") != "1":
        return None
    n_runs = int(os.environ.get("NS_SBC_RUNS", "20"))
    n_warmup = int(os.environ.get("NS_SBC_WARMUP", "200"))
    n_samples = int(os.environ.get("NS_SBC_SAMPLES", "200"))
    seed = int(os.environ.get("NS_SBC_SEED", "0"))
    return n_runs, n_warmup, n_samples, seed


def _sample_posterior_u(
    nextstat_mod,
    model,
    truth_by_name: Dict[str, float],
    *,
    seed: int,
    n_warmup: int,
    n_samples: int,
    rhat_max: float,
    divergence_rate_max: float,
) -> Dict[str, float]:
    r = nextstat_mod.sample(
        model,
        n_chains=2,
        n_warmup=n_warmup,
        n_samples=n_samples,
        seed=seed,
        init_jitter_rel=0.10,
    )
    diag = r["diagnostics"]
    if float(diag["divergence_rate"]) > divergence_rate_max:
        raise AssertionError(
            f"divergence_rate={diag['divergence_rate']} > {divergence_rate_max}"
        )
    for name, v in diag["r_hat"].items():
        if float(v) >= rhat_max:
            raise AssertionError(f"R-hat({name})={v} >= {rhat_max}")

    posterior = r["posterior"]
    out: Dict[str, float] = {}
    for name, truth in truth_by_name.items():
        draws = _flatten_chains(posterior, name)
        out[name] = _ranks_u01(draws, truth)
    return out


def _case_linear_regression_1d(
    nextstat_mod,
    *,
    n_runs: int,
    n_warmup: int,
    n_samples: int,
    seed0: int,
    rhat_max: float,
    divergence_rate_max: float,
) -> Dict[str, Any]:
    n = 25
    x = [[1.0] for _ in range(n)]

    coef_prior_mu = 0.0
    coef_prior_sigma = 1.0

    u_beta1: List[float] = []
    for run in range(n_runs):
        rng = random.Random(seed0 + run)
        beta1 = float(coef_prior_mu + coef_prior_sigma * rng.gauss(0.0, 1.0))
        y = [float(beta1 + rng.gauss(0.0, 1.0)) for _ in range(n)]

        model = nextstat_mod.ComposedGlmModel.linear_regression(
            x,
            y,
            include_intercept=False,
            coef_prior_mu=coef_prior_mu,
            coef_prior_sigma=coef_prior_sigma,
        )
        u = _sample_posterior_u(
            nextstat_mod,
            model,
            {"beta1": beta1},
            seed=seed0 + 10_000 + run,
            n_warmup=n_warmup,
            n_samples=n_samples,
            rhat_max=rhat_max,
            divergence_rate_max=divergence_rate_max,
        )
        u_beta1.append(float(u["beta1"]))

    tol = 0.25 if n_runs < 20 else 0.12
    chk = _assert_sbc_u01(u_beta1, max_mean_delta=tol, max_var_delta=tol)
    return {"name": "linear_regression_1d_mean_only", "u_by_param": {"beta1": u_beta1}, "check": chk}


def _case_linear_regression_2d(
    nextstat_mod,
    *,
    n_runs: int,
    n_warmup: int,
    n_samples: int,
    seed0: int,
    rhat_max: float,
    divergence_rate_max: float,
) -> Dict[str, Any]:
    n = 30
    xs = [(-1.0 + 2.0 * i / (n - 1)) for i in range(n)]
    x = [[1.0, float(v)] for v in xs]

    coef_prior_mu = 0.0
    coef_prior_sigma = 1.0

    u_b1: List[float] = []
    u_b2: List[float] = []
    for run in range(n_runs):
        rng = random.Random(seed0 + 100 + run)
        beta1 = float(coef_prior_mu + coef_prior_sigma * rng.gauss(0.0, 1.0))
        beta2 = float(coef_prior_mu + coef_prior_sigma * rng.gauss(0.0, 1.0))
        y = [float(beta1 + beta2 * float(v) + rng.gauss(0.0, 1.0)) for v in xs]

        model = nextstat_mod.ComposedGlmModel.linear_regression(
            x,
            y,
            include_intercept=False,
            coef_prior_mu=coef_prior_mu,
            coef_prior_sigma=coef_prior_sigma,
        )
        u = _sample_posterior_u(
            nextstat_mod,
            model,
            {"beta1": beta1, "beta2": beta2},
            seed=seed0 + 20_000 + run,
            n_warmup=n_warmup,
            n_samples=n_samples,
            rhat_max=rhat_max,
            divergence_rate_max=divergence_rate_max,
        )
        u_b1.append(float(u["beta1"]))
        u_b2.append(float(u["beta2"]))

    tol = 0.25 if n_runs < 20 else 0.12
    chk1 = _assert_sbc_u01(u_b1, max_mean_delta=tol, max_var_delta=tol)
    chk2 = _assert_sbc_u01(u_b2, max_mean_delta=tol, max_var_delta=tol)
    ok = bool(chk1["ok"] and chk2["ok"])
    return {
        "name": "linear_regression_2d",
        "u_by_param": {"beta1": u_b1, "beta2": u_b2},
        "check": {"ok": ok, "beta1": chk1, "beta2": chk2},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("tmp/apex2_sbc_report.json"))
    ap.add_argument("--cases", type=str, default="lin1d,lin2d", help="Comma-separated: lin1d,lin2d")
    ap.add_argument("--n-runs", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--rhat-max", type=float, default=1.40)
    ap.add_argument("--divergence-rate-max", type=float, default=0.05)
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Make JSON output deterministic (stable ordering; omit timestamps/timings).",
    )
    args = ap.parse_args()

    env_cfg = _require_slow_from_env()
    n_runs = int(args.n_runs if args.n_runs is not None else (env_cfg[0] if env_cfg else 20))
    n_warmup = int(args.warmup if args.warmup is not None else (env_cfg[1] if env_cfg else 200))
    n_samples = int(args.samples if args.samples is not None else (env_cfg[2] if env_cfg else 200))
    seed0 = int(args.seed if args.seed is not None else (env_cfg[3] if env_cfg else 0))

    t0 = time.time()
    report: Dict[str, Any] = {
        "meta": {
            "timestamp": int(t0),
            "python": sys.version.split()[0],
            "n_runs": n_runs,
            "n_warmup": n_warmup,
            "n_samples": n_samples,
            "seed": seed0,
            "cases": args.cases,
            "rhat_max": float(args.rhat_max),
            "divergence_rate_max": float(args.divergence_rate_max),
        },
        "status": "skipped",
        "reason": None,
        "cases": [],
    }

    if os.environ.get("NS_RUN_SLOW") != "1":
        report["reason"] = "NS_RUN_SLOW!=1"
        write_report_json(args.out, report, deterministic=bool(args.deterministic))
        print(f"Wrote: {args.out}")
        return 0

    if n_runs < 10:
        report["reason"] = "n_runs<10"
        write_report_json(args.out, report, deterministic=bool(args.deterministic))
        print(f"Wrote: {args.out}")
        return 0
    if n_warmup < 100 or n_samples < 100:
        report["reason"] = "warmup_or_samples_too_small(<100)"
        write_report_json(args.out, report, deterministic=bool(args.deterministic))
        print(f"Wrote: {args.out}")
        return 0

    try:
        import nextstat  # type: ignore
    except Exception as e:
        report["status"] = "error"
        report["reason"] = f"import_nextstat_failed:{type(e).__name__}:{e}"
        write_report_json(args.out, report, deterministic=bool(args.deterministic))
        print(f"Wrote: {args.out}")
        return 2

    cases = [c.strip() for c in str(args.cases).split(",") if c.strip()]
    any_failed = False
    out_cases: List[Dict[str, Any]] = []
    for c in cases:
        try:
            if c == "lin1d":
                row = _case_linear_regression_1d(
                    nextstat,
                    n_runs=n_runs,
                    n_warmup=n_warmup,
                    n_samples=n_samples,
                    seed0=seed0,
                    rhat_max=float(args.rhat_max),
                    divergence_rate_max=float(args.divergence_rate_max),
                )
            elif c == "lin2d":
                row = _case_linear_regression_2d(
                    nextstat,
                    n_runs=n_runs,
                    n_warmup=n_warmup,
                    n_samples=n_samples,
                    seed0=seed0,
                    rhat_max=float(args.rhat_max),
                    divergence_rate_max=float(args.divergence_rate_max),
                )
            else:
                row = {"name": c, "skipped": True, "reason": f"unknown_case:{c}"}
            if not row.get("skipped") and not bool(row.get("check", {}).get("ok", False)):
                any_failed = True
            out_cases.append(row)
        except Exception as e:
            any_failed = True
            out_cases.append({"name": c, "ok": False, "reason": f"exception:{type(e).__name__}:{e}"})

    report["cases"] = out_cases
    report["status"] = "ok" if not any_failed else "fail"
    report["meta"]["wall_s"] = float(time.time() - t0)

    write_report_json(args.out, report, deterministic=bool(args.deterministic))
    print(f"Wrote: {args.out}")
    return 0 if not any_failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
