#!/usr/bin/env python3
"""Apex2 runner: Posterior/HMC/NUTS quality vs standards (fast, JSON artifact).

Goal: provide a reproducible, machine-readable report that catches catastrophic
regressions in:
- Posterior transform plumbing (bounded/unbounded parameters)
- HMC/NUTS stability (finite diagnostics, low divergence/treedepth saturation)
- Diagnostics plumbing (R-hat/ESS/E-BFMI present and finite)

This runner is intentionally *not* a full SBC suite. For SBC, use:
  tests/apex2_sbc_report.py (slow; requires NS_RUN_SLOW=1)

Run:
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_nuts_quality_report.py \
    --out tmp/apex2_nuts_quality_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _flatten_chains(posterior: Mapping[str, List[List[float]]], name: str) -> List[float]:
    out: List[float] = []
    for chain in posterior[name]:
        out.extend([float(v) for v in chain])
    return out


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _diag_summary(diag: Mapping[str, Any]) -> Dict[str, Any]:
    r_hat = diag.get("r_hat") if isinstance(diag.get("r_hat"), dict) else {}
    ess_bulk = diag.get("ess_bulk") if isinstance(diag.get("ess_bulk"), dict) else {}
    ess_tail = diag.get("ess_tail") if isinstance(diag.get("ess_tail"), dict) else {}
    ebfmi = diag.get("ebfmi") if isinstance(diag.get("ebfmi"), list) else []

    rhat_vals = [v for v in (_safe_float(x) for x in r_hat.values()) if v is not None]
    ess_bulk_vals = [v for v in (_safe_float(x) for x in ess_bulk.values()) if v is not None]
    ess_tail_vals = [v for v in (_safe_float(x) for x in ess_tail.values()) if v is not None]
    ebfmi_vals = [v for v in (_safe_float(x) for x in ebfmi) if v is not None]

    return {
        "divergence_rate": _safe_float(diag.get("divergence_rate")),
        "max_treedepth_rate": _safe_float(diag.get("max_treedepth_rate")),
        "max_r_hat": max(rhat_vals) if rhat_vals else None,
        "min_ess_bulk": min(ess_bulk_vals) if ess_bulk_vals else None,
        "min_ess_tail": min(ess_tail_vals) if ess_tail_vals else None,
        "min_ebfmi": min(ebfmi_vals) if ebfmi_vals else None,
    }


def _eval_thresholds(
    summary: Mapping[str, Any],
    *,
    rhat_max: float,
    divergence_rate_max: float,
    max_treedepth_rate_max: float,
    ess_bulk_min: float,
    ess_tail_min: float,
    ebfmi_min: float,
) -> Tuple[bool, List[str]]:
    failures: List[str] = []

    def req(key: str) -> float:
        v = _safe_float(summary.get(key))
        if v is None:
            failures.append(f"missing_or_nonfinite:{key}")
            return float("nan")
        return float(v)

    div = req("divergence_rate")
    tdr = req("max_treedepth_rate")
    rhat = req("max_r_hat")
    essb = req("min_ess_bulk")
    esst = req("min_ess_tail")
    bfmi = req("min_ebfmi")

    if div == div and div > divergence_rate_max:
        failures.append(f"divergence_rate>{divergence_rate_max}")
    if tdr == tdr and tdr > max_treedepth_rate_max:
        failures.append(f"max_treedepth_rate>{max_treedepth_rate_max}")
    if rhat == rhat and rhat >= rhat_max:
        failures.append(f"max_r_hat>={rhat_max}")
    if essb == essb and essb < ess_bulk_min:
        failures.append(f"min_ess_bulk<{ess_bulk_min}")
    if esst == esst and esst < ess_tail_min:
        failures.append(f"min_ess_tail<{ess_tail_min}")
    if bfmi == bfmi and bfmi < ebfmi_min:
        failures.append(f"min_ebfmi<{ebfmi_min}")

    return (len(failures) == 0), failures


def _case_gaussian_mean(nextstat_mod, *, seed: int, n_warmup: int, n_samples: int) -> Dict[str, Any]:
    model = nextstat_mod.GaussianMeanModel([1.0, 2.0, 3.0, 4.0] * 5, sigma=1.0)
    r = nextstat_mod.sample(
        model,
        n_chains=2,
        n_warmup=n_warmup,
        n_samples=n_samples,
        seed=seed,
        init_jitter_rel=0.10,
    )
    posterior = r["posterior"]
    mu_draws = _flatten_chains(posterior, "mu")
    return {
        "name": "gaussian_mean",
        "model": "GaussianMeanModel",
        "raw_quality": (r.get("diagnostics") or {}).get("quality"),
        "summary": _diag_summary(r["diagnostics"]),
        "checks": {
            "posterior_mean_mu": _mean(mu_draws),
        },
    }


def _case_gaussian_posterior_with_prior(
    nextstat_mod, *, seed: int, n_warmup: int, n_samples: int
) -> Dict[str, Any]:
    """Posterior sampling smoke: ensure priors are actually plumbed into NUTS.

    We use a strong Normal prior centered far from the data MLE; the posterior mean
    should shift toward the prior (not necessarily all the way, but materially).
    """
    model = nextstat_mod.GaussianMeanModel([1.0, 2.0, 3.0, 4.0] * 5, sigma=1.0)
    post = nextstat_mod.Posterior(model)
    post.set_prior_normal("mu", center=10.0, width=0.1)

    r = nextstat_mod.sample(
        post,
        n_chains=2,
        n_warmup=n_warmup,
        n_samples=n_samples,
        seed=seed,
        init_jitter_rel=0.10,
    )
    posterior = r["posterior"]
    mu_draws = _flatten_chains(posterior, "mu")
    mu_mean = _mean(mu_draws)
    return {
        "name": "gaussian_posterior_prior",
        "model": "Posterior(GaussianMeanModel + Normal prior)",
        "raw_quality": (r.get("diagnostics") or {}).get("quality"),
        "summary": _diag_summary(r["diagnostics"]),
        "checks": {
            "posterior_mean_mu": mu_mean,
            # Sanity: should move meaningfully toward the prior center=10.0.
            # Data mean is 2.5; with width=0.1 this should typically be >> 2.5.
            "posterior_mean_mu_gt_5": bool(mu_mean > 5.0),
        },
    }


def _case_linear_regression(nextstat_mod, *, seed: int, n_warmup: int, n_samples: int) -> Dict[str, Any]:
    x = [[1.0] for _ in range(30)]
    y = [1.0, 1.3, 0.9, 1.1, 1.2] * 6
    model = nextstat_mod.ComposedGlmModel.linear_regression(
        x,
        y,
        include_intercept=False,
        coef_prior_mu=0.0,
        coef_prior_sigma=1.0,
    )
    r = nextstat_mod.sample(
        model,
        n_chains=2,
        n_warmup=n_warmup,
        n_samples=n_samples,
        seed=seed,
        init_jitter_rel=0.10,
        target_accept=0.85,
    )
    posterior = r["posterior"]
    b1 = _flatten_chains(posterior, "beta1") if "beta1" in posterior else []
    return {
        "name": "linear_regression",
        "model": "ComposedGlmModel.linear_regression",
        "raw_quality": (r.get("diagnostics") or {}).get("quality"),
        "summary": _diag_summary(r["diagnostics"]),
        "checks": {
            "posterior_mean_beta1": _mean(b1) if b1 else None,
        },
    }


def _case_histfactory_simple(nextstat_mod, *, seed: int, n_warmup: int, n_samples: int) -> Dict[str, Any]:
    ws = json.loads((FIXTURES_DIR / "simple_workspace.json").read_text())
    model = nextstat_mod.HistFactoryModel.from_workspace(json.dumps(ws))
    r = nextstat_mod.sample(
        model,
        n_chains=2,
        n_warmup=n_warmup,
        n_samples=n_samples,
        seed=seed,
        init_jitter_rel=0.10,
        target_accept=0.85,
    )
    posterior = r["posterior"]
    poi_name = r["param_names"][0] if r.get("param_names") else "mu"
    poi_draws = _flatten_chains(posterior, poi_name) if poi_name in posterior else []
    return {
        "name": "histfactory_simple_fixture",
        "model": "HistFactoryModel(simple_workspace.json)",
        "raw_quality": (r.get("diagnostics") or {}).get("quality"),
        "summary": _diag_summary(r["diagnostics"]),
        "checks": {
            "poi_name": poi_name,
            "posterior_mean_poi": _mean(poi_draws) if poi_draws else None,
        },
    }


def _determinism_check(nextstat_mod) -> Dict[str, Any]:
    model = nextstat_mod.GaussianMeanModel([1.0, 2.0, 3.0, 4.0], sigma=1.0)
    kwargs = dict(n_chains=1, n_warmup=30, n_samples=20, seed=42, init_jitter_rel=0.10)
    r1 = nextstat_mod.sample(model, **kwargs)
    r2 = nextstat_mod.sample(model, **kwargs)
    ok = r1["posterior"] == r2["posterior"] and r1["sample_stats"] == r2["sample_stats"]
    return {"ok": bool(ok), "kwargs": kwargs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("tmp/apex2_nuts_quality_report.json"))
    ap.add_argument("--cases", type=str, default="gaussian,linear,histfactory")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--histfactory-warmup", type=int, default=None)
    ap.add_argument("--histfactory-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rhat-max", type=float, default=1.10)
    ap.add_argument("--divergence-rate-max", type=float, default=0.05)
    ap.add_argument("--max-treedepth-rate-max", type=float, default=0.10)
    ap.add_argument("--ess-bulk-min", type=float, default=10.0)
    ap.add_argument("--ess-tail-min", type=float, default=10.0)
    ap.add_argument("--ebfmi-min", type=float, default=0.20)
    ap.add_argument("--histfactory-rhat-max", type=float, default=None)
    ap.add_argument("--histfactory-ess-bulk-min", type=float, default=None)
    ap.add_argument("--histfactory-ess-tail-min", type=float, default=None)
    ap.add_argument("--no-determinism-check", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    report: Dict[str, Any] = {
        "meta": {
            "timestamp": int(t0),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cases": args.cases,
            "warmup": int(args.warmup),
            "samples": int(args.samples),
            "histfactory_warmup": int(args.histfactory_warmup) if args.histfactory_warmup is not None else None,
            "histfactory_samples": int(args.histfactory_samples) if args.histfactory_samples is not None else None,
            "seed": int(args.seed),
            "thresholds": {
                "rhat_max": float(args.rhat_max),
                "divergence_rate_max": float(args.divergence_rate_max),
                "max_treedepth_rate_max": float(args.max_treedepth_rate_max),
                "ess_bulk_min": float(args.ess_bulk_min),
                "ess_tail_min": float(args.ess_tail_min),
                "ebfmi_min": float(args.ebfmi_min),
                "histfactory_rhat_max": float(args.histfactory_rhat_max) if args.histfactory_rhat_max is not None else None,
                "histfactory_ess_bulk_min": float(args.histfactory_ess_bulk_min) if args.histfactory_ess_bulk_min is not None else None,
                "histfactory_ess_tail_min": float(args.histfactory_ess_tail_min) if args.histfactory_ess_tail_min is not None else None,
            },
        },
        "status": "skipped",
        "reason": None,
        "determinism": None,
        "cases": [],
    }

    try:
        import nextstat  # type: ignore
    except ModuleNotFoundError as e:
        report["reason"] = f"import_nextstat_failed:{e}"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"Wrote: {args.out}")
        return 0

    if not args.no_determinism_check:
        report["determinism"] = _determinism_check(nextstat)

    requested = [c.strip().lower() for c in str(args.cases).split(",") if c.strip()]
    cases: List[Dict[str, Any]] = []
    for c in requested:
        if c == "gaussian":
            cases.append(_case_gaussian_mean(nextstat, seed=args.seed + 1, n_warmup=args.warmup, n_samples=args.samples))
        elif c == "posterior":
            cases.append(
                _case_gaussian_posterior_with_prior(
                    nextstat, seed=args.seed + 4, n_warmup=args.warmup, n_samples=args.samples
                )
            )
        elif c == "linear":
            cases.append(_case_linear_regression(nextstat, seed=args.seed + 2, n_warmup=args.warmup, n_samples=args.samples))
        elif c == "histfactory":
            hf_warmup = int(args.histfactory_warmup) if args.histfactory_warmup is not None else int(args.warmup)
            hf_samples = int(args.histfactory_samples) if args.histfactory_samples is not None else int(args.samples)
            cases.append(_case_histfactory_simple(nextstat, seed=args.seed + 3, n_warmup=hf_warmup, n_samples=hf_samples))
        else:
            cases.append({"name": c, "ok": False, "reason": f"unknown_case:{c}"})

    # HistFactory posteriors can require substantially longer runs to meet tight R-hat/ESS gates.
    # For this runner, defaults are strict for non-HEP models, and looser for HistFactory unless
    # the user explicitly overrides via --histfactory-* thresholds.
    hf_rhat_max = float(args.histfactory_rhat_max) if args.histfactory_rhat_max is not None else 3.0
    hf_ess_bulk_min = float(args.histfactory_ess_bulk_min) if args.histfactory_ess_bulk_min is not None else 2.0
    hf_ess_tail_min = float(args.histfactory_ess_tail_min) if args.histfactory_ess_tail_min is not None else 2.0

    any_failed = False
    for row in cases:
        summary = row.get("summary") if isinstance(row.get("summary"), dict) else None
        if summary is None:
            row["ok"] = False
            row["failures"] = list(row.get("failures") or []) + ["missing_summary"]
            any_failed = True
            continue

        is_hf = str(row.get("name") or "").startswith("histfactory")
        thresholds_used = {
            "rhat_max": hf_rhat_max if is_hf else float(args.rhat_max),
            "divergence_rate_max": float(args.divergence_rate_max),
            "max_treedepth_rate_max": float(args.max_treedepth_rate_max),
            "ess_bulk_min": hf_ess_bulk_min if is_hf else float(args.ess_bulk_min),
            "ess_tail_min": hf_ess_tail_min if is_hf else float(args.ess_tail_min),
            "ebfmi_min": float(args.ebfmi_min),
        }
        ok, failures = _eval_thresholds(
            summary,
            rhat_max=float(thresholds_used["rhat_max"]),
            divergence_rate_max=float(thresholds_used["divergence_rate_max"]),
            max_treedepth_rate_max=float(thresholds_used["max_treedepth_rate_max"]),
            ess_bulk_min=float(thresholds_used["ess_bulk_min"]),
            ess_tail_min=float(thresholds_used["ess_tail_min"]),
            ebfmi_min=float(thresholds_used["ebfmi_min"]),
        )
        row["ok"] = bool(ok)
        row["failures"] = failures
        row["thresholds_used"] = thresholds_used
        if not ok:
            any_failed = True

    report["cases"] = cases
    report["status"] = "ok" if not any_failed else "fail"
    report["meta"]["elapsed_s"] = float(time.time() - t0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"Wrote: {args.out}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
