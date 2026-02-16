#!/usr/bin/env python3
"""Calibration benchmark: BCa vs percentile under controlled skew scenarios.

This harness compares interval behavior in two workflows:
- HEP: `unbinned-fit-toys` summary mean CI (`summary.mean_ci`)
- Churn: `churn bootstrap-hr` hazard-ratio CIs

Scenarios are chosen to induce asymmetric estimator behavior:
- HEP: POI near lower bound (boundary skew)
- Churn: heavy censoring + smaller sample size

Output files:
- raw_runs.json
- summary.json
- summary.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


def _find_nextstat_binary(repo_root: Path) -> Path:
    candidates = [
        repo_root / "target" / "release" / "nextstat",
        repo_root.parent / ".nextstat-cargo-target" / "release" / "nextstat",
        repo_root / ".nextstat-cargo-target" / "release" / "nextstat",
        repo_root / "target" / "debug" / "nextstat",
        repo_root.parent / ".nextstat-cargo-target" / "debug" / "nextstat",
        repo_root / ".nextstat-cargo-target" / "debug" / "nextstat",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise RuntimeError(
        "nextstat binary not found. Build first, e.g. `cargo build -p ns-cli`."
    )


def _resolve_nextstat_binary(repo_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.exists():
            raise RuntimeError(f"--nextstat-bin not found: {p}")
        return p
    return _find_nextstat_binary(repo_root)


def _fixture_tree(repo_root: Path) -> Path:
    p = repo_root / "tests" / "fixtures" / "simple_tree.root"
    if not p.exists():
        raise RuntimeError(f"fixture not found: {p}")
    return p


def _write_hep_spec(spec_path: Path, root_path: Path, poi_init: float) -> None:
    spec = {
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                {"name": "mu", "init": poi_init, "bounds": [0.0, 5.0]},
                {"name": "gauss_mu", "init": 125.0, "bounds": [125.0, 125.0]},
                {"name": "gauss_sigma", "init": 30.0, "bounds": [30.0, 30.0]},
            ],
        },
        "channels": [
            {
                "name": "SR",
                "include_in_fit": True,
                "data": {"file": str(root_path), "tree": "events"},
                "observables": [{"name": "mbb", "bounds": [0.0, 500.0]}],
                "processes": [
                    {
                        "name": "p",
                        "pdf": {
                            "type": "gaussian",
                            "observable": "mbb",
                            "params": ["gauss_mu", "gauss_sigma"],
                        },
                        "yield": {"type": "scaled", "base_yield": 1000.0, "scale": "mu"},
                    }
                ],
            }
        ],
    }
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")


def _run_hep_once(
    nextstat_bin: Path,
    spec_path: Path,
    method: str,
    seed: int,
    n_toys: int,
    summary_ci_level: float,
    summary_ci_bootstrap: int,
    threads: int,
) -> dict[str, Any]:
    cmd = [
        str(nextstat_bin),
        "unbinned-fit-toys",
        "--config",
        str(spec_path),
        "--n-toys",
        str(n_toys),
        "--seed",
        str(seed),
        "--threads",
        str(threads),
        "--summary-ci-method",
        method,
        "--summary-ci-level",
        str(summary_ci_level),
        "--summary-ci-bootstrap",
        str(summary_ci_bootstrap),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    wall_s = time.perf_counter() - t0
    out = json.loads(proc.stdout)

    summary = out.get("summary", {})
    mean_ci = summary.get("mean_ci", {})
    lower = mean_ci.get("lower")
    upper = mean_ci.get("upper")
    poi_true = out["results"]["poi_true"]
    diagnostics = mean_ci.get("diagnostics", {})

    return {
        "seed": seed,
        "wall_s": wall_s,
        "poi_true": float(poi_true),
        "lower": float(lower),
        "upper": float(upper),
        "width": float(upper) - float(lower),
        "contains_poi_true": float(lower) <= float(poi_true) <= float(upper),
        "requested_method": mean_ci.get("requested_method"),
        "effective_method": mean_ci.get("method"),
        "fallback_reason": diagnostics.get("fallback_reason"),
        "n_converged": int(out["results"]["n_converged"]),
        "n_toys": int(out["results"]["n_toys"]),
    }


def _generate_churn_data(
    nextstat_bin: Path,
    out_path: Path,
    *,
    n_customers: int,
    n_cohorts: int,
    max_time: float,
    treatment_fraction: float,
    seed: int,
) -> dict[str, Any]:
    cmd = [
        str(nextstat_bin),
        "churn",
        "generate-data",
        "--n-customers",
        str(n_customers),
        "--n-cohorts",
        str(n_cohorts),
        "--max-time",
        str(max_time),
        "--treatment-fraction",
        str(treatment_fraction),
        "--seed",
        str(seed),
        "-o",
        str(out_path),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out_path.read_text(encoding="utf-8"))


def _run_churn_once(
    nextstat_bin: Path,
    data_path: Path,
    *,
    method: str,
    n_bootstrap: int,
    seed: int,
    conf_level: float,
    n_jackknife: int,
    truth_hr: dict[str, float],
) -> dict[str, Any]:
    cmd = [
        str(nextstat_bin),
        "churn",
        "bootstrap-hr",
        "--input",
        str(data_path),
        "--n-bootstrap",
        str(n_bootstrap),
        "--seed",
        str(seed),
        "--conf-level",
        str(conf_level),
        "--ci-method",
        method,
        "--n-jackknife",
        str(n_jackknife),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    wall_s = time.perf_counter() - t0
    out = json.loads(proc.stdout)

    by_coeff: dict[str, Any] = {}
    fallback_count = 0
    effective_bca_count = 0
    widths: list[float] = []
    coverage_hits = 0
    coverage_total = 0

    for c in out.get("coefficients", []):
        name = c["name"]
        lo = float(c["hr_ci_lower"])
        hi = float(c["hr_ci_upper"])
        ci_method = c.get("ci_method")
        diag = c.get("ci_diagnostics", {})
        fallback_reason = diag.get("fallback_reason")
        if fallback_reason:
            fallback_count += 1
        if ci_method == "bca":
            effective_bca_count += 1
        widths.append(hi - lo)

        true_hr = truth_hr.get(name)
        contains_true = None
        if true_hr is not None:
            coverage_total += 1
            contains_true = lo <= true_hr <= hi
            if contains_true:
                coverage_hits += 1

        by_coeff[name] = {
            "hr_true": true_hr,
            "hr_point": float(c["hr_point"]),
            "hr_ci_lower": lo,
            "hr_ci_upper": hi,
            "width": hi - lo,
            "contains_true": contains_true,
            "ci_method": ci_method,
            "fallback_reason": fallback_reason,
        }

    return {
        "seed": seed,
        "wall_s": wall_s,
        "elapsed_s": float(out.get("elapsed_s", wall_s)),
        "n_bootstrap": int(out.get("n_bootstrap", n_bootstrap)),
        "n_converged": int(out.get("n_converged", 0)),
        "n_jackknife_requested": int(out.get("n_jackknife_requested", n_jackknife)),
        "n_jackknife_attempted": int(out.get("n_jackknife_attempted", 0)),
        "ci_method_requested": out.get("ci_method_requested"),
        "fallback_count": fallback_count,
        "effective_bca_count": effective_bca_count,
        "mean_width": statistics.fmean(widths) if widths else None,
        "coverage_rate": (coverage_hits / coverage_total) if coverage_total > 0 else None,
        "coverage_hits": coverage_hits,
        "coverage_total": coverage_total,
        "by_coeff": by_coeff,
    }


def _aggregate_hep(rows: list[dict[str, Any]]) -> dict[str, Any]:
    widths = [float(r["width"]) for r in rows]
    walls = [float(r["wall_s"]) for r in rows]
    center_bias = [((float(r["lower"]) + float(r["upper"])) / 2.0) - float(r["poi_true"]) for r in rows]
    coverage = sum(1 for r in rows if r["contains_poi_true"]) / len(rows)
    fallback_count = sum(1 for r in rows if r.get("fallback_reason"))
    effective_bca_count = sum(1 for r in rows if r.get("effective_method") == "bca")
    return {
        "runs": len(rows),
        "coverage_vs_poi_true": coverage,
        "median_width": statistics.median(widths),
        "mean_width": statistics.fmean(widths),
        "median_wall_s": statistics.median(walls),
        "mean_wall_s": statistics.fmean(walls),
        "median_center_minus_poi_true": statistics.median(center_bias),
        "mean_center_minus_poi_true": statistics.fmean(center_bias),
        "fallback_count": fallback_count,
        "effective_bca_count": effective_bca_count,
    }


def _aggregate_churn(rows: list[dict[str, Any]]) -> dict[str, Any]:
    walls = [float(r["wall_s"]) for r in rows]
    mean_widths = [float(r["mean_width"]) for r in rows if r["mean_width"] is not None]
    fallback_total = sum(int(r["fallback_count"]) for r in rows)
    effective_bca_total = sum(int(r["effective_bca_count"]) for r in rows)
    coverage_hits = sum(int(r["coverage_hits"]) for r in rows)
    coverage_total = sum(int(r["coverage_total"]) for r in rows)
    return {
        "runs": len(rows),
        "median_wall_s": statistics.median(walls),
        "mean_wall_s": statistics.fmean(walls),
        "mean_interval_width": statistics.fmean(mean_widths) if mean_widths else None,
        "fallback_total": fallback_total,
        "effective_bca_total": effective_bca_total,
        "coverage_vs_true_hr": (coverage_hits / coverage_total) if coverage_total > 0 else None,
        "coverage_hits": coverage_hits,
        "coverage_total": coverage_total,
    }


def _write_markdown(summary: dict[str, Any], out_path: Path) -> None:
    lines: list[str] = [
        "# BCa Skew Calibration (HEP + Churn)",
        "",
        "Generated by `scripts/benchmarks/bench_bca_skew_calibration.py`.",
        "",
        "## HEP",
        "",
        "| Scenario | Method | Coverage vs poi_true | Median width | Median wall (s) | Median center bias vs poi_true | Fallbacks |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scenario, methods in summary["hep"].items():
        for method, m in methods.items():
            lines.append(
                f"| {scenario} | {method} | {m['coverage_vs_poi_true']:.3f} | "
                f"{m['median_width']:.6f} | {m['median_wall_s']:.3f} | "
                f"{m['median_center_minus_poi_true']:+.6f} | {m['fallback_count']} |"
            )

    lines.extend(
        [
            "",
            "## Churn",
            "",
            "| Scenario | Method | Coverage vs true HR | Mean interval width | Median wall (s) | Fallbacks |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for scenario, methods in summary["churn"].items():
        for method, m in methods.items():
            coverage = m["coverage_vs_true_hr"]
            coverage_str = f"{coverage:.3f}" if coverage is not None else "n/a"
            width = m["mean_interval_width"]
            width_str = f"{width:.6f}" if width is not None else "n/a"
            lines.append(
                f"| {scenario} | {method} | {coverage_str} | {width_str} | "
                f"{m['median_wall_s']:.3f} | {m['fallback_total']} |"
            )

    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="BCa skew calibration benchmark")
    parser.add_argument("--runs", type=int, default=8, help="seed replicates per scenario")
    parser.add_argument("--seed0", type=int, default=100, help="base seed")
    parser.add_argument("--hep-n-toys", type=int, default=120)
    parser.add_argument("--hep-summary-ci-level", type=float, default=0.68)
    parser.add_argument("--hep-summary-ci-bootstrap", type=int, default=300)
    parser.add_argument("--hep-threads", type=int, default=1)
    parser.add_argument("--churn-n-bootstrap", type=int, default=300)
    parser.add_argument("--churn-conf-level", type=float, default=0.95)
    parser.add_argument("--churn-n-jackknife", type=int, default=120)
    parser.add_argument(
        "--nextstat-bin",
        type=Path,
        default=None,
        help="explicit path to nextstat binary (overrides auto-discovery)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("bench_results/bca_skew_calibration"),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    nextstat_bin = _resolve_nextstat_binary(repo_root, args.nextstat_bin)
    root_path = _fixture_tree(repo_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    methods = ["percentile", "bca"]

    hep_scenarios = [
        {"name": "gauss_midpoi", "poi_init": 1.0},
        {"name": "gauss_boundary_lowpoi", "poi_init": 0.1},
    ]
    churn_scenarios = [
        {
            "name": "baseline",
            "n_customers": 2000,
            "n_cohorts": 6,
            "max_time": 24.0,
            "treatment_fraction": 0.3,
        },
        {
            "name": "heavy_censoring_small_n",
            "n_customers": 400,
            "n_cohorts": 6,
            "max_time": 6.0,
            "treatment_fraction": 0.3,
        },
    ]
    true_hr = {
        "plan_basic": 0.65,
        "plan_premium": 0.40,
        "usage_score": 0.80,
        "support_tickets": 1.0,
    }

    raw: dict[str, Any] = {"hep": {}, "churn": {}}

    with tempfile.TemporaryDirectory(prefix="bca_skew_calib_") as tmp:
        tmp_dir = Path(tmp)

        # HEP matrix
        for s in hep_scenarios:
            scenario_name = s["name"]
            spec_path = tmp_dir / f"hep_spec_{scenario_name}.json"
            _write_hep_spec(spec_path, root_path, float(s["poi_init"]))
            raw["hep"][scenario_name] = {m: [] for m in methods}
            for i in range(args.runs):
                seed = args.seed0 + i
                for method in methods:
                    row = _run_hep_once(
                        nextstat_bin=nextstat_bin,
                        spec_path=spec_path,
                        method=method,
                        seed=seed,
                        n_toys=args.hep_n_toys,
                        summary_ci_level=args.hep_summary_ci_level,
                        summary_ci_bootstrap=args.hep_summary_ci_bootstrap,
                        threads=args.hep_threads,
                    )
                    raw["hep"][scenario_name][method].append(row)
                    print(
                        f"[hep:{scenario_name}:{method}] seed={seed} wall={row['wall_s']:.3f}s "
                        f"width={row['width']:.6f} contains={row['contains_poi_true']} "
                        f"effective={row['effective_method']}"
                    )

        # Churn matrix
        for s in churn_scenarios:
            scenario_name = s["name"]
            raw["churn"][scenario_name] = {m: [] for m in methods}
            for i in range(args.runs):
                seed = args.seed0 + i
                data_path = tmp_dir / f"churn_{scenario_name}_seed{seed}.json"
                ds = _generate_churn_data(
                    nextstat_bin=nextstat_bin,
                    out_path=data_path,
                    n_customers=int(s["n_customers"]),
                    n_cohorts=int(s["n_cohorts"]),
                    max_time=float(s["max_time"]),
                    treatment_fraction=float(s["treatment_fraction"]),
                    seed=seed,
                )
                event_rate = float(ds["n_events"]) / float(ds["n"]) if ds["n"] > 0 else 0.0
                for method in methods:
                    row = _run_churn_once(
                        nextstat_bin=nextstat_bin,
                        data_path=data_path,
                        method=method,
                        n_bootstrap=args.churn_n_bootstrap,
                        seed=seed,
                        conf_level=args.churn_conf_level,
                        n_jackknife=args.churn_n_jackknife,
                        truth_hr=true_hr,
                    )
                    row["event_rate"] = event_rate
                    raw["churn"][scenario_name][method].append(row)
                    coverage = row["coverage_rate"]
                    coverage_str = f"{coverage:.3f}" if coverage is not None else "n/a"
                    print(
                        f"[churn:{scenario_name}:{method}] seed={seed} wall={row['wall_s']:.3f}s "
                        f"coverage={coverage_str} fallback={row['fallback_count']} "
                        f"effective_bca={row['effective_bca_count']} event_rate={event_rate:.3f}"
                    )

    summary = {
        "config": {
            "runs": args.runs,
            "seed0": args.seed0,
            "hep": {
                "n_toys": args.hep_n_toys,
                "summary_ci_level": args.hep_summary_ci_level,
                "summary_ci_bootstrap": args.hep_summary_ci_bootstrap,
                "threads": args.hep_threads,
            },
            "churn": {
                "n_bootstrap": args.churn_n_bootstrap,
                "conf_level": args.churn_conf_level,
                "n_jackknife": args.churn_n_jackknife,
                "truth_hr": true_hr,
            },
            "nextstat_bin": str(nextstat_bin),
            "fixture_root": str(root_path),
        },
        "hep": {
            scenario: {m: _aggregate_hep(rows) for m, rows in methods_map.items()}
            for scenario, methods_map in raw["hep"].items()
        },
        "churn": {
            scenario: {m: _aggregate_churn(rows) for m, rows in methods_map.items()}
            for scenario, methods_map in raw["churn"].items()
        },
    }

    (args.out_dir / "raw_runs.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, args.out_dir / "summary.md")

    print(f"\nSaved: {args.out_dir / 'raw_runs.json'}")
    print(f"Saved: {args.out_dir / 'summary.json'}")
    print(f"Saved: {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
