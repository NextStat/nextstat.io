#!/usr/bin/env python3
"""Churn benchmark: percentile vs BCa in `churn bootstrap-hr`.

Generates one synthetic churn dataset and compares CI methods on the same seeds:
- percentile
- bca

Metrics:
- wall-clock time,
- interval width (mean across coefficients),
- BCa fallback count,
- coverage vs generator truth (optional, with `--use-default-truth`),
- per-coefficient coverage aggregates.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path


TRUE_HR_DEFAULT = {
    "plan_basic": 0.65,
    "plan_premium": 0.40,
    "usage_score": 0.80,
    "support_tickets": 1.0,
}


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


def _generate_data(
    nextstat_bin: Path,
    out_path: Path,
    n_customers: int,
    seed: int,
    n_cohorts: int,
    max_time: float,
    treatment_fraction: float,
) -> None:
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


def _run_bootstrap(
    nextstat_bin: Path,
    data_path: Path,
    method: str,
    n_bootstrap: int,
    seed: int,
    conf_level: float,
    n_jackknife: int,
    truth_hr: dict[str, float],
) -> dict:
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

    coefs = out.get("coefficients", [])
    widths = []
    fallback_count = 0
    effective_bca = 0
    coverage_hits = 0
    coverage_total = 0
    per_coeff: dict[str, dict] = {}
    for c in coefs:
        name = c.get("name")
        lo = c.get("hr_ci_lower")
        hi = c.get("hr_ci_upper")
        point = c.get("hr_point")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            widths.append(float(hi) - float(lo))
            if isinstance(name, str):
                hr_true = truth_hr.get(name)
                contains_true = None
                if isinstance(hr_true, (int, float)):
                    coverage_total += 1
                    if float(lo) <= float(hr_true) <= float(hi):
                        coverage_hits += 1
                        contains_true = True
                    else:
                        contains_true = False
                per_coeff[name] = {
                    "hr_true": hr_true,
                    "hr_point": float(point) if isinstance(point, (int, float)) else None,
                    "hr_ci_lower": float(lo),
                    "hr_ci_upper": float(hi),
                    "width": float(hi) - float(lo),
                    "contains_true": contains_true,
                }
        diag = c.get("ci_diagnostics", {})
        if diag.get("fallback_reason"):
            fallback_count += 1
        if c.get("ci_method") == "bca":
            effective_bca += 1

    return {
        "method_requested": method,
        "method_root": out.get("ci_method_requested"),
        "n_bootstrap": out.get("n_bootstrap"),
        "n_jackknife_requested": out.get("n_jackknife_requested"),
        "n_jackknife_attempted": out.get("n_jackknife_attempted"),
        "n_converged": out.get("n_converged"),
        "elapsed_s": out.get("elapsed_s"),
        "wall_s": wall_s,
        "mean_width": statistics.fmean(widths) if widths else None,
        "fallback_count": fallback_count,
        "effective_bca_count": effective_bca,
        "n_coefficients": len(coefs),
        "coverage_rate": (coverage_hits / coverage_total) if coverage_total > 0 else None,
        "coverage_hits": coverage_hits,
        "coverage_total": coverage_total,
        "by_coefficient": per_coeff,
    }


def _aggregate(rows: list[dict]) -> dict:
    walls = [float(r["wall_s"]) for r in rows]
    mean_widths = [float(r["mean_width"]) for r in rows if r["mean_width"] is not None]
    coverage_hits = sum(int(r.get("coverage_hits", 0)) for r in rows)
    coverage_total = sum(int(r.get("coverage_total", 0)) for r in rows)
    coeff_names: set[str] = set()
    for r in rows:
        coeff_names.update(r.get("by_coefficient", {}).keys())
    per_coefficient: dict[str, dict] = {}
    for name in sorted(coeff_names):
        hits = 0
        total = 0
        widths: list[float] = []
        points: list[float] = []
        true_vals: list[float] = []
        for r in rows:
            c = r.get("by_coefficient", {}).get(name)
            if not isinstance(c, dict):
                continue
            w = c.get("width")
            if isinstance(w, (int, float)):
                widths.append(float(w))
            p = c.get("hr_point")
            if isinstance(p, (int, float)):
                points.append(float(p))
            t = c.get("hr_true")
            if isinstance(t, (int, float)):
                true_vals.append(float(t))
            contains = c.get("contains_true")
            if isinstance(contains, bool):
                total += 1
                if contains:
                    hits += 1
        hr_true = true_vals[0] if true_vals else None
        per_coefficient[name] = {
            "hr_true": hr_true,
            "coverage_hits": hits,
            "coverage_total": total,
            "coverage_vs_true_hr": (hits / total) if total > 0 else None,
            "mean_width": statistics.fmean(widths) if widths else None,
            "mean_hr_point": statistics.fmean(points) if points else None,
        }
    return {
        "runs": len(rows),
        "median_wall_s": statistics.median(walls),
        "mean_wall_s": statistics.fmean(walls),
        "mean_interval_width": statistics.fmean(mean_widths) if mean_widths else None,
        "fallback_total": sum(int(r["fallback_count"]) for r in rows),
        "effective_bca_total": sum(int(r["effective_bca_count"]) for r in rows),
        "coverage_vs_true_hr": (coverage_hits / coverage_total) if coverage_total > 0 else None,
        "coverage_hits": coverage_hits,
        "coverage_total": coverage_total,
        "per_coefficient": per_coefficient,
    }


def _write_markdown(summary: dict, out_path: Path) -> None:
    p = summary["percentile"]
    b = summary["bca"]
    overhead = (b["median_wall_s"] / p["median_wall_s"]) if p["median_wall_s"] > 0 else None

    lines = [
        "# Churn Bootstrap CI Method Benchmark",
        "",
        "Generated by `scripts/benchmarks/bench_churn_bootstrap_ci_methods.py`.",
        "",
        "| Method | Runs | Coverage vs true HR | Median wall (s) | Mean interval width | Fallback total |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| percentile | {p['runs']} | "
            f"{(f'{p['coverage_vs_true_hr']:.6f}' if p['coverage_vs_true_hr'] is not None else 'n/a')} | "
            f"{p['median_wall_s']:.3f} | {p['mean_interval_width']:.6f} | {p['fallback_total']} |"
        ),
        (
            f"| bca | {b['runs']} | "
            f"{(f'{b['coverage_vs_true_hr']:.6f}' if b['coverage_vs_true_hr'] is not None else 'n/a')} | "
            f"{b['median_wall_s']:.3f} | {b['mean_interval_width']:.6f} | {b['fallback_total']} |"
        ),
        "",
    ]
    if overhead is not None:
        lines.append(f"- BCa median wall-time overhead vs percentile: `{overhead:.3f}x`")
    lines.append(
        f"- BCa effective coefficient count: `{b['effective_bca_total']}` across all runs."
    )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark churn bootstrap CI methods")
    parser.add_argument("--n-customers", type=int, default=1000)
    parser.add_argument("--n-cohorts", type=int, default=6)
    parser.add_argument("--max-time", type=float, default=24.0)
    parser.add_argument("--treatment-fraction", type=float, default=0.3)
    parser.add_argument("--n-bootstrap", type=int, default=300)
    parser.add_argument("--conf-level", type=float, default=0.95)
    parser.add_argument("--n-jackknife", type=int, default=120)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed0", type=int, default=42)
    parser.add_argument(
        "--use-default-truth",
        action="store_true",
        help="compute coverage against default generator truth HR values",
    )
    parser.add_argument(
        "--regenerate-data-per-run",
        action="store_true",
        help="generate a fresh churn dataset for each run (dataset-level coverage)",
    )
    parser.add_argument(
        "--nextstat-bin",
        type=Path,
        default=None,
        help="explicit path to nextstat binary (overrides auto-discovery)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("bench_results/churn_bootstrap_ci_methods"),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    nextstat_bin = _resolve_nextstat_binary(repo_root, args.nextstat_bin)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    methods = ["percentile", "bca"]
    raw: dict[str, list[dict]] = {m: [] for m in methods}

    with tempfile.TemporaryDirectory(prefix="churn_ci_bench_") as tmp:
        tmp_dir = Path(tmp)

        fixed_data_path = args.out_dir / "churn_data.json"
        if not args.regenerate_data_per_run:
            _generate_data(
                nextstat_bin,
                fixed_data_path,
                args.n_customers,
                args.seed0,
                args.n_cohorts,
                args.max_time,
                args.treatment_fraction,
            )

        for i in range(args.runs):
            seed = args.seed0 + i
            data_seed = seed if args.regenerate_data_per_run else args.seed0
            data_path = (
                tmp_dir / f"churn_data_seed{data_seed}.json"
                if args.regenerate_data_per_run
                else fixed_data_path
            )
            if args.regenerate_data_per_run:
                _generate_data(
                    nextstat_bin,
                    data_path,
                    args.n_customers,
                    data_seed,
                    args.n_cohorts,
                    args.max_time,
                    args.treatment_fraction,
                )

            for method in methods:
                row = _run_bootstrap(
                    nextstat_bin=nextstat_bin,
                    data_path=data_path,
                    method=method,
                    n_bootstrap=args.n_bootstrap,
                    seed=seed,
                    conf_level=args.conf_level,
                    n_jackknife=args.n_jackknife,
                    truth_hr=TRUE_HR_DEFAULT if args.use_default_truth else {},
                )
                row["data_seed"] = data_seed
                raw[method].append(row)
                coverage = row.get("coverage_rate")
                coverage_str = f"{coverage:.3f}" if isinstance(coverage, (int, float)) else "n/a"
                print(
                    f"[{method}] seed={seed} data_seed={data_seed} wall={row['wall_s']:.3f}s "
                    f"elapsed={row['elapsed_s']:.3f}s conv={row['n_converged']}/{args.n_bootstrap} "
                    f"mean_width={row['mean_width']:.6f} coverage={coverage_str}"
                )

    summary = {
        "config": {
            "n_customers": args.n_customers,
            "n_cohorts": args.n_cohorts,
            "max_time": args.max_time,
            "treatment_fraction": args.treatment_fraction,
            "n_bootstrap": args.n_bootstrap,
            "conf_level": args.conf_level,
            "n_jackknife": args.n_jackknife,
            "runs": args.runs,
            "seed0": args.seed0,
            "regenerate_data_per_run": bool(args.regenerate_data_per_run),
            "nextstat_bin": str(nextstat_bin),
            "use_default_truth": bool(args.use_default_truth),
            "truth_hr_default": TRUE_HR_DEFAULT if args.use_default_truth else None,
        },
        "percentile": _aggregate(raw["percentile"]),
        "bca": _aggregate(raw["bca"]),
    }

    (args.out_dir / "raw_runs.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(summary, args.out_dir / "summary.md")

    print(f"\nSaved: {args.out_dir / 'raw_runs.json'}")
    print(f"Saved: {args.out_dir / 'summary.json'}")
    print(f"Saved: {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
