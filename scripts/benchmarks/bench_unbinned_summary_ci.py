#!/usr/bin/env python3
"""HEP benchmark: percentile vs BCa summary CI for `unbinned-fit-toys`.

This script runs `nextstat unbinned-fit-toys` with:
- `--summary-ci-method percentile`
- `--summary-ci-method bca`

for the same seeds and compares:
- wall-clock overhead,
- CI width,
- empirical inclusion of the generator POI (`results.poi_true`) by `summary.mean_ci`.

Usage:
    python scripts/benchmarks/bench_unbinned_summary_ci.py \
      --runs 8 --n-toys 120 --summary-ci-bootstrap 300 \
      --out-dir bench_results/unbinned_summary_ci
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

from _parse_utils import parse_json_stdout


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


def _write_spec(spec_path: Path, root_path: Path) -> None:
    spec = {
        "schema_version": "nextstat_unbinned_spec_v0",
        "model": {
            "poi": "mu",
            "parameters": [
                {"name": "mu", "init": 1.0, "bounds": [0.0, 5.0]},
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


def _run_once(
    nextstat_bin: Path,
    spec_path: Path,
    method: str,
    seed: int,
    n_toys: int,
    summary_ci_level: float,
    summary_ci_bootstrap: int,
    threads: int,
) -> dict:
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
    out = parse_json_stdout(proc.stdout)

    summary = out.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("missing summary object in unbinned-fit-toys output")
    mean_ci = summary.get("mean_ci")
    if not isinstance(mean_ci, dict):
        raise RuntimeError("missing summary.mean_ci in unbinned-fit-toys output")

    lower = mean_ci.get("lower")
    upper = mean_ci.get("upper")
    if lower is None or upper is None:
        raise RuntimeError("summary.mean_ci lower/upper are null")

    poi_true = out["results"]["poi_true"]
    effective_method = mean_ci.get("method")
    requested_method = mean_ci.get("requested_method")
    width = float(upper) - float(lower)
    contains_true = float(lower) <= float(poi_true) <= float(upper)

    diagnostics = mean_ci.get("diagnostics", {})
    fallback_reason = diagnostics.get("fallback_reason")

    return {
        "seed": seed,
        "requested_method": requested_method,
        "effective_method": effective_method,
        "lower": float(lower),
        "upper": float(upper),
        "width": width,
        "contains_poi_true": bool(contains_true),
        "poi_true": float(poi_true),
        "wall_s": wall_s,
        "n_converged": int(out["results"]["n_converged"]),
        "n_toys": int(out["results"]["n_toys"]),
        "fallback_reason": fallback_reason,
    }


def _aggregate(rows: list[dict]) -> dict:
    widths = [r["width"] for r in rows]
    walls = [r["wall_s"] for r in rows]
    center_bias = [((r["lower"] + r["upper"]) / 2.0) - r["poi_true"] for r in rows]
    coverage = sum(1 for r in rows if r["contains_poi_true"]) / len(rows)
    fallback_count = sum(1 for r in rows if r.get("fallback_reason"))
    bca_effective = sum(1 for r in rows if r.get("effective_method") == "bca")

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
        "effective_bca_count": bca_effective,
    }


def _write_markdown(summary: dict, out_path: Path) -> None:
    p = summary["percentile"]
    b = summary["bca"]
    overhead = (b["median_wall_s"] / p["median_wall_s"]) if p["median_wall_s"] > 0 else None

    lines = [
        "# HEP Summary CI Benchmark (unbinned-fit-toys)",
        "",
        "Generated by `scripts/benchmarks/bench_unbinned_summary_ci.py`.",
        "",
        "| Method | Runs | Coverage vs poi_true | Median width | Median wall (s) | Fallbacks |",
        "|---|---:|---:|---:|---:|---:|",
        f"| percentile | {p['runs']} | {p['coverage_vs_poi_true']:.3f} | {p['median_width']:.6f} | {p['median_wall_s']:.3f} | {p['fallback_count']} |",
        f"| bca | {b['runs']} | {b['coverage_vs_poi_true']:.3f} | {b['median_width']:.6f} | {b['median_wall_s']:.3f} | {b['fallback_count']} |",
        "",
    ]
    if overhead is not None:
        lines.append(f"- BCa median wall-time overhead vs percentile: `{overhead:.3f}x`")
    lines.append(
        f"- BCa effective count: `{b['effective_bca_count']}/{b['runs']}` (others fallback to percentile)."
    )
    lines.append(
        f"- Percentile median CI-center bias vs `poi_true`: `{p['median_center_minus_poi_true']:+.6f}`."
    )
    lines.append(
        f"- BCa median CI-center bias vs `poi_true`: `{b['median_center_minus_poi_true']:+.6f}`."
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark HEP summary CI methods")
    parser.add_argument("--runs", type=int, default=8, help="number of seed replicates")
    parser.add_argument("--seed0", type=int, default=100, help="base seed")
    parser.add_argument("--n-toys", type=int, default=120, help="toys per run")
    parser.add_argument("--summary-ci-level", type=float, default=0.68)
    parser.add_argument("--summary-ci-bootstrap", type=int, default=300)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--nextstat-bin",
        type=Path,
        default=None,
        help="explicit path to nextstat binary (overrides auto-discovery)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("bench_results/unbinned_summary_ci"),
        help="output directory",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    nextstat_bin = _resolve_nextstat_binary(repo_root, args.nextstat_bin)
    root_path = _fixture_tree(repo_root)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = args.out_dir / "unbinned_spec_summary_ci.json"
    _write_spec(spec_path, root_path)

    methods = ["percentile", "bca"]
    raw: dict[str, list[dict]] = {m: [] for m in methods}

    for i in range(args.runs):
        seed = args.seed0 + i
        for method in methods:
            row = _run_once(
                nextstat_bin=nextstat_bin,
                spec_path=spec_path,
                method=method,
                seed=seed,
                n_toys=args.n_toys,
                summary_ci_level=args.summary_ci_level,
                summary_ci_bootstrap=args.summary_ci_bootstrap,
                threads=args.threads,
            )
            raw[method].append(row)
            print(
                f"[{method}] seed={seed} wall={row['wall_s']:.3f}s "
                f"width={row['width']:.6f} contains={row['contains_poi_true']} "
                f"effective={row['effective_method']}"
            )

    summary = {
        "config": {
            "runs": args.runs,
            "seed0": args.seed0,
            "n_toys": args.n_toys,
            "summary_ci_level": args.summary_ci_level,
            "summary_ci_bootstrap": args.summary_ci_bootstrap,
            "threads": args.threads,
            "fixture_root": str(root_path),
            "nextstat_bin": str(nextstat_bin),
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
